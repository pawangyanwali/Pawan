"""
Parallel scan orchestrator for the NASDAQ agent.

Wraps the existing analyse_ticker() function and executes it concurrently
across all tickers using ThreadPoolExecutor.  Feature computation and model
inference release the GIL for I/O-heavy sections, so threads are effective
even for CPU-leaning workloads.

Performance targets (ThreadPoolExecutor, n_workers=8):
  65 tickers  → < 2 s   (vs ~5 s sequential)
  165 tickers → < 5 s
  250 tickers → < 8 s
"""

import logging
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_slow_skip_until: dict[str, float] = {}
_slow_skip_lock = threading.Lock()


# ── Batch XGBoost helper ───────────────────────────────────────────────────────

def batch_predict_xgb(
    model,
    scaler,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    """
    Run predict_proba for N tickers in a single XGBoost call.

    Calling predict_proba() inside a per-ticker loop pays Python overhead and
    XGBoost thread-pool startup cost N times.  Batching reduces that to once.

    Parameters
    ----------
    model : fitted XGBClassifier or CalibratedClassifierCV
    scaler : fitted StandardScaler (or compatible transform with .transform())
    feature_matrix : np.ndarray, shape (N, n_features)
        Each row is the feature vector for one ticker.

    Returns
    -------
    np.ndarray, shape (N,)
        Probability of the positive class for each ticker.
    """
    if feature_matrix.ndim != 2 or feature_matrix.shape[0] == 0:
        return np.full(max(feature_matrix.shape[0], 0), 0.5)

    X_scaled = scaler.transform(feature_matrix)
    proba = model.predict_proba(X_scaled)

    # predict_proba returns (N, n_classes); take the positive-class column.
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    return proba.ravel()


# ── Optimal worker count ───────────────────────────────────────────────────────

def auto_workers() -> int:
    """Return optimal worker count: min(cpu_count, 12)."""
    return min(os.cpu_count() or 4, 12)


# ── Pipeline ───────────────────────────────────────────────────────────────────

class ScanPipeline:
    """
    Parallel ticker scanner.  Wraps the existing analyse_ticker() function
    and runs it concurrently across all tickers using ThreadPoolExecutor.

    Also tracks per-cycle timing metrics that the UI can expose.

    Parameters
    ----------
    n_workers : int
        Number of threads.  Defaults to 8; use auto_workers() for CPU-tuned
        selection (capped at 12 to avoid overwhelming shared ML models).
    """

    def __init__(self, n_workers: int = 8) -> None:
        self.n_workers = n_workers
        self._metrics: dict = {
            "last_cycle_ms": 0,
            "last_scan_count": 0,
            "avg_cycle_ms": 0.0,
            "cycles_completed": 0,
            "errors_last_cycle": 0,
            "timeouts_last_cycle": 0,
            "skipped_slow_cooldown": 0,
            "tickers_per_second": 0.0,
            "slow_tickers_last_cycle": [],
        }
        self._metrics_lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def scan(
        self,
        tickers: list[str],
        data_1m: dict[str, pd.DataFrame],
        data_5m: dict[str, pd.DataFrame],
        data_1h: dict[str, pd.DataFrame],
        data_1d: dict[str, pd.DataFrame],
        on_ticker_done=None,
    ) -> list:
        """
        Parallel scan of all tickers.

        Each ticker is submitted as an independent task to a ThreadPoolExecutor.
        Results are collected until the configured cycle budget is reached, so
        slow tickers cannot stall the whole dashboard cycle. Any exception
        raised by a single ticker is caught, logged at DEBUG level, and counted
        — it never propagates to the caller.

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols to scan.
        data_1m / data_5m / data_1h / data_1d : dict[str, pd.DataFrame]
            Pre-fetched OHLCV data keyed by ticker symbol.  Missing tickers
            receive an empty DataFrame so analyse_ticker() can gate on that.

        Returns
        -------
        list[StockSignal]
            Sorted by abs(signal.score) descending (highest conviction first).
            Tickers that raise an exception or return None are excluded.
        """
        # Lazy import — avoids circular imports at module load time.
        from agent.scanner import analyse_ticker, StockSignal  # noqa: F401

        _empty = pd.DataFrame()
        t0 = time.perf_counter()
        errors = 0
        timeouts = 0
        results: list = []
        slow_tickers: list[tuple[str, float]] = []
        n_requested = len(tickers)
        n_total = n_requested
        n_done = 0
        try:
            slow_threshold_s = max(
                0.0,
                float(os.getenv("NASDAQ_SCAN_SLOW_TICKER_S", "5.0")),
            )
        except ValueError:
            slow_threshold_s = 5.0
        try:
            from agent.config_manager import config as _cfg
            ticker_timeout_s = float(_cfg.get("scanner.ticker_timeout_s") or
                                     os.getenv("NASDAQ_SCAN_TICKER_TIMEOUT_S", "45"))
            cycle_budget_s = float(_cfg.get("scanner.cycle_budget_s", 20.0) or 20.0)
            slow_cooldown_s = float(_cfg.get("scanner.slow_ticker_cooldown_s", 300.0) or 300.0)
        except Exception:
            ticker_timeout_s = float(os.getenv("NASDAQ_SCAN_TICKER_TIMEOUT_S", "45"))
            cycle_budget_s = float(os.getenv("NASDAQ_SCAN_CYCLE_BUDGET_S", "20"))
            slow_cooldown_s = float(os.getenv("NASDAQ_SCAN_SLOW_TICKER_COOLDOWN_S", "300"))
        cycle_budget_s = max(1.0, min(float(ticker_timeout_s), float(cycle_budget_s)))
        slow_cooldown_s = max(0.0, float(slow_cooldown_s))

        now_s = time.time()
        skipped_slow: list[str] = []
        with _slow_skip_lock:
            for _ticker, _until in list(_slow_skip_until.items()):
                if _until <= now_s:
                    del _slow_skip_until[_ticker]
            tickers_to_scan = []
            for ticker in tickers:
                if _slow_skip_until.get(ticker, 0.0) > now_s:
                    skipped_slow.append(ticker)
                else:
                    tickers_to_scan.append(ticker)
        if skipped_slow:
            logger.warning(
                "ScanPipeline skipping %d ticker(s) on slow cooldown: %s",
                len(skipped_slow), ", ".join(skipped_slow[:10]),
            )
        n_total = len(tickers_to_scan)

        def _run(ticker: str):
            task_t0 = time.perf_counter()
            sig = analyse_ticker(
                ticker,
                df_1m=data_1m.get(ticker),
                df_5m=data_5m.get(ticker, _empty),
                df_1h=data_1h.get(ticker, _empty),
                df_1d=data_1d.get(ticker, _empty),
            )
            return sig, time.perf_counter() - task_t0

        executor = ThreadPoolExecutor(
            max_workers=self.n_workers,
            thread_name_prefix="scan",
        )
        pending = set()
        try:
            future_to_ticker = {
                executor.submit(_run, t): t for t in tickers_to_scan
            }

            pending = set(future_to_ticker)
            deadline = time.perf_counter() + cycle_budget_s

            while pending:
                remaining_s = deadline - time.perf_counter()
                if remaining_s <= 0:
                    break
                done, pending = wait(
                    pending,
                    timeout=min(0.5, remaining_s),
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    ticker = future_to_ticker[future]
                    n_done += 1
                    try:
                        sig, task_elapsed_s = future.result()
                        if slow_threshold_s and task_elapsed_s >= slow_threshold_s:
                            slow_tickers.append((ticker, task_elapsed_s))
                        if sig is not None:
                            results.append(sig)
                            if on_ticker_done is not None:
                                try:
                                    on_ticker_done(sig, n_done, n_total)
                                except Exception:
                                    pass
                    except Exception as exc:
                        errors += 1
                        logger.debug("[%s] scan task raised: %s", ticker, exc)

            if pending:
                timed_out = [future_to_ticker[future] for future in pending]
                for future in pending:
                    future.cancel()
                timeouts = len(timed_out)
                if slow_cooldown_s > 0:
                    until = time.time() + slow_cooldown_s
                    with _slow_skip_lock:
                        for ticker in timed_out:
                            _slow_skip_until[ticker] = until
                logger.warning(
                    "ScanPipeline cycle budget %.1fs reached; timed out %d ticker(s): %s",
                    cycle_budget_s,
                    timeouts,
                    ", ".join(timed_out[:10]),
                )
        finally:
            executor.shutdown(wait=not bool(pending), cancel_futures=True)

        elapsed_s = time.perf_counter() - t0
        elapsed_ms = round(elapsed_s * 1000)

        results.sort(key=lambda s: abs(s.score), reverse=True)

        self._update_metrics(
            cycle_ms=elapsed_ms,
            scan_count=len(results),
            errors=errors,
            timeouts=timeouts,
            skipped_slow=len(skipped_slow),
            n_tickers=n_requested,
            elapsed_s=elapsed_s,
            slow_tickers=slow_tickers,
        )

        if slow_tickers:
            top_slow = sorted(slow_tickers, key=lambda item: item[1], reverse=True)[:10]
            logger.warning(
                "ScanPipeline slow tickers >= %.1fs: %s",
                slow_threshold_s,
                ", ".join(f"{ticker}={seconds:.1f}s" for ticker, seconds in top_slow),
            )

        logger.info(
            "ScanPipeline: %d/%d tickers OK, %d errors, %d timeouts, %d skipped, %d ms "
            "(%.1f tickers/s, workers=%d, budget=%.0fs)",
            len(results), n_requested, errors, timeouts, len(skipped_slow),
            elapsed_ms,
            n_requested / elapsed_s if elapsed_s > 0 else 0,
            self.n_workers, cycle_budget_s,
        )

        return results

    def get_metrics(self) -> dict:
        """Return pipeline performance metrics (thread-safe snapshot)."""
        with self._metrics_lock:
            return dict(self._metrics)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _update_metrics(
        self,
        cycle_ms: int,
        scan_count: int,
        errors: int,
        timeouts: int,
        skipped_slow: int,
        n_tickers: int,
        elapsed_s: float,
        slow_tickers: list[tuple[str, float]] | None = None,
    ) -> None:
        with self._metrics_lock:
            prev_cycles = self._metrics["cycles_completed"]
            prev_avg    = self._metrics["avg_cycle_ms"]

            new_cycles  = prev_cycles + 1
            # Cumulative moving average — avoids storing all cycle times.
            new_avg     = (prev_avg * prev_cycles + cycle_ms) / new_cycles

            self._metrics.update({
                "last_cycle_ms":    cycle_ms,
                "last_scan_count":  scan_count,
                "avg_cycle_ms":     round(new_avg, 1),
                "cycles_completed": new_cycles,
                "errors_last_cycle": errors,
                "timeouts_last_cycle": timeouts,
                "skipped_slow_cooldown": skipped_slow,
                "tickers_per_second": round(
                    n_tickers / elapsed_s if elapsed_s > 0 else 0.0, 1
                ),
                "slow_tickers_last_cycle": [
                    {"ticker": ticker, "elapsed_s": round(seconds, 3)}
                    for ticker, seconds in sorted(
                        slow_tickers or [],
                        key=lambda item: item[1],
                        reverse=True,
                    )[:10]
                ],
            })


# ── Singleton access ───────────────────────────────────────────────────────────

_pipeline: ScanPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline(n_workers: int = 8) -> ScanPipeline:
    """
    Return the module-level ScanPipeline singleton, creating it on first call.

    Thread-safe: multiple callers during startup will not create duplicate
    instances.
    """
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = ScanPipeline(n_workers)
    elif n_workers != _pipeline.n_workers:
        # Live worker-count change (PG config edit). The executor is recreated
        # per scan from self.n_workers, so updating the attribute is enough —
        # the next scan picks up the new concurrency cap without a restart.
        _pipeline.n_workers = n_workers
    return _pipeline
