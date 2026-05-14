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
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
            "tickers_per_second": 0.0,
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
    ) -> list:
        """
        Parallel scan of all tickers.

        Each ticker is submitted as an independent task to a ThreadPoolExecutor.
        Results are collected with as_completed() so short tasks don't wait on
        slow ones.  Any exception raised by a single ticker is caught, logged at
        DEBUG level, and counted — it never propagates to the caller.

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
        results: list = []

        def _run(ticker: str):
            return analyse_ticker(
                ticker,
                df_1m=data_1m.get(ticker),
                df_5m=data_5m.get(ticker, _empty),
                df_1h=data_1h.get(ticker, _empty),
                df_1d=data_1d.get(ticker, _empty),
            )

        with ThreadPoolExecutor(max_workers=self.n_workers,
                                thread_name_prefix="scan") as executor:
            future_to_ticker = {
                executor.submit(_run, t): t for t in tickers
            }

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    sig = future.result()
                    if sig is not None:
                        results.append(sig)
                except Exception as exc:
                    errors += 1
                    logger.debug("[%s] scan task raised: %s", ticker, exc)

        elapsed_s = time.perf_counter() - t0
        elapsed_ms = round(elapsed_s * 1000)

        results.sort(key=lambda s: abs(s.score), reverse=True)

        self._update_metrics(
            cycle_ms=elapsed_ms,
            scan_count=len(results),
            errors=errors,
            n_tickers=len(tickers),
            elapsed_s=elapsed_s,
        )

        logger.info(
            "ScanPipeline: %d/%d tickers OK, %d errors, %d ms "
            "(%.1f tickers/s, workers=%d)",
            len(results), len(tickers), errors,
            elapsed_ms,
            len(tickers) / elapsed_s if elapsed_s > 0 else 0,
            self.n_workers,
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
        n_tickers: int,
        elapsed_s: float,
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
                "tickers_per_second": round(
                    n_tickers / elapsed_s if elapsed_s > 0 else 0.0, 1
                ),
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
    return _pipeline
