"""
Walk-forward backtester for NASDAQ agent ML signals.

Architecture
------------
1. Reads resolved signals from the live_backtest SQLite database (bt_signals table).
2. Partitions them into rolling 30-day windows with a 15-day step.
3. For each window, uses the first 20 days as the "training" context and the last
   10 days as the held-out "test" set for win-rate evaluation.
4. Aggregates window results into a BacktestReport with edge detection logic.
5. Persists the report to data/backtest_report.json; loads it back on startup.

Edge criterion: win_rate > 0.53 in more than 60 % of windows that have
at least MIN_SIGNALS_PER_WINDOW signals in the test period.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from agent.db import get_conn

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_PERSIST_PATH = Path(__file__).parent.parent / "data" / "backtest_report.json"

# ── Walk-forward parameters ────────────────────────────────────────────────────
WINDOW_DAYS         = 30    # total days per window
TRAIN_DAYS          = 20    # first N days are "training context" (not evaluated)
TEST_DAYS           = 10    # last M days are the held-out test set
STEP_DAYS           = 15    # slide window by this many days
MIN_SIGNALS_PER_WINDOW = 5  # skip windows with fewer test-period signals
EDGE_WIN_RATE_THRESHOLD = 0.53  # minimum win rate to count a window as "winning"
EDGE_WINDOW_FRACTION    = 0.60  # fraction of windows that must exceed threshold

# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    window_idx:     int
    train_start:    str     # ISO date
    train_end:      str
    test_start:     str
    test_end:       str
    n_signals:      int
    n_wins:         int
    win_rate:       float
    avg_win_pct:    float   # avg gain when correct (positive)
    avg_loss_pct:   float   # avg loss when wrong (negative)
    expectancy:     float   # avg_win * win_rate + avg_loss * (1 - win_rate)
    sharpe:         float   # annualised Sharpe; NaN when insufficient signals
    max_drawdown:   float   # maximum cumulative drawdown (negative)
    tickers_tested: int


@dataclass
class BacktestReport:
    windows:            list[WindowResult]
    overall_win_rate:   float
    overall_expectancy: float
    avg_sharpe:         float
    max_drawdown:       float
    edge_exists:        bool    # True if win_rate > threshold in >60 % of windows
    total_signals:      int
    generated_at:       str     # ISO timestamp
    status:             str     # "complete" | "running" | "error" | "idle"
    progress_pct:       float   # 0–100


# ── Utility helpers ────────────────────────────────────────────────────────────

def _iso_date(dt: datetime) -> str:
    return dt.date().isoformat()


def _compute_sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe ratio from a 1-D array of percentage returns."""
    if len(returns) < 2:
        return float("nan")
    std = float(np.std(returns, ddof=1))
    if std == 0.0:
        return float("nan")
    mean = float(np.mean(returns))
    # Scale: assume each signal resolves in ~1 trading day on average
    # annualise by sqrt(252)
    return float(mean / std * math.sqrt(252))


def _safe_float(v, default: float = 0.0) -> float:
    """Return v as a float, substituting default for NaN/Inf/None."""
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


# ── Database reader ────────────────────────────────────────────────────────────

def _load_resolved_signals() -> pd.DataFrame:
    """
    Read all resolved (non-TRACKING) signals from bt_signals in PostgreSQL.

    Returns an empty DataFrame when the table is empty or unreadable. Never raises.
    """
    try:
        with get_conn() as conn:
            rows = conn.execute("""
                SELECT
                    ticker,
                    direction,
                    entry_price,
                    exit_price,
                    confidence,
                    session,
                    regime,
                    fired_at   AS recorded_at,
                    resolved_at,
                    status     AS outcome,
                    pnl_pct
                FROM bt_signals
                WHERE status IN ('WIN', 'LOSS', 'TIMEOUT')
                ORDER BY fired_at ASC
            """).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows])

        if df.empty:
            return df

        # Parse timestamps; coerce bad values to NaT
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True, errors="coerce")
        df["resolved_at"] = pd.to_datetime(df["resolved_at"], utc=True, errors="coerce")

        # Drop rows with unparseable timestamps or null pnl
        df = df.dropna(subset=["recorded_at"])
        df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce").fillna(0.0)

        logger.debug(f"[Backtester] Loaded {len(df)} resolved signals from DB.")
        return df.reset_index(drop=True)

    except Exception as exc:
        logger.warning(f"[Backtester] DB read failed: {exc}")
        return pd.DataFrame()


# ── Window computation ─────────────────────────────────────────────────────────

def _compute_window(
    window_idx: int,
    train_start: datetime,
    train_end: datetime,
    test_start: datetime,
    test_end: datetime,
    df_test: pd.DataFrame,
) -> WindowResult:
    """
    Compute performance statistics for one walk-forward window.

    df_test must already be filtered to the test period.
    """
    n_signals = len(df_test)

    if n_signals == 0:
        return WindowResult(
            window_idx   = window_idx,
            train_start  = _iso_date(train_start),
            train_end    = _iso_date(train_end),
            test_start   = _iso_date(test_start),
            test_end     = _iso_date(test_end),
            n_signals    = 0,
            n_wins       = 0,
            win_rate     = 0.0,
            avg_win_pct  = 0.0,
            avg_loss_pct = 0.0,
            expectancy   = 0.0,
            sharpe       = float("nan"),
            max_drawdown = 0.0,
            tickers_tested = 0,
        )

    # Win: pnl_pct > 0  (also validate against outcome column when available)
    wins_mask   = df_test["pnl_pct"] > 0
    n_wins      = int(wins_mask.sum())
    win_rate    = n_wins / n_signals

    up_returns  = df_test.loc[wins_mask,   "pnl_pct"]
    dn_returns  = df_test.loc[~wins_mask,  "pnl_pct"]

    avg_win  = _safe_float(up_returns.mean()) if len(up_returns) > 0 else 0.0
    avg_loss = _safe_float(dn_returns.mean()) if len(dn_returns) > 0 else 0.0

    expectancy = avg_win * win_rate + avg_loss * (1.0 - win_rate)

    sharpe = _compute_sharpe(df_test["pnl_pct"].to_numpy())

    # Drawdown on cumulative P&L series
    cum_pnl     = df_test["pnl_pct"].cumsum()
    rolling_max = cum_pnl.cummax()
    max_drawdown = _safe_float((cum_pnl - rolling_max).min(), default=0.0)

    tickers_tested = df_test["ticker"].nunique()

    return WindowResult(
        window_idx   = window_idx,
        train_start  = _iso_date(train_start),
        train_end    = _iso_date(train_end),
        test_start   = _iso_date(test_start),
        test_end     = _iso_date(test_end),
        n_signals    = n_signals,
        n_wins       = n_wins,
        win_rate     = round(win_rate, 4),
        avg_win_pct  = round(avg_win,  4),
        avg_loss_pct = round(avg_loss, 4),
        expectancy   = round(expectancy, 4),
        sharpe       = round(_safe_float(sharpe, float("nan")), 3),
        max_drawdown = round(max_drawdown, 4),
        tickers_tested = tickers_tested,
    )


# ── Persistence ────────────────────────────────────────────────────────────────

def _report_to_dict(report: BacktestReport) -> dict:
    d = asdict(report)
    # Replace NaN floats with None for JSON compatibility
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    return _clean(d)


def _report_from_dict(d: dict) -> BacktestReport:
    windows = [WindowResult(**w) for w in d.get("windows", [])]
    return BacktestReport(
        windows            = windows,
        overall_win_rate   = float(d.get("overall_win_rate",   0.0) or 0.0),
        overall_expectancy = float(d.get("overall_expectancy", 0.0) or 0.0),
        avg_sharpe         = float(d.get("avg_sharpe",         0.0) or 0.0),
        max_drawdown       = float(d.get("max_drawdown",       0.0) or 0.0),
        edge_exists        = bool(d.get("edge_exists", False)),
        total_signals      = int(d.get("total_signals", 0)),
        generated_at       = str(d.get("generated_at", "")),
        status             = str(d.get("status", "idle")),
        progress_pct       = float(d.get("progress_pct", 0.0) or 0.0),
    )


def _save_report(report: BacktestReport, path: Path = _PERSIST_PATH) -> None:
    """Atomic write: write to temp file then rename."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(_report_to_dict(report), indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(data)
            os.replace(tmp, str(path))
        except Exception:
            # Clean up temp file if rename failed
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as exc:
        logger.warning(f"[Backtester] Failed to save report: {exc}")


def _load_report(path: Path = _PERSIST_PATH) -> Optional[BacktestReport]:
    """Load a previously saved report.  Returns None on any error."""
    try:
        if not path.exists():
            return None
        with open(path) as f:
            d = json.load(f)
        report = _report_from_dict(d)
        # Mark as complete so callers know it is a real result
        report.status = "complete"
        logger.info(
            f"[Backtester] Loaded cached report from {path} "
            f"(generated {report.generated_at}, {len(report.windows)} windows)."
        )
        return report
    except Exception as exc:
        logger.warning(f"[Backtester] Failed to load cached report: {exc}")
        return None


# ── Main backtester class ──────────────────────────────────────────────────────

class WalkForwardBacktester:
    """
    Walk-forward validator that reads the live_backtest SQLite database,
    splits resolved signals into rolling time windows, and computes per-window
    win rates, expectancy, Sharpe, and drawdown to determine whether the ML
    signal pipeline has a real statistical edge.

    Usage
    -----
    bt = WalkForwardBacktester()
    bt.run_async()          # non-blocking; background thread updates report
    report = bt.get_report()
    status = bt.get_status()
    """

    def __init__(
        self,
        window_days:      int = WINDOW_DAYS,
        train_days:       int = TRAIN_DAYS,
        step_days:        int = STEP_DAYS,
        min_signals:      int = MIN_SIGNALS_PER_WINDOW,
        persist_path:     Path = _PERSIST_PATH,
    ) -> None:
        self._window_days  = window_days
        self._train_days   = train_days
        self._step_days    = step_days
        self._min_signals  = min_signals
        self._persist_path = persist_path

        self._report:     Optional[BacktestReport] = None
        self._lock        = threading.Lock()
        self._is_running  = False

        self._load_cached()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_async(self) -> None:
        """
        Start the walk-forward backtest in a background daemon thread.
        Non-blocking.  If a run is already in progress, the call is ignored.
        """
        with self._lock:
            if self._is_running:
                logger.debug("[Backtester] Already running — ignoring run_async().")
                return
            self._is_running = True

        thread = threading.Thread(
            target=self._run_safe,
            name="WalkForwardBacktester",
            daemon=True,
        )
        thread.start()
        logger.info("[Backtester] Background run started.")

    def run_sync(self) -> BacktestReport:
        """
        Run the walk-forward backtest on the calling thread.
        Blocks until complete and returns the report.
        """
        with self._lock:
            if self._is_running:
                logger.warning("[Backtester] run_sync() called while already running; waiting.")
        return self._execute()

    def get_report(self) -> Optional[BacktestReport]:
        """Return the latest BacktestReport, or None if not yet available."""
        with self._lock:
            return self._report

    def get_status(self) -> dict:
        """
        Return a compact status dictionary.

        Keys:
          status       – "idle" | "running" | "complete" | "error"
          progress_pct – 0–100
          report_summary – dict or None
        """
        with self._lock:
            report = self._report

        summary = None
        if report is not None:
            summary = {
                "overall_win_rate":   report.overall_win_rate,
                "overall_expectancy": report.overall_expectancy,
                "avg_sharpe":         report.avg_sharpe,
                "max_drawdown":       report.max_drawdown,
                "edge_exists":        report.edge_exists,
                "total_signals":      report.total_signals,
                "n_windows":          len(report.windows),
                "generated_at":       report.generated_at,
            }

        return {
            "status":         report.status if report else "idle",
            "progress_pct":   report.progress_pct if report else 0.0,
            "report_summary": summary,
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_cached(self) -> None:
        """Load last persisted report on construction."""
        cached = _load_report(self._persist_path)
        if cached is not None:
            with self._lock:
                self._report = cached

    def _set_report(self, report: BacktestReport) -> None:
        with self._lock:
            self._report = report

    def _set_progress(self, pct: float, status: str = "running") -> None:
        with self._lock:
            if self._report is not None:
                self._report.progress_pct = round(pct, 1)
                self._report.status = status
            else:
                # Create a placeholder report so callers can poll progress
                self._report = BacktestReport(
                    windows            = [],
                    overall_win_rate   = 0.0,
                    overall_expectancy = 0.0,
                    avg_sharpe         = 0.0,
                    max_drawdown       = 0.0,
                    edge_exists        = False,
                    total_signals      = 0,
                    generated_at       = datetime.now(timezone.utc).isoformat(),
                    status             = status,
                    progress_pct       = round(pct, 1),
                )

    def _run_safe(self) -> None:
        """Wrapper that guarantees _is_running is cleared even on exceptions."""
        try:
            self._execute()
        except Exception as exc:
            logger.error(f"[Backtester] Unhandled error in background run: {exc}", exc_info=True)
            with self._lock:
                if self._report is not None:
                    self._report.status = "error"
                    self._report.progress_pct = 0.0
        finally:
            with self._lock:
                self._is_running = False

    def _execute(self) -> BacktestReport:
        """
        Core walk-forward logic.  Called by both run_sync() and _run_safe().
        """
        logger.info("[Backtester] Starting walk-forward backtest…")
        self._set_progress(0.0, "running")

        # ── 1. Load data ───────────────────────────────────────────────────────
        df = _load_resolved_signals()

        if df.empty:
            logger.info("[Backtester] No resolved signals in DB — returning empty report.")
            report = self._build_empty_report("complete")
            self._set_report(report)
            _save_report(report, self._persist_path)
            return report

        # Ensure chronological order
        df = df.sort_values("recorded_at").reset_index(drop=True)

        # Timezone-aware boundaries
        earliest: datetime = df["recorded_at"].iloc[0].to_pydatetime()
        latest:   datetime = df["recorded_at"].iloc[-1].to_pydatetime()

        # ── 2. Build window schedule ──────────────────────────────────────────
        windows: list[WindowResult] = []
        window_starts: list[datetime] = []

        cursor = earliest.replace(hour=0, minute=0, second=0, microsecond=0)
        while True:
            window_end = cursor + timedelta(days=self._window_days)
            if cursor >= latest:
                break
            window_starts.append(cursor)
            cursor += timedelta(days=self._step_days)

        total_windows = len(window_starts)
        if total_windows == 0:
            logger.info("[Backtester] Not enough date range for even one window.")
            report = self._build_empty_report("complete")
            self._set_report(report)
            _save_report(report, self._persist_path)
            return report

        # ── 3. Process each window ────────────────────────────────────────────
        for idx, w_start in enumerate(window_starts):
            train_start = w_start
            train_end   = w_start + timedelta(days=self._train_days)
            test_start  = train_end
            test_end    = w_start + timedelta(days=self._window_days)

            # Convert to UTC-aware for comparison
            def _ts(dt: datetime) -> pd.Timestamp:
                return pd.Timestamp(dt).tz_localize("UTC") if dt.tzinfo is None else pd.Timestamp(dt)

            ts_test_start = _ts(test_start)
            ts_test_end   = _ts(test_end)

            df_test = df[
                (df["recorded_at"] >= ts_test_start) &
                (df["recorded_at"] <  ts_test_end)
            ].copy()

            if len(df_test) < self._min_signals:
                # Skip windows with insufficient data but still record a null result
                windows.append(WindowResult(
                    window_idx     = idx,
                    train_start    = _iso_date(train_start),
                    train_end      = _iso_date(train_end),
                    test_start     = _iso_date(test_start),
                    test_end       = _iso_date(test_end),
                    n_signals      = len(df_test),
                    n_wins         = 0,
                    win_rate       = 0.0,
                    avg_win_pct    = 0.0,
                    avg_loss_pct   = 0.0,
                    expectancy     = 0.0,
                    sharpe         = float("nan"),
                    max_drawdown   = 0.0,
                    tickers_tested = df_test["ticker"].nunique() if not df_test.empty else 0,
                ))
            else:
                result = _compute_window(
                    window_idx  = idx,
                    train_start = train_start,
                    train_end   = train_end,
                    test_start  = test_start,
                    test_end    = test_end,
                    df_test     = df_test,
                )
                windows.append(result)

            progress = (idx + 1) / total_windows * 95.0   # reserve last 5% for aggregation
            self._set_progress(progress, "running")

        # ── 4. Aggregate across windows ────────────────────────────────────────
        self._set_progress(95.0, "running")
        report = self._aggregate(windows, total_signals=len(df))

        # ── 5. Persist and return ──────────────────────────────────────────────
        _save_report(report, self._persist_path)
        self._set_report(report)

        logger.info(
            f"[Backtester] Complete. {len(windows)} windows | "
            f"win_rate={report.overall_win_rate:.2%} | "
            f"edge_exists={report.edge_exists} | "
            f"total_signals={report.total_signals}"
        )
        return report

    def _aggregate(
        self,
        windows:       list[WindowResult],
        total_signals: int,
    ) -> BacktestReport:
        """Compute aggregate metrics across all walk-forward windows."""

        # Only consider windows that had enough signals for the test period
        qualifying = [w for w in windows if w.n_signals >= self._min_signals]

        if not qualifying:
            return self._build_empty_report("complete", total_signals=total_signals)

        # Overall win rate: weighted by number of signals
        total_wins    = sum(w.n_wins    for w in qualifying)
        total_test    = sum(w.n_signals for w in qualifying)
        overall_win_rate = total_wins / total_test if total_test > 0 else 0.0

        # Overall expectancy: simple mean across windows
        overall_expectancy = float(np.mean([w.expectancy for w in qualifying]))

        # Average Sharpe: ignore NaN windows
        valid_sharpes = [w.sharpe for w in qualifying if not math.isnan(w.sharpe)]
        avg_sharpe = float(np.mean(valid_sharpes)) if valid_sharpes else float("nan")

        # Max drawdown: worst single window
        max_drawdown = float(min(w.max_drawdown for w in qualifying))

        # Edge detection: fraction of qualifying windows with win_rate > threshold
        winning_windows = sum(
            1 for w in qualifying if w.win_rate > EDGE_WIN_RATE_THRESHOLD
        )
        edge_fraction = winning_windows / len(qualifying) if qualifying else 0.0
        edge_exists   = edge_fraction > EDGE_WINDOW_FRACTION

        return BacktestReport(
            windows            = windows,
            overall_win_rate   = round(overall_win_rate,   4),
            overall_expectancy = round(overall_expectancy, 4),
            avg_sharpe         = round(_safe_float(avg_sharpe, 0.0), 3),
            max_drawdown       = round(max_drawdown,       4),
            edge_exists        = edge_exists,
            total_signals      = total_signals,
            generated_at       = datetime.now(timezone.utc).isoformat(),
            status             = "complete",
            progress_pct       = 100.0,
        )

    @staticmethod
    def _build_empty_report(status: str, total_signals: int = 0) -> BacktestReport:
        return BacktestReport(
            windows            = [],
            overall_win_rate   = 0.0,
            overall_expectancy = 0.0,
            avg_sharpe         = 0.0,
            max_drawdown       = 0.0,
            edge_exists        = False,
            total_signals      = total_signals,
            generated_at       = datetime.now(timezone.utc).isoformat(),
            status             = status,
            progress_pct       = 100.0 if status == "complete" else 0.0,
        )


# ── Singleton ──────────────────────────────────────────────────────────────────

_backtester: Optional[WalkForwardBacktester] = None
_singleton_lock = threading.Lock()


def get_backtester() -> WalkForwardBacktester:
    """Return the module-level singleton WalkForwardBacktester, creating it on first call."""
    global _backtester
    if _backtester is None:
        with _singleton_lock:
            if _backtester is None:
                _backtester = WalkForwardBacktester()
    return _backtester
