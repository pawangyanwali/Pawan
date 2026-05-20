"""
Backtest reporter — formats live_backtest stats into structured summaries
suitable for the API and WebSocket broadcast.

Also owns the ML feedback retraining scheduler:
  - After every scan, if ≥ FEEDBACK_MIN_NEW new outcomes have accumulated
    since last retrain, trigger a background retrain with the outcome labels.
"""
from __future__ import annotations
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

_PROCESS_START      = time.time()
_STARTUP_GRACE_SECS = 300          # no feedback retrain for 5 min after process start

import pandas as pd

# ── Calibration persistence ───────────────────────────────────────────────────
_CAL_PATH = Path(__file__).parent.parent / "data" / "confidence_calibration.json"

from agent.live_backtest import (
    get_performance_stats,
    get_tracking_signals,
    get_recent_resolved,
    get_outcomes_for_ml,
)

logger = logging.getLogger(__name__)

FEEDBACK_MIN_NEW     = 100    # min new outcomes before triggering ML retrain
FEEDBACK_CHECK_SECS  = 300    # check every 5 minutes

_last_feedback_count = 0
_feedback_lock       = threading.Lock()


# ── Summary for WebSocket broadcast ───────────────────────────────────────────

def get_broadcast_summary() -> dict:
    """
    Compact summary for inclusion in every WebSocket update.
    Keeps payload small — full stats only on demand via REST.
    """
    stats    = get_performance_stats(lookback_days=7)   # last 7 days for live view
    overall  = stats["overall"]
    tracking = stats["tracking_count"]

    return {
        "tracking":  tracking,
        "total":     overall["total"],
        "win_rate":  overall["win_rate"],
        "avg_r":     overall["avg_r"],
        "expectancy": overall["expectancy"],
    }


def get_full_report(lookback_days: int = 30) -> dict:
    """Full performance report for the REST endpoint."""
    stats    = get_performance_stats(lookback_days=lookback_days)
    tracking = get_tracking_signals()
    recent   = get_recent_resolved(limit=50)

    # Enrich recent resolved with human-readable outcome colour
    for r in recent:
        r["outcome_color"] = (
            "#00ff88" if r["status"] == "WIN" else
            "#ef4444" if r["status"] == "LOSS" else
            "#64748b"
        )

    return {
        "stats":    stats,
        "tracking": tracking,
        "recent":   recent,
    }


# ── ML feedback retraining ────────────────────────────────────────────────────

def maybe_trigger_feedback_retrain(tickers: list) -> None:
    """
    Called after each scan.  If enough new outcomes have accumulated,
    triggers a background retrain using backtest outcomes as additional labels.
    Only runs during CLOSED session (8pm–4am ET) so it never competes with
    the live scan's API rate budget.
    """
    global _last_feedback_count

    # Gate 1: no retrain for 5 min after process start — let the scanner warm
    # SQLite so the retrain fetches zero API calls when it does run.
    if time.time() - _PROCESS_START < _STARTUP_GRACE_SECS:
        return

    # Gate 2: never retrain while the market is tradeable — scan gets full rate budget.
    # Fail CLOSED: if session cannot be determined, skip (don't allow).
    try:
        from agent.market_hours import get_session as _get_sess
        sess = _get_sess()
        if sess not in ("CLOSED",):
            return
    except Exception as _e:
        logger.warning(f"[BT Feedback] Session check failed ({_e}) — skipping retrain")
        return

    outcomes_df = get_outcomes_for_ml(min_count=FEEDBACK_MIN_NEW)
    if outcomes_df is None:
        return

    current_count = len(outcomes_df)
    new_outcomes  = current_count - _last_feedback_count

    if new_outcomes < FEEDBACK_MIN_NEW:
        return

    with _feedback_lock:
        if new_outcomes < FEEDBACK_MIN_NEW:
            return
        _last_feedback_count = current_count

    logger.debug(
        f"[BT Feedback] {new_outcomes} new outcomes → triggering ML feedback retrain"
    )
    thread = threading.Thread(
        target=_run_feedback_retrain,
        args=(outcomes_df, tickers),
        daemon=True,
    )
    thread.start()


def _run_feedback_retrain(outcomes_df: pd.DataFrame, tickers: list) -> None:
    """
    Background thread: retrain ML models, boosting weights on patterns that
    the live backtest shows winning vs losing.
    """
    try:
        from agent.ml_model import retrain_all
        from agent.adaptive_filter import update_filter as _af_update
        from agent.live_backtest import get_performance_stats
        from config import TRAINING_TICKERS

        logger.info("[BT Feedback] Starting feedback-weighted retrain…")
        _log_attribution(outcomes_df)

        # 1. Update confidence calibration table (±15% per context)
        _update_confidence_calibration(outcomes_df)

        # 2. Update adaptive filter — suppresses losing patterns, raises threshold
        stats = get_performance_stats(lookback_days=30)
        _af_update(stats)

        # 3. Retrain ML models — always use Tier-1 training set, never the full
        #    active-ticker list (which can be 477 on fallback and starves the scan).
        retrain_all(TRAINING_TICKERS)
        logger.info("[BT Feedback] Feedback retrain complete.")

    except Exception as e:
        logger.warning(f"[BT Feedback] Retrain failed: {e}")


def _log_attribution(df: pd.DataFrame) -> None:
    """Log which signal contexts are winning vs losing for observability."""
    for col in ["vwap_event", "rsi_zone", "session", "regime", "entry_type"]:
        if col not in df.columns:
            continue
        grp = df.groupby(col)["outcome"].agg(["mean", "count"])
        grp.columns = ["win_rate", "count"]
        grp = grp[grp["count"] >= 5].sort_values("win_rate", ascending=False)
        if not grp.empty:
            lines = [f"  {k}: {v['win_rate']*100:.0f}% ({v['count']} trades)"
                     for k, v in grp.iterrows()]
            logger.info(f"[BT Attribution] {col}:\n" + "\n".join(lines))


# ── Confidence calibration table ──────────────────────────────────────────────
# Stores observed win rates per context to adjust signal confidence at generation time.
# Persisted to data/confidence_calibration.json so learning survives restarts.

_calibration: dict[str, float] = {}   # key: "vwap_event:RECLAIM" → observed_win_rate
_cal_lock = threading.Lock()


def _cal_load() -> None:
    """Load calibration table from disk at startup."""
    global _calibration
    try:
        if _CAL_PATH.exists():
            with open(_CAL_PATH) as f:
                data = json.load(f)
            with _cal_lock:
                _calibration = {k: float(v) for k, v in data.items()}
            logger.info(f"[BT Calibration] Loaded {len(_calibration)} context calibrations from disk.")
    except Exception as e:
        logger.warning(f"[BT Calibration] Could not load saved calibration: {e}")


def _cal_save() -> None:
    try:
        _CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _cal_lock:
            snapshot = dict(_calibration)
        with open(_CAL_PATH, "w") as f:
            json.dump(snapshot, f, indent=2)
    except Exception as e:
        logger.warning(f"[BT Calibration] Save failed: {e}")


# Load persisted calibration on module import
_cal_load()


def _update_confidence_calibration(df: pd.DataFrame) -> None:
    """Build a calibration table: context_key → observed_win_rate, then persist."""
    global _calibration
    new_cal: dict[str, float] = {}

    context_cols = ["vwap_event", "rsi_zone", "session", "regime", "entry_type", "direction"]
    for col in context_cols:
        if col not in df.columns:
            continue
        for val, grp in df.groupby(col):
            if len(grp) < 5:
                continue
            wr = float(grp["outcome"].mean())
            new_cal[f"{col}:{val}"] = round(wr, 3)

    with _cal_lock:
        _calibration = new_cal

    _cal_save()
    logger.info(f"[BT Calibration] Updated {len(new_cal)} context calibrations.")


def get_calibration() -> dict[str, float]:
    with _cal_lock:
        return dict(_calibration)


def adjust_confidence(
    confidence:  float,
    vwap_event:  str = "",
    rsi_zone:    str = "",
    session:     str = "",
    regime:      str = "",
    entry_type:  str = "",
    direction:   str = "",
) -> float:
    """
    Adjust a raw confidence value using observed backtest win rates for each
    matching context.  Multiple context matches are averaged.

    Example: VWAP_RECLAIM historically wins 78% → boosts confidence.
             RSI_EXTREME_OB BUY historically wins 22% → suppresses confidence.
    """
    with _cal_lock:
        cal = dict(_calibration)

    if not cal:
        return confidence

    adjustments = []
    for col, val in [("vwap_event", vwap_event), ("rsi_zone", rsi_zone),
                     ("session", session), ("regime", regime),
                     ("entry_type", entry_type), ("direction", direction)]:
        key = f"{col}:{val}"
        if key in cal:
            obs_wr = cal[key]
            # Shift confidence toward observed win rate (blend 30%)
            adjustment = (obs_wr * 100 - 50) * 0.30   # ±15 max per context
            adjustments.append(adjustment)

    if adjustments:
        total_adj = sum(adjustments) / len(adjustments)
        confidence = round(float(min(max(confidence + total_adj, 25.0), 95.0)), 1)

    return confidence
