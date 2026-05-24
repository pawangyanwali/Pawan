"""
Backtest reporter — formats live_backtest stats into structured summaries
suitable for the API and WebSocket broadcast.

Also owns the ML feedback retraining scheduler:
  - After every scan, if ≥ FEEDBACK_MIN_NEW new outcomes have accumulated
    since last retrain, trigger a background retrain with the outcome labels.

OutcomePredictor
----------------
A lightweight XGBoost model trained exclusively on ACTUAL trade win/loss
results (not on historical price-direction labels like the main ML model).

  Features : direction, confidence, session, regime, vwap_event, rsi_zone,
             entry_type, rr_ratio, mtf_alignment  (all already in bt_signals)
  Label    : 1 = WIN,  0 = LOSS or TIMEOUT

This is the true feedback loop: the model learns from its OWN trading
history and feeds that knowledge back into confidence scoring at signal
generation time.  It replaces the simple ±15% lookup table with a learned
probability estimate.
"""
from __future__ import annotations
import json
import logging
import pickle
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
    """
    Build context win-rate calibration table AND train the OutcomePredictor.

    The simple table is the fast fallback.  The OutcomePredictor (trained here)
    is the primary adjuster once enough data exists — it captures non-linear
    interactions between context features that the table misses.
    """
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

    # Train the outcome predictor on real win/loss labels — true feedback loop.
    train_outcome_predictor(df)


def get_calibration() -> dict[str, float]:
    with _cal_lock:
        return dict(_calibration)


# ── OutcomePredictor — true feedback learning from trade outcomes ──────────────

_OUTCOME_MODEL_PATH = Path(__file__).parent.parent / "data" / "outcome_predictor.pkl"

# Categorical columns and the integer encoding maps learned during training.
_OUTCOME_CATS  = ["direction", "session", "regime", "vwap_event", "rsi_zone", "entry_type"]
_OUTCOME_NUMS  = ["confidence", "rr_ratio", "mtf_alignment", "rsi_value"]

_outcome_model:     Optional[object] = None   # trained XGBoost classifier
_outcome_encoders:  dict = {}                  # col → {value: int_code}
_outcome_lock       = threading.Lock()
_outcome_min_rows   = 50                       # don't train until we have this many resolved


def _outcome_encode(df: "pd.DataFrame") -> "pd.DataFrame":
    """Label-encode categoricals using the global encoder map."""
    import numpy as np
    out = df.copy()
    for col in _OUTCOME_CATS:
        if col not in out.columns:
            out[col] = ""
        enc = _outcome_encoders.get(col, {})
        out[col] = out[col].apply(lambda v: enc.get(str(v), -1)).astype(np.int16)
    for col in _OUTCOME_NUMS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype("float32")
    return out[_OUTCOME_CATS + _OUTCOME_NUMS]


def train_outcome_predictor(outcomes_df: "pd.DataFrame") -> bool:
    """
    Train a lightweight XGBoost classifier on actual trade win/loss outcomes.

    This is the true feedback loop — the model learns which feature combinations
    led to real WINs vs LOSSes from its own trading history, then uses that
    knowledge to adjust confidence scores at signal generation time.

    Called automatically by _update_confidence_calibration() when enough data exists.
    Returns True if a model was successfully trained.
    """
    global _outcome_model, _outcome_encoders
    try:
        from xgboost import XGBClassifier
        import numpy as np

        df = outcomes_df.copy()
        if len(df) < _outcome_min_rows:
            return False

        # Build encoder maps from current data
        encoders: dict = {}
        for col in _OUTCOME_CATS:
            if col not in df.columns:
                df[col] = ""
            unique = sorted(df[col].astype(str).unique())
            encoders[col] = {v: i for i, v in enumerate(unique)}
            df[col] = df[col].astype(str).map(encoders[col]).fillna(-1).astype(np.int16)

        for col in _OUTCOME_NUMS:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")

        X = df[_OUTCOME_CATS + _OUTCOME_NUMS].values
        y = df["outcome"].astype(int).values

        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos < 5 or n_neg < 5:
            return False

        scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

        model = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
            n_jobs=2,
        )
        model.fit(X, y)

        with _outcome_lock:
            _outcome_model    = model
            _outcome_encoders = encoders

        # Persist so it survives restarts
        _OUTCOME_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_OUTCOME_MODEL_PATH, "wb") as f:
            pickle.dump({"model": model, "encoders": encoders}, f)

        acc = float((model.predict(X) == y).mean())
        logger.info(
            f"[OutcomePredictor] Trained on {len(y)} outcomes "
            f"({n_pos} wins / {n_neg} losses) — in-sample acc={acc:.1%}"
        )
        return True

    except Exception as e:
        logger.warning(f"[OutcomePredictor] Training failed: {e}")
        return False


def _load_outcome_predictor() -> None:
    """Load persisted outcome predictor from disk at startup."""
    global _outcome_model, _outcome_encoders
    try:
        if _OUTCOME_MODEL_PATH.exists():
            with open(_OUTCOME_MODEL_PATH, "rb") as f:
                saved = pickle.load(f)
            with _outcome_lock:
                _outcome_model    = saved["model"]
                _outcome_encoders = saved["encoders"]
            logger.info("[OutcomePredictor] Loaded from disk.")
    except Exception as e:
        logger.debug(f"[OutcomePredictor] Could not load from disk: {e}")


_load_outcome_predictor()


def predict_outcome_proba(
    direction:    str   = "",
    session:      str   = "",
    regime:       str   = "",
    vwap_event:   str   = "",
    rsi_zone:     str   = "",
    entry_type:   str   = "",
    confidence:   float = 0.0,
    rr_ratio:     float = 0.0,
    mtf_alignment: float = 0.0,
    rsi_value:    float = 50.0,
) -> Optional[float]:
    """
    Return the outcome predictor's win probability [0,1] for these features,
    or None if the model is not yet trained.
    """
    with _outcome_lock:
        model    = _outcome_model
        encoders = dict(_outcome_encoders)
    if model is None:
        return None
    try:
        import numpy as np
        row = []
        for col, val in zip(_OUTCOME_CATS,
                            [direction, session, regime, vwap_event, rsi_zone, entry_type]):
            row.append(encoders.get(col, {}).get(str(val), -1))
        row += [confidence, rr_ratio, mtf_alignment, rsi_value]
        proba = float(model.predict_proba(np.array([row]))[0][1])
        return round(proba, 3)
    except Exception:
        return None


def adjust_confidence(
    confidence:    float,
    vwap_event:    str   = "",
    rsi_zone:      str   = "",
    session:       str   = "",
    regime:        str   = "",
    entry_type:    str   = "",
    direction:     str   = "",
    rr_ratio:      float = 0.0,
    mtf_alignment: float = 0.0,
    rsi_value:     float = 50.0,
) -> float:
    """
    Adjust raw confidence using the trained OutcomePredictor when available,
    falling back to the simple context-win-rate table otherwise.

    OutcomePredictor path (preferred):
      Converts the model's predicted win probability to a confidence shift.
      A predicted 70% win probability → +6 pts; 30% → -6 pts.  Capped at ±20.

    Fallback (simple table):
      Each matching context contributes ±15 pts based on observed win rate.
      Multiple matches are averaged.
    """
    # ── Path 1: learned outcome model ────────────────────────────────────────
    proba = predict_outcome_proba(
        direction=direction, session=session, regime=regime,
        vwap_event=vwap_event, rsi_zone=rsi_zone, entry_type=entry_type,
        confidence=confidence, rr_ratio=rr_ratio,
        mtf_alignment=mtf_alignment, rsi_value=rsi_value,
    )
    if proba is not None:
        # Map predicted win probability to confidence adjustment:
        #   proba=0.55 → +3 pts,  proba=0.30 → -12 pts,  proba=0.70 → +9 pts
        shift = (proba - 0.50) * 40.0           # ±20 pts max
        shift = max(-20.0, min(20.0, shift))
        return round(float(min(max(confidence + shift, 25.0), 95.0)), 1)

    # ── Path 2: simple context-win-rate table (fallback) ─────────────────────
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
            adjustment = (obs_wr * 100 - 50) * 0.30   # ±15 max per context
            adjustments.append(adjustment)

    if adjustments:
        total_adj = sum(adjustments) / len(adjustments)
        confidence = round(float(min(max(confidence + total_adj, 25.0), 95.0)), 1)

    return confidence
