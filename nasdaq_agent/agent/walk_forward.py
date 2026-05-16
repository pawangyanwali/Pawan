"""
Walk-forward signal replay engine — generates clean, labeled training records
from historical OHLCV data WITHOUT lookahead bias.

The three guarantees of "clean data":

1. No lookahead: features at bar[i] are computed using only bars[0..i].
   compute_features() in feature_engine.py is fully causal — every indicator
   uses only past data at each row.

2. Outcome labels match actual trade logic: entry at bar[i] close, target at
   +ATR*atr_target_mult, stop at -ATR*atr_stop_mult.  We walk forward bar by
   bar checking high/low — whichever triggers first wins.  Time-stop fires at
   max_bars_held bars if neither level is hit.

3. Feature consistency: uses the exact same compute_features() that runs
   during live inference so train/inference distributions are identical.

Signal detection: rule-based (RSI thresholds) — not model-based — so there
is no circular reference between signal generation and model training.

    BUY  when RSI-14 < 35 AND MACD histogram > 0 AND vol_ratio > 1.0
    SELL when RSI-14 > 65 AND MACD histogram < 0 AND vol_ratio > 1.0

Public API
----------
replay_signals(ticker, df, ...) → list[dict]
    Replay a historical OHLCV DataFrame and return labeled signal records.

build_training_df(records) → pd.DataFrame
    Convert raw records to a feature-matrix DataFrame ready for XGBoost.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from agent.feature_engine import FEATURE_COLS_V2, compute_features

logger = logging.getLogger(__name__)

# Session time-of-day filters (hour, minute in ET)
_TRADEABLE_START = (9, 45)   # PRIME opens
_TRADEABLE_END   = (15, 30)  # STANDARD closes → CLOSING_CAUTION


def _is_tradeable_bar(idx: pd.Timestamp) -> bool:
    """True if bar timestamp falls in a regular-session window (ET)."""
    import pytz
    try:
        et = idx.tz_convert("America/New_York") if idx.tzinfo else idx
        h, m = et.hour, et.minute
        after_open  = (h, m) >= _TRADEABLE_START
        before_close = (h, m) <  _TRADEABLE_END
        return after_open and before_close and et.weekday() < 5
    except Exception:
        return True   # can't determine — allow


def replay_signals(
    ticker:           str,
    df:               pd.DataFrame,
    atr_target_mult:  float = 1.5,
    atr_stop_mult:    float = 1.0,
    max_bars_held:    int   = 8,
    rsi_buy_thresh:   float = 35.0,
    rsi_sell_thresh:  float = 65.0,
    cooldown_bars:    int   = 8,
    min_vol_ratio:    float = 1.0,
    filter_session:   bool  = True,
    timeframe:        str   = "5min",
) -> list[dict[str, Any]]:
    """
    Replay historical bars and return a list of labeled signal records.

    Each record contains:
        ticker, bar_dt, direction, entry_price, target, stop,
        outcome (HIT_TARGET | HIT_STOP | TIME_STOP),
        pnl_r (profit in units of R — +1.5, -1.0, or time-exit pnl),
        bars_held,
        + all 32 FEATURE_COLS_V2 values at entry bar
    """
    if df is None or len(df) < 60:
        return []

    # Normalise column names to lowercase
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        logger.debug(f"[WalkFwd] {ticker} missing OHLCV columns — skipping")
        return []

    # ── Step 1: compute ALL features causally on the full history ────────────
    try:
        df_feat = compute_features(df, ticker)
    except Exception as exc:
        logger.warning(f"[WalkFwd] {ticker} feature computation failed: {exc}")
        return []

    # Drop rows where any feature is NaN (early bars without enough lookback)
    df_feat = df_feat.dropna(subset=FEATURE_COLS_V2)
    if len(df_feat) < max_bars_held + 2:
        return []

    records: list[dict[str, Any]] = []
    last_signal_bar = -cooldown_bars - 1

    for i, (idx, row) in enumerate(df_feat.iterrows()):
        # ── Signal cooldown ──────────────────────────────────────────────────
        if i - last_signal_bar < cooldown_bars:
            continue

        # ── Session filter ───────────────────────────────────────────────────
        if filter_session and not _is_tradeable_bar(idx):
            continue

        # ── Feature values for this bar ──────────────────────────────────────
        rsi     = float(row.get("rsi_14", 50.0))
        macd_h  = float(row.get("macd_hist", 0.0))
        vol_r   = float(row.get("vol_ratio", 1.0))
        atr     = float(row.get("atr_14", 0.0))

        if atr <= 0:
            continue

        # ── Signal detection (rule-based, no model circular reference) ───────
        if rsi < rsi_buy_thresh and macd_h > 0 and vol_r >= min_vol_ratio:
            direction = "BUY"
        elif rsi > rsi_sell_thresh and macd_h < 0 and vol_r >= min_vol_ratio:
            direction = "SELL"
        else:
            continue

        # ── Entry levels ─────────────────────────────────────────────────────
        entry = float(row.get("Close", 0.0))
        if entry <= 0:
            continue

        if direction == "BUY":
            target = round(entry + atr * atr_target_mult, 4)
            stop   = round(entry - atr * atr_stop_mult,   4)
        else:
            target = round(entry - atr * atr_target_mult, 4)
            stop   = round(entry + atr * atr_stop_mult,   4)

        # ── Outcome simulation: walk forward bar by bar ──────────────────────
        future_rows = df_feat.iloc[i + 1 : i + 1 + max_bars_held]
        outcome    = "TIME_STOP"
        bars_held  = len(future_rows)
        exit_price = entry

        for j, (_, frow) in enumerate(future_rows.iterrows(), start=1):
            hi = float(frow.get("High", entry))
            lo = float(frow.get("Low",  entry))
            if direction == "BUY":
                if hi >= target:
                    outcome   = "HIT_TARGET"
                    exit_price = target
                    bars_held  = j
                    break
                if lo <= stop:
                    outcome   = "HIT_STOP"
                    exit_price = stop
                    bars_held  = j
                    break
            else:  # SELL
                if lo <= target:
                    outcome   = "HIT_TARGET"
                    exit_price = target
                    bars_held  = j
                    break
                if hi >= stop:
                    outcome   = "HIT_STOP"
                    exit_price = stop
                    bars_held  = j
                    break

        if outcome == "TIME_STOP" and not future_rows.empty:
            exit_price = float(future_rows.iloc[-1].get("close", entry))

        # pnl in units of R (risk = entry → stop distance)
        risk = abs(entry - stop)
        if risk > 0:
            raw_pnl = (exit_price - entry) if direction == "BUY" else (entry - exit_price)
            pnl_r   = round(raw_pnl / risk, 3)
        else:
            pnl_r = 0.0

        last_signal_bar = i

        record: dict[str, Any] = {
            "ticker":      ticker,
            "timeframe":   timeframe,
            "bar_dt":      str(idx),
            "direction":   direction,
            "entry_price": round(entry, 4),
            "target":      target,
            "stop":        stop,
            "exit_price":  round(exit_price, 4),
            "outcome":     outcome,
            "pnl_r":       pnl_r,
            "bars_held":   bars_held,
            "won":         outcome == "HIT_TARGET",
        }
        # Attach all feature values for XGBoost retraining
        for col in FEATURE_COLS_V2:
            record[col] = float(row.get(col, 0.0))

        records.append(record)

    logger.debug(f"[WalkFwd] {ticker}: {len(records)} signals replayed from {len(df_feat)} bars")
    return records


def build_training_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert raw walk-forward records to a feature+label DataFrame.

    X = FEATURE_COLS_V2 columns
    y = 1 (HIT_TARGET) or 0 (HIT_STOP / TIME_STOP)

    Ready to pass directly to XGBClassifier.fit(X, y).
    """
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    # Keep only feature columns + outcome
    keep = FEATURE_COLS_V2 + ["outcome", "won", "ticker", "bar_dt", "pnl_r"]
    df = df[[c for c in keep if c in df.columns]]
    return df


def compute_win_rate_by_context(records: list[dict[str, Any]]) -> dict[str, dict]:
    """
    Compute win rates broken down by direction, RSI zone, and MACD state.
    Used to feed the adaptive filter calibration.
    """
    if not records:
        return {}

    df = pd.DataFrame(records)
    stats: dict[str, dict] = {}

    def _wr(subset):
        if len(subset) == 0:
            return None
        wins = subset["won"].sum()
        return {"win_rate": round(wins / len(subset), 3), "count": len(subset)}

    for direction in ("BUY", "SELL"):
        sub = df[df["direction"] == direction]
        if len(sub) >= 5:
            stats[f"direction:{direction}"] = _wr(sub)

    # RSI zones
    if "rsi_14" in df.columns:
        stats["rsi_zone:OVERSOLD"]  = _wr(df[df["rsi_14"] < 35])
        stats["rsi_zone:OVERBOUGHT"] = _wr(df[df["rsi_14"] > 65])

    return {k: v for k, v in stats.items() if v is not None}
