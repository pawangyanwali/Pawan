"""
Multi-timeframe analysis module.

Resamples 1-minute OHLCV data to 5M, 15M, 30M, 1H and combines with daily
bars to produce a composite directional score and alignment label.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from agent.technical import compute_indicators, score_technical
from agent.price_action import analyze_trend_with_confidence

logger = logging.getLogger(__name__)

# (label, pandas_rule_or_None_for_daily, min_bars_needed, weight_in_composite)
TIMEFRAME_CONFIG = [
    ("1D",  None,    10, 0.35),
    ("1H",  "60min",  6, 0.25),
    ("30M", "30min",  8, 0.20),
    ("15M", "15min", 12, 0.12),
    ("5M",  "5min",  20, 0.08),
]

_NEUTRAL_RESULT = {
    "mtf_score": 0.0,
    "alignment": "MIXED",
    "bull_count": 0,
    "bear_count": 0,
    "timeframes": {},
}

_OHLCV_AGG = {
    "Open":   "first",
    "High":   "max",
    "Low":    "min",
    "Close":  "last",
    "Volume": "sum",
}


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an OHLCV DataFrame using the given pandas offset alias."""
    resampled = df.resample(rule).agg(_OHLCV_AGG).dropna()
    return resampled


def _analyze_timeframe(
    df: pd.DataFrame,
    label: str,
    min_bars: int,
    weight: float,
) -> dict:
    """Run trend + technical analysis on a single timeframe DataFrame."""
    n = len(df)

    if n >= 30:
        try:
            df_ind = compute_indicators(df.copy())
            last_row = df_ind.iloc[-1]
            tech_score = score_technical(last_row)
        except Exception:
            tech_score = 0.0
    else:
        tech_score = 0.0

    if n >= min_bars:
        try:
            trend, trend_prob = analyze_trend_with_confidence(df)
        except Exception:
            trend, trend_prob = "SIDEWAYS", 0.5
    else:
        trend, trend_prob = "SIDEWAYS", 0.5

    if trend == "UPTREND":
        trend_dir = 1
    elif trend == "DOWNTREND":
        trend_dir = -1
    else:
        trend_dir = 0

    tf_contribution = trend_dir * trend_prob * 0.7 + tech_score * 0.3

    return {
        "label":          label,
        "trend":          trend,
        "trend_prob":     trend_prob,
        "score":          round(weight * tf_contribution, 4),
        "_contribution":  tf_contribution,
    }


def _alignment_label(bull_count: int, bear_count: int) -> str:
    if bull_count >= 4:
        return "STRONGLY BULLISH"
    if bull_count == 3:
        return "BULLISH"
    if bear_count >= 4:
        return "STRONGLY BEARISH"
    if bear_count == 3:
        return "BEARISH"
    return "MIXED"


def multi_timeframe_analysis(
    df_1m: pd.DataFrame,
    df_daily: pd.DataFrame,
) -> dict:
    """
    Perform multi-timeframe analysis across 5M, 15M, 30M, 1H, and 1D.

    Parameters
    ----------
    df_1m   : 1-minute OHLCV DataFrame (DatetimeIndex required).
    df_daily: Daily OHLCV DataFrame (DatetimeIndex required, already cached).

    Returns
    -------
    dict with keys:
        mtf_score : float in [-1, +1]
        alignment : str  (STRONGLY BULLISH / BULLISH / MIXED / BEARISH / STRONGLY BEARISH)
        bull_count: int
        bear_count: int
        timeframes: dict[str, dict] — per-TF results keyed by label
    """
    df_1m_valid = (
        df_1m is not None
        and isinstance(df_1m, pd.DataFrame)
        and not df_1m.empty
    )
    df_daily_valid = (
        df_daily is not None
        and isinstance(df_daily, pd.DataFrame)
        and not df_daily.empty
    )

    if not df_1m_valid and not df_daily_valid:
        return _NEUTRAL_RESULT.copy()

    mtf_score = 0.0
    bull_count = 0
    bear_count = 0
    timeframes: dict[str, dict] = {}

    for label, rule, min_bars, weight in TIMEFRAME_CONFIG:
        try:
            if rule is None:
                if not df_daily_valid:
                    logger.debug("[MTF] %s: daily data unavailable, skipping", label)
                    continue
                df_tf = df_daily.copy()
            else:
                if not df_1m_valid:
                    logger.debug("[MTF] %s: 1m data unavailable, skipping", label)
                    continue
                df_tf = _resample(df_1m, rule)

            if df_tf.empty:
                logger.debug("[MTF] %s: empty after resample, skipping", label)
                continue

            result = _analyze_timeframe(df_tf, label, min_bars, weight)
            mtf_score += result["score"]

            if result["trend"] == "UPTREND":
                bull_count += 1
            elif result["trend"] == "DOWNTREND":
                bear_count += 1

            timeframes[label] = {
                "label":      result["label"],
                "trend":      result["trend"],
                "trend_prob": result["trend_prob"],
                "score":      result["score"],
            }

        except Exception as exc:
            logger.warning("[MTF] %s: analysis error — %s", label, exc)

    mtf_score = float(np.clip(mtf_score, -1.0, 1.0))
    alignment = _alignment_label(bull_count, bear_count)

    return {
        "mtf_score":  round(mtf_score, 4),
        "alignment":  alignment,
        "bull_count": bull_count,
        "bear_count": bear_count,
        "timeframes": timeframes,
    }
