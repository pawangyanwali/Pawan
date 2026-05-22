"""
Multi-timeframe analysis module.

With the Grow-377 plan we can afford to fetch 5M and 1H data directly from the API
(cached with TTL) rather than resampling from 1M bars.  This gives far better quality
analysis at each timeframe.

Timeframe hierarchy (top-down, professional trader approach):
  1D  → macro trend, market regime              (weight 0.32)
  4H  → intermediate trend, swing structure     (weight 0.22)
  1H  → intraday trend, setup quality           (weight 0.20)
  30M → near-term momentum, entry timing        (weight 0.13)
  15M → entry precision, pattern confirmation   (weight 0.08)
  5M  → scalp entry trigger                     (weight 0.05)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from agent.technical import compute_indicators, score_technical
from agent.price_action import analyze_trend_with_confidence

logger = logging.getLogger(__name__)


# ── Timeframe config ──────────────────────────────────────────────────────────
# (label, source_key, resample_rule_if_derived, min_bars, weight)
#
# source_key values:
#   "1d"  → df_daily  (directly fetched, cached 24h)
#   "1h"  → df_1h     (directly fetched, cached 1h)
#   "5m"  → df_5m     (directly fetched, cached 5min)
#   "1m"  → resample df_1m using resample_rule
#
TIMEFRAME_CONFIG = [
    # label   source  resample_rule  min_bars  weight
    ("1D",   "1d",   None,          10,        0.32),
    ("4H",   "1h",   "4h",          6,         0.22),   # resample 1H → 4H
    ("1H",   "1h",   None,          8,         0.20),
    ("30M",  "5m",   "30min",       8,         0.13),   # resample 5M → 30M
    ("15M",  "5m",   "15min",       10,        0.08),   # resample 5M → 15M
    ("5M",   "5m",   None,          20,        0.05),
    ("1M",   "1m",   None,          20,        0.03),   # live scalp trigger TF
]

# Short-term gate: require these three TFs to agree before firing intraday signals
_GATE_TIMEFRAMES = {"1M", "5M", "15M"}

_NEUTRAL_RESULT = {
    "mtf_score":          0.0,
    "alignment":          "MIXED",
    "bull_count":         0,
    "bear_count":         0,
    "timeframes":         {},
    "short_tf_alignment": "MIXED",
    "mtf_gate_passed":    False,
}

_OHLCV_AGG = {
    "Open":   "first",
    "High":   "max",
    "Low":    "min",
    "Close":  "last",
    "Volume": "sum",
}


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        return df.resample(rule).agg(_OHLCV_AGG).dropna(subset=["Open", "Close"])
    except Exception as e:
        logger.debug(f"resample({rule}) error: {e}")
        return pd.DataFrame()


def _analyze_timeframe(df: pd.DataFrame, label: str, min_bars: int, weight: float) -> dict:
    """Compute trend + technical score for a single timeframe DataFrame."""
    n = len(df) if df is not None else 0

    tech_score = 0.0
    if n >= 30:
        try:
            df_ind   = compute_indicators(df.copy())
            last_row = df_ind.iloc[-1]
            tech_score = float(score_technical(last_row))
        except Exception:
            pass

    trend, trend_prob = "SIDEWAYS", 0.5
    if n >= min_bars:
        try:
            trend, trend_prob = analyze_trend_with_confidence(df, min_rows=min_bars)
        except Exception:
            pass

    trend_dir = 1 if trend == "UPTREND" else (-1 if trend == "DOWNTREND" else 0)
    contribution = trend_dir * trend_prob * 0.70 + tech_score * 0.30

    return {
        "label":         label,
        "trend":         trend,
        "trend_prob":    round(float(trend_prob), 2),
        "score":         round(float(tech_score), 4),
        "_contribution": float(contribution),
        "_weight":       weight,
    }


def _alignment_label(bull: int, bear: int, total: int) -> str:
    if total == 0:
        return "MIXED"
    if bull == bear:
        return "MIXED"
    bull_frac = bull / total
    bear_frac = bear / total
    if bull_frac >= 0.80:
        return "STRONGLY BULLISH"
    if bull_frac >= 0.60:
        return "BULLISH"
    if bear_frac >= 0.80:
        return "STRONGLY BEARISH"
    if bear_frac >= 0.60:
        return "BEARISH"
    if bull_frac >= 0.50:
        return "BULLISH"
    if bear_frac >= 0.50:
        return "BEARISH"
    return "MIXED"


def multi_timeframe_analysis(
    df_1m:    pd.DataFrame,
    df_5m:    pd.DataFrame,
    df_1h:    pd.DataFrame,
    df_daily: pd.DataFrame,
) -> dict:
    """
    Perform multi-timeframe analysis across 6 timeframes.

    Parameters
    ----------
    df_1m    : 1-minute OHLCV (from live scan).
    df_5m    : 5-minute OHLCV (directly fetched, cached 5 min).
    df_1h    : 1-hour OHLCV   (directly fetched, cached 1 h).
    df_daily : Daily OHLCV    (directly fetched, cached 24 h).

    Returns
    -------
    dict with keys:
        mtf_score  : float in [-1, +1]
        alignment  : str  (STRONGLY BULLISH / BULLISH / MIXED / BEARISH / STRONGLY BEARISH)
        bull_count : int
        bear_count : int
        timeframes : dict[label → dict]  per-TF results
    """
    def _valid(df):
        return df is not None and isinstance(df, pd.DataFrame) and not df.empty

    sources = {
        "1d": df_daily if _valid(df_daily) else None,
        "1h": df_1h    if _valid(df_1h)    else None,
        "5m": df_5m    if _valid(df_5m)    else None,
        "1m": df_1m    if _valid(df_1m)    else None,
    }

    if not any(v is not None for v in sources.values()):
        return _NEUTRAL_RESULT.copy()

    mtf_score  = 0.0
    bull_count = 0
    bear_count = 0
    timeframes: dict[str, dict] = {}

    for label, src_key, resample_rule, min_bars, weight in TIMEFRAME_CONFIG:
        try:
            base_df = sources.get(src_key)
            if base_df is None:
                continue

            df_tf = _resample(base_df, resample_rule) if resample_rule else base_df.copy()

            if df_tf is None or df_tf.empty:
                continue

            result     = _analyze_timeframe(df_tf, label, min_bars, weight)
            weighted   = result["_contribution"] * weight
            mtf_score += weighted

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
            logger.warning(f"[MTF] {label}: {exc}")

    mtf_score = round(float(np.clip(mtf_score, -1.0, 1.0)), 4)
    alignment = _alignment_label(bull_count, bear_count, len(timeframes))

    # Short-TF alignment gate: 1M + 5M + 15M must agree
    gate_trends = [
        timeframes[tf]["trend"]
        for tf in _GATE_TIMEFRAMES
        if tf in timeframes
    ]
    if len(gate_trends) == len(_GATE_TIMEFRAMES):
        if all(t == "UPTREND" for t in gate_trends):
            short_tf_alignment = "BULL"
        elif all(t == "DOWNTREND" for t in gate_trends):
            short_tf_alignment = "BEAR"
        else:
            short_tf_alignment = "MIXED"
    else:
        short_tf_alignment = "MIXED"

    return {
        "mtf_score":          mtf_score,
        "alignment":          alignment,
        "bull_count":         bull_count,
        "bear_count":         bear_count,
        "timeframes":         timeframes,
        "short_tf_alignment": short_tf_alignment,
        "mtf_gate_passed":    short_tf_alignment in ("BULL", "BEAR"),
    }
