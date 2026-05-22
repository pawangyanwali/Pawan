"""
Gap analysis at the open.

Detects gap-up / gap-down vs prior day's close and estimates fill probability
based on gap size and current price position relative to the gap.

Definitions
-----------
gap_pct   : (today_open - prior_close) / prior_close × 100
GAP_UP    : gap_pct ≥ +0.5%
GAP_DOWN  : gap_pct ≤ -0.5%
FLAT      : |gap_pct| < 0.5%

Fill probability heuristic
--------------------------
Small gap  (<1%)  → 70% fill probability
Medium gap (1-3%) → 50% fill probability
Large gap  (>3%)  → 30% fill probability
Already filled    → 100%
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SMALL_GAP  = 1.0   # %
_MEDIUM_GAP = 3.0   # %


def _fill_probability(gap_pct: float, current_price: float,
                      prior_close: float, today_open: float) -> float:
    """Estimate probability (0-1) the gap fills by end of day."""
    abs_gap = abs(gap_pct)

    # Already filled?
    if gap_pct > 0 and current_price <= prior_close:
        return 1.0
    if gap_pct < 0 and current_price >= prior_close:
        return 1.0

    if abs_gap < _SMALL_GAP:
        return 0.70
    elif abs_gap < _MEDIUM_GAP:
        return 0.50
    else:
        return 0.30


def analyse_gap(df_1m: pd.DataFrame, df_1d: pd.DataFrame) -> dict:
    """
    Analyse the opening gap for a ticker.

    Parameters
    ----------
    df_1m : intraday 1-minute bars (today)
    df_1d : daily bars (at least 2 rows — yesterday + today or just recent history)

    Returns dict:
      gap_type         : str  — GAP_UP | GAP_DOWN | FLAT
      gap_pct          : float
      prior_close      : float
      today_open       : float
      fill_probability : float
      gap_filled       : bool
      premarket_high   : float  — highest 1M close before 09:30 ET (0 if unavailable)
      premarket_low    : float
      description      : str
    """
    result = {
        "gap_type":         "FLAT",
        "gap_pct":          0.0,
        "gap_score":        0.0,
        "prior_close":      0.0,
        "today_open":       0.0,
        "fill_probability": 0.0,
        "gap_filled":       False,
        "premarket_high":   0.0,
        "premarket_low":    0.0,
        "description":      "",
    }

    try:
        if df_1d is None or len(df_1d) < 2:
            return result
        if df_1m is None or df_1m.empty:
            return result

        prior_close = float(df_1d["Close"].iloc[-2])
        current     = float(df_1m["Close"].iloc[-1])

        if prior_close <= 0:
            return result

        # Classify pre-market bars (before 09:30 ET) and regular-session bars
        pm_bars = pd.DataFrame()
        mkt_bars = df_1m.copy()
        try:
            import pytz
            et = pytz.timezone("America/New_York")
            # Always localize tz-naive index to UTC first, then convert to ET
            idx = df_1m.index
            if idx.tzinfo is None:
                idx = idx.tz_localize("UTC")
            idx_et = idx.tz_convert(et)
            pm_mask = pd.Series(
                [(t.hour < 9 or (t.hour == 9 and t.minute < 30)) for t in idx_et],
                index=df_1m.index,
            )
            pm_bars  = df_1m[pm_mask.values]
            mkt_bars = df_1m[~pm_mask.values]
        except Exception:
            pass

        # today_open is the first regular-session bar (09:30 ET), not a pre-market bar
        today_open = float(mkt_bars["Open"].iloc[0]) if not mkt_bars.empty else float(df_1m["Open"].iloc[0])
        gap_pct = (today_open - prior_close) / prior_close * 100

        pm_high = float(pm_bars["High"].max()) if not pm_bars.empty else 0.0
        pm_low  = float(pm_bars["Low"].min())  if not pm_bars.empty else 0.0
        result["premarket_high"] = round(pm_high, 4)
        result["premarket_low"]  = round(pm_low,  4)

        if gap_pct >= 0.5:
            gap_type    = "GAP_UP"
            gap_filled  = current <= prior_close
        elif gap_pct <= -0.5:
            gap_type    = "GAP_DOWN"
            gap_filled  = current >= prior_close
        else:
            gap_type   = "FLAT"
            gap_filled = False

        fill_prob = _fill_probability(gap_pct, current, prior_close, today_open)

        if gap_type == "GAP_UP":
            desc = (f"Gap UP {gap_pct:+.2f}% (prior close ${prior_close:.2f} → open ${today_open:.2f}). "
                    f"Fill prob: {fill_prob*100:.0f}%.")
        elif gap_type == "GAP_DOWN":
            desc = (f"Gap DOWN {gap_pct:+.2f}% (prior close ${prior_close:.2f} → open ${today_open:.2f}). "
                    f"Fill prob: {fill_prob*100:.0f}%.")
        else:
            desc = f"Flat open (gap {gap_pct:+.2f}%)."

        abs_gap = abs(gap_pct)
        if gap_type == "GAP_UP":
            if abs_gap >= 3.0:
                gap_score = 0.80
            elif abs_gap >= 1.0:
                gap_score = 0.50
            else:
                gap_score = 0.25
        elif gap_type == "GAP_DOWN":
            if abs_gap >= 3.0:
                gap_score = -0.80
            elif abs_gap >= 1.0:
                gap_score = -0.50
            else:
                gap_score = -0.25
        else:
            gap_score = 0.0

        result.update({
            "gap_type":         gap_type,
            "gap_pct":          round(gap_pct, 3),
            "gap_score":        round(gap_score, 3),
            "prior_close":      round(prior_close, 4),
            "today_open":       round(today_open, 4),
            "fill_probability": round(fill_prob, 2),
            "gap_filled":       gap_filled,
            "description":      desc,
        })
    except Exception as e:
        logger.debug(f"gap_analysis error: {e}")

    return result
