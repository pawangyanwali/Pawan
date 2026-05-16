"""
Relative Strength vs market benchmark (SPY).

RS = (stock intraday return) / (SPY intraday return)

Interpretation
--------------
RS > 1.2  → Leading the market — bullish edge on long setups
RS < 0.8  → Lagging the market — bearish edge on short setups (or avoid longs)
0.8–1.2   → In-line with market
Negative  → Counter-trend (stock down while market up or vice versa)

RS_SCORE mapped to [-1, +1]:
  RS ≥ 2.0  →  +1.0
  RS  1.5   →  +0.6
  RS  1.2   →  +0.3
  RS  1.0   →   0.0
  RS  0.8   →  -0.3
  RS  0.5   →  -0.6
  RS ≤ 0.0  →  -1.0  (counter-trend)
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _intraday_return(df: pd.DataFrame) -> float:
    """Intraday return from first open to last close."""
    if df is None or df.empty:
        return 0.0
    try:
        open_p  = float(df["Open"].iloc[0])
        close_p = float(df["Close"].iloc[-1])
        return (close_p - open_p) / open_p if open_p else 0.0
    except Exception:
        return 0.0


def _rs_to_score(rs: float) -> float:
    """Map RS ratio to [-1, +1] signal contribution."""
    if rs >= 2.0:   return  1.0
    if rs >= 1.5:   return  0.6
    if rs >= 1.2:   return  0.3
    if rs >= 0.9:   return  0.0
    if rs >= 0.5:   return -0.3
    if rs >= 0.0:   return -0.6
    return -1.0     # counter-trend (negative RS)


def compute_relative_strength(df_stock: pd.DataFrame, df_spy: pd.DataFrame) -> dict:
    """
    Compute RS ratio and derived signal score.

    Returns dict:
      rs_ratio  : float  — stock return / spy return
      rs_score  : float  — [-1, +1] signal contribution
      rs_label  : str    — LEADING | IN_LINE | LAGGING | COUNTER
      stock_ret : float  — % intraday return
      spy_ret   : float  — % intraday return
      description : str
    """
    result = {
        "rs_ratio":   1.0,
        "rs_score":   0.0,
        "rs_label":   "IN_LINE",
        "stock_ret":  0.0,
        "spy_ret":    0.0,
        "description": "",
    }
    try:
        stock_ret = _intraday_return(df_stock)
        spy_ret   = _intraday_return(df_spy)

        result["stock_ret"] = round(stock_ret * 100, 3)
        result["spy_ret"]   = round(spy_ret   * 100, 3)

        if abs(spy_ret) < 0.0005:   # market flat — RS undefined, use neutral
            result["description"] = "Market flat — RS undefined."
            return result

        rs_ratio = stock_ret / spy_ret
        rs_score = _rs_to_score(rs_ratio)

        if rs_ratio < 0.0:
            label = "COUNTER"
        elif rs_ratio >= 1.2:
            label = "LEADING"
        elif rs_ratio <= 0.8:
            label = "LAGGING"
        else:
            label = "IN_LINE"

        pct_s = stock_ret * 100
        pct_m = spy_ret   * 100
        result.update({
            "rs_ratio":    round(rs_ratio, 3),
            "rs_score":    round(rs_score, 3),
            "rs_label":    label,
            "description": (
                f"Stock {pct_s:+.2f}% vs SPY {pct_m:+.2f}% → RS {rs_ratio:.2f} ({label})"
            ),
        })
    except Exception as e:
        logger.debug(f"relative_strength error: {e}")
    return result
