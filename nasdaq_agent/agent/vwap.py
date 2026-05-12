"""
VWAP (Volume Weighted Average Price) — primary intraday signal engine.

Events detected
---------------
RECLAIM   price crosses above VWAP from below (strongest long setup)
REJECTION price touches VWAP from above and reverses (short or avoid longs)
EXTENDED  price > 1 VWAP-std above/below (mean-reversion warning)
FLAT      price within ±0.15% of VWAP (no edge)

Signal score returned: [-1.0, +1.0]
  RECLAIM    +1.0  (premium long)
  ABOVE      +0.3  (mild bullish bias)
  FLAT        0.0
  BELOW      -0.3  (mild bearish bias)
  REJECTION  -0.8  (avoid longs / short bias)
  EXTENDED   ±0.8  (mean-reversion setup against the move)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ── Thresholds ────────────────────────────────────────────────────────────────
_FLAT_PCT      = 0.0015   # ±0.15% from VWAP = "flat"
_EXTENDED_PCT  = 0.012    # >1.2% from VWAP = extended / mean-reversion warning
_RECLAIM_BARS  = 3        # price was below VWAP this many bars ago to count as reclaim
_REJECT_BARS   = 3        # price was above VWAP this many bars ago to count as rejection


def compute_vwap_signal(df: pd.DataFrame) -> dict:
    """
    Analyse the last N bars to classify VWAP event and compute signal score.

    Parameters
    ----------
    df : DataFrame with computed VWAP column (added by compute_indicators).
         Must have at least 10 rows.

    Returns dict:
      event       : str   — RECLAIM | REJECTION | EXTENDED_UP | EXTENDED_DOWN | ABOVE | BELOW | FLAT
      score       : float — [-1, +1]
      vwap        : float
      price       : float
      deviation   : float — (price-vwap)/vwap %
      vwap_std    : float — std of close deviations from vwap (last 20 bars)
      description : str
    """
    result = {
        "event":       "FLAT",
        "score":       0.0,
        "vwap":        0.0,
        "price":       0.0,
        "deviation":   0.0,
        "vwap_std":    0.0,
        "description": "",
    }

    if df is None or len(df) < 10 or "vwap" not in df.columns:
        return result

    closes = df["Close"].values
    vwaps  = df["vwap"].values

    current_price = float(closes[-1])
    current_vwap  = float(vwaps[-1])

    if current_vwap <= 0:
        return result

    deviation = (current_price - current_vwap) / current_vwap

    # VWAP standard deviation (last 20 bars)
    tail_len = min(20, len(closes))
    devs     = (closes[-tail_len:] - vwaps[-tail_len:]) / vwaps[-tail_len:]
    vwap_std = float(np.std(devs)) if tail_len > 2 else 0.0

    result.update({
        "vwap":     round(current_vwap, 4),
        "price":    round(current_price, 4),
        "deviation": round(deviation * 100, 3),
        "vwap_std": round(vwap_std * 100, 3),
    })

    # ── Event classification ──────────────────────────────────────────────────
    lookback = min(_RECLAIM_BARS, len(closes) - 1)

    was_below = all(closes[-(lookback+1):-1] < vwaps[-(lookback+1):-1])
    was_above = all(closes[-(lookback+1):-1] > vwaps[-(lookback+1):-1])

    if deviation > _EXTENDED_PCT:
        event = "EXTENDED_UP"
        score = -0.7     # mean-reversion: fade the extension
        desc  = f"Price {deviation*100:+.2f}% above VWAP — extended, mean-reversion risk"

    elif deviation < -_EXTENDED_PCT:
        event = "EXTENDED_DOWN"
        score = +0.7
        desc  = f"Price {deviation*100:+.2f}% below VWAP — extended, potential bounce"

    elif current_price > current_vwap and was_below:
        event = "RECLAIM"
        score = +1.0     # strongest long signal
        desc  = f"VWAP Reclaim — price crossed above VWAP ${current_vwap:.2f} (highest-probability long)"

    elif current_price < current_vwap and was_above:
        event = "REJECTION"
        score = -0.8
        desc  = f"VWAP Rejection — price broke below VWAP ${current_vwap:.2f} (avoid longs / short bias)"

    elif abs(deviation) <= _FLAT_PCT:
        event = "FLAT"
        score = 0.0
        desc  = f"Price at VWAP ${current_vwap:.2f} — no directional edge"

    elif current_price > current_vwap:
        event = "ABOVE"
        score = +0.3
        desc  = f"Price {deviation*100:+.2f}% above VWAP ${current_vwap:.2f} — mild bullish bias"

    else:
        event = "BELOW"
        score = -0.3
        desc  = f"Price {deviation*100:+.2f}% below VWAP ${current_vwap:.2f} — mild bearish bias"

    result.update({"event": event, "score": round(score, 3), "description": desc})
    return result
