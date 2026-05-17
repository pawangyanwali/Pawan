"""
VWAP (Volume Weighted Average Price) — primary intraday signal engine.

Events detected
---------------
RECLAIM       price crosses above VWAP from below (strongest long setup)
REJECTION     price crosses below VWAP from above (avoid longs / short)
AT_2SD_UP     price at +2σ band — high-probability mean-reversion short
AT_2SD_DOWN   price at -2σ band — high-probability mean-reversion long
AT_1SD_UP     price at +1σ — extended; reduce long exposure
AT_1SD_DOWN   price at -1σ — extended down; reduce short exposure
ABOVE         price above VWAP (mild bullish)
BELOW         price below VWAP (mild bearish)
FLAT          price within ±0.15% of VWAP (no edge)

Signal score returned: [-1.0, +1.0]
  RECLAIM      +1.0  (premium long)
  AT_2SD_DOWN  +0.9  (mean-reversion long — dynamic σ-band, not fixed %)
  AT_1SD_DOWN  +0.5
  ABOVE        +0.3  (mild bullish bias)
  FLAT          0.0
  BELOW        -0.3  (mild bearish bias)
  AT_1SD_UP    -0.5
  AT_2SD_UP    -0.9  (mean-reversion short)
  REJECTION    -0.8  (avoid longs / short bias)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ── Thresholds ────────────────────────────────────────────────────────────────
_FLAT_PCT      = 0.0015   # ±0.15% from VWAP = "flat" (used when σ is unavailable)
_RECLAIM_BARS  = 3        # price was below VWAP this many bars ago to count as reclaim
_REJECT_BARS   = 3        # price was above VWAP this many bars ago to count as rejection
_BAND_1SD      = 1.0      # σ multiplier for first band
_BAND_2SD      = 2.0      # σ multiplier for second band (primary mean-reversion zone)
_BAND_3SD      = 3.0      # σ multiplier for extreme band


def compute_vwap_signal(df: pd.DataFrame) -> dict:
    """
    Analyse the last N bars to classify VWAP event and compute signal score.

    Uses dynamic VWAP σ-bands instead of fixed % thresholds. The ±2σ zone
    captures ~82% of intraday price action; moves beyond it have very high
    mean-reversion probability regardless of the stock's volatility regime.

    Returns dict:
      event         : str   — RECLAIM | REJECTION | AT_2SD_UP/DOWN |
                               AT_1SD_UP/DOWN | ABOVE | BELOW | FLAT
      score         : float — [-1, +1]
      vwap          : float
      price         : float
      deviation     : float — (price-vwap)/vwap %
      vwap_std      : float — σ of close deviations from vwap (last 20 bars, %)
      z_score       : float — deviation in σ units
      upper_1/2/3   : float — VWAP + 1/2/3 σ band levels
      lower_1/2/3   : float — VWAP - 1/2/3 σ band levels
      description   : str
    """
    result = {
        "event": "FLAT", "score": 0.0,
        "vwap": 0.0, "price": 0.0, "deviation": 0.0,
        "vwap_std": 0.0, "z_score": 0.0,
        "upper_1": 0.0, "lower_1": 0.0,
        "upper_2": 0.0, "lower_2": 0.0,
        "upper_3": 0.0, "lower_3": 0.0,
        "description": "",
    }

    if df is None or len(df) < 10 or "vwap" not in df.columns:
        return result

    closes = df["Close"].values
    vwaps  = df["vwap"].values
    vols   = df["Volume"].values

    current_price = float(closes[-1])
    current_vwap  = float(vwaps[-1])
    if current_vwap <= 0:
        return result

    deviation = (current_price - current_vwap) / current_vwap

    # ── Dynamic VWAP σ (volume-weighted, last 20 bars) ────────────────────────
    tail  = min(20, len(closes))
    p, v  = closes[-tail:], vols[-tail:]
    total_vol = float(np.sum(v))
    if total_vol > 0:
        vwap_std_abs = float(np.sqrt(np.sum(v * (p - current_vwap) ** 2) / total_vol))
    else:
        # Fallback: simple price std
        vwap_std_abs = float(np.std(p)) if tail > 2 else current_vwap * 0.005
    vwap_std_pct = vwap_std_abs / current_vwap if current_vwap > 0 else 0.0

    z_score = (current_price - current_vwap) / vwap_std_abs if vwap_std_abs > 0 else 0.0

    # Band levels (absolute price)
    u1 = current_vwap + _BAND_1SD * vwap_std_abs
    l1 = current_vwap - _BAND_1SD * vwap_std_abs
    u2 = current_vwap + _BAND_2SD * vwap_std_abs
    l2 = current_vwap - _BAND_2SD * vwap_std_abs
    u3 = current_vwap + _BAND_3SD * vwap_std_abs
    l3 = current_vwap - _BAND_3SD * vwap_std_abs

    result.update({
        "vwap":      round(current_vwap, 4),
        "price":     round(current_price, 4),
        "deviation": round(deviation * 100, 3),
        "vwap_std":  round(vwap_std_pct * 100, 3),
        "z_score":   round(z_score, 2),
        "upper_1": round(u1, 4), "lower_1": round(l1, 4),
        "upper_2": round(u2, 4), "lower_2": round(l2, 4),
        "upper_3": round(u3, 4), "lower_3": round(l3, 4),
    })

    # ── Event classification ──────────────────────────────────────────────────
    lookback  = min(_RECLAIM_BARS, len(closes) - 1)
    was_below = all(closes[-(lookback+1):-1] < vwaps[-(lookback+1):-1])
    was_above = all(closes[-(lookback+1):-1] > vwaps[-(lookback+1):-1])

    # Fresh crossovers take priority over band readings
    if current_price > current_vwap and was_below:
        event = "RECLAIM"
        score = +1.0
        desc  = (f"VWAP Reclaim ${current_vwap:.2f} — "
                 f"premium long (Z={z_score:+.1f}σ)")

    elif current_price < current_vwap and was_above:
        event = "REJECTION"
        score = -0.8
        desc  = (f"VWAP Rejection ${current_vwap:.2f} — "
                 f"avoid longs (Z={z_score:+.1f}σ)")

    # σ-band mean-reversion zones (dynamic, adapts to each stock's volatility)
    elif z_score >= _BAND_2SD:
        event = "AT_2SD_UP"
        score = -0.9
        desc  = (f"+2σ VWAP band ${u2:.2f} — high-probability mean-reversion short "
                 f"(Z={z_score:+.1f}σ, dev={deviation*100:+.2f}%)")

    elif z_score <= -_BAND_2SD:
        event = "AT_2SD_DOWN"
        score = +0.9
        desc  = (f"−2σ VWAP band ${l2:.2f} — high-probability mean-reversion long "
                 f"(Z={z_score:+.1f}σ, dev={deviation*100:+.2f}%)")

    elif z_score >= _BAND_1SD:
        event = "AT_1SD_UP"
        score = -0.5
        desc  = (f"+1σ VWAP band — extended up (Z={z_score:+.1f}σ), "
                 f"reduce longs / trail stops")

    elif z_score <= -_BAND_1SD:
        event = "AT_1SD_DOWN"
        score = +0.5
        desc  = (f"−1σ VWAP band — extended down (Z={z_score:+.1f}σ), "
                 f"mean-reversion potential")

    elif abs(deviation) <= _FLAT_PCT:
        event = "FLAT"
        score = 0.0
        desc  = f"At VWAP ${current_vwap:.2f} — no directional edge"

    elif current_price > current_vwap:
        event = "ABOVE"
        score = +0.3
        desc  = (f"Price {deviation*100:+.2f}% above VWAP ${current_vwap:.2f} "
                 f"(Z={z_score:+.1f}σ) — mild bullish bias")

    else:
        event = "BELOW"
        score = -0.3
        desc  = (f"Price {deviation*100:+.2f}% below VWAP ${current_vwap:.2f} "
                 f"(Z={z_score:+.1f}σ) — mild bearish bias")

    result.update({"event": event, "score": round(score, 3), "description": desc})
    return result
