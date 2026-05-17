"""
Support & Resistance level calculator for the NASDAQ scalping agent.

Three complementary methodologies are combined:
  1. Classic Pivot Points  — derived from the session's High / Low / Close
  2. Swing Highs / Lows   — local extrema across a configurable lookback window,
                            clustered to eliminate near-duplicate levels
  3. Volume Point of Control (POC) — price bucket with the highest cumulative volume

All public functions handle edge-cases gracefully (empty / short DataFrames)
and return zero-filled or empty structures rather than raising exceptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

_CLUSTER_TOLERANCE = 0.004    # 0.4% — tighter clustering creates too many trivial levels
_MIN_ROWS_PIVOT    = 3        # minimum bars needed for pivot calculation
_MIN_ROWS_SWING    = 5        # minimum bars needed for swing detection
_POC_BUCKETS       = 50       # number of price buckets for volume profile


# ── 1. Classic Pivot Points ───────────────────────────────────────────────────

def calculate_pivot_points(df: pd.DataFrame,
                           df_daily: pd.DataFrame | None = None) -> dict:
    """
    Calculate standard floor-trader pivot points from the prior complete session.

    Pass df_daily (daily OHLCV) to use the correct prior-session H/L/C.
    When df_daily is None, falls back to the last bar of df (intraday approximation).

    Returns
    -------
    dict with keys: PP, R1, R2, R3, S1, S2, S3
    All values are float.  Returns zeros when the DataFrame is too short.
    """
    _empty = {k: 0.0 for k in ("PP", "R1", "R2", "R3", "S1", "S2", "S3")}

    # Prefer prior complete session from daily data (correct floor-trader input)
    if df_daily is not None and len(df_daily) >= 2:
        try:
            high  = float(df_daily["High"].iloc[-2])
            low   = float(df_daily["Low"].iloc[-2])
            close = float(df_daily["Close"].iloc[-2])
        except (KeyError, TypeError, ValueError):
            df_daily = None  # fall through to intraday fallback

    if df_daily is None or len(df_daily) < 2:
        if df is None or len(df) < _MIN_ROWS_PIVOT:
            return _empty
        try:
            high  = float(df["High"].iloc[-1])
            low   = float(df["Low"].iloc[-1])
            close = float(df["Close"].iloc[-1])
        except (KeyError, TypeError, ValueError):
            return _empty

    if high <= 0 or low <= 0 or close <= 0:
        return _empty

    pp = (high + low + close) / 3.0

    r1 = 2.0 * pp - low
    r2 = pp + (high - low)
    r3 = high + 2.0 * (pp - low)

    s1 = 2.0 * pp - high
    s2 = pp - (high - low)
    s3 = low - 2.0 * (high - pp)

    return {
        "PP": round(pp, 4),
        "R1": round(r1, 4),
        "R2": round(r2, 4),
        "R3": round(r3, 4),
        "S1": round(s1, 4),
        "S2": round(s2, 4),
        "S3": round(s3, 4),
    }


# ── 2. Swing Highs / Lows ─────────────────────────────────────────────────────

def _cluster_levels(levels: list[float], tolerance: float = _CLUSTER_TOLERANCE) -> list[float]:
    """
    Merge levels that are within `tolerance` (fractional) of each other.
    Returns the mean of each cluster, sorted ascending.
    """
    if not levels:
        return []

    sorted_lvls = sorted(levels)
    clusters: list[list[float]] = [[sorted_lvls[0]]]

    for lvl in sorted_lvls[1:]:
        ref = clusters[-1][-1]
        if ref > 0 and abs(lvl - ref) / ref <= tolerance:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])

    return [round(float(np.mean(c)), 4) for c in clusters]


def find_swing_levels(
    df: pd.DataFrame,
    window: int = 10,
    max_levels: int = 5,
) -> dict:
    """
    Identify swing high and swing low levels using a rolling local-extrema
    approach, then cluster near-duplicate levels.

    Parameters
    ----------
    df         : OHLCV DataFrame with at least ``High`` and ``Low`` columns.
    window     : Number of bars on each side used to confirm a swing high/low.
    max_levels : Maximum number of support and resistance levels to return each.

    Returns
    -------
    dict with keys:
        supports    – list[float], sorted descending (strongest first)
        resistances – list[float], sorted ascending  (nearest first)
    """
    _empty = {"supports": [], "resistances": []}

    if df is None or len(df) < _MIN_ROWS_SWING or window < 1:
        return _empty

    required_len = 2 * window + 1
    if len(df) < required_len:
        return _empty

    try:
        highs  = df["High"].values.astype(float)
        lows   = df["Low"].values.astype(float)
        closes = df["Close"].values.astype(float)
    except (KeyError, TypeError, ValueError):
        return _empty

    current_price = closes[-1]
    if current_price <= 0:
        return _empty

    swing_highs: list[float] = []
    swing_lows:  list[float] = []

    # Scan all bars except the edges (where a full window can't be formed)
    for i in range(window, len(highs) - window):
        local_window_high = highs[max(0, i - window): i + window + 1]
        local_window_low  = lows[max(0, i - window):  i + window + 1]

        if highs[i] == np.max(local_window_high):
            swing_highs.append(float(highs[i]))

        if lows[i] == np.min(local_window_low):
            swing_lows.append(float(lows[i]))

    # Cluster and separate into supports (below price) / resistances (above price)
    clustered_highs = _cluster_levels(swing_highs)
    clustered_lows  = _cluster_levels(swing_lows)

    all_levels = _cluster_levels(clustered_highs + clustered_lows)

    resistances = sorted([lvl for lvl in all_levels if lvl > current_price])
    supports    = sorted([lvl for lvl in all_levels if lvl < current_price], reverse=True)

    return {
        "supports":    [round(s, 4) for s in supports[:max_levels]],
        "resistances": [round(r, 4) for r in resistances[:max_levels]],
    }


# ── 3. Volume Point of Control ────────────────────────────────────────────────

def calculate_volume_poc(df: pd.DataFrame) -> float:
    """
    Calculate the Volume Point of Control (POC) — the price level at which
    the most volume has traded across the provided DataFrame.

    Uses a histogram approach: the High-Low range is divided into
    ``_POC_BUCKETS`` equal bins; each bar's volume is assigned to the bin
    that contains its typical price.

    Returns 0.0 when the DataFrame is insufficient or malformed.
    """
    if df is None or len(df) < 2:
        return 0.0

    try:
        highs   = df["High"].values.astype(float)
        lows    = df["Low"].values.astype(float)
        closes  = df["Close"].values.astype(float)
        volumes = df["Volume"].values.astype(float)
    except (KeyError, TypeError, ValueError):
        return 0.0

    price_min = float(np.nanmin(lows))
    price_max = float(np.nanmax(highs))

    if price_max <= price_min or price_min <= 0:
        return 0.0

    # Build histogram: typical price (HLC/3) weighted by volume
    typical_prices = (highs + lows + closes) / 3.0
    bucket_size    = (price_max - price_min) / _POC_BUCKETS

    vol_profile = np.zeros(_POC_BUCKETS, dtype=float)

    for tp, vol in zip(typical_prices, volumes):
        if not np.isfinite(tp) or not np.isfinite(vol) or vol <= 0:
            continue
        idx = int((tp - price_min) / bucket_size)
        idx = min(idx, _POC_BUCKETS - 1)   # clamp to last bucket
        vol_profile[idx] += vol

    if vol_profile.sum() == 0:
        return 0.0

    poc_idx   = int(np.argmax(vol_profile))
    poc_price = price_min + (poc_idx + 0.5) * bucket_size   # mid-point of the bucket

    return round(float(poc_price), 4)


# ── 4. Combined SR Levels ─────────────────────────────────────────────────────

def get_all_sr_levels(df: pd.DataFrame,
                      df_daily: pd.DataFrame | None = None) -> dict:
    """
    Aggregate all support/resistance methodologies into a single dictionary.

    Pass df_daily to compute pivot points from the prior complete session.

    Returns
    -------
    dict with keys:
        pivots      – dict (PP, R1-R3, S1-S3)
        supports    – list[float] sorted descending
        resistances – list[float] sorted ascending
        poc         – float
    """
    pivots = calculate_pivot_points(df, df_daily=df_daily)
    swings = find_swing_levels(df)
    poc    = calculate_volume_poc(df)

    # Enrich swing levels with pivot-derived levels (filtered by current price)
    try:
        current_price = float(df["Close"].iloc[-1]) if df is not None and len(df) > 0 else 0.0
    except (KeyError, TypeError, ValueError):
        current_price = 0.0

    if current_price > 0:
        pivot_supports    = [v for k, v in pivots.items()
                             if k.startswith("S") and v > 0 and v < current_price]
        pivot_resistances = [v for k, v in pivots.items()
                             if k.startswith("R") and v > 0 and v > current_price]

        combined_supports    = _cluster_levels(swings["supports"] + pivot_supports)
        combined_resistances = _cluster_levels(swings["resistances"] + pivot_resistances)

        # Re-filter after clustering (merging can shift values across current price)
        supports    = sorted([s for s in combined_supports    if s < current_price], reverse=True)
        resistances = sorted([r for r in combined_resistances if r > current_price])
    else:
        supports    = swings["supports"]
        resistances = swings["resistances"]

    return {
        "pivots":      pivots,
        "supports":    supports,
        "resistances": resistances,
        "poc":         poc,
    }


# ── 5. Nearest Level Helpers ──────────────────────────────────────────────────

def nearest_support(price: float, sr: dict) -> float:
    """
    Return the nearest support level below ``price``.

    Falls back to ``price * 0.98`` (2 % below) when no valid support is found.

    Parameters
    ----------
    price : Current market price.
    sr    : SR dict as returned by :func:`get_all_sr_levels`.
    """
    if not price or price <= 0:
        return 0.0

    candidates: list[float] = []

    # From combined swing / pivot supports
    for lvl in sr.get("supports", []):
        if isinstance(lvl, (int, float)) and lvl > 0 and lvl < price:
            candidates.append(float(lvl))

    # Include pivot PP and Sx levels as well
    pivots = sr.get("pivots", {})
    for key in ("PP", "S1", "S2", "S3"):
        lvl = pivots.get(key, 0.0)
        if isinstance(lvl, (int, float)) and lvl > 0 and lvl < price:
            candidates.append(float(lvl))

    # POC as support when below price
    poc = sr.get("poc", 0.0)
    if isinstance(poc, (int, float)) and poc > 0 and poc < price:
        candidates.append(float(poc))

    if not candidates:
        return round(price * 0.98, 4)

    # Closest support = maximum value that is still below price
    return round(max(candidates), 4)


def nearest_resistance(price: float, sr: dict) -> float:
    """
    Return the nearest resistance level above ``price``.

    Falls back to ``price * 1.02`` (2 % above) when no valid resistance is found.

    Parameters
    ----------
    price : Current market price.
    sr    : SR dict as returned by :func:`get_all_sr_levels`.
    """
    if not price or price <= 0:
        return 0.0

    candidates: list[float] = []

    # From combined swing / pivot resistances
    for lvl in sr.get("resistances", []):
        if isinstance(lvl, (int, float)) and lvl > 0 and lvl > price:
            candidates.append(float(lvl))

    # Include pivot PP and Rx levels as well
    pivots = sr.get("pivots", {})
    for key in ("PP", "R1", "R2", "R3"):
        lvl = pivots.get(key, 0.0)
        if isinstance(lvl, (int, float)) and lvl > 0 and lvl > price:
            candidates.append(float(lvl))

    # POC as resistance when above price
    poc = sr.get("poc", 0.0)
    if isinstance(poc, (int, float)) and poc > 0 and poc > price:
        candidates.append(float(poc))

    if not candidates:
        return round(price * 1.02, 4)

    # Closest resistance = minimum value that is still above price
    return round(min(candidates), 4)
