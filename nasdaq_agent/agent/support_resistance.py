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


# ── Module-level fallback constants (used when config_store is unavailable) ───
# All values are readable/writable via config_store keys (sr.*) and exposed
# in Settings → S&R Tuning so they can be changed without a code deploy.

_CLUSTER_TOLERANCE = 0.004    # 0.4% — merge near-duplicate levels
_MIN_ROWS_PIVOT    = 3
_MIN_ROWS_SWING    = 5
_POC_BUCKETS       = 50
_SWING_WINDOW      = 10       # bars on each side to confirm a swing high/low
_SWING_MAX_LEVELS  = 5        # max S/R levels returned per side
_FIB_LOOKBACK      = 50       # bars scanned for Fibonacci swing high/low
_FALLBACK_SUP_PCT  = 0.98     # price × this when no support found
_FALLBACK_RES_PCT  = 1.02     # price × this when no resistance found


def _sr_cfg(key: str, default):
    """Read an sr.* config value from config_store with a module-constant fallback."""
    try:
        from agent.config_manager import config as _cfg
        return _cfg.get(key, default)
    except Exception:
        return default


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
    _min_rows = int(_sr_cfg("sr.min_rows_pivot", _MIN_ROWS_PIVOT))

    # Prefer prior complete session from daily data (correct floor-trader input)
    if df_daily is not None and len(df_daily) >= 2:
        try:
            high  = float(df_daily["High"].iloc[-2])
            low   = float(df_daily["Low"].iloc[-2])
            close = float(df_daily["Close"].iloc[-2])
        except (KeyError, TypeError, ValueError):
            df_daily = None  # fall through to intraday fallback

    if df_daily is None or len(df_daily) < 2:
        if df is None or len(df) < _min_rows:
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

def _cluster_levels(levels: list[float], tolerance: float | None = None) -> list[float]:
    """
    Merge levels that are within `tolerance` (fractional) of each other.
    Returns the mean of each cluster, sorted ascending.
    """
    if tolerance is None:
        tolerance = float(_sr_cfg("sr.cluster_tolerance_pct", _CLUSTER_TOLERANCE * 100)) / 100

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
    window: int | None = None,
    max_levels: int | None = None,
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
    if window is None:
        window = int(_sr_cfg("sr.swing_window_bars", _SWING_WINDOW))
    if max_levels is None:
        max_levels = int(_sr_cfg("sr.swing_max_levels", _SWING_MAX_LEVELS))
    _min_rows = int(_sr_cfg("sr.min_rows_swing", _MIN_ROWS_SWING))

    if df is None or len(df) < _min_rows or window < 1:
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
    n_buckets      = int(_sr_cfg("sr.poc_buckets", _POC_BUCKETS))
    typical_prices = (highs + lows + closes) / 3.0
    bucket_size    = (price_max - price_min) / n_buckets

    vol_profile = np.zeros(n_buckets, dtype=float)

    for tp, vol in zip(typical_prices, volumes):
        if not np.isfinite(tp) or not np.isfinite(vol) or vol <= 0:
            continue
        idx = int((tp - price_min) / bucket_size)
        idx = min(idx, n_buckets - 1)   # clamp to last bucket
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
        return round(price * float(_sr_cfg("sr.fallback_support_pct", _FALLBACK_SUP_PCT)), 4)

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
        return round(price * float(_sr_cfg("sr.fallback_resistance_pct", _FALLBACK_RES_PCT)), 4)

    # Closest resistance = minimum value that is still above price
    return round(min(candidates), 4)


# ── 6. Camarilla Pivot Levels ─────────────────────────────────────────────────

def calculate_camarilla_pivots(df: pd.DataFrame,
                                df_daily: pd.DataFrame | None = None) -> dict:
    """
    Camarilla pivots — tighter intraday S/R levels preferred by scalpers.

    H4/L4: primary breakout/breakdown levels (most-watched)
    H3/L3: entry zones for mean-reversion (price tends to reject H3/L3)
    H2/L2: moderate extension
    H1/L1: minor S/R near prior close

    Formula: HN = Close + (H-L) × multiplier_N
             LN = Close − (H-L) × multiplier_N
    """
    _empty = {k: 0.0 for k in ("H1","H2","H3","H4","L1","L2","L3","L4")}

    if df_daily is not None and len(df_daily) >= 2:
        try:
            h = float(df_daily["High"].iloc[-2])
            l = float(df_daily["Low"].iloc[-2])
            c = float(df_daily["Close"].iloc[-2])
        except Exception:
            df_daily = None
    if df_daily is None or len(df_daily if df_daily is not None else []) < 2:
        if df is None or len(df) < 2:
            return _empty
        h = float(df["High"].max())
        l = float(df["Low"].min())
        c = float(df["Close"].iloc[-1])

    rng = h - l
    if rng <= 0 or c <= 0:
        return _empty

    return {
        "H4": round(c + rng * 1.1 / 2,  4),
        "H3": round(c + rng * 1.1 / 4,  4),
        "H2": round(c + rng * 1.1 / 6,  4),
        "H1": round(c + rng * 1.1 / 12, 4),
        "L1": round(c - rng * 1.1 / 12, 4),
        "L2": round(c - rng * 1.1 / 6,  4),
        "L3": round(c - rng * 1.1 / 4,  4),
        "L4": round(c - rng * 1.1 / 2,  4),
    }


# ── 7. Fibonacci Retracement from Prior Swing ─────────────────────────────────

def calculate_fibonacci_levels(df: pd.DataFrame, lookback: int | None = None) -> dict:
    """
    Calculate Fibonacci retracement and extension levels from the most recent
    significant swing high and swing low over the last `lookback` bars.

    Retracements: 23.6%, 38.2%, 50%, 61.8%, 78.6%
    Extensions:   127.2%, 161.8%, 200%, 261.8%

    Returns dict with keys: swing_high, swing_low, direction, and all Fib levels.
    """
    _empty = {"swing_high": 0.0, "swing_low": 0.0, "direction": "NONE",
              "fib_236": 0.0, "fib_382": 0.0, "fib_500": 0.0,
              "fib_618": 0.0, "fib_786": 0.0,
              "ext_1272": 0.0, "ext_1618": 0.0, "ext_2000": 0.0}

    if lookback is None:
        lookback = int(_sr_cfg("sr.fibonacci_lookback_bars", _FIB_LOOKBACK))

    if df is None or len(df) < 10:
        return _empty

    tail = df.tail(lookback)
    h = float(tail["High"].max())
    l = float(tail["Low"].min())
    c = float(df["Close"].iloc[-1])

    if h <= l or h <= 0:
        return _empty

    rng = h - l
    direction = "UP" if c > (h + l) / 2 else "DOWN"

    if direction == "UP":
        # Retracements of the up-move (from low to high): price may pull back to these
        return {
            "swing_high": round(h, 4),
            "swing_low":  round(l, 4),
            "direction":  direction,
            "fib_236":  round(h - rng * 0.236, 4),
            "fib_382":  round(h - rng * 0.382, 4),
            "fib_500":  round(h - rng * 0.500, 4),
            "fib_618":  round(h - rng * 0.618, 4),
            "fib_786":  round(h - rng * 0.786, 4),
            "ext_1272": round(h + rng * 0.272, 4),
            "ext_1618": round(h + rng * 0.618, 4),
            "ext_2000": round(h + rng * 1.000, 4),
        }
    else:
        # Down-move retracements (price may bounce to these levels)
        return {
            "swing_high": round(h, 4),
            "swing_low":  round(l, 4),
            "direction":  direction,
            "fib_236":  round(l + rng * 0.236, 4),
            "fib_382":  round(l + rng * 0.382, 4),
            "fib_500":  round(l + rng * 0.500, 4),
            "fib_618":  round(l + rng * 0.618, 4),
            "fib_786":  round(l + rng * 0.786, 4),
            "ext_1272": round(l - rng * 0.272, 4),
            "ext_1618": round(l - rng * 0.618, 4),
            "ext_2000": round(l - rng * 1.000, 4),
        }


# ── 8. Volume Profile Value Area (VAH / VAL) ──────────────────────────────────

def calculate_value_area(df: pd.DataFrame, va_pct: float = 0.70) -> dict:
    """
    Calculate Volume Profile Value Area (70% rule).

    VAH (Value Area High): Top boundary where 70% of volume was traded
    VAL (Value Area Low):  Bottom boundary
    POC (Point of Control): Price with highest volume (same as calculate_volume_poc)

    The 80% Rule: if price opens outside the value area and enters it,
    there is an 80% probability of trading to the opposite VA boundary.

    Returns dict with: poc, vah, val, va_pct_actual
    """
    _empty = {"poc": 0.0, "vah": 0.0, "val": 0.0, "va_pct_actual": 0.0}

    if df is None or len(df) < 5:
        return _empty

    try:
        highs   = df["High"].values.astype(float)
        lows    = df["Low"].values.astype(float)
        closes  = df["Close"].values.astype(float)
        volumes = df["Volume"].values.astype(float)
    except Exception:
        return _empty

    price_min = float(np.nanmin(lows))
    price_max = float(np.nanmax(highs))
    if price_max <= price_min or price_min <= 0:
        return _empty

    # Build volume profile histogram
    n_buckets  = int(_sr_cfg("sr.poc_buckets", _POC_BUCKETS))
    bucket_sz  = (price_max - price_min) / n_buckets
    vol_profile = np.zeros(n_buckets)
    tp = (highs + lows + closes) / 3.0

    for p, v in zip(tp, volumes):
        if not np.isfinite(p) or not np.isfinite(v) or v <= 0:
            continue
        idx = min(int((p - price_min) / bucket_sz), n_buckets - 1)
        vol_profile[idx] += v

    total_vol = vol_profile.sum()
    if total_vol == 0:
        return _empty

    poc_idx   = int(np.argmax(vol_profile))
    poc_price = price_min + (poc_idx + 0.5) * bucket_sz

    # Expand outward from POC until 70% of volume is captured
    target_vol = total_vol * va_pct
    va_vol      = vol_profile[poc_idx]
    lo_idx      = poc_idx
    hi_idx      = poc_idx

    while va_vol < target_vol:
        # Expand to whichever side has more volume next
        next_lo = lo_idx - 1
        next_hi = hi_idx + 1
        can_lo  = next_lo >= 0
        can_hi  = next_hi < n_buckets

        if not can_lo and not can_hi:
            break

        vol_lo = vol_profile[next_lo] if can_lo else -1
        vol_hi = vol_profile[next_hi] if can_hi else -1

        if vol_hi >= vol_lo and can_hi:
            hi_idx = next_hi
            va_vol += vol_profile[hi_idx]
        elif can_lo:
            lo_idx = next_lo
            va_vol += vol_profile[lo_idx]
        else:
            break

    vah = price_min + (hi_idx + 1) * bucket_sz
    val = price_min + lo_idx * bucket_sz

    return {
        "poc": round(poc_price, 4),
        "vah": round(vah, 4),
        "val": round(val, 4),
        "va_pct_actual": round(va_vol / total_vol, 3),
    }
