"""
Candlestick pattern detection and market structure analysis for the NASDAQ
scalping agent.

All public functions handle edge-cases gracefully — empty DataFrames or ones
that are too short return neutral / empty values rather than raising exceptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agent.support_resistance import nearest_support, nearest_resistance


# ── Constants ─────────────────────────────────────────────────────────────────

_MIN_ROWS_PATTERN = 3    # bars needed for pattern detection
_MIN_ROWS_TREND   = 20   # bars needed for reliable trend analysis
_EMA_SHORT        = 20
_EMA_LONG         = 50
_EMA_SLOPE_BARS   = 5    # bars used to measure EMA20 slope
_DOJI_BODY_PCT    = 0.05 # body < 5 % of bar range → Doji


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential moving average (no external library dependency)."""
    return series.ewm(span=window, adjust=False).mean()


def _body(o: float, c: float) -> float:
    return abs(c - o)


def _range(h: float, l: float) -> float:
    return h - l if h > l else 1e-9   # guard against zero-range bars


def _upper_wick(o: float, h: float, c: float) -> float:
    return h - max(o, c)


def _lower_wick(o: float, l: float, c: float) -> float:
    return min(o, c) - l


# ── 1. Candlestick Pattern Detection ─────────────────────────────────────────

def detect_patterns(df: pd.DataFrame) -> list[str]:
    """
    Detect candlestick patterns on the last 3 bars of ``df``.

    Patterns checked (in order):
        Single-bar  : Hammer, Shooting Star, Doji, Bullish Pin Bar,
                      Bearish Pin Bar
        Two-bar     : Bullish Engulfing, Bearish Engulfing
        Three-bar   : Morning Star, Evening Star

    Returns
    -------
    list[str]
        Up to 4 detected pattern name strings.  Empty list when data is
        insufficient.
    """
    if df is None or len(df) < _MIN_ROWS_PATTERN:
        return []

    try:
        opens  = df["Open"].values.astype(float)
        highs  = df["High"].values.astype(float)
        lows   = df["Low"].values.astype(float)
        closes = df["Close"].values.astype(float)
    except (KeyError, TypeError, ValueError):
        return []

    # Last 3 bars — bar3 is the most recent
    o3, h3, l3, c3 = opens[-1],  highs[-1],  lows[-1],  closes[-1]
    o2, h2, l2, c2 = opens[-2],  highs[-2],  lows[-2],  closes[-2]
    o1, h1, l1, c1 = opens[-3],  highs[-3],  lows[-3],  closes[-3]

    patterns: list[str] = []

    # ── Derived values for bar3 (most recent) ────────────────────────────────
    rng3   = _range(h3, l3)
    body3  = _body(o3, c3)
    uw3    = _upper_wick(o3, h3, c3)
    lw3    = _lower_wick(o3, l3, c3)
    mid3   = l3 + rng3 / 2.0

    # ── Derived values for bar2 ───────────────────────────────────────────────
    rng2   = _range(h2, l2)
    body2  = _body(o2, c2)
    uw2    = _upper_wick(o2, h2, c2)
    lw2    = _lower_wick(o2, l2, c2)

    # ── Derived values for bar1 (oldest of the 3) ─────────────────────────────
    rng1   = _range(h1, l1)
    body1  = _body(o1, c1)
    mid1   = l1 + rng1 / 2.0

    # ── Single-bar patterns (bar3) ────────────────────────────────────────────

    # Doji — body is negligible relative to the bar's total range
    if body3 < _DOJI_BODY_PCT * rng3:
        patterns.append("Doji")

    # Hammer (bullish reversal): small body at top, long lower wick, tiny upper wick
    elif (body3 > 0
          and lw3 >= 2.0 * body3
          and uw3 <= 0.15 * rng3
          and c3 > l3 + 0.55 * rng3):   # body is in the upper half
        patterns.append("Hammer")

    # Shooting Star (bearish reversal): small body at bottom, long upper wick
    elif (body3 > 0
          and uw3 >= 2.0 * body3
          and lw3 <= 0.15 * rng3
          and c3 < l3 + 0.45 * rng3):   # body is in the lower half
        patterns.append("Shooting Star")

    # Bullish Pin Bar: long lower wick, close above midpoint
    elif (body3 > 0
          and lw3 > 2.0 * body3
          and c3 > mid3):
        patterns.append("Bullish Pin Bar")

    # Bearish Pin Bar: long upper wick, close below midpoint
    elif (body3 > 0
          and uw3 > 2.0 * body3
          and c3 < mid3):
        patterns.append("Bearish Pin Bar")

    # ── Two-bar patterns (bar2 → bar3) ───────────────────────────────────────

    # Bullish Engulfing: bar2 bearish, bar3 bullish and fully engulfs bar2 body
    if (c2 < o2                       # bar2 bearish
            and c3 > o3               # bar3 bullish
            and o3 <= c2              # bar3 opens at or below bar2 close
            and c3 >= o2):            # bar3 closes at or above bar2 open
        patterns.append("Bullish Engulfing")

    # Bearish Engulfing: bar2 bullish, bar3 bearish and fully engulfs bar2 body
    elif (c2 > o2                     # bar2 bullish
              and c3 < o3             # bar3 bearish
              and o3 >= c2            # bar3 opens at or above bar2 close
              and c3 <= o2):          # bar3 closes at or below bar2 open
        patterns.append("Bearish Engulfing")

    # ── Three-bar patterns (bar1 → bar2 → bar3) ──────────────────────────────

    # Morning Star (bullish): large bear → small-body indecision → large bull
    if (c1 < o1                           # bar1: bearish
            and body1 > 0.5 * rng1        # bar1: large body
            and body2 < 0.3 * rng2        # bar2: small body (indecision)
            and c3 > o3                   # bar3: bullish
            and body3 > 0.5 * rng3        # bar3: large body
            and c3 > mid1):               # bar3 closes above bar1's midpoint
        patterns.append("Morning Star")

    # Evening Star (bearish): large bull → small-body indecision → large bear
    elif (c1 > o1                         # bar1: bullish
              and body1 > 0.5 * rng1      # bar1: large body
              and body2 < 0.3 * rng2      # bar2: small body (indecision)
              and c3 < o3                 # bar3: bearish
              and body3 > 0.5 * rng3      # bar3: large body
              and c3 < mid1):             # bar3 closes below bar1's midpoint
        patterns.append("Evening Star")

    return patterns[:4]   # cap at 4 to stay within spec


# ── 2. Trend Analysis ─────────────────────────────────────────────────────────

def analyze_trend_with_confidence(
    df: pd.DataFrame, min_rows: int = _MIN_ROWS_TREND
) -> tuple[str, float]:
    """
    Classify trend and return the fraction of signals (0–1) that agree.

    Parameters
    ----------
    df       : OHLCV DataFrame.
    min_rows : Minimum bars required; callers analysing sparse resampled data
               can lower this (e.g. min_rows=6 for hourly bars).

    Returns
    -------
    (trend, probability)
        trend       – "UPTREND" | "DOWNTREND" | "SIDEWAYS"
        probability – 0.50–1.00 (proportion of the 4 sub-signals that agree)
    """
    if df is None or len(df) < min_rows:
        return "SIDEWAYS", 0.5

    try:
        close = df["Close"].astype(float)
        high  = df["High"].astype(float)
        low   = df["Low"].astype(float)
    except (KeyError, TypeError, ValueError):
        return "SIDEWAYS", 0.5

    ema20 = _ema(close, _EMA_SHORT)
    ema50 = _ema(close, _EMA_LONG) if len(df) >= _EMA_LONG else ema20

    price_now = float(close.iloc[-1])
    e20_now   = float(ema20.iloc[-1])
    e50_now   = float(ema50.iloc[-1])

    signals: list[int] = []   # +1 bullish, -1 bearish, 0 neutral

    signals.append(1 if price_now > e20_now else (-1 if price_now < e20_now else 0))
    signals.append(1 if e20_now > e50_now   else (-1 if e20_now < e50_now   else 0))

    if len(ema20) >= _EMA_SLOPE_BARS:
        slope = float(ema20.iloc[-1]) - float(ema20.iloc[-_EMA_SLOPE_BARS])
        signals.append(1 if slope > 0 else (-1 if slope < 0 else 0))
    else:
        signals.append(0)

    lookback     = min(10, len(df))
    recent_highs = high.iloc[-lookback:].values
    recent_lows  = low.iloc[-lookback:].values
    hh = recent_highs[-1] > recent_highs[0]
    hl = recent_lows[-1]  > recent_lows[0]
    ll = recent_lows[-1]  < recent_lows[0]
    lh = recent_highs[-1] < recent_highs[0]

    if hh and hl:
        signals.append(1)
    elif ll and lh:
        signals.append(-1)
    else:
        signals.append(0)

    n     = len(signals)          # always 4
    bull  = sum(s for s in signals if s > 0)
    bear  = sum(-s for s in signals if s < 0)

    if bull >= 3:
        return "UPTREND",   round(bull / n, 2)
    if bear >= 3:
        return "DOWNTREND", round(bear / n, 2)
    return "SIDEWAYS", round(max(bull, bear) / n, 2)


def analyze_trend(df: pd.DataFrame) -> str:
    """Backward-compatible wrapper around analyze_trend_with_confidence."""
    trend, _ = analyze_trend_with_confidence(df)
    return trend


# ── 3. Price-Action Score ─────────────────────────────────────────────────────

def score_price_action(df: pd.DataFrame, sr: dict) -> tuple[float, list[str]]:
    """
    Score the current price action in the context of support / resistance levels.

    Parameters
    ----------
    df : OHLCV DataFrame with at least a ``Close`` column.
    sr : Support/Resistance dict as returned by
         :func:`agent.support_resistance.get_all_sr_levels`.

    Returns
    -------
    (score, reasons)
        score   – float in [-1, +1]
        reasons – list of human-readable strings, as a trader would say them
    """
    if df is None or len(df) < 1:
        return 0.0, []

    try:
        price = float(df["Close"].iloc[-1])
    except (KeyError, TypeError, ValueError):
        return 0.0, []

    if price <= 0:
        return 0.0, []

    score_parts: list[float] = []
    reasons: list[str] = []

    support    = nearest_support(price, sr)
    resistance = nearest_resistance(price, sr)
    pivot_pp   = sr.get("pivots", {}).get("PP", 0.0)
    poc        = sr.get("poc", 0.0)
    trend      = analyze_trend(df)

    # ── Distance from nearest support ─────────────────────────────────────────
    if support > 0:
        dist_support_pct = (price - support) / price

        if dist_support_pct <= 0.003:   # within 0.3 %
            score_parts.append(0.8)
            reasons.append(
                f"Price is sitting right on support at {support:.2f} — strong bounce zone"
            )
        elif dist_support_pct <= 0.007:  # within 0.7 %
            score_parts.append(0.4)
            reasons.append(
                f"Price is near support at {support:.2f} — buyers likely to defend this level"
            )
        else:
            score_parts.append(0.0)

    # ── Distance from nearest resistance ──────────────────────────────────────
    if resistance > 0:
        dist_resist_pct = (resistance - price) / price

        if dist_resist_pct <= 0.003:   # within 0.3 %
            score_parts.append(-0.8)
            reasons.append(
                f"Price is testing resistance at {resistance:.2f} — expect selling pressure"
            )
        elif dist_resist_pct <= 0.007:  # within 0.7 %
            score_parts.append(-0.4)
            reasons.append(
                f"Price approaching resistance at {resistance:.2f} — upside may be limited"
            )
        else:
            score_parts.append(0.0)

    # ── Price vs Pivot Point ───────────────────────────────────────────────────
    if pivot_pp and pivot_pp > 0:
        if price > pivot_pp:
            score_parts.append(0.3)
            reasons.append(
                f"Price is above pivot point ({pivot_pp:.2f}) — bullish intraday bias"
            )
        elif price < pivot_pp:
            score_parts.append(-0.3)
            reasons.append(
                f"Price is below pivot point ({pivot_pp:.2f}) — bearish intraday bias"
            )

    # ── Price vs Volume POC ───────────────────────────────────────────────────
    if poc and poc > 0:
        poc_dev = (price - poc) / poc
        if poc_dev > 0.002:
            score_parts.append(0.2)
            reasons.append(
                f"Price trading above volume POC ({poc:.2f}) — institutional activity supports upside"
            )
        elif poc_dev < -0.002:
            score_parts.append(-0.2)
            reasons.append(
                f"Price trading below volume POC ({poc:.2f}) — selling dominates value area"
            )

    # ── Trend direction ───────────────────────────────────────────────────────
    if trend == "UPTREND":
        score_parts.append(0.4)
        reasons.append("Price structure shows higher highs and higher lows — trend is up")
    elif trend == "DOWNTREND":
        score_parts.append(-0.4)
        reasons.append("Price structure shows lower highs and lower lows — trend is down")
    else:
        reasons.append("Market is consolidating — no clear directional bias")

    # ── Final score ───────────────────────────────────────────────────────────
    if not score_parts:
        return 0.0, reasons

    avg_score = float(np.clip(np.mean(score_parts), -1.0, 1.0))
    return round(avg_score, 4), reasons
