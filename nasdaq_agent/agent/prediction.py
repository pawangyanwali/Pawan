"""
Professional trade prediction engine for the NASDAQ scalping agent.

Combines technical indicators, volume analysis, ML probability, sentiment,
price-action patterns and support/resistance context into a single, actionable
trade prediction with direction, confidence, targets and risk-reward ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agent.support_resistance import (
    get_all_sr_levels,
    nearest_support,
    nearest_resistance,
)
from agent.price_action import detect_patterns, analyze_trend, score_price_action


# ── Composite score weights ───────────────────────────────────────────────────

_W_TECH    = 0.25
_W_VOLUME  = 0.15
_W_ML      = 0.20
_W_PA      = 0.20   # price-action (SR + trend)
_W_PATTERN = 0.15
_W_SENT    = 0.05

# Direction thresholds
_STRONG_BUY_THRESH  =  0.60
_BUY_THRESH         =  0.25
_SELL_THRESH        = -0.25
_STRONG_SELL_THRESH = -0.60

# Per-pattern score contribution
_BULLISH_PATTERNS = {"Hammer", "Bullish Engulfing", "Bullish Pin Bar", "Morning Star"}
_BEARISH_PATTERNS = {"Shooting Star", "Bearish Engulfing", "Bearish Pin Bar", "Evening Star"}
_PATTERN_UNIT     = 0.5   # score per detected pattern


# ── Empty prediction sentinel ─────────────────────────────────────────────────

def _empty_prediction() -> dict:
    """
    Return a neutral, zero-filled prediction dict for error / edge-case paths.

    Callers can detect this by checking ``direction == "NEUTRAL"`` and
    ``confidence == 0``.
    """
    return {
        "direction":       "NEUTRAL",
        "confidence":      0.0,
        "composite_score": 0.0,
        "target_price":    0.0,
        "stop_loss":       0.0,
        "rr_ratio":        0.0,
        "trend":           "SIDEWAYS",
        "patterns":        [],
        "reasons":         [],
        "supports":        [],
        "resistances":     [],
        "pivots":          {},
        "poc":             0.0,
    }


# ── Pattern scoring ───────────────────────────────────────────────────────────

def _score_patterns(patterns: list[str]) -> tuple[float, list[str]]:
    """
    Convert a list of detected candlestick patterns into a score and reasons.

    Score is capped at ±1.

    Returns
    -------
    (score, reason_strings)
    """
    raw = 0.0
    reasons: list[str] = []

    for p in patterns:
        if p in _BULLISH_PATTERNS:
            raw += _PATTERN_UNIT
            reasons.append(f"{p} pattern detected — bullish reversal signal")
        elif p in _BEARISH_PATTERNS:
            raw -= _PATTERN_UNIT
            reasons.append(f"{p} pattern detected — bearish reversal signal")
        else:
            # Doji / ambiguous patterns — note but do not bias score
            reasons.append(f"{p} pattern detected — market indecision, wait for confirmation")

    return float(np.clip(raw, -1.0, 1.0)), reasons


# ── Indicator-based technical reasons ────────────────────────────────────────

def _build_tech_reasons(last_row: pd.Series) -> list[str]:
    """
    Derive human-readable reasons from the last bar's pre-computed indicator
    columns (as produced by :func:`agent.technical.compute_indicators`).

    Returns
    -------
    list[str] — may be empty when indicator columns are absent.
    """
    reasons: list[str] = []

    # RSI ─────────────────────────────────────────────────────────────────────
    rsi = last_row.get("rsi_14") if hasattr(last_row, "get") else None
    if rsi is None:
        rsi = last_row["rsi_14"] if "rsi_14" in last_row.index else None
    if rsi is not None and pd.notna(rsi):
        rsi = float(rsi)
        if rsi < 30:
            reasons.append(f"RSI {rsi:.1f} — stock is deeply oversold, mean-reversion likely")
        elif rsi < 40:
            reasons.append(f"RSI {rsi:.1f} — oversold conditions building potential snap-back")
        elif rsi > 70:
            reasons.append(f"RSI {rsi:.1f} — overbought, momentum may be exhausted")
        elif rsi > 60:
            reasons.append(f"RSI {rsi:.1f} — approaching overbought territory, watch for reversal")

    # MACD histogram ──────────────────────────────────────────────────────────
    macd_hist = _safe_float(last_row, "macd_hist")
    if macd_hist is not None:
        if macd_hist > 0:
            reasons.append(
                f"MACD histogram positive ({macd_hist:.4f}) — bullish momentum building"
            )
        elif macd_hist < 0:
            reasons.append(
                f"MACD histogram negative ({macd_hist:.4f}) — bearish momentum in control"
            )

    # Bollinger Band position ─────────────────────────────────────────────────
    bb_pct = _safe_float(last_row, "bb_pct")
    if bb_pct is not None:
        if bb_pct < 0.10:
            reasons.append(
                "Price at lower Bollinger Band — classic bounce zone for scalp entries"
            )
        elif bb_pct > 0.90:
            reasons.append(
                "Price at upper Bollinger Band — extended, fade opportunity on confirmation"
            )

    # VWAP deviation ──────────────────────────────────────────────────────────
    vwap  = _safe_float(last_row, "vwap")
    close = _safe_float(last_row, "Close")
    if vwap and close and vwap > 0:
        dev_pct = (close - vwap) / vwap
        if dev_pct > 0.002:
            reasons.append(
                f"Price is {dev_pct * 100:.2f}% above VWAP — institutional buyers are active"
            )
        elif dev_pct < -0.002:
            reasons.append(
                f"Price is {abs(dev_pct) * 100:.2f}% below VWAP — sellers dominating intraday flow"
            )

    return reasons


# ── Volume reasons ────────────────────────────────────────────────────────────

def _build_volume_reasons(last_row: pd.Series) -> list[str]:
    """Generate volume-related reason strings from the last bar."""
    reasons: list[str] = []

    rvol = _safe_float(last_row, "vol_ratio")
    if rvol is None:
        return reasons

    if rvol >= 2.5:
        reasons.append(
            f"Unusual volume spike ({rvol:.1f}x average) — significant interest from large players"
        )
    elif rvol >= 1.5:
        reasons.append(
            f"Elevated relative volume ({rvol:.1f}x average) — above-average participation"
        )

    return reasons


# ── ML reasons ───────────────────────────────────────────────────────────────

def _build_ml_reasons(ml_prob: float) -> list[str]:
    """Convert the ML model's up-probability into a reason string."""
    if ml_prob >= 0.65:
        return [
            f"ML model forecasts upside with {ml_prob * 100:.0f}% probability — algorithmic edge is bullish"
        ]
    if ml_prob <= 0.35:
        return [
            f"ML model flags downside risk ({(1 - ml_prob) * 100:.0f}% bearish probability)"
        ]
    return []


# ── Direction and targets ─────────────────────────────────────────────────────

def _label_direction(composite: float) -> str:
    if composite >= _STRONG_BUY_THRESH:
        return "STRONG BUY"
    if composite >= _BUY_THRESH:
        return "BUY"
    if composite <= _STRONG_SELL_THRESH:
        return "STRONG SELL"
    if composite <= _SELL_THRESH:
        return "SELL"
    return "NEUTRAL"


def _compute_rr(price: float, target: float, stop: float) -> float:
    """Risk-reward ratio, rounded to 2 decimals. Returns 0.0 on bad inputs."""
    try:
        reward = abs(target - price)
        risk   = abs(price - stop)
        if risk == 0:
            return 0.0
        return round(reward / risk, 2)
    except (TypeError, ZeroDivisionError):
        return 0.0


# ── Utility helpers ───────────────────────────────────────────────────────────

def _safe_float(row: pd.Series, key: str) -> float | None:
    """Extract a float from a pandas Series row; return None if missing/NaN."""
    try:
        val = row[key]
        if pd.isna(val):
            return None
        return float(val)
    except (KeyError, TypeError, ValueError):
        return None


# ── Main prediction function ──────────────────────────────────────────────────

def generate_prediction(
    ticker:     str,
    df:         pd.DataFrame,
    tech_score: float,
    vol_score:  float,
    ml_prob:    float,
    sent_score: float,
    last_row:   pd.Series,
) -> dict:
    """
    Generate a complete, actionable scalping prediction for ``ticker``.

    Parameters
    ----------
    ticker     : Ticker symbol (used in fallback label only).
    df         : OHLCV DataFrame with pre-computed indicator columns.
    tech_score : Technical indicator score in [-1, +1].
    vol_score  : Volume score in [-1, +1].
    ml_prob    : ML model's probability of price going up, in [0, 1].
    sent_score : Sentiment score in [-1, +1].
    last_row   : Last row of ``df`` (pd.Series) for fast indicator lookup.

    Returns
    -------
    dict with keys:
        direction, confidence, composite_score, target_price, stop_loss,
        rr_ratio, trend, patterns, reasons, supports, resistances, pivots, poc
    """
    # ── Guard: must have usable data ─────────────────────────────────────────
    if df is None or df.empty or len(df) < 3:
        return _empty_prediction()

    try:
        price = float(df["Close"].iloc[-1])
    except (KeyError, TypeError, ValueError, IndexError):
        return _empty_prediction()

    if price <= 0 or not np.isfinite(price):
        return _empty_prediction()

    # ── 1. Support / Resistance ───────────────────────────────────────────────
    try:
        sr = get_all_sr_levels(df)
    except Exception:
        sr = {"pivots": {}, "supports": [], "resistances": [], "poc": 0.0}

    support    = nearest_support(price, sr)
    resistance = nearest_resistance(price, sr)

    # ── 2. Price action score + patterns ─────────────────────────────────────
    try:
        pa_score, pa_reasons = score_price_action(df, sr)
    except Exception:
        pa_score, pa_reasons = 0.0, []

    try:
        patterns = detect_patterns(df)
    except Exception:
        patterns = []

    pattern_score, pattern_reasons = _score_patterns(patterns)

    # ── 3. Trend ──────────────────────────────────────────────────────────────
    try:
        trend = analyze_trend(df)
    except Exception:
        trend = "SIDEWAYS"

    # ── 4. Indicator-based reasons ────────────────────────────────────────────
    tech_reasons = _build_tech_reasons(last_row)
    vol_reasons  = _build_volume_reasons(last_row)
    ml_reasons   = _build_ml_reasons(float(ml_prob))

    # ── 5. Composite score ────────────────────────────────────────────────────
    ml_score = float(np.clip((float(ml_prob) - 0.5) * 2, -1.0, 1.0))

    composite = (
        _W_TECH    * float(tech_score)   +
        _W_VOLUME  * float(vol_score)    +
        _W_ML      * ml_score            +
        _W_PA      * pa_score            +
        _W_PATTERN * pattern_score       +
        _W_SENT    * float(sent_score)
    )
    composite = round(float(np.clip(composite, -1.0, 1.0)), 4)

    # ── 6. Direction and confidence ───────────────────────────────────────────
    direction  = _label_direction(composite)
    confidence = round(abs(composite) * 100, 1)

    # ── 7. Targets and stop-loss ──────────────────────────────────────────────
    is_bullish = composite >= 0

    if is_bullish:
        target    = round(resistance, 4)
        stop_loss = round(support * 0.998, 4)
    else:
        target    = round(support, 4)
        stop_loss = round(resistance * 1.002, 4)

    rr_ratio = _compute_rr(price, target, stop_loss)

    # ── 8. Consolidate reasons (up to 8) ─────────────────────────────────────
    all_reasons = (
        pa_reasons
        + pattern_reasons
        + tech_reasons
        + vol_reasons
        + ml_reasons
    )
    # Deduplicate while preserving order, then cap at 8
    seen: set[str] = set()
    deduped: list[str] = []
    for r in all_reasons:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    reasons = deduped[:8]

    # ── 9. Return ─────────────────────────────────────────────────────────────
    return {
        "direction":       direction,
        "confidence":      confidence,
        "composite_score": composite,
        "target_price":    target,
        "stop_loss":       stop_loss,
        "rr_ratio":        rr_ratio,
        "trend":           trend,
        "patterns":        patterns,
        "reasons":         reasons,
        "supports":        sr.get("supports", []),
        "resistances":     sr.get("resistances", []),
        "pivots":          sr.get("pivots", {}),
        "poc":             sr.get("poc", 0.0),
    }
