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
from agent.price_action import (
    detect_patterns,
    analyze_trend_with_confidence,
    score_price_action,
)


# ── Direction thresholds ──────────────────────────────────────────────────────

_STRONG_BUY_THRESH  =  0.50
_BUY_THRESH         =  0.15
_SELL_THRESH        = -0.15
_STRONG_SELL_THRESH = -0.50

# ── Pattern classification ────────────────────────────────────────────────────

_BULLISH_PATTERNS = {"Hammer", "Bullish Engulfing", "Bullish Pin Bar", "Morning Star"}
_BEARISH_PATTERNS = {"Shooting Star", "Bearish Engulfing", "Bearish Pin Bar", "Evening Star"}
_PATTERN_UNIT     = 0.5   # score per pattern, capped at ±1


# ── Empty prediction sentinel ─────────────────────────────────────────────────

def _empty_prediction() -> dict:
    return {
        "direction":        "NEUTRAL",
        "confidence":       0.0,
        "composite_score":  0.0,
        "target_price":     0.0,
        "stop_loss":        0.0,
        "rr_ratio":         0.0,
        "trend":            "SIDEWAYS",
        "trend_probability": 0.5,
        "ml_trained":       False,
        "patterns":         [],
        "reasons":          [],
        "supports":         [],
        "resistances":      [],
        "pivots":           {},
        "poc":              0.0,
    }


# ── Pattern scoring ───────────────────────────────────────────────────────────

def _score_patterns(patterns: list[str]) -> tuple[float, list[str]]:
    raw = 0.0
    reasons: list[str] = []
    for p in patterns:
        if p in _BULLISH_PATTERNS:
            raw += _PATTERN_UNIT
            reasons.append(f"{p} pattern — bullish reversal signal")
        elif p in _BEARISH_PATTERNS:
            raw -= _PATTERN_UNIT
            reasons.append(f"{p} pattern — bearish reversal signal")
        else:
            reasons.append(f"{p} — indecision, wait for confirmation")
    return float(np.clip(raw, -1.0, 1.0)), reasons


# ── Indicator-based reasons ───────────────────────────────────────────────────

def _safe_float(row: pd.Series, key: str) -> float | None:
    try:
        val = row[key]
        return None if pd.isna(val) else float(val)
    except (KeyError, TypeError, ValueError):
        return None


def _build_tech_reasons(last_row: pd.Series) -> list[str]:
    reasons: list[str] = []

    rsi = _safe_float(last_row, "rsi_14")
    if rsi is not None:
        if rsi < 30:
            reasons.append(f"RSI {rsi:.1f} — deeply oversold, mean-reversion likely")
        elif rsi < 40:
            reasons.append(f"RSI {rsi:.1f} — oversold, potential snap-back building")
        elif rsi > 70:
            reasons.append(f"RSI {rsi:.1f} — overbought, momentum may be exhausted")
        elif rsi > 60:
            reasons.append(f"RSI {rsi:.1f} — approaching overbought, watch for reversal")

    macd_hist = _safe_float(last_row, "macd_hist")
    if macd_hist is not None:
        if macd_hist > 0:
            reasons.append(f"MACD histogram +{macd_hist:.4f} — bullish momentum building")
        elif macd_hist < 0:
            reasons.append(f"MACD histogram {macd_hist:.4f} — bearish momentum in control")

    bb_pct = _safe_float(last_row, "bb_pct")
    if bb_pct is not None:
        if bb_pct < 0.10:
            reasons.append("Price at lower Bollinger Band — classic bounce zone")
        elif bb_pct > 0.90:
            reasons.append("Price at upper Bollinger Band — extended, fade on confirmation")

    vwap  = _safe_float(last_row, "vwap")
    close = _safe_float(last_row, "Close")
    if vwap and close and vwap > 0:
        dev = (close - vwap) / vwap
        if dev > 0.002:
            reasons.append(f"Price {dev*100:.2f}% above VWAP — institutional buyers active")
        elif dev < -0.002:
            reasons.append(f"Price {abs(dev)*100:.2f}% below VWAP — sellers dominating intraday")

    return reasons


def _build_volume_reasons(last_row: pd.Series) -> list[str]:
    rvol = _safe_float(last_row, "vol_ratio")
    if rvol is None:
        return []
    if rvol >= 2.5:
        return [f"Unusual volume spike ({rvol:.1f}× avg) — significant large-player interest"]
    if rvol >= 1.5:
        return [f"Elevated volume ({rvol:.1f}× avg) — above-average participation"]
    return []


def _build_ml_reasons(ml_prob: float, ml_trained: bool) -> list[str]:
    if not ml_trained:
        return []
    if ml_prob >= 0.65:
        return [f"ML model: {ml_prob*100:.0f}% probability of upside — algorithmic edge bullish"]
    if ml_prob <= 0.35:
        return [f"ML model: {(1-ml_prob)*100:.0f}% probability of downside — algorithmic edge bearish"]
    return []


# ── Composite scoring ─────────────────────────────────────────────────────────

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


def _compute_confidence(
    direction:    str,
    trend:        str,
    trend_prob:   float,
    tech_score:   float,
    vol_score:    float,
    ml_prob:      float,
    ml_trained:   bool,
    pa_score:     float,
    pattern_score: float,
    sent_score:   float,
) -> float:
    """
    Confidence = weighted signal-agreement score (0–95 %).

    Each sub-signal contributes its weight × how strongly it agrees with
    the stated direction.  Agreement is mapped from [-1,+1] to [0,1]:
        agreement = (signal_in_direction + 1) / 2
    so a neutral signal scores 0.50 (partial credit), a contrary signal 0.0.
    """
    if direction == "NEUTRAL":
        return 50.0

    d = 1 if direction in ("BUY", "STRONG BUY") else -1

    votes = 0.0
    total = 0.0

    def _agree(signal: float, weight: float) -> None:
        nonlocal votes, total
        agreement = float(np.clip((signal * d + 1) / 2, 0.0, 1.0))
        votes += weight * agreement
        total += weight

    # Trend  — strongest structural signal
    trend_signal = 1.0 if trend == "UPTREND" else (-1.0 if trend == "DOWNTREND" else 0.0)
    _agree(trend_signal * trend_prob, 30)

    _agree(tech_score,    25)
    _agree(vol_score,     12)
    _agree(pa_score,      15)
    _agree(pattern_score, 10)
    _agree(sent_score,     5)

    if ml_trained:
        # ml_prob in [0,1]; convert to [-1,+1]: (prob-0.5)*2
        ml_signal = float(np.clip((ml_prob - 0.5) * 2, -1.0, 1.0))
        _agree(ml_signal, 20)
    # If untrained: ML weight is simply not included (total stays at 97)

    confidence = (votes / total * 100) if total > 0 else 50.0
    # NEUTRAL cap + absolute bounds
    return round(float(np.clip(confidence, 25.0, 95.0)), 1)


def _compute_rr(price: float, target: float, stop: float) -> float:
    try:
        reward = abs(target - price)
        risk   = abs(price - stop)
        return round(reward / risk, 2) if risk > 0 else 0.0
    except (TypeError, ZeroDivisionError):
        return 0.0


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

    Returns
    -------
    dict with keys:
        direction, confidence, composite_score, target_price, stop_loss,
        rr_ratio, trend, trend_probability, ml_trained,
        patterns, reasons, supports, resistances, pivots, poc
    """
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

    # ── 2. Price action ───────────────────────────────────────────────────────
    try:
        pa_score, pa_reasons = score_price_action(df, sr)
    except Exception:
        pa_score, pa_reasons = 0.0, []

    try:
        patterns = detect_patterns(df)
    except Exception:
        patterns = []

    pattern_score, pattern_reasons = _score_patterns(patterns)

    # ── 3. Trend + probability ────────────────────────────────────────────────
    try:
        trend, trend_prob = analyze_trend_with_confidence(df)
    except Exception:
        trend, trend_prob = "SIDEWAYS", 0.5

    # ── 4. ML state ───────────────────────────────────────────────────────────
    # A model returning exactly 0.5 is untrained; exclude from composite
    ml_trained = abs(float(ml_prob) - 0.5) > 0.02
    ml_score   = float(np.clip((float(ml_prob) - 0.5) * 2, -1.0, 1.0)) if ml_trained else 0.0

    # ── 5. Composite score ────────────────────────────────────────────────────
    # When ML is untrained its weight (0.20) redistributes to tech and PA
    if ml_trained:
        w_tech, w_vol, w_ml, w_pa, w_pat, w_sent = 0.25, 0.15, 0.20, 0.18, 0.12, 0.05
    else:
        w_tech, w_vol, w_ml, w_pa, w_pat, w_sent = 0.30, 0.18,  0.0, 0.25, 0.17, 0.05

    # Direct trend bias: UPTREND pushes composite bullish, DOWNTREND bearish
    trend_signal = 1.0 if trend == "UPTREND" else (-1.0 if trend == "DOWNTREND" else 0.0)
    trend_bias   = 0.20 * trend_signal * trend_prob

    composite = (
        w_tech * float(tech_score)  +
        w_vol  * float(vol_score)   +
        w_ml   * ml_score           +
        w_pa   * pa_score           +
        w_pat  * pattern_score      +
        w_sent * float(sent_score)  +
        trend_bias
    )
    composite = round(float(np.clip(composite, -1.0, 1.0)), 4)

    # Enforce trend–direction consistency:
    # Never label a clear UPTREND stock as SELL / STRONG SELL and vice-versa
    if trend == "UPTREND"   and composite < 0.0:
        composite = max(composite, 0.0)
    if trend == "DOWNTREND" and composite > 0.0:
        composite = min(composite, 0.0)

    # ── 6. Direction and confidence ───────────────────────────────────────────
    direction  = _label_direction(composite)
    confidence = _compute_confidence(
        direction, trend, trend_prob,
        float(tech_score), float(vol_score),
        float(ml_prob), ml_trained,
        pa_score, pattern_score, float(sent_score),
    )

    # ── 7. Targets and stop-loss ──────────────────────────────────────────────
    is_bullish = composite >= 0
    if is_bullish:
        target    = round(resistance, 4)
        stop_loss = round(support * 0.998, 4)
    else:
        target    = round(support, 4)
        stop_loss = round(resistance * 1.002, 4)

    rr_ratio = _compute_rr(price, target, stop_loss)

    # ── 8. Trend reason ───────────────────────────────────────────────────────
    trend_reasons: list[str] = []
    if trend == "UPTREND":
        trend_reasons.append(
            f"Trend is UPTREND ({trend_prob*100:.0f}% signal agreement) — higher highs & higher lows"
        )
    elif trend == "DOWNTREND":
        trend_reasons.append(
            f"Trend is DOWNTREND ({trend_prob*100:.0f}% signal agreement) — lower highs & lower lows"
        )
    else:
        trend_reasons.append("Market is consolidating (SIDEWAYS) — no clear directional bias")

    # ── 9. Build reason list ──────────────────────────────────────────────────
    tech_reasons = _build_tech_reasons(last_row)
    vol_reasons  = _build_volume_reasons(last_row)
    ml_reasons   = _build_ml_reasons(float(ml_prob), ml_trained)

    all_reasons = trend_reasons + pa_reasons + pattern_reasons + tech_reasons + vol_reasons + ml_reasons

    seen: set[str] = set()
    deduped: list[str] = []
    for r in all_reasons:
        if r not in seen:
            seen.add(r)
            deduped.append(r)

    # ── 10. Return ────────────────────────────────────────────────────────────
    return {
        "direction":         direction,
        "confidence":        confidence,
        "composite_score":   composite,
        "target_price":      target,
        "stop_loss":         stop_loss,
        "rr_ratio":          rr_ratio,
        "trend":             trend,
        "trend_probability": round(float(trend_prob), 2),
        "ml_trained":        ml_trained,
        "patterns":          patterns,
        "reasons":           deduped[:8],
        "supports":          sr.get("supports", []),
        "resistances":       sr.get("resistances", []),
        "pivots":            sr.get("pivots", {}),
        "poc":               sr.get("poc", 0.0),
    }
