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

# ── Exhaustion / retest thresholds ────────────────────────────────────────────
_EXTENDED_PCT    = 0.015   # >1.5% from nearest S/R = extended move
_VOL_CLIMAX_MULT = 2.5     # last bar vol > 2.5× 20-bar avg = climax
_RSI_OB          = 70      # RSI overbought (bull exhaustion)
_RSI_OS          = 30      # RSI oversold   (bear exhaustion)

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
        "entry_type":       "IMMEDIATE",
        "retest_level":     0.0,
        "entry_zone_low":   0.0,
        "entry_zone_high":  0.0,
        "exhaustion_flags": [],
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
    mtf_score:    float = 0.0,
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

    # MTF alignment is the strongest standalone signal after trend
    _agree(float(mtf_score), 30)
    _agree(tech_score,       20)
    _agree(vol_score,        10)
    _agree(pa_score,         10)
    _agree(pattern_score,    8)
    _agree(sent_score,       5)

    if ml_trained:
        # ml_prob in [0,1]; convert to [-1,+1]: (prob-0.5)*2
        ml_signal = float(np.clip((ml_prob - 0.5) * 2, -1.0, 1.0))
        _agree(ml_signal, 20)
    # If untrained: ML weight is simply not included (total stays at 97)

    confidence = (votes / total * 100) if total > 0 else 50.0
    # NEUTRAL cap + absolute bounds
    return round(float(np.clip(confidence, 25.0, 95.0)), 1)


def _detect_exhaustion(
    price:     float,
    support:   float,
    resist:    float,
    df:        pd.DataFrame,
    last_row:  pd.Series,
    direction: str,
) -> dict:
    """
    Detect post-breakout exhaustion and determine optimal entry approach.

    Returns entry_type = "IMMEDIATE" when price is at a good entry zone,
    or "WAIT_RETEST" when the move is overextended and a pullback to key
    support/resistance is the higher-probability entry.

    A "WAIT_RETEST" call requires at least 2 of:
      - Price >1.5% from nearest S/R (extended)
      - Volume climax (last bar > 2.5× 20-bar average)
      - RSI overbought/oversold
      - Price at upper/lower Bollinger Band
    """
    base = {
        "entry_type":       "IMMEDIATE",
        "retest_level":     0.0,
        "entry_zone_low":   price,
        "entry_zone_high":  price,
        "exhaustion_flags": [],
        "adjusted_stop":    0.0,
        "adjusted_target":  0.0,
        "adjusted_rr":      0.0,
    }

    if df is None or df.empty or price <= 0:
        return base

    is_bull = direction in ("BUY", "STRONG BUY")
    is_bear = direction in ("SELL", "STRONG SELL")

    if not (is_bull or is_bear):
        return base

    flags: list[str] = []

    # ── 1. Distance from nearest S/R ──────────────────────────────────────────
    if is_bull and support > 0:
        pct = (price - support) / support
        if pct > _EXTENDED_PCT:
            flags.append(
                f"Price {pct*100:.1f}% above key support {support:.2f} — move extended"
            )
    if is_bear and resist > 0:
        pct = (resist - price) / resist
        if pct > _EXTENDED_PCT:
            flags.append(
                f"Price {pct*100:.1f}% below key resistance {resist:.2f} — move extended"
            )

    # ── 2. Volume climax ──────────────────────────────────────────────────────
    try:
        last_vol = float(df["Volume"].iloc[-1])
        avg_vol  = float(df["Volume"].iloc[-20:-1].mean())
        if avg_vol > 0 and last_vol > avg_vol * _VOL_CLIMAX_MULT:
            flags.append(
                f"Volume climax ({last_vol/avg_vol:.1f}× avg) — institutional selling into strength"
                if is_bull else
                f"Volume climax ({last_vol/avg_vol:.1f}× avg) — panic selling exhausted"
            )
    except Exception:
        pass

    # ── 3. RSI extreme ────────────────────────────────────────────────────────
    rsi = _safe_float(last_row, "rsi_14")
    if rsi is not None:
        if is_bull and rsi > _RSI_OB:
            flags.append(f"RSI {rsi:.1f} — overbought, pullback to support likely")
        elif is_bear and rsi < _RSI_OS:
            flags.append(f"RSI {rsi:.1f} — oversold, bounce to resistance likely")

    # ── 4. Bollinger Band extreme ─────────────────────────────────────────────
    bb_pct = _safe_float(last_row, "bb_pct")
    if bb_pct is not None:
        if is_bull and bb_pct > 0.92:
            flags.append("Price at upper Bollinger Band — statistically stretched")
        elif is_bear and bb_pct < 0.08:
            flags.append("Price at lower Bollinger Band — statistically stretched")

    # ── Decision: need ≥2 signals to call WAIT_RETEST ─────────────────────────
    if len(flags) < 2:
        return base

    if is_bull and support > 0:
        retest     = support
        entry_high = round(retest * 1.002, 4)   # slightly above support
        entry_low  = round(retest * 0.997, 4)   # allow wick through
        adj_stop   = round(retest * 0.995, 4)   # tight stop below support
        adj_rr     = _compute_rr(entry_high, resist, adj_stop)
        return {
            "entry_type":       "WAIT_RETEST",
            "retest_level":     round(retest, 4),
            "entry_zone_low":   entry_low,
            "entry_zone_high":  entry_high,
            "exhaustion_flags": flags,
            "adjusted_stop":    adj_stop,
            "adjusted_target":  round(resist, 4),
            "adjusted_rr":      adj_rr,
        }

    if is_bear and resist > 0:
        retest     = resist
        entry_low  = round(retest * 0.998, 4)
        entry_high = round(retest * 1.003, 4)
        adj_stop   = round(retest * 1.005, 4)
        adj_rr     = _compute_rr(entry_low, support, adj_stop)
        return {
            "entry_type":       "WAIT_RETEST",
            "retest_level":     round(retest, 4),
            "entry_zone_low":   entry_low,
            "entry_zone_high":  entry_high,
            "exhaustion_flags": flags,
            "adjusted_stop":    adj_stop,
            "adjusted_target":  round(support, 4),
            "adjusted_rr":      adj_rr,
        }

    return base


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
    mtf_score:  float = 0.0,
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
    # MTF (multi-timeframe alignment) is the dominant signal — professional
    # traders only enter when higher timeframes agree with the setup.
    # When ML is untrained its weight redistributes to tech and MTF.
    if ml_trained:
        # tech  vol   ml    pa    mtf   pat   sent
        w_t, w_v, w_m, w_p, w_f, w_pat, w_s = 0.18, 0.10, 0.15, 0.08, 0.30, 0.12, 0.07
    else:
        w_t, w_v, w_m, w_p, w_f, w_pat, w_s = 0.22, 0.12,  0.0, 0.12, 0.35, 0.12, 0.07

    # Direct trend bias: intraday trend acts as an additional nudge
    trend_signal = 1.0 if trend == "UPTREND" else (-1.0 if trend == "DOWNTREND" else 0.0)
    trend_bias   = 0.12 * trend_signal * trend_prob

    mtf_score_f = float(np.clip(float(mtf_score), -1.0, 1.0))

    composite = (
        w_t   * float(tech_score)  +
        w_v   * float(vol_score)   +
        w_m   * ml_score           +
        w_p   * pa_score           +
        w_f   * mtf_score_f        +
        w_pat * pattern_score      +
        w_s   * float(sent_score)  +
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
        mtf_score_f,
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

    # ── 7b. Exhaustion / retest check ─────────────────────────────────────────
    exhaustion = _detect_exhaustion(
        price, support, resistance, df, last_row, direction
    )
    # When waiting for retest, use retest-based R:R (better entry = better R:R)
    if exhaustion["entry_type"] == "WAIT_RETEST":
        if exhaustion["adjusted_rr"] > 0:
            rr_ratio  = exhaustion["adjusted_rr"]
        if exhaustion["adjusted_stop"] > 0:
            stop_loss = exhaustion["adjusted_stop"]
        if exhaustion["adjusted_target"] > 0:
            target = exhaustion["adjusted_target"]

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

    all_reasons = (
        exhaustion["exhaustion_flags"] +   # exhaustion warnings shown first
        trend_reasons + pa_reasons + pattern_reasons + tech_reasons + vol_reasons + ml_reasons
    )

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
        "reasons":           deduped[:10],
        "supports":          sr.get("supports", []),
        "resistances":       sr.get("resistances", []),
        "pivots":            sr.get("pivots", {}),
        "poc":               sr.get("poc", 0.0),
        "entry_type":        exhaustion["entry_type"],
        "retest_level":      exhaustion["retest_level"],
        "entry_zone_low":    exhaustion["entry_zone_low"],
        "entry_zone_high":   exhaustion["entry_zone_high"],
        "exhaustion_flags":  exhaustion["exhaustion_flags"],
    }
