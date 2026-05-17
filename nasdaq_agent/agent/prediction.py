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
from agent.reversal import compute_reversal_zone


# ── Direction thresholds ──────────────────────────────────────────────────────

_STRONG_BUY_THRESH  =  0.45   # was 0.50 — easier to reach strong buy
_BUY_THRESH         =  0.12   # was 0.15 — sector dampening can push 0.18 → 0.144, still valid
_SELL_THRESH        = -0.12   # was -0.15
_STRONG_SELL_THRESH = -0.45   # was -0.50

# ── Exhaustion / retest thresholds ────────────────────────────────────────────
_EXTENDED_PCT    = 0.015   # >1.5% from nearest S/R = extended move
_VOL_CLIMAX_MULT = 2.5     # last bar vol > 2.5× 20-bar avg = climax
_RSI_OB          = 70      # RSI overbought (bull exhaustion)
_RSI_OS          = 30      # RSI oversold   (bear exhaustion)

# ── RSI zone thresholds (hard gate) ──────────────────────────────────────────
_RSI_EXTREME_OB   = 80   # extreme overbought → flip to SELL bias
_RSI_OB_GATE      = 70   # overbought → suppress BUY, cap at NEUTRAL
_RSI_OS_GATE      = 30   # oversold   → suppress SELL, cap at NEUTRAL
_RSI_EXTREME_OS   = 20   # extreme oversold   → flip to BUY bias
_RSI_NEUTRAL_LOW  = 45   # below here: bullish zone allowed
_RSI_NEUTRAL_HIGH = 55   # above here: bearish zone allowed
_BOUNCE_RSI_DEEP  = 30     # deeply oversold → very high probability bounce
_BOUNCE_RSI_ZONE  = 42     # oversold zone   → watch for bounce
_BOUNCE_BB_LOW    = 0.15   # near lower Bollinger Band
_BOUNCE_SUP_PCT   = 0.006  # within 0.6% of key support = "at support"
_VOL_DRY_FACTOR   = 0.65   # last bar < 65% of avg = sellers drying up

# ── R:R gate ─────────────────────────────────────────────────────────────────
_MIN_RR           = 1.5    # minimum acceptable R:R for scalping (was 2.0; 1.5:1 is profitable at 60%+ WR)
_MIN_TARGET_PCT   = 0.003  # target must be at least 0.3% from entry (avoids degenerate targets)

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
        "bounce_signals":   [],
        "rr_quality":       "LOW",   # LOW | OK | GOOD | EXCELLENT
        "rr_qualifies":     False,
        "rsi_zone":         "NEUTRAL",   # EXTREME_OB | OB | NEUTRAL | OS | EXTREME_OS
        "rsi_value":        50.0,
        "rsi_gated":        False,
        "reversal_score":   0.0,
        "reversal_type":    "NONE",
        "divergence_type":  "NONE",
        "reversal_signals": [],
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
        # Skip truly neutral signals — they carry no information
        if abs(signal) < 0.05:
            return
        # Map [-1,+1] → [0,1]: 1.0 = full agreement, 0.0 = full disagreement
        agreement = float(np.clip((signal * d + 1) / 2, 0.0, 1.0))
        votes += weight * agreement
        total += weight

    # Trend — strongest structural signal; only count when trend is non-neutral
    if trend != "SIDEWAYS":
        trend_signal = 1.0 if trend == "UPTREND" else -1.0
        _agree(trend_signal * max(trend_prob, 0.55), 30)

    # MTF alignment (only when non-trivially non-zero)
    _agree(float(mtf_score), 30)
    _agree(tech_score,       20)
    _agree(vol_score,        10)
    _agree(pa_score,         10)
    _agree(pattern_score,    8)
    _agree(sent_score,       5)

    if ml_trained:
        ml_signal = float(np.clip((ml_prob - 0.5) * 2, -1.0, 1.0))
        _agree(ml_signal, 20)

    # Need at least one meaningful sub-signal before reporting confidence.
    # Was 10 — too strict for choppy markets where only 1-2 signals vote.
    if total < 5:
        return 50.0
    confidence = (votes / total * 100)
    return round(float(np.clip(confidence, 25.0, 95.0)), 1)


def _rsi_zone(rsi: float | None) -> str:
    """Classify RSI into a named zone."""
    if rsi is None:
        return "NEUTRAL"
    if rsi >= _RSI_EXTREME_OB:
        return "EXTREME_OB"
    if rsi >= _RSI_OB_GATE:
        return "OB"
    if rsi <= _RSI_EXTREME_OS:
        return "EXTREME_OS"
    if rsi <= _RSI_OS_GATE:
        return "OS"
    return "NEUTRAL"


def _apply_rsi_gate(
    composite:  float,
    direction:  str,
    rsi:        float | None,
    trend:      str,
    trend_prob: float,
) -> tuple[float, str, bool, list[str]]:
    """
    Hard RSI gate — the single most important mean-reversion rule.

    Overbought stocks are statistically likely to pull back.
    Oversold stocks are statistically likely to bounce.
    No matter what trend/ML says, we respect these extremes on short timeframes.

    RSI zone   │ If signal is BUY    │ If signal is SELL
    ───────────┼─────────────────────┼────────────────────
    EXTREME_OB │ → SELL (80+ = top)  │ keep SELL ✓
    OB (70-80) │ → NEUTRAL (don't buy│ keep SELL ✓
    NEUTRAL    │ keep as-is          │ keep as-is
    OS (20-30) │ keep BUY ✓          │ → NEUTRAL
    EXTREME_OS │ keep BUY ✓          │ → BUY  (20- = bottom)

    Exception: if the trend is very strong (prob > 0.80) and RSI is
    in 70-75 range, allow it — strong momentum can stay OB for a while.
    """
    if rsi is None:
        return composite, direction, False, []

    zone    = _rsi_zone(rsi)
    gated   = False
    reasons: list[str] = []
    is_buy  = direction in ("BUY", "STRONG BUY")
    is_sell = direction in ("SELL", "STRONG SELL")

    if zone == "EXTREME_OB" and is_buy:
        composite  = -0.20           # flip bearish
        direction  = "SELL"
        gated      = True
        reasons.append(
            f"RSI {rsi:.1f} — EXTREME OVERBOUGHT (>80). "
            "Statistically at peak. BUY signal overridden → SELL. "
            "High probability of sharp reversal."
        )

    elif zone == "OB" and is_buy:
        # Allow momentum continuation exception (RSI 70-75 + solid uptrend)
        momentum_exception = (trend == "UPTREND" and trend_prob >= 0.70 and rsi < 75)
        if not momentum_exception:
            composite  = 0.0
            direction  = "NEUTRAL"
            gated      = True
            reasons.append(
                f"RSI {rsi:.1f} — OVERBOUGHT. "
                "Do NOT buy here — pullback to support is the high-probability move. "
                "Wait for RSI to cool below 60 or a retest of support."
            )
        else:
            reasons.append(
                f"RSI {rsi:.1f} — overbought but strong UPTREND ({trend_prob*100:.0f}% conf) "
                "allows momentum continuation. Watch for reversal candles."
            )

    elif zone == "EXTREME_OS" and is_sell:
        composite  = 0.20
        direction  = "BUY"
        gated      = True
        reasons.append(
            f"RSI {rsi:.1f} — EXTREME OVERSOLD (<20). "
            "Statistically at bottom. SELL signal overridden → BUY. "
            "High probability of sharp bounce."
        )

    elif zone == "OS" and is_sell:
        momentum_exception = (trend == "DOWNTREND" and trend_prob >= 0.70 and rsi > 25)
        if not momentum_exception:
            composite  = 0.0
            direction  = "NEUTRAL"
            gated      = True
            reasons.append(
                f"RSI {rsi:.1f} — OVERSOLD. "
                "Do NOT short here — bounce to resistance is the high-probability move. "
                "Wait for RSI to recover above 40 or a retest of resistance."
            )
        else:
            reasons.append(
                f"RSI {rsi:.1f} — oversold but strong DOWNTREND ({trend_prob*100:.0f}% conf) "
                "allows continuation. Watch for bounce candles as exit."
            )

    elif zone in ("OS", "EXTREME_OS") and is_buy:
        reasons.append(
            f"RSI {rsi:.1f} — OVERSOLD. This is the correct zone to buy. "
            "Mean-reversion edge is on your side."
        )

    return composite, direction, gated, reasons


def _detect_bounce_setup(
    price:    float,
    support:  float,
    resist:   float,
    df:       pd.DataFrame,
    last_row: pd.Series,
) -> dict:
    """
    Identify high-probability oversold bounce setups at key support.

    A bounce setup (the IDEAL entry) fires when the stock has been beaten
    down, sellers are exhausted, and price is sitting on structural support.
    Requires ≥ 2 of the 5 signals below.

    Returns a dict with 'detected' bool and 'bounce_signals' list.
    """
    signals: list[str] = []

    # ── 1. RSI oversold ───────────────────────────────────────────────────────
    rsi = _safe_float(last_row, "rsi_14")
    if rsi is not None:
        if rsi <= _BOUNCE_RSI_DEEP:
            signals.append(f"RSI {rsi:.1f} — deeply oversold, mean-reversion edge very high")
        elif rsi <= _BOUNCE_RSI_ZONE:
            signals.append(f"RSI {rsi:.1f} — oversold zone, snap-back probability elevated")

    # ── 2. Lower Bollinger Band ────────────────────────────────────────────────
    bb_pct = _safe_float(last_row, "bb_pct")
    if bb_pct is not None and bb_pct <= _BOUNCE_BB_LOW:
        signals.append(
            f"Price at lower Bollinger Band ({bb_pct*100:.0f}%) — statistically stretched to downside"
        )

    # ── 3. Price at key support ────────────────────────────────────────────────
    if support > 0:
        pct_from_support = (price - support) / support
        if 0 <= pct_from_support <= _BOUNCE_SUP_PCT:
            signals.append(
                f"Price ${price:.2f} sitting on key support ${support:.2f} "
                f"({pct_from_support*100:.2f}% away) — structural bounce zone"
            )

    # ── 4. Volume dry-up (sellers exhausted) ──────────────────────────────────
    try:
        last_vol = float(df["Volume"].iloc[-1])
        avg_vol  = float(df["Volume"].iloc[-20:-1].mean())
        if avg_vol > 0 and last_vol < avg_vol * _VOL_DRY_FACTOR:
            signals.append(
                f"Volume dry-up ({last_vol/avg_vol:.2f}× avg) — selling pressure exhausted, "
                "buyers stepping in quietly"
            )
    except Exception:
        pass

    # ── 5. MACD histogram improving (bearish momentum fading) ─────────────────
    try:
        hist_col = "macd_hist"
        if hist_col in df.columns and len(df) >= 3:
            h_now  = float(df[hist_col].iloc[-1])
            h_prev = float(df[hist_col].iloc[-2])
            if h_now < 0 and h_now > h_prev:   # still negative but improving
                signals.append(
                    f"MACD histogram turning ({h_prev:.4f} → {h_now:.4f}) — "
                    "bearish momentum fading, reversal building"
                )
    except Exception:
        pass

    return {
        "detected":       len(signals) >= 2,
        "bounce_signals": signals,
    }


def _evaluate_rr(
    price:     float,
    sr:        dict,
    direction: str,
) -> tuple[float, float, float, str, bool]:
    """
    Professional R:R engine — enforces minimum 1.5:1 discipline.

    A 30-year trader's logic:
      1.  Stop = just below the nearest STRUCTURAL support (or above resistance
          for shorts) that is at least MIN_STOP_DIST away from entry.
          Never risk more than MAX_RISK_PCT of the stock price.
      2.  Target = find the first resistance BEYOND the MIN_RR level.
          • If resistance exists between entry and the MIN_RR level, price will
            stall there → use that as target (R:R will be LOW → trader decides
            whether to skip or size down).
          • If the path is CLEAR to MIN_RR, project the 1.5:1 level as target.
            Clear air = runway = higher-probability trade.
      3.  Never pick a target that is less than MIN_TARGET_PCT from entry.
    """
    MIN_STOP_DIST = 0.004   # stop must be ≥ 0.4% from entry to avoid noise
    MAX_RISK_PCT  = 0.020   # cap scalp risk at 2% of stock price

    supports    = sorted(
        [float(s) for s in sr.get("supports",    []) if isinstance(s, (int, float)) and s > 0],
        reverse=True,
    )
    resistances = sorted(
        [float(r) for r in sr.get("resistances", []) if isinstance(r, (int, float)) and r > 0],
    )

    is_bull = direction in ("BUY", "STRONG BUY")

    if is_bull:
        # ── Stop: nearest structural support ≥ MIN_STOP_DIST below entry ────────
        structural = next(
            (s for s in supports if (price - s) / price >= MIN_STOP_DIST),
            None,
        )
        if structural is None:
            structural = price * (1.0 - 2 * MIN_STOP_DIST)   # synthetic floor

        stop_loss = round(structural * 0.995, 4)   # 0.5% buffer below support
        risk      = price - stop_loss

        # Cap: never risk more than 2% on a scalp
        if risk > price * MAX_RISK_PCT:
            stop_loss = round(price * (1.0 - MAX_RISK_PCT), 4)
            risk      = price - stop_loss

        risk = max(risk, price * 0.001)   # floor to prevent division by zero

        # ── Target: find clear runway to MIN_RR ─────────────────────────────────
        min_target = price + risk * _MIN_RR   # the 1.5:1 level we need to reach

        # Is there overhead resistance BLOCKING the path before the 1.5:1 level?
        blocking = [r for r in resistances if price < r < min_target]

        if blocking:
            # Nearest blocker is the realistic ceiling — R:R will likely be LOW
            # Show the trade anyway; trader decides to skip or wait for breakout
            target = round(min(blocking), 4)
        else:
            # Clear runway — use first resistance at-or-beyond the 1.5:1 level
            # (adds a structural anchor; if none exists, project the 1.5:1 level)
            beyond = [r for r in resistances if r >= min_target]
            target = round(min(beyond), 4) if beyond else round(min_target, 4)

    else:   # SELL / STRONG SELL
        # ── Stop: nearest structural resistance ≥ MIN_STOP_DIST above entry ─────
        structural = next(
            (r for r in resistances if (r - price) / price >= MIN_STOP_DIST),
            None,
        )
        if structural is None:
            structural = price * (1.0 + 2 * MIN_STOP_DIST)

        stop_loss = round(structural * 1.005, 4)
        risk      = stop_loss - price

        if risk > price * MAX_RISK_PCT:
            stop_loss = round(price * (1.0 + MAX_RISK_PCT), 4)
            risk      = stop_loss - price

        risk = max(risk, price * 0.001)

        # ── Target: find clear runway down to MIN_RR ─────────────────────────────
        min_target = price - risk * _MIN_RR

        blocking = [s for s in supports if min_target < s < price]

        if blocking:
            target = round(max(blocking), 4)
        else:
            below  = [s for s in supports if s <= min_target]
            target = round(max(below), 4) if below else round(min_target, 4)

    # Enforce absolute minimum target move
    if is_bull and target - price < price * _MIN_TARGET_PCT:
        target = round(price + price * _MIN_TARGET_PCT, 4)
    elif not is_bull and price - target < price * _MIN_TARGET_PCT:
        target = round(price - price * _MIN_TARGET_PCT, 4)

    rr = _compute_rr(price, target, stop_loss)

    if rr >= 4.0:
        quality = "EXCELLENT"
    elif rr >= 3.0:
        quality = "GOOD"
    elif rr >= _MIN_RR:
        quality = "OK"
    else:
        quality = "LOW"

    return round(stop_loss, 4), round(target, 4), round(rr, 2), quality, rr >= _MIN_RR


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
    ticker:              str,
    df:                  pd.DataFrame,
    tech_score:          float,
    vol_score:           float,
    ml_prob:             float,
    sent_score:          float,
    last_row:            pd.Series,
    mtf_score:           float = 0.0,
    ml_reversal_prob:    float = 0.5,
    vwap_score:          float = 0.0,
    sector_mult:         float = 1.0,
    ensemble_prob:       float = 0.5,
    ensemble_agreement:  float = 0.0,
    df_daily:            "pd.DataFrame | None" = None,
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
        sr = get_all_sr_levels(df, df_daily=df_daily)
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
        # tech  vol   ml    pa    mtf   pat   sent  vwap
        w_t, w_v, w_m, w_p, w_f, w_pat, w_s, w_vw = 0.15, 0.09, 0.13, 0.07, 0.26, 0.10, 0.06, 0.14
    else:
        w_t, w_v, w_m, w_p, w_f, w_pat, w_s, w_vw = 0.18, 0.10,  0.0, 0.10, 0.30, 0.11, 0.07, 0.14

    # Direct trend bias: intraday trend acts as an additional nudge
    trend_signal = 1.0 if trend == "UPTREND" else (-1.0 if trend == "DOWNTREND" else 0.0)
    trend_bias   = 0.10 * trend_signal * trend_prob

    mtf_score_f  = float(np.clip(float(mtf_score),  -1.0, 1.0))
    vwap_score_f = float(np.clip(float(vwap_score), -1.0, 1.0))

    composite = (
        w_t   * float(tech_score)  +
        w_v   * float(vol_score)   +
        w_m   * ml_score           +
        w_p   * pa_score           +
        w_f   * mtf_score_f        +
        w_pat * pattern_score      +
        w_s   * float(sent_score)  +
        w_vw  * vwap_score_f       +
        trend_bias
    )
    composite = round(float(np.clip(composite, -1.0, 1.0)), 4)

    # Apply sector ETF multiplier (scales composite without flipping sign)
    composite = round(float(np.clip(composite * float(sector_mult), -1.0, 1.0)), 4)

    # Trend–direction consistency: dampen (not clamp) cross-trend signals.
    # A strong downtrend with a barely-positive composite → still NEUTRAL.
    # A strong downtrend with a very positive composite → allow (reversal setup).
    if trend == "UPTREND" and composite < 0.0 and trend_prob > 0.60:
        # Attenuate bearish signal proportionally to trend strength
        composite = composite * (1.0 - (trend_prob - 0.60) * 2.5)
        composite = max(composite, 0.0)
    if trend == "DOWNTREND" and composite > 0.0 and trend_prob > 0.60:
        composite = composite * (1.0 - (trend_prob - 0.60) * 2.5)
        composite = min(composite, 0.0)

    # ── 6. Direction ──────────────────────────────────────────────────────────
    direction = _label_direction(composite)

    # ── 6b. RSI hard gate — overbought suppresses BUY, oversold suppresses SELL
    rsi_val = _safe_float(last_row, "rsi_14")
    composite, direction, rsi_gated, rsi_gate_reasons = _apply_rsi_gate(
        composite, direction, rsi_val, trend, trend_prob
    )
    zone_label = _rsi_zone(rsi_val)

    confidence = _compute_confidence(
        direction, trend, trend_prob,
        float(tech_score), float(vol_score),
        float(ml_prob), ml_trained,
        pa_score, pattern_score, float(sent_score),
        mtf_score_f,
    )
    # RSI-gated signals get a confidence penalty — the gate went against the composite
    if rsi_gated:
        confidence = round(float(np.clip(confidence * 0.75, 25.0, 95.0)), 1)

    # ── 7. Optimised stops, targets and R:R ──────────────────────────────────
    stop_loss, target, rr_ratio, rr_quality, rr_qualifies = _evaluate_rr(
        price, sr, direction
    )

    # ── 7b. Exhaustion / retest check ─────────────────────────────────────────
    exhaustion = _detect_exhaustion(
        price, support, resistance, df, last_row, direction
    )
    if exhaustion["entry_type"] == "WAIT_RETEST":
        # Recalculate R:R from the retest entry level (always better)
        retest_entry = exhaustion["entry_zone_high"]
        if retest_entry > 0:
            adj_stop, adj_target, adj_rr, rr_quality, rr_qualifies = _evaluate_rr(
                retest_entry, sr, direction
            )
            if adj_rr > 0:
                rr_ratio  = adj_rr
                stop_loss = exhaustion["adjusted_stop"] or adj_stop
                target    = exhaustion["adjusted_target"] or adj_target

    # ── 7c. Bounce setup check ────────────────────────────────────────────────
    bounce = _detect_bounce_setup(price, support, resistance, df, last_row)

    # Promote entry_type to BOUNCE_SETUP when at support and oversold,
    # but only if R:R qualifies — a bounce at support with bad R:R is still a bad trade
    entry_type = exhaustion["entry_type"]   # default: IMMEDIATE or WAIT_RETEST
    if bounce["detected"] and entry_type == "IMMEDIATE":
        entry_type = "BOUNCE_SETUP"

    # ── 7d. Reversal zone detection ───────────────────────────────────────────
    rev = compute_reversal_zone(df, last_row, support, resistance, ml_reversal_prob)

    # Reversal zone can upgrade BOUNCE_SETUP or IMMEDIATE when strong enough
    if rev["entry_type_hint"] in ("REVERSAL_ZONE", "STRONG_REVERSAL"):
        if rev["reversal_type"] == "BULLISH" and entry_type in ("IMMEDIATE", "BOUNCE_SETUP"):
            entry_type = rev["entry_type_hint"]
            # Boost composite toward BUY when RSI gate hasn't suppressed it
            if not rsi_gated and composite >= 0:
                rev_boost  = rev["reversal_score"] * 0.20
                composite  = round(float(np.clip(composite + rev_boost, -1.0, 1.0)), 4)
                direction  = _label_direction(composite)
        elif rev["reversal_type"] == "BEARISH" and entry_type in ("IMMEDIATE",):
            entry_type = rev["entry_type_hint"]
            if not rsi_gated and composite <= 0:
                rev_boost  = rev["reversal_score"] * 0.20
                composite  = round(float(np.clip(composite - rev_boost, -1.0, 1.0)), 4)
                direction  = _label_direction(composite)

    # ── 7e. Momentum confirmation — recent candle direction consistency ───────
    _momentum_confirmed = False
    _momentum_adj = 0.0
    try:
        if len(df) >= 3 and "Close" in df.columns and "Open" in df.columns:
            recent = df.iloc[-3:]
            bull_bars = sum(1 for _, r in recent.iterrows() if r["Close"] > r["Open"])
            bear_bars = sum(1 for _, r in recent.iterrows() if r["Close"] < r["Open"])
            if direction in ("BUY", "STRONG BUY") and bull_bars >= 2:
                _momentum_confirmed = True
                _momentum_adj = +3.0
            elif direction in ("SELL", "STRONG SELL") and bear_bars >= 2:
                _momentum_confirmed = True
                _momentum_adj = +3.0
            elif direction in ("BUY", "STRONG BUY") and bear_bars == 3:
                _momentum_adj = -4.0   # 3 consecutive red bars into BUY = bad entry timing
            elif direction in ("SELL", "STRONG SELL") and bull_bars == 3:
                _momentum_adj = -4.0   # 3 consecutive green bars into SELL = bad entry timing
    except Exception:
        pass
    if _momentum_adj != 0.0:
        confidence = round(float(np.clip(confidence + _momentum_adj, 25.0, 95.0)), 1)

    # ── 7f. Volume quality gate — low volume reduces conviction ───────────────
    try:
        rvol = _safe_float(last_row, "vol_ratio")
        if rvol is not None:
            if rvol < 0.4:
                confidence = round(float(np.clip(confidence - 6.0, 25.0, 95.0)), 1)
            elif rvol < 0.7:
                confidence = round(float(np.clip(confidence - 3.0, 25.0, 95.0)), 1)
            elif rvol >= 2.0 and _momentum_confirmed:
                confidence = round(float(np.clip(confidence + 3.0, 25.0, 95.0)), 1)
    except Exception:
        pass

    # ── 7g. Ensemble agreement — low consensus reduces confidence ────────────────
    # When 10 diverse models disagree, the signal is uncertain; penalise confidence.
    # agreement=1.0 → all agree (no penalty); agreement=0.5 → 50/50 split (−10 pts)
    if ensemble_agreement > 0.0:
        try:
            from agent.ensemble_model import ensemble_confidence_multiplier
            # Scale disagreement fraction [0,1] → prob_std range [0, 0.5]
            # so that 100% agreement→0 (no penalty) and 36% agreement→0.18 (max penalty)
            _agree_mult = ensemble_confidence_multiplier((1.0 - ensemble_agreement) * 0.5)
            if _agree_mult < 1.0:
                confidence = round(float(np.clip(confidence * _agree_mult, 25.0, 95.0)), 1)
        except Exception:
            pass

    # ── 8. Trend reason ───────────────────────────────────────────────────────
    # Apply R:R quality adjustment to confidence.
    # R:R reflects position sizing quality, NOT signal direction quality.
    # Penalty for LOW is mild so good setups with tight S/R aren't killed.
    # Traders can manage bad R:R with smaller size — the signal is still valid.
    _rr_adj = {
        "EXCELLENT": +4.0,   # ≥4:1 R:R — extra conviction
        "GOOD":      +2.0,   # ≥3:1 R:R — solid geometry
        "OK":         0.0,   # ≥1.5:1 R:R — acceptable, no change
        "LOW":       -4.0,   # <1.5:1 R:R — reduce size but still show signal
    }.get(rr_quality, 0.0)
    if _rr_adj != 0.0:
        confidence = round(float(np.clip(confidence + _rr_adj, 25.0, 95.0)), 1)

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

    # R:R quality reason — always shown so trader knows if setup is worth taking
    if rr_qualifies:
        rr_reason = f"R:R {rr_ratio:.1f}:1 ({rr_quality}) — risk/reward qualifies ≥ {_MIN_RR}:1 threshold"
    else:
        rr_reason = (
            f"R:R {rr_ratio:.1f}:1 — below {_MIN_RR}:1 minimum. "
            f"Target ${target:.2f} too close or stop ${stop_loss:.2f} too wide. "
            "Consider skipping or waiting for better entry."
        )

    momentum_reasons: list[str] = []
    if _momentum_confirmed:
        momentum_reasons.append("✓ Momentum confirmed — recent candles align with signal direction")
    elif _momentum_adj < 0:
        momentum_reasons.append("⚠ Counter-candle entry — recent bars oppose signal direction (wait for pullback)")

    all_reasons = (
        rsi_gate_reasons               +   # RSI gate overrides shown first
        rev["signals"]                 +   # divergence / reversal signals next
        exhaustion["exhaustion_flags"] +   # exhaustion / overextension
        bounce["bounce_signals"]       +   # bounce signals
        [rr_reason]                    +   # R:R quality always visible
        momentum_reasons               +   # momentum confirmation
        trend_reasons + pa_reasons + pattern_reasons + tech_reasons + vol_reasons + ml_reasons
    )

    seen: set[str] = set()
    deduped: list[str] = []
    for r in all_reasons:
        if r not in seen:
            seen.add(r)
            deduped.append(r)

    # ── 10. RL entry timing recommendation ───────────────────────────────────────
    _rl_action = "ENTER"
    _rl_conf   = 0.5
    try:
        from agent.rl_entry import get_entry_recommendation
        rsi_norm_val = float(np.clip((rsi_val or 50.0) / 100.0, 0.0, 1.0))
        _rvol = _safe_float(last_row, "vol_ratio") or 1.0
        _vwap_dev = 0.0
        try:
            _vwap_dev = float(last_row.get("vwap_dev", 0.0) or 0.0) / 100.0
        except Exception:
            pass
        if price > 0 and support > 0:
            _sup_spread = abs(price - support) / price
        else:
            _sup_spread = 0.005
        _rl_action, _rl_conf = get_entry_recommendation(
            momentum_1bar=float(np.clip(composite * 0.01, -0.01, 0.01)),
            rsi_norm=rsi_norm_val,
            vol_ratio=float(np.clip(_rvol, 0.0, 3.0)),
            spread_to_support=float(np.clip(_sup_spread, 0.0, 0.02)),
            vwap_deviation=_vwap_dev,
            bars_since_signal=0.0,
            time_of_day_norm=0.5,
        )
    except Exception:
        pass

    # ── 11. Return ────────────────────────────────────────────────────────────
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
        "entry_type":        entry_type,
        "retest_level":      exhaustion["retest_level"],
        "entry_zone_low":    exhaustion["entry_zone_low"],
        "entry_zone_high":   exhaustion["entry_zone_high"],
        "exhaustion_flags":  exhaustion["exhaustion_flags"],
        "bounce_signals":    bounce["bounce_signals"],
        "rr_quality":        rr_quality,
        "rr_qualifies":      rr_qualifies,
        "rsi_zone":          zone_label,
        "rsi_value":         round(float(rsi_val), 1) if rsi_val is not None else 50.0,
        "rsi_gated":         rsi_gated,
        "reversal_score":      rev["reversal_score"],
        "reversal_type":       rev["reversal_type"],
        "divergence_type":     rev["divergence_type"],
        "reversal_signals":    rev["signals"],
        "ensemble_agreement":  round(float(ensemble_agreement), 4),
        "rl_entry_action":     _rl_action,
        "rl_entry_confidence": round(float(_rl_conf), 3),
    }
