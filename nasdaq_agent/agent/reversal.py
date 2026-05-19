"""
Reversal zone detection engine.

Identifies high-probability price reversal zones using:
  1. RSI divergence      — price disagrees with momentum (leading indicator)
  2. MACD divergence     — histogram turns before price does
  3. Volume exhaustion   — selling/buying volume collapsing at extremes
  4. Wick analysis       — pin bars / hammers signal rejection
  5. Multi-signal score  — all signals aggregated into [0, 1] reversal probability

A reversal zone requires CONFLUENCE — the more signals agree, the higher
the probability. Used in conjunction with the RSI gate: only BULLISH reversal
zones override NEUTRAL to BUY; BEARISH only at overbought.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_DIV_LOOKBACK      = 30    # bars to scan for divergence swings
_DIV_SWING_WINDOW  = 3     # bars each side to confirm a swing high/low
_DIV_MIN_STRENGTH  = 0.04  # minimum divergence gap to be meaningful (4%)
_REVERSAL_THRESH   = 0.55  # score ≥ 0.55 → REVERSAL_ZONE entry_type
_STRONG_REV_THRESH = 0.72  # score ≥ 0.72 → STRONG_REVERSAL

_WICK_RATIO_MIN    = 0.40  # lower wick must be ≥ 40% of bar range for hammer
_VOL_DRY_FACTOR    = 0.65  # last bar vol < 65% of 20-bar avg = drying up
_MFI_OS            = 25    # Money Flow Index oversold threshold
_STOCH_OS          = 20    # Stochastic oversold threshold
_CCI_OS            = -100  # CCI oversold threshold


# ── Divergence helpers ────────────────────────────────────────────────────────

def _find_swing_lows(series: np.ndarray, window: int = _DIV_SWING_WINDOW) -> list[tuple[int, float]]:
    """Return (index, value) of local minima in series."""
    lows = []
    for i in range(window, len(series) - window):
        segment = series[i - window: i + window + 1]
        if series[i] == segment.min():
            lows.append((i, float(series[i])))
    return lows


def _find_swing_highs(series: np.ndarray, window: int = _DIV_SWING_WINDOW) -> list[tuple[int, float]]:
    """Return (index, value) of local maxima in series."""
    highs = []
    for i in range(window, len(series) - window):
        segment = series[i - window: i + window + 1]
        if series[i] == segment.max():
            highs.append((i, float(series[i])))
    return highs


def detect_rsi_divergence(df: pd.DataFrame) -> tuple[str, float, str]:
    """
    Detect CLASSIC RSI divergence over the last _DIV_LOOKBACK bars.

    Bullish divergence: price makes lower low, RSI makes higher low
    → hidden buying pressure, reversal up likely.

    Bearish divergence: price makes higher high, RSI makes lower high
    → hidden selling pressure, reversal down likely.

    Returns
    -------
    (divergence_type, strength, description)
      divergence_type : "BULLISH" | "BEARISH" | "NONE"
      strength        : 0.0–1.0
      description     : human-readable explanation
    """
    if df is None or len(df) < _DIV_LOOKBACK + 5:
        return "NONE", 0.0, ""

    if "rsi_14" not in df.columns:
        return "NONE", 0.0, ""

    recent = df.tail(_DIV_LOOKBACK)
    prices = recent["Close"].values.astype(float)
    rsi    = recent["rsi_14"].values.astype(float)

    # ── Bullish: look at price lows vs RSI lows ───────────────────────────────
    price_lows = _find_swing_lows(prices)
    rsi_lows   = _find_swing_lows(rsi)

    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        # Compare the two most recent lows
        p1, p2 = price_lows[-2][1], price_lows[-1][1]
        r1, r2 = rsi_lows[-2][1],   rsi_lows[-1][1]

        price_diff = (p1 - p2) / max(p1, 1e-9)   # how much lower? (positive = lower)
        rsi_diff   = (r2 - r1) / max(abs(r1), 1e-9)  # how much higher? (positive = higher)

        if p2 < p1 and r2 > r1 and price_diff > _DIV_MIN_STRENGTH:
            strength = float(np.clip((price_diff + rsi_diff) / 2, 0.0, 1.0))
            desc = (
                f"Bullish RSI divergence: price made lower low "
                f"(${p1:.2f} → ${p2:.2f}, -{price_diff*100:.1f}%) "
                f"but RSI made higher low ({r1:.1f} → {r2:.1f}, +{rsi_diff*100:.1f}%) "
                "— hidden buying pressure, reversal up expected"
            )
            return "BULLISH", round(strength, 3), desc

    # ── Bearish: look at price highs vs RSI highs ─────────────────────────────
    price_highs = _find_swing_highs(prices)
    rsi_highs   = _find_swing_highs(rsi)

    if len(price_highs) >= 2 and len(rsi_highs) >= 2:
        p1, p2 = price_highs[-2][1], price_highs[-1][1]
        r1, r2 = rsi_highs[-2][1],   rsi_highs[-1][1]

        price_diff = (p2 - p1) / max(p1, 1e-9)   # how much higher?
        rsi_diff   = (r1 - r2) / max(abs(r1), 1e-9)  # how much lower?

        if p2 > p1 and r2 < r1 and price_diff > _DIV_MIN_STRENGTH:
            strength = float(np.clip((price_diff + rsi_diff) / 2, 0.0, 1.0))
            desc = (
                f"Bearish RSI divergence: price made higher high "
                f"(${p1:.2f} → ${p2:.2f}, +{price_diff*100:.1f}%) "
                f"but RSI made lower high ({r1:.1f} → {r2:.1f}, -{rsi_diff*100:.1f}%) "
                "— hidden selling pressure, reversal down expected"
            )
            return "BEARISH", round(strength, 3), desc

    return "NONE", 0.0, ""


def detect_hidden_rsi_divergence(df: pd.DataFrame) -> tuple[str, float, str]:
    """
    Detect HIDDEN RSI divergence — a CONTINUATION signal (not reversal).

    Hidden bullish: price makes HIGHER LOW + RSI makes LOWER LOW
    → uptrend is intact with hidden strength; continuation up expected.

    Hidden bearish: price makes LOWER HIGH + RSI makes HIGHER HIGH
    → downtrend is intact with hidden strength; continuation down expected.

    Hidden divergence confirms the primary trend and is the professional
    trader's preferred entry on pullbacks within a trend.

    Returns
    -------
    (divergence_type, strength, description)
      divergence_type : "HIDDEN_BULL" | "HIDDEN_BEAR" | "NONE"
      strength        : 0.0–1.0
      description     : human-readable explanation
    """
    if df is None or len(df) < _DIV_LOOKBACK + 5:
        return "NONE", 0.0, ""
    if "rsi_14" not in df.columns:
        return "NONE", 0.0, ""

    recent = df.tail(_DIV_LOOKBACK)
    prices = recent["Close"].values.astype(float)
    rsi    = recent["rsi_14"].values.astype(float)

    price_lows = _find_swing_lows(prices)
    rsi_lows   = _find_swing_lows(rsi)

    # ── Hidden bullish: price higher low, RSI lower low ───────────────────────
    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        p1, p2 = price_lows[-2][1], price_lows[-1][1]
        r1, r2 = rsi_lows[-2][1],   rsi_lows[-1][1]
        price_diff = (p2 - p1) / max(abs(p1), 1e-9)   # positive = higher low
        rsi_diff   = (r1 - r2) / max(abs(r1), 1e-9)   # positive = lower low

        if p2 > p1 and r2 < r1 and price_diff > _DIV_MIN_STRENGTH * 0.5:
            strength = float(np.clip((price_diff + rsi_diff) / 2, 0.0, 1.0))
            desc = (
                f"Hidden bullish RSI divergence: price made higher low "
                f"(${p1:.2f} → ${p2:.2f}, +{price_diff*100:.1f}%) "
                f"but RSI made lower low ({r1:.1f} → {r2:.1f}) — "
                "uptrend continuation confirmed, buy the dip"
            )
            return "HIDDEN_BULL", round(strength, 3), desc

    price_highs = _find_swing_highs(prices)
    rsi_highs   = _find_swing_highs(rsi)

    # ── Hidden bearish: price lower high, RSI higher high ─────────────────────
    if len(price_highs) >= 2 and len(rsi_highs) >= 2:
        p1, p2 = price_highs[-2][1], price_highs[-1][1]
        r1, r2 = rsi_highs[-2][1],   rsi_highs[-1][1]
        price_diff = (p1 - p2) / max(abs(p1), 1e-9)   # positive = lower high
        rsi_diff   = (r2 - r1) / max(abs(r1), 1e-9)   # positive = higher high

        if p2 < p1 and r2 > r1 and price_diff > _DIV_MIN_STRENGTH * 0.5:
            strength = float(np.clip((price_diff + rsi_diff) / 2, 0.0, 1.0))
            desc = (
                f"Hidden bearish RSI divergence: price made lower high "
                f"(${p1:.2f} → ${p2:.2f}, -{price_diff*100:.1f}%) "
                f"but RSI made higher high ({r1:.1f} → {r2:.1f}) — "
                "downtrend continuation confirmed, sell the bounce"
            )
            return "HIDDEN_BEAR", round(strength, 3), desc

    return "NONE", 0.0, ""


def detect_macd_divergence(df: pd.DataFrame) -> tuple[str, float, str]:
    """
    Detect MACD histogram divergence — histogram turns BEFORE price does.

    Bullish: price lower low + MACD histogram less negative (improving)
    Bearish: price higher high + MACD histogram less positive (weakening)
    """
    if df is None or len(df) < _DIV_LOOKBACK + 5:
        return "NONE", 0.0, ""

    if "macd_hist" not in df.columns:
        return "NONE", 0.0, ""

    recent    = df.tail(_DIV_LOOKBACK)
    prices    = recent["Close"].values.astype(float)
    macd_hist = recent["macd_hist"].values.astype(float)

    price_lows  = _find_swing_lows(prices)
    hist_lows   = _find_swing_lows(macd_hist)

    if len(price_lows) >= 2 and len(hist_lows) >= 2:
        p1, p2 = price_lows[-2][1],  price_lows[-1][1]
        h1, h2 = hist_lows[-2][1],   hist_lows[-1][1]
        price_diff = (p1 - p2) / max(abs(p1), 1e-9)
        if p2 < p1 and h2 > h1 and price_diff > _DIV_MIN_STRENGTH:
            strength = float(np.clip(abs(h2 - h1) * 10, 0.0, 1.0))
            desc = (
                f"Bullish MACD divergence: price lower low but histogram improving "
                f"({h1:.4f} → {h2:.4f}) — momentum reversing before price"
            )
            return "BULLISH", round(strength, 3), desc

    price_highs = _find_swing_highs(prices)
    hist_highs  = _find_swing_highs(macd_hist)

    if len(price_highs) >= 2 and len(hist_highs) >= 2:
        p1, p2 = price_highs[-2][1], price_highs[-1][1]
        h1, h2 = hist_highs[-2][1],  hist_highs[-1][1]
        price_diff = (p2 - p1) / max(abs(p1), 1e-9)
        if p2 > p1 and h2 < h1 and price_diff > _DIV_MIN_STRENGTH:
            strength = float(np.clip(abs(h1 - h2) * 10, 0.0, 1.0))
            desc = (
                f"Bearish MACD divergence: price higher high but histogram weakening "
                f"({h1:.4f} → {h2:.4f}) — momentum fading before price"
            )
            return "BEARISH", round(strength, 3), desc

    return "NONE", 0.0, ""


# ── Wick analysis (pin bar / hammer) ─────────────────────────────────────────

def _wick_analysis(df: pd.DataFrame) -> tuple[str, float, str]:
    """
    Analyse the last candle's wick structure for reversal signals.

    Hammer (bullish):     long lower wick ≥ 40% of range, small body at top
    Shooting star (bear): long upper wick ≥ 40% of range, small body at bottom
    Doji at extreme:      open ≈ close at key level = indecision / reversal
    """
    if df is None or len(df) < 2:
        return "NONE", 0.0, ""
    try:
        last   = df.iloc[-1]
        o, h, l, c = float(last["Open"]), float(last["High"]), float(last["Low"]), float(last["Close"])
        rng = h - l
        if rng < 1e-9:
            return "NONE", 0.0, ""

        body        = abs(c - o)
        lower_wick  = min(o, c) - l
        upper_wick  = h - max(o, c)
        body_ratio  = body / rng
        lower_ratio = lower_wick / rng
        upper_ratio = upper_wick / rng

        if lower_ratio >= _WICK_RATIO_MIN and body_ratio <= 0.35:
            strength = float(np.clip(lower_ratio, 0.0, 1.0))
            return (
                "BULLISH", strength,
                f"Hammer candle: lower wick {lower_ratio*100:.0f}% of range — "
                "buyers aggressively rejected lower prices, reversal signal"
            )

        if upper_ratio >= _WICK_RATIO_MIN and body_ratio <= 0.35:
            strength = float(np.clip(upper_ratio, 0.0, 1.0))
            return (
                "BEARISH", strength,
                f"Shooting star: upper wick {upper_ratio*100:.0f}% of range — "
                "sellers aggressively rejected higher prices, reversal signal"
            )

        if body_ratio <= 0.12:
            return "DOJI", 0.3, "Doji candle — indecision at current level, potential reversal"

    except Exception:
        pass
    return "NONE", 0.0, ""


# ── Multi-oscillator oversold/overbought checks ───────────────────────────────

def _multi_oscillator_signals(last_row: pd.Series) -> list[tuple[str, str]]:
    """
    Check Stochastic, CCI, MFI for oversold/overbought confirmations.
    Returns list of (direction, description) tuples.
    """
    signals = []
    try:
        stoch_k = float(last_row.get("stoch_k", 50))
        stoch_d = float(last_row.get("stoch_d", 50))
        if stoch_k < _STOCH_OS and stoch_d < _STOCH_OS:
            signals.append(("BULLISH", f"Stochastic K={stoch_k:.0f} D={stoch_d:.0f} — both oversold, bullish cross imminent"))
        elif stoch_k > 80 and stoch_d > 80:
            signals.append(("BEARISH", f"Stochastic K={stoch_k:.0f} D={stoch_d:.0f} — both overbought, bearish cross imminent"))
    except Exception:
        pass
    try:
        cci = float(last_row.get("cci_20", 0))
        if cci < _CCI_OS:
            signals.append(("BULLISH", f"CCI {cci:.0f} — extreme oversold (<-100), mean reversion due"))
        elif cci > 100:
            signals.append(("BEARISH", f"CCI {cci:.0f} — extreme overbought (>100), reversal likely"))
    except Exception:
        pass
    try:
        mfi = float(last_row.get("mfi_14", 50))
        if mfi < _MFI_OS:
            signals.append(("BULLISH", f"Money Flow Index {mfi:.0f} — selling exhausted, buyers returning"))
        elif mfi > 75:
            signals.append(("BEARISH", f"Money Flow Index {mfi:.0f} — buyers exhausted, sellers taking over"))
    except Exception:
        pass
    return signals


# ── Volume exhaustion ─────────────────────────────────────────────────────────

def _volume_exhaustion(df: pd.DataFrame, last_row: pd.Series) -> tuple[str, float, str]:
    """Detect volume drying up at a price extreme (sellers/buyers exhausted)."""
    try:
        last_vol = float(df["Volume"].iloc[-1])
        avg_vol  = float(df["Volume"].iloc[-20:-1].mean())
        if avg_vol <= 0:
            return "NONE", 0.0, ""
        ratio = last_vol / avg_vol
        if ratio < _VOL_DRY_FACTOR:
            # Volume drying up — check if price is near extreme
            rsi = float(last_row.get("rsi_14", 50))
            if rsi < 40:
                return (
                    "BULLISH", float(np.clip(1.0 - ratio, 0.0, 1.0)),
                    f"Volume exhaustion ({ratio:.2f}× avg) at oversold RSI {rsi:.0f} — "
                    "sellers leaving the market, supply drying up"
                )
            elif rsi > 60:
                return (
                    "BEARISH", float(np.clip(1.0 - ratio, 0.0, 1.0)),
                    f"Volume exhaustion ({ratio:.2f}× avg) at overbought RSI {rsi:.0f} — "
                    "buyers leaving the market, demand drying up"
                )
    except Exception:
        pass
    return "NONE", 0.0, ""


# ── Main reversal zone scorer ─────────────────────────────────────────────────

def compute_reversal_zone(
    df:        pd.DataFrame,
    last_row:  pd.Series,
    support:   float,
    resist:    float,
    ml_reversal_prob: float = 0.5,
) -> dict:
    """
    Aggregate all reversal signals into a single reversal zone assessment.

    Parameters
    ----------
    df               : OHLCV + indicators DataFrame (1-min bars)
    last_row         : last row of the indicators DataFrame
    support          : nearest support price
    resist           : nearest resistance price
    ml_reversal_prob : probability from ReversalMLModel (0.5 = untrained)

    Returns
    -------
    dict with keys:
        reversal_score   : float [0, 1]
        reversal_type    : "BULLISH" | "BEARISH" | "NONE"
        entry_type_hint  : "REVERSAL_ZONE" | "STRONG_REVERSAL" | "IMMEDIATE"
        divergence_type  : "BULLISH" | "BEARISH" | "NONE"
        signals          : list[str]  — human-readable confirmations
    """
    result = {
        "reversal_score":  0.0,
        "reversal_type":   "NONE",
        "entry_type_hint": "IMMEDIATE",
        "divergence_type": "NONE",
        "signals":         [],
    }

    if df is None or df.empty or len(df) < 20:
        return result

    signals_bull: list[tuple[float, str]] = []   # (weight, description)
    signals_bear: list[tuple[float, str]] = []

    price = float(last_row.get("Close", 0))
    if price <= 0:
        return result

    # ── 1. Classic RSI divergence (highest weight) ───────────────────────────
    rsi_div_type, rsi_div_str, rsi_div_desc = detect_rsi_divergence(df)
    if rsi_div_type == "BULLISH" and rsi_div_desc:
        signals_bull.append((0.30, rsi_div_desc))
        result["divergence_type"] = "BULLISH"
    elif rsi_div_type == "BEARISH" and rsi_div_desc:
        signals_bear.append((0.30, rsi_div_desc))
        result["divergence_type"] = "BEARISH"

    # ── 1b. Hidden RSI divergence (continuation — confirms trend direction) ──
    hidden_div_type, hidden_div_str, hidden_div_desc = detect_hidden_rsi_divergence(df)
    if hidden_div_type == "HIDDEN_BULL" and hidden_div_desc:
        # Hidden bull = uptrend continuation; only meaningful if price is pulling back
        rsi_now = float(last_row.get("rsi_14", 50))
        if rsi_now < 55:   # on a pullback (not already overbought)
            signals_bull.append((0.15, hidden_div_desc))
            if result["divergence_type"] == "NONE":
                result["divergence_type"] = "HIDDEN_BULL"
    elif hidden_div_type == "HIDDEN_BEAR" and hidden_div_desc:
        rsi_now = float(last_row.get("rsi_14", 50))
        if rsi_now > 45:   # on a bounce (not already oversold)
            signals_bear.append((0.15, hidden_div_desc))
            if result["divergence_type"] == "NONE":
                result["divergence_type"] = "HIDDEN_BEAR"

    # ── 2. MACD divergence ────────────────────────────────────────────────────
    macd_div_type, macd_div_str, macd_div_desc = detect_macd_divergence(df)
    if macd_div_type == "BULLISH" and macd_div_desc:
        signals_bull.append((0.20, macd_div_desc))
    elif macd_div_type == "BEARISH" and macd_div_desc:
        signals_bear.append((0.20, macd_div_desc))

    # ── 3. Volume exhaustion ──────────────────────────────────────────────────
    vol_dir, vol_str, vol_desc = _volume_exhaustion(df, last_row)
    if vol_dir == "BULLISH" and vol_desc:
        signals_bull.append((0.15, vol_desc))
    elif vol_dir == "BEARISH" and vol_desc:
        signals_bear.append((0.15, vol_desc))

    # ── 4. Wick / candle structure ────────────────────────────────────────────
    wick_dir, wick_str, wick_desc = _wick_analysis(df)
    if wick_dir == "BULLISH" and wick_desc:
        signals_bull.append((0.12 * wick_str, wick_desc))
    elif wick_dir == "BEARISH" and wick_desc:
        signals_bear.append((0.12 * wick_str, wick_desc))
    elif wick_dir == "DOJI" and wick_desc:
        # Doji is direction-agnostic; add to whichever side is stronger
        rsi_val = float(last_row.get("rsi_14", 50))
        if rsi_val < 45:
            signals_bull.append((0.05, wick_desc))
        elif rsi_val > 55:
            signals_bear.append((0.05, wick_desc))

    # ── 5. Multi-oscillator oversold/overbought ───────────────────────────────
    for osc_dir, osc_desc in _multi_oscillator_signals(last_row):
        if osc_dir == "BULLISH":
            signals_bull.append((0.08, osc_desc))
        elif osc_dir == "BEARISH":
            signals_bear.append((0.08, osc_desc))

    # ── 6. Price proximity to key S/R ────────────────────────────────────────
    if support > 0:
        pct = (price - support) / support
        if 0 <= pct <= 0.008:   # within 0.8% of support
            signals_bull.append((
                0.12 * (1 - pct / 0.008),
                f"Price ${price:.2f} at key support ${support:.2f} "
                f"({pct*100:.2f}% away) — structural reversal zone"
            ))
    if resist > 0:
        pct = (resist - price) / resist
        if 0 <= pct <= 0.008:
            signals_bear.append((
                0.12 * (1 - pct / 0.008),
                f"Price ${price:.2f} at key resistance ${resist:.2f} "
                f"({pct*100:.2f}% away) — structural rejection zone"
            ))

    # ── 7. ML reversal probability ────────────────────────────────────────────
    ml_trained = abs(ml_reversal_prob - 0.5) > 0.03
    if ml_trained:
        ml_signal = (ml_reversal_prob - 0.5) * 2   # [-1, +1]
        if ml_signal > 0.1:
            signals_bull.append((
                0.15 * float(np.clip(ml_signal, 0, 1)),
                f"Reversal ML model: {ml_reversal_prob*100:.0f}% probability of bullish reversal"
            ))
        elif ml_signal < -0.1:
            signals_bear.append((
                0.15 * float(np.clip(-ml_signal, 0, 1)),
                f"Reversal ML model: {(1-ml_reversal_prob)*100:.0f}% probability of bearish reversal"
            ))

    # ── Aggregate ─────────────────────────────────────────────────────────────
    bull_score = float(np.clip(sum(w for w, _ in signals_bull), 0.0, 1.0))
    bear_score = float(np.clip(sum(w for w, _ in signals_bear), 0.0, 1.0))

    if bull_score >= bear_score and bull_score >= 0.20:
        rev_type    = "BULLISH"
        rev_score   = bull_score
        rev_signals = [d for _, d in signals_bull]
    elif bear_score > bull_score and bear_score >= 0.20:
        rev_type    = "BEARISH"
        rev_score   = bear_score
        rev_signals = [d for _, d in signals_bear]
    else:
        return result   # no meaningful reversal signal

    entry_hint = "IMMEDIATE"
    if rev_score >= _STRONG_REV_THRESH:
        entry_hint = "STRONG_REVERSAL"
    elif rev_score >= _REVERSAL_THRESH:
        entry_hint = "REVERSAL_ZONE"

    result.update({
        "reversal_score":  round(rev_score, 3),
        "reversal_type":   rev_type,
        "entry_type_hint": entry_hint,
        "signals":         rev_signals,
    })
    return result


# ── Reversal feature engineering (for ReversalMLModel) ───────────────────────

def compute_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer reversal-specific features on top of the standard indicators.
    Called during ReversalMLModel training and inference.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # RSI slope (momentum of momentum)
    if "rsi_14" in df.columns:
        df["rsi_slope_3"] = df["rsi_14"].diff(3)
        df["rsi_slope_5"] = df["rsi_14"].diff(5)

    # Price slope — clip to ±50% to guard against pct_change on zero-price bars
    if "Close" in df.columns:
        df["price_slope_3"] = df["Close"].pct_change(3).replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5).fillna(0.0)
        df["price_slope_5"] = df["Close"].pct_change(5).replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5).fillna(0.0)

    # RSI / price divergence score (rolling window)
    if "rsi_14" in df.columns and "Close" in df.columns:
        # If RSI is rising (+) but price is falling (-), we have bullish divergence
        # Score in [-1,+1]: positive = bullish div, negative = bearish div
        rsi_dir   = df["rsi_14"].diff(5).apply(np.sign)
        price_dir = df["Close"].pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0).apply(np.sign)
        df["rsi_price_div"] = rsi_dir - price_dir   # +2 = bullish, -2 = bearish

    # MACD histogram slope
    if "macd_hist" in df.columns:
        df["macd_hist_slope"] = df["macd_hist"].diff(3)

    # Lower wick ratio (hammer signal)
    try:
        body_low  = df[["Open", "Close"]].min(axis=1)
        bar_range = df["High"] - df["Low"]
        df["lower_wick_ratio"] = (body_low - df["Low"]) / bar_range.replace(0, np.nan)
        df["upper_wick_ratio"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / bar_range.replace(0, np.nan)
    except Exception:
        pass

    # Volume slope — replace inf from zero-volume bars (data gaps, pre-market)
    if "Volume" in df.columns:
        df["vol_slope_3"] = (
            df["Volume"].replace(0, np.nan).pct_change(3)
            .replace([np.inf, -np.inf], np.nan)
            .clip(-5.0, 5.0).fillna(0.0)
        )

    # Hidden RSI divergence binary flags (1 = active, 0 = not)
    # Computed row-by-row using a rolling window — causal, no look-ahead
    if "rsi_14" in df.columns and "Close" in df.columns:
        prices = df["Close"].values.astype(float)
        rsi_v  = df["rsi_14"].values.astype(float)
        hb = np.zeros(len(df), dtype=float)
        hbr = np.zeros(len(df), dtype=float)
        for i in range(_DIV_LOOKBACK + _DIV_SWING_WINDOW + 1, len(df)):
            p_slice = prices[i - _DIV_LOOKBACK: i]
            r_slice = rsi_v[i - _DIV_LOOKBACK: i]
            p_lows = _find_swing_lows(p_slice)
            r_lows = _find_swing_lows(r_slice)
            if len(p_lows) >= 2 and len(r_lows) >= 2:
                if p_lows[-1][1] > p_lows[-2][1] and r_lows[-1][1] < r_lows[-2][1]:
                    hb[i] = 1.0
            p_highs = _find_swing_highs(p_slice)
            r_highs = _find_swing_highs(r_slice)
            if len(p_highs) >= 2 and len(r_highs) >= 2:
                if p_highs[-1][1] < p_highs[-2][1] and r_highs[-1][1] > r_highs[-2][1]:
                    hbr[i] = 1.0
        df["hidden_div_bull"] = hb
        df["hidden_div_bear"] = hbr

    return df


REVERSAL_FEATURE_COLS = [
    "rsi_14", "rsi_7", "rsi_slope_3", "rsi_slope_5",
    "price_slope_3", "price_slope_5",
    "rsi_price_div",
    "macd_hist", "macd_hist_slope",
    "bb_pct", "bb_width",
    "stoch_k", "stoch_d",
    "cci_20", "mfi_14",
    "vol_ratio", "vol_slope_3",
    "lower_wick_ratio", "upper_wick_ratio",
    "atr_14",
    "hidden_div_bull", "hidden_div_bear",
]
