"""
Market regime detection using SPY and QQQ as proxies.

Regime affects signal confidence:
  BULL_TREND   → longs get full weight, shorts discounted
  BEAR_TREND   → shorts get full weight, longs discounted
  CHOPPY       → all signals discounted
  NEUTRAL      → no adjustment

VIX proxy is estimated from SPY ATR / price — no separate VIX feed needed.
"""
from __future__ import annotations
import logging
import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Regime multipliers: [long_mult, short_mult]
_REGIME_WEIGHT = {
    "BULL_TREND": (1.15, 0.60),
    "BEAR_TREND": (0.60, 1.15),
    "CHOPPY":     (0.70, 0.70),
    "NEUTRAL":    (1.00, 1.00),
}

_REGIME_COLOR = {
    "BULL_TREND": "#22c55e",
    "BEAR_TREND": "#ef4444",
    "CHOPPY":     "#f59e0b",
    "NEUTRAL":    "#94a3b8",
}

_REGIME_LABEL = {
    "BULL_TREND": "Bull Trend",
    "BEAR_TREND": "Bear Trend",
    "CHOPPY":     "Choppy",
    "NEUTRAL":    "Neutral",
}


@dataclass
class RegimeInfo:
    regime:        str   = "NEUTRAL"
    label:         str   = "Neutral"
    color:         str   = "#94a3b8"
    spy_change:    float = 0.0
    qqq_change:    float = 0.0
    vix_proxy:     float = 0.0        # estimated intraday volatility %
    long_mult:     float = 1.0
    short_mult:    float = 1.0
    description:   str   = ""

    def to_dict(self) -> dict:
        return {
            "regime":      self.regime,
            "label":       self.label,
            "color":       self.color,
            "spy_change":  round(self.spy_change, 3),
            "qqq_change":  round(self.qqq_change, 3),
            "vix_proxy":   round(self.vix_proxy, 2),
            "long_mult":   self.long_mult,
            "short_mult":  self.short_mult,
            "description": self.description,
        }


def _vix_proxy(df: pd.DataFrame) -> float:
    """Estimate intraday volatility as ATR(14) / price × 100."""
    if df is None or len(df) < 15:
        return 1.0
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    tr = np.maximum(high[1:] - low[1:],
         np.maximum(np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:]  - close[:-1])))
    atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
    price = float(close[-1])
    return round(atr / price * 100, 3) if price else 1.0


def detect_regime(df_spy: pd.DataFrame, df_qqq: pd.DataFrame) -> RegimeInfo:
    """
    Classify market regime from SPY and QQQ intraday data.

    Uses rolling 60-minute momentum (12 × 5min bars) instead of open-to-now
    to avoid noisy regime flips early in the session.

    Rules (applied in order):
      1. VIX proxy > 2.5% → CHOPPY
      2. VIX proxy spiking (last 5 bars significantly higher) → CHOPPY
      3. Both SPY + QQQ 60-min momentum > +0.3% → BULL_TREND
      4. Both SPY + QQQ 60-min momentum < -0.3% → BEAR_TREND
      5. Disagreement or small move → NEUTRAL
    """
    info = RegimeInfo()

    spy_ok = df_spy is not None and len(df_spy) >= 5
    qqq_ok = df_qqq is not None and len(df_qqq) >= 5

    if spy_ok:
        info.vix_proxy = _vix_proxy(df_spy)
        # 60-min momentum: last close vs close 12 bars ago (or open if fewer bars)
        n_lookback = min(12, len(df_spy) - 1)
        spy_now   = float(df_spy["Close"].iloc[-1])
        spy_ago   = float(df_spy["Close"].iloc[-n_lookback - 1]) if n_lookback > 0 else float(df_spy["Open"].iloc[0])
        info.spy_change = round((spy_now - spy_ago) / spy_ago * 100, 3) if spy_ago else 0.0

    if qqq_ok:
        n_lookback = min(12, len(df_qqq) - 1)
        qqq_now   = float(df_qqq["Close"].iloc[-1])
        qqq_ago   = float(df_qqq["Close"].iloc[-n_lookback - 1]) if n_lookback > 0 else float(df_qqq["Open"].iloc[0])
        info.qqq_change = round((qqq_now - qqq_ago) / qqq_ago * 100, 3) if qqq_ago else 0.0

    if not (spy_ok or qqq_ok):
        info.description = "No SPY/QQQ data — regime unknown."
        return info

    vp = info.vix_proxy
    sc = info.spy_change
    qc = info.qqq_change

    # Detect VIX spike: recent volatility accelerating
    vix_spiking = False
    if spy_ok and len(df_spy) >= 10:
        recent_vix = _vix_proxy(df_spy.iloc[-5:])
        older_vix  = _vix_proxy(df_spy.iloc[-10:-5])
        vix_spiking = recent_vix > older_vix * 1.4 and recent_vix > 1.5

    if vp > 2.5 or vix_spiking:
        regime = "CHOPPY"
        spike_note = " (accelerating)" if vix_spiking else ""
        desc = f"High volatility (VIX proxy {vp:.1f}%{spike_note}) — signals discounted."
    elif sc > 0.3 and qc > 0.3:
        regime = "BULL_TREND"
        desc = f"SPY {sc:+.2f}% / QQQ {qc:+.2f}% (60-min) — longs favoured."
    elif sc < -0.3 and qc < -0.3:
        regime = "BEAR_TREND"
        desc = f"SPY {sc:+.2f}% / QQQ {qc:+.2f}% (60-min) — shorts favoured."
    else:
        regime = "NEUTRAL"
        desc = f"SPY {sc:+.2f}% / QQQ {qc:+.2f}% (60-min) — no directional edge."

    lm, sm          = _REGIME_WEIGHT[regime]
    info.regime     = regime
    info.label      = _REGIME_LABEL[regime]
    info.color      = _REGIME_COLOR[regime]
    info.long_mult  = lm
    info.short_mult = sm
    info.description = desc
    return info


# ── Day Type Classifier ───────────────────────────────────────────────────────
# Classifies the current day's character by 10:30 AM ET.
# Determines whether to apply trend-following or mean-reversion strategy.

_DAY_TYPE_WEIGHT = {
    "TREND_DAY":   {"trend_boost": 1.3, "mr_discount": 0.6, "label": "Trend Day"},
    "RANGE_DAY":   {"trend_boost": 0.7, "mr_discount": 1.3, "label": "Range Day"},
    "UNCERTAIN":   {"trend_boost": 1.0, "mr_discount": 1.0, "label": "Uncertain"},
}


@dataclass
class DayTypeInfo:
    day_type:     str   = "UNCERTAIN"     # TREND_DAY | RANGE_DAY | UNCERTAIN
    label:        str   = "Uncertain"
    or_width:     float = 0.0             # Opening range width % (first 30 min)
    or_width_avg: float = 0.0             # 5-day average OR width %
    vwap_slope:   float = 0.0             # VWAP slope % over first 30 bars
    adx_now:      float = 0.0            # ADX at classification time
    trend_boost:  float = 1.0            # multiplier for trend signals
    mr_discount:  float = 1.0            # multiplier for mean-reversion signals
    description:  str   = ""

    def to_dict(self) -> dict:
        return {
            "day_type":     self.day_type,
            "label":        self.label,
            "or_width":     round(self.or_width, 3),
            "or_width_avg": round(self.or_width_avg, 3),
            "vwap_slope":   round(self.vwap_slope, 3),
            "adx_now":      round(self.adx_now, 1),
            "trend_boost":  self.trend_boost,
            "mr_discount":  self.mr_discount,
            "description":  self.description,
        }


def classify_day_type(df: pd.DataFrame, df_daily: pd.DataFrame | None = None) -> DayTypeInfo:
    """
    Classify the current trading day as TREND_DAY, RANGE_DAY, or UNCERTAIN.

    Classification uses three confluent signals measured after the first 30 min:
      1. Opening Range width vs 5-day average OR width
         - Wider than avg × 1.5 → likely trend day
         - Narrower than avg × 0.7 → likely range day
      2. VWAP slope over first 30 intraday bars
         - Steep slope → directional pressure → trend day
      3. ADX at 10:30 AM
         - ADX > 25 → trend mode
         - ADX < 20 → mean-reversion mode

    Requires intraday DataFrame with computed indicators (adx_14, vwap columns).
    """
    info = DayTypeInfo()

    if df is None or len(df) < 10:
        info.description = "Insufficient data for day type classification."
        return info

    try:
        import pytz
        et = pytz.timezone("America/New_York")
        df_et = df.copy()
        df_et.index = pd.to_datetime(df_et.index)
        if df_et.index.tzinfo is None:
            df_et.index = df_et.index.tz_localize("UTC").tz_convert(et)
        else:
            df_et.index = df_et.index.tz_convert(et)

        today = df_et.index[-1].date()
        today_bars = df_et[df_et.index.date == today]

        # Opening range: 09:30–10:00 (first 30 min)
        or_mask = (
            (today_bars.index.time >= pd.Timestamp("09:30").time()) &
            (today_bars.index.time <  pd.Timestamp("10:00").time())
        )
        or_bars = today_bars[or_mask]
        if or_bars.empty:
            info.description = "Pre-market or no OR bars yet — day type UNCERTAIN."
            return info

        or_high = float(or_bars["High"].max())
        or_low  = float(or_bars["Low"].min())
        mid_price = (or_high + or_low) / 2
        info.or_width = round((or_high - or_low) / mid_price * 100, 3) if mid_price > 0 else 0.0

        # 5-day average OR width from daily data
        if df_daily is not None and len(df_daily) >= 5:
            daily_ranges = (
                (df_daily["High"].iloc[-6:-1] - df_daily["Low"].iloc[-6:-1]) /
                df_daily["Close"].iloc[-6:-1] * 100
            )
            info.or_width_avg = round(float(daily_ranges.mean()), 3)
        else:
            info.or_width_avg = info.or_width   # no history, use current as baseline

        # VWAP slope: change in VWAP over the OR period
        if "vwap" in or_bars.columns and len(or_bars) >= 2:
            vwap_start = float(or_bars["vwap"].iloc[0])
            vwap_end   = float(or_bars["vwap"].iloc[-1])
            info.vwap_slope = round(
                (vwap_end - vwap_start) / vwap_start * 100, 3
            ) if vwap_start > 0 else 0.0

        # ADX at end of OR period (or latest bar)
        adx_col = "adx_14"
        if adx_col in today_bars.columns:
            adx_val = float(today_bars[adx_col].dropna().iloc[-1]) if not today_bars[adx_col].dropna().empty else 20.0
            # adx_14 in feature_engine.py is normalized 0-1; in technical.py it's raw
            # If value is ≤1.0 it's been normalized — scale back up
            info.adx_now = adx_val * 100 if adx_val <= 1.0 else adx_val

        # ── Classification rules (applied with majority vote) ─────────────────
        trend_votes = 0
        range_votes = 0

        avg = info.or_width_avg if info.or_width_avg > 0 else info.or_width
        if info.or_width > avg * 1.5:
            trend_votes += 1
        elif info.or_width < avg * 0.7:
            range_votes += 1

        if abs(info.vwap_slope) > 0.20:
            trend_votes += 1
        elif abs(info.vwap_slope) < 0.05:
            range_votes += 1

        if info.adx_now > 25:
            trend_votes += 1
        elif info.adx_now < 20:
            range_votes += 1

        if trend_votes >= 2:
            day_type = "TREND_DAY"
            desc = (
                f"TREND DAY: OR width {info.or_width:.2f}% vs avg {avg:.2f}%, "
                f"VWAP slope {info.vwap_slope:+.2f}%, ADX {info.adx_now:.0f} — "
                "follow breakouts, avoid fading the move"
            )
        elif range_votes >= 2:
            day_type = "RANGE_DAY"
            desc = (
                f"RANGE DAY: OR width {info.or_width:.2f}% vs avg {avg:.2f}%, "
                f"VWAP slope {info.vwap_slope:+.2f}%, ADX {info.adx_now:.0f} — "
                "fade extremes, buy support / sell resistance, target VWAP"
            )
        else:
            day_type = "UNCERTAIN"
            desc = (
                f"Day type unclear: OR {info.or_width:.2f}%, VWAP slope {info.vwap_slope:+.2f}%, "
                f"ADX {info.adx_now:.0f} — wait for direction, trade smaller"
            )

        weights = _DAY_TYPE_WEIGHT[day_type]
        info.day_type    = day_type
        info.label       = weights["label"]
        info.trend_boost = weights["trend_boost"]
        info.mr_discount = weights["mr_discount"]
        info.description = desc

    except Exception as exc:
        logger.debug("classify_day_type error: %s", exc)
        info.description = "Day type classification error — defaulting to UNCERTAIN."

    return info


_current_day_type: DayTypeInfo = DayTypeInfo()
_day_type_lock = threading.Lock()


def update_day_type(df: pd.DataFrame, df_daily: pd.DataFrame | None = None) -> DayTypeInfo:
    """Update module-level day type singleton. Thread-safe."""
    global _current_day_type
    dt = classify_day_type(df, df_daily)
    with _day_type_lock:
        _current_day_type = dt
    logger.info(f"Day type: {dt.day_type} | {dt.description}")
    return dt


def get_day_type() -> DayTypeInfo:
    with _day_type_lock:
        return _current_day_type


def apply_regime(score: float, regime: RegimeInfo) -> float:
    """Scale composite score by regime multiplier — result stays in [-1, +1]."""
    if score > 0:
        return round(float(np.clip(score * regime.long_mult,  -1.0, 1.0)), 4)
    elif score < 0:
        return round(float(np.clip(score * regime.short_mult, -1.0, 1.0)), 4)
    return score


# ── Module-level singleton so scanner can share one instance ──────────────────

_current_regime: RegimeInfo = RegimeInfo()
_regime_lock = threading.Lock()


def update_regime(df_spy: pd.DataFrame, df_qqq: pd.DataFrame) -> RegimeInfo:
    global _current_regime
    heuristic = detect_regime(df_spy, df_qqq)

    # Consensus vote: blend heuristic with LSTM regime if available.
    # LSTM uses a 20-bar sequence of price/volume features — more stable
    # than single-bar momentum and captures non-linear regime transitions.
    lstm_regime = "UNKNOWN"
    try:
        from agent.lstm_regime import get_lstm_regime
        lstm_regime, _ = get_lstm_regime(df_spy)   # returns (label, confidence)
    except Exception:
        pass

    if lstm_regime not in ("UNKNOWN", heuristic.regime):
        # Disagreement: downgrade to safer regime
        if heuristic.regime in ("BULL_TREND", "BEAR_TREND") and lstm_regime == "CHOPPY":
            heuristic.regime      = "NEUTRAL"
            heuristic.label       = _REGIME_LABEL["NEUTRAL"]
            heuristic.color       = _REGIME_COLOR["NEUTRAL"]
            heuristic.long_mult, heuristic.short_mult = _REGIME_WEIGHT["NEUTRAL"]
            heuristic.description += f" [LSTM disagrees: {lstm_regime} — downgraded to NEUTRAL]"
        elif heuristic.regime == "NEUTRAL" and lstm_regime in ("BULL_TREND", "BEAR_TREND"):
            # LSTM sees a trend the heuristic misses — trust it but label differently
            heuristic.description += f" [LSTM sees {lstm_regime}]"
    elif lstm_regime == heuristic.regime and lstm_regime != "UNKNOWN":
        heuristic.description += f" [LSTM confirms {lstm_regime}]"

    with _regime_lock:
        _current_regime = heuristic
    logger.info(
        f"Regime updated: {_current_regime.regime} (LSTM:{lstm_regime}) | "
        f"SPY {_current_regime.spy_change:+.2f}% / QQQ {_current_regime.qqq_change:+.2f}%"
    )
    return _current_regime


def get_regime() -> RegimeInfo:
    with _regime_lock:
        return _current_regime
