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
