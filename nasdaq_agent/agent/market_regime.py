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

    Rules (applied in order):
      1. If vix_proxy > 2.5%  → CHOPPY
      2. Both SPY + QQQ > +0.4% today → BULL_TREND
      3. Both SPY + QQQ < -0.4% today → BEAR_TREND
      4. Disagreement or small move   → NEUTRAL
    """
    info = RegimeInfo()

    spy_ok = df_spy is not None and len(df_spy) >= 5
    qqq_ok = df_qqq is not None and len(df_qqq) >= 5

    if spy_ok:
        spy_open  = float(df_spy["Open"].iloc[0])
        spy_last  = float(df_spy["Close"].iloc[-1])
        info.spy_change = round((spy_last - spy_open) / spy_open * 100, 3) if spy_open else 0.0
        info.vix_proxy  = _vix_proxy(df_spy)

    if qqq_ok:
        qqq_open  = float(df_qqq["Open"].iloc[0])
        qqq_last  = float(df_qqq["Close"].iloc[-1])
        info.qqq_change = round((qqq_last - qqq_open) / qqq_open * 100, 3) if qqq_open else 0.0

    if not (spy_ok or qqq_ok):
        info.description = "No SPY/QQQ data — regime unknown."
        return info

    vp = info.vix_proxy
    sc = info.spy_change
    qc = info.qqq_change

    if vp > 2.5:
        regime = "CHOPPY"
        desc   = f"High volatility (VIX proxy {vp:.1f}%) — signals discounted."
    elif sc > 0.4 and qc > 0.4:
        regime = "BULL_TREND"
        desc   = f"SPY {sc:+.2f}% / QQQ {qc:+.2f}% — longs favoured."
    elif sc < -0.4 and qc < -0.4:
        regime = "BEAR_TREND"
        desc   = f"SPY {sc:+.2f}% / QQQ {qc:+.2f}% — shorts favoured."
    else:
        regime = "NEUTRAL"
        desc   = f"SPY {sc:+.2f}% / QQQ {qc:+.2f}% — no directional edge."

    lm, sm        = _REGIME_WEIGHT[regime]
    info.regime   = regime
    info.label    = _REGIME_LABEL[regime]
    info.color    = _REGIME_COLOR[regime]
    info.long_mult  = lm
    info.short_mult = sm
    info.description = desc
    return info


def apply_regime(score: float, regime: RegimeInfo) -> float:
    """Scale composite score by regime multiplier based on direction."""
    if score > 0:
        return round(score * regime.long_mult,  4)
    elif score < 0:
        return round(score * regime.short_mult, 4)
    return score


# ── Module-level singleton so scanner can share one instance ──────────────────

_current_regime: RegimeInfo = RegimeInfo()


def update_regime(df_spy: pd.DataFrame, df_qqq: pd.DataFrame) -> RegimeInfo:
    global _current_regime
    _current_regime = detect_regime(df_spy, df_qqq)
    logger.info(
        f"Regime updated: {_current_regime.regime} | "
        f"SPY {_current_regime.spy_change:+.2f}% / QQQ {_current_regime.qqq_change:+.2f}%"
    )
    return _current_regime


def get_regime() -> RegimeInfo:
    return _current_regime
