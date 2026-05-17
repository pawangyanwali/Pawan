"""
Dynamic exit signal generation.

Checks the current bar against the original trade setup and emits one or more
exit signals when conditions for closing the trade are met.

Exit conditions checked (in priority order)
-------------------------------------------
1. TARGET HIT         price ≥ target (BUY) or ≤ target (SELL)
2. STOP HIT           price ≤ stop   (BUY) or ≥ stop   (SELL)
3. CHANDELIER EXIT    price crosses below chandelier_long (BUY) / above chandelier_short (SELL)
4. VWAP LOSS          was above VWAP, now below (BUY trade)
5. VWAP LOSS          was below VWAP, now above (SELL trade)
6. RSI REVERSAL       RSI re-enters OB zone on a BUY, or OS zone on a SELL
7. PARTIAL_1R         price moved 1× initial risk in favor → scale out 25%
8. PARTIAL_2R         price moved 2× initial risk in favor → scale out 50%
9. VOLUME DRY-UP      vol < 50% avg for 2+ bars while price flat → stall
10. MACD CROSS        MACD histogram flips against trade direction
11. TIME STOP         trade open > 15 bars without meaningful progress

Returns list of ExitSignal objects and an aggregate EXIT_NOW / WATCH / HOLD.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd


# ── Thresholds ────────────────────────────────────────────────────────────────
_RSI_OB          = 70
_RSI_OS          = 30
_VOL_DRY_FACTOR  = 0.50    # < 50% avg volume = dry-up
_VOL_DRY_BARS    = 2       # must persist for N bars
_FLAT_PROGRESS   = 0.003   # < 0.3% move from entry = "no progress"
_TIME_STOP_BARS  = 15


@dataclass
class ExitSignal:
    signal:      str    # TARGET_HIT | STOP_HIT | CHANDELIER_EXIT | VWAP_LOSS | RSI_REVERSAL | PARTIAL_1R | PARTIAL_2R | VOL_DRYUP | MACD_CROSS | TIME_STOP
    priority:    str    # CRITICAL | HIGH | MEDIUM
    description: str
    action:      str    # EXIT_NOW | WATCH | SCALE_OUT


@dataclass
class ExitAnalysis:
    recommendation: str = "HOLD"   # EXIT_NOW | SCALE_OUT | WATCH | HOLD
    signals:        list = field(default_factory=list)
    summary:        str = ""

    def to_dict(self) -> dict:
        return {
            "recommendation": self.recommendation,
            "signals":        [asdict(s) for s in self.signals],
            "summary":        self.summary,
        }


def analyse_exits(
    df:          pd.DataFrame,
    direction:   str,
    entry_price: float,
    target:      float,
    stop:        float,
    bars_held:   int = 0,
) -> ExitAnalysis:
    """
    Analyse current conditions and emit exit guidance.

    Parameters
    ----------
    df          : DataFrame with indicators computed (vwap, rsi_14, macd_hist,
                  vol_ratio, chandelier_long, chandelier_short, atr_14 must be present)
    direction   : 'BUY' or 'SELL'
    entry_price : original entry price
    target      : profit target
    stop        : initial stop-loss price
    bars_held   : number of bars since entry
    """
    analysis = ExitAnalysis()
    signals:  list[ExitSignal] = []

    if df is None or len(df) < 3:
        analysis.summary = "Insufficient data for exit analysis."
        return analysis

    last   = df.iloc[-1]
    prev   = df.iloc[-2]
    price  = float(last.get("Close", 0))
    is_buy = direction == "BUY"

    # 1. Target / Stop hit ─────────────────────────────────────────────────────
    if is_buy and price >= target:
        signals.append(ExitSignal("TARGET_HIT", "CRITICAL", f"Target ${target:.2f} reached at ${price:.2f}", "EXIT_NOW"))
    elif not is_buy and price <= target:
        signals.append(ExitSignal("TARGET_HIT", "CRITICAL", f"Target ${target:.2f} reached at ${price:.2f}", "EXIT_NOW"))

    if is_buy and price <= stop:
        signals.append(ExitSignal("STOP_HIT", "CRITICAL", f"Stop ${stop:.2f} breached at ${price:.2f}", "EXIT_NOW"))
    elif not is_buy and price >= stop:
        signals.append(ExitSignal("STOP_HIT", "CRITICAL", f"Stop ${stop:.2f} breached at ${price:.2f}", "EXIT_NOW"))

    # 2. Chandelier Exit — dynamic ATR-based trailing stop ────────────────────
    # chandelier_long  = Highest(High, 22) − 3×ATR(14): long trailing stop
    # chandelier_short = Lowest(Low,  22) + 3×ATR(14): short trailing stop
    # Price closing below chandelier_long on a long = trend reversal signal
    try:
        chan_long  = float(last.get("chandelier_long",  0))
        chan_short = float(last.get("chandelier_short", 0))
        if is_buy and chan_long > 0 and price < chan_long:
            atr = float(last.get("atr_14", 0))
            signals.append(ExitSignal(
                "CHANDELIER_EXIT", "HIGH",
                f"Price ${price:.2f} crossed below Chandelier Exit ${chan_long:.2f} "
                f"(3×ATR={atr:.2f}) — trailing stop triggered, trend may be reversing",
                "EXIT_NOW"
            ))
        elif not is_buy and chan_short > 0 and price > chan_short:
            atr = float(last.get("atr_14", 0))
            signals.append(ExitSignal(
                "CHANDELIER_EXIT", "HIGH",
                f"Price ${price:.2f} crossed above Chandelier Exit ${chan_short:.2f} "
                f"(3×ATR={atr:.2f}) — trailing stop triggered, short thesis weakening",
                "EXIT_NOW"
            ))
    except Exception:
        pass

    # 3. Partial profit at 1R and 2R ──────────────────────────────────────────
    # Initial risk R = |entry_price - stop|. Scale out at 1R (25%) and 2R (50%).
    if entry_price > 0 and stop > 0:
        initial_risk = abs(entry_price - stop)
        if initial_risk > 0:
            if is_buy:
                gain = price - entry_price
                if gain >= 2.0 * initial_risk:
                    signals.append(ExitSignal(
                        "PARTIAL_2R", "HIGH",
                        f"Price +2R (${price:.2f}, +${gain:.2f}) — scale out 50%, "
                        f"move stop to breakeven, let remainder run",
                        "SCALE_OUT"
                    ))
                elif gain >= 1.0 * initial_risk:
                    signals.append(ExitSignal(
                        "PARTIAL_1R", "MEDIUM",
                        f"Price +1R (${price:.2f}, +${gain:.2f}) — scale out 25%, "
                        f"move stop to entry (risk-free trade)",
                        "SCALE_OUT"
                    ))
            else:
                gain = entry_price - price
                if gain >= 2.0 * initial_risk:
                    signals.append(ExitSignal(
                        "PARTIAL_2R", "HIGH",
                        f"Price −2R (${price:.2f}, +${gain:.2f}) — scale out 50%, "
                        f"move stop to breakeven, let remainder run",
                        "SCALE_OUT"
                    ))
                elif gain >= 1.0 * initial_risk:
                    signals.append(ExitSignal(
                        "PARTIAL_1R", "MEDIUM",
                        f"Price −1R (${price:.2f}, +${gain:.2f}) — scale out 25%, "
                        f"move stop to entry (risk-free trade)",
                        "SCALE_OUT"
                    ))

    # 4. VWAP Loss ─────────────────────────────────────────────────────────────
    vwap      = float(last.get("vwap", 0))
    vwap_prev = float(prev.get("vwap", 0))
    if vwap > 0 and vwap_prev > 0:
        prev_above = float(prev.get("Close", 0)) > vwap_prev
        curr_above = price > vwap
        if is_buy and prev_above and not curr_above:
            signals.append(ExitSignal("VWAP_LOSS", "HIGH", f"Lost VWAP ${vwap:.2f} — long thesis invalidated", "EXIT_NOW"))
        elif not is_buy and not prev_above and curr_above:
            signals.append(ExitSignal("VWAP_LOSS", "HIGH", f"Reclaimed VWAP ${vwap:.2f} — short thesis invalidated", "EXIT_NOW"))

    # 5. RSI Reversal ──────────────────────────────────────────────────────────
    rsi = float(last.get("rsi_14", 50))
    rsi_prev = float(prev.get("rsi_14", 50))
    if is_buy and rsi < _RSI_OB and rsi_prev >= _RSI_OB:
        signals.append(ExitSignal("RSI_REVERSAL", "HIGH",
            f"RSI exited overbought ({rsi:.0f}) — momentum reversal, scale out", "SCALE_OUT"))
    elif not is_buy and rsi > _RSI_OS and rsi_prev <= _RSI_OS:
        signals.append(ExitSignal("RSI_REVERSAL", "HIGH",
            f"RSI exited oversold ({rsi:.0f}) — momentum reversal, scale out", "SCALE_OUT"))

    # 9. Volume dry-up ─────────────────────────────────────────────────────────
    if len(df) >= _VOL_DRY_BARS + 1:
        vol_ratios = [float(df.iloc[-(i+1)].get("vol_ratio", 1)) for i in range(_VOL_DRY_BARS)]
        if all(v < _VOL_DRY_FACTOR for v in vol_ratios):
            progress = abs(price - entry_price) / entry_price if entry_price else 0
            if progress < _FLAT_PROGRESS:
                signals.append(ExitSignal("VOL_DRYUP", "MEDIUM",
                    f"Volume dry-up ({_VOL_DRY_BARS} bars) with no progress — momentum stalled", "WATCH"))

    # 10. MACD cross against trade ────────────────────────────────────────────
    hist      = float(last.get("macd_hist", 0))
    hist_prev = float(prev.get("macd_hist", 0))
    if is_buy and hist < 0 and hist_prev >= 0:
        signals.append(ExitSignal("MACD_CROSS", "MEDIUM",
            "MACD histogram crossed negative — momentum shifting down", "WATCH"))
    elif not is_buy and hist > 0 and hist_prev <= 0:
        signals.append(ExitSignal("MACD_CROSS", "MEDIUM",
            "MACD histogram crossed positive — momentum shifting up", "WATCH"))

    # 11. Time stop ────────────────────────────────────────────────────────────
    if bars_held >= _TIME_STOP_BARS:
        progress = (price - entry_price) / entry_price if is_buy else (entry_price - price) / entry_price
        if progress < 0.005:
            signals.append(ExitSignal("TIME_STOP", "MEDIUM",
                f"{bars_held} bars held with < 0.5% progress — time stop triggered", "EXIT_NOW"))

    # ── Aggregate recommendation ──────────────────────────────────────────────
    analysis.signals = signals
    exit_now_signals = {"TARGET_HIT", "STOP_HIT", "VWAP_LOSS", "CHANDELIER_EXIT"}
    if any(s.signal in exit_now_signals and s.action == "EXIT_NOW" for s in signals):
        analysis.recommendation = "EXIT_NOW"
    elif any(s.signal == "TIME_STOP" for s in signals):
        analysis.recommendation = "EXIT_NOW"
    elif any(s.action == "SCALE_OUT" for s in signals):
        analysis.recommendation = "SCALE_OUT"
    elif signals:
        analysis.recommendation = "WATCH"
    else:
        analysis.recommendation = "HOLD"

    if signals:
        analysis.summary = signals[0].description
    else:
        progress = (price - entry_price) / entry_price * 100 if is_buy else (entry_price - price) / entry_price * 100
        analysis.summary = f"Trade on track. Progress: {progress:+.2f}% | Bar {bars_held}"

    return analysis
