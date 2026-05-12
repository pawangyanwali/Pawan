"""
Trade management rules — generated alongside every signal.

Rules applied (in order):
  1. Move stop to breakeven (BE) once price is +1R above entry.
  2. Take 50% off position at +1R (1:1 reward reached).
  3. Time stop — exit after 15 bars if trade is not working.
  4. Trail stop — once +2R reached, trail at 50% of max-gain.

These rules are advisory; they are encoded in the signal output and
displayed in the UI. Actual execution is the trader's responsibility.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

import pandas as pd


@dataclass
class TradeManagement:
    entry:       float = 0.0
    stop:        float = 0.0
    target:      float = 0.0
    rr_ratio:    float = 0.0
    be_level:    float = 0.0      # breakeven = entry (move stop here at +1R)
    partial_exit: float = 0.0    # 50% off at 1:1 (= entry + 1R)
    trail_start:  float = 0.0    # begin trailing at +2R
    trail_stop:   float = 0.0    # current trail stop (= entry + 1R once at +2R)
    time_stop_bars: int = 15     # exit after N bars if no progress
    rules:        list = None    # human-readable checklist

    def __post_init__(self):
        if self.rules is None:
            self.rules = []

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def build_trade_plan(
    entry:   float,
    stop:    float,
    target:  float,
    direction: str = "BUY",
) -> TradeManagement:
    """
    Build a complete trade management plan for a given setup.

    Parameters
    ----------
    entry     : entry price
    stop      : initial stop-loss price
    target    : primary profit target
    direction : BUY or SELL
    """
    plan = TradeManagement(entry=round(entry, 4),
                           stop=round(stop, 4),
                           target=round(target, 4))

    risk   = abs(entry - stop)
    reward = abs(target - entry)
    plan.rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

    one_r = risk  # 1R in price terms

    if direction == "BUY":
        plan.be_level     = round(entry, 4)           # move stop to entry at +1R
        plan.partial_exit = round(entry + one_r, 4)   # take 50% off here
        plan.trail_start  = round(entry + 2 * one_r, 4)
        plan.trail_stop   = round(entry + one_r, 4)   # trail to +1R when at +2R
    else:
        plan.be_level     = round(entry, 4)
        plan.partial_exit = round(entry - one_r, 4)
        plan.trail_start  = round(entry - 2 * one_r, 4)
        plan.trail_stop   = round(entry - one_r, 4)

    plan.time_stop_bars = 15

    plan.rules = [
        f"Entry: ${entry:.2f}  |  Stop: ${stop:.2f}  |  Target: ${target:.2f}",
        f"Risk/Reward: 1 : {plan.rr_ratio:.1f}",
        f"At +1R (${plan.partial_exit:.2f}): take 50% off, move stop to breakeven ${plan.be_level:.2f}",
        f"At +2R (${plan.trail_start:.2f}): trail stop at +1R (${plan.trail_stop:.2f})",
        f"Time stop: exit if trade not working after {plan.time_stop_bars} bars",
    ]

    return plan
