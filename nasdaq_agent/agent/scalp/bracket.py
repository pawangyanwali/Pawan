from __future__ import annotations

from ._utils import finite, positive, round_tick
from .models import BracketGeometry, ScalpSignalConfig, SignalSide


def build_bracket(
    *,
    entry: float,
    side: SignalSide | str,
    atr_14: float,
    spread: float,
    config: ScalpSignalConfig,
) -> BracketGeometry:
    """Build TP1, TP2, and stop exclusively from configured risk math."""
    normalized_side = normalize_side(side)
    if normalized_side is SignalSide.NONE:
        raise ValueError("a LONG or SHORT side is required")
    if not positive(entry) or not positive(atr_14):
        raise ValueError("entry and atr_14 must be positive")
    if not finite(spread) or spread < 0:
        raise ValueError("spread must be non-negative")

    required_risk = max(
        atr_14 * config.stop_atr_multiple,
        entry * config.min_stop_pct,
        spread * config.spread_buffer_mult,
        config.tick_size,
    )
    max_risk = entry * config.max_stop_pct
    risk = min(required_risk, max_risk)
    risk_capped = required_risk > max_risk + 1e-12

    if normalized_side is SignalSide.LONG:
        stop = entry - risk
        tp1 = entry + config.tp1_r * risk
        tp2 = entry + config.reward_r * risk
    else:
        stop = entry + risk
        tp1 = entry - config.tp1_r * risk
        tp2 = entry - config.reward_r * risk

    entry_r = round_tick(entry, config.tick_size)
    stop_r = round_tick(stop, config.tick_size)
    tp1_r = round_tick(tp1, config.tick_size)
    tp2_r = round_tick(tp2, config.tick_size)
    effective_risk = abs(entry_r - stop_r)
    effective_reward = abs(tp2_r - entry_r)
    if effective_risk <= 0:
        raise ValueError("risk collapsed below one tradable tick")

    return BracketGeometry(
        entry=entry_r,
        stop_loss=stop_r,
        tp1=tp1_r,
        tp2=tp2_r,
        risk_per_share=round(effective_risk, 6),
        required_risk=round(required_risk, 6),
        reward_r=config.reward_r,
        rr_ratio=round(effective_reward / effective_risk, 4),
        risk_capped=risk_capped,
    )


def normalize_side(value: SignalSide | str) -> SignalSide:
    if isinstance(value, SignalSide):
        return value
    try:
        return SignalSide(str(value).upper())
    except ValueError:
        return SignalSide.NONE

