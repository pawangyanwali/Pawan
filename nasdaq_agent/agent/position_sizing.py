"""
Position sizing calculator — risk-based sizing.

Formula
-------
  dollar_risk    = account_size × risk_pct / 100
  risk_per_share = |entry - stop|
  shares         = floor(dollar_risk / risk_per_share)
  position_value = shares × entry

Kelly-like confidence scaling (optional, capped at 100%):
  If signal confidence ≥ 70%, allow up to 1.5× normal size.
  If signal confidence < 50%, limit to 0.5× normal size.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict


@dataclass
class PositionSize:
    shares:          int   = 0
    position_value:  float = 0.0
    dollar_risk:     float = 0.0
    risk_per_share:  float = 0.0
    risk_pct_used:   float = 0.0
    account_size:    float = 0.0
    confidence_mult: float = 1.0
    description:     str   = ""

    def to_dict(self) -> dict:
        return asdict(self)


def calculate(
    account_size:  float,
    entry:         float,
    stop:          float,
    risk_pct:      float = 1.0,
    confidence:    float = 50.0,
    max_position_pct: float = 5.0,   # never exceed X% of account in one trade
) -> PositionSize:
    """
    Calculate position size.

    Parameters
    ----------
    account_size  : total account value in dollars
    entry         : entry price per share
    stop          : stop-loss price per share
    risk_pct      : % of account to risk per trade (default 1%)
    confidence    : signal confidence 0–100
    max_position_pct : hard cap as % of account (default 5%)
    """
    ps = PositionSize(account_size=round(account_size, 2))

    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0 or entry <= 0:
        ps.description = "Invalid entry or stop price."
        return ps

    # Confidence-based multiplier — thresholds and multipliers are configurable
    from agent.config_manager import config as _cfg_sz
    _conf_high_thr  = float(_cfg_sz.get("sizing.conf_high_threshold",  75.0))
    _conf_high_mult = float(_cfg_sz.get("sizing.conf_high_mult",       1.25))
    _conf_med_thr   = float(_cfg_sz.get("sizing.conf_medium_threshold", 60.0))
    _conf_med_mult  = float(_cfg_sz.get("sizing.conf_medium_mult",      1.0))
    _conf_low_thr   = float(_cfg_sz.get("sizing.conf_low_threshold",    50.0))
    _conf_low_mult  = float(_cfg_sz.get("sizing.conf_low_mult",         0.5))
    _conf_def_mult  = float(_cfg_sz.get("sizing.conf_default_mult",     0.75))

    if confidence >= _conf_high_thr:
        conf_mult = _conf_high_mult
    elif confidence >= _conf_med_thr:
        conf_mult = _conf_med_mult
    elif confidence < _conf_low_thr:
        conf_mult = _conf_low_mult
    else:
        conf_mult = _conf_def_mult

    dollar_risk    = account_size * (risk_pct / 100) * conf_mult
    shares         = math.floor(dollar_risk / risk_per_share)

    # Apply max position cap
    max_pos_value  = account_size * (max_position_pct / 100)
    max_shares     = math.floor(max_pos_value / entry)
    shares         = min(shares, max_shares)

    if shares <= 0:
        ps.description = "Position too small — risk parameters too tight."
        return ps

    pos_value      = shares * entry
    actual_risk    = shares * risk_per_share
    actual_risk_pct = actual_risk / account_size * 100

    ps.shares           = shares
    ps.position_value   = round(pos_value, 2)
    ps.dollar_risk      = round(actual_risk, 2)
    ps.risk_per_share   = round(risk_per_share, 4)
    ps.risk_pct_used    = round(actual_risk_pct, 3)
    ps.confidence_mult  = conf_mult
    ps.description = (
        f"{shares} shares × ${entry:.2f} = ${pos_value:,.0f} | "
        f"Risk ${actual_risk:.0f} ({actual_risk_pct:.2f}% of account)"
    )
    return ps
