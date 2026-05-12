"""
Market session awareness — knows what time it is and what that means for trading.

Sessions and their implications:
  PRE_MARKET  04:00–09:30 ET  low liquidity, fake levels, avoid
  AVOID_ZONE  09:30–09:45 ET  high volatility open, chop, no entries
  OPEN        09:45–15:00 ET  active session, full signal weight
  POWER_HOUR  15:00–16:00 ET  strongest directional moves of the day
  AFTER_HOURS 16:00–20:00 ET  low liquidity, unreliable
  CLOSED      20:00–04:00 ET  market closed
"""
from __future__ import annotations
from datetime import datetime, time
import pytz

ET = pytz.timezone("America/New_York")

# Session metadata: label, hex colour, tradeable, confidence multiplier
_SESSIONS: dict[str, dict] = {
    "PRE_MARKET":  {"label": "Pre-Market",  "color": "#64748b", "tradeable": False, "mult": 0.40,
                    "advice": "Pre-market — low liquidity, unreliable signals. Wait for open."},
    "AVOID_ZONE":  {"label": "⚠ Avoid Zone","color": "#f59e0b", "tradeable": False, "mult": 0.50,
                    "advice": "First 15 min — extreme chop. Best entries come after 9:45 AM ET."},
    "OPEN":        {"label": "Market Open", "color": "#22c55e", "tradeable": True,  "mult": 1.00,
                    "advice": "Active session — full signal weight. Best setups occur 10am–3pm ET."},
    "POWER_HOUR":  {"label": "⚡ Power Hour","color": "#00ff88", "tradeable": True,  "mult": 1.10,
                    "advice": "Power Hour — strongest directional moves. High-conviction setups only."},
    "AFTER_HOURS": {"label": "After-Hours", "color": "#64748b", "tradeable": False, "mult": 0.30,
                    "advice": "After-hours — thin market. Signals are noise. Wait for next open."},
    "CLOSED":      {"label": "Market Closed","color": "#334155","tradeable": False, "mult": 0.00,
                    "advice": "Market closed. Review setups for tomorrow."},
}


def get_session() -> str:
    """Return current session key."""
    t = datetime.now(ET).time()
    if   time(4,  0) <= t < time(9,  30): return "PRE_MARKET"
    elif time(9,  30) <= t < time(9,  45): return "AVOID_ZONE"
    elif time(9,  45) <= t < time(15,  0): return "OPEN"
    elif time(15,  0) <= t <= time(16,  0): return "POWER_HOUR"
    elif time(16,  0) <  t <= time(20,  0): return "AFTER_HOURS"
    else:                                    return "CLOSED"


def get_session_info() -> dict:
    """Full session info dict for broadcast to frontend."""
    key  = get_session()
    info = _SESSIONS[key].copy()
    now  = datetime.now(ET)
    info["session"]  = key
    info["time_et"]  = now.strftime("%I:%M:%S %p ET")
    info["date_et"]  = now.strftime("%A %b %d, %Y")
    info["is_weekend"] = now.weekday() >= 5
    return info


def confidence_multiplier() -> float:
    return _SESSIONS[get_session()]["mult"]


def is_tradeable() -> bool:
    return _SESSIONS[get_session()]["tradeable"]


def is_trading_day() -> bool:
    return datetime.now(ET).weekday() < 5
