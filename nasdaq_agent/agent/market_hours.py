"""
Market session awareness — session windows for Alpha Strike Trader.

Session schedule (all times ET):
  PRE_MARKET      04:00 – 09:29   Data collection only, no live trades
  RESTRICTED      09:30 – 09:44   First 15 min — price discovery, reduced size (60%)
  PRIME           09:45 – 11:29   Best setups, full position sizing
  LUNCH_BLOCK     11:30 – 13:29   Typically lower volume, reduced size (70%)
  STANDARD        13:30 – 15:29   Good setups, 80% position sizing
  CLOSING_CAUTION 15:30 – 15:44   Momentum trades only, 70% position sizing
  HARD_CLOSE      15:45 – 16:00   Close ALL open positions at market, no new entries
  AFTER_HOURS     16:00 – 20:00   No new trades, data collection only
  CLOSED          20:00 – 04:00   Market closed
"""
from __future__ import annotations
from datetime import datetime, time
import pytz

ET = pytz.timezone("America/New_York")

# Session metadata: label, hex colour, tradeable, confidence multiplier, size_mult
_SESSIONS: dict[str, dict] = {
    "PRE_MARKET": {
        "label": "Pre-Market", "color": "#64748b", "tradeable": False,
        "mult": 0.40, "size_mult": 0.0,
        "advice": "Pre-market — data collection only. Morning scan running.",
    },
    "RESTRICTED": {
        "label": "⚠ Price Discovery (09:30–09:44)", "color": "#f59e0b", "tradeable": True,
        "mult": 0.60, "size_mult": 0.60,
        "advice": "First 15 min — price discovery in progress. Trades active at 60% size.",
    },
    "PRIME": {
        "label": "⚡ Prime Hours", "color": "#22c55e", "tradeable": True,
        "mult": 1.00, "size_mult": 1.0,
        "advice": "Prime hours (9:45–11:30 ET) — highest quality setups. Full position sizing.",
    },
    "LUNCH_BLOCK": {
        "label": "Midday Session (11:30–13:30)", "color": "#94a3b8", "tradeable": True,
        "mult": 0.70, "size_mult": 0.70,
        "advice": "Midday session (11:30–1:30 ET) — typically lower volume. Trades at 70% size.",
    },
    "STANDARD": {
        "label": "Standard Hours", "color": "#3b82f6", "tradeable": True,
        "mult": 0.85, "size_mult": 0.80,
        "advice": "Standard hours (1:30–3:30 ET) — good setups. Position sizing at 80%.",
    },
    "CLOSING_CAUTION": {
        "label": "⚡ Power Hour", "color": "#f97316", "tradeable": True,
        "mult": 0.90, "size_mult": 0.70,
        "advice": "Power hour (3:30–3:45 ET) — momentum trades only. No new scalp entries.",
    },
    "HARD_CLOSE": {
        "label": "🔴 Hard Close Window", "color": "#ef4444", "tradeable": False,
        "mult": 0.00, "size_mult": 0.0,
        "advice": "Hard close (3:45–4:00 PM ET) — ALL positions closing at market. No new entries.",
    },
    "AFTER_HOURS": {
        "label": "After-Hours", "color": "#64748b", "tradeable": False,
        "mult": 0.30, "size_mult": 0.0,
        "advice": "After-hours (4–8 PM ET) — T1·HIGH mega-caps trade at 50% size. T2/T3 monitoring only.",
    },
    "CLOSED": {
        "label": "Market Closed", "color": "#334155", "tradeable": False,
        "mult": 0.00, "size_mult": 0.0,
        "advice": "Market closed. Pre-market scan starts at 4:30 AM ET.",
    },
}


def get_session() -> str:
    """Return current session key based on ET time."""
    t = datetime.now(ET).time()
    if   time(4,   0) <= t < time(9,  30): return "PRE_MARKET"
    elif time(9,  30) <= t < time(9,  45): return "RESTRICTED"
    elif time(9,  45) <= t < time(11, 30): return "PRIME"
    elif time(11, 30) <= t < time(13, 30): return "LUNCH_BLOCK"
    elif time(13, 30) <= t < time(15, 30): return "STANDARD"
    elif time(15, 30) <= t < time(15, 45): return "CLOSING_CAUTION"
    elif time(15, 45) <= t <= time(16,  0): return "HARD_CLOSE"
    elif time(16,  0) <  t <= time(20,  0): return "AFTER_HOURS"
    else:                                    return "CLOSED"


def get_session_info() -> dict:
    """Full session info dict for broadcast to frontend."""
    key  = get_session()
    info = _SESSIONS[key].copy()
    now  = datetime.now(ET)
    info["session"]    = key
    info["time_et"]    = now.strftime("%I:%M:%S %p ET")
    info["date_et"]    = now.strftime("%A %b %d, %Y")
    info["is_weekend"] = now.weekday() >= 5
    return info


def confidence_multiplier() -> float:
    """Confidence multiplier for current session."""
    return _SESSIONS[get_session()]["mult"]


def position_size_multiplier() -> float:
    """Position size multiplier for current session (0.0 = no new trades)."""
    return _SESSIONS[get_session()]["size_mult"]


def is_tradeable() -> bool:
    """True during all market-open sessions except HARD_CLOSE, AFTER_HOURS, CLOSED, PRE_MARKET."""
    return _SESSIONS[get_session()]["tradeable"]


def is_trading_day() -> bool:
    return datetime.now(ET).weekday() < 5


# ── Granular session checks (used by risk_controls and paper_trading) ──────────

def is_restricted() -> bool:
    """True during first 15 min (9:30–9:44 ET) — price discovery, reduced position size."""
    return get_session() == "RESTRICTED"


def is_lunch_block() -> bool:
    """True during 11:30–1:30 PM ET — midday session, reduced position size."""
    return get_session() == "LUNCH_BLOCK"


def is_hard_close_window() -> bool:
    """True from 3:45 PM ET — all positions must close at market."""
    return get_session() in ("HARD_CLOSE",)


def is_closing_caution() -> bool:
    """True from 3:30–3:44 PM ET — momentum trades only, no new scalps."""
    return get_session() == "CLOSING_CAUTION"


def is_pre_market() -> bool:
    return get_session() == "PRE_MARKET"


def is_after_hours() -> bool:
    return get_session() in ("AFTER_HOURS", "CLOSED")


def no_new_entries() -> bool:
    """True only when the market is literally closed or in hard-close wind-down."""
    return get_session() in ("HARD_CLOSE", "AFTER_HOURS", "CLOSED", "PRE_MARKET")


def get_block_reason() -> str:
    """Human-readable reason why new entries are blocked (or '' if not blocked)."""
    s = get_session()
    reasons = {
        "HARD_CLOSE":  "🔴 Hard close window — all positions closing. No new entries.",
        "AFTER_HOURS": "After-hours — market closed for trading.",
        "CLOSED":      "Market closed.",
        "PRE_MARKET":  "Pre-market — no live trading, data collection only.",
    }
    return reasons.get(s, "")


def minutes_until_open() -> int:
    """Minutes until next PRIME session starts (returns 0 if already in PRIME/STANDARD)."""
    now_et = datetime.now(ET)
    t = now_et.time()
    # Already in tradeable session
    if is_tradeable():
        return 0
    # Calculate minutes to next 9:45 AM
    target = now_et.replace(hour=9, minute=45, second=0, microsecond=0)
    if t >= time(9, 45):
        # Past today's open — next open is tomorrow
        from datetime import timedelta
        target = target + timedelta(days=1)
    delta = target - now_et
    return max(0, int(delta.total_seconds() / 60))
