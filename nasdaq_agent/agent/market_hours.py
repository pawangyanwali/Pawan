"""
Market session awareness — session windows for Alpha Strike Trader.

Session schedule (all times ET, weekdays only):
  PRE_MARKET      04:00 – 09:29   Data collection only, no live trades
  RESTRICTED      09:30 – 09:44   First 15 min — price discovery, reduced size (60%)
  PRIME           09:45 – 11:29   Best setups, full position sizing
  LUNCH_BLOCK     11:30 – 13:29   Typically lower volume, reduced size (70%)
  STANDARD        13:30 – 15:29   Good setups, 80% position sizing
  CLOSING_CAUTION 15:30 – 15:44   Momentum trades only, 70% position sizing
  HARD_CLOSE      15:45 – 16:00   Close ALL open positions at market, no new entries
  AFTER_HOURS     16:00 – 20:00   No new trades, data collection only
  CLOSED          20:00 – 04:00   Market closed

Weekends and NYSE/NASDAQ market holidays always return CLOSED.
Half-trading days (Black Friday, Christmas Eve, July 3 when applicable)
close early at 1:00 PM ET — HARD_CLOSE starts at 12:45 PM.
"""
from __future__ import annotations
from datetime import date, datetime, time, timedelta
import logging
import threading
import pytz

ET  = pytz.timezone("America/New_York")
_log = logging.getLogger(__name__)

# ── US Market Holidays (NYSE / NASDAQ) 2025–2026 ────────────────────────────
_MARKET_HOLIDAYS: frozenset[date] = frozenset({
    # 2025
    date(2025,  1,  1),   # New Year's Day
    date(2025,  1, 20),   # Martin Luther King Jr. Day
    date(2025,  2, 17),   # Presidents' Day
    date(2025,  4, 18),   # Good Friday
    date(2025,  5, 26),   # Memorial Day
    date(2025,  6, 19),   # Juneteenth National Independence Day
    date(2025,  7,  4),   # Independence Day
    date(2025,  9,  1),   # Labor Day
    date(2025, 11, 27),   # Thanksgiving Day
    date(2025, 12, 25),   # Christmas Day
    # 2026
    date(2026,  1,  1),   # New Year's Day
    date(2026,  1, 19),   # Martin Luther King Jr. Day
    date(2026,  2, 16),   # Presidents' Day
    date(2026,  4,  3),   # Good Friday
    date(2026,  5, 25),   # Memorial Day
    date(2026,  6, 19),   # Juneteenth National Independence Day
    date(2026,  7,  3),   # Independence Day (observed; Jul 4 falls on Saturday)
    date(2026,  9,  7),   # Labor Day
    date(2026, 11, 26),   # Thanksgiving Day
    date(2026, 12, 25),   # Christmas Day
})

# ── Half-trading days — market closes at 1:00 PM ET ─────────────────────────
_HALF_DAYS: frozenset[date] = frozenset({
    date(2025,  7,  3),   # Day before Independence Day (Jul 4 is Friday)
    date(2025, 11, 28),   # Day after Thanksgiving (Black Friday)
    date(2025, 12, 24),   # Christmas Eve
    date(2026, 11, 27),   # Day after Thanksgiving (Black Friday)
    date(2026, 12, 24),   # Christmas Eve
})

# Session metadata: label, hex colour, tradeable, confidence multiplier, size_mult
_SESSIONS: dict[str, dict] = {
    "PRE_MARKET": {
        "label": "Pre-Market", "color": "#7c3aed", "tradeable": True,
        "mult": 0.45, "size_mult": 0.35,
        "advice": "Pre-market (4–9:30 AM ET) — HIGH-tier at 40% size, MODERATE-tier at 25% size. REGULAR-tier monitoring only.",
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
        "label": "After-Hours", "color": "#4f46e5", "tradeable": True,
        "mult": 0.40, "size_mult": 0.30,
        "advice": "After-hours (4–8 PM ET) — HIGH-tier at 50% size, MODERATE-tier at 30% size. REGULAR-tier monitoring only.",
    },
    "CLOSED": {
        "label": "Market Closed", "color": "#334155", "tradeable": False,
        "mult": 0.00, "size_mult": 0.0,
        "advice": "Market closed. Pre-market scan starts at 4:00 AM ET.",
    },
}


def _is_holiday(d: date) -> bool:
    return d in _MARKET_HOLIDAYS


def _is_half_day(d: date) -> bool:
    return d in _HALF_DAYS


# ── API-backed daily session cache ────────────────────────────────────────────
# Populated once per day from Schwab /markets endpoint (at startup and just
# after midnight ET).  Falls back to rule-based windows when the API is down.
#
# Shape of _day_cache:
#   date            : date            — which trading day this covers
#   is_trading_day  : bool            — False on weekends/holidays
#   regular_open    : time            — normally 09:30
#   regular_close   : time            — 16:00 or 13:00 on half-days
#   pre_open        : time            — normally 04:00 (Schwab often returns 07:00)
#   pre_close       : time            — normally 09:30
#   post_open       : time            — normally 16:00
#   post_close      : time            — normally 20:00
#   source          : str             — "schwab_api" | "rule_based"

_cache_lock: threading.Lock = threading.Lock()
_day_cache:  dict           = {}


def _parse_iso_time(iso: str | None) -> time | None:
    """Extract ET wall-clock time from an ISO-8601 datetime string."""
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).astimezone(ET).time().replace(second=0, microsecond=0)
    except Exception:
        return None


def refresh_market_hours_cache(target_date: date | None = None) -> bool:
    """
    Fetch today's (or target_date's) session boundaries from Schwab /markets.
    Returns True if the Schwab API responded; False if we fell back to rule-based.
    Always writes a valid entry to _day_cache regardless of API availability.

    Safe to call from any thread; designed to be called at startup and at
    midnight ET by the background refresh thread.
    """
    global _day_cache
    d = target_date or datetime.now(ET).date()

    # ── Try Schwab API ────────────────────────────────────────────────────────
    hours: dict = {}
    try:
        from agent.broker.schwab_market_data import fetch_market_hours as _fmh
        hours = _fmh("equity", target_date=d)
    except Exception as exc:
        _log.debug(f"[MarketHours] fetch_market_hours unavailable: {exc}")

    api_ok = hours.get("is_open") is not None   # None = not authorised / network error

    if api_ok:
        reg_open   = _parse_iso_time(hours.get("open_time"))
        reg_close  = _parse_iso_time(hours.get("close_time"))
        pre_open   = _parse_iso_time(hours.get("pre_market_start"))
        pre_close  = _parse_iso_time(hours.get("pre_market_end"))
        post_open  = _parse_iso_time(hours.get("post_market_start"))
        post_close = _parse_iso_time(hours.get("post_market_end"))
        # A trading day has regular session hours; holiday/weekend response has none.
        is_trading = reg_open is not None
        entry: dict = {
            "date":           d,
            "is_trading_day": is_trading,
            "regular_open":   reg_open   or time(9, 30),
            "regular_close":  reg_close  or time(16,  0),
            "pre_open":       pre_open   or time(4,   0),
            "pre_close":      pre_close  or time(9,  30),
            "post_open":      post_open  or time(16,  0),
            "post_close":     post_close or time(20,  0),
            "source":         "schwab_api",
        }
    else:
        # ── Rule-based fallback ───────────────────────────────────────────────
        is_wd      = d.weekday() < 5
        is_hol     = _is_holiday(d)
        is_half    = _is_half_day(d)
        reg_close_t = time(13, 0) if is_half else time(16, 0)
        entry = {
            "date":           d,
            "is_trading_day": is_wd and not is_hol,
            "regular_open":   time(9, 30),
            "regular_close":  reg_close_t,
            "pre_open":       time(4,  0),
            "pre_close":      time(9, 30),
            "post_open":      reg_close_t,
            "post_close":     time(20,  0),
            "source":         "rule_based",
        }

    with _cache_lock:
        _day_cache = entry

    _log.info(
        "[MarketHours] Cache %s: trading=%s regular %s–%s  pre %s–%s  post %s–%s",
        entry["source"], entry["is_trading_day"],
        entry["regular_open"], entry["regular_close"],
        entry["pre_open"], entry["pre_close"],
        entry["post_open"], entry["post_close"],
    )
    return api_ok


def _get_cache() -> dict:
    """Return today's cache entry; triggers a refresh if the date has rolled over."""
    with _cache_lock:
        cached = _day_cache.copy()
    today = datetime.now(ET).date()
    if cached.get("date") != today:
        refresh_market_hours_cache(today)
        with _cache_lock:
            cached = _day_cache.copy()
    return cached


def _start_midnight_refresh() -> None:
    """Daemon thread: re-fetch session boundaries just after midnight ET each day."""
    import time as _time

    def _loop() -> None:
        while True:
            try:
                now_et  = datetime.now(ET)
                # Sleep until 00:01 the next calendar day in ET
                next_day = ET.localize(
                    datetime.combine(now_et.date() + timedelta(days=1), time(0, 1))
                )
                sleep_s = max(0, (next_day - now_et).total_seconds())
                _time.sleep(sleep_s)
                refresh_market_hours_cache()
            except Exception as exc:
                _log.debug(f"[MarketHours] Midnight refresh error: {exc}")
                _time.sleep(60)   # retry in 1 min on unexpected error

    t = threading.Thread(target=_loop, daemon=True, name="market-hours-refresh")
    t.start()


_start_midnight_refresh()


# ── Public 4-state session API ────────────────────────────────────────────────

def get_market_session() -> str:
    """
    Return the canonical market session using the API-backed daily cache.

    REGULAR     — regular trading session  (normally 09:30–16:00 ET)
    PRE_MARKET  — pre-market session       (normally 04:00–09:30 ET)
    AFTER_HOURS — post-market session      (normally 16:00–20:00 ET)
    CLOSED      — outside all sessions (weekends, holidays, overnight)

    Use this in preference to get_session() when you only need to know which
    broad session the market is in (e.g. for data-fetch decisions and ML
    feature flags).  get_session() continues to return the fine-grained
    internal session labels used by risk controls (PRIME, LUNCH_BLOCK, etc.).
    """
    c      = _get_cache()
    now_et = datetime.now(ET)
    t      = now_et.time().replace(second=0, microsecond=0)

    if not c.get("is_trading_day", False):
        return "CLOSED"

    if c["regular_open"] <= t < c["regular_close"]:
        return "REGULAR"
    if c["pre_open"] <= t < c["pre_close"]:
        return "PRE_MARKET"
    if c["post_open"] <= t < c["post_close"]:
        return "AFTER_HOURS"
    return "CLOSED"


def get_session() -> str:
    """Return current session key based on ET time, weekday, and market calendar."""
    now   = datetime.now(ET)
    today = now.date()

    # Weekends are never trading days — no API needed
    if now.weekday() >= 5:
        return "CLOSED"

    # Consult the API-backed cache first so actual API holidays and unexpected
    # early closes (e.g. unscheduled half-days) are respected immediately.
    c = _get_cache()
    cache_is_today = (c.get("date") == today)

    if cache_is_today:
        if not c.get("is_trading_day", True):
            return "CLOSED"
        # If the API says the regular session has ended, map to AH or CLOSED
        reg_close = c.get("regular_close")
        t_now = now.time()
        if reg_close is not None and t_now >= reg_close:
            post_close = c.get("post_close", time(20, 0))
            return "AFTER_HOURS" if t_now < post_close else "CLOSED"
    else:
        # Cache is stale — fall back to hardcoded holiday list
        if _is_holiday(today):
            return "CLOSED"

    t = now.time()

    # Half-day early-close (market closes at 1:00 PM ET) — static fallback
    # only reached when the API cache is unavailable or stale.
    if not cache_is_today and _is_half_day(today):
        if t >= time(13, 0):
            return "CLOSED"
        if t >= time(12, 45):
            return "HARD_CLOSE"
        # Before 12:45 on a half-day: normal session windows apply

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
    now       = datetime.now(ET)
    today     = now.date()
    key       = get_session()
    info      = _SESSIONS[key].copy()

    is_weekend = now.weekday() >= 5
    # Use API cache for holiday/half-day detection when available
    c = _get_cache()
    if c.get("date") == today:
        is_holiday  = not c.get("is_trading_day", True) and not is_weekend
        is_half_day = _is_half_day(today)  # half-day flag still from static list
    else:
        is_holiday  = _is_holiday(today)
        is_half_day = _is_half_day(today)

    # Override label / advice / color for market-closed states
    if is_weekend:
        day_name = now.strftime("%A")
        next_open = "Monday" if now.weekday() == 5 else "Monday"  # Sat→Mon, Sun→Mon
        info["label"]  = f"🔴 {day_name} — Market Closed"
        info["advice"] = (
            f"U.S. stock markets are closed on weekends. "
            f"Next session: {next_open} pre-market at 4:00 AM ET."
        )
        info["color"] = "#334155"
    elif is_holiday:
        info["label"]  = "🔴 Market Holiday — Closed"
        info["advice"] = (
            "U.S. markets are closed today for a scheduled NYSE/NASDAQ holiday. "
            "Normal trading resumes the next business day."
        )
        info["color"] = "#334155"
    elif is_half_day and key in ("CLOSED", "HARD_CLOSE"):
        info["label"]  = "🟡 Half-Day — Early Close (1:00 PM ET)"
        info["advice"] = "Half-trading day — market closed at 1:00 PM ET."
        info["color"] = "#f59e0b"
    elif is_half_day:
        # Still within normal hours on a half-day — annotate advice
        info["advice"] += " ⚠ Half-day: market closes early at 1:00 PM ET."

    info["session"]     = key
    info["time_et"]     = now.strftime("%I:%M:%S %p ET")
    info["date_et"]     = now.strftime("%A, %b %d %Y")
    info["day_name"]    = now.strftime("%A")
    info["is_weekend"]  = is_weekend
    info["is_holiday"]  = is_holiday
    info["is_half_day"] = is_half_day
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
    """True on weekdays that are not NYSE/NASDAQ holidays."""
    now = datetime.now(ET)
    return now.weekday() < 5 and not _is_holiday(now.date())


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
    """True only during the post-market window (16:00–20:00 ET), not when fully closed."""
    return get_session() == "AFTER_HOURS"


def is_ah_eod_close_window() -> bool:
    """True during 19:55–20:05 ET on trading days — hard close window for AH positions."""
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5 or _is_holiday(now_et.date()):
        return False
    t = now_et.time()
    return time(19, 55) <= t < time(20, 6)


def no_new_entries() -> bool:
    """True only when the market is literally closed or in hard-close wind-down.
    Extended-hours sessions (PRE_MARKET, AFTER_HOURS) are NOT fully blocked —
    tier-specific size caps are enforced by check_session_block() in risk_controls.
    """
    return get_session() in ("HARD_CLOSE", "CLOSED")


def get_block_reason() -> str:
    """Human-readable reason why new entries are blocked (or '' if not blocked)."""
    s = get_session()
    now = datetime.now(ET)
    if s == "CLOSED":
        if now.weekday() >= 5:
            return f"Market closed — {now.strftime('%A')}. Trading resumes Monday."
        if _is_holiday(now.date()):
            return "Market closed — NYSE/NASDAQ holiday today."
        return "Market closed."
    reasons = {
        "HARD_CLOSE": "🔴 Hard close window — all positions closing. No new entries.",
    }
    return reasons.get(s, "")


def minutes_until_open() -> int:
    """Minutes until next PRIME session starts (returns 0 if already in PRIME/STANDARD)."""
    now_et = datetime.now(ET)

    # Already in a tradeable session
    if is_tradeable():
        return 0

    # Find the next 9:45 AM on a trading day
    target = now_et.replace(hour=9, minute=45, second=0, microsecond=0)
    if now_et >= target:
        target += timedelta(days=1)

    # Skip weekends and holidays
    while target.weekday() >= 5 or _is_holiday(target.date()):
        target += timedelta(days=1)

    delta = target - now_et
    return max(0, int(delta.total_seconds() / 60))
