"""
Real-time risk controls: daily loss circuit breaker + sector concentration limit.

Daily loss circuit breaker
--------------------------
After paper trades cross a daily loss threshold (default -2% account), all new
BUY/SELL signals are suppressed for the rest of the day.  Prevents a bad morning
from compounding into a catastrophic session — every professional desk has this.

Sector concentration filter
----------------------------
Prevents loading up on correlated positions.  If two or more paper trades are
already open in the same sector, new signals for that sector are suppressed.
Example: NVDA + AMD both open as BUY → QCOM BUY gets blocked.
"""
from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)

# ── Daily loss circuit breaker ─────────────────────────────────────────────────
DAILY_LOSS_LIMIT_PCT  = -2.0   # stop trading if daily P&L drops below -2%
MAX_DAILY_TRADES      = 20     # also stop after this many trades in one day

# ── Sector concentration ───────────────────────────────────────────────────────
MAX_POSITIONS_PER_SECTOR = 2   # never hold more than 2 open positions in same sector

_lock = threading.Lock()
_circuit_open   = False      # True = circuit breaker tripped, no new signals
_circuit_reason = ""
_circuit_date   = None       # date when circuit was tripped (resets each day)


# ── Sector map: ticker → broad sector ─────────────────────────────────────────
# Keeps concentration limits meaningful without requiring a live data call.
_SECTOR_MAP: dict[str, str] = {
    # Semis
    "NVDA":"SEMIS","AMD":"SEMIS","AVGO":"SEMIS","QCOM":"SEMIS","AMAT":"SEMIS",
    "MU":"SEMIS","KLAC":"SEMIS","LRCX":"SEMIS","ADI":"SEMIS","MRVL":"SEMIS",
    "INTC":"SEMIS","SNPS":"SEMIS","CDNS":"SEMIS","MPWR":"SEMIS","ARM":"SEMIS",
    # Mega-cap tech
    "AAPL":"MEGA_TECH","MSFT":"MEGA_TECH","GOOGL":"MEGA_TECH","META":"MEGA_TECH",
    "AMZN":"MEGA_TECH","NFLX":"MEGA_TECH","TSLA":"MEGA_TECH",
    # Cloud / SaaS
    "ADBE":"CLOUD","INTU":"CLOUD","CRM":"CLOUD","WDAY":"CLOUD","SNOW":"CLOUD",
    "DDOG":"CLOUD","ZS":"CLOUD","CRWD":"CLOUD","PANW":"CLOUD","OKTA":"CLOUD",
    "NET":"CLOUD","MDB":"CLOUD","TEAM":"CLOUD","GTLB":"CLOUD","HUBS":"CLOUD",
    "TWLO":"CLOUD","BILL":"CLOUD","DOCU":"CLOUD",
    # Fintech / crypto
    "COIN":"FINTECH","HOOD":"FINTECH","PYPL":"FINTECH","AFRM":"FINTECH",
    "UPST":"FINTECH","MSTR":"FINTECH","MARA":"FINTECH","DKNG":"FINTECH",
    # Biotech / healthcare
    "REGN":"BIOTECH","AMGN":"BIOTECH","ISRG":"BIOTECH","CELH":"BIOTECH",
    # EV / clean energy
    "TSLA":"EV","RIVN":"EV","LCID":"EV","ENPH":"EV",
    # AI / emerging
    "PLTR":"AI_EMERGING","SOUN":"AI_EMERGING","IONQ":"AI_EMERGING",
    # Consumer / retail
    "SBUX":"CONSUMER","COST":"CONSUMER","LULU":"CONSUMER","CHWY":"CONSUMER",
    "BKNG":"CONSUMER","ABNB":"CONSUMER",
    # Travel / rideshare
    "LYFT":"TRAVEL","ROKU":"MEDIA","SNAP":"MEDIA","PINS":"MEDIA","RBLX":"MEDIA",
    "TTD":"MEDIA",
    # Misc
    "ADP":"PAYROLL","AXON":"DEFENSE","CEG":"ENERGY","SMCI":"SERVERS",
    "MELI":"LATAM","APP":"ADTECH","CVNA":"AUTO","ZM":"COMM","FTNT":"CYBERSEC",
}


def get_sector(ticker: str) -> str:
    return _SECTOR_MAP.get(ticker.upper(), "OTHER")


# ── Circuit breaker ───────────────────────────────────────────────────────────

def _reset_if_new_day() -> None:
    global _circuit_open, _circuit_reason, _circuit_date
    today = date.today()
    if _circuit_date != today:
        with _lock:
            _circuit_open   = False
            _circuit_reason = ""
            _circuit_date   = today


def check_circuit_breaker() -> tuple[bool, str]:
    """
    Returns (blocked: bool, reason: str).
    Pulls today's closed paper trade P&L to decide.
    """
    global _circuit_open, _circuit_reason, _circuit_date
    _reset_if_new_day()
    with _lock:
        if _circuit_open:
            return True, _circuit_reason

    try:
        from agent.paper_trading import get_today_pnl
        today_stats = get_today_pnl()
        total_pnl_pct = float(today_stats.get("total_pnl_pct", 0.0) or 0.0)
        trade_count   = int(today_stats.get("total", 0) or 0)

        if total_pnl_pct <= DAILY_LOSS_LIMIT_PCT:
            reason = (
                f"Daily loss circuit breaker: {total_pnl_pct:+.2f}% loss today "
                f"(limit {DAILY_LOSS_LIMIT_PCT}%). No new trades until tomorrow."
            )
            with _lock:
                _circuit_open   = True
                _circuit_reason = reason
                _circuit_date   = date.today()
            logger.warning(f"[RiskControls] {reason}")
            return True, reason

        if trade_count >= MAX_DAILY_TRADES:
            reason = (
                f"Daily trade limit reached ({trade_count}/{MAX_DAILY_TRADES}). "
                "No new signals today."
            )
            with _lock:
                _circuit_open   = True
                _circuit_reason = reason
                _circuit_date   = date.today()
            logger.warning(f"[RiskControls] {reason}")
            return True, reason

    except Exception as e:
        logger.debug(f"[RiskControls] circuit breaker check failed: {e}")

    return False, ""


# ── Sector concentration ───────────────────────────────────────────────────────

def check_sector_concentration(ticker: str, direction: str) -> tuple[bool, str]:
    """
    Returns (blocked: bool, reason: str).
    Blocks a new signal if MAX_POSITIONS_PER_SECTOR are already open in the
    same sector in the same direction (long or short).
    """
    if direction not in ("BUY", "SELL"):
        return False, ""

    sector = get_sector(ticker)
    if sector == "OTHER":
        return False, ""   # unknown sector — don't apply limit

    try:
        from agent.paper_trading import get_open_trades
        open_trades = get_open_trades()
        sector_count = sum(
            1 for t in open_trades
            if get_sector(t.get("ticker", "")) == sector
            and t.get("direction", "") == direction
        )
        if sector_count >= MAX_POSITIONS_PER_SECTOR:
            reason = (
                f"Sector concentration: {sector_count} open {direction} positions "
                f"already in {sector} sector (max {MAX_POSITIONS_PER_SECTOR}). "
                f"Skipping {ticker}."
            )
            logger.debug(f"[RiskControls] {reason}")
            return True, reason
    except Exception as e:
        logger.debug(f"[RiskControls] sector check failed: {e}")

    return False, ""


def get_risk_status() -> dict:
    """Return current risk control state for the dashboard API."""
    _reset_if_new_day()
    with _lock:
        return {
            "circuit_open":        _circuit_open,
            "circuit_reason":      _circuit_reason,
            "daily_loss_limit_pct": DAILY_LOSS_LIMIT_PCT,
            "max_daily_trades":    MAX_DAILY_TRADES,
            "max_positions_per_sector": MAX_POSITIONS_PER_SECTOR,
        }
