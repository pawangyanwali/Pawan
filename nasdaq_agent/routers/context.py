"""
Context / watchlist / after-hours / macro routes:
  GET  /api/context/{ticker}
  GET  /api/watchlist
  POST /api/watchlist/add
  POST /api/watchlist/remove
  GET  /api/after-hours
  GET  /api/macro-calendar
"""

from fastapi import APIRouter, Depends

from auth.dependencies import require_trader, AuthenticatedUser

from agent.macro_calendar import check_macro_event, get_upcoming_events
from agent.after_hours_monitor import get_all_biases as ah_get_all
from config import NASDAQ_TICKERS, load_watchlist, save_watchlist

router = APIRouter(tags=["context"])


@router.get("/api/context/{ticker}")
async def context_snapshot_endpoint(ticker: str):
    """
    Return the latest context intelligence snapshot for a ticker.

    Payload includes rolling sentiment windows (5m/30m/2h/1d), news shock flag,
    context risk score, recent headlines, earnings phase / blackout status,
    and a stale_age_s field indicating how old the data is.

    Source:  Valkey ctx:latest:{ticker} → PostgreSQL fallback → safe defaults.
    Populated by the context-intel service every 30 seconds.
    """
    from agent.context_snapshot import get_context_snapshot
    snapshot = get_context_snapshot(ticker.upper())
    return snapshot


@router.get("/api/watchlist")
async def get_watchlist_endpoint():
    """Return user watchlist + base tickers."""
    return {
        "base":      NASDAQ_TICKERS,
        "watchlist": load_watchlist(),
    }


@router.post("/api/watchlist/add")
async def add_to_watchlist(
    ticker: str,
    _current: AuthenticatedUser = Depends(require_trader),
):
    """Add a ticker to the watchlist."""
    ticker = ticker.upper().strip()
    wl = load_watchlist()
    if ticker not in wl and ticker not in NASDAQ_TICKERS:
        wl.append(ticker)
        save_watchlist(wl)
    return {"watchlist": load_watchlist()}


@router.post("/api/watchlist/remove")
async def remove_from_watchlist(
    ticker: str,
    _current: AuthenticatedUser = Depends(require_trader),
):
    """Remove a ticker from the user watchlist (base tickers cannot be removed)."""
    ticker = ticker.upper().strip()
    wl = [t for t in load_watchlist() if t != ticker]
    save_watchlist(wl)
    return {"watchlist": load_watchlist()}


@router.get("/api/after-hours")
async def after_hours_endpoint():
    """
    Latest after-hours / pre-market snapshot for every scanned ticker.
    Sorted by absolute AH move descending — biggest movers first.
    """
    return {"snapshots": ah_get_all()}


@router.get("/api/macro-calendar")
async def macro_calendar_endpoint():
    """Return current macro event status and upcoming events."""
    return {
        "current": check_macro_event(),
        "upcoming": get_upcoming_events(days=14),
    }
