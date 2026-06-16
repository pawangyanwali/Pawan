"""
Signal-related routes:
  GET  /api/signals
  GET  /api/signal-history
  GET  /api/regime
  GET  /api/credit-usage
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from auth.dependencies import require_viewer, AuthenticatedUser

router = APIRouter(tags=["signals"])


# ── helpers imported lazily to keep start-up light ───────────────────────────

def _snapshot():
    """Import and call _current_signal_snapshot from main at call time."""
    from main import _current_signal_snapshot, _get_universe_total
    from agent.scanner import scanner
    return _current_signal_snapshot, _get_universe_total, scanner


@router.get("/api/signals")
async def get_signals(_user: AuthenticatedUser = Depends(require_viewer)):
    """REST endpoint: returns the latest cached scan results."""
    from main import _current_signal_snapshot, _get_universe_total
    from agent.scanner import scanner
    try:
        from agent.signal_snapshot import read_latest as _snap_read
        snap = _snap_read() or {}
    except Exception:
        snap = {}
    sigs, last_scan, from_cache = _current_signal_snapshot()
    universe_total = int(snap.get("universe_total") or _get_universe_total())
    monitored_count = int(snap.get("monitored_count") or len(sigs))
    active_scan_count = int(snap.get("active_scan_count") or snap.get("scanned_count") or len(sigs))
    analysis_batch_count = int(snap.get("analysis_batch_count") or snap.get("deep_analyzed_count") or snap.get("scanned_count") or len(sigs))
    deep_analyzed_count = int(snap.get("deep_analyzed_count") or snap.get("scanned_count") or len(sigs))
    return {
        "last_scan":  last_scan,
        "count":      active_scan_count,
        "signals":    sigs,
        "from_cache": from_cache,
        "universe_total": universe_total,
        "monitored_count": monitored_count,
        "active_scan_count": active_scan_count,
        "analysis_batch_count": analysis_batch_count,
        "deep_analyzed_count": deep_analyzed_count,
        "preserved_count": int(snap.get("preserved_count") or 0),
        "observation_count": int(snap.get("observation_count") or 0),
        "empty_reason": None if sigs else "no_scanner_signal_cache_or_quote_data",
        "scanning":   scanner.is_running and not sigs,
    }


@router.get("/api/signal-history")
async def signal_history(ticker: str = None, limit: int = 50):
    """Return recent signal history from SQLite tracker."""
    from agent.signal_tracker import get_stats, get_recent_signals
    return {
        "signals": get_recent_signals(limit=limit),
        "stats":   get_stats(ticker=ticker),
    }


@router.get("/api/regime")
async def get_regime_endpoint():
    """Return current market regime (SPY/QQQ based)."""
    from agent.market_regime import get_regime
    from agent.market_hours import get_session_info
    regime = get_regime()
    session = get_session_info()
    return {"regime": regime.to_dict(), "session": session}


@router.get("/api/credit-usage")
async def credit_usage():
    """Schwab Market Data has no credit limits — returns zeros."""
    from agent.data_fetcher import get_credit_usage
    return get_credit_usage()
