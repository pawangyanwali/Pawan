"""
System / infrastructure routes:
  GET  /api/health
  GET  /api/services
  GET  /api/universe
  GET  /api/runtime-health  (alias: /api/health if needed)
  GET  /
  GET  /login   /login.html
  GET  /admin.html
"""

import os

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, FileResponse

from auth.dependencies import require_viewer, AuthenticatedUser

router = APIRouter(tags=["system"])

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "static")


@router.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@router.get("/login", response_class=HTMLResponse)
@router.get("/login.html", response_class=HTMLResponse)
async def login_page():
    return FileResponse(os.path.join(STATIC_DIR, "login.html"))


@router.get("/admin.html", response_class=HTMLResponse)
async def admin_page():
    return FileResponse(os.path.join(STATIC_DIR, "admin.html"))


@router.get("/api/health")
async def health():
    from agent.data_fetcher import get_credit_usage
    from agent.valkey_client import health_status as vk_health
    from main import _current_signal_snapshot
    from routers._deps import manager
    from agent.scanner import scanner
    sigs, last_scan, from_cache = _current_signal_snapshot()
    return {
        "status": "ok",
        "is_running": scanner.is_running,
        "last_scan": last_scan,
        "tickers_tracked": len(sigs),
        "from_cache": from_cache,
        "ws_clients": len(manager.active),
        "api_credits": get_credit_usage(),
        "valkey": vk_health(),
    }


@router.get("/api/services")
async def services_status(_user: AuthenticatedUser = Depends(require_viewer)):
    """
    Aggregate health of all infrastructure services for the dashboard panel.
    Returns connectivity status for: Scanner, MD Poller, Valkey, RDS (PostgreSQL).
    """
    from agent.valkey_client import health_status as vk_health
    from agent.broker.schwab_streamer import get_streamer_status
    from main import _current_signal_snapshot
    from routers._deps import manager
    from agent.scanner import scanner

    streamer = get_streamer_status()
    vk = vk_health()

    # RDS check — lightweight: just try to get a connection from the pool
    rds_ok = False
    rds_error = None
    try:
        import psycopg2, os as _os
        conn = psycopg2.connect(
            host=_os.getenv("PGHOST", ""),
            port=int(_os.getenv("PGPORT", "5432")),
            dbname=_os.getenv("PGDATABASE", "nasdaq_agent"),
            user=_os.getenv("PGUSER", ""),
            password=_os.getenv("PGPASSWORD", ""),
            connect_timeout=3,
        )
        conn.close()
        rds_ok = True
    except Exception as _re:
        rds_error = str(_re)

    ws_st  = streamer.get("ws_streamer", {})
    md_st  = streamer.get("md_poller", {})
    sigs, last_scan, from_cache = _current_signal_snapshot()

    return {
        "scanner": {
            "running":    scanner.is_running,
            "last_scan":  last_scan,
            "tickers":    len(sigs),
            "from_cache": from_cache,
            "ws_clients": len(manager.active),
        },
        "ws_streamer": {
            "running":     ws_st.get("running", False),
            "connected":   ws_st.get("connected", False),
            "live_quotes": ws_st.get("live_quotes", 0),
            "nq_bias":     ws_st.get("nq_bias", 0.0),
            "error":       ws_st.get("error"),
        },
        "md_poller": {
            "running":       md_st.get("running", False),
            "cycle":         md_st.get("cycle", 0),
            "last_ok_ago_s": md_st.get("last_ok_ago_s"),
            "live_quotes":   md_st.get("live_quotes", 0),
            "error":         md_st.get("error"),
        },
        "valkey": vk,
        "rds": {
            "connected": rds_ok,
            "error":     rds_error,
        },
    }


@router.get("/api/universe")
async def universe_status():
    """Ticker universe status: total tracked, active this cycle, tier breakdown."""
    try:
        from agent.ticker_universe import get_universe_manager, TIER1, TIER2, TIER3, FULL_UNIVERSE
        mgr = get_universe_manager()
        active = mgr.get_active_tickers()
        active_set = set(active)
        return {
            "universe_total":  len(FULL_UNIVERSE),
            "active_this_cycle": len(active),
            "tier1_count":     len(TIER1),
            "tier2_count":     len(TIER2),
            "tier3_count":     len(TIER3),
            "tier1_in_active": sum(1 for t in TIER1 if t in active_set),
            "tier2_in_active": sum(1 for t in TIER2 if t in active_set),
            "tier3_in_active": sum(1 for t in TIER3 if t in active_set),
            "active_tickers":  active,
        }
    except Exception as e:
        return {"error": str(e)}
