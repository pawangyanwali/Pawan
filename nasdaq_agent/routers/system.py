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

import json
import os
import time

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse, Response

from auth.dependencies import require_viewer, AuthenticatedUser

router = APIRouter(tags=["system"])

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "static")


@router.get("/robots.txt", response_class=PlainTextResponse)
async def robots_txt():
    return PlainTextResponse(
        "User-agent: *\nDisallow: /api/\nDisallow: /static/\n",
        media_type="text/plain",
    )


@router.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


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


@router.get("/algo", response_class=HTMLResponse)
@router.get("/algo.html", response_class=HTMLResponse)
async def algo_page():
    return FileResponse(os.path.join(STATIC_DIR, "algo.html"))


@router.get("/api/health")
async def health():
    from agent.data_fetcher import get_credit_usage
    from agent.valkey_client import health_status as vk_health
    from main import _current_signal_snapshot
    from routers._deps import manager
    from agent.scanner import scanner
    sigs, last_scan, from_cache = _current_signal_snapshot()
    # Memory pressure — best-effort, never blocks the health response
    _memory_rss_mb: float | None = None
    try:
        import psutil as _psutil
        _memory_rss_mb = round(_psutil.Process().memory_info().rss / 1_048_576, 1)
    except Exception:
        try:
            import resource as _resource
            # Linux: ru_maxrss is in KB; macOS it is in bytes
            _raw = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
            _memory_rss_mb = round(_raw / 1024, 1)   # KB → MB (Linux default)
        except Exception:
            pass
    return {
        "status": "ok",
        "is_running": scanner.is_running,
        "last_scan": last_scan,
        "tickers_tracked": len(sigs),
        "from_cache": from_cache,
        "ws_clients": len(manager.active),
        "api_credits": get_credit_usage(),
        "valkey": vk_health(),
        "memory_rss_mb": _memory_rss_mb,
    }


def _container_health(valkey_connected: bool) -> dict:
    """
    Derive container liveness from uniform service heartbeats.

    Each container writes service:{name}:heartbeat to PostgreSQL + Valkey every
    30 s with a 120 s TTL, so key expiry == container down.

    web-api is always "up" because this process is answering the request.
    """
    now = time.time()
    result: dict = {
        "web-api": {"up": True, "last_seen_ago_s": 0.0, "detail": "serving this response"},
    }

    def _pg_age(key: str) -> float | None:
        try:
            from agent.service_state import get_age_s as _ss_age
            return _ss_age(key)
        except Exception:
            return None

    def _vk_age(key: str) -> float | None:
        if not valkey_connected:
            return None
        try:
            from agent.valkey_client import _get_client as _vk_c
            client = _vk_c()
            if not client:
                return None
            raw = client.get(key)
            if not raw:
                return None
            data = json.loads(raw)
            ts = float(data.get("ts") or 0.0)
            return max(0.0, now - ts) if ts else None
        except Exception:
            return None

    def _age_for(key: str) -> float | None:
        age = _pg_age(key)
        return age if age is not None else _vk_age(key)

    def _entry(age: float | None, ttl_s: int, missing_detail: str, label: str) -> dict:
        if age is None:
            return {"up": False, "last_seen_ago_s": None, "detail": missing_detail}
        rounded = round(age, 1)
        return {
            "up": rounded < ttl_s,
            "last_seen_ago_s": rounded,
            "detail": f"{label} {rounded}s ago",
        }

    def _legacy_key(key: str, ttl_s: int, label: str) -> dict:
        return _entry(_age_for(key), ttl_s, f"no {key} state", label)

    def _heartbeat(service_name: str, ttl_s: int = 120, fallback: dict | None = None) -> dict:
        age = _age_for(f"service:{service_name}:heartbeat")
        if age is not None:
            return _entry(age, ttl_s, f"no service:{service_name}:heartbeat state", "heartbeat")
        if fallback is not None:
            return fallback
        return {"up": False, "last_seen_ago_s": None, "detail": f"no service:{service_name}:heartbeat state"}

    result["market-data"] = _heartbeat(
        "market-data", 120,
        fallback=_legacy_key("scanner:streamer", 90, "streamer status"),
    )
    result["scanner"] = _heartbeat(
        "scanner", 120,
        fallback=_legacy_key("scan:latest", 660, "last scan"),
    )
    result["learner"] = _heartbeat(
        "learner", 180,
        fallback=_legacy_key("learner:status", 300, "learner status"),
    )
    result["scheduler"] = _heartbeat(
        "scheduler", 120,
        fallback=_legacy_key("scheduler:heartbeat", 120, "scheduler heartbeat"),
    )
    result["context-intel"] = _heartbeat(
        "context-intel", 120,
        fallback=_legacy_key("ctx:intel:heartbeat", 120, "context heartbeat"),
    )
    result["watchdog"] = _heartbeat("watchdog", 120)
    return result


@router.get("/api/services")
async def services_status(_user: AuthenticatedUser = Depends(require_viewer)):
    """
    Aggregate health of all infrastructure services for the dashboard panel.
    Returns connectivity status for: Scanner, MD Poller, Valkey, RDS (PostgreSQL),
    and per-container liveness via service heartbeats.
    """
    from agent.valkey_client import health_status as vk_health
    from agent.broker.schwab_streamer import get_streamer_status
    from main import _current_signal_snapshot
    from routers._deps import manager
    from agent.scanner import scanner

    vk = vk_health()
    streamer = get_streamer_status()

    # When market-data runs in its own container, read streamer/poller status
    # from service_state (PostgreSQL first, Valkey fallback).
    if not streamer.get("ws_streamer", {}).get("running"):
        try:
            from agent.service_state import get_state as _ss_get
            _sd = _ss_get("scanner:streamer")
            if _sd:
                streamer = _sd
        except Exception:
            pass
        if not streamer.get("ws_streamer", {}).get("running") and vk.get("connected"):
            try:
                from agent.valkey_client import _get_client as _vk_c
                _vc = _vk_c()
                if _vc:
                    raw = _vc.get("scanner:streamer")
                    if raw:
                        streamer = json.loads(raw)
            except Exception:
                pass

    # scanner.is_running is always False in web-api (scanner runs in its own
    # container).  Derive running state from the scanner heartbeat instead;
    # fall back to the in-process flag for monolith deployments.
    _scanner_running = scanner.is_running
    if not _scanner_running:
        try:
            from agent.service_state import get_age_s as _hb_age
            _age = _hb_age("service:scanner:heartbeat")
            if _age is not None and _age < 120:
                _scanner_running = True
        except Exception:
            pass

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

    ws_st = streamer.get("ws_streamer", {})
    md_st = streamer.get("md_poller", {})
    sigs, last_scan, from_cache = _current_signal_snapshot()

    return {
        "scanner": {
            "running":    _scanner_running,
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
        "containers": _container_health(bool(vk.get("connected"))),
    }


@router.get("/api/runtime-health")
async def runtime_health(_user: AuthenticatedUser = Depends(require_viewer)):
    """SLA snapshot: overall status + actionable alerts for the ops chip."""
    from agent.runtime_sla import evaluate_runtime_sla
    from agent.valkey_client import health_status as vk_health, price_bus_health
    from agent.market_hours import get_market_session

    try:
        vk = vk_health()
        session = get_market_session()
        price_2s = price_bus_health(max_age_s=2.0)
        price_5s = price_bus_health(max_age_s=5.0)

        # Scanner freshness from service state
        scan_age_s = None
        try:
            from agent.service_state import get_age_s as _ss_age
            scan_age_s = _ss_age("service:scanner:heartbeat")
        except Exception:
            pass

        scanner_info = {"scan_age_s": scan_age_s if scan_age_s is not None else -1.0}
        containers = _container_health(bool(vk.get("connected")))

        sla = evaluate_runtime_sla(
            session=session,
            price_2s=price_2s,
            price_5s=price_5s,
            scanner=scanner_info,
            containers=containers,
            valkey=vk,
        )
        return {"sla": sla, "valkey": vk}
    except Exception as exc:
        return {
            "sla": {
                "status": "CRITICAL",
                "alert_count": 1,
                "alerts": [{"severity": "CRITICAL", "component": "web-api",
                             "message": "Runtime health evaluation failed",
                             "detail": str(exc), "action": ""}],
            }
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
