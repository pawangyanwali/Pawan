"""
Real-time streaming routes:
  GET /stream/signals   (SSE or polling fallback)
  WS  /ws              (WebSocket — signals, prices, ticks)
"""

import asyncio
import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse

from routers._deps import manager, _dumps

router = APIRouter(tags=["streaming"])

logger = logging.getLogger(__name__)

_PING_INTERVAL       = 10   # server sends a keepalive ping every N seconds
_PING_TIMEOUT        = 4    # if we can't write the ping within N seconds -> presumed busy
_PING_RETRY_INTERVAL =  5   # after a timeout (event loop busy), retry after this many seconds

# SSE support — gracefully degraded if sse-starlette is not installed
try:
    from sse_starlette.sse import EventSourceResponse as _EventSourceResponse
    _SSE_AVAILABLE = True
except ImportError:
    _EventSourceResponse = None
    _SSE_AVAILABLE = False


async def _safe_ws_close(ws: WebSocket, *, code: int, reason: str = "") -> None:
    """Best-effort WebSocket close that never escapes protocol-race errors."""
    try:
        await ws.close(code=code, reason=reason)
    except Exception as exc:
        logger.debug("WebSocket close ignored: %s", exc)


async def _ws_keepalive(ws: WebSocket) -> None:
    """
    Background task: sends a server-side ping every _PING_INTERVAL seconds.
    This resets the client's 45-second watchdog and keeps NAT/proxy sessions alive.

    Runs as a sibling asyncio.Task alongside the receive loop — completely
    separate from client messages, so there is NEVER a ping-pong feedback loop.
    When the send fails (dead socket) this task exits quietly; the receive loop
    will also error on the next read and close the connection.

    TimeoutError is caught separately and retried — the event loop can be
    briefly saturated during a scanner cycle (477 ticker_update broadcasts)
    which delays the ping write beyond _PING_TIMEOUT.  Retrying keeps the
    keepalive task alive so the next ping succeeds once the burst clears.
    """
    while True:
        try:
            await asyncio.sleep(_PING_INTERVAL)
            await asyncio.wait_for(
                ws.send_json({
                    "type": "ping",
                    "server_ts": time.time(),
                    "source": "backend_ws",
                }),
                timeout=float(_PING_TIMEOUT),
            )
        except asyncio.TimeoutError:
            await asyncio.sleep(_PING_RETRY_INTERVAL)
        except Exception:
            break      # socket gone — receive loop handles cleanup


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    # Accept the connection, then authenticate via either:
    #   Fast path  — valid "token" query-parameter (zero-latency, already in URL)
    #   Slow path  — {"type":"auth","token":"..."} first message within 10 seconds
    # Both paths call the same decode_token / is_blacklisted checks.
    await ws.accept()

    def _verify_token(token: str) -> bool:
        try:
            from auth.utils import decode_token, is_blacklisted
            _p = decode_token(token)
            return _p.get("type") == "access" and not is_blacklisted(_p.get("jti", ""))
        except Exception:
            return False

    _authed = False

    # Fast path: token in query param (client already attached it to the URL)
    _qtoken = ws.query_params.get("token", "")
    if _qtoken:
        _authed = _verify_token(_qtoken)

    # Slow path: wait for first-message auth only when no query token was given.
    # If a query token was provided but was invalid, reject immediately — don't
    # wait for a second auth message (that would deadlock the client).
    if not _authed and not _qtoken:
        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
            msg = json.loads(raw)
            if msg.get("type") == "auth":
                _authed = _verify_token(msg.get("token", ""))
        except (asyncio.TimeoutError, Exception):
            pass

    if not _authed:
        await _safe_ws_close(ws, code=4001)
        return

    await manager.connect(ws)
    logger.info(f"WebSocket client connected. Total: {len(manager.active)}")
    keepalive = asyncio.create_task(_ws_keepalive(ws))
    try:
        from agent.market_regime import get_regime
        from agent.market_hours import get_session_info
        from main import _current_signal_snapshot, _get_universe_total

        regime  = get_regime()
        session = get_session_info()

        # Choose best available signals: live > loaded persisted snapshot
        live_sigs, _last_scan, from_cache = _current_signal_snapshot()

        if live_sigs:
            # Full update so the tab is immediately usable
            await ws.send_text(_dumps({
                "type":          "update",
                "signals":       live_sigs,
                "regime":        regime.to_dict(),
                "session":       session,
                "from_cache":    from_cache,
                "scanned_count": len(live_sigs),
            }))
        else:
            # No data yet (cold start) — send a status frame so the loading
            # screen can show regime/session info rather than spinning blindly.
            await ws.send_text(_dumps({
                "type":       "scan_status",
                "scanning":   True,
                "regime":     regime.to_dict(),
                "session":    session,
                "n_total":    _get_universe_total(),
            }))

        # Drain incoming client messages.
        while True:
            await ws.receive_text()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WebSocket loop error: {e}")
    finally:
        keepalive.cancel()
        manager.disconnect(ws)
        logger.info(f"WebSocket client disconnected. Total: {len(manager.active)}")


if _SSE_AVAILABLE:
    @router.get("/stream/signals")
    async def stream_signals(request: Request):
        """Server-Sent Events stream — pushes signal updates every 5s."""
        async def event_generator():
            while True:
                if await request.is_disconnected():
                    break
                try:
                    from main import _SCANNER_ENABLED
                    if _SCANNER_ENABLED:
                        from agent.scanner import scanner
                        signals = scanner.get_last_signals()
                    else:
                        signals = []
                    data = json.dumps([s.to_dict() for s in signals[:50]])
                    yield {"event": "signals", "data": data}
                except Exception:
                    pass
                await asyncio.sleep(5)
        return _EventSourceResponse(event_generator())
else:
    @router.get("/stream/signals")
    async def stream_signals_fallback(request: Request):
        """
        Fallback polling endpoint (sse-starlette not installed).
        Returns the latest 50 signals as JSON.  Poll every 5 s from the client.
        """
        try:
            from main import _SCANNER_ENABLED
            if _SCANNER_ENABLED:
                from agent.scanner import scanner
                signals = scanner.get_last_signals()
            else:
                signals = []
            return JSONResponse({"signals": [s.to_dict() for s in signals[:50]]})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
