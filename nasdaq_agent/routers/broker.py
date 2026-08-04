"""
Broker / Schwab routes:
  GET  /schwab/auth
  GET  /schwab/callback
  GET  /schwab/auth/at
  GET  /schwab/callback/at
  GET  /schwab/auth/md
  GET  /schwab/callback/md
  GET  /api/broker/status
  POST /api/broker/auth
  GET  /api/broker/positions
  GET  /api/broker/orders
  POST /api/broker/auto-trade/{enabled}
  POST /api/broker/order
  GET  /api/market/streamer
  POST /api/market/streamer/start
  GET  /api/market/movers
  GET  /api/market/hours
  GET  /api/market/iv/{ticker}
"""

import asyncio
import html as _html
import json
import logging
import os
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from auth.dependencies import require_viewer, require_trader, require_admin, AuthenticatedUser

from agent.broker.schwab_auth import (
    load_stored_tokens, load_stored_md_tokens,
    get_token_status, get_md_token_status,
    build_auth_url, exchange_auth_code,
    build_md_auth_url, exchange_md_auth_code,
)
from agent.broker.schwab_client import get_positions, get_account_summary, get_orders
from agent.broker.order_bridge import maybe_place_tos_order, get_daily_status

router = APIRouter(tags=["broker"])

logger = logging.getLogger(__name__)

_tos_auto_trade: bool = os.getenv("SCHWAB_AUTO_TRADE", "false").lower() == "true"


def _schwab_callback_url(request: Request) -> str:
    """
    Return the public callback URL registered in the Schwab Developer Portal.
    Reads SCHWAB_CALLBACK_URL from .env first (required when behind a reverse
    proxy such as IIS, where request.base_url would return http://localhost:8000/).
    Falls back to constructing from the incoming request for local dev.
    """
    explicit = os.getenv("SCHWAB_CALLBACK_URL", "").strip().rstrip("/")
    if explicit:
        return explicit + "/schwab/callback"
    return str(request.base_url).rstrip("/") + "/schwab/callback"


def _get_streamer_status():
    try:
        from agent.broker.schwab_streamer import get_streamer_status

        status = get_streamer_status()
        if not (status.get("ws_streamer") or {}).get("running"):
            remote = _get_cross_container_token_status("market-data:status")
            if remote:
                status = remote
        return status
    except Exception:
        return {"connected": False, "disabled": True}


def _get_cross_container_token_status(key: str) -> dict | None:
    """
    Read token status published by the market-data container.

    web-api never loads Schwab tokens (NASDAQ_MARKET_DATA_ENABLED=0) so its
    in-memory get_token_status() always shows disconnected.  market-data writes
    the real status to Valkey and PostgreSQL every 30s; this reads it back so
    the Schwab & Data Sources panel shows accurate state.

    Try Valkey first (fast, ~1 ms), fall back to PostgreSQL (durable).
    """
    # Valkey fast-path
    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        if client:
            raw = client.get(key)
            if raw:
                return json.loads(raw)
    except Exception:
        pass
    # PostgreSQL fallback
    try:
        from agent.service_state import get_state
        data = get_state(key, ignore_expiry=False)
        if data:
            return data
    except Exception:
        pass
    return None


def _start_md_poller_and_register(tickers):
    try:
        from main import _ensure_tick_broadcast_registered
        _ensure_tick_broadcast_registered()
        if os.getenv("NASDAQ_MARKET_DATA_ENABLED", "1") == "0":
            logger.info(
                "Schwab Market Data OAuth complete; market-data container owns the REST poller."
            )
            return

        from agent.broker.schwab_streamer import start_md_poller, is_streamer_ready
        if not is_streamer_ready():
            start_md_poller(list(tickers), interval=1.0, parallel_batches=2)
    except Exception:
        pass


@router.get("/schwab/auth")
async def schwab_web_auth(request: Request):
    """Redirect browser to Schwab Market Data OAuth login."""
    return RedirectResponse(url="/schwab/auth/md")


@router.get("/schwab/callback")
async def schwab_web_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    """Schwab OAuth callback — handles Market Data app token exchange."""
    if error or not code:
        html = f"""<html><body style="font-family:sans-serif;padding:40px">
        <h2 style="color:#e53e3e">Schwab Auth Failed</h2>
        <p>{_html.escape(error) or 'No code received.'}</p>
        <p><a href="/schwab/auth/md">Try again</a></p></body></html>"""
        return HTMLResponse(html, status_code=400)

    redirect_uri = _schwab_callback_url(request)
    success, reason = exchange_md_auth_code(code, state, redirect_uri)
    if success:
        _start_md_poller_and_register(
            __import__("config").NASDAQ_TICKERS
        )
        html = """<html><body style="font-family:sans-serif;padding:40px;background:#f0fff4">
        <h2 style="color:#276749">&#10003; Schwab Market Data Connected!</h2>
        <p>Tokens saved. REST quotes, IV, movers and price history are now live.</p>
        <p>Real-time 1-second quote poller started for all NASDAQ tickers.</p>
        <p><a href="/">&#8592; Back to Dashboard</a></p></body></html>"""
        return HTMLResponse(html)
    else:
        html = f"""<html><body style="font-family:sans-serif;padding:40px">
        <h2 style="color:#e53e3e">Token Exchange Failed</h2>
        <p><b>Reason:</b> {_html.escape(reason) if reason else 'See server logs.'}</p>
        <p><b>redirect_uri used:</b> <code>{_html.escape(redirect_uri)}</code></p>
        <p>If the redirect_uri above does not match what is registered in the Schwab
        Developer Portal, set <code>SCHWAB_CALLBACK_URL=https://scalpingstocksai.com</code>
        in your <code>.env</code> file and restart.</p>
        <p><a href="/schwab/auth/md">Try again</a></p></body></html>"""
        return HTMLResponse(html, status_code=500)


@router.get("/schwab/auth/at")
async def schwab_at_web_auth(request: Request):
    """Redirect browser to Schwab Accounts+Trading OAuth login (for WebSocket streamer)."""
    from agent.broker.schwab_auth import _trader, build_auth_url
    if not _trader.is_configured():
        return HTMLResponse(
            "<html><body style='font-family:sans-serif;padding:40px'>"
            "<h2>SCHWAB_CLIENT_ID not set in .env</h2>"
            "<p>Add your Accounts+Trading app credentials and restart:</p>"
            "<pre>SCHWAB_CLIENT_ID=&lt;your-app-key&gt;\n"
            "SCHWAB_CLIENT_SECRET=&lt;your-secret&gt;</pre>"
            "<p>Register <code>https://scalpingstocksai.com/schwab/callback/at</code> "
            "as the callback URL in the Schwab Developer Portal.</p>"
            "</body></html>",
            status_code=400,
        )
    redirect_uri = _schwab_callback_url(request).replace("/schwab/callback", "/schwab/callback/at")
    return RedirectResponse(url=build_auth_url(redirect_uri))


@router.get("/schwab/callback/at")
async def schwab_at_web_callback(
    request: Request, code: str = "", state: str = "", error: str = ""
):
    """OAuth callback for the Schwab Accounts+Trading app (WebSocket streamer)."""
    from agent.broker.schwab_auth import exchange_auth_code
    if error or not code:
        html = (f"<html><body style='font-family:sans-serif;padding:40px'>"
                f"<h2 style='color:#e53e3e'>Schwab A+T Auth Failed</h2>"
                f"<p>{_html.escape(error) or 'No code received.'}</p>"
                f"<p><a href='/schwab/auth/at'>Try again</a></p></body></html>")
        return HTMLResponse(html, status_code=400)

    redirect_uri = _schwab_callback_url(request).replace("/schwab/callback", "/schwab/callback/at")
    success, reason = exchange_auth_code(code, state, redirect_uri)
    if success:
        html = ("<html><body style='font-family:sans-serif;padding:40px;background:#f0fff4'>"
                "<h2 style='color:#276749'>&#10003; Schwab Accounts+Trading Connected!</h2>"
                "<p>Tokens saved. token-service and market-data will adopt the new token automatically.</p>"
                "<p>The WebSocket streamer stays owned by the market-data container.</p>"
                "<p><a href='/'>&#8592; Back to Dashboard</a></p></body></html>")
        return HTMLResponse(html)
    else:
        html = (f"<html><body style='font-family:sans-serif;padding:40px'>"
                f"<h2 style='color:#e53e3e'>Token Exchange Failed</h2>"
                f"<p><b>Reason:</b> {_html.escape(reason) if reason else 'See server logs.'}</p>"
                f"<p><a href='/schwab/auth/at'>Try again</a></p></body></html>")
        return HTMLResponse(html, status_code=500)


@router.get("/schwab/auth/md")
async def schwab_md_web_auth(request: Request):
    """Redirect browser to Schwab Market Data OAuth login (for REST quotes/IV/movers)."""
    from agent.broker.schwab_auth import _market_data
    if not _market_data.is_configured():
        return HTMLResponse(
            "<h2>SCHWAB_MD_CLIENT_ID not set in .env</h2>"
            "<p>Add the Market Data app credentials and restart.</p>",
            status_code=400,
        )
    redirect_uri = _schwab_callback_url(request)
    return RedirectResponse(url=build_md_auth_url(redirect_uri))


@router.get("/schwab/callback/md")
async def schwab_md_web_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    """Legacy callback path — redirects to the active /schwab/callback handler."""
    params = str(request.url.query)
    target = f"/schwab/callback?{params}" if params else "/schwab/callback"
    return RedirectResponse(url=target)


@router.get("/api/broker/status")
async def broker_status(_user: AuthenticatedUser = Depends(require_viewer)):
    """Connection status for both Schwab apps, token TTLs, account info."""
    from config import SCHWAB_ENABLED
    try:
        ts    = get_token_status()
        ts_md = get_md_token_status()

        # web-api has NASDAQ_MARKET_DATA_ENABLED=0 and never loads tokens, so its
        # in-memory get_token_status() always returns connected=False / ttl=0.
        # Fall back to the status that market-data publishes every 30s.
        if not ts.get("connected") and ts.get("access_token_ttl_s", 0) == 0:
            remote = _get_cross_container_token_status("schwab:token_status:trader")
            if remote:
                ts = remote
        if not ts_md.get("connected") and ts_md.get("access_token_ttl_s", 0) == 0:
            remote_md = _get_cross_container_token_status("schwab:token_status:marketdata")
            if remote_md:
                ts_md = remote_md

        acct  = {}
        if SCHWAB_ENABLED and ts.get("connected"):
            try:
                acct = get_account_summary()
            except Exception:
                pass
        daily = get_daily_status()
        streamer = _get_streamer_status()
        ws_st = streamer.get("ws_streamer", {})
        md_st = streamer.get("md_poller", {})
        return {
            "connected":          ts_md.get("connected", False) or ts.get("connected", False),
            "schwab_enabled":     SCHWAB_ENABLED,
            "market_data_app":    ts_md,
            "trader_app":         ts,
            "ws_streamer": {
                "running":                  ws_st.get("running", False),
                "connected":                ws_st.get("connected", False),
                "desired_subscriptions":    ws_st.get("desired_subscriptions", 0),
                "sent_subscriptions":       ws_st.get("sent_subscriptions", 0),
                "acknowledged_subscriptions": ws_st.get(
                    "acknowledged_subscriptions", 0
                ),
                "pending_subscription_requests": ws_st.get(
                    "pending_subscription_requests", 0
                ),
                "subscription_coverage_pct": ws_st.get(
                    "subscription_coverage_pct", 0.0
                ),
                "seen_quotes":              ws_st.get("seen_quotes", 0),
                "active_quotes_60s":        ws_st.get("active_quotes_60s", 0),
                "live_quotes":              ws_st.get("live_quotes", 0),
                "fresh_coverage_pct":        ws_st.get("fresh_coverage_pct", 0.0),
                "last_data_age_s":          ws_st.get("last_data_age_s"),
                "auth_url":                 "/schwab/auth/at",
                "error":                    ws_st.get("error"),
            },
            "md_poller": {
                "running":     md_st.get("running", False),
                "cycle":       md_st.get("cycle", 0),
                "live_quotes": md_st.get("live_quotes", 0),
                "auth_url":    "/schwab/auth/md",
                "error":       md_st.get("error"),
            },
            "account":    acct,
            "daily":      daily,
            "auto_trade": _tos_auto_trade,
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


@router.post("/api/broker/auth")
async def broker_auth(
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Initiate Schwab OAuth flow — redirect browser to /schwab/auth instead."""
    from config import SCHWAB_ENABLED
    if not SCHWAB_ENABLED:
        return {"success": False, "error": "Schwab is disabled (SCHWAB_ENABLED=false in .env)"}
    return {"success": False, "error": "Use the browser flow: visit /schwab/auth to authorise"}


@router.get("/api/broker/positions")
async def broker_positions(
    _current: AuthenticatedUser = Depends(require_trader),
):
    """Current open positions in the ThinkorSwim paper account."""
    from config import SCHWAB_ENABLED
    if not SCHWAB_ENABLED:
        return {"positions": [], "schwab_enabled": False}
    try:
        loop = asyncio.get_running_loop()
        positions = await loop.run_in_executor(None, get_positions)
        return {"positions": positions}
    except Exception as e:
        return {"positions": [], "error": str(e)}


@router.get("/api/broker/orders")
async def broker_orders(
    _current: AuthenticatedUser = Depends(require_trader),
):
    """Recent working orders."""
    from config import SCHWAB_ENABLED
    if not SCHWAB_ENABLED:
        return {"orders": [], "schwab_enabled": False}
    try:
        loop = asyncio.get_running_loop()
        orders = await loop.run_in_executor(None, get_orders)
        return {"orders": orders}
    except Exception as e:
        return {"orders": [], "error": str(e)}


@router.post("/api/broker/auto-trade/{enabled}")
async def broker_auto_trade(
    enabled: str,
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Toggle fully-automatic order placement (true/false)."""
    global _tos_auto_trade
    _tos_auto_trade = enabled.lower() == "true"
    return {"auto_trade": _tos_auto_trade}


@router.post("/api/broker/order")
async def broker_manual_order(
    body: dict,
    _current: AuthenticatedUser = Depends(require_trader),
):
    """
    Manually trigger a bracket order for a ticker already in the signal list.
    Body: { "ticker": "NVDA" }
    """
    import types as _types

    _SCANNER_ENABLED = os.getenv("NASDAQ_SCANNER_ENABLED", "1") != "0"

    ticker = body.get("ticker", "").upper()

    sig = None
    if _SCANNER_ENABLED:
        try:
            from agent.scanner import scanner
            sig = next((s for s in scanner.signals if s.ticker == ticker), None)
        except Exception:
            pass

    if sig is None and not _SCANNER_ENABLED:
        try:
            from agent.signal_snapshot import read_latest as _snap_read2
            snap2 = _snap_read2()
            if snap2:
                match = next((s for s in snap2.get("signals", []) if s.get("ticker") == ticker), None)
                if match:
                    sig = _types.SimpleNamespace(**match)
        except Exception:
            pass
    if not sig:
        return {"placed": False, "reason": f"{ticker} not in current scan"}
    result = maybe_place_tos_order(sig)
    return result


@router.get("/api/market/streamer")
async def streamer_status_endpoint():
    """WebSocket streamer health: connected, live quote count, futures bias."""
    try:
        status = _get_streamer_status()
        ts = get_token_status()
        if not ts.get("connected") and ts.get("access_token_ttl_s", 0) == 0:
            remote = _get_cross_container_token_status("schwab:token_status:trader")
            if remote:
                ts = remote
        status["schwab_connected"]       = ts.get("connected", False)
        status["access_token_ttl_s"]     = ts.get("access_token_ttl_s", 0)
        status["refresh_token_ttl_s"]    = ts.get("refresh_token_ttl_s", 0)
        return status
    except Exception as e:
        return {"connected": False, "error": str(e)}


@router.post("/api/market/streamer/start")
async def streamer_start_endpoint(
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Manually (re)start the Schwab WebSocket streamer."""
    ts = get_token_status()
    if not ts.get("connected"):
        return {"started": False, "reason": "Schwab not authenticated — visit /schwab/auth first"}
    try:
        from agent.broker.schwab_streamer import start_streamer
        loop = asyncio.get_running_loop()
        from config import NASDAQ_TICKERS
        await loop.run_in_executor(None, lambda: start_streamer(list(NASDAQ_TICKERS)))
        return {"started": True, "tickers": len(NASDAQ_TICKERS)}
    except Exception as e:
        return {"started": False, "reason": str(e)}


@router.get("/api/market/movers")
async def market_movers(index: str = "$COMPX", sort: str = "PERCENT_CHANGE_UP", freq: int = 0):
    """Top movers for an index via Schwab. index: $COMPX | $SPX | $DJI"""
    try:
        from agent.broker.schwab_market_data import fetch_movers
        loop = asyncio.get_running_loop()
        movers = await loop.run_in_executor(None, lambda: fetch_movers(index, sort, freq))
        return {"movers": movers, "index": index, "sort": sort}
    except Exception as e:
        return {"movers": [], "error": str(e)}


@router.get("/api/market/hours")
async def market_hours_endpoint(market: str = "equity"):
    """Current market session status via Schwab."""
    try:
        from agent.broker.schwab_market_data import fetch_market_hours
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fetch_market_hours(market))
    except Exception as e:
        return {"is_open": None, "error": str(e)}


@router.get("/api/market/iv/{ticker}")
async def ticker_iv(ticker: str):
    """Implied volatility for a single ticker via Schwab option chains."""
    try:
        from agent.broker.schwab_market_data import fetch_iv
        loop = asyncio.get_running_loop()
        iv = await loop.run_in_executor(None, lambda: fetch_iv(ticker.upper()))
        return {"ticker": ticker.upper(), "iv": iv}
    except Exception as e:
        return {"ticker": ticker, "iv": None, "error": str(e)}
