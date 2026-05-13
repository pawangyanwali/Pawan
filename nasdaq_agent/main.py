"""
FastAPI entry point.
Serves the static web dashboard and a WebSocket endpoint that pushes
real-time stock signals to all connected clients.
"""

import asyncio
import json
import logging
import os
import numpy as np
from contextlib import asynccontextmanager
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from agent.scanner import scanner, StockSignal
from agent.market_hours import get_session_info
from agent.market_regime import get_regime
from agent.signal_tracker import get_stats, get_recent_signals, get_observation_summary
from agent.position_sizing import calculate as calc_position
from agent.paper_trading import get_summary as pt_summary, get_open_trades, get_closed_trades, get_daily_pnl, get_today_pnl, get_equity_curve, get_weekly_pnl, get_ticker_pnl
from agent.macro_calendar import check_macro_event, get_upcoming_events
from agent.live_backtest import get_performance_stats, get_tracking_signals, get_recent_resolved, get_price_path
from agent.backtest_reporter import get_broadcast_summary, get_full_report
from agent.adaptive_filter import get_status as af_get_status
from agent.after_hours_monitor import get_all_biases as ah_get_all
from agent.learning_engine import learning_engine, get_learning_log
from agent.broker.schwab_auth import load_stored_tokens, get_token_status, start_auth_flow
from agent.broker.schwab_client import get_positions, get_account_summary, get_orders
from agent.broker.order_bridge import maybe_place_tos_order, get_daily_status
from config import (
    DEFAULT_ACCOUNT_SIZE, DEFAULT_RISK_PCT, MAX_POSITION_PCT,
    load_watchlist, save_watchlist, NASDAQ_TICKERS,
)


import math


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None so json.dumps never crashes."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize(x) for x in obj.tolist()]
    return obj


class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalars to native Python types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _dumps(obj) -> str:
    return json.dumps(_sanitize(obj), cls=_NumpyEncoder)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.active.discard(ws)

    async def broadcast(self, message: str) -> None:
        dead = set()
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active.discard(ws)


manager = ConnectionManager()

# Captured at startup so the scanner background thread can schedule broadcasts
_event_loop: asyncio.AbstractEventLoop | None = None


def _on_signals(signals: list[StockSignal]) -> None:
    """Callback invoked by the scanner thread; schedule a broadcast on the main loop."""
    if _event_loop is None:
        return

    # Build alert list: high-confidence BUY/SELL signals only
    alerts = [
        {"ticker": s.ticker, "direction": s.prediction,
         "confidence": s.confidence, "price": s.price,
         "session": s.session, "regime": s.regime}
        for s in signals
        if s.prediction in ("BUY", "SELL") and s.confidence >= 70
           and s.rr_qualifies and not s.earnings_blocked
    ]

    # Market breadth — computed from current scan signals
    _above_vwap = sum(1 for s in signals if s.vwap_event in ("ABOVE","RECLAIM","EXTENDED_UP"))
    _below_vwap = sum(1 for s in signals if s.vwap_event in ("BELOW","REJECTION","EXTENDED_DOWN"))
    _bullish_signals = sum(1 for s in signals if s.prediction in ("BUY","STRONG BUY"))
    _bearish_signals = sum(1 for s in signals if s.prediction in ("SELL","STRONG SELL"))
    _total = len(signals) or 1
    breadth = {
        "above_vwap":     _above_vwap,
        "below_vwap":     _below_vwap,
        "pct_above_vwap": round(_above_vwap / _total * 100, 1),
        "bullish":        _bullish_signals,
        "bearish":        _bearish_signals,
        "bias":           "BULLISH" if _bullish_signals > _bearish_signals else "BEARISH" if _bearish_signals > _bullish_signals else "NEUTRAL",
    }

    regime  = get_regime()
    session = get_session_info()
    macro   = check_macro_event()

    try:
        bt_summary = get_broadcast_summary()
    except Exception:
        bt_summary = {}

    try:
        learn_summary = af_get_status()
        learn_compact = {
            "win_rate":          learn_summary.get("current_win_rate", 0.0),
            "target_win_rate":   learn_summary.get("target_win_rate", 62.0),
            "dynamic_threshold": learn_summary.get("dynamic_threshold", 60.0),
            "suppressed_count":  learn_summary.get("suppressed_count", 0),
            "blocked_count":     len(learn_summary.get("blocked_contexts", {})),
            "is_learning":       learn_summary.get("is_learning", False),
        }
    except Exception:
        learn_compact = {}

    try:
        from agent.paper_trading import get_open_trades as _pt_open
        open_trades = {t["ticker"]: t for t in _pt_open()}
    except Exception:
        open_trades = {}

    # ThinkorSwim auto-trade: attempt bracket orders for qualifying signals
    if _tos_auto_trade:
        _tos_results = []
        for sig in signals:
            if sig.prediction in ("BUY", "STRONG BUY", "SELL", "STRONG SELL"):
                try:
                    r = maybe_place_tos_order(sig)
                    if r.get("placed"):
                        _tos_results.append(r)
                except Exception as _te:
                    pass
        if _tos_results:
            logging.getLogger(__name__).info(
                f"TOS auto-trade: {len(_tos_results)} orders placed this cycle"
            )

    payload = _dumps({
        "type":        "update",
        "signals":     [s.to_dict() for s in signals],
        "regime":      regime.to_dict(),
        "session":     session,
        "alerts":      alerts,
        "macro":       macro,
        "backtest":    bt_summary,
        "learning":    learn_compact,
        "open_trades": open_trades,
        "breadth":     breadth,
    })
    asyncio.run_coroutine_threadsafe(manager.broadcast(payload), _event_loop)


# ── ThinkorSwim auto-trade toggle ────────────────────────────────────────────
_tos_auto_trade: bool = os.getenv("SCHWAB_AUTO_TRADE", "false").lower() == "true"

# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _event_loop
    _event_loop = asyncio.get_running_loop()
    scanner.register_callback(_on_signals)
    scanner.start_background()
    learning_engine.start()
    # Try to load stored Schwab tokens (silent if not configured)
    try:
        if os.getenv("SCHWAB_CLIENT_ID"):
            ok = load_stored_tokens()
            if ok:
                logging.getLogger(__name__).info("Schwab broker connected from stored tokens.")
    except Exception as _be:
        logging.getLogger(__name__).warning(f"Schwab token load skipped: {_be}")
    yield
    scanner.stop()
    learning_engine.stop()


app = FastAPI(title="NASDAQ Scalping Agent", lifespan=lifespan)

# Allow IIS (port 80) and any other origin to call the FastAPI backend (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Static files (dashboard)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "web", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── HTTP routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/signals")
async def get_signals():
    """REST endpoint: returns the latest cached scan results."""
    return {
        "last_scan": scanner.last_scan,
        "count": len(scanner.signals),
        "signals": [s.to_dict() for s in scanner.signals],
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "is_running": scanner.is_running,
        "last_scan": scanner.last_scan,
        "tickers_tracked": len(scanner.signals),
        "ws_clients": len(manager.active),
    }


@app.get("/api/regime")
async def get_regime_endpoint():
    """Return current market regime (SPY/QQQ based)."""
    regime = get_regime()
    session = get_session_info()
    return {"regime": regime.to_dict(), "session": session}


@app.get("/api/signal-history")
async def signal_history(ticker: str = None, limit: int = 50):
    """Return recent signal history from SQLite tracker."""
    return {
        "signals": get_recent_signals(limit=limit),
        "stats":   get_stats(ticker=ticker),
    }


@app.get("/api/position-size")
async def position_size_endpoint(
    entry:        float,
    stop:         float,
    account_size: float = DEFAULT_ACCOUNT_SIZE,
    risk_pct:     float = DEFAULT_RISK_PCT,
    confidence:   float = 50.0,
    direction:    str   = "BUY",
):
    """Calculate position size for given entry/stop/account parameters."""
    ps = calc_position(
        account_size=account_size,
        entry=entry,
        stop=stop,
        risk_pct=risk_pct,
        confidence=confidence,
        max_position_pct=MAX_POSITION_PCT,
    )
    return ps.to_dict()


@app.get("/api/paper-trading")
async def paper_trading_endpoint():
    """Return paper trading summary, open and recent closed trades."""
    return {
        "summary":       pt_summary(),
        "open_trades":   get_open_trades(),
        "closed_trades": get_closed_trades(limit=30),
    }


@app.get("/api/paper-trading/daily")
async def paper_daily_pnl():
    """Per-day P&L summary for last 14 days."""
    return {"daily": get_daily_pnl(days=14), "today": get_today_pnl()}


@app.get("/api/paper-trading/performance")
async def paper_performance():
    """Full P&L performance dashboard data."""
    return {
        "summary":       pt_summary(),
        "today":         get_today_pnl(),
        "daily":         get_daily_pnl(days=30),
        "weekly":        get_weekly_pnl(),
        "equity_curve":  get_equity_curve(days=60),
        "ticker_pnl":    get_ticker_pnl(),
    }


@app.get("/api/macro-calendar")
async def macro_calendar_endpoint():
    """Return current macro event status and upcoming events."""
    return {
        "current": check_macro_event(),
        "upcoming": get_upcoming_events(days=14),
    }


@app.get("/api/watchlist")
async def get_watchlist_endpoint():
    """Return user watchlist + base tickers."""
    return {
        "base":      NASDAQ_TICKERS,
        "watchlist": load_watchlist(),
    }


@app.post("/api/watchlist/add")
async def add_to_watchlist(ticker: str):
    """Add a ticker to the watchlist."""
    ticker = ticker.upper().strip()
    wl = load_watchlist()
    if ticker not in wl and ticker not in NASDAQ_TICKERS:
        wl.append(ticker)
        save_watchlist(wl)
    return {"watchlist": load_watchlist()}


@app.post("/api/watchlist/remove")
async def remove_from_watchlist(ticker: str):
    """Remove a ticker from the user watchlist (base tickers cannot be removed)."""
    ticker = ticker.upper().strip()
    wl = [t for t in load_watchlist() if t != ticker]
    save_watchlist(wl)
    return {"watchlist": load_watchlist()}


# ── Live backtest endpoints ───────────────────────────────────────────────────

@app.get("/api/backtest/stats")
async def backtest_stats(lookback_days: int = 30):
    """Full backtest performance report with attribution breakdown."""
    return get_full_report(lookback_days=lookback_days)


@app.get("/api/backtest/tracking")
async def backtest_tracking():
    """Currently open (TRACKING) signals being monitored."""
    return {"tracking": get_tracking_signals()}


@app.get("/api/backtest/recent")
async def backtest_recent(limit: int = 50):
    """Recently resolved backtest signals."""
    recent = get_recent_resolved(limit=limit)
    for r in recent:
        r["outcome_color"] = (
            "#00ff88" if r["status"] == "WIN" else
            "#ef4444" if r["status"] == "LOSS" else
            "#64748b"
        )
    return {"recent": recent}


@app.get("/api/backtest/path/{signal_id}")
async def backtest_path(signal_id: str):
    """Price path bars for a specific signal (for replay/chart)."""
    return {"signal_id": signal_id, "path": get_price_path(signal_id)}


@app.get("/api/learning-status")
async def learning_status():
    """Adaptive filter state — blocked contexts, dynamic threshold, win rate progress."""
    return {
        **af_get_status(),
        "engine":       learning_engine.get_status(),
        "observations": get_observation_summary(),
    }


@app.get("/api/learning-log")
async def learning_log_endpoint(limit: int = 100):
    """Last N learning engine log entries for the dashboard live feed."""
    return {
        "log":    get_learning_log(limit=limit),
        "engine": learning_engine.get_status(),
    }


@app.get("/api/after-hours")
async def after_hours_endpoint():
    """
    Latest after-hours / pre-market snapshot for every scanned ticker.
    Sorted by absolute AH move descending — biggest movers first.
    """
    return {"snapshots": ah_get_all()}


# ── ThinkorSwim / Schwab Broker API ──────────────────────────────────────────

@app.get("/api/broker/status")
async def broker_status():
    """Connection status, token TTLs, account info."""
    try:
        ts = get_token_status()
        acct = {}
        if ts.get("connected"):
            try:
                acct = get_account_summary()
            except Exception:
                pass
        daily = get_daily_status()
        return {
            **ts,
            "account":    acct,
            "daily":      daily,
            "auto_trade": _tos_auto_trade,
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


@app.post("/api/broker/auth")
async def broker_auth():
    """
    Initiate Schwab OAuth flow.
    Opens the user's browser to Schwab login (including MFA).
    Blocks until the user completes login (up to 5 min).
    """
    import asyncio
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, start_auth_flow)
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/broker/positions")
async def broker_positions():
    """Current open positions in the ThinkorSwim paper account."""
    try:
        return {"positions": get_positions()}
    except Exception as e:
        return {"positions": [], "error": str(e)}


@app.get("/api/broker/orders")
async def broker_orders():
    """Recent working orders."""
    try:
        return {"orders": get_orders()}
    except Exception as e:
        return {"orders": [], "error": str(e)}


@app.post("/api/broker/auto-trade/{enabled}")
async def broker_auto_trade(enabled: str):
    """Toggle fully-automatic order placement (true/false)."""
    global _tos_auto_trade
    _tos_auto_trade = enabled.lower() == "true"
    return {"auto_trade": _tos_auto_trade}


@app.post("/api/broker/order")
async def broker_manual_order(body: dict):
    """
    Manually trigger a bracket order for a ticker already in the signal list.
    Body: { "ticker": "NVDA" }
    """
    ticker = body.get("ticker", "").upper()
    sig = next((s for s in scanner.signals if s.ticker == ticker), None)
    if not sig:
        return {"placed": False, "reason": f"{ticker} not in current scan"}
    result = maybe_place_tos_order(sig)
    return result


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    logger.info(f"WebSocket client connected. Total: {len(manager.active)}")
    try:
        # Send current state immediately on connect
        if scanner.signals:
            payload = _dumps({
                "type": "update",
                "signals": [s.to_dict() for s in scanner.signals],
            })
            await ws.send_text(payload)

        while True:
            # Keep connection alive; scanner thread pushes updates
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
        logger.info(f"WebSocket client disconnected. Total: {len(manager.active)}")
    except Exception as e:
        manager.disconnect(ws)
        logger.warning(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
