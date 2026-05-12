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
from agent.signal_tracker import get_stats, get_recent_signals
from agent.position_sizing import calculate as calc_position
from agent.paper_trading import get_summary as pt_summary, get_open_trades, get_closed_trades
from agent.macro_calendar import check_macro_event, get_upcoming_events
from config import (
    DEFAULT_ACCOUNT_SIZE, DEFAULT_RISK_PCT, MAX_POSITION_PCT,
    load_watchlist, save_watchlist, NASDAQ_TICKERS,
)


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
    return json.dumps(obj, cls=_NumpyEncoder)

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

    regime  = get_regime()
    session = get_session_info()
    macro   = check_macro_event()

    payload = _dumps({
        "type":    "update",
        "signals": [s.to_dict() for s in signals],
        "regime":  regime.to_dict(),
        "session": session,
        "alerts":  alerts,
        "macro":   macro,
    })
    asyncio.run_coroutine_threadsafe(manager.broadcast(payload), _event_loop)


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _event_loop
    _event_loop = asyncio.get_running_loop()   # capture before spawning thread
    scanner.register_callback(_on_signals)
    scanner.start_background()
    yield
    scanner.stop()


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
