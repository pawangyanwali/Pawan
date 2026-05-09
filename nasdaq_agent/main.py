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
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from agent.scanner import scanner, StockSignal


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
    payload = _dumps({
        "type": "update",
        "signals": [s.to_dict() for s in signals],
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
    }


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
