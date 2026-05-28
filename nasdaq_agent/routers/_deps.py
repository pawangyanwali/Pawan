"""
Shared helpers and globals used across multiple router modules.

ConnectionManager, manager, _event_loop, _sanitize, _NumpyEncoder, _dumps,
and _pt_executor are all defined here so they can be imported by both
main.py (for lifespan / startup callbacks) and any router that needs them,
without creating true circular dependencies.
"""

import asyncio
import json
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Set

from fastapi import WebSocket


# ── Dedicated thread pool for paper-trading DB reads ─────────────────────────
_pt_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="pt_db")


# ── JSON helpers ──────────────────────────────────────────────────────────────

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


# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        # ws is already accepted in websocket_endpoint before auth runs
        self.active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.active.discard(ws)

    async def broadcast(self, message: str) -> None:
        # Snapshot first — prevents RuntimeError if a disconnect() fires during await.
        snapshot = list(self.active)
        if not snapshot:
            return

        async def _send(ws: WebSocket) -> WebSocket | None:
            try:
                # 2s timeout: dead clients cleaned up quickly so they don't delay
                # subsequent broadcasts (keepalive pings, price messages).
                await asyncio.wait_for(ws.send_text(message), timeout=2.0)
                return None
            except Exception:
                return ws

        # Yield once before sending so keepalive pings and price messages can
        # interleave with scanner ticker_update batches on the event loop.
        await asyncio.sleep(0)
        # Send to all clients in parallel — a slow client no longer blocks fast ones.
        dead = await asyncio.gather(*[_send(ws) for ws in snapshot])
        for ws in dead:
            if ws is not None:
                self.active.discard(ws)


manager = ConnectionManager()

# Captured at startup so scanner background threads can schedule broadcasts.
_event_loop: asyncio.AbstractEventLoop | None = None
