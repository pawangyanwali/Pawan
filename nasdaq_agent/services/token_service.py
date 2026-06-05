"""
Schwab Token Service — sole owner of OAuth refresh timers.

Responsibilities:
  - Load tokens on startup: Valkey → PostgreSQL → disk
  - Schedule exactly ONE refresh timer per Schwab app (Trader, MarketData)
  - On each refresh: disk (atomic) + Valkey + PostgreSQL are all updated
    by the existing _store() + _pg_save() + token_store.put_token() chain
  - Publish schwab:tokens_refreshed so market-data and web-api reconnect
  - Subscribe to schwab:new_auth:{app} to adopt tokens written by web-api's
    OAuth callback, then (re)start the refresh timer for that app
  - Expose GET /health on port 8080 for the Docker health check

No other container calls load_stored(schedule_refresh=True) — only this one.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time

sys.path.insert(0, "/app")

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("token-service")


# ── Health HTTP server (port 8080) ────────────────────────────────────────────

def _start_health_server() -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *_):
            pass  # silence default access log

    srv = HTTPServer(("0.0.0.0", 8080), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True, name="HealthHTTP")
    t.start()
    logger.info("[token-service] Health server listening on :8080")


# ── New-auth subscription ─────────────────────────────────────────────────────

def _new_auth_listener() -> None:
    """
    Subscribe to schwab:new_auth:trader and schwab:new_auth:marketdata.

    web-api publishes to these channels after a successful OAuth exchange so
    token-service can reload the fresh tokens and restart its refresh timer
    without requiring a container restart.
    """
    from agent.broker.schwab_auth import _trader, _market_data

    _app_map = {
        "schwab:new_auth:trader":     _trader,
        "schwab:new_auth:marketdata": _market_data,
    }

    while True:
        try:
            from agent.valkey_client import _get_client
            client = _get_client()
            if client is None:
                time.sleep(30)
                continue

            pubsub = client.pubsub()
            for channel in _app_map:
                pubsub.subscribe(channel)
            logger.info("[token-service] Subscribed to schwab:new_auth:{trader,marketdata}")

            for msg in pubsub.listen():
                if msg and msg.get("type") == "message":
                    channel = msg.get("channel", b"")
                    if isinstance(channel, bytes):
                        channel = channel.decode()
                    mgr = _app_map.get(channel)
                    if mgr is None:
                        continue
                    logger.info(
                        "[token-service] New OAuth tokens received for %s — reloading", mgr.name
                    )
                    # Reload fresh token into memory and restart the refresh timer.
                    mgr.load_stored(schedule_refresh=True)

        except Exception as exc:
            logger.warning("[token-service] new_auth listener error: %s — retrying in 30s", exc)
            time.sleep(30)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== token-service starting ===")

    _start_health_server()

    # Import the two singleton _TokenManager instances.
    from agent.broker.schwab_auth import _trader, _market_data

    # Load tokens and start refresh timers.  schedule_refresh=True means this
    # container owns the threading.Timer — no other container does this.
    trader_ok = False
    if _trader.is_configured():
        trader_ok = _trader.load_stored(schedule_refresh=True)
        if not trader_ok:
            logger.warning(
                "[token-service] Trader tokens not found or expired — "
                "visit /schwab/auth/at to re-authenticate."
            )
    else:
        logger.info("[token-service] SCHWAB_CLIENT_ID not set — Trader app disabled.")

    md_ok = False
    if _market_data.is_configured():
        md_ok = _market_data.load_stored(schedule_refresh=True)
        if not md_ok:
            logger.warning(
                "[token-service] MarketData tokens not found or expired — "
                "visit /schwab/auth/md to re-authenticate."
            )
    else:
        logger.info("[token-service] SCHWAB_MD_CLIENT_ID not set — MarketData app disabled.")

    logger.info(
        "[token-service] Ready — Trader=%s MarketData=%s",
        "loaded" if trader_ok else "missing",
        "loaded" if md_ok else "missing",
    )

    # Polling fallback: if the pub/sub message from exchange_code() is dropped
    # (Valkey restart, network blip), token-service would never start a refresh
    # timer.  This loop checks every 30s whether the disk token is newer than
    # what's in memory — if so, it reloads and (re)schedules the timer.
    def _disk_poll_loop() -> None:
        while True:
            time.sleep(30)
            for mgr in (_trader, _market_data):
                if not mgr.is_configured():
                    continue
                try:
                    fresh = mgr._load_from_disk()
                    if not fresh or not fresh.get("access_token"):
                        continue
                    disk_stored_at = fresh.get("stored_at", 0)
                    with mgr._lock:
                        mem_stored_at = mgr._tokens.get("stored_at", 0)
                    if disk_stored_at > mem_stored_at + 5:
                        logger.info(
                            "[token-service] %s disk token is newer (stored_at %s > %s) "
                            "— reloading and rescheduling timer.",
                            mgr.name, disk_stored_at, mem_stored_at,
                        )
                        mgr.load_stored(schedule_refresh=True)
                except Exception as exc:
                    logger.debug("[token-service] disk poll error for %s: %s", mgr.name, exc)

    threading.Thread(target=_disk_poll_loop, daemon=True, name="DiskPoll").start()

    # Listen for new OAuth tokens from web-api in the foreground.
    _new_auth_listener()


if __name__ == "__main__":
    main()
