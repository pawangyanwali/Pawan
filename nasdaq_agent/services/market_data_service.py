#!/usr/bin/env python3
"""
market_data_service — Schwab price publisher (standalone container).

Responsibilities:
  1. Maintain the Schwab WebSocket streamer connection.
  2. Run the REST MD poller as fallback / supplement.
  3. Publish every price update to Valkey (md:prices hash + pub/sub).
  4. Publish 1-min candles to Valkey (md:1m:{ticker} lists) on each bar close
     so the scanner container can read them via data_fetcher Tier A½.
  5. Publish streamer/poller status to Valkey (scanner:streamer, 15s cadence)
     so the web-api Infrastructure panel shows accurate state.

Interface contract:
  WRITES  Valkey md:prices          — spot quotes hash + pub/sub (~300 ms)
  WRITES  Valkey md:1m:{ticker}     — Redis LIST of last 200 1-min candles
  WRITES  Valkey scanner:streamer   — JSON status blob (TTL 60s, every 15s)

Environment variables:
  NASDAQ_MD_STARTUP_DELAY_S   float  poller warm-up delay in seconds (default 10)
  LOG_LEVEL                   DEBUG|INFO|WARNING  (default INFO)
  All Schwab / Valkey env vars inherited from environment / .env
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv
load_dotenv()

from services._base import configure_logging, ServiceRunner

_log    = configure_logging("market-data")
_runner = ServiceRunner("market-data")


# ── Market data startup ───────────────────────────────────────────────────────

def _start(tickers: list[str]) -> None:
    started = False

    if os.getenv("SCHWAB_CLIENT_ID"):
        try:
            from agent.broker.schwab_auth import load_stored_tokens
            from agent.broker.schwab_streamer import start_streamer
            if load_stored_tokens():
                start_streamer(tickers)
                _log.info("Schwab WebSocket streamer started (%d tickers)", len(tickers))
                started = True
            else:
                _log.warning("Schwab A+T tokens missing — visit /schwab/auth/at")
        except Exception as exc:
            _log.warning("Streamer start failed: %s", exc)

    if os.getenv("SCHWAB_MD_CLIENT_ID"):
        try:
            from agent.broker.schwab_auth import load_stored_md_tokens
            from agent.broker.schwab_streamer import start_md_poller
            if load_stored_md_tokens():
                delay = float(os.getenv("NASDAQ_MD_STARTUP_DELAY_S", "10"))
                start_md_poller(
                    tickers, interval=1.0,
                    parallel_batches=2, startup_delay_s=delay,
                )
                _log.info("Schwab MD poller started (delay=%.0fs)", delay)
                started = True
            else:
                _log.warning("Schwab MD tokens missing — visit /schwab/auth/md")
        except Exception as exc:
            _log.warning("MD poller start failed: %s", exc)

    if not started:
        _log.warning("No Schwab data source active — no prices will be published")


# ── Streamer status publisher ─────────────────────────────────────────────────

def _publish_streamer_status_loop() -> None:
    """
    Publish Schwab streamer + MD poller status to Valkey key scanner:streamer
    every 15s so the web-api Infrastructure panel shows accurate state.
    This loop now lives in market_data_service (where the streamer runs).
    """
    import json as _json

    try:
        from agent.valkey_client import _get_client
    except ImportError:
        return

    while not _runner.stopped:
        try:
            client = _get_client()
            if client:
                from agent.broker.schwab_streamer import get_streamer_status
                payload = _json.dumps({"ts": time.time(), **get_streamer_status()})
                client.setex("scanner:streamer", 60, payload)
        except Exception as exc:
            _log.debug("scanner:streamer publish failed: %s", exc)
        time.sleep(15)


# ── Health / stats logger ─────────────────────────────────────────────────────

def _health_loop() -> None:
    """Log Valkey publish stats every 60 s so we can diagnose stale data."""
    while not _runner.stopped:
        try:
            from agent.valkey_client import get_stats
            s = get_stats()
            _log.info(
                "Valkey stats — published: %d  last_ago: %ss  connected: %s",
                s.get("publish_count", 0),
                s.get("last_publish_ago_s", "?"),
                s.get("connected", False),
            )
        except Exception:
            pass
        time.sleep(60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import threading
    from config import NASDAQ_TICKERS

    _log.info("=== market_data_service starting ===")
    _start(list(NASDAQ_TICKERS))

    threading.Thread(target=_health_loop,                  daemon=True, name="md-health").start()
    threading.Thread(target=_publish_streamer_status_loop, daemon=True, name="md-streamer-status").start()

    _log.info("Market data running — waiting for SIGTERM/SIGINT …")
    _runner.register_signals()
    _runner.wait()
    _log.info("=== market_data_service stopped ===")


if __name__ == "__main__":
    main()
