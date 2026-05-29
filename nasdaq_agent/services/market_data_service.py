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
    Publish Schwab streamer + MD poller status every 15s.

    Write order (both best-effort):
      1. PostgreSQL service_state  — durable, expires_at = NOW() + 60s
      2. Valkey SETEX              — fast-path cache, TTL = 60s (backward compat)
    """
    import json as _json

    _TTL = 60  # seconds — matches docker-compose healthcheck expectation

    while not _runner.stopped:
        try:
            from agent.broker.schwab_streamer import get_streamer_status
            payload = {"ts": time.time(), **get_streamer_status()}

            # ── 1. PostgreSQL (source of truth) ───────────────────────────────
            try:
                from agent.service_state import set_state
                set_state("scanner:streamer", payload, ttl_s=_TTL)
            except Exception as exc:
                _log.debug("scanner:streamer PG write failed: %s", exc)

            # ── 2. Valkey (fast-path cache) ───────────────────────────────────
            try:
                from agent.valkey_client import _get_client
                client = _get_client()
                if client:
                    client.setex("scanner:streamer", _TTL, _json.dumps(payload))
            except Exception as exc:
                _log.debug("scanner:streamer Valkey write failed: %s", exc)

        except Exception as exc:
            _log.debug("scanner:streamer status collection failed: %s", exc)
        time.sleep(15)


# ── Token hot-reload ──────────────────────────────────────────────────────────

def _token_reload_loop() -> None:
    """
    Subscribe to the schwab:tokens_refreshed Valkey pub/sub channel.

    When the auth endpoint or background refresh writes new Schwab tokens
    it publishes to this channel.  This loop detects the event and calls
    _start() so the streamer / MD poller reconnect with the new tokens
    immediately — without a container restart.

    Reconnection is non-destructive: _start() calls start_streamer /
    start_md_poller which each guard against double-starting internally.
    """
    while not _runner.stopped:
        try:
            from agent.valkey_client import _get_client
            client = _get_client()
            if client is None:
                time.sleep(30)
                continue

            pubsub = client.pubsub()
            pubsub.subscribe("schwab:tokens_refreshed")
            _log.info("[token_reload] Subscribed to schwab:tokens_refreshed")

            for message in pubsub.listen():
                if _runner.stopped:
                    break
                if message and message.get("type") == "message":
                    _log.info(
                        "[token_reload] Token refresh detected — restarting data sources"
                    )
                    try:
                        from config import NASDAQ_TICKERS
                        _start(list(NASDAQ_TICKERS))
                    except Exception as exc:
                        _log.warning("[token_reload] Restart after token refresh failed: %s", exc)

        except Exception as exc:
            _log.debug("[token_reload] pub/sub error: %s — retrying in 30 s", exc)
            time.sleep(30)


# ── Bar-history gap filler ────────────────────────────────────────────────────

def _bar_accumulator_loop(tickers: list[str]) -> None:
    """
    Fill ohlcv_bars for any ticker that has fewer bars than the training
    minimum so ML retrain never hits Schwab REST cold.

    Waits 120 s at startup so the WebSocket streamer and MD poller finish
    their own initialisation before we add background REST load.
    Repeats every 24 h so non-streamed tickers stay current.
    """
    _log.info(
        "[BarAccumulator] Starting — will fill 1-min/15-min/daily gaps for %d tickers",
        len(tickers),
    )
    time.sleep(120)

    while not _runner.stopped:
        try:
            from agent.historical_cache import fill_history_gaps
            _log.info("[BarAccumulator] Running fill_history_gaps for all intervals…")
            for _iv, _out, _min in [("1min", 3900, 200), ("15min", 5000, 200), ("1day", 500, 100)]:
                fill_history_gaps(tickers, interval=_iv, min_bars=_min, outputsize=_out)
        except Exception as exc:
            _log.warning("[BarAccumulator] error: %s", exc)
        time.sleep(86400)


# ── Token status publisher ────────────────────────────────────────────────────

def _publish_token_status_loop() -> None:
    """
    Publish Schwab token TTL/connection status every 30s so the web-api
    Schwab & Data Sources panel can display accurate state.

    web-api has NASDAQ_MARKET_DATA_ENABLED=0 so it never loads tokens into
    memory; its get_token_status() always returns connected=False.  This loop
    writes the real status from market-data to shared storage so broker_status
    can read it cross-container.
    """
    import json as _json

    _TTL = 90  # 3× the publish interval — stale after 3 missed cycles

    while not _runner.stopped:
        try:
            from agent.broker.schwab_auth import get_token_status, get_md_token_status
            trader_status = get_token_status()
            md_status     = get_md_token_status()

            # Mark the source so web-api knows this is cross-container data
            trader_status["_source"] = "market-data"
            md_status["_source"]     = "market-data"

            # ── PostgreSQL (durable) ──────────────────────────────────────────
            try:
                from agent.service_state import set_state
                set_state("schwab:token_status:trader",     trader_status, ttl_s=_TTL)
                set_state("schwab:token_status:marketdata", md_status,     ttl_s=_TTL)
            except Exception as exc:
                _log.debug("Token status PG write failed: %s", exc)

            # ── Valkey (fast-path) ────────────────────────────────────────────
            try:
                from agent.valkey_client import _get_client
                client = _get_client()
                if client:
                    client.setex("schwab:token_status:trader",     _TTL, _json.dumps(trader_status))
                    client.setex("schwab:token_status:marketdata", _TTL, _json.dumps(md_status))
            except Exception as exc:
                _log.debug("Token status Valkey write failed: %s", exc)

        except Exception as exc:
            _log.debug("Token status collection failed: %s", exc)
        time.sleep(30)


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

    from agent.service_heartbeat import start_service_heartbeat
    start_service_heartbeat("market-data", _runner)

    threading.Thread(target=_health_loop,                  daemon=True, name="md-health").start()
    threading.Thread(target=_publish_streamer_status_loop, daemon=True, name="md-streamer-status").start()
    threading.Thread(target=_publish_token_status_loop,    daemon=True, name="md-token-status").start()
    threading.Thread(target=_token_reload_loop,            daemon=True, name="md-token-reload").start()
    threading.Thread(target=_bar_accumulator_loop,         daemon=True, name="BarAccumulator",
                     args=(list(NASDAQ_TICKERS),)).start()

    _log.info("Market data running — waiting for SIGTERM/SIGINT …")
    _runner.register_signals()
    _runner.wait()
    _log.info("=== market_data_service stopped ===")


if __name__ == "__main__":
    main()
