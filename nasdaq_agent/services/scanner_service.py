#!/usr/bin/env python3
"""
scanner_service — scan loop + market data.

Responsibilities:
  1. Start Schwab WebSocket streamer and/or REST MD poller (market data).
  2. Run the scan loop (Scanner.start_background).
  3. After each cycle, publish scan results to Valkey (scan:latest + scan:notify).
  4. Hot-reload Schwab tokens when web-api publishes schwab:tokens_refreshed.

Interface contract:
  WRITES  Valkey scan:latest  — full JSON snapshot of signals/regime/session
  WRITES  Valkey scan:notify  — lightweight pub/sub wake-up for web-api
  WRITES  Valkey md:prices    — real-time price updates (via schwab_streamer)
  WRITES  PostgreSQL          — signal history, paper trades, backtest records

  The scanner currently reads live prices from the in-process _live_quotes cache
  populated by schwab_streamer.  When data_fetcher is updated to read from
  Valkey instead, market data can be split into its own container.

Environment variables:
  NASDAQ_MARKET_DATA_ENABLED   1|0  start Schwab streamer/poller (default 1)
  NASDAQ_MD_STARTUP_DELAY_S    float  poller warm-up delay in seconds (default 10)
  LOG_LEVEL                    DEBUG|INFO|WARNING  (default INFO)
  All Schwab / Valkey / DB env vars inherited from environment / .env
"""
from __future__ import annotations

import os
import sys
import time

# Bootstrap path before any project imports
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv
load_dotenv()

from services._base import configure_logging, ServiceRunner

_log     = configure_logging("scanner")
_runner  = ServiceRunner("scanner")

_MARKET_DATA_ENABLED = os.getenv("NASDAQ_MARKET_DATA_ENABLED", "1") != "0"


# ── Valkey publish callback ───────────────────────────────────────────────────

def _on_signals(signals: list) -> None:
    """Called by the scanner thread after each completed scan cycle."""
    try:
        from agent.market_regime import get_regime
        from agent.market_hours import get_session_info
        from agent.signal_snapshot import read_latest, write_latest

        sigs_dicts = [s.to_dict() for s in signals]
        if not sigs_dicts:
            previous = read_latest() or {}
            previous_signals = previous.get("signals") or []
            if previous_signals:
                _log.warning(
                    "Scan produced 0 signals; preserving previous dashboard snapshot "
                    "(%d signals) instead of publishing empty results.",
                    len(previous_signals),
                )
                return
            _log.warning(
                "Scan produced 0 signals and no previous dashboard snapshot exists; "
                "skipping empty publish."
            )
            return

        write_latest(
            signals       = sigs_dicts,
            regime        = get_regime().to_dict(),
            session       = get_session_info(),
            scanned_count = len(signals),
        )
        _log.info("Published %d signals to Valkey", len(signals))
    except Exception as exc:
        _log.warning("write_latest failed: %s", exc)


# ── Market data startup ───────────────────────────────────────────────────────

def _start_market_data() -> None:
    """Start Schwab WebSocket streamer and/or REST MD poller."""
    from config import NASDAQ_TICKERS

    # Attempt 1 — WebSocket streamer (real-time Level 1 + 1-min candles)
    if os.getenv("SCHWAB_CLIENT_ID"):
        try:
            from agent.broker.schwab_auth import load_stored_tokens
            from agent.broker.schwab_streamer import start_streamer
            if load_stored_tokens():
                start_streamer(list(NASDAQ_TICKERS))
                _log.info("Schwab WebSocket streamer started")
            else:
                _log.warning("Schwab A+T tokens not found — visit /schwab/auth/at")
        except Exception as exc:
            _log.warning("Schwab streamer not started: %s", exc)

    # Attempt 2 — REST MD poller (fallback / supplement)
    if os.getenv("SCHWAB_MD_CLIENT_ID"):
        try:
            from agent.broker.schwab_auth import load_stored_md_tokens
            from agent.broker.schwab_streamer import start_md_poller
            if load_stored_md_tokens():
                delay = float(os.getenv("NASDAQ_MD_STARTUP_DELAY_S", "10"))
                start_md_poller(
                    list(NASDAQ_TICKERS), interval=1.0,
                    parallel_batches=2, startup_delay_s=delay,
                )
                _log.info("Schwab MD poller started (delay=%.0fs)", delay)
            else:
                _log.warning("Schwab MD tokens not found — visit /schwab/auth/md")
        except Exception as exc:
            _log.warning("Schwab MD poller not started: %s", exc)


# ── Schwab token hot-reload ───────────────────────────────────────────────────

def _token_reload_loop() -> None:
    """
    Subscribe to Valkey channel 'schwab:tokens_refreshed'.
    When the web-api completes a Schwab OAuth flow it publishes to this channel,
    triggering a streamer restart in the scanner container without any manual
    docker restart.
    """
    try:
        from agent.valkey_client import _cfg
        import redis as _r
    except ImportError:
        _log.debug("Valkey/redis not available — Schwab token hot-reload disabled")
        return

    while not _runner.stopped:
        host, port, ssl = _cfg()
        try:
            sub_client = _r.Redis(
                host=host, port=port, ssl=ssl,
                ssl_cert_reqs=None,
                socket_connect_timeout=5,
                socket_timeout=60,
                decode_responses=True,
            )
            pubsub = sub_client.pubsub()
            pubsub.subscribe("schwab:tokens_refreshed")
            _log.info("Subscribed to schwab:tokens_refreshed for token hot-reload")
            for msg in pubsub.listen():
                if _runner.stopped:
                    return
                if msg.get("type") != "message":
                    continue
                _log.info("Schwab tokens refreshed via Valkey — restarting market data")
                try:
                    from agent.broker.schwab_streamer import stop_streamer
                    stop_streamer()
                    time.sleep(1)
                except Exception:
                    pass
                if _MARKET_DATA_ENABLED:
                    _start_market_data()
        except Exception as exc:
            _log.debug("token_reload_loop error: %s — retrying in 30s", exc)
            time.sleep(30)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _log.info("=== scanner_service starting ===")

    import threading
    from agent.scanner import scanner
    from agent.market_hours import refresh_market_hours_cache

    # Warm market-hours cache without blocking startup
    threading.Thread(
        target=refresh_market_hours_cache, daemon=True, name="mh-warm"
    ).start()

    scanner.register_callback(_on_signals)

    if _MARKET_DATA_ENABLED:
        _start_market_data()
    else:
        _log.info("Market data disabled (NASDAQ_MARKET_DATA_ENABLED=0)")

    # Hot-reload: restart market data when new Schwab tokens arrive via OAuth
    # (only relevant when NASDAQ_MARKET_DATA_ENABLED=1 on this container)
    threading.Thread(target=_token_reload_loop, daemon=True, name="token-reload").start()

    scanner.start_background()
    _log.info("Scan loop running — waiting for SIGTERM/SIGINT …")

    _runner.register_signals(on_stop=scanner.stop)
    _runner.wait()

    _log.info("=== scanner_service stopped ===")


if __name__ == "__main__":
    main()
