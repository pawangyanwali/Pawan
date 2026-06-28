#!/usr/bin/env python3
"""
market_data_service — Schwab price publisher (standalone container).

Responsibilities:
  1. Maintain the Schwab WebSocket streamer connection.
  2. Run the REST MD poller as fallback / supplement.
  3. Publish every price update to Valkey (md:prices hash + pub/sub).
  4. Hydrate complete 1-min OHLCV history, then keep it current from
     LEVELONE price and cumulative-volume updates.
  5. Publish streamer/poller status to Valkey (market-data:status, 15s cadence)
     so the web-api Infrastructure panel shows accurate state.

Interface contract:
  WRITES  Valkey md:prices          — spot quotes hash + pub/sub (~300 ms)
  WRITES  Valkey md:1m:{ticker}     — Redis LIST of last 500 1-min candles
  WRITES  Valkey market-data:status — JSON status blob (TTL 60s, every 15s)

Environment variables:
  NASDAQ_MD_STARTUP_DELAY_S   float  poller warm-up delay in seconds (default 10)
  LOG_LEVEL                   DEBUG|INFO|WARNING  (default INFO)
  All Schwab / Valkey env vars inherited from environment / .env
"""
from __future__ import annotations

import os
import json
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
_last_token_generation: dict[str, int] = {}
_last_token_reload_at: dict[str, float] = {}
_bar_hydration_status: dict = {"status": "STARTING", "updated_at": 0.0}


def _validate_universe_candidates(
    candidates: list[str],
    *,
    source: str,
    limit: int | None = None,
    record_failures: bool = False,
) -> int:
    """Validate provider candidates against price, liquidity, and 1m history."""
    from agent.broker.schwab_market_data import (
        fetch_full_quotes,
        fetch_price_history_batch_async,
    )
    from agent.universe_registry import promote_replacement, record_recheck_failure

    symbols = list(dict.fromkeys(str(t).upper() for t in candidates if t))
    if not symbols:
        return 0
    quotes = fetch_full_quotes(symbols)
    frames = fetch_price_history_batch_async(
        symbols,
        interval="1min",
        outputsize=500,
        extended_hours=True,
        background=True,
    )
    promoted = 0
    for ticker in symbols:
        if limit is not None and promoted >= limit:
            break
        frame = frames.get(ticker)
        quote = quotes.get(ticker) or {}
        failure = "NO_USABLE_SCHWAB_1M_HISTORY"
        if frame is not None and not frame.empty:
            volume_column = next(
                (column for column in frame.columns if str(column).lower() == "volume"),
                None,
            )
            if volume_column is not None:
                positive = frame[frame[volume_column] > 0]
                history_bars = len(positive)
                estimated_daily_volume = float(positive[volume_column].tail(390).sum())
                quote_price = float(quote.get("last") or quote.get("mark") or 0.0)
                if promote_replacement(
                    ticker,
                    average_daily_volume=estimated_daily_volume,
                    history_bars=history_bars,
                    quote_price=quote_price,
                    listing_verified=True,
                    source=source,
                ):
                    promoted += 1
                    continue
                if history_bars < 390:
                    failure = f"INSUFFICIENT_1M_HISTORY:{history_bars}/390"
                elif estimated_daily_volume < 500_000:
                    failure = f"INSUFFICIENT_LIQUIDITY:{estimated_daily_volume:.0f}"
                elif quote_price < 1.0:
                    failure = f"PRICE_BELOW_MINIMUM:{quote_price:.4f}"
        if record_failures:
            record_recheck_failure(ticker, reason=failure)
    return promoted


def _discover_universe_replacements() -> int:
    """Promote strictly validated NASDAQ movers until the eligible target is met."""
    try:
        from agent.broker.schwab_market_data import fetch_top_movers_symbols
        from agent.ticker_universe import FULL_UNIVERSE
        from agent.universe_registry import (
            get_universe_registry_summary,
        )

        summary = get_universe_registry_summary()
        target = max(400, int(os.getenv("UNIVERSE_TARGET_ELIGIBLE", "450")))
        slots = max(0, target - int(summary.get("eligible_total") or 0))
        if slots <= 0:
            return 0
        known = set(FULL_UNIVERSE)
        known.update(summary.get("eligible_tickers") or [])
        known.update(
            str(item.get("ticker") or "").upper()
            for item in summary.get("quarantined") or []
        )
        movers = fetch_top_movers_symbols(n=min(40, slots * 2))
        candidates = [
            str(ticker).upper() for ticker in movers
            if ticker and str(ticker).upper() not in known
        ][:max(1, slots * 2)]
        if not candidates:
            return 0

        promoted = _validate_universe_candidates(
            candidates,
            source="SCHWAB_NASDAQ_MOVERS",
            limit=slots,
        )
        if promoted:
            _log.warning(
                "Universe registry promoted %d validated replacement(s)", promoted
            )
        return promoted
    except Exception as exc:
        _log.warning("Universe replacement discovery skipped: %s", exc)
        return 0


# ── Market data startup ───────────────────────────────────────────────────────

def _start(tickers: list[str]) -> None:
    started = False

    if os.getenv("SCHWAB_CLIENT_ID"):
        try:
            from agent.broker.schwab_auth import load_stored_tokens
            from agent.broker.schwab_streamer import start_streamer
            if load_stored_tokens(schedule_refresh=False):
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
            if load_stored_md_tokens(schedule_refresh=False):
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
                set_state("market-data:status", payload, ttl_s=_TTL)
            except Exception as exc:
                _log.debug("market-data:status PG write failed: %s", exc)

            # ── 2. Valkey (fast-path cache) ───────────────────────────────────
            try:
                from agent.valkey_client import _get_client
                client = _get_client()
                if client:
                    client.setex("market-data:status", _TTL, _json.dumps(payload))
            except Exception as exc:
                _log.debug("market-data:status Valkey write failed: %s", exc)

        except Exception as exc:
            _log.debug("market-data:status collection failed: %s", exc)
        time.sleep(15)


def _handle_token_event(payload: dict) -> None:
    """Load rotated Schwab tokens without blindly restarting all data sources."""
    app = str(payload.get("app") or "").lower()
    if app in {"at", "accounts", "accounts_trading"}:
        app = "trader"
    if app not in {"trader", "marketdata"}:
        return

    generation = int(payload.get("generation") or 0)
    if generation and generation <= _last_token_generation.get(app, 0):
        _log.debug("[token_reload] Ignoring duplicate %s generation=%s", app, generation)
        return

    now = time.time()
    if now - _last_token_reload_at.get(app, 0.0) < 5:
        _log.debug("[token_reload] Debounced %s token event", app)
        return
    _last_token_reload_at[app] = now
    if generation:
        _last_token_generation[app] = generation

    from agent.universe_registry import get_runtime_universe

    runtime_tickers = get_runtime_universe()

    if app == "trader":
        from agent.broker.schwab_auth import load_stored_tokens
        from agent.broker.schwab_streamer import get_streamer_status, start_streamer

        if not load_stored_tokens(schedule_refresh=False):
            _log.warning("[token_reload] Trader token event but token could not be loaded")
            return

        status = get_streamer_status()
        ws_status = status.get("ws_streamer", {}) or {}
        if ws_status.get("running") and ws_status.get("connected"):
            _log.info(
                "[token_reload] Trader token loaded (generation=%s); WS already connected",
                generation or "-",
            )
            return

        _log.info(
            "[token_reload] Trader token loaded (generation=%s); starting WS streamer",
            generation or "-",
        )
        start_streamer(runtime_tickers)
        return

    from agent.broker.schwab_auth import load_stored_md_tokens
    from agent.broker.schwab_streamer import is_md_poller_running, start_md_poller

    if not load_stored_md_tokens(schedule_refresh=False):
        _log.warning("[token_reload] MarketData token event but token could not be loaded")
        return

    if is_md_poller_running():
        _log.info(
            "[token_reload] MarketData token loaded (generation=%s); REST poller already running",
            generation or "-",
        )
        return

    delay = float(os.getenv("NASDAQ_MD_STARTUP_DELAY_S", "0"))
    _log.info(
        "[token_reload] MarketData token loaded (generation=%s); starting REST poller",
        generation or "-",
    )
    start_md_poller(
        runtime_tickers,
        interval=1.0,
        parallel_batches=2,
        startup_delay_s=delay,
    )


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
            pubsub.subscribe(
                "schwab:token_rotated:trader",
                "schwab:token_rotated:marketdata",
                "schwab:tokens_refreshed",
            )
            _log.info("[token_reload] Subscribed to Schwab token rotation channels")

            for message in pubsub.listen():
                if _runner.stopped:
                    break
                if message and message.get("type") == "message":
                    try:
                        raw = message.get("data") or "{}"
                        if isinstance(raw, bytes):
                            raw = raw.decode()
                        payload = json.loads(raw)
                        _handle_token_event(payload)
                    except Exception as exc:
                        _log.warning("[token_reload] Token event handling failed: %s", exc)
                    continue

        except Exception as exc:
            _log.debug("[token_reload] pub/sub error: %s — retrying in 30 s", exc)
            time.sleep(30)


# ── Bar-history gap filler ────────────────────────────────────────────────────

def _universe_recheck_loop() -> None:
    """Consume explicit rechecks and hot-reload validated REST subscriptions."""
    while not _runner.stopped:
        try:
            from agent.universe_registry import (
                get_runtime_universe,
                get_universe_registry_summary,
            )
            from agent.valkey_client import _get_client

            client = _get_client()
            raw_symbols = client.smembers("universe:recheck:requested") if client else set()
            queued = {
                item.decode() if isinstance(item, bytes) else str(item)
                for item in raw_symbols
                if item
            }
            # PostgreSQL is the durable command source. Valkey only shortens
            # reaction time and a cache outage cannot strand a CANDIDATE.
            summary = get_universe_registry_summary()
            queued.update(
                str(item.get("ticker") or "").upper()
                for item in summary.get("candidates") or []
                if item.get("ticker")
            )
            symbols = sorted(queued)
            if symbols:
                promoted = _validate_universe_candidates(
                    symbols,
                    source="ADMIN_PROVIDER_RECHECK",
                    record_failures=True,
                )
                if client:
                    client.srem("universe:recheck:requested", *symbols)
                from agent.broker.schwab_streamer import update_md_poller_tickers

                update_md_poller_tickers(get_runtime_universe())
                _log.warning(
                    "Universe recheck completed: %d requested, %d promoted",
                    len(symbols), promoted,
                )
        except Exception as exc:
            _log.warning("Universe recheck cycle failed: %s", exc)
        _runner._stop.wait(30.0)


def _bar_hydration_loop(tickers: list[str]) -> None:
    """Keep the shared scalp bar contract warm from PG and Schwab history."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from agent.market_hours import get_session
    from agent.scalp.bar_hydration import (
        hydrate_one_minute_history,
        missing_valkey_history,
    )

    def hydrate(targets, **kwargs):
        global _bar_hydration_status
        _bar_hydration_status = {
            "status": "RUNNING",
            "requested": len(targets),
            "updated_at": time.time(),
        }
        try:
            metrics = hydrate_one_minute_history(targets, **kwargs)
        except Exception as exc:
            _bar_hydration_status = {
                "status": "FAILED",
                "requested": len(targets),
                "error": str(exc)[:240],
                "updated_at": time.time(),
            }
            raise
        _bar_hydration_status = {
            "status": "READY" if metrics.get("unresolved", 0) == 0 else "DEGRADED",
            "updated_at": time.time(),
            **metrics,
        }
        try:
            from agent.scalp.bar_hydration import missing_valkey_history
            from agent.universe_registry import record_history_audit

            unresolved = set(missing_valkey_history(targets))
            record_history_audit(targets, set(targets) - unresolved)
        except Exception as exc:
            _log.debug("[ScalpBars] universe history audit failed: %s", exc)
        return metrics

    last_session = ""
    last_nightly_refresh = None
    _runner._stop.wait(5.0)

    while not _runner.stopped:
        try:
            from agent.universe_registry import get_runtime_universe

            tickers = get_runtime_universe()
            session = str(get_session() or "CLOSED").upper()
            active = session != "CLOSED"
            session_opened = active and last_session == "CLOSED"
            now_et = datetime.now(ZoneInfo("America/New_York"))
            nightly_due = (
                not active
                and now_et.hour >= 20
                and last_nightly_refresh != now_et.date()
            )
            missing = missing_valkey_history(tickers)
            if not last_session or session_opened:
                targets = tickers
                hydrate(
                    targets,
                    fetch_missing=True,
                    force_refresh=nightly_due,
                    require_fresh=active,
                )
            elif missing:
                _log.warning("[ScalpBars] Repairing %d missing Valkey histories", len(missing))
                hydrate(missing, fetch_missing=active, require_fresh=active)

            if nightly_due:
                if last_session:
                    hydrate(tickers, fetch_missing=True, force_refresh=True)
                from agent.historical_cache import fill_history_gaps
                for interval, outputsize, minimum in (
                    ("15min", 5000, 200),
                    ("1day", 500, 100),
                ):
                    fill_history_gaps(
                        tickers,
                        interval=interval,
                        min_bars=minimum,
                        outputsize=outputsize,
                    )
                last_nightly_refresh = now_et.date()
            last_session = session
        except Exception as exc:
            _log.warning("[ScalpBars] hydration cycle failed: %s", exc)
        _runner._stop.wait(60.0)


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
    from agent.universe_registry import get_runtime_universe

    _log.info("=== market_data_service starting ===")
    try:
        from agent.broker.schwab_auth import load_stored_md_tokens
        if load_stored_md_tokens(schedule_refresh=False):
            _discover_universe_replacements()
    except Exception as exc:
        _log.warning("Universe replacement bootstrap skipped: %s", exc)
    runtime_tickers = get_runtime_universe()
    _log.info(
        "Universe registry selected %d eligible tickers", len(runtime_tickers)
    )
    _start(runtime_tickers)

    from agent.service_heartbeat import start_service_heartbeat
    start_service_heartbeat(
        "market-data",
        _runner,
        extra=lambda: {"bar_hydration": dict(_bar_hydration_status)},
    )

    threading.Thread(target=_health_loop,                  daemon=True, name="md-health").start()
    threading.Thread(target=_publish_streamer_status_loop, daemon=True, name="md-streamer-status").start()
    threading.Thread(target=_publish_token_status_loop,    daemon=True, name="md-token-status").start()
    threading.Thread(target=_token_reload_loop,            daemon=True, name="md-token-reload").start()
    threading.Thread(target=_universe_recheck_loop,        daemon=True, name="md-universe-recheck").start()
    threading.Thread(target=_bar_hydration_loop,           daemon=True, name="ScalpBarHydrator",
                     args=(runtime_tickers,)).start()

    _log.info("Market data running — waiting for SIGTERM/SIGINT …")
    _runner.register_signals()
    _runner.wait()
    _log.info("=== market_data_service stopped ===")


if __name__ == "__main__":
    main()
