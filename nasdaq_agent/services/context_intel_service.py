#!/usr/bin/env python3
"""
context_intel_service — background context intelligence for the scanner.

Four polling loops
------------------
  earnings_poller   every 6 h  : fetch Finnhub earnings calendar → earnings_calendar PG table
  news_poller       every 3 min: market news (1 req) + rotating Tier-1 company news
  feature_compute   every 30 s : compute rolling features from PG → upsert + publish to Valkey
  iv_poller         every 5 min: IV stub (Phase 2); currently publishes a heartbeat

Design principles
-----------------
  * Provider unavailability (Finnhub down, missing API key) is non-fatal.
    Each loop logs a warning and sleeps its interval; existing DB/Valkey data
    remains valid until its TTL expires.
  * Missing FINNHUB_API_KEY: all Finnhub calls return [] gracefully; only
    feature_compute and IV loops run (DB/Valkey only).
  * Scanner reads from Valkey/PG; this service never touches scanner.py.
  * Docker health check: process alive + DB + Valkey (provider NOT checked).

Environment variables
---------------------
  FINNHUB_API_KEY   required for news/earnings polling; optional overall
  VALKEY_HOST/PORT/SSL, PGHOST/…  — inherited from shared .env
  LOG_LEVEL         DEBUG|INFO|WARNING (default INFO)
  LOG_DIR           log file directory (default /app/logs)
"""
from __future__ import annotations

import os
import sys
import time
import threading
import logging

# Bootstrap path before any project imports
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv
load_dotenv()

from services._base import configure_logging, ServiceRunner

_log    = configure_logging("context-intel")
_runner = ServiceRunner("context-intel")


# ── Heartbeat key (for health check) ─────────────────────────────────────────

_HB_KEY = "ctx:intel:heartbeat"
_HB_TTL = 120   # seconds


def _set_heartbeat() -> None:
    try:
        import json
        from agent.valkey_client import _get_client
        c = _get_client()
        if c:
            c.setex(_HB_KEY, _HB_TTL, json.dumps({"ts": time.time()}))
    except Exception:
        pass


# ── Earnings poller — every 6 hours ──────────────────────────────────────────

def _earnings_poller_loop() -> None:
    """
    Fetch Finnhub earnings calendar for the next 90 days (one API call).
    Parse dates into earnings_calendar PG table.
    Runs once on startup, then every 6 hours.
    """
    interval_s = 6 * 3600

    while not _runner.stopped:
        try:
            _run_earnings_poll()
        except Exception as exc:
            _log.warning("[earnings_poller] Unexpected error: %s", exc)

        # Sleep in short chunks so SIGTERM is handled quickly
        _sleep_interruptible(interval_s)


def _run_earnings_poll() -> None:
    from agent.providers.finnhub_provider import fetch_earnings_calendar, is_available
    from agent.context_store import upsert_earnings
    from datetime import datetime, timezone, timedelta

    if not is_available():
        _log.debug("[earnings_poller] FINNHUB_API_KEY not set — skipping")
        return

    now = datetime.now(timezone.utc)
    # Poll -7 days to +30 days:
    #   -7d catches recent earnings for post-earnings cooldown detection
    #   +30d is the realistic window where companies confirm exact dates
    #   (90d produced ~1500 entries but most had null dates = unconfirmed)
    from_dt = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    to_dt   = (now + timedelta(days=30)).strftime("%Y-%m-%d")

    _log.info("[earnings_poller] Fetching calendar %s → %s", from_dt, to_dt)
    raw = fetch_earnings_calendar(from_dt, to_dt)

    if not raw:
        _log.info("[earnings_poller] Finnhub returned 0 earnings records for %s→%s "
                  "(free-tier limit or no confirmed dates in window)", from_dt, to_dt)
        return

    _log.info("[earnings_poller] Finnhub returned %d raw entries", len(raw))

    entries: list[dict] = []
    null_date_count = 0
    for item in raw:
        symbol = (item.get("symbol") or "").upper().strip()
        date_s = item.get("date") or ""          # explicit None → "" handling
        hour   = item.get("hour", "")
        if not symbol:
            continue
        if not date_s:
            null_date_count += 1
            continue   # unconfirmed earnings date — skip
        try:
            d = datetime.strptime(date_s, "%Y-%m-%d")
            if hour == "amc":
                d = d.replace(hour=22, tzinfo=timezone.utc)
            else:
                # bmo / dmh / unknown → 09:30 ET = 13:30 UTC (approximate)
                d = d.replace(hour=13, minute=30, tzinfo=timezone.utc)
        except Exception:
            continue

        entries.append({
            "ticker":       symbol,
            "report_ts":    d,
            "hour":         hour,
            "eps_estimate": item.get("epsEstimate"),
            "rev_estimate": item.get("revenueEstimate"),
        })

    if null_date_count:
        _log.info("[earnings_poller] Skipped %d entries with unconfirmed dates",
                  null_date_count)

    upserted = upsert_earnings(entries)
    _log.info("[earnings_poller] Upserted %d / %d confirmed earnings entries",
              upserted, len(entries))


# ── News poller — every 3 minutes ────────────────────────────────────────────

# Tier-1 tickers split into 10 rotation groups of 10.
# Each news_poller cycle processes 1 group (1 cycle / group × 1 req/ticker = 10 req/30 min).
# At 3-min cadence → 1 group processed per cycle → all TIER1 covered every 30 min.
_TIER1_FOR_ROTATION: list[str] = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","NFLX","AMD",
    "ADBE","INTU","QCOM","AMAT","MU","PANW","CRWD","MRVL","KLAC","LRCX",
    "ADI","SNPS","CDNS","ISRG","REGN","BKNG","ADP","AMGN","SBUX","COST",
    "CMCSA","CSCO","WDAY","MELI","TTD","SMCI","TEAM","FTNT","PYPL","AXON",
    "CEG","ABNB","DDOG","ZS","NET","ANET","OKTA","SNOW","APP","ORLY",
    "CTAS","ROP","CPRT","ROST","FAST","PCAR","PAYX","ODFL","IDXX","VRSK",
    "EA","DLTR","ALGN","ILMN","TTWO","ON","NXPI","MCHP","SWKS","CTSH",
    "NTAP","VRSN","CHKP","EXPE","DXCM","BIIB","MNST","GEHC","GILD","NDAQ",
    "INTC","PEP","LULU","MPWR","CSGP","KDP","SIRI","HON","COIN","HOOD",
    "PLTR","MSTR","HUBS","DKNG","MDB","SNAP","RBLX","SOFI",
]
_ROTATION_GROUP_SIZE = 10
_rotation_index      = 0
_rotation_lock       = threading.Lock()


def _news_poller_loop() -> None:
    interval_s = 180   # 3 minutes

    while not _runner.stopped:
        try:
            _run_news_poll()
        except Exception as exc:
            _log.warning("[news_poller] Unexpected error: %s", exc)
        _sleep_interruptible(interval_s)


def _run_news_poll() -> None:
    global _rotation_index

    from agent.providers.finnhub_provider import (
        fetch_market_news, fetch_company_news, is_available,
    )
    from agent.context_store import ingest_events, map_market_news_to_tickers
    from config import NASDAQ_TICKERS

    if not is_available():
        _log.debug("[news_poller] FINNHUB_API_KEY not set — skipping")
        return

    total_inserted = 0

    # ── 1. Market-wide general news (1 API call) ──────────────────────────────
    market_articles = fetch_market_news("general")
    if market_articles:
        events = map_market_news_to_tickers(market_articles, list(NASDAQ_TICKERS))
        total_inserted += ingest_events(events)
        _log.debug("[news_poller] Market news: %d articles → %d events",
                   len(market_articles), total_inserted)

    # ── 2. Company news: rotating Tier-1 group (10 API calls per cycle) ──────
    with _rotation_lock:
        import math as _math
        _n_groups     = _math.ceil(len(_TIER1_FOR_ROTATION) / _ROTATION_GROUP_SIZE)
        group_start   = _rotation_index * _ROTATION_GROUP_SIZE
        group_tickers = _TIER1_FOR_ROTATION[group_start: group_start + _ROTATION_GROUP_SIZE]
        _rotation_index = (_rotation_index + 1) % _n_groups

    from datetime import datetime, timezone, timedelta
    now      = datetime.now(timezone.utc)
    from_dt  = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    to_dt    = now.strftime("%Y-%m-%d")

    for ticker in group_tickers:
        if _runner.stopped:
            break
        articles = fetch_company_news(ticker, from_dt, to_dt)
        if articles:
            co_events = [
                {
                    "ticker":       ticker,
                    "headline":     a.get("headline", ""),
                    "source":       a.get("source", ""),
                    "url":          a.get("url", ""),
                    "published_ts": int(a.get("datetime", time.time())),
                    "is_market_wide": False,
                }
                for a in articles
            ]
            n = ingest_events(co_events)
            total_inserted += n

    # ── 3. Top movers from Valkey (add company news for active tickers) ───────
    _add_mover_company_news(from_dt, to_dt)

    _log.info("[news_poller] Cycle complete — inserted/deduped %d events", total_inserted)


def _add_mover_company_news(from_dt: str, to_dt: str) -> None:
    """
    Fetch company news for top 5 active movers (by absolute pct_change).
    Reads current prices from Valkey md:prices hash.  Non-fatal if unavailable.
    """
    try:
        from agent.valkey_client import get_all_prices
        from agent.providers.finnhub_provider import fetch_company_news
        from agent.context_store import ingest_events

        prices = get_all_prices()
        if not prices:
            return

        # Sort by absolute pct_change descending
        movers = sorted(
            [(t, abs(float(q.get("pct_change", 0)))) for t, q in prices.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        for ticker, _ in movers:
            if _runner.stopped:
                break
            articles = fetch_company_news(ticker, from_dt, to_dt)
            if articles:
                events = [
                    {
                        "ticker":       ticker,
                        "headline":     a.get("headline", ""),
                        "source":       a.get("source", ""),
                        "url":          a.get("url", ""),
                        "published_ts": int(a.get("datetime", time.time())),
                        "is_market_wide": False,
                    }
                    for a in articles
                ]
                ingest_events(events)
    except Exception as exc:
        _log.debug("[news_poller] mover company news error: %s", exc)


# ── Feature compute loop — every 30 seconds ──────────────────────────────────

def _feature_compute_loop() -> None:
    interval_s = 30

    while not _runner.stopped:
        try:
            _run_feature_compute()
            _set_heartbeat()
        except Exception as exc:
            _log.warning("[feature_compute] Unexpected error: %s", exc)
        _sleep_interruptible(interval_s)


def _run_feature_compute() -> None:
    from agent.context_store import (
        get_tickers_with_events,
        compute_features_for_ticker,
        upsert_features,
        get_next_earnings_from_db,
    )
    from agent.context_snapshot import publish_context_snapshot, build_payload_from_features
    from agent.ticker_universe import TIER1
    from datetime import datetime, timezone

    # Compute for: tickers with recent events + always TIER1
    event_tickers = get_tickers_with_events(max_age_hours=24)
    tickers = list(dict.fromkeys(TIER1 + event_tickers))   # TIER1 first, then others, deduped

    updated = 0
    for ticker in tickers:
        if _runner.stopped:
            break
        try:
            features = compute_features_for_ticker(ticker)
            upsert_features(features)

            # Build earnings fields for the Valkey payload
            earnings_phase     = ""
            earnings_reason    = ""
            earnings_next_date = ""
            earnings_days_away = 999

            next_dt = get_next_earnings_from_db(ticker)
            if next_dt is not None:
                now  = datetime.now(timezone.utc)
                days = (next_dt - now).days
                earnings_days_away = days
                earnings_next_date = next_dt.strftime("%b %d, %Y")
                if 0 <= days <= 3:
                    earnings_phase  = "blackout"
                    earnings_reason = (
                        f"Earnings in {days}d ({next_dt.strftime('%b %d')}) — blackout"
                    )
                elif -1 <= days < 0:
                    earnings_phase  = "cooldown"
                    earnings_reason = "Post-earnings cooldown"
                elif days <= 7:
                    earnings_phase  = "caution"
                    earnings_reason = f"Earnings in {days}d — reduce size"

            payload = build_payload_from_features(
                ticker,
                features,
                earnings_phase     = earnings_phase,
                earnings_reason    = earnings_reason,
                earnings_next_date = earnings_next_date,
                earnings_days_away = earnings_days_away,
            )
            publish_context_snapshot(ticker, payload)
            updated += 1
        except Exception as exc:
            _log.debug("[feature_compute] %s error: %s", ticker, exc)

    if updated:
        _log.debug("[feature_compute] Updated %d tickers", updated)


# ── IV poller — every 5 minutes (Phase 2 stub) ───────────────────────────────

def _iv_poller_loop() -> None:
    """
    Phase 1: stub that logs intent.
    Phase 2: Schwab /chains for ATM IV on active signals + Tier-1 rotation.
    """
    interval_s = 300   # 5 minutes

    while not _runner.stopped:
        try:
            _log.debug("[iv_poller] Phase 2 IV polling not yet implemented")
        except Exception as exc:
            _log.warning("[iv_poller] Unexpected error: %s", exc)
        _sleep_interruptible(interval_s)


# ── Utility ───────────────────────────────────────────────────────────────────

def _sleep_interruptible(total_s: float, chunk_s: float = 5.0) -> None:
    """Sleep in short chunks so SIGTERM is handled within ~5 s."""
    elapsed = 0.0
    while elapsed < total_s and not _runner.stopped:
        time.sleep(min(chunk_s, total_s - elapsed))
        elapsed += chunk_s


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _log.info("=== context_intel_service starting ===")

    from agent.context_store import init_db

    # Ensure DB tables exist (idempotent)
    if not init_db():
        _log.warning("DB not available at startup — will retry on first write")

    threads = [
        threading.Thread(target=_earnings_poller_loop,  daemon=True, name="ctx-earnings"),
        threading.Thread(target=_news_poller_loop,       daemon=True, name="ctx-news"),
        threading.Thread(target=_feature_compute_loop,  daemon=True, name="ctx-features"),
        threading.Thread(target=_iv_poller_loop,         daemon=True, name="ctx-iv"),
    ]

    for t in threads:
        t.start()
    _log.info("Context intel loops started (%d threads)", len(threads))

    _runner.register_signals()
    _runner.wait()

    _log.info("=== context_intel_service stopped ===")


if __name__ == "__main__":
    main()
