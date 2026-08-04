"""
Valkey (ElastiCache Redis-compatible) connection manager.

Roles in the zero-lag architecture:
  MD Poller  → publish_prices(bulk)          write prices every ~300 ms
  Scanner    → get_price(ticker)             read O(1) from hash
  WS handler → subscribe_prices(callback)   receive every batch instantly
  Dashboard  → health_status()              connectivity + queue depth

Key layout:
  HASH  md:prices        field=TICKER  value=JSON(quote)
  PUBSUB md:prices       payload=JSON({ticker: quote, ...})

TLS is required for ElastiCache Valkey.  All config via env vars:
  VALKEY_HOST   (required)
  VALKEY_PORT   (default: 6379)
  VALKEY_SSL    (default: true)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_CHANNEL = "md:prices"
_HASH    = "md:prices"
_STICKY_PRICE_FIELDS = ("open",)
_WS_PRIORITY_TTL_S = max(
    1.0, float(os.getenv("NASDAQ_WS_QUOTE_PRIORITY_TTL_S", "5.0"))
)
_PRICE_FIELDS = (
    "last", "mark", "bid", "ask", "volume", "high", "low", "pct_change",
)

# ── Config ─────────────────────────────────────────────────────────────────────

def _cfg() -> tuple[str, int, bool]:
    host = os.getenv("VALKEY_HOST", "").strip()
    if not host:
        raise RuntimeError("VALKEY_HOST is required")
    port = int(os.getenv("VALKEY_PORT", "6379"))
    ssl  = os.getenv("VALKEY_SSL", "true").lower() not in ("false", "0", "no")
    return host, port, ssl


# ── Singleton client ───────────────────────────────────────────────────────────

_redis = None           # shared redis.Redis connection for commands
_lock  = threading.Lock()
_connected = False
_last_error: Optional[str] = None
_publish_count  = 0
_last_publish_ts: float = 0.0


def _get_client():
    """Return (or lazily create) the shared Redis/Valkey client."""
    global _redis, _connected, _last_error
    if _redis is not None:
        return _redis
    with _lock:
        if _redis is not None:   # double-checked
            return _redis
        try:
            import redis as _redis_lib
            host, port, ssl = _cfg()
            client = _redis_lib.Redis(
                host=host,
                port=port,
                ssl=ssl,
                ssl_cert_reqs=None,      # ElastiCache uses self-signed cert
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
                decode_responses=False,
            )
            client.ping()
            _redis = client
            _connected = True
            _last_error = None
            logger.info(f"[Valkey] Connected → {host}:{port} (ssl={ssl})")
        except Exception as exc:
            _last_error = str(exc)
            _connected = False
            logger.warning(f"[Valkey] Connection failed: {exc}")
            _redis = None
    return _redis


def _reset_client() -> None:
    """Force reconnect on next call (used after connection errors)."""
    global _redis, _connected
    with _lock:
        _redis = None
        _connected = False


# ── Price write (MD Poller → Valkey) ──────────────────────────────────────────

def _positive_float(value) -> float:
    try:
        v = float(value or 0.0)
        return v if v > 0 else 0.0
    except Exception:
        return 0.0


def _merge_sticky_price_fields(incoming: dict, existing: Optional[dict]) -> dict:
    """
    Preserve session fields that Schwab WebSocket ticks can omit.

    LEVELONE_EQUITIES updates are compact and may not include regular-session
    open. Without this merge a fresh WS tick can replace the Valkey ticker JSON
    and erase an open price that REST/bootstrap already supplied.
    """
    if not existing:
        return incoming
    merged = dict(incoming)
    for field in _STICKY_PRICE_FIELDS:
        if _positive_float(merged.get(field)) <= 0:
            prev = _positive_float(existing.get(field))
            if prev > 0:
                merged[field] = prev
    return merged


def _merge_price_sources(
    incoming: dict,
    existing: Optional[dict],
    *,
    now: float | None = None,
) -> dict:
    """Merge quote publishers without letting REST relabel a fresh WS quote.

    WS and REST are complementary: WS owns executable price fields while it is
    fresh; REST may still backfill session metadata such as the opening price.
    Per-source timestamps make the effective source deterministic even when the
    two publisher threads interleave.
    """
    current_time = time.time() if now is None else float(now)
    previous = dict(existing or {})
    merged = _merge_sticky_price_fields(dict(incoming), previous)
    incoming_status = str(merged.get("source_status") or "").upper()
    incoming_ts = float(merged.get("updated_at") or current_time)

    if incoming_status == "LIVE":
        merged["ws_updated_at"] = float(
            merged.get("ws_updated_at") or incoming_ts
        )
        if previous.get("rest_updated_at") is not None:
            merged["rest_updated_at"] = previous["rest_updated_at"]
        return merged

    if incoming_status in {"REST_FALLBACK", "FALLBACK"}:
        merged["rest_updated_at"] = float(
            merged.get("rest_updated_at") or incoming_ts
        )
        ws_updated_at = float(
            previous.get("ws_updated_at")
            or (
                previous.get("updated_at")
                if str(previous.get("source_status") or "").upper() == "LIVE"
                else 0.0
            )
            or 0.0
        )
        if ws_updated_at > 0 and current_time - ws_updated_at <= _WS_PRIORITY_TTL_S:
            # Preserve every executable field from WS. REST remains useful for
            # sticky session metadata and records its own freshness timestamp.
            for field in _PRICE_FIELDS:
                if field in previous:
                    merged[field] = previous[field]
            merged.update(
                updated_at=ws_updated_at,
                ws_updated_at=ws_updated_at,
                source=previous.get("source") or "SCHWAB_WS",
                source_status="LIVE",
                is_live=True,
            )
    return merged


def publish_prices(bulk: dict[str, dict]) -> bool:
    """
    Write a batch of quotes from the MD Poller into Valkey.

    Two operations per batch (pipelined):
      1. HSET md:prices ticker1 <json> ticker2 <json> …  — durable state
      2. PUBLISH md:prices <json_of_bulk>                — real-time fan-out

    Returns True on success, False if Valkey is unavailable.
    bulk format: {ticker: {last, bid, ask, open, high, low, pct_change, volume}}
    """
    global _publish_count, _last_publish_ts, _connected, _last_error
    if not bulk:
        return False

    client = _get_client()
    if client is None:
        return False

    try:
        existing_by_ticker: dict[str, Optional[dict]] = {}
        try:
            keys = list(bulk.keys())
            raw_existing = client.hmget(_HASH, keys) if keys else []
            for ticker, raw in zip(keys, raw_existing):
                if raw is None:
                    existing_by_ticker[ticker] = None
                    continue
                existing_by_ticker[ticker] = json.loads(raw)
        except Exception as exc:
            logger.debug("[Valkey] sticky price merge skipped: %s", exc)
            existing_by_ticker = {}

        now = time.time()
        merged_bulk = {
            ticker: _merge_price_sources(
                quote,
                existing_by_ticker.get(ticker),
                now=now,
            )
            for ticker, quote in bulk.items()
        }

        pipe = client.pipeline(transaction=False)

        # HSET: flatten into field/value pairs
        hset_args: list = []
        for ticker, quote in merged_bulk.items():
            hset_args.append(ticker)
            hset_args.append(json.dumps(quote, separators=(",", ":")))
        pipe.hset(_HASH, mapping=dict(zip(hset_args[::2], hset_args[1::2])))

        # PUBLISH the full batch dict so subscribers get everything at once
        pipe.publish(_CHANNEL, json.dumps(merged_bulk, separators=(",", ":")))

        pipe.execute()
        _publish_count += len(bulk)
        _last_publish_ts = time.time()
        _connected = True
        _last_error = None
        return True

    except Exception as exc:
        _last_error = str(exc)
        _connected = False
        logger.warning(f"[Valkey] publish_prices error: {exc}")
        _reset_client()
        return False


# ── Price read (Scanner / Trade Management → Valkey) ──────────────────────────

def get_price(ticker: str) -> Optional[dict]:
    """
    Return the latest quote dict for a ticker, or None if unavailable.
    O(1) HGET — reads from the durable hash, not from pub/sub.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        raw = client.hget(_HASH, ticker)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.debug(f"[Valkey] get_price({ticker}) error: {exc}")
        _reset_client()
        return None


def get_all_prices() -> dict[str, dict]:
    """
    Return the full price hash as {ticker: quote_dict}.
    Used for bulk reads (e.g. trade management P&L sweep).
    HGETALL — O(N tickers); ~500 tickers ≈ sub-millisecond.
    """
    client = _get_client()
    if client is None:
        return {}
    try:
        raw = client.hgetall(_HASH)
        return {
            k.decode() if isinstance(k, bytes) else k: json.loads(v)
            for k, v in raw.items()
        }
    except Exception as exc:
        logger.debug(f"[Valkey] get_all_prices error: {exc}")
        _reset_client()
        return {}


def price_bus_health(max_age_s: float = 2.0) -> dict:
    """
    Summarize quote freshness across the whole md:prices hash.

    This is intentionally stricter than a process liveness check: Valkey can be
    reachable while prices are stale.  The dashboard and health checks use this
    to distinguish real live Schwab WS quotes from REST fallback or cached scan
    prices.
    """
    prices = get_all_prices()
    # The hash intentionally retains the last quote for quarantined symbols so
    # existing positions can still be marked. Health and UI coverage, however,
    # must be measured against the eligible runtime universe only.
    client = _get_client()
    if client is not None:
        try:
            raw_eligible = client.get("universe:eligible")
            if raw_eligible:
                eligible = {
                    str(ticker).upper()
                    for ticker in json.loads(raw_eligible)
                    if ticker
                }
                prices = {
                    ticker: quote for ticker, quote in prices.items()
                    if ticker.upper() in eligible
                }
        except Exception as exc:
            logger.debug("[Valkey] eligible-universe health filter skipped: %s", exc)
    now = time.time()
    total = len(prices)
    if total == 0:
        return {
            "total": 0,
            "fresh": 0,
            "trusted_fresh": 0,
            "live": 0,
            "fallback": 0,
            "snapshot": 0,
            "stale": 0,
            "fresh_pct": 0.0,
            "trusted_fresh_pct": 0.0,
            "live_pct": 0.0,
            "fallback_pct": 0.0,
            "snapshot_pct": 0.0,
            "oldest_age_s": None,
            "newest_age_s": None,
            "status": "NO_DATA",
        }

    fresh = trusted_fresh = live = fallback = snapshot = stale = 0
    ages: list[float] = []
    for quote in prices.values():
        try:
            ts = float(quote.get("updated_at") or 0.0)
        except Exception:
            ts = 0.0
        age = max(0.0, now - ts) if ts > 0 else float("inf")
        if age != float("inf"):
            ages.append(age)
        source_status = str(quote.get("source_status") or "").upper()
        is_fresh = age <= max_age_s
        if is_fresh:
            fresh += 1
            if source_status == "LIVE":
                live += 1
                trusted_fresh += 1
            elif source_status in ("REST_FALLBACK", "FALLBACK"):
                fallback += 1
                trusted_fresh += 1
            elif source_status in ("SCAN_SNAPSHOT", "STALE_CACHE"):
                snapshot += 1
        else:
            stale += 1

    fresh_pct = fresh / total if total else 0.0
    trusted_fresh_pct = trusted_fresh / total if total else 0.0
    live_pct = live / total if total else 0.0
    fallback_pct = fallback / total if total else 0.0
    snapshot_pct = snapshot / total if total else 0.0
    if live_pct >= 0.95:
        status = "LIVE"
    elif live > 0:
        status = "PARTIAL_LIVE"
    elif fallback_pct >= 0.95:
        status = "REST_FALLBACK"
    elif fallback > 0:
        status = "PARTIAL_FALLBACK"
    elif snapshot > 0:
        status = "SCAN_SNAPSHOT"
    else:
        status = "STALE"

    return {
        "total": total,
        "fresh": fresh,
        "trusted_fresh": trusted_fresh,
        "live": live,
        "fallback": fallback,
        "snapshot": snapshot,
        "stale": stale,
        "fresh_pct": round(fresh_pct * 100, 1),
        "trusted_fresh_pct": round(trusted_fresh_pct * 100, 1),
        "live_pct": round(live_pct * 100, 1),
        "fallback_pct": round(fallback_pct * 100, 1),
        "snapshot_pct": round(snapshot_pct * 100, 1),
        "oldest_age_s": round(max(ages), 1) if ages else None,
        "newest_age_s": round(min(ages), 1) if ages else None,
        "status": status,
        "max_age_s": max_age_s,
    }


# ── Pub/Sub subscriber (WebSocket bridge) ─────────────────────────────────────

_sub_thread: Optional[threading.Thread] = None
_sub_callbacks: list[Callable[[dict], None]] = []
_sub_running = False


def register_price_subscriber(callback: Callable[[dict], None]) -> None:
    """
    Register a callback that is called on every price batch published to Valkey.
    callback(bulk: dict[str, dict]) is called from a background thread.
    Multiple callbacks are supported.
    """
    _sub_callbacks.append(callback)
    _ensure_subscriber_running()


def _ensure_subscriber_running() -> None:
    global _sub_thread, _sub_running
    if _sub_thread and _sub_thread.is_alive():
        return
    with _lock:
        if _sub_thread and _sub_thread.is_alive():
            return
        _sub_running = True
        _sub_thread = threading.Thread(
            target=_subscriber_loop, daemon=True, name="ValkeySubscriber"
        )
        _sub_thread.start()
        logger.info("[Valkey] Subscriber thread started.")


def _subscriber_loop() -> None:
    """Background thread: subscribe to md:prices and fan-out to all callbacks."""
    global _sub_running
    retry_delay = 2.0
    while _sub_running:
        host, port, ssl = _cfg()
        try:
            import redis as _redis_lib
            sub_client = _redis_lib.Redis(
                host=host,
                port=port,
                ssl=ssl,
                ssl_cert_reqs=None,
                socket_connect_timeout=5,
                socket_timeout=60,   # long timeout: blocking listen loop
                decode_responses=False,
            )
            pubsub = sub_client.pubsub()
            pubsub.subscribe(_CHANNEL)
            logger.info(f"[Valkey] Subscribed to channel '{_CHANNEL}'")
            retry_delay = 2.0   # reset on successful connect

            for message in pubsub.listen():
                if not _sub_running:
                    break
                if message.get("type") != "message":
                    continue
                data = message.get("data", b"")
                try:
                    bulk = json.loads(data)
                    for fn in _sub_callbacks:
                        try:
                            fn(bulk)
                        except Exception:
                            pass
                except Exception:
                    pass

        except Exception as exc:
            logger.warning(
                f"[Valkey] Subscriber error ({host}:{port}): {exc} — "
                f"retrying in {retry_delay:.0f}s. "
                f"If this repeats, check VALKEY_HOST env var and VPC peering."
            )
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30)


def stop_subscriber() -> None:
    global _sub_running
    _sub_running = False


# ── Health status (Dashboard) ─────────────────────────────────────────────────

def health_status() -> dict:
    """
    Return Valkey connectivity status for the service dashboard.
    Attempts a lightweight PING — no data read.
    """
    client = _get_client()
    if client is None:
        return {
            "connected": False,
            "error":     _last_error,
            "publish_count": _publish_count,
            "last_publish_ago_s": None,
        }
    try:
        client.ping()
        connected = True
        err = None
    except Exception as exc:
        connected = False
        err = str(exc)
        _reset_client()

    ago = round(time.time() - _last_publish_ts, 1) if _last_publish_ts else None
    return {
        "connected":         connected,
        "error":             err,
        "publish_count":     _publish_count,
        "last_publish_ago_s": ago,
        "subscriber_alive":  bool(_sub_thread and _sub_thread.is_alive()),
    }
