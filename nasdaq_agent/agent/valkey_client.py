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
  VALKEY_HOST   (default: master.nasdaq-cache.zn4aar.use1.cache.amazonaws.com)
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

# ── Config ─────────────────────────────────────────────────────────────────────

def _cfg() -> tuple[str, int, bool]:
    host = os.getenv(
        "VALKEY_HOST",
        "master.nasdaq-cache.zn4aar.use1.cache.amazonaws.com",
    )
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
        pipe = client.pipeline(transaction=False)

        # HSET: flatten into field/value pairs
        hset_args: list = []
        for ticker, quote in bulk.items():
            hset_args.append(ticker)
            hset_args.append(json.dumps(quote, separators=(",", ":")))
        pipe.hset(_HASH, mapping=dict(zip(hset_args[::2], hset_args[1::2])))

        # PUBLISH the full batch dict so subscribers get everything at once
        pipe.publish(_CHANNEL, json.dumps(bulk, separators=(",", ":")))

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
        try:
            import redis as _redis_lib
            host, port, ssl = _cfg()
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
                f"[Valkey] Subscriber error: {exc} — retrying in {retry_delay:.0f}s"
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
