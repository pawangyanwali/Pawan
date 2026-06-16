"""
Durable scan-result state.

Write path  (scanner → PostgreSQL + Valkey):
  1. UPSERT into service_state table (key='scan:latest') — durable, survives
     Valkey restarts, readable by any DB client.
  2. PUBLISH scan:notify to Valkey — lightweight wake-up for web-api subscribers.
  3. SET scan:latest in Valkey — kept as a fast-path cache for containers that
     read it before PostgreSQL is warmed (backward compat, no TTL).

Read path  (web-api → PostgreSQL first, Valkey fallback):
  read_latest() tries PostgreSQL first.  If the DB is unavailable it falls back
  to the Valkey key so the dashboard is never blank due to a transient DB hiccup.

Subscribe path  (web-api → Valkey pub/sub, unchanged):
  subscribe_scan_results() listens on scan:notify; on each notification it calls
  read_latest() which now prefers PostgreSQL.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_PG_KEY  = "scan:latest"
_VK_KEY  = "scan:latest"
_CHANNEL = "scan:notify"


# ── Write (scanner → PostgreSQL + Valkey) ────────────────────────────────────

def write_latest(
    signals:       list[dict[str, Any]],
    regime:        dict[str, Any],
    session:       dict[str, Any],
    scanned_count: int,
    scan_meta:     Optional[dict[str, Any]] = None,
) -> bool:
    """
    Persist the latest scan result to PostgreSQL (durable) and publish a
    wake-up notification via Valkey pub/sub (real-time).

    Returns True when at least one write path succeeded.
    """
    snapshot = {
        "ts":            time.time(),
        "signals":       signals,
        "regime":        regime,
        "session":       session,
        "scanned_count": scanned_count,
    }
    if scan_meta:
        snapshot.update(scan_meta)

    pg_ok = False
    vk_ok = False

    # ── 1. PostgreSQL UPSERT (source of truth) ────────────────────────────────
    try:
        from agent.service_state import set_state
        pg_ok = set_state(_PG_KEY, snapshot, ttl_s=None)
        if not pg_ok:
            logger.warning("[signal_snapshot] PostgreSQL write returned False for scan:latest")
    except Exception as exc:
        logger.warning("[signal_snapshot] PostgreSQL write error for scan:latest: %s", exc)

    # ── 2. Valkey — fast cache + pub/sub trigger ──────────────────────────────
    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        if client:
            raw = json.dumps(snapshot, separators=(",", ":"), default=str)
            pipe = client.pipeline(transaction=False)
            pipe.set(_VK_KEY, raw)           # fast-path cache (no TTL)
            pipe.publish(_CHANNEL, "1")      # wake-up trigger for subscribers
            pipe.execute()
            vk_ok = True
    except Exception as exc:
        logger.debug("[signal_snapshot] Valkey write error: %s", exc)

    if not pg_ok and not vk_ok:
        logger.warning("[signal_snapshot] write_latest: both PG and Valkey failed")
        return False
    return True


# ── Read (web-api → PostgreSQL first, Valkey fallback) ───────────────────────

def read_latest() -> Optional[dict[str, Any]]:
    """
    Return the latest scan snapshot dict, or None if not yet available.

    Preference order:
      1. PostgreSQL service_state (durable, survives Valkey restart)
      2. Valkey scan:latest (fast-path if DB is temporarily unavailable)
    """
    # ── Priority 1: PostgreSQL ────────────────────────────────────────────────
    try:
        from agent.service_state import get_state
        snap = get_state(_PG_KEY, ignore_expiry=True)   # scan:latest has no TTL
        if snap and snap.get("signals") is not None:
            return snap
    except Exception as exc:
        logger.debug("[signal_snapshot] PostgreSQL read error: %s", exc)

    # ── Priority 2: Valkey (fallback) ─────────────────────────────────────────
    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        if client:
            raw = client.get(_VK_KEY)
            if raw:
                return json.loads(raw)
    except Exception as exc:
        logger.debug("[signal_snapshot] Valkey read error: %s", exc)

    return None


# ── Subscribe (web-api background task — Valkey pub/sub, unchanged) ──────────

def subscribe_scan_results(callback: Callable[[dict[str, Any]], None]) -> None:
    """
    Block-subscribe to scan notifications via Valkey pub/sub.  On each
    notification, reads the latest snapshot (preferring PostgreSQL) and calls
    callback(snapshot_dict).

    Runs in a daemon thread — call via threading.Thread(target=..., daemon=True).
    Falls back silently if Valkey is unavailable.
    """
    import threading
    import time as _time

    def _run() -> None:
        while True:
            try:
                from agent.valkey_client import _get_client, _cfg
                import redis as _redis_lib

                if _get_client() is None:
                    _time.sleep(5)
                    continue

                host, port, ssl = _cfg()
                sub = _redis_lib.Redis(
                    host=host, port=port, ssl=ssl,
                    ssl_cert_reqs=None,
                    socket_connect_timeout=5,
                    decode_responses=False,
                )
                ps = sub.pubsub()
                ps.subscribe(_CHANNEL)
                for msg in ps.listen():
                    if msg.get("type") != "message":
                        continue
                    snap = read_latest()
                    if snap:
                        try:
                            callback(snap)
                        except Exception as _ce:
                            logger.debug("[signal_snapshot] callback error: %s", _ce)
            except Exception as exc:
                logger.warning("[signal_snapshot] subscribe error: %s — retrying in 5s", exc)
                _time.sleep(5)

    t = threading.Thread(target=_run, daemon=True, name="scan-snapshot-sub")
    t.start()
