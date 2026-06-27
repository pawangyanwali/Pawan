"""
service_state.py — PostgreSQL-backed durable service state.

Replaces last-known-state Valkey keys with PostgreSQL UPSERT so that
service snapshots survive Valkey restarts and are visible to any DB client.

Keys migrated (Valkey is still the live-bus; PostgreSQL is the source of truth):
  scan:latest          → no expiry   (scalp-engine writes every cycle)
  scalp-learner:status → expires 180s
  scheduler:heartbeat  → expires 90s
  market-data:status   → expires 60s

Keys that stay Valkey-only (high-frequency streaming / pub-sub only):
  md:prices            — 3.3 Hz HASH + pub/sub fan-out
  md:1m:{ticker}       — append-only 1-min candle LISTs, 2 h TTL
  scan:notify          — lightweight pub/sub wake-up trigger
  schwab:tokens_refreshed — pub/sub OAuth trigger

DB interface note:
  agent.db.get_conn() returns _PgConnection — a custom wrapper that exposes
  the sqlite3.Connection API: conn.execute(sql, params) → cursor-like object
  with .fetchone()/.fetchall() returning RealDictRow dicts.  Do NOT call
  conn.cursor() — it does not exist on _PgConnection.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from typing import Any, Optional


def _sanitize_for_jsonb(obj: Any) -> Any:
    """Recursively replace NaN/Infinity with None so PostgreSQL JSONB accepts the payload.

    Python's json.dumps serialises float('nan') as NaN and float('inf') as Infinity
    (JS syntax), which PostgreSQL JSONB rejects as invalid JSON.
    """
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_jsonb(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_jsonb(v) for v in obj]
    return obj

logger = logging.getLogger(__name__)

# ── Lazy auto-init ────────────────────────────────────────────────────────────
# set_state() is called from market-data, learner, and scheduler containers
# that may start before web-api runs init_db() explicitly.  We self-initialize
# once (per process) so callers never need to call init_db() themselves.
_init_lock  = threading.Lock()
_db_ready   = False   # becomes True only after a confirmed successful init_db()


def _ensure_init() -> None:
    """Lazy-initialize once per process. Only marks ready after confirmed success.
    Retries on every set_state() call until init_db() succeeds so a transient
    DB unavailability on cold start does not permanently block table creation.
    """
    global _db_ready
    if _db_ready:
        return
    with _init_lock:
        if not _db_ready:
            _db_ready = init_db()   # True only on real success; False = retry next call


_DDL = """
CREATE TABLE IF NOT EXISTS service_state (
    key        TEXT        PRIMARY KEY,
    value      JSONB       NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NULL
);
CREATE INDEX IF NOT EXISTS idx_service_state_expires
    ON service_state (expires_at)
    WHERE expires_at IS NOT NULL;
"""


def init_db() -> bool:
    """Create the service_state table (idempotent — safe to call on every startup).

    Returns True when the table is confirmed to exist, False on any DB error.
    The return value is used by _ensure_init() to decide whether to retry DDL
    creation on the next set_state() call — a False here means the next write
    will attempt init again rather than silently skipping table setup.

    Uses _PgConnection.execute() — do NOT use conn.cursor() which does not
    exist on the _PgConnection wrapper (agent/db.py).
    """
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            # _PgConnection.execute() translates ? → %s and runs on psycopg2.
            # The DDL contains no parameters so params list is empty.
            conn.execute(_DDL)
        logger.info("[service_state] table ready")
        return True
    except Exception as exc:
        logger.warning("[service_state] init_db error: %s", exc)
        return False


def set_state(key: str, value: dict[str, Any], ttl_s: Optional[int] = None) -> bool:
    """
    Upsert a JSON value for *key*.  Best-effort — returns False on DB error
    so callers (heartbeat loops, status publishers) can log and continue
    without crashing the service.

    ttl_s: if provided, sets expires_at = NOW() + ttl_s seconds.
           get_state() treats rows with expires_at in the past as missing
           (mimics Valkey SETEX semantics).  Pass None for no expiry.

    Self-initializes the service_state table on the first call so that
    market-data, learner, and scheduler containers do not need to call
    init_db() explicitly.
    """
    _ensure_init()
    try:
        from agent.db import get_conn
        expires_fragment = (
            f"NOW() + INTERVAL '{int(ttl_s)} seconds'" if ttl_s else "NULL"
        )
        # %s placeholders — _PgConnection passes them through to psycopg2 as-is
        # (no ? → %s conversion needed since we write %s directly).
        sql = f"""
            INSERT INTO service_state (key, value, updated_at, expires_at)
            VALUES (%s, %s::jsonb, NOW(), {expires_fragment})
            ON CONFLICT (key) DO UPDATE
                SET value      = EXCLUDED.value,
                    updated_at = NOW(),
                    expires_at = EXCLUDED.expires_at
        """
        with get_conn() as conn:
            conn.execute(sql, (key, json.dumps(_sanitize_for_jsonb(value), default=str)))
        return True
    except Exception as exc:
        logger.warning("[service_state] set_state(%s) error: %s", key, exc)
        return False


def get_state(key: str, ignore_expiry: bool = False) -> Optional[dict[str, Any]]:
    """
    Return the stored value dict, or None if the row is missing or expired.

    ignore_expiry=True: return even if past expires_at.  Useful for cold-start
    fallback — stale heartbeat data is better than nothing on dashboard load.

    RealDictCursor returns rows as dict-like objects; access by column name,
    not by index.  psycopg2 automatically deserialises JSONB into Python dicts.
    """
    try:
        from agent.db import get_conn
        expiry_clause = (
            "" if ignore_expiry
            else "AND (expires_at IS NULL OR expires_at > NOW())"
        )
        sql = f"SELECT value FROM service_state WHERE key = %s {expiry_clause}"
        with get_conn() as conn:
            cur = conn.execute(sql, (key,))
            row = cur.fetchone()
        if row is None:
            return None
        # RealDictRow: access by name.  JSONB is auto-parsed to dict by psycopg2;
        # fall back to json.loads() for string payloads (e.g. legacy rows).
        val = row["value"]
        return val if isinstance(val, dict) else json.loads(val)
    except Exception as exc:
        logger.debug("[service_state] get_state(%s) error: %s", key, exc)
        return None


def get_age_s(key: str) -> Optional[float]:
    """
    Return seconds elapsed since the row was last written, or None if not found.
    Does not check expires_at — useful for computing staleness badges regardless
    of whether the row has formally expired.
    """
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            cur = conn.execute(
                "SELECT EXTRACT(EPOCH FROM (NOW() - updated_at)) AS age_s "
                "FROM service_state WHERE key = %s",
                (key,),
            )
            row = cur.fetchone()
        return float(row["age_s"]) if row else None
    except Exception as exc:
        logger.debug("[service_state] get_age_s(%s) error: %s", key, exc)
        return None
