"""
service_state.py — PostgreSQL-backed durable service state.

Replaces last-known-state Valkey keys with PostgreSQL UPSERT so that
service snapshots survive Valkey restarts and are visible to any DB client.

Keys migrated (Valkey is still the live-bus; PostgreSQL is the source of truth):
  scan:latest          → no expiry   (scanner writes after every cycle)
  learner:status       → expires 300s
  scheduler:heartbeat  → expires 90s
  scanner:streamer     → expires 60s

Keys that stay Valkey-only (high-frequency streaming / pub-sub only):
  md:prices            — 3.3 Hz HASH + pub/sub fan-out
  md:1m:{ticker}       — append-only 1-min candle LISTs, 2 h TTL
  scan:notify          — lightweight pub/sub wake-up trigger
  schwab:tokens_refreshed — pub/sub OAuth trigger
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Lazy auto-init ────────────────────────────────────────────────────────────
# set_state() is called from market-data, learner, and scheduler containers
# that may start before web-api runs init_db() explicitly.  We self-initialize
# once (per process) so callers never need to call init_db() themselves.
_init_lock  = threading.Lock()
_db_ready   = False   # becomes True after the first successful init_db()


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
    """
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_DDL)
            conn.commit()
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
        sql = f"""
            INSERT INTO service_state (key, value, updated_at, expires_at)
            VALUES (%s, %s::jsonb, NOW(), {expires_fragment})
            ON CONFLICT (key) DO UPDATE
                SET value      = EXCLUDED.value,
                    updated_at = NOW(),
                    expires_at = EXCLUDED.expires_at
        """
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (key, json.dumps(value, default=str)))
            conn.commit()
        return True
    except Exception as exc:
        logger.debug("[service_state] set_state(%s) error: %s", key, exc)
        return False


def get_state(key: str, ignore_expiry: bool = False) -> Optional[dict[str, Any]]:
    """
    Return the stored value dict, or None if the row is missing or expired.

    ignore_expiry=True: return even if past expires_at.  Useful for cold-start
    fallback — stale heartbeat data is better than nothing on dashboard load.
    """
    try:
        from agent.db import get_conn
        expiry_clause = (
            "" if ignore_expiry
            else "AND (expires_at IS NULL OR expires_at > NOW())"
        )
        sql = f"SELECT value FROM service_state WHERE key = %s {expiry_clause}"
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (key,))
                row = cur.fetchone()
        if row is None:
            return None
        val = row[0]
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
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT EXTRACT(EPOCH FROM (NOW() - updated_at)) "
                    "FROM service_state WHERE key = %s",
                    (key,),
                )
                row = cur.fetchone()
        return float(row[0]) if row else None
    except Exception as exc:
        logger.debug("[service_state] get_age_s(%s) error: %s", key, exc)
        return None
