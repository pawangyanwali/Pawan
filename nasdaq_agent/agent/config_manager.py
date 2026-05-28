"""
Runtime configuration store — Phase 1.

Stores runtime-tunable settings in a PostgreSQL `config_store` table.
Values are JSON-encoded TEXT (supports float, int, bool, str, list).
Changes propagate to all processes via Valkey pub/sub hot-reload without
requiring a container restart.

Usage:
    from agent.config_manager import config

    value = config.get("paper.budget", 50000.0)
    config.set("paper.budget", 75000.0, updated_by="dashboard")
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "paper.budget":                        lambda: float(os.getenv("PAPER_BUDGET", "50000")),
    "paper.max_trade_pct":                 lambda: float(os.getenv("PAPER_MAX_TRADE_PCT", "5.0")),
    "paper.max_allocated_pct":             lambda: float(os.getenv("PAPER_MAX_ALLOCATED_PCT", "40.0")),
    "paper.max_open_trades":               lambda: int(os.getenv("PAPER_MAX_OPEN_TRADES", "10")),
    "paper.min_confidence":                lambda: float(os.getenv("PAPER_TRADE_MIN_CONFIDENCE", "25.0")),
    "scanner.pre_earnings_blackout_days":  lambda: 3,
    "scanner.post_earnings_cooldown_days": lambda: 1,
}

# Legacy column map: config_store key → account_config column name
_LEGACY_COLUMN_MAP: dict[str, str] = {
    "paper.budget":           "total_budget",
    "paper.max_trade_pct":    "max_trade_pct",
    "paper.max_allocated_pct": "max_allocated_pct",
    "paper.max_open_trades":  "max_open_trades",
}

_VALKEY_CHANNEL = "config:changed"

_DDL = """
CREATE TABLE IF NOT EXISTS config_store (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    updated_by TEXT NOT NULL DEFAULT 'system'
)
"""

_UPSERT = """
INSERT INTO config_store (key, value, updated_at, updated_by)
VALUES (?, ?, ?, ?)
ON CONFLICT (key) DO UPDATE SET
    value      = EXCLUDED.value,
    updated_at = EXCLUDED.updated_at,
    updated_by = EXCLUDED.updated_by
"""


# ── ConfigManager ──────────────────────────────────────────────────────────────

class ConfigManager:
    """
    Thread-safe singleton config store backed by PostgreSQL.

    All reads come from an in-memory cache; writes go to DB then cache.
    Valkey pub/sub propagates changes across all processes in real time.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._lock  = threading.RLock()
        self._listener_started = False

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _ensure_table(self) -> None:
        """Create config_store table if it does not exist."""
        try:
            from agent.db import get_conn
            with get_conn() as c:
                c.execute(_DDL)
        except Exception as exc:
            logger.warning("[ConfigManager] _ensure_table failed: %s", exc)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Read all rows from config_store into the in-memory cache.
        Also ensures the table exists.
        """
        self._ensure_table()
        try:
            from agent.db import get_conn
            with get_conn() as c:
                rows = c.execute(
                    "SELECT key, value FROM config_store"
                ).fetchall()
            with self._lock:
                for row in rows:
                    try:
                        self._cache[row["key"]] = json.loads(row["value"])
                    except Exception:
                        self._cache[row["key"]] = row["value"]
            logger.info("[ConfigManager] Loaded %d keys from config_store", len(rows))
        except Exception as exc:
            logger.warning("[ConfigManager] load() failed: %s", exc)

    def seed_defaults(self) -> None:
        """
        For each key in _DEFAULTS not yet in cache:
          1. Try to read from legacy account_config table (paper.* keys).
          2. Fall back to the default factory lambda.

        This preserves user-tuned settings across the migration.
        """
        for key, factory in _DEFAULTS.items():
            with self._lock:
                if key in self._cache:
                    continue

            # Try legacy account_config migration for paper.* keys
            value = None
            legacy_col = _LEGACY_COLUMN_MAP.get(key)
            if legacy_col:
                try:
                    from agent.db import get_conn
                    with get_conn() as c:
                        row = c.execute(
                            f"SELECT {legacy_col} FROM account_config WHERE id=1"
                        ).fetchone()
                    if row and row[legacy_col] is not None:
                        value = row[legacy_col]
                        logger.info(
                            "[ConfigManager] Migrated %s from account_config.%s = %r",
                            key, legacy_col, value,
                        )
                except Exception as exc:
                    logger.debug("[ConfigManager] Legacy read for %s failed: %s", key, exc)

            # Fall back to default factory
            if value is None:
                try:
                    value = factory()
                except Exception as exc:
                    logger.warning("[ConfigManager] Default factory for %s failed: %s", key, exc)
                    continue

            # Persist to DB (only if still absent — race guard)
            with self._lock:
                if key in self._cache:
                    continue

            try:
                self.set(key, value, updated_by="system")
            except Exception as exc:
                logger.warning("[ConfigManager] seed_defaults set(%s) failed: %s", key, exc)

    def get(self, key: str, default: Any = None) -> Any:
        """Thread-safe read from in-memory cache."""
        with self._lock:
            return self._cache.get(key, default)

    def set(self, key: str, value: Any, updated_by: str = "system") -> None:
        """Write value to DB, update cache, publish to Valkey."""
        encoded = json.dumps(value)
        ts      = self._now_iso()
        try:
            from agent.db import get_conn
            with get_conn() as c:
                c.execute(_UPSERT, (key, encoded, ts, updated_by))
        except Exception as exc:
            logger.error("[ConfigManager] set(%s) DB error: %s", key, exc)
            raise

        with self._lock:
            self._cache[key] = value

        # Publish to Valkey for hot-reload across processes
        try:
            from agent.valkey_client import _get_client
            client = _get_client()
            if client is not None:
                payload = json.dumps({"key": key, "value": value})
                client.publish(_VALKEY_CHANNEL, payload)
        except Exception as exc:
            logger.debug("[ConfigManager] Valkey publish failed for %s: %s", key, exc)

    def set_many(self, updates: dict[str, Any], updated_by: str = "system") -> None:
        """Atomic multi-key write — one DB transaction, then publish each key."""
        if not updates:
            return
        ts = self._now_iso()
        rows = [(k, json.dumps(v), ts, updated_by) for k, v in updates.items()]
        try:
            from agent.db import get_conn
            with get_conn() as c:
                for row in rows:
                    c.execute(_UPSERT, row)
        except Exception as exc:
            logger.error("[ConfigManager] set_many DB error: %s", exc)
            raise

        with self._lock:
            for k, v in updates.items():
                self._cache[k] = v

        # Publish each changed key
        try:
            from agent.valkey_client import _get_client
            client = _get_client()
            if client is not None:
                pipe = client.pipeline(transaction=False)
                for k, v in updates.items():
                    payload = json.dumps({"key": k, "value": v})
                    pipe.publish(_VALKEY_CHANNEL, payload)
                pipe.execute()
        except Exception as exc:
            logger.debug("[ConfigManager] Valkey publish (set_many) failed: %s", exc)

    def all(self) -> dict[str, Any]:
        """Return a copy of the full in-memory cache."""
        with self._lock:
            return dict(self._cache)

    def _reload_key(self, key: str) -> None:
        """Re-read a single key from DB into cache (called on hot-reload)."""
        try:
            from agent.db import get_conn
            with get_conn() as c:
                row = c.execute(
                    "SELECT value FROM config_store WHERE key = ?", (key,)
                ).fetchone()
            if row:
                with self._lock:
                    try:
                        self._cache[key] = json.loads(row["value"])
                    except Exception:
                        self._cache[key] = row["value"]
                logger.debug("[ConfigManager] Hot-reloaded key: %s", key)
        except Exception as exc:
            logger.debug("[ConfigManager] _reload_key(%s) failed: %s", key, exc)

    def start_listener(self, runner=None) -> None:
        """
        Start a daemon thread that subscribes to Valkey `config:changed` channel.
        On message: parses JSON, extracts `key`, calls `_reload_key(key)`.
        """
        if self._listener_started:
            return
        self._listener_started = True
        t = threading.Thread(
            target=self._listener_loop,
            daemon=True,
            name="ConfigManagerListener",
        )
        t.start()
        logger.info("[ConfigManager] Valkey listener thread started.")

    def _listener_loop(self) -> None:
        """Background daemon: subscribe to config:changed and hot-reload keys."""
        retry_delay = 2.0
        while True:
            try:
                from agent.valkey_client import _cfg as _vcfg
                import redis as _redis_lib
                host, port, ssl = _vcfg()
                sub_client = _redis_lib.Redis(
                    host=host,
                    port=port,
                    ssl=ssl,
                    ssl_cert_reqs=None,
                    socket_connect_timeout=5,
                    socket_timeout=60,
                    decode_responses=False,
                )
                pubsub = sub_client.pubsub()
                pubsub.subscribe(_VALKEY_CHANNEL)
                logger.info("[ConfigManager] Subscribed to channel '%s'", _VALKEY_CHANNEL)
                retry_delay = 2.0  # reset on successful connect

                for message in pubsub.listen():
                    if message.get("type") != "message":
                        continue
                    data = message.get("data", b"")
                    try:
                        payload = json.loads(data)
                        key = payload.get("key")
                        if key:
                            self._reload_key(key)
                    except Exception:
                        pass

            except Exception as exc:
                logger.warning(
                    "[ConfigManager] Listener error: %s — retrying in %.0fs",
                    exc, retry_delay,
                )
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)


# ── Module-level singleton ─────────────────────────────────────────────────────

config = ConfigManager()
