"""
system_alerts.py — shared PostgreSQL-backed alert primitive.

The single place any service raises an operational alert that must be SURFACED
to the dashboard rather than silently logged. Requirement 5 of production
readiness: errors are surfaced to the dashboard, not suppressed.

Design (mirrors service_state.py):
  - PostgreSQL `system_alerts` is the durable source of truth.
  - Valkey channel `system:alerts` is the live-bus for instant dashboard
    updates (best-effort; PostgreSQL still holds the record if Valkey is down).
  - Alerts deduplicate on `alert_key` (default "{alert_type}:{source}"): a
    repeating failure (e.g. an OAuth refresh retry loop) bumps `occurrences`
    and `last_seen` on the existing OPEN row instead of spamming new rows.
  - resolve_alert() closes an alert when the underlying condition clears (e.g.
    a successful token refresh), so the dashboard banner clears itself.

DB interface note (same as service_state.py):
  agent.db.get_conn() returns _PgConnection — conn.execute(sql, params) →
  cursor with .fetchone()/.fetchall() returning RealDictRow dicts. Do NOT call
  conn.cursor(). We write %s placeholders directly (no ? → %s conversion).
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Valid severities, ordered most→least severe for summary ranking.
SEVERITIES = ("CRITICAL", "WARNING", "INFO")
_SEVERITY_RANK = {"CRITICAL": 3, "WARNING": 2, "INFO": 1}

_VALKEY_CHANNEL = "system:alerts"


def _sanitize_for_jsonb(obj: Any) -> Any:
    """Replace NaN/Infinity with None so PostgreSQL JSONB accepts the payload."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_jsonb(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_jsonb(v) for v in obj]
    return obj


# ── Lazy auto-init ────────────────────────────────────────────────────────────
# raise_alert() is called from any container (market-data, scanner, learner)
# that may start before web-api runs init_db(). Self-initialize once per process.
_init_lock = threading.Lock()
_db_ready  = False


def _ensure_init() -> None:
    global _db_ready
    if _db_ready:
        return
    with _init_lock:
        if not _db_ready:
            _db_ready = init_db()   # True only on real success; retry next call otherwise


_DDL = """
CREATE TABLE IF NOT EXISTS system_alerts (
    id           SERIAL PRIMARY KEY,
    alert_key    TEXT        NOT NULL,
    alert_type   TEXT        NOT NULL,
    severity     TEXT        NOT NULL DEFAULT 'WARNING',
    source       TEXT        NOT NULL DEFAULT '',
    title        TEXT        NOT NULL,
    message      TEXT        NOT NULL DEFAULT '',
    metadata     JSONB       NOT NULL DEFAULT '{}',
    occurrences  INTEGER     NOT NULL DEFAULT 1,
    first_seen   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at  TIMESTAMPTZ NULL,
    resolved_by  TEXT        NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_system_alerts_open_key
    ON system_alerts (alert_key)
    WHERE resolved_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_system_alerts_open
    ON system_alerts (last_seen DESC)
    WHERE resolved_at IS NULL;
"""


def init_db() -> bool:
    """Create the system_alerts table (idempotent). Returns True on confirmed success."""
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            conn.execute(_DDL)
        logger.info("[system_alerts] table ready")
        return True
    except Exception as exc:
        logger.warning("[system_alerts] init_db error: %s", exc)
        return False


def _publish(event: str, payload: dict) -> None:
    """Best-effort Valkey publish for live dashboard updates."""
    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        if client is not None:
            client.publish(_VALKEY_CHANNEL, json.dumps({"event": event, **payload}))
    except Exception as exc:
        logger.debug("[system_alerts] publish failed: %s", exc)


def raise_alert(
    alert_type: str,
    title:      str,
    message:    str = "",
    severity:   str = "WARNING",
    source:     str = "",
    metadata:   Optional[dict] = None,
    dedup_key:  Optional[str]  = None,
) -> bool:
    """
    Raise (or re-raise) an operational alert. Best-effort — returns False on DB
    error so the calling failure path never crashes the service.

    Deduplication: alerts collapse on `dedup_key` (default "{alert_type}:{source}").
    A repeating condition bumps `occurrences` + `last_seen` on the existing OPEN
    row rather than inserting a new one — so an OAuth retry loop produces ONE
    banner with a rising count, not hundreds of rows.

    severity: one of CRITICAL / WARNING / INFO (defaults to WARNING if invalid).
    """
    sev = severity.upper() if severity else "WARNING"
    if sev not in _SEVERITY_RANK:
        sev = "WARNING"
    key = dedup_key or f"{alert_type}:{source}"
    meta = _sanitize_for_jsonb(metadata or {})

    _ensure_init()
    try:
        from agent.db import get_conn
        # Partial-unique index on alert_key WHERE resolved_at IS NULL lets us
        # UPSERT the OPEN alert; closed history rows are never touched.
        sql = """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, source, title, message, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (alert_key) WHERE resolved_at IS NULL
            DO UPDATE SET
                severity    = EXCLUDED.severity,
                title       = EXCLUDED.title,
                message     = EXCLUDED.message,
                metadata    = EXCLUDED.metadata,
                occurrences = system_alerts.occurrences + 1,
                last_seen   = NOW()
            RETURNING id, occurrences
        """
        with get_conn() as conn:
            cur = conn.execute(sql, (
                key, alert_type, sev, source, title, message,
                json.dumps(meta, default=str),
            ))
            row = cur.fetchone()
        alert_id = int(row["id"]) if row else None
        occ      = int(row["occurrences"]) if row else 1
        logger.warning(
            "[system_alerts] %s [%s] %s — %s (occ=%d)",
            sev, alert_type, title, message, occ,
        )
        _publish("raised", {
            "id": alert_id, "alert_key": key, "alert_type": alert_type,
            "severity": sev, "source": source, "title": title,
            "message": message, "occurrences": occ, "ts": time.time(),
        })
        return True
    except Exception as exc:
        # Even if persistence fails, make sure the failure is in the log.
        logger.error("[system_alerts] raise_alert(%s) failed: %s | original: %s/%s",
                     key, exc, title, message)
        return False


def resolve_alert(
    alert_key:   Optional[str] = None,
    alert_type:  Optional[str] = None,
    resolved_by: str = "system",
) -> int:
    """
    Resolve (close) one or more OPEN alerts. Returns the number resolved.

    Pass `alert_key` to close a specific deduplicated alert, or `alert_type`
    to close every OPEN alert of a type (e.g. clear all SCHWAB_AUTH alerts on
    a successful token refresh). One of the two must be provided.
    """
    if not alert_key and not alert_type:
        return 0
    _ensure_init()
    try:
        from agent.db import get_conn
        if alert_key:
            where, param = "alert_key = %s", alert_key
        else:
            where, param = "alert_type = %s", alert_type
        sql = (
            f"UPDATE system_alerts SET resolved_at = NOW(), resolved_by = %s "
            f"WHERE {where} AND resolved_at IS NULL "
            f"RETURNING id"
        )
        with get_conn() as conn:
            cur = conn.execute(sql, (resolved_by, param))
            rows = cur.fetchall()
        n = len(rows) if rows else 0
        if n:
            logger.info("[system_alerts] resolved %d alert(s) for %s",
                       n, alert_key or alert_type)
            _publish("resolved", {
                "alert_key": alert_key, "alert_type": alert_type,
                "resolved_by": resolved_by, "count": n, "ts": time.time(),
            })
        return n
    except Exception as exc:
        logger.warning("[system_alerts] resolve_alert(%s) error: %s",
                      alert_key or alert_type, exc)
        return 0


def get_active_alerts(limit: int = 50) -> list[dict]:
    """Return OPEN alerts, most-severe first then most-recent. [] on error."""
    _ensure_init()
    try:
        from agent.db import get_conn
        sql = """
            SELECT id, alert_key, alert_type, severity, source, title, message,
                   metadata, occurrences,
                   EXTRACT(EPOCH FROM first_seen) AS first_seen_epoch,
                   EXTRACT(EPOCH FROM last_seen)  AS last_seen_epoch,
                   EXTRACT(EPOCH FROM (NOW() - last_seen)) AS age_s
            FROM system_alerts
            WHERE resolved_at IS NULL
            ORDER BY
                CASE severity WHEN 'CRITICAL' THEN 3 WHEN 'WARNING' THEN 2 ELSE 1 END DESC,
                last_seen DESC
            LIMIT %s
        """
        with get_conn() as conn:
            rows = conn.execute(sql, (int(limit),)).fetchall()
        out: list[dict] = []
        for r in (rows or []):
            meta = r["metadata"]
            if not isinstance(meta, dict):
                try:
                    meta = json.loads(meta) if meta else {}
                except Exception:
                    meta = {}
            out.append({
                "id":          int(r["id"]),
                "alert_key":   r["alert_key"],
                "alert_type":  r["alert_type"],
                "severity":    r["severity"],
                "source":      r["source"],
                "title":       r["title"],
                "message":     r["message"],
                "metadata":    meta,
                "occurrences": int(r["occurrences"]),
                "first_seen":  float(r["first_seen_epoch"]) if r["first_seen_epoch"] else None,
                "last_seen":   float(r["last_seen_epoch"])  if r["last_seen_epoch"]  else None,
                "age_s":       float(r["age_s"]) if r["age_s"] is not None else None,
            })
        return out
    except Exception as exc:
        logger.debug("[system_alerts] get_active_alerts error: %s", exc)
        return []


def get_alert_summary() -> dict:
    """Lightweight counts for the dashboard banner. Always returns a dict."""
    alerts = get_active_alerts(limit=200)
    counts = {"CRITICAL": 0, "WARNING": 0, "INFO": 0}
    for a in alerts:
        counts[a["severity"]] = counts.get(a["severity"], 0) + 1
    top_sev = None
    for sev in SEVERITIES:
        if counts.get(sev, 0) > 0:
            top_sev = sev
            break
    return {
        "total":        len(alerts),
        "counts":       counts,
        "top_severity": top_sev,
        "alerts":       alerts[:20],  # banner shows the most relevant few
    }
