"""
audit_log.py — append-only PostgreSQL audit trail for decisions & events.

Production requirement 6: an audit log in PostgreSQL for all errors and events.
system_alerts.py already covers *errors that need operator attention*; this
covers the *decision trail* — why the system did (or did not) act:

  - SIGNAL_SUPPRESSED   — a BUY/SELL signal blocked by the adaptive filter
                          (includes the confidence, learned threshold, and the
                          ML model scores behind the rejected signal).
  - THRESHOLD_CHANGED   — the adaptive filter moved its learned confidence gate
                          (old → new value, win-rate, trigger).
  - (extensible)        — any future decision/config/event sink.

Design (mirrors system_alerts.py / service_state.py):
  - PostgreSQL `audit_log` is the durable, append-only source of truth.
  - High-frequency events (suppressions) are ENQUEUED to an in-memory buffer and
    flushed in batches by a daemon thread, so audit writes never block the
    scanner hot loop and never hammer PostgreSQL one-row-at-a-time. Low-frequency
    critical events (threshold changes) can be written synchronously (sync=True).
  - Best-effort everywhere: audit must never crash or stall trading.
  - Opportunistic retention prune keeps the table bounded (audit.retention_days).

DB interface note (same as service_state.py): get_conn() → conn.execute(sql,
params) with %s placeholders and RealDictRow rows; no conn.cursor(), commit on
context exit.
"""
from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_VALKEY_CHANNEL = "system:audit"

# Async buffer — bounded so a DB outage can't grow memory without limit.
_MAX_QUEUE       = 20000
_FLUSH_INTERVAL_S = 3.0     # drain the buffer every N seconds
_MAX_BATCH        = 500     # rows per flush transaction
_PRUNE_INTERVAL_S = 3600.0  # opportunistic retention prune cadence (per process)

_q: "queue.Queue[dict]" = queue.Queue(maxsize=_MAX_QUEUE)
_flusher_started = False
_flusher_lock    = threading.Lock()


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
_init_lock = threading.Lock()
_db_ready  = False


def _ensure_init() -> None:
    global _db_ready
    if _db_ready:
        return
    with _init_lock:
        if not _db_ready:
            _db_ready = init_db()


def _enabled() -> bool:
    """Audit can be toggled off live via PostgreSQL (audit.enabled)."""
    try:
        from agent.config_manager import config as _cfg
        return bool(_cfg.get("audit.enabled", True))
    except Exception:
        return True


_DDL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGSERIAL   PRIMARY KEY,
    event_type  TEXT        NOT NULL,
    category    TEXT        NOT NULL DEFAULT 'decision',
    ticker      TEXT        NULL,
    source      TEXT        NOT NULL DEFAULT '',
    summary     TEXT        NOT NULL DEFAULT '',
    detail      JSONB       NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_log_created
    ON audit_log (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_type
    ON audit_log (event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_ticker
    ON audit_log (ticker, created_at DESC)
    WHERE ticker IS NOT NULL;
"""


def init_db() -> bool:
    """Create the audit_log table (idempotent). Returns True on confirmed success."""
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            conn.execute(_DDL)
        logger.info("[audit_log] table ready")
        return True
    except Exception as exc:
        logger.warning("[audit_log] init_db error: %s", exc)
        return False


def _publish(payload: dict) -> None:
    """Best-effort Valkey publish so the dashboard can stream new audit rows."""
    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        if client is not None:
            client.publish(_VALKEY_CHANNEL, json.dumps(payload, default=str))
    except Exception as exc:
        logger.debug("[audit_log] publish failed: %s", exc)


def _insert_batch(events: list[dict]) -> None:
    """Write a batch of audit events in ONE transaction (commit on context exit)."""
    if not events:
        return
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            for e in events:
                conn.execute(
                    """
                    INSERT INTO audit_log
                        (event_type, category, ticker, source, summary, detail, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                    """,
                    (
                        e["event_type"], e.get("category", "decision"),
                        e.get("ticker"), e.get("source", ""),
                        e.get("summary", ""),
                        json.dumps(_sanitize_for_jsonb(e.get("detail", {})), default=str),
                        e.get("created_at") or datetime.now(timezone.utc).isoformat(),
                    ),
                )
    except Exception as exc:
        logger.warning("[audit_log] insert batch (%d rows) failed: %s", len(events), exc)


def _prune() -> None:
    """Delete rows older than audit.retention_days (best-effort)."""
    try:
        from agent.config_manager import config as _cfg
        days = int(_cfg.get("audit.retention_days", 14))
    except Exception:
        days = 14
    if days <= 0:
        return
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            conn.execute(
                f"DELETE FROM audit_log WHERE created_at < NOW() - INTERVAL '{days} days'"
            )
        logger.debug("[audit_log] pruned rows older than %d days", days)
    except Exception as exc:
        logger.debug("[audit_log] prune error: %s", exc)


def _flusher_loop() -> None:
    """Daemon: drain the buffer in batches and prune on an interval."""
    last_prune = time.time()
    while True:
        time.sleep(_FLUSH_INTERVAL_S)
        batch: list[dict] = []
        try:
            while len(batch) < _MAX_BATCH:
                batch.append(_q.get_nowait())
        except queue.Empty:
            pass
        if batch:
            _ensure_init()
            _insert_batch(batch)
        now = time.time()
        if now - last_prune >= _PRUNE_INTERVAL_S:
            _ensure_init()
            _prune()
            last_prune = now


def _ensure_flusher() -> None:
    global _flusher_started
    if _flusher_started:
        return
    with _flusher_lock:
        if not _flusher_started:
            t = threading.Thread(target=_flusher_loop, daemon=True, name="audit-flusher")
            t.start()
            _flusher_started = True


def audit(
    event_type: str,
    summary:    str = "",
    *,
    ticker:   Optional[str]  = None,
    source:   str            = "",
    category: str            = "decision",
    detail:   Optional[dict] = None,
    sync:     bool           = False,
) -> None:
    """
    Record an audit event. Best-effort; never raises.

    sync=False (default): enqueue for batched async write — use for
      high-frequency events (suppressions) so the caller never blocks on DB I/O.
    sync=True: write immediately in the calling thread — use for rare, critical
      events (threshold changes) that must survive an abrupt shutdown.
    """
    if not _enabled():
        return
    evt = {
        "event_type": event_type,
        "category":   category,
        "ticker":     ticker,
        "source":     source,
        "summary":    summary,
        "detail":     detail or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if sync:
        _ensure_init()
        _insert_batch([evt])
        _publish({"event_type": event_type, "ticker": ticker, "summary": summary,
                  "source": source, "ts": time.time()})
        return
    try:
        _q.put_nowait(evt)
    except queue.Full:
        # Buffer saturated (DB outage). Drop rather than block trading — the
        # condition itself is already visible via the DB-error logs above.
        logger.debug("[audit_log] buffer full — dropping %s event", event_type)
    _ensure_flusher()


def get_recent(
    limit:      int = 100,
    event_type: Optional[str] = None,
    ticker:     Optional[str] = None,
) -> list[dict]:
    """Return recent audit rows (newest first), optionally filtered. [] on error."""
    _ensure_init()
    try:
        from agent.db import get_conn
        clauses, params = [], []
        if event_type:
            clauses.append("event_type = %s"); params.append(event_type)
        if ticker:
            clauses.append("ticker = %s"); params.append(ticker)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))
        sql = f"""
            SELECT id, event_type, category, ticker, source, summary, detail,
                   EXTRACT(EPOCH FROM created_at) AS created_epoch
            FROM audit_log
            {where}
            ORDER BY id DESC
            LIMIT %s
        """
        with get_conn() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        out: list[dict] = []
        for r in (rows or []):
            det = r["detail"]
            if not isinstance(det, dict):
                try:
                    det = json.loads(det) if det else {}
                except Exception:
                    det = {}
            out.append({
                "id":         int(r["id"]),
                "event_type": r["event_type"],
                "category":   r["category"],
                "ticker":     r["ticker"],
                "source":     r["source"],
                "summary":    r["summary"],
                "detail":     det,
                "created_at": float(r["created_epoch"]) if r["created_epoch"] else None,
            })
        return out
    except Exception as exc:
        logger.debug("[audit_log] get_recent error: %s", exc)
        return []
