"""Persistent, auditable lifecycle state for the trading universe."""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Confirmed by PostgreSQL + Valkey + Schwab history reconciliation on
# 2026-06-27. Keeping this evidence here seeds a fresh database safely; the
# database remains authoritative afterward and manual overrides are preserved.
INITIAL_QUARANTINE: tuple[str, ...] = (
    "CFLT", "ZI", "ALTR", "SMAR", "JAMF", "PAGER", "VMEO", "BIGC",
    "CYBR", "XPERI", "EZCORP", "CCMP", "PSEM", "IIVI", "HOLX", "EXAS",
    "FOLD", "BLUE", "RVNC", "VERV", "NOVA", "MAXN", "AY", "BSIG",
    "REGI", "PYCR", "EVBG", "DISH", "INFN", "DENN", "PACW", "BITF",
    "SDIG", "MIGI", "BSRT", "BFIN", "NKLA", "GOEV", "ZEV", "PTRA",
    "AYRO", "ASTR", "VORB", "OQAL", "KRTX", "RAPT", "GRPH", "ACCD",
    "WISH", "RDFN", "COMS", "CURO", "LAZR", "ISSI", "SWIR", "CREE",
    "APHA", "FLCX", "VIEW", "MTTR", "ORCC", "DESP", "RIDE", "COVA",
    "ATIP", "MNTV", "FRGE", "VCNX", "NTGN", "SRRA", "FARO", "CATC",
)

_DDL = """
CREATE TABLE IF NOT EXISTS universe_registry (
    ticker               TEXT PRIMARY KEY,
    tier                 INTEGER NOT NULL,
    status               TEXT NOT NULL,
    reason               TEXT NOT NULL DEFAULT '',
    source               TEXT NOT NULL DEFAULT '',
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    first_seen_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_checked_at      TIMESTAMPTZ,
    last_usable_at       TIMESTAMPTZ,
    manual_override      BOOLEAN NOT NULL DEFAULT FALSE,
    metadata_json        TEXT NOT NULL DEFAULT '{}'
)
"""

_init_lock = threading.Lock()
_initialized = False


def init_universe_registry() -> None:
    """Create and seed the registry once per process without overriding users."""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        from agent.db import get_conn
        from agent.ticker_universe import FULL_UNIVERSE, TIER1, TIER2

        tier1 = set(TIER1)
        tier2 = set(TIER2)
        with get_conn() as conn:
            conn.execute(_DDL)
            seed_rows = [
                (
                    ticker,
                    1 if ticker in tier1 else 2 if ticker in tier2 else 3,
                )
                for ticker in FULL_UNIVERSE
            ]
            conn.executemany(
                """
                INSERT INTO universe_registry
                  (ticker, tier, status, reason, source)
                VALUES (?, ?, 'ACTIVE', '', 'STATIC_CATALOG')
                ON CONFLICT (ticker) DO NOTHING
                """,
                seed_rows,
            )
            conn.execute(
                """
                UPDATE universe_registry
                SET status='QUARANTINED',
                    reason='NO_USABLE_SCHWAB_1M_HISTORY',
                    source='PRODUCTION_AUDIT_2026_06_27',
                    last_checked_at=NOW(),
                    consecutive_failures=GREATEST(consecutive_failures, 1)
                WHERE ticker = ANY(?) AND manual_override=FALSE
                """,
                (list(INITIAL_QUARANTINE),),
            )
        _initialized = True
        _publish_summary()


def get_runtime_universe() -> list[str]:
    """Return eligible symbols in stable catalog order; fail safely to seed data."""
    from agent.ticker_universe import FULL_UNIVERSE

    try:
        init_universe_registry()
        from agent.db import get_conn
        with get_conn(read_only=True) as conn:
            rows = conn.execute(
                "SELECT ticker FROM universe_registry WHERE status='ACTIVE'"
            ).fetchall()
        active = {str(row["ticker"]).upper() for row in rows}
        ordered = [ticker for ticker in FULL_UNIVERSE if ticker in active]
        ordered.extend(sorted(active - set(FULL_UNIVERSE)))
        return ordered
    except Exception as exc:
        logger.warning("[UniverseRegistry] runtime read failed: %s", exc)
        blocked = set(INITIAL_QUARANTINE)
        return [ticker for ticker in FULL_UNIVERSE if ticker not in blocked]


def record_history_audit(requested: Iterable[str], usable: Iterable[str]) -> None:
    """Record coverage; auto-quarantine only after repeated high-coverage audits."""
    symbols = {str(ticker).upper() for ticker in requested if ticker}
    good = {str(ticker).upper() for ticker in usable if ticker} & symbols
    if not symbols:
        return
    trustworthy = len(symbols) >= 100 and len(good) / len(symbols) >= 0.80
    now = datetime.now(timezone.utc)
    try:
        init_universe_registry()
        from agent.db import get_conn
        with get_conn() as conn:
            for ticker in symbols:
                if ticker in good:
                    conn.execute(
                        """
                        UPDATE universe_registry
                        SET consecutive_failures=0, last_checked_at=?, last_usable_at=?,
                            status=CASE
                              WHEN source='AUTO_HISTORY_AUDIT' AND manual_override=FALSE
                              THEN 'ACTIVE' ELSE status END,
                            reason=CASE
                              WHEN source='AUTO_HISTORY_AUDIT' AND manual_override=FALSE
                              THEN '' ELSE reason END
                        WHERE ticker=?
                        """,
                        (now, now, ticker),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE universe_registry
                        SET consecutive_failures=consecutive_failures+1,
                            last_checked_at=?,
                            status=CASE
                              WHEN ? AND consecutive_failures+1 >= 3
                                   AND manual_override=FALSE
                              THEN 'QUARANTINED' ELSE status END,
                            reason=CASE
                              WHEN ? AND consecutive_failures+1 >= 3
                                   AND manual_override=FALSE
                              THEN 'NO_USABLE_SCHWAB_1M_HISTORY' ELSE reason END,
                            source=CASE
                              WHEN ? AND consecutive_failures+1 >= 3
                                   AND manual_override=FALSE
                              THEN 'AUTO_HISTORY_AUDIT' ELSE source END
                        WHERE ticker=?
                        """,
                        (now, trustworthy, trustworthy, trustworthy, ticker),
                    )
        _publish_summary()
    except Exception as exc:
        logger.warning("[UniverseRegistry] history audit failed: %s", exc)


def promote_replacement(
    ticker: str,
    *,
    average_daily_volume: float,
    history_bars: int,
    quote_price: float,
    listing_verified: bool,
    source: str,
) -> bool:
    """Promote a replacement only after listing, liquidity, and history checks."""
    symbol = str(ticker or "").upper().strip()
    if (
        not symbol
        or not listing_verified
        or float(quote_price) < 1.0
        or float(average_daily_volume) < 500_000
        or int(history_bars) < 390
    ):
        return False
    init_universe_registry()
    from agent.db import get_conn
    now = datetime.now(timezone.utc)
    metadata = json.dumps(
        {
            "average_daily_volume": round(float(average_daily_volume), 2),
            "history_bars": int(history_bars),
            "quote_price": round(float(quote_price), 4),
        },
        separators=(",", ":"),
    )
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO universe_registry
              (ticker, tier, status, reason, source, consecutive_failures,
               first_seen_at, last_checked_at, last_usable_at, metadata_json)
            VALUES (?, 3, 'ACTIVE', 'VALIDATED_REPLACEMENT', ?, 0, ?, ?, ?, ?)
            ON CONFLICT (ticker) DO UPDATE SET
              status=CASE WHEN universe_registry.manual_override THEN universe_registry.status
                          ELSE 'ACTIVE' END,
              reason=CASE WHEN universe_registry.manual_override THEN universe_registry.reason
                          ELSE 'VALIDATED_REPLACEMENT' END,
              source=CASE WHEN universe_registry.manual_override THEN universe_registry.source
                          ELSE EXCLUDED.source END,
              consecutive_failures=0, last_checked_at=EXCLUDED.last_checked_at,
              last_usable_at=EXCLUDED.last_usable_at,
              metadata_json=EXCLUDED.metadata_json
            """,
            (symbol, source, now, now, now, metadata),
        )
    _publish_summary()
    return True


def request_recheck(ticker: str, *, requested_by: str = "SYSTEM") -> dict[str, Any]:
    """Move a known symbol to CANDIDATE and queue provider-side validation."""
    symbol = str(ticker or "").upper().strip()
    if not symbol:
        raise ValueError("Ticker is required")
    init_universe_registry()
    from agent.db import get_conn

    with get_conn() as conn:
        row = conn.execute(
            "SELECT ticker, status FROM universe_registry WHERE ticker=?",
            (symbol,),
        ).fetchone()
        if not row:
            raise ValueError(f"Unknown universe ticker: {symbol}")
        conn.execute(
            """
            UPDATE universe_registry
            SET status='CANDIDATE', reason='RECHECK_REQUESTED',
                source=?, manual_override=FALSE, last_checked_at=NOW()
            WHERE ticker=?
            """,
            (f"ADMIN:{requested_by}"[:120], symbol),
        )
    queued = False
    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        if client is not None:
            client.sadd("universe:recheck:requested", symbol)
            client.publish(
                "universe:recheck",
                json.dumps({"ticker": symbol, "requested_by": requested_by}),
            )
            queued = True
    except Exception as exc:
        logger.warning("[UniverseRegistry] recheck queue failed for %s: %s", symbol, exc)
    _publish_summary()
    return {"ticker": symbol, "status": "CANDIDATE", "queued": queued}


def quarantine_ticker(
    ticker: str,
    *,
    reason: str,
    requested_by: str = "SYSTEM",
) -> dict[str, Any]:
    """Apply an explicit operator quarantine; it cannot auto-promote later."""
    symbol = str(ticker or "").upper().strip()
    detail = str(reason or "").strip()
    if not symbol:
        raise ValueError("Ticker is required")
    if len(detail) < 8:
        raise ValueError("A quarantine reason of at least 8 characters is required")
    init_universe_registry()
    from agent.db import get_conn

    with get_conn() as conn:
        row = conn.execute(
            "SELECT ticker FROM universe_registry WHERE ticker=?", (symbol,)
        ).fetchone()
        if not row:
            raise ValueError(f"Unknown universe ticker: {symbol}")
        conn.execute(
            """
            UPDATE universe_registry
            SET status='QUARANTINED', reason=?, source=?, manual_override=TRUE,
                last_checked_at=NOW()
            WHERE ticker=?
            """,
            (detail[:240], f"ADMIN:{requested_by}"[:120], symbol),
        )
    _publish_summary()
    return {"ticker": symbol, "status": "QUARANTINED", "manual_override": True}


def record_recheck_failure(ticker: str, *, reason: str) -> None:
    """Return a failed candidate to quarantine without touching manual holds."""
    symbol = str(ticker or "").upper().strip()
    if not symbol:
        return
    init_universe_registry()
    from agent.db import get_conn

    with get_conn() as conn:
        conn.execute(
            """
            UPDATE universe_registry
            SET status=CASE WHEN manual_override THEN status ELSE 'QUARANTINED' END,
                reason=CASE WHEN manual_override THEN reason ELSE ? END,
                source=CASE WHEN manual_override THEN source ELSE 'PROVIDER_RECHECK' END,
                consecutive_failures=consecutive_failures+1,
                last_checked_at=NOW()
            WHERE ticker=?
            """,
            (str(reason or "RECHECK_FAILED")[:240], symbol),
        )
    _publish_summary()


def get_universe_registry_summary(*, include_symbols: bool = True) -> dict[str, Any]:
    init_universe_registry()
    from agent.db import get_conn
    from agent.ticker_universe import FULL_UNIVERSE

    with get_conn(read_only=True) as conn:
        counts = conn.execute(
            "SELECT status, COUNT(*) AS count FROM universe_registry GROUP BY status"
        ).fetchall()
        quarantined = conn.execute(
            """
            SELECT ticker, tier, reason, source, consecutive_failures,
                   last_checked_at, last_usable_at, manual_override
            FROM universe_registry
            WHERE status='QUARANTINED'
            ORDER BY tier, ticker
            """
        ).fetchall()
        candidates = conn.execute(
            """
            SELECT ticker, tier, reason, source, consecutive_failures,
                   last_checked_at, last_usable_at, manual_override
            FROM universe_registry
            WHERE status='CANDIDATE'
            ORDER BY tier, ticker
            """
        ).fetchall()
    by_status = {str(row["status"]): int(row["count"]) for row in counts}
    result: dict[str, Any] = {
        "catalog_total": len(FULL_UNIVERSE),
        "eligible_total": by_status.get("ACTIVE", 0),
        "quarantined_total": by_status.get("QUARANTINED", 0),
        "candidate_total": by_status.get("CANDIDATE", 0),
        "status_counts": by_status,
    }
    if include_symbols:
        result["quarantined"] = [dict(row) for row in quarantined]
        result["candidates"] = [dict(row) for row in candidates]
        result["eligible_tickers"] = get_runtime_universe()
    return result


def _publish_summary() -> None:
    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        if client is not None:
            payload = get_universe_registry_summary(include_symbols=False)
            payload["ts"] = datetime.now(timezone.utc).timestamp()
            client.setex("universe:registry", 300, json.dumps(payload))
            client.set(
                "universe:eligible",
                json.dumps(get_runtime_universe(), separators=(",", ":")),
            )
            client.publish("universe:changed", json.dumps(payload))
    except Exception:
        pass
