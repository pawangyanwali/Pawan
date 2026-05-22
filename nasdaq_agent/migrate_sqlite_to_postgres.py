#!/usr/bin/env python3
"""
NASDAQ Agent — SQLite → PostgreSQL full migration
==================================================

BEFORE YOU RUN THIS SCRIPT
---------------------------
1. Run the DDL additions on PostgreSQL (paste into psql/pgAdmin):

    -- Add missing columns & table that weren't in the v1 schema
    ALTER TABLE signals ADD COLUMN IF NOT EXISTS is_suppressed BOOLEAN DEFAULT FALSE;
    ALTER TABLE signals ADD COLUMN IF NOT EXISTS short_outcome  TEXT    DEFAULT 'PENDING';
    ALTER TABLE signals ADD COLUMN IF NOT EXISTS vol_bucket     TEXT    DEFAULT 'NORMAL';
    ALTER TABLE signals ADD COLUMN IF NOT EXISTS trend          TEXT    DEFAULT '';

    ALTER TABLE trades  ADD COLUMN IF NOT EXISTS metadata       JSONB;

    ALTER TABLE signal_observations
        ADD COLUMN IF NOT EXISTS source VARCHAR(20) NOT NULL DEFAULT 'LIVE'
        CHECK (source IN ('LIVE','BACKTEST','WEEKEND'));

    CREATE TABLE IF NOT EXISTS backtest_summary (
        id          BIGSERIAL PRIMARY KEY,
        run_dt      TIMESTAMPTZ NOT NULL,
        timeframe   VARCHAR(10) NOT NULL,
        ticker      VARCHAR(10) NOT NULL,
        total       INTEGER     NOT NULL DEFAULT 0,
        wins        INTEGER     NOT NULL DEFAULT 0,
        win_rate    NUMERIC(5,4),
        avg_pnl_r   NUMERIC(8,4),
        expectancy  NUMERIC(8,4),
        max_dd_r    NUMERIC(8,4),
        sharpe      NUMERIC(8,4),
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (run_dt, timeframe, ticker)
    );

    INSERT INTO schema_migrations (version, description)
    VALUES (2, 'source on signal_observations, backtest_summary table')
    ON CONFLICT DO NOTHING;

2. Copy SQLite files from Lightsail → your machine → bastion:

    # On Lightsail (via SSH):
    tar -czf /tmp/sqlite_bak.tar.gz \\
        /opt/nasdaq-agent/nasdaq_agent/data/signal_history.db \\
        /opt/nasdaq-agent/nasdaq_agent/data/paper_trades.db \\
        /opt/nasdaq-agent/nasdaq_agent/data/live_backtest.db \\
        /opt/nasdaq-agent/nasdaq_agent/data/weekend_learning.db \\
        /opt/nasdaq-agent/nasdaq_agent/data/backtest_mtf.db

    # Copy to bastion (from your local machine):
    scp -i lightsail-key.pem ubuntu@LIGHTSAIL_IP:/tmp/sqlite_bak.tar.gz /tmp/
    scp -i bastion-key.pem /tmp/sqlite_bak.tar.gz ec2-user@3.234.216.250:/tmp/

    # On bastion:
    mkdir -p /tmp/sqlite && cd /tmp/sqlite
    tar -xzf /tmp/sqlite_bak.tar.gz --strip-components=5

3. Install psycopg2 on bastion:
    pip3 install psycopg2-binary --user

USAGE
-----
Dry run (reads SQLite, shows counts, NEVER writes to Postgres):
    python3 migrate_sqlite_to_postgres.py --dry-run \\
        --pg-host trading.xxxxxx.us-east-1.rds.amazonaws.com \\
        --pg-pass YOUR_PASSWORD

Real migration:
    python3 migrate_sqlite_to_postgres.py \\
        --pg-host trading.xxxxxx.us-east-1.rds.amazonaws.com \\
        --pg-pass YOUR_PASSWORD

Sync open trades only (re-run after first migration to refresh status of 14 open trades):
    python3 migrate_sqlite_to_postgres.py --sync-open-trades \\
        --pg-host trading.xxxxxx.us-east-1.rds.amazonaws.com \\
        --pg-pass YOUR_PASSWORD

OPEN TRADE STRATEGY (read this!)
---------------------------------
The 14 open paper trades are actively being managed by the live Lightsail service.
Their state (bars_held, t1_hit, partial_pnl_dollar, shares_remaining, etc.)
changes every minute. This script:

  Phase 1 — run NOW:
    • Migrates all CLOSED trades (they won't change)
    • Also migrates the 14 OPEN trades as a snapshot (status='OPEN')
    • Tags them with metadata->>'sqlite_id' so you can resync later

  Phase 2 — run just before app code cutover to Postgres:
    • --sync-open-trades re-reads paper_trades.db and UPSERTs OPEN rows
    • Run this immediately before deploying the PostgreSQL-native app version
    • After cutover the app writes to PostgreSQL directly; SQLite is retired
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("migrate")

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    log.error("psycopg2 not installed.  Run: pip3 install psycopg2-binary --user")
    sys.exit(1)


# ─── Connection helpers ───────────────────────────────────────────────────────

@contextmanager
def sqlite_ro(path: Path):
    """Open SQLite read-only (URI mode) — never writes lock files."""
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def make_pg_conn(host: str, port: int, db: str, user: str, password: str):
    return psycopg2.connect(
        host=host, port=port, dbname=db, user=user, password=password,
        connect_timeout=15,
        options="-c application_name=sqlite_migration",
    )


# ─── Type-safe helpers ────────────────────────────────────────────────────────

def _f(val: Any) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _i(val: Any) -> Optional[int]:
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _b(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    return bool(int(val))


def _ts(val: Any) -> Optional[str]:
    """Return a TIMESTAMPTZ-compatible string or None."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # Already ISO — PostgreSQL will accept it
    return s


# ─── 1. signal_history.db  →  signals ────────────────────────────────────────

def migrate_signals(sqlite_dir: Path, pg, dry_run: bool) -> dict:
    db_path = sqlite_dir / "signal_history.db"
    if not db_path.exists():
        log.warning("signal_history.db not found — skipping")
        return {"skipped": True}

    with sqlite_ro(db_path) as sc:
        rows = sc.execute("""
            SELECT id, ts, ticker, direction,
                   entry_price, target, stop, confidence,
                   session, regime, trading_tier, vwap_event,
                   rsi_zone, vol_bucket, trend,
                   outcome, short_outcome, exit_price, pnl_pct,
                   bars_held, is_suppressed
            FROM signals
            ORDER BY id
        """).fetchall()

    log.info(f"signal_history.db: {len(rows):,} rows")
    if dry_run:
        return {"total": len(rows), "dry_run": True}

    inserted = updated = 0
    with pg.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO signals (
                fired_at, ticker, direction,
                entry_price, target_price, stop_price, confidence,
                session, regime, trading_tier, vwap_event,
                rsi_zone, vol_bucket, trend,
                outcome, short_outcome, exit_price, pnl_pct,
                bars_held, is_suppressed
            ) VALUES (
                %s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,%s,
                %s,%s
            )
            ON CONFLICT DO NOTHING
            """,
            [
                (
                    _ts(r["ts"]), r["ticker"], r["direction"],
                    _f(r["entry_price"]), _f(r["target"]), _f(r["stop"]), _f(r["confidence"]),
                    r["session"] or "", r["regime"] or "", r["trading_tier"] or "REGULAR",
                    r["vwap_event"] or "", r["rsi_zone"] or "", r["vol_bucket"] or "NORMAL",
                    r["trend"] or "",
                    r["outcome"] or "PENDING", r["short_outcome"] or "PENDING",
                    _f(r["exit_price"]), _f(r["pnl_pct"]),
                    _i(r["bars_held"]) or 0, _b(r["is_suppressed"]),
                )
                for r in rows
            ],
            page_size=500,
        )
        inserted = cur.rowcount
    pg.commit()

    # Rebuild ticker_stats from signals
    _rebuild_ticker_stats(pg)

    log.info(f"  ↳ signals inserted: {len(rows):,}")
    return {"total": len(rows)}


def _rebuild_ticker_stats(pg) -> None:
    """Recompute ticker_stats from the signals table."""
    log.info("Rebuilding ticker_stats …")
    with pg.cursor() as cur:
        cur.execute("""
            INSERT INTO ticker_stats (ticker, total_signals, total_wins, win_rate, avg_pnl_r, last_updated)
            SELECT
                ticker,
                COUNT(*)                                                        AS total_signals,
                COUNT(*) FILTER (WHERE outcome = 'WIN')                         AS total_wins,
                ROUND(
                    COUNT(*) FILTER (WHERE outcome = 'WIN')::NUMERIC
                    / NULLIF(COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS')),0),
                    4
                )                                                               AS win_rate,
                ROUND(AVG(pnl_pct) FILTER (WHERE outcome IN ('WIN','LOSS')), 4) AS avg_pnl_r,
                NOW()
            FROM signals
            GROUP BY ticker
            ON CONFLICT (ticker) DO UPDATE SET
                total_signals = EXCLUDED.total_signals,
                total_wins    = EXCLUDED.total_wins,
                win_rate      = EXCLUDED.win_rate,
                avg_pnl_r     = EXCLUDED.avg_pnl_r,
                last_updated  = NOW()
        """)
    pg.commit()
    log.info("  ↳ ticker_stats rebuilt")


# ─── 2. paper_trades.db  →  trades ───────────────────────────────────────────

def migrate_trades(sqlite_dir: Path, pg, dry_run: bool, open_only: bool = False) -> dict:
    db_path = sqlite_dir / "paper_trades.db"
    if not db_path.exists():
        log.warning("paper_trades.db not found — skipping")
        return {"skipped": True}

    status_filter = "WHERE status = 'OPEN'" if open_only else ""
    with sqlite_ro(db_path) as sc:
        rows = sc.execute(f"""
            SELECT
                id, opened_at, closed_at, ticker, direction,
                entry_price, target, stop, confidence,
                rr_ratio, rr_qualifies, bars_held, status,
                exit_price, exit_reason, pnl_pct, pnl_dollar,
                shares, session, regime, vwap_event, rsi_zone, entry_type,
                t1_hit, t1_price, t2_price, breakeven_set,
                partial_pnl_dollar, shares_remaining,
                order_flow_score, size_mult
            FROM paper_trades
            {status_filter}
            ORDER BY id
        """).fetchall()

    open_count   = sum(1 for r in rows if r["status"] == "OPEN")
    closed_count = len(rows) - open_count
    label = "OPEN-only sync" if open_only else "full"
    log.info(f"paper_trades.db ({label}): {len(rows):,} rows  "
             f"({open_count} OPEN · {closed_count} CLOSED)")

    if dry_run:
        return {"total": len(rows), "open": open_count, "closed": closed_count, "dry_run": True}

    with pg.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO trades (
                opened_at, closed_at, ticker, direction,
                entry_price, target_price, stop_price, confidence,
                rr_ratio, rr_qualifies, bars_held, status,
                exit_price, exit_reason, pnl_pct, pnl_dollar,
                shares, session, regime, vwap_event, rsi_zone, entry_type,
                t1_hit, t1_price, t2_price, breakeven_set,
                partial_pnl_dollar, shares_remaining,
                order_flow_score, size_mult, metadata
            ) VALUES (
                %s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,
                %s,%s,%s
            )
            ON CONFLICT DO NOTHING
            """,
            [
                (
                    _ts(r["opened_at"]), _ts(r["closed_at"]),
                    r["ticker"], r["direction"],
                    _f(r["entry_price"]), _f(r["target"]), _f(r["stop"]),
                    _f(r["confidence"]),
                    _f(r["rr_ratio"]) or 0.0, _b(r["rr_qualifies"]),
                    _i(r["bars_held"]) or 0, r["status"] or "OPEN",
                    _f(r["exit_price"]), r["exit_reason"],
                    _f(r["pnl_pct"]), _f(r["pnl_dollar"]),
                    _i(r["shares"]) or 1,
                    r["session"] or "", r["regime"] or "",
                    r["vwap_event"] or "", r["rsi_zone"] or "", r["entry_type"] or "",
                    _b(r["t1_hit"]), _f(r["t1_price"]) or 0.0, _f(r["t2_price"]) or 0.0,
                    _b(r["breakeven_set"]),
                    _f(r["partial_pnl_dollar"]) or 0.0, _i(r["shares_remaining"]) or 0,
                    _f(r["order_flow_score"]) or 0.0, _f(r["size_mult"]) or 1.0,
                    json.dumps({"sqlite_id": _i(r["id"])}),
                )
                for r in rows
            ],
            page_size=500,
        )
    pg.commit()
    log.info(f"  ↳ trades inserted/refreshed: {len(rows):,}")
    return {"total": len(rows), "open": open_count, "closed": closed_count}


def sync_open_trades(sqlite_dir: Path, pg, dry_run: bool) -> dict:
    """Re-sync the live OPEN trades — call this just before app-code cutover."""
    log.info("=== SYNC OPEN TRADES (pre-cutover refresh) ===")
    db_path = sqlite_dir / "paper_trades.db"
    if not db_path.exists():
        log.warning("paper_trades.db not found — skipping")
        return {"skipped": True}

    with sqlite_ro(db_path) as sc:
        rows = sc.execute("""
            SELECT
                id, opened_at, closed_at, ticker, direction,
                entry_price, target, stop, confidence,
                rr_ratio, rr_qualifies, bars_held, status,
                exit_price, exit_reason, pnl_pct, pnl_dollar,
                shares, session, regime, vwap_event, rsi_zone, entry_type,
                t1_hit, t1_price, t2_price, breakeven_set,
                partial_pnl_dollar, shares_remaining,
                order_flow_score, size_mult
            FROM paper_trades
            WHERE status = 'OPEN'
            ORDER BY id
        """).fetchall()

    log.info(f"  Found {len(rows)} OPEN trades in SQLite to sync")
    if dry_run:
        for r in rows:
            log.info(f"    [DRY] would upsert: #{r['id']} {r['ticker']} {r['direction']} @ {r['entry_price']}")
        return {"open": len(rows), "dry_run": True}

    upserted = 0
    with pg.cursor() as cur:
        for r in rows:
            cur.execute("""
                UPDATE trades
                SET
                    bars_held          = %s,
                    status             = %s,
                    exit_price         = %s,
                    exit_reason        = %s,
                    pnl_pct            = %s,
                    pnl_dollar         = %s,
                    closed_at          = %s,
                    t1_hit             = %s,
                    partial_pnl_dollar = %s,
                    shares_remaining   = %s,
                    breakeven_set      = %s,
                    metadata           = metadata || %s::jsonb
                WHERE
                    metadata->>'sqlite_id' = %s
                    AND status = 'OPEN'
            """, (
                _i(r["bars_held"]) or 0,
                r["status"] or "OPEN",
                _f(r["exit_price"]), r["exit_reason"],
                _f(r["pnl_pct"]), _f(r["pnl_dollar"]),
                _ts(r["closed_at"]),
                _b(r["t1_hit"]),
                _f(r["partial_pnl_dollar"]) or 0.0,
                _i(r["shares_remaining"]) or 0,
                _b(r["breakeven_set"]),
                json.dumps({"last_synced": datetime.now(timezone.utc).isoformat()}),
                str(_i(r["id"])),
            ))
            upserted += cur.rowcount

    pg.commit()
    log.info(f"  ↳ open trades synced: {upserted}")
    return {"open": len(rows), "synced": upserted}


# ─── 3. live_backtest.db  →  signal_observations (source=LIVE) ───────────────

def migrate_live_backtest(sqlite_dir: Path, pg, dry_run: bool) -> dict:
    db_path = sqlite_dir / "live_backtest.db"
    if not db_path.exists():
        log.warning("live_backtest.db not found — skipping")
        return {"skipped": True}

    with sqlite_ro(db_path) as sc:
        rows = sc.execute("""
            SELECT
                signal_id, fired_at, ticker, direction,
                entry_price, target, stop, rr_ratio, confidence,
                session, regime, vwap_event, rsi_zone, rsi_value,
                sector_etf, sector_trend, entry_type, mtf_alignment,
                status, resolved_at, exit_price, exit_reason,
                bars_tracked, max_favorable_r, pnl_pct, r_multiple
            FROM bt_signals
            ORDER BY id
        """).fetchall()

    log.info(f"live_backtest.db: {len(rows):,} rows")
    if dry_run:
        return {"total": len(rows), "dry_run": True}

    with pg.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO signal_observations (
                source, signal_id, fired_at, ticker, direction,
                entry_price, target_price, stop_price, rr_ratio, confidence,
                session, regime, vwap_event, rsi_zone, rsi_value,
                sector_etf, sector_trend, entry_type, mtf_alignment,
                status, resolved_at, exit_price, exit_reason,
                bars_tracked, max_favorable_r, pnl_pct, r_multiple
            ) VALUES (
                'LIVE',%s,%s,%s,%s,
                %s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,%s,%s
            )
            ON CONFLICT (signal_id) DO NOTHING
            """,
            [
                (
                    r["signal_id"], _ts(r["fired_at"]), r["ticker"], r["direction"],
                    _f(r["entry_price"]), _f(r["target"]), _f(r["stop"]),
                    _f(r["rr_ratio"]) or 0.0, _f(r["confidence"]) or 0.0,
                    r["session"] or "", r["regime"] or "", r["vwap_event"] or "",
                    r["rsi_zone"] or "", _f(r["rsi_value"]) or 50.0,
                    r["sector_etf"] or "", r["sector_trend"] or "",
                    r["entry_type"] or "IMMEDIATE", r["mtf_alignment"] or "",
                    r["status"] or "TRACKING", _ts(r["resolved_at"]),
                    _f(r["exit_price"]), r["exit_reason"],
                    _i(r["bars_tracked"]) or 0, _f(r["max_favorable_r"]) or 0.0,
                    _f(r["pnl_pct"]), _f(r["r_multiple"]),
                )
                for r in rows
            ],
            page_size=500,
        )
    pg.commit()
    log.info(f"  ↳ signal_observations (LIVE) inserted: {len(rows):,}")
    return {"total": len(rows)}


# ─── 4. weekend_learning.db  →  signal_observations (source=WEEKEND) ─────────

def migrate_weekend_learning(sqlite_dir: Path, pg, dry_run: bool) -> dict:
    db_path = sqlite_dir / "weekend_learning.db"
    if not db_path.exists():
        log.warning("weekend_learning.db not found — skipping")
        return {"skipped": True}

    with sqlite_ro(db_path) as sc:
        rows = sc.execute("""
            SELECT
                id, weekend_dt, ticker, bar_dt, direction,
                entry_price, target, stop,
                exit_price, outcome, pnl_r, bars_held, won
            FROM signal_records
            ORDER BY id
        """).fetchall()

    log.info(f"weekend_learning.db: {len(rows):,} rows")
    if dry_run:
        return {"total": len(rows), "dry_run": True}

    with pg.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO signal_observations (
                source, signal_id, fired_at, ticker, direction,
                entry_price, target_price, stop_price,
                exit_price, status, r_multiple, bars_tracked, metadata
            ) VALUES (
                'WEEKEND',
                'WKND_' || %s || '_' || %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s, %s::jsonb
            )
            ON CONFLICT (signal_id) DO NOTHING
            """,
            [
                (
                    str(_i(r["id"])), r["ticker"] or "UNK",
                    _ts(r["bar_dt"]) or _ts(r["weekend_dt"]),
                    r["ticker"], r["direction"],
                    _f(r["entry_price"]), _f(r["target"]), _f(r["stop"]),
                    _f(r["exit_price"]),
                    r["outcome"] or ("WIN" if _b(r["won"]) else "LOSS"),
                    _f(r["pnl_r"]),
                    _i(r["bars_held"]) or 0,
                    json.dumps({"weekend_dt": r["weekend_dt"], "sqlite_id": _i(r["id"])}),
                )
                for r in rows
            ],
            page_size=500,
        )
    pg.commit()
    log.info(f"  ↳ signal_observations (WEEKEND) inserted: {len(rows):,}")
    return {"total": len(rows)}


# ─── 5. backtest_mtf.db  →  signal_observations (BACKTEST) + backtest_summary ─

def migrate_backtest_mtf(sqlite_dir: Path, pg, dry_run: bool) -> dict:
    db_path = sqlite_dir / "backtest_mtf.db"
    if not db_path.exists():
        log.warning("backtest_mtf.db not found — skipping")
        return {"skipped": True}

    with sqlite_ro(db_path) as sc:
        trades = sc.execute("""
            SELECT
                id, run_dt, timeframe, ticker, bar_dt, direction,
                entry_price, target, stop,
                exit_price, outcome, pnl_r, bars_held, won
            FROM bt_mtf_trades
            ORDER BY id
        """).fetchall()

        summaries = sc.execute("""
            SELECT run_dt, timeframe, ticker,
                   total, wins, win_rate, avg_pnl_r,
                   expectancy, max_dd_r, sharpe
            FROM bt_mtf_summary
            ORDER BY run_dt, timeframe, ticker
        """).fetchall()

    log.info(f"backtest_mtf.db: {len(trades):,} trades, {len(summaries):,} summaries")
    if dry_run:
        return {"trades": len(trades), "summaries": len(summaries), "dry_run": True}

    # Migrate trades → signal_observations
    with pg.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO signal_observations (
                source, signal_id, fired_at, ticker, direction,
                entry_price, target_price, stop_price,
                exit_price, status, r_multiple, bars_tracked, metadata
            ) VALUES (
                'BACKTEST',
                'BTF_' || %s || '_' || %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s, %s::jsonb
            )
            ON CONFLICT (signal_id) DO NOTHING
            """,
            [
                (
                    str(_i(r["id"])), r["ticker"] or "UNK",
                    _ts(r["bar_dt"]) or _ts(r["run_dt"]),
                    r["ticker"], r["direction"],
                    _f(r["entry_price"]), _f(r["target"]), _f(r["stop"]),
                    _f(r["exit_price"]),
                    r["outcome"] or ("WIN" if _b(r["won"]) else "LOSS"),
                    _f(r["pnl_r"]),
                    _i(r["bars_held"]) or 0,
                    json.dumps({
                        "run_dt": r["run_dt"],
                        "timeframe": r["timeframe"],
                        "sqlite_id": _i(r["id"]),
                    }),
                )
                for r in trades
            ],
            page_size=500,
        )

        # Migrate summaries → backtest_summary
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO backtest_summary (
                run_dt, timeframe, ticker,
                total, wins, win_rate, avg_pnl_r, expectancy, max_dd_r, sharpe
            ) VALUES (%s,%s,%s, %s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (run_dt, timeframe, ticker) DO UPDATE SET
                total      = EXCLUDED.total,
                wins       = EXCLUDED.wins,
                win_rate   = EXCLUDED.win_rate,
                avg_pnl_r  = EXCLUDED.avg_pnl_r,
                expectancy = EXCLUDED.expectancy,
                max_dd_r   = EXCLUDED.max_dd_r,
                sharpe     = EXCLUDED.sharpe
            """,
            [
                (
                    _ts(s["run_dt"]), s["timeframe"], s["ticker"],
                    _i(s["total"]) or 0, _i(s["wins"]) or 0,
                    _f(s["win_rate"]), _f(s["avg_pnl_r"]),
                    _f(s["expectancy"]), _f(s["max_dd_r"]), _f(s["sharpe"]),
                )
                for s in summaries
            ],
            page_size=200,
        )

    pg.commit()
    log.info(f"  ↳ signal_observations (BACKTEST): {len(trades):,}")
    log.info(f"  ↳ backtest_summary: {len(summaries):,}")
    return {"trades": len(trades), "summaries": len(summaries)}


# ─── Pre-flight checks ────────────────────────────────────────────────────────

REQUIRED_TABLES = [
    "signals", "trades", "signal_observations",
    "ticker_stats", "backtest_summary",
]

def preflight(pg, sqlite_dir: Path) -> bool:
    ok = True
    log.info("=== Pre-flight checks ===")

    # Check PostgreSQL tables exist
    with pg.cursor() as cur:
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        existing_tables = {r[0] for r in cur.fetchall()}

    for t in REQUIRED_TABLES:
        if t in existing_tables:
            log.info(f"  [OK] PostgreSQL table '{t}' exists")
        else:
            log.error(f"  [MISSING] PostgreSQL table '{t}' — run DDL additions first")
            ok = False

    # Check signal_observations.source column
    with pg.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'signal_observations' AND column_name = 'source'
        """)
        if cur.fetchone():
            log.info("  [OK] signal_observations.source column exists")
        else:
            log.error("  [MISSING] signal_observations.source — run DDL additions first")
            ok = False

    # Check trades.metadata column
    with pg.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'trades' AND column_name = 'metadata'
        """)
        if cur.fetchone():
            log.info("  [OK] trades.metadata column exists")
        else:
            log.error("  [MISSING] trades.metadata — run DDL additions first")
            ok = False

    # Check SQLite files
    for fname in ["signal_history.db", "paper_trades.db", "live_backtest.db",
                  "weekend_learning.db", "backtest_mtf.db"]:
        p = sqlite_dir / fname
        if p.exists():
            size_mb = p.stat().st_size / 1_048_576
            log.info(f"  [OK] {fname}  ({size_mb:.1f} MB)")
        else:
            log.warning(f"  [MISSING] {fname} (will be skipped)")

    return ok


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Migrate NASDAQ Agent SQLite → PostgreSQL")
    parser.add_argument("--pg-host",     default="localhost",     help="RDS endpoint")
    parser.add_argument("--pg-port",     type=int, default=5432,  help="PostgreSQL port")
    parser.add_argument("--pg-db",       default="trading",       help="Database name")
    parser.add_argument("--pg-user",     default="trading_admin", help="Database user")
    parser.add_argument("--pg-pass",     required=True,           help="Database password")
    parser.add_argument("--sqlite-dir",  default="/tmp/sqlite",   help="Directory with SQLite files")
    parser.add_argument("--dry-run",     action="store_true",     help="Read SQLite, print counts, no writes")
    parser.add_argument("--sync-open-trades", action="store_true",
                        help="Only re-sync the OPEN trades (pre-cutover refresh)")
    parser.add_argument("--skip-preflight", action="store_true",  help="Skip pre-flight table checks")
    args = parser.parse_args()

    sqlite_dir = Path(args.sqlite_dir)
    if not sqlite_dir.exists():
        log.error(f"SQLite directory not found: {sqlite_dir}")
        sys.exit(1)

    log.info(f"Connecting to PostgreSQL at {args.pg_host}:{args.pg_port}/{args.pg_db} …")
    try:
        pg = make_pg_conn(args.pg_host, args.pg_port, args.pg_db, args.pg_user, args.pg_pass)
        pg.autocommit = False
    except Exception as e:
        log.error(f"Cannot connect to PostgreSQL: {e}")
        sys.exit(1)
    log.info("  ↳ connected")

    try:
        if not args.skip_preflight:
            if not preflight(pg, sqlite_dir):
                log.error("Pre-flight failed — fix errors above before running migration")
                sys.exit(1)

        if args.dry_run:
            log.info("\n=== DRY RUN — no data will be written to PostgreSQL ===\n")

        if args.sync_open_trades:
            # Only re-sync OPEN trades; nothing else
            result = sync_open_trades(sqlite_dir, pg, args.dry_run)
            log.info(f"\nSync complete: {result}")
            return

        # Full migration
        log.info("\n=== Phase 1: Signals ===")
        r1 = migrate_signals(sqlite_dir, pg, args.dry_run)

        log.info("\n=== Phase 2: Paper Trades (closed + open snapshot) ===")
        r2 = migrate_trades(sqlite_dir, pg, args.dry_run)

        log.info("\n=== Phase 3: Live Backtest Observations ===")
        r3 = migrate_live_backtest(sqlite_dir, pg, args.dry_run)

        log.info("\n=== Phase 4: Weekend Learning Observations ===")
        r4 = migrate_weekend_learning(sqlite_dir, pg, args.dry_run)

        log.info("\n=== Phase 5: Multi-TF Backtest Observations + Summary ===")
        r5 = migrate_backtest_mtf(sqlite_dir, pg, args.dry_run)

        log.info("\n" + "=" * 60)
        log.info("MIGRATION COMPLETE")
        log.info("=" * 60)
        log.info(f"  signals              : {r1}")
        log.info(f"  trades               : {r2}")
        log.info(f"  observations (LIVE)  : {r3}")
        log.info(f"  observations (WEEKEND): {r4}")
        log.info(f"  observations (BACKTEST): {r5}")

        if not args.dry_run:
            log.info("""
NEXT STEPS
----------
1. Verify row counts in PostgreSQL:
     SELECT 'signals',             COUNT(*) FROM signals
     UNION ALL SELECT 'trades',    COUNT(*) FROM trades
     UNION ALL SELECT 'obs_live',  COUNT(*) FROM signal_observations WHERE source='LIVE'
     UNION ALL SELECT 'obs_wknd',  COUNT(*) FROM signal_observations WHERE source='WEEKEND'
     UNION ALL SELECT 'obs_btf',   COUNT(*) FROM signal_observations WHERE source='BACKTEST'
     UNION ALL SELECT 'bt_summ',   COUNT(*) FROM backtest_summary
     UNION ALL SELECT 'tickers',   COUNT(*) FROM ticker_stats;

2. Check the 14 open trades were migrated:
     SELECT id, ticker, direction, status, opened_at, metadata->>'sqlite_id' AS sqlite_id
     FROM trades WHERE status = 'OPEN' ORDER BY opened_at;

3. When ready to cut the app over to PostgreSQL:
     a. Deploy new app version with PostgreSQL connection
     b. Immediately before deploying, run:
            python3 migrate_sqlite_to_postgres.py --sync-open-trades \\
                --pg-host <HOST> --pg-pass <PASS>
     c. This final sync captures the last-known state of the 14 open trades
     d. The new app version takes over from PostgreSQL
""")

    except KeyboardInterrupt:
        log.warning("Interrupted — rolling back …")
        pg.rollback()
    except Exception as e:
        log.exception(f"Migration failed: {e}")
        pg.rollback()
        sys.exit(1)
    finally:
        pg.close()


if __name__ == "__main__":
    main()
