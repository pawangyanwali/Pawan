#!/usr/bin/env python3
"""
NASDAQ Agent — SQLite → PostgreSQL full migration  (v2 — schema-corrected)
===========================================================================

Actual PostgreSQL column layout (discovered from live DB):

  signals            : ticker, direction, confidence, price, tier, regime, session,
                       vol_bucket, trend, is_suppressed, short_outcome, indicators(JSONB), created_at
  trades             : trade_type, ticker, direction, status, entry_price, entry_at,
                       shares, entry_value, stop_loss, take_profit, exit_price, exit_at,
                       exit_reason, gross_pnl, commission, net_pnl, pnl_pct,
                       hold_minutes, created_at, updated_at, metadata(JSONB)
  signal_observations: signal_id, ticker, direction, entry_price, entry_at,
                       exit_price, exit_at, outcome, pnl_pct, hold_minutes,
                       resolved, created_at, source
  ticker_stats       : ticker, obs_count, win_count, loss_count, scratch_count,
                       win_rate, avg_pnl_pct, avg_hold_min, last_signal_at, updated_at
  backtest_summary   : run_dt, timeframe, ticker, total, wins, win_rate,
                       avg_pnl_r, expectancy, max_dd_r, sharpe, created_at

PREREQUISITE DDL (run once in pgAdmin / psql before running this script):
--------------------------------------------------------------------------
    -- Ensure source column exists on signal_observations
    ALTER TABLE signal_observations ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'LIVE';

    -- Unique constraint so re-runs are idempotent
    CREATE UNIQUE INDEX IF NOT EXISTS uq_signal_obs_signal_id
        ON signal_observations (signal_id);

    -- backtest_summary (may already exist)
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

USAGE
-----
Dry run:
    python3 migrate_sqlite_to_postgres.py --dry-run \\
        --pg-host nasdaq-agent.cozs0qe840sb.us-east-1.rds.amazonaws.com \\
        --pg-pass YOUR_PASSWORD --sqlite-dir /tmp/sqlite

Real run:
    python3 migrate_sqlite_to_postgres.py \\
        --pg-host nasdaq-agent.cozs0qe840sb.us-east-1.rds.amazonaws.com \\
        --pg-pass YOUR_PASSWORD --sqlite-dir /tmp/sqlite
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
    log.error("psycopg2 not installed.  Run: pip3 install psycopg2-binary")
    sys.exit(1)


# ─── Connection helpers ───────────────────────────────────────────────────────

@contextmanager
def sqlite_ro(path: Path):
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def make_pg_conn(host, port, db, user, password):
    return psycopg2.connect(
        host=host, port=port, dbname=db, user=user, password=password,
        connect_timeout=15,
        options="-c application_name=sqlite_migration",
    )


# ─── Type helpers ─────────────────────────────────────────────────────────────

def _f(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None

def _i(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None

def _b(v: Any) -> bool:
    if v is None:
        return False
    return bool(int(v)) if isinstance(v, (int, float)) else str(v).lower() in ('1','true','t','yes')

def _ts(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None

def _confidence(v: Any) -> float:
    """SQLite 0-100 → PostgreSQL NUMERIC(5,4) 0-1."""
    f = _f(v) or 0.0
    return round(f / 100.0, 4) if f > 1.0 else round(f, 4)

_SESSION_MAP = {
    'OPEN': 'REGULAR', 'MARKET': 'REGULAR', 'REGULAR': 'REGULAR',
    'PRE_MARKET': 'PRE_MARKET', 'PRE': 'PRE_MARKET', 'PREMARKET': 'PRE_MARKET',
    'AFTER_HOURS': 'AFTER_HOURS', 'AFTERHOURS': 'AFTER_HOURS',
    'AH': 'AFTER_HOURS', 'EXTENDED': 'AFTER_HOURS',
    'CLOSED': 'CLOSED',
}
def _session(v: Any) -> str:
    return _SESSION_MAP.get(str(v).upper().strip() if v else '', 'REGULAR')

_DIR_MAP = {'BUY': 'BUY', 'LONG': 'BUY', 'SELL': 'SELL', 'SHORT': 'SELL'}
def _direction(v: Any) -> Optional[str]:
    return _DIR_MAP.get(str(v).upper().strip() if v else '', None)

_TIER_MAP = {"HIGH": 1, "REGULAR": 2, "LOW": 3, "TIER1": 1, "TIER2": 2, "TIER3": 3}
def _tier(v: Any) -> int:
    if v is None:
        return 2
    try:
        t = int(v)
        return max(1, min(5, t))   # clamp to 1-5
    except (TypeError, ValueError):
        return _TIER_MAP.get(str(v).upper().strip(), 2)

# trades.exit_reason: only STOP_LOSS, TAKE_PROFIT, SIGNAL, EOD, MANUAL, TIMEOUT
_EXIT_MAP = {
    'TARGET_HIT': 'TAKE_PROFIT', 'T1_HIT': 'TAKE_PROFIT', 'T2_HIT': 'TAKE_PROFIT',
    'TP': 'TAKE_PROFIT', 'TAKE_PROFIT': 'TAKE_PROFIT', 'TARGET': 'TAKE_PROFIT',
    'STOP': 'STOP_LOSS', 'STOP_HIT': 'STOP_LOSS', 'SL': 'STOP_LOSS',
    'STOP_LOSS': 'STOP_LOSS', 'BREAKEVEN_STOP': 'STOP_LOSS', 'BE_STOP': 'STOP_LOSS',
    'STOPLOSS': 'STOP_LOSS',
    'TIME_STOP': 'TIMEOUT', 'TIME_EXIT': 'TIMEOUT', 'BARS_EXCEEDED': 'TIMEOUT',
    'TIMEOUT': 'TIMEOUT', 'TIME': 'TIMEOUT',
    'EOD': 'EOD', 'EOD_CLOSE': 'EOD', 'END_OF_DAY': 'EOD', 'HARD_CLOSE': 'EOD',
    'MANUAL': 'MANUAL', 'SIGNAL': 'SIGNAL',
}
def _exit_reason(v: Any) -> Optional[str]:
    if v is None:
        return None
    return _EXIT_MAP.get(str(v).upper().strip(), 'MANUAL')

# signal_observations.outcome: only WIN, LOSS, SCRATCH (or NULL)
_OUTCOME_MAP = {
    'WIN': 'WIN', 'PROFIT': 'WIN', 'TARGET': 'WIN', 'TARGET_HIT': 'WIN',
    'LOSS': 'LOSS', 'STOPPED': 'LOSS', 'STOP': 'LOSS',
    'SCRATCH': 'SCRATCH', 'BREAKEVEN': 'SCRATCH', 'BE': 'SCRATCH',
}
def _outcome(v: Any) -> Optional[str]:
    """Returns WIN/LOSS/SCRATCH or None — TRACKING/PENDING become NULL."""
    if v is None:
        return None
    return _OUTCOME_MAP.get(str(v).upper().strip(), None)

def _exit_after_entry(entry_ts: Optional[str], exit_ts: Optional[str]) -> Optional[str]:
    """Return exit_ts only if strictly after entry_ts, else NULL (satisfies chk_exit_after_entry)."""
    if not exit_ts or not entry_ts:
        return exit_ts
    return exit_ts if exit_ts > entry_ts else None



# ─── Pre-flight ───────────────────────────────────────────────────────────────

def preflight(pg, sqlite_dir: Path) -> bool:
    ok = True
    log.info("=== Pre-flight checks ===")
    with pg.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        existing = {r[0] for r in cur.fetchall()}
    for t in ['signals','trades','signal_observations','ticker_stats']:
        if t in existing:
            log.info(f"  [OK] table '{t}' exists")
        else:
            log.error(f"  [MISSING] table '{t}'")
            ok = False
    for fname in ["signal_history.db","paper_trades.db","live_backtest.db",
                  "weekend_learning.db","backtest_mtf.db"]:
        p = sqlite_dir / fname
        if p.exists():
            log.info(f"  [OK] {fname}  ({p.stat().st_size/1_048_576:.1f} MB)")
        else:
            log.warning(f"  [MISSING] {fname} — will be skipped")
    return ok


def _table_count(pg, table: str) -> int:
    with pg.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        return cur.fetchone()[0]


# ─── 1. signal_history.db → signals ──────────────────────────────────────────
#
#  SQLite           PostgreSQL
#  ───────────────  ─────────────────────────────────────────────────────────
#  ts               created_at
#  ticker           ticker
#  direction        direction
#  entry_price      price
#  confidence       confidence
#  trading_tier     tier
#  regime           regime
#  session          session
#  vol_bucket       vol_bucket
#  trend            trend
#  is_suppressed    is_suppressed
#  short_outcome    short_outcome
#  target/stop/vwap_event/rsi_zone/outcome/exit_price/pnl_pct/bars_held
#                   → indicators JSONB

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
            FROM signals ORDER BY id
        """).fetchall()

    log.info(f"signal_history.db: {len(rows):,} rows")

    existing = _table_count(pg, "signals")
    if existing > 0:
        log.warning(f"  signals table already has {existing:,} rows — skipping to avoid duplicates")
        log.warning("  To re-migrate: TRUNCATE signals CASCADE; then re-run.")
        return {"skipped_existing": existing}

    if dry_run:
        return {"total": len(rows), "dry_run": True}

    with pg.cursor() as cur:
        # Skip rows that would violate NOT NULL / CHECK constraints
        valid = [r for r in rows if _f(r["entry_price"]) and _f(r["entry_price"]) > 0
                                 and _direction(r["direction"])]
        skipped = len(rows) - len(valid)
        if skipped:
            log.warning(f"  Skipping {skipped} rows with null/zero price or invalid direction")

        psycopg2.extras.execute_batch(cur, """
            INSERT INTO signals (
                ticker, direction, confidence, price, tier,
                regime, session, vol_bucket, trend,
                is_suppressed, short_outcome, indicators, created_at
            ) VALUES (%s,%s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,%s)
        """, [
            (
                r["ticker"], _direction(r["direction"]),
                _confidence(r["confidence"]),
                _f(r["entry_price"]),
                _tier(r["trading_tier"]),
                r["regime"] or "", _session(r["session"]),
                r["vol_bucket"] or "NORMAL", r["trend"] or "",
                _b(r["is_suppressed"]),
                r["short_outcome"] or "PENDING",
                json.dumps({
                    "target":      _f(r["target"]),
                    "stop":        _f(r["stop"]),
                    "vwap_event":  r["vwap_event"] or "",
                    "rsi_zone":    r["rsi_zone"] or "",
                    "outcome":     r["outcome"] or "PENDING",
                    "exit_price":  _f(r["exit_price"]),
                    "pnl_pct":     _f(r["pnl_pct"]),
                    "bars_held":   _i(r["bars_held"]) or 0,
                    "sqlite_id":   _i(r["id"]),
                }),
                _ts(r["ts"]),
            )
            for r in valid
        ], page_size=500)
    pg.commit()

    # Rebuild ticker_stats from the migrated signals
    _rebuild_ticker_stats(pg)

    log.info(f"  ↳ signals inserted: {len(rows):,}")
    return {"total": len(rows)}


def _rebuild_ticker_stats(pg) -> None:
    log.info("Rebuilding ticker_stats …")
    with pg.cursor() as cur:
        cur.execute("""
            INSERT INTO ticker_stats (
                ticker, obs_count, win_count, loss_count, scratch_count,
                win_rate, avg_pnl_pct, avg_hold_min, last_signal_at, updated_at
            )
            SELECT
                ticker,
                COUNT(*)                                                              AS obs_count,
                COUNT(*) FILTER (WHERE indicators->>'outcome' = 'WIN')                AS win_count,
                COUNT(*) FILTER (WHERE indicators->>'outcome' = 'LOSS')               AS loss_count,
                COUNT(*) FILTER (WHERE indicators->>'outcome' NOT IN ('WIN','LOSS')
                                   AND indicators->>'outcome' IS NOT NULL)             AS scratch_count,
                ROUND(
                    COUNT(*) FILTER (WHERE indicators->>'outcome' = 'WIN')::NUMERIC
                    / NULLIF(
                        COUNT(*) FILTER (WHERE indicators->>'outcome' IN ('WIN','LOSS')),
                        0),
                    4)                                                                AS win_rate,
                ROUND(AVG((indicators->>'pnl_pct')::NUMERIC)
                      FILTER (WHERE indicators->>'outcome' IN ('WIN','LOSS')), 4)     AS avg_pnl_pct,
                0                                                                     AS avg_hold_min,
                MAX(created_at)                                                       AS last_signal_at,
                NOW()
            FROM signals
            GROUP BY ticker
            ON CONFLICT (ticker) DO UPDATE SET
                obs_count      = EXCLUDED.obs_count,
                win_count      = EXCLUDED.win_count,
                loss_count     = EXCLUDED.loss_count,
                scratch_count  = EXCLUDED.scratch_count,
                win_rate       = EXCLUDED.win_rate,
                avg_pnl_pct    = EXCLUDED.avg_pnl_pct,
                last_signal_at = EXCLUDED.last_signal_at,
                updated_at     = NOW()
        """)
    pg.commit()
    log.info("  ↳ ticker_stats rebuilt")


# ─── 2. paper_trades.db → trades ─────────────────────────────────────────────
#
#  SQLite           PostgreSQL
#  ───────────────  ─────────────────────────────────────────────────────────
#  opened_at        entry_at, created_at
#  closed_at        exit_at, updated_at
#  ticker           ticker
#  direction        direction
#  status           status
#  entry_price      entry_price
#  target           take_profit
#  stop             stop_loss
#  exit_price       exit_price
#  exit_reason      exit_reason
#  pnl_pct          pnl_pct
#  pnl_dollar       gross_pnl, net_pnl
#  shares           shares
#  bars_held        hold_minutes (1 bar ≈ 1 min)
#  trade_type       'PAPER' (constant)
#  commission       0 (paper)
#  entry_value      entry_price * shares
#  all others       → metadata JSONB

def migrate_trades(sqlite_dir: Path, pg, dry_run: bool) -> dict:
    db_path = sqlite_dir / "paper_trades.db"
    if not db_path.exists():
        log.warning("paper_trades.db not found — skipping")
        return {"skipped": True}

    with sqlite_ro(db_path) as sc:
        rows = sc.execute("""
            SELECT id, opened_at, closed_at, ticker, direction,
                   entry_price, target, stop, confidence,
                   rr_ratio, rr_qualifies, bars_held, status,
                   exit_price, exit_reason, pnl_pct, pnl_dollar,
                   shares, session, regime, vwap_event, rsi_zone, entry_type,
                   t1_hit, t1_price, t2_price, breakeven_set,
                   partial_pnl_dollar, shares_remaining,
                   order_flow_score, size_mult
            FROM paper_trades ORDER BY id
        """).fetchall()

    open_count   = sum(1 for r in rows if r["status"] == "OPEN")
    closed_count = len(rows) - open_count
    log.info(f"paper_trades.db: {len(rows):,} rows  ({open_count} OPEN · {closed_count} CLOSED)")

    existing = _table_count(pg, "trades")
    if existing > 0:
        log.warning(f"  trades table already has {existing:,} rows — skipping to avoid duplicates")
        log.warning("  To re-migrate: TRUNCATE trades CASCADE; then re-run.")
        return {"skipped_existing": existing}

    if dry_run:
        return {"total": len(rows), "open": open_count, "closed": closed_count, "dry_run": True}

    with pg.cursor() as cur:
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO trades (
                trade_type, ticker, direction, status,
                entry_price, entry_at, shares, entry_value,
                stop_loss, take_profit,
                exit_price, exit_at, exit_reason,
                gross_pnl, commission, net_pnl, pnl_pct,
                hold_minutes, created_at, updated_at, metadata
            ) VALUES (
                'PAPER',%s,%s,%s,
                %s,%s,%s,%s,
                %s,%s,
                %s,%s,%s,
                %s,0,%s,%s,
                %s,%s,%s,%s
            )
        """, [
            (
                r["ticker"],
                _direction(r["direction"]) or "BUY",
                r["status"] or "CLOSED",
                _f(r["entry_price"]),
                _ts(r["opened_at"]),
                max(_i(r["shares"]) or 1, 1),
                (_f(r["entry_price"]) or 0) * max(_i(r["shares"]) or 1, 1),
                _f(r["stop"]), _f(r["target"]),
                _f(r["exit_price"]),
                _exit_after_entry(_ts(r["opened_at"]), _ts(r["closed_at"])),
                _exit_reason(r["exit_reason"]),
                _f(r["pnl_dollar"]),
                _f(r["pnl_dollar"]),
                _f(r["pnl_pct"]),
                max(_i(r["bars_held"]) or 0, 0),
                _ts(r["opened_at"]),
                _ts(r["closed_at"]) or _ts(r["opened_at"]),
                json.dumps({
                    "sqlite_id":        _i(r["id"]),
                    "confidence":       _confidence(r["confidence"]),
                    "rr_ratio":         _f(r["rr_ratio"]),
                    "rr_qualifies":     _b(r["rr_qualifies"]),
                    "session":          r["session"] or "",
                    "regime":           r["regime"] or "",
                    "vwap_event":       r["vwap_event"] or "",
                    "rsi_zone":         r["rsi_zone"] or "",
                    "entry_type":       r["entry_type"] or "",
                    "t1_hit":           _b(r["t1_hit"]),
                    "t1_price":         _f(r["t1_price"]),
                    "t2_price":         _f(r["t2_price"]),
                    "breakeven_set":    _b(r["breakeven_set"]),
                    "partial_pnl_dollar": _f(r["partial_pnl_dollar"]),
                    "shares_remaining": _i(r["shares_remaining"]),
                    "order_flow_score": _f(r["order_flow_score"]),
                    "size_mult":        _f(r["size_mult"]),
                }),
            )
            for r in rows
        ], page_size=500)
    pg.commit()
    log.info(f"  ↳ trades inserted: {len(rows):,}")
    return {"total": len(rows), "open": open_count, "closed": closed_count}


# ─── 3. live_backtest.db → signal_observations (LIVE) ────────────────────────
#
#  SQLite           PostgreSQL
#  ───────────────  ─────────────────────────────────────────────────────────
#  signal_id        signal_id  (already unique string)
#  fired_at         entry_at, created_at
#  ticker           ticker
#  direction        direction
#  entry_price      entry_price
#  exit_price       exit_price
#  resolved_at      exit_at
#  status→outcome   outcome  (WIN/LOSS/SCRATCH/PENDING)
#  pnl_pct          pnl_pct
#  bars_tracked     hold_minutes
#  resolved_at!=NULL resolved (bool)
#  source           'LIVE'

def migrate_live_backtest(sqlite_dir: Path, pg, dry_run: bool) -> dict:
    db_path = sqlite_dir / "live_backtest.db"
    if not db_path.exists():
        log.warning("live_backtest.db not found — skipping")
        return {"skipped": True}

    with sqlite_ro(db_path) as sc:
        rows = sc.execute("""
            SELECT signal_id, fired_at, ticker, direction,
                   entry_price, exit_price, resolved_at,
                   status, pnl_pct, bars_tracked
            FROM bt_signals ORDER BY id
        """).fetchall()

    log.info(f"live_backtest.db: {len(rows):,} rows")

    existing = _table_count(pg, "signal_observations")
    if existing > 0:
        log.warning(f"  signal_observations already has {existing:,} rows — skipping")
        log.warning("  To re-migrate: TRUNCATE signal_observations; then re-run.")
        return {"skipped_existing": existing}

    if dry_run:
        return {"total": len(rows), "dry_run": True}

    # signal_id is a FK→signals.id (INTEGER) — we have no matching PG signal IDs,
    # so insert with signal_id=NULL. Filter rows violating NOT NULL / CHECK constraints.
    valid = [r for r in rows
             if _f(r["entry_price"]) and _f(r["entry_price"]) > 0
             and _direction(r["direction"])
             and _ts(r["fired_at"])]
    skipped = len(rows) - len(valid)
    if skipped:
        log.warning(f"  Skipping {skipped} rows with null price/direction/entry_at")

    with pg.cursor() as cur:
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO signal_observations (
                ticker, direction,
                entry_price, entry_at,
                exit_price, exit_at,
                outcome, pnl_pct, hold_minutes,
                resolved, created_at, source
            ) VALUES (%s,%s, %s,%s, %s,%s, %s,%s,%s, %s,%s,'LIVE')
        """, [
            (
                r["ticker"], _direction(r["direction"]),
                _f(r["entry_price"]), _ts(r["fired_at"]),
                _f(r["exit_price"]),
                _exit_after_entry(_ts(r["fired_at"]), _ts(r["resolved_at"])),
                _outcome(r["status"]),
                _f(r["pnl_pct"]),
                max(_i(r["bars_tracked"]) or 0, 0),
                r["resolved_at"] is not None,
                _ts(r["fired_at"]),
            )
            for r in valid
        ], page_size=500)
    pg.commit()
    log.info(f"  ↳ signal_observations (LIVE) inserted: {len(rows):,}")
    return {"total": len(rows)}


# ─── 4. weekend_learning.db → signal_observations (WEEKEND) ──────────────────

def migrate_weekend_learning(sqlite_dir: Path, pg, dry_run: bool) -> dict:
    db_path = sqlite_dir / "weekend_learning.db"
    if not db_path.exists():
        log.warning("weekend_learning.db not found — skipping")
        return {"skipped": True}

    with sqlite_ro(db_path) as sc:
        rows = sc.execute("""
            SELECT id, weekend_dt, ticker, bar_dt, direction,
                   entry_price, exit_price, outcome, pnl_r, bars_held, won
            FROM signal_records ORDER BY id
        """).fetchall()

    log.info(f"weekend_learning.db: {len(rows):,} rows")
    if not rows:
        return {"total": 0}
    if dry_run:
        return {"total": len(rows), "dry_run": True}

    with pg.cursor() as cur:
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO signal_observations (
                ticker, direction,
                entry_price, entry_at,
                exit_price, exit_at,
                outcome, pnl_pct, hold_minutes,
                resolved, created_at, source
            ) VALUES (%s,%s, %s,%s, %s,%s, %s,%s,%s, %s,%s,'WEEKEND')
        """, [
            (
                r["ticker"], _direction(r["direction"]) or "BUY",
                _f(r["entry_price"]),
                _ts(r["bar_dt"]) or _ts(r["weekend_dt"]),
                _f(r["exit_price"]),
                _exit_after_entry(
                    _ts(r["bar_dt"]) or _ts(r["weekend_dt"]),
                    _ts(r["bar_dt"]) or _ts(r["weekend_dt"]),
                ),
                _outcome(r["outcome"]) or ("WIN" if _b(r["won"]) else "LOSS"),
                _f(r["pnl_r"]),
                max(_i(r["bars_held"]) or 0, 0),
                True,
                _ts(r["bar_dt"]) or _ts(r["weekend_dt"]),
            )
            for r in rows
        ], page_size=500)
    pg.commit()
    log.info(f"  ↳ signal_observations (WEEKEND) inserted: {len(rows):,}")
    return {"total": len(rows)}


# ─── 5. backtest_mtf.db → signal_observations (BACKTEST) + backtest_summary ──

def migrate_backtest_mtf(sqlite_dir: Path, pg, dry_run: bool) -> dict:
    db_path = sqlite_dir / "backtest_mtf.db"
    if not db_path.exists():
        log.warning("backtest_mtf.db not found — skipping")
        return {"skipped": True}

    with sqlite_ro(db_path) as sc:
        trades = sc.execute("""
            SELECT id, run_dt, timeframe, ticker, bar_dt, direction,
                   entry_price, exit_price, outcome, pnl_r, bars_held, won
            FROM bt_mtf_trades ORDER BY id
        """).fetchall()
        summaries = sc.execute("""
            SELECT run_dt, timeframe, ticker, total, wins,
                   win_rate, avg_pnl_r, expectancy, max_dd_r, sharpe
            FROM bt_mtf_summary ORDER BY run_dt, timeframe, ticker
        """).fetchall()

    log.info(f"backtest_mtf.db: {len(trades):,} trades, {len(summaries):,} summaries")
    if not trades and not summaries:
        return {"trades": 0, "summaries": 0}
    if dry_run:
        return {"trades": len(trades), "summaries": len(summaries), "dry_run": True}

    with pg.cursor() as cur:
        if trades:
            psycopg2.extras.execute_batch(cur, """
                INSERT INTO signal_observations (
                    ticker, direction,
                    entry_price, entry_at,
                    exit_price, exit_at,
                    outcome, pnl_pct, hold_minutes,
                    resolved, created_at, source
                ) VALUES (%s,%s, %s,%s, %s,%s, %s,%s,%s, %s,%s,'BACKTEST')
            """, [
                (
                    r["ticker"], _direction(r["direction"]) or "BUY",
                    _f(r["entry_price"]),
                    _ts(r["bar_dt"]) or _ts(r["run_dt"]),
                    _f(r["exit_price"]),
                    _exit_after_entry(
                        _ts(r["bar_dt"]) or _ts(r["run_dt"]),
                        _ts(r["bar_dt"]) or _ts(r["run_dt"]),
                    ),
                    _outcome(r["outcome"]) or ("WIN" if _b(r["won"]) else "LOSS"),
                    _f(r["pnl_r"]),
                    max(_i(r["bars_held"]) or 0, 0),
                    True,
                    _ts(r["bar_dt"]) or _ts(r["run_dt"]),
                )
                for r in trades
            ], page_size=500)

        if summaries:
            psycopg2.extras.execute_batch(cur, """
                INSERT INTO backtest_summary (
                    run_dt, timeframe, ticker, total, wins,
                    win_rate, avg_pnl_r, expectancy, max_dd_r, sharpe
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (run_dt, timeframe, ticker) DO UPDATE SET
                    total      = EXCLUDED.total,
                    wins       = EXCLUDED.wins,
                    win_rate   = EXCLUDED.win_rate,
                    avg_pnl_r  = EXCLUDED.avg_pnl_r,
                    expectancy = EXCLUDED.expectancy,
                    max_dd_r   = EXCLUDED.max_dd_r,
                    sharpe     = EXCLUDED.sharpe
            """, [
                (
                    _ts(s["run_dt"]), s["timeframe"], s["ticker"],
                    _i(s["total"]) or 0, _i(s["wins"]) or 0,
                    _f(s["win_rate"]), _f(s["avg_pnl_r"]),
                    _f(s["expectancy"]), _f(s["max_dd_r"]), _f(s["sharpe"]),
                )
                for s in summaries
            ], page_size=200)

    pg.commit()
    log.info(f"  ↳ signal_observations (BACKTEST): {len(trades):,}")
    log.info(f"  ↳ backtest_summary: {len(summaries):,}")
    return {"trades": len(trades), "summaries": len(summaries)}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Migrate NASDAQ Agent SQLite → PostgreSQL")
    ap.add_argument("--pg-host",     default="localhost")
    ap.add_argument("--pg-port",     type=int, default=5432)
    ap.add_argument("--pg-db",       default="trading")
    ap.add_argument("--pg-user",     default="trading_admin")
    ap.add_argument("--pg-pass",     required=True)
    ap.add_argument("--sqlite-dir",  default="/tmp/sqlite")
    ap.add_argument("--dry-run",     action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    sqlite_dir = Path(args.sqlite_dir)
    if not sqlite_dir.exists():
        log.error(f"SQLite directory not found: {sqlite_dir}")
        sys.exit(1)

    log.info(f"Connecting to PostgreSQL at {args.pg_host}:{args.pg_port}/{args.pg_db} …")
    try:
        pg = make_pg_conn(args.pg_host, args.pg_port, args.pg_db, args.pg_user, args.pg_pass)
        pg.autocommit = False
    except Exception as e:
        log.error(f"Cannot connect: {e}")
        sys.exit(1)
    log.info("  ↳ connected")

    try:
        if not args.skip_preflight and not preflight(pg, sqlite_dir):
            sys.exit(1)

        if args.dry_run:
            log.info("\n=== DRY RUN — no data will be written ===\n")

        log.info("\n=== Phase 1: Signals ===")
        r1 = migrate_signals(sqlite_dir, pg, args.dry_run)

        log.info("\n=== Phase 2: Paper Trades ===")
        r2 = migrate_trades(sqlite_dir, pg, args.dry_run)

        log.info("\n=== Phase 3: Live Backtest Observations ===")
        r3 = migrate_live_backtest(sqlite_dir, pg, args.dry_run)

        log.info("\n=== Phase 4: Weekend Learning ===")
        r4 = migrate_weekend_learning(sqlite_dir, pg, args.dry_run)

        log.info("\n=== Phase 5: Multi-TF Backtest ===")
        r5 = migrate_backtest_mtf(sqlite_dir, pg, args.dry_run)

        log.info("\n" + "="*60)
        log.info("MIGRATION COMPLETE")
        log.info("="*60)
        log.info(f"  signals              : {r1}")
        log.info(f"  trades               : {r2}")
        log.info(f"  observations (LIVE)  : {r3}")
        log.info(f"  observations (WEEKEND): {r4}")
        log.info(f"  observations (BACKTEST): {r5}")

        if not args.dry_run:
            log.info("""
Verify in PostgreSQL:
  SELECT 'signals',  COUNT(*) FROM signals
  UNION ALL SELECT 'trades',         COUNT(*) FROM trades
  UNION ALL SELECT 'obs_live',       COUNT(*) FROM signal_observations WHERE source='LIVE'
  UNION ALL SELECT 'obs_weekend',    COUNT(*) FROM signal_observations WHERE source='WEEKEND'
  UNION ALL SELECT 'obs_backtest',   COUNT(*) FROM signal_observations WHERE source='BACKTEST'
  UNION ALL SELECT 'ticker_stats',   COUNT(*) FROM ticker_stats;
""")

    except KeyboardInterrupt:
        log.warning("Interrupted — rolling back")
        pg.rollback()
    except Exception as e:
        log.exception(f"Migration failed: {e}")
        pg.rollback()
        sys.exit(1)
    finally:
        pg.close()


if __name__ == "__main__":
    main()
