"""
Paper broker — order lifecycle recording and execution attribution.

Responsibilities:
  - DDL for paper_orders, paper_fills, trade_execution_attribution tables
  - record_fill(): write a FillResult to paper_fills
  - record_attribution(): write strategy vs actual P&L breakdown
  - init_execution_tables(): idempotent table creation at startup

Fill detection and price determination stay in paper_trading.py.
This module is the recording/attribution layer only.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ── DDL ───────────────────────────────────────────────────────────────────────

_DDL_ORDERS_PG = """
CREATE TABLE IF NOT EXISTS paper_orders (
    id              SERIAL PRIMARY KEY,
    trade_id        INTEGER,
    ticker          TEXT NOT NULL,
    side            TEXT NOT NULL,
    order_type      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'PENDING',
    order_role      TEXT NOT NULL,
    requested_price DOUBLE PRECISION NOT NULL,
    limit_price     DOUBLE PRECISION,
    stop_price      DOUBLE PRECISION,
    quantity        INTEGER NOT NULL,
    filled_quantity INTEGER DEFAULT 0,
    avg_fill_price  DOUBLE PRECISION,
    created_at      TEXT NOT NULL,
    updated_at      TEXT
)"""

_DDL_ORDERS_SQLITE = """
CREATE TABLE IF NOT EXISTS paper_orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id        INTEGER,
    ticker          TEXT NOT NULL,
    side            TEXT NOT NULL,
    order_type      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'PENDING',
    order_role      TEXT NOT NULL,
    requested_price REAL NOT NULL,
    limit_price     REAL,
    stop_price      REAL,
    quantity        INTEGER NOT NULL,
    filled_quantity INTEGER DEFAULT 0,
    avg_fill_price  REAL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT
)"""

_DDL_FILLS_PG = """
CREATE TABLE IF NOT EXISTS paper_fills (
    id              SERIAL PRIMARY KEY,
    order_id        INTEGER,
    trade_id        INTEGER,
    ticker          TEXT NOT NULL,
    side            TEXT NOT NULL,
    fill_price      DOUBLE PRECISION NOT NULL,
    quantity        INTEGER NOT NULL,
    ideal_price     DOUBLE PRECISION,
    spread_bps      DOUBLE PRECISION,
    slippage_bps    DOUBLE PRECISION,
    slippage_dollar DOUBLE PRECISION,
    spread_dollar   DOUBLE PRECISION,
    liquidity_score DOUBLE PRECISION,
    session         TEXT NOT NULL,
    fill_type       TEXT NOT NULL,
    filled_at       TEXT NOT NULL
)"""

_DDL_FILLS_SQLITE = """
CREATE TABLE IF NOT EXISTS paper_fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id        INTEGER,
    trade_id        INTEGER,
    ticker          TEXT NOT NULL,
    side            TEXT NOT NULL,
    fill_price      REAL NOT NULL,
    quantity        INTEGER NOT NULL,
    ideal_price     REAL,
    spread_bps      REAL,
    slippage_bps    REAL,
    slippage_dollar REAL,
    spread_dollar   REAL,
    liquidity_score REAL,
    session         TEXT NOT NULL,
    fill_type       TEXT NOT NULL,
    filled_at       TEXT NOT NULL
)"""

_DDL_ATTRIBUTION_PG = """
CREATE TABLE IF NOT EXISTS trade_execution_attribution (
    trade_id          INTEGER PRIMARY KEY,
    ticker            TEXT,
    direction         TEXT,
    session           TEXT,
    ideal_entry       DOUBLE PRECISION,
    actual_entry      DOUBLE PRECISION,
    ideal_exit        DOUBLE PRECISION,
    actual_exit       DOUBLE PRECISION,
    shares            INTEGER,
    strategy_pnl      DOUBLE PRECISION,
    actual_pnl        DOUBLE PRECISION,
    slippage_pnl      DOUBLE PRECISION,
    entry_slip_bps    DOUBLE PRECISION,
    exit_slip_bps     DOUBLE PRECISION,
    entry_spread_usd  DOUBLE PRECISION,
    signal_quality    TEXT,
    attribution_note  TEXT,
    created_at        TEXT NOT NULL
)"""

_DDL_ATTRIBUTION_SQLITE = """
CREATE TABLE IF NOT EXISTS trade_execution_attribution (
    trade_id          INTEGER PRIMARY KEY,
    ticker            TEXT,
    direction         TEXT,
    session           TEXT,
    ideal_entry       REAL,
    actual_entry      REAL,
    ideal_exit        REAL,
    actual_exit       REAL,
    shares            INTEGER,
    strategy_pnl      REAL,
    actual_pnl        REAL,
    slippage_pnl      REAL,
    entry_slip_bps    REAL,
    exit_slip_bps     REAL,
    entry_spread_usd  REAL,
    signal_quality    TEXT,
    attribution_note  TEXT,
    created_at        TEXT NOT NULL
)"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_postgres() -> bool:
    try:
        from agent.db import using_postgres
        return using_postgres()
    except Exception:
        return False


def init_execution_tables() -> None:
    """Create paper_orders, paper_fills, trade_execution_attribution (idempotent)."""
    try:
        from agent.db import get_conn
        from agent.paper_trading import _run_ddl_autocommit
        if _is_postgres():
            _run_ddl_autocommit([_DDL_ORDERS_PG, _DDL_FILLS_PG, _DDL_ATTRIBUTION_PG])
        else:
            with get_conn() as c:
                c.execute(_DDL_ORDERS_SQLITE)
                c.execute(_DDL_FILLS_SQLITE)
                c.execute(_DDL_ATTRIBUTION_SQLITE)
    except Exception as e:
        logger.warning("[PaperBroker] init_execution_tables failed: %s", e)


def record_fill(
    c,
    trade_id: int,
    ticker: str,
    direction: str,
    shares: int,
    fill_result,                        # FillResult from fill_model
    order_id: Optional[int] = None,
) -> int:
    """Insert a row into paper_fills. Returns fill id (0 on error)."""
    # Determine the side of this fill relative to the market
    if fill_result.fill_type == "ENTRY":
        side = direction
    else:
        # Exit fills invert direction (selling a long, buying back a short)
        side = "SELL" if direction == "BUY" else "BUY"

    try:
        cur = c.execute("""
            INSERT INTO paper_fills
              (order_id, trade_id, ticker, side,
               fill_price, quantity, ideal_price,
               spread_bps, slippage_bps, slippage_dollar, spread_dollar,
               liquidity_score, session, fill_type, filled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order_id, trade_id, ticker, side,
            round(fill_result.fill_price,  4),
            shares,
            round(fill_result.ideal_price, 4),
            round(fill_result.spread_bps,      2),
            round(fill_result.slippage_bps,    2),
            round(fill_result.slippage_dollar, 4),
            round(fill_result.spread_dollar,   4),
            round(fill_result.liquidity_score, 3),
            fill_result.session,
            fill_result.fill_type,
            _now_iso(),
        ))
        return getattr(cur, "lastrowid", 0) or 0
    except Exception as e:
        logger.warning("[PaperBroker] record_fill error: %s", e)
        return 0


def record_attribution(
    c,
    trade_id: int,
    ticker: str,
    direction: str,
    session: str,
    ideal_entry: float,
    actual_entry: float,
    ideal_exit: float,
    actual_exit: float,
    shares: int,
    actual_pnl: float,
    entry_slip_bps: float,
    exit_slip_bps: float,
    entry_spread_usd: float,
    attribution_note: str = "",
) -> None:
    """
    Compute and store strategy P&L vs actual P&L.

    strategy_pnl = what the trade would have earned at ideal prices (no slippage).
    actual_pnl   = what the trade actually earned (with slippage).
    slippage_pnl = actual_pnl − strategy_pnl  (negative = execution hurt us).

    signal_quality labels:
      GOOD_SIGNAL_GOOD_EXEC  — strategy correct AND execution good
      GOOD_SIGNAL_BAD_EXEC   — strategy correct but execution ate the profit
      BAD_SIGNAL_GOOD_EXEC   — strategy wrong but favorable fill salvaged it
      BAD_SIGNAL_BAD_EXEC    — both strategy and execution failed
    """
    if direction == "BUY":
        strategy_pnl = (ideal_exit - ideal_entry) * shares
    else:
        strategy_pnl = (ideal_entry - ideal_exit) * shares

    slippage_pnl = actual_pnl - strategy_pnl

    if strategy_pnl > 0 and actual_pnl > 0:
        signal_quality = "GOOD_SIGNAL_GOOD_EXEC"
    elif strategy_pnl > 0 and actual_pnl <= 0:
        signal_quality = "GOOD_SIGNAL_BAD_EXEC"
    elif strategy_pnl <= 0 and actual_pnl > 0:
        signal_quality = "BAD_SIGNAL_GOOD_EXEC"
    else:
        signal_quality = "BAD_SIGNAL_BAD_EXEC"

    try:
        c.execute("""
            INSERT INTO trade_execution_attribution
              (trade_id, ticker, direction, session,
               ideal_entry, actual_entry, ideal_exit, actual_exit, shares,
               strategy_pnl, actual_pnl, slippage_pnl,
               entry_slip_bps, exit_slip_bps, entry_spread_usd,
               signal_quality, attribution_note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (trade_id) DO UPDATE SET
               ideal_exit     = excluded.ideal_exit,
               actual_exit    = excluded.actual_exit,
               strategy_pnl   = excluded.strategy_pnl,
               actual_pnl     = excluded.actual_pnl,
               slippage_pnl   = excluded.slippage_pnl,
               exit_slip_bps  = excluded.exit_slip_bps,
               signal_quality = excluded.signal_quality,
               attribution_note = excluded.attribution_note
        """, (
            trade_id, ticker, direction, session,
            round(ideal_entry,  4), round(actual_entry, 4),
            round(ideal_exit,   4), round(actual_exit,  4),
            shares,
            round(strategy_pnl,  2), round(actual_pnl,    2),
            round(slippage_pnl,  2),
            round(entry_slip_bps, 2), round(exit_slip_bps, 2),
            round(entry_spread_usd, 4),
            signal_quality, attribution_note, _now_iso(),
        ))
    except Exception as e:
        logger.warning("[PaperBroker] record_attribution error: %s", e)


def get_daily_execution_quality() -> dict:
    """
    Query today's execution metrics from paper_fills + trade_execution_attribution.
    Returns a dict suitable for the dashboard API.
    """
    try:
        from agent.db import get_conn
        with get_conn() as c:
            today_iso = datetime.now(timezone.utc).date().isoformat()

            # Slippage stats from paper_fills
            fills_row = c.execute("""
                SELECT
                    COUNT(*)                                  AS fill_count,
                    COALESCE(AVG(spread_bps),   0)            AS avg_spread_bps,
                    COALESCE(AVG(slippage_bps), 0)            AS avg_slip_bps,
                    COALESCE(SUM(slippage_dollar), 0)         AS total_slip_usd,
                    COALESCE(SUM(spread_dollar),   0)         AS total_spread_usd,
                    COALESCE(AVG(liquidity_score), 1)         AS avg_liq_score
                FROM paper_fills
                WHERE filled_at >= ?
                  AND fill_type IN ('ENTRY','STOP_MARKET','TIME_STOP')
            """, (today_iso,)).fetchone()

            stop_row = c.execute("""
                SELECT
                    COUNT(*)                         AS stop_fills,
                    COALESCE(AVG(slippage_bps), 0)   AS avg_stop_slip_bps,
                    COALESCE(SUM(slippage_dollar), 0) AS total_stop_slip_usd
                FROM paper_fills
                WHERE filled_at >= ? AND fill_type = 'STOP_MARKET'
            """, (today_iso,)).fetchone()

            # Attribution stats
            attr_row = c.execute("""
                SELECT
                    COUNT(*)                           AS closed_trades,
                    COALESCE(SUM(strategy_pnl), 0)     AS strategy_pnl,
                    COALESCE(SUM(actual_pnl),   0)     AS actual_pnl,
                    COALESCE(SUM(slippage_pnl), 0)     AS slippage_pnl,
                    COUNT(CASE WHEN signal_quality='GOOD_SIGNAL_GOOD_EXEC' THEN 1 END) AS good_good,
                    COUNT(CASE WHEN signal_quality='GOOD_SIGNAL_BAD_EXEC'  THEN 1 END) AS good_bad,
                    COUNT(CASE WHEN signal_quality='BAD_SIGNAL_GOOD_EXEC'  THEN 1 END) AS bad_good,
                    COUNT(CASE WHEN signal_quality='BAD_SIGNAL_BAD_EXEC'   THEN 1 END) AS bad_bad
                FROM trade_execution_attribution
                WHERE created_at >= ?
            """, (today_iso,)).fetchone()

        def _f(v): return round(float(v or 0), 2)

        return {
            "fill_count":          int(fills_row["fill_count"]   or 0),
            "avg_spread_bps":      _f(fills_row["avg_spread_bps"]),
            "avg_slip_bps":        _f(fills_row["avg_slip_bps"]),
            "total_slip_usd":      _f(fills_row["total_slip_usd"]),
            "total_spread_usd":    _f(fills_row["total_spread_usd"]),
            "avg_liq_score":       _f(fills_row["avg_liq_score"]),
            "stop_fills":          int(stop_row["stop_fills"]     or 0),
            "avg_stop_slip_bps":   _f(stop_row["avg_stop_slip_bps"]),
            "total_stop_slip_usd": _f(stop_row["total_stop_slip_usd"]),
            "strategy_pnl":        _f(attr_row["strategy_pnl"]),
            "actual_pnl":          _f(attr_row["actual_pnl"]),
            "slippage_pnl":        _f(attr_row["slippage_pnl"]),
            "signal_quality": {
                "good_good": int(attr_row["good_good"] or 0),
                "good_bad":  int(attr_row["good_bad"]  or 0),
                "bad_good":  int(attr_row["bad_good"]  or 0),
                "bad_bad":   int(attr_row["bad_bad"]   or 0),
            },
        }
    except Exception as e:
        logger.warning("[PaperBroker] get_daily_execution_quality error: %s", e)
        return {}
