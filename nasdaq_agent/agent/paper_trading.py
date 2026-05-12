"""
Paper trading simulation — auto-enters and exits paper trades based on
live signal data, stored in SQLite.

How it works
------------
1. When a BUY/SELL signal fires with confidence ≥ threshold AND R:R qualifies,
   a paper trade is opened at the current price.
2. Each scan cycle, open paper trades are updated: exit signals are checked,
   P&L is computed, and trades are closed when conditions are met.
3. A summary of all paper trades is available via get_summary().

Paper trades are separate from signal_tracker.py (which only tracks signal
accuracy, not simulated trades with dynamic exits).
"""
from __future__ import annotations
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent.exit_signals import analyse_exits, ExitAnalysis

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "paper_trades.db"
_lock    = threading.Lock()

_MIN_CONFIDENCE = 0.0   # track all signals — paper trading measures accuracy, not filters it


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH), timeout=10)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                opened_at    TEXT    NOT NULL,
                closed_at    TEXT,
                ticker       TEXT    NOT NULL,
                direction    TEXT    NOT NULL,
                entry_price  REAL    NOT NULL,
                target       REAL    NOT NULL,
                stop         REAL    NOT NULL,
                confidence   REAL    NOT NULL,
                rr_ratio     REAL    DEFAULT 0,
                rr_qualifies INTEGER DEFAULT 0,
                bars_held    INTEGER DEFAULT 0,
                status       TEXT    DEFAULT 'OPEN',   -- OPEN | CLOSED
                exit_price   REAL,
                exit_reason  TEXT,
                pnl_pct      REAL,
                pnl_dollar   REAL
            )
        """)
        # Add columns to existing DBs (safe — ALTER TABLE IF NOT EXISTS column)
        for col, definition in [("rr_ratio", "REAL DEFAULT 0"),
                                 ("rr_qualifies", "INTEGER DEFAULT 0")]:
            try:
                c.execute(f"ALTER TABLE paper_trades ADD COLUMN {col} {definition}")
            except Exception:
                pass
        c.commit()


def maybe_open_trade(
    ticker:       str,
    direction:    str,
    price:        float,
    target:       float,
    stop:         float,
    confidence:   float,
    rr_qualifies: bool  = False,
    rr_ratio:     float = 0.0,
) -> Optional[int]:
    """Open a paper trade for every BUY/SELL signal. Returns trade id or None."""
    if direction not in ("BUY", "SELL"):
        return None

    # Don't open if one already open for this ticker
    with _lock:
        with _conn() as c:
            existing = c.execute(
                "SELECT id FROM paper_trades WHERE ticker=? AND status='OPEN'", (ticker,)
            ).fetchone()
            if existing:
                return None

            cur = c.execute("""
                INSERT INTO paper_trades
                  (opened_at, ticker, direction, entry_price, target, stop,
                   confidence, rr_ratio, rr_qualifies)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                ticker, direction,
                round(price, 4), round(target, 4), round(stop, 4),
                round(confidence, 2), round(rr_ratio, 2), int(rr_qualifies),
            ))
            c.commit()
            rr_flag = "✓ R:R" if rr_qualifies else "✗ R:R"
            logger.info(
                f"[PAPER] Opened {direction} {ticker} @ ${price:.2f} "
                f"T:${target:.2f}  S:${stop:.2f}  conf:{confidence:.0f}%  {rr_flag}"
            )
            return cur.lastrowid


def update_open_trades(ticker: str, df, current_price: float) -> None:
    """Check open trades for ticker and close if exit conditions met."""
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT id, direction, entry_price, target, stop, bars_held
                FROM paper_trades WHERE ticker=? AND status='OPEN'
            """, (ticker,)).fetchall()

            for row in rows:
                bars = (row["bars_held"] or 0) + 1
                c.execute("UPDATE paper_trades SET bars_held=? WHERE id=?", (bars, row["id"]))

                ea: ExitAnalysis = analyse_exits(
                    df=df,
                    direction=row["direction"],
                    entry_price=row["entry_price"],
                    target=row["target"],
                    stop=row["stop"],
                    bars_held=bars,
                )

                if ea.recommendation == "EXIT_NOW":
                    ep    = current_price
                    entry = row["entry_price"] or 1.0
                    if row["direction"] == "BUY":
                        pnl_pct = (ep - entry) / entry * 100
                    else:
                        pnl_pct = (entry - ep) / entry * 100
                    pnl_dollar = pnl_pct / 100 * entry * 100   # assume 100 shares

                    reason = ea.signals[0].signal if ea.signals else "UNKNOWN"
                    c.execute("""
                        UPDATE paper_trades
                        SET status='CLOSED', closed_at=?, exit_price=?,
                            exit_reason=?, pnl_pct=?, pnl_dollar=?
                        WHERE id=?
                    """, (
                        datetime.now(timezone.utc).isoformat(),
                        round(ep, 4), reason,
                        round(pnl_pct, 3), round(pnl_dollar, 2),
                        row["id"]
                    ))
                    outcome = "WIN" if pnl_pct > 0 else "LOSS"
                    logger.info(
                        f"[PAPER] Closed {row['direction']} {ticker} @ ${ep:.2f} | "
                        f"{outcome} {pnl_pct:+.2f}% | Reason: {reason}"
                    )
            c.commit()


def get_open_trades() -> list[dict]:
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT * FROM paper_trades WHERE status='OPEN' ORDER BY id DESC
            """).fetchall()
    return [dict(r) for r in rows]


def get_closed_trades(limit: int = 50) -> list[dict]:
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT * FROM paper_trades WHERE status='CLOSED'
                ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_summary() -> dict:
    with _lock:
        with _conn() as c:
            closed = c.execute("""
                SELECT pnl_pct, direction FROM paper_trades WHERE status='CLOSED'
            """).fetchall()
            open_count = c.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE status='OPEN'"
            ).fetchone()[0]

    total   = len(closed)
    wins    = sum(1 for r in closed if (r["pnl_pct"] or 0) > 0)
    losses  = sum(1 for r in closed if (r["pnl_pct"] or 0) <= 0)
    pnls    = [r["pnl_pct"] for r in closed if r["pnl_pct"] is not None]
    avg_pnl = round(sum(pnls) / len(pnls), 3) if pnls else 0.0
    total_pnl = round(sum(pnls), 2)

    return {
        "open":       open_count,
        "closed":     total,
        "wins":       wins,
        "losses":     losses,
        "win_rate":   round(wins / total * 100, 1) if total > 0 else 0.0,
        "avg_pnl":    avg_pnl,
        "total_pnl":  total_pnl,
    }
