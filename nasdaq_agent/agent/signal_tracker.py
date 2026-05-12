"""
Signal accuracy tracker — persists every signal to SQLite and tracks outcomes.

Schema
------
signals(
  id          INTEGER PRIMARY KEY,
  ts          TEXT,          -- ISO timestamp when signal fired
  ticker      TEXT,
  direction   TEXT,          -- BUY | SELL | NEUTRAL
  entry_price REAL,
  target      REAL,
  stop        REAL,
  confidence  REAL,
  session     TEXT,
  regime      TEXT,
  outcome     TEXT,          -- WIN | LOSS | SCRATCH | PENDING
  exit_price  REAL,
  pnl_pct     REAL,
  bars_held   INTEGER
)

Outcome resolution runs whenever a new scan result is available —
it checks PENDING signals against current price and closes them if
target or stop has been hit.
"""
from __future__ import annotations
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "signal_history.db"
_lock    = threading.Lock()


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH), timeout=10)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                ticker      TEXT    NOT NULL,
                direction   TEXT    NOT NULL,
                entry_price REAL,
                target      REAL,
                stop        REAL,
                confidence  REAL,
                session     TEXT,
                regime      TEXT,
                outcome     TEXT    DEFAULT 'PENDING',
                exit_price  REAL,
                pnl_pct     REAL,
                bars_held   INTEGER DEFAULT 0
            )
        """)
        c.commit()


def record_signal(
    ticker:     str,
    direction:  str,
    entry:      float,
    target:     float,
    stop:       float,
    confidence: float,
    session:    str = "",
    regime:     str = "",
) -> int:
    """Insert a new signal and return its row id."""
    with _lock:
        with _conn() as c:
            cur = c.execute("""
                INSERT INTO signals
                  (ts, ticker, direction, entry_price, target, stop, confidence, session, regime)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                ticker, direction, round(entry, 4),
                round(target, 4), round(stop, 4), round(confidence, 2),
                session, regime,
            ))
            c.commit()
            return cur.lastrowid


def resolve_pending(ticker: str, current_price: float) -> None:
    """Check PENDING signals for ticker and close any that hit target or stop."""
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT id, direction, entry_price, target, stop
                FROM signals WHERE ticker=? AND outcome='PENDING'
            """, (ticker,)).fetchall()

            for row in rows:
                outcome = None
                exit_p  = current_price
                d       = row["direction"]

                if d == "BUY":
                    if current_price >= row["target"]:
                        outcome = "WIN"
                    elif current_price <= row["stop"]:
                        outcome = "LOSS"
                elif d == "SELL":
                    if current_price <= row["target"]:
                        outcome = "WIN"
                    elif current_price >= row["stop"]:
                        outcome = "LOSS"

                if outcome:
                    entry  = row["entry_price"] or 1.0
                    pnl    = (exit_p - entry) / entry * 100
                    if d == "SELL":
                        pnl = -pnl
                    c.execute("""
                        UPDATE signals SET outcome=?, exit_price=?, pnl_pct=?
                        WHERE id=?
                    """, (outcome, round(exit_p, 4), round(pnl, 3), row["id"]))

            c.commit()


def get_stats(ticker: Optional[str] = None, limit: int = 200) -> dict:
    """
    Return accuracy statistics.

    If ticker is provided, stats are filtered to that ticker.
    Returns dict with win_rate, avg_pnl, total, wins, losses, pending.
    """
    with _lock:
        with _conn() as c:
            where  = "WHERE ticker=?" if ticker else ""
            params = (ticker,) if ticker else ()
            rows   = c.execute(
                f"SELECT outcome, pnl_pct FROM signals {where} ORDER BY id DESC LIMIT ?",
                (*params, limit)
            ).fetchall()

    total   = len(rows)
    wins    = sum(1 for r in rows if r["outcome"] == "WIN")
    losses  = sum(1 for r in rows if r["outcome"] == "LOSS")
    pending = sum(1 for r in rows if r["outcome"] == "PENDING")
    closed  = wins + losses
    pnls    = [r["pnl_pct"] for r in rows if r["pnl_pct"] is not None]
    avg_pnl = round(sum(pnls) / len(pnls), 3) if pnls else 0.0

    return {
        "total":    total,
        "wins":     wins,
        "losses":   losses,
        "pending":  pending,
        "win_rate": round(wins / closed * 100, 1) if closed > 0 else 0.0,
        "avg_pnl":  avg_pnl,
    }


def get_recent_signals(limit: int = 50) -> list[dict]:
    """Return last N signals as list of dicts for the UI history table."""
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT ts, ticker, direction, entry_price, target, stop,
                       confidence, session, regime, outcome, exit_price, pnl_pct
                FROM signals ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()
    return [dict(r) for r in rows]
