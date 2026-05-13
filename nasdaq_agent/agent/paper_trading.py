"""
Paper trading simulation — auto-enters and exits paper trades based on
live signal data, stored in SQLite.

How it works
------------
1. When a BUY/SELL signal fires with confidence ≥ adaptive threshold AND R:R qualifies,
   a paper trade is opened with full context (session, regime, vwap_event, etc.).
2. Each scan cycle, open paper trades are updated: exit signals are checked,
   P&L is computed, and trades are closed when conditions are met.
3. When a trade closes, its outcome is immediately fed back to the adaptive filter
   so the system learns in real-time — not just from backtest outcomes.
4. A summary of all paper trades is available via get_summary().
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

_FALLBACK_MIN_CONFIDENCE = 65.0   # used before adaptive filter has enough data


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
                status       TEXT    DEFAULT 'OPEN',
                exit_price   REAL,
                exit_reason  TEXT,
                pnl_pct      REAL,
                pnl_dollar   REAL,
                session      TEXT    DEFAULT '',
                regime       TEXT    DEFAULT '',
                vwap_event   TEXT    DEFAULT '',
                rsi_zone     TEXT    DEFAULT '',
                entry_type   TEXT    DEFAULT ''
            )
        """)
        # Safe migration: add any missing columns to existing DBs
        for col, definition in [
            ("rr_ratio",    "REAL DEFAULT 0"),
            ("rr_qualifies","INTEGER DEFAULT 0"),
            ("session",     "TEXT DEFAULT ''"),
            ("regime",      "TEXT DEFAULT ''"),
            ("vwap_event",  "TEXT DEFAULT ''"),
            ("rsi_zone",    "TEXT DEFAULT ''"),
            ("entry_type",  "TEXT DEFAULT ''"),
        ]:
            try:
                c.execute(f"ALTER TABLE paper_trades ADD COLUMN {col} {definition}")
            except Exception:
                pass
        c.commit()


def _get_min_confidence() -> float:
    """Return the dynamic confidence gate from the adaptive filter."""
    try:
        from agent.adaptive_filter import get_status as _af_status
        return float(_af_status().get("dynamic_threshold", _FALLBACK_MIN_CONFIDENCE))
    except Exception:
        return _FALLBACK_MIN_CONFIDENCE


def maybe_open_trade(
    ticker:       str,
    direction:    str,
    price:        float,
    target:       float,
    stop:         float,
    confidence:   float,
    rr_qualifies: bool  = False,
    rr_ratio:     float = 0.0,
    session:      str   = "",
    regime:       str   = "",
    vwap_event:   str   = "",
    rsi_zone:     str   = "",
    entry_type:   str   = "",
) -> Optional[int]:
    """
    Open a paper trade when signal passes the adaptive confidence gate AND R:R qualifies.
    Full context (session, regime, etc.) is stored so losses can be attributed and learned from.
    Returns trade id or None.
    """
    if direction not in ("BUY", "SELL"):
        return None

    min_conf = _get_min_confidence()
    if confidence < min_conf:
        return None
    if not rr_qualifies:
        return None

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
                   confidence, rr_ratio, rr_qualifies,
                   session, regime, vwap_event, rsi_zone, entry_type)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                ticker, direction,
                round(price, 4), round(target, 4), round(stop, 4),
                round(confidence, 2), round(rr_ratio, 2), int(rr_qualifies),
                session, regime, vwap_event, rsi_zone, entry_type,
            ))
            c.commit()
            logger.info(
                f"[PAPER] Opened {direction} {ticker} @ ${price:.2f} "
                f"T:${target:.2f}  S:${stop:.2f}  conf:{confidence:.0f}%  "
                f"sess:{session}  regime:{regime}  vwap:{vwap_event}"
            )
            return cur.lastrowid


def update_open_trades(ticker: str, df, current_price: float) -> None:
    """Check open trades for ticker and close if exit conditions met."""
    closed_any = False
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
                    closed_any = True
            c.commit()

    if closed_any:
        # Feed outcomes back to adaptive filter immediately after any close
        _trigger_paper_feedback()


def _trigger_paper_feedback() -> None:
    """
    Build context-breakdown stats from all closed paper trades and push them
    into the adaptive filter so it learns from real paper trading outcomes,
    not just backtest simulations.
    """
    try:
        from agent.adaptive_filter import update_from_paper_trades
        stats = _build_paper_stats()
        if stats["overall"]["total"] >= 5:
            update_from_paper_trades(stats)
    except Exception as e:
        logger.debug(f"[PAPER] Filter feedback skipped: {e}")


def _build_paper_stats() -> dict:
    """
    Aggregate closed paper trades into the same stats format that
    live_backtest produces, so adaptive_filter.update_filter() can consume it.
    """
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT direction, pnl_pct, confidence,
                       session, regime, vwap_event, rsi_zone, entry_type
                FROM paper_trades
                WHERE status='CLOSED' AND pnl_pct IS NOT NULL
            """).fetchall()

    trades = [dict(r) for r in rows]
    if not trades:
        return {"overall": {"total": 0, "win_rate": 0.0, "wins": 0}}

    def _stats(group):
        n    = len(group)
        wins = sum(1 for t in group if (t["pnl_pct"] or 0) > 0)
        return {"total": n, "wins": wins, "win_rate": round(wins / n, 4) if n else 0.0}

    def _breakdown(field):
        buckets: dict = {}
        for t in trades:
            val = (t.get(field) or "").strip()
            if not val:
                continue
            buckets.setdefault(val, []).append(t)
        return {k: _stats(v) for k, v in buckets.items()}

    def _conf_band(conf):
        if conf is None:
            return "unknown"
        if conf < 50:   return "<50"
        if conf < 60:   return "50-60"
        if conf < 70:   return "60-70"
        if conf < 80:   return "70-80"
        return "80+"

    conf_buckets: dict = {}
    for t in trades:
        band = _conf_band(t.get("confidence"))
        conf_buckets.setdefault(band, []).append(t)
    by_conf = {k: _stats(v) for k, v in conf_buckets.items()}

    return {
        "overall":       _stats(trades),
        "by_direction":  _breakdown("direction"),
        "by_session":    _breakdown("session"),
        "by_regime":     _breakdown("regime"),
        "by_vwap_event": _breakdown("vwap_event"),
        "by_rsi_zone":   _breakdown("rsi_zone"),
        "by_entry_type": _breakdown("entry_type"),
        "by_confidence": by_conf,
        # not available at paper trade level, but expected by update_filter
        "by_sector_trend": {},
    }


def get_daily_pnl(days: int = 14) -> list[dict]:
    """Return per-day P&L summary for the last N calendar days."""
    conn = _conn()
    try:
        rows = conn.execute("""
            SELECT
                date(closed_at) as trade_date,
                COUNT(*)        as total,
                SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl_pct <= 0 THEN 1 ELSE 0 END) as losses,
                ROUND(SUM(pnl_pct), 2)  as total_pnl_pct,
                ROUND(SUM(COALESCE(pnl_dollar, 0)), 2) as total_pnl_dollar,
                ROUND(AVG(pnl_pct), 2)  as avg_pnl_pct,
                ROUND(MAX(pnl_pct), 2)  as best_trade,
                ROUND(MIN(pnl_pct), 2)  as worst_trade
            FROM paper_trades
            WHERE status='CLOSED' AND closed_at >= date('now', ?)
            GROUP BY date(closed_at)
            ORDER BY trade_date DESC
        """, (f'-{days} days',)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_today_pnl() -> dict:
    """Return today's running P&L stats."""
    conn = _conn()
    try:
        row = conn.execute("""
            SELECT
                COUNT(*)        as total,
                SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                ROUND(SUM(pnl_pct), 2)  as total_pnl_pct,
                ROUND(SUM(COALESCE(pnl_dollar, 0)), 2) as total_pnl_dollar,
                ROUND(MAX(pnl_pct), 2)  as best_trade,
                ROUND(MIN(pnl_pct), 2)  as worst_trade
            FROM paper_trades
            WHERE status='CLOSED' AND date(closed_at) = date('now')
        """).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


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
