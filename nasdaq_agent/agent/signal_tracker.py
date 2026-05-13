"""
Signal accuracy tracker — records every signal with full market context and
resolves outcomes so the self-learning system can learn from what actually
happens in the market, not just from paper trade P&L.

Two outcome paths
-----------------
1. TP/SL resolution  — price hits target or stop (can take hours; reliable)
2. Short-term check  — at the NEXT scan (~90s), did price move ≥ 0.15% in the
   predicted direction?  This gives feedback within seconds and is especially
   important during AFTER_HOURS and CLOSED sessions when TP/SL may never fire.

Context dimensions tracked per signal
--------------------------------------
  session, trading_tier, regime, vwap_event, rsi_zone, rel_volume bucket,
  trend, direction, ah_tier (tier × AH session combo)

This makes the learning system sensitive to:
  - Which sessions have reliable signals (OPEN > AFTER_HOURS)
  - Which tier stocks behave well in AH (HIGH > REGULAR)
  - What volume/VWAP conditions improve accuracy
  - Market regime effects on signal quality
"""
from __future__ import annotations
import logging
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "signal_history.db"
_lock    = threading.Lock()

SHORT_WIN_PCT = 0.35   # price move % needed for a short-term win at next scan
SLIPPAGE_PCT  = 0.05   # realistic bid-ask + fill slippage per side


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH), timeout=10)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            TEXT    NOT NULL,
                ticker        TEXT    NOT NULL,
                direction     TEXT    NOT NULL,
                entry_price   REAL,
                target        REAL,
                stop          REAL,
                confidence    REAL,
                session       TEXT    DEFAULT '',
                regime        TEXT    DEFAULT '',
                trading_tier  TEXT    DEFAULT 'REGULAR',
                vwap_event    TEXT    DEFAULT '',
                rsi_zone      TEXT    DEFAULT '',
                vol_bucket    TEXT    DEFAULT 'NORMAL',
                trend         TEXT    DEFAULT '',
                outcome       TEXT    DEFAULT 'PENDING',
                short_outcome TEXT    DEFAULT 'PENDING',
                exit_price    REAL,
                pnl_pct       REAL,
                bars_held     INTEGER DEFAULT 0,
                is_suppressed INTEGER DEFAULT 0
            )
        """)
        # Add new columns to existing DBs that predate this schema
        for col, typedef in [
            ("trading_tier",  "TEXT DEFAULT 'REGULAR'"),
            ("vwap_event",    "TEXT DEFAULT ''"),
            ("rsi_zone",      "TEXT DEFAULT ''"),
            ("vol_bucket",    "TEXT DEFAULT 'NORMAL'"),
            ("trend",         "TEXT DEFAULT ''"),
            ("short_outcome", "TEXT DEFAULT 'PENDING'"),
            ("is_suppressed", "INTEGER DEFAULT 0"),
        ]:
            try:
                c.execute(f"ALTER TABLE signals ADD COLUMN {col} {typedef}")
            except sqlite3.OperationalError:
                pass   # column already exists
        c.commit()


def record_signal(
    ticker:       str,
    direction:    str,
    entry:        float,
    target:       float,
    stop:         float,
    confidence:   float,
    session:      str   = "",
    regime:       str   = "",
    trading_tier: str   = "REGULAR",
    vwap_event:   str   = "",
    rsi_zone:     str   = "",
    rel_volume:   float = 1.0,
    trend:        str   = "",
) -> int:
    """Insert a new signal and return its row id."""
    vol_bucket = "HIGH" if rel_volume >= 3.0 else "ELEVATED" if rel_volume >= 1.5 else "NORMAL"
    with _lock:
        with _conn() as c:
            cur = c.execute("""
                INSERT INTO signals
                  (ts, ticker, direction, entry_price, target, stop, confidence,
                   session, regime, trading_tier, vwap_event, rsi_zone, vol_bucket, trend)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                ticker, direction,
                round(entry, 4), round(target, 4), round(stop, 4), round(confidence, 2),
                session, regime, trading_tier, vwap_event, rsi_zone, vol_bucket, trend,
            ))
            c.commit()
            return cur.lastrowid


def record_suppressed_signal(
    ticker:       str,
    direction:    str,
    entry:        float,
    confidence:   float,
    suppress_reason: str,
    session:      str   = "",
    regime:       str   = "",
    trading_tier: str   = "REGULAR",
    vwap_event:   str   = "",
    rsi_zone:     str   = "",
    rel_volume:   float = 1.0,
    trend:        str   = "",
) -> int:
    """Record a signal that was suppressed by the adaptive filter.
    These are tracked separately to measure false-negative rate."""
    vol_bucket = "HIGH" if rel_volume >= 3.0 else "ELEVATED" if rel_volume >= 1.5 else "NORMAL"
    with _lock:
        with _conn() as c:
            cur = c.execute("""
                INSERT INTO signals
                  (ts, ticker, direction, entry_price, target, stop, confidence,
                   session, regime, trading_tier, vwap_event, rsi_zone, vol_bucket, trend,
                   outcome, short_outcome, is_suppressed)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                ticker, direction,
                round(entry, 4), 0.0, 0.0, round(confidence, 2),
                session, regime, trading_tier, vwap_event, rsi_zone, vol_bucket, trend,
                "SUPPRESSED", "SUPPRESSED", 1,
            ))
            c.commit()
            return cur.lastrowid


def record_signals_batch(signals: list) -> None:
    """
    Record all actionable signals from a scan cycle with full market context.
    Called from scanner._on_signals() every scan cycle.
    """
    actionable = [s for s in signals
                  if getattr(s, "prediction", "NEUTRAL") not in ("NEUTRAL", "FILTERED")]
    if not actionable:
        return
    with _lock:
        with _conn() as c:
            ts = datetime.now(timezone.utc).isoformat()
            for s in actionable:
                rvol = getattr(s, "rel_volume", 1.0) or 1.0
                vol_bucket = "HIGH" if rvol >= 3.0 else "ELEVATED" if rvol >= 1.5 else "NORMAL"
                c.execute("""
                    INSERT INTO signals
                      (ts, ticker, direction, entry_price, target, stop, confidence,
                       session, regime, trading_tier, vwap_event, rsi_zone, vol_bucket, trend)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    ts,
                    s.ticker,
                    s.prediction,
                    round(getattr(s, "price", 0) or 0, 4),
                    round(getattr(s, "target_price", 0) or 0, 4),
                    round(getattr(s, "stop_loss", 0) or 0, 4),
                    round(getattr(s, "confidence", 50) or 50, 2),
                    getattr(s, "session", "") or "",
                    getattr(s, "regime", "") or "",
                    getattr(s, "trading_tier", "REGULAR") or "REGULAR",
                    getattr(s, "vwap_event", "") or "",
                    getattr(s, "rsi_zone", "") or "",
                    vol_bucket,
                    getattr(s, "trend", "") or "",
                ))
            c.commit()


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

                if d in ("BUY", "STRONG BUY"):
                    if current_price >= row["target"]:
                        outcome = "WIN"
                    elif current_price <= row["stop"]:
                        outcome = "LOSS"
                elif d in ("SELL", "STRONG SELL"):
                    if current_price <= row["target"]:
                        outcome = "WIN"
                    elif current_price >= row["stop"]:
                        outcome = "LOSS"

                if outcome:
                    entry  = row["entry_price"] or 1.0
                    pnl    = (exit_p - entry) / entry * 100
                    if d in ("SELL", "STRONG SELL"):
                        pnl = -pnl
                    c.execute("""
                        UPDATE signals SET outcome=?, exit_price=?, pnl_pct=?
                        WHERE id=?
                    """, (outcome, round(exit_p, 4), round(pnl, 3), row["id"]))
            c.commit()


def resolve_short_term(signals: list) -> None:
    """
    Short-term accuracy check: for every signal that fired on the PREVIOUS scan,
    check if price moved SHORT_WIN_PCT in the right direction by now.
    Updates short_outcome = WIN | LOSS for signals still PENDING.
    Runs every scan cycle — gives the learning engine feedback within 90s.
    """
    if not signals:
        return

    price_map = {s.ticker: getattr(s, "price", 0) for s in signals}

    with _lock:
        with _conn() as c:
            # Only look at signals from the last 10 minutes that lack a short_outcome
            rows = c.execute("""
                SELECT id, ticker, direction, entry_price
                FROM signals
                WHERE short_outcome = 'PENDING'
                  AND ts >= datetime('now', '-10 minutes')
            """).fetchall()

            for row in rows:
                curr_price = price_map.get(row["ticker"], 0)
                entry      = row["entry_price"] or 0
                if curr_price <= 0 or entry <= 0:
                    continue

                pct_move = (curr_price - entry) / entry * 100
                d        = row["direction"]

                _effective_threshold = SHORT_WIN_PCT + SLIPPAGE_PCT
                if d in ("BUY", "STRONG BUY"):
                    short_out = "WIN" if pct_move >= _effective_threshold else "LOSS"
                elif d in ("SELL", "STRONG SELL"):
                    short_out = "WIN" if pct_move <= -_effective_threshold else "LOSS"
                else:
                    continue

                c.execute("UPDATE signals SET short_outcome=? WHERE id=?",
                          (short_out, row["id"]))
            c.commit()


def get_market_breakdown_stats(min_count: int = 5, lookback_days: int = 30) -> dict:
    """
    Build context-breakdown stats from resolved signal outcomes.
    Uses BOTH TP/SL outcomes AND short-term accuracy checks so the system
    gets feedback from every scan cycle, not just when trades close.

    Returns a dict compatible with adaptive_filter.update_filter().
    """
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT direction, session, regime, trading_tier,
                       vwap_event, rsi_zone, vol_bucket, trend,
                       outcome, short_outcome
                FROM signals
                WHERE ts >= datetime('now', ?)
                  AND (outcome != 'PENDING' OR short_outcome != 'PENDING')
            """, (f"-{lookback_days} days",)).fetchall()

    if not rows:
        return {"overall": {"total": 0, "wins": 0, "win_rate": 0.0}}

    # For each row, determine effective win/loss
    # TP/SL takes priority; fall back to short-term outcome
    def _is_win(r) -> Optional[bool]:
        if r["outcome"] in ("WIN", "LOSS"):
            return r["outcome"] == "WIN"
        if r["short_outcome"] in ("WIN", "LOSS"):
            return r["short_outcome"] == "WIN"
        return None

    # Build breakdowns
    dims: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "total": 0}))

    total_wins = total_count = 0

    for r in rows:
        won = _is_win(r)
        if won is None:
            continue
        total_count += 1
        if won:
            total_wins += 1

        def _credit(dim: str, val: str) -> None:
            if not val:
                return
            dims[dim][val]["total"] += 1
            if won:
                dims[dim][val]["wins"] += 1

        _credit("session",      r["session"])
        _credit("trading_tier", r["trading_tier"])
        _credit("regime",       r["regime"])
        _credit("vwap_event",   r["vwap_event"])
        _credit("rsi_zone",     r["rsi_zone"])
        _credit("vol_bucket",   r["vol_bucket"])
        _credit("trend",        r["trend"])

        # AH tier combo
        if r["session"] in ("AFTER_HOURS", "PRE_MARKET", "CLOSED"):
            _credit("ah_tier", r["trading_tier"])

    def _build(dim_data: dict) -> dict:
        result = {}
        for val, counts in dim_data.items():
            t = counts["total"]
            if t < min_count:
                continue
            w = counts["wins"]
            result[val] = {"total": t, "wins": w,
                           "win_rate": round(w / t, 4), "avg_r": 0.0}
        return result

    overall_wr = round(total_wins / total_count, 4) if total_count else 0.0

    return {
        "overall":         {"total": total_count, "wins": total_wins, "win_rate": overall_wr},
        "by_session":      _build(dims["session"]),
        "by_regime":       _build(dims["regime"]),
        "by_trading_tier": _build(dims["trading_tier"]),
        "by_ah_tier":      _build(dims["ah_tier"]),
        "by_vwap_event":   _build(dims["vwap_event"]),
        "by_rsi_zone":     _build(dims["rsi_zone"]),
        "by_vol_bucket":   _build(dims["vol_bucket"]),
        "by_entry_type":   {},
        "by_direction":    {},
        "by_sector_trend": {},
        "by_ah_bias":      {},
        "by_confidence":   {},
    }


def get_stats(ticker: Optional[str] = None, limit: int = 200) -> dict:
    """Return accuracy statistics for a ticker or globally."""
    with _lock:
        with _conn() as c:
            where  = "WHERE ticker=? AND (is_suppressed IS NULL OR is_suppressed=0)" if ticker else "WHERE (is_suppressed IS NULL OR is_suppressed=0)"
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


def get_observation_summary() -> dict:
    """Summary for the learning status API and live log."""
    with _lock:
        with _conn() as c:
            row = c.execute("""
                SELECT
                  COUNT(*) as total,
                  SUM(CASE WHEN outcome != 'PENDING' THEN 1 ELSE 0 END) as tp_sl_resolved,
                  SUM(CASE WHEN short_outcome != 'PENDING' THEN 1 ELSE 0 END) as short_resolved,
                  SUM(CASE WHEN outcome='WIN' OR short_outcome='WIN' THEN 1 ELSE 0 END) as wins
                FROM signals
                WHERE ts >= datetime('now', '-30 days')
            """).fetchone()

    total    = row["total"] or 0
    resolved = max(row["tp_sl_resolved"] or 0, row["short_resolved"] or 0)
    wins     = row["wins"] or 0
    return {
        "total_signals":    total,
        "resolved":         resolved,
        "wins":             wins,
        "observation_wr":   round(wins / resolved * 100, 1) if resolved else 0.0,
    }


def get_ticker_learning_scores(lookback_days: int = 30, min_count: int = 3) -> dict[str, dict]:
    """
    Return per-ticker accuracy stats derived from both TP/SL and short-term outcomes.
    Used by the scanner to attach learning_rank to each signal.

    Returns: { "NVDA": {"win_rate": 0.71, "count": 42}, ... }
    """
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT ticker, outcome, short_outcome
                FROM signals
                WHERE ts >= datetime('now', ?)
                  AND (outcome != 'PENDING' OR short_outcome != 'PENDING')
            """, (f"-{lookback_days} days",)).fetchall()

    from collections import defaultdict
    counts: dict = defaultdict(lambda: {"wins": 0, "total": 0})

    for r in rows:
        # TP/SL takes priority, fall back to short-term
        if r["outcome"] in ("WIN", "LOSS"):
            won = r["outcome"] == "WIN"
        elif r["short_outcome"] in ("WIN", "LOSS"):
            won = r["short_outcome"] == "WIN"
        else:
            continue
        counts[r["ticker"]]["total"] += 1
        if won:
            counts[r["ticker"]]["wins"] += 1

    result = {}
    for ticker, c in counts.items():
        if c["total"] < min_count:
            continue
        result[ticker] = {
            "win_rate": round(c["wins"] / c["total"], 4),
            "count":    c["total"],
        }
    return result


def get_suppressed_stats(lookback_days: int = 7) -> dict:
    """Stats on suppressed signals — used to measure false-negative rate."""
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT direction, session, regime, confidence
                FROM signals
                WHERE is_suppressed = 1
                  AND ts >= datetime('now', ?)
            """, (f"-{lookback_days} days",)).fetchall()
    total = len(rows)
    by_session: dict = {}
    for r in rows:
        s = r["session"] or "UNKNOWN"
        by_session[s] = by_session.get(s, 0) + 1
    return {"total_suppressed": total, "by_session": by_session}


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
