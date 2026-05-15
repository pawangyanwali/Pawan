"""
Paper trading simulation — PRD-compliant position management.

PRD Section 6.3 rules implemented here:
  - Max 3 concurrent positions (MAX_CONCURRENT_TRADES)
  - T1 partial exit: close 50% at 1R profit, move stop to breakeven
  - T2 target: close remaining 50% at 2R profit
  - Time stop: 20 bars (scalp) or 90 bars (intraday) hard close
  - Hard close at 3:45 PM ET — all positions flat regardless of P&L
  - Breakeven rule: when T1 hit, stop moves to entry ± $0.02 (auto, non-overridable)
  - Portfolio heat and session blocks enforced via risk_controls.can_open_trade()
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

_FALLBACK_MIN_CONFIDENCE = 55.0
_MAX_BARS_HELD_SCALP     = 20   # 20-min hard close for scalps (PRD 6.3)
_MAX_BARS_HELD_INTRADAY  = 90   # 90-min hard close for intraday (PRD 6.3)
_MAX_CONCURRENT_TRADES   = 3    # PRD Section 6.3: max 3 simultaneous positions


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH), timeout=10)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                opened_at           TEXT    NOT NULL,
                closed_at           TEXT,
                ticker              TEXT    NOT NULL,
                direction           TEXT    NOT NULL,
                entry_price         REAL    NOT NULL,
                target              REAL    NOT NULL,
                stop                REAL    NOT NULL,
                confidence          REAL    NOT NULL,
                rr_ratio            REAL    DEFAULT 0,
                rr_qualifies        INTEGER DEFAULT 0,
                bars_held           INTEGER DEFAULT 0,
                status              TEXT    DEFAULT 'OPEN',
                exit_price          REAL,
                exit_reason         TEXT,
                pnl_pct             REAL,
                pnl_dollar          REAL,
                shares              INTEGER DEFAULT 1,
                session             TEXT    DEFAULT '',
                regime              TEXT    DEFAULT '',
                vwap_event          TEXT    DEFAULT '',
                rsi_zone            TEXT    DEFAULT '',
                entry_type          TEXT    DEFAULT '',
                -- T1/T2 partial exit tracking (PRD 6.3)
                t1_hit              INTEGER DEFAULT 0,
                t1_price            REAL    DEFAULT 0,
                t2_price            REAL    DEFAULT 0,
                breakeven_set       INTEGER DEFAULT 0,
                partial_pnl_dollar  REAL    DEFAULT 0,
                shares_remaining    INTEGER DEFAULT 0,
                order_flow_score    REAL    DEFAULT 0,
                size_mult           REAL    DEFAULT 1.0
            )
        """)
        # Safe migration: add any missing columns to existing DBs
        _migrate_columns(c)
        c.commit()


def _migrate_columns(c: sqlite3.Connection) -> None:
    existing = {row[1] for row in c.execute("PRAGMA table_info(paper_trades)").fetchall()}
    additions = [
        ("rr_ratio",           "REAL DEFAULT 0"),
        ("rr_qualifies",       "INTEGER DEFAULT 0"),
        ("session",            "TEXT DEFAULT ''"),
        ("regime",             "TEXT DEFAULT ''"),
        ("vwap_event",         "TEXT DEFAULT ''"),
        ("rsi_zone",           "TEXT DEFAULT ''"),
        ("entry_type",         "TEXT DEFAULT ''"),
        ("shares",             "INTEGER DEFAULT 1"),
        ("t1_hit",             "INTEGER DEFAULT 0"),
        ("t1_price",           "REAL DEFAULT 0"),
        ("t2_price",           "REAL DEFAULT 0"),
        ("breakeven_set",      "INTEGER DEFAULT 0"),
        ("partial_pnl_dollar", "REAL DEFAULT 0"),
        ("shares_remaining",   "INTEGER DEFAULT 0"),
        ("order_flow_score",   "REAL DEFAULT 0"),
        ("size_mult",          "REAL DEFAULT 1.0"),
    ]
    for col, definition in additions:
        if col not in existing:
            try:
                c.execute(f"ALTER TABLE paper_trades ADD COLUMN {col} {definition}")
            except Exception:
                pass


def _get_min_confidence() -> float:
    try:
        from agent.adaptive_filter import get_status as _af
        return float(_af().get("dynamic_threshold", _FALLBACK_MIN_CONFIDENCE))
    except Exception:
        return _FALLBACK_MIN_CONFIDENCE


def maybe_open_trade(
    ticker:           str,
    direction:        str,
    price:            float,
    target:           float,
    stop:             float,
    confidence:       float,
    rr_qualifies:     bool  = False,
    rr_ratio:         float = 0.0,
    session:          str   = "",
    regime:           str   = "",
    vwap_event:       str   = "",
    rsi_zone:         str   = "",
    entry_type:       str   = "",
    order_flow_score: float = 0.0,
    size_mult:        float = 1.0,
) -> Optional[int]:
    """
    Open a paper trade when all PRD entry gates pass.
    Returns trade id or None.
    """
    if direction not in ("BUY", "SELL"):
        return None

    min_conf = _get_min_confidence()
    if confidence < min_conf:
        return None

    # ── PRD master entry gate (session / circuit breaker / heat / sector) ──
    from agent.risk_controls import can_open_trade
    allowed, block_reason, gate_size_mult = can_open_trade(ticker, direction, confidence)
    if not allowed:
        logger.debug(f"[PAPER] {ticker} blocked: {block_reason}")
        return None

    # Combine gate size multiplier with signal-level size multiplier.
    # Low R:R setups trade at half size — system still learns from the outcome.
    rr_mult = 1.0 if rr_qualifies else 0.5
    effective_size_mult = round(gate_size_mult * size_mult * rr_mult, 2)
    if effective_size_mult <= 0:
        return None

    # ── Risk-based position sizing ─────────────────────────────────────────
    from agent.position_sizing import calculate as _calc_pos
    from config import DEFAULT_ACCOUNT_SIZE, DEFAULT_RISK_PCT, MAX_POSITION_PCT
    _ps = _calc_pos(
        account_size     = DEFAULT_ACCOUNT_SIZE,
        entry            = price,
        stop             = stop,
        risk_pct         = DEFAULT_RISK_PCT,
        max_position_pct = MAX_POSITION_PCT,
        confidence       = confidence,
    )
    shares = max(1, int(_ps.shares * effective_size_mult))

    # ── T1 and T2 price levels ─────────────────────────────────────────────
    risk_dist = abs(price - stop)
    if direction == "BUY":
        t1_price = round(price + risk_dist, 4)       # 1R
        t2_price = round(price + 2 * risk_dist, 4)  # 2R
    else:
        t1_price = round(price - risk_dist, 4)
        t2_price = round(price - 2 * risk_dist, 4)

    with _lock:
        with _conn() as c:
            existing = c.execute(
                "SELECT id FROM paper_trades WHERE ticker=? AND status='OPEN'", (ticker,)
            ).fetchone()
            if existing:
                return None

            open_count = c.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE status='OPEN'"
            ).fetchone()[0]
            if open_count >= _MAX_CONCURRENT_TRADES:
                return None

            cur = c.execute("""
                INSERT INTO paper_trades
                  (opened_at, ticker, direction, entry_price, target, stop,
                   confidence, rr_ratio, rr_qualifies, shares, shares_remaining,
                   session, regime, vwap_event, rsi_zone, entry_type,
                   t1_price, t2_price, order_flow_score, size_mult)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                ticker, direction,
                round(price, 4), round(target, 4), round(stop, 4),
                round(confidence, 2), round(rr_ratio, 2), int(rr_qualifies),
                shares, shares,  # shares_remaining starts = shares
                session, regime, vwap_event, rsi_zone, entry_type,
                t1_price, t2_price,
                round(order_flow_score, 4), round(effective_size_mult, 2),
            ))
            c.commit()
            logger.info(
                f"[PAPER] Opened {direction} {ticker} @ ${price:.2f} "
                f"T1:${t1_price:.2f}  T2:${t2_price:.2f}  S:${stop:.2f}  "
                f"conf:{confidence:.0f}%  shares:{shares}  OF:{order_flow_score:+.2f}  "
                f"sess:{session}  regime:{regime}"
            )
            return cur.lastrowid


def update_open_trades(ticker: str, df, current_price: float) -> None:
    """
    PRD-compliant position management:
      1. Hard close at 3:45 PM ET (EOD rule)
      2. T1 partial exit at 1R → lock in 50%, move stop to breakeven
      3. T2 full exit at 2R → close remaining position
      4. Stop hit → close remaining shares
      5. Time stop: 20-bar scalp or 90-bar intraday
      6. Exit signal analysis (MACD, RSI reversal, etc.)
    """
    from agent.market_hours import is_hard_close_window

    closed_any = False
    won_any    = None   # last closed outcome for consecutive loss tracking

    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT id, direction, entry_price, target, stop, bars_held,
                       COALESCE(shares, 1) as shares,
                       COALESCE(shares_remaining, shares, 1) as shares_remaining,
                       COALESCE(t1_hit, 0) as t1_hit,
                       COALESCE(breakeven_set, 0) as breakeven_set,
                       COALESCE(partial_pnl_dollar, 0) as partial_pnl_dollar,
                       COALESCE(t1_price, 0) as t1_price,
                       COALESCE(t2_price, 0) as t2_price,
                       COALESCE(entry_type, '') as entry_type
                FROM paper_trades WHERE ticker=? AND status='OPEN'
            """, (ticker,)).fetchall()

            for row in rows:
                bars = (row["bars_held"] or 0) + 1
                c.execute("UPDATE paper_trades SET bars_held=? WHERE id=?", (bars, row["id"]))

                entry           = float(row["entry_price"] or price)
                stop_current    = float(row["stop"])
                t1_price        = float(row["t1_price"] or 0)
                t2_price        = float(row["t2_price"] or 0)
                shares_total    = int(row["shares"] or 1)
                shares_rem      = int(row["shares_remaining"] or shares_total)
                t1_hit          = bool(row["t1_hit"])
                partial_pnl     = float(row["partial_pnl_dollar"] or 0)
                direction       = row["direction"]
                entry_type      = row["entry_type"] or "IMMEDIATE"

                # Determine time stop based on trade type (PRD 6.3)
                is_scalp = entry_type in ("IMMEDIATE", "SCALP") or bars <= 20
                max_bars = _MAX_BARS_HELD_SCALP if is_scalp else _MAX_BARS_HELD_INTRADAY

                ep           = current_price
                exit_reason  = None
                close_shares = 0
                is_partial   = False

                # ── 1. Hard close at 3:45 PM ET (PRD — non-overridable) ────────
                if is_hard_close_window():
                    exit_reason  = "EOD_HARD_CLOSE_3:45PM"
                    close_shares = shares_rem

                # ── 2. T1 partial exit (1R profit) — if not already hit ────────
                elif not t1_hit and t1_price > 0:
                    t1_hit_now = (
                        (direction == "BUY"  and ep >= t1_price) or
                        (direction == "SELL" and ep <= t1_price)
                    )
                    if t1_hit_now:
                        # Exit 50% at T1, move stop to breakeven
                        partial_shares = max(1, shares_rem // 2)
                        t1_pnl = (
                            (t1_price - entry) * partial_shares if direction == "BUY"
                            else (entry - t1_price) * partial_shares
                        )
                        new_partial_pnl = partial_pnl + t1_pnl
                        new_shares_rem  = shares_rem - partial_shares
                        # Breakeven stop: entry ± $0.02 (PRD 6.3 — always auto)
                        be_stop = round(entry - 0.02, 4) if direction == "BUY" else round(entry + 0.02, 4)
                        c.execute("""
                            UPDATE paper_trades
                            SET t1_hit=1, breakeven_set=1, stop=?,
                                partial_pnl_dollar=?, shares_remaining=?
                            WHERE id=?
                        """, (be_stop, round(new_partial_pnl, 2), new_shares_rem, row["id"]))
                        c.commit()
                        logger.info(
                            f"[PAPER] T1 HIT {direction} {ticker} @ ${ep:.2f} | "
                            f"Partial exit {partial_shares} shares, locked ${t1_pnl:+.2f} | "
                            f"Stop → breakeven ${be_stop:.2f} | {new_shares_rem} shares remaining"
                        )
                        # Reload updated row values
                        stop_current = be_stop
                        t1_hit       = True
                        partial_pnl  = new_partial_pnl
                        shares_rem   = new_shares_rem
                        if shares_rem <= 0:
                            exit_reason  = "T1_FULL_EXIT"
                            close_shares = 0   # all shares already accounted for
                            # Record closed trade
                            _record_close(c, row["id"], ep, exit_reason, entry, direction,
                                          shares_total, partial_pnl)
                            closed_any = True
                            won_any    = partial_pnl > 0
                            continue

                # ── 3. T2 full exit (2R profit) — only if T1 already hit ──────
                if not exit_reason and t1_hit and t2_price > 0:
                    t2_hit_now = (
                        (direction == "BUY"  and ep >= t2_price) or
                        (direction == "SELL" and ep <= t2_price)
                    )
                    if t2_hit_now:
                        exit_reason  = "TARGET_T2"
                        close_shares = shares_rem
                        ep           = t2_price  # fill at T2

                # ── 4. Stop hit ────────────────────────────────────────────────
                if not exit_reason:
                    stop_hit = (
                        (direction == "BUY"  and ep <= stop_current) or
                        (direction == "SELL" and ep >= stop_current)
                    )
                    if stop_hit:
                        exit_reason  = "STOP_HIT_BREAKEVEN" if row["breakeven_set"] else "STOP_HIT"
                        close_shares = shares_rem

                # ── 5. Time stop ───────────────────────────────────────────────
                if not exit_reason and bars >= max_bars:
                    exit_reason  = f"TIME_STOP_{max_bars}BARS"
                    close_shares = shares_rem

                # ── 6. Exit signal analysis (technical exits) ─────────────────
                if not exit_reason:
                    ea: ExitAnalysis = analyse_exits(
                        df=df, direction=direction,
                        entry_price=entry, target=row["target"],
                        stop=stop_current, bars_held=bars,
                    )
                    if ea.recommendation == "EXIT_NOW":
                        exit_reason  = ea.signals[0].signal if ea.signals else "SIGNAL_EXIT"
                        close_shares = shares_rem

                # ── Close trade ────────────────────────────────────────────────
                if exit_reason and close_shares > 0:
                    _record_close(c, row["id"], ep, exit_reason, entry, direction,
                                  close_shares, partial_pnl)
                    closed_any = True
                    # Calculate net P&L for consecutive loss tracking
                    final_pnl = (
                        ((ep - entry) * close_shares if direction == "BUY"
                         else (entry - ep) * close_shares)
                        + partial_pnl
                    )
                    won_any = final_pnl > 0

            c.commit()

    if closed_any:
        if won_any is not None:
            try:
                from agent.risk_controls import record_trade_outcome
                record_trade_outcome(won_any)
            except Exception:
                pass
        _trigger_paper_feedback()


def _record_close(
    c: sqlite3.Connection,
    trade_id:     int,
    exit_price:   float,
    exit_reason:  str,
    entry:        float,
    direction:    str,
    close_shares: int,
    partial_pnl:  float = 0.0,
) -> None:
    """Write the final closed state for a trade record."""
    ep = exit_price
    if direction == "BUY":
        pnl_pct    = (ep - entry) / entry * 100
        pnl_dollar = (ep - entry) * close_shares + partial_pnl
    else:
        pnl_pct    = (entry - ep) / entry * 100
        pnl_dollar = (entry - ep) * close_shares + partial_pnl

    # pnl_pct should account for partial exit locked P&L directionally
    if partial_pnl > 0:
        pnl_pct = pnl_dollar / (entry * close_shares + 0.01) * 100

    outcome = "WIN" if pnl_dollar > 0 else "LOSS"
    c.execute("""
        UPDATE paper_trades
        SET status='CLOSED', closed_at=?, exit_price=?,
            exit_reason=?, pnl_pct=?, pnl_dollar=?
        WHERE id=?
    """, (
        datetime.now(timezone.utc).isoformat(),
        round(ep, 4), exit_reason,
        round(pnl_pct, 3), round(pnl_dollar, 2),
        trade_id,
    ))
    logger.info(
        f"[PAPER] Closed {direction} @ ${ep:.2f} | "
        f"{outcome} ${pnl_dollar:+.2f} ({pnl_pct:+.2f}%) | Reason: {exit_reason}"
    )


def close_all_positions_eod() -> int:
    """
    Force-close ALL open paper trades at current price.
    Called at 3:45 PM ET hard close. Returns number of positions closed.
    """
    from agent.data_fetcher import fetch_batch_realtime
    closed = 0
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT id, ticker, direction, entry_price,
                       COALESCE(shares_remaining, shares, 1) as shares_rem,
                       COALESCE(partial_pnl_dollar, 0) as partial_pnl
                FROM paper_trades WHERE status='OPEN'
            """).fetchall()

            if not rows:
                return 0

            tickers = list({r["ticker"] for r in rows})
            try:
                prices = fetch_batch_realtime(tickers)
            except Exception:
                prices = {}

            for row in rows:
                ticker = row["ticker"]
                df = prices.get(ticker)
                ep = float(df.iloc[-1]["Close"]) if (df is not None and not df.empty) else float(row["entry_price"])
                _record_close(
                    c, row["id"], ep, "EOD_HARD_CLOSE_3:45PM",
                    float(row["entry_price"]), row["direction"],
                    int(row["shares_rem"]), float(row["partial_pnl"]),
                )
                closed += 1
            c.commit()

    if closed:
        logger.info(f"[PAPER] EOD hard close: {closed} positions closed at 3:45 PM ET")
        _trigger_paper_feedback()
    return closed


def _trigger_paper_feedback() -> None:
    try:
        from agent.adaptive_filter import update_from_paper_trades
        stats = _build_paper_stats()
        if stats["overall"]["total"] >= 5:
            update_from_paper_trades(stats)
    except Exception as e:
        logger.debug(f"[PAPER] Filter feedback skipped: {e}")


def _build_paper_stats() -> dict:
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
        if conf is None: return "unknown"
        if conf < 50:    return "<50"
        if conf < 60:    return "50-60"
        if conf < 70:    return "60-70"
        if conf < 80:    return "70-80"
        return "80+"

    conf_buckets: dict = {}
    for t in trades:
        band = _conf_band(t.get("confidence"))
        conf_buckets.setdefault(band, []).append(t)

    return {
        "overall":       _stats(trades),
        "by_direction":  _breakdown("direction"),
        "by_session":    _breakdown("session"),
        "by_regime":     _breakdown("regime"),
        "by_vwap_event": _breakdown("vwap_event"),
        "by_rsi_zone":   _breakdown("rsi_zone"),
        "by_entry_type": _breakdown("entry_type"),
        "by_confidence": {k: _stats(v) for k, v in conf_buckets.items()},
        "by_sector_trend": {},
    }


# ── Query functions ───────────────────────────────────────────────────────────

def get_daily_pnl(days: int = 14) -> list[dict]:
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
            rows = c.execute(
                "SELECT * FROM paper_trades WHERE status='OPEN' ORDER BY id DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def get_closed_trades(limit: int = 50) -> list[dict]:
    with _lock:
        with _conn() as c:
            rows = c.execute(
                "SELECT * FROM paper_trades WHERE status='CLOSED' ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def get_summary() -> dict:
    with _lock:
        with _conn() as c:
            closed = c.execute(
                "SELECT pnl_pct, pnl_dollar, direction FROM paper_trades WHERE status='CLOSED'"
            ).fetchall()
            open_count = c.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE status='OPEN'"
            ).fetchone()[0]

    total     = len(closed)
    wins      = sum(1 for r in closed if (r["pnl_pct"] or 0) > 0)
    losses    = sum(1 for r in closed if (r["pnl_pct"] or 0) <= 0)
    pnls      = [r["pnl_pct"] for r in closed if r["pnl_pct"] is not None]
    dollars   = [r["pnl_dollar"] for r in closed if r["pnl_dollar"] is not None]
    avg_pnl   = round(sum(pnls) / len(pnls), 3) if pnls else 0.0
    total_pnl = round(sum(pnls), 2)
    total_dollar_pnl = round(sum(dollars), 2) if dollars else 0.0

    return {
        "open":             open_count,
        "closed":           total,
        "wins":             wins,
        "losses":           losses,
        "win_rate":         round(wins / total * 100, 1) if total > 0 else 0.0,
        "avg_pnl":          avg_pnl,
        "total_pnl":        total_pnl,
        "total_dollar_pnl": total_dollar_pnl,
        "max_concurrent":   _MAX_CONCURRENT_TRADES,
    }


def get_equity_curve(days: int = 30) -> list[dict]:
    conn = _conn()
    try:
        rows = conn.execute("""
            SELECT
                date(closed_at) as trade_date,
                ROUND(SUM(COALESCE(pnl_dollar, 0)), 2) as day_pnl
            FROM paper_trades
            WHERE status='CLOSED' AND closed_at >= date('now', ?)
            GROUP BY date(closed_at)
            ORDER BY trade_date ASC
        """, (f'-{days} days',)).fetchall()
        cumulative = 0.0
        result = []
        for r in rows:
            cumulative = round(cumulative + (r["day_pnl"] or 0), 2)
            result.append({"date": r["trade_date"], "day_pnl": r["day_pnl"], "cumulative": cumulative})
        return result
    finally:
        conn.close()


def get_weekly_pnl() -> list[dict]:
    conn = _conn()
    try:
        rows = conn.execute("""
            SELECT
                strftime('%Y-W%W', closed_at) as week,
                COUNT(*) as total,
                SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                ROUND(SUM(COALESCE(pnl_dollar, 0)), 2) as pnl_dollar,
                ROUND(SUM(pnl_pct), 2) as pnl_pct
            FROM paper_trades
            WHERE status='CLOSED' AND closed_at >= date('now', '-84 days')
            GROUP BY week
            ORDER BY week DESC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_ticker_pnl() -> list[dict]:
    conn = _conn()
    try:
        rows = conn.execute("""
            SELECT
                ticker,
                COUNT(*) as total,
                SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                ROUND(SUM(COALESCE(pnl_dollar, 0)), 2) as pnl_dollar,
                ROUND(AVG(pnl_pct), 2) as avg_pnl_pct,
                ROUND(SUM(pnl_pct), 2) as total_pnl_pct
            FROM paper_trades
            WHERE status='CLOSED'
            GROUP BY ticker
            ORDER BY total DESC
            LIMIT 20
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
