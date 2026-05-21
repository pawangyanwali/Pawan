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

_PAPER_MIN_CONF          = 25.0  # floor confidence for paper trade data collection
_FALLBACK_MIN_CONFIDENCE = 25.0  # used if adaptive filter is unavailable
_MAX_BARS_HELD_SCALP     = 20   # 20-min hard close for scalps (PRD 6.3)
_MAX_BARS_HELD_INTRADAY  = 90   # 90-min hard close for intraday (PRD 6.3)
_MAX_CONCURRENT_TRADES   = 20   # Paper sim: high cap so every signal gets a trade and generates learning data


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH), timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")   # concurrent readers + writer, no locking conflicts
    return c

def _conn_ro() -> sqlite3.Connection:
    """Read-only connection — never blocked by writer threads holding _lock."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH), timeout=5, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA query_only=ON")
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
    """
    Return the minimum confidence required to open a paper trade.

    Paper trading is the system's DATA COLLECTION layer — it needs to capture
    as many signal outcomes as possible so the adaptive filter and ML models
    can learn.  We therefore use a fixed 45% floor rather than the adaptive
    filter's dynamic_threshold (which governs LIVE trading recommendations).

    The adaptive filter's dynamic_threshold is intentionally NOT used here:
      - It starts at 55–65% and can rise further as it learns
      - At 57%+, it would block 80%+ of scanner signals, starving the learner
      - The adaptive filter should OBSERVE 45-55% confidence trades to decide
        whether those contexts are worth blocking — it can't learn without data
    """
    return _PAPER_MIN_CONF


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
    trading_tier:     str   = "REGULAR",
) -> Optional[int]:
    """
    Open a paper trade when all PRD entry gates pass.
    Returns trade id or None.
    """
    if direction not in ("BUY", "SELL"):
        return None

    min_conf = _get_min_confidence()
    if confidence < min_conf:
        logger.debug(f"[PAPER] {ticker} skip: conf {confidence:.0f}% < floor {min_conf:.0f}%")
        return None

    # ── Paper trading is PURE DATA COLLECTION — no blocking on time/session ──
    # Every signal ≥45% confidence must get a trade so the system can learn
    # whether it was right or wrong.  No dead zone, no lunch block, no sector
    # cap, no circuit breaker, no profit-protect mode.  The only limits are:
    #   • one open position per ticker (enforced in the DB transaction below)
    #   • global concurrent cap (enforced in the DB transaction below)
    # Size still scales with R:R so high-conviction setups get more weight.
    rr_mult = round(min(1.0, max(0.20, rr_ratio / 2.0)), 2) if rr_ratio > 0 else 0.20
    effective_size_mult = round(size_mult * rr_mult, 2)
    if effective_size_mult <= 0:
        effective_size_mult = 0.20   # minimum 20% rather than skipping entirely

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
                logger.debug(f"[PAPER] {ticker} skip: already has open trade #{existing[0]}")
                return None

            open_count = c.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE status='OPEN'"
            ).fetchone()[0]
            if open_count >= _MAX_CONCURRENT_TRADES:
                logger.debug(f"[PAPER] {ticker} skip: max concurrent trades ({_MAX_CONCURRENT_TRADES}) reached")
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


def update_open_trades(ticker: str, df, current_price: float,
                       bar_high: float = 0.0, bar_low: float = 0.0) -> None:
    """
    PRD-compliant position management:
      1. Hard close at 3:45 PM ET (EOD rule)
      2. T1 partial exit at 1R → lock in 50%, move stop to breakeven
      3. T2 full exit at 2R → close remaining position
      4. Stop hit → close remaining shares
      5. Time stop: 20-bar scalp or 90-bar intraday
      6. Exit signal analysis (MACD, RSI reversal, etc.)
    """
    from agent.market_hours import is_hard_close_window, is_closing_caution

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

                entry           = float(row["entry_price"] or current_price)
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
                # is_scalp is based solely on entry_type — not bar count, to avoid
                # misclassifying WAIT_RETEST trades that happen to be <20 bars old
                is_scalp = entry_type in ("IMMEDIATE", "SCALP")
                max_bars = _MAX_BARS_HELD_SCALP if is_scalp else _MAX_BARS_HELD_INTRADAY

                ep           = current_price
                exit_reason  = None
                close_shares = 0
                is_partial   = False

                # Intrabar high/low for stop/target detection.  When available,
                # using bar_high/bar_low catches moves that close back inside the
                # range — e.g. a spike through target that doesn't hold the close.
                # Fall back to ep (close) when bar data is unavailable.
                _hi = bar_high if bar_high > 0 else ep
                _lo = bar_low  if bar_low  > 0 else ep

                # unrealized P&L % for smart EOD decisions
                pnl_pct_now = (
                    (ep - entry) / entry * 100 if direction == "BUY"
                    else (entry - ep) / entry * 100
                )

                # ── 1. Hard close at 3:45 PM ET (PRD — non-overridable) ────────
                if is_hard_close_window():
                    exit_reason  = "EOD_HARD_CLOSE_3:45PM"
                    close_shares = shares_rem

                # ── 1.5. Smart EOD pre-close during CLOSING_CAUTION (3:30–3:44) ─
                elif is_closing_caution():
                    momentum_ok = _eod_momentum_favors(df, direction)
                    if pnl_pct_now >= 0.5:
                        if momentum_ok:
                            # Strong winner, momentum still in our favor — trail stop
                            tight_stop = (
                                round(ep * (1 - 0.003), 4) if direction == "BUY"
                                else round(ep * (1 + 0.003), 4)
                            )
                            improves = (
                                (direction == "BUY"  and tight_stop > stop_current) or
                                (direction == "SELL" and tight_stop < stop_current)
                            )
                            if improves:
                                c.execute("UPDATE paper_trades SET stop=? WHERE id=?",
                                          (tight_stop, row["id"]))
                                stop_current = tight_stop
                                logger.info(
                                    f"[PAPER] EOD_TRAIL {direction} {ticker} @ ${ep:.2f} "
                                    f"gain={pnl_pct_now:+.2f}% mom=✓ → stop ${tight_stop:.2f}"
                                )
                            # Fall through — T1/T2/stop checks still run with the new stop
                        else:
                            # Momentum reversing — lock the gain now
                            exit_reason  = "EOD_LOCK_PROFIT_REVERSAL"
                            close_shares = shares_rem
                    elif pnl_pct_now >= 0.1:
                        # Small winner — take it, not worth the risk so close to EOD
                        exit_reason  = "EOD_LOCK_PROFIT"
                        close_shares = shares_rem
                    elif pnl_pct_now >= -0.3:
                        # Breakeven zone — exit
                        exit_reason  = "EOD_BREAKEVEN_EXIT"
                        close_shares = shares_rem
                    else:
                        # Meaningful loss
                        if momentum_ok:
                            # Still moving in our direction — tighten stop, hope for recovery
                            tight_stop = (
                                round(ep * (1 - 0.002), 4) if direction == "BUY"
                                else round(ep * (1 + 0.002), 4)
                            )
                            improves = (
                                (direction == "BUY"  and tight_stop > stop_current) or
                                (direction == "SELL" and tight_stop < stop_current)
                            )
                            if improves:
                                c.execute("UPDATE paper_trades SET stop=? WHERE id=?",
                                          (tight_stop, row["id"]))
                                stop_current = tight_stop
                                logger.info(
                                    f"[PAPER] EOD_TIGHT_RECOVERY {direction} {ticker} "
                                    f"@ ${ep:.2f} pnl={pnl_pct_now:+.2f}% mom=✓ → stop ${tight_stop:.2f}"
                                )
                            else:
                                exit_reason  = "EOD_CUT_LOSS"
                                close_shares = shares_rem
                        else:
                            # Momentum against us — cut the loss
                            exit_reason  = "EOD_CUT_LOSS"
                            close_shares = shares_rem

                # ── 2. T1 partial exit (1R profit) — if not already hit ────────
                if not exit_reason and not t1_hit and t1_price > 0:
                    t1_hit_now = (
                        (direction == "BUY"  and _hi >= t1_price) or
                        (direction == "SELL" and _lo <= t1_price)
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
                        # Breakeven stop: above entry for BUY, below for SELL (locks in ~breakeven)
                        be_stop = round(entry + 0.02, 4) if direction == "BUY" else round(entry - 0.02, 4)
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
                                          shares_total, partial_pnl, shares_total)
                            closed_any = True
                            won_any    = partial_pnl > 0
                            continue

                # ── 3. T2 full exit (2R profit) — only if T1 already hit ──────
                if not exit_reason and t1_hit and t2_price > 0:
                    t2_hit_now = (
                        (direction == "BUY"  and _hi >= t2_price) or
                        (direction == "SELL" and _lo <= t2_price)
                    )
                    if t2_hit_now:
                        exit_reason  = "TARGET_T2"
                        close_shares = shares_rem
                        ep           = t2_price  # fill at T2

                # ── 4. Stop hit ────────────────────────────────────────────────
                if not exit_reason:
                    stop_hit = (
                        (direction == "BUY"  and _lo <= stop_current) or
                        (direction == "SELL" and _hi >= stop_current)
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
                                  close_shares, partial_pnl, shares_total)
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
    total_shares: int   = 0,      # original position size for correct pnl_pct
) -> None:
    """Write the final closed state for a trade record."""
    ep = exit_price
    if direction == "BUY":
        pnl_dollar = (ep - entry) * close_shares + partial_pnl
    else:
        pnl_dollar = (entry - ep) * close_shares + partial_pnl

    # pnl_pct = actual return on the FULL initial position (not just remaining shares).
    # Using total_shares avoids inflating the % when a partial T1 exit has occurred.
    denom_shares = total_shares if total_shares > 0 else close_shares
    pnl_pct = pnl_dollar / (entry * denom_shares + 0.01) * 100

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


def close_all_positions_eod(reason: str = "EOD_HARD_CLOSE_3:45PM") -> int:
    """
    Force-close ALL open paper trades at current price.
    Called at 3:45 PM ET hard close, after-hours, or on startup when market is closed.
    Returns number of positions closed.
    """
    from agent.data_fetcher import fetch_batch_realtime

    # Step 1: read open rows without holding the lock during the API call
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT id, ticker, direction, entry_price,
                       COALESCE(shares_remaining, shares, 1) as shares_rem,
                       COALESCE(partial_pnl_dollar, 0) as partial_pnl,
                       COALESCE(shares, 1) as shares_total
                FROM paper_trades WHERE status='OPEN'
            """).fetchall()
            rows = list(rows)

    if not rows:
        return 0

    # Step 2: fetch prices outside the lock so reads are never blocked
    tickers = list({r["ticker"] for r in rows})
    try:
        prices = fetch_batch_realtime(tickers)
    except Exception:
        prices = {}

    # Step 3: write closes under the lock
    closed = 0
    with _lock:
        with _conn() as c:
            for row in rows:
                ticker = row["ticker"]
                df = prices.get(ticker)
                ep = float(df.iloc[-1]["Close"]) if (df is not None and not df.empty) else float(row["entry_price"])
                _record_close(
                    c, row["id"], ep, reason,
                    float(row["entry_price"]), row["direction"],
                    int(row["shares_rem"]), float(row["partial_pnl"]),
                    int(row["shares_total"]),
                )
                closed += 1
            c.commit()

    if closed:
        logger.info(f"[PAPER] Force-close ({reason}): {closed} positions closed")
        _trigger_paper_feedback()
    return closed


def close_stale_positions() -> int:
    """
    Called at startup and after-hours to sweep any positions that were left
    open when the market closed (scanner may not have been running at 3:45 PM).
    Uses entry price as exit price when live prices are unavailable.
    Returns number of positions closed.
    """
    from agent.market_hours import is_after_hours, no_new_entries
    if not no_new_entries():
        return 0  # Market is open — don't sweep

    with _lock:
        with _conn() as c:
            count = c.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE status='OPEN'"
            ).fetchone()[0]

    if count == 0:
        return 0

    logger.warning(
        f"[PAPER] Found {count} stale open position(s) while market is closed — force-closing"
    )
    return close_all_positions_eod(reason="STALE_MARKET_CLOSED")


def _eod_momentum_favors(df, direction: str, n_bars: int = 4) -> bool:
    """
    Returns True if the last n_bars of price action favor the trade direction.
    Uses simple slope: if closing prices are trending up → favors BUY; down → favors SELL.
    Also checks that the most recent bar is continuing in that direction.
    """
    try:
        if df is None or len(df) < n_bars:
            return False
        closes = [float(df.iloc[-i]["Close"]) for i in range(n_bars, 0, -1)]
        # Overall slope across the window
        slope = closes[-1] - closes[0]
        # Last bar direction (most recent confirmation)
        last_bar_up = closes[-1] >= closes[-2]
        if direction == "BUY":
            return slope > 0 and last_bar_up
        else:  # SELL
            return slope < 0 and not last_bar_up
    except Exception:
        return False


def _apply_eod_action(
    c,
    row_id:     int,
    ep:         float,
    entry:      float,
    direction:  str,
    stop_curr:  float,
    shares_rem: int,
    shares_tot: int,
    partial:    float,
    pnl_pct:    float,
    momentum_ok: bool,
    ticker:     str,
) -> bool:
    """
    Core EOD decision logic shared by update_open_trades() and smart_eod_review().
    Returns True if position was closed or stop was tightened (i.e., acted on).

    Decision matrix:
      pnl >= 0.5% + momentum OK  → tighten trailing stop (let winner run, profit protected)
      pnl >= 0.5% + momentum BAD → exit now (lock the gain before reversal)
      pnl  0.1–0.5%              → exit (small win; not worth overnight or reversal risk)
      pnl -0.3–0.1%              → exit (breakeven zone; no edge left today)
      pnl < -0.3% + momentum OK  → tighten stop very close (recovery attempt)
      pnl < -0.3% + momentum BAD → exit immediately (cut the loss)
    """
    acted = False

    if pnl_pct >= 0.5:
        if momentum_ok:
            # Strong winner with momentum — trail stop to lock in most of the gain
            tight_stop = (
                round(ep * (1 - 0.003), 4) if direction == "BUY"
                else round(ep * (1 + 0.003), 4)
            )
            improves = (
                (direction == "BUY"  and tight_stop > stop_curr) or
                (direction == "SELL" and tight_stop < stop_curr)
            )
            if improves:
                c.execute("UPDATE paper_trades SET stop=? WHERE id=?", (tight_stop, row_id))
                logger.info(
                    f"[PAPER] EOD_TRAIL {direction} {ticker} @ ${ep:.2f} "
                    f"gain={pnl_pct:+.2f}% mom=✓ → stop ${tight_stop:.2f}"
                )
                acted = True
        else:
            # Momentum reversing — take the profit before it evaporates
            _record_close(c, row_id, ep, "EOD_LOCK_PROFIT_REVERSAL",
                          entry, direction, shares_rem, partial, shares_tot)
            logger.info(
                f"[PAPER] EOD_LOCK_PROFIT_REVERSAL {direction} {ticker} @ ${ep:.2f} "
                f"gain={pnl_pct:+.2f}% mom=✗ — taking profit on reversal"
            )
            acted = True

    elif pnl_pct >= 0.1:
        # Small winner — lock it in regardless of momentum (not worth overnight risk)
        _record_close(c, row_id, ep, "EOD_LOCK_PROFIT",
                      entry, direction, shares_rem, partial, shares_tot)
        logger.info(
            f"[PAPER] EOD_LOCK_PROFIT {direction} {ticker} @ ${ep:.2f} gain={pnl_pct:+.2f}%"
        )
        acted = True

    elif pnl_pct >= -0.3:
        # Breakeven zone — exit, no edge left this close to market end
        _record_close(c, row_id, ep, "EOD_BREAKEVEN_EXIT",
                      entry, direction, shares_rem, partial, shares_tot)
        logger.info(
            f"[PAPER] EOD_BREAKEVEN_EXIT {direction} {ticker} @ ${ep:.2f} pnl={pnl_pct:+.2f}%"
        )
        acted = True

    else:
        # Meaningful loss
        if momentum_ok:
            # Price still moving in our favor — tighten stop very close and hope for recovery
            tight_stop = (
                round(ep * (1 - 0.002), 4) if direction == "BUY"
                else round(ep * (1 + 0.002), 4)
            )
            improves = (
                (direction == "BUY"  and tight_stop > stop_curr) or
                (direction == "SELL" and tight_stop < stop_curr)
            )
            if improves:
                c.execute("UPDATE paper_trades SET stop=? WHERE id=?", (tight_stop, row_id))
                logger.info(
                    f"[PAPER] EOD_TIGHT_RECOVERY {direction} {ticker} @ ${ep:.2f} "
                    f"pnl={pnl_pct:+.2f}% mom=✓ → stop ${tight_stop:.2f}"
                )
                acted = True
            else:
                # Stop already tight enough — force close to limit further damage
                _record_close(c, row_id, ep, "EOD_CUT_LOSS",
                              entry, direction, shares_rem, partial, shares_tot)
                logger.info(
                    f"[PAPER] EOD_CUT_LOSS {direction} {ticker} @ ${ep:.2f} pnl={pnl_pct:+.2f}%"
                )
                acted = True
        else:
            # Momentum against us — cut the loss immediately
            _record_close(c, row_id, ep, "EOD_CUT_LOSS",
                          entry, direction, shares_rem, partial, shares_tot)
            logger.info(
                f"[PAPER] EOD_CUT_LOSS {direction} {ticker} @ ${ep:.2f} "
                f"pnl={pnl_pct:+.2f}% mom=✗"
            )
            acted = True

    return acted


def smart_eod_review() -> int:
    """
    Called every scan during CLOSING_CAUTION (3:30–3:44 PM ET).
    Fetches current prices + recent bars for ALL open positions and applies
    momentum-aware pre-close rules so no position carries into the next day.

    Returns number of positions acted on (tightened or closed).
    """
    from agent.data_fetcher import fetch_batch_realtime
    from agent.market_hours import is_closing_caution
    if not is_closing_caution():
        return 0

    # Step 1: read open rows without holding the lock during the API call
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT id, ticker, direction, entry_price, stop,
                       COALESCE(shares_remaining, shares, 1) as shares_rem,
                       COALESCE(shares, 1) as shares_total,
                       COALESCE(partial_pnl_dollar, 0) as partial_pnl
                FROM paper_trades WHERE status='OPEN'
            """).fetchall()
            rows = list(rows)

    if not rows:
        return 0

    # Step 2: fetch prices outside the lock so reads are never blocked
    tickers = list({r["ticker"] for r in rows})
    try:
        prices = fetch_batch_realtime(tickers)
    except Exception:
        prices = {}

    # Step 3: apply EOD actions under the lock
    acted = 0
    with _lock:
        with _conn() as c:
            for row in rows:
                ticker     = row["ticker"]
                direction  = row["direction"]
                entry      = float(row["entry_price"])
                stop_curr  = float(row["stop"])
                shares_rem = int(row["shares_rem"])
                shares_tot = int(row["shares_total"])
                partial    = float(row["partial_pnl"])

                df = prices.get(ticker)
                ep = float(df.iloc[-1]["Close"]) if (df is not None and not df.empty) else None
                if ep is None:
                    continue

                pnl_pct = (
                    (ep - entry) / entry * 100 if direction == "BUY"
                    else (entry - ep) / entry * 100
                )
                momentum_ok = _eod_momentum_favors(df, direction)

                if _apply_eod_action(
                    c, row["id"], ep, entry, direction,
                    stop_curr, shares_rem, shares_tot, partial,
                    pnl_pct, momentum_ok, ticker,
                ):
                    acted += 1

            c.commit()

    if acted:
        _trigger_paper_feedback()
    return acted


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
    conn = _conn_ro()
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
    conn = _conn_ro()
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


def rt_check_positions(ticker: str, last_price: float) -> list[str]:
    """
    Lightweight real-time stop/T1/T2 check using Schwab streaming last price.
    Called every ~5s by the RT monitor — NO df required, NO EOD logic.
    Handles: stop hit, T1 partial exit at 1R, T2 full exit at 2R.
    Full update_open_trades() (EOD, momentum, trailing) still runs every 60s scan.
    Returns list of exit reasons for any positions closed.
    """
    actions: list[str] = []
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT id, direction, entry_price, stop, t1_price, t2_price,
                       COALESCE(t1_hit, 0) as t1_hit,
                       COALESCE(breakeven_set, 0) as breakeven_set,
                       COALESCE(shares, 1) as shares,
                       COALESCE(shares_remaining, shares, 1) as shares_remaining,
                       COALESCE(partial_pnl_dollar, 0) as partial_pnl_dollar
                FROM paper_trades WHERE ticker=? AND status='OPEN'
            """, (ticker,)).fetchall()

            for row in rows:
                d        = row["direction"]
                entry    = float(row["entry_price"])
                stp      = float(row["stop"])
                t1       = float(row["t1_price"] or 0)
                t2       = float(row["t2_price"] or 0)
                t1_hit   = bool(row["t1_hit"])
                shares_r = int(row["shares_remaining"])
                partial  = float(row["partial_pnl_dollar"])

                exit_reason = None

                # ── Stop hit ────────────────────────────────────────────────────
                if (d == "BUY" and last_price <= stp) or (d == "SELL" and last_price >= stp):
                    exit_reason = "STOP_HIT_BREAKEVEN" if row["breakeven_set"] else "STOP_HIT"
                    _record_close(c, row["id"], last_price, exit_reason, entry, d,
                                  int(row["shares"]), partial, shares_r)
                    actions.append(exit_reason)
                    logger.info(
                        f"[PAPER-RT] {exit_reason} {d} {ticker} @ ${last_price:.2f} "
                        f"(real-time stop check)"
                    )
                    continue

                # ── T2 full exit (only if T1 already hit) ───────────────────────
                if t1_hit and t2 > 0:
                    if (d == "BUY" and last_price >= t2) or (d == "SELL" and last_price <= t2):
                        _record_close(c, row["id"], t2, "TARGET_T2", entry, d,
                                      int(row["shares"]), partial, shares_r)
                        actions.append("TARGET_T2")
                        logger.info(
                            f"[PAPER-RT] TARGET_T2 {d} {ticker} @ ${t2:.2f} (real-time)"
                        )
                        continue

                # ── T1 partial exit (1R) ─────────────────────────────────────────
                if not t1_hit and t1 > 0:
                    if (d == "BUY" and last_price >= t1) or (d == "SELL" and last_price <= t1):
                        partial_sh  = max(1, shares_r // 2)
                        t1_pnl      = ((t1 - entry) if d == "BUY" else (entry - t1)) * partial_sh
                        new_partial = partial + t1_pnl
                        new_rem     = shares_r - partial_sh
                        be_stop     = round(entry + 0.02, 4) if d == "BUY" else round(entry - 0.02, 4)
                        c.execute("""
                            UPDATE paper_trades
                            SET t1_hit=1, breakeven_set=1, stop=?,
                                partial_pnl_dollar=?, shares_remaining=?
                            WHERE id=?
                        """, (be_stop, round(new_partial, 2), new_rem, row["id"]))
                        actions.append("T1_HIT_RT")
                        logger.info(
                            f"[PAPER-RT] T1 {d} {ticker} @ ${t1:.2f} "
                            f"partial {partial_sh}sh locked ${t1_pnl:+.2f} "
                            f"stop → breakeven ${be_stop:.2f} (real-time)"
                        )
            c.commit()
    return actions


def get_open_trades() -> list[dict]:
    with _conn_ro() as c:
        rows = c.execute(
            "SELECT * FROM paper_trades WHERE status='OPEN' ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_closed_trades(limit: int = 50) -> list[dict]:
    with _conn_ro() as c:
        rows = c.execute(
            "SELECT * FROM paper_trades WHERE status='CLOSED' ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_summary() -> dict:
    with _conn_ro() as c:
        closed = c.execute(
            "SELECT pnl_pct, pnl_dollar, direction FROM paper_trades WHERE status='CLOSED'"
        ).fetchall()
        open_count = c.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE status='OPEN'"
        ).fetchone()[0]

    total    = len(closed)
    # Win = positive dollar P&L (source of truth — not pnl_pct which can be inflated)
    wins     = sum(1 for r in closed if (r["pnl_dollar"] or 0) > 0)
    losses   = total - wins
    dollars  = [r["pnl_dollar"] for r in closed if r["pnl_dollar"] is not None]
    pnls     = [r["pnl_pct"]    for r in closed if r["pnl_pct"]    is not None]
    total_dollar_pnl = round(sum(dollars), 2) if dollars else 0.0
    avg_pnl_pct      = round(sum(pnls) / len(pnls), 3) if pnls else 0.0

    return {
        "open":             open_count,
        "closed":           total,
        "wins":             wins,
        "losses":           losses,
        "win_rate":         round(wins / total * 100, 1) if total > 0 else 0.0,
        "avg_pnl":          avg_pnl_pct,       # avg % per trade (display metric)
        "total_pnl":        avg_pnl_pct,       # keep key for compat — now equals avg, not sum
        "total_dollar_pnl": total_dollar_pnl,  # actual net dollar P&L
        "max_concurrent":   _MAX_CONCURRENT_TRADES,
    }


def get_equity_curve(days: int = 30) -> list[dict]:
    conn = _conn_ro()
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
    conn = _conn_ro()
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
    conn = _conn_ro()
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
