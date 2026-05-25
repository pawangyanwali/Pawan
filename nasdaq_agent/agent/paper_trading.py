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
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent.exit_signals import analyse_exits, ExitAnalysis
from agent.db import get_conn

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "paper_trades.db"
_lock    = threading.Lock()

_PAPER_MIN_CONF          = 25.0  # floor confidence for paper trade data collection
_FALLBACK_MIN_CONFIDENCE = 25.0  # used if adaptive filter is unavailable
_MAX_BARS_HELD_SCALP     = 20   # 20-min hard close for scalps (PRD 6.3)
_MAX_BARS_HELD_INTRADAY  = 90   # 90-min hard close for intraday (PRD 6.3)
# Loaded from config at runtime so .env changes take effect without code edits
def _max_concurrent() -> int:
    from config import PAPER_MAX_OPEN_TRADES
    return PAPER_MAX_OPEN_TRADES


def _conn():
    return get_conn(_DB_PATH)

def _conn_ro():
    return get_conn(_DB_PATH, read_only=True)


# ── Trade-event callbacks ─────────────────────────────────────────────────────
# Register with register_trade_callback(fn).  Called with ("open"|"close", ticker)
# from a background thread — callbacks must be non-blocking.

_trade_callbacks: list = []

def register_trade_callback(fn) -> None:
    """Register a function(event, ticker) called immediately on open/close."""
    _trade_callbacks.append(fn)

def _fire_trade_event(event: str, ticker: str) -> None:
    for fn in _trade_callbacks:
        try:
            fn(event, ticker)
        except Exception:
            pass


def _run_ddl_autocommit(statements: list[str]) -> None:
    """Run DDL statements each in their own autocommit connection so they are
    committed independently of any surrounding transaction."""
    from agent.db import using_postgres, _get_pool
    if using_postgres():
        pool = _get_pool()
        raw = pool.getconn()
        try:
            raw.autocommit = True
            with raw.cursor() as cur:
                for sql in statements:
                    try:
                        cur.execute(sql)
                    except Exception as e:
                        logger.warning(f"[DB] DDL warning: {e}")
        finally:
            pool.putconn(raw)


def init_db() -> None:
    from agent.db import using_postgres

    # ── DDL statements (always idempotent) ──────────────────────────────────
    create_paper_trades = """
        CREATE TABLE IF NOT EXISTS paper_trades (
            id                  SERIAL PRIMARY KEY,
            opened_at           TEXT    NOT NULL,
            closed_at           TEXT,
            ticker              TEXT    NOT NULL,
            direction           TEXT    NOT NULL,
            entry_price         DOUBLE PRECISION NOT NULL,
            target              DOUBLE PRECISION NOT NULL DEFAULT 0,
            stop                DOUBLE PRECISION NOT NULL DEFAULT 0,
            confidence          DOUBLE PRECISION NOT NULL,
            rr_ratio            DOUBLE PRECISION DEFAULT 0,
            rr_qualifies        INTEGER DEFAULT 0,
            bars_held           INTEGER DEFAULT 0,
            status              TEXT    DEFAULT 'OPEN',
            exit_price          DOUBLE PRECISION,
            exit_reason         TEXT,
            pnl_pct             DOUBLE PRECISION,
            pnl_dollar          DOUBLE PRECISION,
            shares              INTEGER DEFAULT 1,
            session             TEXT    DEFAULT '',
            regime              TEXT    DEFAULT '',
            vwap_event          TEXT    DEFAULT '',
            rsi_zone            TEXT    DEFAULT '',
            entry_type          TEXT    DEFAULT '',
            t1_hit              INTEGER DEFAULT 0,
            t1_price            DOUBLE PRECISION DEFAULT 0,
            t2_price            DOUBLE PRECISION DEFAULT 0,
            breakeven_set       INTEGER DEFAULT 0,
            partial_pnl_dollar  DOUBLE PRECISION DEFAULT 0,
            shares_remaining    INTEGER DEFAULT 0,
            order_flow_score    DOUBLE PRECISION DEFAULT 0,
            size_mult           DOUBLE PRECISION DEFAULT 1.0,
            cost_basis          DOUBLE PRECISION DEFAULT 0,
            algo_name           TEXT    DEFAULT ''
        )
    """ if using_postgres() else """
        CREATE TABLE IF NOT EXISTS paper_trades (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            opened_at           TEXT    NOT NULL,
            closed_at           TEXT,
            ticker              TEXT    NOT NULL,
            direction           TEXT    NOT NULL,
            entry_price         REAL    NOT NULL,
            target              REAL    NOT NULL DEFAULT 0,
            stop                REAL    NOT NULL DEFAULT 0,
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
            t1_hit              INTEGER DEFAULT 0,
            t1_price            REAL    DEFAULT 0,
            t2_price            REAL    DEFAULT 0,
            breakeven_set       INTEGER DEFAULT 0,
            partial_pnl_dollar  REAL    DEFAULT 0,
            shares_remaining    INTEGER DEFAULT 0,
            order_flow_score    REAL    DEFAULT 0,
            size_mult           REAL    DEFAULT 1.0,
            cost_basis          REAL    DEFAULT 0,
            algo_name           TEXT    DEFAULT ''
        )
    """

    create_algo_signal_log = """
        CREATE TABLE IF NOT EXISTS algo_signal_log (
            id           SERIAL PRIMARY KEY,
            logged_at    TEXT              NOT NULL,
            ticker       TEXT              NOT NULL,
            algo         TEXT              NOT NULL,
            direction    TEXT              NOT NULL,
            confidence   DOUBLE PRECISION  DEFAULT 0,
            entry        DOUBLE PRECISION  DEFAULT 0,
            stop         DOUBLE PRECISION  DEFAULT 0,
            target       DOUBLE PRECISION  DEFAULT 0,
            rr           DOUBLE PRECISION  DEFAULT 0,
            trade_opened INTEGER           DEFAULT 0
        )
    """ if using_postgres() else """
        CREATE TABLE IF NOT EXISTS algo_signal_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at    TEXT    NOT NULL,
            ticker       TEXT    NOT NULL,
            algo         TEXT    NOT NULL,
            direction    TEXT    NOT NULL,
            confidence   REAL    DEFAULT 0,
            entry        REAL    DEFAULT 0,
            stop         REAL    DEFAULT 0,
            target       REAL    DEFAULT 0,
            rr           REAL    DEFAULT 0,
            trade_opened INTEGER DEFAULT 0
        )
    """

    create_account_config = """
        CREATE TABLE IF NOT EXISTS account_config (
            id                   INTEGER PRIMARY KEY,
            total_budget         REAL    DEFAULT 50000,
            max_trade_pct        REAL    DEFAULT 5.0,
            max_allocated_pct    REAL    DEFAULT 40.0,
            max_open_trades      INTEGER DEFAULT 10,
            updated_at           TEXT
        )
    """

    if using_postgres():
        create_balance_snapshots = """
            CREATE TABLE IF NOT EXISTS balance_snapshots (
                id              SERIAL PRIMARY KEY,
                snapshot_date   TEXT    NOT NULL UNIQUE,
                closing_equity  REAL    NOT NULL,
                daily_pnl       REAL    DEFAULT 0,
                trade_count     INTEGER DEFAULT 0,
                wins            INTEGER DEFAULT 0,
                losses          INTEGER DEFAULT 0,
                starting_equity REAL    DEFAULT 0
            )
        """
    else:
        create_balance_snapshots = """
            CREATE TABLE IF NOT EXISTS balance_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date   TEXT    NOT NULL UNIQUE,
                closing_equity  REAL    NOT NULL,
                daily_pnl       REAL    DEFAULT 0,
                trade_count     INTEGER DEFAULT 0,
                wins            INTEGER DEFAULT 0,
                losses          INTEGER DEFAULT 0,
                starting_equity REAL    DEFAULT 0
            )
        """

    if using_postgres():
        # Run all DDL via autocommit so each statement commits independently.
        # This prevents a transaction rollback anywhere from wiping the schema.
        # NOTE: ALTER COLUMN TYPE (table rewrites) are intentionally excluded
        # from the boot path — they block the event loop on large tables.
        # The new CREATE TABLE already uses DOUBLE PRECISION; ADD COLUMN also
        # uses DOUBLE PRECISION, so only stale columns need the cast repair.
        # Drop the legacy CHECK constraints that caused NUMERIC(5,4) overflows.
        _ddl = (
            [create_paper_trades, create_account_config, create_balance_snapshots, create_algo_signal_log]
            + [f"ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS {col} {defn}"
               for col, defn in _COLUMN_ADDITIONS]
            + [f"ALTER TABLE paper_trades DROP CONSTRAINT IF EXISTS {con}"
               for con in [
                   "paper_trades_confidence_check", "paper_trades_entry_price_check",
                   "paper_trades_target_check",     "paper_trades_stop_check",
                   "paper_trades_exit_price_check", "paper_trades_pnl_pct_check",
                   "paper_trades_pnl_dollar_check", "paper_trades_rr_ratio_check",
               ]]
        )
        _run_ddl_autocommit(_ddl)
        # Ensure account_config default row (DML — can run in normal transaction)
        with _conn() as c:
            try:
                row = c.execute("SELECT id FROM account_config WHERE id=1").fetchone()
                if not row:
                    from config import PAPER_BUDGET, PAPER_MAX_TRADE_PCT, PAPER_MAX_ALLOCATED_PCT, PAPER_MAX_OPEN_TRADES
                    from datetime import datetime, timezone
                    c.execute(
                        "INSERT INTO account_config (id, total_budget, max_trade_pct, max_allocated_pct, max_open_trades, updated_at) VALUES (1, %s, %s, %s, %s, %s)",
                        (PAPER_BUDGET, PAPER_MAX_TRADE_PCT, PAPER_MAX_ALLOCATED_PCT, PAPER_MAX_OPEN_TRADES,
                         datetime.now(timezone.utc).isoformat()),
                    )
            except Exception:
                pass
    else:
        # SQLite: single transaction is fine
        with _conn() as c:
            c.execute(create_paper_trades)
            c.execute(create_account_config)
            c.execute(create_balance_snapshots)
            c.execute(create_algo_signal_log)
            _migrate_columns(c)
            # Ensure default config row
            try:
                row = c.execute("SELECT id FROM account_config WHERE id=1").fetchone()
                if not row:
                    from config import PAPER_BUDGET, PAPER_MAX_TRADE_PCT, PAPER_MAX_ALLOCATED_PCT, PAPER_MAX_OPEN_TRADES
                    from datetime import datetime, timezone
                    c.execute(
                        "INSERT INTO account_config (id, total_budget, max_trade_pct, max_allocated_pct, max_open_trades, updated_at) VALUES (1, ?, ?, ?, ?, ?)",
                        (PAPER_BUDGET, PAPER_MAX_TRADE_PCT, PAPER_MAX_ALLOCATED_PCT, PAPER_MAX_OPEN_TRADES,
                         datetime.now(timezone.utc).isoformat()),
                    )
            except Exception:
                pass


_COLUMN_ADDITIONS = [
    # Core columns that may be absent in old migrated RDS schemas
    ("target",             "REAL DEFAULT 0"),
    ("stop",               "REAL DEFAULT 0"),
    ("bars_held",          "INTEGER DEFAULT 0"),
    ("status",             "TEXT DEFAULT 'OPEN'"),
    ("exit_price",         "REAL"),
    ("exit_reason",        "TEXT"),
    ("pnl_pct",            "REAL"),
    ("pnl_dollar",         "REAL"),
    # Later additions
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
    ("cost_basis",         "REAL DEFAULT 0"),   # entry_price × shares (allocated capital)
    ("algo_name",          "TEXT DEFAULT ''"),  # algo that triggered the trade ('' = ML)
]


def _migrate_columns(c) -> None:
    """SQLite-only: add missing columns to an existing paper_trades table."""
    existing = {row[1] for row in c.execute("PRAGMA table_info(paper_trades)").fetchall()}
    for col, definition in _COLUMN_ADDITIONS:
        if col not in existing:
            try:
                c.execute(f"ALTER TABLE paper_trades ADD COLUMN {col} {definition}")
            except Exception:
                pass


def _get_min_confidence() -> float:
    """
    Return the minimum confidence required to open a paper trade.

    Paper trading is the DATA COLLECTION layer — the floor is intentionally
    low (25%) so the adaptive filter can observe and learn from low-confidence
    trades. The adaptive filter's dynamic_threshold (55–63%) governs live
    trading recommendations and is NOT used here — doing so would starve the
    learner by blocking 80%+ of signals before any outcome is recorded.

    The floor is read from config so it can be tuned without a code deploy.
    """
    try:
        from config import PAPER_TRADE_MIN_CONFIDENCE
        return float(PAPER_TRADE_MIN_CONFIDENCE)
    except Exception:
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
    algo_name:        str   = "",
) -> Optional[int]:
    """
    Open a paper trade when all PRD entry gates pass.
    Returns trade id or None.
    """
    if direction not in ("BUY", "SELL"):
        return None

    if price <= 0 or stop <= 0:
        logger.debug(f"[PAPER] {ticker} skip: invalid price ({price}) or stop ({stop})")
        return None

    # ── Stop/target geometry validation ─────────────────────────────────────
    # BUY:  stop must be BELOW entry, target must be ABOVE entry.
    # SELL: stop must be ABOVE entry, target must be BELOW entry.
    # Inverted geometry produces negative R:R and corrupts learning data.
    if target > 0:
        if direction == "BUY" and (stop >= price or target <= price):
            logger.debug(
                f"[PAPER] {ticker} BUY geometry invalid: "
                f"entry={price:.2f} stop={stop:.2f} target={target:.2f}"
            )
            return None
        if direction == "SELL" and (stop <= price or target >= price):
            logger.debug(
                f"[PAPER] {ticker} SELL geometry invalid: "
                f"entry={price:.2f} stop={stop:.2f} target={target:.2f}"
            )
            return None

    min_conf = _get_min_confidence()
    if confidence < min_conf:
        logger.debug(f"[PAPER] {ticker} skip: conf {confidence:.0f}% < floor {min_conf:.0f}%")
        return None

    # ── Phase 3: Session gate — hard block for CLOSED, higher bar for extended hours ──
    # Use the live session value from the scanner (passed as `session`).
    # If caller omitted it, fall back to a direct check so the gate is never skipped.
    _live_session = session
    if not _live_session:
        try:
            from agent.market_hours import get_market_session as _gms
            _live_session = _gms()
        except Exception:
            pass

    if _live_session == "CLOSED":
        logger.debug(f"[PAPER] {ticker} skip: market CLOSED — no trades on weekends/overnight")
        return None

    # Extended-hours stop widening: wider stop = smaller shares, less capital at risk
    # on thin ECN spreads (1.5× in pre-market, 2× in after-hours).
    _stop_mult = (
        2.0 if _live_session == "AFTER_HOURS" else
        1.5 if _live_session == "PRE_MARKET"  else
        1.0
    )
    if _stop_mult != 1.0:
        risk_dist_orig = abs(price - stop)
        stop = (
            round(price - risk_dist_orig * _stop_mult, 4) if direction == "BUY"
            else round(price + risk_dist_orig * _stop_mult, 4)
        )

    # Extended-hours confidence floors and tier gate.
    # REGULAR-tier stocks lack the liquidity for AH/PM trades; HIGH/MODERATE allowed with higher bar.
    _EXT_CONF_FLOOR: dict[str, float] = {"HIGH": 70.0, "MODERATE": 60.0}
    if _live_session in ("PRE_MARKET", "AFTER_HOURS"):
        _ext_floor = _EXT_CONF_FLOOR.get(trading_tier)
        if _ext_floor is None:
            logger.debug(
                f"[PAPER] {ticker} skip: REGULAR-tier in {_live_session} — insufficient liquidity"
            )
            return None
        if confidence < _ext_floor:
            logger.debug(
                f"[PAPER] {ticker} skip: conf {confidence:.0f}% < ext-hours floor {_ext_floor:.0f}%"
                f" (tier={trading_tier}, sess={_live_session})"
            )
            return None

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
                logger.debug(f"[PAPER] {ticker} skip: already has open trade #{existing['id']}")
                return None

            open_count = c.execute(
                "SELECT COUNT(*) AS n FROM paper_trades WHERE status='OPEN'"
            ).fetchone()["n"]
            if open_count >= _max_concurrent():
                logger.debug(f"[PAPER] {ticker} skip: max concurrent trades ({_max_concurrent()}) reached")
                return None

            # ── Capital gate: check available capital before sizing ────────────
            cfg = c.execute("SELECT total_budget, max_trade_pct, max_allocated_pct FROM account_config WHERE id=1").fetchone()
            if cfg:
                _budget       = float(cfg["total_budget"])
                _max_trade_v  = _budget * float(cfg["max_trade_pct"])  / 100.0
                _max_alloc_v  = _budget * float(cfg["max_allocated_pct"]) / 100.0
            else:
                from config import PAPER_BUDGET, PAPER_MAX_TRADE_PCT, PAPER_MAX_ALLOCATED_PCT
                _budget      = PAPER_BUDGET
                _max_trade_v = _budget * PAPER_MAX_TRADE_PCT  / 100.0
                _max_alloc_v = _budget * PAPER_MAX_ALLOCATED_PCT / 100.0

            realized_pnl_row = c.execute(
                "SELECT COALESCE(SUM(pnl_dollar),0) AS rpnl FROM paper_trades WHERE status='CLOSED'"
            ).fetchone()
            realized_pnl = float(realized_pnl_row["rpnl"]) if realized_pnl_row else 0.0

            try:
                allocated_row = c.execute(
                    "SELECT COALESCE(SUM(COALESCE(cost_basis, entry_price * shares)),0) AS alloc FROM paper_trades WHERE status='OPEN'"
                ).fetchone()
            except Exception:
                allocated_row = c.execute(
                    "SELECT COALESCE(SUM(entry_price * shares),0) AS alloc FROM paper_trades WHERE status='OPEN'"
                ).fetchone()
            allocated = float(allocated_row["alloc"]) if allocated_row else 0.0

            available = _budget + realized_pnl - allocated
            if available < price:
                logger.debug(f"[PAPER] {ticker} skip: insufficient capital (avail ${available:.0f} < ${price:.2f})")
                return None
            if allocated >= _max_alloc_v:
                logger.debug(f"[PAPER] {ticker} skip: max allocated capital reached (${allocated:.0f} >= ${_max_alloc_v:.0f})")
                return None

            # Refine shares within capital constraints
            cost_basis_per_share = price
            max_by_trade   = max(1, int(_max_trade_v / cost_basis_per_share))
            max_by_capital = max(1, int(available    / cost_basis_per_share))
            shares = min(shares, max_by_trade, max_by_capital)

            cur = c.execute("""
                INSERT INTO paper_trades
                  (opened_at, ticker, direction, entry_price, target, stop,
                   confidence, rr_ratio, rr_qualifies, shares, shares_remaining,
                   session, regime, vwap_event, rsi_zone, entry_type,
                   t1_price, t2_price, order_flow_score, size_mult, cost_basis,
                   algo_name)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                ticker, direction,
                round(price, 4), round(target, 4), round(stop, 4),
                round(confidence, 2), round(rr_ratio, 2), int(rr_qualifies),
                shares, shares,  # shares_remaining starts = shares
                session, regime, vwap_event, rsi_zone, entry_type,
                t1_price, t2_price,
                round(order_flow_score, 4), round(effective_size_mult, 2),
                round(price * shares, 2),
                algo_name,
            ))
            c.commit()
            logger.info(
                f"[PAPER] Opened {direction} {ticker} @ ${price:.2f} "
                f"T1:${t1_price:.2f}  T2:${t2_price:.2f}  S:${stop:.2f}  "
                f"conf:{confidence:.0f}%  shares:{shares}  OF:{order_flow_score:+.2f}  "
                f"sess:{session}  regime:{regime}"
            )
            _fire_trade_event("open", ticker)
            return cur.lastrowid


def update_trade_stop(trade_id: int, new_stop: float, reason: str = "") -> bool:
    """
    Update the stop_loss for an open trade (used by RegimeTransitionHandler).
    Returns True if the trade was found and updated.
    """
    try:
        with _lock:
            with _conn() as c:
                row = c.execute(
                    "SELECT id, stop FROM paper_trades WHERE id=? AND status='OPEN'",
                    (trade_id,)
                ).fetchone()
                if not row:
                    return False
                c.execute(
                    "UPDATE paper_trades SET stop=? WHERE id=?",
                    (round(float(new_stop), 4), trade_id)
                )
                c.commit()
                logger.info(
                    f"[PAPER] Trade #{trade_id} stop updated → ${new_stop:.4f} ({reason})"
                )
                return True
    except Exception as exc:
        logger.warning(f"[PAPER] update_trade_stop error: {exc}")
        return False


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
                                          shares_total, partial_pnl, shares_total, ticker=ticker)
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
                                  close_shares, partial_pnl, shares_total, ticker=ticker)
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
    c,
    trade_id:     int,
    exit_price:   float,
    exit_reason:  str,
    entry:        float,
    direction:    str,
    close_shares: int,
    partial_pnl:  float = 0.0,
    total_shares: int   = 0,      # original position size for correct pnl_pct
    ticker:       str   = "",
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
    cur = c.execute("""
        UPDATE paper_trades
        SET status='CLOSED', closed_at=?, exit_price=?,
            exit_reason=?, pnl_pct=?, pnl_dollar=?
        WHERE id=? AND status='OPEN'
    """, (
        datetime.now(timezone.utc).isoformat(),
        round(ep, 4), exit_reason,
        round(pnl_pct, 3), round(pnl_dollar, 2),
        trade_id,
    ))
    if getattr(cur, 'rowcount', 1) == 0:
        logger.debug(f"[PAPER] Trade #{trade_id} already closed — double-close guard")
        return
    logger.info(
        f"[PAPER] Closed {ticker} {direction} @ ${ep:.2f} | "
        f"{outcome} ${pnl_dollar:+.2f} ({pnl_pct:+.2f}%) | Reason: {exit_reason}"
    )
    if ticker:
        _fire_trade_event("close", ticker)


def close_all_positions_eod(reason: str = "EOD_HARD_CLOSE_3:45PM", extended_hours: bool = False) -> int:
    """
    Force-close ALL open paper trades at current price.
    Called at 3:45 PM ET hard close, after-hours, or on startup when market is closed.
    Pass extended_hours=True when closing during AH session so prices reflect
    the actual after-hours quote rather than the stale regular-session close.
    Returns number of positions closed.
    """
    from agent.data_fetcher import fetch_batch_realtime, get_last_cached_close

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
        prices = fetch_batch_realtime(tickers, extended_hours=extended_hours)
    except Exception:
        prices = {}

    # Step 3: write closes under the lock
    closed = 0
    with _lock:
        with _conn() as c:
            for row in rows:
                ticker = row["ticker"]
                df = prices.get(ticker)
                if df is not None and not df.empty:
                    ep = float(df.iloc[-1]["Close"])
                else:
                    # Live price unavailable (market closed / API down).
                    # Use last in-memory cached price before falling back to entry.
                    ep = get_last_cached_close(ticker)
                    if ep is None:
                        ep = float(row["entry_price"])
                        logger.warning(
                            f"[PAPER] No price for {ticker} at {reason} "
                            f"— using entry ${ep:.2f} (P&L will be $0)"
                        )
                _record_close(
                    c, row["id"], ep, reason,
                    float(row["entry_price"]), row["direction"],
                    int(row["shares_rem"]), float(row["partial_pnl"]),
                    int(row["shares_total"]), ticker=ticker,
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
    Uses the last market close time (not now) so stale closes don't pollute
    today's P&L when the service restarts on a weekend.
    Returns number of positions closed.
    """
    from agent.market_hours import is_after_hours, no_new_entries
    if not no_new_entries():
        return 0  # Market is open — don't sweep

    with _lock:
        with _conn() as c:
            count = c.execute(
                "SELECT COUNT(*) AS n FROM paper_trades WHERE status='OPEN'"
            ).fetchone()["n"]

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
                          entry, direction, shares_rem, partial, shares_tot, ticker=ticker)
            logger.info(
                f"[PAPER] EOD_LOCK_PROFIT_REVERSAL {direction} {ticker} @ ${ep:.2f} "
                f"gain={pnl_pct:+.2f}% mom=✗ — taking profit on reversal"
            )
            acted = True

    elif pnl_pct >= 0.1:
        # Small winner — lock it in regardless of momentum (not worth overnight risk)
        _record_close(c, row_id, ep, "EOD_LOCK_PROFIT",
                      entry, direction, shares_rem, partial, shares_tot, ticker=ticker)
        logger.info(
            f"[PAPER] EOD_LOCK_PROFIT {direction} {ticker} @ ${ep:.2f} gain={pnl_pct:+.2f}%"
        )
        acted = True

    elif pnl_pct >= -0.3:
        # Breakeven zone — exit, no edge left this close to market end
        _record_close(c, row_id, ep, "EOD_BREAKEVEN_EXIT",
                      entry, direction, shares_rem, partial, shares_tot, ticker=ticker)
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
                              entry, direction, shares_rem, partial, shares_tot, ticker=ticker)
                logger.info(
                    f"[PAPER] EOD_CUT_LOSS {direction} {ticker} @ ${ep:.2f} pnl={pnl_pct:+.2f}%"
                )
                acted = True
        else:
            # Momentum against us — cut the loss immediately
            _record_close(c, row_id, ep, "EOD_CUT_LOSS",
                          entry, direction, shares_rem, partial, shares_tot, ticker=ticker)
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

def get_daily_pnl(days: int = 30) -> list[dict]:
    """
    Per-day P&L breakdown for the last N days.
    Includes profit_factor, avg_win, avg_loss, and running equity.
    """
    with _conn_ro() as c:
        rows = c.execute("""
            SELECT
                date(closed_at)   as trade_date,
                COUNT(*)          as total,
                SUM(CASE WHEN pnl_dollar > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl_dollar <= 0 THEN 1 ELSE 0 END) as losses,
                ROUND(SUM(COALESCE(pnl_dollar,0)), 2)  as total_pnl_dollar,
                ROUND(SUM(CASE WHEN pnl_dollar > 0 THEN pnl_dollar ELSE 0 END), 2) as gross_wins,
                ROUND(SUM(CASE WHEN pnl_dollar <= 0 THEN pnl_dollar ELSE 0 END), 2) as gross_losses,
                ROUND(AVG(CASE WHEN pnl_dollar > 0 THEN pnl_dollar END), 2)  as avg_win,
                ROUND(AVG(CASE WHEN pnl_dollar <= 0 THEN pnl_dollar END), 2) as avg_loss,
                ROUND(MAX(pnl_dollar), 2) as best_trade,
                ROUND(MIN(pnl_dollar), 2) as worst_trade
            FROM paper_trades
            WHERE status='CLOSED' AND closed_at >= date('now', ?)
            GROUP BY date(closed_at)
            ORDER BY trade_date ASC
        """, (f'-{days} days',)).fetchall()
        cfg_row = c.execute("SELECT total_budget FROM account_config WHERE id=1").fetchone()

    budget = float(cfg_row["total_budget"]) if cfg_row else 50000.0
    result = []
    running_equity = budget
    for r in rows:
        d = dict(r)
        gw = float(d.get("gross_wins")   or 0)
        gl = float(d.get("gross_losses") or 0)
        day_pnl = float(d.get("total_pnl_dollar") or 0)
        d["profit_factor"]   = round(abs(gw / gl), 3) if gl != 0 else 0.0
        d["day_start_equity"] = round(running_equity, 2)
        running_equity       += day_pnl
        d["day_end_equity"]   = round(running_equity, 2)
        d["day_pnl_pct"]      = round(day_pnl / d["day_start_equity"] * 100, 3) if d["day_start_equity"] else 0.0
        # Remove the old total_pnl_pct (sum of pct — meaningless)
        d.pop("total_pnl_pct", None)
        result.append(d)
    # Return most-recent-first for display
    result.reverse()
    return result


def get_today_pnl() -> dict:
    """Today's closed-trade P&L plus open unrealized P&L."""
    with _conn_ro() as c:
        row = c.execute("""
            SELECT
                COUNT(*)   as total,
                SUM(CASE WHEN pnl_dollar > 0 THEN 1 ELSE 0 END) as wins,
                ROUND(SUM(COALESCE(pnl_dollar, 0)), 2)           as total_pnl_dollar,
                ROUND(SUM(CASE WHEN pnl_dollar > 0 THEN pnl_dollar ELSE 0 END), 2) as gross_wins,
                ROUND(SUM(CASE WHEN pnl_dollar <= 0 THEN pnl_dollar ELSE 0 END), 2) as gross_losses,
                ROUND(MAX(pnl_dollar), 2) as best_trade,
                ROUND(MIN(pnl_dollar), 2) as worst_trade
            FROM paper_trades
            WHERE status='CLOSED' AND date(closed_at) = date('now')
        """).fetchone()
        cfg_row = c.execute("SELECT total_budget FROM account_config WHERE id=1").fetchone()

    budget = float(cfg_row["total_budget"]) if cfg_row else 50000.0
    d = dict(row) if row else {}
    total_dollar = float(d.get("total_pnl_dollar") or 0)
    d["total_pnl_pct"] = round(total_dollar / budget * 100, 3) if budget > 0 else 0.0
    d["budget"] = budget
    return d


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
                                  int(row["shares"]), partial, shares_r, ticker=ticker)
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
                                      int(row["shares"]), partial, shares_r, ticker=ticker)
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
    """Unified paper trade summary including capital state."""
    with _conn_ro() as c:
        closed = c.execute(
            "SELECT pnl_pct, pnl_dollar, direction FROM paper_trades WHERE status='CLOSED'"
        ).fetchall()
        try:
            open_rows = c.execute(
                "SELECT entry_price, shares, COALESCE(cost_basis, entry_price*shares) as cb "
                "FROM paper_trades WHERE status='OPEN'"
            ).fetchall()
        except Exception:
            open_rows = c.execute(
                "SELECT entry_price, shares, entry_price*shares as cb "
                "FROM paper_trades WHERE status='OPEN'"
            ).fetchall()
        cfg_row = c.execute(
            "SELECT total_budget, max_trade_pct, max_allocated_pct, max_open_trades "
            "FROM account_config WHERE id=1"
        ).fetchone()

    budget = float(cfg_row["total_budget"]) if cfg_row else 50000.0
    open_count  = len(open_rows)
    total       = len(closed)
    wins        = sum(1 for r in closed if (r["pnl_dollar"] or 0) > 0)
    losses      = total - wins
    dollars     = [r["pnl_dollar"] for r in closed if r["pnl_dollar"] is not None]
    win_dollars = [d for d in dollars if d > 0]
    los_dollars = [d for d in dollars if d <= 0]

    realized_pnl     = round(sum(dollars), 2) if dollars else 0.0
    gross_wins       = round(sum(win_dollars), 2) if win_dollars else 0.0
    gross_losses     = round(sum(los_dollars), 2) if los_dollars else 0.0
    avg_win          = round(gross_wins  / len(win_dollars), 2) if win_dollars else 0.0
    avg_loss         = round(gross_losses / len(los_dollars), 2) if los_dollars else 0.0
    profit_factor    = round(abs(gross_wins / gross_losses), 3) if gross_losses != 0 else 0.0
    win_rate_dec     = wins / total if total > 0 else 0.0
    loss_rate_dec    = losses / total if total > 0 else 0.0
    expectancy       = round(win_rate_dec * avg_win + loss_rate_dec * avg_loss, 2)

    allocated        = round(sum(float(r["cb"]) for r in open_rows), 2)
    available        = round(budget + realized_pnl - allocated, 2)
    cap_util_pct     = round(allocated / budget * 100, 1) if budget > 0 else 0.0

    return {
        # Capital state
        "starting_balance":   budget,
        "realized_pnl":       realized_pnl,
        "allocated_capital":  allocated,
        "available_capital":  available,
        "capital_util_pct":   cap_util_pct,
        # Trade counts
        "open":               open_count,
        "closed":             total,
        "wins":               wins,
        "losses":             losses,
        "win_rate":           round(win_rate_dec * 100, 1),
        "max_concurrent":     _max_concurrent(),
        # P&L metrics
        "total_dollar_pnl":   realized_pnl,
        "gross_wins":         gross_wins,
        "gross_losses":       gross_losses,
        "avg_win":            avg_win,
        "avg_loss":           avg_loss,
        "profit_factor":      profit_factor,
        "expectancy":         expectancy,
        # Legacy compat
        "avg_pnl":            round(sum(r["pnl_pct"] or 0 for r in closed) / total, 3) if total > 0 else 0.0,
        "total_pnl":          realized_pnl,
    }


def get_account_state(open_prices: dict | None = None) -> dict:
    """
    Full account state: capital, realized + unrealized P&L, risk metrics.

    open_prices: optional {ticker: current_price} dict for unrealized P&L.
    """
    with _conn_ro() as c:
        closed = c.execute(
            "SELECT pnl_dollar FROM paper_trades WHERE status='CLOSED' AND pnl_dollar IS NOT NULL"
        ).fetchall()
        try:
            open_rows = c.execute(
                "SELECT ticker, direction, entry_price, shares, "
                "COALESCE(shares_remaining, shares) as shares_rem, "
                "COALESCE(cost_basis, entry_price*shares) as cb "
                "FROM paper_trades WHERE status='OPEN'"
            ).fetchall()
        except Exception:
            open_rows = c.execute(
                "SELECT ticker, direction, entry_price, shares, "
                "COALESCE(shares_remaining, shares) as shares_rem, "
                "entry_price*shares as cb "
                "FROM paper_trades WHERE status='OPEN'"
            ).fetchall()
        cfg_row = c.execute(
            "SELECT total_budget, max_trade_pct, max_allocated_pct, max_open_trades "
            "FROM account_config WHERE id=1"
        ).fetchone()
        today_row = c.execute("""
            SELECT ROUND(SUM(COALESCE(pnl_dollar,0)),2) as today_pnl
            FROM paper_trades
            WHERE status='CLOSED' AND date(closed_at) = date('now')
        """).fetchone()

    budget        = float(cfg_row["total_budget"]) if cfg_row else 50000.0
    max_trade_pct = float(cfg_row["max_trade_pct"]) if cfg_row else 5.0
    max_alloc_pct = float(cfg_row["max_allocated_pct"]) if cfg_row else 40.0
    max_open      = int(cfg_row["max_open_trades"]) if cfg_row else 10

    dollars      = [float(r["pnl_dollar"]) for r in closed]
    realized_pnl = round(sum(dollars), 2) if dollars else 0.0
    wins         = [d for d in dollars if d > 0]
    losses_d     = [d for d in dollars if d <= 0]
    gross_wins   = round(sum(wins), 2)
    gross_losses = round(sum(losses_d), 2)
    avg_win      = round(gross_wins  / len(wins),     2) if wins     else 0.0
    avg_loss     = round(gross_losses / len(losses_d), 2) if losses_d else 0.0
    profit_factor = round(abs(gross_wins / gross_losses), 3) if gross_losses != 0 else 0.0
    n            = len(dollars)
    win_rate_dec = len(wins) / n if n > 0 else 0.0
    loss_rate_dec = len(losses_d) / n if n > 0 else 0.0
    expectancy   = round(win_rate_dec * avg_win + loss_rate_dec * avg_loss, 2)

    allocated    = round(sum(float(r["cb"]) for r in open_rows), 2)

    # Unrealized P&L using provided prices
    unrealized = 0.0
    if open_prices:
        for r in open_rows:
            cp = open_prices.get(r["ticker"])
            if cp:
                ep  = float(r["entry_price"])
                sh  = int(r["shares_rem"])
                d   = r["direction"]
                unrealized += ((cp - ep) * sh) if d == "BUY" else ((ep - cp) * sh)
    unrealized = round(unrealized, 2)

    available      = round(budget + realized_pnl - allocated, 2)
    account_equity = round(budget + realized_pnl + unrealized, 2)
    today_pnl      = float(today_row["today_pnl"]) if today_row and today_row["today_pnl"] else 0.0

    # Max drawdown from daily snapshots (approximate from closed trades)
    peak = budget
    trough = budget
    running = budget
    max_dd = 0.0
    max_dd_pct = 0.0
    for d in dollars:
        running += d
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd / peak * 100 if peak > 0 else 0.0
    trough = running  # noqa

    # Build per-trade unrealized breakdown for UI
    open_positions = []
    for r in open_rows:
        ep  = float(r["entry_price"])
        sh  = int(r["shares_rem"])
        cb  = float(r["cb"])
        cp  = open_prices.get(r["ticker"]) if open_prices else None
        upnl = None
        if cp:
            upnl = round(((cp - ep) * sh) if r["direction"] == "BUY" else ((ep - cp) * sh), 2)
        open_positions.append({
            "ticker":        r["ticker"],
            "direction":     r["direction"],
            "entry_price":   ep,
            "shares":        sh,
            "cost_basis":    cb,
            "unrealized_pnl": upnl,
        })

    return {
        # Capital
        "starting_balance":   budget,
        "account_equity":     account_equity,
        "available_capital":  available,
        "allocated_capital":  allocated,
        "unrealized_pnl":     unrealized,
        "capital_util_pct":   round(allocated / budget * 100, 1) if budget > 0 else 0.0,
        # P&L
        "realized_pnl":       realized_pnl,
        "gross_wins":         gross_wins,
        "gross_losses":       gross_losses,
        "today_pnl":          round(today_pnl, 2),
        "today_pnl_pct":      round(today_pnl / budget * 100, 3) if budget > 0 else 0.0,
        # Trade stats
        "total_trades":       n,
        "wins":               len(wins),
        "losses":             len(losses_d),
        "win_rate":           round(win_rate_dec * 100, 1),
        "avg_win":            avg_win,
        "avg_loss":           avg_loss,
        "profit_factor":      profit_factor,
        "expectancy":         expectancy,
        # Risk
        "max_drawdown_dollar": round(-max_dd, 2),
        "max_drawdown_pct":    round(-max_dd_pct, 2),
        # Open positions detail
        "open_positions":     open_positions,
        # Config
        "open_trades":        len(open_rows),
        "max_open_trades":    max_open,
        "max_trade_pct":      max_trade_pct,
        "max_allocated_pct":  max_alloc_pct,
    }


def update_account_config(
    total_budget:      float | None = None,
    max_trade_pct:     float | None = None,
    max_allocated_pct: float | None = None,
    max_open_trades:   int   | None = None,
) -> dict:
    """Update account configuration. Returns new config."""
    from datetime import datetime, timezone
    with _lock:
        with _conn() as c:
            row = c.execute("SELECT * FROM account_config WHERE id=1").fetchone()
            cur = dict(row) if row else {}
            new_budget    = total_budget      if total_budget      is not None else cur.get("total_budget", 50000)
            new_trade_pct = max_trade_pct     if max_trade_pct     is not None else cur.get("max_trade_pct", 5.0)
            new_alloc_pct = max_allocated_pct if max_allocated_pct is not None else cur.get("max_allocated_pct", 40.0)
            new_max_open  = max_open_trades   if max_open_trades   is not None else cur.get("max_open_trades", 10)
            c.execute("""
                INSERT OR REPLACE INTO account_config
                (id, total_budget, max_trade_pct, max_allocated_pct, max_open_trades, updated_at)
                VALUES (1, ?, ?, ?, ?, ?)
            """, (new_budget, new_trade_pct, new_alloc_pct, new_max_open,
                  datetime.now(timezone.utc).isoformat()))
            c.commit()
    return {
        "total_budget":      new_budget,
        "max_trade_pct":     new_trade_pct,
        "max_allocated_pct": new_alloc_pct,
        "max_open_trades":   new_max_open,
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


def log_algo_signals(ticker: str, algo_signals: list, trade_opened: bool = False) -> None:
    """
    Persist every fired algo signal to algo_signal_log for performance tracking.
    Call this every time evaluate_trading_algos() returns results.
    """
    if not algo_signals:
        return
    now = datetime.now(timezone.utc).isoformat()
    try:
        with _conn() as c:
            for sig in algo_signals:
                c.execute(
                    """INSERT INTO algo_signal_log
                         (logged_at, ticker, algo, direction, confidence,
                          entry, stop, target, rr, trade_opened)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        now, ticker,
                        sig.get("algo", ""),
                        sig.get("direction", ""),
                        float(sig.get("confidence", 0)),
                        float(sig.get("entry", 0)),
                        float(sig.get("stop", 0)),
                        float(sig.get("target", 0)),
                        float(sig.get("rr", 0)),
                        int(trade_opened),
                    ),
                )
            c.commit()
    except Exception as e:
        logger.debug(f"[ALGO_LOG] {ticker}: {e}")


def get_algo_performance() -> dict:
    """
    Return per-algo performance statistics aggregated from both
    algo_signal_log (all fires) and paper_trades (closed trades).
    """
    try:
        with _conn_ro() as c:
            # Per-algo fire counts and trade-opened counts from signal log
            fire_rows = c.execute("""
                SELECT
                    algo,
                    COUNT(*)                              AS total_fires,
                    SUM(trade_opened)                     AS trades_triggered,
                    AVG(rr)                               AS avg_rr_at_fire,
                    MIN(logged_at)                        AS first_fire,
                    MAX(logged_at)                        AS last_fire
                FROM algo_signal_log
                GROUP BY algo
                ORDER BY total_fires DESC
            """).fetchall()

            # Per-algo closed-trade stats from paper_trades
            trade_rows = c.execute("""
                SELECT
                    algo_name                             AS algo,
                    COUNT(*)                              AS total_trades,
                    SUM(CASE WHEN pnl_dollar > 0 THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN pnl_dollar <= 0 THEN 1 ELSE 0 END) AS losses,
                    AVG(pnl_pct)                          AS avg_pnl_pct,
                    SUM(pnl_dollar)                       AS total_pnl_dollar,
                    AVG(rr_ratio)                         AS avg_rr,
                    AVG(bars_held)                        AS avg_bars_held
                FROM paper_trades
                WHERE status='CLOSED' AND algo_name != ''
                GROUP BY algo_name
                ORDER BY total_pnl_dollar DESC
            """).fetchall()

            # Recent fires (last 100)
            recent_rows = c.execute("""
                SELECT logged_at, ticker, algo, direction, confidence,
                       entry, stop, target, rr, trade_opened
                FROM algo_signal_log
                ORDER BY id DESC LIMIT 100
            """).fetchall()

    except Exception as e:
        logger.debug(f"[ALGO_PERF] query error: {e}")
        return {"algo_stats": [], "recent_fires": [], "error": str(e)}

    fires_by_algo = {r["algo"]: dict(r) for r in fire_rows}
    trades_by_algo = {r["algo"]: dict(r) for r in trade_rows}

    # Merge into a single list
    all_algos = sorted(set(fires_by_algo) | set(trades_by_algo))
    algo_stats = []
    for algo in all_algos:
        f = fires_by_algo.get(algo, {})
        t = trades_by_algo.get(algo, {})
        wins   = int(t.get("wins", 0) or 0)
        losses = int(t.get("losses", 0) or 0)
        total  = wins + losses
        algo_stats.append({
            "algo":            algo,
            "total_fires":     int(f.get("total_fires", 0) or 0),
            "trades_triggered":int(f.get("trades_triggered", 0) or 0),
            "closed_trades":   total,
            "wins":            wins,
            "losses":          losses,
            "win_rate":        round(wins / total * 100, 1) if total > 0 else 0.0,
            "avg_pnl_pct":     round(float(t.get("avg_pnl_pct", 0) or 0), 3),
            "total_pnl_dollar":round(float(t.get("total_pnl_dollar", 0) or 0), 2),
            "avg_rr":          round(float(t.get("avg_rr", 0) or 0), 2),
            "avg_bars_held":   round(float(t.get("avg_bars_held", 0) or 0), 1),
            "avg_rr_at_fire":  round(float(f.get("avg_rr_at_fire", 0) or 0), 2),
            "first_fire":      f.get("first_fire", ""),
            "last_fire":       f.get("last_fire", ""),
        })

    recent_fires = [dict(r) for r in recent_rows]

    return {
        "algo_stats":   algo_stats,
        "recent_fires": recent_fires,
    }


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
