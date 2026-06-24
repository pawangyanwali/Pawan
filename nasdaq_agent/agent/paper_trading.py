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
import collections
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from agent.exit_signals import analyse_exits, ExitAnalysis
from agent.db import get_conn

logger = logging.getLogger(__name__)

_lock    = threading.Lock()

_PAPER_MIN_CONF          = 25.0  # floor confidence for paper trade data collection
_FALLBACK_MIN_CONFIDENCE = 25.0  # used if adaptive filter is unavailable
# Time stops — live-editable via Settings → Trade Rules (paper.max_bars_scalp / paper.max_bars_intraday)
_MAX_BARS_HELD_SCALP     = 20   # fallback default; runtime value from config_store
_MAX_BARS_HELD_INTRADAY  = 90   # fallback default; runtime value from config_store
# Loaded from config at runtime so .env changes take effect without code edits
def _max_concurrent() -> int:
    from config import PAPER_MAX_OPEN_TRADES
    return PAPER_MAX_OPEN_TRADES


def _conn():
    return get_conn()

def _conn_ro():
    return get_conn(read_only=True)


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
            algo_name           TEXT    DEFAULT '',
            mfe_dollar          DOUBLE PRECISION DEFAULT 0,
            mae_dollar          DOUBLE PRECISION DEFAULT 0,
            mfe_pct             DOUBLE PRECISION DEFAULT 0,
            mae_pct             DOUBLE PRECISION DEFAULT 0,
            mfe_r               DOUBLE PRECISION DEFAULT 0,
            mae_r               DOUBLE PRECISION DEFAULT 0
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
            algo_name           TEXT    DEFAULT '',
            mfe_dollar          REAL    DEFAULT 0,
            mae_dollar          REAL    DEFAULT 0,
            mfe_pct             REAL    DEFAULT 0,
            mae_pct             REAL    DEFAULT 0,
            mfe_r               REAL    DEFAULT 0,
            mae_r               REAL    DEFAULT 0
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
            + [f"ALTER TABLE algo_signal_log ADD COLUMN IF NOT EXISTS {col} {defn}"
               for col, defn in _ALGO_SIGNAL_LOG_ADDITIONS]
            + [f"ALTER TABLE param_tune_log ADD COLUMN IF NOT EXISTS {col} {defn}"
               for col, defn in _PARAM_TUNE_LOG_ADDITIONS]
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
            try:
                existing_algo = {row[1] for row in c.execute("PRAGMA table_info(algo_signal_log)").fetchall()}
                for col, definition in _ALGO_SIGNAL_LOG_ADDITIONS:
                    if col not in existing_algo:
                        c.execute(f"ALTER TABLE algo_signal_log ADD COLUMN {col} {definition}")
            except Exception:
                pass
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
    # Execution realism tables (Phase 6 — paper_orders, paper_fills, attribution)
    try:
        from agent.execution.paper_broker import init_execution_tables
        init_execution_tables()
    except Exception as _et:
        logger.warning("[init_db] execution tables init failed: %s", _et)


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
    # ML model scores (added for Algo Intel dashboard)
    ("ml_scalp_prob",          "DOUBLE PRECISION"),
    ("ml_daily_prob",          "DOUBLE PRECISION"),
    ("ml_swing_prob",          "DOUBLE PRECISION"),
    ("ml_deep_prob",           "DOUBLE PRECISION"),
    ("ml_ensemble_score",      "INTEGER"),
    ("feedback_triggered_at",  "TEXT"),
    ("t1_water_mark",          "REAL DEFAULT 0"),  # price high/low since T1 hit, for trailing stop
    # Execution realism (Phase 6) — separates strategy P&L from execution cost
    ("entry_ideal_price",      "REAL DEFAULT 0"),  # signal price before slippage
    ("entry_slip_bps",         "REAL DEFAULT 0"),  # entry slippage in bps
    ("entry_spread_usd",       "REAL DEFAULT 0"),  # entry half-spread cost in dollars
    ("mfe_dollar",             "REAL DEFAULT 0"),
    ("mae_dollar",             "REAL DEFAULT 0"),
    ("mfe_pct",                "REAL DEFAULT 0"),
    ("mae_pct",                "REAL DEFAULT 0"),
    ("mfe_r",                  "REAL DEFAULT 0"),
    ("mae_r",                  "REAL DEFAULT 0"),
]

# Additional columns for algo_signal_log (applied separately)
_ALGO_SIGNAL_LOG_ADDITIONS = [
    ("ml_scalp_prob",  "DOUBLE PRECISION"),
    ("ml_daily_prob",  "DOUBLE PRECISION"),
    ("ml_swing_prob",  "DOUBLE PRECISION"),
    ("ml_deep_prob",   "DOUBLE PRECISION"),
    ("filter_reason",  "TEXT"),
    ("exec_status",    "TEXT"),
]

# Additional columns for param_tune_log (applied separately)
_PARAM_TUNE_LOG_ADDITIONS = [
    ("trigger_trade_id", "INTEGER"),
    ("trigger_ms",       "INTEGER"),
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


# ── Algo family mapping ────────────────────────────────────────────────────────
_ALGO_FAMILY_MAP = {
    "BB_MEAN_REV": "bb_rev",
    "MACD_ACC":    "macd_acc",
    "SUPERTREND":  "supertrend",
    "ORB_ZV":      "orb_zv",
    "RSI2_SNAP":   "rsi2_snap",
    "EMA_PULL":    "ema_pull",
    "VWAP_TREND":  "vwap_trend",
    "VWAP_OFI":    "vwap_ofi",
    "DONCHIAN":    "donchian",
    "SQUEEZE":     "squeeze",
    "VOL_SHOCK":   "vol_shock",
    "KC_FADE":     "keltner",
    "KELTNER_FADE":"keltner",
    "KELTNER":     "keltner",
    "META_ENS":    "meta_ens",
    "PAIR_ARB":    "pair_arb",
    "REGIME_SW":   "regime_sw",
    "REGIME_FADE": "regime_sw",
    "REGIME_TREND":"regime_sw",
    "OFI":         "ofi",
}

def _algo_family(algo_name: str) -> str:
    """Map full algo name to config family key, e.g. 'BB_MEAN_REV_BULL' → 'bb_rev'."""
    upper = (algo_name or "").upper()
    for prefix, family in _ALGO_FAMILY_MAP.items():
        if upper.startswith(prefix):
            return family
    for sfx in ("_BULL", "_BEAR", "_LONG", "_SHORT"):
        if upper.endswith(sfx):
            upper = upper[:-len(sfx)]
    return upper.lower()


def _effective_algo_name(algo_name: str, entry_type: str) -> str:
    """Return a non-empty trade family name for execution learning."""
    raw = (algo_name or "").strip()
    if raw:
        return raw
    entry = (entry_type or "IMMEDIATE").strip().upper().replace(" ", "_")
    entry = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in entry) or "IMMEDIATE"
    return f"PRED_{entry}"


def _family_prefixes(algo_name: str) -> list[str]:
    """SQL LIKE prefixes that identify all algo names in the same family."""
    family = _algo_family(algo_name)
    prefixes = [prefix for prefix, fam in _ALGO_FAMILY_MAP.items() if fam == family]
    if prefixes:
        return prefixes
    upper = (algo_name or "").upper()
    for sfx in ("_BULL", "_BEAR", "_LONG", "_SHORT"):
        if upper.endswith(sfx):
            upper = upper[:-len(sfx)]
    return [upper] if upper else []


def check_family_damage_stop(algo_name: str, session: str = "") -> tuple[bool, str]:
    """DB-backed intraday kill switch for a losing execution family."""
    if not algo_name:
        return False, ""
    try:
        from agent.config_manager import config as _cfg
        if not _cfg.get("risk.family_damage_enabled", True):
            return False, ""
        min_trades = int(_cfg.get("risk.family_damage_min_trades", 3))
        max_losses = int(_cfg.get("risk.family_damage_max_losses", 3))
        loss_usd = abs(float(_cfg.get("risk.family_damage_loss_usd", 100.0)))
        min_wr = float(_cfg.get("risk.family_damage_min_win_rate", 30.0))
        scope_session = bool(_cfg.get("risk.family_damage_scope_session", False))
    except Exception:
        return False, ""

    prefixes = _family_prefixes(algo_name)
    if not prefixes:
        return False, ""

    try:
        from agent.db import using_postgres
        date_filter = (
            "(closed_at::timestamptz AT TIME ZONE 'America/New_York')::date = "
            "(NOW() AT TIME ZONE 'America/New_York')::date"
            if using_postgres()
            else "date(closed_at) = date('now')"
        )
        prefix_clause = " OR ".join("UPPER(algo_name) LIKE ?" for _ in prefixes)
        params: list = [f"{prefix}%" for prefix in prefixes]
        session_clause = ""
        if scope_session and session:
            session_clause = " AND session = ?"
            params.append(session)
        with _conn_ro() as c:
            row = c.execute(f"""
                SELECT
                    COUNT(*) AS total,
                    COALESCE(SUM(CASE WHEN COALESCE(pnl_dollar,0) > 0 THEN 1 ELSE 0 END), 0) AS wins,
                    COALESCE(SUM(CASE WHEN COALESCE(pnl_dollar,0) <= 0 THEN 1 ELSE 0 END), 0) AS losses,
                    COALESCE(SUM(COALESCE(pnl_dollar,0)), 0) AS pnl
                FROM paper_trades
                WHERE status='CLOSED'
                  AND closed_at IS NOT NULL AND closed_at != ''
                  AND {date_filter}
                  AND ({prefix_clause})
                  {session_clause}
            """, tuple(params)).fetchone()
    except Exception as exc:
        logger.debug("[PAPER] family damage check failed for %s: %s", algo_name, exc)
        return False, ""

    total = int(row["total"] or 0) if row else 0
    wins = int(row["wins"] or 0) if row else 0
    losses = int(row["losses"] or 0) if row else 0
    pnl = float(row["pnl"] or 0.0) if row else 0.0
    win_rate = (wins / total * 100.0) if total else 0.0
    family = _algo_family(algo_name)
    if losses >= max_losses:
        return True, (
            f"Family damage stop [{family}]: {losses} losing trades today "
            f"(limit {max_losses}). Shadow learning continues."
        )
    if loss_usd > 0 and pnl <= -loss_usd:
        return True, (
            f"Family damage stop [{family}]: today P&L ${pnl:.0f} "
            f"(loss limit -${loss_usd:.0f}). Shadow learning continues."
        )
    if total >= min_trades and pnl < 0 and win_rate < min_wr:
        return True, (
            f"Family damage stop [{family}]: {win_rate:.0f}% WR over {total} trades "
            f"below {min_wr:.0f}% floor. Shadow learning continues."
        )
    return False, ""


def check_family_open_exposure(algo_name: str, direction: str, session: str = "") -> tuple[bool, str]:
    """Prevent clustered same-family entries before the first loss can close."""
    if not algo_name:
        return False, ""
    try:
        from agent.config_manager import config as _cfg
        max_open = int(_cfg.get("risk.family_max_open_per_direction", 1))
        scope_session = bool(_cfg.get("risk.family_open_scope_session", False))
    except Exception:
        return False, ""
    if max_open <= 0:
        return False, ""

    prefixes = _family_prefixes(algo_name)
    if not prefixes:
        return False, ""
    try:
        prefix_clause = " OR ".join("UPPER(algo_name) LIKE ?" for _ in prefixes)
        params: list = [direction] + [f"{prefix}%" for prefix in prefixes]
        session_clause = ""
        if scope_session and session:
            session_clause = " AND session = ?"
            params.append(session)
        with _conn_ro() as c:
            row = c.execute(f"""
                SELECT COUNT(*) AS n
                FROM paper_trades
                WHERE status='OPEN'
                  AND direction=?
                  AND ({prefix_clause})
                  {session_clause}
            """, tuple(params)).fetchone()
        n_open = int(row["n"] or 0) if row else 0
    except Exception as exc:
        logger.debug("[PAPER] family open exposure check failed for %s: %s", algo_name, exc)
        return False, ""

    if n_open >= max_open:
        family = _algo_family(algo_name)
        return True, (
            f"Family exposure cap [{family} {direction}]: {n_open} open "
            f"(limit {max_open}). Shadow learning continues."
        )
    return False, ""


def check_fast_family_damage_stop(algo_name: str, direction: str, session: str = "") -> tuple[bool, str]:
    """Short-window DB-backed damage stop that reacts after the first clustered losses."""
    if not algo_name:
        return False, ""
    try:
        from agent.config_manager import config as _cfg
        if not _cfg.get("risk.fast_family_damage_enabled", True):
            return False, ""
        window_min = int(_cfg.get("risk.fast_family_loss_window_min", 20))
        max_losses = int(_cfg.get("risk.fast_family_max_losses", 2))
        loss_usd = abs(float(_cfg.get("risk.fast_family_loss_usd", 75.0)))
        scope_session = bool(_cfg.get("risk.family_damage_scope_session", False))
    except Exception:
        return False, ""
    if window_min <= 0:
        return False, ""

    prefixes = _family_prefixes(algo_name)
    if not prefixes:
        return False, ""
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=window_min)).isoformat()
    try:
        prefix_clause = " OR ".join("UPPER(algo_name) LIKE ?" for _ in prefixes)
        params: list = [direction, cutoff] + [f"{prefix}%" for prefix in prefixes]
        session_clause = ""
        if scope_session and session:
            session_clause = " AND session = ?"
            params.append(session)
        with _conn_ro() as c:
            row = c.execute(f"""
                SELECT
                    COUNT(*) AS total,
                    COALESCE(SUM(CASE WHEN COALESCE(pnl_dollar,0) <= 0 THEN 1 ELSE 0 END), 0) AS losses,
                    COALESCE(SUM(COALESCE(pnl_dollar,0)), 0) AS pnl
                FROM paper_trades
                WHERE status='CLOSED'
                  AND direction=?
                  AND closed_at IS NOT NULL AND closed_at != ''
                  AND closed_at >= ?
                  AND ({prefix_clause})
                  {session_clause}
            """, tuple(params)).fetchone()
    except Exception as exc:
        logger.debug("[PAPER] fast family damage check failed for %s: %s", algo_name, exc)
        return False, ""

    total = int(row["total"] or 0) if row else 0
    losses = int(row["losses"] or 0) if row else 0
    pnl = float(row["pnl"] or 0.0) if row else 0.0
    if total <= 0:
        return False, ""
    family = _algo_family(algo_name)
    if losses >= max_losses:
        return True, (
            f"Fast family damage [{family} {direction}]: {losses} losses in "
            f"{window_min}min (limit {max_losses}). Shadow learning continues."
        )
    if loss_usd > 0 and pnl <= -loss_usd:
        return True, (
            f"Fast family damage [{family} {direction}]: ${pnl:.0f} in "
            f"{window_min}min (limit -${loss_usd:.0f}). Shadow learning continues."
        )
    return False, ""


def check_post_auth_quarantine(session: str = "") -> tuple[bool, str]:
    """Block fresh entries briefly after Schwab auth recovery while data warms up."""
    try:
        from agent.config_manager import config as _cfg
        quarantine_min = int(_cfg.get("risk.post_auth_quarantine_min", 20))
    except Exception:
        return False, ""
    if quarantine_min <= 0 or session == "CLOSED":
        return False, ""
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=quarantine_min)
        with _conn_ro() as c:
            row = c.execute("""
                SELECT MAX(resolved_at) AS resolved_at
                FROM system_alerts
                WHERE alert_type='SCHWAB_AUTH'
                  AND severity='CRITICAL'
                  AND resolved_at IS NOT NULL
            """).fetchone()
        raw = (row["resolved_at"] if row else None)
        if not raw:
            return False, ""
        resolved_at = raw if isinstance(raw, datetime) else datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if resolved_at.tzinfo is None:
            resolved_at = resolved_at.replace(tzinfo=timezone.utc)
        if resolved_at >= cutoff:
            age_min = max(0.0, (datetime.now(timezone.utc) - resolved_at).total_seconds() / 60.0)
            return True, (
                f"Post-auth quarantine: Schwab auth recovered {age_min:.1f}min ago; "
                f"paper entries pause for {quarantine_min}min while live data stabilizes."
            )
    except Exception as exc:
        logger.debug("[PAPER] post-auth quarantine check skipped: %s", exc)
    return False, ""


def _check_intraday_geometry_cap(
    ticker: str,
    direction: str,
    price: float,
    stop: float,
    target: float,
) -> tuple[bool, str]:
    """Reject setups whose stop/target geometry is too wide for scalping."""
    try:
        from agent.config_manager import config as _cfg
        max_stop_pct = float(_cfg.get("risk.intraday_max_stop_pct", 2.0))
        max_target_pct = float(_cfg.get("risk.intraday_max_target_pct", 4.0))
    except Exception:
        return False, ""
    if price <= 0:
        return False, ""
    stop_pct = abs(price - stop) / price * 100.0
    target_pct = abs(target - price) / price * 100.0 if target > 0 else 0.0
    if max_stop_pct > 0 and stop_pct > max_stop_pct:
        return True, f"{ticker} stop distance {stop_pct:.1f}% exceeds {max_stop_pct:.1f}% intraday cap"
    if max_target_pct > 0 and target_pct > max_target_pct:
        return True, f"{ticker} target distance {target_pct:.1f}% exceeds {max_target_pct:.1f}% intraday cap"
    return False, ""


def _minutes_to_regular_close() -> float:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    close_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return max(0.0, (close_et - now_et).total_seconds() / 60.0)


def _check_target_reachability(
    ticker: str,
    price: float,
    target: float,
    atr: float,
    session: str,
) -> tuple[bool, str]:
    """Late-day guard: don't open targets that cannot reasonably travel before close."""
    if session not in ("PRIME", "STANDARD", "LUNCH_BLOCK", "CLOSING_CAUTION"):
        return False, ""
    try:
        from agent.config_manager import config as _cfg
        min_minutes = float(_cfg.get("risk.late_day_min_minutes_to_eod", 35))
        atr_fraction = float(_cfg.get("risk.target_reach_atr_fraction", 0.35))
    except Exception:
        return False, ""
    if min_minutes <= 0 or target <= 0 or price <= 0:
        return False, ""
    minutes_left = _minutes_to_regular_close()
    if minutes_left <= 0:
        return False, ""
    target_dist = abs(target - price)
    atr_eff = max(float(atr or 0.0), price * 0.005)
    expected_5m_move = max(atr_eff * max(atr_fraction, 0.05), price * 0.001)
    estimated_minutes = target_dist / expected_5m_move * 5.0
    required = max(min_minutes, estimated_minutes)
    if minutes_left < required:
        return True, (
            f"{ticker} target needs about {estimated_minutes:.0f}min of travel; "
            f"only {minutes_left:.0f}min remain before regular close"
        )
    return False, ""


def _deep_saturation_block(
    ticker: str,
    ml_deep_prob: Optional[float],
    ml_ensemble_score: Optional[int],
) -> tuple[bool, str]:
    """Block deep-model saturation when the ensemble does not confirm it."""
    try:
        from agent.config_manager import config as _cfg
        if not _cfg.get("risk.deep_saturation_guard_enabled", True):
            return False, ""
        saturation = float(_cfg.get("risk.deep_saturation_prob", 0.98))
        min_ensemble = int(_cfg.get("risk.deep_saturation_min_ensemble", 60))
    except Exception:
        return False, ""
    if ml_deep_prob is None or ml_ensemble_score is None:
        return False, ""
    try:
        deep = float(ml_deep_prob)
        ensemble = int(ml_ensemble_score)
    except Exception:
        return False, ""
    if deep >= saturation and ensemble < min_ensemble:
        return True, (
            f"{ticker} deep model saturated ({deep:.2f}) but ensemble={ensemble} "
            f"< {min_ensemble}; routed to shadow learning"
        )
    return False, ""


def _entry_spread_to_risk_block(ticker: str, spread_dollar: float, actual_risk: float) -> tuple[bool, str]:
    """Reject fills where modeled spread consumes too much of the stop risk."""
    try:
        from agent.config_manager import config as _cfg
        max_ratio = float(_cfg.get("risk.max_entry_spread_to_risk", 0.35))
    except Exception:
        return False, ""
    if max_ratio <= 0 or actual_risk <= 0 or spread_dollar <= 0:
        return False, ""
    ratio = float(spread_dollar) / float(actual_risk)
    if ratio > max_ratio:
        return True, (
            f"{ticker} entry spread is {ratio:.0%} of stop risk "
            f"(limit {max_ratio:.0%}); routed to shadow learning"
        )
    return False, ""


def _append_status(out_status: Optional[list], status: str) -> None:
    if out_status is not None and not out_status:
        out_status.append(status)


def _csv_set(value: object) -> set[str]:
    return {
        part.strip().upper()
        for part in str(value or "").split(",")
        if part and part.strip()
    }


def _targeted_pattern_block(
    ticker: str,
    algo_name: str,
    direction: str,
    session: str,
    confidence: float,
    ml_ensemble_score: Optional[int],
) -> tuple[bool, str]:
    """Block specific production-proven weak execution patterns."""
    if not algo_name:
        return False, ""
    try:
        from agent.config_manager import config as _cfg
        upper = algo_name.upper()
        sess = (session or "").upper()
        ens = int(ml_ensemble_score or 0)

        if (
            bool(_cfg.get("risk.kc_fade_bear_lunch_block", True))
            and sess == "LUNCH_BLOCK"
            and direction == "SELL"
            and upper.startswith(("KC_FADE", "KELTNER_FADE", "KELTNER"))
        ):
            return True, (
                f"{ticker} {algo_name} SELL blocked in LUNCH_BLOCK: "
                "recent outcomes show negative follow-through in midday chop"
            )

        if upper.startswith("PRED_IMMEDIATE") and sess == "PRE_MARKET":
            min_ens = int(_cfg.get("risk.pred_immediate_premarket_min_ensemble", 55))
            min_conf = float(_cfg.get("risk.pred_immediate_premarket_min_conf", 80.0))
            if ens < min_ens or float(confidence or 0.0) < min_conf:
                return True, (
                    f"{ticker} PRED_IMMEDIATE pre-market blocked: "
                    f"ensemble={ens} conf={float(confidence or 0):.0f}% "
                    f"requires ensemble>={min_ens} and conf>={min_conf:.0f}%"
                )
    except Exception as exc:
        logger.debug("[PAPER] targeted pattern check skipped for %s: %s", ticker, exc)
    return False, ""


def check_first_loss_probation(
    algo_name: str,
    direction: str,
    session: str,
    confidence: float,
    ml_ensemble_score: Optional[int],
) -> tuple[bool, str, float]:
    """After one recent same-context loss, require stronger evidence or cut size."""
    if not algo_name:
        return False, "", 1.0
    try:
        from agent.config_manager import config as _cfg
        if not bool(_cfg.get("risk.first_loss_probation_enabled", True)):
            return False, "", 1.0
        sess = (session or "").upper()
        sessions = _csv_set(_cfg.get("risk.first_loss_probation_sessions", "PRE_MARKET,LUNCH_BLOCK"))
        if sessions and sess not in sessions:
            return False, "", 1.0
        window_min = int(_cfg.get("risk.first_loss_probation_window_min", 60))
        min_ens = int(_cfg.get("risk.first_loss_probation_min_ensemble", 55))
        conf_bump = float(_cfg.get("risk.first_loss_probation_conf_bump", 8.0))
        size_mult = float(_cfg.get("risk.first_loss_probation_size_mult", 0.50))
    except Exception:
        return False, "", 1.0
    if window_min <= 0:
        return False, "", 1.0

    prefixes = _family_prefixes(algo_name)
    if not prefixes:
        return False, "", 1.0
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=window_min)).isoformat()
    try:
        prefix_clause = " OR ".join("UPPER(algo_name) LIKE ?" for _ in prefixes)
        params: list = [direction, session, cutoff] + [f"{prefix}%" for prefix in prefixes]
        with _conn_ro() as c:
            row = c.execute(f"""
                SELECT
                    COUNT(*) AS losses,
                    COALESCE(SUM(COALESCE(pnl_dollar,0)), 0) AS pnl
                FROM paper_trades
                WHERE status='CLOSED'
                  AND direction=?
                  AND session=?
                  AND closed_at IS NOT NULL AND closed_at != ''
                  AND closed_at >= ?
                  AND COALESCE(pnl_dollar,0) <= 0
                  AND ({prefix_clause})
            """, tuple(params)).fetchone()
    except Exception as exc:
        logger.debug("[PAPER] first-loss probation check failed for %s: %s", algo_name, exc)
        return False, "", 1.0

    losses = int(row["losses"] or 0) if row else 0
    if losses <= 0:
        return False, "", 1.0

    ens = int(ml_ensemble_score or 0)
    conf_floor = _get_min_confidence() + conf_bump
    family = _algo_family(algo_name)
    if ens < min_ens or float(confidence or 0.0) < conf_floor:
        return True, (
            f"First-loss probation [{family} {direction} {session}]: "
            f"{losses} recent loss in {window_min}min; ensemble={ens} "
            f"conf={float(confidence or 0):.0f}% requires ensemble>={min_ens} "
            f"and conf>={conf_floor:.0f}%"
        ), 1.0

    return False, (
        f"First-loss probation [{family} {direction} {session}]: "
        f"{losses} recent loss; stronger signal allowed at {size_mult:.0%} size"
    ), max(0.05, min(1.0, size_mult))


def get_execution_min_rr(algo_name: str = "", entry_type: str = "") -> float:
    """
    Return the configured target reward multiple for this execution path.

    This value builds stop/target geometry; it is not an execution gate. Family
    overrides can raise the reward target, but named algos no longer inherit a
    lower fallback that silently turns a global 1:2 plan into 1.5:1.
    """
    try:
        from agent.config_manager import config as _cfg
        base_target = float(_cfg.get("prediction.min_rr", 1.5))
        effective = _effective_algo_name(algo_name, entry_type)
        if effective.upper().startswith("PRED_"):
            return base_target
        family = _algo_family(effective)
        family_min = float(_cfg.get(f"algos.{family}.exec_min_rr", 0.0) or 0.0)
        if family_min > 0:
            return max(base_target, family_min)
        return base_target
    except Exception:
        return 1.5


def _configured_target(entry: float, stop: float, direction: str, reward_r: float) -> float:
    """Build the profit target from the configured reward multiple."""
    risk_dist = abs(float(entry) - float(stop))
    if risk_dist <= 0:
        return 0.0
    if direction == "BUY":
        return round(float(entry) + float(reward_r) * risk_dist, 4)
    return round(float(entry) - float(reward_r) * risk_dist, 4)


# ── Pre-T1 stop-hit storm circuit ─────────────────────────────────────────────
_pre_t1_storm: dict = {}          # key: "family:session" → deque of (timestamp, pnl)
_pre_t1_storm_lock = threading.Lock()

def record_pre_t1_stop(algo_name: str, session: str, pnl_dollar: float) -> None:
    """Record a pre-T1 stop hit for the storm circuit. Called from trade close logic."""
    if not algo_name:
        return
    try:
        from agent.config_manager import config as _cfg
        if not _cfg.get("risk.pre_t1_storm_enabled", True):
            return
    except Exception:
        pass
    family = _algo_family(algo_name)
    key = f"{family}:{session}"
    now = __import__("time").time()
    with _pre_t1_storm_lock:
        if key not in _pre_t1_storm:
            _pre_t1_storm[key] = collections.deque()
        _pre_t1_storm[key].append((now, pnl_dollar))
        # Prune entries older than 2× the max window to keep memory bounded
        cutoff = now - 3600
        while _pre_t1_storm[key] and _pre_t1_storm[key][0][0] < cutoff:
            _pre_t1_storm[key].popleft()


def check_pre_t1_storm(algo_name: str, session: str) -> tuple:
    """Returns (blocked: bool, reason: str). Trips when a family has excessive pre-T1 stops."""
    if not algo_name:
        return False, ""
    try:
        from agent.config_manager import config as _cfg
        if not _cfg.get("risk.pre_t1_storm_enabled", True):
            return False, ""
        window_min  = int(_cfg.get("risk.pre_t1_storm_window_min", 30))
        max_hits    = int(_cfg.get("risk.pre_t1_storm_max_hits",   8))
        max_loss    = -abs(float(_cfg.get("risk.pre_t1_storm_loss_usd", 250.0)))
    except Exception:
        return False, ""
    family = _algo_family(algo_name)
    key = f"{family}:{session}"
    with _pre_t1_storm_lock:
        entries = list(_pre_t1_storm.get(key, []))
    if not entries:
        return False, ""
    cutoff = __import__("time").time() - window_min * 60
    recent = [(t, p) for t, p in entries if t >= cutoff]
    if not recent:
        return False, ""
    n_hits    = len(recent)
    total_pnl = sum(p for _, p in recent)
    if n_hits >= max_hits:
        return True, (
            f"Pre-T1 storm [{family}]: {n_hits} pre-T1 stop hits in {window_min}min "
            f"— new {family} entries paused (learning continues)"
        )
    if total_pnl <= max_loss:
        return True, (
            f"Pre-T1 storm [{family}]: ${total_pnl:.0f} pre-T1 loss in {window_min}min "
            f"— new {family} entries paused (learning continues)"
        )
    return False, ""


# ── Rolling EV adaptive confidence floor ──────────────────────────────────────
# Tracks per-(family, direction, session) P&L outcomes in a rolling window.
# When average EV is negative enough, exec_min_conf is temporarily raised so
# a struggling combo must show stronger conviction before opening new trades.
_algo_rolling_ev: dict = {}
_algo_rolling_ev_lock = threading.Lock()


def record_algo_ev_outcome(algo_name: str, direction: str, session: str, pnl_dollar: float) -> None:
    """Record a closed trade outcome for the rolling EV adaptive floor."""
    if not algo_name:
        return
    family = _algo_family(algo_name)
    key = f"{family}:{direction}:{session}"
    now = time.time()
    with _algo_rolling_ev_lock:
        if key not in _algo_rolling_ev:
            _algo_rolling_ev[key] = collections.deque()
        _algo_rolling_ev[key].append((now, pnl_dollar))
        cutoff = now - 7200  # prune entries older than 2 hours
        while _algo_rolling_ev[key] and _algo_rolling_ev[key][0][0] < cutoff:
            _algo_rolling_ev[key].popleft()


def check_rolling_ev_suppress(algo_name: str, direction: str, session: str) -> tuple:
    """Returns (suppress: bool, conf_bump: float).
    Trips when rolling EV is below suppress_threshold for at least min_trades."""
    if not algo_name:
        return False, 0.0
    try:
        from agent.config_manager import config as _cfg
        if not _cfg.get("risk.rolling_ev_enabled", True):
            return False, 0.0
        window_min  = int(_cfg.get("risk.rolling_ev_window_min", 120))
        min_trades  = int(_cfg.get("risk.rolling_ev_min_trades", 5))
        threshold   = float(_cfg.get("risk.rolling_ev_suppress_threshold", -2.0))
        conf_bump   = float(_cfg.get("risk.rolling_ev_conf_bump", 15.0))
    except Exception:
        return False, 0.0
    family = _algo_family(algo_name)
    key = f"{family}:{direction}:{session}"
    with _algo_rolling_ev_lock:
        entries = list(_algo_rolling_ev.get(key, []))
    if not entries:
        return False, 0.0
    cutoff = time.time() - window_min * 60
    recent = [(t, p) for t, p in entries if t >= cutoff]
    if len(recent) < min_trades:
        return False, 0.0
    avg_ev = sum(p for _, p in recent) / len(recent)
    if avg_ev < threshold:
        return True, conf_bump
    return False, 0.0


# ── Flash-stop guard (sub-60-second stop hits) ────────────────────────────────
# Catches execution/timing failures faster than the pre-T1 storm circuit.
# A "flash stop" is a stop hit within N seconds of trade open — indicative of
# bad entry timing, spread issues, or momentum reversal on entry.
_flash_stops: dict = {}
_flash_stop_lock = threading.Lock()


def record_flash_stop(algo_name: str, session: str, created_at_str: str) -> None:
    """Record a sub-60-second stop hit for the flash-stop guard."""
    if not algo_name or not created_at_str:
        return
    try:
        from agent.config_manager import config as _cfg
        flash_s = float(_cfg.get("risk.flash_stop_seconds", 60.0))
        opened_dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        elapsed_s = (datetime.now(timezone.utc) - opened_dt).total_seconds()
        if elapsed_s > flash_s:
            return  # not a flash stop
    except Exception:
        return
    family = _algo_family(algo_name)
    key = f"{family}:{session}"
    now = time.time()
    with _flash_stop_lock:
        if key not in _flash_stops:
            _flash_stops[key] = collections.deque()
        _flash_stops[key].append(now)
        cutoff = now - 3600
        while _flash_stops[key] and _flash_stops[key][0] < cutoff:
            _flash_stops[key].popleft()


def check_flash_stop_guard(algo_name: str, session: str) -> tuple:
    """Returns (blocked: bool, reason: str). Trips on repeated sub-60s stop hits."""
    if not algo_name:
        return False, ""
    try:
        from agent.config_manager import config as _cfg
        if not _cfg.get("risk.flash_stop_enabled", True):
            return False, ""
        window_min = int(_cfg.get("risk.flash_stop_window_min", 30))
        max_hits   = int(_cfg.get("risk.flash_stop_max_per_family", 3))
    except Exception:
        return False, ""
    family = _algo_family(algo_name)
    key = f"{family}:{session}"
    with _flash_stop_lock:
        entries = list(_flash_stops.get(key, []))
    if not entries:
        return False, ""
    cutoff = time.time() - window_min * 60
    recent = [t for t in entries if t >= cutoff]
    if len(recent) >= max_hits:
        return True, (
            f"Flash-stop guard [{family}]: {len(recent)} sub-60s stops in {window_min}min "
            f"— new {family} entries paused (learning continues)"
        )
    return False, ""


# ── Ticker-level damage control ───────────────────────────────────────────────
# Blocks a specific ticker after repeated losses or pre-T1 stops.
# Operates independently of algo-family controls — one bad ticker won't
# pollute the algo family's metrics. Config: risk.ticker_* keys.
_ticker_loss_tracker:    dict = {}   # ticker → deque of (timestamp, pnl_dollar)
_ticker_pre_t1_stops:    dict = {}   # ticker → deque of timestamp
_ticker_cooldown_until:  dict = {}   # ticker → epoch when cooldown ends
_ticker_lock = threading.Lock()


def _rcfg(key: str, default):
    try:
        from agent.config_manager import config as _cfg
        return _cfg.get(key, default)
    except Exception:
        return default


def record_ticker_loss(ticker: str, pnl_dollar: float) -> None:
    """Record a loss outcome for ticker-level cooldown tracking."""
    if not ticker or pnl_dollar >= 0:
        return
    now = time.time()
    window_s = int(_rcfg("risk.ticker_loss_window_min", 30)) * 60
    with _ticker_lock:
        if ticker not in _ticker_loss_tracker:
            _ticker_loss_tracker[ticker] = collections.deque()
        _ticker_loss_tracker[ticker].append((now, pnl_dollar))
        cutoff = now - window_s
        while _ticker_loss_tracker[ticker] and _ticker_loss_tracker[ticker][0][0] < cutoff:
            _ticker_loss_tracker[ticker].popleft()
        # Trip cooldown immediately if threshold exceeded
        total = sum(p for _, p in _ticker_loss_tracker[ticker])
        threshold = -abs(float(_rcfg("risk.ticker_loss_cooldown_usd", 50.0)))
        if total < threshold:
            cooldown_s = int(_rcfg("risk.ticker_cooldown_min", 60)) * 60
            _ticker_cooldown_until[ticker] = now + cooldown_s
            logger.info(
                f"[PAPER] Ticker cooldown tripped: {ticker} lost ${total:.2f} "
                f"in {int(_rcfg('risk.ticker_loss_window_min', 30))}min "
                f"→ blocked {int(_rcfg('risk.ticker_cooldown_min', 60))}min"
            )


def record_ticker_pre_t1_stop(ticker: str) -> None:
    """Record a pre-T1 stop for per-ticker fast-stop detection."""
    if not ticker:
        return
    now = time.time()
    window_s = int(_rcfg("risk.ticker_pre_t1_window_min", 30)) * 60
    with _ticker_lock:
        if ticker not in _ticker_pre_t1_stops:
            _ticker_pre_t1_stops[ticker] = collections.deque()
        _ticker_pre_t1_stops[ticker].append(now)
        cutoff = now - window_s
        while _ticker_pre_t1_stops[ticker] and _ticker_pre_t1_stops[ticker][0] < cutoff:
            _ticker_pre_t1_stops[ticker].popleft()
        max_stops = int(_rcfg("risk.ticker_pre_t1_stops_max", 2))
        if len(_ticker_pre_t1_stops[ticker]) >= max_stops:
            cooldown_s = int(_rcfg("risk.ticker_cooldown_min", 60)) * 60
            _ticker_cooldown_until[ticker] = now + cooldown_s
            logger.info(
                f"[PAPER] Ticker cooldown tripped: {ticker} hit {max_stops} pre-T1 stops "
                f"in {int(_rcfg('risk.ticker_pre_t1_window_min', 30))}min "
                f"→ blocked {int(_rcfg('risk.ticker_cooldown_min', 60))}min"
            )


def check_ticker_cooldown(ticker: str) -> tuple:
    """Returns (blocked: bool, reason: str) for ticker-level cooldown."""
    if not ticker:
        return False, ""
    now = time.time()
    with _ticker_lock:
        blocked_until = _ticker_cooldown_until.get(ticker, 0.0)
    if now < blocked_until:
        remaining_min = round((blocked_until - now) / 60, 0)
        return True, f"ticker {ticker} cooling down ({int(remaining_min)}min remaining)"
    return False, ""


def get_ticker_cooldowns() -> list[dict]:
    """Return list of currently active ticker cooldowns (for dashboard)."""
    now = time.time()
    result = []
    with _ticker_lock:
        for ticker, until in _ticker_cooldown_until.items():
            if now < until:
                result.append({
                    "ticker": ticker,
                    "remaining_min": round((until - now) / 60, 1),
                })
    return sorted(result, key=lambda x: x["remaining_min"], reverse=True)


def _get_min_confidence() -> float:
    """
    Return the minimum confidence required to open a paper trade.

    Paper trading is the DATA COLLECTION layer — the floor is intentionally
    low (25%) so the adaptive filter can observe and learn from low-confidence
    trades. The adaptive filter's dynamic_threshold (55–63%) governs live
    trading recommendations and is NOT used here — doing so would starve the
    learner by blocking 80%+ of signals before any outcome is recorded.

    The floor is read from config_store so it can be tuned without a code deploy.
    """
    try:
        from agent.config_manager import config as _cfg
        return float(_cfg.get("paper.min_confidence", _PAPER_MIN_CONF))
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
    ml_scalp_prob:    Optional[float] = None,
    ml_daily_prob:    Optional[float] = None,
    ml_swing_prob:    Optional[float] = None,
    ml_deep_prob:     Optional[float] = None,
    ml_ensemble_score: Optional[int] = None,
    rr_quality:       str   = "",
    atr:              float = 0.0,           # ATR(14) at time of signal — used for fill model
    avg_daily_volume: float = 0.0,           # avg daily shares — used for liquidity penalty
    _out_status:      Optional[list] = None,
) -> Optional[int]:
    """
    Open a paper trade when all PRD entry gates pass.
    Returns trade id or None.
    """
    if direction not in ("BUY", "SELL"):
        _append_status(_out_status, "BLOCKED_BAD_DIRECTION")
        return None

    algo_name = _effective_algo_name(algo_name, entry_type)
    _target_rr = get_execution_min_rr(algo_name, entry_type)
    rr_qualifies = True

    # R:R builds stop/target geometry; it does not hard-gate execution.

    # Block paper execution during RESTRICTED session (9:30–9:44 ET price discovery).
    # Signals continue to algo_signal_log for learning.
    _live_session_early = session or ""
    if _live_session_early == "RESTRICTED":
        from agent.config_manager import config as _cfg_sess
        if _cfg_sess.get("paper.block_restricted_session", True):
            logger.debug(f"[PAPER] {ticker} skip: RESTRICTED session blocked for paper execution")
            _append_status(_out_status, "BLOCKED_RESTRICTED")
            return None

    # Ticker-level damage control — independent of algo-family controls.
    _auth_blocked, _auth_reason = check_post_auth_quarantine(_live_session_early)
    if _auth_blocked:
        logger.info(f"[PAPER] {ticker} skip: {_auth_reason}")
        _append_status(_out_status, "BLOCKED_POST_AUTH_QUARANTINE")
        return None

    _tk_blocked, _tk_reason = check_ticker_cooldown(ticker)
    if _tk_blocked:
        logger.info(f"[PAPER] {ticker} skip: {_tk_reason}")
        _append_status(_out_status, "BLOCKED_TICKER_COOLDOWN")
        return None

    # Per-family execution controls — check BEFORE circuit breaker for fast-path rejection.
    if algo_name:
        from agent.config_manager import config as _cfg_fam
        _fam = _algo_family(algo_name)
        _fam_enabled      = _cfg_fam.get(f"algos.{_fam}.exec_enabled", True)
        _fam_size_mult    = float(_cfg_fam.get(f"algos.{_fam}.exec_size_mult", 1.0) or 1.0)
        _fam_min_conf     = float(_cfg_fam.get(f"algos.{_fam}.exec_min_conf",  0.0) or 0.0)
        _fam_block_sess   = str(_cfg_fam.get(f"algos.{_fam}.exec_block_sessions", "") or "")
        if not _fam_enabled:
            logger.debug(f"[PAPER] {ticker} skip: family {_fam} execution disabled")
            _append_status(_out_status, "BLOCKED_FAMILY_DISABLED")
            return None
        if _fam_min_conf > 0 and confidence < _fam_min_conf:
            logger.debug(f"[PAPER] {ticker} skip: {_fam} conf {confidence:.0f}% < family min {_fam_min_conf:.0f}%")
            _append_status(_out_status, "BLOCKED_FAMILY_CONF")
            return None
        _sess_check = session or _live_session_early
        if _fam_block_sess and _sess_check and _sess_check in _fam_block_sess.split(","):
            logger.debug(f"[PAPER] {ticker} skip: {_fam} blocked in session {_sess_check}")
            _append_status(_out_status, "BLOCKED_FAMILY_SESSION")
            return None
        _targeted_blocked, _targeted_reason = _targeted_pattern_block(
            ticker, algo_name, direction, session or "", confidence, ml_ensemble_score
        )
        if _targeted_blocked:
            logger.info(f"[PAPER] {ticker} skip: {_targeted_reason}")
            _append_status(_out_status, "BLOCKED_PATTERN_CONTEXT")
            return None
        _family_exposure_blocked, _family_exposure_reason = check_family_open_exposure(
            algo_name, direction, session or ""
        )
        if _family_exposure_blocked:
            logger.info(f"[PAPER] {ticker} skip: {_family_exposure_reason}")
            _append_status(_out_status, "BLOCKED_FAMILY_EXPOSURE")
            return None
        _prob_blocked, _prob_reason, _prob_size_mult = check_first_loss_probation(
            algo_name, direction, session or "", confidence, ml_ensemble_score
        )
        if _prob_blocked:
            logger.info(f"[PAPER] {ticker} skip: {_prob_reason}")
            _append_status(_out_status, "BLOCKED_FIRST_LOSS_PROBATION")
            return None
        if _prob_size_mult < 1.0:
            size_mult = round(size_mult * _prob_size_mult, 4)
            logger.info(f"[PAPER] {ticker} probation throttle: {_prob_reason}")
        _fast_damage_blocked, _fast_damage_reason = check_fast_family_damage_stop(
            algo_name, direction, session or ""
        )
        if _fast_damage_blocked:
            logger.info(f"[PAPER] {ticker} skip: {_fast_damage_reason}")
            _append_status(_out_status, "BLOCKED_FAST_FAMILY_DAMAGE")
            return None
        _damage_blocked, _damage_reason = check_family_damage_stop(algo_name, session or "")
        if _damage_blocked:
            logger.info(f"[PAPER] {ticker} skip: {_damage_reason}")
            _append_status(_out_status, "BLOCKED_FAMILY_DAMAGE")
            return None
        # Pre-T1 storm circuit
        _storm_blocked, _storm_reason = check_pre_t1_storm(algo_name, session or "")
        if _storm_blocked:
            logger.info(f"[PAPER] {ticker} skip: {_storm_reason}")
            _append_status(_out_status, "BLOCKED_FAMILY_STORM")
            return None
        # Flash-stop guard — repeated sub-60s stops indicate bad entry timing
        _flash_blocked, _flash_reason = check_flash_stop_guard(algo_name, session or "")
        if _flash_blocked:
            logger.info(f"[PAPER] {ticker} skip: {_flash_reason}")
            _append_status(_out_status, "BLOCKED_FLASH_STORM")
            return None
        # Rolling EV adaptive floor — raise confidence bar AND reduce size when combo is bleeding.
        # Two-tier response: if conf meets bumped floor → allow but halve size.
        #                    if conf below bumped floor → block entirely.
        _ev_blocked, _ev_conf_bump = check_rolling_ev_suppress(algo_name, direction, session or "")
        if _ev_blocked and _ev_conf_bump > 0:
            if bool(_cfg_fam.get("risk.rolling_ev_hard_block", True)):
                logger.info(
                    f"[PAPER] {ticker} skip: {_algo_family(algo_name)}+{direction} rolling EV "
                    "negative — routed to shadow learning"
                )
                _append_status(_out_status, "BLOCKED_EV_SUPPRESS")
                return None
            _ev_min_conf = _get_min_confidence() + _ev_conf_bump
            if confidence < _ev_min_conf:
                logger.info(
                    f"[PAPER] {ticker} skip: {_algo_family(algo_name)}+{direction} rolling EV "
                    f"negative — conf {confidence:.0f}% < bumped floor {_ev_min_conf:.0f}%"
                )
                _append_status(_out_status, "BLOCKED_EV_SUPPRESS")
                return None
            else:
                # Qualifies but combo is struggling — halve size as adaptive throttle
                size_mult = round(size_mult * 0.50, 4)
                logger.info(
                    f"[PAPER] {ticker} EV throttle: {_algo_family(algo_name)}+{direction} "
                    f"rolling EV negative → size halved (conf {confidence:.0f}% ≥ floor {_ev_min_conf:.0f}%)"
                )
        # Apply family size multiplier
        size_mult = round(size_mult * _fam_size_mult, 4)

    if price <= 0 or stop <= 0:
        logger.debug(f"[PAPER] {ticker} skip: invalid price ({price}) or stop ({stop})")
        _append_status(_out_status, "BLOCKED_INVALID_PRICE")
        return None

    target = _configured_target(price, stop, direction, _target_rr)
    rr_ratio = round(float(_target_rr), 2)

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
            _append_status(_out_status, "BLOCKED_BAD_GEOMETRY")
            return None
        if direction == "SELL" and (stop <= price or target >= price):
            logger.debug(
                f"[PAPER] {ticker} SELL geometry invalid: "
                f"entry={price:.2f} stop={stop:.2f} target={target:.2f}"
            )
            _append_status(_out_status, "BLOCKED_BAD_GEOMETRY")
            return None

    _wide_blocked, _wide_reason = _check_intraday_geometry_cap(ticker, direction, price, stop, target)
    if _wide_blocked:
        logger.info(f"[PAPER] {ticker} skip: {_wide_reason}")
        _append_status(_out_status, "BLOCKED_WIDE_GEOMETRY")
        return None

    _reach_blocked, _reach_reason = _check_target_reachability(
        ticker, price, target, atr, session or _live_session_early
    )
    if _reach_blocked:
        logger.info(f"[PAPER] {ticker} skip: {_reach_reason}")
        _append_status(_out_status, "BLOCKED_TARGET_REACHABILITY")
        return None

    _deep_blocked, _deep_reason = _deep_saturation_block(ticker, ml_deep_prob, ml_ensemble_score)
    if _deep_blocked:
        logger.info(f"[PAPER] {ticker} skip: {_deep_reason}")
        _append_status(_out_status, "BLOCKED_DEEP_SATURATION")
        return None

    min_conf = _get_min_confidence()
    if confidence < min_conf:
        logger.debug(f"[PAPER] {ticker} skip: conf {confidence:.0f}% < floor {min_conf:.0f}%")
        _append_status(_out_status, "BLOCKED_CONFIDENCE")
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

    # Confidence floors for PRE_MARKET / AFTER_HOURS by tier — read from config.
    from agent.config_manager import config as _cfg_pt
    _EXT_CONF_FLOOR: dict[str, float] = {
        "HIGH":     float(_cfg_pt.get("paper.ext_hours_high_min_conf",     70.0)),
        "MODERATE": float(_cfg_pt.get("paper.ext_hours_moderate_min_conf", 60.0)),
    }

    if _live_session == "CLOSED":
        _append_status(_out_status, "BLOCKED_CLOSED")
        logger.debug(f"[PAPER] {ticker} skip: market CLOSED — no trades on weekends/overnight")
        return None

    # ── Circuit breaker — HARD gate enforced at the execution layer ──────────
    # The scanner checks this before calling us, but a daily-loss / profit-ceiling
    # halt can trip between that check and this open (e.g. a concurrent close in
    # the same scan batch, or a different caller entirely). Re-checking here closes
    # that race so a halt can NEVER be bypassed regardless of caller. check_circuit
    # _breaker is thread-safe and idempotent. Pass the live session only for HIGH/
    # MODERATE after-hours trades so the AH consecutive-loss reset applies (matching
    # can_open_trade); P&L-based halts persist across all sessions.
    from agent.risk_controls import check_circuit_breaker
    _cb_session = _live_session if (_live_session == "AFTER_HOURS"
                                    and trading_tier in ("HIGH", "MODERATE")) else ""
    _cb_blocked, _cb_reason = check_circuit_breaker(_cb_session)
    if _cb_blocked:
        _append_status(_out_status, "BLOCKED_CIRCUIT")
        logger.info(f"[PAPER] {ticker} BLOCKED at execution by circuit breaker: {_cb_reason}")
        return None

    # Extended-hours stop widening: wider stop = smaller shares, less capital at risk
    # on thin ECN spreads (configurable, default 1.5× pre-market, 2× after-hours).
    from agent.config_manager import config as _cfg_pt
    _stop_mult = (
        float(_cfg_pt.get("paper.after_hours_stop_mult", 2.0)) if _live_session == "AFTER_HOURS" else
        float(_cfg_pt.get("paper.pre_market_stop_mult",  1.5)) if _live_session == "PRE_MARKET"  else
        1.0
    )
    if _stop_mult != 1.0:
        risk_dist_orig = abs(price - stop)
        stop = (
            round(price - risk_dist_orig * _stop_mult, 4) if direction == "BUY"
            else round(price + risk_dist_orig * _stop_mult, 4)
        )
        target = _configured_target(price, stop, direction, _target_rr)
        rr_ratio = round(float(_target_rr), 2)
        _wide_blocked, _wide_reason = _check_intraday_geometry_cap(ticker, direction, price, stop, target)
        if _wide_blocked:
            logger.info(f"[PAPER] {ticker} skip after extended-hours stop widening: {_wide_reason}")
            _append_status(_out_status, "BLOCKED_WIDE_GEOMETRY")
            return None

    # Tier gate: REGULAR-tier stocks lack liquidity for AH/PM trades.
    if _live_session in ("PRE_MARKET", "AFTER_HOURS"):
        _ext_floor = _EXT_CONF_FLOOR.get(trading_tier)
        if _ext_floor is None:
            _append_status(_out_status, "BLOCKED_EXT_HOURS_TIER")
            logger.debug(
                f"[PAPER] {ticker} skip: REGULAR-tier in {_live_session} — insufficient liquidity"
            )
            return None
        if confidence < _ext_floor:
            _append_status(_out_status, "BLOCKED_EXT_HOURS_CONF")
            logger.debug(
                f"[PAPER] {ticker} skip: conf {confidence:.0f}% < ext-hours floor {_ext_floor:.0f}%"
                f" (tier={trading_tier}, sess={_live_session})"
            )
            return None

    from agent.config_manager import config as _cfg_pt
    _rr_mult_min  = float(_cfg_pt.get("paper.rr_size_mult_min", 0.20))
    _rr_denom     = float(_cfg_pt.get("paper.rr_denominator",   2.0))
    rr_mult = round(min(1.0, max(_rr_mult_min, rr_ratio / _rr_denom)), 2) if rr_ratio > 0 else _rr_mult_min
    effective_size_mult = round(size_mult * rr_mult, 2)
    if effective_size_mult <= 0:
        effective_size_mult = _rr_mult_min

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

    # ── T1 and T2 price levels (multiples read from config_store) ─────────────
    from agent.config_manager import config as _cfg_t
    _t1_mult  = float(_cfg_t.get("paper.t1_r_multiple", 1.0))
    _t2_mult  = max(float(_cfg_t.get("paper.t2_r_multiple", _target_rr)), _target_rr)
    if not algo_name.upper().startswith("PRED_"):
        _t2_mult = max(float(_cfg_t.get("paper.algo_t2_r_multiple", _t2_mult)), _target_rr)
    if bool(_cfg_t.get("prediction.use_atr_stops", True)):
        _t2_mult = max(_t2_mult, _target_rr)
    risk_dist = abs(price - stop)
    if direction == "BUY":
        t1_price = round(price + _t1_mult * risk_dist, 4)
        t2_price = round(price + _t2_mult * risk_dist, 4)
    else:
        t1_price = round(price - _t1_mult * risk_dist, 4)
        t2_price = round(price - _t2_mult * risk_dist, 4)

    with _lock:
        with _conn() as c:
            existing = c.execute(
                "SELECT id FROM paper_trades WHERE ticker=? AND status='OPEN'", (ticker,)
            ).fetchone()
            if existing:
                _append_status(_out_status, "BLOCKED_EXISTING_TRADE")
                logger.debug(f"[PAPER] {ticker} skip: already has open trade #{existing['id']}")
                return None

            open_count = c.execute(
                "SELECT COUNT(*) AS n FROM paper_trades WHERE status='OPEN'"
            ).fetchone()["n"]

            # ── Capital gate: check available capital before sizing ────────────
            from agent.config_manager import config as _cfg
            _budget      = float(_cfg.get("paper.budget"))
            _max_trade_v = _budget * float(_cfg.get("paper.max_trade_pct"))  / 100.0
            _max_alloc_v = _budget * float(_cfg.get("paper.max_allocated_pct")) / 100.0
            _max_open    = int(_cfg.get("paper.max_open_trades"))
            if bool(_cfg.get("paper.enforce_risk_controls", True)):
                _risk_max_open = int(_cfg.get("risk.max_concurrent_trades", _max_open) or _max_open)
                _max_open = min(_max_open, _risk_max_open)

            if open_count >= _max_open:
                _append_status(_out_status, "BLOCKED_MAX_OPEN")
                logger.debug(f"[PAPER] {ticker} skip: max concurrent trades ({_max_open}) reached")
                return None

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
                _append_status(_out_status, "BLOCKED_CAPITAL")
                logger.debug(f"[PAPER] {ticker} skip: insufficient capital (avail ${available:.0f} < ${price:.2f})")
                return None
            if allocated >= _max_alloc_v:
                _append_status(_out_status, "BLOCKED_MAX_ALLOCATED")
                logger.debug(f"[PAPER] {ticker} skip: max allocated capital reached (${allocated:.0f} >= ${_max_alloc_v:.0f})")
                return None

            # Refine shares within capital constraints
            cost_basis_per_share = price
            max_by_trade   = max(1, int(_max_trade_v / cost_basis_per_share))
            max_by_capital = max(1, int(available    / cost_basis_per_share))
            shares = min(shares, max_by_trade, max_by_capital)

            # ── Realistic entry fill (Phase 6 execution realism) ─────────────
            from agent.execution.fill_model import compute_entry_fill as _cef
            _atr_eff  = atr if atr > 0 else price * 0.01   # fallback: 1% ATR estimate
            _entry_fill = _cef(
                direction, price, _atr_eff,
                _live_session or session or "REGULAR",
                shares, avg_daily_volume,
            )
            actual_entry = _entry_fill.fill_price   # slippage-adjusted entry

            if direction == "BUY" and stop >= actual_entry:
                _append_status(_out_status, "BLOCKED_BAD_GEOMETRY")
                logger.debug(
                    "[PAPER] %s BUY geometry invalid after fill: entry=%.2f stop=%.2f",
                    ticker, actual_entry, stop,
                )
                return None
            if direction == "SELL" and stop <= actual_entry:
                _append_status(_out_status, "BLOCKED_BAD_GEOMETRY")
                logger.debug(
                    "[PAPER] %s SELL geometry invalid after fill: entry=%.2f stop=%.2f",
                    ticker, actual_entry, stop,
                )
                return None

            # Recompute T1, T2, and R:R from the actual fill price.
            # Stop stays at its signal level (structural anchor); the risk
            # distance naturally reflects actual execution cost.
            _actual_risk = abs(actual_entry - stop)
            _spread_blocked, _spread_reason = _entry_spread_to_risk_block(
                ticker, float(_entry_fill.spread_dollar or 0.0) / max(shares, 1), _actual_risk
            )
            if _spread_blocked:
                _append_status(_out_status, "BLOCKED_SPREAD_RISK")
                logger.info(f"[PAPER] {ticker} skip: {_spread_reason}")
                return None
            if _actual_risk > 0:
                if direction == "BUY":
                    t1_price = round(actual_entry + _t1_mult * _actual_risk, 4)
                    t2_price = round(actual_entry + _t2_mult * _actual_risk, 4)
                else:
                    t1_price = round(actual_entry - _t1_mult * _actual_risk, 4)
                    t2_price = round(actual_entry - _t2_mult * _actual_risk, 4)
                target = t2_price
                rr_ratio = round(_t2_mult, 2)
                rr_qualifies = True
                _wide_blocked, _wide_reason = _check_intraday_geometry_cap(
                    ticker, direction, actual_entry, stop, target
                )
                if _wide_blocked:
                    _append_status(_out_status, "BLOCKED_WIDE_GEOMETRY")
                    logger.info(f"[PAPER] {ticker} skip after fill: {_wide_reason}")
                    return None

            cur = c.execute("""
                INSERT INTO paper_trades
                  (opened_at, ticker, direction, entry_price, target, stop,
                   confidence, rr_ratio, rr_qualifies, shares, shares_remaining,
                   session, regime, vwap_event, rsi_zone, entry_type,
                   t1_price, t2_price, order_flow_score, size_mult, cost_basis,
                   algo_name,
                   ml_scalp_prob, ml_daily_prob, ml_swing_prob, ml_deep_prob,
                   ml_ensemble_score,
                   entry_ideal_price, entry_slip_bps, entry_spread_usd)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                ticker, direction,
                round(actual_entry, 4), round(target, 4), round(stop, 4),
                round(confidence, 2), round(rr_ratio, 2), int(rr_qualifies),
                shares, shares,  # shares_remaining starts = shares
                session, regime, vwap_event, rsi_zone, entry_type,
                t1_price, t2_price,
                round(order_flow_score, 4), round(effective_size_mult, 2),
                round(actual_entry * shares, 2),
                algo_name,
                round(ml_scalp_prob, 4) if ml_scalp_prob is not None else None,
                round(ml_daily_prob, 4) if ml_daily_prob is not None else None,
                round(ml_swing_prob, 4) if ml_swing_prob is not None else None,
                round(ml_deep_prob, 4) if ml_deep_prob is not None else None,
                ml_ensemble_score,
                round(price, 4),                              # entry_ideal_price = signal close
                round(_entry_fill.slippage_bps, 2),           # entry_slip_bps
                round(_entry_fill.spread_dollar, 4),          # entry_spread_usd
            ))
            c.commit()
            logger.info(
                f"[PAPER] Opened {direction} {ticker} @ ${actual_entry:.2f} "
                f"(signal ${price:.2f} slip {_entry_fill.slippage_bps:.1f}bps) "
                f"T1:${t1_price:.2f}  T2:${t2_price:.2f}  S:${stop:.2f}  "
                f"conf:{confidence:.0f}%  shares:{shares}  OF:{order_flow_score:+.2f}  "
                f"sess:{session}  regime:{regime}"
            )
            # Record entry fill in paper_fills
            try:
                from agent.execution.paper_broker import record_fill as _rfill
                _rfill(c, cur.lastrowid, ticker, direction, shares, _entry_fill)
            except Exception as _fe:
                logger.debug("[PAPER] entry fill record error: %s", _fe)
            _fire_trade_event("open", ticker)
            if _out_status is not None:
                _out_status.append("EXECUTED_PAPER")
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
                       COALESCE(entry_type, '') as entry_type,
                       COALESCE(algo_name, '') as algo_name,
                       COALESCE(session, '') as session,
                       COALESCE(t1_water_mark, 0) as t1_water_mark,
                       COALESCE(mfe_dollar, 0) as mfe_dollar,
                       COALESCE(mae_dollar, 0) as mae_dollar,
                       COALESCE(mfe_pct, 0) as mfe_pct,
                       COALESCE(mae_pct, 0) as mae_pct,
                       COALESCE(mfe_r, 0) as mfe_r,
                       COALESCE(mae_r, 0) as mae_r,
                       COALESCE(entry_ideal_price, entry_price) as entry_ideal_price,
                       COALESCE(entry_slip_bps, 0) as entry_slip_bps,
                       COALESCE(entry_spread_usd, 0) as entry_spread_usd,
                       opened_at
                FROM paper_trades WHERE ticker=? AND status='OPEN'
            """, (ticker,)).fetchall()

            for row in rows:
                bars = (row["bars_held"] or 0) + 1
                c.execute("UPDATE paper_trades SET bars_held=? WHERE id=?", (bars, row["id"]))

                entry            = float(row["entry_price"] or current_price)
                stop_current     = float(row["stop"])
                t1_price         = float(row["t1_price"] or 0)
                t2_price         = float(row["t2_price"] or 0)
                shares_total     = int(row["shares"] or 1)
                shares_rem       = int(row["shares_remaining"] or shares_total)
                t1_hit           = bool(row["t1_hit"])
                partial_pnl      = float(row["partial_pnl_dollar"] or 0)
                direction        = row["direction"]
                entry_type       = row["entry_type"] or "IMMEDIATE"
                _algo_nm         = row["algo_name"] or ""
                _sess            = row["session"] or ""
                _entry_ideal     = float(row["entry_ideal_price"] or entry)
                _entry_slip_bps  = float(row["entry_slip_bps"]    or 0)
                _entry_spread    = float(row["entry_spread_usd"]  or 0)
                # Filled in per close-scenario for attribution
                _ideal_exit      = 0.0
                _exit_slip_bps   = 0.0
                _t1_water       = float(row["t1_water_mark"] or 0)
                _created_at     = row["opened_at"] or ""

                # Determine time stop based on trade type (PRD 6.3)
                # is_scalp is based solely on entry_type — not bar count, to avoid
                # misclassifying WAIT_RETEST trades that happen to be <20 bars old
                is_scalp = entry_type in ("IMMEDIATE", "SCALP")
                from agent.config_manager import config as _cfg_bars
                _scalp_bars   = int(_cfg_bars.get("paper.max_bars_scalp",    _MAX_BARS_HELD_SCALP))
                _intra_bars   = int(_cfg_bars.get("paper.max_bars_intraday", _MAX_BARS_HELD_INTRADAY))
                max_bars = _scalp_bars if is_scalp else _intra_bars

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

                _risk_ref = abs(t1_price - entry)
                if _risk_ref <= 0 and t2_price > 0:
                    try:
                        from agent.config_manager import config as _cfg_exc
                        _t2_ref = max(float(_cfg_exc.get("paper.t2_r_multiple", 1.5)), 0.01)
                    except Exception:
                        _t2_ref = 1.5
                    _risk_ref = abs(t2_price - entry) / _t2_ref
                if _risk_ref <= 0:
                    _risk_ref = max(abs(entry - stop_current), 0.0001)
                _notional_ref = max(entry * max(shares_total, 1), 0.01)
                if direction == "BUY":
                    _fav_px = max(0.0, _hi - entry)
                    _adv_px = max(0.0, entry - _lo)
                else:
                    _fav_px = max(0.0, entry - _lo)
                    _adv_px = max(0.0, _hi - entry)
                _mfe_d = max(float(row["mfe_dollar"] or 0), _fav_px * shares_total)
                _mae_d = max(float(row["mae_dollar"] or 0), _adv_px * shares_total)
                if (
                    _mfe_d > float(row["mfe_dollar"] or 0) + 0.005
                    or _mae_d > float(row["mae_dollar"] or 0) + 0.005
                ):
                    c.execute(
                        """UPDATE paper_trades
                           SET mfe_dollar=?, mae_dollar=?, mfe_pct=?, mae_pct=?, mfe_r=?, mae_r=?
                           WHERE id=?""",
                        (
                            round(_mfe_d, 2),
                            round(_mae_d, 2),
                            round(_mfe_d / _notional_ref * 100.0, 3),
                            round(_mae_d / _notional_ref * 100.0, 3),
                            round(_fav_px / _risk_ref, 3),
                            round(_adv_px / _risk_ref, 3),
                            row["id"],
                        ),
                    )

                # unrealized P&L % for smart EOD decisions
                pnl_pct_now = (
                    (ep - entry) / entry * 100 if direction == "BUY"
                    else (entry - ep) / entry * 100
                )

                # ── 0. Post-T1 trailing stop (runs every bar after T1 hit) ──────
                # Advances the ratchet stop: max(profit-lock floor, water_mark − trail_R).
                # Never moves the stop in the wrong direction (ratchet only).
                if t1_hit and t1_price > 0:
                    from agent.config_manager import config as _cfg_trail
                    _profit_lock_r = float(_cfg_trail.get("paper.t1_profit_lock_r", 0.20))
                    _trail_r       = float(_cfg_trail.get("paper.post_t1_trail_r",  0.40))
                    _risk_dist     = abs(t1_price - entry)  # = 1.0R in dollars
                    if _risk_dist > 0:
                        _water_ref = _t1_water if _t1_water > 0 else t1_price
                        if direction == "BUY":
                            new_water        = max(_water_ref, _hi)
                            profit_lock_stop = round(entry + _risk_dist * _profit_lock_r, 4)
                            trailing_stop    = round(new_water - _risk_dist * _trail_r, 4)
                            new_stop         = max(stop_current, trailing_stop, profit_lock_stop)
                        else:
                            new_water        = min(_water_ref, _lo)
                            profit_lock_stop = round(entry - _risk_dist * _profit_lock_r, 4)
                            trailing_stop    = round(new_water + _risk_dist * _trail_r, 4)
                            new_stop         = min(stop_current, trailing_stop, profit_lock_stop)
                        # Only write if water or stop actually moved
                        water_changed = abs(new_water - _t1_water) > 0.0001
                        stop_moved    = abs(new_stop - stop_current) > 0.001
                        if water_changed or stop_moved:
                            c.execute(
                                "UPDATE paper_trades SET stop=?, t1_water_mark=? WHERE id=?",
                                (new_stop, new_water, row["id"]),
                            )
                            c.commit()
                            _t1_water    = new_water
                            stop_current = new_stop

                # ── 1. Hard close at 3:45 PM ET (PRD — non-overridable) ────────
                if is_hard_close_window():
                    exit_reason  = "EOD_HARD_CLOSE_3:45PM"
                    close_shares = shares_rem

                # ── 1.5. Smart EOD pre-close during CLOSING_CAUTION (3:30–3:44) ─
                elif is_closing_caution():
                    momentum_ok = _eod_momentum_favors(df, direction)
                    from agent.config_manager import config as _cfg_eod
                    _eod_strong  = float(_cfg_eod.get("paper.eod_strong_winner_pct",  0.5))
                    _eod_small   = float(_cfg_eod.get("paper.eod_small_winner_pct",   0.1))
                    _eod_loss    = float(_cfg_eod.get("paper.eod_loss_threshold_pct", -0.3))
                    _eod_trail   = float(_cfg_eod.get("paper.eod_trail_stop_pct",     0.003))
                    _eod_recov   = float(_cfg_eod.get("paper.eod_recovery_stop_pct",  0.002))
                    if pnl_pct_now >= _eod_strong:
                        if momentum_ok:
                            # Strong winner, momentum still in our favor — trail stop
                            tight_stop = (
                                round(ep * (1 - _eod_trail), 4) if direction == "BUY"
                                else round(ep * (1 + _eod_trail), 4)
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
                    elif pnl_pct_now >= _eod_small:
                        # Small winner — take it, not worth the risk so close to EOD
                        exit_reason  = "EOD_LOCK_PROFIT"
                        close_shares = shares_rem
                    elif pnl_pct_now >= _eod_loss:
                        # Breakeven zone — exit
                        exit_reason  = "EOD_BREAKEVEN_EXIT"
                        close_shares = shares_rem
                    else:
                        # Meaningful loss
                        if momentum_ok:
                            # Still moving in our direction — tighten stop, hope for recovery
                            tight_stop = (
                                round(ep * (1 - _eod_recov), 4) if direction == "BUY"
                                else round(ep * (1 + _eod_recov), 4)
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
                        # Profit-lock stop: above entry by profit_lock_r × risk for BUY.
                        # Converts breakeven exits into small real wins.
                        from agent.config_manager import config as _cfg_be
                        _profit_lock_r_t1 = float(_cfg_be.get("paper.t1_profit_lock_r", 0.20))
                        _risk_dist_t1     = abs(t1_price - entry)  # 1.0R
                        be_stop = round(
                            entry + _risk_dist_t1 * _profit_lock_r_t1 if direction == "BUY"
                            else entry - _risk_dist_t1 * _profit_lock_r_t1,
                            4,
                        )
                        c.execute("""
                            UPDATE paper_trades
                            SET t1_hit=1, breakeven_set=1, stop=?, t1_water_mark=?,
                                partial_pnl_dollar=?, shares_remaining=?
                            WHERE id=?
                        """, (be_stop, t1_price, round(new_partial_pnl, 2), new_shares_rem, row["id"]))
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
                                          shares_total, partial_pnl, shares_total, ticker=ticker,
                                          ideal_exit_price=t1_price)
                            closed_any = True
                            won_any    = partial_pnl > 0
                            continue

                # ── Execution realism helpers (Phase 6) ───────────────────────
                # ATR and avg_volume for fill model; derived from the bar DataFrame.
                _atr_fm  = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns and not df["atr_14"].isna().all() else stop_current * 0.01
                _avol_fm = float(df["Volume"].mean() * 390) if "Volume" in df.columns else 0.0

                # ── 3. T2 full exit (2R profit) — only if T1 already hit ──────
                if not exit_reason and t1_hit and t2_price > 0:
                    t2_hit_now = (
                        (direction == "BUY"  and _hi >= t2_price) or
                        (direction == "SELL" and _lo <= t2_price)
                    )
                    if t2_hit_now:
                        exit_reason  = "TARGET_T2"
                        close_shares = shares_rem
                        ep           = t2_price   # limit order: fills at T2
                        _ideal_exit  = t2_price
                        _exit_slip_bps = 0.0

                # ── 4. Stop hit ────────────────────────────────────────────────
                if not exit_reason:
                    stop_hit = (
                        (direction == "BUY"  and _lo <= stop_current) or
                        (direction == "SELL" and _hi >= stop_current)
                    )
                    if stop_hit:
                        exit_reason  = "STOP_HIT_BREAKEVEN" if row["breakeven_set"] else "STOP_HIT"
                        close_shares = shares_rem
                        # Realistic stop fill: stop-market fills at stop ± slippage, NOT bar close.
                        # This also fixes the current bug where a stop fills at bar close even
                        # when the bar recovers above the stop after briefly touching it.
                        from agent.execution.fill_model import compute_stop_fill as _csf
                        _stop_fill   = _csf(direction, stop_current, _lo, _hi, _atr_fm, _sess, shares_rem, _avol_fm)
                        ep           = _stop_fill.fill_price
                        _ideal_exit  = stop_current
                        _exit_slip_bps = _stop_fill.slippage_bps
                        # Record pre-T1 stop for storm circuit detection
                        if not t1_hit and _algo_nm:
                            _pnl_calc = (
                                (ep - entry) * shares_rem if direction == "BUY"
                                else (entry - ep) * shares_rem
                            )
                            record_pre_t1_stop(_algo_nm, _sess, _pnl_calc)
                        # Ticker-level pre-T1 stop tracking
                        if not t1_hit:
                            record_ticker_pre_t1_stop(ticker)
                        # Flash-stop guard: record sub-60-second stop hits
                        if _algo_nm:
                            record_flash_stop(_algo_nm, _sess, _created_at)
                        # Record stop fill in paper_fills
                        try:
                            from agent.execution.paper_broker import record_fill as _rfill
                            _rfill(c, row["id"], ticker, direction, shares_rem, _stop_fill)
                        except Exception as _sfe:
                            logger.debug("[PAPER] stop fill record error: %s", _sfe)

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
                    _record_close(
                        c, row["id"], ep, exit_reason, entry, direction,
                        close_shares, partial_pnl, shares_total, ticker=ticker,
                        ideal_exit_price  = _ideal_exit,
                        ideal_entry_price = _entry_ideal,
                        entry_slip_bps    = _entry_slip_bps,
                        exit_slip_bps     = _exit_slip_bps,
                        entry_spread_usd  = _entry_spread,
                        session           = _sess,
                    )
                    closed_any = True
                    # Calculate net P&L for consecutive loss tracking
                    final_pnl = (
                        ((ep - entry) * close_shares if direction == "BUY"
                         else (entry - ep) * close_shares)
                        + partial_pnl
                    )
                    won_any = final_pnl > 0
                    # Ticker-level loss tracking for cooldown
                    if final_pnl < 0:
                        record_ticker_loss(ticker, final_pnl)
                    # Feed rolling EV tracker so adaptive floor adjusts in real-time
                    if _algo_nm:
                        record_algo_ev_outcome(_algo_nm, direction, _sess, final_pnl)

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
    trade_id:          int,
    exit_price:        float,
    exit_reason:       str,
    entry:             float,
    direction:         str,
    close_shares:      int,
    partial_pnl:       float = 0.0,
    total_shares:      int   = 0,
    ticker:            str   = "",
    # Attribution params (Phase 6)
    ideal_exit_price:  float = 0.0,
    ideal_entry_price: float = 0.0,
    entry_slip_bps:    float = 0.0,
    exit_slip_bps:     float = 0.0,
    entry_spread_usd:  float = 0.0,
    session:           str   = "",
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
    # Record execution attribution for strategy P&L vs actual P&L analysis
    try:
        from agent.execution.paper_broker import record_attribution as _ratrib
        _eff_ideal_entry = ideal_entry_price if ideal_entry_price > 0 else ep
        _eff_ideal_exit  = ideal_exit_price  if ideal_exit_price  > 0 else ep
        _eff_shares      = total_shares if total_shares > 0 else close_shares
        _ratrib(
            c, trade_id, ticker, direction, session,
            ideal_entry    = _eff_ideal_entry,
            actual_entry   = entry,
            ideal_exit     = _eff_ideal_exit,
            actual_exit    = ep,
            shares         = _eff_shares,
            actual_pnl     = round(pnl_dollar, 2),
            entry_slip_bps = entry_slip_bps,
            exit_slip_bps  = exit_slip_bps,
            entry_spread_usd = entry_spread_usd,
            attribution_note = exit_reason,
        )
    except Exception as _ae:
        logger.debug("[PAPER] attribution record error: %s", _ae)
    # Publish immediate trade-close event for algo feedback loop
    try:
        import json as _json, time as _time
        from agent.valkey_client import _get_client as _vk_c
        _vc = _vk_c()
        if _vc:
            _row_extra = c.execute(
                "SELECT algo_name, session, regime FROM paper_trades WHERE id=?",
                (trade_id,),
            ).fetchone()
            _vc.publish("trade:closed", _json.dumps({
                "outcome_id": f"paper:{trade_id}",
                "trade_id":   trade_id,
                "ticker":     ticker,
                "algo":       (_row_extra["algo_name"] if _row_extra else "") or "",
                "direction":  direction,
                "pnl_dollar": round(pnl_dollar, 2),
                "pnl_pct":    round(pnl_pct, 4),
                "exit_reason": exit_reason,
                "session":    (_row_extra["session"]  if _row_extra else "") or "",
                "regime":     (_row_extra["regime"]   if _row_extra else "") or "",
                "ts":         _time.time(),
            }))
            c.execute(
                "UPDATE paper_trades SET feedback_triggered_at=? "
                "WHERE id=? AND feedback_triggered_at IS NULL",
                (datetime.now(timezone.utc).isoformat(), trade_id),
            )
    except Exception:
        pass


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

    During AFTER_HOURS: only close regular-session trades (session != 'AFTER_HOURS')
    that leaked past the 3:45 PM hard close — intentional AH positions are left open.
    During HARD_CLOSE/CLOSED: close everything.

    Returns number of positions closed.
    """
    from agent.market_hours import is_after_hours, no_new_entries

    _is_ah = is_after_hours()
    if not no_new_entries() and not _is_ah:
        return 0  # Regular market hours — don't sweep

    with _lock:
        with _conn() as c:
            if _is_ah:
                # Only sweep regular-hour trades that leaked past 3:45 PM.
                # AH-opened positions (session='AFTER_HOURS') are intentional — leave them.
                rows = c.execute(
                    "SELECT id FROM paper_trades WHERE status='OPEN' "
                    "AND (session IS NULL OR session NOT IN ('AFTER_HOURS', 'PRE_MARKET'))"
                ).fetchall()
                stale_ids = [r["id"] for r in rows]
            else:
                count_row = c.execute(
                    "SELECT COUNT(*) AS n FROM paper_trades WHERE status='OPEN'"
                ).fetchone()
                stale_count = int(count_row["n"]) if count_row else 0
                stale_ids = list(range(stale_count))  # used only for len() guard below

    if not stale_ids and _is_ah:
        return 0

    if _is_ah:
        # Force-close only the leaked regular-hour positions
        from agent.data_fetcher import get_last_cached_close
        with _lock:
            with _conn() as c:
                rows = c.execute(
                    "SELECT id, ticker, direction, entry_price, "
                    "COALESCE(shares_remaining, shares, 1) as shares_rem, "
                    "COALESCE(partial_pnl_dollar, 0) as partial_pnl, "
                    "COALESCE(shares, 1) as shares_total "
                    "FROM paper_trades WHERE status='OPEN' "
                    "AND (session IS NULL OR session NOT IN ('AFTER_HOURS', 'PRE_MARKET'))"
                ).fetchall()
                closed = 0
                for row in rows:
                    ep = get_last_cached_close(row["ticker"])
                    if ep is None:
                        ep = float(row["entry_price"])
                    _record_close(
                        c, row["id"], ep, "STALE_REGULAR_HOUR_LEAKED_TO_AH",
                        float(row["entry_price"]), row["direction"],
                        int(row["shares_rem"]), float(row["partial_pnl"]),
                        int(row["shares_total"]), ticker=row["ticker"],
                    )
                    closed += 1
                c.commit()
        if closed:
            logger.warning(
                f"[PAPER] AH stale sweep: closed {closed} regular-hour position(s) "
                "that leaked past 3:45 PM hard close"
            )
            _trigger_paper_feedback()
        return closed

    logger.warning(
        f"[PAPER] Found {len(stale_ids)} stale open position(s) while market is closed — force-closing"
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
    from agent.config_manager import config as _cfg_eod2
    _eod_strong = float(_cfg_eod2.get("paper.eod_strong_winner_pct",  0.5))
    _eod_small  = float(_cfg_eod2.get("paper.eod_small_winner_pct",   0.1))
    _eod_loss   = float(_cfg_eod2.get("paper.eod_loss_threshold_pct", -0.3))
    _eod_trail  = float(_cfg_eod2.get("paper.eod_trail_stop_pct",     0.003))
    _eod_recov  = float(_cfg_eod2.get("paper.eod_recovery_stop_pct",  0.002))

    if pnl_pct >= _eod_strong:
        if momentum_ok:
            # Strong winner with momentum — trail stop to lock in most of the gain
            tight_stop = (
                round(ep * (1 - _eod_trail), 4) if direction == "BUY"
                else round(ep * (1 + _eod_trail), 4)
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

    elif pnl_pct >= _eod_small:
        # Small winner — lock it in regardless of momentum (not worth overnight risk)
        _record_close(c, row_id, ep, "EOD_LOCK_PROFIT",
                      entry, direction, shares_rem, partial, shares_tot, ticker=ticker)
        logger.info(
            f"[PAPER] EOD_LOCK_PROFIT {direction} {ticker} @ ${ep:.2f} gain={pnl_pct:+.2f}%"
        )
        acted = True

    elif pnl_pct >= _eod_loss:
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
                round(ep * (1 - _eod_recov), 4) if direction == "BUY"
                else round(ep * (1 + _eod_recov), 4)
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
        from agent.config_manager import config as _cfg
        stats = _build_paper_stats()
        _min_trades = int(_cfg.get("paper.filter_feedback_min_trades", 5))
        if stats["overall"]["total"] >= _min_trades:
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

    from agent.config_manager import config as _cfg
    budget = float(_cfg.get("paper.budget"))
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
                ROUND(AVG(COALESCE(pnl_pct, 0)), 4)              as avg_pnl_pct,
                ROUND(AVG(CASE WHEN pnl_dollar > 0 THEN pnl_dollar ELSE NULL END), 2) as avg_win_dollar,
                ROUND(AVG(CASE WHEN pnl_dollar <= 0 THEN pnl_dollar ELSE NULL END), 2) as avg_loss_dollar,
                ROUND(SUM(CASE WHEN pnl_dollar > 0 THEN pnl_dollar ELSE 0 END), 2) as gross_wins,
                ROUND(SUM(CASE WHEN pnl_dollar <= 0 THEN pnl_dollar ELSE 0 END), 2) as gross_losses,
                ROUND(MAX(pnl_dollar), 2) as best_trade,
                ROUND(MIN(pnl_dollar), 2) as worst_trade
            FROM paper_trades
            WHERE status='CLOSED'
              AND closed_at IS NOT NULL AND closed_at != ''
              AND (closed_at::timestamptz AT TIME ZONE 'America/New_York')::date
                    = (NOW() AT TIME ZONE 'America/New_York')::date
        """).fetchone()

    from agent.config_manager import config as _cfg
    budget = float(_cfg.get("paper.budget"))
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
                                  shares_r, partial, int(row["shares"]), ticker=ticker,
                                  ideal_exit_price=stp)
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
                                      shares_r, partial, int(row["shares"]), ticker=ticker,
                                      ideal_exit_price=t2)
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
                        from agent.config_manager import config as _cfg_be2
                        _be_off2    = float(_cfg_be2.get("paper.breakeven_stop_offset", 0.02))
                        be_stop     = round(entry + _be_off2, 4) if d == "BUY" else round(entry - _be_off2, 4)
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

    from agent.config_manager import config as _cfg
    budget = float(_cfg.get("paper.budget"))
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
        today_row = c.execute("""
            SELECT ROUND(SUM(COALESCE(pnl_dollar,0)),2) as today_pnl
            FROM paper_trades
            WHERE status='CLOSED'
              AND closed_at IS NOT NULL AND closed_at != ''
              AND (closed_at::timestamptz AT TIME ZONE 'America/New_York')::date
                    = (NOW() AT TIME ZONE 'America/New_York')::date
        """).fetchone()

    from agent.config_manager import config as _cfg
    budget        = float(_cfg.get("paper.budget"))
    max_trade_pct = float(_cfg.get("paper.max_trade_pct"))
    max_alloc_pct = float(_cfg.get("paper.max_allocated_pct"))
    max_open      = int(_cfg.get("paper.max_open_trades"))

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
    from agent.config_manager import config as _cfg
    cur_budget    = _cfg.get("paper.budget")
    cur_trade_pct = _cfg.get("paper.max_trade_pct")
    cur_alloc_pct = _cfg.get("paper.max_allocated_pct")
    cur_max_open  = _cfg.get("paper.max_open_trades")
    new_budget    = float(total_budget)      if total_budget      is not None else float(cur_budget)
    new_trade_pct = float(max_trade_pct)     if max_trade_pct     is not None else float(cur_trade_pct)
    new_alloc_pct = float(max_allocated_pct) if max_allocated_pct is not None else float(cur_alloc_pct)
    new_max_open  = int(max_open_trades)     if max_open_trades   is not None else int(cur_max_open)
    _cfg.set_many({
        "paper.budget": new_budget,
        "paper.max_trade_pct": new_trade_pct,
        "paper.max_allocated_pct": new_alloc_pct,
        "paper.max_open_trades": new_max_open,
    }, updated_by="dashboard")
    return {"total_budget": new_budget, "max_trade_pct": new_trade_pct,
            "max_allocated_pct": new_alloc_pct, "max_open_trades": new_max_open}


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
            single_signal = len(algo_signals) == 1
            for sig in algo_signals:
                # ML scores + filter_reason persist the full ML decision behind
                # every signal (the columns existed but were previously unwritten).
                _ml_scalp = sig.get("ml_scalp_prob")
                _ml_daily = sig.get("ml_daily_prob")
                _ml_swing = sig.get("ml_swing_prob")
                _ml_deep  = sig.get("ml_deep_prob")
                _exec_status = sig.get("exec_status")
                if "trade_opened" in sig:
                    _trade_opened = bool(sig.get("trade_opened"))
                else:
                    _trade_opened = bool(trade_opened and single_signal)
                if _trade_opened:
                    _exec_status = "EXECUTED_PAPER"
                else:
                    _exec_status = _exec_status or "SHADOW_LEARN_ONLY"
                c.execute(
                    """INSERT INTO algo_signal_log
                         (logged_at, ticker, algo, direction, confidence,
                          entry, stop, target, rr, trade_opened,
                          ml_scalp_prob, ml_daily_prob, ml_swing_prob, ml_deep_prob,
                          filter_reason, exec_status)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        now, ticker,
                        sig.get("algo", ""),
                        sig.get("direction", ""),
                        float(sig.get("confidence", 0)),
                        float(sig.get("entry", 0)),
                        float(sig.get("stop", 0)),
                        float(sig.get("target", 0)),
                        float(sig.get("rr", 0)),
                        int(_trade_opened),
                        float(_ml_scalp) if _ml_scalp is not None else None,
                        float(_ml_daily) if _ml_daily is not None else None,
                        float(_ml_swing) if _ml_swing is not None else None,
                        float(_ml_deep)  if _ml_deep  is not None else None,
                        sig.get("filter_reason", ""),
                        _exec_status,
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


# ── Daily trade analysis ───────────────────────────────────────────────────────

def _algo_family_label(algo_name: str | None) -> str:
    """Normalise raw algo_name into a short family label."""
    if not algo_name:
        return "unknown"
    n = algo_name.upper()
    if "BB_MEAN_REV" in n or "BB_REV" in n: return "bb_rev"
    if "MACD_ACC"   in n: return "macd_acc"
    if "SUPERTREND" in n: return "supertrend"
    if "ORB_ZV"     in n or "ORB5" in n or "ORB" in n: return "orb"
    if "RSI2_SNAP"  in n or "RSI2" in n: return "rsi2_snap"
    if "EMA_PULL"   in n or "EMA_PULLBACK" in n: return "ema_pull"
    if "VWAP_OFI"   in n: return "vwap_ofi"
    if "VWAP_TREND" in n or "VWAP_TOUCH" in n: return "vwap_trend"
    if "DONCHIAN"   in n: return "donchian"
    if "SQUEEZE"    in n: return "squeeze"
    if "VOL_SHOCK"  in n: return "vol_shock"
    if "KELTNER"    in n: return "keltner"
    if "META_ENS"   in n: return "meta_ens"
    if "PAIR_ARB"   in n: return "pair_arb"
    if "OFI"        in n: return "ofi"
    return algo_name.lower()[:20]


def _group_stats(rows: list[dict]) -> dict:
    n     = len(rows)
    wins  = sum(1 for r in rows if (r.get("pnl_dollar") or 0) > 0)
    pnls  = [float(r.get("pnl_dollar") or 0) for r in rows]
    gross_w = sum(p for p in pnls if p > 0)
    gross_l = sum(p for p in pnls if p <= 0)
    win_pnls = [p for p in pnls if p > 0]
    los_pnls = [p for p in pnls if p <= 0]
    return {
        "trades":    n,
        "wins":      wins,
        "losses":    n - wins,
        "win_pct":   round(wins / n * 100, 1) if n else 0.0,
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl":   round(sum(pnls) / n, 2) if n else 0.0,
        "avg_win":   round(gross_w / len(win_pnls), 2) if win_pnls else 0.0,
        "avg_loss":  round(gross_l / len(los_pnls), 2) if los_pnls else 0.0,
        "profit_factor": round(abs(gross_w / gross_l), 3) if gross_l else None,
    }


def get_trade_analysis(date_str: str | None = None) -> dict:
    """
    Day-end trade diagnostic.  Returns breakdown of closed trades for a given
    calendar date (ET timezone, defaults to today) across:
      exit_reason · confidence_band · algo_family · session · direction · t1_hit
      top 10 winning and losing tickers
    """
    import time as _time

    # Build the WHERE clause for the target date
    if date_str:
        # Passed as YYYY-MM-DD in ET; compare with AT TIME ZONE cast
        date_filter_pg = f"(closed_at::timestamptz AT TIME ZONE 'America/New_York')::date = '{date_str}'::date"
        date_filter_sq = f"date(closed_at) = '{date_str}'"
    else:
        date_filter_pg = "(closed_at::timestamptz AT TIME ZONE 'America/New_York')::date = (NOW() AT TIME ZONE 'America/New_York')::date"
        date_filter_sq = "date(closed_at) = date('now')"

    conn = _conn_ro()
    try:
        # Detect dialect: psycopg2 rows have .description; sqlite3 rows too — use param style
        try:
            conn.execute("SELECT 1::int")          # PostgreSQL-only syntax
            date_filter = date_filter_pg
        except Exception:
            date_filter = date_filter_sq

        rows = conn.execute(f"""
            SELECT
                ticker, direction, pnl_dollar, pnl_pct,
                exit_reason, confidence, session, regime,
                algo_name, t1_hit, bars_held,
                entry_price, exit_price
            FROM paper_trades
            WHERE status='CLOSED'
              AND closed_at IS NOT NULL AND closed_at != ''
              AND {date_filter}
        """).fetchall()
    except Exception as exc:
        logger.warning("[trade_analysis] query failed: %s", exc)
        return {"error": str(exc), "trades": 0}
    finally:
        conn.close()

    if not rows:
        label = date_str or "today"
        return {"date": label, "trades": 0, "message": f"No closed trades found for {label}"}

    all_rows = [dict(r) for r in rows]
    resolved_date = date_str or __import__("datetime").date.today().isoformat()

    # ── Summary ────────────────────────────────────────────────────────────────
    summary = _group_stats(all_rows)
    summary["date"] = resolved_date

    # ── By exit reason ─────────────────────────────────────────────────────────
    exit_buckets: dict[str, list] = {}
    for r in all_rows:
        key = (r.get("exit_reason") or "unknown").strip()
        exit_buckets.setdefault(key, []).append(r)

    by_exit = sorted(
        [{"exit_reason": k, **_group_stats(v)} for k, v in exit_buckets.items()],
        key=lambda x: x["trades"], reverse=True,
    )

    # ── By confidence band ────────────────────────────────────────────────────
    def _conf_band(conf):
        if conf is None: return "unknown"
        c = float(conf)
        if c < 40:  return "<40%"
        if c < 55:  return "40-55%"
        if c < 70:  return "55-70%"
        if c < 85:  return "70-85%"
        return "85%+"

    conf_buckets: dict[str, list] = {}
    for r in all_rows:
        conf_buckets.setdefault(_conf_band(r.get("confidence")), []).append(r)

    band_order = ["<40%", "40-55%", "55-70%", "70-85%", "85%+", "unknown"]
    by_confidence = [
        {"band": band, **_group_stats(conf_buckets[band])}
        for band in band_order if band in conf_buckets
    ]

    # ── By algo family ────────────────────────────────────────────────────────
    algo_buckets: dict[str, list] = {}
    for r in all_rows:
        algo_buckets.setdefault(_algo_family_label(r.get("algo_name")), []).append(r)

    by_algo = sorted(
        [{"algo": k, **_group_stats(v)} for k, v in algo_buckets.items()],
        key=lambda x: x["trades"], reverse=True,
    )

    # ── By session ────────────────────────────────────────────────────────────
    sess_buckets: dict[str, list] = {}
    for r in all_rows:
        sess_buckets.setdefault((r.get("session") or "unknown").strip(), []).append(r)

    by_session = sorted(
        [{"session": k, **_group_stats(v)} for k, v in sess_buckets.items()],
        key=lambda x: x["trades"], reverse=True,
    )

    # ── By direction ──────────────────────────────────────────────────────────
    dir_buckets: dict[str, list] = {}
    for r in all_rows:
        dir_buckets.setdefault((r.get("direction") or "?").strip(), []).append(r)

    by_direction = [
        {"direction": k, **_group_stats(v)} for k, v in dir_buckets.items()
    ]

    # ── T1 hit vs pre-T1 ──────────────────────────────────────────────────────
    t1_buckets: dict[str, list] = {
        "T1 Reached (partial exit)": [],
        "Pre-T1 stop / time":        [],
    }
    for r in all_rows:
        if r.get("t1_hit"):
            t1_buckets["T1 Reached (partial exit)"].append(r)
        else:
            t1_buckets["Pre-T1 stop / time"].append(r)

    by_t1 = [
        {"label": k, **_group_stats(v)} for k, v in t1_buckets.items() if v
    ]

    # ── By regime ─────────────────────────────────────────────────────────────
    regime_buckets: dict[str, list] = {}
    for r in all_rows:
        regime_buckets.setdefault((r.get("regime") or "unknown").strip(), []).append(r)

    by_regime = sorted(
        [{"regime": k, **_group_stats(v)} for k, v in regime_buckets.items()],
        key=lambda x: x["trades"], reverse=True,
    )

    # ── Top 10 worst and best tickers ─────────────────────────────────────────
    ticker_buckets: dict[str, list] = {}
    for r in all_rows:
        ticker_buckets.setdefault(r.get("ticker", "?"), []).append(r)

    ticker_stats = [
        {"ticker": k, **_group_stats(v)} for k, v in ticker_buckets.items()
        if len(v) >= 2
    ]
    top_losers  = sorted(ticker_stats, key=lambda x: x["total_pnl"])[:10]
    top_winners = sorted(ticker_stats, key=lambda x: x["total_pnl"], reverse=True)[:10]

    return {
        "date":          resolved_date,
        "summary":       summary,
        "by_exit":       by_exit,
        "by_confidence": by_confidence,
        "by_algo":       by_algo,
        "by_session":    by_session,
        "by_direction":  by_direction,
        "by_t1":         by_t1,
        "by_regime":     by_regime,
        "top_losers":    top_losers,
        "top_winners":   top_winners,
    }
