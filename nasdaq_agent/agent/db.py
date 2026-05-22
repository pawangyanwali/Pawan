"""
Database abstraction — wraps psycopg2 (PostgreSQL/RDS) or sqlite3.

The public interface mirrors the sqlite3 Connection API so existing modules
need only swap their import and _conn() implementation.

PostgreSQL is used when PGHOST is set in the environment.
When using PostgreSQL:
  - SQL placeholder '?' is converted to '%s'
  - PRAGMA statements are silently ignored
  - 'INTEGER PRIMARY KEY AUTOINCREMENT' DDL is rewritten to 'SERIAL PRIMARY KEY'
  - lastrowid is populated via SELECT lastval() after INSERT
  - Rows are RealDictRow instances (accessible by column name, same as sqlite3.Row)

Connection pool: ThreadedConnectionPool(1, 5) — shared across all three
SQLite database files (paper_trades, signal_history, live_backtest all map
to the single RDS database).
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_USE_PG: bool = bool(os.getenv("PGHOST"))

# ── PostgreSQL path ────────────────────────────────────────────────────────────

if _USE_PG:
    try:
        import psycopg2
        import psycopg2.extras
        import psycopg2.pool
        _psycopg2_ok = True
    except ImportError:
        _psycopg2_ok = False
        _USE_PG = False
        logger.warning("[DB] psycopg2 not installed — falling back to SQLite")

_pg_pool: Optional[object] = None
_pool_lock = threading.Lock()
_pg_failed = False   # set True on first connection error so we stop retrying


def _get_pool():
    global _pg_pool, _pg_failed, _USE_PG
    if _pg_failed:
        return None
    if _pg_pool is not None:
        return _pg_pool
    with _pool_lock:
        if _pg_pool is not None:
            return _pg_pool
        if _pg_failed:
            return None
        try:
            _pg_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=20,
                host=os.environ["PGHOST"],
                port=int(os.getenv("PGPORT", "5432")),
                dbname=os.getenv("PGDATABASE", "nasdaq_agent"),
                user=os.getenv("PGUSER", "nasdaq"),
                password=os.getenv("PGPASSWORD", ""),
                connect_timeout=5,
                sslmode=os.getenv("PGSSLMODE", "require"),
            )
            logger.info(
                f"[DB] PostgreSQL pool created → "
                f"{os.environ['PGHOST']}:{os.getenv('PGPORT','5432')}"
                f"/{os.getenv('PGDATABASE','nasdaq_agent')}"
            )
        except Exception as exc:
            _pg_failed = True
            _USE_PG = False
            logger.warning(
                f"[DB] PostgreSQL unavailable ({exc}) — falling back to SQLite. "
                f"Set PGHOST correctly in .env and restart to enable RDS."
            )
        return _pg_pool


_PH_RE = re.compile(r"\?")
_AUTOINCREMENT_RE = re.compile(
    r"INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT", re.IGNORECASE
)

# SQLite date/time function patterns — converted to PostgreSQL equivalents.
# Timestamps in the DB are Python ISO strings like '2026-05-22T05:43:04.657000+00:00'.
# to_char with 'YYYY-MM-DD"T"HH24:MI:SS' produces the same prefix so >= comparisons work.
_TS_FMT = "to_char(NOW() AT TIME ZONE 'UTC' + INTERVAL '%s', 'YYYY-MM-DD\"T\"HH24:MI:SS')"

# datetime('now', '-30 days')  or  datetime('now', '-10 minutes')  [fixed modifier]
_DT_FIXED_RE = re.compile(r"datetime\('now'\s*,\s*'([^']+)'\)", re.IGNORECASE)
# datetime('now', ? || ' days')  [parameterised concat — param is like '-7']
_DT_PARAM_CONCAT_RE = re.compile(
    r"datetime\('now'\s*,\s*\?\s*\|\|\s*'\s*days\s*'\)", re.IGNORECASE
)
# datetime('now', ?)  [parameterised — param is like '-30 days']
_DT_PARAM_RE = re.compile(r"datetime\('now'\s*,\s*\?\)", re.IGNORECASE)
# date('now')  e.g.  closed_at >= date('now', '-84 days')  or  = date('now')
_DATE_FIXED_RE = re.compile(r"date\('now'\s*,\s*'([^']+)'\)", re.IGNORECASE)
_DATE_NOW_RE   = re.compile(r"date\('now'\)", re.IGNORECASE)
# date(col_name)  e.g.  date(closed_at) = ...
_DATE_COL_RE   = re.compile(r"\bdate\((\w+)\)", re.IGNORECASE)
# strftime('%Y-W%W', col)  →  TO_CHAR(col::timestamptz, 'IYYY-IW')
_STRFTIME_WEEK_RE = re.compile(r"strftime\('%Y-W%W'\s*,\s*(\w+)\)", re.IGNORECASE)


def _to_pg(sql: str) -> str:
    """Convert SQLite SQL to PostgreSQL SQL."""
    sql = _AUTOINCREMENT_RE.sub("SERIAL PRIMARY KEY", sql)

    # Date/time conversions (process most-specific patterns first, before ? → %s)
    sql = _DT_FIXED_RE.sub(
        lambda m: _TS_FMT % m.group(1),
        sql,
    )
    # datetime('now', ? || ' days') — param is e.g. '-7'; becomes '-7 days' interval
    sql = _DT_PARAM_CONCAT_RE.sub(
        "to_char(NOW() AT TIME ZONE 'UTC' + (? || ' days')::interval,"
        " 'YYYY-MM-DD\"T\"HH24:MI:SS')",
        sql,
    )
    # datetime('now', ?) — param is already a full interval string e.g. '-30 days'
    sql = _DT_PARAM_RE.sub(
        "to_char(NOW() AT TIME ZONE 'UTC' + ?::interval,"
        " 'YYYY-MM-DD\"T\"HH24:MI:SS')",
        sql,
    )
    # date('now', '-84 days') — fixed date modifier
    sql = _DATE_FIXED_RE.sub(
        lambda m: f"(CURRENT_DATE + INTERVAL '{m.group(1)}')::text",
        sql,
    )
    # date('now') standalone — today's date as text for comparison with TEXT col
    sql = _DATE_NOW_RE.sub("CURRENT_DATE::text", sql)
    # date(col) — extract date portion from a TEXT timestamp column
    sql = _DATE_COL_RE.sub(lambda m: f"({m.group(1)}::timestamptz::date::text)", sql)
    # strftime('%Y-W%W', col) — ISO week label for grouping
    sql = _STRFTIME_WEEK_RE.sub(
        lambda m: f"TO_CHAR({m.group(1)}::timestamptz, 'IYYY-IW')", sql
    )

    # ALTER TABLE ADD COLUMN → ADD COLUMN IF NOT EXISTS (idempotent; no abort on dup)
    sql = re.sub(
        r'\bADD\s+COLUMN\b(?!\s+IF\s+NOT\s+EXISTS)',
        'ADD COLUMN IF NOT EXISTS',
        sql, flags=re.IGNORECASE,
    )

    # SQLite ? placeholders → PostgreSQL %s  (must be last)
    sql = _PH_RE.sub("%s", sql)
    return sql


class _NullCursor:
    """Returned for PRAGMA statements on PostgreSQL — always empty."""
    lastrowid = None

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def __iter__(self):
        return iter([])


class _PgCursor:
    """Wraps a psycopg2 RealDictCursor to expose .lastrowid and match sqlite3 Cursor."""

    def __init__(self, cur, lastrowid=None):
        self._cur = cur
        self.lastrowid = lastrowid

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def __iter__(self):
        return iter(self._cur.fetchall())


class _PgConnection:
    """
    Wraps a psycopg2 connection to match the sqlite3.Connection API used by the codebase.
    Translates ? → %s placeholders, suppresses PRAGMAs, and populates lastrowid.
    """

    def __init__(self):
        pool = _get_pool()
        if pool is None:
            raise RuntimeError("PostgreSQL pool unavailable")
        # Retry up to 3 times if pool is temporarily exhausted under burst load
        last_exc: Exception | None = None
        for _attempt in range(3):
            try:
                self._conn = pool.getconn()
                break
            except psycopg2.pool.PoolError as exc:
                last_exc = exc
                if _attempt < 2:
                    time.sleep(0.05 * (2 ** _attempt))   # 50ms, 100ms
        else:
            raise RuntimeError(f"DB pool exhausted after 3 retries: {last_exc}")

        # A pooled connection may have a leftover implicit transaction from its
        # previous use.  set_session (autocommit=False) raises ProgrammingError
        # if called inside a transaction, so we rollback first to get a clean
        # slate, then set autocommit, then run the health check.
        try:
            self._conn.rollback()
        except Exception:
            pass
        self._conn.autocommit = False

        # Health-check: if the connection is stale (RDS idle timeout, network blip)
        # swap it out before any real query fails mid-transaction.
        try:
            _hc = self._conn.cursor()
            _hc.execute("SELECT 1")
            _hc.close()
            self._conn.rollback()   # clean up the implicit txn opened by SELECT 1
        except Exception:
            try:
                pool.putconn(self._conn, close=True)
            except Exception:
                pass
            self._conn = pool.getconn()
            try:
                self._conn.rollback()
            except Exception:
                pass
            self._conn.autocommit = False

    def execute(self, sql: str, params=None) -> "_PgCursor | _NullCursor":
        stripped = sql.strip().upper()
        if stripped.startswith("PRAGMA"):
            return _NullCursor()

        pg_sql = _to_pg(sql)
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Savepoint guard: a failing DDL/DML statement inside a multi-statement
        # transaction aborts the whole PostgreSQL transaction (unlike SQLite which
        # handles per-statement errors independently). Rolling back to a savepoint
        # lets callers catch the error and continue — exactly what _migrate_columns
        # does when it tries to ADD COLUMN for columns that already exist.
        sp = self._conn.cursor()
        sp.execute("SAVEPOINT _dbsp")
        try:
            cur.execute(pg_sql, params or [])
            sp.execute("RELEASE SAVEPOINT _dbsp")
        except Exception:
            sp.execute("ROLLBACK TO SAVEPOINT _dbsp")
            raise

        lastrowid = None
        if stripped.startswith("INSERT"):
            try:
                id_cur = self._conn.cursor()
                id_cur.execute("SELECT lastval()")
                row = id_cur.fetchone()
                lastrowid = row[0] if row else None
                id_cur.close()
            except Exception:
                pass

        return _PgCursor(cur, lastrowid)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type:
                self._conn.rollback()
            else:
                self._conn.commit()
        finally:
            _get_pool().putconn(self._conn)
        return False

    def close(self):
        try:
            _get_pool().putconn(self._conn)
        except Exception:
            pass


# ── SQLite path ────────────────────────────────────────────────────────────────

class _SqliteConnection:
    """
    Thin wrapper around sqlite3.Connection that matches _PgConnection's API.
    Adds row_factory and WAL pragma so callers don't need to set them.
    """

    def __init__(self, path: Path, read_only: bool = False, timeout: int = 10):
        path.parent.mkdir(parents=True, exist_ok=True)
        if read_only:
            self._conn = sqlite3.connect(str(path), timeout=5, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA query_only=ON")
        else:
            self._conn = sqlite3.connect(str(path), timeout=timeout)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")

    def execute(self, sql: str, params=None):
        if params is None:
            return self._conn.execute(sql)
        return self._conn.execute(sql, params)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._conn.rollback()
        else:
            self._conn.commit()
        self._conn.close()
        return False

    def close(self):
        self._conn.close()


# ── Public API ─────────────────────────────────────────────────────────────────

def get_conn(db_path: Path, read_only: bool = False) -> _PgConnection | _SqliteConnection:
    """
    Return a database connection.

    If PGHOST is set → PostgreSQL (db_path ignored).
    Otherwise → SQLite at db_path.

    Use as a context manager:
        with get_conn(path) as c:
            c.execute("INSERT ...", params)
            c.commit()
    """
    if _USE_PG:
        pool = _get_pool()
        if pool is not None:
            return _PgConnection()
        # Pool creation failed — _USE_PG was set False inside _get_pool(); use SQLite
    return _SqliteConnection(db_path, read_only=read_only)


def using_postgres() -> bool:
    """True if the runtime database backend is PostgreSQL."""
    return _USE_PG
