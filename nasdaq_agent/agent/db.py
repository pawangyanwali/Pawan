"""
Database abstraction — PostgreSQL/RDS only via psycopg2.

The public interface mirrors the sqlite3 Connection API so existing callers
work unchanged.  db_path arguments are accepted but ignored (all data lives
in RDS; the path parameter remains for call-site compatibility only).

PostgreSQL notes:
  - SQL placeholder '?' is converted to '%s'
  - PRAGMA statements are silently ignored
  - 'INTEGER PRIMARY KEY AUTOINCREMENT' DDL is rewritten to 'SERIAL PRIMARY KEY'
  - lastrowid is populated via SELECT lastval() after INSERT
  - Rows are RealDictRow instances (accessible by column name, same as sqlite3.Row)

Connection pool: ThreadedConnectionPool(2, 20) shared across all modules.
PGHOST must be set in the environment; raises RuntimeError on startup if missing.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool

logger = logging.getLogger(__name__)

# ── Connection pool ────────────────────────────────────────────────────────────

_pg_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()
_pool_error: Optional[Exception] = None


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pg_pool, _pool_error
    if _pg_pool is not None:
        return _pg_pool
    if _pool_error is not None:
        raise _pool_error
    with _pool_lock:
        if _pg_pool is not None:
            return _pg_pool
        if _pool_error is not None:
            raise _pool_error
        host = os.environ.get("PGHOST")
        if not host:
            _pool_error = RuntimeError(
                "[DB] PGHOST is not set. PostgreSQL is required — "
                "configure PGHOST (and optionally PGPORT, PGDATABASE, PGUSER, "
                "PGPASSWORD) in .env and restart."
            )
            raise _pool_error
        try:
            _pg_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=20,
                host=host,
                port=int(os.getenv("PGPORT", "5432")),
                dbname=os.getenv("PGDATABASE", "nasdaq_agent"),
                user=os.getenv("PGUSER", "nasdaq"),
                password=os.getenv("PGPASSWORD", ""),
                connect_timeout=5,
                sslmode=os.getenv("PGSSLMODE", "require"),
            )
            logger.info(
                "[DB] PostgreSQL pool created → %s:%s/%s",
                host,
                os.getenv("PGPORT", "5432"),
                os.getenv("PGDATABASE", "nasdaq_agent"),
            )
        except Exception as exc:
            _pool_error = RuntimeError(f"[DB] PostgreSQL pool creation failed: {exc}")
            raise _pool_error
        return _pg_pool


# ── SQL dialect translation (SQLite → PostgreSQL) ──────────────────────────────

_PH_RE = re.compile(r"\?")
_AUTOINCREMENT_RE = re.compile(
    r"INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT", re.IGNORECASE
)

# Timestamps in the DB are Python ISO strings like '2026-05-22T05:43:04.657000+00:00'.
_TS_FMT = "to_char(NOW() AT TIME ZONE 'UTC' + INTERVAL '%s', 'YYYY-MM-DD\"T\"HH24:MI:SS')"

_DT_FIXED_RE = re.compile(r"datetime\('now'\s*,\s*'([^']+)'\)", re.IGNORECASE)
_DT_PARAM_CONCAT_RE = re.compile(
    r"datetime\('now'\s*,\s*\?\s*\|\|\s*'\s*days\s*'\)", re.IGNORECASE
)
_DT_PARAM_RE = re.compile(r"datetime\('now'\s*,\s*\?\)", re.IGNORECASE)
_DATE_FIXED_RE = re.compile(r"date\('now'\s*,\s*'([^']+)'\)", re.IGNORECASE)
_DATE_PARAM_RE = re.compile(r"date\('now'\s*,\s*\?\)", re.IGNORECASE)
_DATE_NOW_RE   = re.compile(r"date\('now'\)", re.IGNORECASE)
_DATE_COL_RE   = re.compile(r"\bdate\((\w+)\)", re.IGNORECASE)
_STRFTIME_WEEK_RE = re.compile(r"strftime\('%Y-W%W'\s*,\s*(\w+)\)", re.IGNORECASE)
_ROUND_START_RE = re.compile(r'\bROUND\s*\(', re.IGNORECASE)


def _add_numeric_cast_to_round(sql: str) -> str:
    """Wrap ROUND()'s first arg with ::NUMERIC so it works in PostgreSQL.

    PostgreSQL has ROUND(numeric, int) but NOT ROUND(double precision, int).
    Transforms ROUND(expr, N) → ROUND((expr)::NUMERIC, N) unless the cast is
    already present.  Uses balanced-paren tracking so nested calls are handled
    correctly.
    """
    out: list[str] = []
    i = 0
    while i < len(sql):
        m = _ROUND_START_RE.search(sql, i)
        if m is None:
            out.append(sql[i:])
            break
        out.append(sql[i:m.end()])
        j = m.end()
        depth = 1
        while j < len(sql) and depth > 0:
            if sql[j] == '(':
                depth += 1
            elif sql[j] == ')':
                depth -= 1
            j += 1
        content = sql[m.end():j - 1]
        # Last ', N' is the precision argument; everything before is the expression.
        m2 = re.search(r',\s*\d+\s*$', content)
        if m2 and '::numeric' not in content[:m2.start()].lower():
            expr = content[:m2.start()]
            prec = content[m2.start():]
            out.append(f'({expr})::NUMERIC{prec})')
        else:
            out.append(content + ')')
        i = j
    return ''.join(out)


def _to_pg(sql: str) -> str:
    """Convert SQLite SQL dialect to PostgreSQL."""
    sql = _AUTOINCREMENT_RE.sub("SERIAL PRIMARY KEY", sql)

    sql = _DT_FIXED_RE.sub(lambda m: _TS_FMT % m.group(1), sql)
    sql = _DT_PARAM_CONCAT_RE.sub(
        "to_char(NOW() AT TIME ZONE 'UTC' + (? || ' days')::interval,"
        " 'YYYY-MM-DD\"T\"HH24:MI:SS')",
        sql,
    )
    sql = _DT_PARAM_RE.sub(
        "to_char(NOW() AT TIME ZONE 'UTC' + ?::interval,"
        " 'YYYY-MM-DD\"T\"HH24:MI:SS')",
        sql,
    )
    sql = _DATE_FIXED_RE.sub(
        lambda m: f"(CURRENT_DATE + INTERVAL '{m.group(1)}')::text",
        sql,
    )
    sql = _DATE_PARAM_RE.sub("(CURRENT_DATE + ?::INTERVAL)::text", sql)
    sql = _DATE_NOW_RE.sub("CURRENT_DATE::text", sql)
    sql = _DATE_COL_RE.sub(lambda m: f"({m.group(1)}::timestamptz::date::text)", sql)
    sql = _STRFTIME_WEEK_RE.sub(
        lambda m: f"TO_CHAR({m.group(1)}::timestamptz, 'IYYY-IW')", sql
    )

    sql = _add_numeric_cast_to_round(sql)

    sql = re.sub(
        r'\bADD\s+COLUMN\b(?!\s+IF\s+NOT\s+EXISTS)',
        'ADD COLUMN IF NOT EXISTS',
        sql, flags=re.IGNORECASE,
    )

    # ? → %s must be last (after all regex that use literal ?)
    sql = _PH_RE.sub("%s", sql)
    return sql


# ── Cursor wrappers ────────────────────────────────────────────────────────────

class _NullCursor:
    """Returned for PRAGMA statements — always empty."""
    lastrowid = None

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def __iter__(self):
        return iter([])


class _PgCursor:
    """Wraps a psycopg2 RealDictCursor; exposes .lastrowid."""

    def __init__(self, cur, lastrowid=None):
        self._cur = cur
        self.lastrowid = lastrowid
        self.rowcount = cur.rowcount

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def __iter__(self):
        return iter(self._cur.fetchall())


# ── Connection wrapper ─────────────────────────────────────────────────────────

class _PgConnection:
    """
    Wraps a psycopg2 connection to match the sqlite3.Connection API.
    Translates ? → %s placeholders, suppresses PRAGMAs, populates lastrowid.
    """

    def __init__(self):
        pool = _get_pool()
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

        try:
            self._conn.rollback()
        except Exception:
            pass
        self._conn.autocommit = False

        # Health-check: swap out stale connections before any real query fails.
        try:
            _hc = self._conn.cursor()
            _hc.execute("SELECT 1")
            _hc.close()
            self._conn.rollback()
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

        # Savepoint guard: prevents a failing DDL/DML from aborting the whole
        # transaction (PostgreSQL behaviour differs from SQLite here).
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
            # lastval() fails on tables with no SERIAL/sequence (e.g. composite PKs).
            # A bare `except: pass` leaves the transaction in an aborted state,
            # causing every subsequent execute() in the same with-block to fail.
            # Use a savepoint so a failure rolls back cleanly.
            lv_sp = self._conn.cursor()
            lv_sp.execute("SAVEPOINT _lastval_sp")
            try:
                id_cur = self._conn.cursor()
                id_cur.execute("SELECT lastval()")
                row = id_cur.fetchone()
                lastrowid = row[0] if row else None
                id_cur.close()
                lv_sp.execute("RELEASE SAVEPOINT _lastval_sp")
            except Exception:
                try:
                    lv_sp.execute("ROLLBACK TO SAVEPOINT _lastval_sp")
                except Exception:
                    pass
            finally:
                lv_sp.close()

        return _PgCursor(cur, lastrowid)

    def executemany(self, sql: str, params_list) -> "_PgCursor | _NullCursor":
        stripped = sql.strip().upper()
        if stripped.startswith("PRAGMA"):
            return _NullCursor()
        pg_sql = _to_pg(sql)
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        psycopg2.extras.execute_batch(cur, pg_sql, params_list, page_size=500)
        return _PgCursor(cur)

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


# ── Public API ─────────────────────────────────────────────────────────────────

def get_conn(db_path=None, read_only: bool = False) -> _PgConnection:
    """
    Return a PostgreSQL connection from the shared pool.

    db_path is accepted for call-site compatibility but ignored —
    all data lives in RDS (PGHOST).

    Use as a context manager:
        with get_conn() as c:
            c.execute("INSERT ...", params)
    """
    return _PgConnection()


def using_postgres() -> bool:
    """Always True — PostgreSQL is the only supported backend."""
    return True
