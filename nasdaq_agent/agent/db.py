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
                minconn=1,
                maxconn=8,
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


def _to_pg(sql: str) -> str:
    """Convert SQLite SQL to PostgreSQL SQL."""
    sql = _PH_RE.sub("%s", sql)
    sql = _AUTOINCREMENT_RE.sub("SERIAL PRIMARY KEY", sql)
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
        self._conn = pool.getconn()
        self._conn.autocommit = False

    def execute(self, sql: str, params=None) -> _PgCursor | _NullCursor:
        stripped = sql.strip().upper()
        if stripped.startswith("PRAGMA"):
            return _NullCursor()

        pg_sql = _to_pg(sql)
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(pg_sql, params or [])

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
