"""
Shared fixtures for all tests.
"""
import os
import re
import sqlite3
import sys
import types
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Ensure nasdaq_agent is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── ta library stub ────────────────────────────────────────────────────────────
_TA_IS_REAL = False
try:
    import ta as _ta_check
    _TA_IS_REAL = hasattr(getattr(_ta_check, "trend", None), "ema_indicator")
except ImportError:
    pass

if not _TA_IS_REAL and "ta" not in sys.modules:
    _ta_stub = types.ModuleType("ta")
    _ta_stub._is_stub = True
    for _sub in ("trend", "momentum", "volatility", "volume", "others"):
        _m = types.ModuleType(f"ta.{_sub}")
        _ta_stub.__dict__[_sub] = _m
        sys.modules[f"ta.{_sub}"] = _m
    sys.modules["ta"] = _ta_stub

collect_ignore: list[str] = []
if not _TA_IS_REAL:
    _tests_dir = Path(__file__).parent
    collect_ignore = [
        str(_tests_dir / "test_prediction.py"),
        str(_tests_dir / "test_technical.py"),
    ]


# ── In-memory SQLite adapter (test-only, mimics _PgConnection API) ─────────────

def _sqlite_to_test_sql(sql: str) -> str:
    """Translate SQL from PostgreSQL dialect to SQLite for tests."""
    sql = sql.replace("%s", "?")
    sql = re.sub(r'ON CONFLICT\s*\([^)]+\)\s*DO UPDATE SET[^;]*', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'ON CONFLICT\s*\([^)]+\)\s*DO NOTHING', 'OR IGNORE', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bON CONFLICT DO NOTHING\b', 'OR IGNORE', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\s+RETURNING\s+\w+', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bSERIAL PRIMARY KEY\b', 'INTEGER PRIMARY KEY AUTOINCREMENT', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bDOUBLE PRECISION\b', 'REAL', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bTIMESTAMPTZ\b', 'TEXT', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bJSONB\b', 'TEXT', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bNOW\s*\(\s*\)', "CURRENT_TIMESTAMP", sql, flags=re.IGNORECASE)
    sql = re.sub(r'::\w+', '', sql)
    sql = re.sub(r'\bSAVEPOINT\s+\w+\b', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bRELEASE\s+SAVEPOINT\s+\w+\b', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bROLLBACK\s+TO\s+SAVEPOINT\s+\w+\b', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bHAVING\s+COUNT\s*\(\s*\*\s*\)\s*(>=|>|<=|<|=)\s*\d+', lambda m: f'HAVING COUNT(*) {m.group(1)} {m.group(0).split()[-1]}', sql, flags=re.IGNORECASE)
    return sql.strip()


class _DualRow(dict):
    """Dict-like row that also supports integer index access (for legacy code)."""
    def __init__(self, keys, values):
        super().__init__(zip(keys, values))
        self._vals = list(values)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._vals[key]
        return super().__getitem__(key)


class _TestCursor:
    def __init__(self, cur: sqlite3.Cursor):
        self._cur = cur
        self.lastrowid = cur.lastrowid

    def _wrap(self, row):
        if row is None:
            return None
        desc = self._cur.description or []
        keys = [d[0] for d in desc]
        return _DualRow(keys, row)

    def fetchone(self):
        return self._wrap(self._cur.fetchone())

    def fetchall(self):
        rows = self._cur.fetchall()
        if not rows:
            return []
        desc = self._cur.description or []
        keys = [d[0] for d in desc]
        return [_DualRow(keys, r) for r in rows]

    def __iter__(self):
        return iter(self.fetchall())


class _TestConnection:
    """In-memory SQLite wrapper that mimics _PgConnection's public API."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def execute(self, sql: str, params=None) -> _TestCursor:
        sql = _sqlite_to_test_sql(sql)
        if not sql:
            return _TestCursor(self._conn.cursor())
        try:
            cur = self._conn.execute(sql, params or [])
        except Exception:
            cur = self._conn.cursor()
        tc = _TestCursor(cur)
        tc.lastrowid = cur.lastrowid
        return tc

    def executemany(self, sql: str, params_list) -> _TestCursor:
        sql = _sqlite_to_test_sql(sql)
        if not sql or not params_list:
            return _TestCursor(self._conn.cursor())
        cur = self._conn.executemany(sql, params_list)
        return _TestCursor(cur)

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
        return False

    def close(self):
        pass


# Shared in-memory SQLite per-test-session
_TEST_DB: sqlite3.Connection | None = None


def _reset_test_db():
    global _TEST_DB
    _TEST_DB = sqlite3.connect(":memory:", check_same_thread=False)
    _TEST_DB.execute("""
        CREATE TABLE IF NOT EXISTS system_kv (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)


def _get_test_db() -> sqlite3.Connection:
    global _TEST_DB
    if _TEST_DB is None:
        _TEST_DB = sqlite3.connect(":memory:", check_same_thread=False)
    return _TEST_DB


def _make_test_get_conn():
    def _get_conn(db_path=None, read_only: bool = False) -> _TestConnection:
        if db_path is not None:
            # Use a real SQLite file so tests that verify via sqlite3.connect(path) see the data
            conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
        else:
            conn = _get_test_db()
        return _TestConnection(conn)
    return _get_conn


class _FakeRawConn:
    """Wraps in-memory SQLite to look like psycopg2 raw connection for init_db() calls."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self.autocommit = False

    def cursor(self, **_):
        return _FakeRawCursor(self._conn.cursor())

    def rollback(self):
        try: self._conn.rollback()
        except Exception: pass

    def commit(self):
        try: self._conn.commit()
        except Exception: pass

    def close(self): pass


class _FakeRawCursor:
    def __init__(self, cur: sqlite3.Cursor): self._cur = cur

    def execute(self, sql: str, *args):
        sql = _sqlite_to_test_sql(sql)
        if sql:
            try: self._cur.execute(sql)
            except Exception: pass

    def __enter__(self): return self
    def __exit__(self, *_): pass
    def close(self): pass


def _make_noop_get_pool():
    mock = MagicMock()
    mock.getconn.side_effect = lambda: _FakeRawConn(_get_test_db())
    mock.putconn.return_value = None
    return mock


# ── Autouse fixture ────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def tmp_db_paths(tmp_path, monkeypatch):
    """
    Patch agent.db.get_conn (and all module-level imports of it) so tests use
    an in-memory SQLite database instead of PostgreSQL.
    """
    _reset_test_db()
    _gc = _make_test_get_conn()
    _pool = _make_noop_get_pool()

    import agent.db as _db
    monkeypatch.setattr(_db, "get_conn", _gc)
    monkeypatch.setattr(_db, "_get_pool", lambda: _pool)
    monkeypatch.setattr(_db, "using_postgres", lambda: False)

    # Patch the name in every module that imported it via "from agent.db import ..."
    _modules_to_patch = [
        "agent.live_backtest",
        "agent.paper_trading",
        "agent.signal_tracker",
        "agent.after_hours_monitor",
        "agent.historical_cache",
        "agent.weekend_learner",
        "agent.multi_tf_backtest",
        "agent.backtester",
        "agent.scalp.store",
        "agent.scalp.shadow",
        "agent.scalp.execution_policy",
        "agent.scalp.learning",
        "agent.scalp.ml_trainer",
        "historical.store",
    ]
    import importlib
    for _mod_name in _modules_to_patch:
        try:
            _mod = importlib.import_module(_mod_name)
            if hasattr(_mod, "get_conn"):
                monkeypatch.setattr(_mod, "get_conn", _gc)
            if hasattr(_mod, "_get_pool"):
                monkeypatch.setattr(_mod, "_get_pool", lambda p=_pool: p)
            if _mod_name == "agent.scalp.store":
                monkeypatch.setattr(_mod, "_initialized", False)
            if _mod_name == "agent.scalp.learning":
                monkeypatch.setattr(_mod, "_gate_cache", {})
        except Exception:
            pass

    import agent.live_backtest as lb
    import agent.paper_trading as pt
    import agent.signal_tracker as st

    # NOTE: these modules migrated from per-file SQLite (_DB_PATH) to the shared
    # get_conn() pool, which is already redirected to in-memory SQLite above — so
    # there is no _DB_PATH to patch any more. init_db() creates the schema in the
    # patched test DB; that is the only setup still required.
    lb.init_db()
    pt.init_db()
    st.init_db()


# ── Synthetic OHLCV DataFrame factory ─────────────────────────────────────────
def make_ohlcv(
    n: int = 60,
    start_price: float = 100.0,
    trend: float = 0.0,
    volatility: float = 0.5,
    volume: int = 1_000_000,
) -> pd.DataFrame:
    """Return a realistic OHLCV DataFrame with `n` 1-minute bars."""
    rng = np.random.default_rng(42)
    closes = [start_price]
    for _ in range(n - 1):
        closes.append(closes[-1] + trend + rng.normal(0, volatility))
    closes = np.array(closes)

    noise = rng.uniform(0.1, 0.5, n)
    highs  = closes + noise
    lows   = closes - noise
    opens  = np.clip(np.roll(closes, 1), lows, highs)
    opens[0] = start_price

    idx = pd.date_range("2025-01-10 09:30", periods=n, freq="1min")
    df = pd.DataFrame({
        "Open":   np.round(opens, 4),
        "High":   np.round(highs, 4),
        "Low":    np.round(lows,  4),
        "Close":  np.round(closes, 4),
        "Volume": np.full(n, volume, dtype=int),
    }, index=idx)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    df["vwap"] = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df


@pytest.fixture
def ohlcv():
    return make_ohlcv()


@pytest.fixture
def ohlcv_rising():
    return make_ohlcv(trend=0.10)


@pytest.fixture
def ohlcv_falling():
    return make_ohlcv(trend=-0.10)
