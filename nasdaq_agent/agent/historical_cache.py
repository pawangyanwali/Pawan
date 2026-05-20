"""
Historical OHLCV cache — SQLite-backed, grows over time.

Bars are stored by (ticker, interval) and accumulate with each fetch.
Schwab /pricehistory is the data source (replaces Twelve Data).

Public API
----------
fetch_and_store(tickers, interval, outputsize, broadcast_fn)
    Fetch latest window from Schwab and merge into SQLite.

get_bars(ticker, interval, min_bars)
    Return full cached DataFrame (all stored bars for this ticker/interval).

cache_stats()
    Summary dict: per-interval bar counts and oldest/newest dates.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "ohlcv_cache.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()


# ── Schema ────────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_bars (
            ticker   TEXT NOT NULL,
            interval TEXT NOT NULL,
            dt       TEXT NOT NULL,
            open     REAL,
            high     REAL,
            low      REAL,
            close    REAL,
            volume   REAL,
            PRIMARY KEY (ticker, interval, dt)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_iv
        ON ohlcv_bars (ticker, interval, dt DESC)
    """)
    conn.commit()
    return conn


# ── Internal helpers ──────────────────────────────────────────────────────────

def _upsert_bars(
    conn: sqlite3.Connection, ticker: str, interval: str, df: pd.DataFrame
) -> int:
    """Merge df into ohlcv_bars; return number of new rows inserted."""
    if df.empty:
        return 0
    rows = []
    for idx, row in df.iterrows():
        dt_str = str(idx) if not isinstance(idx, str) else idx
        rows.append((
            ticker, interval, dt_str,
            float(row.get("open",   row.get("Open",   0)) or 0),
            float(row.get("high",   row.get("High",   0)) or 0),
            float(row.get("low",    row.get("Low",    0)) or 0),
            float(row.get("close",  row.get("Close",  0)) or 0),
            float(row.get("volume", row.get("Volume", 0)) or 0),
        ))
    with _lock:
        cur = conn.executemany(
            "INSERT OR IGNORE INTO ohlcv_bars "
            "(ticker,interval,dt,open,high,low,close,volume) "
            "VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        return cur.rowcount


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_and_store(
    tickers:      list[str],
    interval:     str,
    outputsize:   int = 5000,
    end_date:     str | None = None,
    broadcast_fn: Callable | None = None,
) -> dict[str, int]:
    """
    Fetch bars from Schwab and persist to SQLite.
    Returns {ticker: new_bars_inserted}.

    end_date is accepted for API compat but ignored — Schwab does not
    support historical end_date windows.  Only the latest window is fetched.
    """
    from agent.data_fetcher import fetch_batch_interval

    if broadcast_fn:
        broadcast_fn({"phase": "FETCHING", "detail": f"{interval} latest"})

    fetched_all = fetch_batch_interval(tickers, interval, outputsize, ttl=3600)

    inserted: dict[str, int] = {}
    conn = _get_conn()
    try:
        for ticker, df in fetched_all.items():
            n = _upsert_bars(conn, ticker, interval, df)
            inserted[ticker] = n
    finally:
        conn.close()

    total_new = sum(inserted.values())
    logger.info(f"[HistCache] {interval} done — {total_new} new bars stored")
    return inserted


def get_bars(ticker: str, interval: str, min_bars: int = 100) -> pd.DataFrame:
    """
    Return all cached bars for (ticker, interval) as a DataFrame.
    Returns empty DataFrame if fewer than min_bars available.
    """
    conn = _get_conn()
    try:
        df = pd.read_sql_query(
            "SELECT dt,open,high,low,close,volume FROM ohlcv_bars "
            "WHERE ticker=? AND interval=? ORDER BY dt ASC",
            conn,
            params=(ticker, interval),
        )
    finally:
        conn.close()

    if df.empty or len(df) < min_bars:
        return pd.DataFrame()

    df["dt"] = pd.to_datetime(df["dt"])
    df = df.set_index("dt")
    df.index.name = "datetime"
    df.columns = ["open", "high", "low", "close", "volume"]
    return df


def cache_stats() -> dict:
    """Summary of cache contents — used by the dashboard API."""
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT interval,
                   COUNT(DISTINCT ticker) AS tickers,
                   SUM(1)                 AS total_bars,
                   MIN(dt)                AS oldest,
                   MAX(dt)                AS newest
            FROM ohlcv_bars
            GROUP BY interval
        """).fetchall()
    finally:
        conn.close()

    return {
        r[0]: {
            "tickers":    r[1],
            "total_bars": r[2],
            "oldest":     r[3],
            "newest":     r[4],
        }
        for r in rows
    }


def older_window_end_dates(weeks_back: int = 16, step_weeks: int = 8) -> list[str]:
    """Stub — Schwab does not support end_date on /pricehistory."""
    return []
