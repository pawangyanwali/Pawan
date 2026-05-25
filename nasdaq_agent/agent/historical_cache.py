"""
Historical OHLCV cache — PostgreSQL-backed, grows over time.

Bars are stored by (ticker, interval) and accumulate with each fetch.
Schwab /pricehistory is the data source.

Public API
----------
fetch_and_store(tickers, interval, outputsize, broadcast_fn)
    Fetch latest window from Schwab and merge into PostgreSQL.

get_bars(ticker, interval, min_bars)
    Return full cached DataFrame (all stored bars for this ticker/interval).

cache_stats()
    Summary dict: per-interval bar counts and oldest/newest dates.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

import pandas as pd

from agent.db import get_conn, _get_pool

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# ── Schema ─────────────────────────────────────────────────────────────────────

_OHLCV_DDL = """
CREATE TABLE IF NOT EXISTS ohlcv_bars (
    ticker   TEXT NOT NULL,
    interval TEXT NOT NULL,
    dt       TEXT NOT NULL,
    open     DOUBLE PRECISION,
    high     DOUBLE PRECISION,
    low      DOUBLE PRECISION,
    close    DOUBLE PRECISION,
    volume   DOUBLE PRECISION,
    PRIMARY KEY (ticker, interval, dt)
)
"""
_OHLCV_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_iv "
    "ON ohlcv_bars (ticker, interval, dt DESC)"
)


def init_db() -> None:
    """Create ohlcv_bars table if it doesn't exist."""
    pool = _get_pool()
    raw = pool.getconn()
    try:
        raw.autocommit = True
        with raw.cursor() as cur:
            cur.execute(_OHLCV_DDL.strip())
            cur.execute(_OHLCV_IDX)
    finally:
        pool.putconn(raw)
    logger.info("[HistCache] ohlcv_bars table ready (PostgreSQL)")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _upsert_bars(ticker: str, interval: str, df: pd.DataFrame) -> int:
    """Merge df into ohlcv_bars; return number of rows upserted."""
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

    sql = (
        "INSERT INTO ohlcv_bars (ticker, interval, dt, open, high, low, close, volume) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (ticker, interval, dt) DO NOTHING"
    )
    with _lock:
        with get_conn() as conn:
            conn.executemany(sql, rows)
    return len(rows)


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_and_store(
    tickers:      list[str],
    interval:     str,
    outputsize:   int = 5000,
    end_date:     str | None = None,
    broadcast_fn: Callable | None = None,
) -> dict[str, int]:
    """
    Fetch bars from Schwab and persist to PostgreSQL.
    Returns {ticker: new_bars_inserted}.

    end_date is accepted for API compat but ignored — Schwab does not
    support historical end_date windows.
    """
    from agent.data_fetcher import fetch_batch_interval

    if broadcast_fn:
        broadcast_fn({"phase": "FETCHING", "detail": f"{interval} latest"})

    fetched_all = fetch_batch_interval(tickers, interval, outputsize, ttl=3600)

    inserted: dict[str, int] = {}
    for ticker, df in fetched_all.items():
        n = _upsert_bars(ticker, interval, df)
        inserted[ticker] = n

    total_new = sum(inserted.values())
    logger.info(f"[HistCache] {interval} done — {total_new} new bars stored")
    return inserted


def get_bars(ticker: str, interval: str, min_bars: int = 100) -> pd.DataFrame:
    """
    Return all cached bars for (ticker, interval) as a DataFrame.
    Returns empty DataFrame if fewer than min_bars available.
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT dt, open, high, low, close, volume FROM ohlcv_bars "
            "WHERE ticker = %s AND interval = %s ORDER BY dt ASC",
            (ticker, interval),
        ).fetchall()

    if not rows or len(rows) < min_bars:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.set_index("dt")
    df.index.name = "datetime"
    df.columns = ["open", "high", "low", "close", "volume"]
    return df


def cache_stats() -> dict:
    """Summary of cache contents — used by the dashboard API."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT interval,
                   COUNT(DISTINCT ticker) AS tickers,
                   COUNT(*)               AS total_bars,
                   MIN(dt)                AS oldest,
                   MAX(dt)                AS newest
            FROM ohlcv_bars
            GROUP BY interval
        """).fetchall()

    return {
        r["interval"]: {
            "tickers":    r["tickers"],
            "total_bars": r["total_bars"],
            "oldest":     r["oldest"],
            "newest":     r["newest"],
        }
        for r in rows
    }


def older_window_end_dates(weeks_back: int = 16, step_weeks: int = 8) -> list[str]:
    """Stub — Schwab does not support end_date on /pricehistory."""
    return []
