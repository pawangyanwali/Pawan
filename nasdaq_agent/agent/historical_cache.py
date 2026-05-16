"""
Historical OHLCV cache — SQLite-backed, accumulates bars across weekends.

Problem it solves: Twelve Data's /time_series returns at most 5000 bars per
call.  At 5-min resolution that's ~64 trading days.  By persisting bars to
SQLite and fetching an *older* window each weekend (via end_date), the
dataset grows by another ~64 days each week.  After 4 weekends the XGBoost
models train on ~8 months of intraday data instead of 3.

Public API
----------
fetch_and_store(tickers, interval, outputsize, broadcast_fn)
    Fetch latest window from API and merge into SQLite.

fetch_older_window(tickers, interval, end_date, outputsize, broadcast_fn)
    Fetch bars ending at end_date (extends history backwards).

get_bars(ticker, interval, min_bars)
    Return full cached DataFrame (all stored bars for this ticker/interval).

cache_stats()
    Summary dict: per-interval bar counts and oldest/newest dates.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from config import TWELVE_DATA_API_KEY, CALL_GAP, BATCH_SIZE

logger = logging.getLogger(__name__)

_DB_PATH   = Path(__file__).parent.parent / "data" / "ohlcv_cache.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_BASE_URL  = "https://api.twelvedata.com"
_lock      = threading.Lock()

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

def _upsert_bars(conn: sqlite3.Connection, ticker: str, interval: str, df: pd.DataFrame) -> int:
    """Merge df into ohlcv_bars, return number of new rows inserted."""
    if df.empty:
        return 0
    rows = []
    for idx, row in df.iterrows():
        dt_str = str(idx) if not isinstance(idx, str) else idx
        rows.append((ticker, interval, dt_str,
                      float(row.get("open",  row.get("Open",  0)) or 0),
                      float(row.get("high",  row.get("High",  0)) or 0),
                      float(row.get("low",   row.get("Low",   0)) or 0),
                      float(row.get("close", row.get("Close", 0)) or 0),
                      float(row.get("volume",row.get("Volume",0)) or 0)))
    with _lock:
        cur = conn.executemany(
            "INSERT OR IGNORE INTO ohlcv_bars (ticker,interval,dt,open,high,low,close,volume) "
            "VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        return cur.rowcount


def _batch_fetch(batch: list[str], interval: str, outputsize: int,
                 end_date: str | None = None) -> dict[str, pd.DataFrame]:
    """Make one Twelve Data batch request; return {ticker: DataFrame}."""
    payload: dict = {
        "interval":   interval,
        "outputsize": outputsize,
        "symbols":    ",".join(batch),
        "apikey":     TWELVE_DATA_API_KEY,
        "order":      "ASC",
    }
    if end_date:
        payload["end_date"] = end_date

    try:
        resp = requests.post(
            f"{_BASE_URL}/time_series/batch",
            json=payload,
            timeout=45,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning(f"[HistCache] batch fetch error: {exc}")
        return {}

    result: dict[str, pd.DataFrame] = {}
    for sym, val in data.items():
        if not isinstance(val, dict) or "values" not in val:
            continue
        rows = val["values"]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.index.name = "datetime"
        result[sym] = df
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_and_store(
    tickers:      list[str],
    interval:     str,
    outputsize:   int = 5000,
    end_date:     str | None = None,
    broadcast_fn: Callable | None = None,
) -> dict[str, int]:
    """
    Fetch bars from Twelve Data and persist to SQLite.
    Returns {ticker: new_bars_inserted}.
    """
    inserted: dict[str, int] = {}
    conn = _get_conn()
    n_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(tickers), BATCH_SIZE):
        batch      = tickers[i : i + BATCH_SIZE]
        batch_num  = i // BATCH_SIZE + 1
        label      = f"ending {end_date}" if end_date else "latest"
        logger.info(f"[HistCache] {interval} {label} batch {batch_num}/{n_batches}: {batch}")

        if broadcast_fn:
            broadcast_fn({
                "phase": "FETCHING",
                "detail": f"{interval} {label} — batch {batch_num}/{n_batches}",
            })

        fetched = _batch_fetch(batch, interval, outputsize, end_date)
        for ticker, df in fetched.items():
            n = _upsert_bars(conn, ticker, interval, df)
            inserted[ticker] = inserted.get(ticker, 0) + n

        if i + BATCH_SIZE < len(tickers):
            time.sleep(max(CALL_GAP, 1.5))   # respect rate limit

    conn.close()
    total_new = sum(inserted.values())
    logger.info(f"[HistCache] {interval} fetch done — {total_new} new bars stored")
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
                   COUNT(DISTINCT ticker)   AS tickers,
                   SUM(1)                   AS total_bars,
                   MIN(dt)                  AS oldest,
                   MAX(dt)                  AS newest
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
    """
    Generate end_date strings for fetching older windows.
    e.g. weeks_back=16, step=8 → [today-8w, today-16w]
    """
    today = date.today()
    dates: list[str] = []
    for w in range(step_weeks, weeks_back + 1, step_weeks):
        d = today - timedelta(weeks=w)
        # Move to Friday if weekend
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        dates.append(d.strftime("%Y-%m-%d"))
    return dates
