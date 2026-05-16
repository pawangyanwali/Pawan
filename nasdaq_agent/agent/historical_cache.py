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

_BASE_URL = "https://api.twelvedata.com"

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "ohlcv_cache.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_lock    = threading.Lock()

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


def _parse_td_values(values: list) -> pd.DataFrame:
    """Convert a Twelve Data 'values' list → OHLCV DataFrame (oldest first)."""
    if not values:
        return pd.DataFrame()
    try:
        df = pd.DataFrame({
            "open":   [float(v["open"])          for v in values],
            "high":   [float(v["high"])          for v in values],
            "low":    [float(v["low"])           for v in values],
            "close":  [float(v["close"])         for v in values],
            "volume": [float(v.get("volume", 0)) for v in values],
        }, index=pd.to_datetime([v["datetime"] for v in values]))
        df.index.name = "datetime"
        return df.sort_index()
    except Exception as exc:
        logger.debug(f"[HistCache] parse error: {exc}")
        return pd.DataFrame()


def _fetch_with_end_date(batch: list[str], interval: str, outputsize: int,
                         end_date: str) -> dict[str, pd.DataFrame]:
    """
    GET /time_series with end_date to fetch an older window of bars.
    Routes through data_fetcher._get() so the rate limiter is respected.
    """
    try:
        from agent.data_fetcher import _get
        data = _get(
            "/time_series",
            {
                "symbol":     ",".join(batch),
                "interval":   interval,
                "outputsize": outputsize,
                "end_date":   end_date,
                "order":      "ASC",
            },
            n_credits=len(batch),
        )
    except Exception as exc:
        logger.warning(f"[HistCache] older-window fetch error (end_date={end_date}): {exc}")
        return {}

    result: dict[str, pd.DataFrame] = {}

    # Single-ticker response has "values" at the top level
    if "values" in data:
        if data.get("status") != "error" and len(batch) == 1:
            df = _parse_td_values(data["values"])
            if not df.empty:
                result[batch[0]] = df
        return result

    # Multi-ticker response is keyed by symbol
    for ticker in batch:
        td = data.get(ticker, {})
        if not isinstance(td, dict) or td.get("status") == "error":
            logger.debug(f"[HistCache] {ticker} not in older-window response: {td.get('message','')}")
            continue
        df = _parse_td_values(td.get("values", []))
        if not df.empty:
            result[ticker] = df

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

    When end_date is None: delegates to fetch_batch_interval() from
    data_fetcher — uses the proven rate-limited path (GET /time_series).

    When end_date is set: uses _fetch_with_end_date() which makes the same
    GET /time_series call but with an end_date param to pull an older window.
    """
    from agent.data_fetcher import fetch_batch_interval

    inserted: dict[str, int] = {}
    label    = f"ending {end_date}" if end_date else "latest"
    n_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE

    if end_date is None:
        # ── Current window: use data_fetcher's rate-limited path.
        # TTL=3600 lets the in-process cache serve this request if the same
        # interval was already fetched recently (e.g. by ml_model retrain),
        # avoiding duplicate API calls during startup.
        if broadcast_fn:
            broadcast_fn({"phase": "FETCHING",
                          "detail": f"{interval} {label}"})
        fetched_all = fetch_batch_interval(tickers, interval, outputsize, ttl=3600)
        conn = _get_conn()
        try:
            for ticker, df in fetched_all.items():
                n = _upsert_bars(conn, ticker, interval, df)
                inserted[ticker] = n
        finally:
            conn.close()
    else:
        # ── Older window: batched, rate-limited via data_fetcher._get() ─────────
        for i in range(0, len(tickers), BATCH_SIZE):
            batch     = tickers[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            logger.info(
                f"[HistCache] {interval} {label} batch {batch_num}/{n_batches}: "
                f"{len(batch)} symbols"
            )
            if broadcast_fn:
                broadcast_fn({"phase": "FETCHING",
                              "detail": f"{interval} {label} — batch {batch_num}/{n_batches}"})

            fetched = _fetch_with_end_date(batch, interval, outputsize, end_date)
            conn = _get_conn()
            try:
                for ticker, df in fetched.items():
                    n = _upsert_bars(conn, ticker, interval, df)
                    inserted[ticker] = inserted.get(ticker, 0) + n
            finally:
                conn.close()

    total_new = sum(inserted.values())
    logger.info(f"[HistCache] {interval} {label} done — {total_new} new bars stored")
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
