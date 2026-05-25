"""Database read/write for historical price data (PostgreSQL only)."""

import logging

import pandas as pd

from agent.db import get_conn, _get_pool
from historical.schema import ALL_INTERVALS, get_all_ddl, table_name

logger = logging.getLogger(__name__)


def init_tables() -> None:
    """Create all hist_* tables in PostgreSQL if they don't exist."""
    ddl_list = get_all_ddl()
    pool = _get_pool()
    raw = pool.getconn()
    try:
        raw.autocommit = True
        with raw.cursor() as cur:
            for ddl in ddl_list:
                try:
                    cur.execute(ddl.strip())
                except Exception as exc:
                    logger.warning("DDL warning: %s", exc)
    finally:
        pool.putconn(raw)
    logger.info("[HistStore] Tables initialised (PostgreSQL)")


def upsert_bars(interval: str, ticker: str, df: pd.DataFrame) -> int:
    """
    Bulk-insert OHLCV rows for one ticker/interval.
    Returns number of rows inserted.
    """
    if df.empty:
        return 0

    tbl = table_name(interval)
    rows = [
        (
            ticker,
            int(idx.timestamp() * 1000),
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            int(row.get("Volume", 0)),
        )
        for idx, row in df.iterrows()
    ]

    sql = (
        f"INSERT INTO {tbl} (ticker, ts, open, high, low, close, volume) "
        f"VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (ticker, ts) DO NOTHING"
    )

    with get_conn() as conn:
        conn.executemany(sql, rows)

    return len(rows)


def get_last_ts(interval: str, ticker: str) -> int | None:
    """Return the most recent stored epoch-ms for a ticker/interval, or None."""
    tbl = table_name(interval)
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT MAX(ts) AS max_ts FROM {tbl} WHERE ticker = %s", (ticker,)
        ).fetchone()
    return row["max_ts"] if row and row["max_ts"] is not None else None


def row_counts() -> dict[str, int]:
    """Return total row count per interval table."""
    counts: dict[str, int] = {}
    for iv in ALL_INTERVALS:
        tbl = table_name(iv)
        try:
            with get_conn() as conn:
                row = conn.execute(f"SELECT COUNT(*) AS cnt FROM {tbl}").fetchone()
            counts[iv] = row["cnt"] if row else 0
        except Exception:
            counts[iv] = 0
    return counts


def ticker_counts(interval: str) -> dict[str, int]:
    """Return per-ticker bar count for one interval."""
    tbl = table_name(interval)
    try:
        with get_conn() as conn:
            rows = conn.execute(
                f"SELECT ticker, COUNT(*) AS cnt FROM {tbl} GROUP BY ticker ORDER BY ticker"
            ).fetchall()
        return {r["ticker"]: r["cnt"] for r in rows}
    except Exception:
        return {}


def read_ticker_bars(interval: str, ticker: str) -> pd.DataFrame:
    """Load all stored bars for one ticker/interval into a DataFrame."""
    tbl = table_name(interval)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT ts, open, high, low, close, volume FROM {tbl} "
            f"WHERE ticker = %s ORDER BY ts",
            (ticker,),
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    index = pd.to_datetime([r["ts"] for r in rows], unit="ms", utc=True)
    return pd.DataFrame(
        {
            "Open":   [r["open"]   for r in rows],
            "High":   [r["high"]   for r in rows],
            "Low":    [r["low"]    for r in rows],
            "Close":  [r["close"]  for r in rows],
            "Volume": [r["volume"] for r in rows],
        },
        index=index,
    )
