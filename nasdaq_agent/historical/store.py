"""Database read/write for historical price data."""

import logging
from pathlib import Path

import pandas as pd

from agent.db import get_conn, using_postgres
from historical.schema import ALL_INTERVALS, get_all_ddl, table_name

logger = logging.getLogger(__name__)

HISTORY_DB_PATH = Path.home() / ".nasdaq_agent" / "history.db"


def init_tables() -> None:
    """Create all hist_* tables if they don't exist."""
    use_pg = using_postgres()
    ddl_list = get_all_ddl(use_pg)
    HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_conn(HISTORY_DB_PATH) as conn:
        for ddl in ddl_list:
            try:
                if use_pg:
                    # DDL needs autocommit in PostgreSQL
                    raw = conn._conn  # type: ignore[attr-defined]
                    prev = raw.autocommit
                    raw.autocommit = True
                    raw.execute(ddl.strip())
                    raw.autocommit = prev
                else:
                    conn.execute(ddl.strip())
                    conn.commit()
            except Exception as exc:
                logger.warning("DDL warning: %s", exc)
    logger.info("[HistStore] Tables initialised (%s)", "PostgreSQL" if use_pg else "SQLite")


def upsert_bars(interval: str, ticker: str, df: pd.DataFrame) -> int:
    """
    Bulk-insert OHLCV rows for one ticker/interval.
    Returns number of rows inserted.
    """
    if df.empty:
        return 0

    tbl = table_name(interval)
    use_pg = using_postgres()

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

    if use_pg:
        sql = (
            f"INSERT INTO {tbl} (ticker, ts, open, high, low, close, volume) "
            f"VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (ticker, ts) DO NOTHING"
        )
    else:
        sql = (
            f"INSERT OR IGNORE INTO {tbl} (ticker, ts, open, high, low, close, volume) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?)"
        )

    with get_conn(HISTORY_DB_PATH) as conn:
        conn.executemany(sql, rows)
        conn.commit()

    return len(rows)


def get_last_ts(interval: str, ticker: str) -> int | None:
    """Return the most recent stored epoch-ms for a ticker/interval, or None."""
    tbl = table_name(interval)
    use_pg = using_postgres()
    ph = "%s" if use_pg else "?"
    sql = f"SELECT MAX(ts) FROM {tbl} WHERE ticker = {ph}"
    with get_conn(HISTORY_DB_PATH) as conn:
        row = conn.execute(sql, (ticker,)).fetchone()
    return row[0] if row and row[0] is not None else None


def row_counts() -> dict[str, int]:
    """Return total row count per interval table."""
    counts: dict[str, int] = {}
    for iv in ALL_INTERVALS:
        tbl = table_name(iv)
        try:
            with get_conn(HISTORY_DB_PATH) as conn:
                row = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()
            counts[iv] = row[0] if row else 0
        except Exception:
            counts[iv] = 0
    return counts


def ticker_counts(interval: str) -> dict[str, int]:
    """Return per-ticker bar count for one interval."""
    tbl = table_name(interval)
    try:
        with get_conn(HISTORY_DB_PATH) as conn:
            rows = conn.execute(
                f"SELECT ticker, COUNT(*) FROM {tbl} GROUP BY ticker ORDER BY ticker"
            ).fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}


def read_ticker_bars(interval: str, ticker: str) -> pd.DataFrame:
    """Load all stored bars for one ticker/interval into a DataFrame."""
    tbl = table_name(interval)
    use_pg = using_postgres()
    ph = "%s" if use_pg else "?"
    sql = (
        f"SELECT ts, open, high, low, close, volume FROM {tbl} "
        f"WHERE ticker = {ph} ORDER BY ts"
    )
    with get_conn(HISTORY_DB_PATH) as conn:
        rows = conn.execute(sql, (ticker,)).fetchall()

    if not rows:
        return pd.DataFrame()

    index = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)
    return pd.DataFrame(
        {
            "Open":   [r[1] for r in rows],
            "High":   [r[2] for r in rows],
            "Low":    [r[3] for r in rows],
            "Close":  [r[4] for r in rows],
            "Volume": [r[5] for r in rows],
        },
        index=index,
    )
