"""DDL for historical price tables. One table per interval for fast backtesting queries."""

# All stored intervals. 1h/2h/4h are derived from 30min (no native Schwab frequency).
ALL_INTERVALS = ["1min", "5min", "15min", "30min", "1h", "2h", "4h", "1day"]

# Intervals fetched directly from Schwab — each has its own server-side lookback depth.
DIRECT_INTERVALS = ["1min", "5min", "15min", "30min", "1day"]

# Calendar days per chunk per minute interval.
CHUNK_DAYS: dict[str, int] = {
    "1min":  9,
    "5min":  30,
    "15min": 90,
    "30min": 180,
}

# 1h, 2h, 4h are not native Schwab frequencies — derived by resampling 30min.
RESAMPLE_FROM_30MIN: dict[str, str] = {
    "1h": "60min",
    "2h": "120min",
    "4h": "240min",
}


def table_name(interval: str) -> str:
    return f"hist_{interval.replace('min', 'min').replace('h', 'h').replace('day', 'day')}"


def _pg_ddl(interval: str) -> list[str]:
    tbl = table_name(interval)
    return [
        f"""
        CREATE TABLE IF NOT EXISTS {tbl} (
            ticker  TEXT             NOT NULL,
            ts      BIGINT           NOT NULL,
            open    DOUBLE PRECISION NOT NULL,
            high    DOUBLE PRECISION NOT NULL,
            low     DOUBLE PRECISION NOT NULL,
            close   DOUBLE PRECISION NOT NULL,
            volume  BIGINT           NOT NULL DEFAULT 0,
            PRIMARY KEY (ticker, ts)
        )
        """,
        f"CREATE INDEX IF NOT EXISTS idx_{tbl}_ts ON {tbl}(ts)",
    ]


def get_all_ddl() -> list[str]:
    """Return CREATE TABLE / CREATE INDEX statements for all hist_* tables."""
    stmts: list[str] = []
    for iv in ALL_INTERVALS:
        stmts.extend(_pg_ddl(iv))
    return stmts
