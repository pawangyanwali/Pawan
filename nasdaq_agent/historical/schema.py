"""DDL for historical price tables. One table per interval for fast backtesting queries."""

# All intervals stored. 1h and 4h are derived by resampling 30min data.
ALL_INTERVALS = ["1min", "5min", "15min", "30min", "1h", "4h", "1day"]

# Intervals fetched directly from Schwab API
FETCH_INTERVALS = ["1min", "1day"]

# Intervals derived by resampling 1min (pandas resample rule → interval name)
RESAMPLE_FROM_1MIN: dict[str, str] = {
    "5min":  "5min",
    "15min": "15min",
    "30min": "30min",
    "1h":    "60min",
    "4h":    "240min",
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


def _sq_ddl(interval: str) -> list[str]:
    tbl = table_name(interval)
    return [
        f"""
        CREATE TABLE IF NOT EXISTS {tbl} (
            ticker  TEXT    NOT NULL,
            ts      INTEGER NOT NULL,
            open    REAL    NOT NULL,
            high    REAL    NOT NULL,
            low     REAL    NOT NULL,
            close   REAL    NOT NULL,
            volume  INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (ticker, ts)
        )
        """,
        f"CREATE INDEX IF NOT EXISTS idx_{tbl}_ts ON {tbl}(ts)",
    ]


def get_all_ddl(using_pg: bool) -> list[str]:
    stmts: list[str] = []
    for iv in ALL_INTERVALS:
        stmts.extend(_pg_ddl(iv) if using_pg else _sq_ddl(iv))
    return stmts
