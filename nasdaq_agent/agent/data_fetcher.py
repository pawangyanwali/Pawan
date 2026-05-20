"""
Data fetcher — Schwab Market Data API (Twelve Data removed).

Historical OHLCV  → Schwab /pricehistory, one call per ticker, parallelised.
Live 1-min prices → Schwab MD quote poller (_live_quotes), updated every 1s.
Rate limit        → 120 req/min; global lock at 1.5 req/s (90 req/min).

Public API is unchanged so no callers need modification:
  fetch_batch_interval(tickers, interval, outputsize, ttl, extended_hours)
  fetch_batch_realtime(tickers, extended_hours)
  fetch_historical(ticker)       — 1-min bars (~10 days) for XGBoost training
  fetch_historical_daily(ticker) — daily bars (2 years) for DailyML
  fetch_batch_daily(tickers)     — batch daily
  resample_ohlcv(df, freq)       — free resampling, no API call
  get_credit_usage()             — stub; no credits with Schwab
"""

import concurrent.futures
import threading
import time
import logging

import pandas as pd

from config import (
    REALTIME_OUTPUTSIZE,
    DAILY_CACHE_TTL,
    TICKER_NAMES,
)

logger = logging.getLogger(__name__)

# ── In-process memory cache ───────────────────────────────────────────────────
# Structure: {ticker: {interval_str: (DataFrame, fetch_timestamp)}}

_interval_cache: dict[str, dict[str, tuple[pd.DataFrame, float]]] = {}
_interval_cache_lock = threading.Lock()
_CACHE_MAX_TICKERS = 600


def _cache_get(ticker: str, interval: str, ttl: float) -> pd.DataFrame | None:
    with _interval_cache_lock:
        entry = _interval_cache.get(ticker, {}).get(interval)
    if entry and ttl > 0 and (time.time() - entry[1]) < ttl:
        return entry[0]
    return None


def _cache_set(ticker: str, interval: str, df: pd.DataFrame) -> None:
    with _interval_cache_lock:
        if ticker not in _interval_cache:
            if len(_interval_cache) >= _CACHE_MAX_TICKERS:
                oldest = min(
                    _interval_cache,
                    key=lambda t: max(
                        (v[1] for v in _interval_cache[t].values()), default=0
                    ),
                )
                del _interval_cache[oldest]
        _interval_cache.setdefault(ticker, {})[interval] = (df, time.time())


# ── SQLite persistent cache ───────────────────────────────────────────────────
# Consulted for training intervals so restarts never duplicate API calls.

_SQLITE_TRAIN_INTERVALS: frozenset[str] = frozenset(
    {"1min", "5min", "15min", "30min", "1h", "1day"}
)
_SQLITE_TTL_MULT = 4


def _sqlite_get(ticker: str, interval: str, ttl: float) -> pd.DataFrame | None:
    if interval not in _SQLITE_TRAIN_INTERVALS or ttl <= 0:
        return None
    try:
        from agent.historical_cache import _get_conn, get_bars
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT MAX(dt) FROM ohlcv_bars WHERE ticker=? AND interval=?",
                (ticker, interval),
            ).fetchone()
        finally:
            conn.close()
        if not row or not row[0]:
            return None
        newest = pd.Timestamp(row[0])
        if (pd.Timestamp.now() - newest).total_seconds() > ttl * _SQLITE_TTL_MULT:
            return None
        df = get_bars(ticker, interval, min_bars=50)
        if df is None or df.empty:
            return None
        return df.rename(columns={
            "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume",
        })
    except Exception:
        return None


def _sqlite_set(ticker: str, interval: str, df: pd.DataFrame) -> None:
    if interval not in _SQLITE_TRAIN_INTERVALS or df is None or df.empty:
        return
    try:
        from agent.historical_cache import _get_conn, _upsert_bars
        conn = _get_conn()
        try:
            _upsert_bars(conn, ticker, interval, df)
        finally:
            conn.close()
    except Exception:
        pass


# ── Core batch fetcher ────────────────────────────────────────────────────────

def fetch_batch_interval(
    tickers:        list,
    interval:       str,
    outputsize:     int,
    ttl:            float = 0.0,
    extended_hours: bool  = False,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple tickers at a given interval via Schwab.

    Parameters
    ----------
    tickers        : List of ticker symbols.
    interval       : Interval string ('1min', '5min', '15min', '1h', '1day', …).
    outputsize     : Number of bars to return per symbol.
    ttl            : Cache TTL in seconds. 0 = always fetch fresh.
    extended_hours : Include pre/after-market bars (intraday intervals only).

    Returns
    -------
    Dict mapping ticker → DataFrame (oldest bar first, title-case OHLCV columns).
    Only successfully fetched tickers are present.
    """
    interval_key = f"{interval}:ext" if extended_hours else interval

    result: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    # 1) In-process memory cache
    for ticker in tickers:
        cached = _cache_get(ticker, interval_key, ttl)
        if cached is not None:
            result[ticker] = cached
        else:
            to_fetch.append(ticker)

    # 2) SQLite persistent cache (training intervals, ttl > 0, non-extended)
    if to_fetch and ttl > 0 and not extended_hours:
        still_miss: list[str] = []
        for ticker in to_fetch:
            df = _sqlite_get(ticker, interval, ttl)
            if df is not None:
                result[ticker] = df
                _cache_set(ticker, interval_key, df)
            else:
                still_miss.append(ticker)
        to_fetch = still_miss

    if not to_fetch:
        return result

    # 3) Schwab API — parallel, rate-limited via fetch_price_history_batch()
    logger.info(
        f"[Schwab] {interval_key}: fetching {len(to_fetch)} tickers "
        f"(+ {len(tickers) - len(to_fetch)} cached)"
    )
    try:
        from agent.broker.schwab_market_data import fetch_price_history_batch
        fetched = fetch_price_history_batch(
            to_fetch, interval=interval,
            outputsize=outputsize, extended_hours=extended_hours,
        )
    except Exception as e:
        logger.warning(f"[Schwab] batch fetch error {interval}: {e}")
        fetched = {}

    for ticker, df in fetched.items():
        result[ticker] = df
        if ttl > 0:
            _cache_set(ticker, interval_key, df)
            if not extended_hours:
                _sqlite_set(ticker, interval, df)

    ok = len([t for t in to_fetch if t in result])
    logger.info(f"[Schwab] {interval_key}: {ok}/{len(to_fetch)} fetched from API")
    return result


# ── Real-time 1-min with live-quote overlay ───────────────────────────────────

def fetch_batch_realtime(
    tickers: list,
    extended_hours: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    1-min OHLCV bars with the latest live price overlaid on the current bar.

    Historical bars come from Schwab /pricehistory (cached 5 min between
    refreshes).  The current bar's Close is updated from the MD quote
    poller (_live_quotes) which is refreshed every 1 second — so the
    scanner always sees the most recent price without a per-scan API call.
    """
    # Fetch 1-min history (cached 5 min; first call may be slower)
    result = fetch_batch_interval(
        tickers, "1min", REALTIME_OUTPUTSIZE,
        ttl=300, extended_hours=extended_hours,
    )

    # Overlay current live price onto the last bar's Close
    try:
        from agent.broker.schwab_streamer import get_live_quote
        for ticker, df in result.items():
            if df.empty:
                continue
            q = get_live_quote(ticker)
            if not q:
                continue
            last = float(q.get("last") or 0)
            if last <= 0:
                continue
            df = df.copy()
            df.iloc[-1, df.columns.get_loc("Close")] = last
            result[ticker] = df
    except Exception:
        pass

    return result


# ── Convenience wrappers ──────────────────────────────────────────────────────

def fetch_historical(ticker: str) -> pd.DataFrame:
    """~10 days of 1-min bars for XGBoost scalp model training (cached 1h)."""
    result = fetch_batch_interval([ticker], "1min", 3900, ttl=3600)
    return result.get(ticker, pd.DataFrame())


def fetch_historical_daily(ticker: str, outputsize: int = 500) -> pd.DataFrame:
    """~2 years of daily bars for DailyMLModel (cached 24h)."""
    result = fetch_batch_interval([ticker], "1day", outputsize, ttl=DAILY_CACHE_TTL)
    return result.get(ticker, pd.DataFrame())


def fetch_batch_daily(tickers: list) -> dict[str, pd.DataFrame]:
    """Batch-fetch daily OHLCV for all tickers (cached 24h)."""
    return fetch_batch_interval(tickers, "1day", 500, ttl=DAILY_CACHE_TTL)


# ── OHLCV resampling (free — no API calls) ────────────────────────────────────

def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample a higher-frequency OHLCV DataFrame to a lower frequency.
    freq: pandas offset alias — '5min', '15min', '30min', '60min', etc.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        resampled = df.resample(freq).agg({
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }).dropna(subset=["Open", "Close"])
        return resampled
    except Exception as e:
        logger.debug(f"resample_ohlcv({freq}) error: {e}")
        return pd.DataFrame()


# ── Stubs for backward compatibility ─────────────────────────────────────────

def get_credit_usage() -> dict:
    """No API credits with Schwab Market Data — always returns zeros."""
    return {"used": 0, "limit": 0, "pct": 0.0}


def fetch_news(ticker: str) -> list:
    return []


def fetch_ticker_info(ticker: str) -> dict:
    return {
        "name":       TICKER_NAMES.get(ticker, ticker),
        "sector":     "Technology",
        "market_cap": 0,
    }
