"""
Data fetcher using the Twelve Data REST API.

Grow-377 plan:
  - 377 API credits / minute
  - Unlimited daily credits
  - 1 credit = 1 symbol in any /time_series request
  - Batch up to 20 symbols per request → massive throughput improvement

Core function: fetch_batch_interval(tickers, interval, outputsize, ttl)
  - Serves from in-process cache when data is fresh
  - Groups cache misses into batches of BATCH_SIZE
  - Respects CALL_GAP between API calls
"""

import time
import requests
import pandas as pd
import logging
from config import (
    TWELVE_DATA_API_KEY,
    CALL_GAP,
    BATCH_SIZE,
    REALTIME_OUTPUTSIZE,
    DAILY_CACHE_TTL,
    TICKER_NAMES,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.twelvedata.com"

# Twelve Data interval string aliases
_IV = {
    "1m": "1min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1day",
}

# ── Rate limiter ──────────────────────────────────────────────────────────────

_last_call: float = 0.0


def _throttle() -> None:
    global _last_call
    wait = CALL_GAP - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


def _get(endpoint: str, params: dict, _retry: int = 3) -> dict:
    """Throttled GET with rate-limit retry."""
    _throttle()
    params["apikey"] = TWELVE_DATA_API_KEY
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("code") == 429:
            if _retry > 0:
                logger.warning("Rate limit hit — waiting 62s before retry…")
                time.sleep(62)
                return _get(endpoint, params, _retry=_retry - 1)
            return {}
        return data
    except Exception as e:
        logger.warning(f"Twelve Data request failed: {e}")
        return {}


# ── OHLCV parsing ─────────────────────────────────────────────────────────────

def _parse_values(values: list) -> pd.DataFrame:
    """Convert a Twelve Data 'values' list → OHLCV DataFrame (oldest first)."""
    if not values:
        return pd.DataFrame()
    try:
        df = pd.DataFrame({
            "Open":   [float(v["open"])          for v in values],
            "High":   [float(v["high"])          for v in values],
            "Low":    [float(v["low"])           for v in values],
            "Close":  [float(v["close"])         for v in values],
            "Volume": [float(v.get("volume", 0)) for v in values],
        }, index=pd.to_datetime([v["datetime"] for v in values]))
        return df.sort_index()
    except Exception as e:
        logger.debug(f"_parse_values error: {e}")
        return pd.DataFrame()


def _parse_batch_response(data: dict, batch: list) -> dict[str, pd.DataFrame]:
    """Parse a multi-ticker Twelve Data response into per-ticker DataFrames."""
    result: dict[str, pd.DataFrame] = {}

    # Single-ticker path: response has "values" directly
    if "values" in data:
        if data.get("status") != "error" and len(batch) == 1:
            df = _parse_values(data["values"])
            if not df.empty:
                result[batch[0]] = df
        return result

    # Multi-ticker path: response keyed by symbol
    for ticker in batch:
        td = data.get(ticker, {})
        if not isinstance(td, dict) or td.get("status") == "error":
            logger.debug(f"[{ticker}] {td.get('message', 'not in response')}")
            continue
        df = _parse_values(td.get("values", []))
        if not df.empty:
            result[ticker] = df

    return result


# ── Per-interval in-process cache ─────────────────────────────────────────────
# Structure: {ticker: {interval_str: (DataFrame, fetch_timestamp)}}

_interval_cache: dict[str, dict[str, tuple[pd.DataFrame, float]]] = {}


def _cache_get(ticker: str, interval: str, ttl: float) -> pd.DataFrame | None:
    entry = _interval_cache.get(ticker, {}).get(interval)
    if entry and ttl > 0 and (time.time() - entry[1]) < ttl:
        return entry[0]
    return None


def _cache_set(ticker: str, interval: str, df: pd.DataFrame) -> None:
    _interval_cache.setdefault(ticker, {})[interval] = (df, time.time())


# ── Core batch fetcher ────────────────────────────────────────────────────────

def fetch_batch_interval(
    tickers:        list,
    interval:       str,
    outputsize:     int,
    ttl:            float = 0.0,
    extended_hours: bool  = False,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple tickers at a given interval.

    Parameters
    ----------
    tickers        : List of ticker symbols.
    interval       : Twelve Data interval string ('1min', '5min', '1h', '1day', …).
    outputsize     : Number of bars to return per symbol.
    ttl            : Cache TTL in seconds. 0 = always fetch fresh.
    extended_hours : When True, include pre/after-market bars (Twelve Data
                     ``extended_trading_hours`` parameter).  Only meaningful
                     for intraday intervals (≤ 4h).

    Returns
    -------
    Dict mapping ticker → DataFrame (oldest bar first).
    Only successfully-fetched tickers are present.
    """
    # Use a distinct cache key when extended hours data is requested so
    # regular-session and extended-session caches don't collide.
    interval_key = f"{interval}:ext" if extended_hours else interval

    result: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    # Serve cached tickers
    for ticker in tickers:
        cached = _cache_get(ticker, interval_key, ttl)
        if cached is not None:
            result[ticker] = cached
        else:
            to_fetch.append(ticker)

    if not to_fetch:
        return result  # all served from cache — zero API calls

    n_batches = -(-len(to_fetch) // BATCH_SIZE)
    for i in range(0, len(to_fetch), BATCH_SIZE):
        batch     = to_fetch[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        logger.debug(f"[{interval_key}] batch {batch_num}/{n_batches}: {len(batch)} symbols")

        params: dict = {
            "symbol":     ",".join(batch),
            "interval":   interval,
            "outputsize": outputsize,
            "order":      "ASC",
        }
        if extended_hours:
            params["extended_trading_hours"] = "true"

        data = _get("/time_series", params)

        if not data:
            logger.warning(f"[{interval_key}] batch {batch_num} returned empty response")
            continue

        parsed = _parse_batch_response(data, batch)
        for ticker, df in parsed.items():
            result[ticker] = df
            if ttl > 0:
                _cache_set(ticker, interval_key, df)

    fetched = len([t for t in to_fetch if t in result])
    logger.info(f"[{interval_key}] fetched {fetched}/{len(to_fetch)} new + {len(tickers)-len(to_fetch)} cached")
    return result


# ── Convenience wrappers ──────────────────────────────────────────────────────

def fetch_batch_realtime(tickers: list, extended_hours: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch fresh 1-min bars for all tickers (no caching — always live)."""
    return fetch_batch_interval(tickers, "1min", REALTIME_OUTPUTSIZE, ttl=0,
                                extended_hours=extended_hours)


def fetch_historical(ticker: str) -> pd.DataFrame:
    """Fetch ~6 months of 5-min OHLCV for intraday ML training (single ticker)."""
    # 180 days × 78 bars/day = 14040 bars; API caps at 5000 → ~64 days at 5min
    result = fetch_batch_interval([ticker], "5min", 5000, ttl=0)
    return result.get(ticker, pd.DataFrame())


def fetch_historical_daily(ticker: str, outputsize: int = 500) -> pd.DataFrame:
    """Fetch ~2 years of daily OHLCV for daily ML model (cached 24h)."""
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


# ── News & info ───────────────────────────────────────────────────────────────

def fetch_news(ticker: str) -> list:
    """Twelve Data news endpoint requires a paid plan — returns empty on current plan."""
    return []


def fetch_ticker_info(ticker: str) -> dict:
    """Return company name from local cache (no API call needed)."""
    return {
        "name":       TICKER_NAMES.get(ticker, ticker),
        "sector":     "Technology",
        "market_cap": 0,
    }
