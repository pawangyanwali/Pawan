"""
Data fetcher using the Twelve Data REST API.

Free tier limits:
  - 8 API credits / minute
  - 800 API credits / day  (1 credit = 1 symbol in any request)

Daily OHLCV data is cached in-process for 24 hours to avoid wasting credits.
"""

import time
import requests
import pandas as pd
import logging
from config import (
    TWELVE_DATA_API_KEY,
    INTRADAY_INTERVAL,
    REALTIME_INTERVAL,
    REALTIME_OUTPUTSIZE,
    DATA_PERIOD_DAYS,
    DAILY_CACHE_TTL,
    TICKER_NAMES,
)

logger = logging.getLogger(__name__)

BASE_URL   = "https://api.twelvedata.com"
BATCH_SIZE = 1      # 1 symbol per request — free tier = 8 CREDITS/min, not 8 requests
CALL_GAP   = 8.5    # seconds between requests  (60s / 8 credits = 7.5s, use 8.5s for safety)

_IV = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "1d": "1day"}

_last_call: float = 0.0


def _throttle() -> None:
    global _last_call
    wait = CALL_GAP - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


def _get(endpoint: str, params: dict, _retry: int = 3) -> dict:
    """GET request with throttling, rate-limit retry, and basic error handling."""
    _throttle()
    params["apikey"] = TWELVE_DATA_API_KEY
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("code") == 429:
            if _retry > 0:
                logger.warning("Rate limit hit — waiting 62 seconds before retry…")
                time.sleep(62)
                return _get(endpoint, params, _retry=_retry - 1)
            return {}
        return data
    except Exception as e:
        logger.warning(f"Twelve Data request failed: {e}")
        return {}


# ── Parsing helpers ───────────────────────────────────────────────────────────

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
        return df.sort_index()      # ascending — oldest bar first
    except Exception as e:
        logger.debug(f"_parse_values error: {e}")
        return pd.DataFrame()


def _bars_needed(days: int, interval: str) -> int:
    bars_per_day = {"1min": 390, "5min": 78, "15min": 26, "30min": 13, "1h": 6, "1day": 1}
    td_iv = _IV.get(interval, "1min")
    return min(days * bars_per_day.get(td_iv, 390), 5000)


# ── Intraday ML training data ─────────────────────────────────────────────────

def fetch_historical(ticker: str) -> pd.DataFrame:
    """Fetch multi-day intraday OHLCV for intraday ML training."""
    td_iv = _IV.get(INTRADAY_INTERVAL, "5min")
    outputsize = _bars_needed(DATA_PERIOD_DAYS, INTRADAY_INTERVAL)
    data = _get("/time_series", {
        "symbol":     ticker,
        "interval":   td_iv,
        "outputsize": outputsize,
        "order":      "ASC",
    })
    if data.get("status") == "error":
        logger.warning(f"[{ticker}] Twelve Data error: {data.get('message')}")
        return pd.DataFrame()
    return _parse_values(data.get("values", []))


# ── Daily OHLCV (for macro trend + daily ML model) ────────────────────────────

# Cache: ticker → (DataFrame, fetch_timestamp)
_daily_cache: dict[str, tuple[pd.DataFrame, float]] = {}


def fetch_historical_daily(ticker: str, outputsize: int = 500) -> pd.DataFrame:
    """
    Fetch ~2 years of daily OHLCV (500 bars) for daily ML training and macro trend.
    Results are cached for DAILY_CACHE_TTL seconds.
    """
    now = time.time()
    if ticker in _daily_cache:
        df, ts = _daily_cache[ticker]
        if now - ts < DAILY_CACHE_TTL:
            return df

    data = _get("/time_series", {
        "symbol":     ticker,
        "interval":   "1day",
        "outputsize": outputsize,
        "order":      "ASC",
    })
    if data.get("status") == "error":
        logger.warning(f"[{ticker}] daily fetch error: {data.get('message')}")
        df = pd.DataFrame()
    else:
        df = _parse_values(data.get("values", []))

    _daily_cache[ticker] = (df, now)
    return df


def fetch_batch_daily(tickers: list) -> dict[str, pd.DataFrame]:
    """
    Fetch daily OHLCV for all tickers one by one (throttled).
    Returns dict ticker → DataFrame.  Called once at startup and retried after 24h.
    """
    result: dict[str, pd.DataFrame] = {}
    n = len(tickers)
    for i, ticker in enumerate(tickers):
        logger.info(f"Daily data [{i+1}/{n}]: {ticker}")
        df = fetch_historical_daily(ticker)
        if not df.empty:
            result[ticker] = df
    logger.info(f"Daily batch complete: {len(result)}/{n} tickers")
    return result


# ── Realtime scan data (1-min bars) ──────────────────────────────────────────

def fetch_realtime(ticker: str) -> pd.DataFrame:
    """Fetch latest REALTIME_OUTPUTSIZE 1-min bars for a single ticker."""
    td_iv = _IV.get(REALTIME_INTERVAL, "1min")
    data = _get("/time_series", {
        "symbol":     ticker,
        "interval":   td_iv,
        "outputsize": REALTIME_OUTPUTSIZE,
        "order":      "ASC",
    })
    if data.get("status") == "error":
        logger.warning(f"[{ticker}] Twelve Data error: {data.get('message')}")
        return pd.DataFrame()
    return _parse_values(data.get("values", []))


def fetch_batch_realtime(tickers: list) -> dict[str, pd.DataFrame]:
    """
    Fetch 1-min intraday for all tickers using Twelve Data batch requests.
    Each request fetches up to BATCH_SIZE symbols; each symbol costs 1 credit.
    """
    result: dict[str, pd.DataFrame] = {}
    td_iv     = _IV.get(REALTIME_INTERVAL, "1min")
    n_batches = -(-len(tickers) // BATCH_SIZE)

    for i in range(0, len(tickers), BATCH_SIZE):
        batch    = tickers[i: i + BATCH_SIZE]
        symbols  = ",".join(batch)
        batch_num = i // BATCH_SIZE + 1
        logger.info(f"Fetching batch {batch_num}/{n_batches}: {batch}")

        data = _get("/time_series", {
            "symbol":     symbols,
            "interval":   td_iv,
            "outputsize": REALTIME_OUTPUTSIZE,
            "order":      "ASC",
        })

        if not data:
            logger.warning(f"Batch {batch_num} returned empty response")
            continue

        if "values" in data:
            if data.get("status") == "error":
                logger.debug(f"[{batch[0]}] {data.get('message', 'error')}")
            else:
                df = _parse_values(data["values"])
                if not df.empty:
                    result[batch[0]] = df
        else:
            for ticker in batch:
                ticker_data = data.get(ticker, {})
                if ticker_data.get("status") == "error":
                    logger.debug(f"[{ticker}] {ticker_data.get('message', 'error')}")
                    continue
                df = _parse_values(ticker_data.get("values", []))
                if not df.empty:
                    result[ticker] = df

    logger.info(f"Batch fetch complete: {len(result)}/{len(tickers)} tickers OK")
    return result


# ── OHLCV resampling (free — no API calls) ────────────────────────────────────

def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample a higher-frequency OHLCV DataFrame to a lower frequency.

    Parameters
    ----------
    df   : OHLCV DataFrame with a DatetimeIndex (e.g. 1-min bars).
    freq : pandas offset alias — '5min', '15min', '30min', '60min', '1h', etc.

    Returns
    -------
    Resampled DataFrame, oldest bar first, NaN rows dropped.
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
    """Twelve Data news endpoint requires a paid plan — returns empty on free tier."""
    return []


def fetch_ticker_info(ticker: str) -> dict:
    """Return company name from local cache (no API call needed)."""
    return {
        "name":       TICKER_NAMES.get(ticker, ticker),
        "sector":     "Technology",
        "market_cap": 0,
    }
