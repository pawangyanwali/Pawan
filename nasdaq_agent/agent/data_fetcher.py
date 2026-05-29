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

# Per-interval SQLite TTL multipliers.
# Higher-frequency intervals need fresher data; lower-frequency bars are valid
# for much longer.  This lets restarts use SQLite for HTF intervals (5min/1h/1day)
# without hitting the API, so only the 1min interval needs REST calls.
_SQLITE_TTL_MULT = 8   # default / retrain fallback
_SQLITE_TTL_BY_INTERVAL: dict[str, int] = {
    "1min":  4,    # 4 × 300s  =  20 min  — keeps live signals fresh
    "5min":  24,   # 24 × 600s =   4 h    — 5min bars don't change structure
    "15min": 16,   # 16 × 900s =   4 h
    "30min": 12,
    "1h":    24,   # 24 × 3600s = 24 h
    "1day":  30,   # 30 × 86400s = 30 days
}


def _sqlite_get(ticker: str, interval: str, ttl: float) -> pd.DataFrame | None:
    if interval not in _SQLITE_TRAIN_INTERVALS or ttl <= 0:
        return None
    try:
        from agent.historical_cache import get_bars
        df = get_bars(ticker, interval, min_bars=50)
        if df is None or df.empty:
            return None
        # Freshness check: compare newest bar against TTL with per-interval multiplier.
        # historical_cache stores tz-naive UTC datetimes; use utcnow for comparison.
        newest = df.index[-1]
        if hasattr(newest, "tzinfo") and newest.tzinfo is not None:
            newest = newest.tz_convert("UTC").tz_localize(None)
        mult = _SQLITE_TTL_BY_INTERVAL.get(interval, _SQLITE_TTL_MULT)
        if (pd.Timestamp.now("UTC").tz_localize(None) - newest).total_seconds() > ttl * mult:
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
        from agent.historical_cache import _upsert_bars
        _upsert_bars(ticker, interval, df)
    except Exception:
        pass


# ── Core batch fetcher ────────────────────────────────────────────────────────

_CLOSED_SESSION_STALE_TTL = 86_400


def _is_closed_session() -> bool:
    try:
        from agent.market_hours import get_session_info
        return str(get_session_info().get("session", "")).upper() == "CLOSED"
    except Exception:
        return False


def _restore_closed_session_cache(
    tickers: list[str],
    interval: str,
    interval_key: str,
    result: dict[str, pd.DataFrame],
) -> int:
    """Use last-known PostgreSQL bars while markets are closed and REST is down."""
    if not tickers or not _is_closed_session():
        return 0

    restored = 0
    for ticker in tickers:
        if ticker in result:
            continue
        df = _sqlite_get(ticker, interval, _CLOSED_SESSION_STALE_TTL)
        if df is None:
            continue
        result[ticker] = df
        _cache_set(ticker, interval_key, df)
        restored += 1
    return restored


def fetch_batch_interval(
    tickers:        list,
    interval:       str,
    outputsize:     int,
    ttl:            float = 0.0,
    extended_hours: bool  = False,
    background:     bool  = False,
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

    # 2) PostgreSQL persistent cache (training intervals, ttl > 0).
    # Extended-hours data is stored under a distinct key (interval_key = "N:ext")
    # so regular and extended bars never collide in the same cache row.
    if to_fetch and ttl > 0:
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
        from agent.broker.schwab_market_data import fetch_price_history_batch_async
        fetched = fetch_price_history_batch_async(
            to_fetch, interval=interval,
            outputsize=outputsize, extended_hours=extended_hours,
            background=background,
        )
    except Exception:
        try:
            from agent.broker.schwab_market_data import fetch_price_history_batch
            fetched = fetch_price_history_batch(
                to_fetch, interval=interval,
                outputsize=outputsize, extended_hours=extended_hours,
                background=background,
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
    api_ok = ok
    restored = 0
    if ok < len(to_fetch):
        restored = _restore_closed_session_cache(to_fetch, interval, interval_key, result)
        if restored:
            logger.warning(
                "[Schwab] %s: restored %d/%d tickers from PostgreSQL stale cache "
                "during closed session after REST returned partial/empty data",
                interval_key,
                restored,
                len(to_fetch),
            )
            ok += restored
    if restored:
        logger.info(
            f"[Schwab] {interval_key}: {api_ok}/{len(to_fetch)} fetched from API "
            f"(+{restored} stale-cache fallback)"
        )
    else:
        logger.info(f"[Schwab] {interval_key}: {api_ok}/{len(to_fetch)} fetched from API")
    return result


# ── Real-time 1-min with live-quote overlay ───────────────────────────────────

def fetch_batch_realtime(
    tickers: list,
    extended_hours: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    1-min OHLCV bars — streaming-first, REST fallback.

    Tier A (instant, zero API calls):
      When the Schwab WebSocket streamer is connected and has ≥20 bars for a
      ticker, the bars come directly from the in-memory _live_candles deque.
      This eliminates all REST polling for 1-min data in steady state.

    Tier B (REST fallback):
      Tickers not yet in the streamer buffer (startup, reconnect gap, new
      subscriptions) fall back to Schwab /pricehistory via aiohttp batch fetch.
      Results are cached so subsequent calls return instantly.

    Live price overlay (both tiers):
      The current bar's Close is updated from the LEVELONE_EQUITIES quote
      (updated every 250ms by the streamer) so signals always reflect the
      latest traded price.
    """
    result:       dict[str, pd.DataFrame] = {}
    rest_needed:  list[str]               = []

    # ── Tier A: streaming candles ─────────────────────────────────────────────
    try:
        from agent.broker.schwab_streamer import get_live_1m_df, is_streamer_ready, get_streaming_bar_count

        streamer_up = is_streamer_ready()
        if streamer_up:
            _now_et = pd.Timestamp.now(tz="America/New_York")
            _market_open  = _now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
            _market_close = _now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
            _in_session   = _market_open <= _now_et <= _market_close

            for ticker in tickers:
                if get_streaming_bar_count(ticker) >= 20:
                    df = get_live_1m_df(ticker)
                    if df is not None and not df.empty:
                        # Freshness gate: during regular session the newest bar
                        # must be ≤5 min old — stale bars mean the streamer
                        # stopped receiving CHART_EQUITY events for this symbol.
                        if _in_session:
                            newest_bar = df.index[-1]
                            age_s = (_now_et - newest_bar).total_seconds()
                            if age_s > 300:
                                logger.debug(
                                    f"[DataFetcher] {ticker} streaming bar stale "
                                    f"({age_s:.0f}s) — falling back to REST"
                                )
                                rest_needed.append(ticker)
                                continue
                        result[ticker] = df
                        continue
                rest_needed.append(ticker)
        else:
            rest_needed = list(tickers)
    except Exception:
        rest_needed = list(tickers)

    streaming_count = len(result)

    # ── Tier A½: Valkey candle store (cross-container market-data service) ────
    # When the in-process streamer is not running (NASDAQ_MARKET_DATA_ENABLED=0)
    # the market-data container publishes 1-min candles to md:1m:{ticker} lists.
    # This tier reads those candles so the scanner can work without a local
    # Schwab WebSocket connection.
    if rest_needed:
        valkey_hit = []
        try:
            import json as _json
            from agent.valkey_client import _get_client as _vk_get
            _vk = _vk_get()
            if _vk:
                # ── Batch all lrange calls in a single pipeline round-trip ────────
                # N individual lrange calls would cost N × RTT (≥ 1ms each).
                # One pipeline call costs 1 × RTT regardless of N.
                pipe = _vk.pipeline(transaction=False)
                for ticker in rest_needed:
                    pipe.lrange(f"md:1m:{ticker}", 0, -1)
                all_rows_list = pipe.execute()   # list[list[bytes]] — one per ticker

                col_map = {
                    "open": "Open", "high": "High", "low": "Low",
                    "close": "Close", "volume": "Volume",
                    "Open": "Open", "High": "High", "Low": "Low",
                    "Close": "Close", "Volume": "Volume",
                }
                # Stale gate for Tier A½: during regular session (9:30–16:00 ET)
                # reject candles whose newest bar is > 5 min old — this means
                # the market-data container stopped publishing and the REST tier
                # should be used instead.
                try:
                    _now_et_vk  = pd.Timestamp.now(tz="America/New_York")
                    _in_sess_vk = (
                        _now_et_vk.replace(hour=9,  minute=30, second=0, microsecond=0) <=
                        _now_et_vk <=
                        _now_et_vk.replace(hour=16, minute=0,  second=0, microsecond=0)
                    )
                except Exception:
                    _now_et_vk  = None
                    _in_sess_vk = False

                still_rest = []
                for ticker, rows in zip(rest_needed, all_rows_list):
                    if rows and len(rows) >= 20:
                        candles = [_json.loads(r) for r in rows]
                        _df = pd.DataFrame(candles)
                        # Normalise column names to match REST-fetched dataframes
                        _df.rename(columns={c: col_map[c] for c in _df.columns if c in col_map}, inplace=True)
                        # Build a DatetimeIndex from timestamp/time_ms field if present.
                        # WS streamer writes "time_ms" (from _CHART_FIELDS field "7");
                        # REST-sourced rows and some legacy rows use "timestamp".
                        _ts_col = (
                            "time_ms"   if "time_ms"   in _df.columns else
                            "timestamp" if "timestamp" in _df.columns else
                            None
                        )
                        if _ts_col:
                            _df.index = pd.to_datetime(_df[_ts_col], unit="ms", utc=True).dt.tz_convert("America/New_York")
                            _df.drop(columns=[_ts_col], inplace=True, errors="ignore")
                        if not _df.empty and "Close" in _df.columns:
                            # CHART_EQUITY streaming candles omit volume — ensure
                            # the column exists so compute_indicators never KeyErrors.
                            if "Volume" not in _df.columns:
                                _df["Volume"] = 0.0
                            # Freshness gate: during regular session only accept
                            # Valkey candles whose newest bar is ≤ 5 min old.
                            if _in_sess_vk and _now_et_vk is not None:
                                try:
                                    _age_vk = (_now_et_vk - _df.index[-1]).total_seconds()
                                    if _age_vk > 300:
                                        logger.debug(
                                            "[DataFetcher] %s Valkey candles stale "
                                            "(%.0fs) — falling back to REST", ticker, _age_vk
                                        )
                                        still_rest.append(ticker)
                                        continue
                                except Exception:
                                    pass  # index not datetime — accept the data
                            result[ticker] = _df
                            _cache_set(ticker, "1min", _df)
                            valkey_hit.append(ticker)
                            continue
                    still_rest.append(ticker)
                rest_needed = still_rest
                if valkey_hit:
                    logger.debug(f"[DataFetcher] Valkey candles: {len(valkey_hit)} tickers")
        except Exception as _vk_exc:
            logger.debug(f"[DataFetcher] Valkey candle read failed: {_vk_exc}")

    # ── Tier B: async REST fallback ───────────────────────────────────────────
    if rest_needed:
        interval_key = f"1min:ext" if extended_hours else "1min"
        # Check memory cache first
        cache_miss = []
        for ticker in rest_needed:
            cached = _cache_get(ticker, interval_key, 300)
            if cached is not None:
                result[ticker] = cached
            else:
                cache_miss.append(ticker)

        if cache_miss:
            # Try SQLite.  During extended/closed hours we still allow SQLite reads
            # with a longer TTL (4h) — the data is historical so mixing AH bars is
            # acceptable; the live price overlay will correct the last Close anyway.
            _sqlite_ttl = 300 if not extended_hours else 14400   # 5 min regular, 4 h extended
            still_miss = []
            for ticker in cache_miss:
                df = _sqlite_get(ticker, "1min", _sqlite_ttl)
                if df is not None:
                    result[ticker] = df
                    _cache_set(ticker, interval_key, df)
                else:
                    still_miss.append(ticker)

            if still_miss:
                logger.info(
                    f"[Schwab] 1min: fetching {len(still_miss)} tickers "
                    f"(+{len(tickers)-len(still_miss)} stream/cached) "
                    f"[stream:{streaming_count} rest:{len(still_miss)}]"
                )
                try:
                    from agent.broker.schwab_market_data import fetch_price_history_batch_async
                    fetched = fetch_price_history_batch_async(
                        still_miss, interval="1min",
                        outputsize=REALTIME_OUTPUTSIZE,
                        extended_hours=extended_hours,
                    )
                except Exception:
                    from agent.broker.schwab_market_data import fetch_price_history_batch
                    fetched = fetch_price_history_batch(
                        still_miss, interval="1min",
                        outputsize=REALTIME_OUTPUTSIZE,
                        extended_hours=extended_hours,
                    )
                for ticker, df in fetched.items():
                    result[ticker] = df
                    _cache_set(ticker, interval_key, df)
                    if not extended_hours:
                        _sqlite_set(ticker, "1min", df)
                restored = 0
                if len(fetched) < len(still_miss):
                    restored = _restore_closed_session_cache(
                        still_miss, "1min", interval_key, result
                    )
                    if restored:
                        logger.warning(
                            "[Schwab] 1min: restored %d/%d tickers from "
                            "PostgreSQL stale cache during closed session "
                            "after REST returned partial/empty data",
                            restored,
                            len(still_miss),
                        )
                if restored:
                    logger.info(
                        f"[Schwab] 1min: {len(fetched)}/{len(still_miss)} fetched from API "
                        f"(+{restored} stale-cache fallback)"
                    )
                else:
                    logger.info(f"[Schwab] 1min: {len(fetched)}/{len(still_miss)} fetched from API")

    # ── Live price overlay ────────────────────────────────────────────────────
    # Priority 1: in-process _live_quotes (streamer running in this container)
    # Priority 2: Valkey md:prices hash (market-data container in split mode)
    # Without this fallback, scanner container (NASDAQ_MARKET_DATA_ENABLED=0)
    # would serve stale REST Close values and miss the live price update that
    # corrects any trailing zero/incomplete bar from Schwab's API.
    try:
        from agent.broker.schwab_streamer import get_live_quote
        # Bulk-read Valkey prices once per batch call — O(1) HGETALL ≈ 0.5 ms
        _vk_prices: dict = {}
        try:
            from agent.valkey_client import get_all_prices as _vk_all_prices
            _vk_prices = _vk_all_prices()
        except Exception:
            pass
        for ticker, df in result.items():
            if df.empty:
                continue
            q = get_live_quote(ticker)
            if not q:
                # Fallback: read from market-data container's Valkey price hash
                q = _vk_prices.get(ticker, {})
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


def get_last_cached_close(ticker: str) -> float | None:
    """Return the most recently cached Close price for a ticker, ignoring TTL.

    Used as a fallback when live prices are unavailable (after-hours stale-close).
    Checks in-memory cache across all intervals — no API calls, no TTL check.
    """
    for interval in ("1min", "5min", "1day"):
        with _interval_cache_lock:
            entry = _interval_cache.get(ticker, {}).get(interval)
        if entry:
            df, _ = entry
            if not df.empty and "Close" in df.columns:
                return float(df.iloc[-1]["Close"])
    return None


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
    """
    Return recent news for ticker from the context_store DB.
    Falls back to [] when context-intel has not populated any events yet
    (same behaviour as the original stub).
    """
    try:
        from agent.context_store import get_recent_news_for_ticker
        return get_recent_news_for_ticker(ticker, max_items=10)
    except Exception:
        return []


def fetch_ticker_info(ticker: str) -> dict:
    return {
        "name":       TICKER_NAMES.get(ticker, ticker),
        "sector":     "Technology",
        "market_cap": 0,
    }
