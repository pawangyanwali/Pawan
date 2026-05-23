"""
Schwab Market Data API.

Endpoints used:
  /pricehistory              — OHLCV bars (primary historical data source)
  /quotes                    — real-time last price for a list of symbols
  /movers/{index}            — top gainers/losers for NASDAQ / S&P 500
  /chains                    — option chain (IV, OI, Greeks)
  /markets                   — market session hours (open/closed check)
  /instruments               — symbol search and fundamentals

Schwab /pricehistory constraint:
  frequencyType=minute is ONLY valid with periodType=day (max period=10).
  All sub-daily intervals are therefore capped at 10 trading days.
  Valid minute frequencies: 1, 5, 10, 15, 30 — NO 60.
  1h and 4h are derived by fetching 30min and resampling up.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from datetime import date

import aiohttp
import pandas as pd
import requests

from agent.broker.schwab_auth import get_md_token_status

logger = logging.getLogger(__name__)

MARKETDATA_BASE = "https://api.schwabapi.com/marketdata/v1"

# Twelve Data interval → (frequencyType, frequency, periodType, period)
# Valid Schwab minute frequencies: 1, 5, 10, 15, 30 only.
_IV_MAP = {
    "1min":  ("minute",  1,  "day",  10),   # 10 days ≈ 3,900 bars
    "5min":  ("minute",  5,  "day",  10),   # 10 days ≈   780 bars
    "15min": ("minute", 15,  "day",  10),   # 10 days ≈   260 bars
    "30min": ("minute", 30,  "day",  10),   # 10 days ≈   130 bars
    "1day":  ("daily",   1,  "year",  2),   # 2 years  ≈   504 bars
}
# 1h and 4h have no native Schwab frequency — fetch 30min and resample
_RESAMPLE_MAP = {
    "1h": ("30min", "60min"),
    "4h": ("30min", "240min"),
}


def _is_authorised() -> bool:
    """True only when the dedicated Market Data app has its own valid (non-expired) token.
    Returns False if the token has expired so callers skip the request entirely;
    the next _get() call will refresh and retry automatically."""
    try:
        from agent.broker.schwab_auth import _market_data
        if not _market_data.get_access_token():
            return False
        status = _market_data.get_status()
        return status.get("access_token_ttl_s", 0) > 0
    except Exception:
        return False


def _auth_headers() -> dict | None:
    """Return headers using only the MD app token. Returns None if not authorised."""
    try:
        from agent.broker.schwab_auth import _market_data
        token = _market_data.get_access_token()
    except Exception:
        token = None
    if not token:
        return None
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _get(path: str, params: dict, timeout: "int | tuple" = 20) -> dict | list:
    """Authenticated GET to the Schwab Market Data API."""
    # Honour 429 back-off: skip this cycle rather than piling on blocked requests.
    backoff_rem = _backoff_until - time.time()
    if backoff_rem > 5.0:
        return {}   # too long to wait inline; caller retries next cycle
    if backoff_rem > 0:
        time.sleep(backoff_rem)

    headers = _auth_headers()
    if not headers:
        return {}
    try:
        r = requests.get(f"{MARKETDATA_BASE}{path}", headers=headers,
                         params=params, timeout=timeout)
        if r.status_code == 429:
            _on_429()
            return {}
        if r.status_code in (401, 403):
            # Detect CDN/IP-level block — two signals:
            # (a) within 30s of a recent 429 window (Akamai rate-limit), or
            # (b) the local token still has a valid TTL (token is fine, IP is blocked).
            # In both cases: apply back-off and skip; do NOT attempt a token refresh
            # since the token endpoint will also return 403 when the IP is blocked.
            is_cdn_block = (_backoff_until - time.time() > -30)
            if not is_cdn_block and r.status_code == 403:
                try:
                    from agent.broker.schwab_auth import _market_data as _md_cdn
                    ttl = _md_cdn.get_status().get("access_token_ttl_s", 0)
                    if ttl > 60:
                        is_cdn_block = True
                except Exception:
                    pass
            if is_cdn_block:
                _on_429()
                logger.warning(
                    f"[Schwab MD] {r.status_code} on {path} — CDN/IP block detected, "
                    f"backing off (token is still valid)"
                )
                return {}
            # Genuine auth failure (expired/revoked token) — refresh and retry once
            logger.info(f"[Schwab MD] {r.status_code} on {path} — refreshing token…")
            if _try_refresh_md_token():
                headers = _auth_headers()
                if headers:
                    r = requests.get(f"{MARKETDATA_BASE}{path}", headers=headers,
                                     params=params, timeout=timeout)
                    if r.status_code == 429:
                        _on_429()
                        return {}
                    r.raise_for_status()
                    _on_success()
                    return r.json()
            logger.warning("[Schwab MD] Auth refresh failed — re-authenticate at /schwab/auth/md")
            return {}
        r.raise_for_status()
        _on_success()
        return r.json()
    except Exception as e:
        logger.warning(f"[Schwab MD] {path} failed: {e}")
        return {}


# ── Price history ─────────────────────────────────────────────────────────────

# ── 429 adaptive back-off (no artificial pre-throttle) ────────────────────────
# Schwab Market Data Production has NO documented REST rate limit.
# We fire all requests concurrently (zero artificial delay for live scans).
# If Schwab ever returns 429 we back off and retry with exponential delay.
#
# Background retrain tasks use a 1 req/s cap so they never starve live scans.

_rate_lock    = threading.Lock()
_backoff_until: float = 0.0        # epoch time when 429 back-off expires
_backoff_last:  float = 0.0        # duration of the most recent back-off window
_BACKOFF_BASE   = 2.0              # seconds for first 429 back-off
_BACKOFF_MAX    = 30.0             # cap at 30s

# Background-caller throttle: retrain tasks capped at 1 req/s.
_bg_lock = threading.Lock()
_bg_last = 0.0
_BG_GAP  = 1.0   # 1 s between background calls


def _on_429() -> None:
    """Called when Schwab returns HTTP 429. Sets an exponentially growing back-off.

    Concurrent requests often all 429 within the same millisecond.
    Guard: only compound when NOT already inside an active back-off window.
    _backoff_last tracks the previous duration so doubling works correctly
    across retry cycles (remaining * 2 would always collapse to _BACKOFF_BASE).
    """
    global _backoff_until, _backoff_last
    with _rate_lock:
        remaining = _backoff_until - time.time()
        if remaining > 1.0:
            return  # concurrent 429 — already backed off, don't compound
        new_backoff = min(max(_backoff_last * 2.0, _BACKOFF_BASE), _BACKOFF_MAX)
        _backoff_last = new_backoff
        _backoff_until = time.time() + new_backoff
    logger.warning(f"[Schwab MD] 429 → back-off {new_backoff:.1f}s")


def _on_success() -> None:
    """Clear back-off window and reset duration after a clean response."""
    global _backoff_until, _backoff_last
    if _backoff_until > time.time():
        with _rate_lock:
            _backoff_until = 0.0
    _backoff_last = 0.0


# Prevent simultaneous refresh storms when many concurrent calls all get 401/403
_refresh_lock = threading.Lock()
_refresh_last: float = 0.0

def _try_refresh_md_token() -> bool:
    """
    Refresh the Market Data access token.  Coalesces concurrent refresh attempts:
    only one thread calls Schwab at a time; others wait behind _refresh_lock
    and hit the 30-second dedup check instead of hammering the token endpoint.

    _refresh_last is stamped BEFORE the attempt (not only on success) so that
    even a failed refresh prevents re-hammering the token endpoint for 30s.

    If the refresh itself fails (e.g. the token endpoint is also blocked by
    Akamai), _on_429() is called so that _get() skips subsequent API calls
    for the same back-off window instead of looping every second.
    """
    global _refresh_last
    with _refresh_lock:
        if time.time() - _refresh_last < 30:
            return bool(_auth_headers())
        _refresh_last = time.time()   # stamp before attempt — prevents storm on failure
        try:
            from agent.broker.schwab_auth import _market_data as _md_app
            ok = _md_app.refresh()
            if not ok:
                # Token endpoint blocked (CDN/IP block) — back off API calls too
                _on_429()
            return ok
        except Exception as e:
            logger.warning(f"[Schwab MD] Token refresh error: {e}")
            _on_429()
            return False


# ── Async HTTP layer (aiohttp, concurrent, no pre-throttle) ──────────────────
# A single background event loop handles all async HTTP so sync callers can
# submit coroutines via asyncio.run_coroutine_threadsafe() and block for results.

_aio_loop:   asyncio.AbstractEventLoop | None = None
_aio_thread: "threading.Thread | None"        = None
_aio_lock    = threading.Lock()


def _get_aio_loop() -> asyncio.AbstractEventLoop:
    """Return (and lazily start) the shared async event loop thread."""
    global _aio_loop, _aio_thread
    with _aio_lock:
        if _aio_loop is None or _aio_loop.is_closed():
            _aio_loop = asyncio.new_event_loop()
            _aio_thread = threading.Thread(
                target=_aio_loop.run_forever,
                daemon=True,
                name="SchwabAioHTTP",
            )
            _aio_thread.start()
        return _aio_loop


def _run_async(coro, timeout: float = 60.0):
    """Block the calling thread until `coro` completes on the async loop."""
    fut = asyncio.run_coroutine_threadsafe(coro, _get_aio_loop())
    return fut.result(timeout=timeout)


# Async background-task rate state (plain globals — asyncio is single-threaded).
_aio_bg_last: float = 0.0

# ── Async token-bucket rate limiter ──────────────────────────────────────────
# Enforces a global minimum gap between /pricehistory requests so we never
# exceed ~15 req/s (67ms gap) under burst conditions.  This is the proactive
# throttle that was described in the docstring but never implemented.
#
# Using a Lock + timestamp rather than a semaphore so the rate applies across
# ALL concurrent _one() tasks, not just per-slot.  This prevents the startup
# burst where 250 tasks queue up and fire as fast as the semaphore releases.
_aio_rate_lock: "asyncio.Lock | None" = None   # created lazily (loop-bound)
_aio_rate_last: float = 0.0
_AIO_RATE_GAP: float  = 0.067   # 67ms → ≤15 req/s  (leaves headroom vs 120/min cap)


def _get_aio_rate_lock() -> "asyncio.Lock":
    global _aio_rate_lock
    if _aio_rate_lock is None:
        _aio_rate_lock = asyncio.Lock()
    return _aio_rate_lock


async def _aio_rate_wait() -> None:
    """
    Token-bucket gate for async /pricehistory calls.
    Serialises the rate-limit check so only one task updates _aio_rate_last at
    a time; others queue behind the lock and naturally space out their starts.
    """
    global _aio_rate_last
    async with _get_aio_rate_lock():
        now = time.monotonic()
        next_ok = _aio_rate_last + _AIO_RATE_GAP
        if now < next_ok:
            await asyncio.sleep(next_ok - now)
        _aio_rate_last = time.monotonic()


async def _aio_maybe_backoff() -> None:
    """If a 429 back-off window is active, sleep until it expires."""
    remaining = _backoff_until - time.time()
    if remaining > 0:
        await asyncio.sleep(remaining)


async def _aio_bg_wait() -> None:
    """Throttle background callers to 1 req/s so they don't starve live scans."""
    global _aio_bg_last
    now = time.time()
    fire_at = max(now, _aio_bg_last + _BG_GAP)
    _aio_bg_last = fire_at
    delay = fire_at - now
    if delay > 0:
        await asyncio.sleep(delay)


def _rate_wait(background: bool = False) -> None:
    """Sync path: background throttle + 429 back-off wait."""
    if background:
        with _bg_lock:
            gap = time.time() - _bg_last
            if gap < _BG_GAP:
                time.sleep(_BG_GAP - gap)
            _bg_last = time.time()
    # Honour any active 429 back-off
    remaining = _backoff_until - time.time()
    if remaining > 0:
        time.sleep(remaining)


def fetch_price_history(
    ticker:         str,
    interval:       str,
    outputsize:     int  = 300,
    extended_hours: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV for one ticker from Schwab.

    Returns a DataFrame with columns [Open, High, Low, Close, Volume]
    indexed by UTC datetime — identical to data_fetcher output.
    Returns empty DataFrame if not authorised or on any error.

    1h and 4h are fetched as 30min bars and resampled up.
    """
    if not _is_authorised():
        return pd.DataFrame()

    # Handle intervals that require resampling (1h, 4h)
    if interval in _RESAMPLE_MAP:
        src_interval, resample_freq = _RESAMPLE_MAP[interval]
        df = fetch_price_history(ticker, src_interval, outputsize * 2, extended_hours)
        if df.empty:
            return df
        resampled = df.resample(resample_freq).agg({
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }).dropna(subset=["Open", "Close"])
        if outputsize and len(resampled) > outputsize:
            resampled = resampled.iloc[-outputsize:]
        logger.debug(f"[Schwab MD] {ticker}/{interval} (resampled from {src_interval}): {len(resampled)} bars")
        return resampled

    mapping = _IV_MAP.get(interval)
    if not mapping:
        return pd.DataFrame()

    freq_type, freq, period_type, period = mapping
    data = _get("/pricehistory", {
        "symbol":                ticker,
        "periodType":            period_type,
        "period":                period,
        "frequencyType":         freq_type,
        "frequency":             freq,
        "needExtendedHoursData": "true" if extended_hours else "false",
    })
    candles = data.get("candles", []) if isinstance(data, dict) else []
    if not candles:
        return pd.DataFrame()

    try:
        df = pd.DataFrame({
            "Open":   [c["open"]              for c in candles],
            "High":   [c["high"]              for c in candles],
            "Low":    [c["low"]               for c in candles],
            "Close":  [c["close"]             for c in candles],
            "Volume": [float(c.get("volume", 0)) for c in candles],
        }, index=pd.to_datetime([c["datetime"] for c in candles], unit="ms", utc=True))
        df = df.sort_index()
        if outputsize and len(df) > outputsize:
            df = df.iloc[-outputsize:]
        logger.debug(f"[Schwab MD] {ticker}/{interval}: {len(df)} bars")
        return df
    except Exception as e:
        logger.debug(f"[Schwab MD] parse error {ticker}: {e}")
        return pd.DataFrame()


def fetch_price_history_range(
    ticker:         str,
    interval:       str,
    start_ms:       int,
    end_ms:         int,
    extended_hours: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV for an explicit date range (epoch ms).
    Used exclusively by the historical backfill service.

    interval: "1min" | "5min" | "15min" | "30min" | "1day"
    Schwab supports arbitrary start/end ranges; periodType=day is required
    for minute frequencies even when startDate/endDate are supplied.
    """
    if not _is_authorised():
        return pd.DataFrame()

    _RANGE_MAP: dict[str, tuple] = {
        "1min":  ("minute",  1, "day"),
        "5min":  ("minute",  5, "day"),
        "15min": ("minute", 15, "day"),
        "30min": ("minute", 30, "day"),
        "1day":  ("daily",   1, "year"),
    }
    if interval not in _RANGE_MAP:
        logger.warning("[Schwab MD] fetch_price_history_range: unknown interval %s", interval)
        return pd.DataFrame()

    freq_type, freq, period_type = _RANGE_MAP[interval]
    data = _get(
        "/pricehistory",
        {
            "symbol":                ticker,
            "periodType":            period_type,
            "frequencyType":         freq_type,
            "frequency":             freq,
            "startDate":             start_ms,
            "endDate":               end_ms,
            "needExtendedHoursData": "true" if extended_hours else "false",
        },
        timeout=(8, 22),  # 8s connect + 22s read; avoids indefinite stall
    )
    candles = data.get("candles", []) if isinstance(data, dict) else []
    if not candles:
        return pd.DataFrame()

    try:
        df = pd.DataFrame(
            {
                "Open":   [c["open"]               for c in candles],
                "High":   [c["high"]               for c in candles],
                "Low":    [c["low"]                for c in candles],
                "Close":  [c["close"]              for c in candles],
                "Volume": [float(c.get("volume", 0)) for c in candles],
            },
            index=pd.to_datetime([c["datetime"] for c in candles], unit="ms", utc=True),
        )
        df = df.sort_index()
        logger.debug("[Schwab MD] range %s/%s: %d bars", ticker, interval, len(df))
        return df
    except Exception as exc:
        logger.debug("[Schwab MD] range parse error %s: %s", ticker, exc)
        return pd.DataFrame()


def fetch_price_history_batch(
    tickers:        list[str],
    interval:       str  = "1min",
    outputsize:     int  = 300,
    extended_hours: bool = False,
    max_workers:    int  = 5,
    background:     bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Parallel price-history fetch for multiple tickers.
    One Schwab /pricehistory call per ticker, rate-limited to 1.5 req/s.
    Pass background=True for retrain/warmup tasks to limit to 0.4 req/s.
    """
    if not _is_authorised() or not tickers:
        return {}

    result: dict[str, pd.DataFrame] = {}
    lock = threading.Lock()

    def _one(ticker: str) -> None:
        _rate_wait(background=background)
        df = fetch_price_history(ticker, interval, outputsize, extended_hours)
        if not df.empty:
            with lock:
                result[ticker] = df

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        concurrent.futures.wait([ex.submit(_one, t) for t in tickers])

    logger.info(f"[Schwab MD] batch {interval}: {len(result)}/{len(tickers)} tickers")
    return result


async def _fetch_batch_async_coro(
    tickers:        list[str],
    interval:       str,
    outputsize:     int,
    extended_hours: bool,
    background:     bool,
) -> dict[str, pd.DataFrame]:
    """
    Async batch fetch using aiohttp.  Requests are paced by _aio_rate_wait()
    (≤15 req/s token bucket) then limited to 5 in-flight via the semaphore.
    429 back-off is applied on top.  Background tasks use a separate 1 req/s
    throttle so they never starve live scans.
    """
    if not _is_authorised() or not tickers:
        return {}

    from agent.broker.schwab_auth import _market_data as _md_app

    result:    dict[str, pd.DataFrame] = {}
    aio_lock = asyncio.Lock()
    sem      = asyncio.Semaphore(5)    # 5 concurrent /pricehistory calls; MD poller uses 2 more → stays under 120/min

    async def _one(session: "aiohttp.ClientSession", ticker: str) -> None:
        if background:
            await _aio_bg_wait()        # retrain tasks: 1 req/s throttle
        else:
            await _aio_rate_wait()      # live scans: ≤15 req/s token bucket
            await _aio_maybe_backoff()  # honour any active 429 back-off on top
        async with sem:
            try:
                token = _md_app.get_access_token()
                headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

                # Handle resample intervals
                if interval in _RESAMPLE_MAP:
                    src_iv, resample_freq = _RESAMPLE_MAP[interval]
                    df = await _fetch_one_async(session, ticker, src_iv,
                                                 outputsize * 2, extended_hours, headers)
                    if not df.empty:
                        df = df.resample(resample_freq).agg({
                            "Open": "first", "High": "max",
                            "Low": "min", "Close": "last", "Volume": "sum",
                        }).dropna(subset=["Open", "Close"])
                        if outputsize and len(df) > outputsize:
                            df = df.iloc[-outputsize:]
                else:
                    df = await _fetch_one_async(session, ticker, interval,
                                                 outputsize, extended_hours, headers)

                if not df.empty:
                    async with aio_lock:
                        result[ticker] = df
            except Exception as e:
                logger.debug(f"[AioFetch] {ticker}/{interval}: {e}")

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[_one(session, t) for t in tickers])

    logger.info(f"[Schwab AIO] batch {interval}: {len(result)}/{len(tickers)} tickers")
    return result


async def _fetch_one_async(
    session,
    ticker: str,
    interval: str,
    outputsize: int,
    extended_hours: bool,
    headers: dict,
) -> pd.DataFrame:
    """Single async /pricehistory call."""
    mapping = _IV_MAP.get(interval)
    if not mapping:
        return pd.DataFrame()
    freq_type, freq, period_type, period = mapping
    params = {
        "symbol":                ticker,
        "periodType":            period_type,
        "period":                period,
        "frequencyType":         freq_type,
        "frequency":             freq,
        "needExtendedHoursData": "true" if extended_hours else "false",
    }
    url = f"{MARKETDATA_BASE}/pricehistory"
    _to = aiohttp.ClientTimeout(total=15)
    async with session.get(url, headers=headers, params=params, timeout=_to) as resp:
        if resp.status == 429:
            _on_429()
            return pd.DataFrame()
        if resp.status in (401, 403):
            # Run the blocking refresh in the executor so we don't block the event loop
            import asyncio as _aio
            loop = _aio.get_event_loop()
            refreshed = await loop.run_in_executor(None, _try_refresh_md_token)
            if refreshed:
                from agent.broker.schwab_auth import _market_data as _md_app
                new_token = _md_app.get_access_token()
                if new_token:
                    new_headers = {"Authorization": f"Bearer {new_token}", "Accept": "application/json"}
                    async with session.get(url, headers=new_headers, params=params, timeout=_to) as retry:
                        if retry.status != 200:
                            return pd.DataFrame()
                        _on_success()
                        data = await retry.json(content_type=None)
                else:
                    return pd.DataFrame()
            else:
                return pd.DataFrame()
        elif resp.status != 200:
            return pd.DataFrame()
        else:
            _on_success()
            data = await resp.json(content_type=None)
    candles = data.get("candles", []) if isinstance(data, dict) else []
    if not candles:
        return pd.DataFrame()
    try:
        df = pd.DataFrame({
            "Open":   [c["open"]   for c in candles],
            "High":   [c["high"]   for c in candles],
            "Low":    [c["low"]    for c in candles],
            "Close":  [c["close"]  for c in candles],
            "Volume": [float(c.get("volume", 0)) for c in candles],
        }, index=pd.to_datetime([c["datetime"] for c in candles], unit="ms", utc=True))
        df = df.sort_index()
        if outputsize and len(df) > outputsize:
            df = df.iloc[-outputsize:]
        return df
    except Exception:
        return pd.DataFrame()


def fetch_price_history_batch_async(
    tickers:        list[str],
    interval:       str  = "1min",
    outputsize:     int  = 300,
    extended_hours: bool = False,
    background:     bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Sync wrapper: submits the async batch fetch to the shared event loop and
    blocks until complete.  All requests fire concurrently (no pre-throttle);
    Schwab production has no documented REST rate limit.
    """
    # All tickers fire concurrently — timeout is just HTTP round-trip overhead.
    # 60s per ticker × 15s per HTTP call; 120s is generous for any batch size.
    timeout = 120.0
    return _run_async(
        _fetch_batch_async_coro(tickers, interval, outputsize, extended_hours, background),
        timeout=timeout,
    )


# ── Real-time quotes ──────────────────────────────────────────────────────────

def fetch_quotes(tickers: list[str]) -> dict[str, float]:
    """
    Last price for each ticker.
    Returns {ticker: last_price} — empty dict if not authorised.
    """
    if not _is_authorised() or not tickers:
        return {}
    data = _get("/quotes", {"symbols": ",".join(tickers), "fields": "quote"})
    if not isinstance(data, dict):
        return {}
    result = {}
    for ticker, info in data.items():
        try:
            q    = info.get("quote", {})
            last = q.get("lastPrice") or q.get("mark") or 0
            result[ticker] = float(last)
        except Exception:
            pass
    return result


def fetch_full_quotes(tickers: list[str]) -> dict[str, dict]:
    """
    Real-time quote data for all tickers — single API call to /quotes.
    Schwab accepts up to 500 symbols per request; 477-ticker universe fits in one call.
    Returns {ticker: {last, bid, ask, volume, open, high, low, close, pct_change, ...}}.
    Used by the 1-second MD poller for dashboard real-time price updates.
    """
    if not _is_authorised() or not tickers:
        return {}
    # Skip this cycle if a 429 back-off is active. Sleeping inside a thread-pool
    # worker blocks the MDPoller's _cf.wait deadline and causes false "timed out"
    # warnings. Let the next cycle retry once the back-off expires.
    if _backoff_until > time.time():
        return {}
    # Single call — no chunking needed for ≤500 tickers
    # timeout=4 keeps well within the MDPoller's _cf.wait(timeout=interval*5=5s)
    # window so workers never appear "timed out" from a slow-but-valid response.
    chunk = tickers[:500]
    data = _get("/quotes", {"symbols": ",".join(chunk), "fields": "quote"}, timeout=4)
    if not isinstance(data, dict):
        return {}
    result = {}
    for ticker, info in data.items():
        try:
            q = info.get("quote", {})
            _last = float(q.get("lastPrice") or 0)
            _mark = float(q.get("mark")      or 0)
            result[ticker] = {
                # lastPrice = last executed trade; mark = (bid+ask)/2 midpoint.
                # In extended hours lastPrice can be stale (last AH trade minutes ago)
                # while mark reflects current market. We send both so the frontend
                # can pick the most current value per session.
                "last":       _last or _mark,   # fallback to mark if no trades
                "mark":       _mark,
                "bid":        float(q.get("bidPrice")  or 0),
                "ask":        float(q.get("askPrice")  or 0),
                "volume":     float(q.get("totalVolume") or 0),
                "open":       float(q.get("openPrice")  or 0),
                "high":       float(q.get("highPrice")  or 0),
                "low":        float(q.get("lowPrice")   or 0),
                "close":      float(q.get("closePrice") or 0),
                "pct_change": float(q.get("netPercentChangeInDouble") or 0),
            }
        except Exception:
            pass
    return result


# ── Bulk quotes (universe screener) ──────────────────────────────────────────

_QUOTES_BULK_CHUNK = 500   # Schwab /quotes handles up to ~500 symbols per call


def fetch_quotes_bulk(tickers: list[str]) -> dict[str, dict]:
    """
    Bulk quote for any number of tickers — batches into 500-symbol chunks.

    Returns {ticker: {last, volume, pct_change, bid, ask}}.
    Used by the universe screener to rank all ~500 tickers in 1-2 API calls
    before deciding which ones get a full price-history fetch.
    """
    if not _is_authorised() or not tickers:
        return {}

    result: dict[str, dict] = {}
    for i in range(0, len(tickers), _QUOTES_BULK_CHUNK):
        batch = tickers[i:i + _QUOTES_BULK_CHUNK]
        _rate_wait()
        data = _get("/quotes", {"symbols": ",".join(batch), "fields": "quote"})
        if not isinstance(data, dict):
            continue
        for sym, info in data.items():
            try:
                q = info.get("quote", {})
                result[sym] = {
                    "last":       float(q.get("lastPrice") or q.get("mark") or 0),
                    "volume":     int(q.get("totalVolume") or 0),
                    "pct_change": float(q.get("netPercentChangeInDouble") or 0),
                    "bid":        float(q.get("bidPrice") or 0),
                    "ask":        float(q.get("askPrice") or 0),
                }
            except Exception:
                pass
    logger.debug(f"[Schwab MD] bulk quotes: {len(result)}/{len(tickers)} symbols")
    return result


# ── Instruments / symbol search ───────────────────────────────────────────────

def fetch_instruments_search(
    query:      str,
    projection: str = "symbol-search",   # symbol-search | desc-search | fundamental
) -> list[dict]:
    """
    Search Schwab instruments.

    projection="symbol-search"  → find tickers matching a symbol prefix
    projection="desc-search"    → find by company name keywords
    projection="fundamental"    → returns fundamentals (mktCap, avgVol10Days, exchange)

    Returns list of dicts with keys: symbol, description, exchange, assetType,
    plus fundamentals fields when projection="fundamental".
    """
    if not _is_authorised():
        return []
    data = _get("/instruments", {"symbol": query, "projection": projection})
    items = []
    raw = data.get("instruments", data) if isinstance(data, dict) else []
    if isinstance(raw, dict):
        raw = list(raw.values())
    for item in (raw if isinstance(raw, list) else []):
        try:
            fund = item.get("fundamental", {})
            items.append({
                "symbol":      item.get("symbol", ""),
                "description": item.get("description", ""),
                "exchange":    item.get("exchange", ""),
                "asset_type":  item.get("assetType", ""),
                "avg_vol_10d": fund.get("avg10DaysVolume", 0),
                "mkt_cap":     fund.get("marketCapFloat", 0),
            })
        except Exception:
            pass
    return items


# ── Movers ────────────────────────────────────────────────────────────────────

# Schwab index symbols for movers
MOVERS_NASDAQ = "$COMPX"   # NASDAQ Composite
MOVERS_SPX    = "$SPX"     # S&P 500
MOVERS_DJI    = "$DJI"     # Dow Jones


def fetch_movers(
    index:  str = MOVERS_NASDAQ,
    sort:   str = "PERCENT_CHANGE_UP",   # VOLUME | TRADES | PERCENT_CHANGE_UP | PERCENT_CHANGE_DOWN
    freq:   int = 0,                     # 0=all-day, 1, 5, 10, 30, 60 minutes
) -> list[dict]:
    """
    Top movers for a given index.

    Returns a list of dicts with keys:
      symbol, description, last_price, pct_change, volume, trades, direction
    Returns [] if not authorised or on error.
    """
    if not _is_authorised():
        return []
    data = _get(f"/movers/{index}", {"sort": sort, "frequency": freq})
    if not isinstance(data, dict):
        return []
    screeners = data.get("screeners", [])
    result = []
    for item in screeners:
        try:
            result.append({
                "symbol":      item.get("symbol", ""),
                "description": item.get("description", ""),
                "last_price":  float(item.get("lastPrice", 0)),
                "pct_change":  float(item.get("percentChange", 0)),
                "volume":      int(item.get("totalVolume", 0)),
                "trades":      int(item.get("trades", 0)),
                "direction":   item.get("direction", ""),
            })
        except Exception:
            pass
    logger.debug(f"[Schwab MD] movers {index}/{sort}: {len(result)} symbols")
    return result


def fetch_top_movers_symbols(n: int = 20) -> list[str]:
    """
    Return up to n ticker symbols that are moving most (up or down) on NASDAQ.
    Useful for prioritising scanner focus.
    """
    up   = fetch_movers(MOVERS_NASDAQ, sort="PERCENT_CHANGE_UP")
    down = fetch_movers(MOVERS_NASDAQ, sort="PERCENT_CHANGE_DOWN")
    seen, symbols = set(), []
    for item in up + down:
        s = item.get("symbol", "")
        if s and s not in seen:
            seen.add(s)
            symbols.append(s)
        if len(symbols) >= n:
            break
    return symbols


# ── Option chains ─────────────────────────────────────────────────────────────

def fetch_iv(ticker: str) -> float | None:
    """
    Fetch the at-the-money implied volatility for a ticker.
    Returns the volatility as a decimal (e.g. 0.35 = 35%) or None on error.
    """
    if not _is_authorised():
        return None
    data = _get("/chains", {
        "symbol":      ticker,
        "contractType": "ALL",
        "range":       "ATM",
        "optionType":  "S",   # standard options only
    })
    if not isinstance(data, dict):
        return None
    try:
        vix = data.get("volatility")
        if vix is not None:
            return float(vix)
        # Fallback: average IV across ATM calls
        call_map = data.get("callExpDateMap", {})
        ivs = []
        for exp_key, strikes in call_map.items():
            for strike_key, contracts in strikes.items():
                for c in contracts:
                    v = c.get("volatility")
                    if v and v > 0:
                        ivs.append(float(v))
        return sum(ivs) / len(ivs) if ivs else None
    except Exception as e:
        logger.debug(f"[Schwab MD] IV parse error {ticker}: {e}")
        return None


def fetch_option_chain(
    ticker:        str,
    contract_type: str = "ALL",   # CALL | PUT | ALL
    expiration:    str | None = None,  # "YYYY-MM-DD" — None = nearest expiry
) -> dict:
    """
    Fetch full option chain for a ticker.

    Returns a dict with:
      calls: list of call contract dicts
      puts:  list of put contract dicts
      iv:    float (overall implied volatility) or None
    """
    if not _is_authorised():
        return {"calls": [], "puts": [], "iv": None}
    params: dict = {
        "symbol":       ticker,
        "contractType": contract_type,
        "range":        "ALL",
    }
    if expiration:
        params["toExpirationDate"]   = expiration
        params["fromExpirationDate"] = expiration
    data = _get("/chains", params)
    if not isinstance(data, dict):
        return {"calls": [], "puts": [], "iv": None}

    def _flatten(exp_date_map: dict) -> list[dict]:
        contracts = []
        for exp_key, strikes in exp_date_map.items():
            for strike_key, items in strikes.items():
                for c in items:
                    contracts.append({
                        "expiry":       exp_key.split(":")[0],
                        "strike":       float(strike_key),
                        "bid":          c.get("bid", 0),
                        "ask":          c.get("ask", 0),
                        "last":         c.get("last", 0),
                        "volume":       c.get("totalVolume", 0),
                        "open_interest":c.get("openInterest", 0),
                        "iv":           c.get("volatility", 0),
                        "delta":        c.get("delta", 0),
                        "gamma":        c.get("gamma", 0),
                        "theta":        c.get("theta", 0),
                        "vega":         c.get("vega", 0),
                        "in_the_money": c.get("inTheMoney", False),
                    })
        return contracts

    return {
        "calls": _flatten(data.get("callExpDateMap", {})),
        "puts":  _flatten(data.get("putExpDateMap",  {})),
        "iv":    data.get("volatility"),
    }


# ── Market hours ──────────────────────────────────────────────────────────────

def fetch_market_hours(market: str = "equity") -> dict:
    """
    Return market session info for the given market type.
    market: equity | option | bond | future | forex

    Returns dict with keys:
      is_open:     bool
      open_time:   str (ISO) or None
      close_time:  str (ISO) or None
    """
    if not _is_authorised():
        return {"is_open": None, "open_time": None, "close_time": None}
    today = date.today().isoformat()
    data = _get("/markets", {"markets": market, "date": today})
    if not isinstance(data, dict):
        return {"is_open": None, "open_time": None, "close_time": None}
    try:
        mkt = data.get(market, {})
        # Response shape: {market: {product_key: {isOpen, sessionHours: {regularMarket: [{start, end}]}}}}
        for product_key, info in mkt.items():
            is_open   = info.get("isOpen", False)
            sessions  = info.get("sessionHours", {}).get("regularMarket", [])
            open_time  = sessions[0].get("start")  if sessions else None
            close_time = sessions[0].get("end")    if sessions else None
            return {"is_open": is_open, "open_time": open_time, "close_time": close_time}
    except Exception as e:
        logger.debug(f"[Schwab MD] market hours parse error: {e}")
    return {"is_open": None, "open_time": None, "close_time": None}


def is_market_open() -> bool | None:
    """
    Quick check: is the equity market currently open?
    Returns True/False, or None if Schwab is not authorised.
    """
    result = fetch_market_hours("equity")
    return result.get("is_open")
