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
    """True only when the dedicated Market Data app has its own valid token.
    Never falls back to the Accounts+Trading token — that app lacks
    Market Data Production access and will 401."""
    try:
        from agent.broker.schwab_auth import _market_data
        return bool(_market_data.get_access_token())
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


def _get(path: str, params: dict, timeout: int = 20) -> dict | list:
    """Authenticated GET to the Schwab Market Data API."""
    headers = _auth_headers()
    if not headers:
        return {}
    try:
        r = requests.get(f"{MARKETDATA_BASE}{path}", headers=headers,
                         params=params, timeout=timeout)
        if r.status_code == 429:
            _on_429()
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
_BACKOFF_BASE   = 2.0              # seconds for first 429 back-off
_BACKOFF_MAX    = 30.0             # cap at 30s

# Background-caller throttle: retrain tasks capped at 1 req/s.
_bg_lock = threading.Lock()
_bg_last = 0.0
_BG_GAP  = 1.0   # 1 s between background calls


def _on_429() -> None:
    """Called when Schwab returns HTTP 429. Sets a short back-off window."""
    global _backoff_until
    with _rate_lock:
        remaining = _backoff_until - time.time()
        # Double the back-off each consecutive 429 (exponential), cap at max.
        new_backoff = min(max(remaining * 2.0, _BACKOFF_BASE), _BACKOFF_MAX)
        _backoff_until = time.time() + new_backoff
    logger.warning(f"[Schwab MD] 429 received — pausing {new_backoff:.1f}s")


def _on_success() -> None:
    """Clear back-off window after a clean response."""
    global _backoff_until
    if _backoff_until > time.time():
        with _rate_lock:
            _backoff_until = 0.0


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
    Async batch fetch using aiohttp.  All HTTP requests run concurrently;
    the async token bucket (_aio_rate_wait) enforces ≤10 req/s (adaptive 429
    backoff to 1.5 req/s minimum); background tasks capped at 1 req/s.
    """
    if not _is_authorised() or not tickers:
        return {}

    from agent.broker.schwab_auth import _market_data as _md_app

    result:    dict[str, pd.DataFrame] = {}
    aio_lock = asyncio.Lock()

    async def _one(session: "aiohttp.ClientSession", ticker: str) -> None:
        if background:
            await _aio_bg_wait()      # throttle retrain tasks to 1 req/s
        else:
            await _aio_maybe_backoff()  # honour any active 429 back-off, else fire immediately
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
    async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
        if resp.status == 429:
            _on_429()
            return pd.DataFrame()
        if resp.status != 200:
            return pd.DataFrame()
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
    # Single call — no chunking needed for ≤500 tickers
    chunk = tickers[:500]
    data = _get("/quotes", {"symbols": ",".join(chunk), "fields": "quote"}, timeout=8)
    if not isinstance(data, dict):
        return {}
    result = {}
    for ticker, info in data.items():
        try:
            q = info.get("quote", {})
            result[ticker] = {
                "last":       float(q.get("lastPrice") or q.get("mark") or 0),
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
