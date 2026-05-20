"""
Schwab Market Data API.

Endpoints used:
  /pricehistory              — OHLCV bars (fallback for Twelve Data)
  /quotes                    — real-time last price for a list of symbols
  /movers/{index}            — top gainers/losers for NASDAQ / S&P 500
  /chains                    — option chain (IV, OI, Greeks)
  /markets                   — market session hours (open/closed check)

Twelve Data intervals → Schwab parameters:
  "1min"  → frequencyType=minute, frequency=1,  periodType=day,   period=10
  "5min"  → frequencyType=minute, frequency=5,  periodType=month, period=3
  "15min" → frequencyType=minute, frequency=15, periodType=month, period=6
  "30min" → frequencyType=minute, frequency=30, periodType=month, period=6
  "1h"    → frequencyType=minute, frequency=60, periodType=month, period=6
  "1day"  → frequencyType=daily,  frequency=1,  periodType=year,  period=2
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from datetime import date

import pandas as pd
import requests

from agent.broker.schwab_auth import get_md_token_status

logger = logging.getLogger(__name__)

MARKETDATA_BASE = "https://api.schwabapi.com/marketdata/v1"

# Twelve Data interval → (frequencyType, frequency, periodType, period)
_IV_MAP = {
    "1min":  ("minute",  1,  "day",   10),
    "5min":  ("minute",  5,  "month",  3),
    "15min": ("minute", 15,  "month",  6),
    "30min": ("minute", 30,  "month",  6),
    "1h":    ("minute", 60,  "month",  6),
    "4h":    ("minute", 60,  "year",   1),
    "1day":  ("daily",   1,  "year",   2),
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
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"[Schwab MD] {path} failed: {e}")
        return {}


# ── Price history ─────────────────────────────────────────────────────────────

# ── Global rate limiter (shared across all callers) ───────────────────────────
# Schwab Market Data: 120 req/min limit. Use 90 req/min (1.5/s) for headroom.
_rate_lock = threading.Lock()
_rate_last = 0.0
_RATE_GAP  = 1.0 / 1.5   # 0.667 s between requests


def _rate_wait() -> None:
    global _rate_last
    with _rate_lock:
        gap = time.time() - _rate_last
        if gap < _RATE_GAP:
            time.sleep(_RATE_GAP - gap)
        _rate_last = time.time()


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
    """
    if not _is_authorised():
        return pd.DataFrame()

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
) -> dict[str, pd.DataFrame]:
    """
    Parallel price-history fetch for multiple tickers.
    One Schwab /pricehistory call per ticker, rate-limited to 1.5 req/s.
    """
    if not _is_authorised() or not tickers:
        return {}

    result: dict[str, pd.DataFrame] = {}
    lock = threading.Lock()

    def _one(ticker: str) -> None:
        _rate_wait()
        df = fetch_price_history(ticker, interval, outputsize, extended_hours)
        if not df.empty:
            with lock:
                result[ticker] = df

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        concurrent.futures.wait([ex.submit(_one, t) for t in tickers])

    logger.info(f"[Schwab MD] batch {interval}: {len(result)}/{len(tickers)} tickers")
    return result


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
    Full quote data (bid/ask/volume/52w high/low etc.) for each ticker.
    Returns {ticker: {field: value}} — useful for richer signal features.
    """
    if not _is_authorised() or not tickers:
        return {}
    data = _get("/quotes", {"symbols": ",".join(tickers), "fields": "quote,reference"})
    if not isinstance(data, dict):
        return {}
    result = {}
    for ticker, info in data.items():
        try:
            q = info.get("quote", {})
            ref = info.get("reference", {})
            result[ticker] = {
                "last":        q.get("lastPrice") or q.get("mark", 0),
                "bid":         q.get("bidPrice", 0),
                "ask":         q.get("askPrice", 0),
                "volume":      q.get("totalVolume", 0),
                "open":        q.get("openPrice", 0),
                "high":        q.get("highPrice", 0),
                "low":         q.get("lowPrice", 0),
                "close":       q.get("closePrice", 0),
                "pct_change":  q.get("netPercentChangeInDouble", 0),
                "52w_high":    q.get("52WkHigh", 0),
                "52w_low":     q.get("52WkLow", 0),
                "description": ref.get("description", ""),
            }
        except Exception:
            pass
    return result


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
