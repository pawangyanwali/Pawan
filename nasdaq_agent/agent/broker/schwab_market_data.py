"""
Schwab Market Data API — price history fetcher.

Returns OHLCV DataFrames in the same format as data_fetcher so it can
serve as a transparent fallback when Twelve Data returns 429.

Twelve Data intervals map to Schwab parameters:
  "1min"  → frequencyType=minute, frequency=1
  "5min"  → frequencyType=minute, frequency=5
  "15min" → frequencyType=minute, frequency=15
  "30min" → frequencyType=minute, frequency=30
  "1h"    → frequencyType=minute, frequency=60
  "1day"  → frequencyType=daily,  frequency=1
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from agent.broker.schwab_auth import get_access_token, get_token_status

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
    """True if we have a usable access token."""
    try:
        return get_token_status().get("connected", False)
    except Exception:
        return False


def _auth_headers() -> dict | None:
    token = get_access_token()
    if not token:
        return None
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def fetch_price_history(
    ticker:     str,
    interval:   str,
    outputsize: int = 300,
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
    headers = _auth_headers()
    if not headers:
        return pd.DataFrame()

    try:
        r = requests.get(
            f"{MARKETDATA_BASE}/pricehistory",
            headers=headers,
            params={
                "symbol":               ticker,
                "periodType":           period_type,
                "period":               period,
                "frequencyType":        freq_type,
                "frequency":            freq,
                "needExtendedHoursData": False,
            },
            timeout=20,
        )
        r.raise_for_status()
        candles = r.json().get("candles", [])
    except Exception as e:
        logger.warning(f"[Schwab MD] {ticker}/{interval} failed: {e}")
        return pd.DataFrame()

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
        logger.debug(f"[Schwab MD] {ticker}/{interval}: {len(df)} bars")
        return df
    except Exception as e:
        logger.debug(f"[Schwab MD] parse error {ticker}: {e}")
        return pd.DataFrame()


def fetch_quotes(tickers: list[str]) -> dict[str, float]:
    """Last price for each ticker. Returns empty dict if not authorised."""
    if not _is_authorised() or not tickers:
        return {}
    headers = _auth_headers()
    if not headers:
        return {}
    try:
        r = requests.get(
            f"{MARKETDATA_BASE}/quotes",
            headers=headers,
            params={"symbols": ",".join(tickers), "fields": "quote"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning(f"[Schwab MD] quotes failed: {e}")
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
