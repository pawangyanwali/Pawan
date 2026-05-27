"""
Finnhub provider adapter.

Implements:
  fetch_market_news(category)         — broad market / sector news (1 req/call)
  fetch_company_news(ticker, from, to) — symbol-level news (1 req/call)
  fetch_earnings_calendar(from, to)    — all earnings in a date range (1 req/call)

Rate limit: 60 req/min (free tier).  Internal token-bucket holds usage to 55/min
so there is always headroom for callers to burst slightly.

API key: FINNHUB_API_KEY env var.  Every public function returns an empty
list / None gracefully when the key is absent — the rest of the system
continues with stale or default data.

Docs:
  https://finnhub.io/docs/api/market-news
  https://finnhub.io/docs/api/company-news
  https://finnhub.io/docs/api/earnings-calendar
"""
from __future__ import annotations

import logging
import os
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_BASE = "https://finnhub.io/api/v1"
_TIMEOUT_S = 10

# Token-bucket rate limiter: max 55 req / 60-second window (conservative vs 60 limit)
_rate_lock       = threading.Lock()
_req_timestamps: list[float] = []
_WINDOW_S        = 60.0
_MAX_REQUESTS    = 55


# ── Internal helpers ──────────────────────────────────────────────────────────

def _api_key() -> Optional[str]:
    """Return FINNHUB_API_KEY or None."""
    return os.getenv("FINNHUB_API_KEY") or None


def _rate_wait() -> None:
    """Block until a request slot is available within the 60-second window."""
    with _rate_lock:
        now = time.time()
        # Evict timestamps older than 60 s
        _req_timestamps[:] = [t for t in _req_timestamps if now - t < _WINDOW_S]
        if len(_req_timestamps) >= _MAX_REQUESTS:
            oldest = _req_timestamps[0]
            sleep_s = _WINDOW_S - (now - oldest) + 0.05
            if sleep_s > 0:
                time.sleep(sleep_s)
        _req_timestamps.append(time.time())


def _get(path: str, params: dict) -> Optional[object]:
    """
    Execute a rate-limited GET to Finnhub.
    Returns parsed JSON body, or None on any error (network, HTTP, missing key).
    Never raises.
    """
    key = _api_key()
    if not key:
        logger.debug("[Finnhub] FINNHUB_API_KEY not set — request skipped")
        return None

    params = {**params, "token": key}
    _rate_wait()

    try:
        resp = requests.get(f"{_BASE}{path}", params=params, timeout=_TIMEOUT_S)
        if resp.status_code == 429:
            logger.warning("[Finnhub] 429 rate-limit — backing off 30 s")
            time.sleep(30)
            return None
        if resp.status_code != 200:
            logger.debug("[Finnhub] HTTP %d for %s", resp.status_code, path)
            return None
        return resp.json()
    except requests.exceptions.Timeout:
        logger.debug("[Finnhub] Timeout for %s", path)
        return None
    except Exception as exc:
        logger.debug("[Finnhub] Request error for %s: %s", path, exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_market_news(category: str = "general") -> list[dict]:
    """
    Fetch broad market news.  One API call regardless of universe size.

    category: "general" | "forex" | "crypto" | "merger"

    Returns list of articles:
      {id, category, datetime (unix), headline, source, summary, url, image, related}
    Returns [] when FINNHUB_API_KEY is absent or on any error.
    """
    data = _get("/news", {"category": category})
    if not isinstance(data, list):
        return []
    return data


def fetch_company_news(ticker: str, from_date: str, to_date: str) -> list[dict]:
    """
    Fetch company-specific news for one symbol.  One API call.

    from_date, to_date: "YYYY-MM-DD" strings (inclusive).

    Returns same structure as fetch_market_news.
    Returns [] when FINNHUB_API_KEY is absent or on any error.
    """
    data = _get(
        "/company-news",
        {"symbol": ticker, "from": from_date, "to": to_date},
    )
    if not isinstance(data, list):
        return []
    return data


def fetch_earnings_calendar(
    from_date: str,
    to_date: str,
    symbol: Optional[str] = None,
) -> list[dict]:
    """
    Fetch earnings calendar for a date range.
    Without symbol: returns ALL companies reporting in the window (one call).
    With symbol: filtered to that ticker.

    Returns list of entries:
      {symbol, date (YYYY-MM-DD), hour ("bmo"|"amc"|"dmh"),
       epsEstimate, epsActual, revenueEstimate, revenueActual, quarter, year}
    Returns [] on any error.
    """
    params: dict = {"from": from_date, "to": to_date}
    if symbol:
        params["symbol"] = symbol

    data = _get("/calendar/earnings", params)
    if not isinstance(data, dict):
        return []
    return data.get("earningsCalendar", [])


def is_available() -> bool:
    """Return True if FINNHUB_API_KEY is configured (key existence only, no ping)."""
    return bool(_api_key())
