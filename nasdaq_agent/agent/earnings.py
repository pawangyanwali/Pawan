"""
Earnings / event blackout — suppress trading signals within N days of an
earnings announcement.

Strategy:
  - Earnings dates are fetched once per session via yfinance (no API key needed).
  - A 3-day pre-earnings blackout protects against gap risk.
  - A 1-day post-earnings cooldown protects against gap-fill traps.
  - Results are cached for 6 hours so we don't hammer yfinance.
"""
from __future__ import annotations
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# days before next earnings where we suppress signals
_PRE_EARNINGS_DAYS  = 3
_POST_EARNINGS_DAYS = 1
_CACHE_TTL          = 6 * 3600   # 6 hours
# Tickers that consistently 404 on Yahoo Finance — skip earnings check entirely
_KNOWN_MISSING: set[str] = set()

_cache: dict[str, tuple[Optional[datetime], float]] = {}  # ticker → (next_date, fetched_at)


def _fetch_next_earnings(ticker: str) -> Optional[datetime]:
    """Try to get next earnings date via yfinance. Returns None on failure."""
    if ticker in _KNOWN_MISSING:
        return None
    try:
        import yfinance as yf
        # Silence yfinance's own ERROR/WARNING logs — 404s are expected for some tickers
        yf_logger = logging.getLogger("yfinance")
        old_level = yf_logger.level
        yf_logger.setLevel(logging.CRITICAL)
        try:
            t   = yf.Ticker(ticker)
            cal = t.calendar          # dict with 'Earnings Date' key (may be list or Timestamp)
        finally:
            yf_logger.setLevel(old_level)

        if cal is None:
            return None
        dates = cal.get("Earnings Date") or cal.get("Earnings Dates")
        if dates is None:
            return None
        if hasattr(dates, "__iter__") and not isinstance(dates, str):
            dates = list(dates)
            future = [d for d in dates if hasattr(d, "timestamp") and d > datetime.now(timezone.utc)]
            return min(future) if future else None
        if hasattr(dates, "timestamp"):
            return dates
    except Exception as e:
        msg = str(e)
        if "404" in msg or "Not Found" in msg or "Quote not found" in msg:
            _KNOWN_MISSING.add(ticker)
            logger.debug(f"[{ticker}] yfinance 404 — ticker unknown to Yahoo Finance, skipping earnings check")
        else:
            logger.debug(f"[{ticker}] earnings fetch failed: {e}")
    return None


def get_next_earnings(ticker: str) -> Optional[datetime]:
    """Return next earnings datetime (UTC-aware) or None.  Cached 6h."""
    now = time.time()
    if ticker in _cache:
        date, fetched = _cache[ticker]
        if now - fetched < _CACHE_TTL:
            return date
    date = _fetch_next_earnings(ticker)
    _cache[ticker] = (date, now)
    return date


def earnings_blackout(ticker: str) -> dict:
    """
    Return blackout status for ticker.

    Returns dict:
      blocked    : bool — should signal be suppressed?
      reason     : str  — human-readable reason
      next_date  : str  — ISO date string or ""
      days_away  : int  — days until / since earnings (negative = post)
    """
    result = {"blocked": False, "reason": "", "next_date": "", "days_away": 0}
    try:
        next_dt = get_next_earnings(ticker)
        if next_dt is None:
            return result

        # Normalise to UTC-aware
        if next_dt.tzinfo is None:
            next_dt = next_dt.replace(tzinfo=timezone.utc)

        now      = datetime.now(timezone.utc)
        delta    = (next_dt - now).days
        result["next_date"] = next_dt.strftime("%b %d, %Y")
        result["days_away"] = delta

        if 0 <= delta <= _PRE_EARNINGS_DAYS:
            result["blocked"] = True
            result["reason"]  = (
                f"Earnings in {delta}d ({next_dt.strftime('%b %d')}) — "
                "blackout: gap risk too high."
            )
        elif -_POST_EARNINGS_DAYS <= delta < 0:
            result["blocked"] = True
            result["reason"]  = (
                f"Post-earnings cooldown ({abs(delta)}d after) — "
                "gap-fill trap risk."
            )
        elif delta <= _PRE_EARNINGS_DAYS + 4:
            result["reason"] = (
                f"Earnings in {delta}d ({next_dt.strftime('%b %d')}) — "
                "reduce size."
            )
    except Exception as e:
        logger.debug(f"[{ticker}] earnings_blackout error: {e}")
    return result
