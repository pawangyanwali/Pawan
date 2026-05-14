"""
Earnings / event blackout — suppress trading signals within N days of an
earnings announcement.

Strategy:
  - Earnings dates are fetched via Twelve Data /earnings endpoint (same API key).
  - A 3-day pre-earnings blackout protects against gap risk.
  - A 1-day post-earnings cooldown protects against gap-fill traps.
  - Results are cached for 6 hours to stay within credit budget.
"""
from __future__ import annotations
import logging
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_PRE_EARNINGS_DAYS  = 3
_POST_EARNINGS_DAYS = 1
_CACHE_TTL          = 6 * 3600   # 6 hours

_cache: dict[str, tuple[Optional[datetime], float]] = {}  # ticker → (next_date, fetched_at)


def _fetch_next_earnings(ticker: str) -> Optional[datetime]:
    """Fetch next earnings date from Twelve Data /earnings. Returns None on failure."""
    try:
        from agent.data_fetcher import _get
        data = _get("/earnings", {"symbol": ticker, "outputsize": 5})
        earnings = data.get("earnings") or data.get("data") or []
        if not earnings:
            return None
        now = datetime.now(timezone.utc)
        future: list[datetime] = []
        for entry in earnings:
            date_str = entry.get("date") or entry.get("report_date") or ""
            if not date_str:
                continue
            try:
                dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
                if dt > now:
                    future.append(dt)
            except ValueError:
                continue
        return min(future) if future else None
    except Exception as e:
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

        if next_dt.tzinfo is None:
            next_dt = next_dt.replace(tzinfo=timezone.utc)

        now   = datetime.now(timezone.utc)
        delta = (next_dt - now).days
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
