"""
Earnings / event blackout — suppress trading signals within N days of an
earnings announcement.

Strategy:
  - Earnings dates are fetched via Twelve Data /earnings endpoint (same API key).
  - A 3-day pre-earnings blackout protects against gap risk.
  - A 1-day post-earnings cooldown protects against gap-fill traps.
  - Results are cached for 24 hours to minimise API credit consumption.
"""
from __future__ import annotations
import logging
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_PRE_EARNINGS_DAYS  = 3
_POST_EARNINGS_DAYS = 1
_CACHE_TTL          = 24 * 3600   # 24 h for successful fetches
_CACHE_TTL_NONE     = 3600        # 1 h for failed/no-result fetches (retry sooner)

_cache: dict[str, tuple[Optional[datetime], float]] = {}  # ticker → (next_date, fetched_at)


def _fetch_next_earnings(ticker: str) -> Optional[datetime]:
    """
    Look up the next earnings date from the context_store DB (populated by the
    context-intel service from Finnhub).

    Falls back to None when the DB is unavailable or the context-intel service
    has not run yet — earnings blocking remains disabled in that case, which
    is the same behaviour as before Phase 1.
    """
    try:
        from agent.context_store import get_next_earnings_from_db
        return get_next_earnings_from_db(ticker)
    except Exception:
        return None


def get_next_earnings(ticker: str) -> Optional[datetime]:
    """Return next earnings datetime (UTC-aware) or None.  Cached 6h."""
    now = time.time()
    if ticker in _cache:
        date, fetched = _cache[ticker]
        if now - fetched < _CACHE_TTL:
            return date
    date = _fetch_next_earnings(ticker)
    # Failed/no-result fetches use a shorter TTL so we retry sooner
    ttl = _CACHE_TTL if date is not None else _CACHE_TTL_NONE
    _cache[ticker] = (date, now - (_CACHE_TTL - ttl))
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
