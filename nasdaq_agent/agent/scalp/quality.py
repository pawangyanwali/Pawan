"""Canonical classification of market-data completeness failures."""
from __future__ import annotations

MARKET_DATA_MISSING_BLOCKERS = {
    "VALKEY_UNAVAILABLE",
    "BAR_FEED_READ_FAILED",
    "ONE_MINUTE_BARS_MISSING",
    "ONE_MINUTE_BARS_INVALID",
    "RSI_14_MISSING",
    "RSI_7_MISSING",
    "RSI_2_MISSING",
    "MACD_HIST_MISSING",
    "MACD_HIST_PREV_MISSING",
    "ATR_14_MISSING",
    "VWAP_MISSING",
    "RVOL_MISSING",
}

ACTIVE_FEED_BLOCKERS = {
    "QUOTE_STALE",
    "INDICATOR_BAR_STALE",
    "REST_FALLBACK_NOT_TRADABLE",
}


def has_market_data_gap(blockers: list[str], session: str) -> bool:
    """True only for missing canonical market inputs, not context/risk gates."""
    values = {str(blocker).upper() for blocker in blockers}
    if values & MARKET_DATA_MISSING_BLOCKERS:
        return True
    if str(session or "").upper() != "CLOSED" and values & ACTIVE_FEED_BLOCKERS:
        return True
    return False
