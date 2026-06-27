"""Adapter from the shared Valkey price bus to the scalp quote contract."""
from __future__ import annotations

import time

from .models import QuoteSnapshot, QuoteSource


def quote_snapshot_from_price_bus(ticker: str) -> QuoteSnapshot:
    from agent.valkey_client import get_price

    return quote_snapshot_from_payload(ticker, get_price(ticker) or {})


def quote_snapshot_from_payload(ticker: str, quote: dict) -> QuoteSnapshot:
    """Build a quote contract from an already-batched price-bus payload."""
    try:
        updated_at = float(quote.get("updated_at") or 0.0)
    except (TypeError, ValueError):
        updated_at = 0.0
    age_ms = int(max(0.0, time.time() - updated_at) * 1000) if updated_at > 0 else -1
    status = str(quote.get("source_status") or "").upper()
    source = (
        QuoteSource.WS
        if status == "LIVE"
        else QuoteSource.REST
        if status in {"REST_FALLBACK", "FALLBACK"}
        else QuoteSource.STALE
        if status in {"STALE", "STALE_CACHE", "SCAN_SNAPSHOT"}
        else QuoteSource.UNKNOWN
    )
    return QuoteSnapshot(
        ticker=ticker,
        last=_float(quote.get("last") or quote.get("mark")),
        bid=_float(quote.get("bid")),
        ask=_float(quote.get("ask")),
        data_age_ms=age_ms,
        source=source,
    )


def _float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
