"""Authoritative one-minute OHLCV hydration for the scalp runtime."""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

BAR_LIMIT = 500
BAR_TTL_S = 7 * 24 * 60 * 60
MIN_INDICATOR_BARS = 35
MIN_VOLUME_BARS = 10


def frame_is_usable(
    frame: Any, *, require_fresh: bool = False, max_age_s: float = 180.0
) -> bool:
    if frame is None or len(frame) < MIN_INDICATOR_BARS:
        return False
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(frame.columns):
        return False
    numeric = frame[list(required)].apply(pd.to_numeric, errors="coerce")
    if numeric[["Open", "High", "Low", "Close"]].tail(MIN_INDICATOR_BARS).isna().any().any():
        return False
    if int((numeric["Volume"] > 0).sum()) < MIN_VOLUME_BARS:
        return False
    if require_fresh:
        last = frame.index[-1]
        if getattr(last, "tzinfo", None) is None:
            last = pd.Timestamp(last, tz="UTC")
        age_s = time.time() - pd.Timestamp(last).timestamp()
        if age_s < 0 or age_s > max(60.0, float(max_age_s)):
            return False
    return True


def publish_frames_to_valkey(
    frames: dict[str, pd.DataFrame], *, limit: int = BAR_LIMIT, ttl_s: int = BAR_TTL_S
) -> int:
    """Atomically replace each ticker's rolling Valkey bar list."""
    from agent.valkey_client import _get_client

    client = _get_client()
    if client is None:
        return 0
    published = 0
    for ticker, raw_frame in frames.items():
        frame = _normalise_frame(raw_frame).tail(max(MIN_INDICATOR_BARS, int(limit)))
        payload = [_row_payload(index, row) for index, row in frame.iterrows()]
        payload = [item for item in payload if item is not None]
        if not payload:
            continue
        key = f"md:1m:{str(ticker).upper()}"
        pipe = client.pipeline(transaction=True)
        pipe.delete(key)
        pipe.rpush(key, *[json.dumps(item, separators=(",", ":")) for item in payload])
        pipe.ltrim(key, -max(MIN_INDICATOR_BARS, int(limit)), -1)
        pipe.expire(key, max(3600, int(ttl_s)))
        pipe.execute()
        published += 1
    return published


def hydrate_one_minute_history(
    tickers: Iterable[str],
    *,
    fetch_missing: bool = True,
    force_refresh: bool = False,
    require_fresh: bool = False,
) -> dict[str, int]:
    """Restore Valkey from PostgreSQL, then fill incomplete symbols from Schwab."""
    from agent.historical_cache import _upsert_bars, get_recent_bars_bulk

    symbols = [str(ticker).upper() for ticker in dict.fromkeys(tickers) if ticker]
    stored = get_recent_bars_bulk(symbols, "1min", BAR_LIMIT)
    usable = {
        ticker: frame
        for ticker, frame in stored.items()
        if frame_is_usable(frame)
    }
    published = publish_frames_to_valkey(usable)
    current = (
        {
            ticker: frame
            for ticker, frame in usable.items()
            if frame_is_usable(frame, require_fresh=True)
        }
        if require_fresh
        else usable
    )
    missing = symbols if force_refresh else [ticker for ticker in symbols if ticker not in current]
    fetched: dict[str, pd.DataFrame] = {}

    if fetch_missing and missing:
        from agent.broker.schwab_market_data import fetch_price_history_batch_async

        logger.info("[ScalpBars] Fetching authoritative OHLCV for %d tickers", len(missing))
        fetched = fetch_price_history_batch_async(
            missing,
            interval="1min",
            outputsize=BAR_LIMIT,
            extended_hours=True,
            background=True,
        )
        valid_fetched: dict[str, pd.DataFrame] = {}
        for ticker, frame in fetched.items():
            normalised = _normalise_frame(frame)
            if not frame_is_usable(normalised):
                continue
            _upsert_bars(ticker, "1min", normalised)
            valid_fetched[ticker] = normalised
        published += publish_frames_to_valkey(valid_fetched)
        fetched = valid_fetched

    metrics = {
        "requested": len(symbols),
        "postgres_usable": len(usable),
        "rest_requested": len(missing) if fetch_missing else 0,
        "rest_usable": len(fetched),
        "published": published,
        "unresolved": max(0, len(symbols) - len(set(usable) | set(fetched))),
    }
    logger.info("[ScalpBars] Hydration complete: %s", metrics)
    return metrics


def missing_valkey_history(
    tickers: Iterable[str], *, min_bars: int = MIN_INDICATOR_BARS
) -> list[str]:
    """Return symbols whose shared rolling bar list is absent or too short."""
    from agent.valkey_client import _get_client

    symbols = [str(ticker).upper() for ticker in dict.fromkeys(tickers) if ticker]
    client = _get_client()
    if client is None:
        return symbols
    pipe = client.pipeline(transaction=False)
    for ticker in symbols:
        pipe.llen(f"md:1m:{ticker}")
    counts = pipe.execute()
    floor = max(1, int(min_bars))
    return [ticker for ticker, count in zip(symbols, counts) if int(count or 0) < floor]


def _normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    result = frame.copy()
    result.columns = [str(column).title() for column in result.columns]
    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column not in result:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce")
    index = pd.to_datetime(result.index, utc=True, errors="coerce")
    result.index = index
    result = result[~result.index.isna()]
    result = result[(result[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    # A zero-volume row produced by quote polling is not a completed trade bar.
    # Excluding it also repairs already-persisted weekend/off-hours pollution.
    result = result[result["Volume"] > 0]
    return result[["Open", "High", "Low", "Close", "Volume"]].sort_index()


def _row_payload(index: Any, row: pd.Series) -> dict[str, float | int] | None:
    try:
        timestamp = pd.Timestamp(index)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        values = {name: float(row[name]) for name in ("Open", "High", "Low", "Close", "Volume")}
    except (KeyError, TypeError, ValueError):
        return None
    if any(values[name] <= 0 for name in ("Open", "High", "Low", "Close")):
        return None
    return {
        "time_ms": int(timestamp.timestamp() * 1000),
        "open": values["Open"],
        "high": values["High"],
        "low": values["Low"],
        "close": values["Close"],
        "volume": max(0.0, values["Volume"]),
    }
