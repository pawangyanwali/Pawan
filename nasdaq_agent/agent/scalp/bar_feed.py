"""Batched one-minute bar reader for the scalp runtime.

The market-data service owns writes to ``md:1m:{ticker}``.  The scalp engine
reads those lists in one Valkey pipeline so universe coverage does not create
one network round-trip per ticker and never calls a broker API from analysis.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_frame_cache: dict[str, tuple[bytes | str | None, int, int, pd.DataFrame]] = {}


def load_one_minute_frames(
    tickers: Iterable[str], *, limit: int = 120
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Return chronological OHLCV frames and explicit per-ticker errors."""
    symbols = [str(ticker).upper() for ticker in tickers]
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    try:
        from agent.valkey_client import _get_client

        client = _get_client()
        if client is None:
            return {}, {ticker: "VALKEY_UNAVAILABLE" for ticker in symbols}
        probe = client.pipeline(transaction=False)
        for ticker in symbols:
            key = f"md:1m:{ticker}"
            probe.lindex(key, -1)
            probe.llen(key)
        probe_values = probe.execute()
    except Exception as exc:
        logger.warning("[ScalpBars] batched Valkey read failed: %s", exc)
        return {}, {ticker: "BAR_FEED_READ_FAILED" for ticker in symbols}

    requested_limit = max(35, int(limit))
    changed: list[str] = []
    fetch_counts: dict[str, int] = {}
    metadata: dict[str, tuple[bytes | str | None, int]] = {}
    for index, ticker in enumerate(symbols):
        last_raw = probe_values[index * 2]
        row_count = int(probe_values[index * 2 + 1] or 0)
        metadata[ticker] = (last_raw, row_count)
        cached = _frame_cache.get(ticker)
        if (
            cached
            and cached[0] == last_raw
            and cached[1] == row_count
            and cached[2] == requested_limit
        ):
            frames[ticker] = cached[3]
        else:
            changed.append(ticker)
            fetch_counts[ticker] = _delta_fetch_count(
                cached, last_raw, requested_limit
            )

    payload_by_ticker: dict[str, list[Any]] = {}
    if changed:
        try:
            pipe = client.pipeline(transaction=False)
            for ticker in changed:
                pipe.lrange(
                    f"md:1m:{ticker}", -fetch_counts[ticker], -1
                )
            payload_by_ticker = dict(zip(changed, pipe.execute()))
        except Exception as exc:
            logger.warning("[ScalpBars] changed-frame read failed: %s", exc)
            errors.update({ticker: "BAR_FEED_READ_FAILED" for ticker in changed})

    for ticker in changed:
        payload = payload_by_ticker.get(ticker)
        if payload is None:
            continue
        try:
            delta = _frame_from_payload(payload)
            cached = _frame_cache.get(ticker)
            frame = (
                _merge_frame(cached[3], delta, requested_limit)
                if cached and fetch_counts[ticker] < requested_limit
                else delta.tail(requested_limit)
            )
            if frame.empty:
                errors[ticker] = "ONE_MINUTE_BARS_MISSING"
            else:
                frames[ticker] = frame
                last_raw, row_count = metadata[ticker]
                _frame_cache[ticker] = (
                    last_raw,
                    row_count,
                    requested_limit,
                    frame,
                )
        except Exception as exc:
            logger.debug("[ScalpBars] %s parse failed: %s", ticker, exc)
            errors[ticker] = "ONE_MINUTE_BARS_INVALID"

    keep = set(symbols)
    for ticker in tuple(_frame_cache):
        if ticker not in keep:
            _frame_cache.pop(ticker, None)
    return frames, errors


def _delta_fetch_count(
    cached: tuple[bytes | str | None, int, int, pd.DataFrame] | None,
    last_raw: bytes | str | None,
    requested_limit: int,
) -> int:
    if not cached or cached[2] != requested_limit or cached[3].empty:
        return requested_limit
    latest = _payload_timestamp(last_raw)
    cached_latest = cached[3].index[-1]
    if latest is None or cached_latest.tzinfo is None or latest < cached_latest:
        return requested_limit
    gap = max(0, int((latest - cached_latest).total_seconds() // 60))
    return min(requested_limit, max(2, gap + 2))


def _payload_timestamp(raw: bytes | str | None) -> pd.Timestamp | None:
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        item = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        value = item.get("time_ms") or item.get("datetime") or item.get("timestamp")
        if value is None:
            return None
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            unit = "ms" if abs(float(numeric)) > 10_000_000_000 else "s"
            stamp = pd.to_datetime(numeric, unit=unit, utc=True)
        else:
            stamp = pd.to_datetime(value, utc=True, errors="coerce")
        return None if pd.isna(stamp) else stamp
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _merge_frame(
    cached: pd.DataFrame, delta: pd.DataFrame, requested_limit: int
) -> pd.DataFrame:
    if delta.empty:
        return cached.tail(requested_limit)
    merged = pd.concat([cached, delta])
    return (
        merged[~merged.index.duplicated(keep="last")]
        .sort_index()
        .tail(requested_limit)
    )


def _frame_from_payload(payload: list[Any] | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw in payload or []:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        item = json.loads(raw) if isinstance(raw, str) else dict(raw)
        timestamp = item.get("time_ms") or item.get("datetime") or item.get("timestamp")
        if timestamp is None:
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "Open": _number(item.get("open")),
                "High": _number(item.get("high")),
                "Low": _number(item.get("low")),
                "Close": _number(item.get("close")),
                "Volume": max(0.0, _number(item.get("volume"))),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    frame = pd.DataFrame(rows)
    numeric_ts = pd.to_numeric(frame["timestamp"], errors="coerce")
    if numeric_ts.notna().all():
        unit = "ms" if float(numeric_ts.abs().max()) > 10_000_000_000 else "s"
        index = pd.to_datetime(numeric_ts, unit=unit, utc=True, errors="coerce")
    else:
        index = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame.index = index
    frame = frame.drop(columns=["timestamp"])
    frame = frame[~frame.index.isna()]
    frame = frame[(frame[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    frame = frame[frame["Volume"] > 0]
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
