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
        pipe = client.pipeline(transaction=False)
        for ticker in symbols:
            pipe.lrange(f"md:1m:{ticker}", -max(35, int(limit)), -1)
        payloads = pipe.execute()
    except Exception as exc:
        logger.warning("[ScalpBars] batched Valkey read failed: %s", exc)
        return {}, {ticker: "BAR_FEED_READ_FAILED" for ticker in symbols}

    for ticker, payload in zip(symbols, payloads):
        try:
            frame = _frame_from_payload(payload)
            if frame.empty:
                errors[ticker] = "ONE_MINUTE_BARS_MISSING"
            else:
                frames[ticker] = frame
        except Exception as exc:
            logger.debug("[ScalpBars] %s parse failed: %s", ticker, exc)
            errors[ticker] = "ONE_MINUTE_BARS_INVALID"
    return frames, errors


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
