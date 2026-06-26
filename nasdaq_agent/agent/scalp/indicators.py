from __future__ import annotations

import time
from typing import Any

from ._utils import finite
from .models import IndicatorSnapshot


def indicator_snapshot_from_frame(
    frame: Any,
    *,
    now_ms: int | None = None,
) -> IndicatorSnapshot:
    """Extract final-bar values without substituting neutral defaults."""
    if frame is None or len(frame) < 35:
        return IndicatorSnapshot(None, None, None, None, None, None, None, None)

    row = frame.iloc[-1]
    previous = frame.iloc[-2]

    def value(column: str, *, prior: bool = False) -> float | None:
        source = previous if prior else row
        if column not in frame.columns:
            return None
        raw = source[column]
        return float(raw) if finite(raw) else None

    macd_prev = value("macd_hist_prev")
    if macd_prev is None:
        macd_prev = value("macd_hist", prior=True)

    return IndicatorSnapshot(
        rsi_14=value("rsi_14"),
        rsi_7=value("rsi_7"),
        rsi_2=value("rsi_2"),
        macd_hist=value("macd_hist"),
        macd_hist_prev=macd_prev,
        atr_14=value("atr_14"),
        vwap=value("vwap"),
        rvol=value("vol_ratio"),
        vwap_event=_derive_vwap_event(frame),
        bar_age_ms=_frame_bar_age_ms(frame, now_ms=now_ms),
    )


def _derive_vwap_event(frame: Any) -> str:
    if not {"Close", "vwap"}.issubset(set(frame.columns)) or len(frame) < 2:
        return ""
    price = frame["Close"].iloc[-1]
    prior_price = frame["Close"].iloc[-2]
    vwap = frame["vwap"].iloc[-1]
    prior_vwap = frame["vwap"].iloc[-2]
    if not all(finite(v) for v in (price, prior_price, vwap, prior_vwap)):
        return ""
    if prior_price < prior_vwap and price >= vwap:
        return "RECLAIM"
    if prior_price > prior_vwap and price <= vwap:
        return "REJECTION"
    return "ABOVE" if price > vwap else "BELOW" if price < vwap else "AT_VWAP"


def _frame_bar_age_ms(frame: Any, *, now_ms: int | None) -> int | None:
    if frame is None or len(frame) == 0:
        return None
    last_index = frame.index[-1]
    if getattr(last_index, "tzinfo", None) is None:
        return None
    try:
        timestamp_ms = int(last_index.timestamp() * 1000)
    except (AttributeError, OSError, OverflowError, TypeError, ValueError):
        return None
    current_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    return current_ms - timestamp_ms

