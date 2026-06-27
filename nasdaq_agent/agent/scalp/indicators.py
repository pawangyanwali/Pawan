from __future__ import annotations

import time
from typing import Any

import pandas as pd

from ._utils import finite
from .models import IndicatorSnapshot


def calculate_one_minute_indicators(frame: Any) -> Any:
    """Calculate the complete scalp indicator contract from closed 1m bars."""
    if frame is None or len(frame) == 0:
        return frame
    result = frame.copy()
    close = pd.to_numeric(result["Close"], errors="coerce")
    high = pd.to_numeric(result["High"], errors="coerce")
    low = pd.to_numeric(result["Low"], errors="coerce")
    volume = pd.to_numeric(result["Volume"], errors="coerce").fillna(0.0)

    for period in (14, 7, 2):
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = gain / loss.replace(0.0, float("nan"))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
        rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)
        result[f"rsi_{period}"] = rsi

    fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    result["macd_hist"] = macd - signal

    prior_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prior_close).abs(), (low - prior_close).abs()],
        axis=1,
    ).max(axis=1)
    result["atr_14"] = true_range.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()

    typical = (high + low + close) / 3.0
    local_dates = result.index.tz_convert("America/New_York").date
    cumulative_volume = volume.groupby(local_dates).cumsum()
    result["vwap"] = (typical * volume).groupby(local_dates).cumsum() / cumulative_volume.replace(0.0, float("nan"))
    baseline = volume.shift(1).rolling(20, min_periods=10).mean()
    result["vol_ratio"] = volume / baseline.replace(0.0, float("nan"))
    return result


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
