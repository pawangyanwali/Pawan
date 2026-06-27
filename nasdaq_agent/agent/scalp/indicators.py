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

    delta = close.diff()
    for period in (14, 7, 2):
        gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = gain / loss.replace(0.0, float("nan"))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
        rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)
        result[f"rsi_{period}"] = rsi
        result[f"rsi_avg_gain_{period}"] = gain
        result[f"rsi_avg_loss_{period}"] = loss

    fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    result["macd_fast_ema"] = fast
    result["macd_slow_ema"] = slow
    result["macd_signal_ema"] = signal
    result["macd_hist"] = macd - signal

    prior_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prior_close).abs(), (low - prior_close).abs()],
        axis=1,
    ).max(axis=1)
    result["atr_14"] = true_range.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()

    typical = (high + low + close) / 3.0
    local_index = result.index
    if getattr(local_index, "tz", None) is None:
        local_index = local_index.tz_localize("UTC")
    local_index = local_index.tz_convert("America/New_York")
    session_keys = pd.Series(
        [f"{stamp.date()}:{_volume_session(stamp.hour * 60 + stamp.minute)}" for stamp in local_index],
        index=result.index,
    )
    cumulative_volume = volume.groupby(session_keys).cumsum()
    result["vwap"] = (typical * volume).groupby(session_keys).cumsum() / cumulative_volume.replace(0.0, float("nan"))
    baseline = volume.groupby(session_keys).transform(
        lambda values: values.shift(1).rolling(20, min_periods=10).mean()
    )
    result["vol_ratio"] = volume / baseline.replace(0.0, float("nan"))
    return result


def _volume_session(minute_of_day: int) -> str:
    if 4 * 60 <= minute_of_day < 9 * 60 + 30:
        return "PRE_MARKET"
    if 9 * 60 + 30 <= minute_of_day < 16 * 60:
        return "REGULAR"
    if 16 * 60 <= minute_of_day <= 20 * 60:
        return "AFTER_HOURS"
    return "CLOSED"


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
        indicator_close=value("Close"),
        rsi_avg_gain_14=value("rsi_avg_gain_14"),
        rsi_avg_loss_14=value("rsi_avg_loss_14"),
        rsi_avg_gain_7=value("rsi_avg_gain_7"),
        rsi_avg_loss_7=value("rsi_avg_loss_7"),
        rsi_avg_gain_2=value("rsi_avg_gain_2"),
        rsi_avg_loss_2=value("rsi_avg_loss_2"),
        macd_fast_ema=value("macd_fast_ema"),
        macd_slow_ema=value("macd_slow_ema"),
        macd_signal_ema=value("macd_signal_ema"),
    )


def provisional_live_indicators(state: Any, live_price: object) -> dict[str, float] | None:
    """Append one transient quote to the closed-bar EMA state without mutating it."""
    try:
        price = float(live_price)
    except (TypeError, ValueError):
        return None
    close = _state_number(state, "indicator_close")
    if not finite(price) or price <= 0 or not finite(close):
        return None

    result: dict[str, float] = {}
    delta = price - float(close)
    for period in (14, 7, 2):
        gain = _state_number(state, f"rsi_avg_gain_{period}")
        loss = _state_number(state, f"rsi_avg_loss_{period}")
        if not finite(gain) or not finite(loss):
            return None
        alpha = 1.0 / period
        next_gain = alpha * max(delta, 0.0) + (1.0 - alpha) * float(gain)
        next_loss = alpha * max(-delta, 0.0) + (1.0 - alpha) * float(loss)
        if next_loss == 0.0:
            rsi = 100.0 if next_gain > 0.0 else 50.0
        else:
            rsi = 100.0 - (100.0 / (1.0 + next_gain / next_loss))
        result[f"rsi_{period}"] = round(rsi, 6)

    fast = _state_number(state, "macd_fast_ema")
    slow = _state_number(state, "macd_slow_ema")
    signal = _state_number(state, "macd_signal_ema")
    prior_hist = _state_number(state, "macd_hist")
    if not all(finite(value) for value in (fast, slow, signal, prior_hist)):
        return None
    next_fast = (2.0 / 13.0) * price + (11.0 / 13.0) * float(fast)
    next_slow = (2.0 / 27.0) * price + (25.0 / 27.0) * float(slow)
    next_macd = next_fast - next_slow
    next_signal = (2.0 / 10.0) * next_macd + (8.0 / 10.0) * float(signal)
    next_hist = next_macd - next_signal
    result.update(
        macd_hist=round(next_hist, 8),
        macd_slope=round(next_hist - float(prior_hist), 8),
    )
    return result


def _state_number(state: Any, name: str) -> float | None:
    raw = state.get(name) if isinstance(state, dict) else getattr(state, name, None)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if finite(value) else None


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
