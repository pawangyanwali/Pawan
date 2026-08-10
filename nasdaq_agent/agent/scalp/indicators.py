from __future__ import annotations

import time
from dataclasses import replace
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
    session_names = pd.Series(
        [_volume_session(stamp.hour * 60 + stamp.minute) for stamp in local_index],
        index=result.index,
    )
    session_keys = pd.Series(
        [f"{stamp.date()}:{name}" for stamp, name in zip(local_index, session_names)],
        index=result.index,
    )
    cumulative_volume = volume.groupby(session_keys).cumsum()
    result["vwap"] = (typical * volume).groupby(session_keys).cumsum() / cumulative_volume.replace(0.0, float("nan"))
    # The signal contract consumes only the latest RVOL value. Calculate that
    # decision point directly instead of materialising hundreds of tiny
    # groupby/rolling windows for every historical row and every ticker.
    # Prefer the same minute-of-session across prior trading days. This keeps
    # the regular-session U-shaped volume curve from making the open look
    # artificially hot and midday look artificially weak.
    minute_of_day = pd.Series(
        [stamp.hour * 60 + stamp.minute for stamp in local_index],
        index=result.index,
    )
    latest_session = str(session_names.iloc[-1])
    latest_minute = int(minute_of_day.iloc[-1])
    prior = volume.iloc[:-1]
    profile_values = prior[
        (session_names.iloc[:-1] == latest_session)
        & (minute_of_day.iloc[:-1] == latest_minute)
    ].tail(10)
    baseline_value = (
        float(profile_values.median()) if len(profile_values) >= 2 else float("nan")
    )
    if not finite(baseline_value) or baseline_value <= 0:
        session_values = prior[session_names.iloc[:-1] == latest_session].tail(20)
        baseline_value = (
            float(session_values.median()) if len(session_values) >= 10 else float("nan")
        )
    # Sparse PM/AH symbols may not have ten prints in the current session.
    # Prior real traded bars are conservative because regular-session volume
    # normally makes the resulting extended-hours RVOL smaller, not overstated.
    if (
        (not finite(baseline_value) or baseline_value <= 0)
        and latest_session in {"PRE_MARKET", "AFTER_HOURS"}
    ):
        fallback_values = prior.tail(20)
        baseline_value = (
            float(fallback_values.median()) if len(fallback_values) >= 10 else float("nan")
        )
    result["vol_ratio"] = float("nan")
    if finite(baseline_value) and baseline_value > 0:
        result.loc[result.index[-1], "vol_ratio"] = (
            float(volume.iloc[-1]) / baseline_value
        )
    return result


def update_one_minute_indicators(previous: Any, frame: Any) -> Any:
    """Continue a warmed indicator frame using only changed/new 1m bars.

    The full historical frame remains available for RVOL profiling, while the
    recursive indicators advance from their prior EWM state.  A mutable latest
    bar is rolled back one row and recalculated, so quote/bar corrections are
    not hidden by a timestamp-only cache.
    """
    if frame is None or len(frame) == 0:
        return frame
    if previous is None or len(previous) < 2:
        return calculate_one_minute_indicators(frame)
    required_state = {
        "rsi_avg_gain_14", "rsi_avg_loss_14", "rsi_avg_gain_7",
        "rsi_avg_loss_7", "rsi_avg_gain_2", "rsi_avg_loss_2",
        "macd_fast_ema", "macd_slow_ema", "macd_signal_ema", "atr_14",
    }
    if not required_state.issubset(previous.columns):
        return calculate_one_minute_indicators(frame)

    source = frame.sort_index()
    old = previous.sort_index()
    last_old = old.index[-1]
    if source.index[-1] < last_old:
        return calculate_one_minute_indicators(source)

    # Recalculate the current bar if its OHLCV payload changed in place.
    if source.index[-1] == last_old:
        raw_columns = ["Open", "High", "Low", "Close", "Volume"]
        if all(
            _same_number(source.iloc[-1].get(column), old.iloc[-1].get(column))
            for column in raw_columns
        ):
            return old.tail(len(source))
        stable = old.iloc[:-1]
        pending = source.loc[source.index >= last_old]
    else:
        stable = old
        pending = source.loc[source.index > last_old]

    if stable.empty or pending.empty:
        return calculate_one_minute_indicators(source)
    result = stable.copy()
    for stamp, raw in pending.iterrows():
        prior = result.iloc[-1]
        row = {column: raw.get(column) for column in source.columns}
        close = float(raw["Close"])
        prior_close = float(prior["Close"])
        delta = close - prior_close
        for period in (14, 7, 2):
            prior_gain = float(prior[f"rsi_avg_gain_{period}"])
            prior_loss = float(prior[f"rsi_avg_loss_{period}"])
            if not finite(prior_gain) or not finite(prior_loss):
                return calculate_one_minute_indicators(source)
            alpha = 1.0 / period
            gain = alpha * max(delta, 0.0) + (1.0 - alpha) * prior_gain
            loss = alpha * max(-delta, 0.0) + (1.0 - alpha) * prior_loss
            row[f"rsi_avg_gain_{period}"] = gain
            row[f"rsi_avg_loss_{period}"] = loss
            row[f"rsi_{period}"] = (
                100.0 if loss == 0.0 and gain > 0.0
                else 50.0 if loss == 0.0
                else 100.0 - (100.0 / (1.0 + gain / loss))
            )

        fast = (2.0 / 13.0) * close + (11.0 / 13.0) * float(prior["macd_fast_ema"])
        slow = (2.0 / 27.0) * close + (25.0 / 27.0) * float(prior["macd_slow_ema"])
        macd = fast - slow
        signal = (2.0 / 10.0) * macd + (8.0 / 10.0) * float(prior["macd_signal_ema"])
        row.update(
            macd_fast_ema=fast,
            macd_slow_ema=slow,
            macd_signal_ema=signal,
            macd_hist=macd - signal,
        )
        true_range = max(
            abs(float(raw["High"]) - float(raw["Low"])),
            abs(float(raw["High"]) - prior_close),
            abs(float(raw["Low"]) - prior_close),
        )
        row["atr_14"] = (
            true_range / 14.0 + (13.0 / 14.0) * float(prior["atr_14"])
        )
        result.loc[stamp] = row

    result = result[~result.index.duplicated(keep="last")].sort_index()
    _refresh_session_vwap(result, source, pending.index)
    _refresh_latest_rvol(result, source)
    # RSI/MACD/ATR are recursive. Two rows are sufficient to advance the next
    # closed bar and to recalculate an in-place correction of the latest bar.
    return result.tail(2)


def _same_number(left: object, right: object) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-12
    except (TypeError, ValueError):
        return left == right


def _session_parts(index: Any) -> tuple[pd.DatetimeIndex, pd.Series, pd.Series]:
    local = index
    if getattr(local, "tz", None) is None:
        local = local.tz_localize("UTC")
    local = local.tz_convert("America/New_York")
    names = pd.Series(
        [_volume_session(stamp.hour * 60 + stamp.minute) for stamp in local],
        index=index,
    )
    keys = pd.Series(
        [f"{stamp.date()}:{name}" for stamp, name in zip(local, names)],
        index=index,
    )
    return local, names, keys


def _refresh_session_vwap(result: pd.DataFrame, source: pd.DataFrame, changed: Any) -> None:
    close = pd.to_numeric(source["Close"], errors="coerce")
    high = pd.to_numeric(source["High"], errors="coerce")
    low = pd.to_numeric(source["Low"], errors="coerce")
    volume = pd.to_numeric(source["Volume"], errors="coerce").fillna(0.0)
    typical = (high + low + close) / 3.0
    for stamp in changed:
        local_stamp = stamp
        if local_stamp.tzinfo is None:
            local_stamp = local_stamp.tz_localize("UTC")
        local_stamp = local_stamp.tz_convert("America/New_York")
        minute = local_stamp.hour * 60 + local_stamp.minute
        session = _volume_session(minute)
        start_minute = {
            "PRE_MARKET": 4 * 60,
            "REGULAR": 9 * 60 + 30,
            "AFTER_HOURS": 16 * 60,
        }.get(session, 0)
        start_local = local_stamp.normalize() + pd.Timedelta(minutes=start_minute)
        start = start_local.tz_convert(source.index.tz or "UTC")
        mask = (source.index >= start) & (source.index <= stamp)
        total_volume = float(volume.loc[mask].sum())
        result.loc[stamp, "vwap"] = (
            float((typical.loc[mask] * volume.loc[mask]).sum()) / total_volume
            if total_volume > 0 else float("nan")
        )


def _refresh_latest_rvol(result: pd.DataFrame, source: pd.DataFrame) -> None:
    index = source.index
    if getattr(index, "tz", None) is None:
        index = index.tz_localize("UTC")
    local = index.tz_convert("America/New_York")
    minutes = local.hour * 60 + local.minute
    latest_minute = int(minutes[-1])
    latest_session = _volume_session(latest_minute)
    if latest_session == "PRE_MARKET":
        session_mask = (minutes >= 4 * 60) & (minutes < 9 * 60 + 30)
    elif latest_session == "REGULAR":
        session_mask = (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
    elif latest_session == "AFTER_HOURS":
        session_mask = (minutes >= 16 * 60) & (minutes <= 20 * 60)
    else:
        session_mask = (minutes < 4 * 60) | (minutes > 20 * 60)
    volume = pd.to_numeric(source["Volume"], errors="coerce").fillna(0.0)
    prior = volume.iloc[:-1]
    profile = prior[(session_mask[:-1]) & (minutes[:-1] == latest_minute)].tail(10)
    baseline = float(profile.median()) if len(profile) >= 2 else float("nan")
    if not finite(baseline) or baseline <= 0:
        values = prior[session_mask[:-1]].tail(20)
        baseline = float(values.median()) if len(values) >= 10 else float("nan")
    if (
        (not finite(baseline) or baseline <= 0)
        and latest_session in {"PRE_MARKET", "AFTER_HOURS"}
    ):
        values = prior.tail(20)
        baseline = float(values.median()) if len(values) >= 10 else float("nan")
    result["vol_ratio"] = float("nan")
    if finite(baseline) and baseline > 0:
        result.loc[result.index[-1], "vol_ratio"] = float(volume.iloc[-1]) / baseline


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
    bar_close_offset_ms: int = 0,
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
        bar_age_ms=_frame_bar_age_ms(
            frame,
            now_ms=now_ms,
            bar_close_offset_ms=bar_close_offset_ms,
        ),
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


def refresh_indicator_bar_age(
    snapshot: IndicatorSnapshot,
    frame: Any,
    *,
    now_ms: int | None = None,
) -> IndicatorSnapshot:
    """Refresh cached 1m age without recalculating or re-extracting indicators."""
    return replace(
        snapshot,
        bar_age_ms=_frame_bar_age_ms(frame, now_ms=now_ms),
    )


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


def _frame_bar_age_ms(
    frame: Any,
    *,
    now_ms: int | None,
    bar_close_offset_ms: int = 0,
) -> int | None:
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
    return current_ms - timestamp_ms - max(0, int(bar_close_offset_ms))
