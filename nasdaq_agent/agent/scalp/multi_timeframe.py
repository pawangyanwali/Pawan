"""Completed-bar multi-timeframe context for the scalp-only strategy.

Five-minute data is context, never an execution clock. The canonical entry,
stop, targets, and live-data gates remain owned by live quotes and closed 1m
bars. Shadow assessments are deliberately isolated from plan validity.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from ._utils import finite
from .indicators import (
    calculate_one_minute_indicators,
    indicator_snapshot_from_frame,
)
from .models import (
    IndicatorSnapshot,
    MultiTimeframeSnapshot,
    ScalpSignalConfig,
    ScalpSignalPlan,
    SignalSide,
)


@dataclass(frozen=True)
class ShadowAssessment:
    family: str
    side: SignalSide
    ready: bool
    score: float
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]


@dataclass
class FiveMinuteIndicatorState:
    """Incremental closed-5m state retained by the live scalp runtime."""

    completed: pd.DataFrame
    enriched: pd.DataFrame
    snapshot: MultiTimeframeSnapshot


def completed_five_minute_bars(frame: Any) -> pd.DataFrame:
    """Aggregate only fully closed wall-clock 5m buckets from closed 1m bars."""
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    source = frame.sort_index().copy()
    if not isinstance(source.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    if source.index.tz is None:
        source.index = source.index.tz_localize("UTC")

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(source.columns):
        return pd.DataFrame(columns=sorted(required))

    grouped = source.resample("5min", label="left", closed="left").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    # A closed 1m bar timestamp is its minute start. Therefore the latest
    # observable instant is one minute after that timestamp.
    observable_through = source.index[-1] + pd.Timedelta(minutes=1)
    complete = grouped.index + pd.Timedelta(minutes=5) <= observable_through
    grouped = grouped.loc[complete].dropna(subset=["Open", "High", "Low", "Close"])
    return grouped[grouped["Volume"] > 0]


def completed_five_minute_bar_id(frame: Any) -> int:
    """Return the latest possible completed 5m bucket start in epoch ms."""
    if frame is None or len(frame) == 0 or not isinstance(frame.index, pd.DatetimeIndex):
        return 0
    last = frame.index[-1]
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    completed_end = (last + pd.Timedelta(minutes=1)).floor("5min")
    completed_start = completed_end - pd.Timedelta(minutes=5)
    return int(completed_start.timestamp() * 1000)


def five_minute_snapshot(
    frame: Any,
    *,
    now_ms: int | None = None,
    max_bar_age_ms: int = 420_000,
) -> MultiTimeframeSnapshot:
    return build_five_minute_state(
        frame,
        now_ms=now_ms,
        max_bar_age_ms=max_bar_age_ms,
    ).snapshot


def build_five_minute_state(
    frame: Any,
    *,
    now_ms: int | None = None,
    max_bar_age_ms: int = 420_000,
) -> FiveMinuteIndicatorState:
    """Warm the 5m state once; subsequent closed buckets advance incrementally."""
    completed = completed_five_minute_bars(frame)
    if len(completed) < 35:
        return FiveMinuteIndicatorState(
            completed=completed,
            enriched=pd.DataFrame(),
            snapshot=MultiTimeframeSnapshot(completed_bars=len(completed)),
        )

    enriched = calculate_one_minute_indicators(completed)
    close = pd.to_numeric(enriched["Close"], errors="coerce")
    enriched["mtf_ema_fast"] = close.ewm(span=5, adjust=False, min_periods=5).mean()
    enriched["mtf_ema_slow"] = close.ewm(span=13, adjust=False, min_periods=13).mean()
    snapshot = _snapshot_from_enriched(
        enriched,
        completed_bars=len(completed),
        now_ms=now_ms,
        max_bar_age_ms=max_bar_age_ms,
    )
    return FiveMinuteIndicatorState(
        completed=completed.tail(120).copy(),
        enriched=enriched.tail(2).copy(),
        snapshot=snapshot,
    )


def update_five_minute_state(
    state: FiveMinuteIndicatorState | None,
    frame: Any,
    *,
    now_ms: int | None = None,
    max_bar_age_ms: int = 420_000,
) -> FiveMinuteIndicatorState:
    """Advance only newly closed 5m buckets instead of resampling all symbols."""
    if state is None or state.completed.empty or state.enriched.empty:
        return build_five_minute_state(
            frame,
            now_ms=now_ms,
            max_bar_age_ms=max_bar_age_ms,
        )
    target_id = completed_five_minute_bar_id(frame)
    cached_id = int(state.completed.index[-1].timestamp() * 1000)
    if target_id <= cached_id:
        return replace(
            state,
            snapshot=refresh_five_minute_age(
                state.snapshot,
                max_bar_age_ms=max_bar_age_ms,
                now_ms=now_ms,
            ),
        )

    missing_buckets = max(1, (target_id - cached_id) // (5 * 60_000))
    if missing_buckets > 12:
        return build_five_minute_state(
            frame,
            now_ms=now_ms,
            max_bar_age_ms=max_bar_age_ms,
        )
    recent = (
        _latest_completed_five_minute_bar(frame, target_id)
        if missing_buckets == 1
        else completed_five_minute_bars(
            frame.tail(max(15, int(missing_buckets) * 5 + 5))
        )
    )
    pending = recent.loc[recent.index > state.completed.index[-1]]
    if pending.empty:
        return replace(
            state,
            snapshot=refresh_five_minute_age(
                state.snapshot,
                max_bar_age_ms=max_bar_age_ms,
                now_ms=now_ms,
            ),
        )

    completed = pd.concat([state.completed, pending])
    completed = (
        completed[~completed.index.duplicated(keep="last")]
        .sort_index()
        .tail(120)
    )
    enriched = state.enriched.copy()
    for stamp in pending.index:
        enriched = _advance_five_minute_indicators(
            enriched,
            completed.loc[completed.index <= stamp],
            stamp,
        )

    snapshot = _snapshot_from_enriched(
        enriched,
        completed_bars=len(completed),
        now_ms=now_ms,
        max_bar_age_ms=max_bar_age_ms,
    )
    return FiveMinuteIndicatorState(completed, enriched.tail(2).copy(), snapshot)


def _advance_five_minute_indicators(
    enriched: pd.DataFrame,
    completed: pd.DataFrame,
    stamp: pd.Timestamp,
) -> pd.DataFrame:
    """Advance the exact 5m indicator contract with scalar recursive math."""
    prior = enriched.iloc[-1]
    raw = completed.loc[stamp]
    row = {column: raw.get(column) for column in completed.columns}
    close = float(raw["Close"])
    prior_close = float(prior["Close"])
    delta = close - prior_close
    for period in (14, 7, 2):
        prior_gain = float(prior[f"rsi_avg_gain_{period}"])
        prior_loss = float(prior[f"rsi_avg_loss_{period}"])
        if not finite(prior_gain) or not finite(prior_loss):
            return _warm_five_minute_indicators(completed)
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
    row["atr_14"] = true_range / 14.0 + (13.0 / 14.0) * float(prior["atr_14"])
    row["vwap"] = _completed_five_minute_vwap(completed, stamp)
    row["vol_ratio"] = float("nan")
    row["mtf_ema_fast"] = (
        (2.0 / 6.0) * close + (4.0 / 6.0) * float(prior["mtf_ema_fast"])
    )
    row["mtf_ema_slow"] = (
        (2.0 / 14.0) * close + (12.0 / 14.0) * float(prior["mtf_ema_slow"])
    )
    return pd.DataFrame(
        [prior.to_dict(), row],
        index=pd.DatetimeIndex([enriched.index[-1], stamp]),
    )


def _warm_five_minute_indicators(completed: pd.DataFrame) -> pd.DataFrame:
    enriched = calculate_one_minute_indicators(completed)
    close = pd.to_numeric(enriched["Close"], errors="coerce")
    enriched["mtf_ema_fast"] = close.ewm(span=5, adjust=False, min_periods=5).mean()
    enriched["mtf_ema_slow"] = close.ewm(span=13, adjust=False, min_periods=13).mean()
    return enriched.tail(2).copy()


def _completed_five_minute_vwap(
    completed: pd.DataFrame,
    stamp: pd.Timestamp,
) -> float:
    local = stamp
    if local.tzinfo is None:
        local = local.tz_localize("UTC")
    local = local.tz_convert("America/New_York")
    minute = local.hour * 60 + local.minute
    start_minute = (
        4 * 60 if 4 * 60 <= minute < 9 * 60 + 30
        else 9 * 60 + 30 if 9 * 60 + 30 <= minute < 16 * 60
        else 16 * 60 if 16 * 60 <= minute <= 20 * 60
        else 0
    )
    start = (local.normalize() + pd.Timedelta(minutes=start_minute)).tz_convert(
        completed.index.tz or "UTC"
    )
    window = completed.loc[(completed.index >= start) & (completed.index <= stamp)]
    volume = pd.to_numeric(window["Volume"], errors="coerce").fillna(0.0)
    total_volume = float(volume.sum())
    if total_volume <= 0:
        return float("nan")
    typical = (
        pd.to_numeric(window["High"], errors="coerce")
        + pd.to_numeric(window["Low"], errors="coerce")
        + pd.to_numeric(window["Close"], errors="coerce")
    ) / 3.0
    return float((typical * volume).sum() / total_volume)


def _latest_completed_five_minute_bar(
    frame: pd.DataFrame,
    target_id: int,
) -> pd.DataFrame:
    """Aggregate the single newly closed bucket without a pandas resample."""
    if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    index = frame.index
    start = pd.to_datetime(target_id, unit="ms", utc=True)
    if index.tz is None:
        start = start.tz_localize(None)
    else:
        start = start.tz_convert(index.tz)
    end = start + pd.Timedelta(minutes=5)
    bucket = frame.loc[(index >= start) & (index < end)]
    if bucket.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    volume = float(pd.to_numeric(bucket["Volume"], errors="coerce").fillna(0.0).sum())
    if volume <= 0:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    return pd.DataFrame(
        [{
            "Open": float(bucket.iloc[0]["Open"]),
            "High": float(pd.to_numeric(bucket["High"], errors="coerce").max()),
            "Low": float(pd.to_numeric(bucket["Low"], errors="coerce").min()),
            "Close": float(bucket.iloc[-1]["Close"]),
            "Volume": volume,
        }],
        index=pd.DatetimeIndex([start]),
    )


def _snapshot_from_enriched(
    enriched: pd.DataFrame,
    *,
    completed_bars: int,
    now_ms: int | None,
    max_bar_age_ms: int,
) -> MultiTimeframeSnapshot:
    base = indicator_snapshot_from_frame(
        enriched,
        now_ms=now_ms,
        bar_close_offset_ms=5 * 60_000,
    )
    row = enriched.iloc[-1]

    def value(name: str) -> float | None:
        raw = row.get(name)
        return float(raw) if finite(raw) else None

    ema_fast = value("mtf_ema_fast")
    ema_slow = value("mtf_ema_slow")
    state = _context_state(
        close=value("Close"),
        vwap=base.vwap,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        macd_slope=base.macd_slope,
        bar_age_ms=base.bar_age_ms,
        max_bar_age_ms=max_bar_age_ms,
    )
    return MultiTimeframeSnapshot(
        state=state,
        close=value("Close"),
        rsi_14=base.rsi_14,
        macd_hist=base.macd_hist,
        macd_hist_prev=base.macd_hist_prev,
        macd_slope=base.macd_slope,
        atr_14=base.atr_14,
        vwap=base.vwap,
        vwap_event=base.vwap_event,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        bar_age_ms=base.bar_age_ms,
        bar_closed_at_ms=int(
            (enriched.index[-1] + pd.Timedelta(minutes=5)).timestamp() * 1000
        ),
        completed_bars=completed_bars,
    )


def refresh_five_minute_age(
    snapshot: MultiTimeframeSnapshot,
    *,
    max_bar_age_ms: int,
    now_ms: int | None = None,
) -> MultiTimeframeSnapshot:
    """Refresh cached context age so a frozen feed cannot remain tradable."""
    if snapshot.bar_closed_at_ms is None:
        return snapshot
    current_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    age = current_ms - int(snapshot.bar_closed_at_ms)
    state = _context_state(
        close=snapshot.close,
        vwap=snapshot.vwap,
        ema_fast=snapshot.ema_fast,
        ema_slow=snapshot.ema_slow,
        macd_slope=snapshot.macd_slope,
        bar_age_ms=age,
        max_bar_age_ms=max_bar_age_ms,
    )
    return replace(snapshot, state=state, bar_age_ms=age)


def apply_multi_timeframe_shadow(
    plan: ScalpSignalPlan,
    one_minute: IndicatorSnapshot,
    five_minute: MultiTimeframeSnapshot,
    config: ScalpSignalConfig,
    *,
    session: str,
) -> ScalpSignalPlan:
    """Attach MTF evidence without changing canonical execution fields."""
    plan.strategy_family = "REVERSAL" if plan.side is not SignalSide.NONE else "NONE"
    plan.mtf_mode = "OFF" if not config.mtf_enabled else str(config.mtf_mode).upper()
    plan.mtf_state = five_minute.state
    plan.mtf_alignment = _alignment(plan.side, five_minute.state)
    plan.mtf_bar_age_ms = (
        int(five_minute.bar_age_ms) if five_minute.bar_age_ms is not None else -1
    )
    plan.rsi_14_5m = _number(five_minute.rsi_14)
    plan.macd_hist_5m = _number(five_minute.macd_hist)
    plan.macd_hist_prev_5m = _number(five_minute.macd_hist_prev)
    plan.macd_slope_5m = _number(five_minute.macd_slope)
    plan.atr_14_5m = _number(five_minute.atr_14)
    plan.vwap_5m = _number(five_minute.vwap)
    plan.vwap_event_5m = str(five_minute.vwap_event or "")
    plan.ema_fast_5m = _number(five_minute.ema_fast)
    plan.ema_slow_5m = _number(five_minute.ema_slow)

    if plan.mtf_mode != "SHADOW":
        return plan

    assessments: list[ShadowAssessment] = []
    for side in (SignalSide.LONG, SignalSide.SHORT):
        if config.momentum_shadow_enabled:
            assessments.append(
                _momentum_assessment(
                    side, one_minute, five_minute, config, session=session
                )
            )
        if config.reversal_shadow_enabled:
            assessments.append(
                _reversal_assessment(side, one_minute, five_minute, config)
            )
    if not assessments:
        return plan
    selected = max(
        assessments,
        key=lambda item: (item.ready, item.score, item.family == "MOMENTUM_PULLBACK"),
    )
    plan.shadow_strategy_family = selected.family
    plan.shadow_side = selected.side.value
    plan.shadow_setup_ready = selected.ready
    plan.shadow_setup_score = selected.score
    plan.shadow_reasons = list(selected.reasons)
    plan.shadow_blockers = list(selected.blockers)
    return plan


def _momentum_assessment(
    side: SignalSide,
    one: IndicatorSnapshot,
    five: MultiTimeframeSnapshot,
    config: ScalpSignalConfig,
    *,
    session: str,
) -> ShadowAssessment:
    reasons: list[str] = []
    blockers: list[str] = []
    alignment = _alignment(side, five.state)
    if five.state == "NO_DATA":
        blockers.append("MTF_5M_DATA_MISSING_OR_STALE")
    elif alignment != "ALIGNED":
        blockers.append("MTF_5M_TREND_NOT_ALIGNED")
    else:
        reasons.append("MTF_5M_TREND_ALIGNED")

    rsi = _number(one.rsi_14)
    if side is SignalSide.LONG:
        rsi_ok = config.momentum_long_rsi_min <= rsi <= config.momentum_long_rsi_max
        macd_ok = finite(one.macd_slope) and float(one.macd_slope) > 0
        vwap_ok = str(one.vwap_event or "").upper() in {"RECLAIM", "ABOVE"}
    else:
        rsi_ok = config.momentum_short_rsi_min <= rsi <= config.momentum_short_rsi_max
        macd_ok = finite(one.macd_slope) and float(one.macd_slope) < 0
        vwap_ok = str(one.vwap_event or "").upper() in {"REJECTION", "BELOW"}
    _record(rsi_ok, "ONE_MIN_RSI_PULLBACK", "ONE_MIN_RSI_NOT_IN_PULLBACK_ZONE", reasons, blockers)
    _record(macd_ok, "ONE_MIN_MACD_REACCELERATION", "ONE_MIN_MACD_NOT_REACCELERATING", reasons, blockers)
    _record(vwap_ok, "ONE_MIN_VWAP_DIRECTION_CONFIRMED", "ONE_MIN_VWAP_DIRECTION_MISSING", reasons, blockers)

    minimum_rvol = (
        config.min_rvol_extended
        if str(session).upper() in {"PRE_MARKET", "AFTER_HOURS", "EXTENDED"}
        else config.min_rvol_regular
    )
    rvol_ok = finite(one.rvol) and float(one.rvol) >= minimum_rvol
    _record(rvol_ok, "ONE_MIN_RVOL_CONFIRMED", "ONE_MIN_RVOL_TOO_LOW", reasons, blockers)
    return _assessment("MOMENTUM_PULLBACK", side, reasons, blockers)


def _reversal_assessment(
    side: SignalSide,
    one: IndicatorSnapshot,
    five: MultiTimeframeSnapshot,
    config: ScalpSignalConfig,
) -> ShadowAssessment:
    reasons: list[str] = []
    blockers: list[str] = []
    rsi = _number(one.rsi_14)
    if side is SignalSide.LONG:
        rsi_ok = rsi <= config.rsi_oversold
        macd_ok = finite(one.macd_slope) and float(one.macd_slope) > 0
        vwap_ok = str(one.vwap_event or "").upper() in {"RECLAIM", "ABOVE"}
    else:
        rsi_ok = rsi >= config.rsi_overbought
        macd_ok = finite(one.macd_slope) and float(one.macd_slope) < 0
        vwap_ok = str(one.vwap_event or "").upper() in {"REJECTION", "BELOW"}
    _record(rsi_ok, "ONE_MIN_RSI_EXTREME", "ONE_MIN_RSI_NOT_EXTREME", reasons, blockers)
    _record(macd_ok, "ONE_MIN_MACD_TURN", "ONE_MIN_MACD_TURN_MISSING", reasons, blockers)
    _record(vwap_ok, "ONE_MIN_VWAP_REVERSAL_CONFIRMED", "ONE_MIN_VWAP_REVERSAL_MISSING", reasons, blockers)

    alignment = _alignment(side, five.state)
    if five.state == "NO_DATA":
        blockers.append("MTF_5M_DATA_MISSING_OR_STALE")
    elif alignment == "CONFLICT":
        blockers.append("MTF_5M_STRONG_CONFLICT")
    else:
        reasons.append(f"MTF_5M_{alignment}")
    return _assessment("REVERSAL", side, reasons, blockers)


def _context_state(
    *,
    close: float | None,
    vwap: float | None,
    ema_fast: float | None,
    ema_slow: float | None,
    macd_slope: float | None,
    bar_age_ms: int | None,
    max_bar_age_ms: int,
) -> str:
    values = (close, vwap, ema_fast, ema_slow, macd_slope)
    if (
        not all(finite(value) for value in values)
        or bar_age_ms is None
        or bar_age_ms < 0
        or bar_age_ms > max_bar_age_ms
    ):
        return "NO_DATA"
    if float(close) >= float(vwap) and float(ema_fast) > float(ema_slow) and float(macd_slope) >= 0:
        return "BULLISH"
    if float(close) <= float(vwap) and float(ema_fast) < float(ema_slow) and float(macd_slope) <= 0:
        return "BEARISH"
    return "MIXED"


def _alignment(side: SignalSide, state: str) -> str:
    if state == "NO_DATA" or side is SignalSide.NONE:
        return "NO_DATA"
    if state == "MIXED":
        return "MIXED"
    if (side is SignalSide.LONG and state == "BULLISH") or (
        side is SignalSide.SHORT and state == "BEARISH"
    ):
        return "ALIGNED"
    return "CONFLICT"


def _record(
    passed: bool,
    reason: str,
    blocker: str,
    reasons: list[str],
    blockers: list[str],
) -> None:
    (reasons if passed else blockers).append(reason if passed else blocker)


def _assessment(
    family: str,
    side: SignalSide,
    reasons: list[str],
    blockers: list[str],
) -> ShadowAssessment:
    score = max(0.0, min(100.0, 40.0 + 10.0 * len(reasons) - 8.0 * len(blockers)))
    return ShadowAssessment(
        family=family,
        side=side,
        ready=not blockers,
        score=round(score, 1),
        reasons=tuple(dict.fromkeys(reasons)),
        blockers=tuple(dict.fromkeys(blockers)),
    )


def _number(value: Any) -> float:
    return float(value) if finite(value) else 0.0
