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
from .indicators import calculate_one_minute_indicators, indicator_snapshot_from_frame
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
    completed = completed_five_minute_bars(frame)
    if len(completed) < 35:
        return MultiTimeframeSnapshot(completed_bars=len(completed))

    enriched = calculate_one_minute_indicators(completed)
    close = pd.to_numeric(enriched["Close"], errors="coerce")
    enriched["mtf_ema_fast"] = close.ewm(span=5, adjust=False, min_periods=5).mean()
    enriched["mtf_ema_slow"] = close.ewm(span=13, adjust=False, min_periods=13).mean()
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
            (completed.index[-1] + pd.Timedelta(minutes=5)).timestamp() * 1000
        ),
        completed_bars=len(completed),
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
