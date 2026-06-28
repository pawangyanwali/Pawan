"""Scalp-only universe analysis and execution orchestration."""
from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .bar_feed import load_one_minute_frames
from .engine import create_scalp_signal_plan
from .indicators import (
    calculate_one_minute_indicators,
    indicator_snapshot_from_frame,
    refresh_indicator_bar_age,
)
from .learning import apply_context_gate
from .ml_overlay import apply_ml_overlay
from .multi_timeframe import (
    apply_multi_timeframe_shadow,
    completed_five_minute_bar_id,
    five_minute_snapshot,
    refresh_five_minute_age,
)
from .models import IndicatorSnapshot, ScalpSignalConfig, ScalpSignalPlan, SignalSide
from .quality import has_market_data_gap
from .quotes import quote_snapshot_from_payload

logger = logging.getLogger(__name__)


class ScalpRuntime:
    """Owns canonical plan creation; no legacy prediction code is imported."""

    def __init__(self, tickers: list[str]) -> None:
        self.tickers = list(dict.fromkeys(str(t).upper() for t in tickers))
        self._indicator_cache: dict[str, tuple[int, Any, IndicatorSnapshot, list[float], list[float]]] = {}
        self._mtf_cache: dict[str, tuple[int, Any]] = {}
        self._last_execution_bar: dict[str, int] = {}
        self._last_position_bar: dict[str, int] = {}
        self.last_cycle: dict[str, Any] = {}

    def update_tickers(self, tickers: list[str]) -> bool:
        """Replace the eligible universe between cycles and prune stale caches."""
        normalized = list(dict.fromkeys(str(t).upper() for t in tickers if t))
        if normalized == self.tickers:
            return False
        keep = set(normalized)
        self.tickers = normalized
        self._indicator_cache = {
            ticker: value for ticker, value in self._indicator_cache.items()
            if ticker in keep
        }
        self._mtf_cache = {
            ticker: value for ticker, value in self._mtf_cache.items()
            if ticker in keep
        }
        self._last_execution_bar = {
            ticker: value for ticker, value in self._last_execution_bar.items()
            if ticker in keep
        }
        self._last_position_bar = {
            ticker: value for ticker, value in self._last_position_bar.items()
            if ticker in keep
        }
        logger.warning("[ScalpRuntime] Eligible universe updated to %d tickers", len(normalized))
        return True

    def run_cycle(self) -> dict[str, Any]:
        from agent.config_manager import config
        from agent.context_snapshot import get_context_snapshots
        from agent.market_hours import get_session, get_session_info
        from agent.signal_snapshot import write_latest
        from agent.valkey_client import get_all_prices

        started = time.monotonic()
        session = get_session()
        session_info = get_session_info()
        quotes = get_all_prices()
        contexts = get_context_snapshots(self.tickers)
        frames, bar_errors = load_one_minute_frames(
            self.tickers,
            limit=max(390, int(config.get("scalp_runtime.bar_lookback", 500))),
        )
        signal_config = ScalpSignalConfig.from_runtime(config)
        workers = max(1, min(16, int(config.get("scalp_runtime.workers", 8))))

        def analyze(ticker: str) -> tuple[str, ScalpSignalPlan, Any | None, int]:
            quote = quote_snapshot_from_payload(ticker, quotes.get(ticker) or {})
            frame = frames.get(ticker)
            if frame is None or frame.empty:
                plan = create_scalp_signal_plan(
                    quote=quote,
                    indicators=IndicatorSnapshot(None, None, None, None, None, None, None, None),
                    side=SignalSide.NONE,
                    session=session,
                    config=signal_config,
                )
                _add_blocker(plan, bar_errors.get(ticker, "ONE_MINUTE_BARS_MISSING"))
                return ticker, plan, None, 0

            bar_id = int(frame.index[-1].timestamp() * 1000)
            cached = self._indicator_cache.get(ticker)
            if cached and cached[0] == bar_id:
                enriched, cached_indicators, supports, resistances = cached[1:]
                indicators = refresh_indicator_bar_age(cached_indicators, enriched)
            else:
                enriched = calculate_one_minute_indicators(frame)
                indicators = indicator_snapshot_from_frame(enriched)
                supports, resistances = _structure_levels(enriched)
                self._indicator_cache[ticker] = (
                    bar_id, enriched, indicators, supports, resistances
                )

            mtf_bar_id = completed_five_minute_bar_id(frame)
            cached_mtf = self._mtf_cache.get(ticker)
            if cached_mtf and cached_mtf[0] == mtf_bar_id:
                mtf_context = refresh_five_minute_age(
                    cached_mtf[1],
                    max_bar_age_ms=signal_config.mtf_max_bar_age_ms,
                )
            else:
                mtf_context = five_minute_snapshot(
                    frame,
                    max_bar_age_ms=signal_config.mtf_max_bar_age_ms,
                )
                self._mtf_cache[ticker] = (mtf_bar_id, mtf_context)

            candidates = [
                create_scalp_signal_plan(
                    quote=quote,
                    indicators=indicators,
                    side=side,
                    session=session,
                    config=signal_config,
                    supports=supports,
                    resistances=resistances,
                )
                for side in (SignalSide.LONG, SignalSide.SHORT)
            ]
            plan = _select_candidate(candidates, indicators.macd_slope)
            apply_multi_timeframe_shadow(
                plan,
                indicators,
                mtf_context,
                signal_config,
                session=session,
            )
            blocked_sessions = {
                str(value).upper()
                for value in config.get(
                    "scalp_runtime.blocked_sessions",
                    ["CLOSED", "RESTRICTED", "CLOSING_CAUTION", "HARD_CLOSE"],
                )
            }
            if session in blocked_sessions:
                _add_blocker(plan, f"SESSION_{session}_BLOCKED")
            _apply_market_context(plan, contexts.get(ticker) or {}, config)
            if plan.valid:
                plan = apply_ml_overlay(plan)
                plan = apply_context_gate(plan)
            return ticker, plan, enriched, bar_id

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scalp-plan") as pool:
            analyzed = list(pool.map(analyze, self.tickers))

        rows: list[dict[str, Any]] = []
        plans_by_ticker: dict[str, ScalpSignalPlan] = {}
        frames_by_ticker: dict[str, Any] = {}
        bars_by_ticker: dict[str, int] = {}
        for ticker, plan, frame, bar_id in analyzed:
            rows.append({"ticker": ticker, "scalp_plan": plan.to_dict()})
            plans_by_ticker[ticker] = plan
            if frame is not None:
                frames_by_ticker[ticker] = frame
                bars_by_ticker[ticker] = bar_id

        self._manage_positions(frames_by_ticker, bars_by_ticker, quotes)
        if bool(config.get("scalp.execution_enabled", False)):
            self._execute(plans_by_ticker, frames_by_ticker, bars_by_ticker)

        valid_count = sum(1 for plan in plans_by_ticker.values() if plan.valid)
        data_gap_count = sum(
            1 for plan in plans_by_ticker.values()
            if _has_actionable_data_gap(plan.blockers, session)
        )
        elapsed_ms = round((time.monotonic() - started) * 1000.0, 1)
        meta = {
            "runtime": "SCALP_ONLY_V1",
            "universe_total": len(self.tickers),
            "valid_plan_count": valid_count,
            "data_gap_count": data_gap_count,
            "cycle_ms": elapsed_ms,
        }
        write_latest(rows, {}, session_info, len(rows), scan_meta=meta)
        self.last_cycle = {"ts": time.time(), **meta}
        logger.info(
            "Scalp cycle: %d/%d valid, %d data gaps, %.1fms",
            valid_count, len(rows), data_gap_count, elapsed_ms,
        )
        return self.last_cycle

    def run_position_tick(self) -> None:
        """Check live stop, TP1, and TP2 conditions from the current price bus."""
        from agent.paper_trading import get_open_trades, rt_check_positions
        from agent.valkey_client import get_all_prices

        quotes = get_all_prices()
        for ticker in {
            str(row.get("ticker") or "").upper() for row in get_open_trades()
        }:
            if not ticker:
                continue
            quote = quotes.get(ticker) or {}
            try:
                current = float(quote.get("last") or quote.get("mark") or 0.0)
            except (TypeError, ValueError):
                current = 0.0
            if current <= 0:
                continue
            try:
                rt_check_positions(ticker, current)
            except Exception:
                logger.exception("[ScalpRuntime] live position tick failed for %s", ticker)

    def _execute(
        self,
        plans: dict[str, ScalpSignalPlan],
        frames: dict[str, Any],
        bars: dict[str, int],
    ) -> None:
        from agent.paper_trading import maybe_open_trade

        for ticker, plan in plans.items():
            bar_id = bars.get(ticker, 0)
            if not plan.valid or not bar_id or self._last_execution_bar.get(ticker) == bar_id:
                continue
            self._last_execution_bar[ticker] = bar_id
            frame = frames[ticker]
            average_minute_volume = float(frame["Volume"].tail(20).mean())
            if not math.isfinite(average_minute_volume) or average_minute_volume < 0:
                average_minute_volume = 0.0
            avg_daily_volume = average_minute_volume * 390.0
            try:
                maybe_open_trade(
                    ticker=ticker,
                    direction="BUY" if plan.side is SignalSide.LONG else "SELL",
                    price=plan.entry,
                    target=plan.tp2,
                    stop=plan.stop_loss,
                    confidence=plan.confidence,
                    rr_qualifies=True,
                    rr_ratio=plan.rr_ratio,
                    session=plan.session,
                    vwap_event=plan.vwap_event,
                    rsi_zone=plan.rsi_zone,
                    entry_type="SCALP",
                    size_mult=plan.learning_size_mult,
                    trading_tier="HIGH",
                    algo_name="SCALP_V1",
                    atr=plan.atr_14,
                    rsi_value=plan.rsi_14,
                    macd_hist=plan.macd_hist,
                    macd_hist_prev=plan.macd_hist_prev,
                    avg_daily_volume=avg_daily_volume,
                    scalp_plan=plan,
                )
            except Exception:
                logger.exception("[ScalpRuntime] execution failed for %s", ticker)

    def _manage_positions(
        self,
        frames: dict[str, Any],
        bars: dict[str, int],
        quotes: dict[str, dict],
    ) -> None:
        from agent.paper_trading import get_open_trades, update_open_trades

        for row in get_open_trades():
            ticker = str(row.get("ticker") or "").upper()
            frame = frames.get(ticker)
            bar_id = bars.get(ticker, 0)
            if frame is None or not bar_id or self._last_position_bar.get(ticker) == bar_id:
                continue
            current = float((quotes.get(ticker) or {}).get("last") or frame.iloc[-1]["Close"])
            self._last_position_bar[ticker] = bar_id
            try:
                update_open_trades(
                    ticker,
                    frame,
                    current,
                    bar_high=float(frame.iloc[-1]["High"]),
                    bar_low=float(frame.iloc[-1]["Low"]),
                )
            except Exception:
                logger.exception("[ScalpRuntime] position update failed for %s", ticker)


def _select_candidate(
    candidates: list[ScalpSignalPlan], macd_slope: float | None
) -> ScalpSignalPlan:
    valid = [candidate for candidate in candidates if candidate.valid]
    if valid:
        return max(valid, key=lambda candidate: candidate.setup_score)
    preferred = SignalSide.LONG if float(macd_slope or 0.0) >= 0 else SignalSide.SHORT
    return min(
        candidates,
        key=lambda candidate: (
            len(candidate.blockers),
            candidate.side is not preferred,
            -candidate.setup_score,
        ),
    )


def _add_blocker(plan: ScalpSignalPlan, blocker: str) -> None:
    if blocker not in plan.blockers:
        plan.blockers.append(blocker)
    plan.valid = False
    plan.invalid_reason = plan.blockers[0]


def _has_actionable_data_gap(blockers: list[str], session: str) -> bool:
    return has_market_data_gap(blockers, session)


def _apply_market_context(plan: ScalpSignalPlan, context: dict[str, Any], config: Any) -> None:
    age = float(context.get("stale_age_s") or 9999.0)
    maximum_age = max(30.0, float(config.get("scalp_runtime.max_context_age_s", 180.0)))
    plan.context_fresh = bool(context.get("asof_ts")) and age <= maximum_age
    plan.sentiment_30m = float(context.get("sentiment_30m") or 0.0)
    plan.sentiment_velocity = float(context.get("sentiment_velocity") or 0.0)
    plan.news_shock = bool(context.get("news_shock"))
    plan.context_risk_score = float(context.get("context_risk_score") or 0.0)
    plan.earnings_phase = str(context.get("earnings_phase") or "").upper()
    plan.earnings_next_date = str(context.get("earnings_next_date") or "")
    try:
        plan.earnings_days_away = int(context.get("earnings_days_away"))
    except (TypeError, ValueError):
        plan.earnings_days_away = 999
    plan.recent_headlines = [str(value) for value in (context.get("recent_headlines") or [])[:3]]

    if bool(config.get("scalp_runtime.require_context_data", True)) and not plan.context_fresh:
        _add_blocker(plan, "CONTEXT_DATA_MISSING_OR_STALE")
    if plan.earnings_phase == "BLACKOUT":
        _add_blocker(plan, "EARNINGS_BLACKOUT")
    elif plan.earnings_phase == "CAUTION":
        plan.learning_size_mult = min(plan.learning_size_mult, 0.5)
        plan.reasons.append("EARNINGS_CAUTION_SIZE_REDUCED")
    maximum_risk = float(config.get("scalp_runtime.max_context_risk_score", 0.8))
    if plan.context_risk_score >= maximum_risk:
        _add_blocker(plan, "CONTEXT_RISK_TOO_HIGH")
    adverse_threshold = abs(float(config.get("scalp_runtime.adverse_news_sentiment", 0.25)))
    adverse = (
        plan.news_shock
        and (
            (plan.side is SignalSide.LONG and plan.sentiment_30m <= -adverse_threshold)
            or (plan.side is SignalSide.SHORT and plan.sentiment_30m >= adverse_threshold)
        )
    )
    if adverse:
        _add_blocker(plan, "ADVERSE_NEWS_SHOCK")
    elif plan.context_fresh:
        plan.reasons.append("CONTEXT_FRESH")


def _structure_levels(frame: Any, *, lookback: int = 60, window: int = 3) -> tuple[list[float], list[float]]:
    recent = frame.tail(max(15, lookback))
    current = float(recent.iloc[-1]["Close"])
    highs = recent["High"]
    lows = recent["Low"]
    swing_highs = highs[highs == highs.rolling(window * 2 + 1, center=True).max()].dropna()
    swing_lows = lows[lows == lows.rolling(window * 2 + 1, center=True).min()].dropna()
    resistances = sorted({round(float(v), 4) for v in swing_highs if float(v) > current})
    supports = sorted(
        {round(float(v), 4) for v in swing_lows if float(v) < current}, reverse=True
    )
    return supports[:5], resistances[:5]
