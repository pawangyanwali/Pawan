"""Scalp-only universe analysis and execution orchestration."""
from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from .bar_feed import load_one_minute_frames
from .engine import create_scalp_signal_plan
from .indicators import (
    calculate_one_minute_indicators,
    indicator_snapshot_from_frame,
    provisional_live_indicators,
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
from .models import IndicatorSnapshot, QuoteSnapshot, QuoteSource, ScalpSignalConfig, ScalpSignalPlan, SignalSide
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
        self._last_shadow_bar: dict[str, int] = {}
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
        self._last_shadow_bar = {
            ticker: value for ticker, value in self._last_shadow_bar.items()
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
        from agent.scalp.execution_policy import execution_market_health

        started = time.monotonic()
        session = get_session()
        session_info = get_session_info()
        contexts = get_context_snapshots(self.tickers)
        frames, bar_errors = load_one_minute_frames(
            self.tickers,
            limit=max(390, int(config.get("scalp_runtime.bar_lookback", 500))),
        )
        signal_config = ScalpSignalConfig.from_runtime(config)
        workers = max(1, min(16, int(config.get("scalp_runtime.workers", 8))))

        def prepare(ticker: str) -> tuple[str, Any | None, int, Any, list[float], list[float], Any | None, str]:
            frame = frames.get(ticker)
            if frame is None or frame.empty:
                return (
                    ticker, None, 0,
                    IndicatorSnapshot(None, None, None, None, None, None, None, None),
                    [], [], None,
                    bar_errors.get(ticker, "ONE_MINUTE_BARS_MISSING"),
                )

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

            return ticker, enriched, bar_id, indicators, supports, resistances, mtf_context, ""

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scalp-bars") as pool:
            prepared = list(pool.map(prepare, self.tickers))
        market_context = _market_direction_context(prepared)

        # Quotes are deliberately captured after the expensive bar/indicator pass.
        # Plan age therefore measures market-data freshness, not cycle compute time.
        quotes = get_all_prices()
        market_health = execution_market_health(max_age_s=2.0)

        def analyze(item: tuple[str, Any | None, int, Any, list[float], list[float], Any | None, str]) -> tuple[str, ScalpSignalPlan, Any | None, int]:
            ticker, enriched, bar_id, indicators, supports, resistances, mtf_context, bar_error = item
            quote = quote_snapshot_from_payload(ticker, quotes.get(ticker) or {})
            if enriched is None:
                plan = create_scalp_signal_plan(
                    quote=quote,
                    indicators=indicators,
                    side=SignalSide.NONE,
                    session=session,
                    config=signal_config,
                )
                _add_blocker(plan, bar_error)
                return ticker, plan, None, 0

            indicators = _with_provisional_live_indicators(
                indicators,
                quote,
                signal_config,
            )
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
            _apply_directional_quality_filters(plan, market_context, signal_config)
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
            analyzed = list(pool.map(analyze, prepared))

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
        if bool(config.get("scalp.shadow_enabled", True)):
            self._shadow_execute(plans_by_ticker, bars_by_ticker, market_health)
        if bool(config.get("scalp.execution_enabled", False)):
            self._execute(
                plans_by_ticker,
                frames_by_ticker,
                bars_by_ticker,
                market_health,
            )

        valid_count = sum(1 for plan in plans_by_ticker.values() if plan.valid)
        data_gap_count = sum(
            1 for plan in plans_by_ticker.values()
            if _has_actionable_data_gap(plan.blockers, session)
        )
        blocker_counts = _blocker_counts(plans_by_ticker.values())
        elapsed_ms = round((time.monotonic() - started) * 1000.0, 1)
        meta = {
            "runtime": "SCALP_ONLY_V1",
            "universe_total": len(self.tickers),
            "valid_plan_count": valid_count,
            "data_gap_count": data_gap_count,
            "market_context": market_context,
            "blocker_counts": blocker_counts,
            "top_blockers": [
                {"blocker": key, "count": value}
                for key, value in sorted(
                    blocker_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:10]
            ],
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
        try:
            from agent.config_manager import config
            from agent.market_hours import get_session
            from agent.scalp.shadow import mark_shadow_trades

            if bool(config.get("scalp.shadow_enabled", True)):
                mark_shadow_trades(quotes, session=get_session())
        except Exception:
            logger.exception("[ScalpRuntime] shadow position tick failed")
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

    def _shadow_execute(
        self,
        plans: dict[str, ScalpSignalPlan],
        bars: dict[str, int],
        market_health: dict[str, Any],
    ) -> None:
        """Open isolated hypothetical trades once per ticker/bar."""
        from agent.scalp.shadow import open_shadow_trade

        for ticker, plan in plans.items():
            bar_id = bars.get(ticker, 0)
            if not plan.valid or not bar_id or self._last_shadow_bar.get(ticker) == bar_id:
                continue
            self._last_shadow_bar[ticker] = bar_id
            try:
                open_shadow_trade(
                    plan,
                    entry_bar_id=bar_id,
                    market_health=market_health,
                )
            except Exception:
                logger.exception("[ScalpRuntime] shadow entry failed for %s", ticker)

    def _execute(
        self,
        plans: dict[str, ScalpSignalPlan],
        frames: dict[str, Any],
        bars: dict[str, int],
        market_health: dict[str, Any],
    ) -> None:
        from agent.paper_trading import maybe_open_trade
        from agent.scalp.execution_policy import evaluate_execution_policy
        from agent.scalp.store import record_execution_decision

        for ticker, plan in plans.items():
            bar_id = bars.get(ticker, 0)
            if not plan.valid or not bar_id or self._last_execution_bar.get(ticker) == bar_id:
                continue
            self._last_execution_bar[ticker] = bar_id
            policy = evaluate_execution_policy(
                plan,
                mode="PAPER",
                market_health=market_health,
            )
            if not policy.allowed:
                record_execution_decision(
                    plan.plan_id,
                    "BLOCKED",
                    reason=f"POLICY_{policy.reason}",
                    detail={"policy": policy.to_dict(), "entry_bar_id": bar_id},
                )
                continue
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
                    size_mult=plan.learning_size_mult * policy.size_mult,
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


def _with_provisional_live_indicators(
    indicators: IndicatorSnapshot,
    quote: QuoteSnapshot,
    config: ScalpSignalConfig,
) -> IndicatorSnapshot:
    """Project one live quote onto the latest closed-bar indicator state."""
    if not config.use_provisional_live_indicators:
        return indicators
    if quote.data_age_ms < 0 or quote.data_age_ms > config.max_quote_age_ms:
        return indicators
    if quote.normalized_source in {QuoteSource.STALE, QuoteSource.UNKNOWN}:
        return indicators
    bar_age = indicators.bar_age_ms
    if bar_age is None or bar_age < 0 or bar_age > config.provisional_max_bar_age_ms:
        return indicators
    provisional = provisional_live_indicators(indicators, quote.last)
    if not provisional:
        return indicators
    return replace(
        indicators,
        rsi_14=provisional.get("rsi_14"),
        rsi_7=provisional.get("rsi_7"),
        rsi_2=provisional.get("rsi_2"),
        macd_hist=provisional.get("macd_hist"),
        macd_hist_prev=indicators.macd_hist,
        vwap_event=_live_vwap_event(indicators, quote.last),
        bar_age_ms=max(0, int(quote.data_age_ms)),
        indicator_close=quote.last,
    )


def _live_vwap_event(indicators: IndicatorSnapshot, live_price: float) -> str:
    prior_price = _finite_float(indicators.indicator_close)
    vwap = _finite_float(indicators.vwap)
    current = _finite_float(live_price)
    if prior_price is None or vwap is None or current is None:
        return str(indicators.vwap_event or "").upper()
    if prior_price < vwap <= current:
        return "RECLAIM"
    if prior_price > vwap >= current:
        return "REJECTION"
    if current > vwap:
        return "ABOVE"
    if current < vwap:
        return "BELOW"
    return "AT_VWAP"


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _blocker_counts(plans: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for plan in plans:
        for blocker in getattr(plan, "blockers", []) or []:
            key = str(blocker or "").strip()
            if key:
                counts[key] = counts.get(key, 0) + 1
    return counts


def _market_direction_context(
    prepared: list[
        tuple[str, Any | None, int, Any, list[float], list[float], Any | None, str]
    ],
) -> dict[str, Any]:
    """Summarize QQQ/SPY context from the same closed bars used by ticker scans."""
    votes: list[str] = []
    evidence: list[dict[str, Any]] = []
    for ticker, _frame, _bar_id, indicators, _supports, _resistances, mtf, error in prepared:
        if str(ticker).upper() not in {"QQQ", "SPY"}:
            continue
        mtf_state = str(getattr(mtf, "state", "NO_DATA") or "NO_DATA").upper()
        vwap_event = str(getattr(indicators, "vwap_event", "") or "").upper()
        macd_slope = _finite_float(getattr(indicators, "macd_slope", None))
        vote = "UNKNOWN"
        if not error and mtf_state in {"BULLISH", "BEARISH"}:
            vote = mtf_state
        elif macd_slope is not None:
            if macd_slope < 0 and vwap_event in {"BELOW", "REJECTION"}:
                vote = "BEARISH"
            elif macd_slope > 0 and vwap_event in {"ABOVE", "RECLAIM"}:
                vote = "BULLISH"
        if vote != "UNKNOWN":
            votes.append(vote)
        evidence.append(
            {
                "ticker": str(ticker).upper(),
                "state": mtf_state,
                "vwap_event": vwap_event,
                "macd_slope": macd_slope,
                "vote": vote,
            }
        )
    bearish = votes.count("BEARISH")
    bullish = votes.count("BULLISH")
    state = "UNKNOWN"
    if bearish and bearish >= bullish:
        state = "BEARISH"
    elif bullish and bullish > bearish:
        state = "BULLISH"
    return {
        "state": state,
        "bearish_votes": bearish,
        "bullish_votes": bullish,
        "evidence": evidence,
    }


def _apply_directional_quality_filters(
    plan: ScalpSignalPlan,
    market_context: dict[str, Any],
    config: ScalpSignalConfig,
) -> None:
    """Apply production guards that protect weak reversal entries from trend tape."""
    market_state = str((market_context or {}).get("state") or "").upper()
    mtf_state = str(plan.mtf_state or "").upper()
    mtf_alignment = str(plan.mtf_alignment or "").upper()
    if plan.side is SignalSide.LONG:
        if config.long_require_mtf_not_bearish and (
            mtf_state == "BEARISH" or mtf_alignment == "CONFLICT"
        ):
            _add_blocker(plan, "LONG_5M_BEARISH_CONTEXT")
        elif config.long_require_mtf_not_bearish:
            plan.reasons.append("LONG_5M_NOT_BEARISH")
        if config.long_block_bearish_market and market_state == "BEARISH":
            _add_blocker(plan, "LONG_MARKET_BEARISH_CONTEXT")
        elif config.long_block_bearish_market:
            plan.reasons.append("MARKET_NOT_BEARISH_FOR_LONG")
    elif plan.side is SignalSide.SHORT:
        if config.short_require_mtf_not_bullish and (
            mtf_state == "BULLISH" or mtf_alignment == "CONFLICT"
        ):
            _add_blocker(plan, "SHORT_5M_BULLISH_CONTEXT")
        elif config.short_require_mtf_not_bullish:
            plan.reasons.append("SHORT_5M_NOT_BULLISH")
        if config.short_block_bullish_market and market_state == "BULLISH":
            _add_blocker(plan, "SHORT_MARKET_BULLISH_CONTEXT")
        elif config.short_block_bullish_market:
            plan.reasons.append("MARKET_NOT_BULLISH_FOR_SHORT")


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
