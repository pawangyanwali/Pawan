import pandas as pd
import pytest

from agent.scalp import (
    IndicatorSnapshot,
    MultiTimeframeSnapshot,
    QuoteSnapshot,
    QuoteSource,
    ScalpSignalConfig,
    create_scalp_signal_plan,
)
from agent.scalp.multi_timeframe import (
    apply_multi_timeframe_shadow,
    completed_five_minute_bar_id,
    completed_five_minute_bars,
    refresh_five_minute_age,
)


def _frame(periods: int) -> pd.DataFrame:
    index = pd.date_range("2026-06-26T13:30:00Z", periods=periods, freq="min")
    values = [100.0 + offset * 0.05 for offset in range(periods)]
    return pd.DataFrame(
        {
            "Open": values,
            "High": [value + 0.04 for value in values],
            "Low": [value - 0.04 for value in values],
            "Close": [value + 0.02 for value in values],
            "Volume": [1000.0] * periods,
        },
        index=index,
    )


def _quote() -> QuoteSnapshot:
    return QuoteSnapshot(
        ticker="TEST",
        last=100.0,
        bid=99.99,
        ask=100.01,
        data_age_ms=100,
        source=QuoteSource.WS,
    )


def _one_minute(**overrides) -> IndicatorSnapshot:
    values = {
        "rsi_14": 55.0,
        "rsi_7": 52.0,
        "rsi_2": 50.0,
        "macd_hist": 0.04,
        "macd_hist_prev": 0.01,
        "atr_14": 1.0,
        "vwap": 99.8,
        "rvol": 1.2,
        "vwap_event": "ABOVE",
        "bar_age_ms": 30_000,
    }
    values.update(overrides)
    return IndicatorSnapshot(**values)


def _five_minute(**overrides) -> MultiTimeframeSnapshot:
    values = {
        "state": "BULLISH",
        "close": 100.0,
        "rsi_14": 58.0,
        "macd_hist": 0.05,
        "macd_hist_prev": 0.03,
        "macd_slope": 0.02,
        "atr_14": 1.5,
        "vwap": 99.5,
        "vwap_event": "ABOVE",
        "ema_fast": 100.0,
        "ema_slow": 99.5,
        "bar_age_ms": 20_000,
        "completed_bars": 80,
    }
    values.update(overrides)
    return MultiTimeframeSnapshot(**values)


def test_resampler_excludes_the_current_incomplete_five_minute_bucket():
    incomplete = completed_five_minute_bars(_frame(8))
    completed = completed_five_minute_bars(_frame(10))

    assert list(incomplete.index) == [pd.Timestamp("2026-06-26T13:30:00Z")]
    assert incomplete.iloc[-1]["Close"] == pytest.approx(_frame(8).iloc[4]["Close"])
    assert list(completed.index) == [
        pd.Timestamp("2026-06-26T13:30:00Z"),
        pd.Timestamp("2026-06-26T13:35:00Z"),
    ]


def test_future_rows_inside_incomplete_bucket_cannot_change_completed_context():
    source = _frame(8)
    changed = source.copy()
    changed.loc[changed.index[-3]:, ["Open", "High", "Low", "Close"]] = 999.0

    left = completed_five_minute_bars(source)
    right = completed_five_minute_bars(changed)

    pd.testing.assert_frame_equal(left, right)


def test_five_minute_cache_id_changes_only_when_a_bucket_closes():
    before_close = completed_five_minute_bar_id(_frame(8))
    still_open = completed_five_minute_bar_id(_frame(9))
    after_close = completed_five_minute_bar_id(_frame(10))

    assert before_close == still_open
    assert after_close > still_open


def test_shadow_momentum_detects_pullback_without_mutating_canonical_plan():
    one = _one_minute()
    plan = create_scalp_signal_plan(
        quote=_quote(),
        indicators=one,
        side="LONG",
        session="REGULAR",
        resistances=[103.0],
    )
    canonical = (
        plan.valid,
        plan.invalid_reason,
        plan.entry,
        plan.stop_loss,
        plan.tp1,
        plan.tp2,
        plan.confidence,
        tuple(plan.blockers),
    )

    apply_multi_timeframe_shadow(
        plan,
        one,
        _five_minute(),
        ScalpSignalConfig(),
        session="REGULAR",
    )

    assert plan.shadow_strategy_family == "MOMENTUM_PULLBACK"
    assert plan.shadow_side == "LONG"
    assert plan.shadow_setup_ready is True
    assert plan.mtf_alignment == "ALIGNED"
    assert (
        plan.valid,
        plan.invalid_reason,
        plan.entry,
        plan.stop_loss,
        plan.tp1,
        plan.tp2,
        plan.confidence,
        tuple(plan.blockers),
    ) == canonical


def test_five_minute_conflict_is_shadow_evidence_not_live_blocker():
    one = _one_minute(
        rsi_14=25.0,
        rsi_7=20.0,
        rsi_2=8.0,
        macd_hist=-0.02,
        macd_hist_prev=-0.08,
        vwap_event="RECLAIM",
    )
    plan = create_scalp_signal_plan(
        quote=_quote(),
        indicators=one,
        side="LONG",
        session="REGULAR",
        resistances=[103.0],
    )
    assert plan.valid is True

    apply_multi_timeframe_shadow(
        plan,
        one,
        _five_minute(state="BEARISH", macd_slope=-0.02),
        ScalpSignalConfig(),
        session="REGULAR",
    )

    assert plan.valid is True
    assert plan.mtf_alignment == "CONFLICT"
    assert "MTF_5M_STRONG_CONFLICT" in plan.shadow_blockers
    assert "MTF_5M_STRONG_CONFLICT" not in plan.blockers


def test_missing_five_minute_context_fails_open_in_shadow_mode():
    one = _one_minute(
        rsi_14=25.0,
        rsi_7=20.0,
        rsi_2=8.0,
        macd_hist=-0.02,
        macd_hist_prev=-0.08,
        vwap_event="RECLAIM",
    )
    plan = create_scalp_signal_plan(
        quote=_quote(),
        indicators=one,
        side="LONG",
        session="REGULAR",
        resistances=[103.0],
    )

    apply_multi_timeframe_shadow(
        plan,
        one,
        MultiTimeframeSnapshot(),
        ScalpSignalConfig(),
        session="REGULAR",
    )

    assert plan.valid is True
    assert plan.mtf_state == "NO_DATA"
    assert "MTF_5M_DATA_MISSING_OR_STALE" in plan.shadow_blockers
    assert not any(blocker.startswith("MTF_") for blocker in plan.blockers)


def test_cached_five_minute_context_becomes_stale_without_a_new_bar():
    snapshot = _five_minute(bar_closed_at_ms=1_000_000, bar_age_ms=10_000)

    refreshed = refresh_five_minute_age(
        snapshot,
        now_ms=1_500_001,
        max_bar_age_ms=420_000,
    )

    assert refreshed.bar_age_ms == 500_001
    assert refreshed.state == "NO_DATA"


def test_mtf_runtime_configuration_rejects_unsafe_values():
    with pytest.raises(ValueError, match="mtf_mode"):
        ScalpSignalConfig(mtf_mode="BLOCK")
    with pytest.raises(ValueError, match="LONG RSI"):
        ScalpSignalConfig(momentum_long_rsi_min=75, momentum_long_rsi_max=70)
