import math

import pandas as pd
import pytest

from agent.scalp_signal import (
    IndicatorSnapshot,
    PathQuality,
    QuoteSnapshot,
    QuoteSource,
    ScalpSignalConfig,
    SignalSide,
    build_bracket,
    create_scalp_signal_plan,
    detect_scalp_signal_plan,
    indicator_snapshot_from_frame,
)
from agent.scalp.indicators import refresh_indicator_bar_age


def _long_indicators(**overrides):
    values = {
        "rsi_14": 25.0,
        "rsi_7": 20.0,
        "rsi_2": 32.0,
        "macd_hist": -0.04,
        "macd_hist_prev": -0.10,
        "atr_14": 1.0,
        "vwap": 99.80,
        "rvol": 1.20,
        "vwap_event": "RECLAIM",
    }
    values.update(overrides)
    return IndicatorSnapshot(**values)


def _short_indicators(**overrides):
    values = {
        "rsi_14": 75.0,
        "rsi_7": 80.0,
        "rsi_2": 68.0,
        "macd_hist": 0.04,
        "macd_hist_prev": 0.10,
        "atr_14": 1.0,
        "vwap": 100.20,
        "rvol": 1.20,
        "vwap_event": "REJECTION",
    }
    values.update(overrides)
    return IndicatorSnapshot(**values)


def _quote(**overrides):
    values = {
        "ticker": "TEST",
        "last": 100.00,
        "bid": 99.99,
        "ask": 100.01,
        "data_age_ms": 100,
        "source": QuoteSource.WS,
    }
    values.update(overrides)
    return QuoteSnapshot(**values)


def test_build_bracket_long_is_exact_configured_one_to_two():
    bracket = build_bracket(
        entry=100.0,
        side=SignalSide.LONG,
        atr_14=1.0,
        spread=0.02,
        config=ScalpSignalConfig(reward_r=2.0),
    )

    assert bracket.stop_loss == 99.0
    assert bracket.tp1 == 101.0
    assert bracket.tp2 == 102.0
    assert bracket.risk_per_share == 1.0
    assert bracket.rr_ratio == 2.0


def test_build_bracket_short_is_exact_configured_one_to_two():
    bracket = build_bracket(
        entry=100.0,
        side=SignalSide.SHORT,
        atr_14=1.0,
        spread=0.02,
        config=ScalpSignalConfig(reward_r=2.0),
    )

    assert bracket.stop_loss == 101.0
    assert bracket.tp1 == 99.0
    assert bracket.tp2 == 98.0
    assert bracket.rr_ratio == 2.0


def test_valid_long_plan_uses_ask_and_has_complete_audit_context():
    plan = create_scalp_signal_plan(
        quote=_quote(),
        indicators=_long_indicators(),
        side="LONG",
        session="REGULAR",
        resistances=[103.0],
    )

    assert plan.valid is True
    assert plan.entry == 100.01
    assert plan.stop_loss == 99.01
    assert plan.tp1 == 101.01
    assert plan.tp2 == 102.01
    assert plan.rr_ratio == 2.0
    assert plan.rsi_zone == "OS"
    assert plan.bar_age_ms == 0
    assert plan.tp2_path is PathQuality.CLEAR
    assert {"MACD_RISING", "RVOL_CONFIRMED", "TP2_PATH_CLEAR"} <= set(plan.reasons)


def test_valid_short_plan_uses_bid_and_mirrors_bracket():
    plan = create_scalp_signal_plan(
        quote=_quote(),
        indicators=_short_indicators(),
        side="SHORT",
        session="REGULAR",
        supports=[97.0],
    )

    assert plan.valid is True
    assert plan.entry == 99.99
    assert plan.stop_loss == 100.99
    assert plan.tp1 == 98.99
    assert plan.tp2 == 97.99
    assert plan.rr_ratio == 2.0
    assert plan.rsi_zone == "OB"
    assert plan.tp2_path is PathQuality.CLEAR


def test_direction_detection_never_invents_a_trade_from_neutral_rsi():
    plan = detect_scalp_signal_plan(
        quote=_quote(),
        indicators=_long_indicators(rsi_14=50.0),
        session="REGULAR",
    )

    assert plan.side is SignalSide.NONE
    assert plan.valid is False
    assert plan.invalid_reason == "NO_DIRECTIONAL_SETUP"


@pytest.mark.parametrize(
    ("overrides", "blocker"),
    [
        ({"rsi_14": None}, "RSI_14_MISSING"),
        ({"macd_hist": None}, "MACD_HIST_MISSING"),
        ({"macd_hist_prev": math.nan}, "MACD_HIST_PREV_MISSING"),
        ({"atr_14": None}, "ATR_14_MISSING"),
        ({"vwap": None}, "VWAP_MISSING"),
        ({"rvol": None}, "RVOL_MISSING"),
    ],
)
def test_required_indicator_gaps_block_instead_of_becoming_neutral(overrides, blocker):
    plan = create_scalp_signal_plan(
        quote=_quote(),
        indicators=_long_indicators(**overrides),
        side="LONG",
        session="REGULAR",
    )

    assert plan.valid is False
    assert blocker in plan.blockers


def test_stale_quote_blocks_even_when_technicals_are_valid():
    plan = create_scalp_signal_plan(
        quote=_quote(data_age_ms=2_001),
        indicators=_long_indicators(),
        side="LONG",
        session="REGULAR",
    )

    assert plan.valid is False
    assert "QUOTE_STALE" in plan.blockers


def test_rest_fallback_is_visible_and_blocked_by_default():
    plan = create_scalp_signal_plan(
        quote=_quote(source="REST"),
        indicators=_long_indicators(),
        side="LONG",
        session="REGULAR",
    )

    assert plan.source is QuoteSource.REST
    assert "REST_FALLBACK_NOT_TRADABLE" in plan.blockers


def test_spread_to_risk_gate_blocks_expensive_execution():
    config = ScalpSignalConfig(
        max_spread_to_risk=0.10,
        spread_buffer_mult=0.01,
        block_when_risk_capped=False,
    )
    plan = create_scalp_signal_plan(
        quote=_quote(bid=99.80, ask=100.20),
        indicators=_long_indicators(),
        side="LONG",
        session="REGULAR",
        config=config,
    )

    assert "SPREAD_TOO_WIDE_FOR_RISK" in plan.blockers
    assert plan.spread_to_risk > config.max_spread_to_risk


def test_resistance_blocks_path_but_never_rewrites_risk_based_target():
    clear = create_scalp_signal_plan(
        quote=_quote(),
        indicators=_long_indicators(),
        side="LONG",
        session="REGULAR",
        resistances=[103.0],
    )
    blocked = create_scalp_signal_plan(
        quote=_quote(),
        indicators=_long_indicators(),
        side="LONG",
        session="REGULAR",
        resistances=[101.50],
    )

    assert clear.tp2 == blocked.tp2 == 102.01
    assert blocked.valid is False
    assert blocked.tp2_path is PathQuality.BLOCKED_BY_RESISTANCE
    assert "BLOCKED_BY_RESISTANCE" in blocked.blockers


def test_runtime_config_reads_only_new_scalp_namespace():
    cfg = ScalpSignalConfig.from_runtime(
        {
            "scalp.reward_r": 2.5,
            "scalp.max_quote_age_ms": 750,
            "scalp.use_provisional_live_indicators": False,
            "scalp.provisional_max_bar_age_ms": 240_000,
            "scalp.long_require_fast_rsi_confirmation": False,
            "scalp.long_require_vwap_reclaim": False,
            "scalp.long_require_mtf_not_bearish": False,
            "scalp.long_block_bearish_market": False,
            "scalp.short_require_fast_rsi_confirmation": False,
            "scalp.short_premarket_require_vwap_rejection": False,
            "scalp.short_require_mtf_not_bullish": False,
            "scalp.short_block_bullish_market": False,
            "prediction.min_rr": 9.0,
        }
    )

    assert cfg.reward_r == 2.5
    assert cfg.max_quote_age_ms == 750
    assert cfg.use_provisional_live_indicators is False
    assert cfg.provisional_max_bar_age_ms == 240_000
    assert cfg.long_require_fast_rsi_confirmation is False
    assert cfg.long_require_vwap_reclaim is False
    assert cfg.long_require_mtf_not_bearish is False
    assert cfg.long_block_bearish_market is False
    assert cfg.short_require_fast_rsi_confirmation is False
    assert cfg.short_premarket_require_vwap_rejection is False
    assert cfg.short_require_mtf_not_bullish is False
    assert cfg.short_block_bullish_market is False


def test_long_requires_fast_rsi_turn_by_default():
    plan = create_scalp_signal_plan(
        quote=_quote(),
        indicators=_long_indicators(rsi_2=8.0, rsi_7=20.0),
        side="LONG",
        session="REGULAR",
        resistances=[103.0],
    )

    assert plan.valid is False
    assert "LONG_FAST_RSI_NOT_CONFIRMING" in plan.blockers


def test_long_requires_true_vwap_reclaim_by_default():
    blocked = create_scalp_signal_plan(
        quote=_quote(),
        indicators=_long_indicators(vwap_event="ABOVE"),
        side="LONG",
        session="REGULAR",
        resistances=[103.0],
    )
    allowed = create_scalp_signal_plan(
        quote=_quote(),
        indicators=_long_indicators(vwap_event="ABOVE"),
        side="LONG",
        session="REGULAR",
        config=ScalpSignalConfig(long_require_vwap_reclaim=False),
        resistances=[103.0],
    )

    assert "LONG_VWAP_RECLAIM_MISSING" in blocked.blockers
    assert allowed.valid is True


def test_short_requires_fast_rsi_rollover_by_default():
    plan = create_scalp_signal_plan(
        quote=_quote(),
        indicators=_short_indicators(rsi_2=92.0, rsi_7=80.0),
        side="SHORT",
        session="REGULAR",
        supports=[97.0],
    )

    assert plan.valid is False
    assert "SHORT_FAST_RSI_NOT_CONFIRMING" in plan.blockers


def test_premarket_short_requires_true_vwap_rejection_by_default():
    blocked = create_scalp_signal_plan(
        quote=_quote(),
        indicators=_short_indicators(vwap_event="BELOW"),
        side="SHORT",
        session="PRE_MARKET",
        supports=[97.0],
    )
    regular = create_scalp_signal_plan(
        quote=_quote(),
        indicators=_short_indicators(vwap_event="BELOW"),
        side="SHORT",
        session="REGULAR",
        supports=[97.0],
    )

    assert "SHORT_VWAP_REJECTION_MISSING" in blocked.blockers
    assert regular.valid is True


def test_plan_serializes_enums_for_versioned_api_payloads():
    plan = create_scalp_signal_plan(
        quote=_quote(),
        indicators=_long_indicators(),
        side="LONG",
        session="REGULAR",
        resistances=[103.0],
    )

    payload = plan.to_dict()
    assert payload["side"] == "LONG"
    assert payload["source"] == "WS"
    assert payload["tp2_path"] == "CLEAR"


def test_indicator_frame_extraction_uses_real_final_and_previous_values():
    rows = 35
    index = pd.date_range("2026-06-26T13:25:00Z", periods=rows, freq="min")
    frame = pd.DataFrame(
        {
            "Close": [99.0] * 33 + [99.5, 100.5],
            "rsi_14": [25.0] * rows,
            "rsi_7": [20.0] * rows,
            "rsi_2": [8.0] * rows,
            "macd_hist": [-0.2] * 33 + [-0.1, -0.04],
            "atr_14": [1.0] * rows,
            "vwap": [100.0] * rows,
            "vol_ratio": [1.2] * rows,
        },
        index=index,
    )

    snapshot = indicator_snapshot_from_frame(
        frame,
        now_ms=int(index[-1].timestamp() * 1000) + 30_000,
    )

    assert snapshot.macd_hist == -0.04
    assert snapshot.macd_hist_prev == -0.1
    assert snapshot.vwap_event == "RECLAIM"
    assert snapshot.bar_age_ms == 30_000


def test_stale_indicator_bar_blocks_a_live_quote():
    plan = create_scalp_signal_plan(
        quote=_quote(),
        indicators=_long_indicators(bar_age_ms=120_001),
        side="LONG",
        session="REGULAR",
    )

    assert plan.valid is False
    assert "INDICATOR_BAR_STALE" in plan.blockers


def test_cached_indicator_age_refreshes_without_changing_values():
    index = pd.date_range("2026-06-26T13:25:00Z", periods=35, freq="min")
    frame = pd.DataFrame({"Close": [100.0] * 35}, index=index)
    snapshot = _long_indicators(bar_age_ms=1_000)

    refreshed = refresh_indicator_bar_age(
        snapshot,
        frame,
        now_ms=int(index[-1].timestamp() * 1000) + 180_000,
    )

    assert refreshed.bar_age_ms == 180_000
    assert refreshed.rsi_14 == snapshot.rsi_14
    assert refreshed.macd_hist == snapshot.macd_hist


def test_short_frame_returns_invalid_plan_when_bar_depth_is_insufficient():
    snapshot = indicator_snapshot_from_frame(pd.DataFrame({"Close": [100.0] * 10}))
    plan = create_scalp_signal_plan(
        quote=_quote(),
        indicators=snapshot,
        side="LONG",
        session="REGULAR",
    )

    assert plan.valid is False
    assert "RSI_14_MISSING" in plan.blockers
