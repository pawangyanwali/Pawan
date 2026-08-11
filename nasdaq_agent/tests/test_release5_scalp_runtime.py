from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

from agent.scalp.bar_feed import (
    _delta_fetch_count,
    _frame_cache,
    _frame_from_payload,
    _merge_frame,
    load_one_minute_frames,
)
from agent.scalp.indicators import (
    calculate_one_minute_indicators,
    indicator_snapshot_from_frame,
    provisional_live_indicators,
    update_one_minute_indicators,
)
from agent.scalp.quality import has_market_data_gap


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "nasdaq_agent"


def _bars(count: int = 80) -> list[bytes]:
    start = pd.Timestamp("2026-06-25T13:30:00Z")
    payload = []
    for index in range(count):
        price = 100.0 + index * 0.05
        payload.append(
            json.dumps(
                {
                    "time_ms": int((start + pd.Timedelta(minutes=index)).timestamp() * 1000),
                    "open": price - 0.02,
                    "high": price + 0.08,
                    "low": price - 0.08,
                    "close": price,
                    "volume": 1000 + index * 5,
                }
            ).encode()
        )
    return payload


def test_batched_bar_payload_becomes_ordered_utc_frame():
    frame = _frame_from_payload(list(reversed(_bars(50))))
    assert len(frame) == 50
    assert str(frame.index.tz) == "UTC"
    assert frame.index.is_monotonic_increasing
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_warmed_bar_cache_reads_and_merges_only_delta_tail():
    frame = _frame_from_payload(_bars(80))
    latest = _bars(81)[-1]
    cached = (_bars(80)[-1], 2500, frame)
    assert _delta_fetch_count(cached, latest, 2500) == 2
    delta = _frame_from_payload(_bars(81)[-3:])
    merged = _merge_frame(frame, delta, 2500)
    assert len(merged) == 81
    assert merged.index[-1] == delta.index[-1]


def test_warmed_bar_reader_reuses_single_tail_probe(monkeypatch):
    import agent.valkey_client as valkey_client

    class Pipe:
        def __init__(self, owner):
            self.owner = owner
            self.ops = []

        def lrange(self, key, start, end):
            self.ops.append((key, start, end))
            return self

        def execute(self):
            self.owner.batches.append(list(self.ops))
            return [self.owner.payload[start:] for _, start, _ in self.ops]

    class Client:
        def __init__(self):
            self.payload = _bars(50)
            self.batches = []

        def pipeline(self, transaction=False):
            assert transaction is False
            return Pipe(self)

    client = Client()
    _frame_cache.clear()
    monkeypatch.setattr(valkey_client, "_get_client", lambda: client)

    first, errors = load_one_minute_frames(["AAPL"], limit=35)
    assert not errors and len(first["AAPL"]) == 35
    client.payload = _bars(51)
    second, errors = load_one_minute_frames(["AAPL"], limit=35)

    assert not errors and len(second["AAPL"]) == 35
    assert [len(batch) for batch in client.batches] == [1, 1, 1]
    assert client.batches[-1] == [("md:1m:AAPL", -2, -1)]


def test_indicator_contract_is_computed_from_closed_one_minute_bars():
    enriched = calculate_one_minute_indicators(_frame_from_payload(_bars()))
    snapshot = indicator_snapshot_from_frame(
        enriched,
        now_ms=int(enriched.index[-1].timestamp() * 1000) + 60_000,
    )
    assert snapshot.rsi_14 is not None
    assert snapshot.rsi_7 is not None
    assert snapshot.rsi_2 is not None
    assert snapshot.macd_hist is not None
    assert snapshot.macd_hist_prev is not None
    assert snapshot.atr_14 and snapshot.atr_14 > 0
    assert snapshot.vwap and snapshot.vwap > 0
    assert snapshot.rvol and snapshot.rvol > 0
    assert snapshot.bar_age_ms == 60_000


def test_indicator_snapshot_accepts_compact_warmed_recursive_state():
    enriched = calculate_one_minute_indicators(_frame_from_payload(_bars()))

    snapshot = indicator_snapshot_from_frame(enriched.tail(2))

    assert snapshot.rsi_14 is not None
    assert snapshot.rsi_7 is not None
    assert snapshot.rsi_2 is not None
    assert snapshot.macd_hist is not None
    assert snapshot.macd_hist_prev is not None
    assert snapshot.atr_14 and snapshot.atr_14 > 0
    assert snapshot.vwap and snapshot.vwap > 0
    assert snapshot.rvol and snapshot.rvol > 0


def test_extended_hours_inactivity_is_not_misclassified_as_feed_failure():
    blockers = ["INDICATOR_BAR_STALE", "QUOTE_STALE"]

    assert has_market_data_gap(blockers, "AFTER_HOURS") is False
    assert has_market_data_gap(blockers, "REGULAR") is True
    assert has_market_data_gap(["ATR_14_MISSING"], "AFTER_HOURS") is True


def test_rvol_prefers_same_session_and_minute_across_prior_days():
    index = pd.DatetimeIndex([
        "2026-06-22T13:30:00Z",
        "2026-06-23T13:30:00Z",
        "2026-06-24T13:30:00Z",
    ])
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [100.2, 101.2, 102.2],
            "Low": [99.8, 100.8, 101.8],
            "Close": [100.1, 101.1, 102.1],
            "Volume": [100.0, 200.0, 600.0],
        },
        index=index,
    )

    enriched = calculate_one_minute_indicators(frame)

    assert enriched["vol_ratio"].iloc[-1] == pytest.approx(4.0)


def test_provisional_indicators_match_appending_one_live_price_observation():
    frame = _frame_from_payload(_bars())
    enriched = calculate_one_minute_indicators(frame)
    snapshot = indicator_snapshot_from_frame(enriched)
    live_price = float(frame["Close"].iloc[-1]) - 0.37

    provisional = provisional_live_indicators(snapshot, live_price)
    appended = frame.copy()
    next_index = appended.index[-1] + pd.Timedelta(minutes=1)
    appended.loc[next_index] = {
        "Open": live_price,
        "High": live_price,
        "Low": live_price,
        "Close": live_price,
        "Volume": 0.0,
    }
    expected = calculate_one_minute_indicators(appended).iloc[-1]

    assert provisional is not None
    assert provisional["rsi_14"] == pytest.approx(expected["rsi_14"], abs=1e-6)
    assert provisional["rsi_7"] == pytest.approx(expected["rsi_7"], abs=1e-6)
    assert provisional["rsi_2"] == pytest.approx(expected["rsi_2"], abs=1e-6)
    assert provisional["macd_hist"] == pytest.approx(expected["macd_hist"], abs=1e-8)
    assert provisional["macd_slope"] == pytest.approx(
        expected["macd_hist"] - enriched["macd_hist"].iloc[-1], abs=1e-8
    )


@pytest.mark.parametrize("replace_latest", [False, True])
def test_incremental_indicators_match_full_recalculation(replace_latest):
    frame = _frame_from_payload(_bars(240))
    prior_frame = frame.iloc[:-1] if not replace_latest else frame.copy()
    prior = calculate_one_minute_indicators(prior_frame)
    current = frame.copy()
    if replace_latest:
        current.loc[current.index[-1], "Close"] -= 0.21
        current.loc[current.index[-1], "Low"] -= 0.21
        current.loc[current.index[-1], "Volume"] += 275

    incremental = update_one_minute_indicators(prior, current)
    expected = calculate_one_minute_indicators(current)
    for column in (
        "rsi_14", "rsi_7", "rsi_2", "macd_hist", "atr_14", "vwap",
        "vol_ratio", "rsi_avg_gain_14", "rsi_avg_loss_14",
        "macd_fast_ema", "macd_slow_ema", "macd_signal_ema",
    ):
        assert incremental[column].iloc[-1] == pytest.approx(
            expected[column].iloc[-1], rel=1e-10, abs=1e-10
        )


def test_runtime_uses_provisional_live_indicator_snapshot_for_recent_stale_bar():
    from agent.scalp.models import QuoteSnapshot, QuoteSource, ScalpSignalConfig
    from agent.scalp.runtime import _with_provisional_live_indicators

    frame = _frame_from_payload(_bars())
    enriched = calculate_one_minute_indicators(frame)
    snapshot = replace(indicator_snapshot_from_frame(enriched), bar_age_ms=180_000)
    live_price = float(frame["Close"].iloc[-1]) + 0.42
    quote = QuoteSnapshot(
        ticker="AAA",
        last=live_price,
        bid=live_price - 0.01,
        ask=live_price + 0.01,
        data_age_ms=150,
        source=QuoteSource.WS,
    )

    refreshed = _with_provisional_live_indicators(
        snapshot,
        quote,
        ScalpSignalConfig(max_bar_age_ms=120_000, provisional_max_bar_age_ms=300_000),
    )

    assert refreshed.bar_age_ms == 150
    assert refreshed.indicator_close == live_price
    assert refreshed.macd_hist_prev == snapshot.macd_hist
    assert refreshed.macd_hist != snapshot.macd_hist


def test_compose_has_only_canonical_signal_and_learning_owners():
    import yaml

    compose = yaml.safe_load((APP / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert "scalp-engine" in services
    assert "scalp-learner" in services
    assert "scanner" not in services
    assert "learner" not in services
    assert services["scalp-engine"]["command"][-1] == "services.scalp_engine_service"
    assert services["scalp-learner"]["command"][-1] == "services.scalp_learner_service"
    watched = services["watchdog"]["environment"]["WATCHDOG_SERVICES"]
    assert "scalp-engine" in watched and "scalp-learner" in watched
    assert ",scanner," not in f",{watched},"
    assert ",learner," not in f",{watched},"


def test_root_ui_and_deployment_are_scalp_only():
    routes = (APP / "routers" / "system.py").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
    assert 'STATIC_DIR, "scalp.html"' in routes
    assert "_dc up -d --remove-orphans" in workflow
    assert "orphaned legacy runtime container still exists" in workflow
    assert "_svc_health scalp-engine" in workflow
    assert "_svc_health scalp-learner" in workflow
    assert "python -m scripts.activate_scalp_only" in workflow
    assert "python -m scripts.validate_production_release" in workflow
    assert 'EXPECTED_SHA="${{ github.sha }}"' in workflow


def test_activation_enables_shadow_and_keeps_paper_execution_off():
    activation = (APP / "scripts" / "activate_scalp_only.py").read_text(encoding="utf-8")
    assert '"scalp.shadow_enabled": True' in activation
    assert '"scalp.execution_enabled": False' in activation


def test_context_compute_covers_runtime_universe_and_md_has_headroom():
    context = (APP / "services" / "context_intel_service.py").read_text(encoding="utf-8")
    compose = (APP / "docker-compose.yml").read_text(encoding="utf-8")
    market = compose.split("  market-data:", 1)[1].split("  scalp-learner:", 1)[0]
    assert "runtime_tickers = get_runtime_universe()" in context
    assert "runtime_tickers + event_tickers" in context
    assert "memory: 1G" in market


def test_runtime_controls_are_ui_catalogued():
    from agent.config_catalog import build_catalog
    from agent.config_manager import _DEFAULTS

    catalog = build_catalog({}, {key: factory() for key, factory in _DEFAULTS.items()})
    fields = {field["key"]: field for field in catalog["fields"]}
    for key in (
        "scalp_runtime.cycle_interval_s",
        "scalp_runtime.workers",
        "scalp_runtime.bar_lookback",
        "scalp_runtime.mtf_bar_lookback",
        "scalp_runtime.mtf_boundary_batch_size",
        "scalp_runtime.mtf_refresh_batch_size",
        "scalp_runtime.blocked_sessions",
        "scalp.long_require_mtf_not_bearish",
        "scalp.long_block_bearish_market",
        "scalp.short_require_fast_rsi_confirmation",
        "scalp.short_premarket_require_vwap_rejection",
        "scalp.short_require_mtf_not_bullish",
        "scalp.short_block_bullish_market",
        "scalp.shadow_fixed_risk_enabled",
        "scalp.shadow_risk_per_trade_usd",
        "scalp_learn.setup_session_gate_enabled",
        "scalp_learn.setup_session_fast_stop_block_enabled",
        "scalp_learn.setup_session_fast_stop_block_count",
        "scalp_runtime.context_cluster_use_setup_session",
        "scalp_runtime.setup_session_cluster_max_entries",
    ):
        assert key in fields
        assert fields[key]["description"]
    assert fields["scalp_runtime.cycle_interval_s"]["advanced"] is False
    assert fields["scalp_runtime.bar_lookback"]["default"] == 2500
    assert fields["scalp_runtime.mtf_boundary_batch_size"]["default"] == 16
    assert fields["scalp_runtime.mtf_refresh_batch_size"]["default"] == 128


def test_deferred_mtf_refresh_is_bounded_and_fail_closed():
    import inspect
    import agent.scalp.runtime as runtime_module

    source = inspect.getsource(runtime_module.ScalpRuntime.run_cycle)
    assert "scalp_runtime.mtf_boundary_batch_size" in source
    assert "scalp_runtime.mtf_refresh_batch_size" in source
    assert source.index('_add_blocker(plan, "MTF_REFRESH_PENDING")') < source.index(
        "apply_ml_overlay(plan)"
    )
    assert "not mtf_pending" in source


def test_runtime_publishes_every_ticker_even_when_one_has_no_bars(monkeypatch):
    from agent.scalp.runtime import ScalpRuntime
    import agent.scalp.runtime as runtime_module
    import agent.context_snapshot as context_snapshot
    import agent.paper_trading as paper_trading
    import agent.signal_snapshot as signal_snapshot
    import agent.valkey_client as valkey_client

    frame = _frame_from_payload(_bars())
    monkeypatch.setattr(
        runtime_module,
        "load_one_minute_frames",
        lambda tickers, limit: ({"AAA": frame}, {"BBB": "ONE_MINUTE_BARS_MISSING"}),
    )
    monkeypatch.setattr(
        valkey_client,
        "get_all_prices",
        lambda: {
            ticker: {
                "last": 104.0,
                "bid": 103.99,
                "ask": 104.01,
                "updated_at": pd.Timestamp.now(tz="UTC").timestamp(),
                "source_status": "LIVE",
            }
            for ticker in ("AAA", "BBB")
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.market_hours",
        SimpleNamespace(
            get_session=lambda: "CLOSED",
            get_session_info=lambda: {"session": "CLOSED", "tradeable": False},
        ),
    )
    monkeypatch.setattr(paper_trading, "get_open_trades", lambda: [])
    monkeypatch.setattr(
        context_snapshot,
        "get_context_snapshots",
        lambda tickers: {
            ticker: {"asof_ts": pd.Timestamp.now(tz="UTC").timestamp(), "stale_age_s": 1.0}
            for ticker in tickers
        },
    )
    captured = {}

    def write_latest(signals, regime, session, scanned_count, scan_meta=None):
        captured.update(
            signals=signals,
            scanned_count=scanned_count,
            scan_meta=scan_meta,
        )
        return True

    monkeypatch.setattr(signal_snapshot, "write_latest", write_latest)
    result = ScalpRuntime(["AAA", "BBB"]).run_cycle()
    assert result["runtime"] == "SCALP_ONLY_V1"
    assert captured["scanned_count"] == 2
    assert {row["ticker"] for row in captured["signals"]} == {"AAA", "BBB"}
    missing = next(row for row in captured["signals"] if row["ticker"] == "BBB")
    assert "ONE_MINUTE_BARS_MISSING" in missing["scalp_plan"]["blockers"]


def test_fresh_context_is_required_and_earnings_blackout_is_explicit():
    from agent.scalp.models import ScalpSignalPlan, SignalSide
    from agent.scalp.runtime import _apply_market_context

    plan = ScalpSignalPlan(
        ticker="AAA",
        side=SignalSide.LONG,
        valid=True,
        invalid_reason="",
    )
    _apply_market_context(
        plan,
        {
            "asof_ts": 1.0,
            "stale_age_s": 5.0,
            "earnings_phase": "blackout",
            "earnings_days_away": 0,
            "news_shock": True,
            "sentiment_30m": -0.5,
        },
        {
            "scalp_runtime.require_context_data": True,
            "scalp_runtime.max_context_age_s": 180,
            "scalp_runtime.max_context_risk_score": 0.8,
            "scalp_runtime.adverse_news_sentiment": 0.25,
        },
    )
    assert plan.context_fresh is True
    assert plan.earnings_days_away == 0
    assert "EARNINGS_BLACKOUT" in plan.blockers
    assert "ADVERSE_NEWS_SHOCK" in plan.blockers
    assert plan.valid is False


def test_market_context_is_present_before_ml_inference(monkeypatch):
    import inspect
    import agent.scalp.runtime as runtime_module

    source = inspect.getsource(runtime_module.ScalpRuntime.run_cycle)
    assert source.index("_apply_market_context(") < source.index("apply_ml_overlay(")

    observed = {}

    def capture(plan):
        observed.update(
            context_fresh=plan.context_fresh,
            sentiment_30m=plan.sentiment_30m,
            earnings_phase=plan.earnings_phase,
        )
        return plan

    monkeypatch.setattr(runtime_module, "apply_ml_overlay", capture)
    plan = _valid_plan_for_pipeline_test()
    runtime_module._apply_market_context(
        plan,
        {
            "asof_ts": pd.Timestamp.now(tz="UTC").timestamp(),
            "stale_age_s": 1.0,
            "sentiment_30m": 0.4,
            "earnings_phase": "CLEAR",
        },
        _ConfigStub(),
    )
    if plan.valid:
        runtime_module.apply_ml_overlay(plan)

    assert observed == {
        "context_fresh": True,
        "sentiment_30m": 0.4,
        "earnings_phase": "CLEAR",
    }


def test_bearish_market_context_blocks_long_reversal():
    from agent.scalp.models import ScalpSignalConfig, ScalpSignalPlan, SignalSide
    from agent.scalp.runtime import _apply_directional_quality_filters

    plan = ScalpSignalPlan(
        ticker="AAPL",
        side=SignalSide.LONG,
        valid=True,
        invalid_reason="",
        mtf_state="BULLISH",
        mtf_alignment="ALIGNED",
    )
    _apply_directional_quality_filters(
        plan,
        {"state": "BEARISH", "bearish_votes": 1, "bullish_votes": 0},
        ScalpSignalConfig(),
    )

    assert plan.valid is False
    assert "LONG_MARKET_BEARISH_CONTEXT" in plan.blockers


def test_bearish_five_minute_context_blocks_long_reversal():
    from agent.scalp.models import ScalpSignalConfig, ScalpSignalPlan, SignalSide
    from agent.scalp.runtime import _apply_directional_quality_filters

    plan = ScalpSignalPlan(
        ticker="AAPL",
        side=SignalSide.LONG,
        valid=True,
        invalid_reason="",
        mtf_state="BEARISH",
        mtf_alignment="CONFLICT",
    )
    _apply_directional_quality_filters(
        plan,
        {"state": "BULLISH", "bearish_votes": 0, "bullish_votes": 1},
        ScalpSignalConfig(),
    )

    assert plan.valid is False
    assert "LONG_5M_BEARISH_CONTEXT" in plan.blockers


def test_bullish_market_context_blocks_short_reversal():
    from agent.scalp.models import ScalpSignalConfig, ScalpSignalPlan, SignalSide
    from agent.scalp.runtime import _apply_directional_quality_filters

    plan = ScalpSignalPlan(
        ticker="QQQ",
        side=SignalSide.SHORT,
        valid=True,
        invalid_reason="",
        mtf_state="BEARISH",
        mtf_alignment="ALIGNED",
    )
    _apply_directional_quality_filters(
        plan,
        {"state": "BULLISH", "bearish_votes": 0, "bullish_votes": 1},
        ScalpSignalConfig(),
    )

    assert plan.valid is False
    assert "SHORT_MARKET_BULLISH_CONTEXT" in plan.blockers


def test_bullish_five_minute_context_blocks_short_reversal():
    from agent.scalp.models import ScalpSignalConfig, ScalpSignalPlan, SignalSide
    from agent.scalp.runtime import _apply_directional_quality_filters

    plan = ScalpSignalPlan(
        ticker="QQQ",
        side=SignalSide.SHORT,
        valid=True,
        invalid_reason="",
        mtf_state="BULLISH",
        mtf_alignment="CONFLICT",
    )
    _apply_directional_quality_filters(
        plan,
        {"state": "BEARISH", "bearish_votes": 1, "bullish_votes": 0},
        ScalpSignalConfig(),
    )

    assert plan.valid is False
    assert "SHORT_5M_BULLISH_CONTEXT" in plan.blockers


def test_execution_liquidity_classification_does_not_hide_monitored_plan():
    import pandas as pd

    from agent.scalp.models import QuoteSource, ScalpSignalPlan, SignalSide
    from agent.scalp.runtime import _apply_execution_liquidity

    frame = pd.DataFrame(
        {"Close": [100.0] * 40, "Volume": [1_000.0] * 40}
    )
    plan = ScalpSignalPlan(
        ticker="AAPL",
        side=SignalSide.LONG,
        valid=True,
        invalid_reason="",
        source=QuoteSource.WS,
        execution_eligible=True,
        spread_bps=45.0,
    )
    _apply_execution_liquidity(plan, frame, _ConfigStub())

    assert plan.valid is True
    assert plan.blockers == []
    assert plan.execution_eligible is False
    assert plan.execution_liquidity_qualified is False
    assert "EXECUTION_LIQUIDITY_SPREAD" in plan.execution_blockers
    assert plan.execution_median_minute_dollar_volume == 100_000.0


def test_canonical_safety_gates_remain_enabled_by_default():
    from agent.scalp.models import ScalpSignalConfig

    config = ScalpSignalConfig()
    assert config.require_rsi_zone is True
    assert config.require_macd_confirm is True
    assert config.require_vwap_event is True
    assert config.block_when_path_obstructed is True
    assert config.block_when_risk_capped is True


def test_runtime_uses_shared_quote_sla_and_stage_telemetry():
    import inspect

    from agent.scalp.runtime import ScalpRuntime

    source = inspect.getsource(ScalpRuntime.run_cycle)
    assert "execution_market_health()" in source
    assert "execution_market_health(max_age_s=2.0)" not in source
    assert '"stage_ms": stage_ms' in source


def test_live_bar_repair_is_absence_based_and_bounded():
    import inspect

    from services.market_data_service import _bar_hydration_loop

    source = inspect.getsource(_bar_hydration_loop)
    assert "require_fresh=False" in source
    assert "missing[:repair_batch_size]" in source
    assert "require_fresh=active" not in source


class _ConfigStub:
    def get(self, _key, default=None):
        return default


def _valid_plan_for_pipeline_test():
    from agent.scalp.models import ScalpSignalPlan, SignalSide

    return ScalpSignalPlan(
        ticker="AAA",
        side=SignalSide.LONG,
        valid=True,
        invalid_reason="",
    )


def test_learner_retrains_only_after_new_canonical_outcomes():
    service = (APP / "services" / "scalp_learner_service.py").read_text(
        encoding="utf-8"
    )

    assert '"scalp-learner:ml-watermark"' in service
    assert "outcome_count > trained_outcome_watermark" in service
    assert '"WAITING_FOR_NEW_OUTCOMES"' in service
    assert "_save_training_watermark(trained_outcome_watermark)" in service
    load_call = service.index("trained_outcome_watermark = _load_training_watermark")
    first_save = service.index(
        "_save_training_watermark(trained_outcome_watermark)", load_call
    )
    loop_start = service.index("while not _runner.stopped")
    assert load_call < first_save < loop_start
