from __future__ import annotations

import pytest

from agent.scalp.models import QuoteSource, ScalpSignalPlan, SignalSide


def _plan(**overrides) -> ScalpSignalPlan:
    values = {
        "ticker": "TEST",
        "side": SignalSide.LONG,
        "valid": True,
        "invalid_reason": "",
        "entry": 100.0,
        "stop_loss": 99.0,
        "tp1": 101.0,
        "tp2": 102.0,
        "risk_per_share": 1.0,
        "reward_r": 2.0,
        "rr_ratio": 2.0,
        "spread_to_risk": 0.05,
        "rvol": 2.5,
        "atr_bucket": "NORMAL",
        "rsi_zone": "EXTREME_OS",
        "vwap_event": "RECLAIM",
        "setup_type": "OVERSOLD_MACD_TURN_LONG",
        "session": "STANDARD",
        "source": QuoteSource.WS,
        "data_age_ms": 100,
    }
    values.update(overrides)
    return ScalpSignalPlan(**values)


def _config(**overrides):
    values = {
        "scalp.entry_quality_min_score": 65.0,
        "scalp.entry_quality_min_score_extended": 70.0,
        "scalp.entry_confirmation_enabled": True,
        "scalp.entry_confirmation_seconds": 15.0,
        "scalp.entry_confirmation_min_observations": 3,
        "scalp.entry_confirmation_max_chase_r": 0.25,
    }
    values.update(overrides)
    return values


def _health():
    return {
        "status": "LIVE",
        "auth_required": False,
        "token_states": {"trader": "OK", "marketdata": "OK"},
    }


def test_entry_quality_score_separates_strong_and_weak_tp1_evidence():
    from agent.scalp.entry_quality import assess_entry_quality

    strong = assess_entry_quality(_plan(), _config())
    weak = assess_entry_quality(
        _plan(
            spread_to_risk=0.0588,
            rvol=1.19,
            atr_bucket="LOW",
            rsi_zone="OS",
        ),
        _config(),
    )

    assert strong.entry_quality_score == 94.0
    assert strong.entry_quality_gate == "PASS"
    assert weak.entry_quality_score == 63.0
    assert weak.entry_quality_gate == "BELOW_MINIMUM"
    assert weak.confidence == 63.0


def test_entry_quality_rewards_aligned_mtf_and_penalizes_mixed_context():
    from agent.scalp.entry_quality import assess_entry_quality

    aligned_plan = _plan()
    aligned_plan.mtf_alignment = "ALIGNED"
    mixed_plan = _plan()
    mixed_plan.mtf_alignment = "MIXED"

    aligned = assess_entry_quality(aligned_plan, _config())
    mixed = assess_entry_quality(mixed_plan, _config())

    assert aligned.entry_quality_score == 100.0
    assert mixed.entry_quality_score == 86.0
    assert "MTF_ALIGNED" in aligned.entry_quality_reasons
    assert "MTF_MIXED_PENALTY" in mixed.entry_quality_reasons


def test_extended_session_uses_stricter_quality_floor():
    from agent.scalp.entry_quality import assess_entry_quality

    plan = assess_entry_quality(
        _plan(session="AFTER_HOURS"),
        _config(),
    )

    assert plan.entry_quality_min_score == 70.0


def test_confirmation_requires_stability_and_does_not_chase():
    from agent.scalp.entry_quality import confirmation_ready

    pending = {}
    plan = _plan()
    assert confirmation_ready(
        plan, bar_id=100, pending=pending, config=_config(), now=10.0
    ) is False
    assert confirmation_ready(
        plan, bar_id=100, pending=pending, config=_config(), now=20.0
    ) is False
    assert confirmation_ready(
        plan, bar_id=100, pending=pending, config=_config(), now=25.0
    ) is True
    assert plan.entry_confirmation_state == "CONFIRMED"
    assert plan.entry_confirmation_observations == 3

    chased = _plan(ticker="CHASE")
    assert confirmation_ready(
        chased, bar_id=101, pending=pending, config=_config(), now=30.0
    ) is False
    chased.entry = 100.30
    assert confirmation_ready(
        chased, bar_id=101, pending=pending, config=_config(), now=46.0
    ) is False
    assert chased.entry_confirmation_state == "RESET_CHASE"


def test_policy_rejects_low_reachability_score(monkeypatch):
    from agent.config_manager import config
    from agent.scalp.entry_quality import assess_entry_quality
    from agent.scalp.execution_policy import evaluate_execution_policy

    monkeypatch.setitem(config._cache, "paper.enforce_risk_controls", False)
    monkeypatch.setitem(config._cache, "scalp.entry_quality_gate_enabled", True)
    plan = assess_entry_quality(
        _plan(rvol=1.19, atr_bucket="LOW", rsi_zone="OS"),
        _config(),
    )

    decision = evaluate_execution_policy(
        plan, mode="SHADOW", market_health=_health()
    )

    assert decision.allowed is False
    assert decision.reason == "TP1_REACH_SCORE_BELOW_MINIMUM"


def test_negative_empirical_context_is_probe_only_in_shadow(monkeypatch):
    from agent.config_manager import config
    from agent.scalp.entry_quality import assess_entry_quality
    from agent.scalp.execution_policy import evaluate_execution_policy

    for key, value in {
        "paper.enforce_risk_controls": False,
        "scalp.entry_quality_gate_enabled": True,
        "scalp.entry_quality_empirical_gate_enabled": True,
        "scalp.entry_quality_empirical_min_samples": 10,
        "scalp.entry_quality_min_empirical_expectancy_r": 0.0,
        "scalp.entry_quality_shadow_probe_size_mult": 0.10,
        "scalp_runtime.standard_size_mult": 0.80,
    }.items():
        monkeypatch.setitem(config._cache, key, value)
    plan = assess_entry_quality(_plan(), _config())
    plan.learning_sample_count = 15
    plan.learning_mean_expectancy_r = -0.20
    plan.learning_context_scope = "SETUP_SESSION"

    shadow = evaluate_execution_policy(
        plan, mode="SHADOW", market_health=_health()
    )
    paper = evaluate_execution_policy(
        plan, mode="PAPER", market_health=_health()
    )

    assert shadow.allowed is True
    assert shadow.size_mult == pytest.approx(0.08)
    assert shadow.checks["empirical_probe"] is True
    assert paper.allowed is False
    assert paper.reason == "EMPIRICAL_EXPECTANCY_BELOW_MINIMUM"


def test_validated_ml_expected_r_is_a_downside_gate(monkeypatch):
    from agent.config_manager import config
    from agent.scalp.entry_quality import assess_entry_quality
    from agent.scalp.execution_policy import evaluate_execution_policy

    monkeypatch.setitem(config._cache, "paper.enforce_risk_controls", False)
    monkeypatch.setitem(
        config._cache, "scalp.entry_quality_require_positive_ml_ev", True
    )
    monkeypatch.setitem(
        config._cache, "scalp.entry_quality_min_ml_expected_r", 0.05
    )
    plan = assess_entry_quality(_plan(), _config())
    plan.ml_model_version = "champion-v1"
    plan.ml_expected_r = -0.10

    decision = evaluate_execution_policy(
        plan, mode="SHADOW", market_health=_health()
    )

    assert decision.allowed is False
    assert decision.reason == "ML_EXPECTED_R_BELOW_MINIMUM"


def test_pre_tp1_failure_circuit_counts_timeouts(monkeypatch):
    from agent.config_manager import config
    from agent.scalp.learning import BLOCK, SIZE_REDUCE, _decide_gate

    for key, value in {
        "scalp_learn.enabled": True,
        "scalp_learn.pre_tp1_failure_circuit_enabled": True,
        "scalp_learn.pre_tp1_failure_count": 2,
        "scalp_learn.pre_tp1_failure_window_min": 120,
        "scalp_learn.pre_tp1_failure_size_mult": 0.25,
        "scalp_learn.setup_session_pre_tp1_block_enabled": True,
    }.items():
        monkeypatch.setitem(config._cache, key, value)

    exact = _decide_gate(
        2,
        0.25,
        -0.20,
        context_key="OVERSOLD|LONG|STANDARD",
        pre_tp1_failures=2,
    )
    broad = _decide_gate(
        2,
        0.25,
        -0.20,
        context_key="SETUP_SESSION|OVERSOLD|LONG|STANDARD",
        pre_tp1_failures=2,
    )

    assert exact[0] == SIZE_REDUCE
    assert exact[2] == pytest.approx(0.25)
    assert broad[0] == BLOCK
