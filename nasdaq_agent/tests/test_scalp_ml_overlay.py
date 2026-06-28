from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import agent.db as db
from agent.config_manager import config
from agent.scalp import (
    IndicatorSnapshot,
    QuoteSnapshot,
    QuoteSource,
    create_scalp_signal_plan,
)
from agent.scalp.ml_features import (
    FEATURE_NAMES,
    bounded_confidence_adjustment,
    expected_r,
    feature_vector,
)
from agent.scalp.ml_overlay import apply_ml_overlay
from agent.scalp.ml_trainer import TrainingDataset, _promotion_reasons, train_and_maybe_promote
from agent.scalp.store import learning_dashboard_data


class _FakeModel:
    def __init__(self, column: int):
        self.column = column

    def predict_proba(self, values):
        probability = np.where(values[:, self.column] > 0.5, 0.9, 0.1)
        return np.column_stack((1.0 - probability, probability))


def _plan(ticker="MLTEST"):
    return create_scalp_signal_plan(
        quote=QuoteSnapshot(
            ticker=ticker, last=100.0, bid=99.99, ask=100.01,
            data_age_ms=100, source=QuoteSource.WS,
        ),
        indicators=IndicatorSnapshot(
            rsi_14=25.0, rsi_7=20.0, rsi_2=8.0,
            macd_hist=-0.04, macd_hist_prev=-0.10,
            atr_14=1.0, vwap=99.8, rvol=1.2,
            vwap_event="RECLAIM", bar_age_ms=30_000,
        ),
        side="LONG", session="REGULAR", resistances=[103.0],
    )


@pytest.fixture(autouse=True)
def _restore_ml_settings():
    keys = (
        "scalp_ml.training_enabled", "scalp_ml.shadow_enabled",
        "scalp_ml.overlay_enabled", "scalp_ml.minimum_samples",
        "scalp_ml.training_lookback_days", "scalp_ml.maximum_model_age_hours",
        "scalp_ml.minimum_selected_holdout", "scalp_ml.minimum_holdout_sessions",
        "scalp_ml.minimum_session_samples", "scalp_ml.minimum_expectancy_r",
        "scalp_ml.minimum_profit_factor", "scalp_ml.minimum_session_expectancy_r",
        "scalp_ml.minimum_auc", "scalp_ml.minimum_brier_improvement",
        "scalp_ml.selection_expected_r", "scalp_ml.confidence_points_per_r",
        "scalp_ml.max_confidence_raise", "scalp_ml.max_confidence_reduction",
    )
    before = {key: config.get(key) for key in keys}
    yield
    config.set_many(before, updated_by="test_cleanup")


def test_feature_contract_ignores_every_post_trade_field():
    plan = _plan().to_dict()
    original = feature_vector(plan)
    mutated = {
        **plan,
        "tp1_hit": 1,
        "tp2_hit": 1,
        "stop_hit": 0,
        "pnl_r": 9.0,
        "pnl_dollar": 9999.0,
        "mfe_r": 8.0,
        "mae_r": -4.0,
        "exit_reason": "TARGET_T2",
    }

    assert len(original) == len(FEATURE_NAMES)
    assert feature_vector(mutated) == original


def test_feature_contract_includes_pre_entry_market_context():
    plan = _plan().to_dict()
    baseline = feature_vector(plan)
    contextual = feature_vector(
        {
            **plan,
            "context_fresh": True,
            "sentiment_30m": -0.6,
            "sentiment_velocity": -0.25,
            "news_shock": True,
            "context_risk_score": 0.75,
            "earnings_phase": "CAUTION",
            "earnings_days_away": 2,
        }
    )

    assert len(contextual) == len(FEATURE_NAMES)
    assert contextual != baseline


def test_feature_contract_includes_closed_five_minute_context_only():
    plan = _plan().to_dict()
    baseline = feature_vector(plan)
    contextual = feature_vector(
        {
            **plan,
            "mtf_state": "BULLISH",
            "mtf_alignment": "ALIGNED",
            "rsi_14_5m": 58.0,
            "macd_hist_5m": 0.04,
            "macd_slope_5m": 0.02,
            "atr_14_5m": 1.5,
            "vwap_5m": 99.5,
        }
    )

    assert len(contextual) == len(FEATURE_NAMES)
    assert contextual != baseline


def test_expected_r_matches_two_stage_bracket_math():
    assert expected_r(0.0, 0.0, 2.0) == pytest.approx(-1.0)
    assert expected_r(1.0, 0.0, 2.0) == pytest.approx(0.5)
    assert expected_r(1.0, 1.0, 2.0) == pytest.approx(1.5)


def test_overlay_changes_confidence_only(monkeypatch):
    import agent.scalp.ml_overlay as overlay

    config.set_many(
        {
            "scalp_ml.shadow_enabled": True,
            "scalp_ml.overlay_enabled": True,
            "scalp_ml.confidence_points_per_r": 5.0,
            "scalp_ml.max_confidence_raise": 5.0,
            "scalp_ml.max_confidence_reduction": 15.0,
        },
        updated_by="test",
    )
    monkeypatch.setattr(
        overlay,
        "_champion_artifact",
        lambda: {
            "version_id": "champion-1",
            "tp1_model": _FakeModel(0),
            "tp2_model": _FakeModel(0),
        },
    )
    monkeypatch.setattr(overlay, "_record_prediction", lambda _plan: None)
    plan = _plan()
    geometry = (plan.valid, plan.entry, plan.stop_loss, plan.tp1, plan.tp2, plan.learning_size_mult)
    base = plan.confidence

    apply_ml_overlay(plan)

    assert plan.ml_overlay_applied
    assert plan.ml_model_version == "champion-1"
    assert plan.confidence != base
    assert (plan.valid, plan.entry, plan.stop_loss, plan.tp1, plan.tp2, plan.learning_size_mult) == geometry


def test_invalid_plan_can_never_be_rescued_by_ml(monkeypatch):
    import agent.scalp.ml_overlay as overlay

    config.set_many(
        {"scalp_ml.shadow_enabled": True, "scalp_ml.overlay_enabled": True},
        updated_by="test",
    )
    monkeypatch.setattr(overlay, "_champion_artifact", lambda: pytest.fail("model must not load"))
    plan = _plan("INVALID")
    plan.valid = False
    plan.invalid_reason = "QUOTE_STALE"
    plan.blockers.append("QUOTE_STALE")
    original_confidence = plan.confidence

    apply_ml_overlay(plan)

    assert not plan.valid
    assert plan.confidence == original_confidence
    assert not plan.ml_overlay_applied


def test_promotion_requires_economic_and_session_stability():
    metrics = {
        "tp1_auc": 0.80,
        "tp2_auc": 0.75,
        "tp1_brier": 0.15,
        "tp2_brier": 0.16,
        "tp1_baseline_brier": 0.25,
        "tp2_baseline_brier": 0.25,
        "selected_expectancy_r": -0.01,
        "selected_profit_factor": 0.95,
        "session_metrics": [
            {"session_date": "2026-06-24", "count": 10, "expectancy_r": 0.1},
            {"session_date": "2026-06-25", "count": 10, "expectancy_r": -0.1},
        ],
    }

    reasons = _promotion_reasons(metrics, 20, config)

    assert "out-of-sample expectancy below floor" in reasons
    assert "out-of-sample profit factor below floor" in reasons
    assert "recent-session stability gate failed" in reasons


def test_chronological_challenger_promotes_only_after_all_gates(monkeypatch):
    import agent.scalp.ml_trainer as trainer

    n = 100
    tp1 = np.asarray([index % 2 for index in range(n)], dtype=np.int8)
    tp2 = np.asarray([1 if index % 4 == 1 else 0 for index in range(n)], dtype=np.int8)
    x = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    x[:, 0] = tp1
    x[:, 1] = tp2
    start = datetime(2026, 6, 20, 14, 0, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=index // 25, minutes=index % 25) for index in range(n)]
    session_dates = [
        "2026-06-22" if index < 87 else "2026-06-23"
        for index in range(n)
    ]
    dataset = TrainingDataset(
        x=x, tp1=tp1, tp2=tp2,
        pnl_r=np.where(tp1 == 1, 1.0, -1.0).astype(np.float32),
        reward_r=np.full(n, 2.0, dtype=np.float32),
        closed_at=timestamps, session_date=session_dates,
    )
    config.set_many(
        {
            "scalp_ml.minimum_samples": 50,
            "scalp_ml.minimum_selected_holdout": 10,
            "scalp_ml.minimum_holdout_sessions": 2,
            "scalp_ml.minimum_session_samples": 5,
            "scalp_ml.minimum_expectancy_r": 0.05,
            "scalp_ml.minimum_profit_factor": 1.1,
            "scalp_ml.minimum_session_expectancy_r": 0.0,
            "scalp_ml.minimum_auc": 0.52,
            "scalp_ml.minimum_brier_improvement": 0.0,
            "scalp_ml.selection_expected_r": 0.0,
        },
        updated_by="test",
    )
    models = iter((_FakeModel(0), _FakeModel(1)))
    promoted = []
    monkeypatch.setattr(trainer, "load_training_dataset", lambda: dataset)
    monkeypatch.setattr(trainer, "_fit_classifier", lambda _x, _y: next(models))
    monkeypatch.setattr(trainer, "_save_artifact", lambda version, artifact: (trainer._MODEL_DIR / f"{version}.joblib", "abc"))
    monkeypatch.setattr(trainer, "promote_ml_model", lambda metadata: promoted.append(metadata))
    monkeypatch.setattr(trainer, "record_ml_evaluation", lambda metadata: pytest.fail(metadata.get("rejection_reason")))

    result = train_and_maybe_promote()

    assert result["status"] == "CHAMPION"
    assert result["train_count"] == 75
    assert result["holdout_count"] == 25
    assert len(promoted) == 1
    assert promoted[0]["metrics"]["selected_expectancy_r"] > 0


def test_confidence_adjustment_is_asymmetric_and_bounded():
    config.set_many(
        {
            "scalp_ml.confidence_points_per_r": 20.0,
            "scalp_ml.max_confidence_raise": 5.0,
            "scalp_ml.max_confidence_reduction": 15.0,
        },
        updated_by="test",
    )
    assert bounded_confidence_adjustment(2.0) == 5.0
    assert bounded_confidence_adjustment(-2.0) == -15.0


def test_stale_champion_is_not_loaded(monkeypatch):
    import agent.scalp.ml_overlay as overlay
    import agent.scalp.store as store

    config.set("scalp_ml.maximum_model_age_hours", 1, updated_by="test")
    monkeypatch.setattr(
        store,
        "champion_ml_model",
        lambda: {
            "version_id": "stale-1",
            "created_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            "artifact_path": "must-not-be-read.joblib",
            "artifact_sha256": "none",
        },
    )
    overlay.reset_model_cache()

    assert overlay._champion_artifact() is None


def test_insufficient_dataset_is_audited_without_creating_artifact():
    config.set("scalp_ml.minimum_samples", 50, updated_by="test")

    result = train_and_maybe_promote()

    assert result["status"] == "REJECTED"
    assert "insufficient outcomes" in result["rejection_reason"]
    with db.get_conn(read_only=True) as conn:
        row = conn.execute(
            "SELECT status, artifact_path FROM scalp_ml_models WHERE version_id=?",
            (result["version_id"],),
        ).fetchone()
    assert row["status"] == "REJECTED"
    assert row["artifact_path"] == ""
    dashboard = learning_dashboard_data()
    assert dashboard["ml_champion"] is None
    assert dashboard["ml_evaluations"][0]["status"] == "REJECTED"
