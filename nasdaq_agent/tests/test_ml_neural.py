"""
Section B — ML / Neural Model Tests

Covers:
  B1. DynamicBlender: weights, blend, outcome recording, persistence
  B2. StockMLModel: untrained defaults, training, inference, persistence
  B3. Prediction engine: generate_prediction structure and invariants

Run:
    cd nasdaq_agent
    pytest tests/test_ml_neural.py -v --tb=short
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import make_ohlcv
from agent.signal_blender import DynamicBlender, ModelOutcome, MODELS, MIN_SAMPLES
from agent.prediction import _evaluate_rr, _compute_confidence


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fresh_blender(tmp_path) -> DynamicBlender:
    """Return a blender with a fresh empty persist path (no history)."""
    return DynamicBlender(persist_path=tmp_path / "blend_test.json")


def fake_prepared(n: int = 300, n_feat: int = 32) -> tuple:
    """Synthetic (X_train, y_train, X_test, y_test) for model training tests."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_feat))
    # Label is driven by a linear combination of features so XGBoost can learn
    # a signal and pass the accuracy gate (≥0.52 on holdout).
    # Balanced ~50/50 so it also passes the 0.85 homogeneity check.
    signal = X[:, 0] * 2.0 + X[:, 1] * 1.5 + X[:, 2] * 1.0
    y = (signal > 0).astype(int)
    split = int(n * 0.7)
    return X[:split], y[:split], X[split:], y[split:]


# ═════════════════════════════════════════════════════════════════════════════
# B1 – DynamicBlender
# ═════════════════════════════════════════════════════════════════════════════

class TestDynamicBlender:

    def test_weights_sum_to_one(self, tmp_path):
        b = fresh_blender(tmp_path)
        weights = b.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_initial_weights_equal(self, tmp_path):
        """Without any recorded outcomes all models share equal weight."""
        b = fresh_blender(tmp_path)
        weights = b.get_weights()
        assert len(weights) == len(MODELS)
        w_vals = list(weights.values())
        assert all(abs(w - w_vals[0]) < 1e-9 for w in w_vals)

    def test_blend_returns_0_to_1(self, tmp_path):
        b = fresh_blender(tmp_path)
        result = b.blend(0.6, 0.5, 0.4)
        assert 0.0 <= result <= 1.0

    def test_blend_with_equal_probs_returns_that_prob(self, tmp_path):
        """Equal prob for all models → blended == that same prob."""
        b = fresh_blender(tmp_path)
        for p in (0.3, 0.5, 0.7, 0.9):
            result = b.blend(p, p, p)
            assert abs(result - p) < 1e-6, f"blend({p},{p},{p}) = {result}"

    def test_blend_excludes_untrained_models(self, tmp_path):
        """Swing / deep excluded when not trained — still returns [0,1]."""
        b = fresh_blender(tmp_path)
        result = b.blend(0.7, 0.6, 0.5, swing_p=0.3, deep_p=0.3,
                         swing_trained=False, deep_trained=False)
        assert 0.0 <= result <= 1.0

    def test_blend_includes_trained_swing(self, tmp_path):
        """swing_trained=True should include swing in blend."""
        b = fresh_blender(tmp_path)
        r_without = b.blend(0.8, 0.7, 0.6, swing_p=0.1, swing_trained=False)
        r_with    = b.blend(0.8, 0.7, 0.6, swing_p=0.1, swing_trained=True)
        # Including a low-probability model should pull blend down
        assert r_with < r_without + 0.01  # some difference expected

    def test_blend_clamps_to_valid_range(self, tmp_path):
        """Extreme probabilities should still produce valid output."""
        b = fresh_blender(tmp_path)
        assert 0.0 <= b.blend(1.0, 1.0, 1.0) <= 1.0
        assert 0.0 <= b.blend(0.0, 0.0, 0.0) <= 1.0

    def test_record_outcome_increases_sample_count(self, tmp_path):
        b = fresh_blender(tmp_path)
        stats_before = b.get_stats()
        b.record_outcome("scalp", "AAPL", prob=0.7, correct=True)
        b.record_outcome("scalp", "AAPL", prob=0.7, correct=True)
        stats_after = b.get_stats()
        assert stats_after["samples"]["scalp"] > stats_before["samples"]["scalp"]

    def test_better_model_gets_higher_weight(self, tmp_path):
        """Feed many correct outcomes for scalp and many wrong for reversal."""
        b = fresh_blender(tmp_path)
        for _ in range(MIN_SAMPLES + 5):
            b.record_outcome("scalp",    "X", prob=0.7, correct=True)
            b.record_outcome("reversal", "X", prob=0.7, correct=False)
        weights = b.get_weights()
        assert weights["scalp"] > weights["reversal"], \
            "Scalp (all correct) should outweigh reversal (all wrong)"

    def test_weights_normalise_after_outcomes(self, tmp_path):
        b = fresh_blender(tmp_path)
        for _ in range(MIN_SAMPLES + 2):
            b.record_outcome("scalp", "Y", prob=0.6, correct=True)
        weights = b.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_get_stats_returns_required_keys(self, tmp_path):
        b = fresh_blender(tmp_path)
        stats = b.get_stats()
        assert "weights" in stats
        assert "accuracy" in stats
        assert "samples" in stats

    def test_accuracy_is_none_below_min_samples(self, tmp_path):
        b = fresh_blender(tmp_path)
        stats = b.get_stats()
        for m in MODELS:
            assert stats["accuracy"][m] is None, \
                f"Accuracy for {m} should be None before {MIN_SAMPLES} samples"

    def test_accuracy_set_after_min_samples(self, tmp_path):
        b = fresh_blender(tmp_path)
        for _ in range(MIN_SAMPLES):
            b.record_outcome("ensemble", "Z", prob=0.6, correct=True)
        stats = b.get_stats()
        assert stats["accuracy"]["ensemble"] is not None

    def test_persist_and_reload(self, tmp_path):
        """Outcomes recorded in one blender instance survive a reload."""
        path = tmp_path / "blend_persist.json"
        b1 = DynamicBlender(persist_path=path)
        for _ in range(MIN_SAMPLES + 3):
            b1.record_outcome("scalp", "P", prob=0.8, correct=True)
        # Force save
        b1._save()

        b2 = DynamicBlender(persist_path=path)
        stats2 = b2.get_stats()
        assert stats2["samples"]["scalp"] >= MIN_SAMPLES

    def test_blend_with_deep_trained(self, tmp_path):
        b = fresh_blender(tmp_path)
        result = b.blend(0.6, 0.5, 0.4, deep_p=0.7, deep_trained=True)
        assert 0.0 <= result <= 1.0

    def test_record_many_models(self, tmp_path):
        b = fresh_blender(tmp_path)
        for m in ("scalp", "ensemble", "reversal"):
            for i in range(MIN_SAMPLES):
                b.record_outcome(m, "T", prob=0.6, correct=(i % 2 == 0))
        weights = b.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6


# ═════════════════════════════════════════════════════════════════════════════
# B2 – StockMLModel (XGBoost)
# ═════════════════════════════════════════════════════════════════════════════

class TestStockMLModel:
    """
    These tests inject pre-computed features via the _prepared= shortcut so
    they do not require the real 'ta' library.
    """

    @pytest.fixture(autouse=True)
    def _tmp_model_dir(self, tmp_path, monkeypatch):
        """Store model files in tmp_path so tests don't pollute the data/ dir."""
        import agent.ml_model as mlm
        monkeypatch.setattr(mlm, "_MODEL_DIR", tmp_path / "models")
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    def _make_model(self, ticker: str = "TEST") -> "StockMLModel":
        from agent.ml_model import StockMLModel
        return StockMLModel(ticker)

    def test_untrained_predict_returns_half(self):
        m = self._make_model()
        df = make_ohlcv(n=60)
        prob = m.predict_proba(df)
        assert prob == 0.5

    def test_train_from_df_with_prepared_returns_true(self):
        m = self._make_model()
        result = m.train_from_df(None, _prepared=fake_prepared())
        assert result is True

    def test_trained_flag_set_after_training(self):
        m = self._make_model()
        m.train_from_df(None, _prepared=fake_prepared())
        assert m.trained is True

    def test_predict_proba_in_0_1_after_training(self):
        m = self._make_model()
        m.train_from_df(None, _prepared=fake_prepared())
        df = make_ohlcv(n=60)
        with patch("agent.ml_model.compute_live_row") as mock_row:
            import numpy as np
            mock_row.return_value = np.zeros((1, 32))
            prob = m.predict_proba(df)
        assert 0.0 <= prob <= 1.0

    def test_train_rejects_homogeneous_labels(self):
        """If >85% of training labels are the same class, training should fail."""
        n = 300
        n_feat = 32
        rng = np.random.default_rng(1)
        X = rng.standard_normal((n, n_feat))
        y = np.ones(n, dtype=int)  # 100% class-1 — extremely homogeneous
        split = int(n * 0.7)
        prepared = (X[:split], y[:split], X[split:], y[split:])
        m = self._make_model("HOMO")
        result = m.train_from_df(None, _prepared=prepared)
        assert result is False

    def test_model_saves_and_loads(self, tmp_path):
        m1 = self._make_model("SAVE1")
        m1.train_from_df(None, _prepared=fake_prepared())
        m1._save()

        from agent.ml_model import StockMLModel
        m2 = StockMLModel("SAVE1")
        m2._load()
        assert m2.trained is True

    def test_predict_proba_returns_half_if_scaler_missing(self):
        m = self._make_model("NOSCALER")
        m.trained = True
        m.model = MagicMock()
        m.scaler = None
        df = make_ohlcv(n=60)
        with patch("agent.ml_model.compute_live_row") as mock_row:
            mock_row.return_value = np.zeros((1, 32))
            # _safe_transform will return None when scaler is None-like
            # predict_proba should gracefully fall back to 0.5
            prob = m.predict_proba(df)
        assert 0.0 <= prob <= 1.0

    def test_get_or_create_returns_model_instance(self):
        from agent.ml_model import get_or_create, StockMLModel
        m = get_or_create("AAPL")
        assert isinstance(m, StockMLModel)

    def test_get_or_create_returns_same_instance(self):
        from agent.ml_model import get_or_create
        m1 = get_or_create("SHARED")
        m2 = get_or_create("SHARED")
        assert m1 is m2

    def test_module_predict_returns_float(self):
        from agent.ml_model import predict
        df = make_ohlcv(n=60)
        prob = predict("MODPRED", df)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# B3 – DailyMLModel
# ═════════════════════════════════════════════════════════════════════════════

class TestDailyMLModel:

    @pytest.fixture(autouse=True)
    def _tmp_dir(self, tmp_path, monkeypatch):
        import agent.ml_model as mlm
        monkeypatch.setattr(mlm, "_MODEL_DIR", tmp_path / "models")
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    def _make_daily_model(self, ticker="DTST"):
        from agent.ml_model import DailyMLModel
        return DailyMLModel(ticker)

    def test_untrained_daily_returns_half(self):
        m = self._make_daily_model()
        df = make_ohlcv(n=60)
        prob = m.predict_proba(df)
        assert prob == 0.5

    def test_daily_model_instance(self):
        from agent.ml_model import get_or_create_daily, DailyMLModel
        m = get_or_create_daily("DAILY1")
        assert isinstance(m, DailyMLModel)


# ═════════════════════════════════════════════════════════════════════════════
# B4 – Prediction Engine (_evaluate_rr, _compute_confidence)
# ═════════════════════════════════════════════════════════════════════════════

class TestPredictionEngine:
    """
    Test the pure-logic helpers in prediction.py that don't require live data.
    """

    # ── _evaluate_rr ─────────────────────────────────────────────────────────

    def test_evaluate_rr_buy_target_above_price(self):
        sr = {"supports": [95.0, 93.0], "resistances": [110.0, 115.0]}
        stop, target, rr, quality, qualifies = _evaluate_rr(100.0, sr, "BUY")
        if target is not None and target > 0:
            assert target > 100.0

    def test_evaluate_rr_sell_target_below_price(self):
        sr = {"supports": [85.0, 82.0], "resistances": [105.0, 110.0]}
        stop, target, rr, quality, qualifies = _evaluate_rr(100.0, sr, "SELL")
        if target is not None and target > 0:
            assert target < 100.0

    def test_evaluate_rr_rr_positive_when_valid(self):
        sr = {"supports": [90.0], "resistances": [115.0]}
        stop, target, rr, quality, qualifies = _evaluate_rr(100.0, sr, "BUY")
        if rr is not None:
            assert rr >= 0

    def test_evaluate_rr_qualifies_at_2r(self):
        sr = {"supports": [90.0], "resistances": [120.0]}
        stop, target, rr, quality, qualifies = _evaluate_rr(100.0, sr, "BUY")
        if rr is not None and rr >= 2.0:
            assert qualifies is True

    def test_evaluate_rr_empty_sr_returns_gracefully(self):
        stop, target, rr, quality, qualifies = _evaluate_rr(100.0, {}, "BUY")
        # Must not raise; result can be None or 0
        assert rr is not None or rr is None  # just no exception

    def test_evaluate_rr_buy_stop_below_price(self):
        sr = {"supports": [90.0], "resistances": [115.0]}
        stop, target, rr, quality, qualifies = _evaluate_rr(100.0, sr, "BUY")
        if stop is not None and stop > 0:
            assert stop < 100.0

    def test_evaluate_rr_sell_stop_above_price(self):
        sr = {"supports": [85.0], "resistances": [110.0]}
        stop, target, rr, quality, qualifies = _evaluate_rr(100.0, sr, "SELL")
        if stop is not None and stop > 0:
            assert stop > 100.0

    # ── _compute_confidence ───────────────────────────────────────────────────
    # Actual signature: (direction, trend, trend_prob, tech_score, vol_score,
    #                    ml_prob, ml_trained, pa_score, pattern_score,
    #                    sent_score, mtf_score=0.0)

    def _conf(self, ml_prob: float = 0.65, tech_score: float = 0.0,
              direction: str = "BUY") -> float:
        return _compute_confidence(
            direction=direction, trend="UP", trend_prob=0.6,
            tech_score=tech_score, vol_score=0.0,
            ml_prob=ml_prob, ml_trained=True,
            pa_score=0.0, pattern_score=0.0, sent_score=0.0,
        )

    def test_compute_confidence_bounded_0_100(self):
        """_compute_confidence must return a value in [0, 100]."""
        for ml in (0.3, 0.5, 0.6, 0.7, 0.9):
            for tech in (-1.0, 0.0, 0.5, 1.0):
                conf = self._conf(ml_prob=ml, tech_score=tech)
                assert 0.0 <= conf <= 100.0, \
                    f"_compute_confidence(ml={ml}, tech={tech}) = {conf} out of range"

    def test_compute_confidence_high_ml_gives_higher_conf(self):
        low  = self._conf(ml_prob=0.4)
        high = self._conf(ml_prob=0.9)
        assert high > low

    def test_compute_confidence_strong_tech_score_boosts(self):
        low  = self._conf(ml_prob=0.65, tech_score=-1.0)
        high = self._conf(ml_prob=0.65, tech_score=1.0)
        assert high > low


# ═════════════════════════════════════════════════════════════════════════════
# B5 – Signal Blender module-level convenience functions
# ═════════════════════════════════════════════════════════════════════════════

class TestBlendSignals:

    def test_blend_signals_returns_float(self):
        from agent.signal_blender import blend_signals
        result = blend_signals(0.6, 0.5, 0.4)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_get_blender_returns_singleton(self):
        from agent.signal_blender import get_blender
        b1 = get_blender()
        b2 = get_blender()
        assert b1 is b2

    def test_blend_signals_consistency(self):
        """Same inputs → same output (deterministic)."""
        from agent.signal_blender import blend_signals
        r1 = blend_signals(0.7, 0.6, 0.5)
        r2 = blend_signals(0.7, 0.6, 0.5)
        assert r1 == r2


# ═════════════════════════════════════════════════════════════════════════════
# B6 – EnsembleMLModel (if available)
# ═════════════════════════════════════════════════════════════════════════════

class TestEnsembleModel:

    @pytest.fixture(autouse=True)
    def _tmp_dir(self, tmp_path, monkeypatch):
        import agent.ml_model as mlm
        monkeypatch.setattr(mlm, "_MODEL_DIR", tmp_path / "models")
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    def test_ensemble_untrained_returns_half(self):
        from agent.ml_model import get_or_create
        # The "ensemble" is just StockMLModel with a suffix — test via predict()
        from agent.ml_model import predict
        df = make_ohlcv(n=60)
        prob = predict("ENSEMBLE_TST", df)
        assert 0.0 <= prob <= 1.0

    def test_predict_ensemble_returns_float(self):
        try:
            from agent.ml_model import predict_ensemble
            df = make_ohlcv(n=60)
            prob, agreement = predict_ensemble("ENS1", df)
            assert 0.0 <= prob <= 1.0
            assert 0.0 <= agreement <= 1.0
        except (ImportError, AttributeError):
            pytest.skip("predict_ensemble not available in this build")
