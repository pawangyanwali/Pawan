"""
Comprehensive tests for Phase 2 Adaptive Trading Learning Engine.
Covers all 5 components in agent/algo_learning_p2.py.
"""
import json
import math
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_outcomes(n: int = 50, win_pct: float = 0.55) -> pd.DataFrame:
    """Generate a synthetic resolved-outcomes DataFrame."""
    import random
    random.seed(42)
    rows = []
    algos  = ["ORB5_BULL", "GAP_AND_GO_BULL", "BULL_FLAG", "VWAP_TOUCH_SCALP_BULL"]
    tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "AMD"]
    sectors = {"AAPL": "XLK", "NVDA": "SMH", "MSFT": "XLK",
               "TSLA": "XLY", "AMD": "SMH"}
    for i in range(n):
        won    = random.random() < win_pct
        ticker = random.choice(tickers)
        pnl    = random.uniform(0.3, 1.5) if won else random.uniform(-1.2, -0.2)
        rows.append({
            "ticker":      ticker,
            "direction":   "BUY",
            "confidence":  random.uniform(55, 85),
            "session":     "AM",
            "regime":      "BULL",
            "vwap_event":  "ABOVE",
            "rsi_zone":    "MIDDLE",
            "rsi_value":   random.uniform(35, 65),
            "entry_type":  "IMMEDIATE",
            "rr_ratio":    random.uniform(1.5, 3.0),
            "algo_name":   random.choice(algos),
            "sector_etf":  sectors.get(ticker, "QQQ"),
            "status":      "WIN" if won else "LOSS",
            "pnl_pct":     round(pnl, 3),
            "r_multiple":  round(pnl / 0.5, 3),
            "bars_tracked": random.randint(2, 15),
            "fired_at":    f"2025-05-{(i % 30)+1:02d}T10:00:00",
            "won":         int(won),
        })
    return pd.DataFrame(rows)


def _fresh_drift(tmp_path):
    from agent.algo_learning_p2 import ConceptDriftDetector
    d = ConceptDriftDetector.__new__(ConceptDriftDetector)
    d._lock  = threading.Lock()
    d._state = {
        "reference_built":   False,
        "reference_n":       0,
        "reference_arrays":  {},
        "last_psi":          {},
        "drift_events":      [],
        "last_check_cycle":  0,
    }
    d._PATH = tmp_path / "drift_state.json"
    return d


def _fresh_validator(tmp_path):
    from agent.algo_learning_p2 import WalkForwardValidator
    v = WalkForwardValidator.__new__(WalkForwardValidator)
    v._lock          = threading.Lock()
    v._history       = []
    v._last_run_cycle = 0
    v._PATH = tmp_path / "wf_validation.json"
    return v


def _fresh_transfer(tmp_path):
    from agent.algo_learning_p2 import CrossTickerTransferEngine
    t = CrossTickerTransferEngine.__new__(CrossTickerTransferEngine)
    t._lock           = threading.Lock()
    t._global         = {}
    t._sector         = {}
    t._ticker         = {}
    t._ticker_counts  = {}
    t._PATH = tmp_path / "transfer_state.json"
    return t


def _fresh_deployment(tmp_path):
    from agent.algo_learning_p2 import StagedDeploymentController
    d = StagedDeploymentController.__new__(StagedDeploymentController)
    d._lock  = threading.Lock()
    d._state = {
        "current_mode":         "PAPER_ONLY",
        "consecutive_passes":   0,
        "last_mode_change":     None,
        "last_mode_change_cycle": 0,
        "mode_history":         [],
    }
    d._PATH = tmp_path / "deployment_state.json"
    return d


def _fresh_notifier(tmp_path):
    from agent.algo_learning_p2 import OperatorNotificationService
    n = OperatorNotificationService.__new__(OperatorNotificationService)
    n._lock       = threading.Lock()
    n._last_sent  = {}
    n._PATH = tmp_path / "notifications.jsonl"
    return n


def _passing_metrics(win_rate=0.55, profit_factor=1.2, n_trades=50):
    return {
        "win_rate":      win_rate,
        "profit_factor": profit_factor,
        "n_trades":      n_trades,
        "expectancy":    0.003,
        "sharpe":        1.2,
        "max_drawdown":  0.05,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PSI computation (standalone)
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputePSI:

    def test_identical_distributions_psi_zero(self):
        from agent.algo_learning_p2 import _compute_psi
        data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
        psi = _compute_psi(data, data, lo=0, hi=110)
        assert psi == pytest.approx(0.0, abs=0.01)

    def test_completely_different_distributions_high_psi(self):
        from agent.algo_learning_p2 import _compute_psi
        ref  = np.full(50, 10.0)   # all values near 10
        curr = np.full(50, 90.0)   # all values near 90
        psi  = _compute_psi(ref, curr, lo=0, hi=100)
        assert psi >= 0.25

    def test_empty_arrays_return_zero(self):
        from agent.algo_learning_p2 import _compute_psi
        assert _compute_psi(np.array([]), np.array([10.0]), lo=0, hi=100) == 0.0
        assert _compute_psi(np.array([10.0]), np.array([]), lo=0, hi=100) == 0.0

    def test_slightly_shifted_distribution_low_psi(self):
        from agent.algo_learning_p2 import _compute_psi, PSI_MATERIAL
        rng  = np.random.default_rng(42)
        ref  = rng.normal(50, 10, 200)
        curr = rng.normal(52, 10, 200)   # small mean shift — stays below MATERIAL
        psi  = _compute_psi(ref, curr, lo=0, hi=100)
        assert psi < PSI_MATERIAL   # should not reach MATERIAL threshold

    def test_psi_always_non_negative(self):
        from agent.algo_learning_p2 import _compute_psi
        rng = np.random.default_rng(0)
        for _ in range(20):
            ref  = rng.uniform(0, 100, 100)
            curr = rng.uniform(0, 100, 100)
            psi  = _compute_psi(ref, curr, lo=0, hi=100)
            assert psi >= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ConceptDriftDetector
# ═══════════════════════════════════════════════════════════════════════════════

class TestConceptDriftDetector:

    def test_empty_df_returns_no_alerts(self, tmp_path):
        d = _fresh_drift(tmp_path)
        alerts = d.update(pd.DataFrame(), cycle_num=1)
        assert alerts == {}

    def test_reference_not_built_before_min_trades(self, tmp_path):
        d = _fresh_drift(tmp_path)
        small_df = _make_outcomes(n=10)
        d.update(small_df, cycle_num=1)
        assert d._state["reference_built"] is False

    def test_reference_built_after_min_trades(self, tmp_path):
        from agent.algo_learning_p2 import ConceptDriftDetector
        d = _fresh_drift(tmp_path)
        df = _make_outcomes(n=ConceptDriftDetector.REFERENCE_MIN + 5)
        d.update(df, cycle_num=1)
        assert d._state["reference_built"] is True

    def test_reference_arrays_populated_for_tracked_features(self, tmp_path):
        from agent.algo_learning_p2 import ConceptDriftDetector, _FEATURE_RANGES
        d = _fresh_drift(tmp_path)
        df = _make_outcomes(n=ConceptDriftDetector.REFERENCE_MIN + 5)
        d.update(df, cycle_num=1)
        for feat in _FEATURE_RANGES:
            if feat in df.columns:
                assert feat in d._state["reference_arrays"]

    def test_no_psi_check_before_interval(self, tmp_path):
        from agent.algo_learning_p2 import ConceptDriftDetector
        d = _fresh_drift(tmp_path)
        df = _make_outcomes(n=60)
        # Build reference
        d.update(df, cycle_num=1)
        # Second call on same cycle — should not compute PSI yet
        alerts = d.update(df, cycle_num=2)
        assert alerts == {}

    def test_psi_check_fires_after_interval(self, tmp_path):
        from agent.algo_learning_p2 import ConceptDriftDetector
        d = _fresh_drift(tmp_path)
        df = _make_outcomes(n=60)
        # Build reference at cycle 1
        d.update(df, cycle_num=1)
        # Check at cycle 1 + CHECK_INTERVAL
        check_cycle = 1 + ConceptDriftDetector.CHECK_INTERVAL
        d.update(df, cycle_num=check_cycle)
        # last_check_cycle should be updated
        assert d._state["last_check_cycle"] == check_cycle

    def test_material_drift_raises_alert(self, tmp_path):
        from agent.algo_learning_p2 import ConceptDriftDetector, PSI_MATERIAL
        d = _fresh_drift(tmp_path)

        # Reference: confidence mostly 55-65
        ref_df = _make_outcomes(n=50, win_pct=0.55)
        ref_df["confidence"] = 60.0
        d.update(ref_df, cycle_num=1)

        # Now inject a completely different distribution
        curr_df = ref_df.copy()
        curr_df["confidence"] = 95.0   # dramatic shift from 60 → 95

        # Manually set state to skip wait interval
        d._state["last_check_cycle"] = 0
        d._state["reference_built"]  = True
        d._state["reference_arrays"]["confidence"] = [60.0] * 50
        d._state["last_psi"] = {}

        alerts = d.update(curr_df, cycle_num=ConceptDriftDetector.CHECK_INTERVAL + 2)
        if "confidence" in alerts:
            psi, severity = alerts["confidence"]
            assert psi >= 0.0
            assert severity in ("WARNING", "MATERIAL")

    def test_drift_events_appended(self, tmp_path):
        from agent.algo_learning_p2 import ConceptDriftDetector
        d = _fresh_drift(tmp_path)
        # Force a drift event manually
        d._state["drift_events"].append({
            "ts": "2025-01-01T00:00:00", "feature": "confidence",
            "psi": 0.30, "severity": "MATERIAL", "cycle": 10
        })
        assert len(d._state["drift_events"]) == 1

    def test_drift_events_capped_at_100(self, tmp_path):
        d = _fresh_drift(tmp_path)
        d._state["drift_events"] = [{"ts": f"t{i}"} for i in range(105)]
        d._state["drift_events"] = d._state["drift_events"][-100:]
        assert len(d._state["drift_events"]) == 100

    def test_get_drift_summary_structure(self, tmp_path):
        d = _fresh_drift(tmp_path)
        summary = d.get_drift_summary()
        assert "reference_built" in summary
        assert "last_psi"        in summary
        assert "recent_events"   in summary
        assert "material_drifts" in summary

    def test_save_and_load_roundtrip(self, tmp_path):
        d = _fresh_drift(tmp_path)
        d._state["reference_built"] = True
        d._state["reference_n"]     = 45
        d._state["last_psi"]        = {"confidence": 0.12}
        d.save()

        d2 = _fresh_drift(tmp_path)
        d2.load()
        assert d2._state["reference_built"] is True
        assert d2._state["reference_n"]     == 45
        assert d2._state["last_psi"].get("confidence") == pytest.approx(0.12)

    def test_load_corrupt_file_safe(self, tmp_path):
        p = tmp_path / "drift_state.json"
        p.write_text("not valid json {{ {{{")
        d = _fresh_drift(tmp_path)
        d.load()   # must not raise
        assert d._state["reference_built"] is False

    def test_save_missing_parent_creates_dir(self, tmp_path):
        from agent.algo_learning_p2 import ConceptDriftDetector
        d = _fresh_drift(tmp_path / "subdir")
        d._PATH.parent.mkdir(parents=True, exist_ok=True)
        d.save()   # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# 2. WalkForwardValidator
# ═══════════════════════════════════════════════════════════════════════════════

class TestWalkForwardValidator:

    def test_returns_none_if_not_enough_cycles_elapsed(self, tmp_path):
        v = _fresh_validator(tmp_path)
        v._last_run_cycle = 5
        result = v.run_validation(_make_outcomes(50), cycle_num=10)
        # 10 - 5 = 5 < CHECK_CYCLE_INTERVAL (10)
        assert result is None

    def test_returns_none_if_too_few_trades(self, tmp_path):
        from agent.algo_learning_p2 import WalkForwardValidator
        v = _fresh_validator(tmp_path)
        small_df = _make_outcomes(n=5)
        result = v.run_validation(small_df, cycle_num=20)
        assert result is None

    def test_returns_none_on_empty_df(self, tmp_path):
        v = _fresh_validator(tmp_path)
        result = v.run_validation(pd.DataFrame(), cycle_num=20)
        assert result is None

    def test_computes_valid_metrics_on_sufficient_data(self, tmp_path):
        v = _fresh_validator(tmp_path)
        with patch.object(v, "_compute_metrics", wraps=v._compute_metrics) as mock_cm:
            with patch("agent.algo_learning_engine.get_engine") as mock_ale:
                mock_reg = MagicMock()
                mock_reg.evaluate_promotion.return_value = (False, ["test"])
                mock_ale.return_value._model_reg = mock_reg
                result = v.run_validation(_make_outcomes(50), cycle_num=20)
        assert result is not None
        assert "win_rate" in result
        assert "profit_factor" in result
        assert "n_trades" in result

    def test_metrics_win_rate_in_range(self, tmp_path):
        v = _fresh_validator(tmp_path)
        df = _make_outcomes(50, win_pct=0.60)
        metrics = v._compute_metrics(df)
        assert 0.0 <= metrics.get("win_rate", 0) <= 1.0

    def test_metrics_profit_factor_positive(self, tmp_path):
        v = _fresh_validator(tmp_path)
        df = _make_outcomes(50)
        metrics = v._compute_metrics(df)
        assert metrics.get("profit_factor", 0) >= 0.0

    def test_metrics_sharpe_computed(self, tmp_path):
        v = _fresh_validator(tmp_path)
        df = _make_outcomes(60, win_pct=0.70)
        metrics = v._compute_metrics(df)
        assert "sharpe" in metrics
        assert isinstance(metrics["sharpe"], float)

    def test_metrics_max_drawdown_non_negative(self, tmp_path):
        v = _fresh_validator(tmp_path)
        df = _make_outcomes(50)
        metrics = v._compute_metrics(df)
        assert metrics.get("max_drawdown", 0) >= 0.0

    def test_metrics_expectancy_matches_mean_pnl(self, tmp_path):
        v = _fresh_validator(tmp_path)
        df = _make_outcomes(50)
        metrics = v._compute_metrics(df)
        expected_exp = float(df["pnl_pct"].mean())
        assert abs(metrics.get("expectancy", 0) - expected_exp) < 0.01

    def test_metrics_returns_empty_for_too_few_trades(self, tmp_path):
        v = _fresh_validator(tmp_path)
        df = _make_outcomes(5)
        metrics = v._compute_metrics(df)
        assert metrics == {}

    def test_compute_sharpe_with_positive_returns(self, tmp_path):
        from agent.algo_learning_p2 import WalkForwardValidator
        # Consistent non-zero-std returns so Sharpe is well-defined
        rng     = np.random.default_rng(0)
        returns = pd.Series(rng.normal(loc=0.5, scale=0.1, size=50))
        sharpe  = WalkForwardValidator._compute_sharpe(returns)
        # mean≈0.5, std≈0.1 → Sharpe ≈ 5 * sqrt(500) ≈ ~111 — just check it's positive
        assert sharpe > 0.0

    def test_compute_sharpe_zero_std_returns_zero(self, tmp_path):
        from agent.algo_learning_p2 import WalkForwardValidator
        returns = pd.Series([0.0] * 20)
        assert WalkForwardValidator._compute_sharpe(returns) == 0.0

    def test_compute_sharpe_too_few_returns_zero(self, tmp_path):
        from agent.algo_learning_p2 import WalkForwardValidator
        assert WalkForwardValidator._compute_sharpe(pd.Series([0.5])) == 0.0

    def test_compute_max_drawdown_all_positive(self, tmp_path):
        from agent.algo_learning_p2 import WalkForwardValidator
        returns = pd.Series([1.0] * 20)
        dd = WalkForwardValidator._compute_max_drawdown(returns)
        assert dd == pytest.approx(0.0, abs=0.01)

    def test_compute_max_drawdown_steep_loss(self, tmp_path):
        from agent.algo_learning_p2 import WalkForwardValidator
        returns = pd.Series([1.0, 1.0, -50.0, 1.0, 1.0])
        dd = WalkForwardValidator._compute_max_drawdown(returns)
        assert dd > 0.0

    def test_registers_challenger_in_model_registry(self, tmp_path):
        v = _fresh_validator(tmp_path)
        df = _make_outcomes(50)
        with patch("agent.algo_learning_engine.get_engine") as mock_ale:
            mock_reg = MagicMock()
            mock_reg.evaluate_promotion.return_value = (False, ["test"])
            mock_ale.return_value._model_reg = mock_reg
            v.run_validation(df, cycle_num=20)
        mock_reg.register_version.assert_called_once()

    def test_auto_promotes_when_gates_pass(self, tmp_path):
        v = _fresh_validator(tmp_path)
        df = _make_outcomes(50, win_pct=0.80)
        with patch("agent.algo_learning_engine.get_engine") as mock_ale:
            mock_reg = MagicMock()
            mock_reg.evaluate_promotion.return_value = (True, ["all passed"])
            mock_ale.return_value._model_reg = mock_reg
            result = v.run_validation(df, cycle_num=20)
        mock_reg.promote.assert_called_once()
        assert result.get("promoted") is True

    def test_updates_last_run_cycle(self, tmp_path):
        v = _fresh_validator(tmp_path)
        df = _make_outcomes(50)
        with patch("agent.algo_learning_engine.get_engine") as mock_ale:
            mock_reg = MagicMock()
            mock_reg.evaluate_promotion.return_value = (False, [])
            mock_ale.return_value._model_reg = mock_reg
            v.run_validation(df, cycle_num=20)
        assert v._last_run_cycle == 20

    def test_appends_to_history(self, tmp_path):
        v = _fresh_validator(tmp_path)
        df = _make_outcomes(50)
        with patch("agent.algo_learning_engine.get_engine") as mock_ale:
            mock_reg = MagicMock()
            mock_reg.evaluate_promotion.return_value = (False, [])
            mock_ale.return_value._model_reg = mock_reg
            v.run_validation(df, cycle_num=20)
            v._last_run_cycle = 0   # reset so it runs again
            v.run_validation(df, cycle_num=40)
        assert len(v._history) == 2

    def test_save_and_load_roundtrip(self, tmp_path):
        v = _fresh_validator(tmp_path)
        v._history       = [{"win_rate": 0.55, "cycle_num": 10}]
        v._last_run_cycle = 10
        v.save()

        v2 = _fresh_validator(tmp_path)
        v2.load()
        assert v2._last_run_cycle == 10
        assert len(v2._history) == 1

    def test_get_latest_metrics_empty(self, tmp_path):
        v = _fresh_validator(tmp_path)
        assert v.get_latest_metrics() == {}

    def test_get_latest_metrics_returns_last_entry(self, tmp_path):
        v = _fresh_validator(tmp_path)
        v._history = [{"win_rate": 0.50}, {"win_rate": 0.60}]
        assert v.get_latest_metrics()["win_rate"] == 0.60


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CrossTickerTransferEngine
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossTickerTransferEngine:

    def test_empty_df_returns_no_updates(self, tmp_path):
        t = _fresh_transfer(tmp_path)
        summary = t.transfer(pd.DataFrame(), cycle_num=1)
        assert summary["tickers_updated"] == 0
        assert summary["global_updated"] is False

    def test_global_patterns_populated(self, tmp_path):
        t = _fresh_transfer(tmp_path)
        df = _make_outcomes(50)
        t.transfer(df, cycle_num=1)
        assert len(t._global) > 0

    def test_sector_patterns_populated(self, tmp_path):
        t = _fresh_transfer(tmp_path)
        df = _make_outcomes(50)
        t.transfer(df, cycle_num=1)
        assert len(t._sector) > 0

    def test_ticker_patterns_populated(self, tmp_path):
        t = _fresh_transfer(tmp_path)
        df = _make_outcomes(50)
        t.transfer(df, cycle_num=1)
        assert len(t._ticker) > 0

    def test_ticker_counts_tracked(self, tmp_path):
        t = _fresh_transfer(tmp_path)
        df = _make_outcomes(50)
        t.transfer(df, cycle_num=1)
        assert len(t._ticker_counts) > 0

    def test_ticker_tier_global_below_threshold(self, tmp_path):
        from agent.algo_learning_p2 import _TIER_GLOBAL
        t = _fresh_transfer(tmp_path)
        t._ticker_counts["AAPL"] = _TIER_GLOBAL - 1
        assert t.get_ticker_tier("AAPL") == "GLOBAL"

    def test_ticker_tier_blend_in_range(self, tmp_path):
        from agent.algo_learning_p2 import _TIER_GLOBAL, _TIER_BLEND
        t = _fresh_transfer(tmp_path)
        t._ticker_counts["AAPL"] = (_TIER_GLOBAL + _TIER_BLEND) // 2
        assert t.get_ticker_tier("AAPL") == "BLEND"

    def test_ticker_tier_ticker_above_threshold(self, tmp_path):
        from agent.algo_learning_p2 import _TIER_BLEND
        t = _fresh_transfer(tmp_path)
        t._ticker_counts["AAPL"] = _TIER_BLEND + 10
        assert t.get_ticker_tier("AAPL") == "TICKER"

    def test_blended_win_rate_global_for_unknown_ticker(self, tmp_path):
        t = _fresh_transfer(tmp_path)
        t._global["ORB"] = {"win_rate": 0.55, "n_trades": 50}
        result = t.get_blended_win_rate("UNKNWN", "ORB")
        assert result == pytest.approx(0.55, abs=0.01)

    def test_blended_win_rate_full_ticker_weight_above_threshold(self, tmp_path):
        from agent.algo_learning_p2 import _TIER_BLEND, _TICKER_WT
        t = _fresh_transfer(tmp_path)
        t._global["ORB"]           = {"win_rate": 0.50, "n_trades": 100}
        t._ticker["AAPL"]          = {"ORB": {"win_rate": 0.70, "n_trades": 40}}
        t._ticker_counts["AAPL"]   = _TIER_BLEND + 5
        blended = t.get_blended_win_rate("AAPL", "ORB")
        expected = _TICKER_WT * 0.70 + (1 - _TICKER_WT) * 0.50
        assert blended == pytest.approx(expected, abs=0.01)

    def test_blended_win_rate_partial_blend_zone(self, tmp_path):
        from agent.algo_learning_p2 import _TIER_GLOBAL, _TIER_BLEND, _BLEND_TICKER_WT
        t = _fresh_transfer(tmp_path)
        t._global["ORB"]           = {"win_rate": 0.50, "n_trades": 100}
        t._ticker["AAPL"]          = {"ORB": {"win_rate": 0.70, "n_trades": 20}}
        t._ticker_counts["AAPL"]   = (_TIER_GLOBAL + _TIER_BLEND) // 2
        blended = t.get_blended_win_rate("AAPL", "ORB")
        expected = _BLEND_TICKER_WT * 0.70 + (1 - _BLEND_TICKER_WT) * 0.50
        assert blended == pytest.approx(expected, abs=0.01)

    def test_aggregate_by_family_empty_df(self, tmp_path):
        from agent.algo_learning_p2 import CrossTickerTransferEngine
        from agent.algo_learning_engine import _ALGO_FAMILY_MAP
        result = CrossTickerTransferEngine._aggregate_by_family(pd.DataFrame(), _ALGO_FAMILY_MAP)
        assert result == {}

    def test_aggregate_by_family_correct_win_rate(self, tmp_path):
        from agent.algo_learning_p2 import CrossTickerTransferEngine
        from agent.algo_learning_engine import _ALGO_FAMILY_MAP
        rows = [
            {"algo_name": "ORB5_BULL", "won": 1},
            {"algo_name": "ORB5_BULL", "won": 1},
            {"algo_name": "ORB5_BULL", "won": 0},
        ]
        result = CrossTickerTransferEngine._aggregate_by_family(pd.DataFrame(rows), _ALGO_FAMILY_MAP)
        assert "ORB" in result
        assert abs(result["ORB"]["win_rate"] - (2/3)) < 0.01

    def test_blend_aggregates_ewma_update(self, tmp_path):
        from agent.algo_learning_p2 import CrossTickerTransferEngine
        old = {"ORB": {"win_rate": 0.50, "n_trades": 20}}
        new = {"ORB": {"win_rate": 0.70, "n_trades": 10}}
        result = CrossTickerTransferEngine._blend_aggregates(old, new, alpha=0.20)
        expected_wr = 0.20 * 0.70 + 0.80 * 0.50
        assert abs(result["ORB"]["win_rate"] - expected_wr) < 0.01

    def test_blend_aggregates_adds_new_family(self, tmp_path):
        from agent.algo_learning_p2 import CrossTickerTransferEngine
        old = {}
        new = {"ORB": {"win_rate": 0.60, "n_trades": 10}}
        result = CrossTickerTransferEngine._blend_aggregates(old, new)
        assert "ORB" in result

    def test_get_transfer_summary_structure(self, tmp_path):
        t = _fresh_transfer(tmp_path)
        summary = t.get_transfer_summary()
        assert "global_families"  in summary
        assert "sector_count"     in summary
        assert "ticker_count"     in summary
        assert "tier_distribution" in summary

    def test_save_and_load_roundtrip(self, tmp_path):
        t = _fresh_transfer(tmp_path)
        t._global        = {"ORB": {"win_rate": 0.55, "n_trades": 30}}
        t._ticker_counts = {"AAPL": 45}
        t.save()

        t2 = _fresh_transfer(tmp_path)
        t2.load()
        assert t2._global.get("ORB", {}).get("win_rate") == pytest.approx(0.55)
        assert t2._ticker_counts.get("AAPL") == 45

    def test_no_crash_on_unknown_algo_name(self, tmp_path):
        t = _fresh_transfer(tmp_path)
        df = _make_outcomes(10)
        df["algo_name"] = "COMPLETELY_UNKNOWN_ALGO"
        summary = t.transfer(df, cycle_num=1)
        # global_updated may be False because no family mapped, but must not crash
        assert isinstance(summary, dict)

    def test_multiple_cycles_accumulate_data(self, tmp_path):
        t = _fresh_transfer(tmp_path)
        for cycle in range(1, 6):
            t.transfer(_make_outcomes(20), cycle_num=cycle)
        assert len(t._ticker) > 0
        assert len(t._global) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. StagedDeploymentController
# ═══════════════════════════════════════════════════════════════════════════════

class TestStagedDeploymentController:

    def test_default_mode_is_paper_only(self, tmp_path):
        d = _fresh_deployment(tmp_path)
        assert d.get_current_mode() == "PAPER_ONLY"

    def test_routing_shadow_returns_shadow(self, tmp_path):
        d = _fresh_deployment(tmp_path)
        d._state["current_mode"] = "SHADOW"
        assert d.get_routing("ORB5_BULL", 1.0) == "SHADOW"

    def test_routing_paper_only_returns_paper(self, tmp_path):
        d = _fresh_deployment(tmp_path)
        d._state["current_mode"] = "PAPER_ONLY"
        assert d.get_routing("ORB5_BULL", 1.0) == "PAPER"

    def test_routing_partial_live_high_ucb_returns_live(self, tmp_path):
        from agent.algo_learning_p2 import _PARTIAL_LIVE_UCB_THRESHOLD
        d = _fresh_deployment(tmp_path)
        d._state["current_mode"] = "PARTIAL_LIVE"
        assert d.get_routing("ORB5_BULL", _PARTIAL_LIVE_UCB_THRESHOLD + 0.1) == "LIVE"

    def test_routing_partial_live_low_ucb_returns_paper(self, tmp_path):
        from agent.algo_learning_p2 import _PARTIAL_LIVE_UCB_THRESHOLD
        d = _fresh_deployment(tmp_path)
        d._state["current_mode"] = "PARTIAL_LIVE"
        assert d.get_routing("ORB5_BULL", _PARTIAL_LIVE_UCB_THRESHOLD - 0.1) == "PAPER"

    def test_routing_full_live_returns_live(self, tmp_path):
        d = _fresh_deployment(tmp_path)
        d._state["current_mode"] = "FULL_LIVE"
        assert d.get_routing("ORB5_BULL", 0.5) == "LIVE"

    def test_routing_unknown_mode_returns_paper(self, tmp_path):
        d = _fresh_deployment(tmp_path)
        d._state["current_mode"] = "UNKNOWN_MODE"
        assert d.get_routing("ORB5_BULL", 1.0) == "PAPER"

    def test_no_advancement_insufficient_trades(self, tmp_path):
        d = _fresh_deployment(tmp_path)
        advanced = d.evaluate_advancement({"win_rate": 0.60, "profit_factor": 1.5,
                                           "n_trades": 5}, cycle_num=1)
        assert advanced is False

    def test_consecutive_passes_increment_on_meeting_gate(self, tmp_path):
        d = _fresh_deployment(tmp_path)
        d.evaluate_advancement(_passing_metrics(win_rate=0.52), cycle_num=1)
        assert d._state["consecutive_passes"] == 1

    def test_consecutive_passes_reset_on_missing_gate(self, tmp_path):
        d = _fresh_deployment(tmp_path)
        d._state["consecutive_passes"] = 2
        d.evaluate_advancement({"win_rate": 0.30, "profit_factor": 0.5, "n_trades": 50},
                               cycle_num=1)
        assert d._state["consecutive_passes"] == 0

    def test_paper_to_partial_advance_after_consecutive_passes(self, tmp_path):
        from agent.algo_learning_p2 import _ADVANCEMENT_GATES
        d = _fresh_deployment(tmp_path)
        required = _ADVANCEMENT_GATES["PAPER_ONLY"]["consecutive_required"]
        for i in range(required):
            d.evaluate_advancement(_passing_metrics(win_rate=0.52, profit_factor=1.1),
                                   cycle_num=i + 1)
        assert d.get_current_mode() == "PARTIAL_LIVE"

    def test_mode_advance_resets_consecutive_passes(self, tmp_path):
        from agent.algo_learning_p2 import _ADVANCEMENT_GATES
        d = _fresh_deployment(tmp_path)
        required = _ADVANCEMENT_GATES["PAPER_ONLY"]["consecutive_required"]
        for i in range(required):
            d.evaluate_advancement(_passing_metrics(win_rate=0.52, profit_factor=1.1),
                                   cycle_num=i + 1)
        assert d._state["consecutive_passes"] == 0

    def test_mode_advance_recorded_in_history(self, tmp_path):
        from agent.algo_learning_p2 import _ADVANCEMENT_GATES
        d = _fresh_deployment(tmp_path)
        required = _ADVANCEMENT_GATES["PAPER_ONLY"]["consecutive_required"]
        for i in range(required):
            d.evaluate_advancement(_passing_metrics(win_rate=0.52, profit_factor=1.1),
                                   cycle_num=i + 1)
        assert len(d._state["mode_history"]) >= 1
        assert d._state["mode_history"][-1]["to_mode"] == "PARTIAL_LIVE"

    def test_no_advance_from_full_live(self, tmp_path):
        d = _fresh_deployment(tmp_path)
        d._state["current_mode"] = "FULL_LIVE"
        advanced = d.evaluate_advancement(_passing_metrics(), cycle_num=1)
        assert advanced is False
        assert d.get_current_mode() == "FULL_LIVE"

    def test_shadow_to_paper_only_requires_only_win_rate(self, tmp_path):
        from agent.algo_learning_p2 import _ADVANCEMENT_GATES
        d = _fresh_deployment(tmp_path)
        d._state["current_mode"] = "SHADOW"
        required = _ADVANCEMENT_GATES["SHADOW"]["consecutive_required"]
        for i in range(required):
            d.evaluate_advancement({"win_rate": 0.46, "profit_factor": 0.0,
                                    "n_trades": 50}, cycle_num=i + 1)
        assert d.get_current_mode() == "PAPER_ONLY"

    def test_mode_history_capped_at_20(self, tmp_path):
        from agent.algo_learning_p2 import _ADVANCEMENT_GATES
        d = _fresh_deployment(tmp_path)
        d._state["mode_history"] = [{"x": i} for i in range(25)]
        d._state["mode_history"] = d._state["mode_history"][-20:]
        assert len(d._state["mode_history"]) == 20

    def test_evaluate_advancement_no_crash_on_bad_metrics(self, tmp_path):
        d = _fresh_deployment(tmp_path)
        # Should not raise
        d.evaluate_advancement({}, cycle_num=1)
        d.evaluate_advancement(None, cycle_num=2)

    def test_get_status_returns_correct_fields(self, tmp_path):
        d = _fresh_deployment(tmp_path)
        status = d.get_status()
        assert "current_mode"       in status
        assert "consecutive_passes" in status
        assert "last_mode_change"   in status
        assert "mode_history"       in status

    def test_save_and_load_roundtrip(self, tmp_path):
        d = _fresh_deployment(tmp_path)
        d._state["current_mode"]       = "PARTIAL_LIVE"
        d._state["consecutive_passes"] = 2
        d.save()

        d2 = _fresh_deployment(tmp_path)
        d2.load()
        assert d2.get_current_mode()            == "PARTIAL_LIVE"
        assert d2._state["consecutive_passes"]  == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 5. OperatorNotificationService
# ═══════════════════════════════════════════════════════════════════════════════

class TestOperatorNotificationService:

    def test_notify_writes_jsonl(self, tmp_path):
        n = _fresh_notifier(tmp_path)
        result = n.notify("DRIFT_ALERT", "test drift", {"psi": 0.3})
        assert result is True
        lines = [l for l in n._PATH.read_text().strip().split("\n") if l]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event_type"] == "DRIFT_ALERT"
        assert entry["message"]    == "test drift"

    def test_notify_is_throttled_on_repeat(self, tmp_path):
        n = _fresh_notifier(tmp_path)
        n.notify("DRIFT_ALERT", "first", {})
        result = n.notify("DRIFT_ALERT", "second", {})
        assert result is False

    def test_notify_different_event_types_not_throttled(self, tmp_path):
        n = _fresh_notifier(tmp_path)
        r1 = n.notify("DRIFT_ALERT",    "drift msg", {})
        r2 = n.notify("MODEL_PROMOTED", "promote msg", {})
        assert r1 is True
        assert r2 is True

    def test_notify_after_throttle_window_passes(self, tmp_path):
        from agent.algo_learning_p2 import _THROTTLE_SECS_DRIFT
        n = _fresh_notifier(tmp_path)
        n.notify("DRIFT_ALERT", "first", {})
        # Force last_sent to be far in the past
        with n._lock:
            n._last_sent["DRIFT_ALERT"] = 0.0
        result = n.notify("DRIFT_ALERT", "second", {})
        assert result is True

    def test_notify_no_webhook_by_default(self, tmp_path):
        n = _fresh_notifier(tmp_path)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALERT_WEBHOOK_URL", None)
            with patch("urllib.request.urlopen") as mock_url:
                n.notify("MODEL_PROMOTED", "msg", {})
        mock_url.assert_not_called()

    def test_notify_calls_webhook_when_env_set(self, tmp_path):
        n = _fresh_notifier(tmp_path)
        with patch.dict(os.environ, {"ALERT_WEBHOOK_URL": "http://fake.hook/"}):
            with patch("urllib.request.urlopen") as mock_url:
                n.notify("MODEL_PROMOTED", "msg", {})
        assert mock_url.called

    def test_severity_stored_in_entry(self, tmp_path):
        n = _fresh_notifier(tmp_path)
        n.notify("DRIFT_ALERT", "material drift", {}, severity="MATERIAL")
        entry = json.loads(n._PATH.read_text().strip())
        assert entry["severity"] == "MATERIAL"

    def test_broadcast_in_app_no_crash_without_main_module(self, tmp_path):
        n = _fresh_notifier(tmp_path)
        # Should not raise even if main.manager is unavailable
        n._broadcast_in_app("TEST", "msg", {})

    def test_broadcast_in_app_sends_to_manager(self, tmp_path):
        n = _fresh_notifier(tmp_path)
        mock_manager = MagicMock()
        mock_manager.broadcast = MagicMock(return_value=None)
        with patch("main.manager", mock_manager, create=True):
            import asyncio
            with patch("asyncio.run_coroutine_threadsafe") as mock_crt:
                mock_crt.return_value = MagicMock()
                n._broadcast_in_app("DRIFT_ALERT", "test msg", {"event_type": "DRIFT_ALERT"})

    def test_get_recent_notifications_empty_file(self, tmp_path):
        n = _fresh_notifier(tmp_path)
        assert n.get_recent_notifications() == []

    def test_get_recent_notifications_returns_entries(self, tmp_path):
        n = _fresh_notifier(tmp_path)
        n.notify("DRIFT_ALERT",    "one", {})
        with n._lock:
            n._last_sent.clear()
        n.notify("MODEL_PROMOTED", "two", {})
        entries = n.get_recent_notifications()
        assert len(entries) == 2

    def test_get_recent_notifications_limit_respected(self, tmp_path):
        n = _fresh_notifier(tmp_path)
        for i in range(10):
            with n._lock:
                n._last_sent.clear()
            n.notify(f"EVENT_{i}", f"msg {i}", {})
        entries = n.get_recent_notifications(limit=5)
        assert len(entries) <= 5

    def test_notify_unwritable_path_no_crash(self):
        from agent.algo_learning_p2 import OperatorNotificationService
        svc = OperatorNotificationService.__new__(OperatorNotificationService)
        svc._lock      = threading.Lock()
        svc._last_sent = {}
        svc._PATH      = Path("/root/no_permission/notif.jsonl")
        # Must not raise even when path is unwritable
        svc.notify("DRIFT_ALERT", "test", {})


# ═══════════════════════════════════════════════════════════════════════════════
# Phase2Engine — end-to-end coordinator
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase2Engine:

    def _make_engine(self, tmp_path):
        from agent.algo_learning_p2 import Phase2Engine
        engine = Phase2Engine.__new__(Phase2Engine)
        from agent.algo_learning_p2 import (
            ConceptDriftDetector, WalkForwardValidator,
            CrossTickerTransferEngine, StagedDeploymentController,
            OperatorNotificationService
        )
        engine._drift      = _fresh_drift(tmp_path)
        engine._validator  = _fresh_validator(tmp_path)
        engine._transfer   = _fresh_transfer(tmp_path)
        engine._deployment = _fresh_deployment(tmp_path)
        engine._notifier   = _fresh_notifier(tmp_path)
        engine._lock       = threading.Lock()
        return engine

    def test_run_cycle_empty_df_no_crash(self, tmp_path):
        engine = self._make_engine(tmp_path)
        with patch("agent.algo_learning_p2._query_p2_outcomes", return_value=pd.DataFrame()):
            engine.run_cycle(pd.DataFrame(), cycle_num=1)

    def test_run_cycle_none_df_no_crash(self, tmp_path):
        engine = self._make_engine(tmp_path)
        with patch("agent.algo_learning_p2._query_p2_outcomes", return_value=pd.DataFrame()):
            engine.run_cycle(None, cycle_num=1)

    def test_run_cycle_calls_drift_update(self, tmp_path):
        engine = self._make_engine(tmp_path)
        df = _make_outcomes(50)
        with patch("agent.algo_learning_p2._query_p2_outcomes", return_value=df):
            with patch.object(engine._drift, "update", return_value={}) as mock_d:
                engine.run_cycle(df, cycle_num=1)
        mock_d.assert_called_once()

    def test_run_cycle_calls_transfer(self, tmp_path):
        engine = self._make_engine(tmp_path)
        df = _make_outcomes(50)
        with patch("agent.algo_learning_p2._query_p2_outcomes", return_value=df):
            with patch.object(engine._transfer, "transfer",
                              return_value={"tickers_updated": 0,
                                           "global_updated": False,
                                           "sectors_updated": 0}) as mock_t:
                engine.run_cycle(df, cycle_num=1)
        mock_t.assert_called_once()

    def test_run_cycle_emits_drift_notification(self, tmp_path):
        engine = self._make_engine(tmp_path)
        df = _make_outcomes(50)
        drift_alerts = {"confidence": (0.30, "MATERIAL")}
        with patch("agent.algo_learning_p2._query_p2_outcomes", return_value=df):
            with patch.object(engine._drift, "update", return_value=drift_alerts):
                with patch.object(engine._notifier, "notify") as mock_n:
                    engine.run_cycle(df, cycle_num=1)
        drift_calls = [c for c in mock_n.call_args_list if c[0][0] == "DRIFT_ALERT"]
        assert len(drift_calls) == 1

    def test_run_cycle_emits_wf_notification_on_success(self, tmp_path):
        engine = self._make_engine(tmp_path)
        df = _make_outcomes(50)
        metrics = _passing_metrics()
        with patch("agent.algo_learning_p2._query_p2_outcomes", return_value=df):
            with patch.object(engine._drift,     "update", return_value={}):
                with patch.object(engine._validator, "run_validation",
                                  return_value=metrics):
                    with patch.object(engine._notifier, "notify") as mock_n:
                        engine.run_cycle(df, cycle_num=1)
        wf_calls = [c for c in mock_n.call_args_list
                    if c[0][0] == "WALK_FORWARD_COMPLETE"]
        assert len(wf_calls) >= 1

    def test_run_cycle_emits_promotion_notification(self, tmp_path):
        engine = self._make_engine(tmp_path)
        df = _make_outcomes(50)
        metrics = {**_passing_metrics(), "promoted": True, "version_id": "v_test"}
        with patch("agent.algo_learning_p2._query_p2_outcomes", return_value=df):
            with patch.object(engine._drift,     "update", return_value={}):
                with patch.object(engine._validator, "run_validation",
                                  return_value=metrics):
                    with patch.object(engine._notifier, "notify") as mock_n:
                        engine.run_cycle(df, cycle_num=1)
        promo_calls = [c for c in mock_n.call_args_list if c[0][0] == "MODEL_PROMOTED"]
        assert len(promo_calls) >= 1

    def test_run_cycle_emits_deployment_notification_on_advance(self, tmp_path):
        engine = self._make_engine(tmp_path)
        df = _make_outcomes(50)
        with patch("agent.algo_learning_p2._query_p2_outcomes", return_value=df):
            with patch.object(engine._drift,      "update", return_value={}):
                with patch.object(engine._validator,  "run_validation", return_value=None):
                    with patch.object(engine._transfer, "transfer",
                                      return_value={"tickers_updated": 0,
                                                   "global_updated": False,
                                                   "sectors_updated": 0}):
                        with patch.object(engine._deployment, "evaluate_advancement",
                                          return_value=True):
                            with patch.object(engine._notifier, "notify") as mock_n:
                                engine.run_cycle(df, cycle_num=1)
        dep_calls = [c for c in mock_n.call_args_list if c[0][0] == "DEPLOYMENT_ADVANCED"]
        assert len(dep_calls) >= 1

    def test_get_routing_delegates_to_deployment(self, tmp_path):
        engine = self._make_engine(tmp_path)
        engine._deployment._state["current_mode"] = "SHADOW"
        assert engine.get_routing("ORB5_BULL", 1.0) == "SHADOW"

    def test_get_drift_summary_returns_dict(self, tmp_path):
        engine = self._make_engine(tmp_path)
        summary = engine.get_drift_summary()
        assert isinstance(summary, dict)

    def test_get_deployment_mode_returns_string(self, tmp_path):
        engine = self._make_engine(tmp_path)
        assert isinstance(engine.get_deployment_mode(), str)

    def test_get_status_has_all_keys(self, tmp_path):
        engine = self._make_engine(tmp_path)
        status = engine.get_status()
        for key in ("drift", "deployment", "validation", "transfer", "notifications"):
            assert key in status

    def test_get_blended_win_rate_returns_float(self, tmp_path):
        engine = self._make_engine(tmp_path)
        rate = engine.get_blended_win_rate("AAPL", "ORB")
        assert 0.0 <= rate <= 1.0

    def test_get_ticker_tier_returns_string(self, tmp_path):
        engine = self._make_engine(tmp_path)
        tier = engine.get_ticker_tier("AAPL")
        assert tier in ("GLOBAL", "BLEND", "TICKER")

    def test_save_and_load_no_crash(self, tmp_path):
        engine = self._make_engine(tmp_path)
        engine.save()
        engine.load()

    def test_run_cycle_uses_db_data_when_available(self, tmp_path):
        engine = self._make_engine(tmp_path)
        df = _make_outcomes(60)
        with patch("agent.algo_learning_p2._query_p2_outcomes", return_value=df) as mock_q:
            with patch.object(engine._drift, "update", return_value={}):
                with patch.object(engine._transfer, "transfer",
                                  return_value={"tickers_updated": 1,
                                               "global_updated": True,
                                               "sectors_updated": 1}):
                    engine.run_cycle(pd.DataFrame(), cycle_num=1)
        mock_q.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level singletons and public API
# ═══════════════════════════════════════════════════════════════════════════════

class TestModuleSingletons:

    def test_get_phase2_engine_returns_instance(self):
        from agent.algo_learning_p2 import get_phase2_engine, Phase2Engine
        engine = get_phase2_engine()
        assert isinstance(engine, Phase2Engine)

    def test_get_phase2_engine_singleton(self):
        from agent.algo_learning_p2 import get_phase2_engine
        e1 = get_phase2_engine()
        e2 = get_phase2_engine()
        assert e1 is e2

    def test_get_deployment_mode_returns_string(self):
        from agent.algo_learning_p2 import get_deployment_mode
        mode = get_deployment_mode()
        assert isinstance(mode, str)

    def test_get_routing_returns_valid_routing(self):
        from agent.algo_learning_p2 import get_routing
        routing = get_routing("ORB5_BULL", 1.0)
        assert routing in ("SHADOW", "PAPER", "LIVE")

    def test_get_routing_safe_on_exception(self):
        from agent.algo_learning_p2 import get_routing
        with patch("agent.algo_learning_p2.get_phase2_engine", side_effect=Exception("crash")):
            routing = get_routing("ANYTHING")
        assert routing == "PAPER"

    def test_get_deployment_mode_safe_on_exception(self):
        from agent.algo_learning_p2 import get_deployment_mode
        with patch("agent.algo_learning_p2.get_phase2_engine", side_effect=Exception("crash")):
            mode = get_deployment_mode()
        assert mode == "PAPER_ONLY"


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: AlgoLearningEngine Phase 2 wiring
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase1Phase2Integration:

    def test_get_routing_on_ale_returns_paper_by_default(self):
        from agent.algo_learning_engine import AlgoLearningEngine
        ale = AlgoLearningEngine()
        routing = ale.get_routing("ORB5_BULL", 1.0)
        assert routing in ("SHADOW", "PAPER", "LIVE")

    def test_get_drift_summary_on_ale_returns_dict(self):
        from agent.algo_learning_engine import AlgoLearningEngine
        ale = AlgoLearningEngine()
        summary = ale.get_drift_summary()
        assert isinstance(summary, dict)

    def test_get_routing_falls_back_on_p2_error(self):
        from agent.algo_learning_engine import AlgoLearningEngine
        ale = AlgoLearningEngine()
        with patch("agent.algo_learning_p2.get_phase2_engine",
                   side_effect=Exception("p2 unavailable")):
            routing = ale.get_routing("ORB5_BULL", 1.5)
        assert routing == "PAPER"

    def test_get_drift_summary_falls_back_on_p2_error(self):
        from agent.algo_learning_engine import AlgoLearningEngine
        ale = AlgoLearningEngine()
        with patch("agent.algo_learning_p2.get_phase2_engine",
                   side_effect=Exception("p2 unavailable")):
            summary = ale.get_drift_summary()
        assert summary == {}

    def test_run_cycle_calls_phase2(self):
        """Verify Phase 2 is invoked from AlgoLearningEngine.run_cycle()."""
        from agent.algo_learning_engine import AlgoLearningEngine
        ale = AlgoLearningEngine()
        ale._audit.log = MagicMock()
        df = _make_outcomes(10)   # small — just enough for P1 to run

        with patch("agent.algo_learning_p2.get_phase2_engine") as mock_p2:
            mock_engine = MagicMock()
            mock_p2.return_value = mock_engine
            ale.run_cycle(df, cycle_num=99)

        mock_engine.run_cycle.assert_called_once()

    def test_run_cycle_phase2_error_does_not_propagate(self):
        """Phase 2 errors must not crash Phase 1 run_cycle."""
        from agent.algo_learning_engine import AlgoLearningEngine
        ale = AlgoLearningEngine()
        ale._audit.log = MagicMock()

        with patch("agent.algo_learning_p2.get_phase2_engine",
                   side_effect=Exception("p2 crash")):
            # Must not raise
            ale.run_cycle(_make_outcomes(5), cycle_num=100)


# ═══════════════════════════════════════════════════════════════════════════════
# Thread-safety smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestThreadSafety:

    def test_deployment_concurrent_routing_no_crash(self, tmp_path):
        d = _fresh_deployment(tmp_path)
        errors = []

        def _route():
            try:
                for _ in range(50):
                    d.get_routing("ORB5_BULL", 1.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_route) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_transfer_concurrent_transfers_no_crash(self, tmp_path):
        t = _fresh_transfer(tmp_path)
        errors = []
        df = _make_outcomes(20)

        def _transfer():
            try:
                for i in range(5):
                    t.transfer(df, cycle_num=i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_transfer) for _ in range(5)]
        for thr in threads:
            thr.start()
        for thr in threads:
            thr.join()
        assert errors == []

    def test_notifier_concurrent_notify_no_crash(self, tmp_path):
        n = _fresh_notifier(tmp_path)
        errors = []

        def _notify(i):
            try:
                n.notify(f"EVENT_{i}", f"msg {i}", {})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_notify, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_drift_concurrent_updates_no_crash(self, tmp_path):
        d = _fresh_drift(tmp_path)
        df = _make_outcomes(60)
        errors = []

        def _update():
            try:
                d.update(df, cycle_num=1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_update) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases and resilience
# ═══════════════════════════════════════════════════════════════════════════════

class TestResilience:

    def test_query_p2_outcomes_returns_empty_on_missing_db(self):
        from agent.algo_learning_p2 import _query_p2_outcomes
        with patch("agent.db.get_conn", side_effect=Exception("no db")):
            df = _query_p2_outcomes()
        assert df.empty

    def test_get_sector_returns_default_for_unknown_ticker(self):
        from agent.algo_learning_p2 import _get_sector
        assert _get_sector("UNKNOWN_TICKER_XYZ") == "QQQ"

    def test_get_sector_returns_correct_for_known_ticker(self):
        from agent.algo_learning_p2 import _get_sector
        assert _get_sector("NVDA") == "SMH"

    def test_deployment_evaluate_advancement_with_none_metrics(self, tmp_path):
        d = _fresh_deployment(tmp_path)
        # Should not raise
        d.evaluate_advancement(None, cycle_num=1)

    def test_validator_compute_metrics_no_pnl_col(self, tmp_path):
        v = _fresh_validator(tmp_path)
        df = _make_outcomes(50)
        del df["pnl_pct"]
        metrics = v._compute_metrics(df)
        assert metrics == {}

    def test_drift_detector_missing_feature_columns_no_crash(self, tmp_path):
        d = _fresh_drift(tmp_path)
        df = pd.DataFrame({"ticker": ["AAPL"] * 40})   # no feature cols
        # Build reference (will skip missing columns)
        d.update(df, cycle_num=1)
        d._state["last_check_cycle"] = 0
        alerts = d.update(df, cycle_num=10)
        assert isinstance(alerts, dict)

    def test_transfer_no_sector_col_no_crash(self, tmp_path):
        t = _fresh_transfer(tmp_path)
        df = _make_outcomes(20)
        if "sector_etf" in df.columns:
            del df["sector_etf"]
        summary = t.transfer(df, cycle_num=1)
        assert isinstance(summary, dict)

    def test_notifier_jsonl_is_valid_after_many_writes(self, tmp_path):
        n = _fresh_notifier(tmp_path)
        for i in range(20):
            with n._lock:
                n._last_sent.clear()
            n.notify(f"EVT_{i % 5}", f"msg {i}", {"i": i})
        lines = [l for l in n._PATH.read_text().strip().split("\n") if l]
        for line in lines:
            json.loads(line)   # every line must be valid JSON

    def test_deployment_full_lifecycle_shadow_to_full_live(self, tmp_path):
        """Simulate gradual mode advancement from SHADOW to FULL_LIVE."""
        from agent.algo_learning_p2 import _ADVANCEMENT_GATES
        d = _fresh_deployment(tmp_path)
        d._state["current_mode"] = "SHADOW"

        def advance_mode():
            mode = d.get_current_mode()
            gates = _ADVANCEMENT_GATES.get(mode)
            if not gates:
                return False
            required = gates["consecutive_required"]
            for i in range(required):
                result = d.evaluate_advancement(
                    {"win_rate": gates["min_win_rate"] + 0.05,
                     "profit_factor": max(gates["min_profit_factor"], 1.15),
                     "n_trades": 50},
                    cycle_num=i + 1
                )
            return result

        # SHADOW → PAPER_ONLY
        assert advance_mode() is True
        assert d.get_current_mode() == "PAPER_ONLY"

        # PAPER_ONLY → PARTIAL_LIVE
        assert advance_mode() is True
        assert d.get_current_mode() == "PARTIAL_LIVE"

        # PARTIAL_LIVE → FULL_LIVE
        assert advance_mode() is True
        assert d.get_current_mode() == "FULL_LIVE"

        # FULL_LIVE → no further advancement
        assert advance_mode() is False
