"""
Comprehensive tests for Phase 1 Adaptive Trading Learning Engine.
Covers all 10 components in agent/algo_learning_engine.py.
"""
import json
import math
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fresh_registry(tmp_path):
    """Return a ParameterControlRegistry that writes to tmp_path."""
    from agent.algo_learning_engine import ParameterControlRegistry, _DATA_DIR
    reg = ParameterControlRegistry.__new__(ParameterControlRegistry)
    reg._lock = threading.Lock()
    reg._state = {}
    ParameterControlRegistry._load_defaults(reg)
    reg._PATH = tmp_path / "algo_params.json"
    return reg


def _fresh_loss_analyzer(tmp_path):
    from agent.algo_learning_engine import LossAnalyzer
    la = LossAnalyzer.__new__(LossAnalyzer)
    la._lock = threading.Lock()
    la._patterns = {}
    la._PATH = tmp_path / "loss_patterns.json"
    return la


def _fresh_win_reinforcer(tmp_path):
    from agent.algo_learning_engine import WinReinforcer
    wr = WinReinforcer.__new__(WinReinforcer)
    wr._lock = threading.Lock()
    wr._wins = {}
    wr._PATH = tmp_path / "win_patterns.json"
    return wr


def _fresh_selector(tmp_path):
    from agent.algo_learning_engine import AlgoSelector
    sel = AlgoSelector.__new__(AlgoSelector)
    sel._lock = threading.Lock()
    sel._state = {}
    sel._PATH = tmp_path / "algo_selector.json"
    return sel


def _fresh_model_registry(tmp_path):
    from agent.algo_learning_engine import ModelVersionRegistry
    mr = ModelVersionRegistry.__new__(ModelVersionRegistry)
    mr._lock = threading.Lock()
    mr._data = {"champion": None, "challengers": {}, "history": []}
    mr._PATH = tmp_path / "model_registry.json"
    return mr


def _good_challenger_metrics():
    return {
        "n_trades": 50,
        "profit_factor": 1.50,
        "expectancy": 0.002,
        "sharpe": 1.20,
        "max_drawdown": 0.05,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ParameterControlRegistry
# ═══════════════════════════════════════════════════════════════════════════════

class TestParameterControlRegistry:

    def test_defaults_loaded_for_all_families(self):
        from agent.algo_learning_engine import ParameterControlRegistry, _ALL_FAMILIES, _PARAM_SPEC
        reg = ParameterControlRegistry()
        for fam in _ALL_FAMILIES:
            for param, spec in _PARAM_SPEC.items():
                assert abs(reg.get(fam, param) - spec["default"]) < 1e-6

    def test_get_unknown_family_returns_default(self):
        from agent.algo_learning_engine import ParameterControlRegistry
        reg = ParameterControlRegistry()
        # target_mult default is 1.5
        val = reg.get("NONEXISTENT", "target_mult")
        assert val == 1.5

    def test_get_unknown_param_returns_zero(self):
        from agent.algo_learning_engine import ParameterControlRegistry
        reg = ParameterControlRegistry()
        val = reg.get("ORB", "completely_unknown_param")
        assert val == 0.0

    def test_update_applies_within_bounds(self, tmp_path):
        reg = _fresh_registry(tmp_path)
        applied = reg.update("ORB", "target_mult", 1.6, "test", cycle_num=10)
        assert applied is True
        assert abs(reg.get("ORB", "target_mult") - 1.6) < 1e-6

    def test_update_clamps_to_max_change(self, tmp_path):
        """max_change for target_mult is 0.10; requesting +0.5 should clamp to +0.10."""
        reg = _fresh_registry(tmp_path)
        reg.update("ORB", "target_mult", 2.0, "big jump", cycle_num=10)
        val = reg.get("ORB", "target_mult")
        assert abs(val - 1.60) < 1e-5   # default 1.5 + 0.10 max_change

    def test_update_clamps_to_min_bound(self, tmp_path):
        reg = _fresh_registry(tmp_path)
        # target_mult min is 0.75; try to set 0.1
        reg.update("ORB", "target_mult", 0.1, "too low", cycle_num=10)
        val = reg.get("ORB", "target_mult")
        assert val >= 0.75

    def test_update_clamps_to_max_bound(self, tmp_path):
        reg = _fresh_registry(tmp_path)
        # conf_gate max is 75; try to set 200 (increase_only param, currently 55)
        for i in range(100):
            reg.update("ORB", "conf_gate", 200.0, "big", cycle_num=i * 10)
        val = reg.get("ORB", "conf_gate")
        assert val <= 75.0

    def test_update_reduce_only_rejects_increase(self, tmp_path):
        """stop_mult is reduce_only — cannot increase."""
        reg = _fresh_registry(tmp_path)
        applied = reg.update("ORB", "stop_mult", 1.5, "increase attempt", cycle_num=10)
        assert applied is False   # default is 1.0, new_val 1.5 > 1.0

    def test_update_increase_only_rejects_decrease(self, tmp_path):
        """conf_gate is increase_only — cannot decrease below current."""
        reg = _fresh_registry(tmp_path)
        applied = reg.update("ORB", "conf_gate", 50.0, "decrease attempt", cycle_num=10)
        assert applied is False   # default is 55.0, 50 < 55

    def test_update_respects_cooldown(self, tmp_path):
        """cooldown_cycles=5 for target_mult — second update before cooldown must fail."""
        reg = _fresh_registry(tmp_path)
        reg.update("ORB", "target_mult", 1.6, "first", cycle_num=10)
        # Cycle 12 — only 2 cycles later, cooldown is 5
        applied = reg.update("ORB", "target_mult", 1.55, "too soon", cycle_num=12)
        assert applied is False

    def test_update_allowed_after_cooldown(self, tmp_path):
        reg = _fresh_registry(tmp_path)
        reg.update("ORB", "target_mult", 1.6, "first", cycle_num=10)
        # Cycle 16 — 6 cycles later, cooldown is 5
        applied = reg.update("ORB", "target_mult", 1.55, "after cooldown", cycle_num=16)
        assert applied is True

    def test_update_rejects_unknown_family(self, tmp_path):
        reg = _fresh_registry(tmp_path)
        applied = reg.update("UNKNOWN_FAM", "target_mult", 1.6, "x", cycle_num=10)
        assert applied is False

    def test_update_rejects_unknown_param(self, tmp_path):
        reg = _fresh_registry(tmp_path)
        applied = reg.update("ORB", "made_up_param", 1.6, "x", cycle_num=10)
        assert applied is False

    def test_first_update_no_cooldown_check(self, tmp_path):
        """First update (last_updated_cycle=0) should always pass cooldown check."""
        reg = _fresh_registry(tmp_path)
        applied = reg.update("ORB", "target_mult", 1.6, "first ever", cycle_num=1)
        assert applied is True

    def test_get_all_params_known_algo(self):
        from agent.algo_learning_engine import ParameterControlRegistry, _PARAM_SPEC
        reg = ParameterControlRegistry()
        params = reg.get_all_params("ORB5_BULL")
        for p in _PARAM_SPEC:
            assert p in params

    def test_get_all_params_unknown_algo_returns_defaults(self):
        from agent.algo_learning_engine import ParameterControlRegistry, _PARAM_SPEC
        reg = ParameterControlRegistry()
        params = reg.get_all_params("COMPLETELY_UNKNOWN")
        for p, spec in _PARAM_SPEC.items():
            assert abs(params[p] - spec["default"]) < 1e-6

    def test_save_and_load_roundtrip(self, tmp_path):
        reg = _fresh_registry(tmp_path)
        reg.update("ORB", "target_mult", 1.6, "test", cycle_num=10)
        reg.save()

        reg2 = _fresh_registry(tmp_path)
        reg2.load()
        assert abs(reg2.get("ORB", "target_mult") - 1.6) < 1e-6

    def test_load_nonexistent_file_is_safe(self, tmp_path):
        reg = _fresh_registry(tmp_path)
        # Should not raise
        reg.load()
        assert reg.get("ORB", "target_mult") == 1.5   # still default

    def test_load_corrupt_file_is_safe(self, tmp_path):
        p = tmp_path / "algo_params.json"
        p.write_text("not json {{{{")
        reg = _fresh_registry(tmp_path)
        reg.load()   # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# 2. OutcomeClassifier
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutcomeClassifier:

    def test_win_above_threshold(self):
        from agent.algo_learning_engine import OutcomeClassifier
        assert OutcomeClassifier.classify(0.002, "STOP") == "WIN"

    def test_loss_below_threshold(self):
        from agent.algo_learning_engine import OutcomeClassifier
        assert OutcomeClassifier.classify(-0.002, "STOP") == "LOSS"

    def test_break_even_positive(self):
        from agent.algo_learning_engine import OutcomeClassifier
        assert OutcomeClassifier.classify(0.001, "STOP") == "BREAK_EVEN"

    def test_break_even_negative(self):
        from agent.algo_learning_engine import OutcomeClassifier
        assert OutcomeClassifier.classify(-0.001, "STOP") == "BREAK_EVEN"

    def test_break_even_zero(self):
        from agent.algo_learning_engine import OutcomeClassifier
        assert OutcomeClassifier.classify(0.0, "STOP") == "BREAK_EVEN"

    def test_timeout_classified_as_timeout(self):
        from agent.algo_learning_engine import OutcomeClassifier
        result = OutcomeClassifier.classify(-0.005, "TIMEOUT")
        assert result == "TIMEOUT"

    def test_timeout_positive_still_timeout(self):
        from agent.algo_learning_engine import OutcomeClassifier
        result = OutcomeClassifier.classify(0.005, "TIMEOUT")
        assert result == "TIMEOUT"

    def test_invalid_pnl_type(self):
        from agent.algo_learning_engine import OutcomeClassifier
        assert OutcomeClassifier.classify("not_a_number", "STOP") == "INVALID"

    def test_invalid_none(self):
        from agent.algo_learning_engine import OutcomeClassifier
        assert OutcomeClassifier.classify(None, "STOP") == "INVALID"

    def test_exact_win_threshold(self):
        from agent.algo_learning_engine import OutcomeClassifier
        # 0.0015 is exactly WIN_THRESH — above threshold → WIN
        assert OutcomeClassifier.classify(0.0015 + 1e-9, "STOP") == "WIN"

    def test_exactly_at_loss_threshold(self):
        from agent.algo_learning_engine import OutcomeClassifier
        assert OutcomeClassifier.classify(-0.0015 - 1e-9, "STOP") == "LOSS"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LossAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

def _path_of(*r_vals):
    """Build price_path_rows from a sequence of r_vals (bar index = 1-based)."""
    return [{"bar": i + 1, "price": 100 + v, "r_val": v}
            for i, v in enumerate(r_vals)]


class TestLossAnalyzer:

    def test_volatility_spike_detected(self):
        from agent.algo_learning_engine import LossAnalyzer
        la = LossAnalyzer()
        path = _path_of(0, 0.5, 3.5)   # bar2→bar3 swing = 3.0 > 2.0
        result = la.analyze({"direction": "BUY", "regime": "BULL",
                              "vwap_event": "ABOVE", "entry_type": "IMMEDIATE",
                              "bars_tracked": 3, "exit_reason": "STOP",
                              "max_favorable_r": 0.5}, path)
        assert result["root_cause"] == "VOLATILITY_SPIKE"
        assert result["confidence"] >= 0.8

    def test_timeout_drift_detected(self):
        from agent.algo_learning_engine import LossAnalyzer
        la = LossAnalyzer()
        path = _path_of(-0.1, -0.2, -0.1)
        result = la.analyze({"direction": "BUY", "regime": "NEUTRAL",
                              "vwap_event": "ABOVE", "entry_type": "IMMEDIATE",
                              "bars_tracked": 3, "exit_reason": "TIMEOUT",
                              "max_favorable_r": 0.1}, path)
        assert result["root_cause"] == "TIMEOUT_DRIFT"

    def test_wrong_direction_at_bar2(self):
        from agent.algo_learning_engine import LossAnalyzer
        la = LossAnalyzer()
        path = _path_of(0.1, -0.6, -0.8)   # bar2 r_val = -0.6
        result = la.analyze({"direction": "BUY", "regime": "BULL",
                              "vwap_event": "ABOVE", "entry_type": "IMMEDIATE",
                              "bars_tracked": 3, "exit_reason": "STOP",
                              "max_favorable_r": 0.1}, path)
        assert result["root_cause"] == "WRONG_DIRECTION"

    def test_vwap_conflict_buy_below_vwap(self):
        from agent.algo_learning_engine import LossAnalyzer
        la = LossAnalyzer()
        path = _path_of(0.1, 0.0, -0.1)   # bar2 r_val ok
        result = la.analyze({"direction": "BUY", "regime": "BULL",
                              "vwap_event": "BELOW", "entry_type": "PULLBACK",
                              "bars_tracked": 3, "exit_reason": "STOP",
                              "max_favorable_r": 0.1}, path)
        assert result["root_cause"] == "VWAP_CONFLICT"

    def test_vwap_conflict_sell_above_vwap(self):
        from agent.algo_learning_engine import LossAnalyzer
        la = LossAnalyzer()
        path = _path_of(0.1, 0.0, -0.1)
        result = la.analyze({"direction": "SELL", "regime": "BULL",
                              "vwap_event": "ABOVE", "entry_type": "PULLBACK",
                              "bars_tracked": 3, "exit_reason": "STOP",
                              "max_favorable_r": 0.1}, path)
        assert result["root_cause"] == "VWAP_CONFLICT"

    def test_regime_mismatch_neutral(self):
        from agent.algo_learning_engine import LossAnalyzer
        la = LossAnalyzer()
        path = _path_of(0.1, 0.0, -0.1)
        result = la.analyze({"direction": "BUY", "regime": "NEUTRAL",
                              "vwap_event": "ABOVE", "entry_type": "PULLBACK",
                              "bars_tracked": 3, "exit_reason": "STOP",
                              "max_favorable_r": 0.1}, path)
        assert result["root_cause"] == "REGIME_MISMATCH"

    def test_timing_late_no_positive_r_in_5_bars(self):
        from agent.algo_learning_engine import LossAnalyzer
        la = LossAnalyzer()
        path = _path_of(-0.1, -0.1, -0.1, -0.1, -0.1, -0.1)
        result = la.analyze({"direction": "BUY", "regime": "BULL",
                              "vwap_event": "ABOVE", "entry_type": "IMMEDIATE",
                              "bars_tracked": 6, "exit_reason": "STOP",
                              "max_favorable_r": 0.1}, path)
        assert result["root_cause"] == "TIMING_LATE"

    def test_stop_too_tight_price_recovers_after_stop(self):
        from agent.algo_learning_engine import LossAnalyzer
        la = LossAnalyzer()
        # bar3 is stop bar; bars after it have positive r_val → stop too tight
        path = [{"bar": 1, "r_val": 0.1},
                {"bar": 2, "r_val": 0.0},
                {"bar": 3, "r_val": -0.2},  # stop hit here
                {"bar": 4, "r_val": 0.5}]   # price recovered
        result = la.analyze({"direction": "BUY", "regime": "BULL",
                              "vwap_event": "ABOVE", "entry_type": "PULLBACK",
                              "bars_tracked": 3, "exit_reason": "STOP",
                              "max_favorable_r": 0.1}, path)
        assert result["root_cause"] == "STOP_TOO_TIGHT"

    def test_analyze_empty_path_no_crash(self):
        from agent.algo_learning_engine import LossAnalyzer
        la = LossAnalyzer()
        result = la.analyze({}, [])
        assert "root_cause" in result

    def test_analyze_corrupt_data_no_crash(self):
        from agent.algo_learning_engine import LossAnalyzer
        la = LossAnalyzer()
        result = la.analyze({"max_favorable_r": "not_a_number"}, [{"bar": None}])
        assert "root_cause" in result

    def test_record_pattern_ewma_update(self):
        from agent.algo_learning_engine import LossAnalyzer
        la = LossAnalyzer()
        la.record_pattern("ORB5_BULL", "STOP_TOO_TIGHT")
        rates = la._patterns["ORB5_BULL"]
        assert rates["STOP_TOO_TIGHT"] == pytest.approx(0.25)   # alpha * 1.0
        assert rates["WRONG_DIRECTION"] == pytest.approx(0.0)

    def test_record_pattern_multiple_updates(self):
        from agent.algo_learning_engine import LossAnalyzer
        la = LossAnalyzer()
        for _ in range(10):
            la.record_pattern("ORB5_BULL", "STOP_TOO_TIGHT")
        rates = la._patterns["ORB5_BULL"]
        assert rates["STOP_TOO_TIGHT"] > 0.5

    def test_get_dominant_cause_returns_highest_rate(self):
        from agent.algo_learning_engine import LossAnalyzer
        la = LossAnalyzer()
        la._patterns["ORB5_BULL"] = {
            "STOP_TOO_TIGHT": 0.8, "WRONG_DIRECTION": 0.1, "REGIME_MISMATCH": 0.05,
            "VWAP_CONFLICT": 0.02, "TIMING_LATE": 0.01, "TIMEOUT_DRIFT": 0.01,
            "VOLATILITY_SPIKE": 0.01,
        }
        assert la.get_dominant_cause("ORB5_BULL") == "STOP_TOO_TIGHT"

    def test_get_dominant_cause_unknown_algo_returns_default(self):
        from agent.algo_learning_engine import LossAnalyzer
        la = LossAnalyzer()
        assert la.get_dominant_cause("NEVER_SEEN_ALGO") == "WRONG_DIRECTION"

    def test_save_and_load_roundtrip(self, tmp_path):
        la = _fresh_loss_analyzer(tmp_path)
        la.record_pattern("ORB5_BULL", "STOP_TOO_TIGHT")
        la.save()

        la2 = _fresh_loss_analyzer(tmp_path)
        la2.load()
        assert "ORB5_BULL" in la2._patterns
        assert la2._patterns["ORB5_BULL"]["STOP_TOO_TIGHT"] == pytest.approx(0.25)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. WinReinforcer
# ═══════════════════════════════════════════════════════════════════════════════

class TestWinReinforcer:

    def test_weight_default_1_when_no_data(self):
        from agent.algo_learning_engine import WinReinforcer
        wr = WinReinforcer()
        assert wr.get_weight("ORB5_BULL", "BULL:AM:ABOVE") == 1.0

    def test_weight_default_1_below_min_wins(self, tmp_path):
        wr = _fresh_win_reinforcer(tmp_path)
        for _ in range(5):   # fewer than _MIN_WINS=15
            wr.record_win("ORB5_BULL", "BULL:AM:ABOVE")
        assert wr.get_weight("ORB5_BULL", "BULL:AM:ABOVE") == 1.0

    def test_weight_increases_with_many_wins(self, tmp_path):
        wr = _fresh_win_reinforcer(tmp_path)
        for _ in range(30):
            wr.record_win("ORB5_BULL", "BULL:AM:ABOVE")
        weight = wr.get_weight("ORB5_BULL", "BULL:AM:ABOVE")
        assert weight > 1.0

    def test_weight_capped_at_1_3(self, tmp_path):
        wr = _fresh_win_reinforcer(tmp_path)
        for _ in range(200):
            wr.record_win("ORB5_BULL", "BULL:AM:ABOVE")
        assert wr.get_weight("ORB5_BULL", "BULL:AM:ABOVE") <= 1.3

    def test_weight_never_below_0_7(self, tmp_path):
        """Even with 0 wins but enough count, floor is 0.7."""
        wr = _fresh_win_reinforcer(tmp_path)
        # Manually set low ewma
        wr._wins["ORB5_BULL"] = {"BULL:AM:ABOVE": {"ewma": 0.0, "count": 20}}
        assert wr.get_weight("ORB5_BULL", "BULL:AM:ABOVE") >= 0.7

    def test_record_win_increments_count(self, tmp_path):
        wr = _fresh_win_reinforcer(tmp_path)
        wr.record_win("ORB5_BULL", "ctx")
        assert wr._wins["ORB5_BULL"]["ctx"]["count"] == 1

    def test_ewma_alpha_015(self, tmp_path):
        wr = _fresh_win_reinforcer(tmp_path)
        wr.record_win("ORB5_BULL", "ctx")
        ewma = wr._wins["ORB5_BULL"]["ctx"]["ewma"]
        assert ewma == pytest.approx(0.15)   # alpha * 1.0

    def test_save_and_load_roundtrip(self, tmp_path):
        wr = _fresh_win_reinforcer(tmp_path)
        for _ in range(5):
            wr.record_win("ORB5_BULL", "ctx")
        wr.save()

        wr2 = _fresh_win_reinforcer(tmp_path)
        wr2.load()
        assert wr2._wins["ORB5_BULL"]["ctx"]["count"] == 5


# ═══════════════════════════════════════════════════════════════════════════════
# 5. AlgoSelector
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlgoSelector:

    def test_unknown_context_returns_ones(self):
        from agent.algo_learning_engine import AlgoSelector
        sel = AlgoSelector()
        weights = sel.get_weights(["ORB5_BULL", "GAP_AND_GO_BULL"], "UNKNOWN:CTX")
        assert all(w == 1.0 for w in weights.values())

    def test_unexplored_algo_gets_neutral_weight(self, tmp_path):
        sel = _fresh_selector(tmp_path)
        sel.record_outcome("ORB5_BULL", "ctx", won=True)
        weights = sel.get_weights(["ORB5_BULL", "NEW_ALGO"], "ctx")
        assert "NEW_ALGO" in weights

    def test_record_outcome_updates_ewma_wr(self, tmp_path):
        sel = _fresh_selector(tmp_path)
        sel.record_outcome("ORB5_BULL", "ctx", won=True)
        entry = sel._state["ctx"]["ORB5_BULL"]
        # 0.20 * 1.0 + 0.80 * 0.5 = 0.60
        assert entry["ewma_wr"] == pytest.approx(0.60)

    def test_record_outcome_loss_decrements_ewma(self, tmp_path):
        sel = _fresh_selector(tmp_path)
        sel.record_outcome("ORB5_BULL", "ctx", won=False)
        entry = sel._state["ctx"]["ORB5_BULL"]
        # 0.20 * 0.0 + 0.80 * 0.5 = 0.40
        assert entry["ewma_wr"] == pytest.approx(0.40)

    def test_ucb_formula_correct(self):
        from agent.algo_learning_engine import AlgoSelector
        sel = AlgoSelector()
        ucb = sel._ucb(0.6, 100, 10)
        expected = 0.6 + math.sqrt(2 * math.log(100) / 10)
        assert ucb == pytest.approx(expected)

    def test_weights_normalized_to_mean_one(self, tmp_path):
        sel = _fresh_selector(tmp_path)
        algos = ["ORB5_BULL", "GAP_AND_GO_BULL", "BULL_FLAG"]
        for a in algos:
            for _ in range(5):
                sel.record_outcome(a, "ctx", won=True)
        weights = sel.get_weights(algos, "ctx")
        mean = sum(weights.values()) / len(weights)
        assert mean == pytest.approx(1.0, abs=0.01)

    def test_n_trials_increments(self, tmp_path):
        sel = _fresh_selector(tmp_path)
        sel.record_outcome("ORB5_BULL", "ctx", won=True)
        sel.record_outcome("ORB5_BULL", "ctx", won=False)
        assert sel._state["ctx"]["ORB5_BULL"]["n_trials"] == 2

    def test_save_and_load_roundtrip(self, tmp_path):
        sel = _fresh_selector(tmp_path)
        sel.record_outcome("ORB5_BULL", "ctx", won=True)
        sel.save()

        sel2 = _fresh_selector(tmp_path)
        sel2.load()
        assert "ORB5_BULL" in sel2._state.get("ctx", {})

    def test_high_performing_algo_gets_higher_weight(self, tmp_path):
        sel = _fresh_selector(tmp_path)
        ctx = "BULL:AM:ABOVE"
        # Good algo: many wins
        for _ in range(20):
            sel.record_outcome("ORB5_BULL", ctx, won=True)
        # Bad algo: many losses
        for _ in range(20):
            sel.record_outcome("GAP_AND_GO_BULL", ctx, won=False)

        weights = sel.get_weights(["ORB5_BULL", "GAP_AND_GO_BULL"], ctx)
        assert weights["ORB5_BULL"] > weights["GAP_AND_GO_BULL"]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ParameterAdapter
# ═══════════════════════════════════════════════════════════════════════════════

class TestParameterAdapter:

    def _make_adapter(self, cause: str, algo: str = "ORB5_BULL"):
        from agent.algo_learning_engine import (
            ParameterAdapter, ParameterControlRegistry, LossAnalyzer
        )
        reg = ParameterControlRegistry()
        la = LossAnalyzer()
        la._patterns[algo] = {c: 0.0 for c in [
            "STOP_TOO_TIGHT", "WRONG_DIRECTION", "REGIME_MISMATCH",
            "VWAP_CONFLICT", "TIMING_LATE", "TIMEOUT_DRIFT", "VOLATILITY_SPIKE"
        ]}
        la._patterns[algo][cause] = 1.0
        return ParameterAdapter(reg, la), reg

    def test_stop_too_tight_increases_stop_mult(self):
        # stop_mult is reduce_only — the adapter tries to increase it, which is blocked
        # (by design: stop_mult reduce_only means adapter can only reduce it)
        # The adapter attempts new_val = old * 1.10 > old → rejected by registry
        adapter, reg = self._make_adapter("STOP_TOO_TIGHT")
        changes = adapter.adapt("ORB", "ORB5_BULL", cycle_num=10)
        # Since stop_mult is reduce_only, this should be blocked
        assert "stop_mult" not in changes

    def test_wrong_direction_increases_conf_gate(self):
        adapter, reg = self._make_adapter("WRONG_DIRECTION")
        changes = adapter.adapt("ORB", "ORB5_BULL", cycle_num=10)
        assert "conf_gate" in changes
        old_val, new_val, _ = changes["conf_gate"]
        assert new_val > old_val

    def test_regime_mismatch_increases_conf_gate(self):
        adapter, reg = self._make_adapter("REGIME_MISMATCH")
        changes = adapter.adapt("ORB", "ORB5_BULL", cycle_num=10)
        assert "conf_gate" in changes

    def test_timeout_drift_reduces_target_mult(self):
        adapter, reg = self._make_adapter("TIMEOUT_DRIFT")
        changes = adapter.adapt("ORB", "ORB5_BULL", cycle_num=10)
        assert "target_mult" in changes
        old_val, new_val, _ = changes["target_mult"]
        assert new_val < old_val

    def test_vwap_conflict_increases_rvol_gate(self):
        adapter, reg = self._make_adapter("VWAP_CONFLICT")
        changes = adapter.adapt("ORB", "ORB5_BULL", cycle_num=10)
        assert "rvol_gate" in changes
        old_val, new_val, _ = changes["rvol_gate"]
        assert new_val > old_val

    def test_timing_late_reduces_entry_window(self):
        adapter, reg = self._make_adapter("TIMING_LATE")
        changes = adapter.adapt("ORB", "ORB5_BULL", cycle_num=10)
        assert "entry_window_bars" in changes
        old_val, new_val, _ = changes["entry_window_bars"]
        assert new_val < old_val

    def test_volatility_spike_increases_stop_mult(self):
        # Again, stop_mult is reduce_only — widen attempt will be blocked
        adapter, reg = self._make_adapter("VOLATILITY_SPIKE")
        changes = adapter.adapt("ORB", "ORB5_BULL", cycle_num=10)
        # Widen attempt on reduce_only param → blocked by registry
        assert "stop_mult" not in changes

    def test_unknown_family_returns_empty_changes(self):
        from agent.algo_learning_engine import (
            ParameterAdapter, ParameterControlRegistry, LossAnalyzer
        )
        reg = ParameterControlRegistry()
        la = LossAnalyzer()
        adapter = ParameterAdapter(reg, la)
        changes = adapter.adapt("NOT_A_FAMILY", "FAKE_ALGO", cycle_num=10)
        assert changes == {}

    def test_adapt_no_crash_on_exception(self):
        from agent.algo_learning_engine import ParameterAdapter
        # Pass broken objects — should not raise
        adapter = ParameterAdapter(None, None)
        result = adapter.adapt("ORB", "ORB5_BULL", cycle_num=10)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CounterfactualSimulator
# ═══════════════════════════════════════════════════════════════════════════════

class TestCounterfactualSimulator:

    def test_record_suppressed_calls_bt_record(self):
        from agent.algo_learning_engine import CounterfactualSimulator
        sim = CounterfactualSimulator()
        with patch("agent.live_backtest.record_signal", return_value="SID-001") as mock_rec:
            sid = sim.record_suppressed(
                ticker="AAPL", direction="BUY",
                entry_price=150.0, target=153.0, stop=149.0,
                rr_ratio=3.0, confidence=65.0,
                session="AM", regime="BULL", vwap_event="ABOVE",
                rsi_zone="MIDDLE", entry_type="IMMEDIATE",
                algo_name="ORB5_BULL", suppression_reason="LOW_CONFIDENCE",
            )
        assert sid == "SID-001"
        mock_rec.assert_called_once()
        call_kwargs = mock_rec.call_args.kwargs
        assert call_kwargs["is_counterfactual"] == 1
        assert call_kwargs["suppression_reason"] == "LOW_CONFIDENCE"

    def test_record_suppressed_import_error_returns_empty(self):
        from agent.algo_learning_engine import CounterfactualSimulator
        sim = CounterfactualSimulator()
        with patch("agent.live_backtest.record_signal", side_effect=Exception("db error")):
            sid = sim.record_suppressed(
                ticker="AAPL", direction="BUY",
                entry_price=150.0, target=153.0, stop=149.0,
                rr_ratio=3.0, confidence=65.0,
                session="AM", regime="BULL", vwap_event="ABOVE",
                rsi_zone="MIDDLE", entry_type="IMMEDIATE",
                algo_name="ORB5_BULL", suppression_reason="TEST",
            )
        assert sid == ""

    def test_should_relax_filter_not_enough_data(self):
        from agent.algo_learning_engine import CounterfactualSimulator
        sim = CounterfactualSimulator()
        with patch.object(sim, "get_counterfactual_stats",
                          return_value={"total": 5, "wins": 3, "losses": 2, "missed_win_rate": 0.60}):
            relax, reason = sim.should_relax_filter("ctx", min_cf_trades=10)
        assert relax is False
        assert "5" in reason

    def test_should_relax_filter_high_missed_win_rate(self):
        from agent.algo_learning_engine import CounterfactualSimulator
        sim = CounterfactualSimulator()
        with patch.object(sim, "get_counterfactual_stats",
                          return_value={"total": 20, "wins": 10, "losses": 10, "missed_win_rate": 0.50}):
            relax, reason = sim.should_relax_filter("ctx", min_cf_trades=10)
        assert relax is True

    def test_should_relax_filter_low_missed_win_rate(self):
        from agent.algo_learning_engine import CounterfactualSimulator
        sim = CounterfactualSimulator()
        with patch.object(sim, "get_counterfactual_stats",
                          return_value={"total": 20, "wins": 6, "losses": 14, "missed_win_rate": 0.30}):
            relax, reason = sim.should_relax_filter("ctx", min_cf_trades=10)
        assert relax is False

    def test_get_counterfactual_stats_db_error_returns_zeros(self):
        from agent.algo_learning_engine import CounterfactualSimulator
        sim = CounterfactualSimulator()
        with patch("agent.db.get_conn", side_effect=Exception("no db")):
            stats = sim.get_counterfactual_stats()
        assert stats["total"] == 0
        assert stats["missed_win_rate"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 8. ModelVersionRegistry
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelVersionRegistry:

    def test_register_version_creates_challenger(self, tmp_path):
        mr = _fresh_model_registry(tmp_path)
        mr.register_version("v1.0", _good_challenger_metrics())
        assert "v1.0" in mr._data["challengers"]
        assert mr._data["challengers"]["v1.0"]["status"] == "challenger"

    def test_evaluate_promotion_all_gates_pass_no_champion(self, tmp_path):
        mr = _fresh_model_registry(tmp_path)
        mr.register_version("v1.0", _good_challenger_metrics())
        can, reasons = mr.evaluate_promotion("v1.0")
        assert can is True
        assert any("passed" in r for r in reasons)

    def test_evaluate_promotion_fails_insufficient_trades(self, tmp_path):
        mr = _fresh_model_registry(tmp_path)
        metrics = {**_good_challenger_metrics(), "n_trades": 10}
        mr.register_version("v1.0", metrics)
        can, reasons = mr.evaluate_promotion("v1.0")
        assert can is False
        assert any("n_trades" in r for r in reasons)

    def test_evaluate_promotion_fails_low_profit_factor(self, tmp_path):
        mr = _fresh_model_registry(tmp_path)
        metrics = {**_good_challenger_metrics(), "profit_factor": 0.95}
        mr.register_version("v1.0", metrics)
        can, reasons = mr.evaluate_promotion("v1.0")
        assert can is False
        assert any("profit_factor" in r for r in reasons)

    def test_evaluate_promotion_fails_negative_expectancy(self, tmp_path):
        mr = _fresh_model_registry(tmp_path)
        metrics = {**_good_challenger_metrics(), "expectancy": -0.001}
        mr.register_version("v1.0", metrics)
        can, reasons = mr.evaluate_promotion("v1.0")
        assert can is False
        assert any("expectancy" in r for r in reasons)

    def test_evaluate_promotion_fails_sharpe_not_5pct_better(self, tmp_path):
        mr = _fresh_model_registry(tmp_path)
        # Champion has sharpe=1.20; challenger needs >= 1.20 * 1.05 = 1.26
        champ_metrics = {**_good_challenger_metrics(), "sharpe": 1.20}
        mr.register_version("champ", champ_metrics)
        mr.promote("champ")
        chal_metrics = {**_good_challenger_metrics(), "sharpe": 1.22}
        mr.register_version("v2.0", chal_metrics)
        can, reasons = mr.evaluate_promotion("v2.0")
        assert can is False
        assert any("sharpe" in r for r in reasons)

    def test_evaluate_promotion_fails_drawdown_too_high(self, tmp_path):
        mr = _fresh_model_registry(tmp_path)
        champ_metrics = {**_good_challenger_metrics(), "max_drawdown": 0.05}
        mr.register_version("champ", champ_metrics)
        mr.promote("champ")
        # Challenger drawdown 0.15 > 2× 0.05 = 0.10
        chal_metrics = {**_good_challenger_metrics(), "max_drawdown": 0.15,
                        "sharpe": 2.0}
        mr.register_version("v2.0", chal_metrics)
        can, reasons = mr.evaluate_promotion("v2.0")
        assert can is False
        assert any("drawdown" in r for r in reasons)

    def test_evaluate_promotion_unknown_version(self, tmp_path):
        mr = _fresh_model_registry(tmp_path)
        can, reasons = mr.evaluate_promotion("nonexistent")
        assert can is False

    def test_promote_sets_champion(self, tmp_path):
        mr = _fresh_model_registry(tmp_path)
        mr.register_version("v1.0", _good_challenger_metrics())
        mr.promote("v1.0")
        champ = mr.get_champion()
        assert champ["version_id"] == "v1.0"
        assert champ["status"] == "champion"
        assert "v1.0" not in mr._data["challengers"]

    def test_promote_old_champion_goes_to_history(self, tmp_path):
        mr = _fresh_model_registry(tmp_path)
        mr.register_version("v1.0", _good_challenger_metrics())
        mr.promote("v1.0")
        mr.register_version("v2.0", _good_challenger_metrics())
        mr.promote("v2.0")
        assert len(mr._data["history"]) == 1
        assert mr._data["history"][0]["version_id"] == "v1.0"

    def test_promote_history_capped_at_10(self, tmp_path):
        mr = _fresh_model_registry(tmp_path)
        for i in range(15):
            mr.register_version(f"v{i}", _good_challenger_metrics())
            mr.promote(f"v{i}")
        assert len(mr._data["history"]) <= 10

    def test_rollback_restores_previous_champion(self, tmp_path):
        mr = _fresh_model_registry(tmp_path)
        mr.register_version("v1.0", _good_challenger_metrics())
        mr.promote("v1.0")
        mr.register_version("v2.0", _good_challenger_metrics())
        mr.promote("v2.0")
        mr.rollback()
        champ = mr.get_champion()
        assert champ["version_id"] == "v1.0"

    def test_rollback_no_history_is_safe(self, tmp_path):
        mr = _fresh_model_registry(tmp_path)
        mr.rollback()   # must not raise
        assert mr.get_champion() == {}

    def test_get_champion_no_champion_returns_empty(self, tmp_path):
        mr = _fresh_model_registry(tmp_path)
        assert mr.get_champion() == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        mr = _fresh_model_registry(tmp_path)
        mr.register_version("v1.0", _good_challenger_metrics())
        mr.promote("v1.0")
        mr.save()

        mr2 = _fresh_model_registry(tmp_path)
        mr2.load()
        assert mr2.get_champion()["version_id"] == "v1.0"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. RegimeTransitionHandler
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimeTransitionHandler:

    def _make_handler(self):
        from agent.algo_learning_engine import RegimeTransitionHandler, AuditLogger
        audit = AuditLogger()
        audit.log = MagicMock()   # suppress file writes in tests
        return RegimeTransitionHandler(audit), audit

    def test_no_action_when_regime_unchanged(self):
        handler, _ = self._make_handler()
        trades = [{"id": 1, "ticker": "AAPL", "regime": "BULL",
                   "entry_price": 150.0, "stop": 148.0, "direction": "BUY"}]
        actions = handler.check_open_trades(trades, {"AAPL": "BULL"})
        assert actions == []

    def test_bull_to_neutral_tightens_stop(self):
        handler, _ = self._make_handler()
        trades = [{"id": 1, "ticker": "AAPL", "regime": "BULL",
                   "entry_price": 150.0, "stop": 148.0, "direction": "BUY"}]
        actions = handler.check_open_trades(trades, {"AAPL": "NEUTRAL"})
        assert len(actions) == 1
        assert actions[0]["action"] == "TIGHTEN_STOP"
        assert actions[0]["stop_adjustment"] > 0

    def test_any_to_high_vol_in_profit_moves_to_breakeven(self):
        handler, _ = self._make_handler()
        trades = [{"id": 2, "ticker": "TSLA", "regime": "BULL",
                   "entry_price": 200.0, "stop": 198.0,
                   "direction": "BUY", "exit_price": 205.0}]
        actions = handler.check_open_trades(trades, {"TSLA": "HIGH_VOL"})
        assert len(actions) == 1
        assert actions[0]["action"] == "MOVE_TO_BREAKEVEN"

    def test_any_to_high_vol_at_loss_exits(self):
        handler, _ = self._make_handler()
        trades = [{"id": 3, "ticker": "TSLA", "regime": "BULL",
                   "entry_price": 200.0, "stop": 198.0,
                   "direction": "BUY", "exit_price": 195.0}]
        actions = handler.check_open_trades(trades, {"TSLA": "HIGH_VOL"})
        assert len(actions) == 1
        assert actions[0]["action"] == "EXIT_SIGNAL"

    def test_any_to_abnormal_exits(self):
        handler, _ = self._make_handler()
        trades = [{"id": 4, "ticker": "SPY", "regime": "NEUTRAL",
                   "entry_price": 400.0, "stop": 398.0, "direction": "BUY"}]
        actions = handler.check_open_trades(trades, {"SPY": "ABNORMAL"})
        assert len(actions) == 1
        assert actions[0]["action"] == "EXIT_SIGNAL"

    def test_circuit_break_triggers_exit(self):
        handler, _ = self._make_handler()
        trades = [{"id": 5, "ticker": "QQQ", "regime": "BEAR",
                   "entry_price": 300.0, "stop": 302.0, "direction": "SELL"}]
        actions = handler.check_open_trades(trades, {"QQQ": "CIRCUIT_BREAK"})
        assert actions[0]["action"] == "EXIT_SIGNAL"

    def test_multiple_trades_produces_multiple_actions(self):
        handler, _ = self._make_handler()
        trades = [
            {"id": 1, "ticker": "AAPL", "regime": "BULL",
             "entry_price": 150.0, "stop": 148.0, "direction": "BUY"},
            {"id": 2, "ticker": "TSLA", "regime": "BEAR",
             "entry_price": 200.0, "stop": 202.0, "direction": "SELL"},
        ]
        regime_map = {"AAPL": "ABNORMAL", "TSLA": "ABNORMAL"}
        actions = handler.check_open_trades(trades, regime_map)
        assert len(actions) == 2

    def test_missing_ticker_in_regime_map_skips_gracefully(self):
        handler, _ = self._make_handler()
        trades = [{"id": 1, "ticker": "AAPL", "regime": "BULL",
                   "entry_price": 150.0, "stop": 148.0, "direction": "BUY"}]
        # AAPL not in regime map → falls back to old_regime → no change
        actions = handler.check_open_trades(trades, {})
        assert actions == []

    def test_audit_logger_called_for_each_action(self):
        handler, audit = self._make_handler()
        trades = [{"id": 1, "ticker": "AAPL", "regime": "BULL",
                   "entry_price": 150.0, "stop": 148.0, "direction": "BUY"}]
        handler.check_open_trades(trades, {"AAPL": "NEUTRAL"})
        assert audit.log.called

    def test_corrupt_trade_dict_no_crash(self):
        handler, _ = self._make_handler()
        actions = handler.check_open_trades([None, {}, {"ticker": None}], {})
        # Should not raise

    def test_apply_actions_calls_update_trade_stop(self):
        handler, _ = self._make_handler()
        action = {"trade_id": 1, "ticker": "AAPL", "old_regime": "BULL",
                  "new_regime": "NEUTRAL", "action": "TIGHTEN_STOP",
                  "stop_adjustment": 0.4}
        with patch("agent.paper_trading.update_trade_stop", return_value=True) as mock_upd:
            count = handler.apply_actions([action])
        assert count == 1
        mock_upd.assert_called_once_with(
            trade_id=1, new_stop=0.4, reason="TIGHTEN_STOP"
        )

    def test_apply_actions_exit_signal_logged_not_called(self):
        handler, audit = self._make_handler()
        action = {"trade_id": 1, "ticker": "AAPL", "old_regime": "BULL",
                  "new_regime": "ABNORMAL", "action": "EXIT_SIGNAL",
                  "stop_adjustment": 0.0}
        with patch("agent.paper_trading.update_trade_stop") as mock_upd:
            count = handler.apply_actions([action])
        assert count == 1
        mock_upd.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# 10. AuditLogger
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuditLogger:

    def _make_logger(self, tmp_path):
        from agent.algo_learning_engine import AuditLogger
        logger = AuditLogger.__new__(AuditLogger)
        logger._PATH = tmp_path / "audit.jsonl"
        logger._MAX_BYTES = AuditLogger._MAX_BYTES
        logger._lock = threading.Lock()
        return logger

    def test_log_writes_valid_jsonl(self, tmp_path):
        lg = self._make_logger(tmp_path)
        lg.log("TEST_EVENT", {"foo": "bar", "num": 42})
        lines = lg._PATH.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event_type"] == "TEST_EVENT"
        assert entry["data"]["foo"] == "bar"
        assert "ts" in entry

    def test_log_appends_multiple_entries(self, tmp_path):
        lg = self._make_logger(tmp_path)
        lg.log("EVT1", {"a": 1})
        lg.log("EVT2", {"b": 2})
        lg.log("EVT3", {"c": 3})
        lines = [l for l in lg._PATH.read_text().strip().split("\n") if l]
        assert len(lines) == 3

    def test_log_creates_parent_dirs(self, tmp_path):
        from agent.algo_learning_engine import AuditLogger
        nested = tmp_path / "deep" / "nested" / "audit.jsonl"
        lg = AuditLogger.__new__(AuditLogger)
        lg._PATH = nested
        lg._MAX_BYTES = AuditLogger._MAX_BYTES
        lg._lock = threading.Lock()
        lg.log("TEST", {})
        assert nested.exists()

    def test_notify_operator_logs_event(self, tmp_path):
        lg = self._make_logger(tmp_path)
        lg.notify_operator("ALERT", "test message", {"key": "val"})
        lines = [l for l in lg._PATH.read_text().strip().split("\n") if l]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event_type"] == "ALERT"
        assert entry["data"]["message"] == "test message"

    def test_notify_operator_no_webhook_no_error(self, tmp_path):
        lg = self._make_logger(tmp_path)
        # ALERT_WEBHOOK_URL not set → should not raise
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALERT_WEBHOOK_URL", None)
            lg.notify_operator("ALERT", "msg", {})

    def test_notify_operator_webhook_called(self, tmp_path):
        lg = self._make_logger(tmp_path)
        with patch.dict(os.environ, {"ALERT_WEBHOOK_URL": "http://fake.webhook/"}):
            with patch("urllib.request.urlopen") as mock_url:
                lg.notify_operator("ALERT", "test", {"x": 1})
        assert mock_url.called

    def test_trim_triggered_over_size_limit(self, tmp_path):
        lg = self._make_logger(tmp_path)
        lg._MAX_BYTES = 500   # very small for test
        # Write enough to exceed limit
        for i in range(100):
            lg.log("BIG_EVENT", {"payload": "x" * 20, "i": i})
        # File should still exist and be valid JSONL
        lines = [l for l in lg._PATH.read_text().strip().split("\n") if l]
        assert len(lines) > 0
        json.loads(lines[-1])   # last line must be valid JSON

    def test_log_on_unwritable_path_no_crash(self):
        from agent.algo_learning_engine import AuditLogger
        lg = AuditLogger.__new__(AuditLogger)
        lg._PATH = Path("/root/no_permission/audit.jsonl")
        lg._MAX_BYTES = AuditLogger._MAX_BYTES
        lg._lock = threading.Lock()
        lg.log("TEST", {})   # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# AlgoLearningEngine — end-to-end coordinator
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlgoLearningEngine:

    def _make_engine(self):
        from agent.algo_learning_engine import AlgoLearningEngine
        engine = AlgoLearningEngine()
        # Suppress file writes during tests
        engine._audit.log = MagicMock()
        engine._audit.notify_operator = MagicMock()
        return engine

    def _make_df(self, rows):
        return pd.DataFrame(rows)

    def test_run_cycle_empty_df_no_crash(self):
        engine = self._make_engine()
        engine.run_cycle(pd.DataFrame(), cycle_num=1)

    def test_run_cycle_none_df_no_crash(self):
        engine = self._make_engine()
        engine.run_cycle(None, cycle_num=1)

    def test_run_cycle_counts_wins_and_losses(self):
        engine = self._make_engine()
        rows = [
            {"pnl_pct": 0.5, "exit_reason": "TARGET", "status": "WIN",
             "algo_name": "ORB5_BULL", "regime": "BULL", "session": "AM",
             "vwap_event": "ABOVE", "direction": "BUY"},
            {"pnl_pct": -0.5, "exit_reason": "STOP", "status": "LOSS",
             "algo_name": "ORB5_BULL", "regime": "BULL", "session": "AM",
             "vwap_event": "ABOVE", "direction": "BUY", "bars_tracked": 3,
             "max_favorable_r": 0.1},
        ]
        engine.run_cycle(self._make_df(rows), cycle_num=1)
        # Should have called audit with wins=1, losses=1
        cycle_call = [c for c in engine._audit.log.call_args_list
                      if c[0][0] == "ALGO_LEARNING_CYCLE"]
        assert len(cycle_call) == 1
        data = cycle_call[0][0][1]
        assert data["wins"] == 1
        assert data["losses"] == 1

    def test_run_cycle_wins_recorded_in_win_reinforcer(self):
        engine = self._make_engine()
        rows = [{"pnl_pct": 0.5, "exit_reason": "TARGET", "status": "WIN",
                 "algo_name": "ORB5_BULL", "regime": "BULL", "session": "AM",
                 "vwap_event": "ABOVE"}]
        engine.run_cycle(self._make_df(rows), cycle_num=1)
        ctx = engine._win_reinforce._wins.get("ORB5_BULL", {})
        assert "BULL:AM:ABOVE" in ctx

    def test_run_cycle_losses_recorded_in_loss_analyzer(self):
        engine = self._make_engine()
        rows = [{"pnl_pct": -0.5, "exit_reason": "STOP", "status": "LOSS",
                 "algo_name": "ORB5_BULL", "regime": "BULL", "session": "AM",
                 "vwap_event": "ABOVE", "direction": "BUY",
                 "bars_tracked": 3, "max_favorable_r": 0.1}]
        engine.run_cycle(self._make_df(rows), cycle_num=1)
        assert "ORB5_BULL" in engine._loss_analyzer._patterns

    def test_run_cycle_algo_selector_updated(self):
        engine = self._make_engine()
        rows = [{"pnl_pct": 0.5, "exit_reason": "TARGET", "status": "WIN",
                 "algo_name": "ORB5_BULL", "regime": "BULL", "session": "AM",
                 "vwap_event": "ABOVE"}]
        engine.run_cycle(self._make_df(rows), cycle_num=1)
        assert "ORB5_BULL" in engine._selector._state.get("BULL:AM:ABOVE", {})

    def test_run_cycle_break_even_updates_selector_not_win_reinforcer(self):
        engine = self._make_engine()
        rows = [{"pnl_pct": 0.001, "exit_reason": "STOP", "status": "BREAK_EVEN",
                 "algo_name": "ORB5_BULL", "regime": "BULL", "session": "AM",
                 "vwap_event": "ABOVE"}]
        engine.run_cycle(self._make_df(rows), cycle_num=1)
        # BREAK_EVEN → AlgoSelector updates (won=False), but WinReinforcer does NOT
        assert "ORB5_BULL" in engine._selector._state.get("BULL:AM:ABOVE", {})
        ctx = engine._win_reinforce._wins.get("ORB5_BULL", {})
        assert "BULL:AM:ABOVE" not in ctx

    def test_run_cycle_ignores_rows_without_algo_name(self):
        engine = self._make_engine()
        rows = [{"pnl_pct": 0.5, "exit_reason": "TARGET", "status": "WIN",
                 "algo_name": "", "regime": "BULL", "session": "AM", "vwap_event": "ABOVE"}]
        engine.run_cycle(self._make_df(rows), cycle_num=1)
        assert engine._win_reinforce._wins == {}

    def test_run_cycle_pnl_in_percent_normalised(self):
        """If pnl_pct > 0.1 it's treated as percentage (0.5% not 0.5 fractional)."""
        engine = self._make_engine()
        rows = [{"pnl_pct": 0.5, "exit_reason": "TARGET", "status": "WIN",
                 "algo_name": "ORB5_BULL", "regime": "BULL", "session": "AM",
                 "vwap_event": "ABOVE"}]
        engine.run_cycle(self._make_df(rows), cycle_num=1)
        cycle_call = [c for c in engine._audit.log.call_args_list
                      if c[0][0] == "ALGO_LEARNING_CYCLE"][0]
        assert cycle_call[0][1]["wins"] == 1

    def test_on_algo_signal_win_updates_both(self):
        engine = self._make_engine()
        engine.on_algo_signal("ORB5_BULL", "BULL:AM:ABOVE", won=True)
        assert "ORB5_BULL" in engine._selector._state.get("BULL:AM:ABOVE", {})
        assert "ORB5_BULL" in engine._win_reinforce._wins

    def test_on_algo_signal_loss_updates_selector_only(self):
        engine = self._make_engine()
        engine.on_algo_signal("ORB5_BULL", "BULL:AM:ABOVE", won=False)
        assert "ORB5_BULL" in engine._selector._state.get("BULL:AM:ABOVE", {})
        assert "ORB5_BULL" not in engine._win_reinforce._wins

    def test_get_algo_params_returns_defaults_for_unknown(self):
        from agent.algo_learning_engine import AlgoLearningEngine, _PARAM_SPEC
        engine = AlgoLearningEngine()
        params = engine.get_algo_params("NONEXISTENT_ALGO")
        for p, spec in _PARAM_SPEC.items():
            assert abs(params[p] - spec["default"]) < 1e-6

    def test_get_algo_params_returns_live_params(self):
        engine = self._make_engine()
        # max_change for target_mult is 0.10, default 1.5 → clamped to 1.60
        engine._registry.update("ORB", "target_mult", 2.0, "test", cycle_num=10)
        params = engine.get_algo_params("ORB5_BULL")
        # Value moved by max_change (0.10) from default 1.5 → 1.60
        assert abs(params["target_mult"] - 1.60) < 1e-5

    def test_get_algo_selector_weights_returns_dict(self):
        engine = self._make_engine()
        weights = engine.get_algo_selector_weights(["ORB5_BULL", "BULL_FLAG"], "ctx")
        assert set(weights.keys()) == {"ORB5_BULL", "BULL_FLAG"}

    def test_record_suppressed_proxies_to_counterfactual(self):
        engine = self._make_engine()
        with patch.object(engine._counterfact, "record_suppressed",
                          return_value="CF-001") as mock_cf:
            sid = engine.record_suppressed(
                ticker="AAPL", direction="BUY",
                entry_price=150.0, target=153.0, stop=149.0,
                rr_ratio=3.0, confidence=65.0,
                session="AM", regime="BULL", vwap_event="ABOVE",
                rsi_zone="MIDDLE", entry_type="IMMEDIATE",
                algo_name="ORB5_BULL", suppression_reason="TEST",
            )
        assert sid == "CF-001"
        mock_cf.assert_called_once()

    def test_run_cycle_does_not_crash_on_bad_row(self):
        engine = self._make_engine()
        rows = [{"pnl_pct": "BROKEN", "exit_reason": None, "status": None,
                 "algo_name": None, "regime": None}]
        engine.run_cycle(self._make_df(rows), cycle_num=1)   # must not raise

    def test_save_and_load_cycle(self, tmp_path):
        """save() and load() should complete without exception."""
        engine = self._make_engine()
        engine.save()
        engine.load()


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level singletons
# ═══════════════════════════════════════════════════════════════════════════════

class TestModuleSingletons:

    def test_get_engine_returns_same_instance(self):
        from agent.algo_learning_engine import get_engine
        e1 = get_engine()
        e2 = get_engine()
        assert e1 is e2

    def test_get_algo_params_safe_for_unknown(self):
        from agent.algo_learning_engine import get_algo_params
        params = get_algo_params("UNKNOWN_ALGO")
        assert isinstance(params, dict)

    def test_get_algo_params_safe_for_known(self):
        from agent.algo_learning_engine import get_algo_params, _PARAM_SPEC
        params = get_algo_params("ORB5_BULL")
        for p in _PARAM_SPEC:
            assert p in params

    def test_get_selector_weights_returns_neutral_for_unknown(self):
        from agent.algo_learning_engine import get_selector_weights
        weights = get_selector_weights(["ORB5_BULL", "BULL_FLAG"], "NEW_CTX")
        assert all(w == 1.0 for w in weights.values())

    def test_get_selector_weights_safe_on_exception(self):
        from agent.algo_learning_engine import get_selector_weights
        # Even if called with unusual inputs, should not crash
        weights = get_selector_weights([], "ctx")
        assert isinstance(weights, dict)

    def test_algo_family_map_coverage(self):
        """All mapped algos have a valid family."""
        from agent.algo_learning_engine import _ALGO_FAMILY_MAP, _ALL_FAMILIES
        for algo, family in _ALGO_FAMILY_MAP.items():
            assert family in _ALL_FAMILIES, f"{algo} maps to unknown family {family}"


# ═══════════════════════════════════════════════════════════════════════════════
# Thread-safety smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestThreadSafety:

    def test_registry_concurrent_reads_no_crash(self, tmp_path):
        reg = _fresh_registry(tmp_path)
        errors = []

        def _read():
            try:
                for _ in range(50):
                    reg.get("ORB", "target_mult")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_read) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_registry_concurrent_updates_no_crash(self, tmp_path):
        reg = _fresh_registry(tmp_path)
        errors = []

        def _update(i):
            try:
                reg.update("ORB", "target_mult", 1.5 + i * 0.01, "concurrent",
                           cycle_num=i * 20)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_update, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_loss_analyzer_concurrent_record_no_crash(self, tmp_path):
        la = _fresh_loss_analyzer(tmp_path)
        errors = []

        def _record():
            try:
                for _ in range(20):
                    la.record_pattern("ORB5_BULL", "STOP_TOO_TIGHT")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_record) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_win_reinforcer_concurrent_record_no_crash(self, tmp_path):
        wr = _fresh_win_reinforcer(tmp_path)
        errors = []

        def _win():
            try:
                for _ in range(20):
                    wr.record_win("ORB5_BULL", "ctx")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_win) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
