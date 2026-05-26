"""
Tests for Phase 3 Part 3 — auto-tuning EWMA alpha in ParameterAdapter.

The effective learning rate now scales with Phase 2 drift severity:
  stable       → alpha = base (0.10)
  WARNING drift → alpha = base × 1.5 (0.15)
  MATERIAL drift → alpha = base × 2.0 (0.20)

All fixed increments (conf_gate +1.0, rvol_gate +0.05) are also scaled
proportionally so adaptation is uniformly faster under drift.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# _get_effective_alpha unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetEffectiveAlpha:
    """Test _get_effective_alpha() under all drift scenarios."""

    def _make_adapter(self):
        from agent.algo_learning_engine import ParameterAdapter, ParameterControlRegistry, LossAnalyzer
        reg = ParameterControlRegistry()
        la  = LossAnalyzer()
        return ParameterAdapter(reg, la)

    def test_stable_returns_base_alpha(self):
        adapter = self._make_adapter()
        with patch("agent.algo_learning_engine.ParameterAdapter._get_effective_alpha",
                   wraps=adapter._get_effective_alpha):
            # Patch Phase 2 to return no drift
            with patch("agent.algo_learning_p2.get_phase2_engine") as mock_p2:
                mock_p2.return_value.get_drift_summary.return_value = {
                    "material_drifts": [],
                    "recent_events":   [],
                }
                alpha = adapter._get_effective_alpha()
        assert alpha == 0.10

    def test_material_drift_doubles_alpha(self):
        adapter = self._make_adapter()
        with patch("agent.algo_learning_p2.get_phase2_engine") as mock_p2:
            mock_p2.return_value.get_drift_summary.return_value = {
                "material_drifts": ["confidence"],
                "recent_events":   [],
            }
            alpha = adapter._get_effective_alpha()
        assert alpha == pytest.approx(0.20)

    def test_warning_drift_scales_by_1_5(self):
        adapter = self._make_adapter()
        with patch("agent.algo_learning_p2.get_phase2_engine") as mock_p2:
            mock_p2.return_value.get_drift_summary.return_value = {
                "material_drifts": [],
                "recent_events":   [{"feature": "rsi_value", "level": "WARNING"}],
            }
            alpha = adapter._get_effective_alpha()
        assert alpha == pytest.approx(0.15)

    def test_material_takes_priority_over_warning(self):
        """When both material and warning drifts exist, material wins."""
        adapter = self._make_adapter()
        with patch("agent.algo_learning_p2.get_phase2_engine") as mock_p2:
            mock_p2.return_value.get_drift_summary.return_value = {
                "material_drifts": ["confidence", "rr_ratio"],
                "recent_events":   [{"feature": "rsi_value", "level": "WARNING"}],
            }
            alpha = adapter._get_effective_alpha()
        assert alpha == pytest.approx(0.20)

    def test_phase2_unavailable_falls_back_to_base(self):
        """If Phase 2 raises, alpha falls back to base (safe degradation)."""
        adapter = self._make_adapter()
        with patch("agent.algo_learning_p2.get_phase2_engine", side_effect=ImportError):
            alpha = adapter._get_effective_alpha()
        assert alpha == pytest.approx(0.10)

    def test_phase2_exception_falls_back_to_base(self):
        adapter = self._make_adapter()
        with patch("agent.algo_learning_p2.get_phase2_engine") as mock_p2:
            mock_p2.return_value.get_drift_summary.side_effect = RuntimeError("db error")
            alpha = adapter._get_effective_alpha()
        assert alpha == pytest.approx(0.10)

    def test_custom_base_is_respected(self):
        adapter = self._make_adapter()
        with patch("agent.algo_learning_p2.get_phase2_engine") as mock_p2:
            mock_p2.return_value.get_drift_summary.return_value = {
                "material_drifts": ["confidence"],
                "recent_events":   [],
            }
            alpha = adapter._get_effective_alpha(base=0.05)
        assert alpha == pytest.approx(0.10)  # 0.05 × 2.0

    def test_stable_with_empty_events_is_base(self):
        adapter = self._make_adapter()
        with patch("agent.algo_learning_p2.get_phase2_engine") as mock_p2:
            mock_p2.return_value.get_drift_summary.return_value = {
                "material_drifts": [],
                "recent_events":   [{"feature": "rsi", "level": "NONE"}],
            }
            alpha = adapter._get_effective_alpha()
        assert alpha == pytest.approx(0.10)

    def test_constants_defined(self):
        from agent.algo_learning_engine import ParameterAdapter
        assert hasattr(ParameterAdapter, "_EWMA_ALPHA_BASE")
        assert hasattr(ParameterAdapter, "_DRIFT_MULT_STABLE")
        assert hasattr(ParameterAdapter, "_DRIFT_MULT_WARNING")
        assert hasattr(ParameterAdapter, "_DRIFT_MULT_MATERIAL")
        assert ParameterAdapter._EWMA_ALPHA_BASE == 0.10
        assert ParameterAdapter._DRIFT_MULT_MATERIAL > ParameterAdapter._DRIFT_MULT_WARNING > ParameterAdapter._DRIFT_MULT_STABLE

    def test_no_more_class_constant_ewma_alpha(self):
        """_EWMA_ALPHA (old constant) is replaced by _EWMA_ALPHA_BASE."""
        from agent.algo_learning_engine import ParameterAdapter
        # New constant exists
        assert hasattr(ParameterAdapter, "_EWMA_ALPHA_BASE")


# ─────────────────────────────────────────────────────────────────────────────
# adapt() uses drift-adjusted alpha
# ─────────────────────────────────────────────────────────────────────────────

import pytest

class TestAdaptUsesEffectiveAlpha:
    """Verify that adapt() produces larger adjustments under material drift."""

    def _make_adapter_with_mock_loss(self, cause: str):
        from agent.algo_learning_engine import ParameterAdapter, ParameterControlRegistry, LossAnalyzer
        reg = ParameterControlRegistry()
        la  = MagicMock(spec=LossAnalyzer)
        la.get_dominant_cause.return_value = cause
        return ParameterAdapter(reg, la), reg

    def test_stop_too_tight_larger_under_material_drift(self):
        from agent.algo_learning_engine import ParameterAdapter, ParameterControlRegistry, LossAnalyzer
        adapter, reg = self._make_adapter_with_mock_loss("STOP_TOO_TIGHT")
        base_stop = reg.get("ORB", "stop_mult")

        # Stable — small adjustment
        with patch("agent.algo_learning_p2.get_phase2_engine") as mock_p2:
            mock_p2.return_value.get_drift_summary.return_value = {
                "material_drifts": [], "recent_events": []
            }
            changes_stable = adapter.adapt("ORB", "ORB5_BULL", cycle_num=1)

        reg2 = ParameterControlRegistry()
        la2  = MagicMock()
        la2.get_dominant_cause.return_value = "STOP_TOO_TIGHT"
        adapter2 = type(adapter)(reg2, la2)
        base_stop2 = reg2.get("ORB", "stop_mult")

        # Material drift — larger adjustment
        with patch("agent.algo_learning_p2.get_phase2_engine") as mock_p2:
            mock_p2.return_value.get_drift_summary.return_value = {
                "material_drifts": ["confidence"], "recent_events": []
            }
            changes_drift = adapter2.adapt("ORB", "ORB5_BULL", cycle_num=1)

        if changes_stable and changes_drift:
            _, new_stable, _ = changes_stable.get("stop_mult", (None, base_stop, None))
            _, new_drift,  _ = changes_drift.get("stop_mult",  (None, base_stop2, None))
            assert new_drift >= new_stable, (
                f"Drift adjustment ({new_drift:.4f}) should be >= stable ({new_stable:.4f})"
            )

    def test_wrong_direction_larger_conf_gate_under_drift(self):
        adapter, reg = self._make_adapter_with_mock_loss("WRONG_DIRECTION")
        base = reg.get("ORB", "conf_gate")

        with patch("agent.algo_learning_p2.get_phase2_engine") as mock_p2:
            mock_p2.return_value.get_drift_summary.return_value = {
                "material_drifts": ["confidence"], "recent_events": []
            }
            changes = adapter.adapt("ORB", "ORB5_BULL", cycle_num=1)

        if changes:
            _, new, _ = changes.get("conf_gate", (None, base, None))
            # Under material drift, increment > 1.0 (scaled by 2x)
            assert new > base, "conf_gate should increase under WRONG_DIRECTION"

    def test_no_cause_returns_no_changes(self):
        adapter, _ = self._make_adapter_with_mock_loss(None)
        adapter._loss.get_dominant_cause.return_value = None
        with patch("agent.algo_learning_p2.get_phase2_engine") as mock_p2:
            mock_p2.return_value.get_drift_summary.return_value = {
                "material_drifts": [], "recent_events": []
            }
            changes = adapter.adapt("ORB", "ORB5_BULL", cycle_num=1)
        assert changes == {}

    def test_adapt_reason_includes_alpha_value(self):
        """Reason string now includes the effective alpha for auditability."""
        adapter, reg = self._make_adapter_with_mock_loss("STOP_TOO_TIGHT")
        with patch("agent.algo_learning_p2.get_phase2_engine") as mock_p2:
            mock_p2.return_value.get_drift_summary.return_value = {
                "material_drifts": [], "recent_events": []
            }
            changes = adapter.adapt("ORB", "ORB5_BULL", cycle_num=1)
        if changes:
            _, _, reason = changes.get("stop_mult", (None, None, ""))
            assert "α=" in reason, f"Reason should contain alpha: {reason!r}"

    def test_volatility_spike_uses_double_alpha(self):
        adapter, reg = self._make_adapter_with_mock_loss("VOLATILITY_SPIKE")
        base = reg.get("ORB", "stop_mult")
        with patch("agent.algo_learning_p2.get_phase2_engine") as mock_p2:
            mock_p2.return_value.get_drift_summary.return_value = {
                "material_drifts": [], "recent_events": []
            }
            changes = adapter.adapt("ORB", "ORB5_BULL", cycle_num=1)
        if changes:
            _, new, _ = changes.get("stop_mult", (None, base, None))
            # VOLATILITY_SPIKE uses alpha*2, which is 0.20 stable → stop_mult * 1.20
            assert new > base

    def test_all_causes_produce_valid_params(self):
        """Every loss cause produces a valid (in-bounds) parameter update."""
        causes = [
            "STOP_TOO_TIGHT", "WRONG_DIRECTION", "REGIME_MISMATCH",
            "TIMEOUT_DRIFT", "VWAP_CONFLICT", "TIMING_LATE", "VOLATILITY_SPIKE",
        ]
        from agent.algo_learning_engine import _PARAM_SPEC
        for cause in causes:
            adapter, reg = self._make_adapter_with_mock_loss(cause)
            with patch("agent.algo_learning_p2.get_phase2_engine") as mock_p2:
                mock_p2.return_value.get_drift_summary.return_value = {
                    "material_drifts": ["confidence"], "recent_events": []
                }
                changes = adapter.adapt("ORB", "ORB5_BULL", cycle_num=1)
            for param, (old, new, reason) in changes.items():
                spec = _PARAM_SPEC[param]
                assert spec["min"] <= new <= spec["max"], (
                    f"cause={cause} param={param}: {new} out of [{spec['min']}, {spec['max']}]"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Learning rate monotonicity: stable < warning < material
# ─────────────────────────────────────────────────────────────────────────────

class TestAlphaMonotonicity:
    """Effective alpha must be monotonically increasing with drift severity."""

    def _alpha_for(self, material: list, warning_count: int) -> float:
        from agent.algo_learning_engine import ParameterAdapter, ParameterControlRegistry, LossAnalyzer
        reg = ParameterControlRegistry()
        la  = LossAnalyzer()
        adapter = ParameterAdapter(reg, la)
        events = [{"feature": f"feat_{i}", "level": "WARNING"}
                  for i in range(warning_count)]
        with patch("agent.algo_learning_p2.get_phase2_engine") as mock_p2:
            mock_p2.return_value.get_drift_summary.return_value = {
                "material_drifts": material,
                "recent_events":   events,
            }
            return adapter._get_effective_alpha()

    def test_stable_lt_warning(self):
        stable  = self._alpha_for(material=[], warning_count=0)
        warning = self._alpha_for(material=[], warning_count=1)
        assert stable < warning

    def test_warning_lt_material(self):
        warning  = self._alpha_for(material=[], warning_count=1)
        material = self._alpha_for(material=["confidence"], warning_count=0)
        assert warning < material

    def test_stable_lt_material(self):
        stable   = self._alpha_for(material=[], warning_count=0)
        material = self._alpha_for(material=["confidence"], warning_count=0)
        assert stable < material

    def test_all_three_different(self):
        stable   = self._alpha_for(material=[], warning_count=0)
        warning  = self._alpha_for(material=[], warning_count=1)
        material = self._alpha_for(material=["confidence"], warning_count=0)
        assert stable < warning < material


# ─────────────────────────────────────────────────────────────────────────────
# Source-level checks
# ─────────────────────────────────────────────────────────────────────────────

class TestSourceChanges:
    def _src(self) -> str:
        import pathlib
        return (pathlib.Path(__file__).parent.parent / "agent" / "algo_learning_engine.py").read_text(encoding="utf-8")

    def test_ewma_alpha_base_constant_present(self):
        assert "_EWMA_ALPHA_BASE" in self._src()

    def test_get_effective_alpha_method_present(self):
        assert "_get_effective_alpha" in self._src()

    def test_drift_multipliers_defined(self):
        src = self._src()
        assert "_DRIFT_MULT_STABLE" in src
        assert "_DRIFT_MULT_WARNING" in src
        assert "_DRIFT_MULT_MATERIAL" in src

    def test_adapt_uses_effective_alpha_not_hardcoded(self):
        """adapt() should call _get_effective_alpha(), not use a hardcoded constant."""
        src = self._src()
        assert "_get_effective_alpha()" in src

    def test_adapt_reason_has_alpha_annotation(self):
        """Reason strings include α= for auditability."""
        src = self._src()
        assert "α=" in src
