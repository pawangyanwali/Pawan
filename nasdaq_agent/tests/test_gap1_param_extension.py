"""
test_gap1_param_extension.py
============================
Comprehensive tests for Gap 1: _param() helper extension to all algo functions.

Tests cover:
  1. Engine unavailable (_ALE_AVAILABLE = False) → each algo uses hardcoded default rvol_gate
  2. Engine available → _param() is called with the correct family and "rvol_gate"
  3. Signals fire when rvol >= learned gate, don't fire when rvol < learned gate
  4. ORB15 specifically: target_mult is also parameterized correctly
  5. Dedicated tests for at least 3 specific algo functions
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

# Ensure nasdaq_agent is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import agent.trading_algos as ta


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _base_orb15_bull(rvol=1.6, target_mult=1.5) -> SimpleNamespace:
    """Minimal signal that fires ORB15_BULL."""
    return SimpleNamespace(
        orb15_high=101.0,
        orb15_low=100.0,
        orb15_breakout="BULL",
        rel_volume=rvol,
        short_tf_alignment="BULL",
        price=101.5,
    )


def _base_gap_and_go_bull(rvol=2.5) -> SimpleNamespace:
    """Minimal signal that fires GAP_AND_GO_BULL."""
    return SimpleNamespace(
        gap_pct=3.0,
        gap_type="GAP_UP",
        color_vs_prev_close="GREEN",
        rel_volume=rvol,
        short_tf_alignment="BULL",
        price=103.0,
        session_low=100.0,
        session_high=104.0,
        today_open=100.0,
    )


def _base_gap_fade_bear(rvol=1.6) -> SimpleNamespace:
    """Minimal signal that fires GAP_FADE_BEAR (short after gap-up reversal)."""
    return SimpleNamespace(
        gap_pct=2.0,
        gap_type="GAP_UP",
        rel_volume=rvol,
        short_tf_alignment="MIXED",
        price=99.0,       # price < today_open → fading gap
        today_open=100.0,
        prev_day_close=97.0,
        session_high=101.0,
        session_low=98.0,
    )


def _base_pdh_pdl_bull(rvol=1.6) -> SimpleNamespace:
    """Minimal signal that fires PDH_BREAKOUT_BULL."""
    return SimpleNamespace(
        prev_day_high=100.0,
        prev_day_low=98.0,
        rel_volume=rvol,
        short_tf_alignment="BULL",
        price=101.0,
    )


def _base_hod_lod_bull(rvol=1.6) -> SimpleNamespace:
    """Minimal signal that fires HOD_BREAK_BULL."""
    return SimpleNamespace(
        session_high=105.0,
        session_low=100.0,
        orb15_high=102.0,
        orb5_high=0.0,
        rel_volume=rvol,
        short_tf_alignment="BULL",
        price=105.0,   # price == session_high (new HOD, within buf)
    )


def _base_vwap_touch_scalp_bull(rvol=1.4) -> SimpleNamespace:
    """Minimal signal that fires VWAP_TOUCH_SCALP_BULL."""
    return SimpleNamespace(
        vwap_event="RECLAIM",
        vwap_price=100.0,
        vwap_upper_1=101.0,
        vwap_lower_1=99.0,
        rel_volume=rvol,
        short_tf_alignment="BULL",
        price=100.1,
    )


def _base_vwap_hod_scalp(rvol=1.3) -> SimpleNamespace:
    """Minimal signal that fires VWAP_HOD_SCALP."""
    return SimpleNamespace(
        vwap_event="ABOVE",
        vwap_price=100.0,
        vwap_z_score=0.5,
        vwap_lower_1=99.0,
        session_high=105.0,
        rel_volume=rvol,
        short_tf_alignment="BULL",
        price=100.5,
    )


def _base_vwap_lod_scalp(rvol=1.3) -> SimpleNamespace:
    """Minimal signal that fires VWAP_LOD_SCALP."""
    return SimpleNamespace(
        vwap_event="BELOW",
        vwap_price=100.0,
        vwap_z_score=-0.5,
        vwap_upper_1=101.0,
        session_low=95.0,
        rel_volume=rvol,
        short_tf_alignment="BEAR",
        price=99.5,
    )


def _base_level_rejection_scalp_bull(rvol=1.3) -> SimpleNamespace:
    """Minimal signal that fires LEVEL_REJECTION_SCALP_BULL."""
    return SimpleNamespace(
        vwap_event="AT_2SD_DOWN",
        vwap_price=100.0,
        vwap_upper_2=103.0,
        vwap_lower_2=97.0,
        rel_volume=rvol,
        short_tf_alignment="BULL",
        price=97.1,
    )


def _base_micro_pullback_scalp_bull(rvol=1.4) -> SimpleNamespace:
    """Minimal signal that fires MICRO_PULLBACK_SCALP_BULL."""
    return SimpleNamespace(
        vwap_event="ABOVE",
        vwap_price=100.0,
        vwap_z_score=0.5,
        vwap_upper_1=101.5,
        vwap_lower_1=98.5,
        rel_volume=rvol,
        short_tf_alignment="BULL",
        mtf_gate_passed=True,
        price=100.2,
    )


def _base_spy_beta_catchup_bull(rvol=1.2) -> SimpleNamespace:
    """Minimal signal that fires SPY_BETA_CATCHUP_BULL."""
    return SimpleNamespace(
        rs_ratio=0.4,      # < 0.55 → lagging
        rs_label="IN_LINE",
        sector_change=1.0, # > 0.5%
        change_pct=0.1,    # positive
        rel_volume=rvol,
        regime="BULL_TREND",
        price=100.0,
        vwap_price=99.0,
        vwap_upper_1=101.0,
        vwap_lower_1=98.0,
    )


def _base_residual_momentum(rvol=1.3) -> SimpleNamespace:
    """Minimal signal that fires RESIDUAL_MOMENTUM_BULL."""
    return SimpleNamespace(
        rs_ratio=1.8,
        rs_label="LEADING",
        sector_trend="BULLISH",
        short_tf_alignment="BULL",
        rel_volume=rvol,
        price=105.0,
        vwap_price=100.0,
        vwap_lower_1=99.0,
        session_high=110.0,
        session_low=100.0,
    )


def _base_residual_reversion(rvol=1.2) -> SimpleNamespace:
    """Minimal signal that fires RESIDUAL_REVERSION_BEAR."""
    return SimpleNamespace(
        rs_ratio=3.0,
        rs_label="LEADING",
        vwap_z_score=2.0,
        sector_trend="NEUTRAL",
        vwap_price=100.0,
        vwap_upper_2=110.0,
        rel_volume=rvol,
        short_tf_alignment="BEAR",
        price=108.0,
    )


def _base_sector_leader(rvol=1.3) -> SimpleNamespace:
    """Minimal signal that fires SECTOR_LEADER_BULL."""
    return SimpleNamespace(
        stock_vs_sector="LEADING",
        sector_trend="BULLISH",
        sector_change=1.0,
        rel_volume=rvol,
        short_tf_alignment="BULL",
        price=105.0,
        vwap_price=100.0,
        vwap_lower_1=99.0,
        session_high=110.0,
        session_low=100.0,
        sector_etf="XLK",
    )


def _base_sector_laggard_catchup(rvol=1.2) -> SimpleNamespace:
    """Minimal signal that fires SECTOR_LAGGARD_CATCHUP_BULL."""
    return SimpleNamespace(
        stock_vs_sector="LAGGING",
        sector_trend="BULLISH",
        sector_change=0.5,
        rs_label="IN_LINE",
        rel_volume=rvol,
        short_tf_alignment="BULL",
        price=100.0,
        vwap_price=100.5,
        vwap_lower_1=99.0,
        sector_etf="XLK",
    )


def _base_sector_counter_fade(rvol=1.2) -> SimpleNamespace:
    """Minimal signal that fires SECTOR_COUNTER_FADE_BEAR."""
    return SimpleNamespace(
        stock_vs_sector="LEADING",
        sector_trend="BEARISH",
        sector_change=-0.5,
        short_tf_alignment="MIXED",
        rel_volume=rvol,
        vwap_z_score=1.0,
        price=105.0,
        vwap_price=100.0,
        session_high=106.0,
        vwap_upper_2=108.0,
        sector_etf="XLK",
    )


def _base_regime_aligned_long(rvol=1.3) -> SimpleNamespace:
    """Minimal signal that fires REGIME_ALIGNED_LONG."""
    return SimpleNamespace(
        regime="BULL_TREND",
        rs_label="LEADING",
        sector_trend="BULLISH",
        vwap_event="ABOVE",
        short_tf_alignment="BULL",
        rel_volume=rvol,
        price=105.0,
        vwap_price=100.0,
        vwap_lower_1=99.0,
        session_high=110.0,
        session_low=100.0,
        rs_ratio=1.5,
    )


def _base_regime_aligned_short(rvol=1.3) -> SimpleNamespace:
    """Minimal signal that fires REGIME_ALIGNED_SHORT."""
    return SimpleNamespace(
        regime="BEAR_TREND",
        stock_vs_sector="LAGGING",
        sector_trend="BEARISH",
        vwap_event="BELOW",
        short_tf_alignment="BEAR",
        rel_volume=rvol,
        price=95.0,
        vwap_price=100.0,
        vwap_upper_1=101.0,
        session_high=100.0,
        session_low=90.0,
        rs_ratio=0.8,
    )


def _base_sector_breakout_follow_bull(rvol=1.2) -> SimpleNamespace:
    """Minimal signal that fires SECTOR_BREAKOUT_BULL."""
    return SimpleNamespace(
        sector_change=1.0,
        stock_vs_sector="IN_LINE",
        rs_label="IN_LINE",
        vwap_event="ABOVE",
        short_tf_alignment="BULL",
        rel_volume=rvol,
        price=101.0,
        vwap_price=100.0,
        vwap_upper_1=102.0,
        vwap_lower_1=98.0,
        sector_etf="XLK",
    )


def _base_cross_sectional_rs_bull(rvol=1.3) -> SimpleNamespace:
    """Minimal signal that fires CS_RS_RANK_BULL."""
    return SimpleNamespace(
        rs_score=0.8,
        rs_label="LEADING",
        sector_trend="BULLISH",
        rel_volume=rvol,
        short_tf_alignment="BULL",
        price=105.0,
        vwap_price=100.0,
        vwap_upper_1=106.0,
        vwap_lower_1=99.0,
        session_high=110.0,
        session_low=100.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Engine unavailable → hardcoded defaults are used
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineUnavailable:
    """When _ALE_AVAILABLE is False, each algo falls back to its hardcoded default."""

    def _assert_fires_at_default(self, fn, sig_factory, default_gate):
        """Signal fires just at the default rvol threshold, not below."""
        with patch.object(ta, "_ALE_AVAILABLE", False):
            sig_above = sig_factory(rvol=default_gate + 0.01)
            sig_below = sig_factory(rvol=default_gate - 0.01)
            assert fn(sig_above) is not None, (
                f"{fn.__name__} should fire with rvol={default_gate + 0.01} "
                f"(default gate={default_gate})"
            )
            assert fn(sig_below) is None, (
                f"{fn.__name__} should NOT fire with rvol={default_gate - 0.01} "
                f"(default gate={default_gate})"
            )

    def test_orb15_default(self):
        self._assert_fires_at_default(ta.eval_orb15, _base_orb15_bull, 1.5)

    def test_gap_and_go_default(self):
        self._assert_fires_at_default(ta.eval_gap_and_go, _base_gap_and_go_bull, 2.0)

    def test_gap_fade_default(self):
        self._assert_fires_at_default(ta.eval_gap_fade, _base_gap_fade_bear, 1.5)

    def test_pdh_pdl_breakout_default(self):
        self._assert_fires_at_default(ta.eval_pdh_pdl_breakout, _base_pdh_pdl_bull, 1.5)

    def test_hod_lod_break_default(self):
        self._assert_fires_at_default(ta.eval_hod_lod_break, _base_hod_lod_bull, 1.5)

    def test_vwap_touch_scalp_default(self):
        self._assert_fires_at_default(ta.eval_vwap_touch_scalp, _base_vwap_touch_scalp_bull, 1.3)

    def test_vwap_hod_scalp_default(self):
        self._assert_fires_at_default(ta.eval_vwap_hod_scalp, _base_vwap_hod_scalp, 1.2)

    def test_vwap_lod_scalp_default(self):
        self._assert_fires_at_default(ta.eval_vwap_lod_scalp, _base_vwap_lod_scalp, 1.2)

    def test_level_rejection_scalp_default(self):
        self._assert_fires_at_default(ta.eval_level_rejection_scalp, _base_level_rejection_scalp_bull, 1.2)

    def test_micro_pullback_scalp_default(self):
        self._assert_fires_at_default(ta.eval_micro_pullback_scalp, _base_micro_pullback_scalp_bull, 1.3)

    def test_spy_beta_catchup_default(self):
        self._assert_fires_at_default(ta.eval_spy_beta_catchup, _base_spy_beta_catchup_bull, 1.1)

    def test_residual_momentum_default(self):
        self._assert_fires_at_default(ta.eval_residual_momentum, _base_residual_momentum, 1.2)

    def test_residual_reversion_default(self):
        self._assert_fires_at_default(ta.eval_residual_reversion, _base_residual_reversion, 1.1)

    def test_sector_leader_default(self):
        self._assert_fires_at_default(ta.eval_sector_leader, _base_sector_leader, 1.2)

    def test_sector_laggard_catchup_default(self):
        self._assert_fires_at_default(ta.eval_sector_laggard_catchup, _base_sector_laggard_catchup, 1.1)

    def test_sector_counter_fade_default(self):
        self._assert_fires_at_default(ta.eval_sector_counter_fade, _base_sector_counter_fade, 1.1)

    def test_regime_aligned_long_default(self):
        self._assert_fires_at_default(ta.eval_regime_aligned_long, _base_regime_aligned_long, 1.2)

    def test_regime_aligned_short_default(self):
        self._assert_fires_at_default(ta.eval_regime_aligned_short, _base_regime_aligned_short, 1.2)

    def test_sector_breakout_follow_default(self):
        self._assert_fires_at_default(ta.eval_sector_breakout_follow, _base_sector_breakout_follow_bull, 1.1)

    def test_cross_sectional_rs_default(self):
        self._assert_fires_at_default(ta.eval_cross_sectional_rs, _base_cross_sectional_rs_bull, 1.2)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Engine available → _param() called with correct family + "rvol_gate"
# ─────────────────────────────────────────────────────────────────────────────

class TestParamCalledWithCorrectFamily:
    """When engine is available, _param() uses the correct algo family."""

    def _assert_param_called(self, fn, sig_factory, expected_family, rvol=2.0):
        """Patch get_algo_params and verify it's called with the right family."""
        mock_params = {"rvol_gate": rvol - 0.1, "target_mult": 1.5, "stop_mult": 1.0}
        sig = sig_factory(rvol=rvol)
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", return_value=mock_params) as mock_gp:
            fn(sig)
            # _get_algo_params should have been called at least once with the expected family
            calls = [c.args[0] for c in mock_gp.call_args_list]
            assert expected_family in calls, (
                f"{fn.__name__}: expected family '{expected_family}' in calls {calls}"
            )

    def test_orb15_family(self):
        self._assert_param_called(ta.eval_orb15, _base_orb15_bull, "ORB")

    def test_gap_and_go_family(self):
        self._assert_param_called(ta.eval_gap_and_go, _base_gap_and_go_bull, "GAP_TREND", rvol=2.5)

    def test_gap_fade_family(self):
        self._assert_param_called(ta.eval_gap_fade, _base_gap_fade_bear, "GAP_FADE")

    def test_pdh_pdl_breakout_family(self):
        self._assert_param_called(ta.eval_pdh_pdl_breakout, _base_pdh_pdl_bull, "BREAKOUT")

    def test_hod_lod_break_family(self):
        self._assert_param_called(ta.eval_hod_lod_break, _base_hod_lod_bull, "BREAKOUT")

    def test_vwap_touch_scalp_family(self):
        self._assert_param_called(ta.eval_vwap_touch_scalp, _base_vwap_touch_scalp_bull, "VWAP_SCALP")

    def test_vwap_hod_scalp_family(self):
        self._assert_param_called(ta.eval_vwap_hod_scalp, _base_vwap_hod_scalp, "VWAP_SCALP")

    def test_vwap_lod_scalp_family(self):
        self._assert_param_called(ta.eval_vwap_lod_scalp, _base_vwap_lod_scalp, "VWAP_SCALP")

    def test_level_rejection_scalp_family(self):
        self._assert_param_called(ta.eval_level_rejection_scalp, _base_level_rejection_scalp_bull, "LEVEL_SCALP")

    def test_micro_pullback_scalp_family(self):
        self._assert_param_called(ta.eval_micro_pullback_scalp, _base_micro_pullback_scalp_bull, "LEVEL_SCALP")

    def test_spy_beta_catchup_family(self):
        self._assert_param_called(ta.eval_spy_beta_catchup, _base_spy_beta_catchup_bull, "RS_REGIME")

    def test_residual_momentum_family(self):
        self._assert_param_called(ta.eval_residual_momentum, _base_residual_momentum, "RS_REGIME")

    def test_residual_reversion_family(self):
        self._assert_param_called(ta.eval_residual_reversion, _base_residual_reversion, "RS_REGIME")

    def test_sector_leader_family(self):
        self._assert_param_called(ta.eval_sector_leader, _base_sector_leader, "RS_REGIME")

    def test_sector_laggard_catchup_family(self):
        self._assert_param_called(ta.eval_sector_laggard_catchup, _base_sector_laggard_catchup, "RS_REGIME")

    def test_sector_counter_fade_family(self):
        self._assert_param_called(ta.eval_sector_counter_fade, _base_sector_counter_fade, "RS_REGIME")

    def test_regime_aligned_long_family(self):
        self._assert_param_called(ta.eval_regime_aligned_long, _base_regime_aligned_long, "RS_REGIME")

    def test_regime_aligned_short_family(self):
        self._assert_param_called(ta.eval_regime_aligned_short, _base_regime_aligned_short, "RS_REGIME")

    def test_sector_breakout_follow_family(self):
        self._assert_param_called(ta.eval_sector_breakout_follow, _base_sector_breakout_follow_bull, "RS_REGIME")

    def test_cross_sectional_rs_family(self):
        self._assert_param_called(ta.eval_cross_sectional_rs, _base_cross_sectional_rs_bull, "RS_REGIME")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Learned gate controls signal firing (parameterized)
# ─────────────────────────────────────────────────────────────────────────────

class TestLearnedGateControlsFiring:
    """
    Verify that when the engine returns a LEARNED rvol_gate, signals fire/don't fire
    according to that learned value rather than the hardcoded default.
    """

    def _test_learned_gate(self, fn, sig_factory, learned_gate, default_gate):
        """
        With a learned rvol_gate that differs from the default:
          - rvol just above learned_gate → fires
          - rvol between learned_gate and default_gate → may or may not fire depending
            on direction, but we confirm the LEARNED gate is honoured
        """
        mock_params = {"rvol_gate": learned_gate, "target_mult": 1.5, "stop_mult": 1.0}
        sig_above = sig_factory(rvol=learned_gate + 0.05)
        sig_below = sig_factory(rvol=learned_gate - 0.05)

        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", return_value=mock_params):
            result_above = fn(sig_above)
            result_below = fn(sig_below)

        assert result_above is not None, (
            f"{fn.__name__}: should fire with rvol={learned_gate + 0.05} "
            f"(learned gate={learned_gate})"
        )
        assert result_below is None, (
            f"{fn.__name__}: should NOT fire with rvol={learned_gate - 0.05} "
            f"(learned gate={learned_gate})"
        )

    def test_orb15_learned_gate_lower(self):
        """Learned gate 1.2 (lower than default 1.5) → fires at 1.25."""
        self._test_learned_gate(ta.eval_orb15, _base_orb15_bull, 1.2, 1.5)

    def test_orb15_learned_gate_higher(self):
        """Learned gate 2.0 (higher than default 1.5) → fires at 2.05, not 1.95."""
        self._test_learned_gate(ta.eval_orb15, _base_orb15_bull, 2.0, 1.5)

    def test_gap_and_go_learned_gate_lower(self):
        """GAP_TREND: learned gate 1.5 (lower than default 2.0)."""
        self._test_learned_gate(ta.eval_gap_and_go, _base_gap_and_go_bull, 1.5, 2.0)

    def test_gap_and_go_learned_gate_higher(self):
        """GAP_TREND: learned gate 2.5 (higher than default 2.0)."""
        self._test_learned_gate(ta.eval_gap_and_go, _base_gap_and_go_bull, 2.5, 2.0)

    def test_gap_fade_learned_gate(self):
        """GAP_FADE: learned gate 2.0 (higher than default 1.5)."""
        self._test_learned_gate(ta.eval_gap_fade, _base_gap_fade_bear, 2.0, 1.5)

    def test_vwap_touch_scalp_learned_gate(self):
        """VWAP_SCALP: learned gate 1.6 (higher than default 1.3)."""
        self._test_learned_gate(ta.eval_vwap_touch_scalp, _base_vwap_touch_scalp_bull, 1.6, 1.3)

    def test_vwap_hod_scalp_learned_gate(self):
        """VWAP_SCALP: learned gate 1.5 (higher than default 1.2)."""
        self._test_learned_gate(ta.eval_vwap_hod_scalp, _base_vwap_hod_scalp, 1.5, 1.2)

    def test_level_rejection_scalp_learned_gate(self):
        """LEVEL_SCALP: learned gate 1.5 (higher than default 1.2)."""
        self._test_learned_gate(ta.eval_level_rejection_scalp, _base_level_rejection_scalp_bull, 1.5, 1.2)

    def test_micro_pullback_scalp_learned_gate(self):
        """LEVEL_SCALP: learned gate 1.6 (higher than default 1.3)."""
        self._test_learned_gate(ta.eval_micro_pullback_scalp, _base_micro_pullback_scalp_bull, 1.6, 1.3)

    def test_spy_beta_catchup_learned_gate(self):
        """RS_REGIME: learned gate 1.4 (higher than default 1.1)."""
        self._test_learned_gate(ta.eval_spy_beta_catchup, _base_spy_beta_catchup_bull, 1.4, 1.1)

    def test_residual_momentum_learned_gate(self):
        """RS_REGIME: learned gate 1.5 (higher than default 1.2)."""
        self._test_learned_gate(ta.eval_residual_momentum, _base_residual_momentum, 1.5, 1.2)

    def test_sector_leader_learned_gate(self):
        """RS_REGIME: learned gate 1.0 (lower than default 1.2)."""
        self._test_learned_gate(ta.eval_sector_leader, _base_sector_leader, 1.0, 1.2)

    def test_sector_laggard_catchup_learned_gate(self):
        """RS_REGIME: learned gate 1.4 (higher than default 1.1)."""
        self._test_learned_gate(ta.eval_sector_laggard_catchup, _base_sector_laggard_catchup, 1.4, 1.1)

    def test_sector_counter_fade_learned_gate(self):
        """RS_REGIME: learned gate 1.4 (higher than default 1.1)."""
        self._test_learned_gate(ta.eval_sector_counter_fade, _base_sector_counter_fade, 1.4, 1.1)

    def test_regime_aligned_long_learned_gate(self):
        """RS_REGIME: learned gate 1.5 (higher than default 1.2)."""
        self._test_learned_gate(ta.eval_regime_aligned_long, _base_regime_aligned_long, 1.5, 1.2)

    def test_regime_aligned_short_learned_gate(self):
        """RS_REGIME: learned gate 1.5 (higher than default 1.2)."""
        self._test_learned_gate(ta.eval_regime_aligned_short, _base_regime_aligned_short, 1.5, 1.2)

    def test_sector_breakout_follow_learned_gate(self):
        """RS_REGIME: learned gate 1.4 (higher than default 1.1)."""
        self._test_learned_gate(ta.eval_sector_breakout_follow, _base_sector_breakout_follow_bull, 1.4, 1.1)

    def test_cross_sectional_rs_learned_gate(self):
        """RS_REGIME: learned gate 1.5 (higher than default 1.2)."""
        self._test_learned_gate(ta.eval_cross_sectional_rs, _base_cross_sectional_rs_bull, 1.5, 1.2)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: ORB15 target_mult is also parameterized
# ─────────────────────────────────────────────────────────────────────────────

class TestORB15TargetMultParam:
    """ORB15 should also parameterize target_mult via _param("ORB", "target_mult", 1.5)."""

    def test_target_mult_default_no_engine(self):
        """Without engine, target = entry + or_range * 1.5 (default)."""
        sig = _base_orb15_bull(rvol=2.0)
        with patch.object(ta, "_ALE_AVAILABLE", False):
            result = ta.eval_orb15(sig)
        assert result is not None
        # or_range = 101.0 - 100.0 = 1.0; entry = 101.5; target = 101.5 + 1.0 * 1.5 = 103.0
        assert abs(result.target - 103.0) < 0.01, f"Expected target ~103.0, got {result.target}"

    def test_target_mult_learned(self):
        """With engine returning target_mult=2.0, target should be entry + range*2.0."""
        sig = _base_orb15_bull(rvol=2.0)
        mock_params = {"rvol_gate": 1.5, "target_mult": 2.0, "stop_mult": 1.0}
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", return_value=mock_params):
            result = ta.eval_orb15(sig)
        assert result is not None
        # or_range = 1.0; entry = 101.5; target = 101.5 + 1.0 * 2.0 = 103.5
        assert abs(result.target - 103.5) < 0.01, f"Expected target ~103.5, got {result.target}"

    def test_target_mult_learned_bear(self):
        """Bear breakout with learned target_mult=2.0."""
        sig = SimpleNamespace(
            orb15_high=101.0,
            orb15_low=100.0,
            orb15_breakout="BEAR",
            rel_volume=2.0,
            short_tf_alignment="BEAR",
            price=99.5,
        )
        mock_params = {"rvol_gate": 1.5, "target_mult": 2.0, "stop_mult": 1.0}
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", return_value=mock_params):
            result = ta.eval_orb15(sig)
        assert result is not None
        # or_range = 1.0; entry = 99.5; target = 99.5 - 1.0 * 2.0 = 97.5
        assert abs(result.target - 97.5) < 0.01, f"Expected target ~97.5, got {result.target}"

    def test_target_mult_get_algo_params_called_for_ORB(self):
        """Verify _get_algo_params is called with 'ORB' for target_mult."""
        sig = _base_orb15_bull(rvol=2.0)
        mock_params = {"rvol_gate": 1.5, "target_mult": 1.5, "stop_mult": 1.0}
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", return_value=mock_params) as mock_gp:
            ta.eval_orb15(sig)
        # Both rvol_gate and target_mult are fetched via "ORB" family
        families_called = [c.args[0] for c in mock_gp.call_args_list]
        assert "ORB" in families_called

    def test_orb5_does_NOT_parameterize_new_target_mult(self):
        """ORB5 also uses target_mult but was already done in original code — verify it still works."""
        sig = SimpleNamespace(
            orb5_high=101.0,
            orb5_low=100.0,
            orb5_breakout="BULL",
            rel_volume=2.0,
            short_tf_alignment="BULL",
            price=101.5,
        )
        # ORB5 already had both _rvol_gate and _target_mult in original code
        mock_params = {"rvol_gate": 1.5, "target_mult": 2.0, "stop_mult": 1.0}
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", return_value=mock_params):
            result = ta.eval_orb5(sig)
        assert result is not None
        assert abs(result.target - 103.5) < 0.01  # entry + 1.0 * 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Dedicated deep tests for specific algo functions
# ─────────────────────────────────────────────────────────────────────────────

class TestGapAndGoDetailed:
    """Deep tests for eval_gap_and_go parameterization."""

    def test_fires_at_exactly_default_gate(self):
        """Signal fires when rvol == default gate (2.0)."""
        sig = _base_gap_and_go_bull(rvol=2.0)
        with patch.object(ta, "_ALE_AVAILABLE", False):
            result = ta.eval_gap_and_go(sig)
        assert result is not None
        assert result.algo == "GAP_AND_GO_BULL"

    def test_does_not_fire_just_below_default_gate(self):
        """Signal does not fire when rvol = 1.99 (just below 2.0)."""
        sig = _base_gap_and_go_bull(rvol=1.99)
        with patch.object(ta, "_ALE_AVAILABLE", False):
            result = ta.eval_gap_and_go(sig)
        assert result is None

    def test_learned_lower_gate_fires_at_1_5(self):
        """With learned gate of 1.5, signal fires at rvol=1.6 (default would block)."""
        sig = _base_gap_and_go_bull(rvol=1.6)
        mock_params = {"rvol_gate": 1.5, "target_mult": 1.5, "stop_mult": 1.0}
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", return_value=mock_params):
            result = ta.eval_gap_and_go(sig)
        assert result is not None, "With learned gate=1.5, rvol=1.6 should fire"

    def test_learned_higher_gate_blocks_at_2_0(self):
        """With learned gate of 2.5, signal does not fire at rvol=2.2."""
        sig = _base_gap_and_go_bull(rvol=2.2)
        mock_params = {"rvol_gate": 2.5, "target_mult": 1.5, "stop_mult": 1.0}
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", return_value=mock_params):
            result = ta.eval_gap_and_go(sig)
        assert result is None, "With learned gate=2.5, rvol=2.2 should NOT fire"

    def test_bear_side_uses_same_gate(self):
        """Bear gap-and-go also uses the learned rvol_gate."""
        sig = SimpleNamespace(
            gap_pct=-3.0,
            gap_type="GAP_DOWN",
            color_vs_prev_close="RED",
            rel_volume=2.3,
            short_tf_alignment="BEAR",
            price=97.0,
            session_low=96.0,
            session_high=100.0,
            today_open=100.0,
        )
        mock_params = {"rvol_gate": 2.5, "target_mult": 1.5, "stop_mult": 1.0}
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", return_value=mock_params):
            result = ta.eval_gap_and_go(sig)
        assert result is None, "rvol=2.3 < learned gate=2.5 should NOT fire bear"


class TestVWAPTouchScalpDetailed:
    """Deep tests for eval_vwap_touch_scalp parameterization."""

    def test_fires_at_exactly_default_gate_1_3(self):
        """Signal fires at rvol == 1.3 (exact default)."""
        sig = _base_vwap_touch_scalp_bull(rvol=1.3)
        with patch.object(ta, "_ALE_AVAILABLE", False):
            result = ta.eval_vwap_touch_scalp(sig)
        assert result is not None
        assert result.algo == "VWAP_TOUCH_SCALP_BULL"

    def test_does_not_fire_below_default_gate(self):
        """Signal blocked at rvol=1.29."""
        sig = _base_vwap_touch_scalp_bull(rvol=1.29)
        with patch.object(ta, "_ALE_AVAILABLE", False):
            result = ta.eval_vwap_touch_scalp(sig)
        assert result is None

    def test_learned_gate_1_5_blocks_at_1_4(self):
        """With learned gate=1.5, rvol=1.4 should not fire."""
        sig = _base_vwap_touch_scalp_bull(rvol=1.4)
        mock_params = {"rvol_gate": 1.5, "target_mult": 1.5, "stop_mult": 1.0}
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", return_value=mock_params):
            result = ta.eval_vwap_touch_scalp(sig)
        assert result is None

    def test_rejection_side_also_uses_learned_gate(self):
        """VWAP REJECTION (bear) also uses _rvol_gate."""
        sig = SimpleNamespace(
            vwap_event="REJECTION",
            vwap_price=100.0,
            vwap_upper_1=101.0,
            vwap_lower_1=99.0,
            rel_volume=1.4,
            short_tf_alignment="BEAR",
            price=99.9,
        )
        mock_params = {"rvol_gate": 1.5, "target_mult": 1.5, "stop_mult": 1.0}
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", return_value=mock_params):
            result = ta.eval_vwap_touch_scalp(sig)
        assert result is None, "rvol=1.4 < learned gate=1.5 should NOT fire REJECTION"

    def test_rejection_fires_at_learned_gate(self):
        """REJECTION fires when rvol >= learned gate."""
        sig = SimpleNamespace(
            vwap_event="REJECTION",
            vwap_price=100.0,
            vwap_upper_1=101.0,
            vwap_lower_1=99.0,
            rel_volume=1.6,
            short_tf_alignment="BEAR",
            price=99.9,
        )
        mock_params = {"rvol_gate": 1.5, "target_mult": 1.5, "stop_mult": 1.0}
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", return_value=mock_params):
            result = ta.eval_vwap_touch_scalp(sig)
        assert result is not None, "rvol=1.6 >= learned gate=1.5 should fire REJECTION"
        assert result.algo == "VWAP_TOUCH_SCALP_BEAR"


class TestSectorBreakoutFollowDetailed:
    """Deep tests for eval_sector_breakout_follow — two rvol checks use _rvol_gate."""

    def test_fires_bull_at_exactly_default_gate_1_1(self):
        """Bull side fires at rvol == 1.1 (exact default)."""
        sig = _base_sector_breakout_follow_bull(rvol=1.1)
        with patch.object(ta, "_ALE_AVAILABLE", False):
            result = ta.eval_sector_breakout_follow(sig)
        assert result is not None
        assert result.algo == "SECTOR_BREAKOUT_BULL"

    def test_does_not_fire_below_1_1_default(self):
        """Bull side blocked at rvol=1.09."""
        sig = _base_sector_breakout_follow_bull(rvol=1.09)
        with patch.object(ta, "_ALE_AVAILABLE", False):
            result = ta.eval_sector_breakout_follow(sig)
        assert result is None

    def test_bear_side_also_uses_learned_gate(self):
        """Bear side (sector_change < -0.8) also uses _rvol_gate."""
        sig = SimpleNamespace(
            sector_change=-1.2,
            stock_vs_sector="IN_LINE",
            rs_label="IN_LINE",
            vwap_event="BELOW",
            short_tf_alignment="BEAR",
            rel_volume=1.3,
            price=99.0,
            vwap_price=100.0,
            vwap_upper_1=101.0,
            vwap_lower_1=98.0,
            sector_etf="XLK",
        )
        mock_params = {"rvol_gate": 1.5, "target_mult": 1.5, "stop_mult": 1.0}
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", return_value=mock_params):
            result = ta.eval_sector_breakout_follow(sig)
        assert result is None, "rvol=1.3 < learned gate=1.5 should block bear side"

    def test_bear_fires_at_learned_gate(self):
        """Bear side fires when rvol >= learned gate."""
        sig = SimpleNamespace(
            sector_change=-1.2,
            stock_vs_sector="IN_LINE",
            rs_label="IN_LINE",
            vwap_event="BELOW",
            short_tf_alignment="BEAR",
            rel_volume=1.6,
            price=99.0,
            vwap_price=100.0,
            vwap_upper_1=101.0,
            vwap_lower_1=98.0,
            sector_etf="XLK",
        )
        mock_params = {"rvol_gate": 1.5, "target_mult": 1.5, "stop_mult": 1.0}
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", return_value=mock_params):
            result = ta.eval_sector_breakout_follow(sig)
        assert result is not None, "rvol=1.6 >= learned gate=1.5 should fire bear side"
        assert result.algo == "SECTOR_BREAKOUT_BEAR"

    def test_learned_lower_gate_allows_at_1_05(self):
        """With learned gate=1.0, rvol=1.05 should fire (default 1.1 would block)."""
        sig = _base_sector_breakout_follow_bull(rvol=1.05)
        mock_params = {"rvol_gate": 1.0, "target_mult": 1.5, "stop_mult": 1.0}
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", return_value=mock_params):
            result = ta.eval_sector_breakout_follow(sig)
        assert result is not None, "With learned gate=1.0, rvol=1.05 should fire"


class TestResidualMomentumDetailed:
    """Deep tests for eval_residual_momentum parameterization."""

    def test_fires_at_default_gate_1_2(self):
        """Fires at rvol == 1.2 (exact default)."""
        sig = _base_residual_momentum(rvol=1.2)
        with patch.object(ta, "_ALE_AVAILABLE", False):
            result = ta.eval_residual_momentum(sig)
        assert result is not None
        assert result.algo == "RESIDUAL_MOMENTUM_BULL"

    def test_blocked_below_default_gate(self):
        """Blocked at rvol=1.19."""
        sig = _base_residual_momentum(rvol=1.19)
        with patch.object(ta, "_ALE_AVAILABLE", False):
            result = ta.eval_residual_momentum(sig)
        assert result is None

    def test_learned_gate_1_5_blocks_at_1_4(self):
        """With learned gate=1.5, rvol=1.4 does not fire."""
        sig = _base_residual_momentum(rvol=1.4)
        mock_params = {"rvol_gate": 1.5, "target_mult": 1.5, "stop_mult": 1.0}
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", return_value=mock_params):
            result = ta.eval_residual_momentum(sig)
        assert result is None

    def test_learned_gate_1_0_fires_at_1_1(self):
        """With learned gate=1.0, rvol=1.1 fires (default 1.2 would block)."""
        sig = _base_residual_momentum(rvol=1.1)
        mock_params = {"rvol_gate": 1.0, "target_mult": 1.5, "stop_mult": 1.0}
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", return_value=mock_params):
            result = ta.eval_residual_momentum(sig)
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Edge cases and engine exception handling
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineExceptionFallback:
    """When engine raises an exception, _param() falls back to default."""

    def test_orb15_falls_back_on_exception(self):
        """If get_algo_params raises, rvol_gate falls back to 1.5."""
        sig = _base_orb15_bull(rvol=1.6)
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", side_effect=RuntimeError("engine down")):
            result = ta.eval_orb15(sig)
        # Should still fire because default 1.5 <= 1.6
        assert result is not None

    def test_gap_and_go_falls_back_on_exception(self):
        """If get_algo_params raises, rvol_gate falls back to 2.0."""
        sig_above = _base_gap_and_go_bull(rvol=2.1)
        sig_below = _base_gap_and_go_bull(rvol=1.9)
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", side_effect=RuntimeError("engine down")):
            assert ta.eval_gap_and_go(sig_above) is not None  # rvol 2.1 >= default 2.0
            assert ta.eval_gap_and_go(sig_below) is None      # rvol 1.9 < default 2.0

    def test_vwap_touch_scalp_falls_back_on_exception(self):
        """If get_algo_params raises, rvol_gate falls back to 1.3."""
        sig = _base_vwap_touch_scalp_bull(rvol=1.3)
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params", side_effect=ValueError("bad data")):
            result = ta.eval_vwap_touch_scalp(sig)
        assert result is not None  # default 1.3 <= 1.3 → fires


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Bull/bear flag algos are UNCHANGED (no rvol gate)
# ─────────────────────────────────────────────────────────────────────────────

class TestFlagAlgosUnchanged:
    """eval_bull_flag and eval_bear_flag have no rvol gate — they must still work."""

    def test_bull_flag_fires_no_rvol_gate(self):
        """Bull flag fires regardless of rvol (no _rvol_gate added)."""
        sig = SimpleNamespace(
            bull_flag=True,
            flag_high=102.0,
            flag_low=100.0,
            pole_pct=1.5,
            rel_volume=0.5,  # very low rvol — should still fire
            short_tf_alignment="BULL",
            price=102.1,
        )
        result = ta.eval_bull_flag(sig)
        assert result is not None
        assert result.algo == "BULL_FLAG"

    def test_bear_flag_fires_no_rvol_gate(self):
        """Bear flag fires regardless of rvol."""
        sig = SimpleNamespace(
            bear_flag=True,
            flag_high=102.0,
            flag_low=100.0,
            pole_pct=-1.5,
            rel_volume=0.5,  # very low rvol — should still fire
            short_tf_alignment="BEAR",
            price=99.9,
        )
        result = ta.eval_bear_flag(sig)
        assert result is not None
        assert result.algo == "BEAR_FLAG"

    def test_get_algo_params_not_called_for_bull_flag(self):
        """_get_algo_params is never called for eval_bull_flag."""
        sig = SimpleNamespace(
            bull_flag=True,
            flag_high=102.0,
            flag_low=100.0,
            pole_pct=1.5,
            rel_volume=1.5,
            short_tf_alignment="BULL",
            price=102.1,
        )
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params") as mock_gp:
            ta.eval_bull_flag(sig)
            mock_gp.assert_not_called()

    def test_get_algo_params_not_called_for_bear_flag(self):
        """_get_algo_params is never called for eval_bear_flag."""
        sig = SimpleNamespace(
            bear_flag=True,
            flag_high=102.0,
            flag_low=100.0,
            pole_pct=-1.5,
            rel_volume=1.5,
            short_tf_alignment="BEAR",
            price=99.9,
        )
        with patch.object(ta, "_ALE_AVAILABLE", True), \
             patch.object(ta, "_get_algo_params") as mock_gp:
            ta.eval_bear_flag(sig)
            mock_gp.assert_not_called()
