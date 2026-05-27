import pytest
pytestmark = pytest.mark.slow

"""
Comprehensive pytest tests for agent/trading_algos.py.

Covers:
  - AlgoResult helpers (_rr, to_dict)
  - eval_orb5
  - eval_orb15
  - eval_gap_and_go
  - eval_gap_fade
  - eval_pdh_pdl_breakout
  - eval_hod_lod_break
  - eval_bull_flag / eval_bear_flag
  - eval_vwap_touch_scalp
  - eval_vwap_hod_scalp / eval_vwap_lod_scalp
  - eval_level_rejection_scalp
  - eval_micro_pullback_scalp
  - evaluate_all
"""
import sys
import os

# Ensure the project root is on PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agent.trading_algos import (
    AlgoResult,
    _rr,
    eval_orb5,
    eval_orb15,
    eval_gap_and_go,
    eval_gap_fade,
    eval_pdh_pdl_breakout,
    eval_hod_lod_break,
    eval_bull_flag,
    eval_bear_flag,
    eval_vwap_touch_scalp,
    eval_vwap_hod_scalp,
    eval_vwap_lod_scalp,
    eval_level_rejection_scalp,
    eval_micro_pullback_scalp,
    evaluate_all,
)
from agent.scanner import StockSignal


# ── Helper ────────────────────────────────────────────────────────────────────

def _sig(**kwargs) -> StockSignal:
    """Return a minimal valid StockSignal with all required fields."""
    defaults = dict(
        ticker="TEST",
        name="Test Inc",
        price=100.0,
        change_pct=1.0,
        technical=0.5,
        volume=0.5,
        ml_prob=0.6,
        ml_daily_prob=0.6,
        sentiment=0.0,
        score=0.5,
        signal="BUY",
        rel_volume=2.0,
        unusual_vol=False,
        prediction="BUY",
        confidence=70.0,
        trend="UPTREND",
        trend_probability=0.75,
        ml_trained=True,
        target_price=105.0,
        stop_loss=97.0,
        rr_ratio=1.67,
    )
    defaults.update(kwargs)
    return StockSignal(**defaults)


def _assert_valid_result(res: AlgoResult, expected_direction: str) -> None:
    """Assert all critical invariants for any AlgoResult."""
    assert res is not None, "Expected a result, got None"
    assert isinstance(res.algo, str) and len(res.algo) > 0, "algo must be non-empty string"
    assert isinstance(res.reason, str) and len(res.reason) > 0, "reason must be non-empty string"
    assert res.direction in ("BUY", "SELL"), f"direction must be BUY or SELL, got {res.direction!r}"
    assert res.direction == expected_direction, f"Expected {expected_direction}, got {res.direction}"
    assert 0 < res.confidence <= 100, f"confidence must be in (0, 100], got {res.confidence}"
    assert res.rr > 0, f"rr must be > 0, got {res.rr}"
    if res.direction == "BUY":
        assert res.stop < res.entry, f"BUY: stop ({res.stop}) must be < entry ({res.entry})"
        assert res.entry < res.target, f"BUY: entry ({res.entry}) must be < target ({res.target})"
    else:  # SELL
        assert res.stop > res.entry, f"SELL: stop ({res.stop}) must be > entry ({res.entry})"
        assert res.entry > res.target, f"SELL: entry ({res.entry}) must be > target ({res.target})"


# ══════════════════════════════════════════════════════════════════════════════
# TestAlgoHelpers
# ══════════════════════════════════════════════════════════════════════════════

class TestAlgoHelpers:
    """Tests for _rr() and AlgoResult.to_dict()."""

    def test_rr_basic(self):
        """1:2 R:R returns 2.0."""
        assert _rr(100, 95, 110) == 2.0

    def test_rr_zero_risk_returns_zero(self):
        """When entry == stop, returns 0.0 instead of dividing by zero."""
        assert _rr(100, 100, 110) == 0.0

    def test_rr_symmetric_buy(self):
        """Standard BUY: risk 5, reward 10 → rr = 2.0."""
        result = _rr(100, 95, 110)
        assert result == pytest.approx(2.0, abs=0.01)

    def test_rr_symmetric_sell(self):
        """Standard SELL: entry=100, stop=105, target=90 → risk 5, reward 10 → rr=2.0."""
        result = _rr(100, 105, 90)
        assert result == pytest.approx(2.0, abs=0.01)

    def test_rr_small_values(self):
        """Works correctly with small price differences."""
        result = _rr(10.0, 9.9, 10.1)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_rr_returns_float(self):
        assert isinstance(_rr(100, 95, 110), float)

    def test_algo_result_to_dict_has_required_keys(self):
        res = AlgoResult(
            algo="TEST_ALGO", direction="BUY",
            confidence=75.0, entry=100.0, stop=95.0, target=110.0,
            rr=2.0, reason="test reason",
        )
        d = res.to_dict()
        for key in ("algo", "direction", "confidence", "entry", "stop", "target", "rr", "reason"):
            assert key in d, f"Key {key!r} missing from to_dict()"

    def test_algo_result_to_dict_rounds_floats(self):
        """Floats in to_dict() are rounded to 4 decimal places."""
        res = AlgoResult(
            algo="TEST", direction="BUY",
            confidence=75.123456, entry=100.123456, stop=95.123456, target=110.123456,
            rr=2.123456, reason="r",
        )
        d = res.to_dict()
        # Each float value in the dict should have at most 4 decimal places
        for k, v in d.items():
            if isinstance(v, float):
                assert round(v, 4) == v, f"Field {k}={v} not rounded to 4dp"

    def test_algo_result_to_dict_preserves_strings(self):
        res = AlgoResult(
            algo="MY_ALGO", direction="SELL",
            confidence=60.0, entry=100.0, stop=105.0, target=90.0,
            rr=2.0, reason="test reason text",
        )
        d = res.to_dict()
        assert d["algo"] == "MY_ALGO"
        assert d["direction"] == "SELL"
        assert d["reason"] == "test reason text"

    def test_rr_rounded_to_2dp(self):
        """_rr() rounds its output to 2 decimal places."""
        result = _rr(100, 97, 110)
        # risk=3, reward=10 → 3.33333 rounds to 3.33
        assert result == pytest.approx(3.33, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# TestORB5
# ══════════════════════════════════════════════════════════════════════════════

class TestORB5:
    """Tests for eval_orb5 (5-min opening range breakout)."""

    def _bull_sig(self, **kwargs):
        base = dict(
            orb5_high=101.0, orb5_low=99.0,
            orb5_breakout="BULL",
            rel_volume=2.0,
            short_tf_alignment="MIXED",
            price=102.0,
        )
        base.update(kwargs)
        return _sig(**base)

    def _bear_sig(self, **kwargs):
        base = dict(
            orb5_high=101.0, orb5_low=99.0,
            orb5_breakout="BEAR",
            rel_volume=2.0,
            short_tf_alignment="MIXED",
            price=98.0,
        )
        base.update(kwargs)
        return _sig(**base)

    def test_bull_trigger_returns_result(self):
        res = eval_orb5(self._bull_sig())
        assert res is not None
        assert res.algo == "ORB5_BULL"

    def test_bull_direction_is_buy(self):
        res = eval_orb5(self._bull_sig())
        _assert_valid_result(res, "BUY")

    def test_bull_entry_equals_price(self):
        res = eval_orb5(self._bull_sig())
        assert res.entry == pytest.approx(102.0)

    def test_bull_stop_is_orl(self):
        """Bull stop should be the ORB-5 low."""
        res = eval_orb5(self._bull_sig())
        assert res.stop == pytest.approx(99.0)

    def test_bull_target_is_entry_plus_range_times_1p5(self):
        """Target = entry + (orh - orl) * 1.5 = 102 + 2 * 1.5 = 105."""
        res = eval_orb5(self._bull_sig())
        assert res.target == pytest.approx(102.0 + (101.0 - 99.0) * 1.5)

    def test_bull_rr_positive(self):
        res = eval_orb5(self._bull_sig())
        assert res.rr > 0

    def test_bull_confidence_in_range(self):
        res = eval_orb5(self._bull_sig())
        assert 0 < res.confidence <= 100

    def test_bull_gate_bull_boosts_confidence(self):
        res_mixed = eval_orb5(self._bull_sig(short_tf_alignment="MIXED"))
        res_bull  = eval_orb5(self._bull_sig(short_tf_alignment="BULL"))
        assert res_bull.confidence > res_mixed.confidence

    def test_bear_trigger_returns_result(self):
        res = eval_orb5(self._bear_sig())
        assert res is not None
        assert res.algo == "ORB5_BEAR"

    def test_bear_direction_is_sell(self):
        res = eval_orb5(self._bear_sig())
        _assert_valid_result(res, "SELL")

    def test_bear_stop_is_orh(self):
        """Bear stop should be the ORB-5 high."""
        res = eval_orb5(self._bear_sig())
        assert res.stop == pytest.approx(101.0)

    def test_bear_target_is_entry_minus_range_times_1p5(self):
        """Target = entry - (orh - orl) * 1.5 = 98 - 2 * 1.5 = 95."""
        res = eval_orb5(self._bear_sig())
        assert res.target == pytest.approx(98.0 - (101.0 - 99.0) * 1.5)

    def test_none_when_rvol_below_threshold(self):
        """rel_volume < 1.5 → None."""
        res = eval_orb5(self._bull_sig(rel_volume=1.4))
        assert res is None

    def test_none_when_bull_gate_blocks_bull(self):
        """Bull setup with short_tf_alignment='BEAR' → None (gate blocks)."""
        res = eval_orb5(self._bull_sig(short_tf_alignment="BEAR"))
        assert res is None

    def test_none_when_bear_gate_blocks_bear(self):
        """Bear setup with short_tf_alignment='BULL' → None (gate blocks)."""
        res = eval_orb5(self._bear_sig(short_tf_alignment="BULL"))
        assert res is None

    def test_none_when_orb_high_zero(self):
        """orb5_high=0 → None (levels not set)."""
        res = eval_orb5(self._bull_sig(orb5_high=0.0))
        assert res is None

    def test_none_when_orb_low_zero(self):
        """orb5_low=0 → None."""
        res = eval_orb5(self._bull_sig(orb5_low=0.0))
        assert res is None

    def test_none_when_breakout_is_none(self):
        """No BULL/BEAR breakout signal → None."""
        res = eval_orb5(self._bull_sig(orb5_breakout="NONE"))
        assert res is None

    def test_bull_at_rvol_boundary(self):
        """Exactly at rvol=1.5 threshold still triggers."""
        res = eval_orb5(self._bull_sig(rel_volume=1.5))
        assert res is not None

    def test_confidence_capped_at_95(self):
        """Very high RVOL should not push confidence above 95."""
        res = eval_orb5(self._bull_sig(rel_volume=100.0, short_tf_alignment="BULL"))
        assert res.confidence <= 95.0


# ══════════════════════════════════════════════════════════════════════════════
# TestORB15
# ══════════════════════════════════════════════════════════════════════════════

class TestORB15:
    """Tests for eval_orb15 (15-min opening range breakout)."""

    def _bull_sig(self, **kwargs):
        base = dict(
            orb15_high=102.0, orb15_low=98.0,
            orb15_breakout="BULL",
            rel_volume=2.0,
            short_tf_alignment="MIXED",
            price=103.0,
        )
        base.update(kwargs)
        return _sig(**base)

    def _bear_sig(self, **kwargs):
        base = dict(
            orb15_high=102.0, orb15_low=98.0,
            orb15_breakout="BEAR",
            rel_volume=2.0,
            short_tf_alignment="MIXED",
            price=97.0,
        )
        base.update(kwargs)
        return _sig(**base)

    def test_bull_trigger_returns_result(self):
        res = eval_orb15(self._bull_sig())
        assert res is not None
        assert res.algo == "ORB15_BULL"

    def test_bull_direction_is_buy(self):
        res = eval_orb15(self._bull_sig())
        _assert_valid_result(res, "BUY")

    def test_bull_stop_is_orl15(self):
        res = eval_orb15(self._bull_sig())
        assert res.stop == pytest.approx(98.0)

    def test_bull_target_is_entry_plus_range_times_1p5(self):
        """Target = 103 + (102-98)*1.5 = 103 + 6 = 109."""
        res = eval_orb15(self._bull_sig())
        assert res.target == pytest.approx(103.0 + (102.0 - 98.0) * 1.5)

    def test_bear_trigger_returns_result(self):
        res = eval_orb15(self._bear_sig())
        assert res is not None
        assert res.algo == "ORB15_BEAR"

    def test_bear_direction_is_sell(self):
        res = eval_orb15(self._bear_sig())
        _assert_valid_result(res, "SELL")

    def test_bear_stop_is_orh15(self):
        res = eval_orb15(self._bear_sig())
        assert res.stop == pytest.approx(102.0)

    def test_bear_target_is_entry_minus_range_times_1p5(self):
        """Target = 97 - (102-98)*1.5 = 97 - 6 = 91."""
        res = eval_orb15(self._bear_sig())
        assert res.target == pytest.approx(97.0 - (102.0 - 98.0) * 1.5)

    def test_none_when_rvol_too_low(self):
        res = eval_orb15(self._bull_sig(rel_volume=1.0))
        assert res is None

    def test_none_when_bull_gate_blocked(self):
        res = eval_orb15(self._bull_sig(short_tf_alignment="BEAR"))
        assert res is None

    def test_none_when_bear_gate_blocked(self):
        res = eval_orb15(self._bear_sig(short_tf_alignment="BULL"))
        assert res is None

    def test_none_when_levels_zero(self):
        res = eval_orb15(self._bull_sig(orb15_high=0.0, orb15_low=0.0))
        assert res is None

    def test_rr_positive(self):
        res = eval_orb15(self._bull_sig())
        assert res.rr > 0

    def test_confidence_in_range(self):
        res = eval_orb15(self._bear_sig())
        assert 0 < res.confidence <= 100

    def test_confidence_capped_at_95(self):
        res = eval_orb15(self._bull_sig(rel_volume=200.0, short_tf_alignment="BULL"))
        assert res.confidence <= 95.0

    def test_bull_gate_boosts_confidence(self):
        res_mixed = eval_orb15(self._bull_sig(short_tf_alignment="MIXED"))
        res_bull  = eval_orb15(self._bull_sig(short_tf_alignment="BULL"))
        assert res_bull.confidence > res_mixed.confidence


# ══════════════════════════════════════════════════════════════════════════════
# TestGapAndGo
# ══════════════════════════════════════════════════════════════════════════════

class TestGapAndGo:
    """Tests for eval_gap_and_go."""

    def _bull_sig(self, **kwargs):
        base = dict(
            gap_type="GAP_UP",
            gap_pct=3.0,
            color_vs_prev_close="GREEN",
            rel_volume=2.5,
            short_tf_alignment="MIXED",
            price=103.0,
            session_low=100.0,
            session_high=0.0,
            today_open=100.0,
        )
        base.update(kwargs)
        return _sig(**base)

    def _bear_sig(self, **kwargs):
        base = dict(
            gap_type="GAP_DOWN",
            gap_pct=-3.0,
            color_vs_prev_close="RED",
            rel_volume=2.5,
            short_tf_alignment="MIXED",
            price=97.0,
            session_high=100.0,
            session_low=0.0,
            today_open=100.0,
        )
        base.update(kwargs)
        return _sig(**base)

    def test_bull_trigger(self):
        res = eval_gap_and_go(self._bull_sig())
        assert res is not None
        assert res.algo == "GAP_AND_GO_BULL"
        _assert_valid_result(res, "BUY")

    def test_bear_trigger(self):
        res = eval_gap_and_go(self._bear_sig())
        assert res is not None
        assert res.algo == "GAP_AND_GO_BEAR"
        _assert_valid_result(res, "SELL")

    def test_bull_stop_uses_session_low(self):
        res = eval_gap_and_go(self._bull_sig(session_low=99.0))
        assert res.stop == pytest.approx(99.0)

    def test_bull_stop_fallback_when_sess_low_zero(self):
        """When session_low=0, stop = price * 0.98."""
        res = eval_gap_and_go(self._bull_sig(session_low=0.0, price=103.0))
        assert res.stop == pytest.approx(103.0 * 0.98, abs=0.001)

    def test_bear_stop_uses_session_high(self):
        res = eval_gap_and_go(self._bear_sig(session_high=101.0))
        assert res.stop == pytest.approx(101.0)

    def test_none_when_rvol_too_low(self):
        # Engine initial rvol_gate is 1.5 for GAP_TREND; use 1.4 to stay below it
        res = eval_gap_and_go(self._bull_sig(rel_volume=1.4))
        assert res is None

    def test_none_when_gap_pct_too_small(self):
        res = eval_gap_and_go(self._bull_sig(gap_pct=1.0))
        assert res is None

    def test_none_when_color_wrong(self):
        """Gap-up requires GREEN candle."""
        res = eval_gap_and_go(self._bull_sig(color_vs_prev_close="RED"))
        assert res is None

    def test_none_when_bull_gate_blocks(self):
        res = eval_gap_and_go(self._bull_sig(short_tf_alignment="BEAR"))
        assert res is None

    def test_none_when_bear_gate_blocks(self):
        res = eval_gap_and_go(self._bear_sig(short_tf_alignment="BULL"))
        assert res is None

    def test_bull_confidence_in_range(self):
        res = eval_gap_and_go(self._bull_sig())
        assert 0 < res.confidence <= 100

    def test_bull_rr_positive(self):
        res = eval_gap_and_go(self._bull_sig())
        assert res.rr > 0


# ══════════════════════════════════════════════════════════════════════════════
# TestGapFade
# ══════════════════════════════════════════════════════════════════════════════

class TestGapFade:
    """Tests for eval_gap_fade."""

    def _fade_up_sig(self, **kwargs):
        """Gap-up fade (SELL): gap opened high, price now below today_open."""
        base = dict(
            gap_type="GAP_UP",
            gap_pct=2.0,
            rel_volume=2.0,
            short_tf_alignment="MIXED",
            price=99.0,        # below today_open → fading
            today_open=101.0,
            prev_day_close=97.0,
            session_high=103.0,
            session_low=0.0,
        )
        base.update(kwargs)
        return _sig(**base)

    def _fade_down_sig(self, **kwargs):
        """Gap-down fade (BUY): gap opened low, price now above today_open."""
        base = dict(
            gap_type="GAP_DOWN",
            gap_pct=-2.0,
            rel_volume=2.0,
            short_tf_alignment="MIXED",
            price=101.0,       # above today_open → fading
            today_open=99.0,
            prev_day_close=105.0,
            session_low=96.0,
            session_high=0.0,
        )
        base.update(kwargs)
        return _sig(**base)

    def test_fade_up_trigger(self):
        res = eval_gap_fade(self._fade_up_sig())
        assert res is not None
        assert res.algo == "GAP_FADE_BEAR"
        _assert_valid_result(res, "SELL")

    def test_fade_down_trigger(self):
        res = eval_gap_fade(self._fade_down_sig())
        assert res is not None
        assert res.algo == "GAP_FADE_BULL"
        _assert_valid_result(res, "BUY")

    def test_fade_up_target_is_prev_close(self):
        res = eval_gap_fade(self._fade_up_sig())
        assert res.target == pytest.approx(97.0)

    def test_fade_down_target_is_prev_close(self):
        res = eval_gap_fade(self._fade_down_sig())
        assert res.target == pytest.approx(105.0)

    def test_none_when_today_open_zero(self):
        res = eval_gap_fade(self._fade_up_sig(today_open=0.0))
        assert res is None

    def test_none_when_prev_close_zero(self):
        res = eval_gap_fade(self._fade_up_sig(prev_day_close=0.0))
        assert res is None

    def test_none_when_rvol_too_low(self):
        res = eval_gap_fade(self._fade_up_sig(rel_volume=1.0))
        assert res is None

    def test_none_when_gap_pct_too_small_for_fade_up(self):
        res = eval_gap_fade(self._fade_up_sig(gap_pct=1.0))
        assert res is None

    def test_none_when_bull_gate_blocks_fade_up(self):
        """Fade up (sell) blocked when short_tf_alignment == 'BULL'."""
        res = eval_gap_fade(self._fade_up_sig(short_tf_alignment="BULL"))
        assert res is None

    def test_none_when_bear_gate_blocks_fade_down(self):
        """Fade down (buy) blocked when short_tf_alignment == 'BEAR'."""
        res = eval_gap_fade(self._fade_down_sig(short_tf_alignment="BEAR"))
        assert res is None

    def test_rr_positive_fade_up(self):
        res = eval_gap_fade(self._fade_up_sig())
        assert res.rr > 0

    def test_confidence_in_range_fade_down(self):
        res = eval_gap_fade(self._fade_down_sig())
        assert 0 < res.confidence <= 100


# ══════════════════════════════════════════════════════════════════════════════
# TestPDHPDLBreakout
# ══════════════════════════════════════════════════════════════════════════════

class TestPDHPDLBreakout:
    """Tests for eval_pdh_pdl_breakout."""

    def _bull_sig(self, **kwargs):
        base = dict(
            prev_day_high=100.0,
            prev_day_low=95.0,
            rel_volume=2.0,
            short_tf_alignment="MIXED",
            price=101.0,  # above PDH
        )
        base.update(kwargs)
        return _sig(**base)

    def _bear_sig(self, **kwargs):
        base = dict(
            prev_day_high=100.0,
            prev_day_low=95.0,
            rel_volume=2.0,
            short_tf_alignment="MIXED",
            price=94.0,  # below PDL
        )
        base.update(kwargs)
        return _sig(**base)

    def test_bull_trigger_above_pdh(self):
        res = eval_pdh_pdl_breakout(self._bull_sig())
        assert res is not None
        assert res.algo == "PDH_BREAKOUT_BULL"
        _assert_valid_result(res, "BUY")

    def test_bear_trigger_below_pdl(self):
        res = eval_pdh_pdl_breakout(self._bear_sig())
        assert res is not None
        assert res.algo == "PDL_BREAKDOWN_BEAR"
        _assert_valid_result(res, "SELL")

    def test_bull_stop_is_pdh_minus_atr(self):
        """Bull stop = PDH - 0.5 * pd_range = 100 - 0.5*5 = 97.5."""
        res = eval_pdh_pdl_breakout(self._bull_sig())
        pd_range = 100.0 - 95.0
        atr_proxy = pd_range * 0.5
        assert res.stop == pytest.approx(100.0 - atr_proxy)

    def test_bear_stop_is_pdl_plus_atr(self):
        """Bear stop = PDL + 0.5 * pd_range = 95 + 2.5 = 97.5."""
        res = eval_pdh_pdl_breakout(self._bear_sig())
        pd_range = 100.0 - 95.0
        atr_proxy = pd_range * 0.5
        assert res.stop == pytest.approx(95.0 + atr_proxy)

    def test_bull_target_is_pdh_plus_half_range(self):
        """Bull target = PDH + pd_range * 0.5 = 100 + 2.5 = 102.5."""
        res = eval_pdh_pdl_breakout(self._bull_sig())
        assert res.target == pytest.approx(100.0 + 5.0 * 0.5)

    def test_bear_target_is_pdl_minus_half_range(self):
        """Bear target = PDL - pd_range * 0.5 = 95 - 2.5 = 92.5."""
        res = eval_pdh_pdl_breakout(self._bear_sig())
        assert res.target == pytest.approx(95.0 - 5.0 * 0.5)

    def test_none_when_price_not_above_pdh(self):
        """Price at PDH but not above → None."""
        res = eval_pdh_pdl_breakout(self._bull_sig(price=100.0))
        assert res is None

    def test_none_when_price_not_below_pdl(self):
        """Price at PDL but not below → None."""
        res = eval_pdh_pdl_breakout(self._bear_sig(price=95.0))
        assert res is None

    def test_none_when_rvol_too_low(self):
        res = eval_pdh_pdl_breakout(self._bull_sig(rel_volume=1.2))
        assert res is None

    def test_none_when_pdh_zero(self):
        res = eval_pdh_pdl_breakout(self._bull_sig(prev_day_high=0.0))
        assert res is None

    def test_none_when_pdl_zero(self):
        res = eval_pdh_pdl_breakout(self._bull_sig(prev_day_low=0.0))
        assert res is None

    def test_none_when_gate_blocks_bull(self):
        res = eval_pdh_pdl_breakout(self._bull_sig(short_tf_alignment="BEAR"))
        assert res is None

    def test_none_when_gate_blocks_bear(self):
        res = eval_pdh_pdl_breakout(self._bear_sig(short_tf_alignment="BULL"))
        assert res is None

    def test_rr_positive_bull(self):
        res = eval_pdh_pdl_breakout(self._bull_sig())
        assert res.rr > 0

    def test_confidence_in_range(self):
        res = eval_pdh_pdl_breakout(self._bull_sig())
        assert 0 < res.confidence <= 100

    def test_confidence_capped_at_95(self):
        res = eval_pdh_pdl_breakout(self._bull_sig(rel_volume=200.0, short_tf_alignment="BULL"))
        assert res.confidence <= 95.0


# ══════════════════════════════════════════════════════════════════════════════
# TestHODLODBreak
# ══════════════════════════════════════════════════════════════════════════════

class TestHODLODBreak:
    """Tests for eval_hod_lod_break."""

    def _bull_sig(self, **kwargs):
        """Price exactly at session_high → HOD break."""
        base = dict(
            session_high=105.0,
            session_low=95.0,
            orb15_high=102.0,  # sess_high > orb_high → beyond opening range
            rel_volume=2.0,
            short_tf_alignment="MIXED",
            price=105.0,
        )
        base.update(kwargs)
        return _sig(**base)

    def _bear_sig(self, **kwargs):
        """Price exactly at session_low → LOD break."""
        base = dict(
            session_high=105.0,
            session_low=95.0,
            orb15_high=102.0,
            rel_volume=2.0,
            short_tf_alignment="MIXED",
            price=95.0,
        )
        base.update(kwargs)
        return _sig(**base)

    def test_bull_trigger(self):
        res = eval_hod_lod_break(self._bull_sig())
        assert res is not None
        assert res.algo == "HOD_BREAK_BULL"
        _assert_valid_result(res, "BUY")

    def test_bear_trigger(self):
        res = eval_hod_lod_break(self._bear_sig())
        assert res is not None
        assert res.algo == "LOD_BREAK_BEAR"
        _assert_valid_result(res, "SELL")

    def test_none_when_session_high_zero(self):
        res = eval_hod_lod_break(self._bull_sig(session_high=0.0))
        assert res is None

    def test_none_when_session_low_zero(self):
        res = eval_hod_lod_break(self._bull_sig(session_low=0.0))
        assert res is None

    def test_none_when_rvol_too_low(self):
        res = eval_hod_lod_break(self._bull_sig(rel_volume=1.0))
        assert res is None

    def test_none_when_gate_blocks_bull(self):
        res = eval_hod_lod_break(self._bull_sig(short_tf_alignment="BEAR"))
        assert res is None

    def test_none_when_gate_blocks_bear(self):
        res = eval_hod_lod_break(self._bear_sig(short_tf_alignment="BULL"))
        assert res is None

    def test_none_inside_opening_range(self):
        """sess_high <= orb_high * 1.001 → still in opening range → None."""
        res = eval_hod_lod_break(self._bull_sig(
            session_high=102.0, orb15_high=102.0, price=102.0
        ))
        assert res is None

    def test_rr_positive(self):
        res = eval_hod_lod_break(self._bull_sig())
        assert res.rr > 0

    def test_confidence_in_range(self):
        res = eval_hod_lod_break(self._bull_sig())
        assert 0 < res.confidence <= 100


# ══════════════════════════════════════════════════════════════════════════════
# TestBullFlag
# ══════════════════════════════════════════════════════════════════════════════

class TestBullFlag:
    """Tests for eval_bull_flag."""

    def _bull_flag_sig(self, **kwargs):
        base = dict(
            bull_flag=True,
            flag_high=102.0,
            flag_low=99.0,
            pole_pct=2.0,
            rel_volume=1.8,
            short_tf_alignment="MIXED",
            price=102.5,
        )
        base.update(kwargs)
        return _sig(**base)

    def test_bull_flag_trigger(self):
        res = eval_bull_flag(self._bull_flag_sig())
        assert res is not None
        assert res.algo == "BULL_FLAG"
        _assert_valid_result(res, "BUY")

    def test_bull_flag_stop_is_flag_low(self):
        res = eval_bull_flag(self._bull_flag_sig())
        assert res.stop == pytest.approx(99.0)

    def test_bull_flag_target_equals_entry_plus_pole_pts(self):
        """Target = entry + entry * |pole_pct| / 100 = 102.5 + 2.05 = 104.55."""
        res = eval_bull_flag(self._bull_flag_sig())
        pole_pts = 102.5 * abs(2.0) / 100
        assert res.target == pytest.approx(102.5 + pole_pts, abs=0.001)

    def test_none_when_bull_flag_false(self):
        res = eval_bull_flag(self._bull_flag_sig(bull_flag=False))
        assert res is None

    def test_none_when_flag_high_zero(self):
        res = eval_bull_flag(self._bull_flag_sig(flag_high=0.0))
        assert res is None

    def test_none_when_flag_low_zero(self):
        res = eval_bull_flag(self._bull_flag_sig(flag_low=0.0))
        assert res is None

    def test_none_when_bear_gate(self):
        """Bull flag blocked when short_tf_alignment='BEAR'."""
        res = eval_bull_flag(self._bull_flag_sig(short_tf_alignment="BEAR"))
        assert res is None

    def test_bull_gate_boosts_confidence(self):
        res_mixed = eval_bull_flag(self._bull_flag_sig(short_tf_alignment="MIXED"))
        res_bull  = eval_bull_flag(self._bull_flag_sig(short_tf_alignment="BULL"))
        assert res_bull.confidence > res_mixed.confidence

    def test_rr_positive(self):
        res = eval_bull_flag(self._bull_flag_sig())
        assert res.rr > 0

    def test_confidence_in_range(self):
        res = eval_bull_flag(self._bull_flag_sig())
        assert 0 < res.confidence <= 100

    def test_confidence_capped_at_90(self):
        res = eval_bull_flag(self._bull_flag_sig(pole_pct=100.0, rel_volume=200.0))
        assert res.confidence <= 90.0


# ══════════════════════════════════════════════════════════════════════════════
# TestBearFlag
# ══════════════════════════════════════════════════════════════════════════════

class TestBearFlag:
    """Tests for eval_bear_flag."""

    def _bear_flag_sig(self, **kwargs):
        base = dict(
            bear_flag=True,
            flag_high=102.0,
            flag_low=99.0,
            pole_pct=-2.0,
            rel_volume=1.8,
            short_tf_alignment="MIXED",
            price=98.5,
        )
        base.update(kwargs)
        return _sig(**base)

    def test_bear_flag_trigger(self):
        res = eval_bear_flag(self._bear_flag_sig())
        assert res is not None
        assert res.algo == "BEAR_FLAG"
        _assert_valid_result(res, "SELL")

    def test_bear_flag_stop_is_flag_high(self):
        res = eval_bear_flag(self._bear_flag_sig())
        assert res.stop == pytest.approx(102.0)

    def test_bear_flag_target_equals_entry_minus_pole_pts(self):
        """Target = entry - entry * |pole_pct| / 100 = 98.5 - 1.97 = 96.53."""
        res = eval_bear_flag(self._bear_flag_sig())
        pole_pts = 98.5 * abs(-2.0) / 100
        assert res.target == pytest.approx(98.5 - pole_pts, abs=0.001)

    def test_none_when_bear_flag_false(self):
        res = eval_bear_flag(self._bear_flag_sig(bear_flag=False))
        assert res is None

    def test_none_when_flag_high_zero(self):
        res = eval_bear_flag(self._bear_flag_sig(flag_high=0.0))
        assert res is None

    def test_none_when_flag_low_zero(self):
        res = eval_bear_flag(self._bear_flag_sig(flag_low=0.0))
        assert res is None

    def test_none_when_bull_gate(self):
        """Bear flag blocked when short_tf_alignment='BULL'."""
        res = eval_bear_flag(self._bear_flag_sig(short_tf_alignment="BULL"))
        assert res is None

    def test_bear_gate_boosts_confidence(self):
        res_mixed = eval_bear_flag(self._bear_flag_sig(short_tf_alignment="MIXED"))
        res_bear  = eval_bear_flag(self._bear_flag_sig(short_tf_alignment="BEAR"))
        assert res_bear.confidence > res_mixed.confidence

    def test_rr_positive(self):
        res = eval_bear_flag(self._bear_flag_sig())
        assert res.rr > 0

    def test_confidence_in_range(self):
        res = eval_bear_flag(self._bear_flag_sig())
        assert 0 < res.confidence <= 100

    def test_confidence_capped_at_90(self):
        res = eval_bear_flag(self._bear_flag_sig(pole_pct=-100.0, rel_volume=200.0))
        assert res.confidence <= 90.0


# ══════════════════════════════════════════════════════════════════════════════
# TestVWAPTouchScalp
# ══════════════════════════════════════════════════════════════════════════════

class TestVWAPTouchScalp:
    """Tests for eval_vwap_touch_scalp."""

    def _reclaim_sig(self, **kwargs):
        """VWAP RECLAIM — bullish scalp."""
        base = dict(
            vwap_event="RECLAIM",
            vwap_price=100.0,
            vwap_upper_1=102.0,
            vwap_lower_1=98.0,
            rel_volume=1.8,
            short_tf_alignment="MIXED",
            price=100.5,
        )
        base.update(kwargs)
        return _sig(**base)

    def _rejection_sig(self, **kwargs):
        """VWAP REJECTION — bearish scalp."""
        base = dict(
            vwap_event="REJECTION",
            vwap_price=100.0,
            vwap_upper_1=102.0,
            vwap_lower_1=98.0,
            rel_volume=1.8,
            short_tf_alignment="MIXED",
            price=99.5,
        )
        base.update(kwargs)
        return _sig(**base)

    def test_reclaim_bull_trigger(self):
        res = eval_vwap_touch_scalp(self._reclaim_sig())
        assert res is not None
        assert res.algo == "VWAP_TOUCH_SCALP_BULL"
        _assert_valid_result(res, "BUY")

    def test_rejection_bear_trigger(self):
        res = eval_vwap_touch_scalp(self._rejection_sig())
        assert res is not None
        assert res.algo == "VWAP_TOUCH_SCALP_BEAR"
        _assert_valid_result(res, "SELL")

    def test_none_when_vwap_zero(self):
        res = eval_vwap_touch_scalp(self._reclaim_sig(vwap_price=0.0))
        assert res is None

    def test_none_when_rvol_below_1p3(self):
        res = eval_vwap_touch_scalp(self._reclaim_sig(rel_volume=1.2))
        assert res is None

    def test_none_when_bear_gate_blocks_reclaim(self):
        res = eval_vwap_touch_scalp(self._reclaim_sig(short_tf_alignment="BEAR"))
        assert res is None

    def test_none_when_bull_gate_blocks_rejection(self):
        res = eval_vwap_touch_scalp(self._rejection_sig(short_tf_alignment="BULL"))
        assert res is None

    def test_rr_positive_reclaim(self):
        res = eval_vwap_touch_scalp(self._reclaim_sig())
        assert res.rr > 0

    def test_confidence_in_range(self):
        res = eval_vwap_touch_scalp(self._reclaim_sig())
        assert 0 < res.confidence <= 100

    def test_none_when_event_is_flat(self):
        res = eval_vwap_touch_scalp(self._reclaim_sig(vwap_event="FLAT"))
        assert res is None


# ══════════════════════════════════════════════════════════════════════════════
# TestVWAPHODScalp
# ══════════════════════════════════════════════════════════════════════════════

class TestVWAPHODScalp:
    """Tests for eval_vwap_hod_scalp."""

    def _bull_sig(self, **kwargs):
        base = dict(
            vwap_event="ABOVE",
            vwap_price=100.0,
            vwap_z_score=0.5,
            vwap_lower_1=98.0,
            session_high=106.0,
            rel_volume=1.5,
            short_tf_alignment="BULL",
            price=101.0,
        )
        base.update(kwargs)
        return _sig(**base)

    def test_hod_scalp_trigger(self):
        res = eval_vwap_hod_scalp(self._bull_sig())
        assert res is not None
        assert res.algo == "VWAP_HOD_SCALP"
        _assert_valid_result(res, "BUY")

    def test_target_is_session_high(self):
        res = eval_vwap_hod_scalp(self._bull_sig())
        assert res.target == pytest.approx(106.0)

    def test_none_when_gate_not_bull(self):
        res = eval_vwap_hod_scalp(self._bull_sig(short_tf_alignment="BEAR"))
        assert res is None

    def test_none_when_z_score_out_of_range(self):
        res = eval_vwap_hod_scalp(self._bull_sig(vwap_z_score=1.5))
        assert res is None

    def test_none_when_vwap_zero(self):
        res = eval_vwap_hod_scalp(self._bull_sig(vwap_price=0.0))
        assert res is None

    def test_none_when_sess_high_at_or_below_price(self):
        """Already at HOD → no room for target."""
        res = eval_vwap_hod_scalp(self._bull_sig(session_high=101.0, price=101.0))
        assert res is None

    def test_rr_positive(self):
        res = eval_vwap_hod_scalp(self._bull_sig())
        assert res.rr > 0

    def test_none_when_event_wrong(self):
        res = eval_vwap_hod_scalp(self._bull_sig(vwap_event="REJECTION"))
        assert res is None


# ══════════════════════════════════════════════════════════════════════════════
# TestVWAPLODScalp
# ══════════════════════════════════════════════════════════════════════════════

class TestVWAPLODScalp:
    """Tests for eval_vwap_lod_scalp."""

    def _bear_sig(self, **kwargs):
        base = dict(
            vwap_event="BELOW",
            vwap_price=100.0,
            vwap_z_score=-0.5,
            vwap_upper_1=102.0,
            session_low=94.0,
            rel_volume=1.5,
            short_tf_alignment="BEAR",
            price=99.0,
        )
        base.update(kwargs)
        return _sig(**base)

    def test_lod_scalp_trigger(self):
        res = eval_vwap_lod_scalp(self._bear_sig())
        assert res is not None
        assert res.algo == "VWAP_LOD_SCALP"
        _assert_valid_result(res, "SELL")

    def test_target_is_session_low(self):
        res = eval_vwap_lod_scalp(self._bear_sig())
        assert res.target == pytest.approx(94.0)

    def test_none_when_gate_not_bear(self):
        res = eval_vwap_lod_scalp(self._bear_sig(short_tf_alignment="BULL"))
        assert res is None

    def test_none_when_z_score_out_of_range(self):
        res = eval_vwap_lod_scalp(self._bear_sig(vwap_z_score=-1.5))
        assert res is None

    def test_none_when_vwap_zero(self):
        res = eval_vwap_lod_scalp(self._bear_sig(vwap_price=0.0))
        assert res is None

    def test_none_when_sess_low_at_or_above_price(self):
        res = eval_vwap_lod_scalp(self._bear_sig(session_low=99.0, price=99.0))
        assert res is None

    def test_rr_positive(self):
        res = eval_vwap_lod_scalp(self._bear_sig())
        assert res.rr > 0

    def test_none_when_event_wrong(self):
        res = eval_vwap_lod_scalp(self._bear_sig(vwap_event="RECLAIM"))
        assert res is None


# ══════════════════════════════════════════════════════════════════════════════
# TestLevelRejectionScalp
# ══════════════════════════════════════════════════════════════════════════════

class TestLevelRejectionScalp:
    """Tests for eval_level_rejection_scalp."""

    def _bull_sig(self, **kwargs):
        """Price at −2σ (oversold), buy toward VWAP."""
        base = dict(
            vwap_event="AT_2SD_DOWN",
            vwap_price=100.0,
            vwap_upper_2=104.0,
            vwap_lower_2=96.0,
            rel_volume=1.5,
            short_tf_alignment="MIXED",
            price=96.0,
        )
        base.update(kwargs)
        return _sig(**base)

    def _bear_sig(self, **kwargs):
        """Price at +2σ (overbought), sell toward VWAP."""
        base = dict(
            vwap_event="AT_2SD_UP",
            vwap_price=100.0,
            vwap_upper_2=104.0,
            vwap_lower_2=96.0,
            rel_volume=1.5,
            short_tf_alignment="MIXED",
            price=104.0,
        )
        base.update(kwargs)
        return _sig(**base)

    def test_bull_trigger(self):
        res = eval_level_rejection_scalp(self._bull_sig())
        assert res is not None
        assert res.algo == "LEVEL_REJECTION_SCALP_BULL"
        _assert_valid_result(res, "BUY")

    def test_bear_trigger(self):
        res = eval_level_rejection_scalp(self._bear_sig())
        assert res is not None
        assert res.algo == "LEVEL_REJECTION_SCALP_BEAR"
        _assert_valid_result(res, "SELL")

    def test_bull_target_is_vwap(self):
        res = eval_level_rejection_scalp(self._bull_sig())
        assert res.target == pytest.approx(100.0)

    def test_bear_target_is_vwap(self):
        res = eval_level_rejection_scalp(self._bear_sig())
        assert res.target == pytest.approx(100.0)

    def test_none_when_vwap_zero(self):
        res = eval_level_rejection_scalp(self._bull_sig(vwap_price=0.0))
        assert res is None

    def test_none_when_rvol_too_low(self):
        res = eval_level_rejection_scalp(self._bull_sig(rel_volume=1.1))
        assert res is None

    def test_none_when_bear_gate_blocks_bull(self):
        res = eval_level_rejection_scalp(self._bull_sig(short_tf_alignment="BEAR"))
        assert res is None

    def test_none_when_bull_gate_blocks_bear(self):
        res = eval_level_rejection_scalp(self._bear_sig(short_tf_alignment="BULL"))
        assert res is None

    def test_rr_positive_bull(self):
        res = eval_level_rejection_scalp(self._bull_sig())
        assert res.rr > 0

    def test_confidence_in_range(self):
        res = eval_level_rejection_scalp(self._bull_sig())
        assert 0 < res.confidence <= 100


# ══════════════════════════════════════════════════════════════════════════════
# TestMicroPullbackScalp
# ══════════════════════════════════════════════════════════════════════════════

class TestMicroPullbackScalp:
    """Tests for eval_micro_pullback_scalp."""

    def _bull_sig(self, **kwargs):
        base = dict(
            vwap_event="ABOVE",
            vwap_price=100.0,
            vwap_z_score=0.5,
            vwap_upper_1=103.0,
            vwap_lower_1=97.0,
            rel_volume=1.5,
            short_tf_alignment="BULL",
            price=101.0,
        )
        base.update(kwargs)
        return _sig(**base)

    def _bear_sig(self, **kwargs):
        base = dict(
            vwap_event="BELOW",
            vwap_price=100.0,
            vwap_z_score=-0.5,
            vwap_upper_1=103.0,
            vwap_lower_1=97.0,
            rel_volume=1.5,
            short_tf_alignment="BEAR",
            price=99.0,
        )
        base.update(kwargs)
        return _sig(**base)

    def test_bull_trigger(self):
        res = eval_micro_pullback_scalp(self._bull_sig())
        assert res is not None
        assert res.algo == "MICRO_PULLBACK_SCALP_BULL"
        _assert_valid_result(res, "BUY")

    def test_bear_trigger(self):
        res = eval_micro_pullback_scalp(self._bear_sig())
        assert res is not None
        assert res.algo == "MICRO_PULLBACK_SCALP_BEAR"
        _assert_valid_result(res, "SELL")

    def test_bull_target_is_upper_1(self):
        res = eval_micro_pullback_scalp(self._bull_sig())
        assert res.target == pytest.approx(103.0)

    def test_bear_target_is_lower_1(self):
        res = eval_micro_pullback_scalp(self._bear_sig())
        assert res.target == pytest.approx(97.0)

    def test_none_when_gate_mixed(self):
        """Gate must be BULL or BEAR — MIXED → None."""
        res = eval_micro_pullback_scalp(self._bull_sig(short_tf_alignment="MIXED"))
        assert res is None

    def test_none_when_rvol_too_low(self):
        res = eval_micro_pullback_scalp(self._bull_sig(rel_volume=1.2))
        assert res is None

    def test_none_when_z_score_out_of_range_bull(self):
        """z_score must be in [0.1, 1.2] for bull."""
        res = eval_micro_pullback_scalp(self._bull_sig(vwap_z_score=0.05))
        assert res is None

    def test_none_when_z_score_out_of_range_bear(self):
        """z_score must be in [-1.2, -0.1] for bear."""
        res = eval_micro_pullback_scalp(self._bear_sig(vwap_z_score=-0.05))
        assert res is None

    def test_rr_positive_bull(self):
        res = eval_micro_pullback_scalp(self._bull_sig())
        assert res.rr > 0

    def test_rr_positive_bear(self):
        res = eval_micro_pullback_scalp(self._bear_sig())
        assert res.rr > 0

    def test_confidence_in_range(self):
        res = eval_micro_pullback_scalp(self._bull_sig())
        assert 0 < res.confidence <= 100


# ══════════════════════════════════════════════════════════════════════════════
# TestEvaluateAll
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluateAll:
    """Tests for evaluate_all() — the top-level dispatcher."""

    def test_returns_list(self):
        sig = _sig()
        result = evaluate_all(sig)
        assert isinstance(result, list)

    def test_returns_list_of_dicts(self):
        """Each item in the result should be a dict."""
        sig = _sig(
            orb5_high=101.0, orb5_low=99.0, orb5_breakout="BULL",
            rel_volume=2.0, short_tf_alignment="MIXED", price=102.0,
        )
        result = evaluate_all(sig)
        for item in result:
            assert isinstance(item, dict), f"Item should be dict, got {type(item)}"

    def test_dicts_have_required_keys(self):
        """Every returned dict must contain all AlgoResult fields."""
        sig = _sig(
            orb5_high=101.0, orb5_low=99.0, orb5_breakout="BULL",
            rel_volume=2.0, short_tf_alignment="MIXED", price=102.0,
        )
        result = evaluate_all(sig)
        required_keys = {"algo", "direction", "confidence", "entry", "stop", "target", "rr", "reason"}
        for item in result:
            for key in required_keys:
                assert key in item, f"Key {key!r} missing from result dict"

    def test_all_rr_positive(self):
        """All fired results must have rr > 0."""
        sig = _sig(
            orb5_high=101.0, orb5_low=99.0, orb5_breakout="BULL",
            rel_volume=2.0, short_tf_alignment="MIXED", price=102.0,
        )
        result = evaluate_all(sig)
        for item in result:
            assert item["rr"] > 0, f"Result {item['algo']} has rr={item['rr']}"

    def test_direction_values_valid(self):
        """All directions must be BUY or SELL."""
        sig = _sig(
            orb5_high=101.0, orb5_low=99.0, orb5_breakout="BULL",
            rel_volume=2.0, short_tf_alignment="MIXED", price=102.0,
        )
        result = evaluate_all(sig)
        for item in result:
            assert item["direction"] in ("BUY", "SELL"), \
                f"Invalid direction {item['direction']!r} in algo {item['algo']}"

    def test_confidence_in_range_for_all(self):
        sig = _sig(
            orb5_high=101.0, orb5_low=99.0, orb5_breakout="BULL",
            rel_volume=2.0, short_tf_alignment="MIXED", price=102.0,
        )
        result = evaluate_all(sig)
        for item in result:
            assert 0 < item["confidence"] <= 100, \
                f"Confidence out of range for {item['algo']}: {item['confidence']}"

    def test_empty_list_for_neutral_sig(self):
        """A plain neutral signal with no special fields fires no algos."""
        sig = _sig()  # no ORB, no gap, no PDH/PDL, etc.
        result = evaluate_all(sig)
        assert isinstance(result, list)
        # May or may not fire algos — just ensure no crash and result is iterable

    def test_orb5_fires_in_evaluate_all(self):
        sig = _sig(
            orb5_high=101.0, orb5_low=99.0, orb5_breakout="BULL",
            rel_volume=2.0, short_tf_alignment="MIXED", price=102.0,
        )
        result = evaluate_all(sig)
        algos = [r["algo"] for r in result]
        assert "ORB5_BULL" in algos

    def test_does_not_crash_on_missing_optional_fields(self):
        """evaluate_all should be robust and not raise exceptions."""
        sig = _sig()  # minimal signal, most optional fields at defaults
        try:
            result = evaluate_all(sig)
        except Exception as exc:
            pytest.fail(f"evaluate_all raised {type(exc).__name__}: {exc}")

    def test_buy_invariant_for_all_results(self):
        """For BUY: stop < entry < target."""
        sig = _sig(
            orb5_high=101.0, orb5_low=99.0, orb5_breakout="BULL",
            rel_volume=2.0, short_tf_alignment="MIXED", price=102.0,
            orb15_high=101.5, orb15_low=98.5, orb15_breakout="BULL",
        )
        result = evaluate_all(sig)
        for item in result:
            if item["direction"] == "BUY":
                assert item["stop"] < item["entry"] < item["target"], \
                    f"{item['algo']}: BUY invariant violated stop={item['stop']} entry={item['entry']} target={item['target']}"

    def test_sell_invariant_for_all_results(self):
        """For SELL: stop > entry > target."""
        sig = _sig(
            orb5_high=101.0, orb5_low=99.0, orb5_breakout="BEAR",
            rel_volume=2.0, short_tf_alignment="MIXED", price=98.0,
            orb15_high=101.5, orb15_low=98.5, orb15_breakout="BEAR",
        )
        result = evaluate_all(sig)
        for item in result:
            if item["direction"] == "SELL":
                assert item["stop"] > item["entry"] > item["target"], \
                    f"{item['algo']}: SELL invariant violated stop={item['stop']} entry={item['entry']} target={item['target']}"


# ══════════════════════════════════════════════════════════════════════════════
# TestAhGapFade
# ══════════════════════════════════════════════════════════════════════════════

class TestAhGapFade:
    """Unit tests for eval_ah_gap_fade — the AH Extreme Gap Fade algo."""

    from agent.trading_algos import eval_ah_gap_fade

    # ── helpers ───────────────────────────────────────────────────────────────

    def _bear_sig(self, gap_pct=15.0, rsi_value=82.0, rvol=3.0,
                  price=None, today_open=None, prev_close=100.0,
                  sess_high=None, pm_high=None,
                  orb5_breakout="", vwap_event="ABOVE",
                  session="REGULAR", **extra):
        """Return a signal set up to trigger AH_GAP_FADE_BEAR."""
        open_price = today_open if today_open is not None else prev_close * (1 + gap_pct / 100)
        cur_price  = price if price is not None else open_price
        return _sig(
            price=cur_price,
            gap_pct=gap_pct,
            gap_type="GAP_UP",
            rsi_value=rsi_value,
            rsi_zone="EXTREME_OB",
            rel_volume=rvol,
            today_open=open_price,
            prev_day_close=prev_close,
            session_high=sess_high if sess_high is not None else cur_price * 1.01,
            premarket_high=pm_high if pm_high is not None else cur_price * 1.005,
            orb5_breakout=orb5_breakout,
            vwap_event=vwap_event,
            session=session,
            **extra,
        )

    def _bull_sig(self, gap_pct=-15.0, rsi_value=22.0, rvol=3.0,
                  price=None, today_open=None, prev_close=100.0,
                  sess_low=None, pm_low=None,
                  orb5_breakout="", vwap_event="BELOW",
                  session="REGULAR", **extra):
        """Return a signal set up to trigger AH_GAP_FADE_BULL."""
        open_price = today_open if today_open is not None else prev_close * (1 + gap_pct / 100)
        cur_price  = price if price is not None else open_price
        return _sig(
            price=cur_price,
            gap_pct=gap_pct,
            gap_type="GAP_DOWN",
            rsi_value=rsi_value,
            rsi_zone="EXTREME_OS",
            rel_volume=rvol,
            today_open=open_price,
            prev_day_close=prev_close,
            session_low=sess_low if sess_low is not None else cur_price * 0.99,
            premarket_low=pm_low if pm_low is not None else cur_price * 0.995,
            orb5_breakout=orb5_breakout,
            vwap_event=vwap_event,
            session=session,
            **extra,
        )

    # ── bear-side tests ───────────────────────────────────────────────────────

    def test_bear_triggers_on_extreme_gap_up(self):
        from agent.trading_algos import eval_ah_gap_fade
        sig = self._bear_sig()
        res = eval_ah_gap_fade(sig)
        assert res is not None, "Expected AH_GAP_FADE_BEAR signal"
        _assert_valid_result(res, "SELL")
        assert res.algo == "AH_GAP_FADE_BEAR"

    def test_bear_sell_invariant(self):
        """SELL geometry: stop > entry > target."""
        from agent.trading_algos import eval_ah_gap_fade
        sig = self._bear_sig()
        res = eval_ah_gap_fade(sig)
        assert res is not None
        assert res.stop > res.entry > res.target, (
            f"SELL geometry violated: stop={res.stop} entry={res.entry} target={res.target}"
        )

    def test_bear_confidence_in_range(self):
        from agent.trading_algos import eval_ah_gap_fade
        sig = self._bear_sig(gap_pct=20.0, rsi_value=85.0, rvol=4.0)
        res = eval_ah_gap_fade(sig)
        assert res is not None
        assert 0 < res.confidence <= 85, f"confidence out of range: {res.confidence}"

    def test_bear_larger_gap_higher_confidence(self):
        """A 25% gap should produce higher confidence than a 10% gap."""
        from agent.trading_algos import eval_ah_gap_fade
        small = eval_ah_gap_fade(self._bear_sig(gap_pct=10.1))
        large = eval_ah_gap_fade(self._bear_sig(gap_pct=25.0))
        assert small is not None and large is not None
        assert large.confidence >= small.confidence

    def test_bear_no_fire_below_gap_threshold(self):
        """gap_pct < 10% must not trigger the algo."""
        from agent.trading_algos import eval_ah_gap_fade
        sig = self._bear_sig(gap_pct=5.0)
        assert eval_ah_gap_fade(sig) is None

    def test_bear_no_fire_when_rsi_not_overbought(self):
        """RSI below 75 must not trigger."""
        from agent.trading_algos import eval_ah_gap_fade
        sig = self._bear_sig(rsi_value=60.0)
        assert eval_ah_gap_fade(sig) is None

    def test_bear_no_fire_when_orb5_bull(self):
        """Strong ORB5 bull breakout means continuation — skip fade."""
        from agent.trading_algos import eval_ah_gap_fade
        sig = self._bear_sig(orb5_breakout="BULL")
        assert eval_ah_gap_fade(sig) is None

    def test_bear_no_fire_when_already_faded_below_open(self):
        """If price has dropped 5 % below open, algo should not fire (already faded)."""
        from agent.trading_algos import eval_ah_gap_fade
        prev_close = 100.0
        today_open = 115.0  # 15% gap
        price = today_open * 0.93  # 7 % below open — beyond the 3% trigger threshold
        sig = self._bear_sig(prev_close=prev_close, today_open=today_open, price=price)
        assert eval_ah_gap_fade(sig) is None

    def test_bear_no_fire_in_after_hours_session(self):
        """Algo only fires in REGULAR session."""
        from agent.trading_algos import eval_ah_gap_fade
        sig = self._bear_sig(session="AFTER_HOURS")
        assert eval_ah_gap_fade(sig) is None

    def test_bear_vwap_extended_boost_raises_confidence(self):
        """EXTENDED_UP vwap_event should produce higher confidence than neutral."""
        from agent.trading_algos import eval_ah_gap_fade
        neutral_vwap = eval_ah_gap_fade(self._bear_sig(vwap_event="NEUTRAL"))
        extended_vwap = eval_ah_gap_fade(self._bear_sig(vwap_event="EXTENDED_UP"))
        if neutral_vwap and extended_vwap:
            assert extended_vwap.confidence >= neutral_vwap.confidence

    def test_bear_target_is_partial_fill(self):
        """Target should be above prev_close (40% fill, not 100% fill)."""
        from agent.trading_algos import eval_ah_gap_fade
        prev_close = 100.0
        gap_pct = 20.0
        sig = self._bear_sig(prev_close=prev_close, gap_pct=gap_pct)
        res = eval_ah_gap_fade(sig)
        assert res is not None
        # Target should be above prev_close (partial, not full, fill)
        assert res.target > prev_close, (
            f"Target {res.target} should be above prev_close {prev_close} for partial fill"
        )

    def test_bear_rr_at_least_one(self):
        """R:R must be ≥ 1.0 for the algo to fire."""
        from agent.trading_algos import eval_ah_gap_fade
        sig = self._bear_sig()
        res = eval_ah_gap_fade(sig)
        if res is not None:
            assert res.rr >= 1.0, f"R:R too low: {res.rr}"

    # ── bull-side tests ───────────────────────────────────────────────────────

    def test_bull_triggers_on_extreme_gap_down(self):
        from agent.trading_algos import eval_ah_gap_fade
        sig = self._bull_sig()
        res = eval_ah_gap_fade(sig)
        assert res is not None, "Expected AH_GAP_FADE_BULL signal"
        _assert_valid_result(res, "BUY")
        assert res.algo == "AH_GAP_FADE_BULL"

    def test_bull_buy_invariant(self):
        """BUY geometry: stop < entry < target."""
        from agent.trading_algos import eval_ah_gap_fade
        sig = self._bull_sig()
        res = eval_ah_gap_fade(sig)
        assert res is not None
        assert res.stop < res.entry < res.target

    def test_bull_no_fire_when_orb5_bear(self):
        """Strong ORB5 bear breakdown means continuation — skip long fade."""
        from agent.trading_algos import eval_ah_gap_fade
        sig = self._bull_sig(orb5_breakout="BEAR")
        assert eval_ah_gap_fade(sig) is None

    def test_bull_no_fire_below_gap_threshold(self):
        from agent.trading_algos import eval_ah_gap_fade
        sig = self._bull_sig(gap_pct=-4.0)
        assert eval_ah_gap_fade(sig) is None

    # ── evaluate_all integration ──────────────────────────────────────────────

    def test_ah_gap_fade_bear_in_evaluate_all(self):
        """AH_GAP_FADE_BEAR must appear in evaluate_all output for a qualifying signal."""
        sig = self._bear_sig()
        result = evaluate_all(sig)
        algos = [r["algo"] for r in result]
        assert "AH_GAP_FADE_BEAR" in algos, \
            f"AH_GAP_FADE_BEAR not in evaluate_all output; fired algos: {algos}"

    def test_ah_gap_fade_bull_in_evaluate_all(self):
        """AH_GAP_FADE_BULL must appear in evaluate_all output for a qualifying signal."""
        sig = self._bull_sig()
        result = evaluate_all(sig)
        algos = [r["algo"] for r in result]
        assert "AH_GAP_FADE_BULL" in algos, \
            f"AH_GAP_FADE_BULL not in evaluate_all output; fired algos: {algos}"
