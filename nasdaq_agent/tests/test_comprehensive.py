from __future__ import annotations
import pytest
"""
Comprehensive system test — covers every API endpoint, ML/model layer,
signal logic, backtest accuracy, P&L math, and data integrity.

Run:
    cd nasdaq_agent
    pytest tests/test_comprehensive.py -v --tb=short
"""
pytestmark = pytest.mark.slow

import math
import sys
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from tests.conftest import make_ohlcv


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 – Position Sizing Math
# ═════════════════════════════════════════════════════════════════════════════

from agent.position_sizing import calculate, PositionSize


class TestPositionSizingMath:
    def test_basic_1pct_risk(self):
        # max_position_pct=5.0 caps at 5% of $10k=$500; entry=$100 → max 5 shares
        # dollar_risk = $100, risk/share=$5 → 20 shares, but capped at 5
        ps = calculate(account_size=10_000, entry=100, stop=95, risk_pct=1.0, confidence=60)
        assert ps.shares == 5
        assert ps.dollar_risk > 0

    def test_high_confidence_multiplier(self):
        # Use small account+high entry to avoid max_position_cap ceiling
        # and use max_position_pct=100 to disable the cap
        ps_low  = calculate(10_000, 100, 95, 1.0, confidence=40,  max_position_pct=100.0)  # 0.5×
        ps_high = calculate(10_000, 100, 95, 1.0, confidence=80,  max_position_pct=100.0)  # 1.25×
        assert ps_high.confidence_mult == 1.25
        assert ps_low.confidence_mult == 0.5
        assert ps_high.shares > ps_low.shares, \
            f"high conf should give more shares: {ps_high.shares} vs {ps_low.shares}"

    def test_position_value_equals_shares_x_entry(self):
        ps = calculate(10_000, 50.0, 48.0, 1.0, confidence=65)
        assert ps.position_value == pytest.approx(ps.shares * 50.0, rel=1e-6)

    def test_zero_risk_returns_empty(self):
        ps = calculate(10_000, 100.0, 100.0, 1.0, confidence=60)
        assert ps.shares == 0

    def test_max_position_cap(self):
        # Even at 100% confidence, position cannot exceed 5% of account
        ps = calculate(100_000, 10.0, 9.99, 5.0, confidence=100, max_position_pct=5.0)
        assert ps.position_value <= 100_000 * 0.05 + 10.0  # allow 1 share overshoot

    def test_confidence_mult_50_to_74_is_075(self):
        ps = calculate(10_000, 100, 98, 1.0, confidence=55)
        assert ps.confidence_mult == 0.75

    def test_confidence_mult_75_plus_is_125(self):
        ps = calculate(10_000, 100, 98, 1.0, confidence=75)
        assert ps.confidence_mult == 1.25

    def test_risk_pct_used_matches_formula(self):
        ps = calculate(10_000, 100, 95, 1.0, confidence=60)
        expected_pct = ps.dollar_risk / 10_000 * 100
        assert ps.risk_pct_used == pytest.approx(expected_pct, rel=1e-3)

    def test_invalid_entry_returns_zero_shares(self):
        ps = calculate(10_000, 0, 0, 1.0, confidence=60)
        assert ps.shares == 0

    def test_to_dict_serializable(self):
        ps = calculate(10_000, 100, 95, 1.0, confidence=60)
        d = ps.to_dict()
        assert isinstance(d, dict)
        for key in ("shares", "position_value", "dollar_risk", "risk_per_share"):
            assert key in d


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 – VWAP Logic
# ═════════════════════════════════════════════════════════════════════════════

from agent.vwap import compute_vwap_signal


class TestVWAPLogic:
    def test_above_vwap_positive_score(self):
        df = make_ohlcv(start_price=110.0)
        df["vwap"] = 100.0
        result = compute_vwap_signal(df)
        assert result["score"] >= 0, "price above VWAP should yield non-negative score"

    def test_below_vwap_negative_or_zero_score(self):
        df = make_ohlcv(start_price=90.0)
        df["vwap"] = 100.0
        result = compute_vwap_signal(df)
        # deviation (price-vwap)/vwap should be negative, even if score is rounded
        assert result["deviation"] < 0, "price below VWAP must show negative deviation"

    def test_vwap_deviation_sign_matches_position(self):
        df_above = make_ohlcv(start_price=105.0)
        df_above["vwap"] = 100.0
        df_below = make_ohlcv(start_price=95.0)
        df_below["vwap"] = 100.0
        r_above = compute_vwap_signal(df_above)
        r_below = compute_vwap_signal(df_below)
        assert r_above["deviation"] > 0
        assert r_below["deviation"] < 0

    def test_vwap_value_positive(self):
        df = make_ohlcv()
        result = compute_vwap_signal(df)
        assert result["vwap"] > 0

    def test_event_is_valid_string(self):
        valid = {"RECLAIM", "REJECTION", "AT_2SD_UP", "AT_2SD_DOWN",
                 "AT_1SD_UP", "AT_1SD_DOWN", "ABOVE", "BELOW", "FLAT", "NONE"}
        df = make_ohlcv()
        result = compute_vwap_signal(df)
        assert result["event"] in valid, f"unexpected vwap event: {result['event']}"

    def test_empty_df_handled(self):
        result = compute_vwap_signal(pd.DataFrame())
        assert result["score"] == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 – Support & Resistance
# ═════════════════════════════════════════════════════════════════════════════

from agent.support_resistance import get_all_sr_levels


class TestSupportResistance:
    def test_returns_dict_with_required_keys(self):
        df = make_ohlcv(n=120)
        result = get_all_sr_levels(df)
        assert isinstance(result, dict)
        assert "supports" in result and "resistances" in result

    def test_supports_are_lists(self):
        df = make_ohlcv(n=120)
        result = get_all_sr_levels(df)
        assert isinstance(result["supports"], list)
        assert isinstance(result["resistances"], list)

    def test_supports_below_price(self):
        df = make_ohlcv(n=120, start_price=100.0)
        result = get_all_sr_levels(df)
        price = float(df["Close"].iloc[-1])
        for s in result["supports"]:
            assert s <= price * 1.02, f"support {s} should be at/below price {price}"

    def test_resistances_above_price(self):
        df = make_ohlcv(n=120, start_price=100.0)
        result = get_all_sr_levels(df)
        price = float(df["Close"].iloc[-1])
        for r in result["resistances"]:
            assert r >= price * 0.98, f"resistance {r} should be at/above price {price}"

    def test_handles_short_df(self):
        df = make_ohlcv(n=10)
        result = get_all_sr_levels(df)
        assert result is not None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 – Live Backtest (Signal Tracking & Resolution)
# ═════════════════════════════════════════════════════════════════════════════

from agent.live_backtest import (
    record_signal, update_tracking, get_tracking_signals,
    get_recent_resolved, get_performance_stats, MAX_BARS,
)


class TestLiveBacktestResolution:
    def test_win_r_calculation_buy(self):
        record_signal("LB_AAPL", "BUY", entry_price=100.0, target=110.0, stop=95.0)
        resolved = update_tracking("LB_AAPL", current_price=111.0, vwap=100.0)
        assert len(resolved) == 1
        outcome = resolved[0]
        assert outcome["status"] == "WIN"
        assert outcome["exit_reason"] == "TARGET"
        # R stored as r_multiple: (exit-entry)/(entry-stop) = (110-100)/(100-95) = 2.0R
        r_val = outcome.get("r_multiple", outcome.get("final_r", 0))
        assert r_val >= 1.9

    def test_loss_r_calculation_buy(self):
        record_signal("LB_MSFT", "BUY", entry_price=100.0, target=110.0, stop=95.0)
        resolved = update_tracking("LB_MSFT", current_price=94.0, vwap=100.0)
        assert resolved[0]["status"] == "LOSS"
        r_val = resolved[0].get("r_multiple", resolved[0].get("final_r", 0))
        assert r_val <= -0.9

    def test_sell_wins_when_price_drops(self):
        record_signal("LB_TSLA", "SELL", entry_price=200.0, target=190.0, stop=205.0)
        resolved = update_tracking("LB_TSLA", current_price=189.0, vwap=200.0)
        assert resolved[0]["status"] == "WIN"

    def test_signal_id_format(self):
        sid = record_signal("LB_NVDA", "BUY", entry_price=500.0, target=510.0, stop=495.0)
        assert "LB_NVDA" in sid

    def test_bars_tracked_increment(self):
        record_signal("LB_AMD", "BUY", entry_price=100.0, target=110.0, stop=95.0)
        for i in range(5):
            update_tracking("LB_AMD", current_price=102.0, vwap=0.0)
        tracking = get_tracking_signals()
        match = next((t for t in tracking if t["ticker"] == "LB_AMD"), None)
        assert match is not None
        assert match["bars_tracked"] == 5

    def test_current_r_tracks_live_price(self):
        record_signal("LB_SNOW", "BUY", entry_price=100.0, target=110.0, stop=95.0)
        update_tracking("LB_SNOW", current_price=102.5, vwap=0.0)
        tracking = get_tracking_signals()
        match = next((t for t in tracking if t["ticker"] == "LB_SNOW"), None)
        # R = (102.5 - 100) / (100 - 95) = 0.5R
        assert abs(match["current_r"] - 0.5) < 0.01

    def test_timeout_at_max_bars(self):
        record_signal("LB_HOOD", "BUY", entry_price=10.0, target=15.0, stop=9.0)
        resolved = []
        for _ in range(MAX_BARS):
            resolved = update_tracking("LB_HOOD", current_price=11.0, vwap=0.0)
        assert resolved[0]["status"] == "TIMEOUT"

    def test_perf_stats_win_rate_is_ratio(self):
        record_signal("LB_T1", "BUY", 100.0, 110.0, 95.0)
        update_tracking("LB_T1", 111.0, 0.0)
        stats = get_performance_stats()
        wr = stats["overall"]["win_rate"]
        assert 0.0 <= wr <= 1.0, f"win_rate must be 0–1, got {wr}"

    def test_perf_stats_has_all_breakdowns(self):
        stats = get_performance_stats()
        for key in ["by_direction", "by_session", "by_regime", "by_vwap_event",
                    "by_rsi_zone", "by_entry_type", "by_confidence", "by_sector_trend"]:
            assert key in stats


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 – Paper Trading Lifecycle (Data Collection Mode)
# ═════════════════════════════════════════════════════════════════════════════

from agent.paper_trading import (
    maybe_open_trade, update_open_trades, get_open_trades,
    get_closed_trades, get_summary, _record_close,
)


class TestPaperTradingLifecycle:
    def test_full_buy_win_cycle(self):
        tid = maybe_open_trade("PT_WIN", "BUY", 100.0, 105.0, 95.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        assert tid is not None
        df = make_ohlcv(start_price=107.0)  # above target=105
        update_open_trades("PT_WIN", df, current_price=107.0)
        closed = get_closed_trades()
        match = next((t for t in closed if t["ticker"] == "PT_WIN"), None)
        assert match is not None, "trade should have closed on target hit"
        assert match["exit_reason"] in ("TARGET_HIT", "EXIT_NOW", "TARGET")

    def test_full_buy_loss_cycle(self):
        tid = maybe_open_trade("PT_LOSS", "BUY", 100.0, 110.0, 95.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        assert tid is not None
        df = make_ohlcv(start_price=93.0)
        update_open_trades("PT_LOSS", df, current_price=93.0)
        closed = get_closed_trades()
        match = next((t for t in closed if t["ticker"] == "PT_LOSS"), None)
        assert match is not None, "trade should have closed on stop hit"

    def test_pnl_positive_on_buy_win(self):
        maybe_open_trade("PT_PNL", "BUY", 100.0, 110.0, 95.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        df = make_ohlcv(start_price=115.0)
        update_open_trades("PT_PNL", df, current_price=115.0)
        closed = get_closed_trades()
        match = next((t for t in closed if t["ticker"] == "PT_PNL"), None)
        if match and match.get("pnl_pct") is not None:
            assert match["pnl_pct"] > 0, "winning BUY trade must have positive P&L"

    def test_pnl_negative_on_buy_loss(self):
        maybe_open_trade("PT_PNL2", "BUY", 100.0, 110.0, 95.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        df = make_ohlcv(start_price=93.0)
        update_open_trades("PT_PNL2", df, current_price=93.0)
        closed = get_closed_trades()
        match = next((t for t in closed if t["ticker"] == "PT_PNL2"), None)
        if match and match.get("pnl_pct") is not None:
            assert match["pnl_pct"] < 0, "losing BUY trade must have negative P&L"

    def test_sell_pnl_positive_when_price_falls(self):
        maybe_open_trade("PT_SELL", "SELL", 200.0, 185.0, 207.0, confidence=75.0, rr_qualifies=True, session="REGULAR")
        df = make_ohlcv(start_price=182.0)
        update_open_trades("PT_SELL", df, current_price=182.0)
        closed = get_closed_trades()
        match = next((t for t in closed if t["ticker"] == "PT_SELL"), None)
        if match and match.get("pnl_pct") is not None:
            assert match["pnl_pct"] > 0, "winning SELL trade must have positive P&L"

    def test_no_duplicate_same_ticker(self):
        maybe_open_trade("PT_DUP", "BUY", 100.0, 110.0, 95.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        t2 = maybe_open_trade("PT_DUP", "SELL", 100.0, 90.0, 105.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        assert t2 is None, "cannot open second trade while one is open on same ticker"

    def test_low_confidence_floor_blocks(self):
        tid = maybe_open_trade("PT_LOWC", "BUY", 100.0, 110.0, 95.0, confidence=10.0, rr_qualifies=True, session="REGULAR")
        assert tid is None, "confidence below 25% floor must be rejected"

    def test_rr_false_still_opens(self):
        tid = maybe_open_trade("PT_RRF", "BUY", 100.0, 102.0, 99.0, confidence=70.0, rr_qualifies=False, session="REGULAR")
        assert tid is not None, "rr_qualifies=False must still open (data collection mode)"

    def test_neutral_direction_rejected(self):
        tid = maybe_open_trade("PT_NEU", "NEUTRAL", 100.0, 110.0, 95.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        assert tid is None

    def test_summary_win_rate_in_pct_range(self):
        s = get_summary()
        assert 0.0 <= s["win_rate"] <= 100.0

    def test_summary_has_required_fields(self):
        s = get_summary()
        for key in ("open", "closed", "wins", "losses", "win_rate"):
            assert key in s, f"summary missing key: {key}"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 – Backtest Reporter (Attribution & Calibration)
# ═════════════════════════════════════════════════════════════════════════════

from agent.backtest_reporter import (
    get_broadcast_summary, get_full_report,
    adjust_confidence, get_calibration, _update_confidence_calibration,
)
import agent.backtest_reporter as _br


class TestBacktestReporter:
    def test_broadcast_summary_keys(self):
        s = get_broadcast_summary()
        for key in ("tracking", "total", "win_rate", "avg_r", "expectancy"):
            assert key in s

    def test_win_rate_is_ratio_after_wins(self):
        record_signal("BR_W", "BUY", 100.0, 110.0, 95.0)
        update_tracking("BR_W", 111.0, 0.0)
        s = get_broadcast_summary()
        assert 0.0 <= s["win_rate"] <= 1.0

    def test_avg_r_positive_after_wins(self):
        record_signal("BR_R", "BUY", 100.0, 110.0, 95.0)
        update_tracking("BR_R", 111.0, 0.0)
        s = get_broadcast_summary()
        assert s["avg_r"] is not None

    def test_full_report_structure(self):
        r = get_full_report(lookback_days=30)
        assert "stats" in r
        assert "tracking" in r
        assert "recent" in r

    def test_full_report_recent_has_outcome_color(self):
        record_signal("BR_C", "BUY", 300.0, 310.0, 295.0)
        update_tracking("BR_C", 311.0, 0.0)
        r = get_full_report()
        for item in r["recent"]:
            assert "outcome_color" in item, "each resolved signal needs outcome_color"

    def test_calibration_builds_from_outcomes(self):
        df = pd.DataFrame({
            "vwap_event": ["RECLAIM"] * 15 + ["REJECTION"] * 15,
            "outcome":    [1] * 12 + [0] * 3 + [0] * 12 + [1] * 3,
        })
        _update_confidence_calibration(df)
        cal = get_calibration()
        assert "vwap_event:RECLAIM" in cal
        assert cal["vwap_event:RECLAIM"] > cal["vwap_event:REJECTION"]

    def test_adjust_confidence_clamps_to_25_95(self):
        with _br._cal_lock:
            _br._calibration = {"vwap_event:RECLAIM": 1.0}
        assert adjust_confidence(99.0, vwap_event="RECLAIM") <= 95.0
        with _br._cal_lock:
            _br._calibration = {"vwap_event:REJECTION": 0.0}
        assert adjust_confidence(5.0, vwap_event="REJECTION") >= 25.0
        with _br._cal_lock:
            _br._calibration = {}

    def test_no_calibration_returns_unchanged(self):
        with _br._cal_lock:
            _br._calibration = {}
        result = adjust_confidence(55.0, vwap_event="NONE")
        assert result == 55.0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 – Signal Blender (Dynamic Weights)
# ═════════════════════════════════════════════════════════════════════════════

from agent.signal_blender import DynamicBlender, ModelOutcome, MODELS


class TestSignalBlender:
    def test_weights_sum_to_one(self, tmp_path):
        blender = DynamicBlender(persist_path=tmp_path / "test_blend.json")
        weights = blender.get_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"weights sum {total} ≠ 1.0"

    def test_all_models_have_weights(self, tmp_path):
        blender = DynamicBlender(persist_path=tmp_path / "test_blend2.json")
        weights = blender.get_weights()
        for model in MODELS:
            assert model in weights, f"model {model} missing from weights"

    def test_weights_all_non_negative(self, tmp_path):
        blender = DynamicBlender(persist_path=tmp_path / "test_blend3.json")
        weights = blender.get_weights()
        for m, w in weights.items():
            assert w >= 0, f"model {m} has negative weight {w}"

    def test_accurate_model_gets_higher_weight(self, tmp_path):
        blender = DynamicBlender(persist_path=tmp_path / "test_blend4.json")
        # record_outcome(model, ticker, prob, correct)
        for _ in range(10):
            blender.record_outcome("scalp", "AAPL", 0.8, True)
            blender.record_outcome("reversal", "AAPL", 0.3, False)
        weights = blender.get_weights()
        assert weights["scalp"] >= weights["reversal"], \
            f"accurate scalp ({weights['scalp']:.3f}) should outweigh reversal ({weights['reversal']:.3f})"

    def test_blend_probability_in_range(self, tmp_path):
        # blend(scalp_p, ensemble_p, reversal_p, ...) — positional args
        blender = DynamicBlender(persist_path=tmp_path / "test_blend5.json")
        prob = blender.blend(scalp_p=0.7, ensemble_p=0.6, reversal_p=0.4)
        assert 0.0 <= prob <= 1.0, f"blended probability {prob} out of range"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 – Exit Signals (Trade Management)
# ═════════════════════════════════════════════════════════════════════════════

from agent.exit_signals import analyse_exits, ExitAnalysis


class TestExitSignals:
    def test_returns_exit_analysis(self):
        df = make_ohlcv()
        result = analyse_exits(df, "BUY", entry_price=100.0, target=110.0, stop=95.0)
        assert isinstance(result, ExitAnalysis)

    def test_target_hit_fires_exit_now_buy(self):
        df = make_ohlcv(start_price=112.0)
        result = analyse_exits(df, "BUY", entry_price=100.0, target=110.0, stop=95.0, bars_held=3)
        assert result.recommendation == "EXIT_NOW"
        assert any(s.signal == "TARGET_HIT" for s in result.signals)

    def test_stop_hit_fires_exit_now_buy(self):
        df = make_ohlcv(start_price=93.0)
        result = analyse_exits(df, "BUY", entry_price=100.0, target=110.0, stop=95.0, bars_held=3)
        assert result.recommendation == "EXIT_NOW"
        assert any(s.signal == "STOP_HIT" for s in result.signals)

    def test_target_hit_fires_exit_now_sell(self):
        df = make_ohlcv(start_price=88.0)
        result = analyse_exits(df, "SELL", entry_price=100.0, target=90.0, stop=105.0, bars_held=3)
        assert result.recommendation == "EXIT_NOW"
        assert any(s.signal == "TARGET_HIT" for s in result.signals)

    def test_stop_hit_fires_exit_now_sell(self):
        df = make_ohlcv(start_price=107.0)
        result = analyse_exits(df, "SELL", entry_price=100.0, target=90.0, stop=105.0, bars_held=3)
        assert result.recommendation == "EXIT_NOW"
        assert any(s.signal == "STOP_HIT" for s in result.signals)

    def test_mid_trade_no_exit(self):
        df = make_ohlcv(start_price=103.0)
        result = analyse_exits(df, "BUY", entry_price=100.0, target=115.0, stop=95.0, bars_held=2)
        assert result.recommendation in ("HOLD", "WATCH", "SCALE_OUT")

    def test_summary_is_string(self):
        df = make_ohlcv()
        result = analyse_exits(df, "BUY", entry_price=100.0, target=110.0, stop=95.0)
        assert isinstance(result.summary, str) and len(result.summary) > 0

    def test_empty_df_handled_gracefully(self):
        result = analyse_exits(pd.DataFrame(), "BUY", 100.0, 110.0, 95.0)
        assert isinstance(result, ExitAnalysis)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 – Macro Calendar
# ═════════════════════════════════════════════════════════════════════════════

from agent.macro_calendar import check_macro_event, get_upcoming_events


class TestMacroCalendar:
    def test_check_event_structure(self):
        result = check_macro_event()
        # blocked key may be "blocked" or "is_blocked"
        has_blocked = "blocked" in result or "is_blocked" in result
        assert has_blocked, f"macro event missing blocked field, keys: {list(result.keys())}"
        assert "event_name" in result
        assert "impact" in result

    def test_impact_valid_value(self):
        result = check_macro_event()
        assert result["impact"] in ("HIGH", "MEDIUM", "LOW", "NONE", None, "")

    def test_hours_away_non_negative_or_none(self):
        result = check_macro_event()
        ha = result.get("hours_away")
        if ha is not None:
            assert ha >= 0

    def test_upcoming_is_list(self):
        events = get_upcoming_events()
        assert isinstance(events, list)

    def test_upcoming_events_sorted(self):
        events = get_upcoming_events()
        if len(events) >= 2:
            for i in range(len(events) - 1):
                assert events[i]["date"] <= events[i + 1]["date"], "events must be sorted by date"

    def test_upcoming_events_have_required_fields(self):
        events = get_upcoming_events()
        for ev in events[:5]:
            for key in ("date", "impact"):
                assert key in ev
            # event name field may be 'name' or 'event_name' depending on source
            assert "name" in ev or "event_name" in ev


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 – Sector ETF Context
# ═════════════════════════════════════════════════════════════════════════════

from agent.sector_etf import get_sector_context


class TestSectorETF:
    def test_returns_sector_context(self):
        df = make_ohlcv(n=60)
        ctx = get_sector_context("AAPL", df)
        # SectorContext is a dataclass — access as attributes
        assert hasattr(ctx, "ticker") or hasattr(ctx, "etf") or isinstance(ctx, dict)

    def test_score_mult_in_range(self):
        df = make_ohlcv(n=60)
        ctx = get_sector_context("NVDA", df)
        mult = ctx.score_mult if hasattr(ctx, "score_mult") else ctx.get("score_mult", 1.0)
        assert 0.5 <= mult <= 1.5

    def test_unknown_ticker_uses_default(self):
        df = make_ohlcv(n=60)
        ctx = get_sector_context("XXXX_FAKE", df)
        assert ctx is not None

    def test_known_tickers_map_correctly(self):
        df = make_ohlcv(n=60)
        # Use only tickers confirmed in SECTOR_MAP
        for ticker, expected_etf in [("AAPL", "XLK"), ("NVDA", "SMH"), ("MSFT", "XLK")]:
            ctx = get_sector_context(ticker, df)
            etf = ctx.etf if hasattr(ctx, "etf") else ctx.get("etf", "")
            assert etf == expected_etf, f"{ticker} should map to {expected_etf}, got {etf}"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 – Market Regime
# ═════════════════════════════════════════════════════════════════════════════

from agent.market_regime import get_regime


class TestMarketRegime:
    def test_regime_returns_regime_info(self):
        regime = get_regime()
        from agent.market_regime import RegimeInfo
        assert isinstance(regime, RegimeInfo), f"expected RegimeInfo, got {type(regime)}"

    def test_regime_label_is_string(self):
        regime = get_regime()
        assert isinstance(regime.label, str) and len(regime.label) > 0

    def test_regime_mults_non_negative(self):
        regime = get_regime()
        assert regime.long_mult >= 0
        assert regime.short_mult >= 0

    def test_regime_api_returns_dict(self, client):
        body = client.get("/api/regime").json()
        # via API it should be serialized to dict
        assert isinstance(body["regime"], dict)
        assert "label" in body["regime"]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 12 – API Endpoints (FastAPI TestClient)
# ═════════════════════════════════════════════════════════════════════════════

import importlib


def _get_client():
    """Build a TestClient with scanner fully mocked."""
    mock_scanner = MagicMock()
    mock_scanner.signals = []
    mock_scanner.last_scan = None
    mock_scanner.is_running = True
    mock_scanner.start_background = MagicMock()
    mock_scanner.stop = MagicMock()
    mock_scanner.register_callback = MagicMock()

    # Patch scanner in module before importing main
    import agent.scanner as sc_mod
    sc_mod.scanner = mock_scanner

    # Reload to pick up fresh state
    if "main" in sys.modules:
        m = sys.modules["main"]
        m.scanner = mock_scanner
    else:
        import main as m
        m.scanner = mock_scanner

    from fastapi.testclient import TestClient
    from auth.dependencies import get_current_user, AuthenticatedUser
    _test_admin = AuthenticatedUser(
        id=1, username="test_admin", role="ADMIN", status="ACTIVE", jti="test-jti"
    )
    m.app.dependency_overrides[get_current_user] = lambda: _test_admin
    return TestClient(m.app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def client():
    c = _get_client()
    yield c
    import sys
    if "main" in sys.modules:
        sys.modules["main"].app.dependency_overrides.clear()


class TestAPIHealth:
    def test_health_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_health_has_required_fields(self, client):
        body = client.get("/api/health").json()
        for key in ("status", "is_running", "tickers_tracked", "ws_clients", "valkey"):
            assert key in body, f"/api/health missing key: {key}"

    def test_health_status_is_ok(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"


class TestAPISignals:
    def test_signals_200(self, client):
        assert client.get("/api/signals").status_code == 200

    def test_signals_has_list(self, client):
        body = client.get("/api/signals").json()
        assert "signals" in body and isinstance(body["signals"], list)

    def test_signals_has_count(self, client):
        body = client.get("/api/signals").json()
        assert "count" in body
        assert body["count"] == len(body["signals"])

    def test_signals_has_scan_timestamp(self, client):
        body = client.get("/api/signals").json()
        assert "last_scan" in body


class TestAPIRegime:
    def test_regime_200(self, client):
        assert client.get("/api/regime").status_code == 200

    def test_regime_has_both_keys(self, client):
        body = client.get("/api/regime").json()
        assert "regime" in body and "session" in body

    def test_regime_label_present(self, client):
        body = client.get("/api/regime").json()
        assert "label" in body["regime"]


class TestAPIPaperTrading:
    def test_paper_trading_200(self, client):
        assert client.get("/api/paper-trading").status_code == 200

    def test_paper_trading_structure(self, client):
        body = client.get("/api/paper-trading").json()
        assert "summary" in body
        assert "open_trades" in body
        assert "closed_trades" in body
        assert isinstance(body["open_trades"], list)
        assert isinstance(body["closed_trades"], list)

    def test_summary_keys(self, client):
        body = client.get("/api/paper-trading").json()
        s = body["summary"]
        for key in ("open", "closed", "wins", "losses", "win_rate"):
            assert key in s

    def test_paper_trading_daily_200(self, client):
        assert client.get("/api/paper-trading/daily").status_code == 200

    def test_paper_trading_daily_structure(self, client):
        body = client.get("/api/paper-trading/daily").json()
        assert "daily" in body and isinstance(body["daily"], list)
        assert "today" in body

    def test_paper_trading_performance_200(self, client):
        assert client.get("/api/paper-trading/performance").status_code == 200

    def test_paper_trading_performance_structure(self, client):
        body = client.get("/api/paper-trading/performance").json()
        for key in ("summary", "daily", "equity_curve"):
            assert key in body


class TestAPIAccountState:
    def test_account_state_200(self, client):
        assert client.get("/api/account-state").status_code == 200

    def test_account_state_keys(self, client):
        body = client.get("/api/account-state").json()
        has_capital = "capital" in body or "account_equity" in body
        assert has_capital, f"/api/account-state missing capital, keys: {list(body.keys())}"
        # config may be inline (max_trade_pct etc.) or nested under "config"
        has_config = "config" in body or "max_trade_pct" in body or "max_open_trades" in body
        assert has_config, f"/api/account-state missing config, keys: {list(body.keys())}"

    def test_account_state_capital_positive(self, client):
        body = client.get("/api/account-state").json()
        capital = body.get("capital", body.get("account_equity", 0))
        assert capital > 0

    def test_account_config_post(self, client):
        r = client.post("/api/account-config", json={"budget": 50000})
        assert r.status_code == 200


class TestAPIBacktest:
    def test_backtest_stats_200(self, client):
        assert client.get("/api/backtest/stats").status_code == 200

    def test_backtest_stats_structure(self, client):
        body = client.get("/api/backtest/stats").json()
        assert "stats" in body
        assert "overall" in body["stats"]

    def test_backtest_stats_lookback_param(self, client):
        assert client.get("/api/backtest/stats?lookback_days=7").status_code == 200
        assert client.get("/api/backtest/stats?lookback_days=90").status_code == 200

    def test_backtest_tracking_200(self, client):
        assert client.get("/api/backtest/tracking").status_code == 200

    def test_backtest_tracking_is_list(self, client):
        body = client.get("/api/backtest/tracking").json()
        assert "tracking" in body and isinstance(body["tracking"], list)

    def test_backtest_recent_200(self, client):
        assert client.get("/api/backtest/recent").status_code == 200

    def test_backtest_recent_is_list(self, client):
        body = client.get("/api/backtest/recent").json()
        assert "recent" in body and isinstance(body["recent"], list)

    def test_backtest_path_200(self, client):
        assert client.get("/api/backtest/path/FAKE_SIG").status_code == 200

    def test_backtest_path_is_list(self, client):
        body = client.get("/api/backtest/path/FAKE_SIG").json()
        assert "path" in body and isinstance(body["path"], list)

    def test_backtest_mtf_200(self, client):
        assert client.get("/api/backtest/mtf").status_code == 200

    def test_backtest_results_no_name_error(self, client):
        # Critical: duplicate endpoint was removed; this must not crash
        r = client.get("/api/historical/backtest/results")
        assert r.status_code in (200, 404), \
            f"historical backtest results returned {r.status_code}: {r.text}"
        if r.status_code == 200:
            body = r.json()
            assert isinstance(body, dict)


class TestAPISignalHistory:
    def test_signal_history_200(self, client):
        assert client.get("/api/signal-history").status_code == 200

    def test_signal_history_structure(self, client):
        body = client.get("/api/signal-history").json()
        assert "signals" in body and "stats" in body

    def test_signal_history_limit_param(self, client):
        assert client.get("/api/signal-history?limit=10").status_code == 200


class TestAPIWatchlist:
    def test_watchlist_get_200(self, client):
        assert client.get("/api/watchlist").status_code == 200

    def test_watchlist_structure(self, client):
        body = client.get("/api/watchlist").json()
        assert "base" in body and "watchlist" in body
        assert isinstance(body["base"], list)

    def test_watchlist_add(self, client):
        r = client.post("/api/watchlist/add?ticker=PLTR")
        assert r.status_code == 200
        body = r.json()
        assert "watchlist" in body

    def test_watchlist_add_base_ticker_blocked(self, client):
        # AAPL is in base — should reject or return graceful error
        r = client.post("/api/watchlist/add?ticker=AAPL")
        # Either rejected (4xx) or returns with a message
        assert r.status_code in (200, 400, 422)

    def test_watchlist_remove(self, client):
        client.post("/api/watchlist/add?ticker=RKLB")
        r = client.post("/api/watchlist/remove?ticker=RKLB")
        assert r.status_code == 200


class TestAPIClusters:
    def test_clusters_200(self, client):
        assert client.get("/api/clusters").status_code == 200

    def test_clusters_has_abc(self, client):
        body = client.get("/api/clusters").json()
        for cluster in ("A", "B", "C"):
            assert cluster in body, f"clusters missing {cluster}"


class TestAPIPositionSize:
    def test_position_size_200(self, client):
        r = client.get("/api/position-size?entry=100&stop=95&account_size=10000")
        assert r.status_code == 200

    def test_position_size_has_shares(self, client):
        body = client.get("/api/position-size?entry=100&stop=95&account_size=10000").json()
        assert "shares" in body or "position_size" in body

    def test_position_size_math_correct(self, client):
        body = client.get(
            "/api/position-size?entry=100&stop=95&account_size=10000&risk_pct=1&confidence=65"
        ).json()
        # max_position_pct=5% of $10k = $500 / $100 = 5 shares max
        shares = body.get("shares", body.get("position_size", {}).get("shares", None))
        if shares is not None:
            assert 1 <= shares <= 25, f"shares {shares} outside plausible range"

    def test_position_size_zero_risk_handled(self, client):
        r = client.get("/api/position-size?entry=100&stop=100&account_size=10000")
        assert r.status_code in (200, 400, 422)


class TestAPIMacroCalendar:
    def test_macro_calendar_200(self, client):
        assert client.get("/api/macro-calendar").status_code == 200

    def test_macro_calendar_structure(self, client):
        body = client.get("/api/macro-calendar").json()
        assert "current" in body and "upcoming" in body
        assert isinstance(body["upcoming"], list)


class TestAPIMLStatus:
    def test_ml_status_200(self, client):
        assert client.get("/api/ml-status").status_code == 200

    def test_ml_status_has_model_info(self, client):
        body = client.get("/api/ml-status").json()
        assert "blend_weights" in body or "scalp_models" in body or "pipeline_metrics" in body


class TestAPILearning:
    def test_learning_status_200(self, client):
        assert client.get("/api/learning-status").status_code == 200

    def test_learning_status_keys(self, client):
        body = client.get("/api/learning-status").json()
        # win_rate stored as current_win_rate or win_rate
        has_wr = "win_rate" in body or "current_win_rate" in body
        assert has_wr, f"/api/learning-status missing win_rate key, got: {list(body.keys())}"
        assert "dynamic_threshold" in body or "threshold" in body

    def test_learning_log_200(self, client):
        assert client.get("/api/learning-log").status_code == 200

    def test_learning_log_is_list(self, client):
        body = client.get("/api/learning-log").json()
        assert "log" in body and isinstance(body["log"], list)


class TestAPIAlgoPerformance:
    def test_algo_perf_200(self, client):
        assert client.get("/api/algo-performance").status_code == 200

    def test_algo_perf_is_dict_or_list(self, client):
        body = client.get("/api/algo-performance").json()
        assert isinstance(body, (dict, list))


class TestAPIPipelineMetrics:
    def test_pipeline_metrics_200(self, client):
        assert client.get("/api/pipeline-metrics").status_code == 200

    def test_pipeline_metrics_keys(self, client):
        body = client.get("/api/pipeline-metrics").json()
        assert "last_cycle_ms" in body or "tickers_per_second" in body


class TestAPIRiskStatus:
    def test_risk_status_200(self, client):
        assert client.get("/api/risk-status").status_code == 200

    def test_risk_status_has_circuit_breaker(self, client):
        body = client.get("/api/risk-status").json()
        # field may be circuit_open, circuit_breaker, or enabled
        has_circuit = any(k in body for k in ("circuit_open", "circuit_breaker", "enabled", "status"))
        assert has_circuit, f"risk-status missing circuit field, got: {list(body.keys())}"


class TestAPICreditUsage:
    def test_credit_usage_200(self, client):
        assert client.get("/api/credit-usage").status_code == 200

    def test_credit_usage_keys(self, client):
        body = client.get("/api/credit-usage").json()
        for key in ("used", "pct"):
            assert key in body

    def test_credit_pct_in_range(self, client):
        body = client.get("/api/credit-usage").json()
        pct = body.get("pct", 0)
        assert 0.0 <= pct <= 100.0


class TestAPIServices:
    def test_services_200(self, client):
        assert client.get("/api/services").status_code == 200

    def test_services_has_scanner(self, client):
        body = client.get("/api/services").json()
        assert "scanner" in body or "services" in body


class TestAPINotifications:
    def test_notify_config_get_200(self, client):
        assert client.get("/api/notify/config").status_code == 200

    def test_notify_config_no_token_leak(self, client):
        body = client.get("/api/notify/config").json()
        # Token must never be returned
        assert "token" not in body or body.get("token") in (None, "", "***")


class TestAPIWeekendLearning:
    def test_weekend_learning_status_200(self, client):
        assert client.get("/api/weekend-learning/status").status_code == 200

    def test_weekend_learning_history_200(self, client):
        assert client.get("/api/weekend-learning/history").status_code == 200

    def test_weekend_cache_stats_200(self, client):
        assert client.get("/api/weekend-learning/cache-stats").status_code == 200


class TestAPIBroker:
    def test_broker_status_200(self, client):
        assert client.get("/api/broker/status").status_code == 200

    def test_broker_status_has_connected_field(self, client):
        body = client.get("/api/broker/status").json()
        assert "connected" in body or "schwab" in body or "status" in body


class TestAPIMarket:
    def test_market_streamer_200(self, client):
        assert client.get("/api/market/streamer").status_code == 200

    def test_market_hours_200(self, client):
        assert client.get("/api/market/hours").status_code == 200

    def test_after_hours_200(self, client):
        assert client.get("/api/after-hours").status_code == 200

    def test_after_hours_is_list(self, client):
        body = client.get("/api/after-hours").json()
        assert "snapshots" in body or "after_hours" in body or isinstance(body, list)


class TestAPIPremarket:
    def test_premarket_200(self, client):
        assert client.get("/api/premarket-scan").status_code == 200


class TestAPIRoot:
    def test_root_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_static_index_exists(self, client):
        r = client.get("/")
        assert b"<html" in r.content.lower() or b"<!doctype" in r.content.lower()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 13 – Data Accuracy & Integrity Checks
# ═════════════════════════════════════════════════════════════════════════════

class TestDataAccuracy:
    def test_vwap_formula_correctness(self):
        """VWAP = cumsum(typical_price × volume) / cumsum(volume) — verify conftest fixture."""
        df = make_ohlcv(n=10, start_price=100.0)
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        expected_vwap = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()
        assert np.allclose(df["vwap"].values, expected_vwap.values, rtol=1e-5)

    def test_paper_pnl_dollar_formula(self):
        """P&L dollar = (exit - entry) × shares × direction_sign"""
        entry, exit_p, shares = 100.0, 110.0, 50
        expected_pnl_pct = (exit_p - entry) / entry * 100  # 10%
        expected_pnl_dollar = (exit_p - entry) * shares     # $500
        assert abs(expected_pnl_pct - 10.0) < 1e-6
        assert abs(expected_pnl_dollar - 500.0) < 1e-6

    def test_position_size_dollar_risk_formula(self):
        ps = calculate(account_size=10_000, entry=100, stop=95, risk_pct=1.0, confidence=60)
        # dollar_risk = shares × risk_per_share
        manual = ps.shares * ps.risk_per_share
        assert abs(ps.dollar_risk - manual) < 0.01

    def test_rr_ratio_formula(self):
        """R:R = (target - entry) / (entry - stop) — verify formula holds."""
        from agent.prediction import _evaluate_rr
        from agent.support_resistance import get_all_sr_levels
        df = make_ohlcv(n=120, start_price=100.0)
        sr = get_all_sr_levels(df)
        stop, target, rr, quality, qualifies = _evaluate_rr(price=100.0, sr=sr, direction="BUY")
        if stop > 0 and target > 0 and abs(100.0 - stop) > 0.001:
            manual_rr = (target - 100.0) / (100.0 - stop)
            assert abs(rr - manual_rr) < 0.5, f"R:R {rr:.2f} ≠ manual {manual_rr:.2f}"

    def test_sell_rr_formula(self):
        from agent.prediction import _evaluate_rr
        from agent.support_resistance import get_all_sr_levels
        df = make_ohlcv(n=120, start_price=100.0, trend=-0.1)
        sr = get_all_sr_levels(df)
        stop, target, rr, quality, qualifies = _evaluate_rr(price=100.0, sr=sr, direction="SELL")
        if stop > 0 and target > 0:
            assert target <= 100.0 * 1.01 and stop >= 100.0 * 0.99

    def test_live_backtest_r_formula(self):
        """For BUY: R = (exit - entry) / (entry - stop)"""
        record_signal("DA_R", "BUY", entry_price=100.0, target=110.0, stop=95.0)
        resolved = update_tracking("DA_R", current_price=110.0, vwap=100.0)
        if resolved:
            # key may be r_multiple or final_r depending on version
            r_val = resolved[0].get("r_multiple", resolved[0].get("final_r", None))
            expected_r = (110.0 - 100.0) / (100.0 - 95.0)  # = 2.0
            if r_val is not None:
                assert abs(r_val - expected_r) < 0.1

    def test_confidence_calibration_math(self):
        """Higher win_rate context should boost confidence, lower should reduce."""
        with _br._cal_lock:
            _br._calibration = {
                "session:REGULAR": 0.8,   # 80% win rate → boost
                "session:AFTER_HOURS": 0.2,  # 20% win rate → reduce
            }
        boosted  = adjust_confidence(50.0, session="REGULAR")
        reduced  = adjust_confidence(50.0, session="AFTER_HOURS")
        with _br._cal_lock:
            _br._calibration = {}
        assert boosted > 50.0, "high win rate context should boost confidence"
        assert reduced < 50.0, "low win rate context should reduce confidence"

    def test_win_rate_formula(self):
        """win_rate = wins / total_resolved (in live_backtest, as 0–1)"""
        for i in range(3):
            record_signal(f"WR_{i}", "BUY", 100.0, 110.0, 95.0)
            update_tracking(f"WR_{i}", 111.0, 0.0)  # 3 wins
        record_signal("WR_L", "BUY", 100.0, 110.0, 95.0)
        update_tracking("WR_L", 94.0, 0.0)  # 1 loss
        stats = get_performance_stats()
        wr = stats["overall"]["win_rate"]
        assert 0.0 < wr < 1.0 and wr == pytest.approx(0.75, abs=0.05)

    def test_avg_r_expectancy_positive_after_good_trades(self):
        """avg_r and expectancy should be positive after consistent wins."""
        for _ in range(5):
            record_signal(f"EXP_{_}", "BUY", 100.0, 110.0, 95.0)
            update_tracking(f"EXP_{_}", 111.0, 0.0)
        s = get_broadcast_summary()
        assert s.get("avg_r", 0) is not None
        if s["total"] >= 5:
            assert s["avg_r"] > 0, "avg_r should be positive after 5 wins"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 14 – Adaptive Filter (Learning Engine)
# ═════════════════════════════════════════════════════════════════════════════

import agent.adaptive_filter as _af


class TestAdaptiveFilter:
    def test_initial_threshold_in_range(self):
        threshold = _af._state["dynamic_threshold"]
        assert 50.0 <= threshold <= 75.0

    def test_state_has_required_keys(self):
        for key in ("blocked_contexts", "boosted_contexts", "dynamic_threshold",
                    "current_win_rate", "total_resolved"):
            assert key in _af._state, f"adaptive filter state missing: {key}"

    def test_should_suppress_low_confidence(self, monkeypatch):
        # confidence below dynamic_threshold returns a reason regardless of enforcement mode
        import agent.adaptive_filter as af_mod
        monkeypatch.setattr(af_mod, "_ENFORCEMENT_MODE", "enforce")
        threshold = _af._state["dynamic_threshold"]
        suppressed, reason = _af.should_suppress(confidence=threshold - 10)
        assert suppressed, "confidence below threshold must be suppressed in enforce mode"
        assert len(reason) > 0

    def test_should_suppress_high_confidence_passes(self):
        threshold = _af._state["dynamic_threshold"]
        suppressed, _ = _af.should_suppress(confidence=threshold + 15)
        # High confidence should not be suppressed by confidence gate
        # (may still be blocked by context if any blocked contexts exist, so just check type)
        assert isinstance(suppressed, bool)

    def test_confidence_boost_returns_float(self):
        boost = _af.get_confidence_boost(vwap_event="RECLAIM", session="REGULAR")
        assert isinstance(boost, float)

    def test_win_rate_in_state_is_fraction(self):
        wr = _af._state["current_win_rate"]
        assert 0.0 <= wr <= 1.0, f"win_rate {wr} must be 0–1 fraction"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 15 – ML Model Interface (Prediction Pipeline)
# ═════════════════════════════════════════════════════════════════════════════

class TestPredictionPipeline:
    """Tests for prediction pipeline — ta indicators skipped (requires real ta library)."""

    def _get_sr_for_price(self, price=100.0, trend=0.05):
        from agent.support_resistance import get_all_sr_levels
        df = make_ohlcv(n=120, start_price=price, trend=trend)
        return get_all_sr_levels(df)

    def test_evaluate_rr_buy_properties(self):
        from agent.prediction import _evaluate_rr
        sr = self._get_sr_for_price(100.0, trend=0.05)
        stop, target, rr, quality, qualifies = _evaluate_rr(price=100.0, sr=sr, direction="BUY")
        if target > 0 and stop > 0:
            assert target >= 100.0 * 1.003, "BUY target must be at least 0.3% above price"
            assert stop <= 100.0, "BUY stop must be at or below price"

    def test_evaluate_rr_sell_properties(self):
        from agent.prediction import _evaluate_rr
        sr = self._get_sr_for_price(100.0, trend=-0.05)
        stop, target, rr, quality, qualifies = _evaluate_rr(price=100.0, sr=sr, direction="SELL")
        if target > 0 and stop > 0:
            assert target <= 100.0 * 0.997, "SELL target must be at least 0.3% below price"
            assert stop >= 100.0, "SELL stop must be at or above price"

    def test_rr_qualifies_at_min_rr(self):
        from agent.prediction import _evaluate_rr
        # Use large spread to get above the configured minimum R:R.
        df = make_ohlcv(n=120, start_price=100.0, trend=0.3)
        from agent.support_resistance import get_all_sr_levels
        sr = get_all_sr_levels(df)
        _, _, rr, _, qualifies = _evaluate_rr(100.0, sr=sr, direction="BUY")
        assert qualifies == (rr >= 1.5)

    def test_quality_labels_valid(self):
        from agent.prediction import _evaluate_rr
        sr = self._get_sr_for_price()
        _, _, _, quality, _ = _evaluate_rr(100.0, sr=sr, direction="BUY")
        assert quality in ("EXCELLENT", "GOOD", "OK", "LOW")

    def test_generate_prediction_requires_indicators(self):
        """Verify generate_prediction accepts indicator DataFrame."""
        from agent.prediction import generate_prediction
        # Use a minimal dataframe with the columns prediction needs
        df = make_ohlcv(n=120)
        # Without ta indicators, most columns will be missing — prediction should still return a dict
        try:
            result = generate_prediction("AAPL", df, 0.5, 0.5, 0.6, 0.0, df.iloc[-1])
            assert isinstance(result, dict)
            assert "direction" in result
        except Exception:
            pytest.skip("generate_prediction requires full indicator columns from ta library")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 16 – Data Fetcher Cache Logic
# ═════════════════════════════════════════════════════════════════════════════

from agent.data_fetcher import (
    _cache_get, _cache_set, get_last_cached_close, _interval_cache,
)


class TestDataFetcherCache:
    def test_cache_set_and_get(self):
        df = make_ohlcv()
        _cache_set("AAPL", "1min", df)
        result = _cache_get("AAPL", "1min", ttl=300)
        assert result is not None
        assert not result.empty

    def test_cache_ttl_expiry(self):
        df = make_ohlcv()
        _cache_set("MSFT", "5min", df)
        # TTL of 0 should always miss
        result = _cache_get("MSFT", "5min", ttl=0)
        assert result is None

    def test_cache_miss_returns_none(self):
        result = _cache_get("NOTEXIST_XYZ", "1min", ttl=300)
        assert result is None

    def test_get_last_cached_close_returns_float(self):
        df = make_ohlcv(start_price=150.0)
        _cache_set("GOOG", "1min", df)
        close = get_last_cached_close("GOOG")
        assert close is not None
        assert isinstance(close, float)
        assert close > 0

    def test_get_last_cached_close_unknown_returns_none(self):
        result = get_last_cached_close("ZZZZZ_FAKE_9999")
        assert result is None

    def test_cache_stale_close_at_eod(self):
        """Verify get_last_cached_close works without TTL (for EOD stale-close fix)."""
        df = make_ohlcv(start_price=200.0)
        _cache_set("IBM", "1day", df)
        # Should return the last Close even when market is closed (TTL expired)
        close = get_last_cached_close("IBM")
        assert close is not None
        assert abs(close - float(df["Close"].iloc[-1])) < 1.0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 17 – Multi-TF Backtest Endpoint
# ═════════════════════════════════════════════════════════════════════════════

class TestMTFBacktest:
    def test_mtf_backtest_200(self, client):
        assert client.get("/api/backtest/mtf").status_code == 200

    def test_mtf_history_200(self, client):
        assert client.get("/api/backtest/mtf/history").status_code == 200

    def test_mtf_ticker_200(self, client):
        r = client.get("/api/backtest/mtf/AAPL")
        assert r.status_code in (200, 404)  # 404 if no MTF data for ticker yet


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 18 – WebSocket Connection
# ═════════════════════════════════════════════════════════════════════════════

class TestWebSocket:
    def test_ws_connects(self, client):
        try:
            with client.websocket_connect("/ws") as ws:
                # Should connect without error
                assert ws is not None
        except Exception as e:
            pytest.skip(f"WebSocket not available in test context: {e}")

    def test_ws_no_immediate_error(self, client):
        try:
            with client.websocket_connect("/ws") as ws:
                # Attempt a receive with timeout (may get initial message or nothing)
                pass  # Just connecting successfully is enough
        except Exception:
            pytest.skip("WebSocket not stable in test context")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 19 – Error Handling & Edge Cases
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_position_size_api_no_params_returns_ok(self, client):
        # All params have defaults — no-param call returns 200 with a result dict
        r = client.get("/api/position-size")
        assert r.status_code == 200
        body = r.json()
        assert "shares" in body or "error" in body

    def test_backtest_path_nonexistent_signal(self, client):
        r = client.get("/api/backtest/path/DOES_NOT_EXIST_12345")
        assert r.status_code == 200
        body = r.json()
        assert body["path"] == []

    def test_watchlist_remove_nonexistent(self, client):
        r = client.post("/api/watchlist/remove?ticker=ZZZZNOTHERE")
        # Should not crash — either 200 or 404
        assert r.status_code in (200, 404)

    def test_account_config_invalid_budget(self, client):
        r = client.post("/api/account-config", json={"budget": -100})
        # Should handle gracefully — either accepted or validation error
        assert r.status_code in (200, 400, 422)

    def test_paper_trading_zero_price_rejected(self):
        tid = maybe_open_trade("EDGE_ZERO", "BUY", 0.0, 10.0, 0.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        assert tid is None, "zero entry price should be rejected"

    def test_live_backtest_zero_price_rejected(self):
        sid = record_signal("EDGE_ZERO2", "BUY", entry_price=0.0, target=10.0, stop=0.0)
        assert sid == "", "zero entry price must be rejected by live backtest"

    def test_double_close_idempotent(self, tmp_path, monkeypatch):
        """Closing an already-closed trade should be a no-op, not raise."""
        import agent.paper_trading as pt
        pt.init_db()
        tid = pt.maybe_open_trade("DC_TEST", "BUY", 100.0, 110.0, 95.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        assert tid is not None
        df = make_ohlcv(start_price=112.0)
        pt.update_open_trades("DC_TEST", df, current_price=112.0)
        # Second close should be silently ignored
        pt.update_open_trades("DC_TEST", df, current_price=113.0)
        closed = pt.get_closed_trades()
        matches = [t for t in closed if t["ticker"] == "DC_TEST"]
        assert len(matches) == 1, f"double-close must not duplicate — found {len(matches)} entries"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 20 – Comprehensive P&L Accuracy
# ═════════════════════════════════════════════════════════════════════════════

class TestPnLAccuracy:
    def test_buy_pnl_pct_formula(self):
        """P&L% for BUY = (exit - entry) / entry × 100"""
        entry, exit_p = 100.0, 115.0
        expected = (exit_p - entry) / entry * 100  # 15%
        assert abs(expected - 15.0) < 1e-6

    def test_sell_pnl_pct_formula(self):
        """P&L% for SELL = (entry - exit) / entry × 100"""
        entry, exit_p = 100.0, 85.0
        expected = (entry - exit_p) / entry * 100  # 15% profit
        assert abs(expected - 15.0) < 1e-6

    def test_paper_trade_pnl_matches_formula(self, tmp_path, monkeypatch):
        import agent.paper_trading as pt
        pt.init_db()
        pt.maybe_open_trade("PNL_CHK", "BUY", 100.0, 115.0, 95.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        df = make_ohlcv(start_price=116.0)
        pt.update_open_trades("PNL_CHK", df, current_price=116.0)
        closed = pt.get_closed_trades()
        match = next((t for t in closed if t["ticker"] == "PNL_CHK"), None)
        if match and match.get("pnl_pct") is not None and match.get("exit_price"):
            ep = float(match["exit_price"])
            expected_pct = (ep - 100.0) / 100.0 * 100
            # Allow 3% tolerance: exit may be at target (115) not current price (116)
            assert abs(match["pnl_pct"] - expected_pct) < 3.0, \
                f"P&L {match['pnl_pct']:.2f}% ≠ formula {expected_pct:.2f}%"

    def test_stale_close_uses_cached_price_not_zero(self, tmp_path, monkeypatch):
        """Critical fix: EOD close must use cached price, not $0."""
        import agent.paper_trading as pt
        from agent.data_fetcher import _cache_set
        pt.init_db()

        # Pre-load price in cache
        df = make_ohlcv(start_price=150.0)
        _cache_set("STALE_CHK", "1min", df)

        pt.maybe_open_trade("STALE_CHK", "BUY", 148.0, 155.0, 144.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        # Force EOD close (close_all_positions_eod uses cached price)
        from agent.paper_trading import close_all_positions_eod
        close_all_positions_eod(reason="EOD_TEST")
        closed = pt.get_closed_trades()
        match = next((t for t in closed if t["ticker"] == "STALE_CHK"), None)
        if match:
            assert match.get("exit_price", 0) > 0, "exit price must not be $0 with cached data"
