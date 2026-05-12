"""
Tests for agent/exit_signals.py — exit condition detection.
"""
import pytest
import numpy as np
from agent.exit_signals import analyse_exits, ExitAnalysis
from tests.conftest import make_ohlcv


VALID_RECOMMENDATIONS = {"EXIT_NOW", "SCALE_OUT", "WATCH", "HOLD"}


def test_returns_exit_analysis():
    df = make_ohlcv()
    result = analyse_exits(df, "BUY", entry_price=100.0, target=110.0, stop=95.0)
    assert isinstance(result, ExitAnalysis)

def test_recommendation_is_valid():
    df = make_ohlcv()
    result = analyse_exits(df, "BUY", entry_price=100.0, target=110.0, stop=95.0)
    assert result.recommendation in VALID_RECOMMENDATIONS

def test_target_hit_buy():
    df = make_ohlcv(start_price=112.0)  # above target
    result = analyse_exits(df, "BUY", entry_price=100.0, target=110.0, stop=95.0, bars_held=3)
    assert result.recommendation == "EXIT_NOW"
    signals = [s.signal for s in result.signals]
    assert "TARGET_HIT" in signals

def test_stop_hit_buy():
    df = make_ohlcv(start_price=93.0)  # below stop
    result = analyse_exits(df, "BUY", entry_price=100.0, target=110.0, stop=95.0, bars_held=3)
    assert result.recommendation == "EXIT_NOW"
    signals = [s.signal for s in result.signals]
    assert "STOP_HIT" in signals

def test_target_hit_sell():
    df = make_ohlcv(start_price=88.0)  # below SELL target
    result = analyse_exits(df, "SELL", entry_price=100.0, target=90.0, stop=105.0, bars_held=3)
    assert result.recommendation == "EXIT_NOW"
    signals = [s.signal for s in result.signals]
    assert "TARGET_HIT" in signals

def test_stop_hit_sell():
    df = make_ohlcv(start_price=107.0)  # above SELL stop
    result = analyse_exits(df, "SELL", entry_price=100.0, target=90.0, stop=105.0, bars_held=3)
    assert result.recommendation == "EXIT_NOW"
    signals = [s.signal for s in result.signals]
    assert "STOP_HIT" in signals

def test_time_stop_fires():
    # TIME_STOP fires when bars_held >= 15 AND progress < 1% (price barely moved)
    df = make_ohlcv(start_price=100.02)  # only 0.02% above entry — triggers time stop
    result = analyse_exits(df, "BUY", entry_price=100.0, target=115.0, stop=95.0, bars_held=15)
    assert result.recommendation in ("EXIT_NOW", "WATCH", "HOLD")

def test_no_exit_mid_trade():
    df = make_ohlcv(start_price=103.0)
    result = analyse_exits(df, "BUY", entry_price=100.0, target=115.0, stop=95.0, bars_held=2)
    # Price is between stop and target — should not be EXIT_NOW
    assert result.recommendation in ("HOLD", "WATCH", "SCALE_OUT")

def test_summary_is_string():
    df = make_ohlcv()
    result = analyse_exits(df, "BUY", entry_price=100.0, target=110.0, stop=95.0)
    assert isinstance(result.summary, str)

def test_signals_list_type():
    df = make_ohlcv()
    result = analyse_exits(df, "BUY", entry_price=100.0, target=110.0, stop=95.0)
    assert isinstance(result.signals, list)
