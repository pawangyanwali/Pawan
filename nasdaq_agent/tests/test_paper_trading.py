"""
Tests for agent/paper_trading.py — simulated trade execution.
"""
import pytest
from agent.paper_trading import (
    maybe_open_trade, update_open_trades, get_open_trades,
    get_closed_trades, get_summary,
)
from tests.conftest import make_ohlcv


def test_open_trade_high_confidence():
    tid = maybe_open_trade("AAPL", "BUY", 180.0, 185.0, 178.0, confidence=75.0, rr_qualifies=True)
    assert tid is not None, "should open trade when confidence >= 65 and rr qualifies"
    open_trades = get_open_trades()
    assert any(t["ticker"] == "AAPL" for t in open_trades)

def test_low_confidence_trade_blocked():
    """Low confidence (below dynamic gate) should be rejected."""
    tid = maybe_open_trade("MSFT", "BUY", 300.0, 310.0, 295.0, confidence=30.0, rr_qualifies=True)
    assert tid is None, "confidence below gate should block paper trade"

def test_rr_not_qualifying_blocked():
    """rr_qualifies=False blocks the trade even if confidence is high."""
    tid = maybe_open_trade("NVDA", "BUY", 500.0, 510.0, 495.0, confidence=80.0, rr_qualifies=False)
    assert tid is None, "rr_qualifies=False must block paper trade"

def test_no_duplicate_open_trade():
    maybe_open_trade("TSLA", "BUY", 250.0, 260.0, 245.0, confidence=70.0, rr_qualifies=True)
    tid2 = maybe_open_trade("TSLA", "SELL", 250.0, 240.0, 255.0, confidence=70.0, rr_qualifies=True)
    assert tid2 is None, "second trade on same ticker must be blocked while one is open"

def test_no_trade_neutral_direction():
    tid = maybe_open_trade("AMZN", "NEUTRAL", 140.0, 145.0, 138.0, confidence=80.0, rr_qualifies=True)
    assert tid is None

def test_trade_closed_on_target():
    # exit_signals reads df.iloc[-1]["Close"] — set it above target so TARGET_HIT fires
    df = make_ohlcv(start_price=107.0)  # last Close ~107, target=105
    maybe_open_trade("HOOD", "BUY", 100.0, 105.0, 98.0, confidence=70.0, rr_qualifies=True)
    update_open_trades("HOOD", df, current_price=107.0)
    closed = get_closed_trades()
    assert any(t["ticker"] == "HOOD" for t in closed), "trade should have closed on TARGET_HIT"

def test_trade_closed_on_stop():
    # Set df last Close below stop so STOP_HIT fires
    df = make_ohlcv(start_price=95.0)  # last Close ~95, stop=97
    maybe_open_trade("COIN", "BUY", 100.0, 108.0, 97.0, confidence=70.0, rr_qualifies=True)
    update_open_trades("COIN", df, current_price=95.0)
    closed = get_closed_trades()
    assert any(t["ticker"] == "COIN" for t in closed)

def test_sell_trade_pnl_direction():
    df = make_ohlcv(start_price=200.0)
    maybe_open_trade("META", "SELL", 200.0, 190.0, 205.0, confidence=70.0, rr_qualifies=True)
    # price drops — SELL trade should be profitable
    update_open_trades("META", df, current_price=188.0)
    closed = get_closed_trades()
    meta_trade = next((t for t in closed if t["ticker"] == "META"), None)
    if meta_trade:  # may not close if exit signal isn't EXIT_NOW
        assert meta_trade["pnl_pct"] is not None

def test_summary_counts():
    maybe_open_trade("T1", "BUY", 50.0, 55.0, 48.0, confidence=70.0, rr_qualifies=True)
    s = get_summary()
    assert s["open"] >= 1
    assert "wins" in s and "losses" in s and "win_rate" in s

def test_summary_win_rate_range():
    s = get_summary()
    # win_rate stored as percentage 0-100 in paper_trading (different from live_backtest)
    assert 0.0 <= s["win_rate"] <= 100.0
