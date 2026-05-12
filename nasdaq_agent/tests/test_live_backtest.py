"""
Tests for agent/live_backtest.py — the live backtesting engine.
"""
import time
import pytest
from agent.live_backtest import (
    record_signal, update_tracking, get_tracking_signals,
    get_recent_resolved, get_performance_stats, _empty_stats, MAX_BARS,
)


# ── record_signal ─────────────────────────────────────────────────────────────

def test_record_buy_signal():
    sid = record_signal("AAPL", "BUY", entry_price=180.0, target=185.0, stop=178.0)
    assert sid != "", "should return a non-empty signal_id"
    assert "AAPL" in sid

def test_record_sell_signal():
    sid = record_signal("MSFT", "SELL", entry_price=300.0, target=295.0, stop=302.0)
    assert sid != ""

def test_record_neutral_ignored():
    sid = record_signal("NVDA", "NEUTRAL", entry_price=500.0, target=510.0, stop=495.0)
    assert sid == "", "NEUTRAL direction must not be recorded"

def test_record_invalid_prices_ignored():
    sid = record_signal("TSLA", "BUY", entry_price=0.0, target=100.0, stop=90.0)
    assert sid == "", "zero entry price must be rejected"

def test_deduplication_same_ticker_direction():
    record_signal("GOOGL", "BUY", entry_price=140.0, target=145.0, stop=138.0)
    sid2 = record_signal("GOOGL", "BUY", entry_price=141.0, target=146.0, stop=139.0)
    assert sid2 == "", "duplicate BUY for same ticker must be skipped"

def test_opposite_directions_both_recorded():
    s1 = record_signal("AMD", "BUY",  entry_price=160.0, target=165.0, stop=158.0)
    s2 = record_signal("AMD", "SELL", entry_price=160.0, target=155.0, stop=162.0)
    assert s1 != "" and s2 != "", "BUY and SELL for same ticker are independent"

def test_record_stores_context():
    sid = record_signal(
        "META", "BUY", 500.0, 510.0, 495.0,
        confidence=78.5, session="REGULAR", regime="BULL_TREND",
        vwap_event="RECLAIM", rsi_zone="NEUTRAL",
    )
    tracking = get_tracking_signals()
    match = next((t for t in tracking if t["signal_id"] == sid), None)
    assert match is not None
    assert match["confidence"] == 78.5
    assert match["session"] == "REGULAR"
    assert match["vwap_event"] == "RECLAIM"


# ── update_tracking / resolution ──────────────────────────────────────────────

def test_update_tracking_win_buy():
    record_signal("AMZN", "BUY", entry_price=180.0, target=185.0, stop=178.0)
    resolved = update_tracking("AMZN", current_price=186.0, vwap=181.0)
    assert len(resolved) == 1
    assert resolved[0]["status"] == "WIN"
    assert resolved[0]["exit_reason"] == "TARGET"

def test_update_tracking_loss_buy():
    record_signal("NFLX", "BUY", entry_price=600.0, target=615.0, stop=595.0)
    resolved = update_tracking("NFLX", current_price=594.0, vwap=601.0)
    assert resolved[0]["status"] == "LOSS"
    assert resolved[0]["exit_reason"] == "STOP"

def test_update_tracking_win_sell():
    record_signal("TSLA", "SELL", entry_price=250.0, target=240.0, stop=253.0)
    resolved = update_tracking("TSLA", current_price=239.0, vwap=249.0)
    assert resolved[0]["status"] == "WIN"

def test_update_tracking_loss_sell():
    record_signal("COIN", "SELL", entry_price=200.0, target=190.0, stop=204.0)
    resolved = update_tracking("COIN", current_price=205.0, vwap=199.0)
    assert resolved[0]["status"] == "LOSS"

def test_update_tracking_vwap_loss_buy():
    """BUY signal entered above VWAP — loses when price drops below VWAP.
    Condition: entry >= vwap AND current_price < vwap."""
    # entry=522.0 >= vwap=521.0, then price drops to 519.0 < vwap=521.0
    record_signal("NVDA", "BUY", entry_price=522.0, target=532.0, stop=517.0)
    resolved = update_tracking("NVDA", current_price=519.0, vwap=521.0)
    assert len(resolved) == 1
    assert resolved[0]["status"] == "LOSS"
    assert resolved[0]["exit_reason"] == "VWAP_LOSS"

def test_update_tracking_timeout():
    record_signal("HOOD", "BUY", entry_price=10.0, target=15.0, stop=9.0)
    resolved = []
    # Advance MAX_BARS without hitting target or stop
    for _ in range(MAX_BARS):
        resolved = update_tracking("HOOD", current_price=11.0, vwap=0.0)
    assert resolved[0]["status"] == "TIMEOUT"

def test_update_tracking_no_resolution_mid_bar():
    record_signal("ROKU", "BUY", entry_price=60.0, target=65.0, stop=58.0)
    resolved = update_tracking("ROKU", current_price=61.0, vwap=60.5)
    assert resolved == [], "mid-trade update should not resolve"

def test_bars_tracked_increments():
    record_signal("CRWD", "BUY", entry_price=300.0, target=310.0, stop=295.0)
    for _ in range(3):
        update_tracking("CRWD", current_price=302.0, vwap=0.0)
    tracking = get_tracking_signals()
    match = next((t for t in tracking if t["ticker"] == "CRWD"), None)
    assert match is not None
    assert match["bars_tracked"] == 3

def test_current_r_in_tracking():
    record_signal("SNOW", "BUY", entry_price=100.0, target=110.0, stop=95.0)
    update_tracking("SNOW", current_price=105.0, vwap=0.0)  # R = (105-100)/(100-95) = 1.0
    tracking = get_tracking_signals()
    match = next((t for t in tracking if t["ticker"] == "SNOW"), None)
    assert match is not None
    assert abs(match["current_r"] - 1.0) < 0.01


# ── get_performance_stats ─────────────────────────────────────────────────────

def test_performance_stats_empty():
    stats = get_performance_stats()
    assert stats["overall"]["total"] == 0
    assert stats["overall"]["win_rate"] == 0.0

def test_performance_stats_win_rate_is_ratio():
    """win_rate must be 0-1, not 0-100."""
    record_signal("AAPL", "BUY", 180.0, 185.0, 178.0)
    update_tracking("AAPL", 186.0, 0.0)  # WIN
    record_signal("MSFT", "BUY", 300.0, 310.0, 295.0)
    update_tracking("MSFT", 294.0, 0.0)  # LOSS
    stats = get_performance_stats()
    wr = stats["overall"]["win_rate"]
    assert 0.0 <= wr <= 1.0, f"win_rate should be 0-1, got {wr}"

def test_performance_stats_counts():
    record_signal("A1", "BUY",  100.0, 110.0, 95.0)
    update_tracking("A1", 111.0, 0.0)  # WIN
    record_signal("A2", "SELL", 200.0, 190.0, 205.0)
    update_tracking("A2", 206.0, 0.0)  # LOSS
    stats = get_performance_stats()
    o = stats["overall"]
    assert o["wins"] == 1
    assert o["losses"] == 1
    assert o["total"] == 2

def test_performance_stats_has_breakdowns():
    stats = get_performance_stats()
    for key in ["by_direction", "by_session", "by_regime",
                "by_vwap_event", "by_rsi_zone", "by_entry_type",
                "by_confidence", "by_sector_trend"]:
        assert key in stats, f"missing breakdown key: {key}"

def test_get_recent_resolved():
    record_signal("X1", "BUY", 50.0, 55.0, 48.0)
    update_tracking("X1", 56.0, 0.0)
    recent = get_recent_resolved(limit=10)
    assert len(recent) >= 1
    assert recent[0]["status"] in ("WIN", "LOSS", "TIMEOUT")
