"""
Tests for agent/backtest_reporter.py — attribution, calibration, confidence adjustment.
"""
import pytest
from agent.backtest_reporter import (
    get_broadcast_summary, get_full_report,
    adjust_confidence, get_calibration,
    _update_confidence_calibration,
)
from agent.live_backtest import record_signal, update_tracking
import pandas as pd


def test_broadcast_summary_structure():
    s = get_broadcast_summary()
    for key in ("tracking", "total", "win_rate", "avg_r", "expectancy"):
        assert key in s, f"missing key: {key}"

def test_broadcast_summary_win_rate_ratio():
    record_signal("AAPL", "BUY", 180.0, 185.0, 178.0)
    update_tracking("AAPL", 186.0, 0.0)  # WIN
    s = get_broadcast_summary()
    assert 0.0 <= s["win_rate"] <= 1.0

def test_full_report_structure():
    r = get_full_report(lookback_days=30)
    assert "stats" in r
    assert "tracking" in r
    assert "recent" in r

def test_full_report_recent_has_outcome_color():
    record_signal("MSFT", "BUY", 300.0, 310.0, 295.0)
    update_tracking("MSFT", 311.0, 0.0)  # WIN
    r = get_full_report()
    for item in r["recent"]:
        assert "outcome_color" in item

def test_adjust_confidence_no_calibration():
    """With no calibration data, confidence should be returned unchanged."""
    import agent.backtest_reporter as br
    with br._cal_lock:
        br._calibration = {}   # ensure empty — other tests may have populated it
    result = adjust_confidence(60.0, vwap_event="RECLAIM")
    assert result == 60.0

def test_adjust_confidence_clamps_to_range():
    """Result must always stay in [25, 95]."""
    # Manually insert a calibration entry with extreme win rate
    import agent.backtest_reporter as br
    import threading
    with br._cal_lock:
        br._calibration = {"vwap_event:RECLAIM": 1.0}  # 100% win rate

    result = adjust_confidence(90.0, vwap_event="RECLAIM")
    assert result <= 95.0

    with br._cal_lock:
        br._calibration = {"vwap_event:REJECTION": 0.0}  # 0% win rate

    result = adjust_confidence(30.0, vwap_event="REJECTION")
    assert result >= 25.0

    # Reset
    with br._cal_lock:
        br._calibration = {}

def test_calibration_update_from_dataframe():
    df = pd.DataFrame({
        "vwap_event": ["RECLAIM"] * 10 + ["REJECTION"] * 10,
        "outcome":    [1] * 8 + [0] * 2 + [0] * 8 + [1] * 2,  # 80% vs 20%
    })
    _update_confidence_calibration(df)
    cal = get_calibration()
    assert "vwap_event:RECLAIM" in cal
    assert cal["vwap_event:RECLAIM"] > cal["vwap_event:REJECTION"]

def test_adjust_confidence_multiple_contexts():
    import agent.backtest_reporter as br
    with br._cal_lock:
        br._calibration = {
            "vwap_event:RECLAIM": 0.75,
            "session:REGULAR":    0.65,
        }
    result = adjust_confidence(50.0, vwap_event="RECLAIM", session="REGULAR")
    # Both contexts should boost confidence above 50
    assert result > 50.0
    with br._cal_lock:
        br._calibration = {}


def test_feedback_retrain_defers_deep_phase(monkeypatch):
    import agent.adaptive_filter as adaptive_filter
    import agent.backtest_reporter as br
    import agent.live_backtest as live_backtest
    import agent.ml_model as ml_model
    import config

    calls = []
    monkeypatch.setattr(config, "TRAINING_TICKERS", ["AAPL", "MSFT"])
    monkeypatch.setattr(br, "_log_attribution", lambda df: None)
    monkeypatch.setattr(br, "_update_confidence_calibration", lambda df: None)
    monkeypatch.setattr(adaptive_filter, "update_filter", lambda stats: None)
    monkeypatch.setattr(
        live_backtest,
        "get_performance_stats",
        lambda lookback_days=30: {"win_rate": 0.5},
    )
    monkeypatch.setattr(
        ml_model,
        "retrain_all",
        lambda tickers, **kwargs: calls.append((list(tickers), kwargs)),
    )

    outcomes = pd.DataFrame({"outcome": [1, 0, 1, 0]})

    br._run_feedback_retrain(outcomes, ["SHOULD_NOT_USE"])

    assert calls == [(["AAPL", "MSFT"], {"skip_deep": True})]
