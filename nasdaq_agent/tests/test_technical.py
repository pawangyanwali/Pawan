"""
Tests for agent/technical.py — indicator computation and scoring.
"""
import pytest
import numpy as np

pytest.importorskip("ta", reason="ta library not available in this environment")

from agent.technical import compute_indicators, score_technical
from tests.conftest import make_ohlcv


def test_compute_indicators_returns_dataframe():
    df = make_ohlcv()
    result = compute_indicators(df)
    assert hasattr(result, "iloc"), "should return a DataFrame"

def test_compute_indicators_expected_columns():
    df = make_ohlcv()
    result = compute_indicators(df)
    expected = ["rsi", "macd", "ema9", "ema21", "bb_upper", "bb_lower", "vwap"]
    for col in expected:
        assert col in result.columns, f"missing column: {col}"

def test_rsi_in_valid_range():
    df = make_ohlcv(n=60)
    result = compute_indicators(df)
    rsi = result["rsi"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI must be 0-100"

def test_score_technical_returns_float():
    df = make_ohlcv()
    ind = compute_indicators(df)
    score = score_technical(ind.iloc[-1])
    assert isinstance(score, (float, int, np.floating))

def test_score_technical_in_range():
    df = make_ohlcv()
    ind = compute_indicators(df)
    score = score_technical(ind.iloc[-1])
    assert -1.0 <= score <= 1.0, f"score out of [-1, 1]: {score}"

def test_score_technical_rising_trend_positive():
    """Strongly rising price should produce a positive technical score."""
    df = make_ohlcv(n=80, trend=0.3, volatility=0.05)
    ind = compute_indicators(df)
    score = score_technical(ind.iloc[-1])
    assert score > 0, f"rising trend should give positive score, got {score}"

def test_score_technical_falling_trend_negative():
    """Strongly falling price should produce a negative technical score."""
    df = make_ohlcv(n=80, trend=-0.3, volatility=0.05)
    ind = compute_indicators(df)
    score = score_technical(ind.iloc[-1])
    assert score < 0, f"falling trend should give negative score, got {score}"

def test_compute_indicators_no_nan_at_end():
    df = make_ohlcv(n=100)
    result = compute_indicators(df)
    last = result.iloc[-1]
    critical = ["rsi", "macd", "ema9", "ema21"]
    for col in critical:
        if col in last.index:
            assert not np.isnan(last[col]), f"{col} is NaN at last bar"
