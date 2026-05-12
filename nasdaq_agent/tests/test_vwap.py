"""
Tests for agent/vwap.py — VWAP signal event classification.
"""
import numpy as np
import pandas as pd
import pytest
from agent.vwap import compute_vwap_signal
from tests.conftest import make_ohlcv


def _df_with_vwap(price_above_vwap: bool = True, n: int = 60):
    """Build a DataFrame where last Close is above or below VWAP."""
    df = make_ohlcv(n=n, start_price=100.0)
    # Override VWAP to be clearly above or below the last close
    last_close = float(df["Close"].iloc[-1])
    df["vwap"] = last_close * (0.98 if price_above_vwap else 1.02)
    return df


def test_returns_required_keys():
    df = make_ohlcv()
    result = compute_vwap_signal(df)
    for key in ("event", "score", "vwap", "price", "deviation", "description"):
        assert key in result, f"missing key: {key}"

def test_score_is_float():
    df = make_ohlcv()
    result = compute_vwap_signal(df)
    assert isinstance(result["score"], float)

def test_event_is_string():
    df = make_ohlcv()
    result = compute_vwap_signal(df)
    assert isinstance(result["event"], str)
    assert len(result["event"]) > 0

def test_price_above_vwap_score_in_range():
    # Price above VWAP can be ABOVE(+0.3) or EXTENDED_UP(-0.7 exhaustion) — both valid
    df = _df_with_vwap(price_above_vwap=True)
    result = compute_vwap_signal(df)
    assert -1.0 <= result["score"] <= 1.0

def test_price_below_vwap_score_in_range():
    # Price below VWAP can be BELOW(-0.3) or EXTENDED_DOWN(+0.7 bounce) — both valid
    df = _df_with_vwap(price_above_vwap=False)
    result = compute_vwap_signal(df)
    assert -1.0 <= result["score"] <= 1.0

def test_vwap_value_positive():
    df = make_ohlcv()
    result = compute_vwap_signal(df)
    assert result["vwap"] > 0

def test_handles_missing_vwap_column_gracefully():
    df = make_ohlcv()
    df = df.drop(columns=["vwap"])
    # Should not raise — must return a safe default
    result = compute_vwap_signal(df)
    assert "event" in result

def test_deviation_sign_matches_position():
    df = _df_with_vwap(price_above_vwap=True)
    result = compute_vwap_signal(df)
    assert result["deviation"] >= 0.0, "deviation should be positive when price > VWAP"
