"""
Tests for agent/prediction.py — trade prediction and R:R calculation.
"""
import pytest
import numpy as np

pytest.importorskip("ta", reason="ta library not available in this environment")

from agent.prediction import generate_prediction, _evaluate_rr
from agent.technical import compute_indicators
from tests.conftest import make_ohlcv


def _make_last(trend=0.1):
    df = make_ohlcv(n=100, trend=trend)
    ind = compute_indicators(df)
    return ind, ind.iloc[-1]


def test_generate_prediction_returns_dict():
    ind, last = _make_last()
    result = generate_prediction("AAPL", ind, 0.5, 0.5, 0.6, 0.0, last)
    assert isinstance(result, dict)

def test_generate_prediction_required_keys():
    ind, last = _make_last()
    result = generate_prediction("AAPL", ind, 0.5, 0.5, 0.6, 0.0, last)
    for key in ("direction", "confidence", "target_price", "stop_loss",
                "rr_ratio", "trend", "patterns", "reasons", "rr_qualifies"):
        assert key in result, f"missing key: {key}"

def test_direction_valid():
    ind, last = _make_last()
    result = generate_prediction("AAPL", ind, 0.5, 0.5, 0.6, 0.0, last)
    assert result["direction"] in ("BUY", "SELL", "NEUTRAL", "STRONG BUY", "STRONG SELL")

def test_confidence_in_range():
    ind, last = _make_last()
    result = generate_prediction("AAPL", ind, 0.5, 0.5, 0.6, 0.0, last)
    assert 0.0 <= result["confidence"] <= 100.0

def test_target_above_entry_for_buy(monkeypatch):
    ind, last = _make_last(trend=0.3)
    result = generate_prediction("NVDA", ind, 0.8, 0.8, 0.8, 0.8, last)
    price = float(ind["Close"].iloc[-1])
    if result["direction"] == "BUY" and result["target_price"] > 0:
        assert result["target_price"] > price, "BUY target must be above entry"

def test_stop_below_entry_for_buy():
    ind, last = _make_last(trend=0.3)
    result = generate_prediction("NVDA", ind, 0.8, 0.8, 0.8, 0.8, last)
    price = float(ind["Close"].iloc[-1])
    if result["direction"] == "BUY" and result["stop_loss"] > 0:
        assert result["stop_loss"] < price, "BUY stop must be below entry"

def test_target_below_entry_for_sell():
    ind, last = _make_last(trend=-0.3)
    result = generate_prediction("TSLA", ind, -0.8, -0.8, 0.2, -0.8, last)
    price = float(ind["Close"].iloc[-1])
    if result["direction"] == "SELL" and result["target_price"] > 0:
        assert result["target_price"] < price, "SELL target must be below entry"


# ── _evaluate_rr ──────────────────────────────────────────────────────────────

def _sr(support: float, resist: float) -> dict:
    """Build a minimal SR dict for _evaluate_rr."""
    return {"supports": [support], "resistances": [resist], "pivots": {}, "poc": 0.0}

def test_evaluate_rr_buy_basic():
    stop, target, rr, quality, qualifies = _evaluate_rr(
        price=100.0, sr=_sr(96.0, 110.0), direction="BUY"
    )
    assert target > 100.0, "BUY target must be above price"
    assert stop < 100.0,   "BUY stop must be below price"
    assert rr > 0

def test_evaluate_rr_sell_basic():
    stop, target, rr, quality, qualifies = _evaluate_rr(
        price=100.0, sr=_sr(92.0, 104.0), direction="SELL"
    )
    assert target < 100.0, "SELL target must be below price"
    assert stop > 100.0,   "SELL stop must be above price"

def test_minimum_target_distance_buy():
    """Target must be at least 0.3% above price even if resist is very close."""
    stop, target, rr, quality, qualifies = _evaluate_rr(
        price=100.0, sr=_sr(99.0, 100.01), direction="BUY"
    )
    assert target >= 100.0 * 1.003, f"target too close to entry: {target}"

def test_minimum_target_distance_sell():
    """Target must be at least 0.3% below price even if support is very close."""
    stop, target, rr, quality, qualifies = _evaluate_rr(
        price=100.0, sr=_sr(99.99, 101.0), direction="SELL"
    )
    assert target <= 100.0 * 0.997, f"target too close to entry: {target}"

def test_rr_qualifies_threshold():
    stop, target, rr, quality, qualifies = _evaluate_rr(
        price=100.0, sr=_sr(96.0, 110.0), direction="BUY"
    )
    assert qualifies is True

def test_atr_rr_uses_min_rr_when_t2_is_lower(monkeypatch):
    from agent.config_manager import config

    values = {
        "prediction.use_atr_stops": True,
        "prediction.stop_atr_multiple": 1.0,
        "paper.t2_r_multiple": 1.5,
        "prediction.min_rr": 2.0,
        "prediction.min_target_pct": 0.003,
        "prediction.min_stop_dist_pct": 0.004,
        "prediction.max_risk_pct": 0.020,
    }
    monkeypatch.setattr(config, "get", lambda key, default=None: values.get(key, default))

    stop, target, rr, quality, qualifies = _evaluate_rr(
        price=100.0,
        sr={"supports": [96.0], "resistances": [103.0], "pivots": {}, "poc": 0.0},
        direction="BUY",
        atr=1.0,
    )

    assert stop == pytest.approx(99.0)
    assert target == pytest.approx(102.0)
    assert rr == pytest.approx(2.0)
    assert qualifies is True

def test_atr_buy_rr_blocks_resistance_inside_target_path():
    stop, target, rr, quality, qualifies = _evaluate_rr(
        price=100.0,
        sr=_sr(96.0, 100.75),
        direction="BUY",
        atr=1.0,
    )
    assert target > 100.0
    assert quality == "LOW"
    assert qualifies is True

def test_atr_sell_rr_blocks_support_inside_target_path():
    stop, target, rr, quality, qualifies = _evaluate_rr(
        price=100.0,
        sr=_sr(99.25, 104.0),
        direction="SELL",
        atr=1.0,
    )
    assert target < 100.0
    assert quality == "LOW"
    assert qualifies is True

def test_quality_labels():
    _, _, rr, quality, _ = _evaluate_rr(
        price=100.0, sr=_sr(96.0, 115.0), direction="BUY"
    )
    assert quality in ("EXCELLENT", "GOOD", "OK", "LOW")
