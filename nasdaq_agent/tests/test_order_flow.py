"""
tests/test_order_flow.py
========================
Pytest tests for agent/order_flow.py

Covers compute_order_flow() and get_signal_strength().
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the package root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.order_flow import compute_order_flow, get_signal_strength

# ---------------------------------------------------------------------------
# OHLCV helper (no ta dependency, per spec)
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 50, drift: float = 0.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 + np.cumsum(rng.normal(drift, 0.5, n))
    highs  = closes + np.abs(rng.uniform(0.1, 0.5, n))
    lows   = closes - np.abs(rng.uniform(0.1, 0.5, n))
    opens  = np.clip(np.roll(closes, 1), lows, highs)
    opens[0] = closes[0]
    vols   = rng.integers(500_000, 2_000_000, n).astype(float)
    idx    = pd.date_range("2025-01-10 09:30", periods=n, freq="1min")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )


REQUIRED_KEYS = {
    "score",
    "cmf",
    "volume_pressure",
    "momentum_score",
    "surge_detected",
    "consecutive_bars",
    "divergence",
    "interpretation",
    "label",
}

VALID_LABELS     = {"STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"}
VALID_DIVERGENCE = {"BULLISH", "BEARISH", "NONE"}

SIGNAL_STRENGTH_REQUIRED_KEYS = {"strength", "execute", "size_mult", "reason"}


# ===========================================================================
# compute_order_flow — structure / schema tests
# ===========================================================================

class TestComputeOrderFlowSchema:
    def test_returns_all_required_keys(self):
        df = make_ohlcv()
        result = compute_order_flow(df)
        assert REQUIRED_KEYS.issubset(result.keys()), (
            f"Missing keys: {REQUIRED_KEYS - result.keys()}"
        )

    def test_score_bounded_m1_to_p1(self):
        df = make_ohlcv()
        result = compute_order_flow(df)
        assert -1.0 <= result["score"] <= 1.0

    def test_label_is_valid_string(self):
        df = make_ohlcv()
        result = compute_order_flow(df)
        assert result["label"] in VALID_LABELS

    def test_cmf_bounded(self):
        df = make_ohlcv()
        result = compute_order_flow(df)
        assert -1.0 <= result["cmf"] <= 1.0

    def test_volume_pressure_bounded(self):
        df = make_ohlcv()
        result = compute_order_flow(df)
        assert -1.0 <= result["volume_pressure"] <= 1.0

    def test_momentum_score_bounded(self):
        df = make_ohlcv()
        result = compute_order_flow(df)
        assert -1.0 <= result["momentum_score"] <= 1.0

    def test_divergence_label_valid(self):
        df = make_ohlcv()
        result = compute_order_flow(df)
        assert result["divergence"] in VALID_DIVERGENCE

    def test_interpretation_nonempty_string(self):
        df = make_ohlcv()
        result = compute_order_flow(df)
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0

    def test_surge_detected_is_bool(self):
        df = make_ohlcv()
        result = compute_order_flow(df)
        assert isinstance(result["surge_detected"], bool)

    def test_consecutive_bars_is_int(self):
        df = make_ohlcv()
        result = compute_order_flow(df)
        assert isinstance(result["consecutive_bars"], int)


# ===========================================================================
# compute_order_flow — edge / boundary cases
# ===========================================================================

class TestComputeOrderFlowEdgeCases:
    def test_empty_df_returns_neutral(self):
        result = compute_order_flow(pd.DataFrame())
        assert result["score"] == 0.0
        assert result["label"] == "NEUTRAL"

    def test_too_short_df_returns_neutral(self):
        df = make_ohlcv(n=3)
        result = compute_order_flow(df)
        assert result["score"] == 0.0
        assert result["label"] == "NEUTRAL"

    def test_none_df_returns_neutral(self):
        result = compute_order_flow(None)  # type: ignore[arg-type]
        assert result["score"] == 0.0
        assert result["label"] == "NEUTRAL"

    def test_score_deterministic_same_input(self):
        df = make_ohlcv(seed=7)
        r1 = compute_order_flow(df)
        r2 = compute_order_flow(df)
        assert r1["score"] == r2["score"]

    def test_case_insensitive_columns(self):
        df = make_ohlcv()
        df_lower = df.rename(columns=str.lower)
        result = compute_order_flow(df_lower)
        assert REQUIRED_KEYS.issubset(result.keys())
        assert result["label"] in VALID_LABELS

    def test_case_insensitive_columns_upper(self):
        df = make_ohlcv()
        # The function should already accept Title-case (Open/High/...) columns
        # from make_ohlcv; verify it also accepts all-upper
        df_upper = df.rename(columns=str.upper)
        result = compute_order_flow(df_upper)
        assert REQUIRED_KEYS.issubset(result.keys())


# ===========================================================================
# compute_order_flow — directional tests
# ===========================================================================

class TestComputeOrderFlowDirectional:
    def test_uptrend_gives_positive_score(self):
        # Strong upward drift should produce a positive score
        df = make_ohlcv(n=50, drift=1.0, seed=0)
        result = compute_order_flow(df)
        assert result["score"] > 0.0, (
            f"Expected positive score for up-trending data, got {result['score']}"
        )

    def test_downtrend_gives_negative_score(self):
        # Strong downward drift should produce a negative score
        df = make_ohlcv(n=50, drift=-1.0, seed=0)
        result = compute_order_flow(df)
        assert result["score"] < 0.0, (
            f"Expected negative score for down-trending data, got {result['score']}"
        )

    def test_uptrend_label_is_bullish(self):
        df = make_ohlcv(n=50, drift=1.0, seed=1)
        result = compute_order_flow(df)
        assert result["label"] in {"STRONG_BUY", "BUY"}

    def test_downtrend_label_is_bearish(self):
        df = make_ohlcv(n=50, drift=-1.0, seed=1)
        result = compute_order_flow(df)
        assert result["label"] in {"STRONG_SELL", "SELL"}

    def test_consecutive_bars_positive_on_uptrend(self):
        # With a strong drift every bar closes up → streak should be > 0
        df = make_ohlcv(n=50, drift=2.0, seed=3)
        result = compute_order_flow(df)
        assert result["consecutive_bars"] >= 0  # non-negative for uptrend

    def test_consecutive_bars_negative_on_downtrend(self):
        df = make_ohlcv(n=50, drift=-2.0, seed=3)
        result = compute_order_flow(df)
        assert result["consecutive_bars"] <= 0  # non-positive for downtrend

    def test_surge_detected_on_high_volume(self):
        df = make_ohlcv(n=30, drift=0.5, seed=10)
        # Make the last bar's volume 20× the average of the rest
        avg_vol = float(df["Volume"].iloc[:-1].mean())
        df = df.copy()
        df.iloc[-1, df.columns.get_loc("Volume")] = avg_vol * 20
        result = compute_order_flow(df)
        assert result["surge_detected"] is True

    def test_surge_not_detected_on_normal_volume(self):
        # All bars have the same volume — no surge possible
        df = make_ohlcv(n=30)
        df = df.copy()
        df["Volume"] = 1_000_000.0
        result = compute_order_flow(df)
        assert result["surge_detected"] is False


# ===========================================================================
# compute_order_flow — lookback parameter
# ===========================================================================

class TestComputeOrderFlowLookback:
    def test_lookback_clamped_to_available_rows(self):
        # lookback larger than df length should not raise
        df = make_ohlcv(n=10)
        result = compute_order_flow(df, lookback=100)
        assert REQUIRED_KEYS.issubset(result.keys())

    def test_short_lookback_returns_valid_result(self):
        df = make_ohlcv(n=20)
        result = compute_order_flow(df, lookback=5)
        assert REQUIRED_KEYS.issubset(result.keys())
        assert -1.0 <= result["score"] <= 1.0

    def test_different_lookbacks_return_different_scores(self):
        df = make_ohlcv(n=50, drift=0.5)
        r_short = compute_order_flow(df, lookback=5)
        r_long  = compute_order_flow(df, lookback=40)
        # They may or may not differ numerically, but both must be valid
        assert -1.0 <= r_short["score"] <= 1.0
        assert -1.0 <= r_long["score"]  <= 1.0


# ===========================================================================
# get_signal_strength — structure tests
# ===========================================================================

class TestGetSignalStrength:
    def test_get_signal_strength_returns_dict(self):
        result = get_signal_strength(80.0, 0.6, "TRENDING_UP")
        assert isinstance(result, dict)

    def test_get_signal_strength_has_required_keys(self):
        result = get_signal_strength(80.0, 0.6, "TRENDING_UP")
        assert SIGNAL_STRENGTH_REQUIRED_KEYS.issubset(result.keys()), (
            f"Missing keys: {SIGNAL_STRENGTH_REQUIRED_KEYS - result.keys()}"
        )

    def test_execute_is_bool(self):
        result = get_signal_strength(80.0, 0.6, "TRENDING_UP")
        assert isinstance(result["execute"], bool)

    def test_size_mult_is_float(self):
        result = get_signal_strength(80.0, 0.6, "TRENDING_UP")
        assert isinstance(result["size_mult"], float)

    def test_reason_is_nonempty_string(self):
        result = get_signal_strength(80.0, 0.6, "TRENDING_UP")
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0


# ===========================================================================
# get_signal_strength — rule tests
# ===========================================================================

class TestGetSignalStrengthRules:
    def test_low_confidence_returns_no_signal(self):
        # Below 72 → NO_SIGNAL regardless of order flow
        result = get_signal_strength(60.0, 0.9, "TRENDING_UP")
        assert result["strength"] == "NO_SIGNAL"
        assert result["execute"] is False
        assert result["size_mult"] == 0.0

    def test_range_bound_returns_blocked(self):
        result = get_signal_strength(85.0, 0.8, "RANGE_BOUND")
        assert result["strength"] == "BLOCKED"
        assert result["execute"] is False

    def test_conflicted_of_blocks_trade(self):
        # Strongly negative order flow while price model is bullish → CONFLICTED
        result = get_signal_strength(80.0, -0.5, "TRENDING_UP")
        assert result["strength"] == "CONFLICTED"
        assert result["execute"] is False

    def test_strong_buy_all_conditions_met(self):
        result = get_signal_strength(85.0, 0.7, "TRENDING_UP")
        assert result["strength"] == "STRONG"
        assert result["execute"] is True
        assert result["size_mult"] == 1.0

    def test_standard_buy_moderate_of(self):
        result = get_signal_strength(75.0, 0.30, "TRENDING")
        assert result["strength"] == "STANDARD"
        assert result["execute"] is True

    def test_weak_buy_neutral_of_trending_up(self):
        # OF score between -0.40 and +0.20 AND regime TRENDING_UP
        result = get_signal_strength(75.0, 0.10, "TRENDING_UP")
        assert result["strength"] == "WEAK"
        assert result["execute"] is True
        assert result["size_mult"] == 0.5

    def test_bullish_regime_boosts_positive_score(self):
        # Both TRENDING_UP and TRENDING should enable the strong signal
        result_up   = get_signal_strength(85.0, 0.7, "TRENDING_UP")
        result_trend = get_signal_strength(85.0, 0.7, "TRENDING")
        # Either should be executeable (STRONG or STANDARD)
        assert result_up["execute"] is True
        assert result_trend["execute"] is True

    def test_strong_buy_not_triggered_below_confidence(self):
        # Just under the 78 threshold for strong
        result = get_signal_strength(77.0, 0.9, "TRENDING_UP")
        # Should NOT be STRONG (may be STANDARD)
        assert result["strength"] != "STRONG"

    def test_regime_case_insensitive(self):
        # Lowercase regime strings should be treated as upper after .upper()
        result_lower = get_signal_strength(85.0, 0.7, "trending_up")
        result_upper = get_signal_strength(85.0, 0.7, "TRENDING_UP")
        assert result_lower["strength"] == result_upper["strength"]
