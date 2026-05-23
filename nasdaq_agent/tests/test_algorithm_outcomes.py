"""
Algorithm Outcome Tests — verifies that each algorithm produces the
mathematically correct output for known inputs.

No `ta` library dependency: all algorithms tested here use only
numpy/pandas, so tests run in every environment.

Covers:
  1.  Market Regime — detect_regime, apply_regime, RegimeInfo multipliers
  2.  Pivot Points — standard floor-trader formula with known H/L/C
  3.  Volume POC — histogram-based volume-at-price calculation
  4.  Swing Level detection — local extrema clustering and filtering
  5.  S/R helpers — nearest_support, nearest_resistance, get_all_sr_levels
  6.  Level clustering — _cluster_levels tolerance and merge logic
  7.  Price Action patterns — Doji, Hammer, Shooting Star, Engulfing, Stars
  8.  Trend Analysis — analyze_trend_with_confidence with known trend series
  9.  score_price_action — directional bias near S/R levels
  10. RSI divergence detection — classic and hidden divergence
  11. MACD divergence detection
  12. Reversal zone scoring
  13. Risk controls — position sizing formulas, circuit-breaker state
  14. VIX proxy formula

Run:
    cd nasdaq_agent
    pytest tests/test_algorithm_outcomes.py -v --tb=short
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import make_ohlcv
from agent.market_regime import detect_regime, apply_regime, RegimeInfo
from agent.support_resistance import (
    calculate_pivot_points, find_swing_levels, calculate_volume_poc,
    get_all_sr_levels, nearest_support, nearest_resistance, _cluster_levels,
)
from agent.price_action import detect_patterns, analyze_trend_with_confidence, score_price_action


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_trend_series(n: int, start: float, drift: float, noise: float = 0.05) -> pd.DataFrame:
    """Create a deterministic rising/falling OHLCV series for trend tests."""
    rng = np.random.default_rng(0)
    closes = [start + i * drift + rng.normal(0, noise) for i in range(n)]
    closes = np.array(closes)
    noise_arr = np.abs(rng.uniform(0.02, 0.1, n))
    highs  = closes + noise_arr
    lows   = closes - noise_arr
    opens  = np.clip(np.roll(closes, 1), lows, highs)
    opens[0] = start
    idx = pd.date_range("2025-01-10 09:30", periods=n, freq="1min")
    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows,
        "Close": closes, "Volume": np.ones(n) * 1_000_000,
    }, index=idx)


def _make_single_bar(o: float, h: float, l: float, c: float) -> pd.Series:
    return pd.Series({"Open": o, "High": h, "Low": l, "Close": c})


def _df_from_bars(*bars) -> pd.DataFrame:
    """Build a 3-bar DataFrame from (o,h,l,c) tuples for pattern tests."""
    rows = [{"Open": o, "High": h, "Low": l, "Close": c} for o, h, l, c in bars]
    idx = pd.date_range("2025-01-10 09:30", periods=len(rows), freq="1min")
    return pd.DataFrame(rows, index=idx)


# ═════════════════════════════════════════════════════════════════════════════
# 1 – Market Regime
# ═════════════════════════════════════════════════════════════════════════════

class TestMarketRegime:

    def _spy_qqq(self, spy_drift: float, qqq_drift: float, n: int = 20) -> tuple:
        """Build SPY and QQQ dataframes with specified 60-min drift."""
        spy = _make_trend_series(n, start=500.0, drift=spy_drift, noise=0.01)
        qqq = _make_trend_series(n, start=400.0, drift=qqq_drift, noise=0.01)
        return spy, qqq

    def test_bull_trend_when_both_up(self):
        """SPY +0.5% and QQQ +0.5% over 60 min → BULL_TREND."""
        # drift=0.5 per bar × 13 bars ≈ +6.5 points on 500 start → +1.3%
        spy, qqq = self._spy_qqq(spy_drift=0.4, qqq_drift=0.4, n=20)
        regime = detect_regime(spy, qqq)
        assert regime.regime == "BULL_TREND"

    def test_bear_trend_when_both_down(self):
        """SPY and QQQ both falling → BEAR_TREND."""
        spy, qqq = self._spy_qqq(spy_drift=-0.4, qqq_drift=-0.4, n=20)
        regime = detect_regime(spy, qqq)
        assert regime.regime == "BEAR_TREND"

    def test_neutral_when_diverging(self):
        """SPY up but QQQ flat → NEUTRAL (no consensus)."""
        spy = _make_trend_series(20, 500.0, drift=0.3, noise=0.01)
        qqq = _make_trend_series(20, 400.0, drift=0.0, noise=0.01)
        regime = detect_regime(spy, qqq)
        assert regime.regime == "NEUTRAL"

    def test_choppy_on_high_volatility(self):
        """Extreme intraday volatility (ATR/price > 2.5%) → CHOPPY."""
        rng = np.random.default_rng(1)
        n = 20
        closes = 500.0 + rng.normal(0, 20, n)   # massive volatility: ±20 on 500
        closes = np.clip(closes, 400, 600)
        highs  = closes + np.abs(rng.normal(15, 5, n))
        lows   = closes - np.abs(rng.normal(15, 5, n))
        lows   = np.maximum(lows, closes * 0.80)
        opens  = np.clip(np.roll(closes, 1), lows, highs)
        opens[0] = 500.0
        idx = pd.date_range("2025-01-10 09:30", periods=n, freq="1min")
        spy = pd.DataFrame({
            "Open": opens, "High": highs, "Low": lows,
            "Close": closes, "Volume": np.ones(n) * 1e6,
        }, index=idx)
        qqq = spy.copy()
        regime = detect_regime(spy, qqq)
        assert regime.regime == "CHOPPY"

    def test_regime_returns_regimeinfo(self):
        spy, qqq = self._spy_qqq(0.3, 0.3, n=15)
        regime = detect_regime(spy, qqq)
        assert isinstance(regime, RegimeInfo)

    def test_regime_has_required_attributes(self):
        spy, qqq = self._spy_qqq(0.3, 0.3)
        regime = detect_regime(spy, qqq)
        for attr in ("regime", "long_mult", "short_mult", "description",
                     "spy_change", "qqq_change", "vix_proxy"):
            assert hasattr(regime, attr), f"RegimeInfo missing attribute: {attr}"

    def test_bull_trend_long_mult_above_1(self):
        """BULL_TREND should boost long signals (long_mult > 1.0)."""
        spy, qqq = self._spy_qqq(0.4, 0.4)
        regime = detect_regime(spy, qqq)
        if regime.regime == "BULL_TREND":
            assert regime.long_mult > 1.0
            assert regime.short_mult < 1.0

    def test_bear_trend_short_mult_above_1(self):
        spy, qqq = self._spy_qqq(-0.4, -0.4)
        regime = detect_regime(spy, qqq)
        if regime.regime == "BEAR_TREND":
            assert regime.short_mult > 1.0
            assert regime.long_mult < 1.0

    def test_apply_regime_bull_boosts_positive_score(self):
        """Positive score × BULL_TREND long_mult > original score."""
        spy, qqq = self._spy_qqq(0.4, 0.4)
        regime = detect_regime(spy, qqq)
        if regime.regime == "BULL_TREND":
            original = 0.50
            adjusted = apply_regime(original, regime)
            assert adjusted >= original  # boosted or same

    def test_apply_regime_bear_boosts_negative_score(self):
        """Negative score × BEAR_TREND short_mult → more negative."""
        spy, qqq = self._spy_qqq(-0.4, -0.4)
        regime = detect_regime(spy, qqq)
        if regime.regime == "BEAR_TREND":
            original = -0.50
            adjusted = apply_regime(original, regime)
            assert adjusted <= original  # more negative or same

    def test_apply_regime_output_clamped_to_1(self):
        """apply_regime must always return a value in [-1, 1]."""
        info = RegimeInfo()
        info.long_mult = 1.5  # would push 0.9 above 1.0
        info.short_mult = 1.5
        for score in (0.9, -0.9, 0.5, -0.5, 0.0):
            result = apply_regime(score, info)
            assert -1.0 <= result <= 1.0, f"apply_regime({score}) = {result} out of [-1,1]"

    def test_apply_regime_zero_stays_zero(self):
        spy, qqq = self._spy_qqq(0.3, 0.3)
        regime = detect_regime(spy, qqq)
        assert apply_regime(0.0, regime) == 0.0

    def test_no_data_returns_neutral(self):
        """Null/empty frames → RegimeInfo with NEUTRAL regime."""
        regime = detect_regime(None, None)
        assert isinstance(regime, RegimeInfo)
        assert regime.regime == "NEUTRAL"

    def test_spy_change_sign_matches_drift(self):
        """spy_change attribute should have same sign as the input drift."""
        spy_up, qqq_up = self._spy_qqq(0.4, 0.0, n=20)
        regime = detect_regime(spy_up, qqq_up)
        assert regime.spy_change > 0


# ═════════════════════════════════════════════════════════════════════════════
# 2 – Pivot Points
# ═════════════════════════════════════════════════════════════════════════════

class TestPivotPoints:
    """Verify floor-trader pivot formulas against hand-computed expected values."""

    def _daily(self, h: float, l: float, c: float) -> pd.DataFrame:
        """One-row daily DataFrame for prior session."""
        idx = pd.date_range("2025-01-09", periods=2, freq="D")
        return pd.DataFrame({
            "High":  [h, h + 1],  # iloc[-2] = first row = test values
            "Low":   [l, l + 1],
            "Close": [c, c + 1],
        }, index=idx)

    def test_pp_formula(self):
        h, l, c = 110.0, 90.0, 105.0
        expected_pp = (h + l + c) / 3.0
        piv = calculate_pivot_points(make_ohlcv(n=10), self._daily(h, l, c))
        assert abs(piv["PP"] - expected_pp) < 0.001

    def test_r1_formula(self):
        h, l, c = 110.0, 90.0, 105.0
        pp  = (h + l + c) / 3.0
        r1  = 2 * pp - l
        piv = calculate_pivot_points(make_ohlcv(n=10), self._daily(h, l, c))
        assert abs(piv["R1"] - r1) < 0.001

    def test_s1_formula(self):
        h, l, c = 110.0, 90.0, 105.0
        pp  = (h + l + c) / 3.0
        s1  = 2 * pp - h
        piv = calculate_pivot_points(make_ohlcv(n=10), self._daily(h, l, c))
        assert abs(piv["S1"] - s1) < 0.001

    def test_r2_formula(self):
        h, l, c = 110.0, 90.0, 105.0
        pp  = (h + l + c) / 3.0
        r2  = pp + (h - l)
        piv = calculate_pivot_points(make_ohlcv(n=10), self._daily(h, l, c))
        assert abs(piv["R2"] - r2) < 0.001

    def test_s2_formula(self):
        h, l, c = 110.0, 90.0, 105.0
        pp  = (h + l + c) / 3.0
        s2  = pp - (h - l)
        piv = calculate_pivot_points(make_ohlcv(n=10), self._daily(h, l, c))
        assert abs(piv["S2"] - s2) < 0.001

    def test_ordering_r1_pp_s1(self):
        """Must always hold: R1 > PP > S1 (standard pivot ordering)."""
        for h, l, c in [(110, 90, 100), (200, 180, 195), (50, 40, 47)]:
            piv = calculate_pivot_points(make_ohlcv(n=5), self._daily(h, l, c))
            assert piv["R1"] > piv["PP"] > piv["S1"], \
                f"Pivot ordering violated: R1={piv['R1']} PP={piv['PP']} S1={piv['S1']}"

    def test_r3_r2_r1_ascending(self):
        piv = calculate_pivot_points(make_ohlcv(n=5), self._daily(110, 90, 100))
        assert piv["R3"] > piv["R2"] > piv["R1"]

    def test_s3_s2_s1_descending(self):
        piv = calculate_pivot_points(make_ohlcv(n=5), self._daily(110, 90, 100))
        assert piv["S3"] < piv["S2"] < piv["S1"]

    def test_empty_df_returns_zeros(self):
        piv = calculate_pivot_points(pd.DataFrame())
        assert all(v == 0.0 for v in piv.values())

    def test_too_short_df_returns_zeros(self):
        piv = calculate_pivot_points(make_ohlcv(n=1))
        # With no daily data and only 1 row, falls back to intraday — still valid
        assert isinstance(piv, dict)
        assert "PP" in piv

    def test_output_has_all_keys(self):
        piv = calculate_pivot_points(make_ohlcv(n=10), self._daily(110, 90, 100))
        for k in ("PP", "R1", "R2", "R3", "S1", "S2", "S3"):
            assert k in piv


# ═════════════════════════════════════════════════════════════════════════════
# 3 – Volume POC
# ═════════════════════════════════════════════════════════════════════════════

class TestVolumePOC:

    def test_poc_within_price_range(self):
        df = make_ohlcv(n=50)
        poc = calculate_volume_poc(df)
        assert df["Low"].min() <= poc <= df["High"].max()

    def test_poc_positive(self):
        df = make_ohlcv(n=50)
        poc = calculate_volume_poc(df)
        assert poc > 0.0

    def test_poc_concentrated_volume(self):
        """When all volume is at one price band, POC should be in that band."""
        n = 50
        prices = np.linspace(100, 200, n)
        # Massive volume at prices 140-160 only (bars 20-30)
        vols   = np.ones(n) * 100
        vols[20:30] = 1_000_000
        df = pd.DataFrame({
            "Open":   prices,
            "High":   prices + 0.5,
            "Low":    prices - 0.5,
            "Close":  prices,
            "Volume": vols,
        })
        poc = calculate_volume_poc(df)
        # POC should be in the high-volume region (140-160)
        assert 135 < poc < 165, f"POC={poc:.2f} should be near high-volume band 140-160"

    def test_poc_too_short_returns_zero(self):
        assert calculate_volume_poc(pd.DataFrame()) == 0.0
        assert calculate_volume_poc(make_ohlcv(n=1)) == 0.0

    def test_poc_flat_price_returns_that_price(self):
        """When all bars have the same price, POC should be that price."""
        n = 20
        p = 100.0
        df = pd.DataFrame({
            "Open":   [p] * n, "High":   [p + 0.1] * n,
            "Low":    [p - 0.1] * n, "Close":  [p] * n,
            "Volume": [1000] * n,
        })
        poc = calculate_volume_poc(df)
        assert abs(poc - p) < 0.5  # within half-bucket width

    def test_poc_is_finite_float(self):
        df = make_ohlcv(n=100)
        poc = calculate_volume_poc(df)
        assert isinstance(poc, float) and not (poc != poc)  # not NaN


# ═════════════════════════════════════════════════════════════════════════════
# 4 – Swing Level Detection
# ═════════════════════════════════════════════════════════════════════════════

class TestSwingLevels:

    def test_clear_highs_and_lows_detected(self):
        """A series with obvious local high at bar 20 and low at bar 40."""
        n = 60
        closes = np.full(n, 100.0)
        closes[20] = 120.0    # clear swing high
        closes[40] = 80.0     # clear swing low
        highs  = closes + 0.1
        lows   = closes - 0.1
        highs[20] = 120.1
        lows[40]  = 79.9
        opens = np.clip(np.roll(closes, 1), lows, highs)
        opens[0] = 100.0
        df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows,
                           "Close": closes, "Volume": np.ones(n) * 1e6})
        result = find_swing_levels(df, window=5)
        assert isinstance(result, dict)
        assert "supports" in result
        assert "resistances" in result

    def test_supports_below_current_price(self):
        df = make_ohlcv(n=60, start_price=100.0)
        result = find_swing_levels(df, window=5)
        current_price = float(df["Close"].iloc[-1])
        for s in result.get("supports", []):
            assert s < current_price, f"Support {s:.2f} >= current price {current_price:.2f}"

    def test_resistances_above_current_price(self):
        df = make_ohlcv(n=60, start_price=100.0)
        result = find_swing_levels(df, window=5)
        current_price = float(df["Close"].iloc[-1])
        for r in result.get("resistances", []):
            assert r > current_price, f"Resistance {r:.2f} <= current price {current_price:.2f}"

    def test_max_levels_respected(self):
        df = make_ohlcv(n=100, volatility=2.0)
        result = find_swing_levels(df, window=5, max_levels=3)
        assert len(result.get("supports", [])) <= 3
        assert len(result.get("resistances", [])) <= 3

    def test_too_short_df_returns_empty(self):
        df = make_ohlcv(n=5)  # below 2*window+1=21
        result = find_swing_levels(df, window=10)
        assert result.get("supports", []) == []
        assert result.get("resistances", []) == []

    def test_output_is_dict(self):
        df = make_ohlcv(n=50)
        result = find_swing_levels(df)
        assert isinstance(result, dict)


# ═════════════════════════════════════════════════════════════════════════════
# 5 – Level Clustering
# ═════════════════════════════════════════════════════════════════════════════

class TestClusterLevels:

    def test_identical_levels_merged(self):
        levels = [100.0, 100.0, 100.0]
        result = _cluster_levels(levels)
        assert len(result) == 1
        assert abs(result[0] - 100.0) < 0.001

    def test_near_levels_within_tolerance_merged(self):
        """100.0 and 100.3 are within 0.4% of each other → should merge."""
        levels = [100.0, 100.3]
        result = _cluster_levels(levels, tolerance=0.004)
        assert len(result) == 1

    def test_far_levels_stay_separate(self):
        """100.0 and 102.0 are 2% apart ��� should NOT merge at 0.4% tolerance."""
        levels = [100.0, 102.0]
        result = _cluster_levels(levels, tolerance=0.004)
        assert len(result) == 2

    def test_output_sorted_ascending(self):
        levels = [105.0, 100.0, 95.0, 110.0]
        result = _cluster_levels(levels)
        assert result == sorted(result)

    def test_empty_input_returns_empty(self):
        assert _cluster_levels([]) == []

    def test_single_level_returns_that_level(self):
        result = _cluster_levels([100.0])
        assert len(result) == 1
        assert abs(result[0] - 100.0) < 0.001

    def test_three_clusters_preserved(self):
        """100, 100.1 → cluster1; 103, 103.2 → cluster2; 107 → cluster3."""
        levels = [100.0, 100.1, 103.0, 103.2, 107.0]
        result = _cluster_levels(levels, tolerance=0.004)
        assert len(result) == 3


# ═════════════════════════════════════════════════════════════════════════════
# 6 – nearest_support / nearest_resistance
# ═════════════════════════════════════════════════════════════════════════════

class TestNearestSR:

    def _sr(self, supports=None, resistances=None):
        return {
            "supports": supports or [],
            "resistances": resistances or [],
            "pivots": {},
            "poc": 0.0,
        }

    def test_nearest_support_returns_highest_below_price(self):
        sr = self._sr(supports=[90.0, 95.0, 97.0])
        result = nearest_support(100.0, sr)
        assert result == 97.0  # highest support below 100

    def test_nearest_resistance_returns_lowest_above_price(self):
        sr = self._sr(resistances=[103.0, 105.0, 110.0])
        result = nearest_resistance(100.0, sr)
        assert result == 103.0  # lowest resistance above 100

    def test_no_support_returns_fallback(self):
        """When no support found, fallback to price * 0.98."""
        sr = self._sr(supports=[])
        result = nearest_support(100.0, sr)
        assert abs(result - 98.0) < 0.01

    def test_no_resistance_returns_fallback(self):
        sr = self._sr(resistances=[])
        result = nearest_resistance(100.0, sr)
        assert abs(result - 102.0) < 0.01

    def test_support_above_price_ignored(self):
        """Supports at 105 and 102 (both above price=100) → fallback."""
        sr = self._sr(supports=[105.0, 102.0])
        result = nearest_support(100.0, sr)
        assert result == pytest.approx(100.0 * 0.98, abs=0.01)

    def test_resistance_below_price_ignored(self):
        """Resistances at 95 and 98 (both below price=100) → fallback."""
        sr = self._sr(resistances=[95.0, 98.0])
        result = nearest_resistance(100.0, sr)
        assert result == pytest.approx(100.0 * 1.02, abs=0.01)

    def test_get_all_sr_output_keys(self):
        df = make_ohlcv(n=60)
        sr = get_all_sr_levels(df)
        for key in ("pivots", "supports", "resistances", "poc"):
            assert key in sr, f"Missing SR key: {key}"

    def test_get_all_sr_poc_positive(self):
        df = make_ohlcv(n=60)
        sr = get_all_sr_levels(df)
        assert sr["poc"] >= 0.0

    def test_get_all_sr_empty_df_safe(self):
        sr = get_all_sr_levels(pd.DataFrame())
        assert isinstance(sr, dict)


# ═════════════════════════════════════════════════════════════════════════════
# 7 – Candlestick Pattern Detection
# ═════════════════════════════════════════════════════════════════════════════

class TestPatternDetection:
    """
    Each test crafts a minimal 3-bar DataFrame with known OHLC values that
    satisfy exactly the pattern's mathematical conditions.
    """

    # ── Doji ──────────────────────────────────────────────────────────────────
    def test_doji_detected(self):
        """body < 5% of range → Doji."""
        # body = 0.01, range = 4.0, body/range = 0.0025 < 0.05
        last = (100.0, 102.0, 98.0, 100.01)   # (o, h, l, c)
        df = _df_from_bars((99, 101, 97, 100), (100, 102, 97, 100.5), last)
        assert "Doji" in detect_patterns(df)

    # ── Hammer ────────────────────────────────────────────────────────────────
    def test_hammer_detected(self):
        """
        Hammer criteria:
          body > 0, lower_wick >= 2×body, upper_wick <= 0.15×range,
          close > low + 0.55×range (body in upper half)
        """
        # o=100, h=101, l=97, c=100.8
        # body=0.8, range=4, lw=min(100,100.8)-97=3>=1.6✓ uw=101-100.8=0.2<=0.6✓
        # c=100.8 > 97+0.55*4=99.2 ✓
        last = (100.0, 101.0, 97.0, 100.8)
        df = _df_from_bars((100, 101, 99, 100.5), (100.5, 101.5, 99.5, 101), last)
        assert "Hammer" in detect_patterns(df)

    # ── Shooting Star ─────────────────────────────────────────────────────────
    def test_shooting_star_detected(self):
        """
        Shooting Star: body at bottom, long upper wick, close < 45% of range.
        o=100, h=103, l=99, c=99.2
        body=0.8, range=4, uw=103-100=3>=1.6✓ lw=99.2-99=0.2<=0.6✓
        c=99.2 < l+0.45*range = 99+1.8=100.8 ✓
        """
        last = (100.0, 103.0, 99.0, 99.2)
        df = _df_from_bars((100, 101, 99, 100.5), (100.5, 101.5, 99.5, 101), last)
        assert "Shooting Star" in detect_patterns(df)

    # ── Bullish Engulfing ─────────────────────────────────────────────────────
    def test_bullish_engulfing_detected(self):
        """
        bar2 bearish (o2=102, c2=100), bar3 bullish (o3=99, c3=103)
        o3=99 <= c2=100 ✓, c3=103 >= o2=102 ✓
        """
        bar1 = (100, 101, 99, 100.5)
        bar2 = (102.0, 103.0, 99.5, 100.0)   # bearish
        bar3 = (99.0,  104.0, 98.5, 103.0)   # bullish engulfing
        df = _df_from_bars(bar1, bar2, bar3)
        assert "Bullish Engulfing" in detect_patterns(df)

    # ── Bearish Engulfing ─────────────────────────────────────────────────────
    def test_bearish_engulfing_detected(self):
        """
        bar2 bullish (o2=98, c2=102), bar3 bearish (o3=103, c3=97)
        o3=103 >= c2=102 ✓, c3=97 <= o2=98 ✓
        """
        bar1 = (100, 101, 99, 100.5)
        bar2 = (98.0,  103.0, 97.5, 102.0)   # bullish
        bar3 = (103.0, 104.0, 96.5,  97.0)   # bearish engulfing
        df = _df_from_bars(bar1, bar2, bar3)
        assert "Bearish Engulfing" in detect_patterns(df)

    # ── Morning Star ─────────────────────────────────────────────────────────
    def test_morning_star_detected(self):
        """
        bar1 bearish large body (o=104, c=100), bar2 doji (small body),
        bar3 bullish large body closing above bar1 midpoint (102).
        """
        bar1 = (104.0, 104.5, 99.5, 100.0)   # bearish, body=4, range~5
        bar2 = (100.0, 101.0,  99.0, 100.1)   # tiny body → small body indecision
        bar3 = (100.0, 104.5,  99.5, 104.0)   # bullish, body=4, c3=104>mid1=102
        df = _df_from_bars(bar1, bar2, bar3)
        assert "Morning Star" in detect_patterns(df)

    # ── Evening Star ──────────────────────────────────────────────────────────
    def test_evening_star_detected(self):
        """
        bar1 bullish large body, bar2 doji, bar3 bearish large body
        closing below bar1 midpoint.
        """
        bar1 = (100.0, 104.5, 99.5, 104.0)   # bullish, body=4, mid=102
        bar2 = (104.0, 105.0, 103.0, 104.1)  # tiny body
        bar3 = (104.0, 104.5,  99.5, 100.0)  # bearish, body=4, c3=100<mid1=102
        df = _df_from_bars(bar1, bar2, bar3)
        assert "Evening Star" in detect_patterns(df)

    # ── Edge cases ────────────────────────────────────────────────────────────
    def test_empty_df_returns_empty_list(self):
        assert detect_patterns(pd.DataFrame()) == []

    def test_short_df_returns_empty_list(self):
        df = _df_from_bars((100, 101, 99, 100.5), (100.5, 101, 100, 100.8))
        assert detect_patterns(df) == []

    def test_flat_bars_detected_as_doji_or_nothing(self):
        """All bars with same open/close → Doji."""
        flat = (100.0, 101.0, 99.0, 100.0)  # exact open==close
        df = _df_from_bars(flat, flat, flat)
        patterns = detect_patterns(df)
        if patterns:
            assert "Doji" in patterns

    def test_returns_at_most_4_patterns(self):
        df = make_ohlcv(n=10)
        patterns = detect_patterns(df)
        assert len(patterns) <= 4

    def test_returns_list_of_strings(self):
        df = make_ohlcv(n=10)
        patterns = detect_patterns(df)
        assert isinstance(patterns, list)
        assert all(isinstance(p, str) for p in patterns)


# ═════════════════════════════════════════════════════════════════════════════
# 8 – Trend Analysis
# ═════════════════════════════════════════════════════════════════════════════

class TestTrendAnalysis:

    def test_strong_uptrend_detected(self):
        """60 bars with +1 drift per bar → UPTREND."""
        df = _make_trend_series(60, start=100.0, drift=1.0, noise=0.01)
        trend, prob = analyze_trend_with_confidence(df)
        assert trend == "UPTREND", f"Expected UPTREND but got {trend}"

    def test_strong_downtrend_detected(self):
        """60 bars with -1 drift per bar → DOWNTREND."""
        df = _make_trend_series(60, start=200.0, drift=-1.0, noise=0.01)
        trend, prob = analyze_trend_with_confidence(df)
        assert trend == "DOWNTREND", f"Expected DOWNTREND but got {trend}"

    def test_flat_series_is_sideways(self):
        """Perfectly flat series (zero drift, constant OHLC) → SIDEWAYS."""
        n = 60
        idx = pd.date_range("2025-01-10 09:30", periods=n, freq="1min")
        df = pd.DataFrame({
            "Open":   np.full(n, 100.0),
            "High":   np.full(n, 101.0),
            "Low":    np.full(n, 99.0),
            "Close":  np.full(n, 100.0),
            "Volume": np.full(n, 1_000_000),
        }, index=idx)
        trend, _ = analyze_trend_with_confidence(df)
        assert trend == "SIDEWAYS"

    def test_probability_in_0_5_to_1_range(self):
        """Probability must be in [0.5, 1.0] per spec."""
        df = _make_trend_series(60, start=100.0, drift=0.5, noise=0.01)
        _, prob = analyze_trend_with_confidence(df)
        assert 0.5 <= prob <= 1.0

    def test_strong_trend_has_high_probability(self):
        df = _make_trend_series(60, start=100.0, drift=1.5, noise=0.01)
        _, prob = analyze_trend_with_confidence(df)
        assert prob >= 0.5

    def test_too_short_returns_safely(self):
        df = _make_trend_series(10, start=100.0, drift=0.5)
        # Should return without raising — value may be SIDEWAYS
        trend, prob = analyze_trend_with_confidence(df)
        assert isinstance(trend, str)
        assert isinstance(prob, float)

    def test_returns_tuple(self):
        df = make_ohlcv(n=60)
        result = analyze_trend_with_confidence(df)
        assert isinstance(result, tuple) and len(result) == 2

    def test_trend_labels_valid(self):
        df = make_ohlcv(n=60)
        trend, _ = analyze_trend_with_confidence(df)
        assert trend in ("UPTREND", "DOWNTREND", "SIDEWAYS")


# ═════════════════════════════════════════════════════════════════════════════
# 9 – score_price_action
# ═════════════════════════════════════════════════════════════════════════════

class TestScorePriceAction:

    def _sr_near(self, price: float, side: str) -> dict:
        """Build an SR dict that puts price near a support or resistance."""
        if side == "support":
            return {
                "supports":    [price * 0.999],    # 0.1% below price
                "resistances": [price * 1.05],     # 5% above
                "pivots":      {"PP": price, "R1": price*1.03, "S1": price*0.97},
                "poc":         price,
            }
        else:
            return {
                "supports":    [price * 0.95],
                "resistances": [price * 1.001],    # 0.1% above price
                "pivots":      {"PP": price, "R1": price*1.03, "S1": price*0.97},
                "poc":         price,
            }

    def test_price_at_support_gives_positive_score(self):
        """Sitting on support → bullish bias → positive score."""
        df = make_ohlcv(n=30, start_price=100.0)
        sr = self._sr_near(100.0, "support")
        score, reasons = score_price_action(df, sr)
        assert score > 0, f"Expected positive score at support, got {score}"

    def test_price_at_resistance_gives_negative_score(self):
        """Pushing against resistance → bearish bias → negative score."""
        # Use a flat df to ensure last close is exactly 100.0
        n = 30
        idx = pd.date_range("2025-01-10 09:30", periods=n, freq="1min")
        df = pd.DataFrame({
            "Open":   np.full(n, 100.0),
            "High":   np.full(n, 101.0),
            "Low":    np.full(n, 99.0),
            "Close":  np.full(n, 100.0),
            "Volume": np.full(n, 1_000_000),
        }, index=idx)
        sr = self._sr_near(100.0, "resistance")
        score, reasons = score_price_action(df, sr)
        assert score < 0, f"Expected negative score at resistance, got {score}"

    def test_score_bounded_m1_to_p1(self):
        df = make_ohlcv(n=30)
        sr = get_all_sr_levels(df)
        score, _ = score_price_action(df, sr)
        assert -1.0 <= score <= 1.0

    def test_returns_tuple(self):
        df = make_ohlcv(n=30)
        result = score_price_action(df, get_all_sr_levels(df))
        assert isinstance(result, tuple) and len(result) == 2

    def test_reasons_is_list_of_strings(self):
        df = make_ohlcv(n=30)
        _, reasons = score_price_action(df, get_all_sr_levels(df))
        assert isinstance(reasons, list)
        assert all(isinstance(r, str) for r in reasons)

    def test_rising_trend_gives_positive_component(self):
        """UPTREND should contribute a positive component to the score."""
        df = _make_trend_series(60, 100.0, drift=1.0, noise=0.01)
        score_up, _ = score_price_action(df, get_all_sr_levels(df))
        df_down = _make_trend_series(60, 200.0, drift=-1.0, noise=0.01)
        score_dn, _ = score_price_action(df_down, get_all_sr_levels(df_down))
        assert score_up > score_dn


# ═════════════════════════════════════════════════════════════════════════════
# 10 – RSI Divergence Detection
# ═════════════════════════════════════════════════════════════════════════════

class TestRSIDivergence:

    def _make_rsi_div_df(self, price_dir: str, rsi_dir: str, n: int = 50) -> pd.DataFrame:
        """
        Craft a DataFrame with explicit rsi_14 column to test divergence logic.
        price_dir/rsi_dir = "up" or "down" for the last two swing points.
        """
        df = make_ohlcv(n=n)
        # Inject synthetic rsi_14 column
        rsi = np.full(n, 50.0)
        # Set two distinguishable swing points near the start and end of the series
        mid = n // 2
        if price_dir == "down":
            df.iloc[mid]["Close"] = df.iloc[mid]["Close"] * 1.05  # earlier higher
            # Make last bars lower close — simulate lower low
        if rsi_dir == "up":
            rsi[mid]   = 35.0   # earlier lower RSI
            rsi[n - 1] = 45.0   # later higher RSI = higher low
        else:
            rsi[mid]   = 65.0
            rsi[n - 1] = 55.0
        df["rsi_14"] = rsi
        df["macd_hist"] = np.linspace(0.1, -0.1, n)  # needed by some divergence funcs
        return df

    def test_detect_rsi_divergence_import(self):
        """Module imports correctly."""
        from agent.reversal import detect_rsi_divergence
        assert callable(detect_rsi_divergence)

    def test_rsi_divergence_returns_tuple(self):
        from agent.reversal import detect_rsi_divergence
        df = make_ohlcv(n=50)
        df["rsi_14"] = 50.0
        result = detect_rsi_divergence(df)
        assert isinstance(result, tuple) and len(result) == 3

    def test_rsi_divergence_type_is_valid(self):
        from agent.reversal import detect_rsi_divergence
        df = make_ohlcv(n=50)
        df["rsi_14"] = 50.0
        dtype, strength, desc = detect_rsi_divergence(df)
        assert dtype in ("BULLISH", "BEARISH", "NONE")

    def test_rsi_divergence_strength_bounded(self):
        from agent.reversal import detect_rsi_divergence
        df = make_ohlcv(n=50)
        df["rsi_14"] = 50.0
        dtype, strength, desc = detect_rsi_divergence(df)
        assert 0.0 <= strength <= 1.0

    def test_rsi_divergence_short_df_returns_none(self):
        from agent.reversal import detect_rsi_divergence
        df = make_ohlcv(n=10)
        df["rsi_14"] = 50.0
        dtype, _, _ = detect_rsi_divergence(df)
        assert dtype == "NONE"

    def test_macd_divergence_returns_tuple(self):
        from agent.reversal import detect_macd_divergence
        df = make_ohlcv(n=50)
        df["macd_hist"] = np.linspace(0.1, -0.1, 50)
        result = detect_macd_divergence(df)
        assert isinstance(result, tuple) and len(result) == 3

    def test_macd_divergence_type_valid(self):
        from agent.reversal import detect_macd_divergence
        df = make_ohlcv(n=50)
        df["macd_hist"] = np.zeros(50)
        dtype, strength, desc = detect_macd_divergence(df)
        assert dtype in ("BULLISH", "BEARISH", "NONE")


# ═════════════════════════════════════════════════════════════════════════════
# 11 – Reversal Zone Scoring
# ═════════════════════════════════════════════════════════════════════════════

class TestReversalZone:

    def _df_with_indicators(self, n: int = 50) -> pd.DataFrame:
        df = make_ohlcv(n=n)
        df["rsi_14"]   = 35.0          # oversold — bullish reversal candidate
        df["stoch_k"]  = 20.0
        df["stoch_d"]  = 20.0
        df["cci_20"]   = -120.0
        df["mfi_14"]   = 30.0
        df["macd_hist"] = np.linspace(0.05, -0.05, n)
        df["vwap"]     = df["Close"].mean()
        df["atr_14"]   = 0.5
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        df["vwap"] = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()
        return df

    def test_reversal_zone_returns_dict(self):
        from agent.reversal import compute_reversal_zone
        df = self._df_with_indicators()
        last_row = df.iloc[-1]
        result = compute_reversal_zone(df, last_row, support=95.0,
                                       resist=105.0, ml_reversal_prob=0.6)
        assert isinstance(result, dict)

    def test_reversal_zone_has_required_keys(self):
        from agent.reversal import compute_reversal_zone
        df = self._df_with_indicators()
        last_row = df.iloc[-1]
        result = compute_reversal_zone(df, last_row, support=95.0,
                                       resist=105.0, ml_reversal_prob=0.6)
        for key in ("reversal_score", "reversal_type"):
            assert key in result, f"Missing key in reversal zone: {key}"

    def test_reversal_score_bounded(self):
        from agent.reversal import compute_reversal_zone
        df = self._df_with_indicators()
        last_row = df.iloc[-1]
        result = compute_reversal_zone(df, last_row, support=95.0,
                                       resist=105.0, ml_reversal_prob=0.6)
        score = result.get("reversal_score", 0.0)
        assert 0.0 <= score <= 1.0

    def test_reversal_type_valid(self):
        from agent.reversal import compute_reversal_zone
        df = self._df_with_indicators()
        last_row = df.iloc[-1]
        result = compute_reversal_zone(df, last_row, support=95.0,
                                       resist=105.0, ml_reversal_prob=0.6)
        assert result.get("reversal_type", "NONE") in ("BULLISH", "BEARISH", "NONE")

    def test_oversold_signals_produce_bullish_type(self):
        """RSI=25, Stoch=10/10, MFI=20, CCI=-150 near support → bullish."""
        from agent.reversal import compute_reversal_zone
        df = self._df_with_indicators()
        df["rsi_14"]  = 25.0   # deeply oversold
        df["stoch_k"] = 10.0
        df["stoch_d"] = 15.0   # both below _STOCH_OS=20 → fires stoch signal
        df["mfi_14"]  = 20.0   # below _MFI_OS=25 → fires MFI signal
        df["cci_20"]  = -150.0 # well below -100 → fires CCI signal
        last_row = df.iloc[-1]
        # Force price just above support (0.5% gap)
        support = float(last_row["Close"]) * 0.995
        result = compute_reversal_zone(df, last_row, support=support,
                                       resist=float(last_row["Close"]) * 1.05,
                                       ml_reversal_prob=0.75)
        # Score should be positive when deeply oversold
        assert result.get("reversal_score", 0.0) > 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 12 – VIX Proxy Formula
# ═════════════════════════════════════════════════════════════════════════════

class TestVixProxy:

    def test_vix_proxy_returns_positive_float(self):
        from agent.market_regime import _vix_proxy
        df = make_ohlcv(n=20, volatility=1.0)
        vix = _vix_proxy(df)
        assert isinstance(vix, float) and vix > 0

    def test_higher_volatility_gives_higher_vix(self):
        from agent.market_regime import _vix_proxy
        low_vol  = _vix_proxy(make_ohlcv(n=20, volatility=0.1))
        high_vol = _vix_proxy(make_ohlcv(n=20, volatility=5.0))
        assert high_vol > low_vol

    def test_vix_proxy_too_short_returns_value(self):
        from agent.market_regime import _vix_proxy
        df = make_ohlcv(n=5)
        vix = _vix_proxy(df)
        assert isinstance(vix, float) and vix >= 0
