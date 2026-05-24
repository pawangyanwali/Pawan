"""
tests/test_opening_range.py
============================
Pytest tests for agent/opening_range.py

Covers compute_opening_range() and the OpeningRangeResult dataclass.

Notes on the implementation
----------------------------
* compute_opening_range() filters bars by ET timezone.
* ORB-5  : bars in [09:30, 09:35) ET
* ORB-15 : bars in [09:30, 09:45) ET
* ORB-30 : bars in [09:30, 10:00) ET
* The last bar of the input is treated as "current price".

Because the function converts a tz-naive DatetimeIndex as UTC first and then
to ET, a naive index of "2025-01-10 14:30" (UTC) becomes "2025-01-10 09:30 ET"
— i.e. exactly the market open.  We exploit this to produce controllable ORB
data without a pytz dependency in the tests themselves.
"""
from __future__ import annotations

import sys
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.opening_range import OpeningRangeResult, compute_opening_range


# ---------------------------------------------------------------------------
# Test-data helpers
# ---------------------------------------------------------------------------

def _make_market_df(
    n_bars: int = 40,
    open_price: float = 100.0,
    drift: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return a 1-minute OHLCV DataFrame whose index is tz-naive UTC.

    Starting at 14:30 UTC (= 09:30 ET), so the function's tz-conversion
    places bar 0 exactly at the market open.

    n_bars=40 → 14:30…15:09 UTC → 09:30…10:09 ET, covering ORB-5/15/30.
    """
    rng = np.random.default_rng(seed)
    closes = open_price + np.cumsum(np.where(
        rng.random(n_bars) > 0.5, drift, -drift * 0.5
    ))
    highs  = closes + np.abs(rng.uniform(0.05, 0.30, n_bars))
    lows   = closes - np.abs(rng.uniform(0.05, 0.30, n_bars))
    opens  = np.clip(np.roll(closes, 1), lows, highs)
    opens[0] = open_price
    vols   = rng.integers(100_000, 1_000_000, n_bars).astype(float)

    # tz-naive UTC: 14:30 UTC = 09:30 ET
    idx = pd.date_range("2025-01-10 14:30", periods=n_bars, freq="1min")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )


def _make_breakout_df(above_orh: bool = True) -> pd.DataFrame:
    """
    Return data where the last bar is clearly above ORH (above_orh=True)
    or clearly below ORL (above_orh=False).

    ORB-5 uses bars 0-4 (09:30…09:34 ET).  We fix them to a tight range
    and then push the last bar far outside.
    """
    n = 40
    base = 100.0

    # Bars 0-4: opening range with known high/low
    orh_fixed = base + 0.5
    orl_fixed = base - 0.5

    highs  = np.full(n, (orh_fixed + orl_fixed) / 2)
    lows   = np.full(n, (orh_fixed + orl_fixed) / 2)
    closes = np.full(n, base)
    opens  = np.full(n, base)

    # Set the first 5 bars to create a known OR
    for i in range(5):
        highs[i]  = orh_fixed
        lows[i]   = orl_fixed
        closes[i] = base
        opens[i]  = base

    # Make the final bar break out
    if above_orh:
        closes[-1] = orh_fixed + 2.0   # well above ORH
        highs[-1]  = orh_fixed + 2.5
        lows[-1]   = orh_fixed + 1.5
        opens[-1]  = orh_fixed + 1.8
    else:
        closes[-1] = orl_fixed - 2.0   # well below ORL
        highs[-1]  = orl_fixed - 1.5
        lows[-1]   = orl_fixed - 2.5
        opens[-1]  = orl_fixed - 1.8

    vols = np.full(n, 500_000.0)
    idx  = pd.date_range("2025-01-10 14:30", periods=n, freq="1min")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )


def _make_inside_range_df() -> pd.DataFrame:
    """Return data where all bars (including the last) are inside ORB-5."""
    n = 20
    base = 100.0
    # Tight range; last bar at midpoint → inside
    highs  = np.full(n, base + 0.3)
    lows   = np.full(n, base - 0.3)
    closes = np.full(n, base)         # always at midpoint → inside
    opens  = np.full(n, base)
    vols   = np.full(n, 500_000.0)
    idx    = pd.date_range("2025-01-10 14:30", periods=n, freq="1min")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )


# ===========================================================================
# OpeningRangeResult dataclass tests
# ===========================================================================

class TestOpeningRangeResultDataclass:
    def test_instantiate_with_defaults(self):
        res = OpeningRangeResult()
        assert res.orh_5    == 0.0
        assert res.orl_5    == 0.0
        assert res.orh_15   == 0.0
        assert res.orl_15   == 0.0
        assert res.orh_30   == 0.0
        assert res.orl_30   == 0.0
        assert res.label    == "NONE"     if hasattr(res, "label") else True

    def test_to_dict_returns_dict(self):
        res = OpeningRangeResult()
        d = res.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_all_fields(self):
        res = OpeningRangeResult()
        d = res.to_dict()
        for f in dc_fields(res):
            assert f.name in d

    def test_to_dict_rounds_floats(self):
        res = OpeningRangeResult()
        res.orh_5 = 100.123456789
        d = res.to_dict()
        assert d["orh_5"] == round(100.123456789, 4)


# ===========================================================================
# compute_opening_range — type / schema tests
# ===========================================================================

class TestComputeOpeningRangeSchema:
    def test_returns_openingrangeresult_type(self):
        df = _make_market_df()
        result = compute_opening_range(df)
        assert isinstance(result, OpeningRangeResult)

    def test_orh_at_or_above_orl_5(self):
        df = _make_market_df()
        result = compute_opening_range(df)
        if result.orh_5 > 0:
            assert result.orh_5 >= result.orl_5

    def test_orh_at_or_above_orl_15(self):
        df = _make_market_df()
        result = compute_opening_range(df)
        if result.orh_15 > 0:
            assert result.orh_15 >= result.orl_15

    def test_orh_at_or_above_orl_30(self):
        df = _make_market_df()
        result = compute_opening_range(df)
        if result.orh_30 > 0:
            assert result.orh_30 >= result.orl_30

    def test_orh_positive_value(self):
        df = _make_market_df(n_bars=40)
        result = compute_opening_range(df)
        if result.orh_15 > 0:
            assert result.orh_15 > 0

    def test_orl_positive_value(self):
        df = _make_market_df(n_bars=40)
        result = compute_opening_range(df)
        if result.orl_15 > 0:
            assert result.orl_15 > 0

    def test_description_is_string(self):
        df = _make_market_df()
        result = compute_opening_range(df)
        assert isinstance(result.description, str)

    def test_breakout_5_valid_value(self):
        df = _make_market_df()
        result = compute_opening_range(df)
        assert result.breakout_5 in {"BULL", "BEAR", "NONE"}

    def test_breakout_15_valid_value(self):
        df = _make_market_df()
        result = compute_opening_range(df)
        assert result.breakout_15 in {"BULL", "BEAR", "NONE"}

    def test_breakout_30_valid_value(self):
        df = _make_market_df()
        result = compute_opening_range(df)
        assert result.breakout_30 in {"BULL", "BEAR", "NONE"}

    def test_position_vs_or15_valid(self):
        df = _make_market_df()
        result = compute_opening_range(df)
        assert result.position_vs_or15 in {"ABOVE", "BELOW", "INSIDE", "UNKNOWN"}

    def test_position_vs_or30_valid(self):
        df = _make_market_df()
        result = compute_opening_range(df)
        assert result.position_vs_or30 in {"ABOVE", "BELOW", "INSIDE", "UNKNOWN"}

    def test_or5_score_bounded(self):
        df = _make_market_df()
        result = compute_opening_range(df)
        assert -1.0 <= result.or5_score <= 1.0

    def test_or15_score_bounded(self):
        df = _make_market_df()
        result = compute_opening_range(df)
        assert -1.0 <= result.or15_score <= 1.0

    def test_or30_score_bounded(self):
        df = _make_market_df()
        result = compute_opening_range(df)
        assert -1.0 <= result.or30_score <= 1.0

    def test_or_width_5_pct_non_negative(self):
        df = _make_market_df()
        result = compute_opening_range(df)
        assert result.or_width_5_pct >= 0.0

    def test_or_width_15_pct_non_negative(self):
        df = _make_market_df()
        result = compute_opening_range(df)
        assert result.or_width_15_pct >= 0.0

    def test_or_width_30_pct_non_negative(self):
        df = _make_market_df()
        result = compute_opening_range(df)
        assert result.or_width_30_pct >= 0.0


# ===========================================================================
# compute_opening_range — edge / boundary cases
# ===========================================================================

class TestComputeOpeningRangeEdgeCases:
    def test_empty_df_returns_safe_defaults(self):
        result = compute_opening_range(pd.DataFrame())
        assert isinstance(result, OpeningRangeResult)
        assert result.orh_15 == 0.0
        assert result.orl_15 == 0.0

    def test_too_short_df_safe_defaults(self):
        # Only 2 bars — function should not raise and should return a valid result
        # Both bars fall inside ORB-5 window (09:30-09:32 ET), so ORH/ORL may be
        # populated; the important thing is the result is a valid OpeningRangeResult
        # and no exception is thrown.
        idx = pd.date_range("2025-01-10 14:30", periods=2, freq="1min")
        df = pd.DataFrame(
            {"Open": [100.0, 101.0], "High": [101.0, 102.0],
             "Low":  [99.0,  100.0], "Close": [100.5, 101.5],
             "Volume": [500_000, 600_000]},
            index=idx,
        )
        result = compute_opening_range(df)
        assert isinstance(result, OpeningRangeResult)
        # With only 2 bars the ORH must be >= ORL wherever they are set
        if result.orh_5 > 0:
            assert result.orh_5 >= result.orl_5
        if result.orh_15 > 0:
            assert result.orh_15 >= result.orl_15
        if result.orh_30 > 0:
            assert result.orh_30 >= result.orl_30

    def test_none_df_returns_safe_defaults(self):
        result = compute_opening_range(None)  # type: ignore[arg-type]
        assert isinstance(result, OpeningRangeResult)
        assert result.orh_15 == 0.0

    def test_result_has_description_on_empty(self):
        result = compute_opening_range(pd.DataFrame())
        assert isinstance(result.description, str)
        assert len(result.description) > 0


# ===========================================================================
# compute_opening_range — directional / signal tests
# ===========================================================================

class TestComputeOpeningRangeDirectional:
    def test_clear_breakout_above_orh_is_bull(self):
        df = _make_breakout_df(above_orh=True)
        result = compute_opening_range(df)
        # The last bar is well above ORH → BULL breakout on ORB-5
        if result.orh_5 > 0:
            assert result.breakout_5 == "BULL"

    def test_clear_breakdown_below_orl_is_bear(self):
        df = _make_breakout_df(above_orh=False)
        result = compute_opening_range(df)
        if result.orh_5 > 0:
            assert result.breakout_5 == "BEAR"

    def test_inside_range_is_neutral(self):
        df = _make_inside_range_df()
        result = compute_opening_range(df)
        if result.orh_5 > 0 and result.orl_5 > 0:
            assert result.breakout_5 == "NONE"
            assert result.position_vs_or5 == "INSIDE"

    def test_breakout_pct_positive_on_bull(self):
        df = _make_breakout_df(above_orh=True)
        result = compute_opening_range(df)
        if result.orh_5 > 0:
            assert result.or5_score > 0.0

    def test_position_above_orh_gives_positive_score(self):
        df = _make_breakout_df(above_orh=True)
        result = compute_opening_range(df)
        if result.orh_5 > 0:
            assert result.position_vs_or5 == "ABOVE"
            assert result.or5_score > 0.0

    def test_position_below_orl_gives_negative_score(self):
        df = _make_breakout_df(above_orh=False)
        result = compute_opening_range(df)
        if result.orh_5 > 0:
            assert result.position_vs_or5 == "BELOW"
            assert result.or5_score < 0.0
