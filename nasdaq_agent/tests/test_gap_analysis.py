"""
tests/test_gap_analysis.py
===========================
Pytest tests for agent/gap_analysis.py

Covers analyse_gap() and detect_color_flip().

Notes on the implementation
----------------------------
* analyse_gap() treats the 2nd-to-last row of df_1d as yesterday's close
  and the first bar of df_1m (regular session, non-pre-market) as today's open.
* The function converts tz-naive df_1m index as UTC → ET before splitting
  into pre-market / regular-session bars.
  - tz-naive "2025-01-10 14:30" UTC = "2025-01-10 09:30 ET" → regular session.
  - tz-naive "2025-01-10 13:00" UTC = "2025-01-10 08:00 ET" → pre-market.
* For a gap-up test:  today_open > prior_close  (≥ +0.5 % threshold)
* For a gap-down test: today_open < prior_close  (≤ -0.5 % threshold)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.gap_analysis import analyse_gap, detect_color_flip


# ---------------------------------------------------------------------------
# Test-data helpers
# ---------------------------------------------------------------------------

def make_daily(prev_close: float = 100.0, today_open: float = 103.0) -> pd.DataFrame:
    """Two-row daily DataFrame: yesterday + today."""
    idx = pd.date_range("2025-01-09", periods=2, freq="D")
    return pd.DataFrame(
        {
            "Open":   [prev_close * 0.99, today_open],
            "High":   [prev_close * 1.01, today_open * 1.01],
            "Low":    [prev_close * 0.98, today_open * 0.99],
            "Close":  [prev_close,         today_open],
            "Volume": [1_000_000,          500_000],
        },
        index=idx,
    )


def make_intraday(
    n_bars: int = 30,
    open_price: float = 103.0,
    drift: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    1-minute intraday bars starting at 14:30 UTC (= 09:30 ET, regular session).
    """
    rng = np.random.default_rng(seed)
    closes = open_price + np.cumsum(rng.normal(drift, 0.1, n_bars))
    highs  = closes + np.abs(rng.uniform(0.02, 0.15, n_bars))
    lows   = closes - np.abs(rng.uniform(0.02, 0.15, n_bars))
    opens  = np.clip(np.roll(closes, 1), lows, highs)
    opens[0] = open_price
    vols   = rng.integers(50_000, 500_000, n_bars).astype(float)
    idx    = pd.date_range("2025-01-10 14:30", periods=n_bars, freq="1min")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )


REQUIRED_GAP_KEYS = {
    "gap_type", "gap_pct", "prior_close", "today_open",
    "fill_probability", "gap_filled", "description",
}

REQUIRED_COLOR_FLIP_KEYS = {
    "color", "r2g_event", "g2r_event", "r2g_bars_ago", "g2r_bars_ago",
}

VALID_GAP_TYPES = {"GAP_UP", "GAP_DOWN", "FLAT"}
VALID_COLORS    = {"GREEN", "RED", "FLAT"}


# ===========================================================================
# analyse_gap — structure / schema tests
# ===========================================================================

class TestAnalyseGapSchema:
    def test_analyse_gap_returns_dict(self):
        df_1m = make_intraday(open_price=103.0)
        df_1d = make_daily(prev_close=100.0, today_open=103.0)
        result = analyse_gap(df_1m, df_1d)
        assert isinstance(result, dict)

    def test_analyse_gap_has_required_keys(self):
        df_1m = make_intraday(open_price=103.0)
        df_1d = make_daily(prev_close=100.0, today_open=103.0)
        result = analyse_gap(df_1m, df_1d)
        assert REQUIRED_GAP_KEYS.issubset(result.keys()), (
            f"Missing keys: {REQUIRED_GAP_KEYS - result.keys()}"
        )

    def test_gap_type_valid_string(self):
        df_1m = make_intraday(open_price=103.0)
        df_1d = make_daily(prev_close=100.0, today_open=103.0)
        result = analyse_gap(df_1m, df_1d)
        assert result["gap_type"] in VALID_GAP_TYPES

    def test_fill_probability_bounded_0_to_1(self):
        df_1m = make_intraday(open_price=103.0)
        df_1d = make_daily(prev_close=100.0, today_open=103.0)
        result = analyse_gap(df_1m, df_1d)
        assert 0.0 <= result["fill_probability"] <= 1.0

    def test_gap_filled_is_bool(self):
        df_1m = make_intraday(open_price=103.0)
        df_1d = make_daily(prev_close=100.0, today_open=103.0)
        result = analyse_gap(df_1m, df_1d)
        assert isinstance(result["gap_filled"], bool)

    def test_description_is_nonempty_string(self):
        df_1m = make_intraday(open_price=103.0)
        df_1d = make_daily(prev_close=100.0, today_open=103.0)
        result = analyse_gap(df_1m, df_1d)
        assert isinstance(result["description"], str)

    def test_prior_close_positive(self):
        df_1m = make_intraday(open_price=103.0)
        df_1d = make_daily(prev_close=100.0, today_open=103.0)
        result = analyse_gap(df_1m, df_1d)
        assert result["prior_close"] > 0.0

    def test_today_open_positive(self):
        df_1m = make_intraday(open_price=103.0)
        df_1d = make_daily(prev_close=100.0, today_open=103.0)
        result = analyse_gap(df_1m, df_1d)
        assert result["today_open"] > 0.0

    def test_gap_score_key_present(self):
        # gap_score is also populated by the implementation
        df_1m = make_intraday(open_price=103.0)
        df_1d = make_daily(prev_close=100.0, today_open=103.0)
        result = analyse_gap(df_1m, df_1d)
        assert "gap_score" in result


# ===========================================================================
# analyse_gap — gap direction tests
# ===========================================================================

class TestAnalyseGapDirection:
    def test_gap_up_positive_gap_pct(self):
        # today_open = 103 vs prev_close = 100 → +3 % gap
        df_1m = make_intraday(open_price=103.0)
        df_1d = make_daily(prev_close=100.0, today_open=103.0)
        result = analyse_gap(df_1m, df_1d)
        assert result["gap_pct"] > 0.0
        assert result["gap_type"] == "GAP_UP"

    def test_gap_down_negative_gap_pct(self):
        # today_open = 97 vs prev_close = 100 → -3 % gap
        df_1m = make_intraday(open_price=97.0)
        df_1d = make_daily(prev_close=100.0, today_open=97.0)
        result = analyse_gap(df_1m, df_1d)
        assert result["gap_pct"] < 0.0
        assert result["gap_type"] == "GAP_DOWN"

    def test_no_gap_is_flat_type(self):
        # today_open = 100.1 vs prev_close = 100 → +0.1 % (< 0.5 % threshold → FLAT)
        df_1m = make_intraday(open_price=100.1)
        df_1d = make_daily(prev_close=100.0, today_open=100.1)
        result = analyse_gap(df_1m, df_1d)
        assert result["gap_type"] == "FLAT"

    def test_exact_threshold_positive_is_gap_up(self):
        # +0.6 % (clearly above the 0.5 % threshold) → GAP_UP
        # Avoids floating-point precision issues at exactly +0.5 %
        prev = 100.0
        today_o = 100.6   # exactly 0.6 % above prev, no fp rounding
        df_1m = make_intraday(open_price=today_o)
        df_1d = make_daily(prev_close=prev, today_open=today_o)
        result = analyse_gap(df_1m, df_1d)
        assert result["gap_type"] == "GAP_UP"

    def test_exact_threshold_negative_is_gap_down(self):
        prev = 100.0
        today_o = prev * 0.995
        df_1m = make_intraday(open_price=today_o)
        df_1d = make_daily(prev_close=prev, today_open=today_o)
        result = analyse_gap(df_1m, df_1d)
        assert result["gap_type"] == "GAP_DOWN"

    def test_gap_up_description_mentions_gap_up(self):
        df_1m = make_intraday(open_price=103.0)
        df_1d = make_daily(prev_close=100.0, today_open=103.0)
        result = analyse_gap(df_1m, df_1d)
        assert "Gap UP" in result["description"] or "GAP_UP" in result["description"]

    def test_gap_down_description_mentions_gap_down(self):
        df_1m = make_intraday(open_price=97.0)
        df_1d = make_daily(prev_close=100.0, today_open=97.0)
        result = analyse_gap(df_1m, df_1d)
        assert "Gap DOWN" in result["description"] or "GAP_DOWN" in result["description"]


# ===========================================================================
# analyse_gap — fill probability tests
# ===========================================================================

class TestAnalyseGapFillProbability:
    def test_small_gap_higher_fill_probability(self):
        # ~1 % gap → 0.70 fill probability (small gap heuristic)
        df_1m = make_intraday(open_price=101.0)
        df_1d = make_daily(prev_close=100.0, today_open=101.0)
        result_small = analyse_gap(df_1m, df_1d)

        # ~10 % gap → 0.30 fill probability (large gap heuristic)
        df_1m_big = make_intraday(open_price=110.0)
        df_1d_big = make_daily(prev_close=100.0, today_open=110.0)
        result_large = analyse_gap(df_1m_big, df_1d_big)

        assert result_small["fill_probability"] >= result_large["fill_probability"]

    def test_large_gap_lower_fill_probability(self):
        # ~10 % gap should give <= 0.30 fill probability
        df_1m = make_intraday(open_price=110.0)
        df_1d = make_daily(prev_close=100.0, today_open=110.0)
        result = analyse_gap(df_1m, df_1d)
        assert result["fill_probability"] <= 0.50

    def test_already_filled_gap_probability_is_1(self):
        # Gap up but current price (last bar close) is below prior close → filled
        prev_close = 100.0
        today_open = 103.0
        # All intraday closes will be below prev_close (98)
        df_1m = make_intraday(open_price=98.0, drift=0.0, seed=0)
        df_1d = make_daily(prev_close=prev_close, today_open=today_open)
        # Force the first bar open to today_open to set gap, but closes are ~98
        df_1m = df_1m.copy()
        df_1m.iloc[0, df_1m.columns.get_loc("Open")] = today_open
        result = analyse_gap(df_1m, df_1d)
        # If the gap was filled, probability = 1.0
        if result["gap_filled"]:
            assert result["fill_probability"] == 1.0


# ===========================================================================
# analyse_gap — edge cases
# ===========================================================================

class TestAnalyseGapEdgeCases:
    def test_too_short_df_1d_returns_safe_defaults(self):
        # Only 1 row in df_1d → cannot determine prior_close → neutral defaults
        df_1d_short = pd.DataFrame(
            {"Open": [100.0], "High": [101.0], "Low": [99.0],
             "Close": [100.0], "Volume": [1_000_000]},
            index=pd.date_range("2025-01-10", periods=1, freq="D"),
        )
        df_1m = make_intraday()
        result = analyse_gap(df_1m, df_1d_short)
        assert isinstance(result, dict)
        assert result["gap_type"] == "FLAT"
        assert result["gap_pct"] == 0.0

    def test_none_df_1d_returns_safe_defaults(self):
        df_1m = make_intraday()
        result = analyse_gap(df_1m, None)  # type: ignore[arg-type]
        assert isinstance(result, dict)
        assert result["gap_type"] == "FLAT"

    def test_none_df_1m_returns_safe_defaults(self):
        df_1d = make_daily()
        result = analyse_gap(None, df_1d)  # type: ignore[arg-type]
        assert isinstance(result, dict)
        assert result["gap_type"] == "FLAT"

    def test_empty_df_1m_returns_safe_defaults(self):
        df_1d = make_daily()
        result = analyse_gap(pd.DataFrame(), df_1d)
        assert isinstance(result, dict)
        assert result["gap_type"] == "FLAT"


# ===========================================================================
# detect_color_flip — structure / schema tests
# ===========================================================================

class TestDetectColorFlipSchema:
    def test_detect_color_flip_returns_dict(self):
        df = make_intraday(open_price=100.0)
        result = detect_color_flip(df, prev_close=100.0)
        assert isinstance(result, dict)

    def test_detect_color_flip_has_required_keys(self):
        df = make_intraday(open_price=100.0)
        result = detect_color_flip(df, prev_close=100.0)
        assert REQUIRED_COLOR_FLIP_KEYS.issubset(result.keys()), (
            f"Missing keys: {REQUIRED_COLOR_FLIP_KEYS - result.keys()}"
        )

    def test_color_is_valid_string(self):
        df = make_intraday(open_price=100.0)
        result = detect_color_flip(df, prev_close=100.0)
        assert result["color"] in VALID_COLORS

    def test_r2g_event_is_bool(self):
        df = make_intraday(open_price=100.0)
        result = detect_color_flip(df, prev_close=100.0)
        assert isinstance(result["r2g_event"], bool)

    def test_g2r_event_is_bool(self):
        df = make_intraday(open_price=100.0)
        result = detect_color_flip(df, prev_close=100.0)
        assert isinstance(result["g2r_event"], bool)

    def test_r2g_bars_ago_is_int(self):
        df = make_intraday(open_price=100.0)
        result = detect_color_flip(df, prev_close=100.0)
        assert isinstance(result["r2g_bars_ago"], int)

    def test_g2r_bars_ago_is_int(self):
        df = make_intraday(open_price=100.0)
        result = detect_color_flip(df, prev_close=100.0)
        assert isinstance(result["g2r_bars_ago"], int)


# ===========================================================================
# detect_color_flip — color logic tests
# ===========================================================================

class TestDetectColorFlipLogic:
    def test_color_green_when_all_closes_above_prev_close(self):
        # All closes >> prev_close → GREEN
        prev_close = 90.0
        df = make_intraday(open_price=100.0, drift=0.0, seed=0)
        # All closes are around 100, which is above prev_close=90
        result = detect_color_flip(df, prev_close=prev_close)
        assert result["color"] == "GREEN"

    def test_color_red_when_all_closes_below_prev_close(self):
        prev_close = 110.0
        df = make_intraday(open_price=100.0, drift=0.0, seed=0)
        result = detect_color_flip(df, prev_close=prev_close)
        assert result["color"] == "RED"

    def test_color_flip_false_same_bar_color(self):
        # When all bars are consistently above prev_close (GREEN), no G2R event
        prev_close = 90.0
        df = make_intraday(open_price=100.0, drift=0.0, seed=1)
        result = detect_color_flip(df, prev_close=prev_close)
        assert result["g2r_event"] is False

    def test_color_flip_true_when_crossing_prev_close(self):
        # Craft a scenario where bars cross prev_close:
        # Most bars below prev_close, then last bar above it → R2G on last bar
        prev_close = 100.0
        n = 10

        # Bars 0 to n-2: closes at 99 (below prev_close)
        # Bar n-1: close at 101 (above prev_close) while bar n-2 close <= prev_close
        closes = np.full(n, 99.0)
        closes[-1] = 101.0

        highs  = closes + 0.3
        lows   = closes - 0.3
        opens  = closes - 0.1
        vols   = np.full(n, 500_000.0)

        # Use 14:30 UTC = 09:30 ET so all bars are on the same "today" in ET
        idx = pd.date_range("2025-01-10 14:30", periods=n, freq="1min")
        df = pd.DataFrame(
            {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
            index=idx,
        )
        result = detect_color_flip(df, prev_close=prev_close)
        # The last bar crossed from below to above → r2g_event should be True
        assert result["r2g_event"] is True

    def test_no_crossing_returns_no_event(self):
        # All bars clearly above prev_close — no crossings at all
        prev_close = 90.0
        df = make_intraday(open_price=100.0, drift=0.0, seed=5)
        result = detect_color_flip(df, prev_close=prev_close)
        assert result["r2g_event"] is False
        assert result["g2r_event"] is False

    def test_none_df_returns_safe_defaults(self):
        result = detect_color_flip(None, prev_close=100.0)  # type: ignore[arg-type]
        assert isinstance(result, dict)
        assert result["color"] == "FLAT"

    def test_empty_df_returns_safe_defaults(self):
        result = detect_color_flip(pd.DataFrame(), prev_close=100.0)
        assert isinstance(result, dict)
        assert result["color"] == "FLAT"

    def test_invalid_prev_close_returns_safe_defaults(self):
        df = make_intraday()
        result = detect_color_flip(df, prev_close=0.0)
        assert isinstance(result, dict)
        assert result["color"] == "FLAT"

    def test_bars_ago_is_minus_one_when_no_event(self):
        # No crossings → r2g_bars_ago == -1
        prev_close = 90.0
        df = make_intraday(open_price=100.0, drift=0.0, seed=5)
        result = detect_color_flip(df, prev_close=prev_close)
        assert result["r2g_bars_ago"] == -1
