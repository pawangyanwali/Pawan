import pytest
pytestmark = pytest.mark.slow

"""
Supplementary tests covering smaller untested modules:
  - agent/relative_strength.py
  - agent/signal_blender.py
  - agent/ensemble_model.py
  - agent/walk_forward.py
  - agent/ticker_universe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── OHLCV helper ──────────────────────────────────────────────────────────────

def make_ohlcv(n: int = 50, drift: float = 0.0, seed: int = 42, start: float = 100.0) -> pd.DataFrame:
    """Return a simple OHLCV DataFrame with n bars."""
    rng = np.random.default_rng(seed)
    closes = start + np.cumsum(rng.normal(drift, 0.3, n))
    highs = closes + np.abs(rng.uniform(0.1, 0.3, n))
    lows  = closes - np.abs(rng.uniform(0.1, 0.3, n))
    opens = np.clip(np.roll(closes, 1), lows, highs)
    opens[0] = closes[0]
    idx = pd.date_range("2025-01-10 09:30", periods=n, freq="1min")
    return pd.DataFrame(
        {
            "Open":   opens,
            "High":   highs,
            "Low":    lows,
            "Close":  closes,
            "Volume": np.full(n, 1e6),
        },
        index=idx,
    )


def make_ohlcv_trending(n: int = 10, start: float = 100.0, end_price: float = 110.0) -> pd.DataFrame:
    """Return an OHLCV DataFrame that moves linearly from start to end_price."""
    closes = np.linspace(start, end_price, n)
    highs  = closes + 0.1
    lows   = closes - 0.1
    opens  = np.roll(closes, 1)
    opens[0] = start
    idx = pd.date_range("2025-01-10 09:30", periods=n, freq="1min")
    return pd.DataFrame(
        {
            "Open":   opens,
            "High":   highs,
            "Low":    lows,
            "Close":  closes,
            "Volume": np.full(n, 1e6),
        },
        index=idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestRelativeStrength
# ─────────────────────────────────────────────────────────────────────────────

class TestRelativeStrength:
    """Tests for agent.relative_strength.compute_relative_strength."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from agent.relative_strength import compute_relative_strength
        self.compute = compute_relative_strength

    def test_returns_dict(self):
        df = make_ohlcv(50)
        result = self.compute(df, df)
        assert isinstance(result, dict)

    def test_has_score_and_label_keys(self):
        df = make_ohlcv(50)
        result = self.compute(df, df)
        assert "rs_score" in result
        assert "rs_label" in result

    def test_score_bounded(self):
        df = make_ohlcv(50, drift=0.2)
        spy = make_ohlcv(50, drift=0.0, seed=99)
        result = self.compute(df, spy)
        assert -1.0 <= result["rs_score"] <= 1.0

    def test_outperformer_has_positive_score(self):
        # Stock clearly outperforms: +10% vs SPY +0.1% (non-flat so RS is defined)
        df_stock = make_ohlcv_trending(10, 100.0, 110.0)   # +10%
        df_spy   = make_ohlcv_trending(10, 100.0, 100.1)   # +0.1%
        result = self.compute(df_stock, df_spy)
        assert result["rs_score"] > 0.0

    def test_underperformer_has_negative_score(self):
        # Stock -5% vs SPY +5% → negative RS ratio → negative score
        df_stock = make_ohlcv_trending(10, 100.0, 95.0)    # -5%
        df_spy   = make_ohlcv_trending(10, 100.0, 105.0)   # +5%
        result = self.compute(df_stock, df_spy)
        assert result["rs_score"] < 0.0

    def test_label_is_valid_string(self):
        df = make_ohlcv(50)
        spy = make_ohlcv(50, seed=99)
        result = self.compute(df, spy)
        assert result["rs_label"] in {"LEADING", "IN_LINE", "LAGGING", "COUNTER"}

    def test_too_short_df_returns_safe_defaults(self):
        # A DataFrame with a single row should not raise; fallback defaults apply
        df_short = make_ohlcv(1)
        spy_short = make_ohlcv(1)
        result = self.compute(df_short, spy_short)
        assert isinstance(result, dict)
        assert "rs_label" in result

    def test_has_stock_ret_and_spy_ret(self):
        df = make_ohlcv(50)
        spy = make_ohlcv(50, seed=99)
        result = self.compute(df, spy)
        assert "stock_ret" in result
        assert "spy_ret" in result

    def test_market_flat_returns_neutral_label(self):
        # When SPY has zero intraday return the function sets rs_label = 'IN_LINE'
        df_spy_flat = make_ohlcv_trending(10, 100.0, 100.0)  # no movement
        df_stock = make_ohlcv_trending(10, 100.0, 105.0)
        result = self.compute(df_stock, df_spy_flat)
        # Should return safely and rs_label is a string
        assert isinstance(result["rs_label"], str)


# ─────────────────────────────────────────────────────────────────────────────
# TestBlendSignals
# ─────────────────────────────────────────────────────────────────────────────

class TestBlendSignals:
    """Tests for agent.signal_blender.blend_signals (module-level convenience fn)."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from agent.signal_blender import blend_signals
        self.blend = blend_signals

    def test_returns_float(self):
        result = self.blend(0.5, 0.5, 0.5)
        assert isinstance(result, float)

    def test_all_0_5_returns_near_0_5(self):
        result = self.blend(0.5, 0.5, 0.5)
        assert abs(result - 0.5) < 0.05

    def test_result_bounded_0_to_1(self):
        for scalp, ens, rev in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.3, 0.7, 0.5)]:
            r = self.blend(scalp, ens, rev)
            assert 0.0 <= r <= 1.0

    def test_high_prob_returns_above_0_5(self):
        result = self.blend(0.9, 0.9, 0.9)
        assert result > 0.5

    def test_low_prob_returns_below_0_5(self):
        result = self.blend(0.1, 0.1, 0.1)
        assert result < 0.5

    def test_trained_models_influence_result(self):
        # With swing_trained and deep_trained active (p=1.0), result should shift higher
        base    = self.blend(0.5, 0.5, 0.5, swing_trained=False, deep_trained=False)
        boosted = self.blend(0.5, 0.5, 0.5, swing_p=0.9, deep_p=0.9,
                             swing_trained=True, deep_trained=True)
        # boosted should be > base because optional model probs are high
        assert boosted > base

    def test_optional_untrained_models_excluded(self):
        # swing/deep not trained, their probability value should not affect result
        r1 = self.blend(0.6, 0.6, 0.6, swing_p=0.0, deep_p=0.0,
                        swing_trained=False, deep_trained=False)
        r2 = self.blend(0.6, 0.6, 0.6, swing_p=1.0, deep_p=1.0,
                        swing_trained=False, deep_trained=False)
        assert r1 == pytest.approx(r2, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# TestEnsembleModel
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsembleModel:
    """Tests for agent.ensemble_model functions."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from agent.ensemble_model import (
            MetaEnsemble,
            ensemble_confidence_multiplier,
            get_meta_prediction,
            get_or_create_meta,
        )
        self.mult    = ensemble_confidence_multiplier
        self.get_or  = get_or_create_meta
        self.predict = get_meta_prediction
        self.Meta    = MetaEnsemble

    def test_confidence_multiplier_returns_float(self):
        result = self.mult(0.05)
        assert isinstance(result, float)

    def test_confidence_multiplier_bounded(self):
        for std in [0.0, 0.05, 0.10, 0.18, 0.25]:
            result = self.mult(std)
            assert 0.0 <= result <= 2.0

    def test_confidence_multiplier_lower_on_high_std(self):
        low_std  = self.mult(0.02)
        high_std = self.mult(0.20)
        assert high_std < low_std

    def test_confidence_multiplier_full_agreement(self):
        # Very low std → maximum confidence (1.0)
        assert self.mult(0.0) == 1.0

    def test_confidence_multiplier_high_disagreement(self):
        # std >= AGREE_LOW (0.18) → minimum multiplier (0.4)
        assert self.mult(0.25) == pytest.approx(0.4, abs=1e-6)

    def test_get_or_create_returns_meta_ensemble(self):
        meta = self.get_or("TSLA")
        assert isinstance(meta, self.Meta)

    def test_get_or_create_same_ticker_same_object(self):
        m1 = self.get_or("GOOG")
        m2 = self.get_or("GOOG")
        assert m1 is m2

    def test_get_meta_prediction_returns_tuple(self):
        result = self.predict("AAPL")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_meta_prediction_probability_bounded(self):
        prob, mult = self.predict("NVDA", scalp_prob=0.7, ensemble_prob=0.6,
                                  daily_prob=0.65, reversal_prob=0.55)
        assert 0.0 <= prob <= 1.0

    def test_get_meta_prediction_mult_bounded(self):
        prob, mult = self.predict("AMZN")
        assert 0.0 <= mult <= 2.0


# ─────────────────────────────────────────────────────────────────────────────
# TestWalkForward
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForward:
    """Tests for agent.walk_forward.build_training_df and compute_win_rate_by_context."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from agent.walk_forward import build_training_df, compute_win_rate_by_context
        from agent.feature_engine import FEATURE_COLS_V2
        self.build_df = build_training_df
        self.win_rate = compute_win_rate_by_context
        self.FEAT     = FEATURE_COLS_V2

    def _make_records(self, n: int = 6, direction: str = "BUY") -> list[dict]:
        """Build minimal fake records with all required feature columns."""
        records = []
        for i in range(n):
            r: dict = {
                "ticker":    "AAPL",
                "bar_dt":    "2025-01-10",
                "direction": direction,
                "pnl_r":     1.0 if i % 2 == 0 else -0.5,
                "won":       i % 2 == 0,
                "outcome":   "HIT_TARGET" if i % 2 == 0 else "HIT_STOP",
                "rsi_14":    30.0 if direction == "BUY" else 70.0,
            }
            for col in self.FEAT:
                r.setdefault(col, 0.5)
            records.append(r)
        return records

    def test_build_training_df_returns_dataframe(self):
        records = self._make_records()
        result = self.build_df(records)
        assert isinstance(result, pd.DataFrame)

    def test_build_training_df_empty_input_returns_empty(self):
        result = self.build_df([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_build_training_df_has_won_column(self):
        records = self._make_records()
        df = self.build_df(records)
        assert "won" in df.columns

    def test_build_training_df_row_count_matches_records(self):
        records = self._make_records(8)
        df = self.build_df(records)
        assert len(df) == 8

    def test_compute_win_rate_returns_dict(self):
        records = self._make_records(10)
        result = self.win_rate(records)
        assert isinstance(result, dict)

    def test_compute_win_rate_empty_returns_empty_dict(self):
        result = self.win_rate([])
        assert result == {}

    def test_compute_win_rate_keys_are_context_strings(self):
        records = self._make_records(10, direction="BUY")
        result = self.win_rate(records)
        for key in result:
            assert isinstance(key, str)
            # Keys follow "context:value" pattern
            assert ":" in key

    def test_compute_win_rate_values_are_dicts(self):
        records = self._make_records(10, direction="BUY")
        result = self.win_rate(records)
        for val in result.values():
            assert isinstance(val, dict)
            assert "win_rate" in val
            assert "count" in val


# ─────────────────────────────────────────────────────────────────────────────
# TestTickerUniverse
# ─────────────────────────────────────────────────────────────────────────────

class TestTickerUniverse:
    """Tests for agent.ticker_universe.get_universe and get_active_tickers_universe."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from agent.ticker_universe import get_active_tickers_universe, get_universe
        self.get_universe = get_universe
        self.get_active   = get_active_tickers_universe

    def test_get_universe_returns_list(self):
        result = self.get_universe()
        assert isinstance(result, list)

    def test_universe_nonempty(self):
        result = self.get_universe()
        assert len(result) > 0

    def test_all_tickers_are_strings(self):
        result = self.get_universe()
        for ticker in result:
            assert isinstance(ticker, str)

    def test_get_active_tickers_universe_returns_list(self):
        result = self.get_active()
        assert isinstance(result, list)

    def test_universe_contains_known_tickers(self):
        result = self.get_universe()
        # These are in TIER1 so must always be present
        assert "AAPL" in result
        assert "MSFT" in result
        assert "NVDA" in result

    def test_universe_has_no_empty_strings(self):
        result = self.get_universe()
        for ticker in result:
            assert ticker.strip() != ""

    def test_get_active_contains_tier1(self):
        # Tier 1 is always included in the active universe
        active = self.get_active()
        # At minimum, AAPL (top TIER1) should appear
        assert "AAPL" in active

    def test_universe_returns_copy(self):
        # Modifying the returned list should not affect subsequent calls
        u1 = self.get_universe()
        original_len = len(u1)
        u1.append("FAKE_TICKER_XYZ")
        u2 = self.get_universe()
        assert len(u2) == original_len
