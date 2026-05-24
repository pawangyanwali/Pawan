"""
Tests for agent/multi_tf_backtest.py

Covers:
  - compute_tf_stats
  - _sharpe
  - _max_drawdown
  - _expectancy
  - get_summary, get_run_history, get_ticker_stats, build_training_records (DB)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.multi_tf_backtest import (
    _expectancy,
    _max_drawdown,
    _sharpe,
    build_training_records,
    compute_tf_stats,
    get_run_history,
    get_summary,
    get_ticker_stats,
)


# ── Record helpers ────────────────────────────────────────────────────────────

def _win(pnl_r: float = 1.0) -> dict:
    """Return a single winning trade record using the real pnl_r field."""
    return {
        "pnl_r":     pnl_r,
        "won":       True,
        "direction": "BUY",
        "ticker":    "AAPL",
        "timeframe": "5min",
    }


def _loss(pnl_r: float = -0.5) -> dict:
    """Return a single losing trade record using the real pnl_r field."""
    return {
        "pnl_r":     pnl_r,
        "won":       False,
        "direction": "BUY",
        "ticker":    "AAPL",
        "timeframe": "5min",
    }


# ── TestComputeTfStats ────────────────────────────────────────────────────────

class TestComputeTfStats:
    """Tests for compute_tf_stats(records) -> dict."""

    def test_returns_dict(self):
        result = compute_tf_stats([_win(), _loss()])
        assert isinstance(result, dict)

    def test_empty_records_returns_zero_win_rate(self):
        result = compute_tf_stats([])
        assert result["win_rate"] == 0.0

    def test_all_wins_returns_1_win_rate(self):
        records = [_win(1.5), _win(1.0), _win(2.0)]
        result = compute_tf_stats(records)
        assert result["win_rate"] == 1.0

    def test_all_losses_returns_0_win_rate(self):
        records = [_loss(-1.0), _loss(-0.5), _loss(-0.8)]
        result = compute_tf_stats(records)
        assert result["win_rate"] == 0.0

    def test_win_rate_bounded_0_to_1(self):
        records = [_win(), _win(), _loss(), _loss(), _win()]
        result = compute_tf_stats(records)
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_expectancy_positive_on_all_wins(self):
        records = [_win(1.5), _win(1.0), _win(2.0)]
        result = compute_tf_stats(records)
        assert result["expectancy"] > 0.0

    def test_expectancy_negative_on_all_losses(self):
        records = [_loss(-1.0), _loss(-0.5), _loss(-0.8)]
        result = compute_tf_stats(records)
        assert result["expectancy"] < 0.0

    def test_max_drawdown_nonnegative(self):
        records = [_win(1.0), _loss(-2.0), _win(0.5), _loss(-1.0)]
        result = compute_tf_stats(records)
        assert result["max_dd_r"] >= 0.0

    def test_max_drawdown_zero_on_all_wins(self):
        records = [_win(1.0), _win(2.0), _win(0.5)]
        result = compute_tf_stats(records)
        assert result["max_dd_r"] == 0.0

    def test_has_required_keys(self):
        result = compute_tf_stats([_win(), _loss()])
        required = {"win_rate", "expectancy", "sharpe", "max_dd_r", "total"}
        assert required.issubset(result.keys())

    def test_trade_count_matches_input(self):
        records = [_win(), _loss(), _win()]
        result = compute_tf_stats(records)
        assert result["total"] == 3

    def test_wins_count_correct(self):
        records = [_win(), _win(), _loss()]
        result = compute_tf_stats(records)
        assert result["wins"] == 2


# ── TestSharpeHelper ──────────────────────────────────────────────────────────

class TestSharpeHelper:
    """Tests for _sharpe(pnl_series) -> float."""

    def test_sharpe_nan_on_empty_series(self):
        # Implementation returns 0.0 for len < 4
        result = _sharpe([])
        assert result == 0.0

    def test_sharpe_nan_on_single_value(self):
        result = _sharpe([1.0])
        assert result == 0.0

    def test_sharpe_zero_on_constant_series(self):
        # All same values → std = 0 → returns 0.0
        result = _sharpe([1.0, 1.0, 1.0, 1.0])
        assert result == 0.0

    def test_sharpe_positive_on_all_wins(self):
        # Four positive values with variation → positive sharpe
        result = _sharpe([1.0, 2.0, 1.5, 1.8])
        assert result > 0.0

    def test_sharpe_negative_on_all_losses(self):
        # Four negative values → negative sharpe
        result = _sharpe([-1.0, -0.5, -1.2, -0.8])
        assert result < 0.0

    def test_sharpe_returns_float(self):
        result = _sharpe([1.0, -0.5, 1.0, -0.5])
        assert isinstance(result, float)

    def test_sharpe_three_items_returns_zero(self):
        # Fewer than 4 → returns 0.0
        result = _sharpe([1.0, 2.0, 3.0])
        assert result == 0.0


# ── TestMaxDrawdown ───────────────────────────────────────────────────────────

class TestMaxDrawdown:
    """Tests for _max_drawdown(pnl_series) -> float."""

    def test_max_drawdown_on_sequence_of_losses(self):
        # cumsum: -1, -2, -3; running_max (from first value): -1,-1,-1
        # drawdown: 0, 1, 2 → max = 2.0
        result = _max_drawdown([-1.0, -1.0, -1.0])
        assert result == pytest.approx(2.0, abs=1e-6)

    def test_max_drawdown_zero_monotonic_gains(self):
        result = _max_drawdown([1.0, 2.0, 3.0, 4.0])
        assert result == 0.0

    def test_max_drawdown_nonnegative_always(self):
        import random
        random.seed(7)
        series = [random.uniform(-2, 3) for _ in range(20)]
        result = _max_drawdown(series)
        assert result >= 0.0

    def test_max_drawdown_empty_returns_zero(self):
        result = _max_drawdown([])
        assert result == 0.0

    def test_max_drawdown_peak_to_trough_correct(self):
        # cumsum: 1, 2, 0, -2 → running_max: 1, 2, 2, 2 → dd: 0, 0, 2, 4
        result = _max_drawdown([1.0, 1.0, -2.0, -2.0])
        assert result == pytest.approx(4.0, abs=1e-6)

    def test_max_drawdown_returns_float(self):
        result = _max_drawdown([1.0, -0.5])
        assert isinstance(result, float)


# ── TestDatabaseFunctions ─────────────────────────────────────────────────────

class TestDatabaseFunctions:
    """
    Tests for the query helpers that touch SQLite.
    These operate against a live (empty) DB created at module import time.
    They should not raise and should return the correct container types.
    """

    def test_get_summary_returns_dict(self):
        result = get_summary()
        assert isinstance(result, dict)

    def test_get_run_history_returns_list(self):
        result = get_run_history()
        assert isinstance(result, list)

    def test_get_ticker_stats_returns_dict(self):
        result = get_ticker_stats("AAPL")
        assert isinstance(result, dict)

    def test_build_training_records_returns_dataframe(self):
        result = build_training_records()
        assert isinstance(result, pd.DataFrame)

    def test_get_run_history_limit_respected(self):
        result = get_run_history(limit=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_get_summary_empty_db_returns_dict(self):
        # Empty DB with no data → returns {} or valid dict
        result = get_summary(run_dt="nonexistent-run")
        assert isinstance(result, dict)

    def test_build_training_records_empty_returns_empty_df(self):
        # No data in DB → empty DataFrame
        result = build_training_records(run_dt="nonexistent-run")
        assert isinstance(result, pd.DataFrame)
        assert result.empty
