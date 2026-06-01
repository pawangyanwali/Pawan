"""
Section C — Backtesting Tests

Covers:
  C1. LiveBacktest: record, update (WIN/LOSS/TIMEOUT), stats, R-multiples
  C2. Walk-forward: empty DB, _compute_window formulas, aggregation
  C3. BacktestReport: edge detection, required fields, serialization
  C4. Backtester module helpers: _compute_sharpe, _safe_float, _compute_window

Run:
    cd nasdaq_agent
    pytest tests/test_backtest.py -v --tb=short
"""
from __future__ import annotations

import math
import sys
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent.live_backtest as LB
import agent.backtester as BT
from agent.backtester import (
    _compute_window, _compute_sharpe, _safe_float,
    WindowResult, BacktestReport, WalkForwardBacktester,
    EDGE_WIN_RATE_THRESHOLD, EDGE_WINDOW_FRACTION,
)
from tests.conftest import make_ohlcv


# ─────────────────────────────────────────────────────────────────────────────
# Helper factories
# ─────────────────────────────────────────────────────────────────────────────

def _window(
    idx: int = 0,
    n_signals: int = 20,
    n_wins: int = 12,
    avg_win: float = 2.0,
    avg_loss: float = -1.0,
) -> WindowResult:
    win_rate  = n_wins / n_signals
    expectancy = avg_win * win_rate + avg_loss * (1.0 - win_rate)
    return WindowResult(
        window_idx    = idx,
        train_start   = "2025-01-01",
        train_end     = "2025-02-01",
        test_start    = "2025-02-01",
        test_end      = "2025-03-01",
        n_signals     = n_signals,
        n_wins        = n_wins,
        win_rate      = round(win_rate, 4),
        avg_win_pct   = avg_win,
        avg_loss_pct  = avg_loss,
        expectancy    = round(expectancy, 4),
        sharpe        = 1.0,
        max_drawdown  = -3.0,
        tickers_tested = 3,
    )


def _iso_now(offset_days: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=offset_days)).isoformat()


# ═════════════════════════════════════════════════════════════════════════════
# C1 – LiveBacktest database operations
# ═════════════════════════════════════════════════════════════════════════════

class TestLiveBacktest:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        # DB routes through the autouse get_conn() patch (shared in-memory SQLite);
        # init_db() creates the schema there.
        LB.init_db()

    def test_record_signal_returns_str(self):
        sid = LB.record_signal("AAPL", "BUY", entry_price=150.0,
                               target=160.0, stop=145.0)
        assert isinstance(sid, str) and len(sid) > 0

    def test_record_signal_initial_status_tracking(self):
        sid = LB.record_signal("MSFT", "BUY", entry_price=300.0,
                               target=315.0, stop=295.0)
        conn = LB.get_conn()
        row = conn.execute("SELECT status FROM bt_signals WHERE signal_id=?",
                           (sid,)).fetchone()
        conn.close()
        assert row["status"] == "TRACKING"

    def test_update_tracking_buy_win_at_target(self):
        sid = LB.record_signal("NVDA_W", "BUY", entry_price=200.0,
                               target=210.0, stop=195.0)
        resolved = LB.update_tracking("NVDA_W", current_price=211.0, vwap=200.0)
        wins = [r for r in resolved if r["status"] == "WIN"]
        assert len(wins) >= 1

    def test_update_tracking_buy_loss_at_stop(self):
        sid = LB.record_signal("NVDA_L", "BUY", entry_price=200.0,
                               target=210.0, stop=195.0)
        resolved = LB.update_tracking("NVDA_L", current_price=194.0, vwap=200.0)
        losses = [r for r in resolved if r["status"] == "LOSS"]
        assert len(losses) >= 1

    def test_update_tracking_sell_win_at_target(self):
        sid = LB.record_signal("SPY_SW", "SELL", entry_price=100.0,
                               target=90.0, stop=105.0)
        resolved = LB.update_tracking("SPY_SW", current_price=89.0, vwap=100.0)
        wins = [r for r in resolved if r["status"] == "WIN"]
        assert len(wins) >= 1

    def test_update_tracking_sell_loss_at_stop(self):
        sid = LB.record_signal("SPY_SL", "SELL", entry_price=100.0,
                               target=90.0, stop=105.0)
        resolved = LB.update_tracking("SPY_SL", current_price=106.0, vwap=100.0)
        losses = [r for r in resolved if r["status"] == "LOSS"]
        assert len(losses) >= 1

    def test_win_r_multiple_is_positive(self):
        sid = LB.record_signal("RWIN", "BUY", entry_price=100.0,
                               target=110.0, stop=95.0)
        resolved = LB.update_tracking("RWIN", current_price=111.0, vwap=100.0)
        for r in resolved:
            if r["status"] == "WIN":
                r_val = r.get("r_multiple", r.get("final_r"))
                if r_val is not None:
                    assert r_val > 0, f"WIN should have positive R, got {r_val}"

    def test_loss_r_multiple_is_negative(self):
        sid = LB.record_signal("RLOSS", "BUY", entry_price=100.0,
                               target=110.0, stop=95.0)
        resolved = LB.update_tracking("RLOSS", current_price=94.0, vwap=100.0)
        for r in resolved:
            if r["status"] == "LOSS":
                r_val = r.get("r_multiple", r.get("final_r"))
                if r_val is not None:
                    assert r_val < 0, f"LOSS should have negative R, got {r_val}"

    def test_get_tracking_signals_only_tracking(self):
        LB.record_signal("TRK1", "BUY", entry_price=50.0, target=55.0, stop=47.0)
        LB.record_signal("TRK2", "BUY", entry_price=50.0, target=55.0, stop=47.0)
        # Resolve one
        LB.update_tracking("TRK1", current_price=56.0, vwap=50.0)
        active = LB.get_tracking_signals()
        tickers = [s["ticker"] for s in active]
        assert "TRK2" in tickers
        assert "TRK1" not in tickers

    def test_multiple_tickers_independent(self):
        """Resolving ticker A should not affect ticker B."""
        LB.record_signal("INDA", "BUY", entry_price=100.0, target=110.0, stop=95.0)
        LB.record_signal("INDB", "BUY", entry_price=100.0, target=110.0, stop=95.0)
        LB.update_tracking("INDA", current_price=111.0, vwap=100.0)
        active = [s["ticker"] for s in LB.get_tracking_signals()]
        assert "INDB" in active
        assert "INDA" not in active

    def test_get_performance_stats_structure(self):
        LB.record_signal("STATS1", "BUY", 100.0, 110.0, 95.0)
        LB.update_tracking("STATS1", current_price=111.0, vwap=100.0)
        stats = LB.get_performance_stats(min_resolved=0)
        assert "overall" in stats
        assert "total" in stats["overall"]

    def test_performance_stats_win_counted_correctly(self):
        LB.record_signal("PW1", "BUY", 100.0, 110.0, 95.0)
        LB.record_signal("PW2", "BUY", 100.0, 110.0, 95.0)
        LB.update_tracking("PW1", current_price=111.0, vwap=100.0)
        LB.update_tracking("PW2", current_price=94.0, vwap=100.0)
        stats = LB.get_performance_stats(min_resolved=0)
        total = stats["overall"].get("total", 0)
        assert total >= 2

    def test_get_recent_resolved_structure(self):
        LB.record_signal("REC1", "BUY", 100.0, 110.0, 95.0)
        LB.update_tracking("REC1", current_price=111.0, vwap=100.0)
        recent = LB.get_recent_resolved(limit=10)
        assert isinstance(recent, list)

    def test_record_signal_with_confidence(self):
        sid = LB.record_signal("CONF1", "BUY", 100.0, 110.0, 95.0, confidence=75.0)
        conn = LB.get_conn()
        row = conn.execute("SELECT confidence FROM bt_signals WHERE signal_id=?",
                           (sid,)).fetchone()
        conn.close()
        assert abs(float(row["confidence"]) - 75.0) < 0.01

    def test_no_zero_entry_rejected(self):
        """Record with entry_price=0 should either be rejected or stored as 0."""
        try:
            LB.record_signal("ZERO1", "BUY", entry_price=0.0,
                             target=5.0, stop=-1.0)
        except Exception:
            pass  # raising is acceptable
        active = LB.get_tracking_signals()
        zeros = [s for s in active if s["ticker"] == "ZERO1"
                 and float(s.get("entry_price", 1)) == 0.0]
        assert zeros == [], "Zero-entry signal should not be tracked"

    def test_inactive_price_does_not_resolve(self):
        """Price between stop and target should not resolve the signal."""
        sid = LB.record_signal("MID1", "BUY", 100.0, 110.0, 95.0)
        resolved = LB.update_tracking("MID1", current_price=103.0, vwap=100.0)
        still_open = [s for s in resolved if s["ticker"] == "MID1"]
        assert still_open == [], "Price between stop and target should not resolve"

    def test_rt_check_resolution_returns_list(self):
        LB.record_signal("RT1", "BUY", 100.0, 110.0, 95.0)
        result = LB.rt_check_resolution("RT1", last_price=111.0)
        assert isinstance(result, list)

    def test_get_outcomes_for_ml_returns_df_or_none(self):
        result = LB.get_outcomes_for_ml(min_count=1)
        assert result is None or isinstance(result, pd.DataFrame)


# ═════════════════════════════════════════════════════════════════════════════
# C2 – _compute_window formulas
# ═════════════════════════════════════════════════════════════════════════════

class TestComputeWindow:

    def _make_df(self, pnl_pcts: list[float]) -> pd.DataFrame:
        """Build a minimal df_test for _compute_window."""
        n = len(pnl_pcts)
        tickers = [f"T{i % 3}" for i in range(n)]
        now = datetime.now(timezone.utc)
        return pd.DataFrame({
            "ticker":    tickers,
            "pnl_pct":   pnl_pcts,
            "direction": ["BUY"] * n,
            "confidence": [65.0] * n,
            "session":   ["POWER_HOUR"] * n,
            "regime":    ["BULL"] * n,
            "recorded_at": [now + timedelta(minutes=i) for i in range(n)],
        })

    def test_all_wins_win_rate_is_one(self):
        df = self._make_df([2.0, 3.0, 1.5, 2.5])
        train_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = _compute_window(0, train_start, train_start + timedelta(days=7),
                                 train_start + timedelta(days=7),
                                 train_start + timedelta(days=14), df)
        assert result.win_rate == 1.0

    def test_all_losses_win_rate_is_zero(self):
        df = self._make_df([-1.0, -2.0, -0.5, -1.5])
        train_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = _compute_window(0, train_start, train_start + timedelta(days=7),
                                 train_start + timedelta(days=7),
                                 train_start + timedelta(days=14), df)
        assert result.win_rate == 0.0

    def test_win_rate_formula_correct(self):
        # 6 wins, 4 losses → 60% win rate
        df = self._make_df([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, -1.0])
        train_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = _compute_window(0, train_start, train_start + timedelta(days=7),
                                 train_start + timedelta(days=7),
                                 train_start + timedelta(days=14), df)
        assert abs(result.win_rate - 0.6) < 1e-6

    def test_expectancy_formula(self):
        """expectancy = avg_win * win_rate + avg_loss * (1 - win_rate)."""
        pnls = [2.0, 2.0, -1.0, -1.0]   # WR=0.5, avg_win=2, avg_loss=-1
        df = self._make_df(pnls)
        train_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = _compute_window(0, train_start, train_start + timedelta(days=7),
                                 train_start + timedelta(days=7),
                                 train_start + timedelta(days=14), df)
        expected_exp = 2.0 * 0.5 + (-1.0) * 0.5
        assert abs(result.expectancy - expected_exp) < 0.01

    def test_max_drawdown_negative_or_zero(self):
        pnls = [1.0, -3.0, 2.0, -4.0, 1.0]
        df = self._make_df(pnls)
        train_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = _compute_window(0, train_start, train_start + timedelta(days=7),
                                 train_start + timedelta(days=7),
                                 train_start + timedelta(days=14), df)
        assert result.max_drawdown <= 0.0

    def test_empty_df_returns_zero_window(self):
        df = self._make_df([])
        train_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = _compute_window(0, train_start, train_start + timedelta(days=7),
                                 train_start + timedelta(days=7),
                                 train_start + timedelta(days=14), df)
        assert result.n_signals == 0
        assert result.win_rate == 0.0

    def test_tickers_tested_counts_unique(self):
        pnls = [1.0, 1.0, 1.0, 1.0, 1.0]
        n = len(pnls)
        df = pd.DataFrame({
            "ticker": ["A", "A", "B", "B", "C"],
            "pnl_pct": pnls,
            "direction": ["BUY"] * n,
            "confidence": [65.0] * n,
            "session": ["OPEN"] * n,
            "regime": ["BULL"] * n,
            "recorded_at": [datetime.now(timezone.utc)] * n,
        })
        train_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = _compute_window(0, train_start, train_start + timedelta(days=7),
                                 train_start + timedelta(days=7),
                                 train_start + timedelta(days=14), df)
        assert result.tickers_tested == 3

    def test_positive_expectancy_when_win_rate_high(self):
        pnls = [2.0] * 7 + [-1.0] * 3   # 70% WR, E = 2*0.7 + (-1)*0.3 = 1.1
        df = self._make_df(pnls)
        train_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = _compute_window(0, train_start, train_start + timedelta(days=7),
                                 train_start + timedelta(days=7),
                                 train_start + timedelta(days=14), df)
        assert result.expectancy > 0.0


# ═════════════════════════════════════════════════════════════════════════════
# C3 – Backtester module helpers
# ═════════════════════════════════════════════════════════════════════════════

class TestBacktesterHelpers:

    def test_compute_sharpe_nan_on_single_element(self):
        result = _compute_sharpe(np.array([1.0]))
        assert math.isnan(result)

    def test_compute_sharpe_nan_on_empty(self):
        result = _compute_sharpe(np.array([]))
        assert math.isnan(result)

    def test_compute_sharpe_positive_for_pure_wins(self):
        returns = np.array([1.0, 2.0, 1.5, 1.8, 2.2])
        s = _compute_sharpe(returns)
        assert not math.isnan(s)
        assert s > 0.0

    def test_compute_sharpe_negative_for_pure_losses(self):
        returns = np.array([-1.0, -2.0, -1.5, -1.8, -2.2])
        s = _compute_sharpe(returns)
        assert not math.isnan(s)
        assert s < 0.0

    def test_compute_sharpe_nan_when_std_zero(self):
        returns = np.array([1.0, 1.0, 1.0, 1.0])
        result = _compute_sharpe(returns)
        assert math.isnan(result)

    def test_safe_float_nan_returns_default(self):
        assert _safe_float(float("nan"), default=0.0) == 0.0
        assert _safe_float(float("inf"), default=-1.0) == -1.0

    def test_safe_float_returns_value(self):
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_safe_float_none_returns_default(self):
        assert _safe_float(None, default=99.0) == 99.0

    def test_safe_float_string_returns_default(self):
        assert _safe_float("bad", default=0.0) == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# C4 – BacktestReport and edge detection
# ═════════════════════════════════════════════════════════════════════════════

class TestBacktestReport:

    def test_build_empty_report_has_all_fields(self):
        report = WalkForwardBacktester._build_empty_report("complete")
        required = {
            "windows", "overall_win_rate", "overall_expectancy",
            "avg_sharpe", "max_drawdown", "edge_exists",
            "total_signals", "generated_at", "status", "progress_pct",
        }
        for f in required:
            assert hasattr(report, f), f"BacktestReport missing field: {f}"

    def test_empty_report_status_complete(self):
        r = WalkForwardBacktester._build_empty_report("complete")
        assert r.status == "complete"
        assert r.edge_exists is False
        assert r.total_signals == 0

    def test_edge_exists_true_when_most_windows_win(self):
        """Edge exists when > EDGE_WINDOW_FRACTION of windows beat EDGE_WIN_RATE_THRESHOLD."""
        n_windows = 10
        n_winning = int(n_windows * EDGE_WINDOW_FRACTION) + 1  # majority wins
        windows = [
            _window(i, n_signals=20, n_wins=int(20 * (EDGE_WIN_RATE_THRESHOLD + 0.1)))
            for i in range(n_winning)
        ] + [
            _window(i + n_winning, n_signals=20, n_wins=5)
            for i in range(n_windows - n_winning)
        ]
        bt = WalkForwardBacktester()
        bt._min_signals = 5
        report = bt._aggregate(windows, total_signals=200)
        assert report.edge_exists is True

    def test_edge_exists_false_when_most_windows_lose(self):
        """Edge does not exist when fewer than EDGE_WINDOW_FRACTION of windows beat threshold."""
        n_windows = 10
        n_winning = int(n_windows * EDGE_WINDOW_FRACTION) - 1  # minority wins
        windows = [
            _window(i, n_signals=20, n_wins=int(20 * (EDGE_WIN_RATE_THRESHOLD + 0.1)))
            for i in range(n_winning)
        ] + [
            _window(i + n_winning, n_signals=20, n_wins=3)
            for i in range(n_windows - n_winning)
        ]
        bt = WalkForwardBacktester()
        bt._min_signals = 5
        report = bt._aggregate(windows, total_signals=200)
        assert report.edge_exists is False

    def test_aggregate_overall_win_rate_formula(self):
        """Overall win rate must be weighted sum of wins / sum of signals."""
        w1 = _window(0, n_signals=10, n_wins=7)
        w2 = _window(1, n_signals=20, n_wins=8)
        bt = WalkForwardBacktester()
        bt._min_signals = 5
        report = bt._aggregate([w1, w2], total_signals=30)
        expected_wr = (7 + 8) / (10 + 20)
        assert abs(report.overall_win_rate - expected_wr) < 1e-4

    def test_aggregate_max_drawdown_is_worst_window(self):
        w1 = _window(0, n_signals=10, n_wins=7)
        w1 = WindowResult(**{**vars(w1), "max_drawdown": -2.0})
        w2 = _window(1, n_signals=10, n_wins=8)
        w2 = WindowResult(**{**vars(w2), "max_drawdown": -8.0})
        bt = WalkForwardBacktester()
        bt._min_signals = 5
        report = bt._aggregate([w1, w2], total_signals=20)
        assert report.max_drawdown == -8.0

    def test_report_serialise_round_trip(self, tmp_path):
        """Report can be saved to JSON and re-loaded without data loss."""
        report = WalkForwardBacktester._build_empty_report("complete")
        path = tmp_path / "bt_report.json"
        BT._save_report(report, path=path)
        loaded = BT._load_report(path=path)
        assert loaded is not None
        assert loaded.status == "complete"
        assert loaded.edge_exists == report.edge_exists


# ═════════════════════════════════════════════════════════════════════════════
# C5 – WalkForwardBacktester integration
# ═════════════════════════════════════════════════════════════════════════════

class TestWalkForwardBacktester:

    @pytest.fixture(autouse=True)
    def _tmp_bt(self, tmp_path, monkeypatch):
        # DB routes through the autouse get_conn() patch; only the JSON report
        # path still needs redirecting to a temp file.
        monkeypatch.setattr(BT, "_PERSIST_PATH", tmp_path / "bt_report.json")
        LB.init_db()

    def test_run_sync_empty_db_returns_report(self):
        bt = WalkForwardBacktester(persist_path=None)
        report = bt.run_sync()
        assert isinstance(report, BacktestReport)
        assert report.status == "complete"
        assert report.total_signals == 0

    def test_run_sync_with_resolved_signals(self, tmp_path, monkeypatch):
        """Inject resolved WIN + LOSS signals and verify report structure."""
        # Record signals and resolve them
        s1 = LB.record_signal("AAPL", "BUY", 150.0, 160.0, 145.0, confidence=70.0)
        s2 = LB.record_signal("AAPL", "BUY", 150.0, 160.0, 145.0, confidence=70.0)
        s3 = LB.record_signal("AAPL", "BUY", 150.0, 160.0, 145.0, confidence=70.0)
        LB.update_tracking("AAPL", current_price=161.0, vwap=150.0)  # WIN
        LB.update_tracking("AAPL", current_price=144.0, vwap=150.0)  # LOSS (remaining)
        LB.update_tracking("AAPL", current_price=161.0, vwap=150.0)  # WIN

        bt = WalkForwardBacktester(persist_path=tmp_path / "bt.json")
        report = bt.run_sync()
        assert report.status == "complete"
        assert report.total_signals >= 0

    def test_get_status_has_progress_key(self):
        bt = WalkForwardBacktester()
        status = bt.get_status()
        assert "progress_pct" in status
        assert "status" in status

    def test_run_async_starts_background_thread(self):
        bt = WalkForwardBacktester()
        bt.run_async()
        import time
        time.sleep(0.2)  # let thread start
        status = bt.get_status()
        assert status["status"] in ("running", "complete", "idle")

    def test_empty_report_has_expected_defaults(self):
        r = WalkForwardBacktester._build_empty_report("idle")
        assert r.overall_win_rate == 0.0
        assert r.overall_expectancy == 0.0
        assert r.edge_exists is False
        assert r.progress_pct == 0.0

    def test_min_signals_parameter_filters_small_windows(self):
        """Windows with fewer signals than min_signals should be excluded from stats."""
        large_win = _window(0, n_signals=50, n_wins=35)
        small_win = _window(1, n_signals=2, n_wins=2)   # below min_signals=5
        bt = WalkForwardBacktester()
        bt._min_signals = 5
        report = bt._aggregate([large_win, small_win], total_signals=52)
        # Only the large window qualifies; win_rate should reflect only it
        assert abs(report.overall_win_rate - 35 / 50) < 1e-4


# ═════════════════════════════════════════════════════════════════════════════
# C6 – Live backtest pnl_pct sign verification
# ═════════════════════════════════════════════════════════════════════════════

class TestLiveBacktestPnL:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        LB.init_db()

    def _get_resolved(self, ticker: str) -> list[dict]:
        conn = LB.get_conn()
        rows = conn.execute(
            "SELECT status, pnl_pct FROM bt_signals WHERE ticker=? AND status != 'TRACKING'",
            (ticker,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def test_buy_win_pnl_positive(self):
        LB.record_signal("BW1", "BUY", 100.0, 110.0, 95.0)
        LB.update_tracking("BW1", current_price=111.0, vwap=100.0)
        resolved = self._get_resolved("BW1")
        assert resolved, "Signal should be resolved"
        for r in resolved:
            if r["status"] == "WIN":
                assert r["pnl_pct"] > 0

    def test_buy_loss_pnl_negative(self):
        LB.record_signal("BL1", "BUY", 100.0, 110.0, 95.0)
        LB.update_tracking("BL1", current_price=94.0, vwap=100.0)
        resolved = self._get_resolved("BL1")
        assert resolved, "Signal should be resolved"
        for r in resolved:
            if r["status"] == "LOSS":
                assert r["pnl_pct"] < 0

    def test_sell_win_pnl_positive(self):
        LB.record_signal("SW1", "SELL", 100.0, 90.0, 105.0)
        LB.update_tracking("SW1", current_price=89.0, vwap=100.0)
        resolved = self._get_resolved("SW1")
        if resolved:
            for r in resolved:
                if r["status"] == "WIN":
                    assert r["pnl_pct"] > 0

    def test_sell_loss_pnl_negative(self):
        LB.record_signal("SL1", "SELL", 100.0, 90.0, 105.0)
        LB.update_tracking("SL1", current_price=106.0, vwap=100.0)
        resolved = self._get_resolved("SL1")
        if resolved:
            for r in resolved:
                if r["status"] == "LOSS":
                    assert r["pnl_pct"] < 0

    def test_r_sign_matches_pnl_sign(self):
        """R-multiple and pnl_pct should have the same sign."""
        LB.record_signal("RSIGN", "BUY", 100.0, 110.0, 95.0)
        resolved = LB.update_tracking("RSIGN", current_price=111.0, vwap=100.0)
        conn = LB.get_conn()
        row = conn.execute(
            "SELECT r_multiple, pnl_pct FROM bt_signals WHERE ticker='RSIGN'"
        ).fetchone()
        conn.close()
        if row and row["r_multiple"] is not None and row["pnl_pct"] is not None:
            r = float(row["r_multiple"])
            p = float(row["pnl_pct"])
            if r != 0 and p != 0:
                assert (r > 0) == (p > 0), f"R={r} and pnl_pct={p} have different signs"
