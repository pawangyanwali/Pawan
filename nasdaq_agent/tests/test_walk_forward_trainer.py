import pytest
pytestmark = pytest.mark.slow

"""
Comprehensive tests for Phase 3 Part 1: WalkForwardTrainer.

Tests cover:
  1. run() with synthetic OHLCV DataFrames (no API calls)
  2. Parameter recommendations applied per win-rate thresholds
  3. Low win-rate triggers conf_gate increase
  4. High win-rate triggers rvol_gate decrease
  5. Negative avg_pnl_r triggers target_mult decrease
  6. High avg_pnl_r triggers target_mult increase
  7. get_status() returns expected keys
  8. Empty OHLCV graceful handling (no crash)
  9. All-NaN OHLCV graceful handling (no crash)
 10. Singleton pattern works correctly
 11+ Additional edge-case coverage
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── helpers ────────────────────────────────────────────────────────────────────

def _make_ohlcv(
    n: int = 200,
    start: float = 100.0,
    freq: str = "5min",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic OHLCV DataFrame with `n` bars suitable for replay_signals.
    The DatetimeIndex starts on a weekday (Monday 09:45) in market hours.
    Columns are lowercase: open, high, low, close, volume.
    """
    rng = np.random.default_rng(seed)
    closes = [start]
    for _ in range(n - 1):
        closes.append(max(1.0, closes[-1] + rng.normal(0, 0.5)))
    closes = np.array(closes)
    noise  = rng.uniform(0.1, 0.5, n)
    highs  = closes + noise
    lows   = np.maximum(0.1, closes - noise)
    opens  = np.clip(np.roll(closes, 1), lows, highs)
    opens[0] = start

    idx = pd.date_range("2025-01-06 09:45", periods=n, freq=freq, tz="America/New_York")
    return pd.DataFrame({
        "open":   np.round(opens, 4),
        "high":   np.round(highs, 4),
        "low":    np.round(lows,  4),
        "close":  np.round(closes, 4),
        "volume": rng.integers(500_000, 2_000_000, n),
    }, index=idx)


def _make_ohlcv_map(tickers, n=200, seed=42):
    """Build a dict ohlcv_map[ticker][tf] = DataFrame."""
    return {
        ticker: {
            "5min":  _make_ohlcv(n=n, freq="5min",  seed=seed),
            "15min": _make_ohlcv(n=n, freq="15min", seed=seed + 1),
        }
        for ticker in tickers
    }


# A pre-baked list of walk-forward-style records for unit testing the
# aggregation / recommendation logic without running the full feature engine.
def _make_records(n: int, win_rate: float, avg_pnl_r: float, direction: str = "BUY"):
    """Synthetic walk-forward records for aggregation tests."""
    rng = np.random.default_rng(7)
    records = []
    for i in range(n):
        won = rng.random() < win_rate
        pnl = avg_pnl_r + rng.normal(0, 0.1)
        records.append({
            "ticker":     "TEST",
            "timeframe":  "5min",
            "bar_dt":     f"2025-01-06 09:{45+i:02d}",
            "direction":  direction,
            "entry_price": 100.0,
            "target":      101.5,
            "stop":        99.0,
            "exit_price":  101.5 if won else 99.0,
            "outcome":     "HIT_TARGET" if won else "HIT_STOP",
            "pnl_r":       float(pnl),
            "bars_held":   3,
            "won":         bool(won),
        })
    return records


# ── Fresh trainer factory (isolated from module singleton) ─────────────────────

def _fresh_trainer(tmp_path: Path):
    """Return a WalkForwardTrainer that writes state to tmp_path."""
    from agent.walk_forward_trainer import WalkForwardTrainer, _STATE_PATH
    trainer = WalkForwardTrainer.__new__(WalkForwardTrainer)
    trainer._last_summary = {}
    # Override path to temp dir
    import agent.walk_forward_trainer as wft_mod
    wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"
    return trainer


# ── Mock AlgoLearningEngine so tests never touch the real engine/registry ──────

class _MockRegistry:
    """Minimal mock that tracks update() calls."""

    def __init__(self, defaults=None):
        self._vals = defaults or {}
        self.calls: list[dict] = []

    def get(self, family, param):
        return self._vals.get((family, param), 1.5)

    def update(self, family, param, new_val, reason, cycle_num):
        self.calls.append(dict(
            family=family, param=param, new_val=new_val,
            reason=reason, cycle_num=cycle_num
        ))
        return True


class _MockEngine:
    def __init__(self):
        self._registry = _MockRegistry()


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestWalkForwardTrainerRun:
    """Tests for WalkForwardTrainer.run() end-to-end (synthetic OHLCV)."""

    def test_run_returns_summary_dict(self, tmp_path):
        """run() returns a dict with standard summary keys."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        trainer = WalkForwardTrainer()
        ohlcv_map = _make_ohlcv_map(["AAPL"])

        with patch("agent.algo_learning_engine.get_engine", return_value=_MockEngine()):
            result = trainer.run(["AAPL"], ohlcv_map, cycle_num=1)

        assert isinstance(result, dict)
        for key in ("started_at", "finished_at", "total_records", "overall_win_rate",
                    "family_stats", "recommendations", "cycle_num"):
            assert key in result, f"Missing key: {key}"

    def test_run_cycle_num_propagated(self, tmp_path):
        """cycle_num is preserved in the returned summary."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        trainer = WalkForwardTrainer()
        ohlcv_map = _make_ohlcv_map(["MSFT"])

        with patch("agent.algo_learning_engine.get_engine", return_value=_MockEngine()):
            result = trainer.run(["MSFT"], ohlcv_map, cycle_num=7)

        assert result["cycle_num"] == 7

    def test_run_with_multiple_tickers(self, tmp_path):
        """run() processes multiple tickers without crashing."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        tickers = ["AAPL", "MSFT", "NVDA"]
        trainer = WalkForwardTrainer()
        ohlcv_map = _make_ohlcv_map(tickers)

        with patch("agent.algo_learning_engine.get_engine", return_value=_MockEngine()):
            result = trainer.run(tickers, ohlcv_map, cycle_num=0)

        assert result["tickers_processed"] == 3

    def test_run_persists_state_to_file(self, tmp_path):
        """run() writes results to the JSON state file."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        state_path = tmp_path / "wf_trainer.json"
        wft_mod._STATE_PATH = state_path

        trainer = WalkForwardTrainer()
        ohlcv_map = _make_ohlcv_map(["TSLA"])

        with patch("agent.algo_learning_engine.get_engine", return_value=_MockEngine()):
            trainer.run(["TSLA"], ohlcv_map)

        assert state_path.exists(), "State file was not written"
        data = json.loads(state_path.read_text())
        assert "total_records" in data

    def test_run_with_only_5min_data(self, tmp_path):
        """run() works when only 5min data is provided (15min missing)."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        trainer = WalkForwardTrainer()
        ohlcv_map = {"AAPL": {"5min": _make_ohlcv(n=200, freq="5min")}}

        with patch("agent.algo_learning_engine.get_engine", return_value=_MockEngine()):
            result = trainer.run(["AAPL"], ohlcv_map)

        assert isinstance(result, dict)

    def test_run_with_only_15min_data(self, tmp_path):
        """run() works when only 15min data is provided (5min missing)."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        trainer = WalkForwardTrainer()
        ohlcv_map = {"AAPL": {"15min": _make_ohlcv(n=200, freq="15min")}}

        with patch("agent.algo_learning_engine.get_engine", return_value=_MockEngine()):
            result = trainer.run(["AAPL"], ohlcv_map)

        assert isinstance(result, dict)


class TestEmptyAndNaN:
    """Tests for graceful handling of degenerate OHLCV inputs."""

    def test_empty_ohlcv_map_no_crash(self, tmp_path):
        """run() with empty ohlcv_map returns a summary without crashing."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        trainer = WalkForwardTrainer()

        with patch("agent.algo_learning_engine.get_engine", return_value=_MockEngine()):
            result = trainer.run(["AAPL"], ohlcv_map={})

        assert result["total_records"] == 0
        assert result["tickers_skipped"] == 1

    def test_empty_dataframe_no_crash(self, tmp_path):
        """run() with empty DataFrames inside ohlcv_map doesn't crash."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        trainer = WalkForwardTrainer()
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        ohlcv_map = {"AAPL": {"5min": empty_df, "15min": empty_df}}

        with patch("agent.algo_learning_engine.get_engine", return_value=_MockEngine()):
            result = trainer.run(["AAPL"], ohlcv_map)

        assert result["total_records"] == 0

    def test_all_nan_ohlcv_no_crash(self, tmp_path):
        """run() with all-NaN OHLCV doesn't raise an exception."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        trainer = WalkForwardTrainer()
        nan_df = pd.DataFrame(
            np.nan,
            index=pd.date_range("2025-01-06 09:45", periods=50, freq="5min"),
            columns=["open", "high", "low", "close", "volume"],
        )
        ohlcv_map = {"AAPL": {"5min": nan_df}}

        with patch("agent.algo_learning_engine.get_engine", return_value=_MockEngine()):
            result = trainer.run(["AAPL"], ohlcv_map)

        assert isinstance(result, dict)

    def test_ticker_not_in_ohlcv_map_skipped(self, tmp_path):
        """Tickers absent from ohlcv_map are counted as skipped."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        trainer = WalkForwardTrainer()
        ohlcv_map = {}  # no data at all

        with patch("agent.algo_learning_engine.get_engine", return_value=_MockEngine()):
            result = trainer.run(["AAPL", "MSFT"], ohlcv_map)

        assert result["tickers_skipped"] == 2

    def test_too_few_bars_no_crash(self, tmp_path):
        """OHLCV with fewer bars than the minimum (60) is skipped gracefully."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        trainer = WalkForwardTrainer()
        short_df = _make_ohlcv(n=20, freq="5min")
        ohlcv_map = {"AAPL": {"5min": short_df}}

        with patch("agent.algo_learning_engine.get_engine", return_value=_MockEngine()):
            result = trainer.run(["AAPL"], ohlcv_map)

        assert isinstance(result, dict)


class TestParameterRecommendations:
    """Tests for _apply_recommendations logic using mock registry."""

    def _run_with_records(self, records, cycle_num, tmp_path, mock_engine=None):
        """Helper: build a trainer, patch aggregate step with pre-built records, run."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        if mock_engine is None:
            mock_engine = _MockEngine()

        trainer = WalkForwardTrainer()

        # Patch replay_signals to return our synthetic records directly
        with patch("agent.walk_forward.replay_signals", return_value=records):
            with patch("agent.algo_learning_engine.get_engine", return_value=mock_engine):
                ohlcv_map = {"TEST": {"5min": _make_ohlcv(n=200)}}
                result = trainer.run(["TEST"], ohlcv_map, cycle_num=cycle_num)

        return result, mock_engine._registry.calls

    def test_low_win_rate_triggers_conf_gate_increase(self, tmp_path):
        """
        Win rate < 0.35 with n >= 20 should attempt to raise conf_gate.
        """
        records = _make_records(n=30, win_rate=0.20, avg_pnl_r=0.5)
        mock_engine = _MockEngine()
        mock_engine._registry._vals = {(f, "conf_gate"): 55.0
                                        for f in ["ORB", "GAP_TREND", "GAP_FADE",
                                                   "BREAKOUT", "FLAG", "VWAP_SCALP",
                                                   "LEVEL_SCALP", "RS_REGIME"]}

        result, calls = self._run_with_records(records, cycle_num=1, tmp_path=tmp_path,
                                               mock_engine=mock_engine)

        conf_calls = [c for c in calls if c["param"] == "conf_gate"]
        assert conf_calls, "Expected conf_gate update calls for low win_rate"
        for c in conf_calls:
            assert c["new_val"] > 55.0, f"Expected conf_gate raised, got {c['new_val']}"

    def test_high_win_rate_triggers_rvol_gate_decrease(self, tmp_path):
        """
        Win rate > 0.60 with n >= 20 should attempt to lower rvol_gate.
        """
        records = _make_records(n=30, win_rate=0.80, avg_pnl_r=0.5)
        mock_engine = _MockEngine()
        mock_engine._registry._vals = {(f, "rvol_gate"): 1.5
                                        for f in ["ORB", "GAP_TREND", "GAP_FADE",
                                                   "BREAKOUT", "FLAG", "VWAP_SCALP",
                                                   "LEVEL_SCALP", "RS_REGIME"]}

        result, calls = self._run_with_records(records, cycle_num=1, tmp_path=tmp_path,
                                               mock_engine=mock_engine)

        rvol_calls = [c for c in calls if c["param"] == "rvol_gate"]
        assert rvol_calls, "Expected rvol_gate update calls for high win_rate"
        for c in rvol_calls:
            assert c["new_val"] < 1.5, f"Expected rvol_gate lowered, got {c['new_val']}"

    def test_low_pnl_r_triggers_target_mult_decrease(self, tmp_path):
        """
        avg_pnl_r < -0.5 with n >= 15 should attempt to lower target_mult.
        """
        records = _make_records(n=25, win_rate=0.45, avg_pnl_r=-0.8)
        mock_engine = _MockEngine()
        mock_engine._registry._vals = {(f, "target_mult"): 1.5
                                        for f in ["ORB", "GAP_TREND", "GAP_FADE",
                                                   "BREAKOUT", "FLAG", "VWAP_SCALP",
                                                   "LEVEL_SCALP", "RS_REGIME"]}

        result, calls = self._run_with_records(records, cycle_num=1, tmp_path=tmp_path,
                                               mock_engine=mock_engine)

        tm_calls = [c for c in calls if c["param"] == "target_mult"]
        assert tm_calls, "Expected target_mult update calls for low avg_pnl_r"
        for c in tm_calls:
            assert c["new_val"] < 1.5, f"Expected target_mult lowered, got {c['new_val']}"

    def test_high_pnl_r_triggers_target_mult_increase(self, tmp_path):
        """
        avg_pnl_r > 1.2 with n >= 15 should attempt to raise target_mult.
        """
        records = _make_records(n=25, win_rate=0.45, avg_pnl_r=1.8)
        mock_engine = _MockEngine()
        mock_engine._registry._vals = {(f, "target_mult"): 1.5
                                        for f in ["ORB", "GAP_TREND", "GAP_FADE",
                                                   "BREAKOUT", "FLAG", "VWAP_SCALP",
                                                   "LEVEL_SCALP", "RS_REGIME"]}

        result, calls = self._run_with_records(records, cycle_num=1, tmp_path=tmp_path,
                                               mock_engine=mock_engine)

        tm_calls = [c for c in calls if c["param"] == "target_mult"]
        assert tm_calls, "Expected target_mult update calls for high avg_pnl_r"
        for c in tm_calls:
            assert c["new_val"] > 1.5, f"Expected target_mult raised, got {c['new_val']}"

    def test_below_min_n_conf_no_recommendation(self, tmp_path):
        """
        n < 20 should NOT trigger conf_gate or rvol_gate updates.
        """
        records = _make_records(n=10, win_rate=0.10, avg_pnl_r=0.3)
        result, calls = self._run_with_records(records, cycle_num=1, tmp_path=tmp_path)

        conf_calls = [c for c in calls if c["param"] in ("conf_gate", "rvol_gate")]
        assert not conf_calls, f"Unexpected conf/rvol updates for n<20: {conf_calls}"

    def test_below_min_n_target_no_recommendation(self, tmp_path):
        """
        n < 15 should NOT trigger target_mult updates.
        """
        records = _make_records(n=8, win_rate=0.10, avg_pnl_r=-1.5)
        result, calls = self._run_with_records(records, cycle_num=1, tmp_path=tmp_path)

        tm_calls = [c for c in calls if c["param"] == "target_mult"]
        assert not tm_calls, f"Unexpected target_mult updates for n<15: {tm_calls}"

    def test_neutral_stats_no_recommendation(self, tmp_path):
        """
        Win rate in (0.35, 0.60) and avg_pnl_r in (-0.5, 1.2) → no updates.
        """
        records = _make_records(n=30, win_rate=0.50, avg_pnl_r=0.3)
        result, calls = self._run_with_records(records, cycle_num=1, tmp_path=tmp_path)

        assert not calls, f"Unexpected registry updates for neutral stats: {calls}"

    def test_recommendations_list_in_summary(self, tmp_path):
        """
        The returned summary['recommendations'] is a list.
        """
        records = _make_records(n=30, win_rate=0.20, avg_pnl_r=0.5)
        result, _ = self._run_with_records(records, cycle_num=1, tmp_path=tmp_path)
        assert isinstance(result["recommendations"], list)

    def test_recommendations_have_expected_fields(self, tmp_path):
        """
        Each recommendation dict has algo_family, param, new_val, reason.
        """
        records = _make_records(n=30, win_rate=0.20, avg_pnl_r=0.5)
        mock_engine = _MockEngine()
        mock_engine._registry._vals = {(f, "conf_gate"): 55.0
                                        for f in ["ORB", "GAP_TREND", "GAP_FADE",
                                                   "BREAKOUT", "FLAG", "VWAP_SCALP",
                                                   "LEVEL_SCALP", "RS_REGIME"]}
        result, _ = self._run_with_records(records, cycle_num=1, tmp_path=tmp_path,
                                           mock_engine=mock_engine)

        for rec in result["recommendations"]:
            for field in ("algo_family", "param", "new_val", "reason"):
                assert field in rec, f"Missing field '{field}' in recommendation"


class TestGetStatus:
    """Tests for WalkForwardTrainer.get_status()."""

    def test_get_status_returns_dict(self, tmp_path):
        """get_status() returns a dict (possibly empty on fresh trainer)."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        trainer = WalkForwardTrainer()
        status = trainer.get_status()
        assert isinstance(status, dict)

    def test_get_status_after_run_has_expected_keys(self, tmp_path):
        """After run(), get_status() returns a dict with standard keys."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        trainer = WalkForwardTrainer()
        ohlcv_map = _make_ohlcv_map(["AAPL"])

        with patch("agent.algo_learning_engine.get_engine", return_value=_MockEngine()):
            trainer.run(["AAPL"], ohlcv_map, cycle_num=3)

        status = trainer.get_status()
        for key in ("started_at", "finished_at", "total_records", "overall_win_rate",
                    "family_stats", "recommendations", "cycle_num"):
            assert key in status, f"get_status() missing key: {key}"

    def test_get_status_loads_persisted_state(self, tmp_path):
        """get_status() loads from JSON file if _last_summary is empty."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        state_path = tmp_path / "wf_trainer.json"
        wft_mod._STATE_PATH = state_path

        # Write a fake state file
        fake_state = {"total_records": 42, "overall_win_rate": 0.55, "cycle_num": 9}
        state_path.write_text(json.dumps(fake_state))

        # Fresh trainer (no in-memory summary)
        trainer = WalkForwardTrainer()
        status = trainer.get_status()
        assert status.get("total_records") == 42

    def test_get_status_handles_missing_file(self, tmp_path):
        """get_status() returns {} when state file doesn't exist."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "nonexistent_wf.json"

        trainer = WalkForwardTrainer()
        status = trainer.get_status()
        assert isinstance(status, dict)

    def test_get_status_handles_corrupt_json(self, tmp_path):
        """get_status() doesn't crash on corrupt JSON file."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        state_path = tmp_path / "wf_trainer.json"
        wft_mod._STATE_PATH = state_path
        state_path.write_text("{this is not valid json}")

        trainer = WalkForwardTrainer()
        status = trainer.get_status()
        assert isinstance(status, dict)


class TestSingleton:
    """Tests for the module-level singleton pattern."""

    def test_singleton_returns_same_instance(self, tmp_path):
        """get_walk_forward_trainer() returns the same object on repeated calls."""
        import agent.walk_forward_trainer as wft_mod
        wft_mod._trainer_instance = None   # reset for test isolation
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        from agent.walk_forward_trainer import get_walk_forward_trainer
        a = get_walk_forward_trainer()
        b = get_walk_forward_trainer()
        assert a is b, "Singleton should return the same instance"

    def test_singleton_is_walk_forward_trainer_type(self, tmp_path):
        """The singleton is an instance of WalkForwardTrainer."""
        import agent.walk_forward_trainer as wft_mod
        wft_mod._trainer_instance = None   # reset
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        from agent.walk_forward_trainer import get_walk_forward_trainer, WalkForwardTrainer
        instance = get_walk_forward_trainer()
        assert isinstance(instance, WalkForwardTrainer)

    def test_singleton_thread_safety(self, tmp_path):
        """Concurrent calls to get_walk_forward_trainer() return the same object."""
        import agent.walk_forward_trainer as wft_mod
        wft_mod._trainer_instance = None
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        from agent.walk_forward_trainer import get_walk_forward_trainer
        instances = []

        def _get():
            instances.append(get_walk_forward_trainer())

        threads = [threading.Thread(target=_get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = instances[0]
        assert all(i is first for i in instances), "All threads should get the same instance"


class TestAggregateByFamily:
    """Unit tests for _aggregate_by_family() method."""

    def test_aggregate_splits_by_direction(self, tmp_path):
        """_aggregate_by_family groups records by direction key."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"
        trainer = WalkForwardTrainer()

        buy_recs  = _make_records(10, 0.7, 0.5, "BUY")
        sell_recs = _make_records(10, 0.3, -0.5, "SELL")
        stats = trainer._aggregate_by_family(buy_recs + sell_recs)

        assert "BUY"  in stats
        assert "SELL" in stats

    def test_aggregate_win_rate_correct(self, tmp_path):
        """_aggregate_by_family computes win_rate correctly."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"
        trainer = WalkForwardTrainer()

        # 10 wins out of 10 records
        records = [{"direction": "BUY", "won": True, "pnl_r": 1.5} for _ in range(10)]
        stats = trainer._aggregate_by_family(records)
        assert stats["BUY"]["win_rate"] == 1.0

    def test_aggregate_empty_records(self, tmp_path):
        """_aggregate_by_family on empty list returns empty dict."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"
        trainer = WalkForwardTrainer()

        stats = trainer._aggregate_by_family([])
        assert stats == {}

    def test_aggregate_avg_pnl_r_correct(self, tmp_path):
        """_aggregate_by_family computes avg_pnl_r as mean of pnl_r."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"
        trainer = WalkForwardTrainer()

        records = [{"direction": "BUY", "won": True, "pnl_r": float(v)}
                   for v in [1.0, 2.0, 3.0]]
        stats = trainer._aggregate_by_family(records)
        assert abs(stats["BUY"]["avg_pnl_r"] - 2.0) < 1e-4


class TestFetchOhlcvHelper:
    """Tests for the _fetch_ohlcv_for_ticker helper function."""

    def test_fetch_returns_dict(self, tmp_path):
        """_fetch_ohlcv_for_ticker returns a dict (possibly empty)."""
        from agent.walk_forward_trainer import _fetch_ohlcv_for_ticker

        with patch("agent.historical_cache.get_bars", return_value=pd.DataFrame()):
            result = _fetch_ohlcv_for_ticker("AAPL")

        assert isinstance(result, dict)

    def test_fetch_graceful_on_import_error(self, tmp_path):
        """_fetch_ohlcv_for_ticker returns {} if historical_cache unavailable."""
        from agent.walk_forward_trainer import _fetch_ohlcv_for_ticker

        with patch("builtins.__import__", side_effect=ImportError("no cache")):
            result = _fetch_ohlcv_for_ticker("AAPL")

        # Should not raise — returns empty dict or similar
        assert isinstance(result, dict)


class TestAlgoLearningEngineUnavailable:
    """Test trainer gracefully handles missing AlgoLearningEngine."""

    def test_run_survives_missing_engine(self, tmp_path):
        """run() survives if get_engine() raises an ImportError."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        trainer = WalkForwardTrainer()
        ohlcv_map = {"AAPL": {"5min": _make_ohlcv(n=200)}}

        with patch("agent.walk_forward_trainer.WalkForwardTrainer._apply_recommendations",
                   side_effect=Exception("engine unavailable")):
            # Should NOT raise — returns partial summary
            result = trainer.run(["AAPL"], ohlcv_map)

        assert isinstance(result, dict)

    def test_apply_recommendations_import_error(self, tmp_path):
        """_apply_recommendations returns [] if algo_learning_engine can't be imported."""
        from agent.walk_forward_trainer import WalkForwardTrainer
        import agent.walk_forward_trainer as wft_mod
        wft_mod._STATE_PATH = tmp_path / "wf_trainer.json"

        trainer = WalkForwardTrainer()
        with patch.dict("sys.modules", {"agent.algo_learning_engine": None}):
            result = trainer._apply_recommendations(
                {"BUY": {"n": 30, "win_rate": 0.20, "avg_pnl_r": 0.3}},
                cycle_num=1
            )
        assert isinstance(result, list)
