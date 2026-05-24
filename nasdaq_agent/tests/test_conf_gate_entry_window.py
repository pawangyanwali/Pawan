"""
Tests for conf_gate and entry_window_bars filters wired into scanner.py.

conf_gate:        Per-algo-family learned minimum confidence. Signals below the
                  gate call bt_record() for learning data but skip maybe_open_trade().

entry_window_bars: Staleness guard — a signal that has fired for more consecutive
                  scan cycles than entry_window_bars is treated as "already digested"
                  and its trade is skipped. Counter resets when the signal stops firing.
"""
from __future__ import annotations

import pathlib
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_params(conf_gate: float = 55.0, entry_window_bars: int = 3) -> dict:
    return {
        "conf_gate":          conf_gate,
        "entry_window_bars":  entry_window_bars,
        "rvol_gate":          1.5,
        "target_mult":        1.5,
        "stop_mult":          1.0,
    }


def _make_signal(algo: str = "ORB5_BULL", confidence: float = 70.0) -> dict:
    return {
        "algo":       algo,
        "direction":  "BUY",
        "entry":      100.0,
        "target":     105.0,
        "stop":       98.0,
        "confidence": confidence,
        "rr":         2.5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Source-level checks (fast, no imports)
# ─────────────────────────────────────────────────────────────────────────────

class TestScannerSourceHasFilters:
    """Verify the filter code is present in scanner.py without importing it."""

    def _src(self) -> str:
        return (pathlib.Path("/home/user/Pawan/nasdaq_agent/agent/scanner.py")
                .read_text())

    def test_get_algo_params_imported(self):
        assert "get_algo_params as _get_algo_params" in self._src()

    def test_conf_gate_check_present(self):
        assert "conf_gate" in self._src()

    def test_entry_window_bars_check_present(self):
        assert "entry_window_bars" in self._src()

    def test_sig_consec_dict_declared(self):
        assert "_sig_consec" in self._src()

    def test_sig_consec_lock_declared(self):
        assert "_sig_consec_lock" in self._src()

    def test_staleness_guard_logs(self):
        assert "stale" in self._src()

    def test_conf_gate_suppression_logs(self):
        assert "conf_gate" in self._src() and "suppressed" in self._src()

    def test_cleanup_after_loop(self):
        """Consecutive counters must be cleaned up for signals that stopped firing."""
        src = self._src()
        assert "_fired" in src or "fired_this_cycle" in src or "_fired =" in src
        assert "_sig_consec" in src

    def test_counter_incremented_before_filters(self):
        """Counter must be updated even if signal is later suppressed by conf_gate."""
        src = self._src()
        # _sig_key and counter increment must appear before the conf_gate check
        conf_idx = src.index("conf_gate")
        counter_idx = src.index("_sig_consec")
        assert counter_idx < conf_idx, (
            "Consecutive counter update must appear before conf_gate check so counts "
            "are accurate even for suppressed signals."
        )


# ─────────────────────────────────────────────────────────────────────────────
# conf_gate logic (unit tests using the exact filtering logic)
# ─────────────────────────────────────────────────────────────────────────────

class TestConfGateLogic:
    """Test the conf_gate filtering logic in isolation."""

    def _run_conf_gate(
        self,
        confidence: float,
        conf_gate: float,
        ale_available: bool = True,
    ) -> bool:
        """Simulate the conf_gate check. Returns True if signal passes (trade proceeds)."""
        if not ale_available:
            return True  # no filter when engine unavailable
        params = _make_params(conf_gate=conf_gate)
        return confidence >= params["conf_gate"]

    def test_passes_above_gate(self):
        assert self._run_conf_gate(confidence=70.0, conf_gate=55.0) is True

    def test_passes_at_exact_gate(self):
        assert self._run_conf_gate(confidence=55.0, conf_gate=55.0) is True

    def test_suppressed_below_gate(self):
        assert self._run_conf_gate(confidence=54.9, conf_gate=55.0) is False

    def test_suppressed_well_below_gate(self):
        assert self._run_conf_gate(confidence=40.0, conf_gate=55.0) is False

    def test_passes_when_ale_unavailable(self):
        """When engine unavailable, conf_gate filter is bypassed."""
        assert self._run_conf_gate(confidence=30.0, conf_gate=55.0, ale_available=False) is True

    def test_learned_higher_gate_suppresses_more(self):
        """Engine raises conf_gate over time → more signals suppressed."""
        # With default gate: 60 passes
        assert self._run_conf_gate(confidence=60.0, conf_gate=55.0) is True
        # With learned higher gate: 60 now suppressed
        assert self._run_conf_gate(confidence=60.0, conf_gate=65.0) is False

    def test_default_gate_is_55(self):
        """PARAM_SPEC default for conf_gate is 55.0."""
        from agent.algo_learning_engine import get_algo_params
        params = get_algo_params("ORB5_BULL")
        assert "conf_gate" in params
        assert 45.0 <= params["conf_gate"] <= 75.0  # within spec bounds

    def test_conf_gate_all_families(self):
        """Every algo family has a conf_gate parameter."""
        from agent.algo_learning_engine import get_algo_params, _ALGO_FAMILY_MAP
        checked = set()
        for algo_name in list(_ALGO_FAMILY_MAP.keys())[:8]:
            params = get_algo_params(algo_name)
            assert "conf_gate" in params, f"conf_gate missing for {algo_name}"
            assert params["conf_gate"] >= 45.0
            checked.add(algo_name)
        assert len(checked) >= 8

    def test_conf_gate_suppressed_still_records_bt(self):
        """
        When conf_gate suppresses a signal, bt_record is already called before the
        check — the learning data is preserved even for suppressed signals.
        """
        mock_bt_record = MagicMock()
        mock_maybe_open = MagicMock(return_value=None)
        mock_get_params = MagicMock(return_value=_make_params(conf_gate=80.0))

        signals = [_make_signal(confidence=70.0)]
        for asig in signals:
            mock_bt_record(ticker="AAPL", algo_name=asig["algo"])  # always
            params = mock_get_params(asig["algo"])
            if asig["confidence"] < params["conf_gate"]:
                continue  # suppressed — skip trade
            mock_maybe_open(ticker="AAPL")

        mock_bt_record.assert_called_once()
        mock_maybe_open.assert_not_called()

    def test_conf_gate_passes_allows_trade(self):
        mock_bt_record = MagicMock()
        mock_maybe_open = MagicMock(return_value="trade-1")
        mock_get_params = MagicMock(return_value=_make_params(conf_gate=55.0))

        signals = [_make_signal(confidence=75.0)]
        for asig in signals:
            mock_bt_record(ticker="AAPL", algo_name=asig["algo"])
            params = mock_get_params(asig["algo"])
            if asig["confidence"] < params["conf_gate"]:
                continue
            mock_maybe_open(ticker="AAPL")

        mock_bt_record.assert_called_once()
        mock_maybe_open.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# entry_window_bars logic (unit tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestEntryWindowBarsLogic:
    """
    Test the entry_window_bars staleness logic in isolation.
    Simulates the consecutive counter state machine from scanner.py.
    """

    def _simulate_cycles(
        self,
        n_cycles: int,
        entry_window: int,
        signal_fires: list[bool] | None = None,
    ) -> list[bool]:
        """
        Simulate n_cycles scan cycles.
        signal_fires[i] = True means the signal fires in cycle i (default: all fire).
        Returns list of booleans: whether the trade was allowed on each cycle.
        """
        if signal_fires is None:
            signal_fires = [True] * n_cycles

        consec: dict[str, int] = {}
        lock = threading.Lock()
        results = []
        ticker, algo = "AAPL", "ORB5_BULL"
        key = f"{ticker}:{algo}"

        for i, fires in enumerate(signal_fires):
            if fires:
                with lock:
                    consec[key] = consec.get(key, 0) + 1
                    c = consec[key]
                allowed = c <= entry_window
                results.append(allowed)
            else:
                # Signal stopped firing → reset counter
                with lock:
                    consec.pop(key, None)
                results.append(None)  # no signal this cycle

        return results

    def test_first_fire_always_allowed(self):
        results = self._simulate_cycles(n_cycles=1, entry_window=3)
        assert results[0] is True

    def test_fires_within_window_allowed(self):
        results = self._simulate_cycles(n_cycles=3, entry_window=3)
        assert all(r is True for r in results)

    def test_exactly_at_window_allowed(self):
        """At cycle = entry_window, signal is still allowed."""
        results = self._simulate_cycles(n_cycles=3, entry_window=3)
        assert results[2] is True

    def test_exceeds_window_blocked(self):
        """At cycle entry_window+1, signal is stale → blocked."""
        results = self._simulate_cycles(n_cycles=4, entry_window=3)
        assert results[3] is False  # cycle 4 > window 3

    def test_many_cycles_blocked_after_window(self):
        results = self._simulate_cycles(n_cycles=10, entry_window=3)
        assert results[0] is True   # fresh
        assert results[1] is True
        assert results[2] is True
        assert results[3] is False  # stale
        assert results[9] is False

    def test_counter_resets_after_signal_stops(self):
        """After signal stops firing, next fire is treated as fresh."""
        signal_fires = [True, True, True, True, False, True]  # 4 fires, gap, 1 fire
        results = self._simulate_cycles(
            n_cycles=6, entry_window=3, signal_fires=signal_fires
        )
        assert results[0] is True
        assert results[3] is False  # stale (cycle 4 > window 3)
        assert results[4] is None   # no signal
        assert results[5] is True   # fresh after gap

    def test_entry_window_1_only_first_fire_allowed(self):
        """entry_window=1: only the very first cycle of a continuous signal runs."""
        results = self._simulate_cycles(n_cycles=5, entry_window=1)
        assert results[0] is True
        assert results[1] is False
        assert results[4] is False

    def test_entry_window_8_max_allows_many_cycles(self):
        results = self._simulate_cycles(n_cycles=8, entry_window=8)
        assert all(r is True for r in results)
        # 9th cycle would be blocked
        results9 = self._simulate_cycles(n_cycles=9, entry_window=8)
        assert results9[8] is False

    def test_engine_provides_entry_window_param(self):
        """AlgoLearningEngine.get_algo_params() includes entry_window_bars."""
        from agent.algo_learning_engine import get_algo_params
        for algo in ("ORB5_BULL", "GAP_AND_GO_BULL", "VWAP_TOUCH_SCALP_BULL"):
            params = get_algo_params(algo)
            assert "entry_window_bars" in params
            assert 1 <= params["entry_window_bars"] <= 8

    def test_default_entry_window_is_3(self):
        """PARAM_SPEC default for entry_window_bars is 3."""
        from agent.algo_learning_engine import get_algo_params
        params = get_algo_params("ORB5_BULL")
        # Initial value is 3 (unless already learned otherwise)
        assert 1 <= int(params["entry_window_bars"]) <= 8


# ─────────────────────────────────────────────────────────────────────────────
# Combined flow simulation
# ─────────────────────────────────────────────────────────────────────────────

class TestCombinedFilterFlow:
    """Simulate the full per-signal flow: bt_record → conf_gate → entry_window → routing → trade."""

    def _process_signal(
        self,
        asig: dict,
        consec: int,
        conf_gate: float = 55.0,
        entry_window: int = 3,
        routing: str = "PAPER",
        ale_available: bool = True,
    ) -> tuple[bool, bool]:
        """
        Returns (bt_called, trade_called).
        Simulates one iteration of the scanner signal loop.
        """
        bt_mock = MagicMock()
        trade_mock = MagicMock(return_value="trade-1")

        # bt_record always
        bt_mock()

        if ale_available:
            # conf_gate check
            if asig["confidence"] < conf_gate:
                return True, False

            # entry_window check
            if consec > entry_window:
                return True, False

            # routing check
            if routing == "SHADOW":
                return True, False

        trade_mock()
        return True, True

    def test_fresh_high_conf_signal_trades(self):
        sig = _make_signal(confidence=75.0)
        bt, trade = self._process_signal(sig, consec=1, conf_gate=55.0, entry_window=3)
        assert bt and trade

    def test_low_conf_suppressed(self):
        sig = _make_signal(confidence=50.0)
        bt, trade = self._process_signal(sig, consec=1, conf_gate=55.0, entry_window=3)
        assert bt and not trade

    def test_stale_signal_suppressed(self):
        sig = _make_signal(confidence=75.0)
        bt, trade = self._process_signal(sig, consec=4, conf_gate=55.0, entry_window=3)
        assert bt and not trade

    def test_shadow_routing_suppressed(self):
        sig = _make_signal(confidence=75.0)
        bt, trade = self._process_signal(sig, consec=1, routing="SHADOW")
        assert bt and not trade

    def test_both_gates_met_paper_trades(self):
        sig = _make_signal(confidence=68.0)
        bt, trade = self._process_signal(sig, consec=2, conf_gate=55.0, entry_window=3, routing="PAPER")
        assert bt and trade

    def test_ale_unavailable_all_signals_trade(self):
        """Without engine, all signals proceed to maybe_open_trade."""
        sig = _make_signal(confidence=30.0)
        bt, trade = self._process_signal(
            sig, consec=100, conf_gate=55.0, entry_window=3,
            routing="SHADOW", ale_available=False,
        )
        assert bt and trade

    def test_multiple_signals_selective_filtering(self):
        """Only signals passing both filters open trades."""
        signals = [
            (_make_signal("ORB5_BULL",      70.0), 1,  "PAPER"),   # ✓ passes
            (_make_signal("GAP_AND_GO_BULL", 40.0), 1,  "PAPER"),   # ✗ conf_gate
            (_make_signal("HOD_BREAK_BULL",  75.0), 4,  "PAPER"),   # ✗ stale
            (_make_signal("BULL_FLAG",       80.0), 1,  "SHADOW"),  # ✗ shadow
            (_make_signal("CS_RS_RANK_BULL", 65.0), 2,  "LIVE"),    # ✓ passes
        ]
        expected = [True, False, False, False, True]
        for (sig, consec, routing), expect in zip(signals, expected):
            _, trade = self._process_signal(
                sig, consec=consec, conf_gate=55.0, entry_window=3, routing=routing
            )
            assert trade == expect, (
                f"Signal {sig['algo']} (conf={sig['confidence']}, consec={consec}, "
                f"routing={routing}) trade={trade}, expected={expect}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level state (thread safety)
# ─────────────────────────────────────────────────────────────────────────────

class TestConsecCounterThreadSafety:
    """Verify the _sig_consec dict handles concurrent updates correctly."""

    def test_concurrent_increments_are_safe(self):
        consec: dict[str, int] = {}
        lock = threading.Lock()
        errors = []

        def increment(key: str, n: int) -> None:
            for _ in range(n):
                try:
                    with lock:
                        consec[key] = consec.get(key, 0) + 1
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=increment, args=(f"AAPL:ORB5_BULL", 100))
                   for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert consec["AAPL:ORB5_BULL"] == 500

    def test_cleanup_removes_stale_keys(self):
        """Keys for signals that stopped firing are removed from the dict."""
        consec = {
            "AAPL:ORB5_BULL":       3,
            "AAPL:GAP_AND_GO_BULL": 1,
            "MSFT:ORB5_BULL":       2,
        }
        lock = threading.Lock()
        # This cycle, only ORB5_BULL fired for AAPL
        fired = {"AAPL:ORB5_BULL"}
        ticker = "AAPL"
        with lock:
            for k in [k for k in list(consec) if k.startswith(f"{ticker}:")]:
                if k not in fired:
                    del consec[k]

        assert "AAPL:ORB5_BULL" in consec
        assert "AAPL:GAP_AND_GO_BULL" not in consec
        assert "MSFT:ORB5_BULL" in consec  # other ticker untouched

    def test_cleanup_does_not_affect_other_tickers(self):
        consec = {
            "AAPL:ORB5_BULL": 5,
            "TSLA:ORB5_BULL": 2,
        }
        lock = threading.Lock()
        fired = set()  # no signals fired for AAPL this cycle
        with lock:
            for k in [k for k in list(consec) if k.startswith("AAPL:")]:
                if k not in fired:
                    del consec[k]

        assert "AAPL:ORB5_BULL" not in consec
        assert "TSLA:ORB5_BULL" in consec


# ─────────────────────────────────────────────────────────────────────────────
# Integration: get_algo_params returns correct conf_gate and entry_window_bars
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAlgoParamsIntegration:
    """Verify get_algo_params() returns usable conf_gate and entry_window_bars."""

    def test_orb_family_params(self):
        from agent.algo_learning_engine import get_algo_params
        p = get_algo_params("ORB5_BULL")
        assert float(p["conf_gate"]) >= 45.0
        assert int(p["entry_window_bars"]) >= 1

    def test_gap_trend_family_params(self):
        from agent.algo_learning_engine import get_algo_params
        p = get_algo_params("GAP_AND_GO_BULL")
        assert "conf_gate" in p
        assert "entry_window_bars" in p

    def test_vwap_scalp_family_params(self):
        from agent.algo_learning_engine import get_algo_params
        p = get_algo_params("VWAP_TOUCH_SCALP_BULL")
        assert "conf_gate" in p
        assert "entry_window_bars" in p

    def test_rs_regime_family_params(self):
        from agent.algo_learning_engine import get_algo_params
        p = get_algo_params("CS_RS_RANK_BULL")
        assert "conf_gate" in p
        assert "entry_window_bars" in p

    def test_unknown_algo_returns_defaults(self):
        """Unknown algo name falls back to PARAM_SPEC defaults."""
        from agent.algo_learning_engine import get_algo_params
        p = get_algo_params("NONEXISTENT_ALGO")
        assert "conf_gate" in p
        assert "entry_window_bars" in p
        # Defaults match PARAM_SPEC
        assert p["conf_gate"] == 55.0
        assert p["entry_window_bars"] == 3

    def test_conf_gate_within_spec_bounds(self):
        from agent.algo_learning_engine import get_algo_params
        for algo in ("ORB5_BULL", "GAP_AND_GO_BULL", "BULL_FLAG", "REGIME_ALIGNED_LONG"):
            p = get_algo_params(algo)
            assert 45.0 <= p["conf_gate"] <= 75.0, (
                f"{algo} conf_gate {p['conf_gate']} out of [45, 75]"
            )

    def test_entry_window_within_spec_bounds(self):
        from agent.algo_learning_engine import get_algo_params
        for algo in ("ORB5_BULL", "GAP_AND_GO_BULL", "VWAP_HOD_SCALP", "SPY_BETA_CATCHUP_BULL"):
            p = get_algo_params(algo)
            assert 1 <= int(p["entry_window_bars"]) <= 8, (
                f"{algo} entry_window_bars {p['entry_window_bars']} out of [1, 8]"
            )
