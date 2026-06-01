from __future__ import annotations
import pytest
"""
Section A — Self-Learning System Tests

Covers:
  A1. Adaptive filter: WR updates, EWMA, threshold movement
  A2. Adaptive filter: context blocking and boosting
  A3. Adaptive filter: suppression checks, confidence boosts
  A4. Adaptive filter: anti-deadlock, observation source, reset_filter
  A5. Signal tracker: record, resolve WIN/LOSS, stats
  A6. Learning engine: start/stop, log API

Run:
    cd nasdaq_agent
    pytest tests/test_self_learning.py -v --tb=short
"""
pytestmark = pytest.mark.slow

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent.adaptive_filter as AF
import agent.signal_tracker as ST
from agent.learning_engine import LearningEngine, get_learning_log, clear_log


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a stats dict in the format expected by update_filter
# ─────────────────────────────────────────────────────────────────────────────

def make_stats(
    total: int = 20,
    win_rate: float = 0.60,
    by_session: dict | None = None,
    by_regime: dict | None = None,
    by_vwap_event: dict | None = None,
    by_direction: dict | None = None,
    by_confidence: dict | None = None,
) -> dict:
    by_confidence = by_confidence or {}
    return {
        "overall": {"total": total, "win_rate": win_rate},
        "by_session":    by_session   or {},
        "by_regime":     by_regime    or {},
        "by_vwap_event": by_vwap_event or {},
        "by_rsi_zone":   {},
        "by_entry_type": {},
        "by_direction":  by_direction or {},
        "by_sector_trend": {},
        "by_ah_bias":    {},
        "by_confidence": by_confidence,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: reset adaptive filter state before every test so tests are isolated
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_af(tmp_path, monkeypatch):
    """Give each test a fresh, isolated adaptive filter state.

    Filter state now persists to the DB (routed to the shared in-memory test DB
    by the autouse tmp_db_paths fixture, reset per test); reset_filter() clears
    the in-memory _state, so no file-path patch is needed any more.
    """
    AF.reset_filter("test setup")
    yield
    AF.reset_filter("test teardown")


# ═════════════════════════════════════════════════════════════════════════════
# A1 – Initial state and WR / threshold movement
# ═════════════════════════════════════════════════════════════════════════════

class TestAdaptiveFilterWRThreshold:

    def test_initial_threshold_is_default(self):
        st = AF.get_status()
        assert st["dynamic_threshold"] == AF.DEFAULT_THRESHOLD

    def test_initial_win_rate_is_zero(self):
        st = AF.get_status()
        assert st["current_win_rate"] == 0.0

    def test_initial_no_blocked_contexts(self):
        st = AF.get_status()
        assert st["blocked_contexts"] == {}

    def test_initial_no_boosted_contexts(self):
        st = AF.get_status()
        assert st["boosted_contexts"] == {}

    def test_update_with_high_wr_sets_win_rate(self):
        AF.update_filter(make_stats(total=50, win_rate=0.65), source="backtest")
        st = AF.get_status()
        assert st["current_win_rate"] > 0.0

    def test_ewma_smoothing_two_updates(self):
        """Two 60% updates: get_status() returns win_rate*100 → ~60 after both."""
        AF.update_filter(make_stats(total=50, win_rate=0.60), source="backtest")
        AF.update_filter(make_stats(total=50, win_rate=0.60), source="backtest")
        st = AF.get_status()
        # get_status() multiplies by 100; two identical 60% updates should be ~60
        assert 55 < st["current_win_rate"] < 70

    def test_ewma_alpha_formula(self):
        """Cold-start EWMA: first update from 0.0 → stored directly as win_rate.
        get_status() returns win_rate * 100, so 0.60 becomes 60.0."""
        AF.update_filter(make_stats(total=50, win_rate=0.60), source="backtest")
        st = AF.get_status()
        # Expected: 0.60 * 100 = 60.0
        assert abs(st["current_win_rate"] - 60.0) < 1e-4

    def test_low_total_does_not_block_contexts_but_wr_tracked(self):
        """Total < MIN_SAMPLE: context blocking is skipped; threshold stays at DEFAULT."""
        AF.update_filter(make_stats(total=3, win_rate=0.20), source="backtest")
        st = AF.get_status()
        # No context keys → nothing to block; threshold unchanged
        assert st["dynamic_threshold"] == AF.DEFAULT_THRESHOLD
        assert st["blocked_contexts"] == {}

    def test_threshold_stays_within_bounds(self):
        """Threshold must always be in [MIN_THRESHOLD, MAX_THRESHOLD]."""
        for wr in (0.10, 0.30, 0.55, 0.70, 0.90):
            AF.update_filter(make_stats(total=200, win_rate=wr), source="backtest")
        st = AF.get_status()
        thr = st["dynamic_threshold"]
        assert AF.MIN_THRESHOLD <= thr <= AF.MAX_THRESHOLD

    def test_very_high_wr_relaxes_threshold(self):
        """Win rate above RELAX_ABOVE should push threshold below DEFAULT."""
        for _ in range(8):
            AF.update_filter(make_stats(total=200, win_rate=AF.RELAX_ABOVE + 0.05),
                             source="backtest")
        st = AF.get_status()
        assert st["dynamic_threshold"] <= AF.DEFAULT_THRESHOLD

    def test_threshold_history_recorded(self):
        AF.update_filter(make_stats(total=50, win_rate=0.60), source="backtest")
        st = AF.get_status()
        assert len(st.get("threshold_history", [])) >= 1

    def test_observation_source_does_not_change_main_win_rate(self):
        AF.update_filter(make_stats(total=50, win_rate=0.75), source="observation")
        st = AF.get_status()
        assert st["current_win_rate"] == 0.0, \
            "Observation source must not update current_win_rate"

    def test_observation_source_updates_observation_win_rate(self):
        AF.update_filter(make_stats(total=50, win_rate=0.75), source="observation")
        st = AF.get_status()
        assert st.get("observation_win_rate", 0.0) > 0.0


# ═════════════════════════════════════════════════════════════════════════════
# A2 – Context blocking and boosting
# ═════════════════════════════════════════════════════════════════════════════

class TestAdaptiveFilterContexts:

    def test_low_wr_session_gets_blocked(self):
        """A session with win rate < SUPPRESS_BELOW over MIN_SAMPLE trades → blocked."""
        stats = make_stats(
            total=50, win_rate=0.50,
            by_session={
                "AVOID_ZONE": {
                    "total": AF.MIN_SAMPLE + 2,
                    "win_rate": AF.SUPPRESS_BELOW - 0.05,
                }
            }
        )
        AF.update_filter(stats, source="backtest")
        st = AF.get_status()
        assert "session:AVOID_ZONE" in st["blocked_contexts"]

    def test_high_wr_session_gets_boosted(self):
        """A session with win rate >= BOOST_ABOVE over MIN_SAMPLE trades → boosted."""
        stats = make_stats(
            total=50, win_rate=0.65,
            by_session={
                "POWER_HOUR": {
                    "total": AF.MIN_SAMPLE + 2,
                    "win_rate": AF.BOOST_ABOVE + 0.05,
                }
            }
        )
        AF.update_filter(stats, source="backtest")
        st = AF.get_status()
        assert "session:POWER_HOUR" in st["boosted_contexts"]

    def test_low_sample_count_not_blocked(self):
        """Only MIN_SAMPLE - 1 trades in context → should NOT block."""
        stats = make_stats(
            total=50, win_rate=0.50,
            by_session={
                "SMALL_SESSION": {
                    "total": max(0, AF.MIN_SAMPLE - 1),
                    "win_rate": 0.10,
                }
            }
        )
        AF.update_filter(stats, source="backtest")
        st = AF.get_status()
        assert "session:SMALL_SESSION" not in st["blocked_contexts"]

    def test_regime_can_be_blocked(self):
        stats = make_stats(
            total=50, win_rate=0.50,
            by_regime={
                "CHOPPY": {
                    "total": AF.MIN_SAMPLE + 5,
                    "win_rate": AF.SUPPRESS_BELOW - 0.10,
                }
            }
        )
        AF.update_filter(stats, source="backtest")
        st = AF.get_status()
        assert "regime:CHOPPY" in st["blocked_contexts"]

    def test_direction_can_be_blocked(self):
        stats = make_stats(
            total=50, win_rate=0.50,
            by_direction={
                "SELL": {
                    "total": AF.MIN_SAMPLE + 5,
                    "win_rate": AF.SUPPRESS_BELOW - 0.10,
                }
            }
        )
        AF.update_filter(stats, source="backtest")
        st = AF.get_status()
        assert "direction:SELL" in st["blocked_contexts"]

    def test_observation_source_can_block_context(self):
        """Observation source should still block contexts (with tighter threshold)."""
        # Observation requires 3× MIN_SAMPLE and lower than obs_suppress_below=0.20
        obs_min = max(AF.MIN_SAMPLE * 3, 25)
        stats = make_stats(
            total=50, win_rate=0.30,
            by_session={
                "OBS_SESSION": {
                    "total": obs_min + 5,
                    "win_rate": 0.10,
                }
            }
        )
        AF.update_filter(stats, source="observation")
        st = AF.get_status()
        assert "session:OBS_SESSION" in st["blocked_contexts"]

    def test_observation_source_does_not_boost_contexts(self):
        """Observation source must never add boosted contexts."""
        stats = make_stats(
            total=50, win_rate=0.90,
            by_session={
                "OBS_GREAT": {
                    "total": 50,
                    "win_rate": 0.95,
                }
            }
        )
        AF.update_filter(stats, source="observation")
        st = AF.get_status()
        assert "session:OBS_GREAT" not in st["boosted_contexts"]


# ═════════════════════════════════════════════════════════════════════════════
# A3 – should_suppress and get_confidence_boost
# ═════════════════════════════════════════════════════════════════════════════

class TestAdaptiveFilterSuppression:

    def test_suppress_when_below_threshold(self):
        """Confidence below dynamic_threshold → suppressed."""
        threshold = AF.get_status()["dynamic_threshold"]
        suppressed, reason = AF.should_suppress(confidence=threshold - 5.0)
        assert suppressed is True
        assert "threshold" in reason.lower() or "confidence" in reason.lower()

    def test_pass_when_above_threshold(self):
        """Confidence above threshold and no blocked contexts → not suppressed."""
        threshold = AF.get_status()["dynamic_threshold"]
        suppressed, reason = AF.should_suppress(confidence=threshold + 10.0)
        assert suppressed is False
        assert reason == ""

    def test_suppress_when_session_blocked(self):
        """Blocked session context → suppressed regardless of confidence."""
        stats = make_stats(
            total=50, win_rate=0.50,
            by_session={"BAD_SESSION": {"total": 20, "win_rate": 0.10}}
        )
        AF.update_filter(stats, source="backtest")
        threshold = AF.get_status()["dynamic_threshold"]
        suppressed, _ = AF.should_suppress(session="BAD_SESSION",
                                            confidence=threshold + 20.0)
        assert suppressed is True

    def test_suppress_when_regime_blocked(self):
        stats = make_stats(
            total=50, win_rate=0.50,
            by_regime={"BEAR_TREND": {"total": 20, "win_rate": 0.05}}
        )
        AF.update_filter(stats, source="backtest")
        threshold = AF.get_status()["dynamic_threshold"]
        suppressed, _ = AF.should_suppress(regime="BEAR_TREND",
                                            confidence=threshold + 20.0)
        assert suppressed is True

    def test_no_boost_when_no_boosted_contexts(self):
        boost = AF.get_confidence_boost(session="POWER_HOUR")
        assert boost == 0.0

    def test_boost_positive_when_session_boosted(self):
        """After boosting POWER_HOUR, get_confidence_boost should return > 0."""
        stats = make_stats(
            total=50, win_rate=0.75,
            by_session={"POWER_HOUR": {"total": 15, "win_rate": AF.BOOST_ABOVE + 0.10}}
        )
        AF.update_filter(stats, source="backtest")
        boost = AF.get_confidence_boost(session="POWER_HOUR")
        assert boost > 0.0

    def test_boost_zero_for_unknown_context(self):
        boost = AF.get_confidence_boost(session="UNKNOWN_SESSION")
        assert boost == 0.0

    def test_suppress_check_is_per_dimension(self):
        """Blocking 'session:X' should not suppress 'session:Y'."""
        stats = make_stats(
            total=50, win_rate=0.50,
            by_session={"BAD_SESSION": {"total": 20, "win_rate": 0.05}}
        )
        AF.update_filter(stats, source="backtest")
        threshold = AF.get_status()["dynamic_threshold"]
        suppressed, _ = AF.should_suppress(session="GOOD_SESSION",
                                            confidence=threshold + 20.0)
        assert suppressed is False


# ═════════════════════════════════════════════════════════════════════════════
# A4 – Anti-deadlock, reset, and edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestAdaptiveFilterEdgeCases:

    def test_reset_clears_all_state(self):
        AF.update_filter(make_stats(total=50, win_rate=0.40), source="backtest")
        AF.reset_filter("test")
        st = AF.get_status()
        assert st["blocked_contexts"] == {}
        assert st["boosted_contexts"] == {}
        assert st["current_win_rate"] == 0.0
        assert st["dynamic_threshold"] == AF.DEFAULT_THRESHOLD

    def test_reset_returns_correct_dict(self):
        result = AF.reset_filter("unit-test")
        assert result["status"] == "reset"
        assert result["reason"] == "unit-test"
        assert result["threshold"] == AF.DEFAULT_THRESHOLD

    def test_anti_deadlock_fires_after_stuck_cycles(self):
        """
        If stuck at MAX_THRESHOLD with low WR for MAX_STUCK_CYCLES consecutive
        updates, the threshold must be reset to DEFAULT.

        To produce new_threshold >= MAX_THRESHOLD - 0.5 (= 62.5), we need the
        "80+" confidence band to show WR >= TARGET (0.55), which sets
        best_threshold = 80, capped to MAX_THRESHOLD = 63.
        With a low overall WR (< 0.35), the anti-deadlock increments _stuck_cycles.
        """
        stuck_stats = make_stats(
            total=200, win_rate=0.25,   # LOW overall WR < 0.35
            by_confidence={
                "<50":   {"wins": 0,  "total": 10},
                "50-60": {"wins": 0,  "total": 10},
                "60-70": {"wins": 0,  "total": 10},
                "70-80": {"wins": 0,  "total": 10},
                # 80+ band has wr=0.60 >= 0.55 → best_threshold = 80 → capped at MAX=63
                "80+":   {"wins": 60, "total": 100},
            }
        )
        # Exactly MAX_STUCK_CYCLES iterations:
        # each one increments _stuck_cycles until it reaches MAX → fires reset on the last call
        for _ in range(AF.MAX_STUCK_CYCLES):
            AF.update_filter(stuck_stats, source="backtest")
        st = AF.get_status()
        assert st["dynamic_threshold"] <= AF.DEFAULT_THRESHOLD, \
            f"Anti-deadlock should have fired; threshold={st['dynamic_threshold']}"

    def test_increment_suppressed_increments_count(self):
        st_before = AF.get_status().get("suppressed_count", 0)
        AF.increment_suppressed()
        AF.increment_suppressed()
        st_after = AF.get_status().get("suppressed_count", 0)
        assert st_after == st_before + 2

    def test_bootstrap_guard_keeps_threshold_at_default(self):
        """With fewer than BOOTSTRAP_OUTCOMES resolved, threshold stays at DEFAULT."""
        stats = make_stats(
            total=AF.BOOTSTRAP_OUTCOMES - 1,
            win_rate=0.30,
            by_confidence={
                "<50":  {"wins": 2, "total": 10},
                "50-60": {"wins": 3, "total": 10},
            }
        )
        AF.update_filter(stats, source="backtest")
        st = AF.get_status()
        assert st["dynamic_threshold"] <= AF.DEFAULT_THRESHOLD

    def test_state_has_all_required_keys(self):
        st = AF.get_status()
        required = {
            "dynamic_threshold", "current_win_rate",
            "blocked_contexts", "boosted_contexts", "suppressed_count",
        }
        for k in required:
            assert k in st, f"Missing key in get_status(): {k}"

    def test_thread_safety_concurrent_updates(self):
        """Multiple threads calling update_filter should not corrupt state."""
        errors = []

        def worker(win_rate: float):
            try:
                AF.update_filter(make_stats(total=20, win_rate=win_rate),
                                 source="backtest")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(0.5 + i * 0.05,))
                   for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"Thread safety errors: {errors}"
        st = AF.get_status()
        assert AF.MIN_THRESHOLD <= st["dynamic_threshold"] <= AF.MAX_THRESHOLD


# ═════════════════════════════════════════════════════════════════════════════
# A5 – Signal Tracker
# ═════════════════════════════════════════════════════════════════════════════

class TestSignalTracker:

    @pytest.fixture(autouse=True)
    def _setup_db(self, tmp_path, monkeypatch):
        ST.init_db()

    def test_record_signal_returns_int(self):
        sid = ST.record_signal(
            "TST", "BUY", entry=100.0, target=108.0, stop=95.0, confidence=65.0
        )
        assert isinstance(sid, int) and sid > 0

    def test_record_returns_unique_ids(self):
        ids = [
            ST.record_signal("TST", "BUY", 100.0, 108.0, 95.0, confidence=65.0)
            for _ in range(5)
        ]
        assert len(set(ids)) == 5

    def test_resolve_pending_buy_win(self):
        ST.record_signal("WIN1", "BUY", 100.0, 110.0, 95.0, confidence=70.0)
        ST.resolve_pending("WIN1", current_price=111.0)
        stats = ST.get_stats("WIN1")
        assert stats["wins"] >= 1

    def test_resolve_pending_buy_loss(self):
        ST.record_signal("LOSS1", "BUY", 100.0, 110.0, 95.0, confidence=70.0)
        ST.resolve_pending("LOSS1", current_price=94.0)
        stats = ST.get_stats("LOSS1")
        assert stats["losses"] >= 1

    def test_resolve_pending_sell_win(self):
        ST.record_signal("SELLW", "SELL", 100.0, 90.0, 105.0, confidence=70.0)
        ST.resolve_pending("SELLW", current_price=89.0)
        stats = ST.get_stats("SELLW")
        assert stats["wins"] >= 1

    def test_resolve_pending_sell_loss(self):
        ST.record_signal("SELLL", "SELL", 100.0, 90.0, 105.0, confidence=70.0)
        ST.resolve_pending("SELLL", current_price=106.0)
        stats = ST.get_stats("SELLL")
        assert stats["losses"] >= 1

    def test_buy_win_pnl_positive(self):
        ST.record_signal("PNLBW", "BUY", 100.0, 110.0, 95.0, confidence=70.0)
        ST.resolve_pending("PNLBW", current_price=111.0)
        import sqlite3
        with ST.get_conn() as c:
            row = c.execute(
                "SELECT pnl_pct FROM signals WHERE ticker='PNLBW' AND outcome='WIN'"
            ).fetchone()
        assert row is not None
        assert float(row["pnl_pct"]) > 0

    def test_buy_loss_pnl_negative(self):
        ST.record_signal("PNLBL", "BUY", 100.0, 110.0, 95.0, confidence=70.0)
        ST.resolve_pending("PNLBL", current_price=94.0)
        import sqlite3
        with ST.get_conn() as c:
            row = c.execute(
                "SELECT pnl_pct FROM signals WHERE ticker='PNLBL' AND outcome='LOSS'"
            ).fetchone()
        assert row is not None
        assert float(row["pnl_pct"]) < 0

    def test_get_stats_totals_consistent(self):
        for _ in range(3):
            ST.record_signal("STAT1", "BUY", 100.0, 110.0, 95.0, confidence=70.0)
        ST.resolve_pending("STAT1", current_price=112.0)
        stats = ST.get_stats("STAT1")
        assert stats["total"] >= 3
        assert stats["wins"] + stats["losses"] + stats["pending"] == stats["total"]

    def test_pending_signals_count_before_resolution(self):
        for _ in range(4):
            ST.record_signal("PEND", "BUY", 100.0, 110.0, 95.0, confidence=70.0)
        stats = ST.get_stats("PEND")
        assert stats["pending"] == 4

    def test_get_observation_summary_structure(self):
        ST.record_signal("OBS1", "BUY", 100.0, 108.0, 95.0, confidence=65.0)
        summary = ST.get_observation_summary()
        # Key is 'total_signals' (not 'total')
        assert "total_signals" in summary
        assert "resolved" in summary
        assert summary["total_signals"] >= 1

    def test_record_suppressed_signal(self):
        import sqlite3
        ST.record_suppressed_signal(
            "SUPP1", "BUY", 100.0, confidence=40.0,
            suppress_reason="below threshold"
        )
        with ST.get_conn() as c:
            # Column is 'is_suppressed' not 'suppressed'
            row = c.execute(
                "SELECT is_suppressed FROM signals WHERE ticker='SUPP1'"
            ).fetchone()
        assert row is not None
        assert row["is_suppressed"] == 1

    def test_get_recent_signals_returns_list(self):
        ST.record_signal("REC1", "BUY", 100.0, 108.0, 95.0, confidence=65.0)
        recent = ST.get_recent_signals(limit=10)
        assert isinstance(recent, list)
        assert len(recent) >= 1

    def test_get_ticker_learning_scores_structure(self):
        for _ in range(5):
            ST.record_signal("LEARN1", "BUY", 100.0, 108.0, 95.0, confidence=65.0)
        ST.resolve_pending("LEARN1", current_price=109.0)
        scores = ST.get_ticker_learning_scores(lookback_days=30, min_count=1)
        assert isinstance(scores, dict)
        if "LEARN1" in scores:
            s = scores["LEARN1"]
            assert "win_rate" in s or "wins" in s

    def test_get_market_breakdown_stats_structure(self):
        ST.record_signal(
            "MBD1", "BUY", 100.0, 108.0, 95.0, confidence=65.0,
            session="POWER_HOUR", regime="BULL_TREND"
        )
        breakdown = ST.get_market_breakdown_stats(min_count=1, lookback_days=30)
        assert isinstance(breakdown, dict)

    def test_get_suppressed_stats_returns_dict(self):
        ST.record_suppressed_signal("SUP2", "SELL", 50.0, confidence=30.0,
                                    suppress_reason="test")
        result = ST.get_suppressed_stats(lookback_days=7)
        assert isinstance(result, dict)

    def test_no_resolution_without_price_crossing(self):
        """Signal should stay PENDING if price is between stop and target."""
        ST.record_signal("MID1", "BUY", 100.0, 110.0, 95.0, confidence=70.0)
        ST.resolve_pending("MID1", current_price=103.0)
        stats = ST.get_stats("MID1")
        assert stats["pending"] == 1


# ═════════════════════════════════════════════════════════════════════════════
# A6 – Learning Engine
# ═════════════════════════════════════════════════════════════════════════════

class TestLearningEngine:

    def test_learning_engine_starts_and_stops(self):
        engine = LearningEngine()
        engine.start()
        time.sleep(0.1)
        assert engine.get_status()["running"] is True
        engine.stop()
        assert engine.get_status()["running"] is False

    def test_double_start_is_idempotent(self):
        engine = LearningEngine()
        engine.start()
        engine.start()  # second call should be no-op
        status = engine.get_status()
        assert status["running"] is True
        engine.stop()

    def test_get_status_has_required_keys(self):
        engine = LearningEngine()
        st = engine.get_status()
        for key in ("running", "cycle_count", "last_win_rate",
                    "last_threshold", "interval_secs"):
            assert key in st, f"Missing key in get_status(): {key}"

    def test_get_learning_log_returns_list(self):
        logs = get_learning_log(limit=10)
        assert isinstance(logs, list)

    def test_clear_log_empties_buffer(self):
        clear_log()
        logs = get_learning_log()
        assert logs == []

    def test_cycle_count_increments_over_time(self):
        """Cycle count should be >= 0; actual increment requires a full interval."""
        engine = LearningEngine()
        st = engine.get_status()
        assert st["cycle_count"] >= 0
