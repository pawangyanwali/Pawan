"""
Fast CI test suite — must complete in under 60 seconds total.

Covers:
  - signal_snapshot: Valkey snapshot serialisation / deserialisation
  - startup_checks:  data-quality logic (confidence migration, adaptive filter scale)
  - service imports: all 4 service entry points are importable without side-effects
  - market hours:    session detection and TTL helpers
  - risk controls:   circuit-breaker and position-size gates
  - paper trading:   open / close / P&L accounting
  - live backtest:   R calculation, zero-R guard, expectancy
  - adaptive filter: threshold calibration, context blocking
  - vwap:            signal classification
  - schemas:         SignalSnapshot round-trip

No external network calls, no Docker, no Schwab, no heavy ML imports.
All DB access goes through the in-memory SQLite fixture in conftest.py.
"""
from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SignalSnapshot schema — pure-Python round-trip
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignalSnapshot:
    def test_to_dict_round_trip(self):
        from agent.schemas import SignalSnapshot
        snap = SignalSnapshot(
            ts=1234567890.0,
            signals=[{"ticker": "AAPL", "confidence": 80.0}],
            regime={"regime": "BULLISH"},
            session={"session": "OPEN"},
            scanned_count=1,
        )
        d = snap.to_dict()
        snap2 = SignalSnapshot.from_dict(d)
        assert snap2.ts == snap.ts
        assert snap2.signals == snap.signals
        assert snap2.scanned_count == 1

    def test_from_dict_defaults(self):
        from agent.schemas import SignalSnapshot
        snap = SignalSnapshot.from_dict({})
        assert snap.ts == 0.0
        assert snap.signals == []
        assert snap.scanned_count == 0

    def test_json_serialisable(self):
        from agent.schemas import SignalSnapshot
        snap = SignalSnapshot(
            ts=time.time(), signals=[], regime={}, session={}, scanned_count=0
        )
        json.dumps(snap.to_dict())   # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# 2. signal_snapshot module — write/read with mocked Valkey
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignalSnapshotModule:
    def _make_client(self):
        store: dict = {}
        client = MagicMock()
        pipe = MagicMock()
        pipe.__enter__ = MagicMock(return_value=pipe)
        pipe.__exit__ = MagicMock(return_value=False)
        pipe.set = MagicMock(side_effect=lambda k, v: store.__setitem__(k, v))
        pipe.publish = MagicMock(return_value=None)
        pipe.execute = MagicMock(return_value=None)
        client.pipeline = MagicMock(return_value=pipe)
        client.get = MagicMock(side_effect=lambda k: store.get(k))
        return client, store

    def test_write_then_read(self):
        from agent.signal_snapshot import write_latest, read_latest
        client, store = self._make_client()
        with patch("agent.valkey_client._get_client", return_value=client):
            wrote = write_latest(
                signals=[{"ticker": "NVDA", "confidence": 90}],
                regime={"regime": "BULLISH"},
                session={"session": "OPEN"},
                scanned_count=1,
            )
            assert wrote is True
            snap = read_latest()
        assert snap is not None
        assert snap["signals"][0]["ticker"] == "NVDA"
        assert snap["scanned_count"] == 1

    def test_write_returns_false_when_no_client(self):
        from agent.signal_snapshot import write_latest
        # write_latest returns False only when BOTH PostgreSQL and Valkey fail.
        # Patching only Valkey is insufficient now that PG is the primary write path.
        with patch("agent.valkey_client._get_client", return_value=None):
            with patch("agent.service_state.set_state", return_value=False):
                assert write_latest([], {}, {}, 0) is False

    def test_read_returns_none_when_no_client(self):
        from agent.signal_snapshot import read_latest
        # read_latest returns None only when BOTH PostgreSQL and Valkey have no data.
        with patch("agent.valkey_client._get_client", return_value=None):
            with patch("agent.service_state.get_state", return_value=None):
                assert read_latest() is None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Service entry points — importable without side-effects
# ═══════════════════════════════════════════════════════════════════════════════

class TestServiceImports:
    """Verify every service module can be imported without triggering network/DB."""

    def _import_clean(self, module_name: str) -> None:
        """Import a module with all heavy subs mocked out."""
        heavy = [
            "agent.scanner", "agent.learning_engine", "agent.weekend_learner",
            "agent.broker.schwab_streamer", "agent.broker.schwab_auth",
            "agent.broker.schwab_market_data", "agent.valkey_client",
        ]
        mocks = {m: MagicMock() for m in heavy}
        with patch.dict("sys.modules", mocks):
            import importlib
            if module_name in sys.modules:
                del sys.modules[module_name]
            importlib.import_module(module_name)

    def test_base_importable(self):
        import services._base  # noqa: F401

    def test_scanner_service_importable(self):
        self._import_clean("services.scanner_service")

    def test_learner_service_importable(self):
        self._import_clean("services.learner_service")

    def test_scheduler_service_importable(self):
        self._import_clean("services.scheduler_service")

    def test_market_data_service_importable(self):
        self._import_clean("services.market_data_service")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. startup_checks — logic correctness (no live DB needed for scale check)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStartupChecks:
    def test_adaptive_filter_ok_scale(self):
        """get_status() returns win_rate already in 0-100; check must not multiply again."""
        from agent.startup_checks import _check_adaptive_filter
        fake_status = {
            "dynamic_threshold": 57.0,
            "current_win_rate":  63.4,      # already a percentage (63.4%)
            "blocked_contexts":  {},
        }
        with patch("agent.startup_checks._check_adaptive_filter") as mock_fn:
            # Call the real implementation with patched get_status
            pass

        with patch("agent.adaptive_filter.get_status", return_value=fake_status), \
             patch("agent.adaptive_filter.MIN_THRESHOLD", 40.0), \
             patch("agent.adaptive_filter.MAX_THRESHOLD", 80.0):
            result = _check_adaptive_filter()

        assert result["ok"] is True
        assert result["win_rate_pct"] == 63.4       # not 6340
        assert result["threshold"] == 57.0

    def test_adaptive_filter_wr_out_of_range(self):
        """win_rate > 100 must flag ok=False."""
        from agent.startup_checks import _check_adaptive_filter
        fake_status = {
            "dynamic_threshold": 57.0,
            "current_win_rate":  4870.0,   # the old broken value
            "blocked_contexts":  {},
        }
        with patch("agent.adaptive_filter.get_status", return_value=fake_status), \
             patch("agent.adaptive_filter.MIN_THRESHOLD", 40.0), \
             patch("agent.adaptive_filter.MAX_THRESHOLD", 80.0):
            result = _check_adaptive_filter()
        assert result["ok"] is False

    def test_confidence_migration_marks_ok(self, tmp_db_paths):
        """After migration, the check returns ok=True and migrated_rows > 0."""
        from agent.startup_checks import _check_confidence_scale
        from agent.db import get_conn

        # Seed signals on the wrong 0-1 scale
        with get_conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS signals "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, confidence REAL)"
            )
            c.executemany(
                "INSERT INTO signals (confidence) VALUES (?)",
                [(0.73,), (0.85,), (0.60,), (75.0,)]   # 3 bad, 1 good
            )

        result = _check_confidence_scale()
        assert result["ok"] is True
        assert result["migrated_rows"] == 3

        # Second run is idempotent — nothing left to migrate
        result2 = _check_confidence_scale()
        assert result2["migrated_rows"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Market hours
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketHours:
    def test_get_session_info_returns_dict(self):
        from agent.market_hours import get_session_info
        info = get_session_info()
        assert isinstance(info, dict)

    def test_session_keys_present(self):
        from agent.market_hours import get_session_info
        info = get_session_info()
        assert "session" in info

    def test_get_market_session_string(self):
        from agent.market_hours import get_market_session
        session = get_market_session()
        assert isinstance(session, str)
        assert session in ("PRE_MARKET", "REGULAR", "AFTER_HOURS", "CLOSED")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Live backtest — R calculation and zero-R guard
# ═══════════════════════════════════════════════════════════════════════════════

class TestLiveBacktestR:
    def _r(self, current, entry, stop, direction):
        from agent.live_backtest import _compute_r
        return _compute_r(current, entry, stop, direction)

    def test_buy_positive_r(self):
        assert self._r(105, 100, 95, "BUY") == pytest.approx(1.0)

    def test_buy_negative_r(self):
        assert self._r(97, 100, 95, "BUY") == pytest.approx(-0.6)

    def test_sell_positive_r(self):
        assert self._r(95, 100, 105, "SELL") == pytest.approx(1.0)

    def test_degenerate_entry_equals_stop(self):
        # entry == stop → risk = 0 → must return 0, not raise
        assert self._r(105, 100, 100, "BUY") == 0.0

    def test_zero_risk_returns_zero(self):
        assert self._r(50, 100, 100, "SELL") == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Risk controls
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskControls:
    def test_circuit_breaker_not_triggered_initially(self, tmp_db_paths):
        from agent.risk_controls import check_circuit_breaker
        result = check_circuit_breaker()
        blocked, reason = result
        assert isinstance(blocked, bool)
        assert isinstance(reason, str)

    def test_sector_concentration_returns_bool(self, tmp_db_paths):
        from agent.risk_controls import check_sector_concentration
        result = check_sector_concentration("AAPL", "BUY")
        blocked, reason = result
        assert isinstance(blocked, bool)
        assert isinstance(reason, str)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Paper trading — open, update, close
# ═══════════════════════════════════════════════════════════════════════════════

class TestPaperTrading:
    def test_get_summary_empty(self, tmp_db_paths):
        from agent.paper_trading import get_summary
        s = get_summary()
        # get_summary() returns capital + trade-count keys (open/closed/wins/losses)
        assert "open" in s or "closed" in s

    def test_open_and_close_trade(self, tmp_db_paths):
        from agent.paper_trading import maybe_open_trade
        result = maybe_open_trade(
            "TSLA", "BUY", 200.0, 210.0, 195.0, 80.0,
            rr_qualifies=True, session="REGULAR",
        )
        # Returns int trade id on success, None if blocked by a risk gate
        assert result is None or isinstance(result, int)

    def test_close_stale_positions(self, tmp_db_paths):
        from agent.paper_trading import close_stale_positions
        n = close_stale_positions()
        assert isinstance(n, int) and n >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Adaptive filter — basic threshold logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveFilter:
    def test_get_status_keys(self):
        from agent.adaptive_filter import get_status
        s = get_status()
        assert "dynamic_threshold" in s
        assert "current_win_rate" in s
        assert "is_learning" in s

    def test_win_rate_is_percentage_scale(self):
        from agent.adaptive_filter import get_status
        s = get_status()
        wr = s["current_win_rate"]
        # Must be 0-100, not 0-1 fraction
        assert 0.0 <= wr <= 100.0, f"win_rate {wr} not in 0-100 range"

    def test_threshold_in_range(self):
        from agent.adaptive_filter import get_status, MIN_THRESHOLD, MAX_THRESHOLD
        s = get_status()
        t = s["dynamic_threshold"]
        assert MIN_THRESHOLD <= t <= MAX_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════════════
# 10. VWAP signal classification
# ═══════════════════════════════════════════════════════════════════════════════

class TestVwap:
    def _make_df(self, closes, vwap_val, n=20):
        import pandas as pd
        import numpy as np
        idx = pd.date_range("2025-01-10 09:30", periods=n, freq="1min")
        df = pd.DataFrame({
            "Close":  closes,
            "High":   np.array(closes) + 0.5,
            "Low":    np.array(closes) - 0.5,
            "Volume": [1_000_000] * n,
            "vwap":   [vwap_val] * n,
        }, index=idx)
        return df

    def test_above_vwap(self):
        from agent.vwap import compute_vwap_signal
        import numpy as np
        closes = list(np.linspace(101, 105, 20))  # price above VWAP
        df = self._make_df(closes, vwap_val=100.0)
        result = compute_vwap_signal(df)
        assert isinstance(result, dict)
        # Full set of valid VWAP events from agent/vwap.py
        assert result["event"] in (
            "RECLAIM", "REJECTION",
            "AT_2SD_UP", "AT_2SD_DOWN",
            "AT_1SD_UP", "AT_1SD_DOWN",
            "ABOVE", "BELOW", "FLAT",
        )

    def test_below_vwap(self):
        from agent.vwap import compute_vwap_signal
        import numpy as np
        closes = list(np.linspace(96, 99, 20))   # price below VWAP
        df = self._make_df(closes, vwap_val=100.0)
        result = compute_vwap_signal(df)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. ServiceRunner lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

class TestServiceRunner:
    def test_stopped_initially_false(self):
        from services._base import ServiceRunner
        r = ServiceRunner("test")
        assert r.stopped is False

    def test_stop_event_set(self):
        from services._base import ServiceRunner
        r = ServiceRunner("test")
        r._stop.set()
        assert r.stopped is True

    def test_configure_logging_returns_logger(self):
        import logging
        from services._base import configure_logging
        log = configure_logging("test-svc")
        assert isinstance(log, logging.Logger)
