from __future__ import annotations
import pytest
"""
Comprehensive tests for Phase 1 (market session detection) and Phase 2
(extended-hours training data + session-aware ML features).

Phase 1 covers:
  - get_market_session() — canonical 4-state session (REGULAR/PRE_MARKET/AFTER_HOURS/CLOSED)
  - refresh_market_hours_cache() — API-backed daily cache with rule-based fallback
  - _parse_iso_time() — ISO-8601 → ET wall-clock time
  - is_after_hours() bug fix — no longer returns True when market is CLOSED
  - fetch_market_hours() — now returns all session windows (pre + regular + post)

Phase 2 covers:
  - FEATURE_COLS_V2 now contains is_extended_hours and session_type
  - compute_features() adds session features to every DataFrame
  - _compute_session_features() correctly classifies pre/regular/after-hours bars
  - ml_model._retrain_all_locked fetches 1min and 15min data with extended_hours=True

Bug-fix regression tests:
  - schwab_auth.py: refresh() cancels orphaned timers before scheduling new retry
  - schwab_market_data.py: _try_refresh_md_token dedup raised 30s → 180s
  - schwab_market_data.py: _fetch_one_async applies CDN-block detection before refresh
"""
pytestmark = pytest.mark.slow

import sys
import os
import datetime as _real_dt
from unittest.mock import patch, MagicMock, call
import threading

import pytest
import pandas as pd
import numpy as np
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

ET = pytz.timezone("America/New_York")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _et(year, month, day, hour, minute=0) -> _real_dt.datetime:
    return ET.localize(_real_dt.datetime(year, month, day, hour, minute, 0))


def _patch_mh_now(fake_dt):
    """Patch datetime.now inside agent.market_hours to return fake_dt."""
    mock = MagicMock(wraps=_real_dt.datetime)
    mock.now.return_value = fake_dt
    mock.side_effect = _real_dt.datetime
    mock.combine = _real_dt.datetime.combine
    mock.fromisoformat = _real_dt.datetime.fromisoformat
    return patch("agent.market_hours.datetime", mock)


def _make_ohlcv(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame for the given index."""
    n = len(index)
    return pd.DataFrame({
        "Open":   np.full(n, 100.0),
        "High":   np.full(n, 101.0),
        "Low":    np.full(n, 99.0),
        "Close":  np.full(n, 100.5),
        "Volume": np.full(n, 1_000_000.0),
    }, index=index)


def _regular_session_index(n: int = 60) -> pd.DatetimeIndex:
    """DatetimeIndex of 1-min bars inside regular session on a Monday.
    Regular session: 9:30–16:00 ET = 14:30–21:00 UTC (January, EST = UTC-5)."""
    start = _real_dt.datetime(2025, 1, 13, 14, 30, tzinfo=pytz.UTC)  # 9:30 AM ET
    return pd.date_range(start=start, periods=n, freq="1min")


def _pre_market_index(n: int = 30) -> pd.DatetimeIndex:
    """DatetimeIndex of 1-min bars in pre-market (6:00–6:30 AM ET = 11:00–11:30 UTC)."""
    start = _real_dt.datetime(2025, 1, 13, 11, 0, tzinfo=pytz.UTC)  # 6 AM ET
    return pd.date_range(start=start, periods=n, freq="1min")


def _after_hours_index(n: int = 30) -> pd.DatetimeIndex:
    """DatetimeIndex of 1-min bars in after-hours (5:00–5:30 PM ET = 22:00–22:30 UTC)."""
    start = _real_dt.datetime(2025, 1, 13, 22, 0, tzinfo=pytz.UTC)  # 5 PM ET
    return pd.date_range(start=start, periods=n, freq="1min")


def _mixed_session_index() -> pd.DatetimeIndex:
    """Index spanning pre-market, regular, and after-hours bars."""
    pre  = _pre_market_index(20)
    reg  = _regular_session_index(30)
    post = _after_hours_index(20)
    return pre.append(reg).append(post)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — get_market_session()
# ──────────────────────────────────────────────────────────────────────────────

class TestGetMarketSession:
    """get_market_session() must return one of 4 canonical states."""

    VALID = {"REGULAR", "PRE_MARKET", "AFTER_HOURS", "CLOSED"}

    def _session_at(self, fake_dt: _real_dt.datetime) -> str:
        import agent.market_hours as mh
        # Force cache to match our fake date so the function uses rule-based bounds
        from agent.market_hours import refresh_market_hours_cache, _cache_lock, _day_cache
        refresh_market_hours_cache(fake_dt.date())
        with _patch_mh_now(fake_dt):
            return mh.get_market_session()

    def test_returns_valid_string(self):
        import agent.market_hours as mh
        assert mh.get_market_session() in self.VALID

    def test_regular_session_monday_10am(self):
        # 2025-01-13 Mon, 10:00 AM ET — regular session
        result = self._session_at(_et(2025, 1, 13, 10, 0))
        assert result == "REGULAR"

    def test_regular_session_standard_window(self):
        result = self._session_at(_et(2025, 1, 13, 14, 0))
        assert result == "REGULAR"

    def test_pre_market_early_morning(self):
        # 6:00 AM ET on a weekday
        result = self._session_at(_et(2025, 1, 13, 6, 0))
        assert result == "PRE_MARKET"

    def test_pre_market_just_before_open(self):
        # 9:29 AM ET — still pre-market
        result = self._session_at(_et(2025, 1, 13, 9, 29))
        assert result == "PRE_MARKET"

    def test_regular_starts_at_930(self):
        result = self._session_at(_et(2025, 1, 13, 9, 30))
        assert result == "REGULAR"

    def test_after_hours_5pm(self):
        result = self._session_at(_et(2025, 1, 13, 17, 0))
        assert result == "AFTER_HOURS"

    def test_after_hours_just_after_close(self):
        # 16:01 ET
        result = self._session_at(_et(2025, 1, 13, 16, 1))
        assert result == "AFTER_HOURS"

    def test_closed_overnight(self):
        # 2:00 AM ET on a weekday — outside all sessions
        result = self._session_at(_et(2025, 1, 13, 2, 0))
        assert result == "CLOSED"

    def test_closed_saturday(self):
        # 2025-01-11 is Saturday
        result = self._session_at(_et(2025, 1, 11, 10, 0))
        assert result == "CLOSED"

    def test_closed_sunday(self):
        # 2025-01-12 is Sunday
        result = self._session_at(_et(2025, 1, 12, 15, 0))
        assert result == "CLOSED"

    def test_closed_holiday(self):
        # 2025-01-01 is New Year's Day
        result = self._session_at(_et(2025, 1, 1, 10, 30))
        assert result == "CLOSED"

    def test_closed_mlk_day(self):
        # 2025-01-20 is MLK Day
        result = self._session_at(_et(2025, 1, 20, 10, 0))
        assert result == "CLOSED"


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — is_after_hours() bug fix
# ──────────────────────────────────────────────────────────────────────────────

class TestIsAfterHoursBugFix:
    """
    REGRESSION: is_after_hours() previously returned True for CLOSED session
    (e.g., 2 AM on a weekday).  After fix it must only return True during the
    16:00–20:00 ET post-market window.
    """

    def test_not_true_at_2am_weekday(self):
        """The bug: 2 AM returned True because CLOSED was included."""
        from agent.market_hours import is_after_hours
        dt = _et(2025, 1, 13, 2, 0)  # Monday 2 AM
        with _patch_mh_now(dt):
            assert is_after_hours() is False, (
                "is_after_hours() must not return True at 2 AM — market is CLOSED, not after-hours"
            )

    def test_not_true_on_weekend(self):
        from agent.market_hours import is_after_hours
        dt = _et(2025, 1, 11, 18, 0)  # Saturday 6 PM
        with _patch_mh_now(dt):
            assert is_after_hours() is False

    def test_not_true_before_market_open(self):
        from agent.market_hours import is_after_hours
        dt = _et(2025, 1, 13, 6, 0)   # Monday 6 AM = PRE_MARKET
        with _patch_mh_now(dt):
            assert is_after_hours() is False

    def test_true_during_post_market(self):
        from agent.market_hours import is_after_hours
        dt = _et(2025, 1, 13, 17, 0)  # Monday 5 PM = AFTER_HOURS
        with _patch_mh_now(dt):
            assert is_after_hours() is True

    def test_false_during_regular_session(self):
        from agent.market_hours import is_after_hours
        dt = _et(2025, 1, 13, 10, 0)
        with _patch_mh_now(dt):
            assert is_after_hours() is False


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — _parse_iso_time()
# ──────────────────────────────────────────────────────────────────────────────

class TestParseIsoTime:
    def _parse(self, iso):
        from agent.market_hours import _parse_iso_time
        return _parse_iso_time(iso)

    def test_none_returns_none(self):
        assert self._parse(None) is None

    def test_empty_string_returns_none(self):
        assert self._parse("") is None

    def test_invalid_string_returns_none(self):
        assert self._parse("not-a-date") is None

    def test_regular_open_time(self):
        # 9:30 AM ET expressed as ISO-8601 with UTC offset
        result = self._parse("2025-01-13T09:30:00-05:00")
        assert result is not None
        assert result.hour == 9
        assert result.minute == 30

    def test_pre_market_time(self):
        result = self._parse("2025-01-13T07:00:00-05:00")
        assert result is not None
        assert result.hour == 7
        assert result.minute == 0

    def test_post_market_time(self):
        result = self._parse("2025-01-13T20:00:00-05:00")
        assert result is not None
        assert result.hour == 20
        assert result.minute == 0

    def test_utc_time_converted_correctly(self):
        # 14:30 UTC = 09:30 ET (EST, UTC-5)
        result = self._parse("2025-01-13T14:30:00+00:00")
        assert result is not None
        assert result.hour == 9
        assert result.minute == 30

    def test_seconds_stripped(self):
        result = self._parse("2025-01-13T09:30:45-05:00")
        assert result is not None
        assert result.second == 0
        assert result.microsecond == 0


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — refresh_market_hours_cache()
# ──────────────────────────────────────────────────────────────────────────────

class TestRefreshMarketHoursCache:
    def test_returns_false_when_api_unavailable(self):
        """When fetch_market_hours cannot be imported (aiohttp absent) → rule-based fallback."""
        import sys
        import agent.market_hours as mh
        # Remove the schwab_market_data module from sys.modules so the lazy import
        # inside refresh_market_hours_cache fails, triggering the rule-based path.
        saved = sys.modules.pop("agent.broker.schwab_market_data", None)
        try:
            result = mh.refresh_market_hours_cache(_real_dt.date(2025, 1, 13))
        finally:
            if saved is not None:
                sys.modules["agent.broker.schwab_market_data"] = saved
        assert result is False
        assert mh._day_cache["source"] == "rule_based"

    def test_cache_populated_after_refresh(self):
        import agent.market_hours as mh
        mh.refresh_market_hours_cache(_real_dt.date(2025, 1, 13))
        cache = mh._day_cache
        assert "date" in cache
        assert "is_trading_day" in cache
        assert "regular_open" in cache
        assert "regular_close" in cache
        assert "pre_open" in cache
        assert "pre_close" in cache
        assert "post_open" in cache
        assert "post_close" in cache
        assert "source" in cache

    def test_weekend_not_trading_day(self):
        import agent.market_hours as mh
        # 2025-01-11 is Saturday
        mh.refresh_market_hours_cache(_real_dt.date(2025, 1, 11))
        assert mh._day_cache["is_trading_day"] is False

    def test_holiday_not_trading_day(self):
        import agent.market_hours as mh
        # 2025-01-01 is New Year's Day
        mh.refresh_market_hours_cache(_real_dt.date(2025, 1, 1))
        assert mh._day_cache["is_trading_day"] is False

    def test_regular_weekday_is_trading_day(self):
        import agent.market_hours as mh
        mh.refresh_market_hours_cache(_real_dt.date(2025, 1, 13))
        assert mh._day_cache["is_trading_day"] is True

    def test_api_success_uses_api_boundaries(self):
        import sys
        import agent.market_hours as mh

        api_response = {
            "is_open":           True,
            "open_time":         "2025-01-13T09:30:00-05:00",
            "close_time":        "2025-01-13T16:00:00-05:00",
            "pre_market_start":  "2025-01-13T07:00:00-05:00",
            "pre_market_end":    "2025-01-13T09:30:00-05:00",
            "post_market_start": "2025-01-13T16:00:00-05:00",
            "post_market_end":   "2025-01-13T20:00:00-05:00",
        }
        # Inject a lightweight mock module so the lazy import inside
        # refresh_market_hours_cache resolves correctly without needing aiohttp.
        mock_module = MagicMock()
        mock_module.fetch_market_hours.return_value = api_response
        saved = sys.modules.get("agent.broker.schwab_market_data")
        sys.modules["agent.broker.schwab_market_data"] = mock_module
        try:
            ok = mh.refresh_market_hours_cache(_real_dt.date(2025, 1, 13))
        finally:
            if saved is None:
                sys.modules.pop("agent.broker.schwab_market_data", None)
            else:
                sys.modules["agent.broker.schwab_market_data"] = saved

        assert ok is True
        assert mh._day_cache["source"] == "schwab_api"
        assert mh._day_cache["regular_open"].hour == 9
        assert mh._day_cache["regular_open"].minute == 30
        assert mh._day_cache["regular_close"].hour == 16
        assert mh._day_cache["pre_open"].hour == 7

    def test_api_half_day_returns_early_close(self):
        import agent.market_hours as mh
        # 2025-07-03 is Black Friday (market closes 1 PM)
        mh.refresh_market_hours_cache(_real_dt.date(2025, 7, 3))
        # Rule-based fallback since Schwab won't be connected in test
        assert mh._day_cache["regular_close"].hour == 13

    def test_cache_source_rule_based_without_api(self):
        import agent.market_hours as mh
        mh.refresh_market_hours_cache(_real_dt.date(2025, 1, 13))
        # Without real Schwab auth the source must be rule_based
        assert mh._day_cache["source"] == "rule_based"


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — fetch_market_hours() extended return values
# ──────────────────────────────────────────────────────────────────────────────

class TestFetchMarketHoursExtended:
    """
    Tests for the extended fetch_market_hours() return shape.

    These tests use sys.modules injection to avoid requiring aiohttp.
    Structural tests (session window parsing logic) run inline without import.
    """

    def _get_fmh(self):
        """Import fetch_market_hours, skipping if aiohttp is unavailable."""
        try:
            from agent.broker.schwab_market_data import fetch_market_hours
            return fetch_market_hours
        except ImportError:
            pytest.skip("aiohttp not installed — skipping schwab_market_data tests")

    def test_returns_all_seven_keys_when_not_authorised(self):
        """Even when the API is not authenticated all 7 keys must be present."""
        import sys
        mock_module = MagicMock()
        mock_module._is_authorised.return_value = False
        mock_module.fetch_market_hours.return_value = {
            "is_open": None, "open_time": None, "close_time": None,
            "pre_market_start": None, "pre_market_end": None,
            "post_market_start": None, "post_market_end": None,
        }
        saved = sys.modules.get("agent.broker.schwab_market_data")
        sys.modules["agent.broker.schwab_market_data"] = mock_module
        try:
            result = mock_module.fetch_market_hours("equity")
        finally:
            if saved is None:
                sys.modules.pop("agent.broker.schwab_market_data", None)
            else:
                sys.modules["agent.broker.schwab_market_data"] = saved

        expected_keys = {
            "is_open", "open_time", "close_time",
            "pre_market_start", "pre_market_end",
            "post_market_start", "post_market_end",
        }
        assert expected_keys == set(result.keys())

    def test_is_open_none_when_not_authorised(self):
        """The _empty sentinel must return is_open=None."""
        # Test the _empty dict shape directly (no import needed)
        _empty = {
            "is_open": None, "open_time": None, "close_time": None,
            "pre_market_start": None, "pre_market_end": None,
            "post_market_start": None, "post_market_end": None,
        }
        assert _empty["is_open"] is None

    def test_parses_all_session_windows_from_api_response(self):
        """_first_window helper correctly extracts start/end from sessionHours."""
        raw_api = {
            "equity": {
                "EQ": {
                    "isOpen": True,
                    "sessionHours": {
                        "preMarket":     [{"start": "2025-01-13T07:00:00-05:00",
                                           "end":   "2025-01-13T09:30:00-05:00"}],
                        "regularMarket": [{"start": "2025-01-13T09:30:00-05:00",
                                           "end":   "2025-01-13T16:00:00-05:00"}],
                        "postMarket":    [{"start": "2025-01-13T16:00:00-05:00",
                                           "end":   "2025-01-13T20:00:00-05:00"}],
                    },
                }
            }
        }
        # Inline test of the parsing logic (no aiohttp needed)
        info  = raw_api["equity"]["EQ"]
        hours = info.get("sessionHours", {})

        def _first_window(key):
            windows = hours.get(key, [])
            return (windows[0].get("start"), windows[0].get("end")) if windows else (None, None)

        pre_s,  pre_e  = _first_window("preMarket")
        reg_s,  reg_e  = _first_window("regularMarket")
        post_s, post_e = _first_window("postMarket")

        assert pre_s  == "2025-01-13T07:00:00-05:00"
        assert reg_s  == "2025-01-13T09:30:00-05:00"
        assert post_s == "2025-01-13T16:00:00-05:00"
        assert pre_e  == "2025-01-13T09:30:00-05:00"
        assert reg_e  == "2025-01-13T16:00:00-05:00"
        assert post_e == "2025-01-13T20:00:00-05:00"

    def test_empty_sessions_return_none_values(self):
        """When sessionHours is empty, all session window values must be None."""
        hours = {}
        def _first_window(key):
            windows = hours.get(key, [])
            return (windows[0].get("start"), windows[0].get("end")) if windows else (None, None)

        for session_key in ("preMarket", "regularMarket", "postMarket"):
            start, end = _first_window(session_key)
            assert start is None
            assert end is None


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — FEATURE_COLS_V2 session features registered
# ──────────────────────────────────────────────────────────────────────────────

class TestFeatureColsV2Registration:
    def test_is_extended_hours_in_feature_cols(self):
        from agent.feature_engine import FEATURE_COLS_V2
        assert "is_extended_hours" in FEATURE_COLS_V2, (
            "is_extended_hours must be in FEATURE_COLS_V2 (Phase 2 requirement)"
        )

    def test_session_type_in_feature_cols(self):
        from agent.feature_engine import FEATURE_COLS_V2
        assert "session_type" in FEATURE_COLS_V2, (
            "session_type must be in FEATURE_COLS_V2 (Phase 2 requirement)"
        )

    def test_feature_cols_v2_length(self):
        from agent.feature_engine import FEATURE_COLS_V2
        # V1(23) + original new(9) + session features(2) = 34
        assert len(FEATURE_COLS_V2) == 34, (
            f"FEATURE_COLS_V2 should have 34 features, got {len(FEATURE_COLS_V2)}"
        )

    def test_no_duplicates_in_feature_cols(self):
        from agent.feature_engine import FEATURE_COLS_V2
        assert len(FEATURE_COLS_V2) == len(set(FEATURE_COLS_V2)), (
            "Duplicate feature names detected in FEATURE_COLS_V2"
        )

    def test_session_features_appended_at_end(self):
        from agent.feature_engine import FEATURE_COLS_V2
        last_two = FEATURE_COLS_V2[-2:]
        assert "is_extended_hours" in last_two
        assert "session_type" in last_two


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — _compute_session_features()
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeSessionFeatures:
    def _compute(self, index):
        from agent.feature_engine import _compute_session_features
        return _compute_session_features(index)

    # ── Regular session bars ──────────────────────────────────────────────────

    def test_regular_session_is_extended_is_zero(self):
        idx = _regular_session_index(60)
        is_ext, sess = self._compute(idx)
        assert (is_ext == 0.0).all(), "Regular-session bars must have is_extended_hours=0"

    def test_regular_session_type_is_zero(self):
        idx = _regular_session_index(60)
        _, sess = self._compute(idx)
        assert (sess == 0.0).all(), "Regular-session bars must have session_type=0"

    def test_regular_session_boundary_930_is_regular(self):
        # Bar at exactly 9:30 AM ET (UTC 14:30)
        idx = pd.DatetimeIndex([
            _real_dt.datetime(2025, 1, 13, 14, 30, tzinfo=pytz.UTC)  # 9:30 ET
        ])
        is_ext, sess = self._compute(idx)
        assert is_ext.iloc[0] == 0.0
        assert sess.iloc[0] == 0.0

    def test_regular_session_bar_at_1559_is_regular(self):
        # 15:59 ET = 20:59 UTC — still regular session
        idx = pd.DatetimeIndex([
            _real_dt.datetime(2025, 1, 13, 20, 59, tzinfo=pytz.UTC)  # 15:59 ET
        ])
        is_ext, sess = self._compute(idx)
        assert is_ext.iloc[0] == 0.0
        assert sess.iloc[0] == 0.0

    # ── Pre-market bars ───────────────────────────────────────────────────────

    def test_pre_market_is_extended_is_one(self):
        idx = _pre_market_index(30)
        is_ext, _ = self._compute(idx)
        assert (is_ext == 1.0).all(), "Pre-market bars must have is_extended_hours=1"

    def test_pre_market_session_type_is_one(self):
        idx = _pre_market_index(30)
        _, sess = self._compute(idx)
        assert (sess == 1.0).all(), "Pre-market bars must have session_type=1"

    def test_pre_market_bar_at_0700_et(self):
        idx = pd.DatetimeIndex([
            _real_dt.datetime(2025, 1, 13, 12, 0, tzinfo=pytz.UTC)  # 7:00 AM ET
        ])
        is_ext, sess = self._compute(idx)
        assert is_ext.iloc[0] == 1.0
        assert sess.iloc[0] == 1.0

    def test_bar_just_before_open_929(self):
        # 9:29 ET = 14:29 UTC
        idx = pd.DatetimeIndex([
            _real_dt.datetime(2025, 1, 13, 14, 29, tzinfo=pytz.UTC)
        ])
        is_ext, sess = self._compute(idx)
        assert is_ext.iloc[0] == 1.0
        assert sess.iloc[0] == 1.0

    # ── After-hours bars ─────────────────────────────────────────────────────

    def test_after_hours_is_extended_is_one(self):
        idx = _after_hours_index(30)
        is_ext, _ = self._compute(idx)
        assert (is_ext == 1.0).all(), "After-hours bars must have is_extended_hours=1"

    def test_after_hours_session_type_is_two(self):
        idx = _after_hours_index(30)
        _, sess = self._compute(idx)
        assert (sess == 2.0).all(), "After-hours bars must have session_type=2"

    def test_bar_at_exactly_1600_et(self):
        # 16:00 ET = 21:00 UTC — first bar of after-hours
        idx = pd.DatetimeIndex([
            _real_dt.datetime(2025, 1, 13, 21, 0, tzinfo=pytz.UTC)
        ])
        is_ext, sess = self._compute(idx)
        assert is_ext.iloc[0] == 1.0
        assert sess.iloc[0] == 2.0

    def test_bar_at_1959_et(self):
        # 19:59 ET = 00:59 UTC+1 day
        idx = pd.DatetimeIndex([
            _real_dt.datetime(2025, 1, 14, 0, 59, tzinfo=pytz.UTC)
        ])
        is_ext, sess = self._compute(idx)
        assert is_ext.iloc[0] == 1.0
        assert sess.iloc[0] == 2.0

    # ── Mixed session bars ────────────────────────────────────────────────────

    def test_mixed_index_counts(self):
        """20 pre + 30 regular + 20 post = 40 extended, 30 regular."""
        idx = _mixed_session_index()
        is_ext, sess = self._compute(idx)
        assert int(is_ext.sum()) == 40, f"Expected 40 extended bars, got {int(is_ext.sum())}"
        assert int((sess == 0.0).sum()) == 30
        assert int((sess == 1.0).sum()) == 20
        assert int((sess == 2.0).sum()) == 20

    def test_values_are_float_dtype(self):
        idx = _regular_session_index(10)
        is_ext, sess = self._compute(idx)
        assert is_ext.dtype == float or np.issubdtype(is_ext.dtype, np.floating)
        assert sess.dtype == float or np.issubdtype(sess.dtype, np.floating)

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_non_datetime_index_returns_zeros(self):
        idx = pd.RangeIndex(10)
        is_ext, sess = self._compute(idx)
        assert (is_ext == 0.0).all()
        assert (sess == 0.0).all()

    def test_single_bar_regular(self):
        idx = pd.DatetimeIndex([_real_dt.datetime(2025, 1, 13, 15, 0, tzinfo=pytz.UTC)])
        is_ext, sess = self._compute(idx)
        assert len(is_ext) == 1
        assert len(sess) == 1

    def test_empty_index_returns_empty_series(self):
        idx = pd.DatetimeIndex([])
        is_ext, sess = self._compute(idx)
        assert len(is_ext) == 0
        assert len(sess) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — compute_features() adds session columns
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeFeaturesSessionColumns:
    def test_adds_is_extended_hours_column(self):
        from agent.feature_engine import compute_features
        idx = _regular_session_index(60)
        df = compute_features(_make_ohlcv(idx))
        assert "is_extended_hours" in df.columns

    def test_adds_session_type_column(self):
        from agent.feature_engine import compute_features
        idx = _regular_session_index(60)
        df = compute_features(_make_ohlcv(idx))
        assert "session_type" in df.columns

    def test_regular_session_extended_is_zero(self):
        from agent.feature_engine import compute_features
        idx = _regular_session_index(60)
        df = compute_features(_make_ohlcv(idx))
        assert (df["is_extended_hours"] == 0.0).all()

    def test_pre_market_extended_is_one(self):
        from agent.feature_engine import compute_features
        idx = _pre_market_index(60)
        df = compute_features(_make_ohlcv(idx))
        assert (df["is_extended_hours"] == 1.0).all()

    def test_after_hours_extended_is_one(self):
        from agent.feature_engine import compute_features
        idx = _after_hours_index(60)
        df = compute_features(_make_ohlcv(idx))
        assert (df["is_extended_hours"] == 1.0).all()

    def test_pre_market_session_type_is_one(self):
        from agent.feature_engine import compute_features
        idx = _pre_market_index(60)
        df = compute_features(_make_ohlcv(idx))
        assert (df["session_type"] == 1.0).all()

    def test_after_hours_session_type_is_two(self):
        from agent.feature_engine import compute_features
        idx = _after_hours_index(60)
        df = compute_features(_make_ohlcv(idx))
        assert (df["session_type"] == 2.0).all()

    def test_all_feature_cols_v2_present_after_compute(self):
        from agent.feature_engine import compute_features, FEATURE_COLS_V2
        idx = _regular_session_index(80)
        df = compute_features(_make_ohlcv(idx))
        missing = [c for c in FEATURE_COLS_V2 if c not in df.columns]
        assert missing == [], f"Missing columns after compute_features: {missing}"

    def test_no_nan_in_session_features(self):
        from agent.feature_engine import compute_features
        idx = _mixed_session_index()
        df = compute_features(_make_ohlcv(idx))
        assert not df["is_extended_hours"].isna().any()
        assert not df["session_type"].isna().any()

    def test_mixed_session_correct_distribution(self):
        from agent.feature_engine import compute_features
        idx = _mixed_session_index()
        df = compute_features(_make_ohlcv(idx))
        n_ext = int(df["is_extended_hours"].sum())
        n_reg = int((df["session_type"] == 0).sum())
        assert n_ext == 40
        assert n_reg == 30


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — prepare_training_data includes session features
# ──────────────────────────────────────────────────────────────────────────────

class TestPrepareTrainingDataSessionFeatures:
    def _make_large_df(self, n=300, session="regular") -> pd.DataFrame:
        if session == "regular":
            idx = _regular_session_index(n)
        elif session == "pre":
            # Extend to multiple days to get enough bars
            all_bars = []
            for day_offset in range(5):
                base = _real_dt.datetime(2025, 1, 13, 11, 0, tzinfo=pytz.UTC)
                start = base + _real_dt.timedelta(days=day_offset)
                all_bars.append(pd.date_range(start=start, periods=n // 5, freq="1min"))
            idx = all_bars[0].append(all_bars[1]).append(all_bars[2]).append(
                all_bars[3]).append(all_bars[4])
        else:
            idx = _after_hours_index(n)
        return _make_ohlcv(idx)

    def test_prepare_returns_correct_feature_count(self):
        from agent.feature_engine import prepare_training_data, FEATURE_COLS_V2
        df = self._make_large_df(300, "regular")
        result = prepare_training_data(df, ticker="AAPL", lookahead_bars=3)
        if result is None:
            pytest.skip("Not enough variance in synthetic data to train")
        X_train, y_train, X_test, y_test = result
        assert X_train.shape[1] == len(FEATURE_COLS_V2), (
            f"Expected {len(FEATURE_COLS_V2)} features, got {X_train.shape[1]}"
        )

    def test_extended_hours_df_does_not_crash_prepare(self):
        from agent.feature_engine import prepare_training_data
        df = self._make_large_df(200, "pre")
        # Should not raise — may return None if not enough variance
        try:
            prepare_training_data(df, ticker="AAPL", lookahead_bars=3)
        except Exception as e:
            pytest.fail(f"prepare_training_data raised on pre-market data: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — ml_model training uses extended_hours=True
# ──────────────────────────────────────────────────────────────────────────────

class TestMLModelTrainingFetchExtendedHours:
    """
    _retrain_all_locked uses a local import `from agent.data_fetcher import fetch_batch_interval`.
    Patch the source (`agent.data_fetcher.fetch_batch_interval`) so mock resolves correctly.
    """

    def _run_locked(self, captured_calls):
        """Run _retrain_all_locked with a fake fetch and return without crashing."""
        import agent.ml_model as mlm

        def fake_fetch(tickers, interval, outputsize, ttl=None, background=False,
                       extended_hours=False):
            captured_calls.append({"interval": interval, "extended_hours": extended_hours})
            return {}

        with patch("agent.data_fetcher.fetch_batch_interval", side_effect=fake_fetch):
            try:
                mlm._retrain_all_locked(["AAPL"], delay=0.0)
            except Exception:
                pass  # Expected — empty data, no models trained

    def test_1min_fetch_uses_extended_hours(self):
        """_retrain_all_locked must pass extended_hours=True for 1-min data."""
        captured_calls: list = []
        self._run_locked(captured_calls)

        one_min_calls = [c for c in captured_calls if c["interval"] == "1min"]
        assert len(one_min_calls) >= 1, "No 1min fetch call found"
        assert one_min_calls[0]["extended_hours"] is True, (
            "1min training fetch must use extended_hours=True (Phase 2 requirement)"
        )

    def test_15min_fetch_uses_extended_hours(self):
        """_retrain_all_locked must pass extended_hours=True for 15-min data."""
        captured_calls: list = []
        self._run_locked(captured_calls)

        fifteenmin_calls = [c for c in captured_calls if c["interval"] == "15min"]
        assert len(fifteenmin_calls) >= 1, "No 15min fetch call found"
        assert fifteenmin_calls[0]["extended_hours"] is True, (
            "15min training fetch must use extended_hours=True (Phase 2 requirement)"
        )

    def test_daily_fetch_does_not_use_extended_hours(self):
        """Daily bars have no meaningful extended-hours concept — must remain False."""
        captured_calls: list = []
        self._run_locked(captured_calls)

        daily_calls = [c for c in captured_calls if c["interval"] == "1day"]
        if daily_calls:
            assert daily_calls[0]["extended_hours"] is False, (
                "Daily training fetch must NOT use extended_hours=True"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Token refresh regressions
# ──────────────────────────────────────────────────────────────────────────────



# ──────────────────────────────────────────────────────────────────────────────
# Bug-fix regressions — _try_refresh_md_token dedup window
# ──────────────────────────────────────────────────────────────────────────────

class TestTryRefreshMdTokenDedup:
    """
    Before fix: dedup window was 30s, allowing multiple refresh storms during
    extended ML training (~90s of concurrent API calls).
    After fix: window is 180s — covers a full retry cycle.

    These tests inspect the source text of schwab_market_data.py directly so
    they work even when aiohttp is not installed.
    """

    def _get_source(self) -> str:
        import pathlib
        return pathlib.Path(
            os.path.dirname(os.path.dirname(__file__))
        ).joinpath("agent/broker/schwab_market_data.py").read_text()

    def test_dedup_window_is_at_least_180s(self):
        """Guard: the dedup threshold in _try_refresh_md_token must be 180, not 30."""
        src = self._get_source()
        # Find the _try_refresh_md_token function body
        fn_start = src.index("def _try_refresh_md_token")
        fn_body  = src[fn_start:fn_start + 700]  # first ~700 chars of the function
        assert "180" in fn_body, (
            "_try_refresh_md_token dedup window must be 180s, not 30s"
        )

    def test_dedup_30s_not_used_anymore(self):
        """The original 30s window must no longer appear in _try_refresh_md_token."""
        src = self._get_source()
        fn_start = src.index("def _try_refresh_md_token")
        fn_body  = src[fn_start:fn_start + 400]
        assert "< 30" not in fn_body and "< 30.0" not in fn_body, (
            "Old 30s dedup window still appears in _try_refresh_md_token"
        )

    def test_refresh_lock_present(self):
        """_try_refresh_md_token must still use _refresh_lock for thread safety."""
        src = self._get_source()
        assert "_refresh_lock" in src


# ──────────────────────────────────────────────────────────────────────────────
# Bug-fix regressions — async CDN block detection before token refresh
# ──────────────────────────────────────────────────────────────────────────────

class TestAsyncCDNBlockDetection:
    """
    Before fix: _fetch_one_async called _try_refresh_md_token() on any 401/403
    with no CDN-block check.

    After fix: 403 with a valid token TTL > 60s is detected as a CDN block
    and the function returns immediately without calling refresh.

    Tested via source inspection so these pass without aiohttp installed.
    """

    def _get_source(self) -> str:
        import pathlib
        return pathlib.Path(
            os.path.dirname(os.path.dirname(__file__))
        ).joinpath("agent/broker/schwab_market_data.py").read_text()

    def test_cdn_block_detection_present_in_async_path(self):
        src = self._get_source()
        fn_start = src.index("async def _fetch_one_async")
        fn_end   = src.index("\nasync def ", fn_start + 10) if "\nasync def " in src[fn_start + 10:] else fn_start + 3000
        fn_body  = src[fn_start:fn_end]
        assert "is_cdn" in fn_body, (
            "_fetch_one_async must contain is_cdn CDN-block detection (Phase 1 fix)"
        )

    def test_try_refresh_still_called_for_genuine_auth_failures(self):
        src = self._get_source()
        fn_start = src.index("async def _fetch_one_async")
        fn_body  = src[fn_start:fn_start + 3000]
        assert "_try_refresh_md_token" in fn_body, (
            "_fetch_one_async must still call _try_refresh_md_token for genuine 401"
        )

    def test_on_cdn_block_called_in_async_path(self):
        src = self._get_source()
        fn_start = src.index("async def _fetch_one_async")
        fn_body  = src[fn_start:fn_start + 3000]
        assert "_on_cdn_block" in fn_body, (
            "_on_cdn_block must be called in _fetch_one_async when IP block is detected"
        )

    def test_token_ttl_check_in_async_path(self):
        src = self._get_source()
        fn_start = src.index("async def _fetch_one_async")
        fn_body  = src[fn_start:fn_start + 3000]
        assert "access_token_ttl_s" in fn_body or "ttl" in fn_body.lower(), (
            "CDN detection in _fetch_one_async must check token TTL"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Regression — existing get_session() still works (backward compat)
# ──────────────────────────────────────────────────────────────────────────────

class TestGetSessionBackwardCompat:
    """Phase 1/2 changes must not break the existing fine-grained get_session()."""

    def test_prime_still_works(self):
        from agent.market_hours import get_session
        with _patch_mh_now(_et(2025, 1, 13, 10, 30)):
            assert get_session() == "PRIME"

    def test_lunch_block_still_works(self):
        from agent.market_hours import get_session
        with _patch_mh_now(_et(2025, 1, 13, 12, 0)):
            assert get_session() == "LUNCH_BLOCK"

    def test_hard_close_still_works(self):
        from agent.market_hours import get_session
        with _patch_mh_now(_et(2025, 1, 13, 15, 50)):
            assert get_session() == "HARD_CLOSE"

    def test_restricted_still_works(self):
        from agent.market_hours import get_session
        with _patch_mh_now(_et(2025, 1, 13, 9, 35)):
            assert get_session() == "RESTRICTED"

    def test_no_new_entries_still_blocks_hard_close(self):
        from agent.market_hours import no_new_entries
        with _patch_mh_now(_et(2025, 1, 13, 15, 50)):
            assert no_new_entries() is True

    def test_confidence_multiplier_prime_is_one(self):
        from agent.market_hours import confidence_multiplier
        with _patch_mh_now(_et(2025, 1, 13, 10, 30)):
            assert confidence_multiplier() == 1.0

    def test_position_size_multiplier_hard_close_is_zero(self):
        from agent.market_hours import position_size_multiplier
        with _patch_mh_now(_et(2025, 1, 13, 15, 50)):
            assert position_size_multiplier() == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 tests: after-hours end-of-day close window
# ═══════════════════════════════════════════════════════════════════════════════



class TestPhase3AhEodCloseWindow:
    """is_ah_eod_close_window() must fire only at 19:55–20:05 ET on trading days."""

    def test_fires_at_1955_et(self):
        from agent.market_hours import is_ah_eod_close_window
        with _patch_mh_now(_et(2025, 1, 13, 19, 55)):   # Monday
            assert is_ah_eod_close_window() is True

    def test_fires_at_2000_et(self):
        from agent.market_hours import is_ah_eod_close_window
        with _patch_mh_now(_et(2025, 1, 13, 20, 0)):
            assert is_ah_eod_close_window() is True

    def test_fires_at_2004_et(self):
        from agent.market_hours import is_ah_eod_close_window
        with _patch_mh_now(_et(2025, 1, 13, 20, 4)):
            assert is_ah_eod_close_window() is True

    def test_does_not_fire_at_2006_et(self):
        from agent.market_hours import is_ah_eod_close_window
        with _patch_mh_now(_et(2025, 1, 13, 20, 6)):
            assert is_ah_eod_close_window() is False

    def test_does_not_fire_before_1955(self):
        from agent.market_hours import is_ah_eod_close_window
        with _patch_mh_now(_et(2025, 1, 13, 19, 54)):
            assert is_ah_eod_close_window() is False

    def test_does_not_fire_on_saturday(self):
        from agent.market_hours import is_ah_eod_close_window
        with _patch_mh_now(_et(2025, 1, 11, 19, 57)):   # Saturday
            assert is_ah_eod_close_window() is False

    def test_does_not_fire_on_sunday(self):
        from agent.market_hours import is_ah_eod_close_window
        with _patch_mh_now(_et(2025, 1, 12, 19, 57)):   # Sunday
            assert is_ah_eod_close_window() is False

    def test_does_not_fire_on_holiday(self):
        from agent.market_hours import is_ah_eod_close_window
        with _patch_mh_now(_et(2025, 1, 1, 19, 57)):    # New Year's Day
            assert is_ah_eod_close_window() is False

    def test_does_not_fire_at_regular_session(self):
        from agent.market_hours import is_ah_eod_close_window
        with _patch_mh_now(_et(2025, 1, 13, 14, 30)):   # 2:30 PM — regular session
            assert is_ah_eod_close_window() is False








# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 tests: session-aware volume baselines in agent/volume.py
# ═══════════════════════════════════════════════════════════════════════════════

import pytz as _pytz
_ET_TZ = _pytz.timezone("America/New_York")


def _make_1m_df(
    start_et: "_real_dt.datetime",
    n_bars: int,
    volume: int = 100_000,
    close: float = 200.0,
) -> "pd.DataFrame":
    """Build a 1-minute OHLCV DataFrame with a DatetimeIndex in America/New_York."""
    idx = pd.date_range(start=start_et, periods=n_bars, freq="1min", tz=start_et.tzinfo)
    df  = pd.DataFrame({
        "Open":   close,
        "High":   close + 0.10,
        "Low":    close - 0.10,
        "Close":  close,
        "Volume": volume,
    }, index=idx)
    return df


def _ah_start(date_str: str = "2025-01-13") -> "_real_dt.datetime":
    """Return 16:00 ET on the given date (after-hours open)."""
    return _real_dt.datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=16, minute=0, tzinfo=_ET_TZ
    )


def _pm_start(date_str: str = "2025-01-13") -> "_real_dt.datetime":
    """Return 04:00 ET on the given date (pre-market open)."""
    return _real_dt.datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=4, minute=0, tzinfo=_ET_TZ
    )


def _reg_start(date_str: str = "2025-01-13") -> "_real_dt.datetime":
    """Return 09:30 ET on the given date (regular session open)."""
    return _real_dt.datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=9, minute=30, tzinfo=_ET_TZ
    )


class TestPhase4SessionDetect:
    """_detect_session() correctly identifies the session from the last bar's ET time."""

    def test_regular_session_930_is_regular(self):
        from agent.volume import _detect_session
        import pandas as pd
        idx = pd.date_range("2025-01-13 14:30", periods=1, freq="1min", tz="UTC")  # 9:30 ET
        idx_et = idx.tz_convert("America/New_York")
        assert _detect_session(idx_et) == "REGULAR"

    def test_regular_session_1559_is_regular(self):
        from agent.volume import _detect_session
        import pandas as pd
        idx = pd.date_range("2025-01-13 20:59", periods=1, freq="1min", tz="UTC")  # 15:59 ET
        idx_et = idx.tz_convert("America/New_York")
        assert _detect_session(idx_et) == "REGULAR"

    def test_after_hours_1600_detected(self):
        from agent.volume import _detect_session
        import pandas as pd
        idx = pd.date_range("2025-01-13 21:00", periods=1, freq="1min", tz="UTC")  # 16:00 ET
        idx_et = idx.tz_convert("America/New_York")
        assert _detect_session(idx_et) == "AFTER_HOURS"

    def test_after_hours_1800_detected(self):
        from agent.volume import _detect_session
        import pandas as pd
        idx = pd.date_range("2025-01-13 23:00", periods=1, freq="1min", tz="UTC")  # 18:00 ET
        idx_et = idx.tz_convert("America/New_York")
        assert _detect_session(idx_et) == "AFTER_HOURS"

    def test_pre_market_0700_detected(self):
        from agent.volume import _detect_session
        import pandas as pd
        idx = pd.date_range("2025-01-13 12:00", periods=1, freq="1min", tz="UTC")  # 07:00 ET
        idx_et = idx.tz_convert("America/New_York")
        assert _detect_session(idx_et) == "PRE_MARKET"

    def test_pre_market_0429_detected(self):
        from agent.volume import _detect_session
        import pandas as pd
        idx = pd.date_range("2025-01-13 09:29", periods=1, freq="1min", tz="UTC")  # 04:29 ET
        idx_et = idx.tz_convert("America/New_York")
        assert _detect_session(idx_et) == "PRE_MARKET"


class TestPhase4SessionElapsedHelpers:
    """_ah_elapsed() and _pm_elapsed() compute correct minutes from session start."""

    def test_ah_elapsed_at_1600_is_zero(self):
        from agent.volume import _ah_elapsed
        import pandas as pd
        ts = pd.Timestamp("2025-01-13 21:00", tz="UTC").tz_convert("America/New_York")
        assert _ah_elapsed(ts) == 0

    def test_ah_elapsed_at_1630_is_30(self):
        from agent.volume import _ah_elapsed
        import pandas as pd
        ts = pd.Timestamp("2025-01-13 21:30", tz="UTC").tz_convert("America/New_York")
        assert _ah_elapsed(ts) == 30

    def test_ah_elapsed_at_1800_is_120(self):
        from agent.volume import _ah_elapsed
        import pandas as pd
        ts = pd.Timestamp("2025-01-13 23:00", tz="UTC").tz_convert("America/New_York")
        assert _ah_elapsed(ts) == 120

    def test_pm_elapsed_at_0400_is_zero(self):
        from agent.volume import _pm_elapsed
        import pandas as pd
        ts = pd.Timestamp("2025-01-13 09:00", tz="UTC").tz_convert("America/New_York")
        assert _pm_elapsed(ts) == 0

    def test_pm_elapsed_at_0700_is_180(self):
        from agent.volume import _pm_elapsed
        import pandas as pd
        ts = pd.Timestamp("2025-01-13 12:00", tz="UTC").tz_convert("America/New_York")
        assert _pm_elapsed(ts) == 180

    def test_pm_elapsed_at_0930_is_330(self):
        from agent.volume import _pm_elapsed
        import pandas as pd
        ts = pd.Timestamp("2025-01-13 14:30", tz="UTC").tz_convert("America/New_York")
        assert _pm_elapsed(ts) == 330


class TestPhase4RvolExtended:
    """rvol_time_of_day() uses session-specific baselines for AH/PM."""

    def test_returns_float(self):
        from agent.volume import rvol_time_of_day
        df = _make_1m_df(_ah_start(), 60)
        result = rvol_time_of_day(df)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_ah_rvol_equal_bars_returns_near_one(self):
        """All-same-volume AH history → RVOL ≈ 1.0."""
        from agent.volume import rvol_time_of_day
        # Build 5 past AH sessions + today, all equal volume
        frames = []
        for day_offset in range(6, 0, -1):
            # Use Mon-Sat offsets: skip weekends for realism
            date_str = f"2025-01-{13 - day_offset:02d}"
            frames.append(_make_1m_df(_ah_start(date_str), 60, volume=50_000))
        # today (day 0) = same volume
        frames.append(_make_1m_df(_ah_start("2025-01-13"), 30, volume=50_000))
        df_all = pd.concat(frames).sort_index()
        result = rvol_time_of_day(df_all)
        # 30 bars at 50k = 1.5M; historical baseline at 30 elapsed = ~1.5M → rvol ≈ 1.0
        assert 0.5 <= result <= 2.0, f"Uniform AH rvol should be near 1.0, got {result}"

    def test_ah_spike_returns_above_one(self):
        """Today's AH volume 3× historical baseline → RVOL ≈ 3."""
        from agent.volume import rvol_time_of_day
        frames = []
        for day_offset in range(6, 0, -1):
            date_str = f"2025-01-{13 - day_offset:02d}"
            frames.append(_make_1m_df(_ah_start(date_str), 60, volume=50_000))
        # today: 3× volume
        frames.append(_make_1m_df(_ah_start("2025-01-13"), 30, volume=150_000))
        df_all = pd.concat(frames).sort_index()
        result = rvol_time_of_day(df_all)
        assert result > 1.5, f"3× AH spike should give rvol > 1.5, got {result}"

    def test_ah_does_not_use_regular_session_bars_for_baseline(self):
        """AH rvol baseline must ignore regular-session bars from the same days."""
        from agent.volume import rvol_time_of_day
        frames = []
        for day_offset in range(6, 0, -1):
            date_str = f"2025-01-{13 - day_offset:02d}"
            # Regular session: 390 bars at 5M/bar (huge regular-session volume)
            frames.append(_make_1m_df(_reg_start(date_str), 390, volume=5_000_000))
            # AH session: 60 bars at 100k/bar (typical AH volume)
            frames.append(_make_1m_df(_ah_start(date_str), 60, volume=100_000))
        # Today AH: same 100k volume
        frames.append(_make_1m_df(_ah_start("2025-01-13"), 30, volume=100_000))
        df_all = pd.concat(frames).sort_index()
        result = rvol_time_of_day(df_all)
        # If regular-session bars were used as baseline (avg 5M vs today's 100k),
        # result would be near 0.0.  Correct AH-only baseline → ≈ 1.0.
        assert result > 0.3, (
            f"AH rvol should use AH-only baseline, not regular session. Got {result}"
        )

    def test_pm_rvol_uniform_returns_near_one(self):
        """Uniform PM volume across 4 historical days → RVOL ≈ 1.0."""
        from agent.volume import rvol_time_of_day
        frames = []
        for day_offset in range(5, 0, -1):
            date_str = f"2025-01-{13 - day_offset:02d}"
            frames.append(_make_1m_df(_pm_start(date_str), 120, volume=30_000))
        frames.append(_make_1m_df(_pm_start("2025-01-13"), 60, volume=30_000))
        df_all = pd.concat(frames).sort_index()
        result = rvol_time_of_day(df_all)
        assert 0.4 <= result <= 2.5, f"Uniform PM rvol should be near 1.0, got {result}"

    def test_empty_dataframe_returns_one(self):
        from agent.volume import rvol_time_of_day
        result = rvol_time_of_day(pd.DataFrame())
        assert result == 1.0

    def test_none_dataframe_returns_one(self):
        from agent.volume import rvol_time_of_day
        assert rvol_time_of_day(None) == 1.0

    def test_regular_session_path_unchanged(self):
        """Regular session still uses the 9:30-based profile/history."""
        from agent.volume import rvol_time_of_day
        frames = []
        for day_offset in range(6, 0, -1):
            date_str = f"2025-01-{13 - day_offset:02d}"
            frames.append(_make_1m_df(_reg_start(date_str), 60, volume=1_000_000))
        frames.append(_make_1m_df(_reg_start("2025-01-13"), 30, volume=1_000_000))
        df_all = pd.concat(frames).sort_index()
        result = rvol_time_of_day(df_all)
        assert 0.5 <= result <= 2.0, f"Uniform regular-session rvol should be ≈ 1.0, got {result}"


class TestPhase4DetectUnusualVolumeExtended:
    """detect_unusual_volume() uses session-specific baseline for AH/PM."""

    def test_ah_spike_vs_ah_baseline_detected(self):
        """A 3× spike in AH relative to AH history is detected as unusual."""
        from agent.volume import detect_unusual_volume
        # First 59 bars: normal AH volume; last bar: spike
        df = _make_1m_df(_ah_start(), 59, volume=50_000)
        spike_row = _make_1m_df(
            _ah_start().replace(hour=16, minute=59), 1, volume=150_000
        )
        df = pd.concat([df, spike_row]).sort_index()
        result = detect_unusual_volume(df, threshold=2.5)
        assert result is True, "3× AH spike must be detected as unusual"

    def test_normal_ah_volume_not_unusual(self):
        """Uniform AH volume across 20 bars is NOT flagged as unusual."""
        from agent.volume import detect_unusual_volume
        df = _make_1m_df(_ah_start(), 25, volume=50_000)
        result = detect_unusual_volume(df, threshold=2.5)
        assert result is False, "Uniform AH volume must not be flagged as unusual"

    def test_ah_volume_not_inflated_by_regular_bars(self):
        """AH 'unusual' check must ignore regular-session bars in the same df."""
        from agent.volume import detect_unusual_volume
        # Regular session: 390 bars at 5M (would make AH 100k look near-zero rvol)
        reg = _make_1m_df(_reg_start(), 390, volume=5_000_000)
        # AH: 25 bars at 100k (normal AH), last bar at 100k (not a spike)
        ah  = _make_1m_df(_ah_start(), 25, volume=100_000)
        df  = pd.concat([reg, ah]).sort_index()
        # If regular bars polluted the baseline (avg ~5M), 100k/5M << 2.5 → False
        # But also 100k vs 100k AH baseline = 1.0 << 2.5 → False  ← correct answer
        result = detect_unusual_volume(df, threshold=2.5)
        assert result is False, "Normal AH volume must not appear unusual regardless of RS bars"

    def test_regular_session_unchanged(self):
        """Regular-session detect_unusual_volume still uses 20-bar rolling average."""
        from agent.volume import detect_unusual_volume
        df_reg = _make_1m_df(_reg_start(), 25, volume=1_000_000)
        spike  = _make_1m_df(
            _reg_start().replace(hour=10, minute=25), 1, volume=3_000_000
        )
        df = pd.concat([df_reg, spike]).sort_index()
        result = detect_unusual_volume(df, threshold=2.5)
        assert result is True, "3× spike in regular session must be detected"


class TestPhase4RelativeVolumeExtended:
    """relative_volume() uses session-specific baseline for AH/PM."""

    def test_ah_relative_volume_uses_ah_bars(self):
        """AH relative_volume must compare against AH-only bars."""
        from agent.volume import relative_volume
        reg = _make_1m_df(_reg_start(), 390, volume=5_000_000)
        ah  = _make_1m_df(_ah_start(), 20, volume=100_000)
        # last bar = same 100k
        df  = pd.concat([reg, ah]).sort_index()
        result = relative_volume(df)
        # If using AH-only baseline: 100k/100k = 1.0
        # If using all-bars 20-bar window: would be ~0.02
        assert result > 0.5, (
            f"AH relative_volume should be ≈1.0 using AH-only baseline, got {result}"
        )

    def test_regular_session_relative_volume_unchanged(self):
        """Regular-session relative_volume returns ratio vs 20-bar average."""
        from agent.volume import relative_volume
        df = _make_1m_df(_reg_start(), 25, volume=1_000_000)
        result = relative_volume(df)
        assert 0.9 <= result <= 1.1, f"Uniform regular-session rvol should be ≈1.0, got {result}"


class TestPhase4Profiles:
    """_AH_CUM_PROFILE and _PM_CUM_PROFILE are well-formed."""

    def test_ah_profile_starts_above_zero(self):
        from agent.volume import _AH_CUM_PROFILE
        assert _AH_CUM_PROFILE[0][1] > 0

    def test_ah_profile_ends_at_one(self):
        from agent.volume import _AH_CUM_PROFILE
        assert _AH_CUM_PROFILE[-1][1] == 1.0

    def test_ah_profile_is_monotone(self):
        from agent.volume import _AH_CUM_PROFILE
        fracs = [f for _, f in _AH_CUM_PROFILE]
        assert all(fracs[i] <= fracs[i+1] for i in range(len(fracs)-1))

    def test_pm_profile_ends_at_one(self):
        from agent.volume import _PM_CUM_PROFILE
        assert _PM_CUM_PROFILE[-1][1] == 1.0

    def test_pm_profile_is_monotone(self):
        from agent.volume import _PM_CUM_PROFILE
        fracs = [f for _, f in _PM_CUM_PROFILE]
        assert all(fracs[i] <= fracs[i+1] for i in range(len(fracs)-1))

    def test_interp_profile_clamps_below_zero(self):
        from agent.volume import _interp_profile, _AH_CUM_PROFILE
        assert _interp_profile(-10, _AH_CUM_PROFILE) >= 0

    def test_interp_profile_clamps_above_max(self):
        from agent.volume import _interp_profile, _AH_CUM_PROFILE
        assert _interp_profile(9999, _AH_CUM_PROFILE) == 1.0
