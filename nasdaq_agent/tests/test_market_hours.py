"""
Tests for agent/market_hours.py

Strategy:
  - Type/range checks against live current time (always valid regardless of session).
  - Patched-time tests to exercise specific session branches deterministically.

The module calls `datetime.now(ET)` directly (not via a re-bound name), so we
patch `agent.market_hours.datetime` with a MagicMock whose `.now()` returns a
known aware datetime.
"""
from __future__ import annotations

import sys
import os
import datetime as _real_datetime
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import agent.market_hours as mh
from agent.market_hours import (
    get_session,
    get_session_info,
    is_tradeable,
    is_trading_day,
    is_lunch_block,
    is_hard_close_window,
    is_closing_caution,
    is_pre_market,
    is_after_hours,
    confidence_multiplier,
    position_size_multiplier,
    minutes_until_open,
    no_new_entries,
    get_block_reason,
    _SESSIONS,
    ET,
)

# ── helpers ───────────────────────────────────────────────────────────────────

VALID_SESSIONS = set(_SESSIONS.keys())


def _make_et(year: int, month: int, day: int, hour: int, minute: int = 0) -> _real_datetime.datetime:
    """Return an ET-aware datetime for the given local ET components."""
    naive = _real_datetime.datetime(year, month, day, hour, minute, 0)
    return ET.localize(naive)


def _patch_now(fake_et_dt):
    """Context manager: patch datetime.now inside market_hours to return fake_et_dt."""
    mock_dt = MagicMock(wraps=_real_datetime.datetime)
    mock_dt.now.return_value = fake_et_dt
    # Keep real date/time/timedelta accessible since module uses them
    mock_dt.side_effect = _real_datetime.datetime
    return patch("agent.market_hours.datetime", mock_dt)


# ── Live (real-time) type/range tests — always pass regardless of session ─────

def test_get_session_returns_valid_string():
    session = get_session()
    assert isinstance(session, str)
    assert session in VALID_SESSIONS


def test_get_session_info_returns_dict():
    info = get_session_info()
    assert isinstance(info, dict)


def test_get_session_info_has_required_keys():
    info = get_session_info()
    required = {"session", "label", "color", "tradeable", "mult", "size_mult",
                "advice", "time_et", "date_et", "day_name", "is_weekend",
                "is_holiday", "is_half_day"}
    for key in required:
        assert key in info, f"Missing key: {key}"


def test_get_session_info_session_is_valid():
    info = get_session_info()
    assert info["session"] in VALID_SESSIONS


def test_is_tradeable_returns_bool():
    result = is_tradeable()
    assert isinstance(result, bool)


def test_is_trading_day_returns_bool():
    result = is_trading_day()
    assert isinstance(result, bool)


def test_confidence_multiplier_positive():
    mult = confidence_multiplier()
    assert isinstance(mult, float)
    assert mult >= 0.0


def test_confidence_multiplier_bounded():
    mult = confidence_multiplier()
    assert 0.0 <= mult <= 2.0


def test_position_size_multiplier_positive():
    mult = position_size_multiplier()
    assert isinstance(mult, float)
    assert mult >= 0.0


def test_position_size_multiplier_bounded():
    mult = position_size_multiplier()
    assert 0.0 <= mult <= 2.0


def test_minutes_until_open_nonnegative():
    result = minutes_until_open()
    assert isinstance(result, int)
    assert result >= 0


def test_is_lunch_block_returns_bool():
    assert isinstance(is_lunch_block(), bool)


def test_is_hard_close_window_returns_bool():
    assert isinstance(is_hard_close_window(), bool)


def test_is_pre_market_returns_bool():
    assert isinstance(is_pre_market(), bool)


def test_is_after_hours_returns_bool():
    assert isinstance(is_after_hours(), bool)


def test_no_new_entries_returns_bool():
    assert isinstance(no_new_entries(), bool)


def test_get_block_reason_returns_string():
    reason = get_block_reason()
    assert isinstance(reason, str)


# ── Consistency checks (invariants that must hold at any time) ─────────────

def test_tradeable_and_session_consistent():
    """is_tradeable() must agree with _SESSIONS[get_session()]['tradeable']."""
    session = get_session()
    expected = _SESSIONS[session]["tradeable"]
    assert is_tradeable() == expected


def test_confidence_multiplier_matches_session():
    session = get_session()
    expected = _SESSIONS[session]["mult"]
    assert confidence_multiplier() == expected


def test_position_size_multiplier_matches_session():
    session = get_session()
    expected = _SESSIONS[session]["size_mult"]
    assert position_size_multiplier() == expected


def test_no_new_entries_only_when_hard_close_or_closed():
    session = get_session()
    if no_new_entries():
        assert session in ("HARD_CLOSE", "CLOSED")
    else:
        assert session not in ("HARD_CLOSE", "CLOSED")


# ── Patched-time tests for PRIME session (Monday 10:30 AM ET, not a holiday) ──

# 2025-01-13 is a Monday; no holiday; 10:30 AM ET = PRIME session.
_PRIME_DT = _make_et(2025, 1, 13, 10, 30)


def test_prime_session_identified():
    with _patch_now(_PRIME_DT):
        assert get_session() == "PRIME"


def test_prime_is_tradeable():
    with _patch_now(_PRIME_DT):
        assert is_tradeable() is True


def test_prime_confidence_multiplier():
    with _patch_now(_PRIME_DT):
        assert confidence_multiplier() == 1.0


def test_prime_no_new_entries_false():
    with _patch_now(_PRIME_DT):
        assert no_new_entries() is False


def test_prime_block_reason_empty():
    with _patch_now(_PRIME_DT):
        assert get_block_reason() == ""


# ── Patched-time tests for PRE_MARKET (Monday 07:00 AM ET) ────────────────────

_PRE_MARKET_DT = _make_et(2025, 1, 13, 7, 0)


def test_pre_market_session_identified():
    with _patch_now(_PRE_MARKET_DT):
        assert get_session() == "PRE_MARKET"


def test_pre_market_is_pre_market():
    with _patch_now(_PRE_MARKET_DT):
        assert is_pre_market() is True


def test_pre_market_minutes_until_open_is_zero():
    """PRE_MARKET is tradeable so minutes_until_open returns 0."""
    with _patch_now(_PRE_MARKET_DT):
        assert minutes_until_open() == 0


# ── Patched-time tests for LUNCH_BLOCK (Monday 12:00 PM ET) ──────────────────

_LUNCH_DT = _make_et(2025, 1, 13, 12, 0)


def test_lunch_block_session_identified():
    with _patch_now(_LUNCH_DT):
        assert get_session() == "LUNCH_BLOCK"


def test_lunch_block_is_lunch_block():
    with _patch_now(_LUNCH_DT):
        assert is_lunch_block() is True


def test_lunch_block_is_tradeable():
    with _patch_now(_LUNCH_DT):
        assert is_tradeable() is True


# ── Patched-time tests for HARD_CLOSE (Monday 15:50 ET) ──────────────────────

_HARD_CLOSE_DT = _make_et(2025, 1, 13, 15, 50)


def test_hard_close_session_identified():
    with _patch_now(_HARD_CLOSE_DT):
        assert get_session() == "HARD_CLOSE"


def test_hard_close_is_hard_close_window():
    with _patch_now(_HARD_CLOSE_DT):
        assert is_hard_close_window() is True


def test_hard_close_no_new_entries():
    with _patch_now(_HARD_CLOSE_DT):
        assert no_new_entries() is True


def test_hard_close_not_tradeable():
    with _patch_now(_HARD_CLOSE_DT):
        assert is_tradeable() is False


def test_hard_close_block_reason_non_empty():
    with _patch_now(_HARD_CLOSE_DT):
        reason = get_block_reason()
        assert isinstance(reason, str)
        assert len(reason) > 0


# ── Patched-time tests for AFTER_HOURS (Monday 17:00 ET) ─────────────────────

_AFTER_HOURS_DT = _make_et(2025, 1, 13, 17, 0)


def test_after_hours_session_identified():
    with _patch_now(_AFTER_HOURS_DT):
        assert get_session() == "AFTER_HOURS"


def test_after_hours_is_after_hours():
    with _patch_now(_AFTER_HOURS_DT):
        assert is_after_hours() is True


# ── Patched-time tests for CLOSED on a weekend (Saturday) ────────────────────

# 2025-01-11 is a Saturday
_WEEKEND_DT = _make_et(2025, 1, 11, 14, 0)


def test_weekend_session_is_closed():
    with _patch_now(_WEEKEND_DT):
        assert get_session() == "CLOSED"


def test_weekend_is_not_trading_day():
    with _patch_now(_WEEKEND_DT):
        assert is_trading_day() is False


def test_weekend_no_new_entries():
    with _patch_now(_WEEKEND_DT):
        assert no_new_entries() is True


def test_weekend_block_reason_mentions_day():
    with _patch_now(_WEEKEND_DT):
        reason = get_block_reason()
        assert "Saturday" in reason or "Sunday" in reason or "Market closed" in reason


def test_weekend_minutes_until_open_positive():
    with _patch_now(_WEEKEND_DT):
        mins = minutes_until_open()
        assert mins > 0


# ── Patched-time tests for CLOSED on a holiday ───────────────────────────────

# 2025-01-01 is New Year's Day (holiday)
_HOLIDAY_DT = _make_et(2025, 1, 1, 10, 30)


def test_holiday_session_is_closed():
    with _patch_now(_HOLIDAY_DT):
        assert get_session() == "CLOSED"


def test_holiday_is_not_trading_day():
    with _patch_now(_HOLIDAY_DT):
        assert is_trading_day() is False


def test_holiday_block_reason_mentions_holiday():
    with _patch_now(_HOLIDAY_DT):
        reason = get_block_reason()
        assert "holiday" in reason.lower() or "Holiday" in reason or "closed" in reason.lower()


# ── Patched-time tests for CLOSING_CAUTION (Monday 15:32 ET) ─────────────────

_CLOSING_CAUTION_DT = _make_et(2025, 1, 13, 15, 32)


def test_closing_caution_session_identified():
    with _patch_now(_CLOSING_CAUTION_DT):
        assert get_session() == "CLOSING_CAUTION"


def test_closing_caution_is_tradeable():
    with _patch_now(_CLOSING_CAUTION_DT):
        assert is_tradeable() is True


def test_closing_caution_is_closing_caution():
    with _patch_now(_CLOSING_CAUTION_DT):
        assert is_closing_caution() is True


# ── Patched-time tests for RESTRICTED (Monday 09:35 ET) ──────────────────────

_RESTRICTED_DT = _make_et(2025, 1, 13, 9, 35)


def test_restricted_session_identified():
    with _patch_now(_RESTRICTED_DT):
        assert get_session() == "RESTRICTED"


def test_restricted_is_tradeable():
    with _patch_now(_RESTRICTED_DT):
        assert is_tradeable() is True


# ── get_session_info additional checks ────────────────────────────────────────

def test_session_info_is_weekend_true_on_saturday():
    with _patch_now(_WEEKEND_DT):
        info = get_session_info()
        assert info["is_weekend"] is True


def test_session_info_is_holiday_true_on_holiday():
    with _patch_now(_HOLIDAY_DT):
        info = get_session_info()
        assert info["is_holiday"] is True


def test_session_info_normal_weekday_not_weekend():
    with _patch_now(_PRIME_DT):
        info = get_session_info()
        assert info["is_weekend"] is False
