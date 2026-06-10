"""
Tests for agent/macro_calendar.py — FOMC/CPI/NFP blackout logic.
"""
from datetime import datetime, timezone

import pytest
from agent.macro_calendar import check_macro_event, get_upcoming_events


def test_check_macro_event_structure():
    result = check_macro_event()
    for key in ("blocked", "impact", "event_name", "hours_away", "description"):
        assert key in result, f"missing key: {key}"

def test_blocked_is_bool():
    result = check_macro_event()
    assert isinstance(result["blocked"], bool)

def test_impact_valid_value():
    result = check_macro_event()
    assert result["impact"] in ("HIGH", "MEDIUM", "LOW", "NONE")

def test_hours_away_non_negative():
    result = check_macro_event()
    # Active macro windows can be post-event and therefore negative; inactive
    # upcoming events should remain non-negative.
    if result["hours_away"] is not None and not (result["blocked"] or result.get("throttled")):
        assert result["hours_away"] >= 0


def test_high_impact_macro_hard_blocks_only_tight_window():
    result = check_macro_event(
        check_dt=datetime(2026, 6, 10, 13, 45, tzinfo=timezone.utc)
    )

    assert result["event_name"] == "CPI Release"
    assert result["blocked"] is True
    assert result["throttled"] is False
    assert result["size_mult"] == 0.0


def test_high_impact_macro_throttles_outside_hard_window():
    result = check_macro_event(
        check_dt=datetime(2026, 6, 10, 16, 30, tzinfo=timezone.utc)
    )

    assert result["event_name"] == "CPI Release"
    assert result["blocked"] is False
    assert result["throttled"] is True
    assert 0.0 < result["size_mult"] < 1.0
    assert result["min_confidence"] >= 70.0

def test_upcoming_events_is_list():
    events = get_upcoming_events(days=14)
    assert isinstance(events, list)

def test_upcoming_events_structure():
    events = get_upcoming_events(days=60)  # wider window to catch something
    for ev in events:
        assert "name" in ev
        assert "date" in ev
        assert "impact" in ev

def test_upcoming_events_sorted_by_date():
    events = get_upcoming_events(days=90)
    if len(events) >= 2:
        dates = [e["date"] for e in events]
        assert dates == sorted(dates), "events should be sorted chronologically"

def test_check_macro_event_returns_consistently():
    """Calling check_macro_event() twice should return the same blocked state."""
    r1 = check_macro_event()
    r2 = check_macro_event()
    assert r1["blocked"] == r2["blocked"]
    assert r1["impact"] == r2["impact"]
