"""
Macro event calendar — FOMC, CPI, NFP, OpEx dates.

Signals are suppressed (or confidence reduced) in the 24h window around
major scheduled macro events due to extreme unpredictability.

Dates are hard-coded for 2025-2026 and refreshed here annually.
"""
from __future__ import annotations
from datetime import date, timedelta, timezone, datetime

# ── Event calendar 2025-2026 ─────────────────────────────────────────────────
# Format: (date_str, event_name, impact)
# impact: HIGH (suppress signals) | MEDIUM (warn) | LOW (note only)
_EVENTS: list[tuple[str, str, str]] = [
    # FOMC Meetings 2025
    ("2025-01-29", "FOMC Rate Decision", "HIGH"),
    ("2025-03-19", "FOMC Rate Decision", "HIGH"),
    ("2025-05-07", "FOMC Rate Decision", "HIGH"),
    ("2025-06-18", "FOMC Rate Decision", "HIGH"),
    ("2025-07-30", "FOMC Rate Decision", "HIGH"),
    ("2025-09-17", "FOMC Rate Decision", "HIGH"),
    ("2025-10-29", "FOMC Rate Decision", "HIGH"),
    ("2025-12-10", "FOMC Rate Decision", "HIGH"),
    # FOMC Meetings 2026
    ("2026-01-28", "FOMC Rate Decision", "HIGH"),
    ("2026-03-18", "FOMC Rate Decision", "HIGH"),
    ("2026-04-29", "FOMC Rate Decision", "HIGH"),
    ("2026-06-17", "FOMC Rate Decision", "HIGH"),
    ("2026-07-29", "FOMC Rate Decision", "HIGH"),
    ("2026-09-16", "FOMC Rate Decision", "HIGH"),
    ("2026-11-04", "FOMC Rate Decision", "HIGH"),
    ("2026-12-16", "FOMC Rate Decision", "HIGH"),
    # CPI Releases 2025 (approximate — 2nd or 3rd Wed of month)
    ("2025-01-15", "CPI Release", "HIGH"),
    ("2025-02-12", "CPI Release", "HIGH"),
    ("2025-03-12", "CPI Release", "HIGH"),
    ("2025-04-10", "CPI Release", "HIGH"),
    ("2025-05-13", "CPI Release", "HIGH"),
    ("2025-06-11", "CPI Release", "HIGH"),
    ("2025-07-15", "CPI Release", "HIGH"),
    ("2025-08-12", "CPI Release", "HIGH"),
    ("2025-09-10", "CPI Release", "HIGH"),
    ("2025-10-15", "CPI Release", "HIGH"),
    ("2025-11-13", "CPI Release", "HIGH"),
    ("2025-12-10", "CPI Release", "HIGH"),
    # NFP (Non-Farm Payrolls) — first Friday of month
    ("2025-01-10", "NFP Report", "HIGH"),
    ("2025-02-07", "NFP Report", "HIGH"),
    ("2025-03-07", "NFP Report", "HIGH"),
    ("2025-04-04", "NFP Report", "HIGH"),
    ("2025-05-02", "NFP Report", "HIGH"),
    ("2025-06-06", "NFP Report", "HIGH"),
    ("2025-07-03", "NFP Report", "HIGH"),
    ("2025-08-01", "NFP Report", "HIGH"),
    ("2025-09-05", "NFP Report", "HIGH"),
    ("2025-10-03", "NFP Report", "HIGH"),
    ("2025-11-07", "NFP Report", "HIGH"),
    ("2025-12-05", "NFP Report", "HIGH"),
    # Options Expiration (monthly OpEx — 3rd Friday)
    ("2025-01-17", "Monthly OpEx", "MEDIUM"),
    ("2025-02-21", "Monthly OpEx", "MEDIUM"),
    ("2025-03-21", "Monthly OpEx", "MEDIUM"),
    ("2025-04-17", "Monthly OpEx", "MEDIUM"),
    ("2025-05-16", "Monthly OpEx", "MEDIUM"),
    ("2025-06-20", "Monthly OpEx", "MEDIUM"),
    ("2025-07-18", "Monthly OpEx", "MEDIUM"),
    ("2025-08-15", "Monthly OpEx", "MEDIUM"),
    ("2025-09-19", "Monthly OpEx", "MEDIUM"),
    ("2025-10-17", "Monthly OpEx", "MEDIUM"),
    ("2025-11-21", "Monthly OpEx", "MEDIUM"),
    ("2025-12-19", "Monthly OpEx", "MEDIUM"),
    # Quarterly OpEx (quadruple witching — March, June, Sep, Dec 3rd Fri)
    ("2025-03-21", "Quad Witching OpEx", "HIGH"),
    ("2025-06-20", "Quad Witching OpEx", "HIGH"),
    ("2025-09-19", "Quad Witching OpEx", "HIGH"),
    ("2025-12-19", "Quad Witching OpEx", "HIGH"),
    ("2026-03-20", "Quad Witching OpEx", "HIGH"),
    ("2026-06-19", "Quad Witching OpEx", "HIGH"),
]

# Blackout window in hours before/after event
_HIGH_BLACKOUT_HOURS   = 4
_MEDIUM_BLACKOUT_HOURS = 2


def _parse(d: str) -> date:
    return date.fromisoformat(d)


def check_macro_event(check_date: date | None = None) -> dict:
    """
    Check if today (or a given date) is within the blackout window of a macro event.

    Returns dict:
      blocked     : bool
      impact      : str  — HIGH | MEDIUM | LOW | NONE
      event_name  : str
      event_date  : str
      hours_away  : float  — negative = past
      description : str
    """
    result = {
        "blocked":    False,
        "impact":     "NONE",
        "event_name": "",
        "event_date": "",
        "hours_away": 0.0,
        "description": "",
    }

    now = datetime.now(timezone.utc)
    today = check_date or now.date()

    nearest = None
    nearest_hours = float("inf")

    for date_str, name, impact in _EVENTS:
        ev_date = _parse(date_str)
        ev_dt   = datetime(ev_date.year, ev_date.month, ev_date.day,
                           14, 0, tzinfo=timezone.utc)   # assume 2pm UTC (9am ET)
        hours_away = (ev_dt - now).total_seconds() / 3600

        blackout = _HIGH_BLACKOUT_HOURS if impact == "HIGH" else _MEDIUM_BLACKOUT_HOURS

        if abs(hours_away) < abs(nearest_hours):
            nearest_hours = hours_away
            nearest = (date_str, name, impact, hours_away)

        if abs(hours_away) <= blackout:
            result.update({
                "blocked":     impact == "HIGH",
                "impact":      impact,
                "event_name":  name,
                "event_date":  date_str,
                "hours_away":  round(hours_away, 1),
                "description": (
                    f"{'🚫' if impact=='HIGH' else '⚠'} {name} on {date_str} "
                    f"({'+' if hours_away>=0 else ''}{hours_away:.1f}h) — "
                    f"{'Signals suppressed' if impact=='HIGH' else 'Reduce size'}"
                ),
            })
            return result   # return closest / most impactful match

    # No blackout, but show next upcoming event as info
    if nearest:
        date_str, name, impact, hours_away = nearest
        if 0 < hours_away < 48:
            result["description"] = (
                f"Next: {name} in {hours_away:.0f}h ({date_str})"
            )

    return result


def get_upcoming_events(days: int = 7) -> list[dict]:
    """Return macro events in the next N days."""
    now   = datetime.now(timezone.utc)
    out   = []
    for date_str, name, impact in _EVENTS:
        ev_date = _parse(date_str)
        ev_dt   = datetime(ev_date.year, ev_date.month, ev_date.day,
                           14, 0, tzinfo=timezone.utc)
        hours_away = (ev_dt - now).total_seconds() / 3600
        if 0 <= hours_away <= days * 24:
            out.append({
                "date":       date_str,
                "name":       name,
                "impact":     impact,
                "hours_away": round(hours_away, 1),
            })
    return sorted(out, key=lambda x: x["hours_away"])
