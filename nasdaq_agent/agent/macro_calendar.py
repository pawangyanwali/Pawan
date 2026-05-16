"""
Macro event calendar — FOMC, CPI, NFP, OpEx dates.

Signals are suppressed (or confidence reduced) in the blackout window around
major scheduled macro events due to extreme unpredictability.

Dates are hard-coded for 2025-2026 and refreshed here annually.
"""
from __future__ import annotations
from datetime import date, timezone, datetime

# ── Event calendar 2025-2026 ─────────────────────────────────────────────────
# Format: (date_str, event_name, impact, hour_utc, minute_utc)
# Correct release times (ET → UTC, EST = UTC-5, EDT = UTC-4):
#   NFP / CPI           8:30am ET  → 13:30 UTC
#   FOMC Rate Decision  2:00pm ET  → 18:00 UTC (EDT) / 19:00 UTC (EST)
#   Monthly OpEx        Market open → 14:30 UTC (approx)
_EVENTS: list[tuple[str, str, str, int, int]] = [
    # ── FOMC Rate Decisions (2:00pm ET) ──────────────────────────────────────
    ("2025-01-29", "FOMC Rate Decision", "HIGH", 19, 0),   # EST
    ("2025-03-19", "FOMC Rate Decision", "HIGH", 18, 0),   # EDT
    ("2025-05-07", "FOMC Rate Decision", "HIGH", 18, 0),
    ("2025-06-18", "FOMC Rate Decision", "HIGH", 18, 0),
    ("2025-07-30", "FOMC Rate Decision", "HIGH", 18, 0),
    ("2025-09-17", "FOMC Rate Decision", "HIGH", 18, 0),
    ("2025-10-29", "FOMC Rate Decision", "HIGH", 18, 0),
    ("2025-12-10", "FOMC Rate Decision", "HIGH", 19, 0),   # EST
    ("2026-01-28", "FOMC Rate Decision", "HIGH", 19, 0),
    ("2026-03-18", "FOMC Rate Decision", "HIGH", 18, 0),
    ("2026-04-29", "FOMC Rate Decision", "HIGH", 18, 0),
    ("2026-06-17", "FOMC Rate Decision", "HIGH", 18, 0),
    ("2026-07-29", "FOMC Rate Decision", "HIGH", 18, 0),
    ("2026-09-16", "FOMC Rate Decision", "HIGH", 18, 0),
    ("2026-11-04", "FOMC Rate Decision", "HIGH", 19, 0),
    ("2026-12-16", "FOMC Rate Decision", "HIGH", 19, 0),
    # ── CPI Releases (8:30am ET = 13:30 UTC) ─────────────────────────────────
    ("2025-01-15", "CPI Release", "HIGH", 13, 30),
    ("2025-02-12", "CPI Release", "HIGH", 13, 30),
    ("2025-03-12", "CPI Release", "HIGH", 13, 30),
    ("2025-04-10", "CPI Release", "HIGH", 13, 30),
    ("2025-05-13", "CPI Release", "HIGH", 13, 30),
    ("2025-06-11", "CPI Release", "HIGH", 13, 30),
    ("2025-07-15", "CPI Release", "HIGH", 13, 30),
    ("2025-08-12", "CPI Release", "HIGH", 13, 30),
    ("2025-09-10", "CPI Release", "HIGH", 13, 30),
    ("2025-10-15", "CPI Release", "HIGH", 13, 30),
    ("2025-11-13", "CPI Release", "HIGH", 13, 30),
    ("2025-12-10", "CPI Release", "HIGH", 13, 30),
    ("2026-01-14", "CPI Release", "HIGH", 13, 30),
    ("2026-02-11", "CPI Release", "HIGH", 13, 30),
    ("2026-03-11", "CPI Release", "HIGH", 13, 30),
    ("2026-04-09", "CPI Release", "HIGH", 13, 30),
    ("2026-05-13", "CPI Release", "HIGH", 13, 30),
    ("2026-06-10", "CPI Release", "HIGH", 13, 30),
    ("2026-07-15", "CPI Release", "HIGH", 13, 30),
    ("2026-08-12", "CPI Release", "HIGH", 13, 30),
    ("2026-09-09", "CPI Release", "HIGH", 13, 30),
    ("2026-10-14", "CPI Release", "HIGH", 13, 30),
    ("2026-11-12", "CPI Release", "HIGH", 13, 30),
    ("2026-12-09", "CPI Release", "HIGH", 13, 30),
    # ── NFP (8:30am ET = 13:30 UTC, first Friday of month) ───────────────────
    ("2025-01-10", "NFP Report", "HIGH", 13, 30),
    ("2025-02-07", "NFP Report", "HIGH", 13, 30),
    ("2025-03-07", "NFP Report", "HIGH", 13, 30),
    ("2025-04-04", "NFP Report", "HIGH", 13, 30),
    ("2025-05-02", "NFP Report", "HIGH", 13, 30),
    ("2025-06-06", "NFP Report", "HIGH", 13, 30),
    ("2025-07-03", "NFP Report", "HIGH", 13, 30),
    ("2025-08-01", "NFP Report", "HIGH", 13, 30),
    ("2025-09-05", "NFP Report", "HIGH", 13, 30),
    ("2025-10-03", "NFP Report", "HIGH", 13, 30),
    ("2025-11-07", "NFP Report", "HIGH", 13, 30),
    ("2025-12-05", "NFP Report", "HIGH", 13, 30),
    ("2026-01-09", "NFP Report", "HIGH", 13, 30),
    ("2026-02-06", "NFP Report", "HIGH", 13, 30),
    ("2026-03-06", "NFP Report", "HIGH", 13, 30),
    ("2026-04-03", "NFP Report", "HIGH", 13, 30),
    ("2026-05-01", "NFP Report", "HIGH", 13, 30),
    ("2026-06-05", "NFP Report", "HIGH", 13, 30),
    ("2026-07-02", "NFP Report", "HIGH", 13, 30),
    ("2026-08-07", "NFP Report", "HIGH", 13, 30),
    ("2026-09-04", "NFP Report", "HIGH", 13, 30),
    ("2026-10-02", "NFP Report", "HIGH", 13, 30),
    ("2026-11-06", "NFP Report", "HIGH", 13, 30),
    ("2026-12-04", "NFP Report", "HIGH", 13, 30),
    # ── Monthly OpEx (3rd Friday) ─────────────────────────────────────────────
    ("2025-01-17", "Monthly OpEx", "MEDIUM", 14, 30),
    ("2025-02-21", "Monthly OpEx", "MEDIUM", 14, 30),
    ("2025-04-17", "Monthly OpEx", "MEDIUM", 14, 30),
    ("2025-05-16", "Monthly OpEx", "MEDIUM", 14, 30),
    ("2025-07-18", "Monthly OpEx", "MEDIUM", 14, 30),
    ("2025-08-15", "Monthly OpEx", "MEDIUM", 14, 30),
    ("2025-10-17", "Monthly OpEx", "MEDIUM", 14, 30),
    ("2025-11-21", "Monthly OpEx", "MEDIUM", 14, 30),
    ("2026-01-16", "Monthly OpEx", "MEDIUM", 14, 30),
    ("2026-02-20", "Monthly OpEx", "MEDIUM", 14, 30),
    ("2026-04-17", "Monthly OpEx", "MEDIUM", 14, 30),
    ("2026-05-15", "Monthly OpEx", "MEDIUM", 14, 30),
    ("2026-07-17", "Monthly OpEx", "MEDIUM", 14, 30),
    ("2026-08-21", "Monthly OpEx", "MEDIUM", 14, 30),
    ("2026-10-16", "Monthly OpEx", "MEDIUM", 14, 30),
    ("2026-11-20", "Monthly OpEx", "MEDIUM", 14, 30),
    # ── Quad Witching OpEx (Mar/Jun/Sep/Dec 3rd Fri) ─────────────────────────
    ("2025-03-21", "Quad Witching OpEx", "HIGH", 14, 30),
    ("2025-06-20", "Quad Witching OpEx", "HIGH", 14, 30),
    ("2025-09-19", "Quad Witching OpEx", "HIGH", 14, 30),
    ("2025-12-19", "Quad Witching OpEx", "HIGH", 14, 30),
    ("2026-03-20", "Quad Witching OpEx", "HIGH", 14, 30),
    ("2026-06-19", "Quad Witching OpEx", "HIGH", 14, 30),
    ("2026-09-18", "Quad Witching OpEx", "HIGH", 14, 30),
    ("2026-12-18", "Quad Witching OpEx", "HIGH", 14, 30),
]

_IMPACT_RANK = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
_HIGH_BLACKOUT_HOURS   = 4
_MEDIUM_BLACKOUT_HOURS = 2


def _parse(d: str) -> date:
    return date.fromisoformat(d)


def _ev_dt(date_str: str, hour_utc: int, minute_utc: int) -> datetime:
    d = _parse(date_str)
    return datetime(d.year, d.month, d.day, hour_utc, minute_utc, tzinfo=timezone.utc)


def check_macro_event(check_date: date | None = None) -> dict:
    """
    Check if now is within the blackout window of any macro event.

    Returns the HIGHEST-IMPACT matching event (not first-listed).
    'Nearest upcoming' info shows only future events.
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

    # Collect all events currently inside their blackout window
    active_matches: list[tuple[int, dict]] = []  # (impact_rank, result_dict)
    nearest_future: tuple[float, str, str, str] | None = None  # (hours, date, name, impact)

    for date_str, name, impact, h_utc, m_utc in _EVENTS:
        ev_dt      = _ev_dt(date_str, h_utc, m_utc)
        hours_away = (ev_dt - now).total_seconds() / 3600
        blackout   = _HIGH_BLACKOUT_HOURS if impact == "HIGH" else _MEDIUM_BLACKOUT_HOURS

        # Track nearest future event for informational display
        if hours_away > 0:
            if nearest_future is None or hours_away < nearest_future[0]:
                nearest_future = (hours_away, date_str, name, impact)

        if abs(hours_away) <= blackout:
            active_matches.append((_IMPACT_RANK.get(impact, 0), {
                "blocked":     impact == "HIGH",
                "impact":      impact,
                "event_name":  name,
                "event_date":  date_str,
                "hours_away":  round(hours_away, 1),
                "description": (
                    f"{'🚫' if impact=='HIGH' else '⚠'} {name} on {date_str} "
                    f"({'+'if hours_away>=0 else ''}{hours_away:.1f}h) — "
                    f"{'Signals suppressed' if impact=='HIGH' else 'Reduce size'}"
                ),
            }))

    if active_matches:
        # Return the highest-impact match (sort descending by rank)
        active_matches.sort(key=lambda x: x[0], reverse=True)
        result.update(active_matches[0][1])
        return result

    # No blackout — show next upcoming event as info
    if nearest_future:
        hours, date_str, name, _ = nearest_future
        if hours < 48:
            result["description"] = f"Next: {name} in {hours:.0f}h ({date_str})"

    return result


def get_upcoming_events(days: int = 7) -> list[dict]:
    """Return macro events in the next N days, sorted by time."""
    now = datetime.now(timezone.utc)
    out = []
    for date_str, name, impact, h_utc, m_utc in _EVENTS:
        ev_dt      = _ev_dt(date_str, h_utc, m_utc)
        hours_away = (ev_dt - now).total_seconds() / 3600
        if 0 <= hours_away <= days * 24:
            out.append({
                "date":       date_str,
                "name":       name,
                "impact":     impact,
                "hours_away": round(hours_away, 1),
            })
    return sorted(out, key=lambda x: x["hours_away"])
