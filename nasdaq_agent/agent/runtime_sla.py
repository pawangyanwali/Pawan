"""Runtime SLA evaluation for the production dashboard.

The health endpoint already exposes raw process and price-bus facts.  This
module turns those facts into one operator-facing status so production can
distinguish "running" from "safe to trust for trading decisions."
"""

from __future__ import annotations

import time
from typing import Any


ACTIVE_SESSIONS = {"PRE_MARKET", "REGULAR", "STANDARD", "AFTER_HOURS"}
CRITICAL = "CRITICAL"
WARN = "WARN"
INFO = "INFO"


def _num(data: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    try:
        return float((data or {}).get(key, default) or default)
    except Exception:
        return default


def _alert(
    alerts: list[dict[str, Any]],
    severity: str,
    component: str,
    message: str,
    *,
    detail: str = "",
    action: str = "",
) -> None:
    alerts.append(
        {
            "id": f"{component}:{message}".lower().replace(" ", "_"),
            "severity": severity,
            "component": component,
            "message": message,
            "detail": detail,
            "action": action,
        }
    )


def _is_up(entry: dict[str, Any] | None) -> bool:
    return bool((entry or {}).get("up") is True)


def evaluate_runtime_sla(
    *,
    session: str,
    price_2s: dict[str, Any],
    price_5s: dict[str, Any],
    scanner: dict[str, Any],
    containers: dict[str, dict[str, Any]],
    valkey: dict[str, Any],
) -> dict[str, Any]:
    """Return OK/DEGRADED/CRITICAL plus actionable alerts.

    Active sessions use a strict 2-second price freshness gate because the UI
    has a hard 1-second dashboard update requirement.  Closed sessions do not
    fail just because quotes are stale, but learning/watchdog/container health
    still remains mandatory.
    """
    session_key = str(session or "CLOSED").upper()
    active_session = session_key in ACTIVE_SESSIONS
    alerts: list[dict[str, Any]] = []

    if not bool((valkey or {}).get("connected")):
        _alert(
            alerts,
            CRITICAL,
            "valkey",
            "Valkey disconnected",
            detail=str((valkey or {}).get("error") or "cache unavailable"),
            action="Restore Valkey connectivity; price fan-out and service heartbeats depend on it.",
        )

    required_services = (
        "web-api",
        "market-data",
        "scalp-engine",
        "scalp-learner",
        "scheduler",
        "context-intel",
        "watchdog",
    )
    for name in required_services:
        entry = (containers or {}).get(name, {})
        if not _is_up(entry):
            _alert(
                alerts,
                CRITICAL,
                name,
                f"{name} heartbeat missing",
                detail=str(entry.get("detail") or "service heartbeat is stale or absent"),
                action=f"Restart or inspect the {name} container before trusting production.",
            )

    # Learning must keep running even outside market hours.
    learner = (containers or {}).get("scalp-learner", {})
    if _is_up(learner) and _num(learner, "last_seen_ago_s", 0.0) > 180:
        _alert(
            alerts,
            WARN,
            "scalp-learner",
            "Learner heartbeat slow",
            detail=f"last heartbeat {learner.get('last_seen_ago_s')}s ago",
            action="Inspect learner logs; continuous learning should not pause.",
        )

    total_prices = int(_num(price_2s, "total", 0.0))
    status_2s = str((price_2s or {}).get("status") or "NO_DATA").upper()
    trusted_2s = _num(price_2s, "trusted_fresh_pct", 0.0)
    live_2s = _num(price_2s, "live_pct", 0.0)
    trusted_5s = _num(price_5s, "trusted_fresh_pct", 0.0)

    if active_session:
        if total_prices <= 0:
            _alert(
                alerts,
                CRITICAL,
                "prices",
                "No price data",
                action="Restore Schwab market data before relying on the dashboard.",
            )
        elif trusted_2s < 80.0:
            _alert(
                alerts,
                CRITICAL,
                "prices",
                "Price freshness below SLA",
                detail=f"2s trusted freshness {trusted_2s:.1f}% across {total_prices} tickers",
                action="Check market-data, Schwab tokens, Valkey price bus, and REST fallback health.",
            )
        elif trusted_2s < 95.0:
            # Treat missing price_5s data (startup, no 5-s window yet) as WARN not CRITICAL.
            severity = WARN if (not price_5s or trusted_5s >= 95.0) else CRITICAL
            _alert(
                alerts,
                severity,
                "prices",
                "Price cadence degraded",
                detail=f"2s trusted freshness {trusted_2s:.1f}%; 5s trusted freshness {trusted_5s:.1f}%",
                action="Dashboard may not meet the 1-second update target; inspect market-data load.",
            )

        if status_2s in {"SCAN_SNAPSHOT", "STALE", "NO_DATA"}:
            _alert(
                alerts,
                CRITICAL,
                "market-data",
                "No trusted live/fallback prices",
                detail=f"price status {status_2s}",
                action="Do not trade from stale scanner snapshots; restore market-data service.",
            )
        elif status_2s in {"REST_FALLBACK", "PARTIAL_FALLBACK"} or live_2s < 80.0:
            _alert(
                alerts,
                WARN,
                "market-data",
                "Live stream degraded",
                detail=f"price status {status_2s}; live coverage {live_2s:.1f}%",
                action="Fallback quotes are fresh but not live streaming; verify Schwab streamer auth and connection.",
            )

        scan_age = _num(scanner, "scan_age_s", -1.0)
        if scan_age < 0:
            _alert(
                alerts,
                CRITICAL,
                "scalp-engine",
                "Scalp engine freshness unknown",
                action="Confirm scalp-engine writes canonical scan:latest snapshots.",
            )
        elif scan_age > 180:
            _alert(
                alerts,
                CRITICAL,
                "scalp-engine",
                "Scalp engine stale during active session",
                detail=f"last scan {scan_age:.1f}s ago",
                action="Inspect scalp-engine workers and market-data dependencies.",
            )

        universe_total = int(_num(scanner, "universe_total", 0.0))
        data_gaps = int(_num(scanner, "data_gap_count", 0.0))
        if universe_total > 0:
            gap_pct = data_gaps / universe_total * 100.0
            if gap_pct > 15.0:
                _alert(
                    alerts,
                    CRITICAL,
                    "scalp-engine",
                    "Canonical plan data gaps above SLA",
                    detail=f"{data_gaps}/{universe_total} plans blocked by data gaps ({gap_pct:.1f}%)",
                    action="Keep execution disabled; inspect quote ordering, one-minute bars, and context coverage.",
                )
            elif gap_pct > 5.0:
                _alert(
                    alerts,
                    WARN,
                    "scalp-engine",
                    "Canonical plan data gaps elevated",
                    detail=f"{data_gaps}/{universe_total} plans blocked by data gaps ({gap_pct:.1f}%)",
                    action="Review missing plan inputs before enabling canonical execution.",
                )
    else:
        if total_prices <= 0:
            _alert(
                alerts,
                INFO,
                "prices",
                "No closed-session price cache",
                action="Expected outside active sessions unless closed-session polling is enabled.",
            )
        scan_age = _num(scanner, "scan_age_s", -1.0)
        if scan_age > 900:
            _alert(
                alerts,
                WARN,
                "scalp-engine",
                "Scalp snapshot old",
                detail=f"last scan {scan_age:.1f}s ago",
                action="Closed-session scanning can be slower, but stale snapshots should recover before pre-market.",
            )

    severities = {a["severity"] for a in alerts}
    if CRITICAL in severities:
        overall = CRITICAL
    elif WARN in severities:
        overall = "DEGRADED"
    else:
        overall = "OK"

    return {
        "status": overall,
        "active_session": active_session,
        "generated_at": time.time(),
        "alert_count": len([a for a in alerts if a["severity"] in {CRITICAL, WARN}]),
        "alerts": alerts,
        "thresholds": {
            "active_price_trusted_fresh_pct_warn": 95.0,
            "active_price_trusted_fresh_pct_critical": 80.0,
            "active_live_pct_warn": 80.0,
            "active_scan_age_s_critical": 180,
            "active_plan_data_gap_pct_warn": 5.0,
            "active_plan_data_gap_pct_critical": 15.0,
        },
    }
