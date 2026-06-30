from agent.runtime_sla import evaluate_runtime_sla


def _containers(up=True):
    return {
        name: {"up": up, "last_seen_ago_s": 10, "detail": "heartbeat 10s ago"}
        for name in (
            "web-api",
            "market-data",
            "scalp-engine",
            "scalp-learner",
            "scheduler",
            "context-intel",
            "watchdog",
        )
    }


def _price(status="LIVE", trusted=100.0, live=100.0, total=477):
    return {
        "status": status,
        "total": total,
        "trusted_fresh_pct": trusted,
        "live_pct": live,
        "fresh_pct": trusted,
    }


def test_active_session_live_prices_and_services_are_ok():
    sla = evaluate_runtime_sla(
        session="REGULAR",
        price_2s=_price(),
        price_5s=_price(),
        scanner={"scan_age_s": 30, "signals": 250},
        containers=_containers(),
        valkey={"connected": True},
    )

    assert sla["status"] == "OK"
    assert sla["alert_count"] == 0


def test_active_session_rest_fallback_is_degraded_not_reported_as_live():
    sla = evaluate_runtime_sla(
        session="AFTER_HOURS",
        price_2s=_price(status="REST_FALLBACK", trusted=100.0, live=0.0),
        price_5s=_price(status="REST_FALLBACK", trusted=100.0, live=0.0),
        scanner={"scan_age_s": 45, "signals": 250},
        containers=_containers(),
        valkey={"connected": True},
    )

    assert sla["status"] == "DEGRADED"
    assert any(a["component"] == "market-data" for a in sla["alerts"])


def test_active_session_stale_prices_are_critical():
    sla = evaluate_runtime_sla(
        session="PRE_MARKET",
        price_2s=_price(status="STALE", trusted=10.0, live=0.0),
        price_5s=_price(status="STALE", trusted=20.0, live=0.0),
        scanner={"scan_age_s": 20, "signals": 250},
        containers=_containers(),
        valkey={"connected": True},
    )

    assert sla["status"] == "CRITICAL"
    assert any(a["component"] == "prices" for a in sla["alerts"])


def test_active_session_scan_snapshots_are_critical_even_when_fresh():
    sla = evaluate_runtime_sla(
        session="REGULAR",
        price_2s=_price(status="SCAN_SNAPSHOT", trusted=100.0, live=0.0),
        price_5s=_price(status="SCAN_SNAPSHOT", trusted=100.0, live=0.0),
        scanner={"scan_age_s": 20, "signals": 250},
        containers=_containers(),
        valkey={"connected": True},
    )

    assert sla["status"] == "CRITICAL"
    assert any(a["message"] == "No trusted live/fallback prices" for a in sla["alerts"])


def test_active_session_plan_data_gaps_are_critical():
    sla = evaluate_runtime_sla(
        session="REGULAR",
        price_2s=_price(),
        price_5s=_price(),
        scanner={
            "scan_age_s": 10,
            "universe_total": 400,
            "valid_plan_count": 0,
            "data_gap_count": 400,
        },
        containers=_containers(),
        valkey={"connected": True},
    )

    assert sla["status"] == "CRITICAL"
    assert any(
        alert["message"] == "Canonical plan data gaps above SLA"
        for alert in sla["alerts"]
    )


def test_closed_session_still_requires_scalp_learner_and_watchdog():
    containers = _containers()
    containers["scalp-learner"] = {"up": False, "detail": "no scalp learner heartbeat"}
    containers["watchdog"] = {"up": False, "detail": "no watchdog heartbeat"}

    sla = evaluate_runtime_sla(
        session="CLOSED",
        price_2s=_price(status="STALE", trusted=0.0, live=0.0, total=477),
        price_5s=_price(status="STALE", trusted=0.0, live=0.0, total=477),
        scanner={"scan_age_s": 600, "signals": 0},
        containers=containers,
        valkey={"connected": True},
    )

    assert sla["status"] == "CRITICAL"
    assert {a["component"] for a in sla["alerts"]} >= {"scalp-learner", "watchdog"}
