from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_runtime_health_exposes_release4_sla_snapshot():
    # Endpoint moved from main.py to routers/system.py
    src = _src("routers/system.py")

    assert '@router.get("/api/runtime-health")' in src
    assert "from agent.runtime_sla import evaluate_runtime_sla" in src
    assert "sla = evaluate_runtime_sla(" in src
    assert '"sla": sla' in src


def test_release4_sla_rules_cover_active_prices_and_required_services():
    src = _src("agent/runtime_sla.py")

    assert "ACTIVE_SESSIONS" in src
    assert "active_price_trusted_fresh_pct_critical" in src
    assert "Live stream degraded" in src
    for service_name in (
        "web-api",
        "market-data",
        "scalp-engine",
        "scalp-learner",
        "scheduler",
        "context-intel",
        "watchdog",
    ):
        assert f'"{service_name}"' in src


def test_command_center_surfaces_market_and_plan_freshness():
    src = _src("web/static/scalp.html")

    assert 'id="md-chip"' in src
    assert 'id="scan-chip"' in src
    assert "Market data" in src
    assert "d.scan_ts" in src
    assert "setTimeout(load,1000)" in src
