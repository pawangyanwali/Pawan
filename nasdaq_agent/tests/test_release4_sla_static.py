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
        "scanner",
        "learner",
        "scheduler",
        "context-intel",
        "watchdog",
    ):
        assert f'"{service_name}"' in src


def test_dashboard_surfaces_runtime_sla_without_touching_price_chip():
    src = _src("web/static/index.html")

    assert 'id="ops-sla-chip"' in src
    assert "function _setOpsSlaChip(sla)" in src
    assert "async function pollRuntimeSla()" in src
    assert "/api/runtime-health" in src
    assert "setInterval(pollRuntimeSla, 10000)" in src
    assert "setInterval(() => {" in src
    assert "document.getElementById('data-freshness')" in src
