from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEALTH = ROOT / "web" / "static" / "health.html"
SYSTEM = ROOT / "routers" / "system.py"
SCALP = ROOT / "web" / "static" / "scalp.html"


def test_health_page_has_dedicated_route_and_navigation():
    routes = SYSTEM.read_text(encoding="utf-8")
    dashboard = SCALP.read_text(encoding="utf-8")

    assert '@router.get("/health"' in routes
    assert 'STATIC_DIR, "health.html"' in routes
    assert 'href="/health"' in dashboard


def test_health_page_monitors_all_authoritative_domains():
    page = HEALTH.read_text(encoding="utf-8")

    for endpoint in (
        "/api/health",
        "/api/services",
        "/api/runtime-health",
        "/api/broker/status",
        "/api/scalp/dashboard",
    ):
        assert endpoint in page
    for service in (
        "Application Services",
        "Market Data Pipeline",
        "Infrastructure",
        "Schwab Applications",
        "Runtime SLA Alerts",
        "Endpoint Monitor",
    ):
        assert service in page


def test_health_page_keeps_schwab_polling_conservative_and_exposes_reauth():
    page = HEALTH.read_text(encoding="utf-8")

    assert "setInterval(loadCore,5000)" in page
    assert "setInterval(loadBroker,30000)" in page
    assert "if(state.coreLoading)return" in page
    assert "if(state.brokerLoading)return" in page
    assert "/schwab/auth/at" in page
    assert "/schwab/auth/md" in page
    assert 'id="settings-modal"' not in page
