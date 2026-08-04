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
    assert "acknowledged_subscriptions" in page
    assert "desired_subscriptions" in page
    assert "subscription_coverage_pct" in page
    assert "active_quotes_60s" in page
    assert 'id="settings-modal"' not in page


def test_broker_streamer_status_uses_market_data_owner_state(monkeypatch):
    import agent.broker.schwab_streamer as streamer
    import routers.broker as broker

    monkeypatch.setattr(
        streamer,
        "get_streamer_status",
        lambda: {"ws_streamer": {"running": False, "connected": False}},
    )
    remote = {
        "ws_streamer": {
            "running": True,
            "connected": True,
            "desired_subscriptions": 477,
            "acknowledged_subscriptions": 477,
        }
    }
    monkeypatch.setattr(
        broker,
        "_get_cross_container_token_status",
        lambda key: remote if key == "market-data:status" else None,
    )
    assert broker._get_streamer_status() == remote
