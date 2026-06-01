from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _repo(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


def test_service_heartbeat_publishes_to_postgres_and_valkey():
    src = _src("agent/service_heartbeat.py")

    assert "service:{service_name}:heartbeat" in src
    assert "set_state(heartbeat_key, payload, ttl_s=ttl_s)" in src
    assert "client.setex(heartbeat_key, ttl_s" in src
    assert "start_service_heartbeat" in src


def test_all_standalone_services_publish_uniform_heartbeats():
    service_files = {
        "market-data": "services/market_data_service.py",
        "scanner": "services/scanner_service.py",
        "learner": "services/learner_service.py",
        "scheduler": "services/scheduler_service.py",
        "context-intel": "services/context_intel_service.py",
        "watchdog": "services/watchdog_service.py",
    }

    for service_name, path in service_files.items():
        src = _src(path)
        assert "start_service_heartbeat" in src
        assert f'start_service_heartbeat("{service_name}", _runner)' in src


def test_api_services_reports_full_release3_container_health():
    # Container health logic moved from main.py to routers/system.py
    src = _src("routers/system.py")

    assert "def _container_health(valkey_connected: bool)" in src
    for service_name in (
        "market-data",
        "scanner",
        "learner",
        "scheduler",
        "context-intel",
        "watchdog",
    ):
        assert f'service:{{service_name}}:heartbeat' in _src("agent/service_heartbeat.py")
        assert f'"{service_name}"' in src

    assert '"scanner:streamer"' in src
    assert '"ctx:intel:heartbeat"' in src
    # fresh_coverage_pct is published by the streamer into service state
    assert "fresh_coverage_pct" in _src("agent/broker/schwab_streamer.py")


def test_dashboard_container_health_includes_all_release3_services():
    src = _src("web/static/index.html")

    for dom_id in (
        "ct-webapi",
        "ct-marketdata",
        "ct-scanner",
        "ct-learner",
        "ct-scheduler",
        "ct-context",
        "ct-watchdog",
    ):
        assert f'id="{dom_id}-badge"' in src
        assert f"id+'-badge'" in src or f"{dom_id}','" in src

    assert "repeat(auto-fit,minmax(72px,1fr))" in src
    assert "['ct-marketdata','market-data']" in src
    assert "['ct-context','context-intel']" in src
    assert "['ct-watchdog','watchdog']" in src


def test_compose_resource_budget_matches_t3a_xlarge_release3():
    src = _repo("docker-compose.yml")

    assert "t3a.xlarge resource budget: 4 vCPU / 16 GB RAM" in src
    assert "mem_limit: 1500m" in src
    assert 'cpus: "0.75"' in src
    assert "mem_limit: 5000m" in src
    assert 'cpus: "1.75"' in src
    assert "mem_limit: 4000m" in src
    assert 'cpus: "1.25"' in src
    assert "mem_limit: 768m" in src
