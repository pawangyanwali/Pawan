from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _repo(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_service_heartbeat_publishes_to_postgres_and_valkey():
    src = _src("agent/service_heartbeat.py")

    assert "service:{service_name}:heartbeat" in src
    assert "set_state(heartbeat_key, payload, ttl_s=ttl_s)" in src
    assert "client.setex(heartbeat_key, ttl_s" in src
    assert "start_service_heartbeat" in src


def test_all_standalone_services_publish_uniform_heartbeats():
    service_files = {
        "market-data": "services/market_data_service.py",
        "scalp-engine": "services/scalp_engine_service.py",
        "scalp-learner": "services/scalp_learner_service.py",
        "scheduler": "services/scheduler_service.py",
        "context-intel": "services/context_intel_service.py",
        "watchdog": "services/watchdog_service.py",
    }

    for service_name, path in service_files.items():
        src = _src(path)
        assert "start_service_heartbeat" in src
        assert f'"{service_name}",' in src


def test_api_services_reports_full_release3_container_health():
    # Container health logic moved from main.py to routers/system.py
    src = _src("routers/system.py")

    assert "def _container_health(valkey_connected: bool)" in src
    for service_name in (
        "market-data",
        "scalp-engine",
        "scalp-learner",
        "scheduler",
        "context-intel",
        "watchdog",
    ):
        assert f'service:{{service_name}}:heartbeat' in _src("agent/service_heartbeat.py")
        assert f'"{service_name}"' in src

    assert '"market-data:status"' in src
    assert '"ctx:intel:heartbeat"' in src
    # fresh_coverage_pct is published by the streamer into service state
    assert "fresh_coverage_pct" in _src("agent/broker/schwab_streamer.py")


def test_command_center_exposes_runtime_health_and_one_second_refresh():
    src = _src("web/static/scalp.html")
    assert 'id="md-chip"' in src
    assert 'id="scan-chip"' in src
    assert 'id="execution-chip"' in src
    assert "state.timer=setTimeout(load,1000)" in src


def test_compose_resource_budget_matches_scalp_only_runtime():
    src = _repo("docker-compose.yml")
    assert "scalp-engine:" in src
    assert "scalp-learner:" in src
    assert 'cpus: "1.5"' in src
    assert 'cpus: "1.0"' in src
    assert "memory: 2G" in src
