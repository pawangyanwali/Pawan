from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_scalp_learner_runs_canonical_outcome_and_ml_loop():
    src = _src("services/scalp_learner_service.py")
    assert "train_and_maybe_promote" in src
    assert '"scalp-learner:status"' in src
    assert 'mode="OBSERVING"' in src
    assert "Immediate context learning runs on every canonical trade close" in src


def test_compose_enables_scalp_only_learning_contracts():
    import yaml

    src = _src("docker-compose.yml")
    services = yaml.safe_load(src)["services"]
    assert "scalp-learner" in services
    assert services["scalp-learner"]["command"][-1] == "services.scalp_learner_service"
    assert "learner" not in services


def test_scalp_learner_healthcheck_requires_fresh_status():
    src = _src("services/scalp_learner_healthcheck.py")
    assert "scalp-learner:status" in src
    assert "service:scalp-learner:heartbeat" in src
    assert "status stale" in src
