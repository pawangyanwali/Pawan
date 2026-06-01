from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _repo(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


def test_learner_service_runs_continuous_deep_loop_and_publishes_state():
    src = _src("services/learner_service.py")

    assert "def _continuous_deep_loop()" in src
    assert "def _run_deep_cycle()" in src
    assert "retrain_deep_all(hist_15m)" in src
    assert '"deep":            _deep_state_snapshot()' in src
    assert 'threading.Thread(target=_continuous_deep_loop' in src
    assert "Continuous learning active" in src


def test_learning_engine_can_retrain_during_active_sessions_inside_learner():
    src = _src("agent/learning_engine.py")

    assert "LEARNING_RETRAIN_ACTIVE_SESSIONS" in src
    assert "RETRAIN_ACTIVE_SESSIONS and _session != \"CLOSED\"" in src
    assert "inside learner container" in src
    assert "skip_deep=_skip_deep" in src
    assert '"active_session_retrain_enabled": RETRAIN_ACTIVE_SESSIONS' in src


def test_adaptive_filter_observe_mode_does_not_block_trade_signals(monkeypatch):
    from agent import adaptive_filter as af

    monkeypatch.setattr(af, "_ENFORCEMENT_MODE", "observe")
    with af._lock:
        old_threshold = af._state["dynamic_threshold"]
        old_blocked = dict(af._state["blocked_contexts"])
        af._state["dynamic_threshold"] = 99.0
        af._state["blocked_contexts"] = {}
    try:
        blocked, reason = af.should_suppress(confidence=1.0)
        assert blocked is False
        assert "below learned threshold" in reason
    finally:
        with af._lock:
            af._state["dynamic_threshold"] = old_threshold
            af._state["blocked_contexts"] = old_blocked


def test_deep_model_has_concurrency_guard():
    src = _src("agent/deep_model.py")

    assert "if _is_training_now:" in src
    assert "another deep retrain is already running" in src
    assert "return False" in src


def test_compose_enables_release2_continuous_learning_contracts():
    src = _repo("docker-compose.yml")

    assert 'LEARNING_RETRAIN_ACTIVE_SESSIONS: "1"' in src
    assert 'LEARNING_RETRAIN_INCLUDE_DEEP: "1"' in src
    assert 'LEARNER_DEEP_ENABLED:        "1"' in src
    assert 'ADAPTIVE_FILTER_ENFORCEMENT_MODE: "observe"' in src
    assert 'test: ["CMD", "python3", "/app/services/learner_healthcheck.py"]' in src


def test_learner_healthcheck_requires_fresh_running_engine_status():
    src = _src("services/learner_healthcheck.py")

    assert "learner:status" in src
    assert "learning engine not running" in src
    assert "learner:status stale" in src
    assert "deep learner status missing" in src
