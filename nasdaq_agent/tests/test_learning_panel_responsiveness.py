from pathlib import Path


def _main_source() -> str:
    return (Path(__file__).parent.parent / "main.py").read_text()


def test_learning_params_endpoint_does_not_queue_executor_jobs():
    src = _main_source()
    start = src.index("async def learning_params_status")
    body = src[start:start + 2200]
    assert "_LEARNING_PARAMS_CACHE_TTL_SECS" in src
    assert "_learning_params_cache" in body
    assert "results = [_safe_params(algo) for algo in _FAMILY_REPRESENTATIVES.values()]" in body
    assert "run_in_executor" not in body


def test_learning_phase2_endpoint_uses_in_process_status_read():
    src = _main_source()
    start = src.index("async def learning_phase2_status")
    body = src[start:start + 700]
    assert "status = _get_p2_engine().get_status()" in body
    assert "run_in_executor" not in body
