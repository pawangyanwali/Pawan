from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_root_serves_the_scalp_command_center() -> None:
    src = _src("routers/system.py")
    assert 'STATIC_DIR, "scalp.html"' in src
    assert 'STATIC_DIR, "index.html"' not in src


def test_historical_workers_keep_durable_cross_worker_status() -> None:
    router = _src("routers/backtest.py")
    entrypoint = _src("historical/__main__.py")
    assert "def _job_status" in router
    assert "tracking_via_status_file" in router
    assert "def _write_job_status" in entrypoint
    assert 'Path(os.getenv("LOG_DIR", "/app/logs"))' in entrypoint
