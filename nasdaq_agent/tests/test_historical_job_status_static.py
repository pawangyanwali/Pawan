from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_historical_jobs_show_live_dashboard_progress_after_start() -> None:
    src = _src("web/static/index.html")

    assert "function _histActive(job)" in src
    assert "setInterval(loadHistoricalStatus, 3000)" in src
    assert "_histSetButton('hist-retrain-btn', rtActive" in src
    assert "_histSetButton('hist-bt-btn', btActive" in src
    assert "Starting historical retrain subprocess" in src
    assert "Starting historical backtest subprocess" in src
    assert "setTimeout(loadHistoricalStatus, 500)" in src
    assert "cross-worker status" in src
    assert "Complete — 0 signals" in src


def test_historical_status_survives_multi_worker_status_polling() -> None:
    router = _src("routers/backtest.py")
    retrain = _src("historical/retrain.py")
    backtest = _src("historical/backtest.py")

    assert "def _job_status" in router
    assert "fresh status file is therefore the" in router
    assert "tracking_via_status_file" in router
    assert "_write_status(_RETRAIN_STATUS" in router
    assert "_write_status(_BACKTEST_STATUS" in router
    assert "\"status\": \"starting\"" in router
    assert "done >= total" in router
    assert "state[\"updated_at\"] = time.time()" in retrain
    assert "state[\"updated_at\"] = time.time()" in backtest
    assert "if i % 10 == 0 or i == n" not in backtest
