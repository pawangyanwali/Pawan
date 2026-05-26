from pathlib import Path


def _scanner_source() -> str:
    return (Path(__file__).parent.parent / "agent" / "scanner.py").read_text()


def test_startup_training_waits_for_two_completed_scans():
    src = _scanner_source()
    assert "_second_scan_done" in src
    assert "self._second_scan_done.wait(timeout=900)" in src
    assert "self._scan_count >= 2" in src


def test_startup_training_defers_deep_phase():
    src = _scanner_source()
    start = src.index("def _train_ml_background")
    body = src[start:start + 1800]
    assert "retrain_all(TRAINING_TICKERS, daily_data=daily_data, skip_deep=True)" in body
