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
    body = src[start:start + 2400]
    assert "retrain_all(TRAINING_TICKERS, daily_data=daily_data, skip_deep=True)" in body


def test_startup_training_waits_for_closed_market_window():
    src = _scanner_source()
    start = src.index("def _train_ml_background")
    body = src[start:start + 2400]
    assert "while not self._training_allowed_now()" in body
    assert "Startup retrain" in body
    assert "waiting for CLOSED window" in src


def test_scheduled_training_and_deep_finetune_are_live_session_gated():
    src = _scanner_source()
    assert "from agent.market_hours import get_session_info, get_market_session" in src
    assert "def _training_allowed_now" in src
    assert "return self._training_session() == \"CLOSED\"" in src

    retrain_start = src.index("def _should_retrain")
    retrain_body = src[retrain_start:retrain_start + 900]
    assert "session = self._training_session()" in retrain_body
    assert "session != \"CLOSED\"" in retrain_body
    assert "Scheduled retrain" in retrain_body

    deep_start = src.index("def _should_finetune_deep")
    deep_body = src[deep_start:deep_start + 900]
    assert "session = self._training_session()" in deep_body
    assert "session != \"CLOSED\"" in deep_body
    assert "Deep BiLSTM fine-tune" in deep_body
