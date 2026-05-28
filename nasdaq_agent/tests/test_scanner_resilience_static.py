from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_scanner_service_does_not_publish_empty_snapshot_over_previous_data():
    src = _src("services/scanner_service.py")

    assert "from agent.signal_snapshot import read_latest, write_latest" in src
    assert "if not sigs_dicts:" in src
    assert "previous = read_latest() or {}" in src
    assert "preserving previous dashboard snapshot" in src
    assert "skipping empty publish" in src


def test_scanner_full_cycle_preserves_previous_signals_on_zero_result_scan():
    src = _src("agent/scanner.py")

    zero_guard = src.index("if not results:")
    assign = src.index("self.signals   = results")
    assert zero_guard < assign
    assert "preserving previous" in src[zero_guard:assign]
    assert "return list(self.signals)" in src[zero_guard:assign]
    assert "leaving the dashboard snapshot unchanged" in src[zero_guard:assign]


def test_data_fetcher_closed_session_stale_cache_fallback_is_scoped():
    src = _src("agent/data_fetcher.py")

    assert "def _restore_closed_session_cache(" in src
    assert 'upper() == "CLOSED"' in src
    assert "_restore_closed_session_cache(to_fetch, interval, interval_key, result)" in src
    assert '_restore_closed_session_cache(\n                        still_miss, "1min", interval_key, result' in src
    assert "stale-cache fallback" in src
