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
    assign = src.index("_actual_by_ticker = {s.ticker: s for s in results}")
    assert zero_guard < assign
    assert "preserving previous" in src[zero_guard:assign]
    assert "return list(self.signals)" in src[zero_guard:assign]
    assert "leaving the dashboard snapshot unchanged" in src[zero_guard:assign]


def test_scanner_merges_partial_results_before_publishing():
    src = _src("agent/scanner.py")

    assert "_actual_by_ticker = {s.ticker: s for s in results}" in src
    assert "_previous_by_ticker = {s.ticker: s for s in self.signals}" in src
    assert "_observation_signal(" in src
    assert "self.signals   = merged_results" in src
    assert "self._notify(merged_results)" in src


def test_pipeline_does_not_slow_cooldown_cycle_budget_deferrals():
    src = _src("agent/pipeline.py")
    pending_block = src[src.index("if pending:"):src.index("finally:", src.index("if pending:"))]

    assert "deferred %d ticker" in pending_block
    assert "_slow_skip_until" not in pending_block


def test_scanner_closed_session_uses_5min_cache_as_proxy_when_1min_unavailable():
    src = _src("agent/scanner.py")

    assert 'if _sess.get("session", "").upper() == "CLOSED":' in src
    assert "batch_1m[_ticker] = _df_proxy" in src
    assert "Closed-session scan using 5min cached bars as a 1min proxy" in src


def test_scanner_skips_deep_inference_while_market_closed():
    src = _src("agent/scanner.py")

    assert '_session_name_for_ml = str(get_session_info().get("session", "")).upper()' in src
    assert 'if _has_15m and _session_name_for_ml != "CLOSED"' in src
    assert "else 0.5" in src


def test_data_fetcher_closed_session_stale_cache_fallback_is_scoped():
    src = _src("agent/data_fetcher.py")

    assert "def _restore_closed_session_cache(" in src
    assert 'upper() == "CLOSED"' in src
    assert "_restore_closed_session_cache(to_fetch, interval, interval_key, result)" in src
    assert '_restore_closed_session_cache(\n                        still_miss, "1min", interval_key, result' in src
    assert "stale-cache fallback" in src
