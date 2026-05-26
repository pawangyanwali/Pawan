import json
import os
import time
from datetime import datetime, timedelta, timezone

import main


def _write_cache(path, *, age_seconds: int, signals: list[dict]) -> str:
    ts = (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).isoformat()
    path.write_text(json.dumps({"ts": ts, "signals": signals}))
    mtime = time.time() - age_seconds
    os.utime(path, (mtime, mtime))
    return ts


def _reset_cache_state(monkeypatch, cache_file):
    monkeypatch.setattr(main, "_SIGNAL_CACHE_FILE", str(cache_file))
    main._last_signals_dicts = []
    main._last_signals_ts = ""
    main.scanner.signals = []
    main.scanner.last_scan = None


def test_closed_session_loads_long_weekend_signal_cache(tmp_path, monkeypatch):
    cache_file = tmp_path / "signal_cache.json"
    signals = [{"ticker": "AAPL", "prediction": "NEUTRAL", "confidence": 50.0}]
    ts = _write_cache(cache_file, age_seconds=3 * 24 * 3600, signals=signals)
    _reset_cache_state(monkeypatch, cache_file)
    monkeypatch.setattr(
        main,
        "get_session_info",
        lambda: {"session": "CLOSED", "is_holiday": True, "is_weekend": False},
    )

    main._load_signal_cache()

    assert main._last_signals_dicts == signals
    assert main._last_signals_ts == ts

    sigs, last_scan, from_cache = main._current_signal_snapshot()
    assert sigs == signals
    assert last_scan == ts
    assert from_cache is True


def test_active_session_rejects_old_signal_cache(tmp_path, monkeypatch):
    cache_file = tmp_path / "signal_cache.json"
    signals = [{"ticker": "AAPL", "prediction": "BUY", "confidence": 75.0}]
    _write_cache(cache_file, age_seconds=main._SIGNAL_CACHE_ACTIVE_MAX_AGE_SECS + 60, signals=signals)
    _reset_cache_state(monkeypatch, cache_file)
    monkeypatch.setattr(
        main,
        "get_session_info",
        lambda: {"session": "PRIME", "is_holiday": False, "is_weekend": False},
    )

    main._load_signal_cache()

    assert main._last_signals_dicts == []
    assert main._last_signals_ts == ""


def test_loaded_closed_cache_expires_when_session_becomes_active(tmp_path, monkeypatch):
    cache_file = tmp_path / "signal_cache.json"
    signals = [{"ticker": "MSFT", "prediction": "SELL", "confidence": 71.0}]
    ts = _write_cache(cache_file, age_seconds=3 * 24 * 3600, signals=signals)
    _reset_cache_state(monkeypatch, cache_file)
    monkeypatch.setattr(
        main,
        "get_session_info",
        lambda: {"session": "CLOSED", "is_holiday": False, "is_weekend": True},
    )
    main._load_signal_cache()
    assert main._last_signals_dicts == signals

    monkeypatch.setattr(
        main,
        "get_session_info",
        lambda: {"session": "PRIME", "is_holiday": False, "is_weekend": False},
    )

    sigs, last_scan, from_cache = main._current_signal_snapshot()

    assert sigs == []
    assert last_scan == ts
    assert from_cache is False
