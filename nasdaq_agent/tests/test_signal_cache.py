import json
import os
import time
from datetime import datetime, timedelta, timezone

import main


class _Regime:
    regime = "NEUTRAL"
    label = "Neutral"
    color = "#94a3b8"


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
    monkeypatch.setattr(main, "get_regime", lambda: _Regime())
    monkeypatch.setattr(main, "get_live_quotes_snapshot", lambda max_age_s=None: {})


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


def test_observation_rows_fill_dashboard_when_signals_and_cache_empty(tmp_path, monkeypatch):
    cache_file = tmp_path / "missing_signal_cache.json"
    _reset_cache_state(monkeypatch, cache_file)
    monkeypatch.setattr(main, "NASDAQ_TICKERS", ["AAPL", "MSFT"])
    monkeypatch.setattr(
        main,
        "get_session_info",
        lambda: {"session": "PRE_MARKET", "is_holiday": False, "is_weekend": False},
    )
    monkeypatch.setattr(
        main,
        "get_live_quotes_snapshot",
        lambda max_age_s=None: {
            "AAPL": {"last": 199.0, "mark": 200.0, "net_pct_change": 1.25, "updated_at": time.time()},
            "MSFT": {"last": 399.5, "mark": 400.0, "net_pct_change": -0.5, "updated_at": time.time()},
        },
    )

    sigs, last_scan, from_cache = main._current_signal_snapshot()

    assert len(sigs) == 2
    assert sigs[0]["ticker"] == "AAPL"
    assert sigs[0]["price"] == 200.0
    assert sigs[0]["prediction"] == "NEUTRAL"
    assert sigs[0]["is_observation"] is True
    assert last_scan is not None
    assert from_cache is False


def test_active_observation_rows_require_fresh_quotes(tmp_path, monkeypatch):
    cache_file = tmp_path / "missing_signal_cache.json"
    _reset_cache_state(monkeypatch, cache_file)
    monkeypatch.setattr(main, "NASDAQ_TICKERS", ["AAPL"])
    monkeypatch.setattr(
        main,
        "get_session_info",
        lambda: {"session": "PRIME", "is_holiday": False, "is_weekend": False},
    )
    seen = {}

    def _quotes(max_age_s=None):
        seen["max_age_s"] = max_age_s
        return {"AAPL": {"last": 199.0, "updated_at": time.time()}}

    monkeypatch.setattr(main, "get_live_quotes_snapshot", _quotes)

    sigs, _, _ = main._current_signal_snapshot()

    assert len(sigs) == 1
    assert seen["max_age_s"] == 10.0


def test_valid_cache_is_augmented_with_missing_observation_rows(tmp_path, monkeypatch):
    cache_file = tmp_path / "signal_cache.json"
    cached = [{"ticker": "AAPL", "prediction": "BUY", "confidence": 75.0}]
    ts = _write_cache(cache_file, age_seconds=60, signals=cached)
    _reset_cache_state(monkeypatch, cache_file)
    monkeypatch.setattr(main, "NASDAQ_TICKERS", ["AAPL", "MSFT"])
    monkeypatch.setattr(
        main,
        "get_session_info",
        lambda: {"session": "CLOSED", "is_holiday": False, "is_weekend": False},
    )
    monkeypatch.setattr(
        main,
        "get_live_quotes_snapshot",
        lambda max_age_s=None: {
            "AAPL": {"last": 199.0, "updated_at": time.time()},
            "MSFT": {"last": 400.0, "updated_at": time.time()},
        },
    )
    main._load_signal_cache()

    sigs, last_scan, from_cache = main._current_signal_snapshot()

    assert [s["ticker"] for s in sigs] == ["AAPL", "MSFT"]
    assert sigs[0] == cached[0]
    assert sigs[1]["is_observation"] is True
    assert last_scan is not None and last_scan != ts
    assert from_cache is True
