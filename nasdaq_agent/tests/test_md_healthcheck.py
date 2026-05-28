from services import md_healthcheck


def test_regular_session_keeps_strict_default_threshold(monkeypatch):
    monkeypatch.delenv("MD_HEALTH_MIN_FRESH_PCT", raising=False)
    monkeypatch.delenv("MD_HEALTH_MIN_FRESH_PCT_EXTENDED", raising=False)

    assert md_healthcheck._min_fresh_pct_for_session("REGULAR") == 80.0
    assert md_healthcheck._min_fresh_pct_for_session("STANDARD") == 80.0


def test_extended_sessions_use_extended_threshold(monkeypatch):
    monkeypatch.setenv("MD_HEALTH_MIN_FRESH_PCT", "90")
    monkeypatch.setenv("MD_HEALTH_MIN_FRESH_PCT_EXTENDED", "60")

    assert md_healthcheck._min_fresh_pct_for_session("PRE_MARKET") == 60.0
    assert md_healthcheck._min_fresh_pct_for_session("AFTER_HOURS") == 60.0


def test_exact_session_threshold_overrides_extended(monkeypatch):
    monkeypatch.setenv("MD_HEALTH_MIN_FRESH_PCT_EXTENDED", "60")
    monkeypatch.setenv("MD_HEALTH_MIN_FRESH_PCT_PRE_MARKET", "75")

    assert md_healthcheck._min_fresh_pct_for_session("PRE_MARKET") == 75.0


def test_bad_env_value_falls_back_safely(monkeypatch):
    monkeypatch.setenv("MD_HEALTH_MIN_FRESH_PCT_EXTENDED", "bad-value")

    assert md_healthcheck._min_fresh_pct_for_session("AFTER_HOURS") == 60.0
