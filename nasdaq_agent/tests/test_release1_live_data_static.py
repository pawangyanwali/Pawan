from pathlib import Path
import ast
import time


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _repo(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_market_data_quotes_are_source_labeled_and_rest_does_not_mask_live_ws():
    src = _src("agent/broker/schwab_streamer.py")

    assert 'quote["source"] = "SCHWAB_WS"' in src
    assert 'quote["source_status"] = "LIVE"' in src
    assert '"source": "SCHWAB_REST"' in src
    assert '"source_status": "REST_FALLBACK"' in src
    assert "ws_fresh_coverage(max_age_s=2.0)" in src
    assert "NASDAQ_WS_STANDDOWN_FRESH_PCT" in src


def test_price_bus_health_measures_whole_dashboard_not_single_ticker():
    src = _src("agent/valkey_client.py")

    assert "def price_bus_health(max_age_s: float = 2.0)" in src
    assert "prices = get_all_prices()" in src
    assert '"live_pct"' in src
    assert '"fallback_pct"' in src
    assert '"trusted_fresh_pct"' in src
    assert '"SCAN_SNAPSHOT"' in src
    assert '"PARTIAL_LIVE"' in src
    assert '"REST_FALLBACK"' in src


def test_price_bus_health_does_not_treat_scanner_snapshots_as_trusted(monkeypatch):
    from agent import valkey_client

    now = time.time()
    monkeypatch.setattr(
        valkey_client,
        "get_all_prices",
        lambda: {
            "AAPL": {"updated_at": now, "source_status": "SCAN_SNAPSHOT"},
            "MSFT": {"updated_at": now, "source_status": "SCAN_SNAPSHOT"},
        },
    )

    health = valkey_client.price_bus_health(max_age_s=5)

    assert health["fresh_pct"] == 100.0
    assert health["trusted_fresh_pct"] == 0.0
    assert health["status"] == "SCAN_SNAPSHOT"


def test_price_bus_preserves_open_when_ws_tick_omits_it():
    from agent.valkey_client import _merge_sticky_price_fields

    merged = _merge_sticky_price_fields(
        {"last": 101.0, "open": 0.0, "source_status": "LIVE"},
        {"last": 100.0, "open": 98.5, "source_status": "REST_FALLBACK"},
    )

    assert merged["last"] == 101.0
    assert merged["open"] == 98.5
    assert merged["source_status"] == "LIVE"


def test_price_bus_rest_cannot_overwrite_fresh_ws_source():
    from agent.valkey_client import _merge_price_sources

    merged = _merge_price_sources(
        {
            "last": 99.0,
            "bid": 98.99,
            "ask": 99.01,
            "open": 97.0,
            "updated_at": 104.0,
            "source": "SCHWAB_REST",
            "source_status": "REST_FALLBACK",
            "is_live": False,
        },
        {
            "last": 101.0,
            "bid": 100.99,
            "ask": 101.01,
            "updated_at": 100.0,
            "ws_updated_at": 100.0,
            "source": "SCHWAB_WS",
            "source_status": "LIVE",
            "is_live": True,
        },
        now=104.0,
    )

    assert merged["last"] == 101.0
    assert merged["open"] == 97.0
    assert merged["updated_at"] == 100.0
    assert merged["rest_updated_at"] == 104.0
    assert merged["source_status"] == "LIVE"


def test_price_bus_rest_takes_over_after_ws_priority_window():
    from agent.valkey_client import _merge_price_sources

    merged = _merge_price_sources(
        {
            "last": 99.0,
            "updated_at": 110.0,
            "source_status": "REST_FALLBACK",
        },
        {
            "last": 101.0,
            "updated_at": 100.0,
            "ws_updated_at": 100.0,
            "source_status": "LIVE",
        },
        now=110.0,
    )

    assert merged["last"] == 99.0
    assert merged["source_status"] == "REST_FALLBACK"


def test_streamer_seeds_open_from_rest_without_downgrading_live_ws():
    src = _src("agent/broker/schwab_streamer.py")
    valkey = _src("agent/valkey_client.py")

    assert '"open":       float(quote.get("open") or 0)' in src
    assert "rest_open = float(q.get(\"open\") or 0)" in src
    assert "need_open_backfill: list[str] = []" in src
    assert "_schedule_open_backfill(need_open_backfill)" in src
    assert "fetch_price_history_batch_async(" in src
    assert 'interval="1min"' in src
    assert "background=True" in src
    assert 'quote.get("source_status") == "LIVE"' in src
    assert '"source_status": "LIVE"' in src
    assert "_STICKY_PRICE_FIELDS = (\"open\",)" in valkey
    assert "client.hmget(_HASH, keys)" in valkey


def test_streamer_extracts_today_open_from_regular_session_history():
    import pandas as pd
    from agent.broker.schwab_streamer import _extract_today_open_from_df

    today = pd.Timestamp.now(tz="America/New_York").normalize()
    idx = pd.DatetimeIndex([
        today + pd.Timedelta(hours=9, minutes=30),
        today + pd.Timedelta(hours=9, minutes=31),
    ]).tz_convert("UTC")
    df = pd.DataFrame({"Open": [42.25, 43.0]}, index=idx)

    assert _extract_today_open_from_df(df) == 42.25


def test_scalp_engine_has_no_training_or_broker_data_fetch_ownership():
    src = _src("services/scalp_engine_service.py") + _src("agent/scalp/runtime.py")
    compose = _repo("docker-compose.yml")

    assert "retrain_all" not in src
    assert "retrain_deep_all" not in src
    assert "fetch_batch_interval" not in src
    assert "scalp-engine:" in compose
    assert "scanner:" not in compose.replace("scanner:streamer", "")


def test_scanner_snapshots_are_not_reported_as_live_prices():
    # Snapshot classification lives in valkey_client.price_bus_health()
    vk_src = _src("agent/valkey_client.py")
    assert '"SCAN_SNAPSHOT"' in vk_src
    assert "source_status" in vk_src
    assert "price_bus_health" in vk_src

    # Non-live quotes must carry is_live=False in the streamer
    streamer_src = _src("agent/broker/schwab_streamer.py")
    assert '"is_live": False' in streamer_src

    # md_healthcheck uses price_bus_health to gate liveness
    mdhc_src = _src("services/md_healthcheck.py")
    assert "price_bus_health(max_age_s=max_age_s)" in mdhc_src


def test_command_center_displays_price_source_truth():
    src = _src("web/static/scalp.html")
    assert 'id="md-chip"' in src
    assert "h.live||0" in src
    assert "h.fallback||0" in src
    assert "h.stale||0" in src
    assert "Market data" in src


def test_market_data_healthcheck_uses_price_bus_coverage_thresholds():
    src = _src("services/md_healthcheck.py")
    compose = _repo("docker-compose.yml")

    assert "price_bus_health(max_age_s=max_age_s)" in src
    assert "trusted_fresh_pct" in src
    assert "MD_HEALTH_MIN_FRESH_PCT" in src
    assert "MD_HEALTH_MAX_PRICE_AGE_S" in src
    assert "market-data:" in compose
    assert 'services.md_healthcheck' in compose


def test_market_data_token_reload_delegates_once_before_continue():
    src = _src("services/market_data_service.py")
    tree = ast.parse(src)
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_token_reload_loop"
    )

    token_event_ifs = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and "message.get('type') == 'message'" in ast.unparse(node.test)
    ]
    assert token_event_ifs

    body = token_event_ifs[0].body
    handler_index = next(
        i for i, node in enumerate(body)
        if "_handle_token_event(payload)" in ast.unparse(node)
    )
    continue_index = next(i for i, node in enumerate(body) if isinstance(node, ast.Continue))

    assert handler_index < continue_index
    assert "_start(list(NASDAQ_TICKERS))" not in ast.unparse(token_event_ifs[0])


def test_watchdog_service_restarts_stopped_or_unhealthy_containers():
    src = _src("services/watchdog_service.py")
    compose = _repo("docker-compose.yml")
    healthcheck = _src("services/watchdog_healthcheck.py")

    assert "WATCHDOG_SERVICES" in src
    assert "/containers/json?all=1" in src
    assert "/var/run/docker.sock" in src
    assert '"health=unhealthy"' in src
    assert "WATCHDOG_UNHEALTHY_STRIKES" in src
    assert 'command: ["python", "-m", "services.watchdog_service"]' in compose
    assert "user: root" in compose
    assert "/var/run/docker.sock:/var/run/docker.sock" in compose
    assert 'test: ["CMD", "python", "-m", "services.watchdog_healthcheck"]' in compose
    assert 'os.getenv("WATCHDOG_DOCKER_SOCKET", "/var/run/docker.sock")' in healthcheck
    assert "GET /_ping HTTP/1.1" in healthcheck
    assert 'b"200 OK"' in healthcheck
