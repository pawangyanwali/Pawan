from pathlib import Path
import ast
import time


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _repo(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


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


def test_scanner_side_training_is_disabled_by_default_in_production():
    src = _src("agent/scanner.py")
    compose = _repo("docker-compose.yml")

    assert 'os.getenv("NASDAQ_SCANNER_TRAINING_ENABLED", "0")' in src
    assert "if _scanner_training_enabled():" in src
    assert "if _scanner_training_enabled() and self._should_retrain()" in src
    assert "elif _scanner_training_enabled() and self._should_finetune_deep()" in src
    assert "Scanner-side ML training disabled; learner services own retraining." in src
    assert 'NASDAQ_SCANNER_TRAINING_ENABLED: "0"' in compose


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


def test_dashboard_displays_price_source_truth_instead_of_reconnecting_on_snapshots():
    src = _src("web/static/index.html")

    assert "function _updatePriceBusState(prices, fallbackStatus='UNKNOWN')" in src
    assert "let _priceBusCache = {}" in src
    assert "Evaluate the whole dashboard cache" in src
    assert 'id="price-source-breakdown"' in src
    assert "function _setPriceSourceBreakdown(cov, state='UNKNOWN')" in src
    assert "`WS ${live} | REST ${fallback} | STALE ${stale}`" in src
    assert "`WS ${live} | REST ${fallback} | SNAP ${snapshot} | STALE ${stale}`" in src
    assert "REST FALLBACK" in src
    assert "HYBRID LIVE" in src
    assert "WS /" in src
    assert "REST /" in src
    assert "SCAN SNAPSHOT" in src
    assert "MARKET CLOSED" in src
    assert "wsSilent" in src


def test_market_data_healthcheck_uses_price_bus_coverage_thresholds():
    src = _src("services/md_healthcheck.py")
    compose = _repo("docker-compose.yml")

    assert "price_bus_health(max_age_s=max_age_s)" in src
    assert "trusted_fresh_pct" in src
    assert "MD_HEALTH_MIN_FRESH_PCT" in src
    assert "MD_HEALTH_MAX_PRICE_AGE_S" in src
    assert 'MD_HEALTH_MIN_FRESH_PCT:     "80"' in compose
    assert 'MD_HEALTH_MIN_FRESH_PCT_EXTENDED: "60"' in compose


def test_market_data_token_reload_restart_is_reachable_before_continue():
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
    start_index = next(
        i for i, node in enumerate(body)
        if "_start(list(NASDAQ_TICKERS))" in ast.unparse(node)
    )
    continue_index = next(i for i, node in enumerate(body) if isinstance(node, ast.Continue))

    assert start_index < continue_index


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
    assert 'user: "0:0"' in compose
    assert "/var/run/docker.sock:/var/run/docker.sock" in compose
    assert 'test: ["CMD", "python3", "/app/services/watchdog_healthcheck.py"]' in compose
    assert 'os.getenv("WATCHDOG_DOCKER_SOCKET", "/var/run/docker.sock")' in healthcheck
    assert "GET /_ping HTTP/1.1" in healthcheck
    assert 'b"200 OK"' in healthcheck
