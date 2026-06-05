from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_token_load_defaults_are_read_only():
    auth = _read("agent/broker/schwab_auth.py")

    assert "def load_stored(self, schedule_refresh: bool = False)" in auth
    assert "def load_stored_tokens(schedule_refresh: bool = False)" in auth
    assert "def load_stored_md_tokens(schedule_refresh: bool = False)" in auth


def test_market_data_client_requests_token_service_refresh_only():
    market_data = _read("agent/broker/schwab_market_data.py")

    assert "schwab:refresh_requested:marketdata" in market_data
    assert "._market_data.refresh(" not in market_data
    assert "_md_app.refresh(" not in market_data
    assert "return bool(_auth_headers())" not in market_data


def test_token_service_owns_refresh_requests():
    token_service = _read("services/token_service.py")

    assert "schwab:refresh_requested:trader" in token_service
    assert "schwab:refresh_requested:marketdata" in token_service
    assert "refresh_mgr.refresh()" in token_service
    assert 'os.environ["SCHWAB_TOKEN_OWNER"] = "1"' in token_service


def test_token_service_initializes_postgres_token_table():
    auth = _read("agent/broker/schwab_auth.py")
    token_service = _read("services/token_service.py")

    assert "def init_schwab_token_store()" in auth
    assert "CREATE TABLE IF NOT EXISTS schwab_tokens" in auth
    assert "def _ensure_schwab_token_table(conn)" in auth
    assert "from agent.db import get_conn" in auth
    assert "from agent.db import get_pool" not in auth
    assert "pool.connection()" not in auth
    assert "init_schwab_token_store()" in token_service


def test_market_data_reacts_to_app_specific_token_events():
    service = _read("services/market_data_service.py")

    assert "schwab:token_rotated:trader" in service
    assert "schwab:token_rotated:marketdata" in service
    assert "_handle_token_event(payload)" in service
    assert "_last_token_generation" in service
    assert "WS already connected" in service


def test_web_oauth_callback_does_not_start_streamer():
    broker = _read("routers/broker.py")
    start = broker.index("async def schwab_at_web_callback")
    end = broker.index("@router.get(\"/schwab/auth/md\")")
    callback = broker[start:end]

    assert "start_streamer" not in callback
    assert "tokens_refreshed" not in callback


def test_scanner_algo_path_passes_rr_quality_to_paper_execution():
    scanner = _read("agent/scanner.py")

    assert "_algo_rr_quality" in scanner
    assert 'rr_quality        = ""' not in scanner
