import pytest
pytestmark = pytest.mark.slow

"""
Tests for REST API endpoints via FastAPI TestClient.
Uses a mock scanner so no real Twelve Data API calls are made.
"""
import pytest
pytest.importorskip("ta", reason="ta library not available — API tests skipped")

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with scanner mocked out."""
    mock_scanner = MagicMock()
    mock_scanner.signals = []
    mock_scanner.last_scan = None
    mock_scanner.is_running = True
    mock_scanner.start_background = MagicMock()
    mock_scanner.stop = MagicMock()
    mock_scanner.register_callback = MagicMock()

    with patch("agent.scanner.scanner", mock_scanner), \
         patch("main.scanner", mock_scanner):
        from main import app
        from auth.dependencies import get_current_user, AuthenticatedUser

        _test_admin = AuthenticatedUser(
            id=1, username="test_admin", role="ADMIN", status="ACTIVE", jti="test-jti"
        )
        app.dependency_overrides[get_current_user] = lambda: _test_admin
        yield TestClient(app)
        app.dependency_overrides.clear()


def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert body["status"] == "ok"

def test_signals_endpoint(client):
    r = client.get("/api/signals")
    assert r.status_code == 200
    body = r.json()
    assert "signals" in body
    assert isinstance(body["signals"], list)

def test_regime_endpoint(client):
    r = client.get("/api/regime")
    assert r.status_code == 200
    body = r.json()
    assert "regime" in body
    assert "session" in body

def test_macro_calendar_endpoint(client):
    r = client.get("/api/macro-calendar")
    assert r.status_code == 200
    body = r.json()
    assert "current" in body
    assert "upcoming" in body
    assert isinstance(body["upcoming"], list)

def test_paper_trading_endpoint(client):
    r = client.get("/api/paper-trading")
    assert r.status_code == 200
    body = r.json()
    assert "summary" in body
    assert "open_trades" in body
    assert "closed_trades" in body

def test_watchlist_get(client):
    r = client.get("/api/watchlist")
    assert r.status_code == 200
    body = r.json()
    assert "base" in body
    assert "watchlist" in body
    assert isinstance(body["base"], list)
    assert isinstance(body["watchlist"], list)

def test_watchlist_add(client):
    r = client.post("/api/watchlist/add?ticker=PLTR")
    assert r.status_code == 200
    body = r.json()
    assert "watchlist" in body

def test_watchlist_remove(client):
    # Add first
    client.post("/api/watchlist/add?ticker=RKLB")
    r = client.post("/api/watchlist/remove?ticker=RKLB")
    assert r.status_code == 200

def test_backtest_stats_endpoint(client):
    r = client.get("/api/backtest/stats")
    assert r.status_code == 200
    body = r.json()
    assert "stats" in body
    assert "overall" in body["stats"]

def test_backtest_tracking_endpoint(client):
    r = client.get("/api/backtest/tracking")
    assert r.status_code == 200
    body = r.json()
    assert "tracking" in body
    assert isinstance(body["tracking"], list)

def test_backtest_recent_endpoint(client):
    r = client.get("/api/backtest/recent")
    assert r.status_code == 200
    body = r.json()
    assert "recent" in body
    assert isinstance(body["recent"], list)

def test_backtest_path_endpoint(client):
    r = client.get("/api/backtest/path/FAKE_SIGNAL_ID")
    assert r.status_code == 200
    body = r.json()
    assert "path" in body
    assert isinstance(body["path"], list)

def test_position_size_endpoint(client):
    r = client.get("/api/position-size?entry=100&stop=95&account_size=10000")
    assert r.status_code == 200
    body = r.json()
    assert "shares" in body or "position_size" in body or "dollar_risk" in body

def test_signal_history_endpoint(client):
    r = client.get("/api/signal-history")
    assert r.status_code == 200
    body = r.json()
    assert "signals" in body
    assert "stats" in body

def test_root_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
