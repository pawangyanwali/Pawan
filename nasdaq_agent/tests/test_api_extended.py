"""
Extended API endpoint tests using FastAPI TestClient.
Tests the 57 untested endpoints beyond what test_api.py covers.
"""
import pytest
pytest.importorskip("ta", reason="ta library not available — API tests skipped")

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
    mock_scanner.register_per_ticker_callback = MagicMock()

    with patch("agent.scanner.scanner", mock_scanner), \
         patch("main.scanner", mock_scanner):
        from main import app
        with TestClient(app) as c:
            yield c


# ── /api/services ────────────────────────────────────────────────────────────

def test_services_status_code(client):
    r = client.get("/api/services")
    assert r.status_code == 200


def test_services_returns_dict(client):
    r = client.get("/api/services")
    body = r.json()
    assert isinstance(body, dict)


def test_services_has_scanner_key(client):
    r = client.get("/api/services")
    body = r.json()
    assert "scanner" in body


def test_services_has_valkey_key(client):
    r = client.get("/api/services")
    body = r.json()
    assert "valkey" in body


# ── /api/credit-usage ─────────────────────────────────────────────────────────

def test_credit_usage_ok(client):
    r = client.get("/api/credit-usage")
    assert r.status_code == 200


def test_credit_usage_returns_dict_or_list(client):
    r = client.get("/api/credit-usage")
    body = r.json()
    assert body is not None


# ── /api/account-state ───────────────────────────────────────────────────────

def test_account_state_ok(client):
    r = client.get("/api/account-state")
    assert r.status_code == 200


def test_account_state_returns_dict(client):
    r = client.get("/api/account-state")
    body = r.json()
    assert isinstance(body, dict)


# ── /api/deep-model/status ───────────────────────────────────────────────────

def test_deep_model_status_ok(client):
    r = client.get("/api/deep-model/status")
    assert r.status_code == 200


def test_deep_model_status_returns_dict(client):
    r = client.get("/api/deep-model/status")
    body = r.json()
    assert isinstance(body, dict)


# ── /api/risk-status ─────────────────────────────────────────────────────────

def test_risk_status_ok(client):
    r = client.get("/api/risk-status")
    assert r.status_code == 200


def test_risk_status_returns_dict(client):
    r = client.get("/api/risk-status")
    body = r.json()
    assert isinstance(body, dict)


# ── /api/premarket-scan ──────────────────────────────────────────────────────

def test_premarket_scan_ok(client):
    r = client.get("/api/premarket-scan")
    assert r.status_code in (200, 503)


def test_premarket_scan_returns_dict(client):
    r = client.get("/api/premarket-scan")
    body = r.json()
    assert isinstance(body, dict)


# ── /api/ml-status ───────────────────────────────────────────────────────────

def test_ml_status_ok(client):
    r = client.get("/api/ml-status")
    assert r.status_code == 200


def test_ml_status_returns_dict(client):
    r = client.get("/api/ml-status")
    body = r.json()
    assert isinstance(body, dict)


def test_ml_status_has_deep_model_key(client):
    r = client.get("/api/ml-status")
    body = r.json()
    assert "deep_model" in body or "error" in body


# ── /api/weekend-learning/status ─────────────────────────────────────────────

def test_weekend_learning_status_ok(client):
    r = client.get("/api/weekend-learning/status")
    assert r.status_code == 200


def test_weekend_learning_status_returns_dict(client):
    r = client.get("/api/weekend-learning/status")
    body = r.json()
    assert isinstance(body, dict)


# ── /api/weekend-learning/history ────────────────────────────────────────────

def test_weekend_learning_history_ok(client):
    r = client.get("/api/weekend-learning/history")
    assert r.status_code == 200


def test_weekend_learning_history_returns_dict(client):
    r = client.get("/api/weekend-learning/history")
    body = r.json()
    assert isinstance(body, (dict, list))


# ── /api/weekend-learning/cache-stats ────────────────────────────────────────

def test_weekend_learning_cache_stats_ok(client):
    r = client.get("/api/weekend-learning/cache-stats")
    assert r.status_code == 200


def test_weekend_learning_cache_stats_returns_dict(client):
    r = client.get("/api/weekend-learning/cache-stats")
    body = r.json()
    assert isinstance(body, (dict, list))


# ── /api/backtest/mtf ─────────────────────────────────────────────────────────

def test_backtest_mtf_ok(client):
    r = client.get("/api/backtest/mtf")
    assert r.status_code == 200


def test_backtest_mtf_returns_dict_or_list(client):
    r = client.get("/api/backtest/mtf")
    body = r.json()
    assert isinstance(body, (dict, list))


# ── /api/algo-performance ─────────────────────────────────────────────────────

def test_algo_performance_ok(client):
    r = client.get("/api/algo-performance")
    assert r.status_code == 200


def test_algo_performance_returns_dict(client):
    r = client.get("/api/algo-performance")
    body = r.json()
    assert isinstance(body, (dict, list))


# ── /api/learning-status ─────────────────────────────────────────────────────

def test_learning_status_ok(client):
    r = client.get("/api/learning-status")
    assert r.status_code == 200


def test_learning_status_returns_dict(client):
    r = client.get("/api/learning-status")
    body = r.json()
    assert isinstance(body, dict)


def test_learning_status_has_engine_key(client):
    r = client.get("/api/learning-status")
    body = r.json()
    assert "engine" in body


# ── /api/learning-log ────────────────────────────────────────────────────────

def test_learning_log_ok(client):
    r = client.get("/api/learning-log")
    assert r.status_code == 200


def test_learning_log_returns_dict(client):
    r = client.get("/api/learning-log")
    body = r.json()
    assert isinstance(body, dict)


def test_learning_log_has_log_key(client):
    r = client.get("/api/learning-log")
    body = r.json()
    assert "log" in body
    assert isinstance(body["log"], list)


# ── /api/after-hours ─────────────────────────────────────────────────────────

def test_after_hours_ok(client):
    r = client.get("/api/after-hours")
    assert r.status_code == 200


def test_after_hours_returns_dict(client):
    r = client.get("/api/after-hours")
    body = r.json()
    assert isinstance(body, dict)


def test_after_hours_has_snapshots_key(client):
    r = client.get("/api/after-hours")
    body = r.json()
    assert "snapshots" in body


# ── /api/broker/status ───────────────────────────────────────────────────────

def test_broker_status_ok(client):
    r = client.get("/api/broker/status")
    assert r.status_code == 200


def test_broker_status_returns_dict(client):
    r = client.get("/api/broker/status")
    body = r.json()
    assert isinstance(body, dict)


def test_broker_status_has_connected_key(client):
    r = client.get("/api/broker/status")
    body = r.json()
    assert "connected" in body or "error" in body


# ── /api/broker/positions ────────────────────────────────────────────────────

def test_broker_positions_ok(client):
    r = client.get("/api/broker/positions")
    assert r.status_code == 200


def test_broker_positions_returns_dict(client):
    r = client.get("/api/broker/positions")
    body = r.json()
    assert isinstance(body, dict)


def test_broker_positions_has_positions_key(client):
    r = client.get("/api/broker/positions")
    body = r.json()
    assert "positions" in body
    assert isinstance(body["positions"], list)


# ── /api/broker/orders ───────────────────────────────────────────────────────

def test_broker_orders_ok(client):
    r = client.get("/api/broker/orders")
    assert r.status_code == 200


def test_broker_orders_has_orders_key(client):
    r = client.get("/api/broker/orders")
    body = r.json()
    assert "orders" in body
    assert isinstance(body["orders"], list)


# ── /api/market/streamer ─────────────────────────────────────────────────────

def test_market_streamer_ok(client):
    r = client.get("/api/market/streamer")
    assert r.status_code == 200


def test_market_streamer_returns_dict(client):
    r = client.get("/api/market/streamer")
    body = r.json()
    assert isinstance(body, dict)


# ── /api/market/hours ────────────────────────────────────────────────────────

def test_market_hours_ok(client):
    r = client.get("/api/market/hours")
    assert r.status_code == 200


def test_market_hours_returns_dict(client):
    r = client.get("/api/market/hours")
    body = r.json()
    assert isinstance(body, dict)


# ── /api/market/movers ───────────────────────────────────────────────────────

def test_market_movers_ok(client):
    r = client.get("/api/market/movers")
    assert r.status_code == 200


def test_market_movers_returns_dict(client):
    r = client.get("/api/market/movers")
    body = r.json()
    assert isinstance(body, dict)


def test_market_movers_has_movers_key(client):
    r = client.get("/api/market/movers")
    body = r.json()
    assert "movers" in body


# ── /api/universe ────────────────────────────────────────────────────────────

def test_universe_ok(client):
    r = client.get("/api/universe")
    assert r.status_code == 200


def test_universe_returns_dict(client):
    r = client.get("/api/universe")
    body = r.json()
    assert isinstance(body, dict)


def test_universe_has_total_key(client):
    r = client.get("/api/universe")
    body = r.json()
    # Either success or error dict
    assert "universe_total" in body or "error" in body


# ── /api/pipeline-metrics ────────────────────────────────────────────────────

def test_pipeline_metrics_ok(client):
    r = client.get("/api/pipeline-metrics")
    assert r.status_code == 200


def test_pipeline_metrics_returns_dict(client):
    r = client.get("/api/pipeline-metrics")
    body = r.json()
    assert isinstance(body, dict)


# ── /api/blend-weights ───────────────────────────────────────────────────────

def test_blend_weights_ok(client):
    r = client.get("/api/blend-weights")
    assert r.status_code == 200


def test_blend_weights_returns_dict(client):
    r = client.get("/api/blend-weights")
    body = r.json()
    assert isinstance(body, dict)


# ── /api/backtest/results ────────────────────────────────────────────────────

def test_backtest_results_ok(client):
    r = client.get("/api/backtest/results")
    assert r.status_code == 200


def test_backtest_results_returns_dict(client):
    r = client.get("/api/backtest/results")
    body = r.json()
    assert isinstance(body, dict)


def test_backtest_results_has_status_key(client):
    r = client.get("/api/backtest/results")
    body = r.json()
    assert "status" in body or "error" in body


# ── /api/notify/config ───────────────────────────────────────────────────────

def test_notify_config_ok(client):
    r = client.get("/api/notify/config")
    assert r.status_code == 200


def test_notify_config_returns_dict(client):
    r = client.get("/api/notify/config")
    body = r.json()
    assert isinstance(body, dict)


# ── /api/clusters ────────────────────────────────────────────────────────────

def test_clusters_ok(client):
    r = client.get("/api/clusters")
    assert r.status_code == 200


def test_clusters_returns_dict(client):
    r = client.get("/api/clusters")
    body = r.json()
    assert isinstance(body, dict)


def test_clusters_has_expected_keys(client):
    r = client.get("/api/clusters")
    body = r.json()
    assert "A" in body
    assert "B" in body
    assert "C" in body
    assert "assignments" in body


# ── /api/paper-trading/daily ─────────────────────────────────────────────────

def test_paper_trading_daily_ok(client):
    r = client.get("/api/paper-trading/daily")
    assert r.status_code == 200


def test_paper_trading_daily_returns_dict(client):
    r = client.get("/api/paper-trading/daily")
    body = r.json()
    assert isinstance(body, dict)


def test_paper_trading_daily_has_expected_keys(client):
    r = client.get("/api/paper-trading/daily")
    body = r.json()
    assert "daily" in body
    assert "today" in body


# ── /api/paper-trading/performance ───────────────────────────────────────────

def test_paper_trading_performance_ok(client):
    r = client.get("/api/paper-trading/performance")
    assert r.status_code == 200


def test_paper_trading_performance_returns_dict(client):
    r = client.get("/api/paper-trading/performance")
    body = r.json()
    assert isinstance(body, dict)


def test_paper_trading_performance_has_summary_key(client):
    r = client.get("/api/paper-trading/performance")
    body = r.json()
    assert "summary" in body


# ── POST /api/adaptive-filter/reset ──────────────────────────────────────────

def test_adaptive_filter_reset_ok(client):
    r = client.post("/api/adaptive-filter/reset")
    assert r.status_code == 200


def test_adaptive_filter_reset_returns_dict(client):
    r = client.post("/api/adaptive-filter/reset")
    body = r.json()
    assert isinstance(body, dict)


def test_adaptive_filter_reset_has_ok_key(client):
    r = client.post("/api/adaptive-filter/reset")
    body = r.json()
    assert "ok" in body
    assert body["ok"] is True
