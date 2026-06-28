from __future__ import annotations

import json
from pathlib import Path

import agent.config_manager as config_manager
import agent.db as db
import agent.universe_registry as registry
import agent.valkey_client as valkey
import routers.scalp as scalp_router
import routers.system as system_router
from agent.scalp.runtime import ScalpRuntime


ROOT = Path(__file__).resolve().parents[1]


class _Result:
    def __init__(self, row=None, rows=None):
        self.row = row
        self.rows = rows or []

    def fetchone(self):
        return self.row

    def fetchall(self):
        return list(self.rows)


class _Conn:
    def __init__(self, row=None):
        self.row = row
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        return _Result(row=self.row)


class _Valkey:
    def __init__(self):
        self.sets = {}
        self.published = []

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)

    def publish(self, channel, payload):
        self.published.append((channel, json.loads(payload)))


def test_recheck_is_persisted_and_queued(monkeypatch):
    conn = _Conn(row={"ticker": "AAPL", "status": "QUARANTINED"})
    client = _Valkey()
    monkeypatch.setattr(registry, "init_universe_registry", lambda: None)
    monkeypatch.setattr(registry, "_publish_summary", lambda: None)
    monkeypatch.setattr(db, "get_conn", lambda: conn)
    monkeypatch.setattr(valkey, "_get_client", lambda: client)

    result = registry.request_recheck("aapl", requested_by="admin")

    assert result == {"ticker": "AAPL", "status": "CANDIDATE", "queued": True}
    assert client.sets["universe:recheck:requested"] == {"AAPL"}
    assert any("status='CANDIDATE'" in sql for sql, _ in conn.executed)


def test_manual_quarantine_requires_an_explanation(monkeypatch):
    monkeypatch.setattr(registry, "init_universe_registry", lambda: None)
    try:
        registry.quarantine_ticker("AAPL", reason="bad")
    except ValueError as exc:
        assert "at least 8 characters" in str(exc)
    else:
        raise AssertionError("short quarantine reason was accepted")


def test_runtime_universe_hot_reload_prunes_ticker_caches():
    runtime = ScalpRuntime(["AAPL", "MSFT"])
    runtime._indicator_cache["MSFT"] = (1, None, None, [], [])
    runtime._last_execution_bar["MSFT"] = 1

    assert runtime.update_tickers(["AAPL", "NVDA"]) is True
    assert runtime.tickers == ["AAPL", "NVDA"]
    assert "MSFT" not in runtime._indicator_cache
    assert "MSFT" not in runtime._last_execution_bar
    assert runtime.update_tickers(["AAPL", "NVDA"]) is False


def test_readiness_reports_enabled_services_and_waiting_learning(monkeypatch):
    monkeypatch.setattr(
        scalp_router,
        "_dashboard_snapshot",
        lambda: {
            "plans": [{"ticker": f"T{i}"} for i in range(400)],
            "counts": {"data_gap": 0},
            "session": {"session": "CLOSED"},
            "risk": {
                "budget": 150_000,
                "max_open_positions": 10,
                "tp1_r": 1.0,
                "tp2_r": 2.0,
                "execution_enabled": True,
            },
            "learning": {
                "outcome_count": 0,
                "context_count": 0,
                "action_count": 0,
                "active_actions": [],
                "ml_champion": None,
            },
        },
    )
    monkeypatch.setattr(
        registry,
        "get_universe_registry_summary",
        lambda include_symbols=False: {
            "catalog_total": 477,
            "eligible_total": 405,
            "quarantined_total": 72,
            "candidate_total": 0,
        },
    )
    monkeypatch.setattr(valkey, "health_status", lambda: {"connected": True})
    monkeypatch.setattr(
        valkey,
        "price_bus_health",
        lambda max_age_s=2: {"live": 0, "fallback": 0, "stale": 405},
    )
    monkeypatch.setattr(
        system_router,
        "_container_health",
        lambda connected: {
            "market-data": {"up": True},
            "scalp-engine": {"up": True},
            "scalp-learner": {"up": True},
        },
    )
    monkeypatch.setattr(config_manager.config, "get", lambda key, default=None: True)

    result = scalp_router._readiness_snapshot()

    assert result["status"] == "WARN"
    assert result["enabled"]["paper_execution"] is True
    assert result["enabled"]["immediate_learning"] is True
    assert result["enabled"]["scheduled_ml_training"] is True
    assert next(c for c in result["checks"] if c["name"] == "Fresh quote coverage")["state"] == "N/A"
    assert next(c for c in result["checks"] if c["name"] == "Immediate outcome learning")["state"] == "WAITING"


def test_universe_page_exposes_readiness_and_control_paths():
    page = (ROOT / "web" / "static" / "universe.html").read_text(encoding="utf-8")
    system = (ROOT / "routers" / "system.py").read_text(encoding="utf-8")
    market = (ROOT / "services" / "market_data_service.py").read_text(encoding="utf-8")
    engine = (ROOT / "services" / "scalp_engine_service.py").read_text(encoding="utf-8")

    assert "/api/scalp/readiness" in page
    assert "/api/universe/${encodeURIComponent(ticker)}/action" in page
    assert 'href="/universe"' in page
    assert '@router.get("/universe"' in system
    assert "_universe_recheck_loop" in market
    assert 'summary.get("candidates")' in market
    assert "update_md_poller_tickers" in market
    assert "runtime.update_tickers(get_runtime_universe())" in engine


def test_rest_poller_requires_nearly_complete_ws_coverage_before_standdown():
    streamer = (
        ROOT / "agent" / "broker" / "schwab_streamer.py"
    ).read_text(encoding="utf-8")
    assert "max(\n                    _WS_STANDDOWN_FRESH_PCT, 0.999\n                )" in streamer
    assert "for batch in cycle_batches" in streamer
