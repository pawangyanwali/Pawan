from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

import agent.db as db
import agent.broker.schwab_market_data as schwab_market_data
import agent.universe_registry as registry
import agent.valkey_client as valkey
from agent.ticker_universe import FULL_UNIVERSE, TIER1


ROOT = Path(__file__).resolve().parents[1]


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


class _Connection:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        return _Rows(self.rows)


def test_seed_quarantine_is_auditable_and_does_not_remove_tier_one():
    assert len(registry.INITIAL_QUARANTINE) == 72
    assert len(set(registry.INITIAL_QUARANTINE)) == 72
    assert not set(registry.INITIAL_QUARANTINE) & set(TIER1)


def test_runtime_universe_uses_registry_and_preserves_catalog_order(monkeypatch):
    connection = _Connection(
        [{"ticker": "NVDA"}, {"ticker": "AAPL"}, {"ticker": "NEWC"}]
    )
    monkeypatch.setattr(registry, "init_universe_registry", lambda: None)
    monkeypatch.setattr(db, "get_conn", lambda read_only=False: connection)

    assert registry.get_runtime_universe() == ["AAPL", "NVDA", "NEWC"]


def test_runtime_universe_failure_uses_known_safe_catalog(monkeypatch):
    monkeypatch.setattr(
        registry,
        "init_universe_registry",
        lambda: (_ for _ in ()).throw(RuntimeError("db unavailable")),
    )
    result = registry.get_runtime_universe()
    assert "AAPL" in result
    assert not set(registry.INITIAL_QUARANTINE) & set(result)
    assert len(result) == len(FULL_UNIVERSE) - len(registry.INITIAL_QUARANTINE)


def test_replacement_requires_listing_liquidity_price_and_history(monkeypatch):
    monkeypatch.setattr(registry, "init_universe_registry", lambda: None)
    assert registry.promote_replacement(
        "NEWC", average_daily_volume=1_000_000, history_bars=500,
        quote_price=20, listing_verified=False, source="TEST",
    ) is False
    assert registry.promote_replacement(
        "NEWC", average_daily_volume=100_000, history_bars=500,
        quote_price=20, listing_verified=True, source="TEST",
    ) is False
    assert registry.promote_replacement(
        "NEWC", average_daily_volume=1_000_000, history_bars=100,
        quote_price=20, listing_verified=True, source="TEST",
    ) is False

    connection = _Connection([])
    monkeypatch.setattr(db, "get_conn", lambda: connection)
    monkeypatch.setattr(registry, "_publish_summary", lambda: None)
    assert registry.promote_replacement(
        "NEWC", average_daily_volume=1_000_000, history_bars=500,
        quote_price=20, listing_verified=True, source="TEST",
    ) is True
    assert any("VALIDATED_REPLACEMENT" in sql for sql, _ in connection.executed)


def test_price_bus_health_ignores_quarantined_hash_entries(monkeypatch):
    now = time.time()
    monkeypatch.setattr(
        valkey,
        "get_all_prices",
        lambda: {
            "AAPL": {"updated_at": now, "source_status": "LIVE"},
            "MSFT": {"updated_at": now, "source_status": "REST_FALLBACK"},
            "OLD": {"updated_at": 1, "source_status": "LIVE"},
        },
    )

    class Client:
        def get(self, key):
            assert key == "universe:eligible"
            return json.dumps(["AAPL", "MSFT"])

    monkeypatch.setattr(valkey, "_get_client", lambda: Client())
    health = valkey.price_bus_health(max_age_s=2)
    assert health["total"] == 2
    assert health["live"] == 1
    assert health["fallback"] == 1
    assert health["stale"] == 0


def test_canonical_services_use_the_shared_runtime_universe():
    market = (ROOT / "services" / "market_data_service.py").read_text(encoding="utf-8")
    scalp = (ROOT / "services" / "scalp_engine_service.py").read_text(encoding="utf-8")
    system = (ROOT / "routers" / "system.py").read_text(encoding="utf-8")
    assert "runtime_tickers = get_runtime_universe()" in market
    assert "tickers = get_runtime_universe()" in scalp
    assert '"quarantined_total"' in system
    assert '"eligible_total"' in system
    assert "_discover_universe_replacements()" in market
    assert "UNIVERSE_TARGET_ELIGIBLE" in (
        ROOT / "docker-compose.yml"
    ).read_text(encoding="utf-8")


def test_replacement_discovery_promotes_only_validated_external_movers(monkeypatch):
    import services.market_data_service as market_data_service

    frame = pd.DataFrame(
        {
            "Open": [10.0] * 400,
            "High": [10.1] * 400,
            "Low": [9.9] * 400,
            "Close": [10.0] * 400,
            "Volume": [2000.0] * 400,
        },
        index=pd.date_range("2026-06-25T13:30:00Z", periods=400, freq="min"),
    )
    monkeypatch.setenv("UNIVERSE_TARGET_ELIGIBLE", "406")
    monkeypatch.setattr(
        registry,
        "get_universe_registry_summary",
        lambda: {
            "eligible_total": 405,
            "eligible_tickers": ["AAPL"],
            "quarantined": [],
        },
    )
    monkeypatch.setattr(
        schwab_market_data,
        "fetch_top_movers_symbols",
        lambda n: ["NEWC", "AAPL"],
    )
    monkeypatch.setattr(
        schwab_market_data,
        "fetch_full_quotes",
        lambda tickers: {"NEWC": {"last": 10.0}},
    )
    monkeypatch.setattr(
        schwab_market_data,
        "fetch_price_history_batch_async",
        lambda *args, **kwargs: {"NEWC": frame},
    )
    promoted = []
    monkeypatch.setattr(
        registry,
        "promote_replacement",
        lambda ticker, **kwargs: promoted.append((ticker, kwargs)) or True,
    )

    assert market_data_service._discover_universe_replacements() == 1
    assert promoted[0][0] == "NEWC"
    assert promoted[0][1]["listing_verified"] is True
    assert promoted[0][1]["history_bars"] == 400
    assert promoted[0][1]["average_daily_volume"] == 780_000
