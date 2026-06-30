from datetime import datetime, timezone
from pathlib import Path

import pytest

import agent.paper_trading as paper
import agent.signal_snapshot as signal_snapshot
import agent.valkey_client as valkey
import agent.scalp.store as scalp_store
import routers.scalp as scalp_router
from routers.config_router import _validate_scalp_learning


ROOT = Path(__file__).resolve().parents[1]


def test_command_center_is_a_dedicated_authenticated_live_page():
    html = (ROOT / "web" / "static" / "scalp.html").read_text(encoding="utf-8")
    system = (ROOT / "routers" / "system.py").read_text(encoding="utf-8")
    main = (ROOT / "main.py").read_text(encoding="utf-8")

    assert '@router.get("/scalp"' in system
    assert "scalp.html" in system
    assert "app.include_router(scalp_router)" in main
    assert "authFetch('/api/scalp/dashboard')" in html
    assert "setTimeout(load,1000)" in html
    assert "canonical scalp plans" in html.lower()
    assert "Learning Guard" in html
    assert "Open Positions" in html
    assert "Recent Learned Outcomes" in html
    assert "Shadow Validation" in html
    assert 'href="/settings"' in html


def test_dashboard_snapshot_joins_plans_prices_risk_and_learning(monkeypatch):
    now = datetime.now(timezone.utc).timestamp()
    monkeypatch.setattr(
        signal_snapshot,
        "read_latest",
        lambda: {
            "ts": now,
            "session": {"session": "REGULAR"},
            "regime": {"regime": "BULLISH"},
            "signals": [
                {
                    "scalp_plan": {
                        "plan_id": "plan-1",
                        "ticker": "AAPL",
                        "side": "LONG",
                        "valid": True,
                        "confidence": 82,
                        "entry": 100,
                        "stop_loss": 99,
                        "tp1": 101,
                        "tp2": 102,
                        "reasons": ["LONG_SETUP"],
                        "blockers": [],
                    }
                }
            ],
        },
    )
    monkeypatch.setattr(
        paper,
        "get_open_trades",
        lambda: [
            {
                "ticker": "AAPL",
                "direction": "BUY",
                "entry_price": 100,
                "stop": 99,
                "shares": 10,
                "shares_remaining": 10,
            }
        ],
    )
    monkeypatch.setattr(paper, "get_today_pnl", lambda: {"total_pnl_dollar": 25})
    monkeypatch.setattr(
        valkey,
        "get_all_prices",
        lambda: {
            "AAPL": {
                "last": 101,
                "updated_at": now,
                "source_status": "WS_LIVE",
            }
        },
    )
    monkeypatch.setattr(
        valkey,
        "price_bus_health",
        lambda max_age_s: {"status": "LIVE", "live": 1, "fallback": 0, "stale": 0},
    )
    monkeypatch.setattr(
        scalp_store,
        "learning_dashboard_data",
        lambda **_kwargs: {
            "contexts": [],
            "recent_actions": [],
            "recent_outcomes": [],
            "counts": {"outcomes": 0, "contexts": 0, "actions": 0},
        },
    )
    monkeypatch.setattr(scalp_router, "_cache_value", None)
    monkeypatch.setattr(scalp_router, "_cache_ts", 0.0)

    result = scalp_router._dashboard_snapshot()

    assert result["counts"]["actionable"] == 1
    assert result["counts"]["long"] == 1
    assert result["plans"][0]["state"] == "ACTIONABLE"
    assert result["plans"][0]["stop_r"] == pytest.approx(1.0)
    assert result["plans"][0]["tp1_r"] == pytest.approx(1.0)
    assert result["plans"][0]["tp2_r"] == pytest.approx(2.0)
    assert result["positions"][0]["unrealized_pnl"] == pytest.approx(10.0)
    assert result["positions"][0]["price_source"] == "WS_LIVE"
    assert result["risk"]["realized_pnl"] == pytest.approx(25.0)
    assert result["risk"]["open_risk"] == pytest.approx(10.0)
    assert result["risk"]["max_open_positions"] <= result["risk"]["configured_max_open_positions"]
    assert "max_allocated_pct" in result["risk"]
    assert "max_portfolio_heat_pct" in result["risk"]
    assert "effective_daily_loss_halt_usd" in result["risk"]
    assert result["learning"]["outcome_count"] == 0
    assert result["risk"]["shadow_enabled"] is True
    assert "shadow" in result


def test_hard_learning_block_is_not_presented_as_watch():
    assert scalp_router._plan_state(
        {
            "side": "LONG",
            "valid": False,
            "blockers": ["LEARNING_CONTEXT_BLOCK"],
        }
    ) == "BLOCKED"


def test_closed_market_staleness_is_not_reported_as_data_loss():
    plan = {
        "session": "CLOSED",
        "side": "NONE",
        "valid": False,
        "blockers": ["QUOTE_STALE", "INDICATOR_BAR_STALE", "SESSION_CLOSED_BLOCKED"],
    }
    assert scalp_router._plan_state(plan) == "BLOCKED"
    assert scalp_router._plan_reason(plan) == "MARKET_CLOSED_LAST_SESSION_DATA"
    plan["blockers"].append("VWAP_MISSING")
    assert scalp_router._plan_state(plan) == "DATA_GAP"


def test_context_unavailability_is_blocked_not_mislabeled_as_market_data_gap():
    plan = {
        "session": "REGULAR",
        "side": "LONG",
        "valid": False,
        "blockers": ["CONTEXT_DATA_MISSING_OR_STALE"],
    }
    assert scalp_router._plan_state(plan) == "BLOCKED"


def test_rest_fallback_is_a_data_gap_during_an_active_session():
    plan = {
        "session": "REGULAR",
        "side": "LONG",
        "valid": False,
        "blockers": ["REST_FALLBACK_NOT_TRADABLE"],
    }
    assert scalp_router._plan_state(plan) == "DATA_GAP"


def test_plan_live_telemetry_uses_fresh_quote_and_hides_ema_state(monkeypatch):
    now = datetime.now(timezone.utc).timestamp()
    monkeypatch.setattr(scalp_router.time, "time", lambda: now)
    plan = {
        "ticker": "AAPL", "entry": 100, "stop_loss": 99, "tp1": 101, "tp2": 102,
        "indicator_close": 100.0, "macd_hist": 0.01,
        "rsi_avg_gain_14": 0.2, "rsi_avg_loss_14": 0.1,
        "rsi_avg_gain_7": 0.2, "rsi_avg_loss_7": 0.1,
        "rsi_avg_gain_2": 0.2, "rsi_avg_loss_2": 0.1,
        "macd_fast_ema": 100.1, "macd_slow_ema": 99.9, "macd_signal_ema": 0.15,
    }
    result = scalp_router._enrich_plan(
        plan,
        {"AAPL": {"last": 100.5, "bid": 100.49, "ask": 100.51,
                  "updated_at": now - 0.2, "source_status": "LIVE"}},
    )
    assert result["live_price"] == pytest.approx(100.5)
    assert result["live_price_age_ms"] == 200
    assert result["indicator_mode"] == "PROVISIONAL_LIVE"
    assert result["live_rsi_14"] is not None
    assert result["live_macd_hist"] is not None
    assert "macd_fast_ema" not in result
    assert "rsi_avg_gain_14" not in result


def test_stale_quote_does_not_claim_provisional_live_indicators(monkeypatch):
    now = datetime.now(timezone.utc).timestamp()
    monkeypatch.setattr(scalp_router.time, "time", lambda: now)
    result = scalp_router._enrich_plan(
        {"ticker": "AAPL", "entry": 100, "stop_loss": 99, "tp1": 101, "tp2": 102},
        {"AAPL": {"last": 100.5, "updated_at": now - 10, "source_status": "LIVE"}},
    )
    assert result["indicator_mode"] == "CLOSED_1M"
    assert result["live_rsi_14"] is None
    assert result["live_macd_hist"] is None


def test_opportunity_rows_show_live_price_rsi_and_macd():
    html = (ROOT / "web" / "static" / "scalp.html").read_text(encoding="utf-8")
    assert "Live price" in html
    assert "RSI 14" in html
    assert "MACD hist" in html
    assert "Provisional RSI 14 / 7 / 2" in html
    assert "Provisional MACD hist / slope" in html
    assert "Closed 1m RSI 14 / 7 / 2" in html
    assert "Closed 5m context" in html
    assert "Closed 5m RSI / MACD slope" in html
    assert "Shadow scalp family" in html
    assert "Shadow strategy evidence" in html
    assert "p.live_rsi_14??p.rsi_14" in html
    assert "p.live_macd_hist??p.macd_hist" in html
    assert "configured_max_open_positions" in html
    assert "Closed scalp outcomes" in html
    assert "No SCALP_PLAN_V1 outcomes yet" in html


def test_bulk_history_load_uses_indexed_lateral_lookup():
    source = (ROOT / "agent" / "historical_cache.py").read_text(encoding="utf-8")
    assert "CROSS JOIN LATERAL" in source
    assert "FROM unnest(%s::text[])" in source
    assert "ROW_NUMBER() OVER (PARTITION BY ticker" not in source


def test_scalp_learning_configuration_rejects_unsafe_ordering():
    values = {
        "scalp_learn.rolling_window_min": 120,
        "scalp_learn.min_samples_to_adjust": 5,
        "scalp_learn.min_samples_to_block": 12,
        "scalp_learn.ewma_alpha": 0.25,
        "scalp_learn.negative_reduce_r": -0.05,
        "scalp_learn.negative_block_r": -0.20,
        "scalp_learn.block_win_rate": 0.40,
        "scalp_learn.confidence_win_rate": 0.48,
        "scalp_learn.base_confidence_floor": 60,
        "scalp_learn.confidence_raise_step": 10,
        "scalp_learn.size_reduce_mult": 0.5,
        "scalp_learn.action_ttl_min": 60,
    }
    _validate_scalp_learning(values)

    invalid = dict(values, **{"scalp_learn.min_samples_to_block": 2})
    with pytest.raises(ValueError, match="cannot be below"):
        _validate_scalp_learning(invalid)

    invalid = dict(values, **{"scalp_learn.size_reduce_mult": 1.2})
    with pytest.raises(ValueError, match="no greater than 1"):
        _validate_scalp_learning(invalid)


def test_scalp_runtime_rejects_truncated_session_history(monkeypatch):
    from fastapi import HTTPException
    import routers.config_router as config_router

    class Config:
        def all(self): return {}
        def set_many(self, *_args, **_kwargs): raise AssertionError("must validate first")

    import agent.config_manager as manager
    monkeypatch.setattr(manager, "config", Config())
    with pytest.raises(HTTPException, match="at least 390"):
        import asyncio
        asyncio.run(
            config_router.update_config(
                {"scalp_runtime.bar_lookback": 120},
                user=type("User", (), {"username": "test"})(),
            )
        )
