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


def test_hard_learning_block_is_not_presented_as_watch():
    assert scalp_router._plan_state(
        {
            "side": "LONG",
            "valid": False,
            "blockers": ["LEARNING_CONTEXT_BLOCK"],
        }
    ) == "BLOCKED"


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
