from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from agent.scalp.models import QuoteSource, ScalpSignalPlan, SignalSide


def _plan() -> ScalpSignalPlan:
    return ScalpSignalPlan(
        ticker="AAPL",
        side=SignalSide.LONG,
        valid=True,
        invalid_reason="",
        entry=100.0,
        stop_loss=99.0,
        tp1=101.0,
        tp2=102.0,
        risk_per_share=1.0,
        reward_r=2.0,
        rr_ratio=2.0,
        setup_type="OVERSOLD_MACD_TURN_LONG",
        session="PRIME",
        confidence=82.0,
        source=QuoteSource.WS,
        data_age_ms=100,
    )


def _health() -> dict:
    return {
        "status": "LIVE",
        "auth_required": False,
        "token_states": {"trader": "OK", "marketdata": "OK"},
    }


def test_market_day_boundary_is_new_york_not_utc():
    from agent.scalp.execution_policy import market_day_start_utc

    # 01:00 UTC on June 30 is still June 29 in New York.
    observed = market_day_start_utc(
        datetime(2026, 6, 30, 1, 0, tzinfo=timezone.utc)
    )
    assert observed == datetime(2026, 6, 29, 4, 0, tzinfo=timezone.utc)


def test_canonical_paper_preflight_obeys_daily_loss_halt(monkeypatch):
    from agent.config_manager import config
    from agent.db import get_conn
    from agent.paper_trading import init_db
    from agent.scalp.execution_policy import (
        evaluate_execution_policy,
        market_day_start_utc,
    )

    overrides = {
        "paper.enforce_risk_controls": True,
        "paper.budget": 50_000.0,
        "paper.daily_loss_halt_usd": 300.0,
        "paper.daily_loss_halt_pct": 1.0,
        "paper.max_daily_trades": 75,
        "paper.max_open_trades": 10,
        "risk.max_concurrent_trades": 10,
        "scalp_runtime.execution_policy_enabled": True,
        "scalp_runtime.require_live_execution_data": True,
    }
    for key, value in overrides.items():
        monkeypatch.setitem(config._cache, key, value)

    init_db()
    opened_at = market_day_start_utc().isoformat()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO paper_trades
              (opened_at, closed_at, ticker, direction, entry_price, target,
               stop, confidence, status, pnl_dollar, shares, shares_remaining)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                opened_at, opened_at, "MSFT", "BUY", 100.0, 102.0, 99.0,
                80.0, "CLOSED", -350.0, 100, 0,
            ),
        )

    decision = evaluate_execution_policy(
        _plan(),
        mode="PAPER",
        market_health=_health(),
    )
    assert decision.allowed is False
    assert decision.reason == "DAILY_LOSS_HALT"


def test_same_context_cluster_throttle_blocks_stampede(monkeypatch):
    from agent.config_manager import config
    from agent.db import get_conn
    from agent.scalp.execution_policy import evaluate_execution_policy
    from agent.scalp.store import init_scalp_tables

    monkeypatch.setitem(config._cache, "paper.enforce_risk_controls", False)
    monkeypatch.setitem(config._cache, "scalp_runtime.execution_policy_enabled", True)
    monkeypatch.setitem(config._cache, "scalp_runtime.require_live_execution_data", True)
    monkeypatch.setitem(
        config._cache, "scalp_runtime.context_cluster_throttle_enabled", True
    )
    monkeypatch.setitem(config._cache, "scalp_runtime.context_cluster_window_min", 10)
    monkeypatch.setitem(config._cache, "scalp_runtime.context_cluster_max_entries", 3)

    plan = _plan()
    payload = json.dumps(plan.to_dict(), separators=(",", ":"))
    now = datetime.now(timezone.utc)
    init_scalp_tables()
    with get_conn() as conn:
        for offset in range(3):
            conn.execute(
                """
                INSERT INTO scalp_shadow_trades
                  (plan_id, entry_bar_id, opened_at, closed_at, ticker, side,
                   setup_type, session, status, entry_fill, current_price,
                   stop_loss, original_stop, tp1, tp2, risk_per_share, shares,
                   shares_remaining, high_watermark, low_watermark, plan_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    f"cluster-{offset}", 10_000 + offset,
                    (now - timedelta(minutes=offset)).isoformat(),
                    (now - timedelta(minutes=offset, seconds=-30)).isoformat(),
                    f"SYM{offset}", plan.side.value, plan.setup_type,
                    plan.session, "CLOSED", plan.entry, plan.entry,
                    plan.stop_loss, plan.stop_loss, plan.tp1, plan.tp2,
                    plan.risk_per_share, 100, 0, plan.entry, plan.entry,
                    payload,
                ),
            )

    decision = evaluate_execution_policy(
        plan,
        mode="SHADOW",
        market_health=_health(),
    )

    assert decision.allowed is False
    assert decision.reason == "CONTEXT_CLUSTER_THROTTLE"
    assert decision.checks["context_cluster"]["entry_count"] == 3
