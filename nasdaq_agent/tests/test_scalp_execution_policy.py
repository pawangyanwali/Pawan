from __future__ import annotations

from datetime import datetime, timezone

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
