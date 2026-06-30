from __future__ import annotations

import pytest

from agent.scalp.models import ScalpSignalPlan, SignalSide


def _plan(side: SignalSide = SignalSide.LONG) -> ScalpSignalPlan:
    long = side is SignalSide.LONG
    return ScalpSignalPlan(
        ticker="AAPL",
        side=side,
        valid=True,
        invalid_reason="",
        entry=100.0,
        stop_loss=99.0 if long else 101.0,
        tp1=101.0 if long else 99.0,
        tp2=102.0 if long else 98.0,
        risk_per_share=1.0,
        reward_r=2.0,
        rr_ratio=2.0,
        setup_type="REVERSAL",
        session="REGULAR",
        confidence=80.0,
    )


def test_shadow_trade_is_isolated_and_resolves_at_tp2():
    from agent.scalp.shadow import (
        mark_shadow_trades,
        open_shadow_trade,
        shadow_dashboard_data,
    )

    assert open_shadow_trade(_plan(), entry_bar_id=1234) is True
    assert open_shadow_trade(_plan(), entry_bar_id=1234) is False

    opened = shadow_dashboard_data()
    assert opened["metrics"]["open_count"] == 1
    assert opened["metrics"]["closed_today"] == 0

    mark_shadow_trades(
        {"AAPL": {"last": 102.1, "bid": 102.0, "ask": 102.2}},
        session="REGULAR",
    )
    closed = shadow_dashboard_data()
    assert closed["metrics"]["open_count"] == 0
    assert closed["metrics"]["closed_today"] == 1
    assert closed["metrics"]["wins"] == 1
    assert closed["recent_closed"][0]["exit_reason"] == "TP2"
    assert closed["recent_closed"][0]["pnl_r"] >= 1.5


def test_shadow_trade_uses_executable_ask_for_short_stop():
    from agent.scalp.shadow import (
        mark_shadow_trades,
        open_shadow_trade,
        shadow_dashboard_data,
    )

    assert open_shadow_trade(_plan(SignalSide.SHORT), entry_bar_id=4321) is True
    mark_shadow_trades(
        {"AAPL": {"last": 100.9, "bid": 100.8, "ask": 101.0}},
        session="REGULAR",
    )
    closed = shadow_dashboard_data()
    assert closed["recent_closed"][0]["exit_reason"] == "STOP"
    assert closed["recent_closed"][0]["pnl_r"] == pytest.approx(-1.0)


def test_runtime_reads_quotes_after_indicator_preparation():
    import inspect

    from agent.scalp.runtime import ScalpRuntime

    source = inspect.getsource(ScalpRuntime.run_cycle)
    assert source.index("prepared = list(pool.map(prepare") < source.index("quotes = get_all_prices()")
    assert source.index("quotes = get_all_prices()") < source.index("list(pool.map(analyze, prepared))")
