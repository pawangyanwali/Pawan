from __future__ import annotations

import json
import pytest

import time
from datetime import date

from agent.scalp.models import QuoteSource, ScalpSignalPlan, SignalSide


_LIVE_HEALTH = {
    "status": "LIVE",
    "auth_required": False,
    "token_states": {"trader": "OK", "marketdata": "OK"},
}


def _quote(*, last: float, bid: float, ask: float) -> dict:
    return {
        "last": last,
        "bid": bid,
        "ask": ask,
        "updated_at": time.time(),
        "source_status": "LIVE",
    }


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
        source=QuoteSource.WS,
        data_age_ms=100,
    )


def test_shadow_trade_is_isolated_and_resolves_at_tp2():
    from agent.scalp.shadow import (
        mark_shadow_trades,
        open_shadow_trade,
        shadow_dashboard_data,
    )

    assert open_shadow_trade(_plan(), entry_bar_id=1234, market_health=_LIVE_HEALTH) is True
    assert open_shadow_trade(_plan(), entry_bar_id=1234, market_health=_LIVE_HEALTH) is False

    opened = shadow_dashboard_data()
    assert opened["metrics"]["open_count"] == 1
    assert opened["metrics"]["closed_today"] == 0

    mark_shadow_trades(
        {"AAPL": _quote(last=102.1, bid=102.0, ask=102.2)},
        session="REGULAR",
    )
    closed = shadow_dashboard_data()
    assert closed["metrics"]["open_count"] == 0
    assert closed["metrics"]["closed_today"] == 1
    assert closed["metrics"]["wins"] == 1
    assert closed["recent_closed"][0]["exit_reason"] == "TP2"
    assert closed["recent_closed"][0]["pnl_r"] >= 1.5


def test_shadow_daily_report_persists_root_cause_summary():
    from agent.db import get_conn
    from agent.scalp.shadow_report import (
        generate_shadow_daily_report,
        latest_shadow_daily_reports,
        shadow_daily_reports_for_range,
    )
    from agent.scalp.store import init_scalp_tables

    target = date(2026, 7, 2)
    init_scalp_tables()
    plan_json = json.dumps(
        {
            "setup_type": "OVERSOLD_MACD_TURN_LONG",
            "rsi_zone": "EXTREME_OS",
            "vwap_event": "RECLAIM",
            "confidence": 82,
            "rvol": 1.7,
            "reasons": ["RSI_EXTREME", "MACD_TURN"],
        }
    )
    with get_conn() as conn:
        for offset in range(3):
            conn.execute(
                """
                INSERT INTO scalp_shadow_trades
                  (plan_id, entry_bar_id, opened_at, closed_at, ticker, side,
                   setup_type, session, status, entry_fill, current_price,
                   stop_loss, original_stop, tp1, tp2, risk_per_share, shares,
                   shares_remaining, t1_hit, t2_hit, pnl_r, pnl_dollar, mfe_r,
                   mae_r, high_watermark, low_watermark, exit_reason, plan_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    f"report-{offset}", 777 + offset,
                    f"2026-07-02T14:0{offset}:00+00:00",
                    f"2026-07-02T14:0{offset + 1}:00+00:00", "AAPL", "LONG",
                    "OVERSOLD_MACD_TURN_LONG", "STANDARD", "CLOSED", 100.0,
                    99.0, 99.0, 99.0, 101.0, 102.0, 1.0, 100, 0, 0, 0,
                    -1.0, -100.0, 0.2, -1.0, 100.2, 99.0, "STOP", plan_json,
                ),
            )

    report = generate_shadow_daily_report(target, persist=True)
    latest = latest_shadow_daily_reports(limit=1)

    assert report["market_date"] == "2026-07-02"
    assert report["status"] == "NEGATIVE"
    assert report["summary"]["closed"] == 3
    assert report["groups"]["exit_reason"][0]["exit_reason"] == "STOP"
    assert "Pre-TP1 stops" in " ".join(report["findings"])
    assert "Worst setup: OVERSOLD_MACD_TURN_LONG" in " ".join(report["findings"])
    assert latest[0]["market_date"] == "2026-07-02"

    ranged = shadow_daily_reports_for_range("2026-07-02", "2026-07-06")
    assert [row["market_date"] for row in ranged] == [
        "2026-07-06",
        "2026-07-03",
        "2026-07-02",
    ]
    assert {row["market_date"] for row in ranged}.isdisjoint({"2026-07-04", "2026-07-05"})
    assert ranged[-1]["summary"]["closed"] == 3

    with pytest.raises(ValueError, match="limited"):
        shadow_daily_reports_for_range("2026-01-01", "2026-03-01")


def test_shadow_trade_uses_executable_ask_for_short_stop():
    from agent.scalp.shadow import (
        mark_shadow_trades,
        open_shadow_trade,
        shadow_dashboard_data,
    )

    assert open_shadow_trade(
        _plan(SignalSide.SHORT),
        entry_bar_id=4321,
        market_health=_LIVE_HEALTH,
    ) is True
    mark_shadow_trades(
        {"AAPL": _quote(last=101.0, bid=100.8, ask=101.0)},
        session="REGULAR",
    )
    closed = shadow_dashboard_data()
    assert closed["recent_closed"][0]["exit_reason"] == "STOP"
    assert closed["recent_closed"][0]["pnl_r"] == pytest.approx(-1.0)


def test_short_stop_is_not_triggered_by_ask_spike_alone():
    from agent.scalp.shadow import (
        mark_shadow_trades,
        open_shadow_trade,
        shadow_dashboard_data,
    )

    assert open_shadow_trade(
        _plan(SignalSide.SHORT),
        entry_bar_id=2222,
        market_health=_LIVE_HEALTH,
    ) is True
    mark_shadow_trades(
        {"AAPL": _quote(last=100.5, bid=100.4, ask=103.0)},
        session="REGULAR",
    )
    result = shadow_dashboard_data()
    assert result["metrics"]["open_count"] == 1
    assert result["metrics"]["closed_today"] == 0


def test_shadow_daily_loss_halt_rejects_next_candidate(monkeypatch):
    from agent.config_manager import config
    from agent.db import get_conn
    from agent.scalp.execution_policy import market_day_start_utc
    from agent.scalp.shadow import open_shadow_trade, shadow_dashboard_data
    from agent.scalp.store import init_scalp_tables

    overrides = {
        "paper.enforce_risk_controls": True,
        "paper.budget": 50_000.0,
        "paper.daily_loss_halt_usd": 300.0,
        "paper.daily_loss_halt_pct": 1.0,
        "scalp_runtime.execution_policy_enabled": True,
        "scalp_runtime.require_live_execution_data": True,
    }
    for key, value in overrides.items():
        monkeypatch.setitem(config._cache, key, value)
    init_scalp_tables()
    now = market_day_start_utc().isoformat()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO scalp_shadow_trades
              (plan_id, entry_bar_id, opened_at, closed_at, ticker, side,
               status, entry_fill, current_price, stop_loss, original_stop,
               tp1, tp2, risk_per_share, shares, shares_remaining,
               pnl_r, pnl_dollar, high_watermark, low_watermark, plan_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "loss", 1, now, now, "MSFT", "LONG", "CLOSED", 100.0,
                95.0, 99.0, 99.0, 101.0, 102.0, 1.0, 100, 0,
                -3.5, -350.0, 100.0, 95.0, "{}",
            ),
        )

    assert open_shadow_trade(
        _plan(),
        entry_bar_id=3333,
        market_health=_LIVE_HEALTH,
    ) is False
    dashboard = shadow_dashboard_data()
    assert dashboard["policy"]["halted"] is True
    assert dashboard["policy"]["reason"] == "DAILY_LOSS_HALT"
    assert dashboard["metrics"]["rejection_reasons"]["DAILY_LOSS_HALT"] == 1


def test_lunch_candidate_is_observed_but_not_executed():
    from agent.scalp.shadow import open_shadow_trade, shadow_dashboard_data

    plan = _plan()
    plan.session = "LUNCH_BLOCK"
    assert open_shadow_trade(
        plan,
        entry_bar_id=4444,
        market_health=_LIVE_HEALTH,
    ) is False
    dashboard = shadow_dashboard_data()
    assert dashboard["metrics"]["candidate_count"] == 1
    assert dashboard["metrics"]["approved_count"] == 0
    assert dashboard["metrics"]["rejection_reasons"][
        "SESSION_LUNCH_BLOCK_EXECUTION_BLOCKED"
    ] == 1


def test_premarket_policy_multiplier_reduces_shadow_size():
    from agent.scalp.shadow import open_shadow_trade, shadow_dashboard_data

    plan = _plan()
    plan.session = "PRE_MARKET"
    assert open_shadow_trade(
        plan,
        entry_bar_id=5555,
        market_health=_LIVE_HEALTH,
    ) is True
    opened = shadow_dashboard_data()["open"][0]
    assert opened["policy_size_mult"] == pytest.approx(0.35)
    assert opened["shares"] == 8


def test_auth_required_rejects_new_shadow_risk():
    from agent.scalp.shadow import open_shadow_trade, shadow_dashboard_data

    health = {
        "status": "PARTIAL_LIVE",
        "auth_required": True,
        "token_states": {"trader": "AUTH_REQUIRED", "marketdata": "AUTH_REQUIRED"},
    }
    assert open_shadow_trade(
        _plan(),
        entry_bar_id=6666,
        market_health=health,
    ) is False
    dashboard = shadow_dashboard_data()
    assert dashboard["metrics"]["rejection_reasons"]["SCHWAB_AUTH_REQUIRED"] == 1


def test_runtime_reads_quotes_after_indicator_preparation():
    import inspect

    from agent.scalp.runtime import ScalpRuntime

    source = inspect.getsource(ScalpRuntime.run_cycle)
    assert source.index("prepared = list(pool.map(prepare") < source.index("quotes = get_all_prices()")
    assert source.index("quotes = get_all_prices()") < source.index("list(pool.map(analyze, prepared))")
