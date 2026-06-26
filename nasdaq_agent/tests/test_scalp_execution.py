import pytest

from agent.config_manager import config
import agent.db as db
from agent.paper_trading import get_open_trades, maybe_open_trade, rt_check_positions
from agent.scalp import (
    IndicatorSnapshot,
    QuoteSnapshot,
    QuoteSource,
    create_scalp_signal_plan,
)


@pytest.fixture(autouse=True)
def _restore_cutover_flag():
    previous = bool(config.get("scalp.execution_enabled", False))
    yield
    config.set("scalp.execution_enabled", previous, updated_by="test_cleanup")


def _valid_plan(ticker="SCALP1"):
    return create_scalp_signal_plan(
        quote=QuoteSnapshot(
            ticker=ticker,
            last=100.0,
            bid=99.99,
            ask=100.01,
            data_age_ms=100,
            source=QuoteSource.WS,
        ),
        indicators=IndicatorSnapshot(
            rsi_14=25.0,
            rsi_7=20.0,
            rsi_2=8.0,
            macd_hist=-0.04,
            macd_hist_prev=-0.10,
            atr_14=1.0,
            vwap=99.8,
            rvol=1.2,
            vwap_event="RECLAIM",
            bar_age_ms=30_000,
        ),
        side="LONG",
        session="REGULAR",
        resistances=[103.0],
    )


def test_execution_cutover_requires_a_scalp_plan():
    config.set("scalp.execution_enabled", True, updated_by="test")
    status = []

    trade_id = maybe_open_trade(
        "NOPLAN",
        "BUY",
        100.0,
        102.0,
        99.0,
        confidence=90.0,
        session="REGULAR",
        _out_status=status,
    )

    assert trade_id is None
    assert status == ["BLOCKED_SCALP_PLAN_REQUIRED"]


def test_valid_scalp_plan_is_persisted_and_linked_to_paper_trade():
    config.set_many(
        {
            "scalp.execution_enabled": True,
            "paper.min_confidence": 25.0,
            "risk.intraday_max_stop_pct": 5.0,
            "risk.intraday_max_target_pct": 5.0,
        },
        updated_by="test",
    )
    plan = _valid_plan()
    status = []

    trade_id = maybe_open_trade(
        plan.ticker,
        "BUY",
        plan.price,
        plan.tp2,
        plan.stop_loss,
        confidence=plan.confidence,
        session="REGULAR",
        atr=plan.atr_14,
        avg_daily_volume=10_000_000,
        scalp_plan=plan,
        _out_status=status,
    )

    assert trade_id is not None, status
    trade = next(row for row in get_open_trades() if row["id"] == trade_id)
    assert trade["scalp_plan_id"] == plan.plan_id
    assert trade["execution_contract"] == "SCALP_PLAN_V1"
    assert trade["t1_price"] == pytest.approx(plan.tp1)
    assert trade["t2_price"] == pytest.approx(plan.tp2)
    assert trade["entry_ideal_price"] == pytest.approx(plan.entry)
    assert trade["entry_price"] >= plan.ask
    assert status == ["EXECUTED_PAPER"]

    with db.get_conn(read_only=True) as conn:
        saved = conn.execute(
            "SELECT ticker, valid FROM scalp_signal_plans WHERE plan_id=?",
            (plan.plan_id,),
        ).fetchone()
        decision = conn.execute(
            "SELECT decision, trade_id FROM scalp_execution_decisions WHERE plan_id=?",
            (plan.plan_id,),
        ).fetchone()

    assert saved["ticker"] == plan.ticker
    assert saved["valid"] == 1
    assert decision["decision"] == "EXECUTED_PAPER"
    assert decision["trade_id"] == trade_id


def test_invalid_plan_cannot_open_even_with_high_confidence():
    config.set("scalp.execution_enabled", True, updated_by="test")
    plan = _valid_plan("BLOCKED1")
    plan.valid = False
    plan.invalid_reason = "QUOTE_STALE"
    plan.blockers.append("QUOTE_STALE")
    status = []

    trade_id = maybe_open_trade(
        plan.ticker,
        "BUY",
        plan.price,
        plan.tp2,
        plan.stop_loss,
        confidence=99.0,
        session="REGULAR",
        scalp_plan=plan,
        _out_status=status,
    )

    assert trade_id is None
    assert status == ["BLOCKED_INVALID_SCALP_PLAN"]

    with db.get_conn(read_only=True) as conn:
        saved = conn.execute(
            "SELECT valid, invalid_reason FROM scalp_signal_plans WHERE plan_id=?",
            (plan.plan_id,),
        ).fetchone()
        decision = conn.execute(
            "SELECT decision, reason FROM scalp_execution_decisions WHERE plan_id=?",
            (plan.plan_id,),
        ).fetchone()

    assert saved["valid"] == 0
    assert saved["invalid_reason"] == "QUOTE_STALE"
    assert decision["decision"] == "BLOCKED"
    assert decision["reason"] == "BLOCKED_INVALID_SCALP_PLAN"


def test_realtime_close_immediately_records_scalp_learning_outcome():
    config.set_many(
        {
            "scalp.execution_enabled": True,
            "paper.min_confidence": 25.0,
            "risk.intraday_max_stop_pct": 5.0,
            "risk.intraday_max_target_pct": 5.0,
        },
        updated_by="test",
    )
    plan = _valid_plan("CLOSE1")
    trade_id = maybe_open_trade(
        plan.ticker,
        "BUY",
        plan.price,
        plan.tp2,
        plan.stop_loss,
        confidence=plan.confidence,
        session="REGULAR",
        atr=plan.atr_14,
        avg_daily_volume=10_000_000,
        scalp_plan=plan,
    )

    assert trade_id is not None
    actions = rt_check_positions(plan.ticker, plan.stop_loss - 0.01)

    assert "STOP_HIT" in actions
    with db.get_conn(read_only=True) as conn:
        outcome = conn.execute(
            "SELECT trade_id, pnl_r, context_key FROM scalp_trade_outcomes WHERE trade_id=?",
            (trade_id,),
        ).fetchone()
        stats = conn.execute(
            "SELECT sample_count FROM scalp_context_stats WHERE context_key=?",
            (outcome["context_key"],),
        ).fetchone()

    assert outcome["trade_id"] == trade_id
    assert outcome["pnl_r"] < 0
    assert stats["sample_count"] == 1
