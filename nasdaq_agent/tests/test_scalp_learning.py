from datetime import datetime, timedelta, timezone

import pytest

import agent.db as db
from agent.config_manager import config
from agent.scalp import (
    IndicatorSnapshot,
    QuoteSnapshot,
    QuoteSource,
    create_scalp_signal_plan,
)
from agent.scalp.learning import (
    ALLOW,
    BLOCK,
    SIZE_REDUCE,
    apply_context_gate,
    context_key_for_plan,
    get_context_gate,
    record_closed_trade,
)
from agent.scalp.store import init_scalp_tables, save_plan


def _plan(ticker: str = "LEARN1"):
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


def _closed_trade(conn, plan, trade_id: int, pnl_r: float) -> None:
    save_plan(plan)
    risk_capital = plan.risk_per_share * 100
    pnl_dollar = risk_capital * pnl_r
    conn.execute(
        """
        INSERT INTO paper_trades
          (id, opened_at, closed_at, ticker, direction, entry_price, target,
           stop, confidence, status, exit_price, exit_reason, pnl_dollar,
           shares, session, t1_hit, mfe_r, mae_r, scalp_plan_id,
           execution_contract)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            trade_id,
            (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
            datetime.now(timezone.utc).isoformat(),
            plan.ticker,
            "BUY",
            plan.entry,
            plan.tp2,
            plan.stop_loss,
            plan.confidence,
            "CLOSED",
            plan.stop_loss if pnl_r < 0 else plan.tp1,
            "STOP_HIT" if pnl_r < 0 else "TARGET_T1",
            pnl_dollar,
            100,
            "REGULAR",
            int(pnl_r > 0),
            max(0.0, pnl_r),
            min(0.0, pnl_r),
            plan.plan_id,
            "SCALP_PLAN_V1",
        ),
    )


@pytest.fixture(autouse=True)
def _learning_defaults():
    config.set_many(
        {
            "scalp_learn.enabled": True,
            "scalp_learn.rolling_window_min": 120,
            "scalp_learn.min_samples_to_adjust": 2,
            "scalp_learn.min_samples_to_block": 4,
            "scalp_learn.ewma_alpha": 0.5,
            "scalp_learn.negative_reduce_r": -0.05,
            "scalp_learn.negative_block_r": -0.20,
            "scalp_learn.block_win_rate": 0.40,
            "scalp_learn.confidence_win_rate": 0.48,
            "scalp_learn.base_confidence_floor": 60,
            "scalp_learn.confidence_raise_step": 10,
            "scalp_learn.size_reduce_mult": 0.5,
            "scalp_learn.action_ttl_min": 60,
        },
        updated_by="test",
    )
    yield


def test_context_key_is_stable_and_excludes_ticker():
    left = _plan("AAA")
    right = _plan("BBB")

    assert context_key_for_plan(left) == context_key_for_plan(right)
    assert "REGULAR" in context_key_for_plan(left)
    assert "RECLAIM" in context_key_for_plan(left)


def test_closed_trade_is_idempotent_and_updates_context_gate():
    init_scalp_tables()
    plans = [_plan(f"LOSS{i}") for i in range(4)]
    with db.get_conn() as conn:
        for trade_id, plan in enumerate(plans, start=1):
            _closed_trade(conn, plan, trade_id, -1.0)
            assert record_closed_trade(conn, trade_id) is not None
        assert record_closed_trade(conn, 4) is not None

        count = conn.execute(
            "SELECT COUNT(*) AS n FROM scalp_trade_outcomes"
        ).fetchone()["n"]
        stats = conn.execute(
            "SELECT * FROM scalp_context_stats WHERE context_key=?",
            (context_key_for_plan(plans[0]),),
        ).fetchone()

    assert count == 4
    assert stats["sample_count"] == 4
    assert stats["gate_state"] == BLOCK
    assert stats["ewma_expectancy_r"] == pytest.approx(-1.0)


def test_learning_gate_tightens_size_without_rewriting_bracket():
    init_scalp_tables()
    plan = _plan("SIZE1")
    original = (plan.entry, plan.stop_loss, plan.tp1, plan.tp2)
    context_key = context_key_for_plan(plan)
    now = datetime.now(timezone.utc)
    with db.get_conn() as conn:
        conn.execute(
            """
            INSERT INTO scalp_context_stats
              (context_key, updated_at, sample_count, wins, losses,
               posterior_win_rate, ewma_expectancy_r, mean_expectancy_r,
               gate_state, confidence_floor, size_mult, expires_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                context_key, now.isoformat(), 6, 3, 3, 0.5, -0.1, -0.1,
                SIZE_REDUCE, 0.0, 0.5,
                (now + timedelta(minutes=30)).isoformat(),
            ),
        )

    apply_context_gate(plan)

    assert plan.learning_gate == SIZE_REDUCE
    assert plan.learning_size_mult == pytest.approx(0.5)
    assert (plan.entry, plan.stop_loss, plan.tp1, plan.tp2) == original
    assert plan.valid


def test_expired_learning_action_fails_open():
    init_scalp_tables()
    plan = _plan("EXPIRED1")
    context_key = context_key_for_plan(plan)
    now = datetime.now(timezone.utc)
    with db.get_conn() as conn:
        conn.execute(
            """
            INSERT INTO scalp_context_stats
              (context_key, updated_at, sample_count, wins, losses,
               posterior_win_rate, ewma_expectancy_r, mean_expectancy_r,
               gate_state, confidence_floor, size_mult, expires_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                context_key, now.isoformat(), 20, 2, 18, 0.14, -0.8, -0.8,
                BLOCK, 0.0, 1.0,
                (now - timedelta(seconds=1)).isoformat(),
            ),
        )

    gate = get_context_gate(context_key)
    apply_context_gate(plan)

    assert gate["gate_state"] == ALLOW
    assert plan.learning_gate == ALLOW
    assert plan.valid
