from __future__ import annotations

import time

import pytest

from agent.scalp.models import QuoteSource, ScalpSignalPlan, SignalSide


def _plan(
    *,
    ticker: str = "AAPL",
    side: SignalSide = SignalSide.LONG,
) -> ScalpSignalPlan:
    long = side is SignalSide.LONG
    return ScalpSignalPlan(
        ticker=ticker,
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
        setup_type="MOMENTUM_PULLBACK",
        strategy_family="MOMENTUM_PULLBACK",
        session="REGULAR",
        confidence=80.0,
        source=QuoteSource.WS,
        data_age_ms=100,
    )


def _quote(
    *,
    last: float,
    bid: float,
    ask: float,
    source: str = "LIVE",
) -> dict:
    return {
        "last": last,
        "bid": bid,
        "ask": ask,
        "updated_at": time.time(),
        "source_status": source,
    }


def test_candidate_registration_is_deduplicated_and_observational():
    from agent.db import get_conn
    from agent.scalp.candidate_tracker import (
        CANONICAL_CANDIDATE,
        register_candidate,
    )

    plan = _plan()
    assert register_candidate(
        plan,
        candidate_type=CANONICAL_CANDIDATE,
        entry_bar_id=1234,
    )
    assert not register_candidate(
        plan,
        candidate_type=CANONICAL_CANDIDATE,
        entry_bar_id=1234,
    )
    assert not register_candidate(
        plan,
        candidate_type=CANONICAL_CANDIDATE,
        entry_bar_id=1235,
    )

    with get_conn(read_only=True) as conn:
        candidates = conn.execute(
            "SELECT COUNT(*) AS n FROM scalp_candidate_trials"
        ).fetchone()["n"]
        learned = conn.execute(
            "SELECT COUNT(*) AS n FROM scalp_trade_outcomes"
        ).fetchone()["n"]
    assert candidates == 1
    assert learned == 0


def test_long_candidate_resolves_at_tp2_without_learning_contamination():
    from agent.db import get_conn
    from agent.scalp.candidate_tracker import (
        CANONICAL_CANDIDATE,
        candidate_dashboard_data,
        mark_candidate_trials,
        register_candidate,
    )

    register_candidate(
        _plan(),
        candidate_type=CANONICAL_CANDIDATE,
        entry_bar_id=2001,
    )
    assert mark_candidate_trials(
        {"AAPL": _quote(last=102.1, bid=102.0, ask=102.2)},
        session="REGULAR",
    ) == 1

    dashboard = candidate_dashboard_data()
    row = dashboard["recent"][0]
    assert row["status"] == "CLOSED"
    assert row["exit_reason"] == "TP2"
    assert row["t1_hit"] == 1
    assert row["t2_hit"] == 1
    assert row["pnl_r"] == pytest.approx(1.5)
    assert dashboard["metrics"]["closed"] == 1
    assert dashboard["metrics"]["by_type"][CANONICAL_CANDIDATE][
        "expectancy_r"
    ] == pytest.approx(1.5)
    with get_conn(read_only=True) as conn:
        learned = conn.execute(
            "SELECT COUNT(*) AS n FROM scalp_trade_outcomes"
        ).fetchone()["n"]
    assert learned == 0


def test_candidate_batch_registers_multiple_tickers_in_one_scan():
    from agent.db import get_conn
    from agent.scalp.candidate_tracker import (
        CANONICAL_CANDIDATE,
        MTF_CANDIDATE,
        register_candidate_batch,
    )

    attempted = register_candidate_batch([
        (CANONICAL_CANDIDATE, _plan(ticker="AAPL"), 2101, {}),
        (
            MTF_CANDIDATE,
            _plan(ticker="MSFT", side=SignalSide.SHORT),
            2101,
            {"observation_only": True},
        ),
    ])
    assert attempted == 2
    with get_conn(read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT ticker, admission_state FROM scalp_candidate_trials
            ORDER BY ticker
            """
        ).fetchall()
    assert [(row["ticker"], row["admission_state"]) for row in rows] == [
        ("AAPL", "OBSERVED_BEFORE_CONFIRMATION"),
        ("MSFT", "MTF_OBSERVATION_ONLY"),
    ]


def test_resolved_candidate_requires_independent_episode_cooldown():
    from datetime import datetime, timedelta, timezone

    from agent.db import get_conn
    from agent.scalp.candidate_tracker import (
        CANONICAL_CANDIDATE,
        mark_candidate_trials,
        register_candidate,
    )

    assert register_candidate(
        _plan(),
        candidate_type=CANONICAL_CANDIDATE,
        entry_bar_id=2201,
    )
    assert mark_candidate_trials(
        {"AAPL": _quote(last=98.9, bid=98.8, ask=99.0)},
        session="REGULAR",
    ) == 1
    assert not register_candidate(
        _plan(),
        candidate_type=CANONICAL_CANDIDATE,
        entry_bar_id=2202,
    )

    old = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    with get_conn() as conn:
        conn.execute(
            "UPDATE scalp_candidate_trials SET resolved_at=? WHERE ticker='AAPL'",
            (old,),
        )
    assert register_candidate(
        _plan(),
        candidate_type=CANONICAL_CANDIDATE,
        entry_bar_id=2203,
    )
    with get_conn(read_only=True) as conn:
        versions = conn.execute(
            "SELECT episode_version FROM scalp_candidate_trials ORDER BY id"
        ).fetchall()
    assert [row["episode_version"] for row in versions] == [2, 2]


def test_short_candidate_uses_ask_for_stop_resolution():
    from agent.scalp.candidate_tracker import (
        MTF_CANDIDATE,
        candidate_dashboard_data,
        mark_candidate_trials,
        register_candidate,
    )

    register_candidate(
        _plan(ticker="MSFT", side=SignalSide.SHORT),
        candidate_type=MTF_CANDIDATE,
        entry_bar_id=2002,
    )
    mark_candidate_trials(
        {"MSFT": _quote(last=101.05, bid=100.95, ask=101.1)},
        session="REGULAR",
    )

    row = candidate_dashboard_data()["recent"][0]
    assert row["status"] == "CLOSED"
    assert row["exit_reason"] == "STOP"
    assert row["exit_price"] == pytest.approx(101.1)
    assert row["pnl_r"] == pytest.approx(-1.1)


def test_candidate_admission_records_confirmation_and_policy_disposition():
    from agent.scalp.candidate_tracker import (
        CANONICAL_CANDIDATE,
        candidate_dashboard_data,
        register_candidate,
        update_candidate_admission,
    )

    register_candidate(
        _plan(),
        candidate_type=CANONICAL_CANDIDATE,
        entry_bar_id=3001,
        admission_state="OBSERVED_BEFORE_CONFIRMATION",
    )
    assert update_candidate_admission(
        ticker="AAPL",
        entry_bar_id=3002,
        admission_state="SHADOW_REJECTED",
        admission_reason="DAILY_LOSS_HALT",
    ) == 1

    dashboard = candidate_dashboard_data()
    row = dashboard["recent"][0]
    assert row["admission_state"] == "SHADOW_REJECTED"
    assert row["admission_reason"] == "DAILY_LOSS_HALT"
    assert dashboard["metrics"]["admissions"]["SHADOW_REJECTED"] == 1


def test_stale_candidate_quote_is_not_marked():
    from agent.scalp.candidate_tracker import (
        MTF_CANDIDATE,
        candidate_dashboard_data,
        mark_candidate_trials,
        register_candidate,
    )

    register_candidate(
        _plan(),
        candidate_type=MTF_CANDIDATE,
        entry_bar_id=4001,
    )
    stale = _quote(last=102.1, bid=102.0, ask=102.2)
    stale["updated_at"] = time.time() - 30
    assert mark_candidate_trials({"AAPL": stale}, session="REGULAR") == 0
    row = candidate_dashboard_data()["recent"][0]
    assert row["status"] == "OPEN"
    assert row["current_price"] == pytest.approx(100.0)
