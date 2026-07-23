"""Bounded, auditable outcome learning for the greenfield scalp engine."""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from agent.db import get_conn

from .models import ScalpSignalPlan
from .store import init_scalp_tables

logger = logging.getLogger(__name__)

ALLOW = "ALLOW"
SIZE_REDUCE = "SIZE_REDUCE"
CONFIDENCE_RAISE = "CONFIDENCE_RAISE"
BLOCK = "BLOCK"

_gate_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_cache_lock = threading.Lock()
_CACHE_TTL_S = 2.0


def context_key_for_plan(plan: ScalpSignalPlan | dict[str, Any]) -> str:
    read = (
        plan.get
        if isinstance(plan, dict)
        else lambda key, default="": getattr(plan, key, default)
    )
    parts = (
        read("setup_type", "UNKNOWN") or "UNKNOWN",
        _side_value(read("side", "NONE")),
        read("session", "UNKNOWN") or "UNKNOWN",
        read("rsi_zone", "UNKNOWN") or "UNKNOWN",
        _macd_state(read("macd_slope", 0.0)),
        read("vwap_event", "UNKNOWN") or "UNKNOWN",
        _spread_bucket(read("spread_to_risk", 0.0)),
        read("atr_bucket", "UNKNOWN") or "UNKNOWN",
    )
    return "|".join(str(part).upper().replace("|", "_") for part in parts)


def setup_session_key_for_plan(plan: ScalpSignalPlan | dict[str, Any]) -> str:
    """Broader adaptive key: setup family, side, and session only."""
    read = (
        plan.get
        if isinstance(plan, dict)
        else lambda key, default="": getattr(plan, key, default)
    )
    parts = (
        "SETUP_SESSION",
        read("setup_type", "UNKNOWN") or "UNKNOWN",
        _side_value(read("side", "NONE")),
        read("session", "UNKNOWN") or "UNKNOWN",
    )
    return "|".join(str(part).upper().replace("|", "_") for part in parts)


def apply_context_gate(plan: ScalpSignalPlan) -> ScalpSignalPlan:
    """Apply only expiring risk-tightening actions; bracket levels never change."""
    from agent.config_manager import config

    plan.context_key = context_key_for_plan(plan)
    if not bool(config.get("scalp_learn.enabled", True)):
        return plan
    gate = get_context_gate(plan.context_key)
    plan.learning_gate = gate["gate_state"]
    plan.learned_expectancy_r = float(gate.get("ewma_expectancy_r") or 0.0)
    plan.learned_win_rate = float(gate.get("posterior_win_rate") or 0.0)
    plan.learning_size_mult = min(
        1.0, max(0.05, float(gate.get("size_mult") or 1.0))
    )
    plan.learning_confidence_floor = float(gate.get("confidence_floor") or 0.0)
    plan.learning_action_expires_at = str(gate.get("expires_at") or "")

    if plan.learning_gate == BLOCK:
        _block_plan(plan, "LEARNING_CONTEXT_BLOCK")
    elif (
        plan.learning_gate == CONFIDENCE_RAISE
        and plan.confidence < plan.learning_confidence_floor
    ):
        _block_plan(plan, "LEARNING_CONFIDENCE_FLOOR")
    elif plan.learning_gate == SIZE_REDUCE:
        plan.reasons.append("LEARNING_SIZE_REDUCED")

    if bool(config.get("scalp_learn.setup_session_gate_enabled", True)):
        setup_gate = get_context_gate(setup_session_key_for_plan(plan))
        setup_state = str(setup_gate.get("gate_state") or ALLOW)
        setup_floor = float(setup_gate.get("confidence_floor") or 0.0)
        setup_size_mult = min(
            1.0, max(0.05, float(setup_gate.get("size_mult") or 1.0))
        )
        if setup_state == BLOCK:
            plan.learning_gate = BLOCK
            plan.learned_expectancy_r = min(
                plan.learned_expectancy_r,
                float(setup_gate.get("ewma_expectancy_r") or 0.0),
            )
            _block_plan(plan, "LEARNING_SETUP_SESSION_BLOCK")
        elif setup_state == CONFIDENCE_RAISE and plan.confidence < setup_floor:
            plan.learning_gate = CONFIDENCE_RAISE
            plan.learning_confidence_floor = max(
                plan.learning_confidence_floor, setup_floor
            )
            _block_plan(plan, "LEARNING_SETUP_SESSION_CONFIDENCE_FLOOR")
        elif setup_state == SIZE_REDUCE:
            plan.learning_gate = SIZE_REDUCE
            plan.learning_size_mult = min(plan.learning_size_mult, setup_size_mult)
            plan.learned_expectancy_r = min(
                plan.learned_expectancy_r,
                float(setup_gate.get("ewma_expectancy_r") or 0.0),
            )
            plan.reasons.append("LEARNING_SETUP_SESSION_SIZE_REDUCED")
    return plan


def record_closed_trade(conn, trade_id: int) -> dict[str, Any] | None:
    """Persist one idempotent scalp outcome and synchronously refresh its gate."""
    from agent.config_manager import config

    init_scalp_tables()
    row = conn.execute(
        """
        SELECT id, scalp_plan_id, execution_contract, closed_at, ticker,
               direction, entry_price, exit_price, shares, t1_hit,
               pnl_dollar, mfe_r, mae_r, exit_reason, session
        FROM paper_trades WHERE id=? AND status='CLOSED'
        """,
        (trade_id,),
    ).fetchone()
    if (
        not row
        or row["execution_contract"] != "SCALP_PLAN_V1"
        or not row["scalp_plan_id"]
    ):
        return None
    plan_row = conn.execute(
        "SELECT plan_json FROM scalp_signal_plans WHERE plan_id=?",
        (row["scalp_plan_id"],),
    ).fetchone()
    if not plan_row:
        logger.warning("[ScalpLearning] plan missing for closed trade %s", trade_id)
        return None
    plan_json = str(plan_row["plan_json"] or "")
    plan = json.loads(plan_json)
    context_key = context_key_for_plan({**plan, "session": row["session"]})
    shares = max(1, int(row["shares"] or 1))
    risk_capital = abs(float(plan.get("risk_per_share") or 0.0)) * shares
    pnl_dollar = float(row["pnl_dollar"] or 0.0)
    pnl_r = pnl_dollar / risk_capital if risk_capital > 0 else 0.0
    reason = str(row["exit_reason"] or "")
    outcome = {
        "plan_id": row["scalp_plan_id"],
        "trade_id": int(row["id"]),
        "closed_at": str(row["closed_at"]),
        "ticker": str(row["ticker"]),
        "side": _side_value(plan.get("side") or row["direction"]),
        "context_key": context_key,
        "setup_type": str(plan.get("setup_type") or ""),
        "session": str(row["session"] or ""),
        "rsi_zone": str(plan.get("rsi_zone") or ""),
        "macd_state": _macd_state(plan.get("macd_slope", 0.0)),
        "vwap_event": str(plan.get("vwap_event") or ""),
        "spread_bucket": _spread_bucket(plan.get("spread_to_risk", 0.0)),
        "atr_bucket": str(plan.get("atr_bucket") or "UNKNOWN"),
        "entry_fill": float(row["entry_price"] or 0.0),
        "exit_fill": float(row["exit_price"] or 0.0),
        "tp1_hit": int(bool(row["t1_hit"])),
        "tp2_hit": int(reason == "TARGET_T2"),
        "stop_hit": int("STOP" in reason),
        "time_stop": int("TIME" in reason or "MAX_BARS" in reason),
        "pnl_r": round(pnl_r, 6),
        "pnl_dollar": round(pnl_dollar, 2),
        "mfe_r": float(row["mfe_r"] or 0.0),
        "mae_r": float(row["mae_r"] or 0.0),
        "exit_reason": reason,
        "plan_json": plan_json,
    }
    keys = (
        "plan_id", "trade_id", "closed_at", "ticker", "side", "context_key",
        "setup_type", "session", "rsi_zone", "macd_state", "vwap_event",
        "spread_bucket", "atr_bucket", "entry_fill", "exit_fill", "tp1_hit", "tp2_hit",
        "stop_hit", "time_stop", "pnl_r", "pnl_dollar", "mfe_r", "mae_r",
        "exit_reason", "plan_json",
    )
    existing = conn.execute(
        "SELECT id FROM scalp_trade_outcomes WHERE trade_id=?", (trade_id,)
    ).fetchone()
    if existing:
        conn.execute(
            """
            UPDATE scalp_trade_outcomes
            SET pnl_r=?, pnl_dollar=?, mfe_r=?, mae_r=?, exit_reason=?,
                plan_json=?
            WHERE trade_id=?
            """,
            (
                outcome["pnl_r"], outcome["pnl_dollar"], outcome["mfe_r"],
                outcome["mae_r"], outcome["exit_reason"], outcome["plan_json"],
                trade_id,
            ),
        )
    else:
        conn.execute(
            """
            INSERT INTO scalp_trade_outcomes
              (plan_id, trade_id, closed_at, ticker, side, context_key,
               setup_type, session, rsi_zone, macd_state, vwap_event,
               spread_bucket, atr_bucket, entry_fill, exit_fill, tp1_hit, tp2_hit,
               stop_hit, time_stop, pnl_r, pnl_dollar, mfe_r, mae_r,
               exit_reason, plan_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            tuple(outcome[key] for key in keys),
        )
    _refresh_context(conn, context_key)
    if bool(config.get("scalp_learn.setup_session_gate_enabled", True)):
        setup_key = setup_session_key_for_plan({**plan, "session": row["session"]})
        _refresh_setup_session_context(conn, setup_key)
        _invalidate_gate(setup_key)
    _invalidate_gate(context_key)
    return outcome


def record_closed_shadow_trade(conn, shadow_trade_id: int) -> dict[str, Any] | None:
    """Persist one closed shadow trade as an immediate learning outcome.

    Shadow outcomes remain isolated from account P&L and paper_trades.  They use
    negative trade IDs in scalp_trade_outcomes so they are auditable, idempotent,
    and cannot collide with real paper trade IDs.
    """
    from agent.config_manager import config

    if not bool(config.get("scalp_learn.enabled", True)):
        return None
    if not bool(config.get("scalp_learn.shadow_outcomes_enabled", True)):
        return None
    init_scalp_tables()
    row = conn.execute(
        """
        SELECT * FROM scalp_shadow_trades
        WHERE id=? AND status='CLOSED'
        """,
        (shadow_trade_id,),
    ).fetchone()
    if not row:
        return None
    plan_json = str(row["plan_json"] or "{}")
    plan = json.loads(plan_json)
    context_key = context_key_for_plan({**plan, "session": row["session"]})
    exit_reason = str(row.get("exit_reason") or "")
    shadow_outcome_id = -abs(int(row["id"]))
    outcome = {
        "plan_id": str(row.get("plan_id") or plan.get("plan_id") or ""),
        "trade_id": shadow_outcome_id,
        "closed_at": str(row.get("closed_at") or datetime.now(timezone.utc).isoformat()),
        "ticker": str(row.get("ticker") or ""),
        "side": _side_value(plan.get("side") or row.get("side")),
        "context_key": context_key,
        "setup_type": str(row.get("setup_type") or plan.get("setup_type") or ""),
        "session": str(row.get("session") or plan.get("session") or ""),
        "rsi_zone": str(plan.get("rsi_zone") or ""),
        "macd_state": _macd_state(plan.get("macd_slope", 0.0)),
        "vwap_event": str(plan.get("vwap_event") or ""),
        "spread_bucket": _spread_bucket(plan.get("spread_to_risk", 0.0)),
        "atr_bucket": str(plan.get("atr_bucket") or "UNKNOWN"),
        "entry_fill": float(row.get("entry_fill") or 0.0),
        "exit_fill": float(row.get("exit_fill") or row.get("current_price") or 0.0),
        "tp1_hit": int(bool(row.get("t1_hit"))),
        "tp2_hit": int(bool(row.get("t2_hit"))),
        "stop_hit": int("STOP" in exit_reason),
        "time_stop": int("TIME" in exit_reason or "MAX_BARS" in exit_reason),
        "pnl_r": round(float(row.get("pnl_r") or 0.0), 6),
        "pnl_dollar": round(float(row.get("pnl_dollar") or 0.0), 2),
        "mfe_r": float(row.get("mfe_r") or 0.0),
        "mae_r": float(row.get("mae_r") or 0.0),
        "exit_reason": f"SHADOW_{exit_reason or 'CLOSED'}",
        "plan_json": plan_json,
    }
    keys = (
        "plan_id", "trade_id", "closed_at", "ticker", "side", "context_key",
        "setup_type", "session", "rsi_zone", "macd_state", "vwap_event",
        "spread_bucket", "atr_bucket", "entry_fill", "exit_fill", "tp1_hit", "tp2_hit",
        "stop_hit", "time_stop", "pnl_r", "pnl_dollar", "mfe_r", "mae_r",
        "exit_reason", "plan_json",
    )
    existing = conn.execute(
        "SELECT id FROM scalp_trade_outcomes WHERE trade_id=?",
        (shadow_outcome_id,),
    ).fetchone()
    if existing:
        conn.execute(
            """
            UPDATE scalp_trade_outcomes
            SET pnl_r=?, pnl_dollar=?, mfe_r=?, mae_r=?, exit_reason=?,
                closed_at=?, plan_json=?
            WHERE trade_id=?
            """,
            (
                outcome["pnl_r"], outcome["pnl_dollar"], outcome["mfe_r"],
                outcome["mae_r"], outcome["exit_reason"], outcome["closed_at"],
                outcome["plan_json"], shadow_outcome_id,
            ),
        )
    else:
        conn.execute(
            """
            INSERT INTO scalp_trade_outcomes
              (plan_id, trade_id, closed_at, ticker, side, context_key,
               setup_type, session, rsi_zone, macd_state, vwap_event,
               spread_bucket, atr_bucket, entry_fill, exit_fill, tp1_hit, tp2_hit,
               stop_hit, time_stop, pnl_r, pnl_dollar, mfe_r, mae_r,
               exit_reason, plan_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            tuple(outcome[key] for key in keys),
        )
    _refresh_context(conn, context_key)
    if bool(config.get("scalp_learn.setup_session_gate_enabled", True)):
        setup_key = setup_session_key_for_plan({**plan, "session": row["session"]})
        _refresh_setup_session_context(conn, setup_key)
        _invalidate_gate(setup_key)
    _invalidate_gate(context_key)
    return outcome


def get_context_gate(context_key: str) -> dict[str, Any]:
    now = time.time()
    with _cache_lock:
        cached = _gate_cache.get(context_key)
        if cached and now - cached[0] <= _CACHE_TTL_S:
            return dict(cached[1])
    init_scalp_tables()
    try:
        with get_conn(read_only=True) as conn:
            row = conn.execute(
                "SELECT * FROM scalp_context_stats WHERE context_key=?",
                (context_key,),
            ).fetchone()
        result = dict(row) if row else _allow_gate(context_key)
    except Exception as exc:
        logger.debug("[ScalpLearning] gate read failed: %s", exc)
        result = _allow_gate(context_key)
    if _is_expired(result.get("expires_at")):
        result.update(gate_state=ALLOW, confidence_floor=0.0, size_mult=1.0)
    with _cache_lock:
        _gate_cache[context_key] = (now, dict(result))
    return result


def _refresh_context(conn, context_key: str) -> dict[str, Any]:
    from agent.config_manager import config

    now = datetime.now(timezone.utc)
    rows = conn.execute(
        """
        SELECT closed_at, pnl_r, pnl_dollar, exit_reason FROM scalp_trade_outcomes
        WHERE context_key=? ORDER BY closed_at ASC
        """,
        (context_key,),
    ).fetchall()
    return _refresh_gate_from_rows(conn, context_key, rows, now=now, config=config)


def _refresh_setup_session_context(conn, setup_key: str) -> dict[str, Any]:
    from agent.config_manager import config

    parts = str(setup_key or "").split("|")
    if len(parts) < 4 or parts[0] != "SETUP_SESSION":
        return _allow_gate(setup_key)
    now = datetime.now(timezone.utc)
    rows = conn.execute(
        """
        SELECT closed_at, pnl_r, pnl_dollar, exit_reason FROM scalp_trade_outcomes
        WHERE setup_type=? AND side=? AND session=?
        ORDER BY closed_at ASC
        """,
        (parts[1], parts[2], parts[3]),
    ).fetchall()
    return _refresh_gate_from_rows(conn, setup_key, rows, now=now, config=config)


def _refresh_gate_from_rows(
    conn,
    context_key: str,
    rows: list[Any],
    *,
    now: datetime,
    config: Any,
) -> dict[str, Any]:
    window_min = max(1, int(config.get("scalp_learn.rolling_window_min", 120)))
    cutoff = now - timedelta(minutes=window_min)
    samples = [row for row in rows if _parse_ts(row["closed_at"]) >= cutoff]
    pnl_values = [float(row["pnl_r"] or 0.0) for row in samples]
    dollar_values = [float(row["pnl_dollar"] or 0.0) for row in samples]
    fast_window_min = max(1, int(config.get("scalp_learn.fast_stop_window_min", 10)))
    fast_cutoff = now - timedelta(minutes=fast_window_min)
    fast_stop_losses = sum(
        1
        for row in samples
        if _parse_ts(row["closed_at"]) >= fast_cutoff
        and float(row["pnl_r"] or 0.0) <= 0
        and "STOP" in str(row["exit_reason"] or "").upper()
    )
    wins = sum(1 for value in pnl_values if value > 0)
    losses = len(pnl_values) - wins
    posterior = (wins + 1.0) / (len(pnl_values) + 2.0)
    alpha = min(
        1.0, max(0.01, float(config.get("scalp_learn.ewma_alpha", 0.25)))
    )
    ewma = 0.0
    for index, value in enumerate(pnl_values):
        ewma = value if index == 0 else alpha * value + (1.0 - alpha) * ewma
    mean_r = sum(pnl_values) / len(pnl_values) if pnl_values else 0.0
    sum_dollar = sum(dollar_values)
    mean_dollar = sum_dollar / len(dollar_values) if dollar_values else 0.0
    old = conn.execute(
        """
        SELECT gate_state, confidence_floor, size_mult
        FROM scalp_context_stats WHERE context_key=?
        """,
        (context_key,),
    ).fetchone()
    old_state = str(old["gate_state"] if old else ALLOW)
    gate_state, confidence_floor, size_mult, reason = _decide_gate(
        len(pnl_values), posterior, ewma, fast_stop_losses, sum_dollar,
        mean_dollar, context_key=context_key,
    )
    ttl = max(1, int(config.get("scalp_learn.action_ttl_min", 60)))
    expires = now + timedelta(minutes=ttl) if gate_state != ALLOW else None
    values = (
        now.isoformat(), len(pnl_values), wins, losses, round(posterior, 6),
        round(ewma, 6), round(mean_r, 6), round(sum_dollar, 2),
        round(mean_dollar, 2), gate_state, confidence_floor, size_mult,
        expires.isoformat() if expires else None,
    )
    if old:
        conn.execute(
            """
            UPDATE scalp_context_stats
            SET updated_at=?, sample_count=?, wins=?, losses=?,
                posterior_win_rate=?, ewma_expectancy_r=?,
                mean_expectancy_r=?, sum_pnl_dollar=?, mean_pnl_dollar=?,
                gate_state=?, confidence_floor=?, size_mult=?, expires_at=?
            WHERE context_key=?
            """,
            values + (context_key,),
        )
    else:
        conn.execute(
            """
            INSERT INTO scalp_context_stats
              (context_key, updated_at, sample_count, wins, losses,
               posterior_win_rate, ewma_expectancy_r, mean_expectancy_r,
               sum_pnl_dollar, mean_pnl_dollar, gate_state, confidence_floor,
               size_mult, expires_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (context_key,) + values,
        )
    if old_state != gate_state:
        old_value = float(old["size_mult"] or 0.0) if old else 0.0
        new_value = size_mult if gate_state == SIZE_REDUCE else confidence_floor
        conn.execute(
            """
            INSERT INTO scalp_learning_actions
              (action_ts, context_key, action_type, old_state, new_state,
               old_value, new_value, reason, expires_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                now.isoformat(), context_key, gate_state, old_state, gate_state,
                old_value, new_value, reason,
                expires.isoformat() if expires else None,
            ),
        )
    return {
        "context_key": context_key,
        "sample_count": len(pnl_values),
        "wins": wins,
        "losses": losses,
        "posterior_win_rate": posterior,
        "ewma_expectancy_r": ewma,
        "mean_expectancy_r": mean_r,
        "sum_pnl_dollar": sum_dollar,
        "mean_pnl_dollar": mean_dollar,
        "gate_state": gate_state,
        "confidence_floor": confidence_floor,
        "size_mult": size_mult,
        "expires_at": expires.isoformat() if expires else None,
        "fast_stop_losses": fast_stop_losses,
    }


def _decide_gate(
    samples: int,
    posterior: float,
    ewma: float,
    fast_stop_losses: int = 0,
    sum_dollar: float = 0.0,
    mean_dollar: float = 0.0,
    context_key: str = "",
) -> tuple[str, float, float, str]:
    from agent.config_manager import config

    if not bool(config.get("scalp_learn.enabled", True)):
        return ALLOW, 0.0, 1.0, "learning disabled"
    min_adjust = max(1, int(config.get("scalp_learn.min_samples_to_adjust", 5)))
    min_block = max(
        min_adjust, int(config.get("scalp_learn.min_samples_to_block", 12))
    )
    block_r = float(config.get("scalp_learn.negative_block_r", -0.20))
    block_wr = float(config.get("scalp_learn.block_win_rate", 0.40))
    reduce_r = float(config.get("scalp_learn.negative_reduce_r", -0.05))
    confidence_wr = float(config.get("scalp_learn.confidence_win_rate", 0.48))
    is_setup_session = str(context_key or "").upper().startswith("SETUP_SESSION|")
    if samples >= min_block and ewma <= block_r and posterior <= block_wr:
        return BLOCK, 0.0, 1.0, f"EWMA {ewma:.3f}R; posterior WR {posterior:.1%}"
    if bool(config.get("scalp_learn.fast_stop_circuit_enabled", True)):
        if is_setup_session and bool(
            config.get("scalp_learn.setup_session_fast_stop_block_enabled", True)
        ):
            block_count = max(
                1,
                int(config.get("scalp_learn.setup_session_fast_stop_block_count", 2)),
            )
            if fast_stop_losses >= block_count:
                fast_window = max(
                    1, int(config.get("scalp_learn.fast_stop_window_min", 10))
                )
                return (
                    BLOCK,
                    0.0,
                    1.0,
                    f"{fast_stop_losses} setup/session stop exits in {fast_window}m",
                )
        fast_count = max(1, int(config.get("scalp_learn.fast_stop_count", 3)))
        if fast_stop_losses >= fast_count:
            fast_window = max(
                1, int(config.get("scalp_learn.fast_stop_window_min", 10))
            )
            mult = min(
                1.0,
                max(0.05, float(config.get("scalp_learn.fast_stop_size_mult", 0.25))),
            )
            return (
                SIZE_REDUCE,
                0.0,
                mult,
                f"{fast_stop_losses} stop exits in {fast_window}m",
            )
    if samples < min_adjust:
        return ALLOW, 0.0, 1.0, f"observing {samples}/{min_adjust} samples"
    if bool(config.get("scalp_learn.dollar_guard_enabled", True)):
        block_dollar = float(config.get("scalp_learn.negative_block_dollar", -100.0))
        reduce_dollar = float(config.get("scalp_learn.negative_reduce_dollar", -25.0))
        if (
            samples >= min_block
            and sum_dollar <= block_dollar
            and posterior <= confidence_wr
        ):
            return (
                BLOCK,
                0.0,
                1.0,
                f"dollar P&L ${sum_dollar:.2f}; avg ${mean_dollar:.2f}; posterior WR {posterior:.1%}",
            )
        if sum_dollar <= reduce_dollar:
            mult = min(
                1.0,
                max(0.05, float(config.get("scalp_learn.size_reduce_mult", 0.50))),
            )
            return (
                SIZE_REDUCE,
                0.0,
                mult,
                f"dollar P&L ${sum_dollar:.2f}; avg ${mean_dollar:.2f}",
            )
    if ewma <= reduce_r:
        mult = min(
            1.0,
            max(0.05, float(config.get("scalp_learn.size_reduce_mult", 0.50))),
        )
        return SIZE_REDUCE, 0.0, mult, f"EWMA {ewma:.3f}R"
    if posterior < confidence_wr:
        base = float(config.get("scalp_learn.base_confidence_floor", 60.0))
        step = max(
            0.0, float(config.get("scalp_learn.confidence_raise_step", 10.0))
        )
        floor = min(100.0, base + step)
        return CONFIDENCE_RAISE, floor, 1.0, f"posterior WR {posterior:.1%}"
    return ALLOW, 0.0, 1.0, "context recovered"


def _block_plan(plan: ScalpSignalPlan, blocker: str) -> None:
    if blocker not in plan.blockers:
        plan.blockers.append(blocker)
    plan.valid = False
    if not plan.invalid_reason:
        plan.invalid_reason = blocker


def _allow_gate(context_key: str) -> dict[str, Any]:
    return {
        "context_key": context_key,
        "sample_count": 0,
        "posterior_win_rate": 0.0,
        "ewma_expectancy_r": 0.0,
        "gate_state": ALLOW,
        "confidence_floor": 0.0,
        "size_mult": 1.0,
        "expires_at": None,
    }


def _side_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "NONE").upper()


def _macd_state(value: Any) -> str:
    try:
        slope = float(value or 0.0)
    except (TypeError, ValueError):
        slope = 0.0
    return "RISING" if slope > 1e-9 else "FALLING" if slope < -1e-9 else "FLAT"


def _spread_bucket(value: Any) -> str:
    try:
        ratio = float(value or 0.0)
    except (TypeError, ValueError):
        ratio = 0.0
    return "TIGHT" if ratio <= 0.10 else "NORMAL" if ratio <= 0.25 else "WIDE"


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return datetime.min.replace(tzinfo=timezone.utc)
    return dt.replace(tzinfo=dt.tzinfo or timezone.utc).astimezone(timezone.utc)


def _is_expired(value: Any) -> bool:
    return bool(value) and _parse_ts(value) <= datetime.now(timezone.utc)


def _invalidate_gate(context_key: str) -> None:
    with _cache_lock:
        _gate_cache.pop(context_key, None)
