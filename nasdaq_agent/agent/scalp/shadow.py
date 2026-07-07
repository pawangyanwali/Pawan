"""Isolated shadow execution for canonical scalp plans.

Shadow trades never write to paper_trades or account P&L.  When enabled, closed
shadow trades can also become negative-ID scalp learning outcomes so the system
can tighten bad contexts without waiting for a human report review.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any

from agent.db import get_conn

from .models import ScalpSignalPlan, SignalSide
from .store import init_scalp_tables

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _opened_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return _now()


def _position_size(plan: ScalpSignalPlan, policy_size_mult: float = 1.0) -> int:
    from agent.config_manager import config

    budget = max(0.0, _float(config.get("paper.budget", 50_000.0)))
    max_notional = budget * max(0.0, _float(config.get("paper.max_trade_pct", 5.0))) / 100.0
    risk_pct = max(0.0, _float(config.get("trading.risk_pct", 1.0)))
    risk_budget = budget * risk_pct / 100.0
    risk = max(plan.risk_per_share, abs(plan.entry - plan.stop_loss))
    by_notional = int(max_notional / plan.entry) if plan.entry > 0 else 0
    by_risk = int(risk_budget / risk) if risk > 0 else 0
    shares = min(value for value in (by_notional, by_risk) if value > 0) if by_notional > 0 or by_risk > 0 else 0
    shares = int(
        shares
        * max(0.0, min(1.0, plan.learning_size_mult))
        * max(0.0, min(1.0, policy_size_mult))
    )
    return max(0, shares)


def _record_shadow_decision(
    plan: ScalpSignalPlan,
    decision: str,
    reason: str,
    *,
    entry_bar_id: int,
    detail: dict[str, Any] | None = None,
) -> None:
    from .store import record_execution_decision

    payload = {"entry_bar_id": int(entry_bar_id), **(detail or {})}
    record_execution_decision(
        plan.plan_id,
        decision,
        reason=reason,
        detail=payload,
    )


def open_shadow_trade(
    plan: ScalpSignalPlan,
    *,
    entry_bar_id: int,
    market_health: dict[str, Any] | None = None,
) -> bool:
    """Persist one hypothetical entry for a valid plan and closed bar."""
    if not plan.valid or plan.side not in {SignalSide.LONG, SignalSide.SHORT}:
        return False
    if plan.entry <= 0 or plan.risk_per_share <= 0 or entry_bar_id <= 0:
        return False

    from agent.config_manager import config

    init_scalp_tables()
    from .execution_policy import evaluate_execution_policy

    policy = evaluate_execution_policy(
        plan,
        mode="SHADOW",
        market_health=market_health,
    )
    if not policy.allowed:
        _record_shadow_decision(
            plan,
            "SHADOW_REJECTED",
            policy.reason,
            entry_bar_id=entry_bar_id,
            detail={"policy": policy.to_dict()},
        )
        logger.info("[Shadow] rejected %s: %s", plan.ticker, policy.reason)
        return False

    shares = _position_size(plan, policy.size_mult)
    if shares <= 0:
        _record_shadow_decision(
            plan,
            "SHADOW_REJECTED",
            "POSITION_SIZE_ZERO",
            entry_bar_id=entry_bar_id,
            detail={"policy": policy.to_dict()},
        )
        return False
    payload = json.dumps(plan.to_dict(), separators=(",", ":"))
    policy_payload = json.dumps(policy.to_dict(), separators=(",", ":"))

    try:
        with get_conn() as conn:
            existing = conn.execute(
                "SELECT id FROM scalp_shadow_trades WHERE ticker=? AND status='OPEN'",
                (plan.ticker,),
            ).fetchone()
            if existing:
                _record_shadow_decision(
                    plan,
                    "SHADOW_REJECTED",
                    "EXISTING_TICKER_POSITION",
                    entry_bar_id=entry_bar_id,
                    detail={"policy": policy.to_dict()},
                )
                return False
            allocation = conn.execute(
                """
                SELECT COALESCE(SUM(entry_fill * shares_remaining), 0) AS allocated
                FROM scalp_shadow_trades WHERE status='OPEN'
                """
            ).fetchone() or {}
            proposed_notional = plan.entry * shares
            max_allocated = _float(policy.checks.get("max_allocated"))
            allocated = _float(allocation.get("allocated"))
            if (
                bool(policy.checks.get("risk_controls_enabled", True))
                and max_allocated > 0
                and allocated + proposed_notional > max_allocated
            ):
                _record_shadow_decision(
                    plan,
                    "SHADOW_REJECTED",
                    "MAX_ALLOCATED_CAPITAL",
                    entry_bar_id=entry_bar_id,
                    detail={
                        "policy": policy.to_dict(),
                        "allocated": allocated,
                        "proposed_notional": proposed_notional,
                    },
                )
                return False
            conn.execute(
                """
                INSERT INTO scalp_shadow_trades
                  (plan_id, entry_bar_id, opened_at, ticker, side, setup_type,
                   session, entry_fill, current_price, stop_loss, original_stop,
                   tp1, tp2, risk_per_share, shares, shares_remaining,
                   high_watermark, low_watermark, policy_size_mult, policy_json,
                   plan_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    plan.plan_id, int(entry_bar_id), _iso(), plan.ticker,
                    plan.side.value, plan.setup_type, plan.session, plan.entry,
                    plan.entry, plan.stop_loss, plan.stop_loss, plan.tp1, plan.tp2,
                    plan.risk_per_share, shares, shares, plan.entry, plan.entry,
                    policy.size_mult, policy_payload, payload,
                ),
            )
        _record_shadow_decision(
            plan,
            "SHADOW_OPENED",
            "ALLOWED",
            entry_bar_id=entry_bar_id,
            detail={"policy": policy.to_dict(), "shares": shares},
        )
        logger.info(
            "[Shadow] opened %s %s at %.4f (%d shares, %.2fR target, %.2fx policy size)",
            plan.ticker, plan.side.value, plan.entry, shares, plan.reward_r,
            policy.size_mult,
        )
        return True
    except Exception as exc:
        # A concurrent cycle/restart can race the unique open-ticker index.
        if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
            return False
        raise


def _executable_price(row: dict[str, Any], quote: dict[str, Any]) -> float:
    side = str(row.get("side") or "").upper()
    if side == "LONG":
        return _float(quote.get("bid") or quote.get("last") or quote.get("mark"))
    return _float(quote.get("ask") or quote.get("last") or quote.get("mark"))


def _trigger_price(quote: dict[str, Any]) -> float:
    """Use the consolidated trade/mark to trigger exits, never one-sided NBBO alone."""
    last = _float(quote.get("last"))
    if last > 0:
        return last
    mark = _float(quote.get("mark"))
    if mark > 0:
        return mark
    bid = _float(quote.get("bid"))
    ask = _float(quote.get("ask"))
    return (bid + ask) / 2.0 if bid > 0 and ask > 0 and ask >= bid else 0.0


def _quote_age_ms(quote: dict[str, Any]) -> int:
    updated_at = _float(quote.get("updated_at"))
    if updated_at <= 0:
        return 2_147_483_647
    return max(0, int((_now().timestamp() - updated_at) * 1000.0))


def _spread_bps(quote: dict[str, Any], reference: float) -> float:
    bid = _float(quote.get("bid"))
    ask = _float(quote.get("ask"))
    if bid <= 0 or ask <= 0 or ask < bid or reference <= 0:
        return 0.0
    return (ask - bid) / reference * 10_000.0


def mark_shadow_trades(quotes: dict[str, dict], *, session: str) -> int:
    """Mark, trail, and resolve all open shadow trades from executable prices."""
    from agent.config_manager import config

    init_scalp_tables()
    with get_conn(read_only=True) as conn:
        rows = [dict(row) for row in conn.execute(
            "SELECT * FROM scalp_shadow_trades WHERE status='OPEN' ORDER BY opened_at"
        ).fetchall()]

    changed = 0
    profit_lock_r = max(0.0, _float(config.get("paper.t1_profit_lock_r", 0.20)))
    trail_r = max(0.0, _float(config.get("paper.post_t1_trail_r", 0.40)))
    max_minutes = max(1, int(config.get("paper.max_bars_scalp", 20)))
    close_session = str(session or "").upper() in {"CLOSED", "HARD_CLOSE"}

    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        quote = quotes.get(ticker) or {}
        trigger = _trigger_price(quote)
        executable = _executable_price(row, quote)
        age_ms = _quote_age_ms(quote)
        max_age_ms = max(
            1,
            int(config.get("scalp_runtime.position_max_quote_age_ms", 5_000)),
        )
        source = str(quote.get("source_status") or "UNKNOWN").upper()
        if trigger <= 0 or executable <= 0 or age_ms > max_age_ms:
            continue
        if source in {"SCAN_SNAPSHOT", "STALE_CACHE", "UNKNOWN"}:
            continue
        side = str(row.get("side") or "").upper()
        direction = 1.0 if side == "LONG" else -1.0
        entry = _float(row.get("entry_fill"))
        risk = max(_float(row.get("risk_per_share")), abs(entry - _float(row.get("original_stop"))))
        if entry <= 0 or risk <= 0:
            continue

        shares = max(1, int(row.get("shares") or 1))
        remaining = max(0, int(row.get("shares_remaining") or shares))
        t1_hit = bool(row.get("t1_hit"))
        t2_hit = bool(row.get("t2_hit"))
        partial = _float(row.get("realized_partial"))
        high = max(_float(row.get("high_watermark"), entry), trigger)
        low = min(_float(row.get("low_watermark"), entry), trigger)
        favorable_r = direction * (trigger - entry) / risk
        mfe_r = max(_float(row.get("mfe_r")), direction * ((high if direction > 0 else low) - entry) / risk)
        mae_r = min(_float(row.get("mae_r")), direction * ((low if direction > 0 else high) - entry) / risk)
        stop = _float(row.get("stop_loss"))
        tp1 = _float(row.get("tp1"))
        tp2 = _float(row.get("tp2"))

        reached_t1 = trigger >= tp1 if direction > 0 else trigger <= tp1
        reached_t2 = trigger >= tp2 if direction > 0 else trigger <= tp2
        if reached_t1 and not t1_hit:
            partial_shares = max(1, shares // 2)
            partial += direction * (tp1 - entry) * partial_shares
            remaining = max(0, shares - partial_shares)
            t1_hit = True

        if t1_hit and remaining > 0:
            lock = entry + direction * profit_lock_r * risk
            trail = (high - trail_r * risk) if direction > 0 else (low + trail_r * risk)
            stop = max(stop, lock, trail) if direction > 0 else min(stop, lock, trail)

        hit_stop = trigger <= stop if direction > 0 else trigger >= stop
        timed_out = _now() - _opened_at(row.get("opened_at")) >= timedelta(minutes=max_minutes)
        exit_reason = ""
        exit_fill = executable
        if reached_t2:
            t2_hit = True
            exit_reason = "TP2"
            exit_fill = tp2
        elif hit_stop:
            exit_reason = "TRAIL_STOP" if t1_hit else "STOP"
            exit_fill = executable
        elif close_session:
            exit_reason = "SESSION_CLOSE"
        elif timed_out:
            exit_reason = "TIME_STOP"

        pnl_dollar = partial + direction * (exit_fill - entry) * remaining
        pnl_r = pnl_dollar / (risk * shares)
        closed = bool(exit_reason)
        spread_bps = _spread_bps(quote, trigger)
        slippage_bps = (
            abs(exit_fill - trigger) / trigger * 10_000.0
            if closed and trigger > 0 else 0.0
        )
        with get_conn() as conn:
            conn.execute(
                """
                UPDATE scalp_shadow_trades SET
                  current_price=?, stop_loss=?, shares_remaining=?, t1_hit=?,
                  t2_hit=?, realized_partial=?, pnl_r=?, pnl_dollar=?, mfe_r=?,
                  mae_r=?, high_watermark=?, low_watermark=?, trigger_price=?,
                  trigger_source=?, trigger_quote_age_ms=?, exit_bid=?, exit_ask=?,
                  exit_spread_bps=?, exit_slippage_bps=?, status=?, closed_at=?,
                  exit_fill=?, exit_reason=?
                WHERE id=? AND status='OPEN'
                """,
                (
                    trigger, stop, 0 if closed else remaining, int(t1_hit), int(t2_hit),
                    partial, pnl_r, pnl_dollar, mfe_r, mae_r, high, low,
                    trigger, source, age_ms, _float(quote.get("bid")),
                    _float(quote.get("ask")), spread_bps, slippage_bps,
                    "CLOSED" if closed else "OPEN", _iso() if closed else None,
                    exit_fill if closed else None, exit_reason, row["id"],
                ),
            )
            if closed:
                try:
                    from .learning import record_closed_shadow_trade

                    record_closed_shadow_trade(conn, int(row["id"]))
                except Exception:
                    logger.exception(
                        "[Shadow] learning outcome failed for shadow trade %s",
                        row.get("id"),
                    )
        changed += 1
        if closed:
            logger.info(
                "[Shadow] closed %s %s %.3fR $%.2f via %s",
                ticker, side, pnl_r, pnl_dollar, exit_reason,
            )
    return changed


def shadow_dashboard_data(*, recent_limit: int = 30) -> dict[str, Any]:
    """Return isolated shadow positions and same-day validation metrics."""
    init_scalp_tables()
    from .execution_policy import execution_policy_status, market_day_start_utc

    day_start = market_day_start_utc().isoformat()
    with get_conn(read_only=True) as conn:
        open_rows = [dict(row) for row in conn.execute(
            "SELECT * FROM scalp_shadow_trades WHERE status='OPEN' ORDER BY opened_at DESC"
        ).fetchall()]
        all_closed = [dict(row) for row in conn.execute(
            """
            SELECT * FROM scalp_shadow_trades
            WHERE status='CLOSED' AND closed_at>=?
            ORDER BY closed_at DESC
            """,
            (day_start,),
        ).fetchall()]
        decision_rows = [dict(row) for row in conn.execute(
            """
            SELECT decision, reason, COUNT(*) AS count
            FROM scalp_execution_decisions
            WHERE decided_at>=? AND decision IN ('SHADOW_OPENED', 'SHADOW_REJECTED')
            GROUP BY decision, reason
            """,
            (day_start,),
        ).fetchall()]
    closed_rows = all_closed[:max(1, min(int(recent_limit), 200))]
    wins = sum(_float(row.get("pnl_r")) > 0 for row in all_closed)
    losses = sum(_float(row.get("pnl_r")) <= 0 for row in all_closed)
    total_r = sum(_float(row.get("pnl_r")) for row in all_closed)
    total_dollar = sum(_float(row.get("pnl_dollar")) for row in all_closed)
    gross_win = sum(max(0.0, _float(row.get("pnl_r"))) for row in all_closed)
    gross_loss = abs(sum(min(0.0, _float(row.get("pnl_r"))) for row in all_closed))
    total = len(all_closed)
    opened_candidates = sum(
        int(row.get("count") or 0)
        for row in decision_rows if row.get("decision") == "SHADOW_OPENED"
    )
    rejected_candidates = sum(
        int(row.get("count") or 0)
        for row in decision_rows if row.get("decision") == "SHADOW_REJECTED"
    )
    rejection_reasons = {
        str(row.get("reason") or "UNKNOWN"): int(row.get("count") or 0)
        for row in decision_rows if row.get("decision") == "SHADOW_REJECTED"
    }
    return {
        "open": open_rows,
        "recent_closed": closed_rows,
        "policy": execution_policy_status("SHADOW"),
        "metrics": {
            "open_count": len(open_rows),
            "closed_today": total,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(wins / total * 100.0, 1) if total else 0.0,
            "expectancy_r": round(total_r / total, 4) if total else 0.0,
            "pnl_r": round(total_r, 4),
            "pnl_dollar": round(total_dollar, 2),
            "profit_factor": round(gross_win / gross_loss, 3) if gross_loss else (999.0 if gross_win else 0.0),
            "candidate_count": opened_candidates + rejected_candidates,
            "approved_count": opened_candidates,
            "rejected_count": rejected_candidates,
            "rejection_reasons": rejection_reasons,
        },
    }
