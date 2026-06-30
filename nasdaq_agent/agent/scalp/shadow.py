"""Isolated shadow execution for canonical scalp plans.

Shadow trades never write to paper_trades or scalp_trade_outcomes.  They exist
only to measure how valid plans would have behaved with executable bid/ask
prices while canonical paper execution is disabled.
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


def _position_size(plan: ScalpSignalPlan) -> int:
    from agent.config_manager import config

    budget = max(0.0, _float(config.get("paper.budget", 50_000.0)))
    max_notional = budget * max(0.0, _float(config.get("paper.max_trade_pct", 5.0))) / 100.0
    risk_pct = max(0.0, _float(config.get("trading.risk_pct", 1.0)))
    risk_budget = budget * risk_pct / 100.0
    risk = max(plan.risk_per_share, abs(plan.entry - plan.stop_loss))
    by_notional = int(max_notional / plan.entry) if plan.entry > 0 else 0
    by_risk = int(risk_budget / risk) if risk > 0 else 0
    shares = min(value for value in (by_notional, by_risk) if value > 0) if by_notional > 0 or by_risk > 0 else 0
    shares = int(shares * max(0.0, min(1.0, plan.learning_size_mult)))
    return max(1, shares)


def open_shadow_trade(plan: ScalpSignalPlan, *, entry_bar_id: int) -> bool:
    """Persist one hypothetical entry for a valid plan and closed bar."""
    if not plan.valid or plan.side not in {SignalSide.LONG, SignalSide.SHORT}:
        return False
    if plan.entry <= 0 or plan.risk_per_share <= 0 or entry_bar_id <= 0:
        return False

    from agent.config_manager import config

    init_scalp_tables()
    configured_max = max(1, int(config.get("paper.max_open_trades", 10)))
    risk_max = max(1, int(config.get("risk.max_concurrent_trades", configured_max)))
    max_open = min(configured_max, risk_max)
    shares = _position_size(plan)
    payload = json.dumps(plan.to_dict(), separators=(",", ":"))

    try:
        with get_conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS count FROM scalp_shadow_trades WHERE status='OPEN'"
            ).fetchone()
            if int((count or {}).get("count") or 0) >= max_open:
                return False
            existing = conn.execute(
                "SELECT id FROM scalp_shadow_trades WHERE ticker=? AND status='OPEN'",
                (plan.ticker,),
            ).fetchone()
            if existing:
                return False
            conn.execute(
                """
                INSERT INTO scalp_shadow_trades
                  (plan_id, entry_bar_id, opened_at, ticker, side, setup_type,
                   session, entry_fill, current_price, stop_loss, original_stop,
                   tp1, tp2, risk_per_share, shares, shares_remaining,
                   high_watermark, low_watermark, plan_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    plan.plan_id, int(entry_bar_id), _iso(), plan.ticker,
                    plan.side.value, plan.setup_type, plan.session, plan.entry,
                    plan.entry, plan.stop_loss, plan.stop_loss, plan.tp1, plan.tp2,
                    plan.risk_per_share, shares, shares, plan.entry, plan.entry,
                    payload,
                ),
            )
        logger.info(
            "[Shadow] opened %s %s at %.4f (%d shares, %.2fR target)",
            plan.ticker, plan.side.value, plan.entry, shares, plan.reward_r,
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
        current = _executable_price(row, quotes.get(ticker) or {})
        if current <= 0:
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
        high = max(_float(row.get("high_watermark"), entry), current)
        low = min(_float(row.get("low_watermark"), entry), current)
        favorable_r = direction * (current - entry) / risk
        mfe_r = max(_float(row.get("mfe_r")), direction * ((high if direction > 0 else low) - entry) / risk)
        mae_r = min(_float(row.get("mae_r")), direction * ((low if direction > 0 else high) - entry) / risk)
        stop = _float(row.get("stop_loss"))
        tp1 = _float(row.get("tp1"))
        tp2 = _float(row.get("tp2"))

        reached_t1 = current >= tp1 if direction > 0 else current <= tp1
        reached_t2 = current >= tp2 if direction > 0 else current <= tp2
        if reached_t1 and not t1_hit:
            partial_shares = max(1, shares // 2)
            partial += direction * (tp1 - entry) * partial_shares
            remaining = max(0, shares - partial_shares)
            t1_hit = True

        if t1_hit and remaining > 0:
            lock = entry + direction * profit_lock_r * risk
            trail = (high - trail_r * risk) if direction > 0 else (low + trail_r * risk)
            stop = max(stop, lock, trail) if direction > 0 else min(stop, lock, trail)

        hit_stop = current <= stop if direction > 0 else current >= stop
        timed_out = _now() - _opened_at(row.get("opened_at")) >= timedelta(minutes=max_minutes)
        exit_reason = ""
        exit_fill = current
        if reached_t2:
            t2_hit = True
            exit_reason = "TP2"
            exit_fill = tp2
        elif hit_stop:
            exit_reason = "TRAIL_STOP" if t1_hit else "STOP"
            exit_fill = current
        elif close_session:
            exit_reason = "SESSION_CLOSE"
        elif timed_out:
            exit_reason = "TIME_STOP"

        pnl_dollar = partial + direction * (exit_fill - entry) * remaining
        pnl_r = pnl_dollar / (risk * shares)
        closed = bool(exit_reason)
        with get_conn() as conn:
            conn.execute(
                """
                UPDATE scalp_shadow_trades SET
                  current_price=?, stop_loss=?, shares_remaining=?, t1_hit=?,
                  t2_hit=?, realized_partial=?, pnl_r=?, pnl_dollar=?, mfe_r=?,
                  mae_r=?, high_watermark=?, low_watermark=?, status=?,
                  closed_at=?, exit_fill=?, exit_reason=?
                WHERE id=? AND status='OPEN'
                """,
                (
                    current, stop, 0 if closed else remaining, int(t1_hit), int(t2_hit),
                    partial, pnl_r, pnl_dollar, mfe_r, mae_r, high, low,
                    "CLOSED" if closed else "OPEN", _iso() if closed else None,
                    exit_fill if closed else None, exit_reason, row["id"],
                ),
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
    day_start = _now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
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
    closed_rows = all_closed[:max(1, min(int(recent_limit), 200))]
    wins = sum(_float(row.get("pnl_r")) > 0 for row in all_closed)
    losses = sum(_float(row.get("pnl_r")) <= 0 for row in all_closed)
    total_r = sum(_float(row.get("pnl_r")) for row in all_closed)
    total_dollar = sum(_float(row.get("pnl_dollar")) for row in all_closed)
    gross_win = sum(max(0.0, _float(row.get("pnl_r"))) for row in all_closed)
    gross_loss = abs(sum(min(0.0, _float(row.get("pnl_r"))) for row in all_closed))
    total = len(all_closed)
    return {
        "open": open_rows,
        "recent_closed": closed_rows,
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
        },
    }
