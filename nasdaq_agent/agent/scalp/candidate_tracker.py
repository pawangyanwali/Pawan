"""Counterfactual outcome tracking for scalp candidates.

Candidate trials are observational. They never create a paper position, consume
risk budget, or feed the learning tables. Their only purpose is to measure what
would have happened to setups rejected by confirmation, policy, or strategy
selection so those gates can be evaluated without weakening production safety.
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

CANONICAL_CANDIDATE = "CANONICAL_VALID"
MTF_CANDIDATE = "MTF_SHADOW_READY"
_last_purge_date: str = ""

_INSERT_SQL = """
    INSERT INTO scalp_candidate_trials
      (candidate_key, plan_id, observed_at, ticker, side,
       candidate_type, strategy_family, session, entry_bar_id,
       admission_state, admission_reason, entry_price, current_price,
       stop_loss, original_stop, tp1, tp2, risk_per_share,
       high_watermark, low_watermark, trigger_source,
       trigger_quote_age_ms, quality_score, metadata_json, plan_json)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT DO NOTHING
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _opened_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            parsed = _now()
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)


def candidate_key(
    plan: ScalpSignalPlan,
    *,
    candidate_type: str,
    entry_bar_id: int,
    strategy_family: str = "",
) -> str:
    family = str(strategy_family or plan.strategy_family or "NONE").upper()
    return ":".join(
        (
            str(candidate_type or "UNKNOWN").upper(),
            str(plan.ticker or "").upper(),
            plan.side.value,
            family,
            str(int(entry_bar_id)),
        )
    )


def register_candidate(
    plan: ScalpSignalPlan,
    *,
    candidate_type: str,
    entry_bar_id: int,
    strategy_family: str = "",
    admission_state: str = "OBSERVED",
    admission_reason: str = "",
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Insert one immutable open episode per type/ticker/side/family."""
    values = _candidate_values(
        plan,
        candidate_type=candidate_type,
        entry_bar_id=entry_bar_id,
        strategy_family=strategy_family,
        admission_state=admission_state,
        admission_reason=admission_reason,
        metadata=metadata,
    )
    if values is None:
        return False
    init_scalp_tables()
    _purge_if_due()
    with get_conn() as conn:
        cursor = conn.execute(_INSERT_SQL, values)
    return bool(getattr(cursor, "rowcount", 0))


def register_candidate_batch(
    candidates: list[
        tuple[str, ScalpSignalPlan, int, dict[str, Any]]
    ],
) -> int:
    """Persist one scan's candidate episodes in a single transaction."""
    values = []
    for candidate_type, plan, entry_bar_id, metadata in candidates:
        row = _candidate_values(
            plan,
            candidate_type=candidate_type,
            entry_bar_id=entry_bar_id,
            strategy_family=plan.strategy_family,
            admission_state=(
                "MTF_OBSERVATION_ONLY"
                if str(candidate_type).upper() == MTF_CANDIDATE
                else "OBSERVED_BEFORE_CONFIRMATION"
            ),
            metadata=metadata,
        )
        if row is not None:
            values.append(row)
    if not values:
        return 0
    init_scalp_tables()
    _purge_if_due()
    with get_conn() as conn:
        conn.executemany(_INSERT_SQL, values)
    return len(values)


def _candidate_values(
    plan: ScalpSignalPlan,
    *,
    candidate_type: str,
    entry_bar_id: int,
    strategy_family: str = "",
    admission_state: str = "OBSERVED",
    admission_reason: str = "",
    metadata: dict[str, Any] | None = None,
) -> tuple[Any, ...] | None:
    if plan.side not in {SignalSide.LONG, SignalSide.SHORT}:
        return None
    if (
        int(entry_bar_id or 0) <= 0
        or plan.entry <= 0
        or plan.stop_loss <= 0
        or plan.tp1 <= 0
        or plan.tp2 <= 0
        or plan.risk_per_share <= 0
    ):
        return None
    family = str(strategy_family or plan.strategy_family or "NONE").upper()
    kind = str(candidate_type or "UNKNOWN").upper()
    key = candidate_key(
        plan,
        candidate_type=kind,
        entry_bar_id=entry_bar_id,
        strategy_family=family,
    )
    payload = json.dumps(plan.to_dict(), separators=(",", ":"))
    meta = json.dumps(metadata or {}, separators=(",", ":"))
    return (
        key,
        plan.plan_id,
        _iso(),
        plan.ticker,
        plan.side.value,
        kind,
        family,
        plan.session,
        int(entry_bar_id),
        str(admission_state or "OBSERVED").upper(),
        str(admission_reason or ""),
        plan.entry,
        plan.entry,
        plan.stop_loss,
        plan.stop_loss,
        plan.tp1,
        plan.tp2,
        plan.risk_per_share,
        plan.entry,
        plan.entry,
        getattr(plan.source, "value", str(plan.source)),
        int(plan.data_age_ms or 0),
        float(plan.entry_quality_score or plan.setup_score or 0.0),
        meta,
        payload,
    )


def update_candidate_admission(
    *,
    ticker: str,
    entry_bar_id: int,
    candidate_type: str = CANONICAL_CANDIDATE,
    admission_state: str,
    admission_reason: str = "",
) -> int:
    """Attach the actual confirmation/policy disposition to an observed trial."""
    init_scalp_tables()
    ticker = str(ticker or "").upper()
    kind = str(candidate_type or CANONICAL_CANDIDATE).upper()
    with get_conn() as conn:
        match = conn.execute(
            """
            SELECT id FROM scalp_candidate_trials
            WHERE ticker=? AND candidate_type=?
              AND (entry_bar_id=? OR status='OPEN')
            ORDER BY
              CASE WHEN entry_bar_id=? THEN 0 ELSE 1 END,
              observed_at DESC
            LIMIT 1
            """,
            (ticker, kind, int(entry_bar_id or 0), int(entry_bar_id or 0)),
        ).fetchone()
        if not match:
            return 0
        cursor = conn.execute(
            """
            UPDATE scalp_candidate_trials
            SET admission_state=?, admission_reason=?
            WHERE id=?
            """,
            (
                str(admission_state or "UNKNOWN").upper(),
                str(admission_reason or ""),
                match["id"],
            ),
        )
    return max(0, int(getattr(cursor, "rowcount", 0) or 0))


def _quote_age_ms(quote: dict[str, Any]) -> int:
    updated = _number(quote.get("updated_at"))
    if updated <= 0:
        return 2_147_483_647
    return max(0, int((_now().timestamp() - updated) * 1000.0))


def _trigger_price(quote: dict[str, Any]) -> float:
    return _number(quote.get("last") or quote.get("mark"))


def _executable_price(side: str, quote: dict[str, Any]) -> float:
    if str(side).upper() == "LONG":
        return _number(quote.get("bid") or quote.get("last") or quote.get("mark"))
    return _number(quote.get("ask") or quote.get("last") or quote.get("mark"))


def mark_candidate_trials(quotes: dict[str, dict], *, session: str) -> int:
    """Mark all open trials from the one-second price bus and resolve brackets."""
    from agent.config_manager import config

    init_scalp_tables()
    with get_conn(read_only=True) as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT * FROM scalp_candidate_trials
                WHERE status='OPEN'
                ORDER BY observed_at
                """
            ).fetchall()
        ]

    updates: list[tuple[Any, ...]] = []
    maximum_age_ms = max(
        1, int(config.get("scalp_runtime.position_max_quote_age_ms", 5_000))
    )
    max_minutes = max(1, int(config.get("paper.max_bars_scalp", 20)))
    profit_lock_r = max(0.0, _number(config.get("paper.t1_profit_lock_r", 0.20)))
    trail_r = max(0.0, _number(config.get("paper.post_t1_trail_r", 0.40)))
    close_session = str(session or "").upper() in {"CLOSED", "HARD_CLOSE"}

    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        quote = quotes.get(ticker) or {}
        trigger = _trigger_price(quote)
        executable = _executable_price(str(row.get("side") or ""), quote)
        age_ms = _quote_age_ms(quote)
        source = str(quote.get("source_status") or "UNKNOWN").upper()
        if (
            trigger <= 0
            or executable <= 0
            or age_ms > maximum_age_ms
            or source in {"SCAN_SNAPSHOT", "STALE_CACHE", "UNKNOWN"}
        ):
            continue

        side = str(row.get("side") or "").upper()
        direction = 1.0 if side == "LONG" else -1.0
        entry = _number(row.get("entry_price"))
        risk = max(
            _number(row.get("risk_per_share")),
            abs(entry - _number(row.get("original_stop"))),
        )
        if entry <= 0 or risk <= 0:
            continue

        high = max(_number(row.get("high_watermark"), entry), trigger)
        low = min(_number(row.get("low_watermark"), entry), trigger)
        mfe_r = max(
            _number(row.get("mfe_r")),
            direction * ((high if direction > 0 else low) - entry) / risk,
        )
        mae_r = min(
            _number(row.get("mae_r")),
            direction * ((low if direction > 0 else high) - entry) / risk,
        )
        stop = _number(row.get("stop_loss"))
        tp1 = _number(row.get("tp1"))
        tp2 = _number(row.get("tp2"))
        t1_hit = bool(row.get("t1_hit"))
        t2_hit = bool(row.get("t2_hit"))
        partial_r = _number(row.get("realized_partial_r"))
        remaining = max(0.0, min(1.0, _number(row.get("remaining_fraction"), 1.0)))

        reached_t1 = trigger >= tp1 if direction > 0 else trigger <= tp1
        reached_t2 = trigger >= tp2 if direction > 0 else trigger <= tp2
        if reached_t1 and not t1_hit:
            partial_r += direction * (tp1 - entry) / risk * 0.5
            remaining = 0.5
            t1_hit = True

        if t1_hit and remaining > 0:
            lock = entry + direction * profit_lock_r * risk
            trail = high - trail_r * risk if direction > 0 else low + trail_r * risk
            stop = max(stop, lock, trail) if direction > 0 else min(stop, lock, trail)

        hit_stop = trigger <= stop if direction > 0 else trigger >= stop
        timed_out = _now() - _opened_at(row.get("observed_at")) >= timedelta(
            minutes=max_minutes
        )
        exit_reason = ""
        exit_price = executable
        if reached_t2:
            t2_hit = True
            exit_reason = "TP2"
            exit_price = tp2
        elif hit_stop:
            exit_reason = "TRAIL_STOP" if t1_hit else "STOP"
        elif close_session:
            exit_reason = "SESSION_CLOSE"
        elif timed_out:
            exit_reason = "TIME_STOP"

        pnl_r = partial_r + direction * (exit_price - entry) / risk * remaining
        closed = bool(exit_reason)
        updates.append((
            trigger,
            stop,
            int(t1_hit),
            int(t2_hit),
            partial_r,
            0.0 if closed else remaining,
            pnl_r,
            mfe_r,
            mae_r,
            high,
            low,
            source,
            age_ms,
            "CLOSED" if closed else "OPEN",
            _iso() if closed else None,
            exit_price if closed else None,
            exit_reason,
            row["id"],
        ))
    if updates:
        with get_conn() as conn:
            conn.executemany(
                """
                UPDATE scalp_candidate_trials SET
                  current_price=?, stop_loss=?, t1_hit=?, t2_hit=?,
                  realized_partial_r=?, remaining_fraction=?, pnl_r=?,
                  mfe_r=?, mae_r=?, high_watermark=?, low_watermark=?,
                  trigger_source=?, trigger_quote_age_ms=?, status=?,
                  resolved_at=?, exit_price=?, exit_reason=?
                WHERE id=? AND status='OPEN'
                """,
                updates,
            )
    return len(updates)


def _purge_if_due() -> None:
    """Bound observational storage without putting cleanup in the hot tick path."""
    global _last_purge_date
    today = _now().date().isoformat()
    if _last_purge_date == today:
        return
    from agent.config_manager import config

    retention_days = max(
        7, int(config.get("scalp.candidate_retention_days", 30))
    )
    cutoff = (_now() - timedelta(days=retention_days)).isoformat()
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM scalp_candidate_trials WHERE observed_at<?",
            (cutoff,),
        )
    _last_purge_date = today


def candidate_dashboard_data(*, recent_limit: int = 40) -> dict[str, Any]:
    """Return today's counterfactual gate evidence without learning contamination."""
    from .execution_policy import market_day_start_utc

    init_scalp_tables()
    day_start = market_day_start_utc().isoformat()
    limit = max(1, min(int(recent_limit), 200))
    with get_conn(read_only=True) as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT * FROM scalp_candidate_trials
                WHERE observed_at>=?
                ORDER BY observed_at DESC
                """,
                (day_start,),
            ).fetchall()
        ]
    closed = [row for row in rows if str(row.get("status") or "") == "CLOSED"]
    grouped: dict[str, dict[str, Any]] = {}
    for row in closed:
        key = str(row.get("candidate_type") or "UNKNOWN")
        bucket = grouped.setdefault(
            key,
            {"closed": 0, "wins": 0, "pnl_r": 0.0, "tp1_hits": 0, "tp2_hits": 0},
        )
        pnl_r = _number(row.get("pnl_r"))
        bucket["closed"] += 1
        bucket["wins"] += int(pnl_r > 0)
        bucket["pnl_r"] += pnl_r
        bucket["tp1_hits"] += int(bool(row.get("t1_hit")))
        bucket["tp2_hits"] += int(bool(row.get("t2_hit")))
    for bucket in grouped.values():
        count = int(bucket["closed"])
        bucket["win_rate_pct"] = round(bucket["wins"] / count * 100.0, 1) if count else 0.0
        bucket["expectancy_r"] = round(bucket["pnl_r"] / count, 4) if count else 0.0
        bucket["pnl_r"] = round(bucket["pnl_r"], 4)

    admissions: dict[str, int] = {}
    for row in rows:
        state = str(row.get("admission_state") or "UNKNOWN")
        admissions[state] = admissions.get(state, 0) + 1
    total_r = sum(_number(row.get("pnl_r")) for row in closed)
    wins = sum(_number(row.get("pnl_r")) > 0 for row in closed)
    return {
        "metrics": {
            "observed_today": len(rows),
            "open": sum(str(row.get("status") or "") == "OPEN" for row in rows),
            "closed": len(closed),
            "wins": wins,
            "losses": len(closed) - wins,
            "win_rate_pct": round(wins / len(closed) * 100.0, 1) if closed else 0.0,
            "expectancy_r": round(total_r / len(closed), 4) if closed else 0.0,
            "pnl_r": round(total_r, 4),
            "admissions": admissions,
            "by_type": grouped,
        },
        "recent": rows[:limit],
        "learning_isolation": True,
    }
