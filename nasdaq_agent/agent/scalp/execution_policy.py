"""Shared execution preflight for canonical paper and shadow scalp trades."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from agent.db import get_conn

from .models import ScalpSignalPlan

ET = ZoneInfo("America/New_York")

_SESSION_SIZE_KEYS = {
    "PRE_MARKET": "scalp_runtime.pre_market_size_mult",
    "RESTRICTED": "scalp_runtime.restricted_size_mult",
    "PRIME": "scalp_runtime.prime_size_mult",
    "LUNCH_BLOCK": "scalp_runtime.lunch_size_mult",
    "STANDARD": "scalp_runtime.standard_size_mult",
    "CLOSING_CAUTION": "scalp_runtime.closing_size_mult",
    "AFTER_HOURS": "scalp_runtime.after_hours_size_mult",
    "HARD_CLOSE": "scalp_runtime.hard_close_size_mult",
    "CLOSED": "scalp_runtime.closed_size_mult",
}

_SESSION_SIZE_DEFAULTS = {
    "PRE_MARKET": 0.35,
    "RESTRICTED": 0.0,
    "PRIME": 1.0,
    "LUNCH_BLOCK": 0.0,
    "STANDARD": 0.8,
    "CLOSING_CAUTION": 0.0,
    "AFTER_HOURS": 0.3,
    "HARD_CLOSE": 0.0,
    "CLOSED": 0.0,
}


def market_day_start_utc(now: datetime | None = None) -> datetime:
    """Return midnight New York time as a UTC timestamp."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    local = current.astimezone(ET)
    return local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _source_name(plan: ScalpSignalPlan) -> str:
    source = getattr(plan.source, "value", plan.source)
    return str(source or "UNKNOWN").upper()


@dataclass(frozen=True)
class ExecutionPolicyDecision:
    allowed: bool
    reason: str
    size_mult: float
    checks: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def execution_market_health(*, max_age_s: float = 2.0) -> dict[str, Any]:
    """Combine price-bus freshness with durable Schwab token state."""
    try:
        from agent.valkey_client import price_bus_health

        result = dict(price_bus_health(max_age_s=max_age_s))
    except Exception as exc:
        result = {"status": "UNKNOWN", "health_error": str(exc)}

    token_states: dict[str, str] = {}
    try:
        with get_conn(read_only=True) as conn:
            rows = conn.execute(
                "SELECT app, status FROM schwab_tokens WHERE app IN (?, ?)",
                ("trader", "marketdata"),
            ).fetchall()
        token_states = {
            str(row.get("app") or "").lower(): str(row.get("status") or "OK").upper()
            for row in rows
        }
    except Exception:
        # Fresh installations and isolated unit tests may not have the token table.
        token_states = {}

    result["token_states"] = token_states
    result["auth_required"] = any(
        state == "AUTH_REQUIRED" for state in token_states.values()
    )
    return result


def _ledger_state(mode: str) -> dict[str, Any]:
    shadow = str(mode or "SHADOW").upper() == "SHADOW"
    if shadow:
        from .store import init_scalp_tables

        init_scalp_tables()
    else:
        from agent.paper_trading import init_db

        init_db()
    table = "scalp_shadow_trades" if shadow else "paper_trades"
    entry_column = "entry_fill" if shadow else "entry_price"
    start = market_day_start_utc().isoformat()
    with get_conn(read_only=True) as conn:
        open_row = conn.execute(
            f"""
            SELECT COUNT(*) AS count,
                   COALESCE(SUM({entry_column} * COALESCE(NULLIF(shares_remaining, 0), shares)), 0) AS allocated
            FROM {table} WHERE status='OPEN'
            """
        ).fetchone() or {}
        day_row = conn.execute(
            f"""
            SELECT COUNT(*) AS count,
                   COALESCE(SUM(CASE WHEN status='CLOSED' THEN COALESCE(pnl_dollar, 0) ELSE 0 END), 0) AS pnl
            FROM {table} WHERE opened_at>=?
            """,
            (start,),
        ).fetchone() or {}
    return {
        "open_count": int(open_row.get("count") or 0),
        "allocated": round(_number(open_row.get("allocated")), 2),
        "daily_trade_count": int(day_row.get("count") or 0),
        "daily_pnl": round(_number(day_row.get("pnl")), 2),
        "day_start_utc": start,
    }


def _risk_limits(config: Any) -> dict[str, Any]:
    budget = max(0.0, _number(config.get("paper.budget", 50_000.0)))
    usd_limit = abs(_number(config.get("paper.daily_loss_halt_usd", 300.0)))
    pct_limit = abs(_number(config.get("paper.daily_loss_halt_pct", 0.25)))
    pct_limit_usd = budget * pct_limit / 100.0
    candidates = [value for value in (usd_limit, pct_limit_usd) if value > 0]
    configured_open = max(1, int(config.get("paper.max_open_trades", 10)))
    risk_open = max(1, int(config.get("risk.max_concurrent_trades", configured_open)))
    return {
        "risk_controls_enabled": bool(config.get("paper.enforce_risk_controls", True)),
        "budget": budget,
        "daily_loss_halt_usd": min(candidates) if candidates else 0.0,
        "max_daily_trades": max(1, int(config.get("paper.max_daily_trades", 75))),
        "max_open_trades": min(configured_open, risk_open),
        "max_allocated": budget
        * max(0.0, _number(config.get("paper.max_allocated_pct", 40.0)))
        / 100.0,
    }


def _session_size_mult(session: str, config: Any) -> float:
    name = str(session or "").upper()
    key = _SESSION_SIZE_KEYS.get(name)
    default = _SESSION_SIZE_DEFAULTS.get(name, 1.0)
    value = _number(config.get(key, default), default) if key else default
    return max(0.0, min(1.0, value))


def _same_context_entries(
    plan: ScalpSignalPlan,
    *,
    window_min: int,
    mode: str,
) -> dict[str, Any]:
    """Count recent entries with the same learned context across shadow/paper."""
    from .learning import context_key_for_plan

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(1, int(window_min)))
    current_key = context_key_for_plan(plan)
    rows: list[dict[str, Any]] = []
    try:
        with get_conn(read_only=True) as conn:
            rows.extend(
                dict(row)
                for row in conn.execute(
                    """
                    SELECT opened_at, status, plan_json
                    FROM scalp_shadow_trades
                    WHERE opened_at >= ?
                    ORDER BY opened_at DESC
                    LIMIT 250
                    """,
                    (cutoff.isoformat(),),
                ).fetchall()
            )
            try:
                rows.extend(
                    dict(row)
                    for row in conn.execute(
                        """
                        SELECT p.opened_at, p.status, s.plan_json
                        FROM paper_trades p
                        LEFT JOIN scalp_signal_plans s ON s.plan_id=p.scalp_plan_id
                        WHERE p.opened_at >= ?
                          AND p.execution_contract='SCALP_PLAN_V1'
                        ORDER BY p.opened_at DESC
                        LIMIT 250
                        """,
                        (cutoff.isoformat(),),
                    ).fetchall()
                )
            except Exception:
                # Some isolated tests initialize only the shadow ledger.
                pass
    except Exception as exc:
        return {
            "enabled": True,
            "context_key": current_key,
            "window_min": window_min,
            "entry_count": 0,
            "open_count": 0,
            "error": str(exc),
        }

    matches = []
    for row in rows:
        try:
            payload = json.loads(row.get("plan_json") or "{}")
        except Exception:
            payload = {}
        if payload and context_key_for_plan(payload) == current_key:
            matches.append(row)
    return {
        "enabled": True,
        "mode": str(mode or "").upper(),
        "context_key": current_key,
        "window_min": window_min,
        "entry_count": len(matches),
        "open_count": sum(
            1 for row in matches if str(row.get("status") or "").upper() == "OPEN"
        ),
    }


def evaluate_execution_policy(
    plan: ScalpSignalPlan,
    *,
    mode: str,
    market_health: dict[str, Any] | None = None,
) -> ExecutionPolicyDecision:
    """Evaluate controls that must agree across shadow and paper execution."""
    from agent.config_manager import config

    session = str(plan.session or "").upper()
    health = dict(market_health or execution_market_health())
    ledger = _ledger_state(mode)
    limits = _risk_limits(config)
    size_mult = _session_size_mult(session, config)
    checks = {
        "mode": str(mode or "").upper(),
        "session": session,
        "source": _source_name(plan),
        "quote_age_ms": int(plan.data_age_ms or 0),
        "market_status": str(health.get("status") or "UNKNOWN").upper(),
        "auth_required": bool(health.get("auth_required")),
        "token_states": health.get("token_states") or {},
        "session_size_mult": size_mult,
        **ledger,
        **limits,
    }

    if not bool(config.get("scalp_runtime.execution_policy_enabled", True)):
        return ExecutionPolicyDecision(True, "POLICY_DISABLED", size_mult, checks)

    blocked_sessions = {
        str(value).upper()
        for value in config.get(
            "scalp_runtime.execution_blocked_sessions",
            ["CLOSED", "RESTRICTED", "LUNCH_BLOCK", "CLOSING_CAUTION", "HARD_CLOSE"],
        )
    }
    if session in blocked_sessions or size_mult <= 0:
        return ExecutionPolicyDecision(False, f"SESSION_{session}_EXECUTION_BLOCKED", 0.0, checks)

    if bool(config.get("scalp_runtime.require_live_execution_data", True)):
        if checks["auth_required"]:
            return ExecutionPolicyDecision(False, "SCHWAB_AUTH_REQUIRED", 0.0, checks)
        if checks["market_status"] in {"NO_DATA", "STALE", "SCAN_SNAPSHOT", "UNKNOWN"}:
            return ExecutionPolicyDecision(False, "MARKET_DATA_NOT_TRUSTED", 0.0, checks)
        allowed_sources = {"WS"}
        if bool(config.get("scalp.allow_rest_fallback_trading", False)):
            allowed_sources.add("REST")
        if checks["source"] not in allowed_sources:
            return ExecutionPolicyDecision(False, "QUOTE_SOURCE_NOT_TRADABLE", 0.0, checks)
        maximum_age = max(1, int(config.get("scalp.max_quote_age_ms", 2_000)))
        if checks["quote_age_ms"] > maximum_age:
            return ExecutionPolicyDecision(False, "QUOTE_TOO_OLD", 0.0, checks)

    if bool(limits["risk_controls_enabled"]):
        halt = _number(limits["daily_loss_halt_usd"])
        if halt > 0 and _number(ledger["daily_pnl"]) <= -halt:
            return ExecutionPolicyDecision(False, "DAILY_LOSS_HALT", 0.0, checks)
        if int(ledger["daily_trade_count"]) >= int(limits["max_daily_trades"]):
            return ExecutionPolicyDecision(False, "MAX_DAILY_TRADES", 0.0, checks)
        if int(ledger["open_count"]) >= int(limits["max_open_trades"]):
            return ExecutionPolicyDecision(False, "MAX_OPEN_TRADES", 0.0, checks)

    if bool(config.get("scalp_runtime.context_cluster_throttle_enabled", True)):
        cluster_window = max(
            1, int(config.get("scalp_runtime.context_cluster_window_min", 10))
        )
        cluster_max = max(
            1, int(config.get("scalp_runtime.context_cluster_max_entries", 3))
        )
        cluster = _same_context_entries(
            plan,
            window_min=cluster_window,
            mode=mode,
        )
        checks["context_cluster"] = cluster
        checks["context_cluster_max_entries"] = cluster_max
        if int(cluster.get("entry_count") or 0) >= cluster_max:
            return ExecutionPolicyDecision(
                False,
                "CONTEXT_CLUSTER_THROTTLE",
                0.0,
                checks,
            )

    return ExecutionPolicyDecision(True, "ALLOWED", size_mult, checks)


def execution_policy_status(mode: str = "SHADOW") -> dict[str, Any]:
    """Return current ledger limits and halt state for dashboard observability."""
    from agent.config_manager import config

    try:
        ledger = _ledger_state(mode)
        limits = _risk_limits(config)
    except Exception as exc:
        return {"halted": False, "reason": "STATUS_UNAVAILABLE", "error": str(exc)}
    halt = _number(limits["daily_loss_halt_usd"])
    reason = ""
    if halt > 0 and _number(ledger["daily_pnl"]) <= -halt:
        reason = "DAILY_LOSS_HALT"
    elif int(ledger["daily_trade_count"]) >= int(limits["max_daily_trades"]):
        reason = "MAX_DAILY_TRADES"
    return {"halted": bool(reason), "reason": reason, **ledger, **limits}
