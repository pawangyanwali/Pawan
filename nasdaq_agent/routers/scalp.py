"""Read-only command-center API for the greenfield scalping platform."""
from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends

from auth.dependencies import AuthenticatedUser, require_viewer

router = APIRouter(tags=["scalp"])

_cache_lock = threading.Lock()
_cache_ts = 0.0
_cache_value: dict[str, Any] | None = None
_CACHE_TTL_S = 0.75


@router.get("/api/scalp/dashboard")
async def scalp_dashboard(_user: AuthenticatedUser = Depends(require_viewer)):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _dashboard_snapshot)


@router.get("/api/scalp/learning")
async def scalp_learning(_user: AuthenticatedUser = Depends(require_viewer)):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _learning_snapshot)


def _dashboard_snapshot() -> dict[str, Any]:
    global _cache_ts, _cache_value
    now = time.time()
    with _cache_lock:
        if _cache_value is not None and now - _cache_ts <= _CACHE_TTL_S:
            return _cache_value

    from agent.config_manager import config
    from agent.paper_trading import get_open_trades, get_today_pnl
    from agent.signal_snapshot import read_latest
    from agent.valkey_client import get_all_prices, price_bus_health

    snapshot = read_latest() or {}
    prices = get_all_prices()
    plans = []
    for signal in snapshot.get("signals") or []:
        plan = dict(signal.get("scalp_plan") or {})
        if not plan:
            continue
        plans.append(_enrich_plan(plan, prices))
    plans.sort(
        key=lambda plan: (
            plan["state"] != "ACTIONABLE",
            plan["state"] != "WATCH",
            -float(plan.get("confidence") or 0.0),
            str(plan.get("ticker") or ""),
        )
    )

    positions = [_enrich_position(row, prices) for row in get_open_trades()]
    today = get_today_pnl()
    budget = float(config.get("paper.budget", 50000.0))
    configured_max_open = int(config.get("paper.max_open_trades", 10))
    risk_max_open = int(config.get("risk.max_concurrent_trades", configured_max_open))
    effective_max_open = min(configured_max_open, risk_max_open)
    max_trade_pct = float(config.get("paper.max_trade_pct", 5.0))
    max_allocated_pct = float(config.get("paper.max_allocated_pct", 40.0))
    max_portfolio_heat_pct = float(config.get("risk.max_portfolio_heat_pct", 1.5))
    daily_halt_usd = abs(float(config.get("paper.daily_loss_halt_usd", 300.0)))
    daily_halt_pct = abs(float(config.get("paper.daily_loss_halt_pct", 0.25)))
    percent_halt_usd = budget * daily_halt_pct / 100.0
    effective_daily_halt_usd = min(
        threshold for threshold in (daily_halt_usd, percent_halt_usd) if threshold > 0
    ) if daily_halt_usd > 0 or percent_halt_usd > 0 else 0.0
    open_risk = sum(
        abs(float(row.get("entry_price") or 0) - float(row.get("stop") or 0))
        * int(row.get("shares_remaining") or row.get("shares") or 0)
        for row in positions
    )
    allocated = sum(
        float(row.get("entry_price") or 0)
        * int(row.get("shares_remaining") or row.get("shares") or 0)
        for row in positions
    )
    learning = _learning_snapshot()
    health = price_bus_health(max_age_s=2.0)
    counts = {
        "actionable": sum(plan["state"] == "ACTIONABLE" for plan in plans),
        "watch": sum(plan["state"] == "WATCH" for plan in plans),
        "blocked": sum(plan["state"] == "BLOCKED" for plan in plans),
        "data_gap": sum(plan["state"] == "DATA_GAP" for plan in plans),
        "long": sum(plan.get("side") == "LONG" and plan["state"] == "ACTIONABLE" for plan in plans),
        "short": sum(plan.get("side") == "SHORT" and plan["state"] == "ACTIONABLE" for plan in plans),
    }
    result = {
        "schema_version": 1,
        "asof_ts": datetime.now(timezone.utc).isoformat(),
        "scan_ts": snapshot.get("ts"),
        "session": snapshot.get("session") or {},
        "regime": snapshot.get("regime") or {},
        "market_data_health": health,
        "counts": counts,
        "plans": plans,
        "positions": positions,
        "risk": {
            "budget": round(budget, 2),
            "realized_pnl": round(float(today.get("total_pnl_dollar") or 0.0), 2),
            "open_unrealized_pnl": round(sum(float(row.get("unrealized_pnl") or 0.0) for row in positions), 2),
            "open_risk": round(open_risk, 2),
            "open_risk_pct": round(open_risk / budget * 100, 3) if budget else 0.0,
            "allocated": round(allocated, 2),
            "allocated_pct": round(allocated / budget * 100, 2) if budget else 0.0,
            "open_positions": len(positions),
            "max_open_positions": effective_max_open,
            "configured_max_open_positions": configured_max_open,
            "risk_max_open_positions": risk_max_open,
            "max_trade_pct": round(max_trade_pct, 3),
            "max_allocated_pct": round(max_allocated_pct, 3),
            "max_portfolio_heat_pct": round(max_portfolio_heat_pct, 3),
            "daily_loss_halt_usd": round(daily_halt_usd, 2),
            "daily_loss_halt_pct": round(daily_halt_pct, 3),
            "effective_daily_loss_halt_usd": round(effective_daily_halt_usd, 2),
            "tp1_r": round(float(config.get("scalp.tp1_r", 1.0)), 3),
            "tp2_r": round(float(config.get("scalp.reward_r", 2.0)), 3),
            "execution_enabled": bool(config.get("scalp.execution_enabled", False)),
        },
        "learning": learning,
    }
    with _cache_lock:
        _cache_ts = now
        _cache_value = result
    return result


def _learning_snapshot() -> dict[str, Any]:
    from agent.scalp.store import learning_dashboard_data

    data = learning_dashboard_data(
        context_limit=100, action_limit=30, outcome_limit=30
    )
    stats = data["contexts"]
    now = datetime.now(timezone.utc)
    active = [
        row for row in stats
        if row.get("gate_state") != "ALLOW" and not _expired(row.get("expires_at"), now)
    ]
    gates = {row.get("context_key"): row.get("gate_state", "ALLOW") for row in stats}
    outcomes = [
        {**row, "context_gate": gates.get(row.get("context_key"), "ALLOW")}
        for row in data["recent_outcomes"]
    ]
    champion = _decode_ml_metadata(data.get("ml_champion"))
    evaluations = [
        _decode_ml_metadata(row) for row in data.get("ml_evaluations", [])
    ]
    return {
        "active_actions": active,
        "recent_actions": data["recent_actions"],
        "recent_outcomes": outcomes,
        "contexts": stats,
        "ml_champion": champion,
        "ml_evaluations": evaluations,
        "outcome_count": int((data.get("counts") or {}).get("outcomes") or 0),
        "context_count": int((data.get("counts") or {}).get("contexts") or 0),
        "action_count": int((data.get("counts") or {}).get("actions") or 0),
    }


def _enrich_plan(plan: dict[str, Any], prices: dict[str, dict] | None = None) -> dict[str, Any]:
    from agent.scalp.indicators import provisional_live_indicators

    result = dict(plan)
    entry = float(result.get("entry") or 0.0)
    stop = float(result.get("stop_loss") or 0.0)
    tp1 = float(result.get("tp1") or 0.0)
    tp2 = float(result.get("tp2") or 0.0)
    risk = float(result.get("risk_per_share") or abs(entry - stop))
    ticker = str(result.get("ticker") or "").upper()
    quote = (prices or {}).get(ticker) or {}
    live_price = _as_float(quote.get("last") or quote.get("mark"))
    quote_ts = _as_float(quote.get("updated_at"))
    quote_age_ms = int(max(0.0, time.time() - quote_ts) * 1000) if quote_ts > 0 else -1
    source_status = str(quote.get("source_status") or "UNKNOWN").upper()
    live_sources = {"LIVE", "WS_LIVE"}
    rest_sources = {"REST_FALLBACK", "FALLBACK"}
    fresh_quote = (
        live_price > 0
        and 0 <= quote_age_ms <= 2_500
        and source_status in live_sources | rest_sources
    )
    live = provisional_live_indicators(result, live_price) if fresh_quote else None
    indicator_mode = (
        "PROVISIONAL_LIVE"
        if live and source_status in live_sources
        else "PROVISIONAL_REST"
        if live
        else "CLOSED_1M"
    )
    result.update(
        state=_plan_state(result),
        display_reason=_plan_reason(result),
        stop_r=round(abs(stop - entry) / risk, 4) if risk > 0 else 0.0,
        tp1_r=round(abs(tp1 - entry) / risk, 4) if risk > 0 else 0.0,
        tp2_r=round(abs(tp2 - entry) / risk, 4) if risk > 0 else 0.0,
        live_price=round(live_price, 4) if live_price > 0 else 0.0,
        live_bid=round(_as_float(quote.get("bid")), 4),
        live_ask=round(_as_float(quote.get("ask")), 4),
        live_price_source=source_status,
        live_price_age_ms=quote_age_ms,
        live_rsi_14=(live or {}).get("rsi_14"),
        live_rsi_7=(live or {}).get("rsi_7"),
        live_rsi_2=(live or {}).get("rsi_2"),
        live_macd_hist=(live or {}).get("macd_hist"),
        live_macd_slope=(live or {}).get("macd_slope"),
        indicator_mode=indicator_mode,
    )
    for key in _INDICATOR_STATE_FIELDS:
        result.pop(key, None)
    return result


def _as_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


_INDICATOR_STATE_FIELDS = (
    "indicator_close",
    "rsi_avg_gain_14", "rsi_avg_loss_14",
    "rsi_avg_gain_7", "rsi_avg_loss_7",
    "rsi_avg_gain_2", "rsi_avg_loss_2",
    "macd_fast_ema", "macd_slow_ema", "macd_signal_ema",
)


def _plan_state(plan: dict[str, Any]) -> str:
    from agent.scalp.quality import has_market_data_gap

    blockers = [str(item) for item in plan.get("blockers") or []]
    if has_market_data_gap(blockers, str(plan.get("session") or "")):
        return "DATA_GAP"
    if bool(plan.get("valid")) and plan.get("side") in {"LONG", "SHORT"}:
        return "ACTIONABLE"
    hard_prefixes = (
        "SESSION_", "EARNINGS_", "MACRO_", "LEARNING_", "BRACKET_",
        "REQUIRED_", "SPREAD_", "RVOL_", "BLOCKED_BY_", "CONTEXT_",
    )
    if blockers and any(item.startswith(hard_prefixes) for item in blockers):
        return "BLOCKED"
    if plan.get("side") in {"LONG", "SHORT"}:
        return "WATCH"
    return "BLOCKED"


def _plan_reason(plan: dict[str, Any]) -> str:
    if str(plan.get("session") or "").upper() == "CLOSED":
        return "MARKET_CLOSED_LAST_SESSION_DATA"
    if plan.get("valid"):
        reasons = plan.get("reasons") or []
        return str(reasons[0] if reasons else "VALID_SETUP")
    return str(plan.get("invalid_reason") or "NO_VALID_SETUP")


def _enrich_position(row: dict[str, Any], prices: dict[str, dict]) -> dict[str, Any]:
    position = dict(row)
    ticker = str(position.get("ticker") or "").upper()
    quote = prices.get(ticker) or {}
    current = float(quote.get("last") or quote.get("mark") or 0.0)
    entry = float(position.get("entry_price") or 0.0)
    shares = int(position.get("shares_remaining") or position.get("shares") or 0)
    direction = str(position.get("direction") or "BUY").upper()
    unrealized = 0.0
    if current > 0 and entry > 0 and shares > 0:
        unrealized = (
            (current - entry) * shares
            if direction == "BUY"
            else (entry - current) * shares
        )
    position.update(
        current_price=round(current, 4),
        unrealized_pnl=round(unrealized, 2),
        price_source=str(quote.get("source_status") or "UNKNOWN"),
        price_age_s=round(max(0.0, time.time() - float(quote.get("updated_at") or 0)), 2) if quote.get("updated_at") else None,
    )
    return position


def _expired(value: Any, now: datetime) -> bool:
    if not value:
        return False
    if isinstance(value, datetime):
        expiry = value
    else:
        try:
            expiry = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return True
    return expiry.replace(tzinfo=expiry.tzinfo or timezone.utc) <= now


def _decode_ml_metadata(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    result = dict(row)
    raw = result.get("metrics_json")
    if isinstance(raw, str):
        try:
            import json
            result["metrics"] = json.loads(raw)
        except (TypeError, ValueError):
            result["metrics"] = {}
    elif isinstance(raw, dict):
        result["metrics"] = raw
    else:
        result["metrics"] = {}
    result.pop("metrics_json", None)
    return result
