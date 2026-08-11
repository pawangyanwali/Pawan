"""Evidence gate for canonical paper execution activation."""
from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from agent.scalp.quality import CONTINUOUS_BAR_SESSIONS

_EVIDENCE_SESSIONS = CONTINUOUS_BAR_SESSIONS
_cache_lock = threading.Lock()
_cache: tuple[float, dict[str, Any]] | None = None


def execution_activation_report(*, force: bool = False) -> dict[str, Any]:
    global _cache
    now = time.time()
    with _cache_lock:
        if not force and _cache and now - _cache[0] < 60:
            return dict(_cache[1])
    report = _build_report()
    with _cache_lock:
        _cache = (now, report)
    return dict(report)


def _build_report() -> dict[str, Any]:
    from agent.config_manager import config
    from agent.db import get_conn
    from agent.scalp.store import init_scalp_tables

    init_scalp_tables()
    required_days = max(5, int(config.get("scalp_activation.required_market_days", 5)))
    min_cycles = max(5, int(config.get("scalp_activation.min_cycles_per_day", 300)))
    max_p95_ms = max(1000.0, float(config.get("scalp_activation.max_cycle_p95_ms", 8000)))
    max_gap_pct = max(0.0, float(config.get("scalp_activation.max_data_gap_pct", 5.0)))
    min_fresh_pct = min(100.0, float(config.get("scalp_activation.min_quote_coverage_pct", 95.0)))
    min_trials = max(30, int(config.get("scalp_activation.min_canonical_trials", 100)))
    min_expectancy = float(config.get("scalp_activation.min_expectancy_r", 0.0))
    min_pf = max(1.0, float(config.get("scalp_activation.min_profit_factor", 1.10)))
    since = (datetime.now(timezone.utc) - timedelta(days=21)).isoformat()
    with get_conn(read_only=True) as conn:
        cycles = conn.execute(
            """SELECT bucket_ts, session, universe_total, data_gap_count,
                      execution_universe_total, execution_data_gap_count,
                      cycle_ms, live_count, rest_count, stale_count
               FROM scalp_cycle_metrics WHERE bucket_ts >= ?
               ORDER BY bucket_ts""",
            (since,),
        ).fetchall()
        trials = conn.execute(
            """SELECT pnl_r FROM scalp_candidate_trials
               WHERE candidate_type='CANONICAL_VALID' AND status='CLOSED'
                 AND episode_version>=2"""
        ).fetchall()

    eastern = ZoneInfo("America/New_York")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in cycles:
        row = dict(raw)
        if str(row.get("session") or "").upper() not in _EVIDENCE_SESSIONS:
            continue
        stamp = _as_datetime(row.get("bucket_ts"))
        if stamp is None:
            continue
        grouped.setdefault(stamp.astimezone(eastern).date().isoformat(), []).append(row)

    market_dates = _required_market_dates(grouped, required_days)
    days = []
    for market_date in market_dates:
        rows = grouped.get(market_date, [])
        latencies = sorted(float(row.get("cycle_ms") or 0.0) for row in rows)
        p95 = latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)] if latencies else 0.0
        monitored_universe = sum(
            max(0, int(row.get("universe_total") or 0)) for row in rows
        )
        execution_universe = sum(
            max(0, int(row.get("execution_universe_total") or 0)) for row in rows
        )
        gaps = sum(
            max(0, int(row.get("execution_data_gap_count") or 0)) for row in rows
        )
        fresh = sum(
            max(0, int(row.get("live_count") or 0))
            + max(0, int(row.get("rest_count") or 0))
            for row in rows
        )
        gap_pct = gaps / execution_universe * 100.0 if execution_universe else 100.0
        fresh_pct = fresh / monitored_universe * 100.0 if monitored_universe else 0.0
        passed = (
            len(rows) >= min_cycles
            and p95 <= max_p95_ms
            and gap_pct <= max_gap_pct
            and fresh_pct >= min_fresh_pct
        )
        days.append({
            "market_date": market_date,
            "cycles": len(rows),
            "cycle_p95_ms": round(p95, 1),
            "execution_universe_observations": execution_universe,
            "data_gap_pct": round(gap_pct, 2),
            "quote_coverage_pct": round(fresh_pct, 2),
            "passed": passed,
        })

    pnl = [float(dict(row).get("pnl_r") or 0.0) for row in trials]
    wins = sum(value for value in pnl if value > 0)
    losses = abs(sum(value for value in pnl if value < 0))
    expectancy = sum(pnl) / len(pnl) if pnl else 0.0
    profit_factor = wins / losses if losses > 0 else (float("inf") if wins > 0 else 0.0)
    operational_ready = len(days) == required_days and all(day["passed"] for day in days)
    statistical_ready = (
        len(pnl) >= min_trials
        and expectancy > min_expectancy
        and profit_factor >= min_pf
    )
    reasons = []
    if not operational_ready:
        reasons.append("FIVE_CONSECUTIVE_MARKET_DAYS_NOT_SLA_COMPLIANT")
    if not statistical_ready:
        reasons.append("CANONICAL_EVIDENCE_NOT_STATISTICALLY_READY")
    return {
        "ready": operational_ready and statistical_ready,
        "required_market_days": required_days,
        "days": days,
        "operational_ready": operational_ready,
        "statistical_ready": statistical_ready,
        "canonical_trials": len(pnl),
        "canonical_evidence_contract": "INDEPENDENT_EPISODE_V2",
        "expectancy_r": round(expectancy, 4),
        "profit_factor": round(profit_factor, 4) if math.isfinite(profit_factor) else None,
        "thresholds": {
            "min_cycles_per_day": min_cycles,
            "max_cycle_p95_ms": max_p95_ms,
            "max_data_gap_pct": max_gap_pct,
            "min_quote_coverage_pct": min_fresh_pct,
            "min_canonical_trials": min_trials,
            "min_expectancy_r": min_expectancy,
            "min_profit_factor": min_pf,
        },
        "reasons": reasons,
    }


def _as_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=value.tzinfo or timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)
    except (TypeError, ValueError):
        return None


def _required_market_dates(
    grouped: dict[str, list[dict[str, Any]]], required_days: int
) -> list[str]:
    del grouped
    cursor = _latest_completed_market_date()
    result = []
    from agent.market_hours import _is_holiday

    while len(result) < required_days:
        if cursor.weekday() < 5 and not _is_holiday(cursor):
            result.append(cursor.isoformat())
        cursor -= timedelta(days=1)
    return list(reversed(result))


def _latest_completed_market_date(*, now: datetime | None = None):
    """Return the latest market date with a complete 4am-8pm evidence window."""
    from agent.market_hours import _is_holiday

    eastern = ZoneInfo("America/New_York")
    local = (now or datetime.now(timezone.utc)).astimezone(eastern)
    cursor = local.date()
    if local.hour < 20:
        cursor -= timedelta(days=1)
    while cursor.weekday() >= 5 or _is_holiday(cursor):
        cursor -= timedelta(days=1)
    return cursor
