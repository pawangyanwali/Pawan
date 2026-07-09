"""Daily diagnostics for isolated shadow scalp execution.

The shadow ledger is intentionally separate from canonical paper outcomes, but
it still needs first-class post-session analysis.  This module turns raw shadow
trades and policy decisions into a durable daily report that the UI can show
without somebody manually querying PostgreSQL.
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from agent.db import get_conn

from .store import init_scalp_tables

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
_GROUP_VALUE_FIELDS = {
    "setup": "setup_type",
    "side": "side",
    "session": "session",
    "exit_reason": "exit_reason",
    "rsi_zone": "rsi_zone",
    "vwap_event": "vwap_event",
    "trigger_source": "trigger_source",
}


def generate_shadow_daily_report(
    market_date: date | str | None = None,
    *,
    persist: bool = True,
) -> dict[str, Any]:
    """Build and optionally persist one ET-date shadow execution report."""
    init_scalp_tables()
    target_date = _coerce_date(market_date)
    start_utc, end_utc = _et_bounds(target_date)

    trades = _rows(
        """
        SELECT * FROM scalp_shadow_trades
        WHERE (opened_at >= ? AND opened_at < ?)
           OR (closed_at >= ? AND closed_at < ?)
        ORDER BY opened_at ASC
        """,
        (
            start_utc.isoformat(),
            end_utc.isoformat(),
            start_utc.isoformat(),
            end_utc.isoformat(),
        ),
    )
    decisions = _rows(
        """
        SELECT decided_at, decision, reason, detail_json
        FROM scalp_execution_decisions
        WHERE decided_at >= ? AND decided_at < ?
        ORDER BY decided_at ASC
        """,
        (start_utc.isoformat(), end_utc.isoformat()),
    )
    learning_actions = _rows(
        """
        SELECT action_ts, context_key, action_type, old_state, new_state,
               old_value, new_value, reason, expires_at
        FROM scalp_learning_actions
        WHERE action_ts >= ? AND action_ts < ?
        ORDER BY action_ts DESC
        LIMIT 100
        """,
        (start_utc.isoformat(), end_utc.isoformat()),
    )

    for trade in trades:
        _enrich_trade(trade)

    closed = [row for row in trades if str(row.get("status", "")).upper() == "CLOSED"]
    overall = _summarize(trades)
    report = {
        "schema_version": 1,
        "market_date": target_date.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {
            "timezone": "America/New_York",
            "start_utc": start_utc.isoformat(),
            "end_utc": end_utc.isoformat(),
        },
        "status": _status(overall),
        "summary": overall,
        "groups": {
            "setup": _group_stats(closed, "setup_type"),
            "side": _group_stats(closed, "side"),
            "session": _group_stats(closed, "session"),
            "exit_reason": _group_stats(closed, "exit_reason"),
            "rsi_zone": _group_stats(closed, "rsi_zone"),
            "vwap_event": _group_stats(closed, "vwap_event"),
            "trigger_source": _group_stats(closed, "trigger_source"),
        },
        "decisions": _decision_summary(decisions),
        "learning": {
            "actions_count": len(learning_actions),
            "recent_actions": [_clean(row) for row in learning_actions[:20]],
        },
        "exit_calibration": _exit_calibration(closed),
        "worst_trades": _worst_trades(closed),
        "findings": [],
        "recommendations": [],
    }
    report["findings"], report["recommendations"] = _diagnose(report)
    report["autonomous_diagnosis"] = _autonomous_diagnosis(report)
    report["findings"] = _dedupe(
        report["findings"]
        + [
            item.get("finding", "")
            for item in report["autonomous_diagnosis"].get("actions", [])
            if item.get("finding")
        ]
    )
    report["recommendations"] = _dedupe(
        report["recommendations"]
        + [
            item.get("recommendation", "")
            for item in report["autonomous_diagnosis"].get("actions", [])
            if item.get("recommendation")
        ]
    )

    if persist:
        _persist_report(report)
    return report


def generate_recent_shadow_reports(days: int = 7) -> list[dict[str, Any]]:
    """Backfill recent weekday ET reports, newest first."""
    today = datetime.now(ET).date()
    reports = []
    cursor = today
    target_count = max(1, int(days))
    while len(reports) < target_count:
        if cursor.weekday() < 5:
            reports.append(generate_shadow_daily_report(cursor))
        cursor -= timedelta(days=1)
    return reports


def latest_shadow_daily_reports(limit: int = 5) -> list[dict[str, Any]]:
    """Read latest persisted daily shadow diagnostics."""
    init_scalp_tables()
    rows = _rows(
        """
        SELECT report_json FROM scalp_shadow_daily_reports
        ORDER BY market_date DESC LIMIT ?
        """,
        (max(1, min(int(limit), 30)),),
    )
    reports = []
    for row in rows:
        try:
            reports.append(json.loads(row.get("report_json") or "{}"))
        except Exception:
            continue
    return reports


def shadow_daily_reports_for_range(
    start_date: date | str,
    end_date: date | str,
    *,
    generate_missing: bool = True,
    include_weekends: bool = False,
    max_days: int = 45,
) -> list[dict[str, Any]]:
    """Read or build reports for an inclusive ET date range, newest first."""
    init_scalp_tables()
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    if start > end:
        start, end = end, start
    span_days = (end - start).days + 1
    if span_days > max(1, int(max_days)):
        raise ValueError(f"shadow report range is limited to {max_days} days")

    market_dates = [
        (start + timedelta(days=offset)).isoformat()
        for offset in range(span_days)
        if include_weekends or (start + timedelta(days=offset)).weekday() < 5
    ]
    if not market_dates:
        return []
    existing = _read_reports_by_date(market_dates)
    if generate_missing:
        for market_date in market_dates:
            if market_date not in existing:
                existing[market_date] = generate_shadow_daily_report(market_date)
    return [
        existing[market_date]
        for market_date in sorted(market_dates, reverse=True)
        if market_date in existing
    ]


def _coerce_date(value: date | str | None) -> date:
    if value is None:
        return datetime.now(ET).date()
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.astimezone(ET).date() if value.tzinfo else value.date()
    return date.fromisoformat(str(value)[:10])


def _read_reports_by_date(market_dates: list[str]) -> dict[str, dict[str, Any]]:
    if not market_dates:
        return {}
    placeholders = ",".join("?" for _ in market_dates)
    rows = _rows(
        f"""
        SELECT market_date, report_json FROM scalp_shadow_daily_reports
        WHERE market_date IN ({placeholders})
        """,
        tuple(market_dates),
    )
    reports: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            reports[str(row.get("market_date"))] = json.loads(row.get("report_json") or "{}")
        except Exception:
            continue
    return reports


def _et_bounds(market_date: date) -> tuple[datetime, datetime]:
    start = datetime(
        market_date.year,
        market_date.month,
        market_date.day,
        tzinfo=ET,
    )
    end = start + timedelta(days=1)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def _rows(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with get_conn(read_only=True) as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def _execute(sql: str, params: tuple[Any, ...]) -> None:
    with get_conn() as conn:
        conn.execute(sql, params)


def _safe_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _enrich_trade(trade: dict[str, Any]) -> None:
    plan = _safe_json(trade.get("plan_json"))
    trade["_plan"] = plan
    trade["setup_type"] = trade.get("setup_type") or plan.get("setup_type") or "UNKNOWN"
    trade["rsi_zone"] = plan.get("rsi_zone") or "UNKNOWN"
    trade["vwap_event"] = plan.get("vwap_event") or "UNKNOWN"
    trade["atr_bucket"] = plan.get("atr_bucket") or "UNKNOWN"
    trade["data_source"] = plan.get("source") or plan.get("data_source") or "UNKNOWN"
    confidence = _float(plan.get("confidence"))
    if confidence >= 80:
        bucket = ">=80"
    elif confidence >= 70:
        bucket = "70-79"
    elif confidence >= 60:
        bucket = "60-69"
    else:
        bucket = "<60"
    trade["confidence_bucket"] = bucket


def _summarize(trades: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [row for row in trades if str(row.get("status", "")).upper() == "CLOSED"]
    open_rows = [row for row in trades if str(row.get("status", "")).upper() == "OPEN"]
    wins = [row for row in closed if _float(row.get("pnl_r")) > 0]
    losses = [row for row in closed if _float(row.get("pnl_r")) <= 0]
    gross_win = sum(max(0.0, _float(row.get("pnl_r"))) for row in closed)
    gross_loss = abs(sum(min(0.0, _float(row.get("pnl_r"))) for row in closed))
    return {
        "opened": len(trades),
        "closed": len(closed),
        "open": len(open_rows),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": _round(len(wins) / len(closed) * 100.0) if closed else 0.0,
        "pnl_r": _round(sum(_float(row.get("pnl_r")) for row in closed), 4),
        "expectancy_r": _round(
            sum(_float(row.get("pnl_r")) for row in closed) / len(closed), 4
        )
        if closed
        else 0.0,
        "pnl_dollar": _round(sum(_float(row.get("pnl_dollar")) for row in closed), 2),
        "avg_win_r": _round(
            sum(_float(row.get("pnl_r")) for row in wins) / len(wins), 4
        )
        if wins
        else 0.0,
        "avg_loss_r": _round(
            sum(_float(row.get("pnl_r")) for row in losses) / len(losses), 4
        )
        if losses
        else 0.0,
        "avg_win_dollar": _round(
            sum(_float(row.get("pnl_dollar")) for row in wins) / len(wins), 2
        )
        if wins
        else 0.0,
        "avg_loss_dollar": _round(
            sum(_float(row.get("pnl_dollar")) for row in losses) / len(losses), 2
        )
        if losses
        else 0.0,
        "profit_factor": _round(gross_win / gross_loss, 3)
        if gross_loss
        else (999.0 if gross_win else 0.0),
        "tp1_hit_rate_pct": _round(
            sum(_int(row.get("t1_hit")) for row in closed) / len(closed) * 100.0
        )
        if closed
        else 0.0,
        "tp2_hit_rate_pct": _round(
            sum(_int(row.get("t2_hit")) for row in closed) / len(closed) * 100.0
        )
        if closed
        else 0.0,
        "avg_mfe_r": _round(
            sum(_float(row.get("mfe_r")) for row in closed) / len(closed), 4
        )
        if closed
        else 0.0,
        "avg_mae_r": _round(
            sum(_float(row.get("mae_r")) for row in closed) / len(closed), 4
        )
        if closed
        else 0.0,
        "avg_exit_spread_bps": _round(
            sum(_float(row.get("exit_spread_bps")) for row in closed) / len(closed), 3
        )
        if closed
        else 0.0,
        "avg_exit_slippage_bps": _round(
            sum(_float(row.get("exit_slippage_bps")) for row in closed) / len(closed), 3
        )
        if closed
        else 0.0,
    }


def _group_stats(
    trades: list[dict[str, Any]], key: str, limit: int = 10
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        groups[str(trade.get(key) or "UNKNOWN")].append(trade)
    result = []
    for name, values in groups.items():
        summary = _summarize(values)
        summary[key] = name
        result.append(summary)
    return sorted(
        result,
        key=lambda item: (item.get("closed", 0), abs(item.get("pnl_r", 0.0))),
        reverse=True,
    )[:limit]


def _decision_summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(str(row.get("decision") or "UNKNOWN") for row in decisions)
    by_reason = Counter(
        f"{row.get('decision') or 'UNKNOWN'}:{row.get('reason') or 'UNKNOWN'}"
        for row in decisions
    )
    return {
        "total": len(decisions),
        "by_type": dict(by_type.most_common()),
        "by_reason": dict(by_reason.most_common(20)),
    }


def _exit_calibration(closed: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(closed)
    tp1_hits = sum(_int(row.get("t1_hit")) for row in closed)
    tp2_hits = sum(_int(row.get("t2_hit")) for row in closed)
    stop_exits = sum(
        1
        for row in closed
        if str(row.get("exit_reason") or "").upper() in {"STOP", "STOP_HIT"}
    )
    trail_exits = sum(
        1
        for row in closed
        if "TRAIL" in str(row.get("exit_reason") or "").upper()
    )
    breakeven_like = sum(
        1
        for row in closed
        if _int(row.get("t1_hit"))
        and not _int(row.get("t2_hit"))
        and abs(_float(row.get("pnl_r"))) < 0.25
    )
    return {
        "closed": total,
        "tp1_hits": tp1_hits,
        "tp2_hits": tp2_hits,
        "tp1_to_tp2_conversion_pct": _round(tp2_hits / tp1_hits * 100.0)
        if tp1_hits
        else 0.0,
        "tp1_without_tp2": max(0, tp1_hits - tp2_hits),
        "stop_exits": stop_exits,
        "trail_exits": trail_exits,
        "breakeven_like_after_tp1": breakeven_like,
        "avg_mfe_r": _round(
            sum(_float(row.get("mfe_r")) for row in closed) / total, 4
        )
        if total
        else 0.0,
        "avg_mae_r": _round(
            sum(_float(row.get("mae_r")) for row in closed) / total, 4
        )
        if total
        else 0.0,
    }


def _worst_trades(closed: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    result = []
    for trade in sorted(closed, key=lambda row: _float(row.get("pnl_r")))[:limit]:
        plan = trade.get("_plan") or {}
        result.append(
            {
                "ticker": trade.get("ticker"),
                "side": trade.get("side"),
                "setup_type": trade.get("setup_type"),
                "session": trade.get("session"),
                "opened_at": _clean(trade.get("opened_at")),
                "closed_at": _clean(trade.get("closed_at")),
                "exit_reason": trade.get("exit_reason"),
                "pnl_r": _round(_float(trade.get("pnl_r")), 4),
                "pnl_dollar": _round(_float(trade.get("pnl_dollar")), 2),
                "mfe_r": _round(_float(trade.get("mfe_r")), 4),
                "mae_r": _round(_float(trade.get("mae_r")), 4),
                "t1_hit": _int(trade.get("t1_hit")),
                "t2_hit": _int(trade.get("t2_hit")),
                "confidence": _round(_float(plan.get("confidence")), 2),
                "rsi_14": _round(_float(plan.get("rsi_14")), 2),
                "rsi_7": _round(_float(plan.get("rsi_7")), 2),
                "rsi_2": _round(_float(plan.get("rsi_2")), 2),
                "vwap_event": plan.get("vwap_event") or "UNKNOWN",
                "rvol": _round(_float(plan.get("rvol")), 3),
                "reasons": list(plan.get("reasons") or [])[:8],
            }
        )
    return result


def _diagnose(report: dict[str, Any]) -> tuple[list[str], list[str]]:
    summary = report["summary"]
    findings: list[str] = []
    recommendations: list[str] = []
    closed = int(summary.get("closed") or 0)
    if closed == 0:
        findings.append("No closed shadow trades were recorded for this ET date.")
        recommendations.append(
            "Confirm whether the date was a market holiday or whether data/auth blocked signal generation."
        )
        return findings, recommendations

    expectancy = _float(summary.get("expectancy_r"))
    profit_factor = _float(summary.get("profit_factor"))
    win_rate = _float(summary.get("win_rate_pct"))
    if expectancy < 0:
        findings.append(
            f"Shadow expectancy was negative at {expectancy:.3f}R with {win_rate:.1f}% win rate."
        )
        recommendations.append(
            "Keep canonical paper execution disabled until the daily shadow report is non-negative for multiple active sessions."
        )
    elif profit_factor < 1.2:
        findings.append(
            f"Shadow expectancy was slightly positive but fragile: PF {profit_factor:.2f}."
        )
        recommendations.append(
            "Treat this as observation-only; require stronger profit factor before activation."
        )
    else:
        findings.append(
            f"Shadow validation was positive: expectancy {expectancy:.3f}R, PF {profit_factor:.2f}."
        )

    stop_group = _find_group(report, "exit_reason", "STOP")
    if stop_group and _float(stop_group.get("pnl_r")) < 0:
        findings.append(
            f"Pre-TP1 stops drove {stop_group['closed']} trades and {stop_group['pnl_r']:.2f}R."
        )
        recommendations.append(
            "Tighten entry confirmation before allowing immediate entries; most damage comes before TP1."
        )

    if _float(summary.get("tp1_hit_rate_pct")) < 35.0:
        findings.append(
            f"TP1 conversion was weak at {summary['tp1_hit_rate_pct']:.1f}%."
        )
        recommendations.append(
            "Require one extra confirmation for scalp entries when TP1 hit rate is below 35%."
        )
    if _float(summary.get("tp2_hit_rate_pct")) < 10.0:
        findings.append(
            f"TP2 conversion was very low at {summary['tp2_hit_rate_pct']:.1f}%."
        )
        recommendations.append(
            "Use TP1/trailing statistics as the main profitability signal until TP2 conversion improves."
        )
    exit_cal = report.get("exit_calibration") or {}
    if int(exit_cal.get("tp1_hits") or 0) >= 3 and _float(
        exit_cal.get("tp1_to_tp2_conversion_pct")
    ) < 20.0:
        findings.append(
            "TP1 hit but TP2 rarely converted: "
            f"{exit_cal.get('tp1_to_tp2_conversion_pct', 0):.1f}% of TP1 hits reached TP2."
        )
        recommendations.append(
            "Review TP2 distance and trailing capture by setup before allowing canonical execution."
        )

    for key in ("setup", "session", "vwap_event", "rsi_zone"):
        bad = _worst_group(report, key)
        if bad:
            bad_value = _group_value(bad, key)
            findings.append(
                f"Worst {key}: {bad_value} ({bad['closed']} trades, {bad['expectancy_r']:.3f}R EV)."
            )
            recommendations.append(
                f"Shadow-tighten or block {key}={bad_value} until it recovers in later reports."
            )

    if _float(summary.get("avg_exit_slippage_bps")) > 25.0:
        findings.append(
            f"Average exit slippage was high at {summary['avg_exit_slippage_bps']:.1f} bps."
        )
        recommendations.append(
            "Audit executable bid/ask fills and avoid activation during degraded liquidity."
        )
    if int((report.get("learning") or {}).get("actions_count") or 0) == 0:
        findings.append("No durable learning actions were created for this date.")
        recommendations.append(
            "Promote this report into an automated risk recommendation feed before enabling self-mutation."
        )

    return _dedupe(findings), _dedupe(recommendations)


def _autonomous_diagnosis(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") or {}
    exit_cal = report.get("exit_calibration") or {}
    closed = int(summary.get("closed") or 0)
    actions: list[dict[str, Any]] = []
    severity = "PASS"

    def add(action: str, finding: str, recommendation: str, sev: str = "WARN") -> None:
        nonlocal severity
        order = {"PASS": 0, "INFO": 1, "WARN": 2, "CRITICAL": 3}
        if order.get(sev, 0) > order.get(severity, 0):
            severity = sev
        actions.append(
            {
                "action": action,
                "severity": sev,
                "finding": finding,
                "recommendation": recommendation,
            }
        )

    if closed == 0:
        add(
            "CHECK_DATA_SIGNAL_PIPELINE",
            "No closed shadow trades were available for diagnosis.",
            "Verify market session, Schwab data, and policy rejections before changing strategy.",
            "WARN",
        )
        return {
            "severity": severity,
            "summary": "No trade sample available.",
            "actions": actions,
        }

    pnl_r = _float(summary.get("pnl_r"))
    pnl_dollar = _float(summary.get("pnl_dollar"))
    expectancy = _float(summary.get("expectancy_r"))
    if pnl_dollar < 0 <= pnl_r:
        add(
            "DOLLAR_GUARD",
            f"R outcome was non-negative ({pnl_r:.2f}R) but real dollar P&L was ${pnl_dollar:.2f}.",
            "Keep dollar-aware context reduction enabled so same-context trades shrink when fills or sizing lose money.",
            "WARN",
        )
    elif expectancy < 0 or pnl_dollar < 0:
        add(
            "KEEP_SHADOW_ONLY",
            f"Daily shadow expectancy was {expectancy:.3f}R with dollar P&L ${pnl_dollar:.2f}.",
            "Keep canonical execution disabled and let learning gates tighten the losing contexts.",
            "CRITICAL",
        )

    if int(exit_cal.get("tp1_hits") or 0) >= 3 and _float(
        exit_cal.get("tp1_to_tp2_conversion_pct")
    ) < 20.0:
        add(
            "TP2_TRAIL_CALIBRATION",
            f"Only {exit_cal.get('tp1_to_tp2_conversion_pct', 0):.1f}% of TP1 hits reached TP2.",
            "Compare setup-level MFE against TP2 distance and prefer trailing capture until TP2 is statistically reachable.",
            "WARN",
        )

    if int(exit_cal.get("stop_exits") or 0) >= 3:
        add(
            "ENTRY_CLUSTER_THROTTLE",
            f"{exit_cal.get('stop_exits')} trades exited at the initial stop.",
            "Use same-context cluster throttling and one extra confirmation after clustered pre-TP1 stops.",
            "WARN",
        )

    if int((report.get("learning") or {}).get("actions_count") or 0) == 0:
        add(
            "VERIFY_LEARNING_MUTATION",
            "No expiring learning action was written during this session.",
            "If bad trades occurred, verify context keys, outcome persistence, and learning thresholds.",
            "WARN",
        )

    if not actions:
        add(
            "CONTINUE_OBSERVATION",
            "No material defect pattern exceeded autonomous thresholds.",
            "Continue shadow observation and require multiple positive sessions before promotion.",
            "INFO",
        )

    return {
        "severity": severity,
        "summary": actions[0]["finding"],
        "actions": actions,
    }


def _find_group(report: dict[str, Any], group: str, value: str) -> dict[str, Any] | None:
    for row in ((report.get("groups") or {}).get(group) or []):
        if str(row.get(group) or "").upper() == value.upper():
            return row
    return None


def _worst_group(report: dict[str, Any], group: str) -> dict[str, Any] | None:
    rows = [
        row for row in ((report.get("groups") or {}).get(group) or [])
        if int(row.get("closed") or 0) >= 3 and _float(row.get("expectancy_r")) < 0
    ]
    if not rows:
        return None
    return sorted(rows, key=lambda row: _float(row.get("expectancy_r")))[0]


def _group_value(row: dict[str, Any], group: str) -> str:
    field = _GROUP_VALUE_FIELDS.get(group, group)
    return str(row.get(field) or row.get(group) or "UNKNOWN")


def _persist_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    payload = json.dumps(_clean(report), separators=(",", ":"), sort_keys=True)
    _execute(
        """
        INSERT INTO scalp_shadow_daily_reports
          (market_date, generated_at, status, closed_count, win_rate_pct,
           expectancy_r, pnl_r, pnl_dollar, report_json)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT (market_date) DO UPDATE SET
          generated_at=excluded.generated_at,
          status=excluded.status,
          closed_count=excluded.closed_count,
          win_rate_pct=excluded.win_rate_pct,
          expectancy_r=excluded.expectancy_r,
          pnl_r=excluded.pnl_r,
          pnl_dollar=excluded.pnl_dollar,
          report_json=excluded.report_json
        """,
        (
            report["market_date"],
            report["generated_at"],
            report["status"],
            int(summary.get("closed") or 0),
            _float(summary.get("win_rate_pct")),
            _float(summary.get("expectancy_r")),
            _float(summary.get("pnl_r")),
            _float(summary.get("pnl_dollar")),
            payload,
        ),
    )


def _status(summary: dict[str, Any]) -> str:
    if int(summary.get("closed") or 0) == 0:
        return "NO_TRADES"
    expectancy = _float(summary.get("expectancy_r"))
    profit_factor = _float(summary.get("profit_factor"))
    if expectancy < 0:
        return "NEGATIVE"
    if expectancy < 0.05 or profit_factor < 1.2:
        return "FRAGILE"
    return "POSITIVE"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value if value is not None else default)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _round(value: float, digits: int = 2) -> float:
    return round(_float(value), digits)


def _clean(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, tuple):
        return [_clean(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
