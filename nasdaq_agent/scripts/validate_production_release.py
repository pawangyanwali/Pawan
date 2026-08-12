#!/usr/bin/env python3
"""Functional post-deploy gate for the scalp-only production runtime."""
from __future__ import annotations

import json
import sys
import time


_CLOSED_SESSIONS = {"CLOSED", "WEEKEND", "HOLIDAY", "UNKNOWN"}


def _validate_schwab_health(
    *,
    active_session: bool,
    market_data: dict,
    token_statuses: dict[str, dict],
) -> tuple[dict[str, object], list[str]]:
    """Validate durable auth always and live streaming only when it is expected."""
    failures: list[str] = []
    ws = market_data.get("ws_streamer") or {}
    details: dict[str, object] = {
        "ws": {
            key: ws.get(key)
            for key in (
                "connected", "desired_subscriptions", "acknowledged_subscriptions",
                "subscription_coverage_pct", "pending_subscription_requests",
            )
        },
        "schwab_tokens": {},
    }

    if not market_data:
        failures.append("market-data status unavailable")

    desired = int(ws.get("desired_subscriptions") or 0)
    if desired < 400:
        failures.append("Schwab desired subscription universe is below 400")

    for app in ("trader", "marketdata"):
        status = token_statuses.get(app) or {}
        connected = bool(status.get("connected"))
        details["schwab_tokens"][app] = {
            "connected": connected,
            "access_token_ttl_s": int(status.get("access_token_ttl_s") or 0),
            "refresh_token_ttl_s": int(status.get("refresh_token_ttl_s") or 0),
            "source": status.get("_source"),
        }
        if not connected:
            failures.append(f"Schwab {app} token is not connected")

    # Schwab streaming may be intentionally offline overnight. During every
    # tradable session, however, the release remains fail-closed on both the
    # transport and the acknowledged subscription universe.
    if active_session:
        if not bool(ws.get("connected")):
            failures.append("Schwab WebSocket is not connected")
        if float(ws.get("subscription_coverage_pct") or 0.0) < 95.0:
            failures.append("Schwab subscription ACK coverage is below 95%")

    return details, failures


def main() -> int:
    from agent.config_manager import config
    from agent.scalp.store import init_scalp_tables
    from agent.service_state import get_state
    from agent.signal_snapshot import read_latest
    from agent.universe_registry import get_runtime_universe
    from agent.valkey_client import _get_client, price_bus_health

    failures: list[str] = []
    details: dict[str, object] = {}
    config.load()
    init_scalp_tables()

    tickers = get_runtime_universe()
    details["universe_total"] = len(tickers)
    if len(tickers) < 400:
        failures.append(f"runtime universe too small: {len(tickers)}")

    snapshot = read_latest() or {}
    snapshot_age = time.time() - float(snapshot.get("ts") or 0.0)
    details["snapshot_age_s"] = round(snapshot_age, 1)
    details["snapshot_runtime"] = snapshot.get("runtime")
    details["snapshot_universe"] = int(snapshot.get("universe_total") or 0)
    details["cycle_ms"] = float(snapshot.get("cycle_ms") or 0.0)
    details["data_gap_count"] = int(snapshot.get("data_gap_count") or 0)
    details["execution_universe_total"] = int(
        snapshot.get("execution_universe_total") or 0
    )
    details["execution_data_gap_count"] = int(
        snapshot.get("execution_data_gap_count") or 0
    )
    if snapshot.get("runtime") != "SCALP_ONLY_V1":
        failures.append("latest snapshot is not SCALP_ONLY_V1")
    if snapshot_age < 0 or snapshot_age > 60:
        failures.append(f"scalp snapshot stale: {snapshot_age:.1f}s")
    if int(snapshot.get("universe_total") or 0) < 400:
        failures.append("scalp snapshot does not cover the production universe")
    universe_total = int(snapshot.get("universe_total") or 0)
    monitored_gap_pct = (
        int(snapshot.get("data_gap_count") or 0) / universe_total * 100.0
        if universe_total else 100.0
    )
    details["monitored_data_gap_pct"] = round(monitored_gap_pct, 2)
    session_value = snapshot.get("session")
    if isinstance(session_value, dict):
        session_value = session_value.get("session") or session_value.get("name")
    session_name = str(session_value or "UNKNOWN").upper()
    active_session = session_name not in _CLOSED_SESSIONS
    from agent.scalp.quality import continuous_bar_session
    data_gap_sla_session = continuous_bar_session(session_name)
    details["session"] = session_name
    details["active_session_sla_enforced"] = active_session
    details["data_gap_sla_enforced"] = data_gap_sla_session
    cycle_ms = float(snapshot.get("cycle_ms") or 0.0)
    execution_universe = int(snapshot.get("execution_universe_total") or 0)
    gap_pct = (
        int(snapshot.get("execution_data_gap_count") or 0)
        / execution_universe * 100.0
        if execution_universe else 100.0
    )
    details["execution_data_gap_pct"] = round(gap_pct, 2)
    if active_session and (cycle_ms <= 0 or cycle_ms > 12_000):
        failures.append(f"full-universe cycle exceeds 12s SLA: {cycle_ms:.1f}ms")
    if data_gap_sla_session and gap_pct > 5.0:
        failures.append(
            f"execution-universe data-gap rate exceeds 5%: {gap_pct:.1f}%"
        )

    prices = price_bus_health(max_age_s=5.0)
    details["price_bus"] = {
        key: prices.get(key)
        for key in (
            "status", "total", "trusted_fresh_pct", "live_pct",
            "fallback_pct", "stale",
        )
    }
    if int(prices.get("total") or 0) < 400:
        failures.append("price bus covers fewer than 400 eligible tickers")
    if active_session and float(prices.get("trusted_fresh_pct") or 0.0) < 95.0:
        failures.append("trusted five-second quote coverage is below 95%")

    market_data = get_state("market-data:status") or {}
    schwab_details, schwab_failures = _validate_schwab_health(
        active_session=active_session,
        market_data=market_data,
        token_statuses={
            "trader": get_state("schwab:token_status:trader") or {},
            "marketdata": get_state("schwab:token_status:marketdata") or {},
        },
    )
    details.update(schwab_details)
    failures.extend(schwab_failures)

    client = _get_client()
    if client is None:
        failures.append("Valkey unavailable")
    else:
        pipe = client.pipeline(transaction=False)
        for ticker in tickers:
            pipe.llen(f"md:1m:{ticker}")
        counts = [int(value or 0) for value in pipe.execute()]
        usable = sum(value >= 35 for value in counts)
        bar_coverage = usable / max(1, len(tickers)) * 100.0
        details["bar_coverage_pct"] = round(bar_coverage, 1)
        details["bar_depth_min"] = min(counts) if counts else 0
        details["bar_depth_median"] = sorted(counts)[len(counts) // 2] if counts else 0
        if bar_coverage < 95.0:
            failures.append("usable one-minute bar coverage is below 95%")

    from agent.scalp.activation import execution_activation_report
    activation = execution_activation_report(force=True)
    details["execution_activation"] = activation
    if bool(config.get("scalp.execution_enabled", False)) and not activation.get("ready"):
        failures.append("canonical execution enabled before five-day evidence gate passed")

    details["ok"] = not failures
    details["failures"] = failures
    print(json.dumps(details, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
