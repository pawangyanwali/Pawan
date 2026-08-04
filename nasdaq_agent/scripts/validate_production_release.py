#!/usr/bin/env python3
"""Functional post-deploy gate for the scalp-only production runtime."""
from __future__ import annotations

import json
import sys
import time


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

    if bool(config.get("scalp.execution_enabled", False)):
        failures.append("canonical paper execution must remain disabled")

    tickers = get_runtime_universe()
    details["universe_total"] = len(tickers)
    if len(tickers) < 400:
        failures.append(f"runtime universe too small: {len(tickers)}")

    snapshot = read_latest() or {}
    snapshot_age = time.time() - float(snapshot.get("ts") or 0.0)
    details["snapshot_age_s"] = round(snapshot_age, 1)
    details["snapshot_runtime"] = snapshot.get("runtime")
    details["snapshot_universe"] = int(snapshot.get("universe_total") or 0)
    if snapshot.get("runtime") != "SCALP_ONLY_V1":
        failures.append("latest snapshot is not SCALP_ONLY_V1")
    if snapshot_age < 0 or snapshot_age > 60:
        failures.append(f"scalp snapshot stale: {snapshot_age:.1f}s")
    if int(snapshot.get("universe_total") or 0) < 400:
        failures.append("scalp snapshot does not cover the production universe")

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
    if float(prices.get("trusted_fresh_pct") or 0.0) < 80.0:
        failures.append("trusted five-second quote coverage is below 80%")

    market_data = get_state("market-data:status") or {}
    ws = market_data.get("ws_streamer") or {}
    details["ws"] = {
        key: ws.get(key)
        for key in (
            "connected", "desired_subscriptions", "acknowledged_subscriptions",
            "subscription_coverage_pct", "pending_subscription_requests",
        )
    }
    if not bool(ws.get("connected")):
        failures.append("Schwab WebSocket is not connected")
    if int(ws.get("desired_subscriptions") or 0) < 400:
        failures.append("Schwab desired subscription universe is below 400")
    if float(ws.get("subscription_coverage_pct") or 0.0) < 95.0:
        failures.append("Schwab subscription ACK coverage is below 95%")

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

    details["ok"] = not failures
    details["failures"] = failures
    print(json.dumps(details, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
