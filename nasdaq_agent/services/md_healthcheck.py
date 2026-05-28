#!/usr/bin/env python3
"""
market-data container health check.

Exit 0 healthy, exit 1 unhealthy.

Checks:
  1. Valkey ping.
  2. scanner:streamer status publisher freshness.
  3. md:prices trusted freshness across the whole universe during active sessions.

During CLOSED sessions the quote freshness check is skipped because live quotes
are not expected to move, but the publisher thread must still be alive.

PRE_MARKET and AFTER_HOURS are intentionally less strict than REGULAR.  Many
symbols in the 477-ticker universe do not print every few seconds outside the
regular session; restarting a healthy market-data container for that sparse
coverage creates a LIVE/FALLBACK flap on the dashboard.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/app")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _min_fresh_pct_for_session(session: str) -> float:
    session_key = (session or "CLOSED").upper()
    exact_name = f"MD_HEALTH_MIN_FRESH_PCT_{session_key}"
    if exact_name in os.environ:
        return _float_env(exact_name, 80.0)
    if session_key in {"PRE_MARKET", "AFTER_HOURS"}:
        return _float_env("MD_HEALTH_MIN_FRESH_PCT_EXTENDED", 60.0)
    return _float_env("MD_HEALTH_MIN_FRESH_PCT", 80.0)


def main() -> int:
    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        if not client or not client.ping():
            print("FAIL: Valkey unreachable")
            return 1
    except Exception as exc:
        print(f"FAIL: Valkey error: {exc}")
        return 1

    try:
        raw = client.get("scanner:streamer")
        if raw:
            age = time.time() - json.loads(raw).get("ts", 0)
            if age > 90:
                print(f"FAIL: scanner:streamer stale ({age:.0f}s)")
                return 1
    except Exception as exc:
        print(f"WARN: scanner:streamer check error: {exc}")

    try:
        from agent.market_hours import get_market_session
        session = get_market_session()
    except Exception:
        session = "CLOSED"

    if session != "CLOSED":
        try:
            from agent.valkey_client import price_bus_health
            max_age_s = float(os.getenv("MD_HEALTH_MAX_PRICE_AGE_S", "5"))
            min_fresh_pct = _min_fresh_pct_for_session(session)
            health = price_bus_health(max_age_s=max_age_s)
            if health["total"] == 0:
                print(f"FAIL: md:prices empty during {session} session")
                return 1
            trusted_pct = float(health.get("trusted_fresh_pct") or 0.0)
            if trusted_pct < min_fresh_pct:
                print(
                    f"FAIL: md:prices trusted freshness {trusted_pct}% < {min_fresh_pct}% "
                    f"(status={health['status']}, max_age={max_age_s}s, session={session})"
                )
                return 1
            if health.get("status") in {"REST_FALLBACK", "PARTIAL_FALLBACK"}:
                print(
                    f"OK: REST fallback active "
                    f"(trusted={trusted_pct}%, live={health.get('live_pct')}%, session={session})"
                )
                return 0
        except Exception as exc:
            print(f"FAIL: md:prices freshness check error: {exc}")
            return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
