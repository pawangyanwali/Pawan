#!/usr/bin/env python3
"""
context_healthcheck.py — Docker health probe for the context-intel container.

Three-tier check (provider reachability is NOT checked — see design notes):
  1. Valkey PING              — always checked
  2. DB reachable             — SELECT 1 from ticker_context_features
  3. Feature-compute liveness — ctx:intel:heartbeat key age < 120 s
                                 (only enforced after 2-minute warm-up)

Design notes
------------
  * Provider (Finnhub) outage → warning in /api/services/status, NOT a health fail.
    Restarting the container because Finnhub is down would not help and could
    cause restart loops.
  * The heartbeat key has a 120-second TTL set by feature_compute_loop every 30 s.
    If the feature compute thread hangs or crashes, the key expires and the health
    check reports unhealthy, which does restart the container (one legitimate case).

Exit codes
----------
  0  healthy
  1  unhealthy (message written to stderr)
"""
import json
import os
import sys
import time

# Bootstrap import path
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _check_valkey() -> str | None:
    """Return error string or None."""
    try:
        from agent.valkey_client import _get_client
        c = _get_client()
        if c is None:
            return "Valkey client unavailable"
        c.ping()
        return None
    except Exception as exc:
        return f"Valkey PING failed: {exc}"


def _check_db() -> str | None:
    """Return error string or None."""
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            conn.execute("SELECT 1 FROM ticker_context_features LIMIT 1")
        return None
    except Exception as exc:
        return f"DB check failed: {exc}"


def _check_heartbeat() -> str | None:
    """
    Return error string or None.
    Skip the heartbeat check for the first 2 minutes after container start
    to give feature_compute_loop time to run its first cycle.
    """
    # Allow 2-minute warm-up window using container uptime heuristic:
    # if /proc/uptime < 120s we skip this check.
    try:
        with open("/proc/uptime") as f:
            uptime_s = float(f.read().split()[0])
        if uptime_s < 120:
            return None   # still warming up — don't penalise
    except Exception:
        pass

    try:
        from agent.valkey_client import _get_client
        c = _get_client()
        if c is None:
            return None   # Valkey already failed above; don't double-report
        raw = c.get("ctx:intel:heartbeat")
        if raw is None:
            return "ctx:intel:heartbeat missing — feature_compute may be stuck"
        data = json.loads(raw)
        age  = time.time() - float(data.get("ts", 0))
        if age > 120:
            return f"ctx:intel:heartbeat stale ({age:.0f}s ago)"
        return None
    except Exception as exc:
        return f"Heartbeat check error: {exc}"


def main() -> None:
    errors: list[str] = []

    for check_fn in (_check_valkey, _check_db, _check_heartbeat):
        err = check_fn()
        if err:
            errors.append(err)

    if errors:
        print(f"UNHEALTHY: {'; '.join(errors)}", file=sys.stderr)
        sys.exit(1)

    print("OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
