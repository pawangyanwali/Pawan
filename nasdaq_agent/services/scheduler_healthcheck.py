#!/usr/bin/env python3
"""
scheduler container health check — called by Docker HEALTHCHECK.

Exit 0  healthy
Exit 1  unhealthy

Two checks:

  1. Valkey ping  — proves network + Valkey are reachable from this container.

  2. scheduler:heartbeat freshness  — the scheduler publishes this key every
     SCHEDULER_HEARTBEAT_S seconds (default 30) with TTL = 3× that interval
     (default 90s).  A stale or absent key means the heartbeat loop has hung
     or the container has silently died without triggering a SIGTERM.

     PostgreSQL service_state is checked first (source of truth); Valkey is
     the fast-path fallback — matching the dual-write pattern used by every
     other service.

Age threshold:  SCHEDULER_HEARTBEAT_S × 3  (default 90s)
During CLOSED sessions the EOD watchdog is idle but the heartbeat loop must
still be alive, so the age check is always enforced.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/app")

_HEARTBEAT_KEY      = "scheduler:heartbeat"
_HEARTBEAT_INTERVAL = int(os.getenv("SCHEDULER_HEARTBEAT_S", "30"))
_MAX_AGE_S          = _HEARTBEAT_INTERVAL * 3   # 90s default


def main() -> int:
    # ── 1. Valkey reachability ────────────────────────────────────────────────
    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        if not client or not client.ping():
            print("FAIL: Valkey unreachable")
            return 1
    except Exception as exc:
        print(f"FAIL: Valkey error: {exc}")
        return 1

    # ── 2. Heartbeat freshness ────────────────────────────────────────────────
    ts: float | None = None

    # PostgreSQL first (source of truth)
    try:
        from agent.service_state import get_age_s
        age_pg = get_age_s(_HEARTBEAT_KEY)
        if age_pg is not None:
            ts = time.time() - age_pg
    except Exception:
        pass

    # Valkey fast-path fallback
    if ts is None:
        try:
            raw = client.get(_HEARTBEAT_KEY)
            if raw:
                ts = float(json.loads(raw).get("ts") or 0) or None
        except Exception:
            pass

    if ts is None:
        print(f"FAIL: {_HEARTBEAT_KEY} not found in PG or Valkey")
        return 1

    age = time.time() - ts
    if age > _MAX_AGE_S:
        print(f"FAIL: {_HEARTBEAT_KEY} stale ({age:.0f}s > {_MAX_AGE_S}s) — heartbeat loop may be hung")
        return 1

    print(f"OK: scheduler heartbeat age {age:.0f}s (threshold {_MAX_AGE_S}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
