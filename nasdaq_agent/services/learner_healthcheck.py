#!/usr/bin/env python3
"""Docker health check for the continuous learner container."""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/app")


def _read_status() -> dict | None:
    try:
        from agent.service_state import get_state
        state = get_state("learner:status")
        if state:
            return state
    except Exception:
        pass

    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        raw = client.get("learner:status") if client else None
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def main() -> int:
    status = _read_status()
    if not status:
        print("FAIL: learner:status missing")
        return 1

    age = time.time() - float(status.get("ts") or 0.0)
    if age > 180:
        print(f"FAIL: learner:status stale ({age:.0f}s)")
        return 1

    engine = status.get("engine") or {}
    if engine.get("running") is not True:
        print("FAIL: learning engine not running")
        return 1

    deep = status.get("deep") or {}
    if deep.get("enabled") is True and "running" not in deep:
        print("FAIL: deep learner status missing")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
