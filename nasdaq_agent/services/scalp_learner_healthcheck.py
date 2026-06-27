#!/usr/bin/env python3
"""Health check for canonical scalp learning."""
from __future__ import annotations

import sys
import time


def main() -> int:
    try:
        from agent.service_state import get_age_s, get_state

        heartbeat_age = get_age_s("service:scalp-learner:heartbeat")
        if heartbeat_age is None or heartbeat_age > 45:
            print(f"FAIL: scalp-learner heartbeat age={heartbeat_age}")
            return 1
        status = get_state("scalp-learner:status") or {}
        age = time.time() - float(status.get("ts") or 0.0)
        if status.get("running") is not True or age > 90:
            print(f"FAIL: scalp learner status stale ({age:.1f}s)")
            return 1
        print("OK")
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
