#!/usr/bin/env python3
"""Health check for the canonical scalp engine."""
from __future__ import annotations

import sys
import time


def main() -> int:
    try:
        from agent.service_state import get_age_s
        from agent.signal_snapshot import read_latest

        heartbeat_age = get_age_s("service:scalp-engine:heartbeat")
        if heartbeat_age is None or heartbeat_age > 45:
            print(f"FAIL: scalp-engine heartbeat age={heartbeat_age}")
            return 1
        snapshot = read_latest() or {}
        age = time.time() - float(snapshot.get("ts") or 0.0)
        if snapshot.get("runtime") != "SCALP_ONLY_V1":
            print("FAIL: canonical scalp snapshot missing")
            return 1
        if age > 45:
            print(f"FAIL: scalp snapshot stale ({age:.1f}s)")
            return 1
        if int(snapshot.get("universe_total") or 0) < 400:
            print("FAIL: universe coverage below 400")
            return 1
        print("OK")
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
