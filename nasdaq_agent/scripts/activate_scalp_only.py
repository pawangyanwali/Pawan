#!/usr/bin/env python3
"""One-time, health-gated activation for the Release 5 paper runtime."""
from __future__ import annotations

import sys
import time


def main() -> int:
    from agent.config_manager import config
    from agent.service_state import get_age_s
    from agent.signal_snapshot import read_latest

    snapshot = read_latest() or {}
    snapshot_age = time.time() - float(snapshot.get("ts") or 0.0)
    engine_age = get_age_s("service:scalp-engine:heartbeat")
    learner_age = get_age_s("service:scalp-learner:heartbeat")
    context_age = get_age_s("service:context-intel:heartbeat")
    checks = {
        "runtime": snapshot.get("runtime") == "SCALP_ONLY_V1",
        "snapshot_fresh": snapshot_age <= 60,
        "universe_covered": int(snapshot.get("universe_total") or 0) >= 400,
        "engine_healthy": engine_age is not None and engine_age <= 45,
        "learner_healthy": learner_age is not None and learner_age <= 45,
        "context_healthy": context_age is not None and context_age <= 120,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print("Activation refused; failed checks: " + ", ".join(failed))
        return 1

    config.load()
    config.set_many(
        {
            "scalp.shadow_enabled": False,
            "scalp.execution_enabled": True,
            "scalp_runtime.bar_lookback": 500,
        },
        updated_by="release5_health_gated_cutover",
    )
    print(
        "SCALP_ONLY_V1 activated: canonical plans are now the sole paper-entry contract "
        f"({snapshot.get('universe_total')} tickers, snapshot age {snapshot_age:.1f}s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
