#!/usr/bin/env python3
"""Production entrypoint for the scalp-only signal runtime."""
from __future__ import annotations

import sys
import threading

from services._base import ServiceRunner, configure_logging

_log = configure_logging("scalp-engine")
_runner = ServiceRunner("scalp-engine")


def main() -> int:
    from agent.config_manager import config
    from agent.scalp.runtime import ScalpRuntime
    from agent.scalp.store import init_scalp_tables
    from agent.service_heartbeat import start_service_heartbeat
    from agent.universe_registry import get_runtime_universe

    tickers = get_runtime_universe()
    _log.info("=== scalp_engine_service starting (%d eligible tickers) ===", len(tickers))
    config.load()
    config.seed_defaults()
    config.start_listener(_runner)
    init_scalp_tables()
    runtime = ScalpRuntime(tickers)
    start_service_heartbeat(
        "scalp-engine",
        _runner,
        interval_s=15,
        extra=lambda: dict(runtime.last_cycle),
    )
    _runner.register_signals()

    def _position_loop() -> None:
        while not _runner.stopped:
            try:
                runtime.run_position_tick()
            except Exception:
                _log.exception("Live position monitor failed")
            _runner._stop.wait(1.0)

    threading.Thread(
        target=_position_loop,
        daemon=True,
        name="scalp-position-monitor",
    ).start()
    while not _runner.stopped:
        try:
            runtime.run_cycle()
        except Exception:
            _log.exception("Scalp cycle failed")
        interval = max(1.0, float(config.get("scalp_runtime.cycle_interval_s", 5.0)))
        _runner._stop.wait(interval)
    _log.info("=== scalp_engine_service stopped ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
