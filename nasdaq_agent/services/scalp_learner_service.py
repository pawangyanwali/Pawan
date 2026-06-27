#!/usr/bin/env python3
"""Canonical outcome learner and advisory ML trainer."""
from __future__ import annotations

import sys
import time
from typing import Any

from services._base import ServiceRunner, configure_logging

_log = configure_logging("scalp-learner")
_runner = ServiceRunner("scalp-learner")
_status: dict[str, Any] = {"ts": 0.0, "running": False, "mode": "STARTING"}


def _publish_status(**updates: Any) -> None:
    _status.update(ts=time.time(), running=True, **updates)
    try:
        from agent.service_state import set_state

        set_state("scalp-learner:status", dict(_status), ttl_s=180)
    except Exception:
        _log.debug("Status publish failed", exc_info=True)


def main() -> int:
    from agent.config_manager import config
    from agent.scalp.ml_trainer import train_and_maybe_promote
    from agent.scalp.store import init_scalp_tables
    from agent.service_heartbeat import start_service_heartbeat

    _log.info("=== scalp_learner_service starting ===")
    config.load()
    config.seed_defaults()
    config.start_listener(_runner)
    init_scalp_tables()
    start_service_heartbeat(
        "scalp-learner",
        _runner,
        interval_s=15,
        extra=lambda: dict(_status),
    )
    _runner.register_signals()
    last_attempt = 0.0
    _publish_status(mode="OBSERVING", detail="Immediate context learning runs on every canonical trade close")
    while not _runner.stopped:
        enabled = bool(config.get("scalp_ml.training_enabled", False))
        interval_s = max(
            300.0,
            float(config.get("scalp_ml.training_interval_min", 60)) * 60.0,
        )
        if enabled and time.time() - last_attempt >= interval_s:
            last_attempt = time.time()
            _publish_status(mode="TRAINING", training_enabled=True)
            try:
                result = train_and_maybe_promote()
                _publish_status(
                    mode="OBSERVING",
                    training_enabled=True,
                    last_training_result=result,
                )
            except Exception as exc:
                _log.exception("Scalp ML training failed")
                _publish_status(mode="DEGRADED", training_enabled=True, error=str(exc))
        else:
            _publish_status(mode="OBSERVING", training_enabled=enabled)
        _runner._stop.wait(30)
    _log.info("=== scalp_learner_service stopped ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
