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


def _load_training_watermark(fallback: int = 0) -> int:
    try:
        from agent.service_state import get_state

        state = get_state("scalp-learner:ml-watermark", ignore_expiry=True) or {}
        return max(int(fallback), int(state.get("outcome_count") or 0))
    except Exception:
        return int(fallback)


def _save_training_watermark(outcome_count: int) -> None:
    from agent.service_state import set_state

    set_state(
        "scalp-learner:ml-watermark",
        {"outcome_count": int(outcome_count), "evaluated_at": time.time()},
        ttl_s=None,
    )


def main() -> int:
    from agent.config_manager import config
    from agent.scalp.ml_trainer import train_and_maybe_promote
    from agent.scalp.store import (
        init_scalp_tables,
        latest_ml_sample_count,
        scalp_outcome_count,
    )
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
    try:
        trained_outcome_watermark = _load_training_watermark(
            latest_ml_sample_count()
        )
    except Exception:
        _log.exception("ML outcome watermark load failed")
        trained_outcome_watermark = 0
    _publish_status(mode="OBSERVING", detail="Immediate context learning runs on every canonical trade close")
    while not _runner.stopped:
        manually_enabled = bool(config.get("scalp_ml.training_enabled", False))
        auto_armed = bool(config.get("scalp_ml.auto_train_when_ready", True))
        promotion_minimum_samples = max(50, int(config.get("scalp_ml.minimum_samples", 200)))
        bootstrap_minimum_samples = max(
            50,
            int(config.get("scalp_ml.bootstrap_minimum_samples", 75)),
        )
        bootstrap_enabled = bool(config.get("scalp_ml.bootstrap_training_enabled", True))
        minimum_samples = (
            min(promotion_minimum_samples, bootstrap_minimum_samples)
            if bootstrap_enabled
            else promotion_minimum_samples
        )
        try:
            outcome_count = scalp_outcome_count()
        except Exception as exc:
            _log.exception("Canonical outcome count failed")
            _publish_status(
                mode="DEGRADED",
                training_enabled=False,
                auto_armed=auto_armed,
                error=str(exc),
            )
            _runner._stop.wait(30)
            continue
        sample_ready = outcome_count >= minimum_samples
        samples_until_training = max(0, minimum_samples - outcome_count)
        enabled = manually_enabled or (auto_armed and sample_ready)
        interval_s = max(
            300.0,
            float(config.get("scalp_ml.training_interval_min", 60)) * 60.0,
        )
        has_new_outcomes = outcome_count > trained_outcome_watermark
        if enabled and has_new_outcomes and time.time() - last_attempt >= interval_s:
            last_attempt = time.time()
            _publish_status(mode="TRAINING", training_enabled=True)
            try:
                result = train_and_maybe_promote()
                trained_outcome_watermark = outcome_count
                _save_training_watermark(trained_outcome_watermark)
                _publish_status(
                    mode="OBSERVING",
                    training_enabled=True,
                    auto_armed=auto_armed,
                    sample_ready=sample_ready,
                    outcome_count=outcome_count,
                    minimum_samples=minimum_samples,
                    promotion_minimum_samples=promotion_minimum_samples,
                    bootstrap_minimum_samples=bootstrap_minimum_samples,
                    bootstrap_enabled=bootstrap_enabled,
                    samples_until_training=samples_until_training,
                    trained_outcome_watermark=trained_outcome_watermark,
                    last_training_result=result,
                )
            except Exception as exc:
                _log.exception("Scalp ML training failed")
                _publish_status(mode="DEGRADED", training_enabled=True, error=str(exc))
        else:
            _publish_status(
                mode=(
                    "WAITING_FOR_NEW_OUTCOMES" if enabled and not has_new_outcomes
                    else "OBSERVING" if enabled
                    else "WAITING_FOR_SAMPLES" if auto_armed
                    else "DISABLED"
                ),
                training_enabled=enabled,
                auto_armed=auto_armed,
                sample_ready=sample_ready,
                outcome_count=outcome_count,
                minimum_samples=minimum_samples,
                promotion_minimum_samples=promotion_minimum_samples,
                bootstrap_minimum_samples=bootstrap_minimum_samples,
                bootstrap_enabled=bootstrap_enabled,
                samples_until_training=samples_until_training,
                trained_outcome_watermark=trained_outcome_watermark,
                new_outcomes=max(0, outcome_count - trained_outcome_watermark),
            )
        _runner._stop.wait(30)
    _log.info("=== scalp_learner_service stopped ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
