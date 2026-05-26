#!/usr/bin/env python3
"""
learner_service — off-hours ML training and adaptive filter.

Responsibilities:
  1. Run the adaptive learning engine (win-rate tracking, threshold calibration).
  2. Run the weekend deep-learner when market is closed.
  3. Gate ALL activity behind market-hours checks — never train during market hours.
  4. Publish adaptive-filter + engine state to Valkey every 60s so that web-api
     can serve fresh /api/learning-status without a local learner.

Interface contract:
  READS   PostgreSQL  — signal history, trade outcomes
  WRITES  PostgreSQL  — updated model weights, learning logs
  WRITES  local disk  — XGBoost / BiLSTM model files (agent/models/)
  WRITES  Valkey      — learner:status (TTL 300s)

Environment variables:
  LOG_LEVEL  DEBUG|INFO|WARNING  (default INFO)
  All DB env vars inherited from environment / .env
"""
from __future__ import annotations

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv
load_dotenv()

from services._base import configure_logging, ServiceRunner

_log    = configure_logging("learner")
_runner = ServiceRunner("learner")

# Hard gate: never train within MARKET_HOURS_BUFFER_S of market open/close.
# The learning engine has its own internal gate; this is a belt-and-suspenders check.
_POLL_INTERVAL_S = int(os.getenv("LEARNER_POLL_INTERVAL_S", "30"))


# ── Market-hours guard ────────────────────────────────────────────────────────

def _is_market_hours() -> bool:
    """Return True if we are currently within regular trading hours."""
    try:
        from agent.market_hours import get_market_session
        session = get_market_session()
        return session in ("OPEN", "PRE_MARKET", "POST_MARKET")
    except Exception:
        return True   # fail-safe: assume market open, block training


# ── Learning engine lifecycle ─────────────────────────────────────────────────

def _run_learning_engine() -> None:
    """Start the adaptive learning engine and keep it alive."""
    try:
        from agent.learning_engine import learning_engine
        learning_engine.start()
        _log.info("Adaptive learning engine started")
    except Exception as exc:
        _log.error("Learning engine failed to start: %s", exc, exc_info=True)


def _run_weekend_learner() -> None:
    """Start the weekend deep-learner if conditions are met."""
    try:
        import agent.weekend_learner as weekend_learner
        weekend_learner.maybe_start()
        _log.info("Weekend learner evaluated (may not start if market is open)")
    except Exception as exc:
        _log.warning("Weekend learner error: %s", exc)


# ── Valkey status publisher ───────────────────────────────────────────────────

def _publish_status_loop() -> None:
    """
    Publish adaptive-filter + engine state to Valkey every 60s so that the
    web-api container (which runs no learner) can serve fresh data from
    GET /api/learning-status without polling the DB.

    Key: learner:status  TTL: 300s  (expires if learner dies)
    """
    try:
        from agent.valkey_client import _get_client
    except ImportError:
        _log.debug("Valkey client not available — learner:status will not be published")
        return

    while not _runner.stopped:
        try:
            client = _get_client()
            if client:
                from agent.learning_engine import learning_engine
                from agent.adaptive_filter import get_status as af_status
                payload = json.dumps({
                    "ts":             time.time(),
                    "engine":         learning_engine.get_status(),
                    "adaptive_filter": af_status(),
                })
                client.setex("learner:status", 300, payload)
                _log.debug("learner:status published to Valkey")
        except Exception as exc:
            _log.debug("learner:status publish failed: %s", exc)
        time.sleep(60)


# ── Main loop ─────────────────────────────────────────────────────────────────

def _monitor_loop() -> None:
    """
    Periodic health check.  Logs learning engine state every 5 minutes so
    operators can confirm it is running during off-hours.
    """
    while not _runner.stopped:
        try:
            from agent.adaptive_filter import get_status as af_get_status
            s = af_get_status()
            _log.info(
                "Learning — win_rate=%.1f%%  threshold=%.1f%%  suppressed=%d  is_learning=%s",
                s.get("current_win_rate", 0.0),
                s.get("dynamic_threshold", 60.0),
                s.get("suppressed_count", 0),
                s.get("is_learning", False),
            )
        except Exception:
            pass
        time.sleep(300)   # every 5 min


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import threading

    _log.info("=== learner_service starting ===")

    if _is_market_hours():
        _log.warning(
            "Market appears to be open — learner will start but internal gates "
            "will block training until market closes."
        )

    _run_learning_engine()
    _run_weekend_learner()

    threading.Thread(target=_monitor_loop,        daemon=True, name="learner-monitor").start()
    threading.Thread(target=_publish_status_loop, daemon=True, name="learner-status-pub").start()

    _log.info("Learner running — waiting for SIGTERM/SIGINT …")

    def _on_stop() -> None:
        try:
            from agent.learning_engine import learning_engine
            learning_engine.stop()
        except Exception:
            pass

    _runner.register_signals(on_stop=_on_stop)
    _runner.wait()
    _log.info("=== learner_service stopped ===")


if __name__ == "__main__":
    main()
