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
import threading
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv
load_dotenv()

from services._base import configure_logging, ServiceRunner

_log    = configure_logging("learner")
_runner = ServiceRunner("learner")

_POLL_INTERVAL_S = int(os.getenv("LEARNER_POLL_INTERVAL_S", "30"))
_DEEP_ENABLED = os.getenv("LEARNER_DEEP_ENABLED", "1").lower() in ("1", "true", "yes")
_DEEP_INTERVAL_S = max(300, int(os.getenv("LEARNER_DEEP_INTERVAL_S", "3600")))
_DEEP_STARTUP_DELAY_S = max(0, int(os.getenv("LEARNER_DEEP_STARTUP_DELAY_S", "300")))
_DEEP_TICKER_LIMIT = max(1, int(os.getenv("LEARNER_DEEP_TICKER_LIMIT", "100")))
_DEEP_STATE: dict = {
    "enabled": _DEEP_ENABLED,
    "running": False,
    "cycle": 0,
    "interval_s": _DEEP_INTERVAL_S,
    "ticker_limit": _DEEP_TICKER_LIMIT,
    "last_started_at": None,
    "last_finished_at": None,
    "last_success_at": None,
    "last_duration_s": None,
    "last_tickers": 0,
    "last_error": None,
}
_DEEP_LOCK = threading.Lock()


# ── Market-hours guard ────────────────────────────────────────────────────────

def _is_market_hours() -> bool:
    """Return True if we are currently within regular trading hours."""
    try:
        from agent.market_hours import get_market_session
        session = get_market_session()
        return session in ("REGULAR", "PRE_MARKET", "AFTER_HOURS")
    except Exception:
        return True   # fail-safe: assume market open, block training


# ── Learning engine lifecycle ─────────────────────────────────────────────────

def _current_session() -> str:
    try:
        from agent.market_hours import get_market_session
        return str(get_market_session())
    except Exception:
        return "UNKNOWN"


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

def _deep_state_snapshot() -> dict:
    with _DEEP_LOCK:
        return dict(_DEEP_STATE)


def _set_deep_state(**updates) -> None:
    with _DEEP_LOCK:
        _DEEP_STATE.update(updates)


def _continuous_deep_loop() -> None:
    """
    Keep Deep BiLSTM learning alive in the learner container.

    The scanner never runs this work. Docker CPU and memory limits keep this
    background learner from starving the market-data and scanner containers.
    Training is skipped while the market is open to avoid CPU contention with
    the live scanner — consistent with the module's market-hours contract.
    """
    if not _DEEP_ENABLED:
        _log.info("Continuous deep learner disabled by LEARNER_DEEP_ENABLED=0")
        return

    if _DEEP_STARTUP_DELAY_S:
        _log.info("Continuous deep learner waiting %ss before first cycle", _DEEP_STARTUP_DELAY_S)
        if _runner._stop.wait(_DEEP_STARTUP_DELAY_S):
            return

    while not _runner.stopped:
        if _is_market_hours():
            _log.debug("Skipping deep cycle — market is open; rechecking in 5 min")
            _runner._stop.wait(300)
            continue
        _run_deep_cycle()
        _runner._stop.wait(_DEEP_INTERVAL_S)


def _run_deep_cycle() -> None:
    start = time.time()
    _set_deep_state(
        running=True,
        cycle=int(_deep_state_snapshot().get("cycle", 0)) + 1,
        last_started_at=start,
        last_finished_at=None,
        last_error=None,
    )
    try:
        from agent.data_fetcher import fetch_batch_interval
        from agent.deep_model import retrain_deep_all
        from config import TRAINING_TICKERS

        tickers = list(TRAINING_TICKERS)[:_DEEP_TICKER_LIMIT]
        _log.info("Continuous deep cycle starting (%d tickers)", len(tickers))
        hist_15m = fetch_batch_interval(
            tickers,
            "15min",
            5000,
            ttl=86400,
            background=True,
            extended_hours=True,
        )
        ok = retrain_deep_all(hist_15m)
        finished = time.time()
        _set_deep_state(
            running=False,
            last_finished_at=finished,
            last_duration_s=round(finished - start, 1),
            last_success_at=finished if ok else _deep_state_snapshot().get("last_success_at"),
            last_tickers=len(hist_15m),
            last_error=None if ok else "deep retrain returned false",
        )
        _log.info("Continuous deep cycle finished ok=%s tickers=%d", ok, len(hist_15m))
    except Exception as exc:
        finished = time.time()
        _set_deep_state(
            running=False,
            last_finished_at=finished,
            last_duration_s=round(finished - start, 1),
            last_error=str(exc),
        )
        _log.warning("Continuous deep cycle failed: %s", exc, exc_info=True)


def _publish_status_loop() -> None:
    """
    Publish adaptive-filter + learning-engine state every 60s.

    Write order (both best-effort — a failure must never crash the learner):
      1. PostgreSQL service_state  — durable, expires_at = NOW() + 300s
      2. Valkey SETEX              — fast-path cache, TTL = 300s (backward compat)

    Key: learner:status  TTL: 300s  (marks stale if learner dies)
    """
    try:
        from agent.valkey_client import _get_client
    except ImportError:
        _get_client = lambda: None   # noqa: E731 — Valkey unavailable, PG-only path

    _TTL = 300  # seconds — matches docker-compose healthcheck expectation

    while not _runner.stopped:
        try:
            from agent.learning_engine import learning_engine, get_learning_log
            from agent.adaptive_filter import get_status as af_status

            phase2_status: dict = {}
            try:
                from agent.algo_learning_p2 import get_phase2_engine
                phase2_status = get_phase2_engine().get_status()
            except Exception:
                pass

            payload = {
                "ts":              time.time(),
                "engine":          learning_engine.get_status(),
                "adaptive_filter": af_status(),
                "log":             get_learning_log(limit=50),
                "phase2":          phase2_status,
                "deep":            _deep_state_snapshot(),
                "service": {
                    "mode": "continuous",
                    "poll_interval_s": _POLL_INTERVAL_S,
                    "session": _current_session(),
                },
            }

            # ── 1. PostgreSQL (source of truth) ───────────────────────────────
            try:
                from agent.service_state import set_state
                set_state("learner:status", payload, ttl_s=_TTL)
                _log.debug("learner:status published to PostgreSQL")
            except Exception as exc:
                _log.debug("learner:status PG write failed: %s", exc)

            # ── 2. Valkey (fast-path cache) ───────────────────────────────────
            try:
                client = _get_client()
                if client:
                    client.setex("learner:status", _TTL, json.dumps(payload))
                    _log.debug("learner:status published to Valkey")
            except Exception as exc:
                _log.debug("learner:status Valkey write failed: %s", exc)

        except Exception as exc:
            _log.debug("learner:status publish failed: %s", exc)
        if _runner._stop.wait(60):
            break


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
        if _runner._stop.wait(300):
            break


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import threading
    from agent.service_heartbeat import start_service_heartbeat

    _log.info("=== learner_service starting ===")
    _log.info("Continuous learning active; training work stays inside learner container limits.")

    _run_learning_engine()
    _run_weekend_learner()
    start_service_heartbeat("learner", _runner)

    threading.Thread(target=_monitor_loop,        daemon=True, name="learner-monitor").start()
    threading.Thread(target=_publish_status_loop, daemon=True, name="learner-status-pub").start()
    threading.Thread(target=_continuous_deep_loop, daemon=True, name="continuous-deep").start()

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
