#!/usr/bin/env python3
"""
learner_service — continuous ML training and adaptive filter.

Responsibilities:
  1. Run the adaptive learning engine (win-rate tracking, threshold calibration)
     continuously, every 90s, in all sessions.
  2. Continuously fine-tune the Deep BiLSTM:
       - market hours : lightweight 3-epoch fine-tune every
                        learner.deep_market_interval_s (default 30min) on fresh
                        intraday bars, so the model keeps learning during the day.
       - CLOSED       : full cycle every learner.deep_interval_s (default 1h).
     A manual request via Valkey key "deep:train:requested" runs immediately in
     any session. All training stays inside the learner container's 1.5-CPU limit
     so it never starves the scanner.
  3. Publish adaptive-filter + engine state to Valkey every 60s so that web-api
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
# Market-hours fine-tune cadence (0 = disabled). The learner keeps the BiLSTM
# learning intraday with lightweight 3-epoch fine-tunes; isolation is guaranteed
# by the learner container's 1.5-CPU cgroup limit so the scanner is never starved.
_DEEP_MARKET_INTERVAL_S = max(0, int(os.getenv("LEARNER_DEEP_MARKET_INTERVAL_S", "1800")))
_DEEP_STARTUP_DELAY_S = max(0, int(os.getenv("LEARNER_DEEP_STARTUP_DELAY_S", "300")))

# 15-min bar cache TTL by session, passed to fetch_batch_interval.
#   CLOSED      : 24h — data isn't changing; large TTL survives Schwab token gaps
#                 (regular-session bars stay served from the PostgreSQL cache).
#   market hrs  : 120s — fetch_batch_interval marks a 15-min row stale once the
#                 newest bar is older than ttl × 16 (the 15min interval multiplier
#                 in data_fetcher), so 120 × 16 ≈ 32 min forces fresh intraday bars
#                 on each ~30 min fine-tune cycle. Fetched bars are still persisted
#                 to PostgreSQL (ttl > 0), so durability is preserved.
_DEEP_DATA_TTL_CLOSED_S = 86400
_DEEP_DATA_TTL_MARKET_S = 120
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


def _publish_deep_state_now() -> None:
    """Immediately push current deep state into learner:status in Valkey + PostgreSQL.
    Called after significant state transitions (start, end, error) so the dashboard
    clears the "Training in progress" banner within seconds rather than waiting for
    the 60s background publisher tick."""
    try:
        from agent.valkey_client import _get_client
        from agent.service_state import set_state
        import json as _json

        deep_snap = _deep_state_snapshot()
        # Merge into existing learner:status so we don't clobber other fields
        existing: dict = {}
        try:
            client = _get_client()
            if client:
                raw = client.get("learner:status")
                if raw:
                    existing = _json.loads(raw)
        except Exception:
            pass
        existing["deep"] = deep_snap
        existing["ts"] = time.time()
        try:
            set_state("learner:status", existing, ttl_s=300)
        except Exception:
            pass
        try:
            client = _get_client()
            if client:
                client.setex("learner:status", 300, _json.dumps(existing, default=str))
        except Exception:
            pass
    except Exception as exc:
        _log.debug("_publish_deep_state_now failed: %s", exc)


# ── Durable training-history persistence (PostgreSQL source of truth + Valkey) ──
#
# Architecture: the loss-curve history must survive learner restarts, so it is
# persisted to PostgreSQL service_state under "deep:history" with NO TTL (durable).
# Valkey holds a live mirror for fast reads and per-epoch pub/sub updates. The
# in-process deep_model._training_history is just a working copy, reloaded from
# PostgreSQL on startup so no data is ever lost.

_DEEP_HISTORY_KEY = "deep:history"


def _persist_deep_history(history: list[dict]) -> None:
    """Write training history durably to PostgreSQL (no TTL) and mirror to Valkey."""
    if not history:
        return
    payload = {"history": history[-200:], "updated_at": time.time()}
    try:
        from agent.service_state import set_state
        set_state(_DEEP_HISTORY_KEY, payload, ttl_s=None)   # ttl_s=None → durable, never expires
    except Exception as exc:
        _log.debug("persist deep:history to PG failed: %s", exc)
    try:
        from agent.valkey_client import _get_client
        import json as _json
        client = _get_client()
        if client:
            client.set(_DEEP_HISTORY_KEY, _json.dumps(payload, default=str))
            client.publish("deep:history:updated", _json.dumps(payload, default=str))
    except Exception as exc:
        _log.debug("mirror deep:history to Valkey failed: %s", exc)


def _load_deep_history() -> None:
    """Restore training history from PostgreSQL into deep_model on startup."""
    try:
        from agent.service_state import get_state
        from agent.deep_model import set_history
        row = get_state(_DEEP_HISTORY_KEY, ignore_expiry=True)
        hist = (row or {}).get("history", []) if row else []
        if hist:
            set_history(hist)
            _log.info("Restored %d deep training-history points from PostgreSQL", len(hist))
    except Exception as exc:
        _log.debug("load deep:history failed: %s", exc)


_last_epoch_publish = 0.0


def _on_epoch(entry: dict, full_history: list[dict]) -> None:
    """Live per-epoch hook (registered into deep_model). Streams the growing loss
    curve to Valkey on every epoch and checkpoints to PostgreSQL at most every 10s."""
    global _last_epoch_publish
    # Always update the live deep state so the dashboard loss curve grows in real time
    _set_deep_state(running=True, history=full_history[-120:],
                    last_epoch=entry.get("epoch"), last_loss=entry.get("loss"))
    _publish_deep_state_now()
    # Durable PostgreSQL checkpoint, rate-limited to avoid hammering the DB per epoch
    now = time.time()
    if now - _last_epoch_publish >= 10.0:
        _last_epoch_publish = now
        _persist_deep_history(full_history)


_MANUAL_REQUEST_KEY = "deep:train:requested"
_MARKET_HOURS_CHECK_INTERVAL_S = 30   # poll for manual requests every 30s during market hours


def _check_and_clear_manual_request() -> bool:
    """Return True and consume the manual training request from Valkey if one exists."""
    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        if client and client.delete(_MANUAL_REQUEST_KEY):
            return True
    except Exception:
        pass
    return False


def _deep_interval_for_session(session: str) -> int:
    """
    Cadence between deep-training cycles, by market session.

    CLOSED        → learner.deep_interval_s         (default 3600s) — full cycle
    market hours  → learner.deep_market_interval_s  (default 1800s) — fine-tune

    During market hours the model files already exist, so _train_one_cluster
    automatically runs a lightweight FINE_TUNE_EPOCHS (3) pass rather than a full
    10-epoch retrain — keeping the BiLSTM learning intraday without starving the
    scanner (the learner container is capped at 1.5 CPU). Returning 0 for market
    hours disables intraday training (CLOSED-only behaviour).
    """
    try:
        from agent.config_manager import config as _cfg
        if session == "CLOSED":
            return max(300, int(_cfg.get("learner.deep_interval_s", _DEEP_INTERVAL_S)))
        return max(0, int(_cfg.get("learner.deep_market_interval_s", _DEEP_MARKET_INTERVAL_S)))
    except Exception:
        return _DEEP_INTERVAL_S if session == "CLOSED" else _DEEP_MARKET_INTERVAL_S


def _wait_for_next_cycle(interval_s: int) -> bool:
    """
    Sleep up to interval_s in _MARKET_HOURS_CHECK_INTERVAL_S slices so a manual
    "Train BiLSTM" request is picked up within ~30s. Returns True if a manual
    request was consumed (caller should run a cycle immediately) or the service
    is stopping; False when the interval simply elapsed.
    """
    slept = 0
    while slept < interval_s and not _runner.stopped:
        if _runner._stop.wait(_MARKET_HOURS_CHECK_INTERVAL_S):
            return True
        slept += _MARKET_HOURS_CHECK_INTERVAL_S
        if _check_and_clear_manual_request():
            _log.info("Manual BiLSTM training request received — running now")
            _run_deep_cycle()
            return True
    return False


def _continuous_deep_loop() -> None:
    """
    Keep the Deep BiLSTM continuously learning — during BOTH market hours and the
    CLOSED window.

    - CLOSED        : full cycle every learner.deep_interval_s (default 1h).
    - market hours  : lightweight 3-epoch fine-tune every
                      learner.deep_market_interval_s (default 30min). Set that key
                      to 0 to revert to CLOSED-only training.

    A manual "deep:train:requested" key (the "Train BiLSTM" button) is detected
    within ~30s and runs immediately regardless of session.

    Isolation: all training runs inside the learner container, which is capped at
    1.5 CPU by docker-compose, so intraday fine-tuning never starves the scanner
    (2.5 CPU) or web-api on the 4-vCPU host.
    """
    if _DEEP_STARTUP_DELAY_S:
        _log.info("Continuous deep learner waiting %ss before first cycle", _DEEP_STARTUP_DELAY_S)
        if _runner._stop.wait(_DEEP_STARTUP_DELAY_S):
            return

    while not _runner.stopped:
        try:
            from agent.config_manager import config as _cfg
            deep_on = _cfg.get("learner.deep_enabled", _DEEP_ENABLED)
        except Exception:
            deep_on = _DEEP_ENABLED
        if not deep_on:
            _log.debug("Deep learner disabled via config; rechecking in 60s")
            _runner._stop.wait(60)
            continue

        # Manual request always wins, regardless of session.
        if _check_and_clear_manual_request():
            _log.info("Manual BiLSTM training request received — running now")
            _run_deep_cycle()
            continue

        session = _current_session()
        interval = _deep_interval_for_session(session)

        if interval <= 0:
            # Intraday training disabled (market hours + deep_market_interval_s=0).
            # Idle, but still poll for manual requests every 30s.
            _log.debug("Deep intraday training disabled (session=%s); idling", session)
            if _wait_for_next_cycle(300):
                continue
            continue

        _mode = "fine-tune" if session != "CLOSED" else "full"
        _log.info("Deep cycle starting (session=%s, mode=%s, next in %ss)",
                  session, _mode, interval)
        _run_deep_cycle()

        # Cool down for `interval`, waking early on a manual request.
        _wait_for_next_cycle(interval)


def _run_deep_cycle() -> None:
    start = time.time()
    _set_deep_state(
        running=True,
        cycle=int(_deep_state_snapshot().get("cycle", 0)) + 1,
        last_started_at=start,
        last_finished_at=None,
        last_error=None,
    )
    _publish_deep_state_now()   # immediately show "Training in progress" on dashboard
    try:
        # Ensure Schwab tokens are current before attempting a live data fetch.
        # Tokens expire every 30 min; refresh them if needed so _is_authorised() is True.
        try:
            from agent.broker.schwab_auth import load_stored_md_tokens, load_stored_tokens, _market_data as _sm
            if not _sm.get_access_token():
                load_stored_md_tokens()
                load_stored_tokens()
        except Exception:
            pass

        from agent.data_fetcher import fetch_batch_interval
        from agent.deep_model import retrain_deep_all
        from config import TRAINING_TICKERS

        try:
            from agent.config_manager import config as _cfg
            _ticker_limit = int(_cfg.get("learner.deep_ticker_limit", _DEEP_TICKER_LIMIT))
        except Exception:
            _ticker_limit = _DEEP_TICKER_LIMIT

        # Sample tickers proportionally across all three clusters so that every
        # cluster gets training data.  TRAINING_TICKERS is ordered TIER1→TIER2→TIER3
        # so a plain [:limit] slice only ever hits Cluster A (all of TIER1=100).
        from config import CLUSTER_A_TICKERS, CLUSTER_B_TICKERS, CLUSTER_C_TICKERS
        _per_cluster = max(1, _ticker_limit // 3)
        tickers = (
            list(CLUSTER_A_TICKERS)[:_per_cluster]
            + list(CLUSTER_B_TICKERS)[:_per_cluster]
            + list(CLUSTER_C_TICKERS)[:(_ticker_limit - 2 * _per_cluster)]
        )
        # Fresh intraday bars during market hours (small TTL), resilient cache when
        # CLOSED (large TTL). Either way ttl > 0 so bars persist to PostgreSQL.
        _session = _current_session()
        _data_ttl = _DEEP_DATA_TTL_CLOSED_S if _session == "CLOSED" else _DEEP_DATA_TTL_MARKET_S
        _log.info("Continuous deep cycle starting (session=%s, %d tickers: A=%d B=%d C=%d, data_ttl=%ds)",
                  _session, len(tickers), _per_cluster, _per_cluster,
                  len(tickers) - 2 * _per_cluster, _data_ttl)
        hist_15m = fetch_batch_interval(
            tickers,
            "15min",
            5000,
            ttl=_data_ttl,
            background=True,
            # extended_hours=False (default) — regular-session bars are sufficient
            # for directional BiLSTM training AND this enables the PostgreSQL cache,
            # so data survives Schwab token expiry across training cycles.
        )
        if not hist_15m:
            _log.warning(
                "[deep-cycle] fetch_batch_interval returned 0 tickers — "
                "Schwab may be unauthorized or tokens expired. Training skipped."
            )
            _set_deep_state(
                running=False,
                last_finished_at=time.time(),
                last_duration_s=round(time.time() - start, 1),
                last_error="fetch returned 0 tickers — check Schwab auth",
            )
            _publish_deep_state_now()
            return
        ok = retrain_deep_all(hist_15m)
        finished = time.time()

        # Capture training history (loss curve) and trained-cluster list from the
        # deep_model module — these live in THIS process's memory after training.
        # We publish them into learner:status so the web-api container (which has
        # its own empty copy of deep_model state) can render the loss curve and
        # trained badges without re-training.
        try:
            from agent.deep_model import get_training_history, get_model_info
            _hist = get_training_history()[-120:]   # last ~120 epoch points
            _info = get_model_info()
            _trained_clusters = _info.get("trained_clusters", [])
        except Exception:
            _hist, _trained_clusters = [], []

        _set_deep_state(
            running=False,
            last_finished_at=finished,
            last_duration_s=round(finished - start, 1),
            last_success_at=finished if ok else _deep_state_snapshot().get("last_success_at"),
            last_tickers=len(hist_15m),
            last_error=None if ok else (
                f"retrain returned false — {len(hist_15m)} tickers fetched; "
                "check learner logs for NaN loss (inf in features) or < 500 train sequences"
            ),
            history=_hist,
            trained_clusters=_trained_clusters,
            trained=bool(_trained_clusters),
        )
        _log.info("Continuous deep cycle finished ok=%s tickers=%d clusters=%s history=%d",
                  ok, len(hist_15m), _trained_clusters, len(_hist))
        _persist_deep_history(_hist)   # final durable checkpoint to PostgreSQL + Valkey
        _publish_deep_state_now()      # immediately clear "Training in progress" banner
    except Exception as exc:
        finished = time.time()
        _set_deep_state(
            running=False,
            last_finished_at=finished,
            last_duration_s=round(finished - start, 1),
            last_error=str(exc),
        )
        _log.warning("Continuous deep cycle failed: %s", exc, exc_info=True)
        _publish_deep_state_now()   # immediately reflect failure on dashboard


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
            parameter_governor: dict = {}
            try:
                from agent.algo_learning_p2 import get_phase2_engine
                phase2_status = get_phase2_engine().get_status()
            except Exception:
                pass
            try:
                from agent.algo_learning_engine import get_engine
                parameter_governor = get_engine().get_parameter_governor_status()
            except Exception:
                pass

            payload = {
                "ts":              time.time(),
                "engine":          learning_engine.get_status(),
                "adaptive_filter": af_status(),
                "log":             get_learning_log(limit=50),
                "phase2":          phase2_status,
                "parameter_governor": parameter_governor,
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

    from agent.config_manager import config as _cfg
    _cfg.load()
    _cfg.seed_defaults()
    _cfg.start_listener(_runner)
    _log.info("Runtime config loaded (%d keys)", len(_cfg.all()))

    # Load Schwab tokens from disk so fetch_batch_interval can authenticate.
    # Every other service (scanner, market-data, web-api) does this at startup.
    # Without it, _market_data._tokens stays empty, _is_authorised() is always
    # False, and all 15-min data fetches silently return {}.
    try:
        from agent.broker.schwab_auth import load_stored_md_tokens, load_stored_tokens
        ok_md = load_stored_md_tokens()
        ok_tr = load_stored_tokens()
        _log.info("Schwab tokens loaded — md=%s trader=%s", ok_md, ok_tr)
        if not ok_md and not ok_tr:
            _log.warning(
                "Schwab tokens NOT loaded — BiLSTM training will only work from "
                "PostgreSQL cache. Re-authenticate at /schwab/auth/md to fix."
            )
    except Exception as exc:
        _log.warning("Schwab token load failed: %s", exc)

    # Restore durable training history from PostgreSQL and register the live
    # per-epoch hook so the loss curve streams to Valkey + PostgreSQL during training.
    _load_deep_history()
    try:
        from agent.deep_model import set_epoch_hook
        set_epoch_hook(_on_epoch)
        _log.info("Registered live per-epoch training hook")
    except Exception as exc:
        _log.debug("epoch hook registration failed: %s", exc)

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
