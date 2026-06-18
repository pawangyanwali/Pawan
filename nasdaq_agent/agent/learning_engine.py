"""
Continuous background learning engine.

Runs as a single daemon thread 24/7 — completely independent of the scan loop.
Every LEARN_INTERVAL_SECS it:

  1. Reads latest resolved outcomes from live_backtest and closed paper trades
  2. Builds combined context-breakdown stats (session, regime, vwap_event …)
  3. Pushes fresh stats into the adaptive filter
  4. Optionally triggers ML model feedback retrain when enough new outcomes exist

Works during all sessions including AFTER_HOURS and PRE_MARKET — the learning
never stops just because the market is closed.

Log strategy
------------
Routine per-cycle messages go to a rotating in-memory ring buffer (last 200 lines)
exposed via get_learning_log() for the /api/learning-log endpoint.
Only significant events (new blocked context, threshold shift ≥ 3 pts, retrain
triggered) emit a single INFO line to the root logger so the console stays clean.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROCESS_START         = time.time()
_STARTUP_RETRAIN_GRACE = 360     # no ML retrain for 6 min — lets scanner warm SQLite first

# ── Tunable parameters ────────────────────────────────────────────────────────
LEARN_INTERVAL_SECS   = 90      # how often to run a learning cycle (1.5 min)
RETRAIN_MIN_NEW       = 15      # new outcomes needed to trigger ML retrain
RETRAIN_COOLDOWN_SECS = 600     # don't retrain more often than every 10 min
RETRAIN_ACTIVE_SESSIONS = os.getenv("LEARNING_RETRAIN_ACTIVE_SESSIONS", "1").lower() in ("1", "true", "yes")
RETRAIN_INCLUDE_DEEP = os.getenv("LEARNING_RETRAIN_INCLUDE_DEEP", "1").lower() in ("1", "true", "yes")

# ── Persistent log file (JSONL — one entry per line, append-only) ─────────────
_LOG_PATH   = Path(__file__).parent.parent / "data" / "learning_log.jsonl"
_MAX_LOG_LINES = 5000   # keep last 5000 lines; older entries trimmed on startup

# ── In-memory ring buffer for dashboard log ───────────────────────────────────
_LOG_BUFFER: deque[dict] = deque(maxlen=500)
_buf_lock   = threading.Lock()
_log_file_lock = threading.Lock()


def _load_log_from_disk() -> None:
    """Pre-populate the in-memory buffer from the persisted JSONL file."""
    try:
        if not _LOG_PATH.exists():
            return
        lines = _LOG_PATH.read_text().splitlines()
        # Trim to max on startup to prevent unbounded growth
        if len(lines) > _MAX_LOG_LINES:
            lines = lines[-_MAX_LOG_LINES:]
            _LOG_PATH.write_text("\n".join(lines) + "\n")
        with _buf_lock:
            for line in lines[-500:]:   # fill buffer with last 500
                try:
                    _LOG_BUFFER.append(json.loads(line))
                except Exception:
                    pass
        logger.info(f"[Learning] Restored {min(500, len(lines))} log entries from disk.")
    except Exception as e:
        logger.warning(f"[Learning] Could not load log from disk: {e}")


def _log(msg: str, level: str = "INFO", significant: bool = False) -> None:
    """Append to ring buffer and persist to JSONL file."""
    entry = {
        "ts":      datetime.now(timezone.utc).isoformat(),
        "level":   level,
        "message": msg,
    }
    with _buf_lock:
        _LOG_BUFFER.append(entry)
    # Append to disk (non-blocking, fire-and-forget)
    try:
        with _log_file_lock:
            _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
    if significant:
        logger.info(f"[Learning] {msg}")


def get_learning_log(limit: int = 100) -> list[dict]:
    """Return the last `limit` log entries for the dashboard API."""
    with _buf_lock:
        entries = list(_LOG_BUFFER)
    return entries[-limit:]


def clear_log() -> None:
    """Wipe both the in-memory buffer and the on-disk JSONL file."""
    with _buf_lock:
        _LOG_BUFFER.clear()
    try:
        with _log_file_lock:
            if _LOG_PATH.exists():
                _LOG_PATH.write_text("")
    except Exception as e:
        logger.warning(f"[Learning] Could not clear log file: {e}")
    logger.info("[Learning] Log cleared on service restart.")


# Load persisted log entries on module import so dashboard shows history immediately
_load_log_from_disk()


# ── Learning engine ───────────────────────────────────────────────────────────

class LearningEngine:
    """
    Singleton background learning thread.

    Usage:
        engine = LearningEngine()
        engine.start()          # call once at app startup
        engine.stop()           # call on shutdown
        engine.get_status()     # returns current state for API
    """

    def __init__(self):
        self._thread:           Optional[threading.Thread] = None
        self._feedback_thread:  Optional[threading.Thread] = None
        self._running:          bool  = False
        self._cycle_count:      int   = 0
        self._last_retrain_t:   float = 0.0
        self._last_bt_count:    int   = 0
        self._last_pt_count:    int   = 0
        self._last_cycle_ts:    Optional[str] = None
        self._last_win_rate:    float = 0.0
        self._last_threshold:   float = 65.0
        self._last_blocked_n:   int   = 0     # track to only log NEW blocked contexts
        self._last_feedback_at: Optional[float] = None  # epoch of last trade-close feedback
        self._feedback_count:   int   = 0
        self._restore_state_from_db()

    def _restore_state_from_db(self) -> None:
        """
        Restore outcome counts from PostgreSQL so restarts don't trigger false
        ML retrains (counts would otherwise reset to 0, making every existing
        outcome appear "new").
        """
        try:
            from agent.db import get_conn, using_postgres
            if not using_postgres():
                return
            with get_conn() as c:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS system_kv (
                        key        TEXT PRIMARY KEY,
                        value      TEXT NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                row = c.execute(
                    "SELECT value FROM system_kv WHERE key = 'learning_engine_state'"
                ).fetchone()
            if row:
                saved = json.loads(row["value"])
                self._last_bt_count = int(saved.get("last_bt_count", 0))
                self._last_pt_count = int(saved.get("last_pt_count", 0))
                self._last_threshold = float(saved.get("last_threshold", 65.0))
                logger.info(
                    f"[Learning] Restored state from DB — "
                    f"bt_count={self._last_bt_count} pt_count={self._last_pt_count}"
                )
        except Exception as e:
            logger.debug(f"[Learning] State restore skipped: {e}")

    def _save_state_to_db(self) -> None:
        """Persist restart-sensitive engine state to PostgreSQL."""
        try:
            from agent.db import get_conn, using_postgres
            if not using_postgres():
                return
            import json as _json
            payload = _json.dumps({
                "last_bt_count":  self._last_bt_count,
                "last_pt_count":  self._last_pt_count,
                "last_threshold": self._last_threshold,
                "cycle_count":    self._cycle_count,
                "updated_at":     datetime.now(timezone.utc).isoformat(),
            })
            with get_conn() as c:
                c.execute("""
                    INSERT INTO system_kv (key, value, updated_at)
                    VALUES ('learning_engine_state', %s, NOW())
                    ON CONFLICT (key) DO UPDATE
                        SET value = EXCLUDED.value, updated_at = NOW()
                """, (payload,))
                c.commit()
        except Exception as e:
            logger.debug(f"[Learning] State save skipped: {e}")

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, name="LearningEngine", daemon=True
        )
        self._thread.start()
        self._feedback_thread = threading.Thread(
            target=self._trade_feedback_loop, name="LearningFeedback", daemon=True
        )
        self._feedback_thread.start()
        _log("Learning engine started — running every "
             f"{LEARN_INTERVAL_SECS}s, 24/7 including after-hours", significant=True)

    def _trade_feedback_loop(self) -> None:
        """
        Subscribe to Valkey trade:closed pub/sub channel.
        On each trade close, immediately run a mini learning cycle for that algo's family.
        Target latency: < 5 seconds from trade close to param update.
        """
        import json as _json
        while self._running:
            try:
                from agent.valkey_client import _get_client
                client = _get_client()
                if client is None:
                    time.sleep(30)
                    continue
                pubsub = client.pubsub()
                pubsub.subscribe("trade:closed")
                _log("Trade feedback loop subscribed to trade:closed channel")
                for message in pubsub.listen():
                    if not self._running:
                        break
                    if not (message and message.get("type") == "message"):
                        continue
                    try:
                        data = _json.loads(message["data"])
                        self._run_trade_feedback(data)
                    except Exception as _msg_err:
                        _log(f"[Feedback] message parse error: {_msg_err}", level="DEBUG")
            except Exception as exc:
                _log(f"[Feedback] pub/sub error: {exc} — retrying in 15s", level="DEBUG")
                time.sleep(15)

    def _run_trade_feedback(self, trade_data: dict) -> None:
        """Run a mini learning cycle for the algo family of a just-closed trade."""
        try:
            import pandas as pd
            algo  = trade_data.get("algo", "") or ""
            if not algo:
                return

            from agent.algo_learning_engine import get_engine as _get_ale, _ALGO_FAMILY_MAP
            family = _ALGO_FAMILY_MAP.get(algo, "")
            if not family:
                return

            # Build a single-row outcomes dataframe for this trade
            row = {
                "outcome_id":  trade_data.get("outcome_id") or f"paper:{trade_data.get('trade_id', '')}",
                "pnl_pct":     trade_data.get("pnl_pct", 0.0),
                "pnl_dollar":  trade_data.get("pnl_dollar", 0.0),
                "exit_reason": trade_data.get("exit_reason", ""),
                "status":      "WIN" if float(trade_data.get("pnl_dollar", 0.0) or 0.0) > 0 else "LOSS",
                "algo_name":   algo,
                "direction":   trade_data.get("direction", ""),
                "regime":      trade_data.get("regime", ""),
                "session":     trade_data.get("session", ""),
                "vwap_event":  "",
            }
            df = pd.DataFrame([row])
            self._cycle_count += 1
            _get_ale().run_cycle(df, self._cycle_count)
            self._last_feedback_at = time.time()
            self._feedback_count  += 1
            _log(
                f"[Feedback] Mini-cycle for {family} ({algo}) — "
                f"trade #{trade_data.get('trade_id','?')} pnl={trade_data.get('pnl_pct',0):+.2f}%",
                level="DEBUG",
            )
        except Exception as exc:
            _log(f"[Feedback] run_trade_feedback error: {exc}", level="DEBUG")

    def stop(self) -> None:
        self._running = False

    def get_status(self) -> dict:
        try:
            from agent.config_manager import config as _cfg
            retrain_cooldown = int(
                _cfg.get("learner.feedback_retrain_cooldown_s", RETRAIN_COOLDOWN_SECS)
            )
            retrain_min_new = int(
                _cfg.get("learner.feedback_retrain_min_new", RETRAIN_MIN_NEW)
            )
        except Exception:
            retrain_cooldown = RETRAIN_COOLDOWN_SECS
            retrain_min_new = RETRAIN_MIN_NEW
        return {
            "running":        self._running,
            "cycle_count":    self._cycle_count,
            "last_cycle":     self._last_cycle_ts,
            "last_win_rate":  round(self._last_win_rate, 1),
            "last_threshold": round(self._last_threshold, 1),
            "interval_secs":  LEARN_INTERVAL_SECS,
            "last_retrain_at": self._last_retrain_t or None,
            "retrain_cooldown_secs": retrain_cooldown,
            "retrain_min_new": retrain_min_new,
            "active_session_retrain_enabled": RETRAIN_ACTIVE_SESSIONS,
            "deep_retrain_enabled": RETRAIN_INCLUDE_DEEP,
            "feedback_loop_active": (
                self._feedback_thread is not None and self._feedback_thread.is_alive()
            ),
            "last_feedback_at": self._last_feedback_at,
            "feedback_count":   self._feedback_count,
        }

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            try:
                self._run_cycle()
            except Exception as e:
                _log(f"Cycle error: {e}", level="ERROR")
            time.sleep(LEARN_INTERVAL_SECS)

    def _run_cycle(self) -> None:
        self._cycle_count += 1
        ts = datetime.now(timezone.utc).isoformat()
        _log(f"Cycle #{self._cycle_count} starting…", level="DEBUG")

        # ── 1. Fetch latest resolved stats from live backtest ─────────────────
        bt_stats, bt_count = self._get_bt_stats()

        # ── 2. Fetch latest paper trade stats ────────────────────────────────
        pt_stats, pt_count = self._get_pt_stats()

        # ── 3. Fetch real-time market observation stats ───────────────────────
        # SHORT-TERM direction checks — fast feedback but SYSTEMATICALLY BIASED
        # for mean-reversion signals (price moves against reversal for first few
        # bars).  Kept SEPARATE from trade stats so it only informs CONTEXT
        # pattern learning, never the main win rate or confidence threshold.
        obs_stats, obs_count = self._get_observation_stats()

        # ── 4. Merge ONLY high-quality trade sources ──────────────────────────
        # Observation stats are sent separately as source="observation" so they
        # cannot corrupt current_win_rate or dynamic_threshold.
        merged = self._merge_stats(bt_stats, pt_stats)

        if merged["overall"]["total"] < 3:
            # Still push observation context patterns even when trade data is thin
            if obs_count >= 5:
                from agent.adaptive_filter import update_filter as _obs_update
                _obs_update(obs_stats, source="observation")
            _log(f"Cycle #{self._cycle_count}: only {merged['overall']['total']} "
                 "trade outcomes — waiting for more", level="DEBUG")
            self._last_cycle_ts = ts
            return

        # ── 5. Push merged trade stats into adaptive filter ───────────────────
        from agent.adaptive_filter import update_filter, get_status as af_status
        prev_threshold = self._last_threshold
        update_filter(merged, source="backtest")

        # Also push observation stats for context-only learning
        if obs_count >= 5:
            update_filter(obs_stats, source="observation")

        status = af_status()
        new_wr        = float(status.get("current_win_rate", 0.0))
        new_threshold = float(status.get("dynamic_threshold", 65.0))
        blocked_n     = len(status.get("blocked_contexts", {}))
        boosted_n     = len(status.get("boosted_contexts", {}))

        # Log summary to ring buffer (always)
        _log(
            f"Cycle #{self._cycle_count} — WR:{new_wr:.1f}%  "
            f"gate:{new_threshold:.1f}%  blocked:{blocked_n}  boosted:{boosted_n}  "
            f"bt:{bt_count} pt:{pt_count} obs:{obs_count} total:{merged['overall']['total']}"
        )

        # Only surface significant changes to console
        if abs(new_threshold - prev_threshold) >= 3.0:
            _log(
                f"Threshold shifted {prev_threshold:.1f}% → {new_threshold:.1f}%  "
                f"(WR {new_wr:.1f}%)", significant=True
            )
        if blocked_n > self._last_blocked_n:
            _log(f"New context blocked — {blocked_n} total suppressed contexts",
                 significant=True)
            self._last_blocked_n = blocked_n

        self._last_win_rate  = new_wr
        self._last_threshold = new_threshold
        self._last_cycle_ts  = ts

        # ── Wire in algo learning engine ──────────────────────────────────────
        try:
            from agent.algo_learning_engine import get_engine as _get_ale
            from agent.live_backtest import get_outcomes_for_ml
            outcomes_df = get_outcomes_for_ml(min_count=1)
            if outcomes_df is not None and not outcomes_df.empty:
                _get_ale().run_cycle(outcomes_df, self._cycle_count)
        except Exception as _ale_err:
            _log(f"AlgoLearningEngine cycle error: {_ale_err}", level="WARNING")

        # ── 5. Maybe trigger ML model retrain (heavier operation) ────────────
        new_bt_outcomes = bt_count - self._last_bt_count
        new_pt_outcomes = pt_count - self._last_pt_count
        total_new       = new_bt_outcomes + new_pt_outcomes
        try:
            from agent.config_manager import config as _cfg
            retrain_min_new = int(
                _cfg.get("learner.feedback_retrain_min_new", RETRAIN_MIN_NEW)
            )
            retrain_cooldown = int(
                _cfg.get("learner.feedback_retrain_cooldown_s", RETRAIN_COOLDOWN_SECS)
            )
        except Exception:
            retrain_min_new = RETRAIN_MIN_NEW
            retrain_cooldown = RETRAIN_COOLDOWN_SECS
        cooldown_ok = (time.time() - self._last_retrain_t) > retrain_cooldown

        if total_new >= retrain_min_new and cooldown_ok:
            self._last_bt_count  = bt_count
            self._last_pt_count  = pt_count
            # Startup grace: suppress retrain until the scanner has warmed SQLite.
            # The loop fires immediately at startup; without this gate it hits the
            # Schwab API before the first scan, starving the scanner's rate budget.
            if time.time() - _PROCESS_START < _STARTUP_RETRAIN_GRACE:
                _log("Retrain deferred — startup SQLite warm-up in progress", level="INFO")
            else:
                self._last_retrain_t = time.time()
                self._trigger_retrain(merged)
        else:
            # Update counts even when not retraining (avoid stale baseline)
            self._last_bt_count = max(self._last_bt_count, bt_count)
            self._last_pt_count = max(self._last_pt_count, pt_count)

        # Persist state so restarts don't reset counts and trigger false retrains
        self._save_state_to_db()

    # ── Data fetchers ─────────────────────────────────────────────────────────

    def _get_bt_stats(self) -> tuple[dict, int]:
        """Fetch live backtest resolved stats (last 30 days)."""
        try:
            from agent.live_backtest import get_performance_stats
            stats = get_performance_stats(lookback_days=30)
            count = stats.get("overall", {}).get("total", 0)
            return stats, count
        except Exception as e:
            _log(f"bt_stats error: {e}", level="WARNING")
            return {}, 0

    def _get_pt_stats(self) -> tuple[dict, int]:
        """Fetch paper trading closed-trade stats."""
        try:
            from agent.paper_trading import _build_paper_stats
            stats = _build_paper_stats()
            count = stats.get("overall", {}).get("total", 0)
            return stats, count
        except Exception as e:
            _log(f"pt_stats error: {e}", level="WARNING")
            return {}, 0

    def _get_observation_stats(self) -> tuple[dict, int]:
        """
        Fetch real-time market observation stats from signal_tracker.
        These are computed from short-term price checks (every 90s) across all
        context dimensions: session, trading tier, AH tier, regime, VWAP, RSI,
        volume bucket. Updates continuously regardless of paper trade activity.
        """
        try:
            from agent.signal_tracker import get_market_breakdown_stats, get_observation_summary
            stats   = get_market_breakdown_stats(min_count=5, lookback_days=30)
            summary = get_observation_summary()
            count   = summary.get("resolved", 0)
            if count > 0 and count != getattr(self, "_last_obs_count", -1):
                _log(
                    f"Market observations: {count} resolved, "
                    f"WR={summary.get('observation_wr', 0):.1f}% "
                    f"across {stats.get('overall', {}).get('total', 0)} signals",
                    level="DEBUG",
                )
                self._last_obs_count = count  # type: ignore[attr-defined]
            return stats, count
        except Exception as e:
            _log(f"obs_stats error: {e}", level="WARNING")
            return {}, 0

    # ── Stat merger ───────────────────────────────────────────────────────────

    def _merge_stats(self, bt: dict, pt: dict) -> dict:
        """
        Merge backtest and paper trading stats into a single stats dict.
        For each breakdown dimension, combines win/total counts before computing
        the merged win rate — giving us more data points for each context bucket.
        """
        if not bt and not pt:
            return {"overall": {"total": 0, "wins": 0, "win_rate": 0.0}}
        if not bt:
            return pt
        if not pt:
            return bt

        def _merge_breakdowns(bd_bt: dict, bd_pt: dict) -> dict:
            merged: dict = {}
            keys = set(bd_bt) | set(bd_pt)
            for k in keys:
                a = bd_bt.get(k, {"total": 0, "wins": 0, "win_rate": 0.0})
                b = bd_pt.get(k, {"total": 0, "wins": 0, "win_rate": 0.0})
                total = (a.get("total", 0) or 0) + (b.get("total", 0) or 0)
                wins  = (a.get("wins",  0) or 0) + (b.get("wins",  0) or 0)
                merged[k] = {
                    "total":    total,
                    "wins":     wins,
                    "win_rate": round(wins / total, 4) if total else 0.0,
                }
            return merged

        # Merge overall
        bt_o   = bt.get("overall", {})
        pt_o   = pt.get("overall", {})
        tot    = (bt_o.get("total", 0) or 0) + (pt_o.get("total", 0) or 0)
        wins   = (bt_o.get("wins",  0) or 0) + (pt_o.get("wins",  0) or 0)
        overall = {
            "total":    tot,
            "wins":     wins,
            "win_rate": round(wins / tot, 4) if tot else 0.0,
        }

        dims = [
            "by_vwap_event", "by_session", "by_regime", "by_rsi_zone",
            "by_entry_type", "by_direction", "by_sector_trend",
            "by_confidence", "by_ah_bias",
        ]
        result = {"overall": overall}
        for dim in dims:
            result[dim] = _merge_breakdowns(bt.get(dim, {}), pt.get(dim, {}))
        return result

    # ── ML retrain ───────────────────────────────────────────────────────────

    def _trigger_retrain(self, stats: dict) -> None:
        """
        Background ML model retrain — runs in its own thread so the learning
        loop isn't blocked. Uses outcome-weighted samples from both backtest
        and paper trading.
        """
        # Skip full retrain when weekend_learner is running its own deep retrain
        # to avoid CPU/API contention.
        try:
            from agent.weekend_learner import is_running as _wl_running
            if _wl_running():
                _log("Skipping retrain — weekend learning in progress", level="DEBUG")
                return
        except Exception:
            pass

        try:
            from agent.market_hours import get_market_session
            _session = get_market_session()
        except Exception as exc:
            _session = "UNKNOWN"
            _log(f"ML feedback retrain session check failed: {exc}; continuing in learner container", level="WARNING")
        _actual_session = _session
        if RETRAIN_ACTIVE_SESSIONS and _session != "CLOSED":
            _log(
                f"ML feedback retrain continuing during session={_session} inside learner container",
                level="INFO",
            )
            _session = "CLOSED"

        if _session != "CLOSED":
            _log(f"ML feedback retrain deferred — session={_session}; waiting for CLOSED window", level="INFO")
            return

        _log(f"ML feedback retrain triggered — "
             f"{stats['overall']['total']} total outcomes, session={_actual_session}", significant=True)

        def _do_retrain():
            try:
                from agent.backtest_reporter import (
                    _log_attribution, _update_confidence_calibration,
                )
                from agent.live_backtest import get_outcomes_for_ml
                from agent.ml_model import retrain_all
                from config import TRAINING_TICKERS

                outcomes_df = get_outcomes_for_ml(min_count=5)
                if outcomes_df is not None and not outcomes_df.empty:
                    _log_attribution(outcomes_df)
                    _update_confidence_calibration(outcomes_df)

                # During market hours the learner's continuous_deep_loop owns deep
                # BiLSTM fine-tuning (dashboard-visible, persistence-aware). Skip
                # deep here in active sessions so we never run two competing deep
                # passes that would just contend on deep_model._is_training_now.
                _skip_deep = (not RETRAIN_INCLUDE_DEEP) or (_actual_session != "CLOSED")
                retrain_all(TRAINING_TICKERS, skip_deep=_skip_deep)
                _log(f"ML retrain complete (deep={'skipped' if _skip_deep else 'included'})",
                     significant=True)
            except Exception as e:
                _log(f"ML retrain failed: {e}", level="ERROR", significant=True)

        threading.Thread(target=_do_retrain, name="LearningRetrain", daemon=True).start()


# ── Module-level singleton ────────────────────────────────────────────────────
learning_engine = LearningEngine()
