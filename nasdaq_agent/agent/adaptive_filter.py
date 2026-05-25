"""
Adaptive signal filter — self-learning system that suppresses losing patterns
and dynamically raises the confidence gate until win rate targets are met.

Architecture (two-tier data quality)
-------------------------------------
TIER 1  "trade" sources  ("backtest", "paper", "weekend_walk_forward")
        → Actual TP/SL resolved trades — HIGH quality, slow feedback (~hours)
        → Updates: current_win_rate, dynamic_threshold, context blocks/boosts
        → Uses EWMA so a single bad batch doesn't destroy accumulated history

TIER 2  "observation" source  (short-term 90s direction checks from scanner)
        → Fast feedback (~90s) but SYSTEMATICALLY BIASED for mean-reversion:
          price almost always moves against a reversal signal for the first
          few bars, so 90s-resolution accuracy is ~15-30% even for trades
          that eventually hit target.
        → Updates: observation_win_rate + context blocks/boosts ONLY
        → Does NOT touch current_win_rate or dynamic_threshold

Anti-deadlock
-------------
Scenario: low WR → high threshold → signals suppressed → no trades → no new
data → WR stays low → threshold stays high → forever stuck.

Fix: if dynamic_threshold has been at MAX and WR hasn't improved for
MAX_STUCK_CYCLES consecutive "trade" updates → auto-relax to DEFAULT.
This lets signals flow again so the system can generate new learning data.

EWMA
----
current_win_rate is updated as an exponential moving average (α = 0.40).
At α=0.40: a single update provides 40% of the new value, so ~5 "trade"
updates are needed to fully reflect a regime change. Fast enough to react
to real shifts, slow enough to resist random noise.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Tunable parameters ────────────────────────────────────────────────────────
TARGET_WIN_RATE   = 0.55   # goal win rate — system tightens until reached
SUPPRESS_BELOW    = 0.35   # suppress context pattern if win_rate < this
BOOST_ABOVE       = 0.72   # boost confidence if win_rate >= this
MIN_SAMPLE        = 8      # minimum resolved trades before suppressing a context
RELAX_ABOVE       = 0.85   # if win rate exceeds this, slightly relax threshold
DEFAULT_THRESHOLD = 55.0   # starting confidence gate
MIN_THRESHOLD     = 50.0   # never go below this
MAX_THRESHOLD     = 63.0   # never require more than this — high gates kill signal flow
BOOTSTRAP_OUTCOMES = 30    # outcomes needed before threshold can rise above DEFAULT
EWMA_ALPHA        = 0.40   # exponential smoothing: 0.40 = fast adaptation (~5 updates)
MAX_STUCK_CYCLES  = 4      # consecutive max-threshold trade updates before anti-deadlock fires

# Sources that produce reliable TP/SL-resolved trade data
_TRADE_SOURCES = {"backtest", "paper", "weekend_walk_forward"}

_FILTER_PATH = Path(__file__).parent.parent / "data" / "adaptive_filter.json"
_lock = threading.Lock()

# ── In-memory state ───────────────────────────────────────────────────────────
_state: dict = {
    "blocked_contexts":    {},
    "boosted_contexts":    {},
    "dynamic_threshold":   DEFAULT_THRESHOLD,
    "current_win_rate":    0.0,   # EWMA of trade-source win rates (fraction 0-1)
    "observation_win_rate": 0.0,  # short-term direction accuracy (display only)
    "total_resolved":      0,
    "last_updated":        None,
    "suppressed_count":    0,
    "threshold_history":   [],
    "false_negative_count": 0,
    "_stuck_cycles":       0,     # consecutive cycles at MAX with low WR
}


# ── Persistence ───────────────────────────────────────────────────────────────

_KV_KEY = "adaptive_filter_state"


def _load_from_db() -> dict | None:
    """Restore adaptive filter state from PostgreSQL system_kv table."""
    try:
        from agent.db import get_conn, using_postgres
        if not using_postgres():
            return None
        with get_conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS system_kv (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            row = c.execute(
                "SELECT value FROM system_kv WHERE key = %s", (_KV_KEY,)
            ).fetchone()
        if row:
            return json.loads(row["value"])
    except Exception as e:
        logger.debug(f"[AdaptiveFilter] DB restore skipped: {e}")
    return None


def _save_to_db(snapshot: dict) -> None:
    """Backup adaptive filter state to PostgreSQL system_kv table."""
    try:
        from agent.db import get_conn, using_postgres
        if not using_postgres():
            return
        payload = json.dumps(snapshot)
        with get_conn() as c:
            c.execute("""
                INSERT INTO system_kv (key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = NOW()
            """, (_KV_KEY, payload))
            c.commit()
    except Exception as e:
        logger.debug(f"[AdaptiveFilter] DB backup skipped: {e}")


def _load():
    global _state
    try:
        if _FILTER_PATH.exists():
            with open(_FILTER_PATH) as f:
                saved = json.load(f)
            with _lock:
                _state.update(saved)
                if _state["dynamic_threshold"] > MAX_THRESHOLD:
                    _state["dynamic_threshold"] = DEFAULT_THRESHOLD
                    logger.info(
                        f"[AdaptiveFilter] Reset persisted threshold to DEFAULT "
                        f"{DEFAULT_THRESHOLD}% (was above MAX {MAX_THRESHOLD}%)"
                    )
            logger.info(
                f"[AdaptiveFilter] Loaded — threshold={_state['dynamic_threshold']:.1f}%  "
                f"blocked={len(_state['blocked_contexts'])}  "
                f"trade_WR={_state['current_win_rate']*100:.1f}%  "
                f"obs_WR={_state.get('observation_win_rate', 0)*100:.1f}%"
            )
            return
    except Exception as e:
        logger.warning(f"[AdaptiveFilter] Could not load local state: {e}")

    # Local file missing (fresh deployment) — try PostgreSQL backup
    saved = _load_from_db()
    if saved:
        with _lock:
            _state.update(saved)
            if _state["dynamic_threshold"] > MAX_THRESHOLD:
                _state["dynamic_threshold"] = DEFAULT_THRESHOLD
        logger.info(
            f"[AdaptiveFilter] Restored from DB — threshold={_state['dynamic_threshold']:.1f}%  "
            f"blocked={len(_state['blocked_contexts'])}  "
            f"trade_WR={_state['current_win_rate']*100:.1f}%"
        )


def _save():
    try:
        _FILTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            snapshot = dict(_state)
        with open(_FILTER_PATH, "w") as f:
            json.dump(snapshot, f, indent=2)
        # Mirror to PostgreSQL so state survives fresh deployments
        _save_to_db(snapshot)
    except Exception as e:
        logger.warning(f"[AdaptiveFilter] Could not save state: {e}")


_load()

# ── Startup poisoned-state guard ──────────────────────────────────────────────
# If we previously crashed into a state with very low WR and many blocked
# contexts (false-positive flood from bad VWAP_LOSS logic), auto-reset rather
# than letting the system start in a permanently broken configuration.
def _auto_reset_if_poisoned() -> None:
    with _lock:
        wr      = _state.get("current_win_rate", 0.0)
        blocked = len(_state.get("blocked_contexts", {}))
    if wr < 0.30 and blocked > 10:
        logger.warning(
            f"[AdaptiveFilter] Poisoned state detected on load "
            f"(WR={wr*100:.1f}%, {blocked} blocked contexts) — auto-resetting."
        )
        with _lock:
            _state["blocked_contexts"]     = {}
            _state["boosted_contexts"]     = {}
            _state["dynamic_threshold"]    = DEFAULT_THRESHOLD
            _state["current_win_rate"]     = 0.0
            _state["observation_win_rate"] = 0.0
            _state["suppressed_count"]     = 0
            _state["_stuck_cycles"]        = 0
            _state["threshold_history"]    = []
            _state["last_updated"]         = None
        _save()
        logger.info(
            f"[AdaptiveFilter] Auto-reset complete — threshold → {DEFAULT_THRESHOLD}%"
        )

_auto_reset_if_poisoned()

# ── Core update ───────────────────────────────────────────────────────────────

def update_filter(stats: dict, source: str = "backtest") -> None:
    """
    Update adaptive filter from new performance stats.

    source="backtest" | "paper" | "weekend_walk_forward"
        → High-quality TP/SL data: updates win rate, threshold, and context patterns.

    source="observation"
        → Short-term direction checks: only updates context patterns and
          observation_win_rate.  Does NOT touch current_win_rate or threshold.
    """
    _apply_stats(stats, source=source)


def update_from_paper_trades(stats: dict) -> None:
    """Called immediately after each paper trade closes."""
    _apply_stats(stats, source="paper")


def _apply_stats(stats: dict, source: str = "backtest") -> None:
    overall    = stats.get("overall", {})
    total      = overall.get("total", 0)
    current_wr = float(overall.get("win_rate", 0.0))

    is_trade_source = source in _TRADE_SOURCES

    # ── Update win rate (trade sources only, with EWMA) ───────────────────────
    if is_trade_source and total > 0 and current_wr > 0:
        with _lock:
            old_wr = _state["current_win_rate"]
            # EWMA: blend new reading into history — prevents single-batch crashes
            if old_wr > 0:
                smoothed = EWMA_ALPHA * current_wr + (1.0 - EWMA_ALPHA) * old_wr
            else:
                smoothed = current_wr
            _state["current_win_rate"] = smoothed
            _state["total_resolved"]   = total
        _save()

    # ── Update observation win rate (separate, display-only) ──────────────────
    if not is_trade_source and total > 0 and current_wr > 0:
        with _lock:
            old_obs = _state.get("observation_win_rate", 0.0)
            if old_obs > 0:
                _state["observation_win_rate"] = 0.3 * current_wr + 0.7 * old_obs
            else:
                _state["observation_win_rate"] = current_wr

    if total < MIN_SAMPLE:
        return

    # ── Context blocking / boosting (ALL sources contribute) ─────────────────
    context_keys = [
        ("vwap_event",   stats.get("by_vwap_event",  {})),
        ("session",      stats.get("by_session",      {})),
        ("regime",       stats.get("by_regime",       {})),
        ("rsi_zone",     stats.get("by_rsi_zone",     {})),
        ("entry_type",   stats.get("by_entry_type",   {})),
        ("direction",    stats.get("by_direction",    {})),
        ("sector_trend", stats.get("by_sector_trend", {})),
        ("ah_bias",      stats.get("by_ah_bias",      {})),
    ]

    new_blocked: dict = {}
    new_boosted: dict = {}

    # For observation source: use higher MIN_SAMPLE and tighter SUPPRESS_BELOW
    # to avoid killing context patterns based on noisy short-term checks
    obs_suppress_below = 0.20 if not is_trade_source else SUPPRESS_BELOW
    obs_min_sample     = max(MIN_SAMPLE * 3, 25) if not is_trade_source else MIN_SAMPLE

    for dim, breakdown in context_keys:
        for val, s in breakdown.items():
            count = s.get("total", 0)
            wr    = float(s.get("win_rate", 0.0))
            if count < obs_min_sample:
                continue
            key = f"{dim}:{val}"
            if wr < obs_suppress_below:
                new_blocked[key] = {
                    "win_rate": round(wr, 3),
                    "count":    count,
                    "reason":   f"{dim}={val} wins only {wr*100:.0f}% ({count} trades) [{source}]",
                }
            elif wr >= BOOST_ABOVE and is_trade_source:
                # Only boost from high-quality trade data
                new_boosted[key] = {"win_rate": round(wr, 3), "count": count}

    # Observation source: only merge in NEW blocks, don't wipe existing trade-derived blocks
    if not is_trade_source:
        with _lock:
            existing_blocked = dict(_state["blocked_contexts"])
            existing_boosted = dict(_state["boosted_contexts"])
            existing_blocked.update(new_blocked)  # add/update obs-derived blocks
            _state["blocked_contexts"] = existing_blocked
            # Don't touch boosted from observations
        _save()
        return

    # ── Trade source: full update of threshold + contexts ─────────────────────
    with _lock:
        smoothed_wr   = _state["current_win_rate"]
        total_resolved = _state["total_resolved"]

    new_threshold = _compute_threshold(
        stats.get("by_confidence", {}), smoothed_wr, total_resolved
    )

    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()

    with _lock:
        # Anti-deadlock: if stuck at max threshold with very low WR → relax
        if (new_threshold >= MAX_THRESHOLD - 0.5 and smoothed_wr < 0.35):
            _state["_stuck_cycles"] = _state.get("_stuck_cycles", 0) + 1
            if _state["_stuck_cycles"] >= MAX_STUCK_CYCLES:
                new_threshold = DEFAULT_THRESHOLD
                _state["_stuck_cycles"] = 0
                logger.warning(
                    f"[AdaptiveFilter] ANTI-DEADLOCK: threshold reset to "
                    f"{DEFAULT_THRESHOLD}% after {MAX_STUCK_CYCLES} stuck cycles "
                    f"(WR={smoothed_wr*100:.1f}%)"
                )
        else:
            _state["_stuck_cycles"] = 0

        _state["blocked_contexts"]  = new_blocked
        _state["boosted_contexts"]  = new_boosted
        _state["dynamic_threshold"] = new_threshold
        _state["last_updated"]      = ts
        history = _state.setdefault("threshold_history", [])
        history.append({"threshold": new_threshold, "win_rate": smoothed_wr, "ts": ts})
        _state["threshold_history"] = history[-10:]

    _save()

    logger.debug(
        f"[AdaptiveFilter:{source}] trade_WR={smoothed_wr*100:.1f}% (raw={current_wr*100:.1f}%)  "
        f"threshold={new_threshold:.1f}%  "
        f"blocked={len(new_blocked)}  boosted={len(new_boosted)}  total={total}"
    )


def _compute_threshold(
    by_confidence:  dict,
    current_wr:     float,
    total_resolved: int = 0,
) -> float:
    """
    Find the minimum confidence level where historical win rate >= TARGET_WIN_RATE.

    Key change from original: max raise is capped at +4 points per update
    (original formula could raise by +11 points in a single update at low WR,
    which caused catastrophic gate-tightening and learning starvation).

    Bootstrap guard: don't raise above DEFAULT during first BOOTSTRAP_OUTCOMES.
    """
    band_order = [("<50", 0), ("50-60", 50), ("60-70", 60), ("70-80", 70), ("80+", 80)]

    best_threshold   = DEFAULT_THRESHOLD
    cumulative_wins  = 0
    cumulative_total = 0

    for band_label, lower_bound in reversed(band_order):
        s = by_confidence.get(band_label, {})
        wins  = s.get("wins", 0)
        total = s.get("total", 0)
        cumulative_wins  += wins
        cumulative_total += total
        if cumulative_total < MIN_SAMPLE:
            continue
        cum_wr = cumulative_wins / cumulative_total
        if cum_wr >= TARGET_WIN_RATE:
            best_threshold = float(lower_bound) if lower_bound > 0 else DEFAULT_THRESHOLD

    if current_wr >= RELAX_ABOVE:
        best_threshold = max(MIN_THRESHOLD, best_threshold - 5.0)

    if total_resolved < BOOTSTRAP_OUTCOMES:
        return round(float(max(MIN_THRESHOLD, min(DEFAULT_THRESHOLD, best_threshold))), 1)

    # Post-bootstrap: gentle raise — max +4 points per update cycle.
    # Original formula could raise +11 points (gap*25*scale), causing deadlock.
    # At 11% WR: gap=0.439, scale→1.0, raise=min(4, 0.439*10*1.0)=4.0 ✓
    if current_wr < TARGET_WIN_RATE:
        gap          = TARGET_WIN_RATE - current_wr
        scale        = min(1.0, (total_resolved - BOOTSTRAP_OUTCOMES) / 200.0)
        raise_amount = min(4.0, gap * 10 * scale)
        raised       = min(MAX_THRESHOLD, DEFAULT_THRESHOLD + raise_amount)
        best_threshold = max(best_threshold, raised)

    return round(float(max(MIN_THRESHOLD, min(MAX_THRESHOLD, best_threshold))), 1)


# ── Signal suppression check ──────────────────────────────────────────────────

def record_false_negative_check(direction: str, price_moved: float) -> None:
    from agent.signal_tracker import SHORT_WIN_PCT, SLIPPAGE_PCT
    threshold = SHORT_WIN_PCT + SLIPPAGE_PCT
    d = 1 if "BUY" in direction else -1
    if (price_moved * d) >= threshold:
        with _lock:
            _state["false_negative_count"] = _state.get("false_negative_count", 0) + 1
        _save()


def should_suppress(
    vwap_event:  str = "",
    session:     str = "",
    regime:      str = "",
    rsi_zone:    str = "",
    entry_type:  str = "",
    direction:   str = "",
    sector_trend: str = "",
    confidence:  float = 0.0,
) -> tuple[bool, str]:
    with _lock:
        blocked   = dict(_state["blocked_contexts"])
        threshold = float(_state["dynamic_threshold"])

    for dim, val in [
        ("vwap_event", vwap_event), ("session", session), ("regime", regime),
        ("rsi_zone", rsi_zone), ("entry_type", entry_type),
        ("direction", direction), ("sector_trend", sector_trend),
    ]:
        if not val:
            continue
        key = f"{dim}:{val}"
        if key in blocked:
            return True, f"Suppressed: {blocked[key]['reason']}"

    if confidence < threshold:
        return True, f"Confidence {confidence:.1f}% below learned threshold {threshold:.1f}%"

    return False, ""


def get_confidence_boost(
    vwap_event: str = "",
    session:    str = "",
    regime:     str = "",
    rsi_zone:   str = "",
    entry_type: str = "",
    direction:  str = "",
) -> float:
    with _lock:
        boosted = dict(_state["boosted_contexts"])

    boosts = []
    for dim, val in [("vwap_event", vwap_event), ("session", session),
                     ("regime", regime), ("rsi_zone", rsi_zone),
                     ("entry_type", entry_type), ("direction", direction)]:
        key = f"{dim}:{val}"
        if key in boosted:
            wr = boosted[key]["win_rate"]
            boosts.append((wr - BOOST_ABOVE) * 30)

    return round(sum(boosts) / len(boosts), 1) if boosts else 0.0


def get_status() -> dict:
    """Return current filter state for the API and dashboard."""
    with _lock:
        return {
            "dynamic_threshold":    _state["dynamic_threshold"],
            "current_win_rate":     round(_state["current_win_rate"] * 100, 1),
            "observation_win_rate": round(_state.get("observation_win_rate", 0.0) * 100, 1),
            "target_win_rate":      round(TARGET_WIN_RATE * 100, 1),   # fix 55.000000001% display
            "total_resolved":       _state["total_resolved"],
            "blocked_contexts":     _state["blocked_contexts"],
            "boosted_contexts":     _state["boosted_contexts"],
            "suppressed_count":     _state["suppressed_count"],
            "false_negative_count": _state.get("false_negative_count", 0),
            "last_updated":         _state["last_updated"],
            "threshold_history":    _state["threshold_history"],
            "is_learning":          _state["total_resolved"] >= MIN_SAMPLE,
            "stuck_cycles":         _state.get("_stuck_cycles", 0),
        }


def reset_filter(reason: str = "manual") -> dict:
    """
    Reset the learned state: clear all blocked/boosted contexts and recalibrate
    win rate from zero.  The confidence gate is restored to DEFAULT (55%).

    Called when:
      - User requests a manual reset via /api/reset-learning
      - System detects poisoned state at startup (win_rate < 25% with many blocks)

    Does NOT clear the live_backtest.db signal history — new outcomes will
    naturally rebuild correct context statistics from clean data.
    """
    global _state
    with _lock:
        _state["blocked_contexts"]     = {}
        _state["boosted_contexts"]     = {}
        _state["dynamic_threshold"]    = DEFAULT_THRESHOLD
        _state["current_win_rate"]     = 0.0
        _state["observation_win_rate"] = 0.0
        _state["suppressed_count"]     = 0
        _state["_stuck_cycles"]        = 0
        _state["threshold_history"]    = []
        _state["last_updated"]         = None
        snapshot = dict(_state)
    _save()
    logger.info(
        f"[AdaptiveFilter] RESET ({reason}): cleared all blocked/boosted contexts, "
        f"win_rate → 0.0, threshold → {DEFAULT_THRESHOLD}%"
    )
    return {"status": "reset", "reason": reason, "threshold": DEFAULT_THRESHOLD}


def increment_suppressed():
    with _lock:
        _state["suppressed_count"] = _state.get("suppressed_count", 0) + 1


