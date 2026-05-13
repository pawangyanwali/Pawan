"""
Adaptive signal filter — self-learning system that suppresses losing patterns
and dynamically raises the confidence gate until win rate targets are met.

How it works
------------
1.  After every feedback retrain cycle (≥20 new outcomes) the filter reads
    the live backtest performance stats.
2.  For each context dimension (vwap_event, session, regime, rsi_zone,
    entry_type, direction) it identifies contexts where win_rate < SUPPRESS_BELOW
    with at least MIN_SAMPLE resolved trades.
3.  Suppressed contexts are stored in a JSON file so they survive restarts.
4.  The dynamic confidence threshold is computed by finding the lowest confidence
    band that historically achieves >= TARGET_WIN_RATE.
5.  In scanner.py, before a signal is emitted, should_suppress() is called.
    If suppressed the signal direction is forced to NEUTRAL so it never reaches
    paper trading or the signal table as a tradeable idea.

Target: 62% win rate. The filter tightens automatically until reached.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Tunable parameters ────────────────────────────────────────────────────────
TARGET_WIN_RATE   = 0.62   # goal win rate — system tightens until reached
SUPPRESS_BELOW    = 0.35   # suppress context if win_rate < this
BOOST_ABOVE       = 0.72   # boost confidence if win_rate >= this
MIN_SAMPLE        = 8      # minimum resolved trades before suppressing a context
RELAX_ABOVE       = 0.92   # if win rate exceeds this, slightly relax threshold
DEFAULT_THRESHOLD = 60.0   # starting dynamic confidence gate
MIN_THRESHOLD     = 50.0   # never go below this (avoids suppressing all signals)
MAX_THRESHOLD     = 85.0   # never require more than this

_FILTER_PATH = Path(__file__).parent.parent / "data" / "adaptive_filter.json"
_lock = threading.Lock()

# ── In-memory state ───────────────────────────────────────────────────────────
_state: dict = {
    "blocked_contexts":    {},     # "vwap_event:REJECTION" → {win_rate, count, reason}
    "boosted_contexts":    {},     # "vwap_event:RECLAIM"   → {win_rate, count}
    "dynamic_threshold":   DEFAULT_THRESHOLD,
    "current_win_rate":    0.0,
    "total_resolved":      0,
    "last_updated":        None,
    "suppressed_count":    0,      # how many signals suppressed this session
    "threshold_history":   [],     # [{threshold, win_rate, ts}] — last 10
}


# ── Persistence ───────────────────────────────────────────────────────────────

def _load():
    global _state
    try:
        if _FILTER_PATH.exists():
            with open(_FILTER_PATH) as f:
                saved = json.load(f)
            with _lock:
                _state.update(saved)
            logger.info(
                f"[AdaptiveFilter] Loaded — threshold={_state['dynamic_threshold']:.1f}%  "
                f"blocked={len(_state['blocked_contexts'])}  "
                f"win_rate={_state['current_win_rate']*100:.1f}%"
            )
    except Exception as e:
        logger.warning(f"[AdaptiveFilter] Could not load saved state: {e}")


def _save():
    try:
        _FILTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            snapshot = dict(_state)
        with open(_FILTER_PATH, "w") as f:
            json.dump(snapshot, f, indent=2)
    except Exception as e:
        logger.warning(f"[AdaptiveFilter] Could not save state: {e}")


# Load persisted state at import time
_load()


# ── Core update ───────────────────────────────────────────────────────────────

def update_filter(stats: dict) -> None:
    """
    Called after each backtest feedback retrain with fresh performance stats.
    Updates blocked/boosted context lists and the dynamic confidence threshold.
    """
    _apply_stats(stats, source="backtest")


def update_from_paper_trades(stats: dict) -> None:
    """
    Called immediately after each paper trade closes.
    Merges paper trading outcomes into the filter — complementing backtest data
    with real simulated P&L so the system learns continuously, not just every
    20 backtest resolutions.
    """
    _apply_stats(stats, source="paper")


def _apply_stats(stats: dict, source: str = "backtest") -> None:
    """Shared logic for update_filter and update_from_paper_trades."""
    overall = stats.get("overall", {})
    total   = overall.get("total", 0)
    if total < MIN_SAMPLE:
        return

    current_wr = float(overall.get("win_rate", 0.0))
    new_blocked: dict = {}
    new_boosted: dict = {}

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

    for dim, breakdown in context_keys:
        for val, s in breakdown.items():
            count = s.get("total", 0)
            wr    = float(s.get("win_rate", 0.0))
            if count < MIN_SAMPLE:
                continue
            key = f"{dim}:{val}"
            if wr < SUPPRESS_BELOW:
                new_blocked[key] = {
                    "win_rate": round(wr, 3),
                    "count":    count,
                    "reason":   f"{dim}={val} wins only {wr*100:.0f}% ({count} trades)",
                }
            elif wr >= BOOST_ABOVE:
                new_boosted[key] = {"win_rate": round(wr, 3), "count": count}

    # Dynamic threshold: find confidence band achieving TARGET_WIN_RATE
    new_threshold = _compute_threshold(stats.get("by_confidence", {}), current_wr)

    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()

    with _lock:
        _state["blocked_contexts"]  = new_blocked
        _state["boosted_contexts"]  = new_boosted
        _state["dynamic_threshold"] = new_threshold
        _state["current_win_rate"]  = current_wr
        _state["total_resolved"]    = total
        _state["last_updated"]      = ts
        history = _state.setdefault("threshold_history", [])
        history.append({"threshold": new_threshold, "win_rate": current_wr, "ts": ts})
        _state["threshold_history"] = history[-10:]  # keep last 10

    _save()

    # Routine stats go to DEBUG — only surface to console if threshold shifted significantly
    logger.debug(
        f"[AdaptiveFilter:{source}] Updated — win_rate={current_wr*100:.1f}%  "
        f"threshold={new_threshold:.1f}%  "
        f"blocked={len(new_blocked)}  boosted={len(new_boosted)}  "
        f"total={total}"
    )
    if new_blocked:
        for k, v in new_blocked.items():
            logger.debug(f"  [SUPPRESS] {v['reason']}")


def _compute_threshold(by_confidence: dict, current_wr: float) -> float:
    """
    Find the minimum confidence level where historical win rate >= TARGET_WIN_RATE.

    Bands (from backtest stats): '<50', '50-60', '60-70', '70-80', '80+'
    We want the lowest band's lower-bound where cumulative above that band
    achieves the target.
    """
    band_order = [("<50", 0), ("50-60", 50), ("60-70", 60), ("70-80", 70), ("80+", 80)]

    # Build cumulative stats starting from the highest band and working down
    # "signals with confidence >= X" — find the lowest X giving >= 90% WR
    cumulative_wins   = 0
    cumulative_total  = 0
    best_threshold    = MAX_THRESHOLD  # default: very selective

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
            # This band and above achieves the target — lower threshold to this band
            best_threshold = float(lower_bound) if lower_bound > 0 else DEFAULT_THRESHOLD

    # If overall win rate is above target already, we can relax slightly
    if current_wr >= RELAX_ABOVE:
        best_threshold = max(MIN_THRESHOLD, best_threshold - 5.0)

    # If no band achieves 90%, progressively raise the threshold
    if best_threshold == MAX_THRESHOLD and current_wr < TARGET_WIN_RATE:
        gap = TARGET_WIN_RATE - current_wr
        # Raise proportionally: 10% below target → raise by 10 points
        best_threshold = min(MAX_THRESHOLD, DEFAULT_THRESHOLD + gap * 100)

    return round(float(max(MIN_THRESHOLD, min(MAX_THRESHOLD, best_threshold))), 1)


# ── Signal suppression check ──────────────────────────────────────────────────

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
    """
    Check whether a signal should be suppressed based on learned losing patterns.

    Returns (suppress: bool, reason: str).
    suppress=True means change direction to NEUTRAL — don't trade this setup.
    """
    with _lock:
        blocked  = dict(_state["blocked_contexts"])
        threshold = float(_state["dynamic_threshold"])

    checks = [
        ("vwap_event",   vwap_event),
        ("session",      session),
        ("regime",       regime),
        ("rsi_zone",     rsi_zone),
        ("entry_type",   entry_type),
        ("direction",    direction),
        ("sector_trend", sector_trend),
    ]

    for dim, val in checks:
        if not val:
            continue
        key = f"{dim}:{val}"
        if key in blocked:
            info = blocked[key]
            return True, f"Suppressed: {info['reason']}"

    # Confidence gate — use dynamic threshold learned from outcomes
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
    """
    Return a confidence boost (positive) for high-win-rate contexts.
    Applied in addition to backtest_reporter.adjust_confidence().
    """
    with _lock:
        boosted = dict(_state["boosted_contexts"])

    boosts = []
    for dim, val in [("vwap_event", vwap_event), ("session", session),
                     ("regime", regime), ("rsi_zone", rsi_zone),
                     ("entry_type", entry_type), ("direction", direction)]:
        key = f"{dim}:{val}"
        if key in boosted:
            wr = boosted[key]["win_rate"]
            boosts.append((wr - BOOST_ABOVE) * 30)  # up to +8.4 per context

    return round(sum(boosts) / len(boosts), 1) if boosts else 0.0


def get_status() -> dict:
    """Return current filter state for the API and dashboard."""
    with _lock:
        return {
            "dynamic_threshold":  _state["dynamic_threshold"],
            "current_win_rate":   round(_state["current_win_rate"] * 100, 1),
            "target_win_rate":    TARGET_WIN_RATE * 100,
            "total_resolved":     _state["total_resolved"],
            "blocked_contexts":   _state["blocked_contexts"],
            "boosted_contexts":   _state["boosted_contexts"],
            "suppressed_count":   _state["suppressed_count"],
            "last_updated":       _state["last_updated"],
            "threshold_history":  _state["threshold_history"],
            "is_learning":        _state["total_resolved"] >= MIN_SAMPLE,
        }


def increment_suppressed():
    with _lock:
        _state["suppressed_count"] = _state.get("suppressed_count", 0) + 1
