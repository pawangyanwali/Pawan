"""
Phase 1 Adaptive Trading Learning Engine — AI-TRD-CL-002.

Self-contained module with 10 components, all persisted to data/.
Every component has graceful fallback so the system never crashes if files
are missing.

Components:
  1. ParameterControlRegistry  — algo parameter tuning with bounds/cooldown
  2. OutcomeClassifier          — WIN/LOSS/BREAK_EVEN/TIMEOUT/INVALID
  3. LossAnalyzer               — 7 root-cause patterns for losses
  4. WinReinforcer              — EWMA context-win tracking
  5. AlgoSelector               — UCB-based algo weighting
  6. ParameterAdapter           — wires LossAnalyzer → ParameterControlRegistry
  7. CounterfactualSimulator    — shadow-trade what would have happened if filter passed
  8. ModelVersionRegistry       — champion/challenger with promotion gates
  9. RegimeTransitionHandler    — acts on open trades when regime changes
 10. AuditLogger                — append-only JSONL event log

AlgoLearningEngine — top-level coordinator (run_cycle is called by LearningEngine).

Module-level singletons are exposed at the bottom for clean import.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. ParameterControlRegistry ───────────────────────────────────────────────

# Algo → family mapping
_ALGO_FAMILY_MAP: dict[str, str] = {
    # ORB family
    "ORB5_BULL": "ORB", "ORB5_BEAR": "ORB",
    "ORB15_BULL": "ORB", "ORB15_BEAR": "ORB",
    # GAP_TREND family
    "GAP_AND_GO_BULL": "GAP_TREND", "GAP_AND_GO_BEAR": "GAP_TREND",
    # GAP_FADE family
    "GAP_FADE_BULL": "GAP_FADE", "GAP_FADE_BEAR": "GAP_FADE",
    # BREAKOUT family
    "PDH_BREAKOUT_BULL": "BREAKOUT", "PDL_BREAKDOWN_BEAR": "BREAKOUT",
    "HOD_BREAK_BULL": "BREAKOUT", "LOD_BREAK_BEAR": "BREAKOUT",
    # FLAG family
    "BULL_FLAG": "FLAG", "BEAR_FLAG": "FLAG",
    # VWAP_SCALP family
    "VWAP_TOUCH_SCALP_BULL": "VWAP_SCALP", "VWAP_TOUCH_SCALP_BEAR": "VWAP_SCALP",
    "VWAP_HOD_SCALP": "VWAP_SCALP", "VWAP_LOD_SCALP": "VWAP_SCALP",
    # LEVEL_SCALP family
    "LEVEL_REJECTION_SCALP_BULL": "LEVEL_SCALP", "LEVEL_REJECTION_SCALP_BEAR": "LEVEL_SCALP",
    "MICRO_PULLBACK_SCALP_BULL": "LEVEL_SCALP", "MICRO_PULLBACK_SCALP_BEAR": "LEVEL_SCALP",
    # RS_REGIME family
    "SPY_BETA_CATCHUP_BULL": "RS_REGIME", "SPY_BETA_CATCHUP_BEAR": "RS_REGIME",
    "RESIDUAL_MOMENTUM_BULL": "RS_REGIME", "RESIDUAL_REVERSION_BEAR": "RS_REGIME",
    "SECTOR_LEADER_BULL": "RS_REGIME", "SECTOR_LAGGARD_CATCHUP_BULL": "RS_REGIME",
    "SECTOR_COUNTER_FADE_BEAR": "RS_REGIME",
    "REGIME_ALIGNED_LONG": "RS_REGIME", "REGIME_ALIGNED_SHORT": "RS_REGIME",
    "SECTOR_BREAKOUT_BULL": "RS_REGIME", "SECTOR_BREAKOUT_BEAR": "RS_REGIME",
    "CS_RS_RANK_BULL": "RS_REGIME", "CS_RS_RANK_BEAR": "RS_REGIME",
}

_ALL_FAMILIES = {"ORB", "GAP_TREND", "GAP_FADE", "BREAKOUT", "FLAG", "VWAP_SCALP", "LEVEL_SCALP", "RS_REGIME"}

# Default parameter spec — applied to every family
_PARAM_SPEC: dict[str, dict] = {
    "target_mult": {
        "default": 1.5, "min": 0.75, "max": 3.0,
        "max_change": 0.10, "min_samples": 15, "cooldown_cycles": 5,
        "auto": "both",
    },
    "stop_mult": {
        "default": 1.0, "min": 0.50, "max": 2.0,
        "max_change": 0.05, "min_samples": 15, "cooldown_cycles": 5,
        "auto": "reduce_only",
    },
    "rvol_gate": {
        "default": 1.5, "min": 1.0, "max": 3.0,
        "max_change": 0.10, "min_samples": 10, "cooldown_cycles": 3,
        "auto": "both",
    },
    "conf_gate": {
        "default": 55.0, "min": 45.0, "max": 75.0,
        "max_change": 2.0, "min_samples": 20, "cooldown_cycles": 3,
        "auto": "increase_only",
    },
    "entry_window_bars": {
        "default": 3, "min": 1, "max": 8,
        "max_change": 1, "min_samples": 10, "cooldown_cycles": 5,
        "auto": "both",
    },
}


class ParameterControlRegistry:
    """
    Persists current tuned parameters per algo-family to data/algo_params.json.
    Enforces bounds, cooldown periods, and directional-only updates.
    """

    _PATH = _DATA_DIR / "algo_params.json"

    def __init__(self):
        self._lock = threading.Lock()
        # Structure: {family: {param: {current, previous, rollback, last_updated_cycle}}}
        self._state: dict[str, dict] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        for family in _ALL_FAMILIES:
            self._state[family] = {}
            for param, spec in _PARAM_SPEC.items():
                self._state[family][param] = {
                    "current": spec["default"],
                    "previous": spec["default"],
                    "rollback": spec["default"],
                    "last_updated_cycle": 0,
                }

    def load(self) -> None:
        try:
            if self._PATH.exists():
                raw = json.loads(self._PATH.read_text())
                with self._lock:
                    for family, params in raw.items():
                        if family in self._state:
                            for param, vals in params.items():
                                if param in self._state[family]:
                                    self._state[family][param].update(vals)
        except Exception as exc:
            logger.warning(f"[ParamRegistry] load error: {exc}")

    def save(self) -> None:
        try:
            with self._lock:
                data = {f: {p: dict(v) for p, v in params.items()}
                        for f, params in self._state.items()}
            self._PATH.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning(f"[ParamRegistry] save error: {exc}")

    def get(self, family: str, param: str) -> float:
        try:
            spec = _PARAM_SPEC.get(param, {})
            with self._lock:
                return float(self._state.get(family, {}).get(param, {}).get(
                    "current", spec.get("default", 0.0)
                ))
        except Exception:
            return float(_PARAM_SPEC.get(param, {}).get("default", 0.0))

    def update(self, family: str, param: str, new_val: float, reason: str, cycle_num: int) -> bool:
        """
        Attempt to update a parameter. Enforces:
          - family/param must exist
          - bounds (min, max)
          - max_change per update
          - cooldown_cycles between updates
          - directional constraint (auto = reduce_only | increase_only | both)
        Returns True if update was applied.
        """
        if family not in _ALL_FAMILIES:
            return False
        spec = _PARAM_SPEC.get(param)
        if spec is None:
            return False

        with self._lock:
            entry = self._state[family][param]
            current = float(entry["current"])
            last_cycle = int(entry.get("last_updated_cycle", 0))

            # Cooldown check
            if cycle_num - last_cycle < spec["cooldown_cycles"] and last_cycle > 0:
                return False

            # Directional constraint
            auto = spec["auto"]
            if auto == "increase_only" and new_val < current:
                return False
            if auto == "reduce_only" and new_val > current:
                return False

            # Bound clamp
            new_val = max(spec["min"], min(spec["max"], new_val))

            # Max change check
            if abs(new_val - current) > spec["max_change"]:
                # Clamp change to max_change
                if new_val > current:
                    new_val = current + spec["max_change"]
                else:
                    new_val = current - spec["max_change"]
                new_val = max(spec["min"], min(spec["max"], new_val))

            if abs(new_val - current) < 1e-9:
                return False

            entry["rollback"] = entry["previous"]
            entry["previous"] = current
            entry["current"] = round(new_val, 6)
            entry["last_updated_cycle"] = cycle_num

        logger.info(f"[ParamRegistry] {family}.{param}: {current} → {new_val:.4f} ({reason})")
        return True

    def get_all_params(self, algo_name: str) -> dict:
        """Return all current params for the algo's family. Falls back to defaults."""
        family = _ALGO_FAMILY_MAP.get(algo_name, "")
        if not family:
            return {p: s["default"] for p, s in _PARAM_SPEC.items()}
        with self._lock:
            fam_state = self._state.get(family, {})
            result = {}
            for param, spec in _PARAM_SPEC.items():
                result[param] = fam_state.get(param, {}).get("current", spec["default"])
        return result


# ── 2. OutcomeClassifier ──────────────────────────────────────────────────────

class OutcomeClassifier:
    """Pure-function outcome classifier. No persistence."""

    _WIN_THRESH      = 0.0015   # +0.15%
    _LOSS_THRESH     = -0.0015  # -0.15%

    @classmethod
    def classify(cls, pnl_pct: float, exit_reason: str) -> str:
        """
        Returns: WIN | LOSS | BREAK_EVEN | TIMEOUT | INVALID
        pnl_pct is fractional (e.g. 0.002 = +0.2%).
        TIMEOUT is returned as a label; also sub-classified economically.
        """
        try:
            pnl = float(pnl_pct)
        except (TypeError, ValueError):
            return "INVALID"

        if exit_reason == "TIMEOUT":
            # Economic sub-classification for timeouts
            if pnl > cls._WIN_THRESH:
                return "TIMEOUT"   # technically open but trending well
            return "TIMEOUT"

        if pnl > cls._WIN_THRESH:
            return "WIN"
        if pnl < cls._LOSS_THRESH:
            return "LOSS"
        return "BREAK_EVEN"


# ── 3. LossAnalyzer ──────────────────────────────────────────────────────────

_LOSS_CAUSES = [
    "STOP_TOO_TIGHT", "WRONG_DIRECTION", "REGIME_MISMATCH",
    "VWAP_CONFLICT", "TIMING_LATE", "TIMEOUT_DRIFT", "VOLATILITY_SPIKE",
]


class LossAnalyzer:
    """
    Identifies root causes of losses and tracks per-algo EWMA pattern rates.
    Persisted to data/loss_patterns.json.
    """

    _PATH = _DATA_DIR / "loss_patterns.json"

    def __init__(self):
        self._lock = threading.Lock()
        # {algo_name: {cause: ewma_rate}}
        self._patterns: dict[str, dict[str, float]] = {}

    def load(self) -> None:
        try:
            if self._PATH.exists():
                raw = json.loads(self._PATH.read_text())
                with self._lock:
                    self._patterns = raw
        except Exception as exc:
            logger.warning(f"[LossAnalyzer] load error: {exc}")

    def save(self) -> None:
        try:
            with self._lock:
                data = dict(self._patterns)
            self._PATH.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning(f"[LossAnalyzer] save error: {exc}")

    def analyze(self, signal_row: dict, price_path_rows: list[dict]) -> dict:
        """
        Determine the dominant root cause for a loss.
        signal_row keys: direction, regime, vwap_event, entry_type, bars_tracked,
                         exit_reason, max_favorable_r
        price_path_rows: [{bar, price, r_val}, ...]
        Returns: {root_cause, confidence, details}
        """
        try:
            direction      = signal_row.get("direction", "")
            regime         = signal_row.get("regime", "")
            vwap_event     = signal_row.get("vwap_event", "")
            entry_type     = signal_row.get("entry_type", "IMMEDIATE")
            bars_tracked   = int(signal_row.get("bars_tracked", 0) or 0)
            exit_reason    = signal_row.get("exit_reason", "")
            max_fav_r      = float(signal_row.get("max_favorable_r", 0) or 0)

            path = sorted(price_path_rows, key=lambda x: x.get("bar", 0))

            # VOLATILITY_SPIKE: r_val swing > 2.0 in single bar
            for i in range(1, len(path)):
                swing = abs(path[i].get("r_val", 0) - path[i - 1].get("r_val", 0))
                if swing > 2.0:
                    return {"root_cause": "VOLATILITY_SPIKE", "confidence": 0.85,
                            "details": f"R swing {swing:.2f} at bar {path[i].get('bar')}"}

            # TIMEOUT_DRIFT: timeout + never moved favorably
            if exit_reason == "TIMEOUT" and max_fav_r < 0.3:
                return {"root_cause": "TIMEOUT_DRIFT", "confidence": 0.80,
                        "details": f"max_favorable_r={max_fav_r:.2f}"}

            # WRONG_DIRECTION: r_val at bar 2 already <= -0.5
            bar2 = next((p for p in path if p.get("bar") == 2), None)
            if bar2 and bar2.get("r_val", 0) <= -0.5:
                return {"root_cause": "WRONG_DIRECTION", "confidence": 0.78,
                        "details": f"r_val at bar2={bar2['r_val']:.2f}"}

            # VWAP_CONFLICT: vwap event conflicts with direction
            if (vwap_event == "BELOW" and direction == "BUY") or \
               (vwap_event == "ABOVE" and direction == "SELL"):
                return {"root_cause": "VWAP_CONFLICT", "confidence": 0.72,
                        "details": f"vwap_event={vwap_event}, direction={direction}"}

            # REGIME_MISMATCH: neutral regime, algo works best in BULL/BEAR
            if regime == "NEUTRAL":
                return {"root_cause": "REGIME_MISMATCH", "confidence": 0.65,
                        "details": f"regime={regime}"}

            # TIMING_LATE: IMMEDIATE entry but >5 bars before first positive r_val
            if entry_type == "IMMEDIATE":
                first_pos = next((p for p in path if p.get("r_val", 0) > 0), None)
                if first_pos is None or first_pos.get("bar", 0) > 5:
                    return {"root_cause": "TIMING_LATE", "confidence": 0.60,
                            "details": "no positive r_val in first 5 bars"}

            # STOP_TOO_TIGHT: after stop hit, price eventually reached target direction
            stop_bar = next((p for p in path if p.get("bar") == bars_tracked), None)
            if stop_bar:
                post_stop_r = [p.get("r_val", 0) for p in path
                               if p.get("bar", 0) > stop_bar.get("bar", 0)]
                if any(r > 0.3 for r in post_stop_r):
                    return {"root_cause": "STOP_TOO_TIGHT", "confidence": 0.70,
                            "details": "price recovered after stop hit"}

            return {"root_cause": "WRONG_DIRECTION", "confidence": 0.40,
                    "details": "no specific pattern matched"}
        except Exception as exc:
            logger.debug(f"[LossAnalyzer] analyze error: {exc}")
            return {"root_cause": "WRONG_DIRECTION", "confidence": 0.30, "details": str(exc)}

    def record_pattern(self, algo_name: str, root_cause: str, ewma_alpha: float = 0.25) -> None:
        """EWMA update of pattern rates per algo."""
        with self._lock:
            if algo_name not in self._patterns:
                self._patterns[algo_name] = {c: 0.0 for c in _LOSS_CAUSES}
            rates = self._patterns[algo_name]
            for cause in _LOSS_CAUSES:
                hit = 1.0 if cause == root_cause else 0.0
                rates[cause] = ewma_alpha * hit + (1 - ewma_alpha) * rates.get(cause, 0.0)

    def get_dominant_cause(self, algo_name: str) -> str:
        """Return the most frequent (highest EWMA rate) root cause for algo."""
        with self._lock:
            rates = self._patterns.get(algo_name, {})
        if not rates:
            return "WRONG_DIRECTION"
        return max(rates, key=lambda c: rates.get(c, 0.0))


# ── 4. WinReinforcer ─────────────────────────────────────────────────────────

class WinReinforcer:
    """
    Tracks wins per algo+context and computes weight multipliers.
    Persisted to data/win_patterns.json.
    """

    _PATH   = _DATA_DIR / "win_patterns.json"
    _MIN_WINS = 15

    def __init__(self):
        self._lock = threading.Lock()
        # {algo_name: {context_key: {ewma, count}}}
        self._wins: dict[str, dict[str, dict]] = {}

    def load(self) -> None:
        try:
            if self._PATH.exists():
                with self._lock:
                    self._wins = json.loads(self._PATH.read_text())
        except Exception as exc:
            logger.warning(f"[WinReinforcer] load error: {exc}")

    def save(self) -> None:
        try:
            with self._lock:
                data = {a: {k: dict(v) for k, v in ctx.items()}
                        for a, ctx in self._wins.items()}
            self._PATH.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning(f"[WinReinforcer] save error: {exc}")

    def record_win(self, algo_name: str, context_key: str, ewma_alpha: float = 0.15) -> None:
        with self._lock:
            if algo_name not in self._wins:
                self._wins[algo_name] = {}
            ctx = self._wins[algo_name]
            if context_key not in ctx:
                ctx[context_key] = {"ewma": 0.0, "count": 0}
            entry = ctx[context_key]
            entry["ewma"]  = ewma_alpha * 1.0 + (1 - ewma_alpha) * entry["ewma"]
            entry["count"] = entry.get("count", 0) + 1

    def get_weight(self, algo_name: str, context_key: str) -> float:
        """Returns multiplier: 0.7–1.3. Only applies when >= 15 recorded wins."""
        with self._lock:
            ctx = self._wins.get(algo_name, {}).get(context_key, {})
        if ctx.get("count", 0) < self._MIN_WINS:
            return 1.0
        ewma = float(ctx.get("ewma", 0.5))
        # Scale: ewma 0.8+ → 1.3, ewma 0.2- → 0.7, 0.5 → 1.0
        weight = 0.7 + ewma * 0.6
        return round(max(0.7, min(1.3, weight)), 4)


# ── 5. AlgoSelector ──────────────────────────────────────────────────────────

class AlgoSelector:
    """
    UCB-based algo weighting. Persisted to data/algo_selector.json.
    """

    _PATH = _DATA_DIR / "algo_selector.json"

    def __init__(self):
        self._lock = threading.Lock()
        # {context_key: {algo_name: {ewma_wr, n_trials, total_trials}}}
        self._state: dict[str, dict[str, dict]] = {}

    def load(self) -> None:
        try:
            if self._PATH.exists():
                with self._lock:
                    self._state = json.loads(self._PATH.read_text())
        except Exception as exc:
            logger.warning(f"[AlgoSelector] load error: {exc}")

    def save(self) -> None:
        try:
            with self._lock:
                data = {k: {a: dict(v) for a, v in algos.items()}
                        for k, algos in self._state.items()}
            self._PATH.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning(f"[AlgoSelector] save error: {exc}")

    def record_outcome(self, algo_name: str, context_key: str, won: bool,
                       ewma_alpha: float = 0.20) -> None:
        with self._lock:
            if context_key not in self._state:
                self._state[context_key] = {}
            ctx = self._state[context_key]
            if algo_name not in ctx:
                ctx[algo_name] = {"ewma_wr": 0.5, "n_trials": 0, "total_trials": 0}
            entry = ctx[algo_name]
            entry["ewma_wr"]    = ewma_alpha * (1.0 if won else 0.0) + (1 - ewma_alpha) * entry["ewma_wr"]
            entry["n_trials"]   = entry.get("n_trials", 0) + 1
            entry["total_trials"] = entry.get("total_trials", 0) + 1

    def _ucb(self, ewma_wr: float, total_trials: int, n_trials: int) -> float:
        explore = math.sqrt(2 * math.log(max(total_trials, 1)) / max(n_trials, 1))
        return ewma_wr + explore

    def get_weights(self, algo_names: list[str], context_key: str) -> dict[str, float]:
        """
        Returns a weight multiplier per algo (default 1.0 for unknown algos).
        Uses UCB to balance exploitation (high ewma_wr) + exploration.
        """
        with self._lock:
            ctx = self._state.get(context_key, {})

        if not ctx:
            return {a: 1.0 for a in algo_names}

        # Compute UCB scores
        scores: dict[str, float] = {}
        for algo in algo_names:
            if algo in ctx:
                e = ctx[algo]
                scores[algo] = self._ucb(
                    float(e.get("ewma_wr", 0.5)),
                    int(e.get("total_trials", 1)),
                    int(e.get("n_trials", 1)),
                )
            else:
                scores[algo] = 1.0  # unexplored gets neutral weight

        # Normalize so mean = 1.0
        vals = list(scores.values())
        mean = sum(vals) / len(vals) if vals else 1.0
        if mean <= 0:
            return {a: 1.0 for a in algo_names}
        return {a: round(v / mean, 4) for a, v in scores.items()}


# ── 6. ParameterAdapter ──────────────────────────────────────────────────────

class ParameterAdapter:
    """
    Wires LossAnalyzer → ParameterControlRegistry to suggest param adjustments.
    Uses a very slow EWMA (alpha=0.10) to avoid over-fitting to recent trades.
    """

    _EWMA_ALPHA = 0.10

    def __init__(self, registry: ParameterControlRegistry, loss_analyzer: LossAnalyzer):
        self._reg  = registry
        self._loss = loss_analyzer

    def adapt(self, algo_family: str, algo_name: str, cycle_num: int) -> dict:
        """
        Analyse dominant loss cause for this algo and nudge params accordingly.
        Returns dict of {param: (old_val, new_val, reason)} for changes applied.
        """
        changes: dict[str, tuple] = {}
        try:
            cause = self._loss.get_dominant_cause(algo_name)
            if not cause:
                return changes

            adjustments: list[tuple[str, float, str]] = []  # (param, delta, reason)

            if cause == "STOP_TOO_TIGHT":
                # Increase stop_mult to give trades more room
                old = self._reg.get(algo_family, "stop_mult")
                new = old * (1 + self._EWMA_ALPHA)
                adjustments.append(("stop_mult", new, f"STOP_TOO_TIGHT: widen stop"))

            elif cause == "WRONG_DIRECTION":
                # Raise conf_gate to filter lower-conviction entries
                old = self._reg.get(algo_family, "conf_gate")
                new = old + 1.0
                adjustments.append(("conf_gate", new, f"WRONG_DIRECTION: raise conf gate"))

            elif cause == "REGIME_MISMATCH":
                old = self._reg.get(algo_family, "conf_gate")
                new = old + 1.0
                adjustments.append(("conf_gate", new, f"REGIME_MISMATCH: raise conf gate"))

            elif cause == "TIMEOUT_DRIFT":
                # Target is too far — pull it in
                old = self._reg.get(algo_family, "target_mult")
                new = old * (1 - self._EWMA_ALPHA)
                adjustments.append(("target_mult", new, f"TIMEOUT_DRIFT: reduce target_mult"))

            elif cause == "VWAP_CONFLICT":
                old = self._reg.get(algo_family, "rvol_gate")
                new = old + 0.05
                adjustments.append(("rvol_gate", new, f"VWAP_CONFLICT: raise rvol gate"))

            elif cause == "TIMING_LATE":
                old = self._reg.get(algo_family, "entry_window_bars")
                new = old - 1
                adjustments.append(("entry_window_bars", new, f"TIMING_LATE: tighten entry window"))

            elif cause == "VOLATILITY_SPIKE":
                old = self._reg.get(algo_family, "stop_mult")
                new = old * (1 + self._EWMA_ALPHA * 2)
                adjustments.append(("stop_mult", new, f"VOLATILITY_SPIKE: widen stop for volatility"))

            for param, new_val, reason in adjustments:
                old_val = self._reg.get(algo_family, param)
                applied = self._reg.update(algo_family, param, new_val, reason, cycle_num)
                if applied:
                    actual_new = self._reg.get(algo_family, param)
                    changes[param] = (old_val, actual_new, reason)

        except Exception as exc:
            logger.debug(f"[ParameterAdapter] adapt error: {exc}")

        return changes


# ── 7. CounterfactualSimulator ────────────────────────────────────────────────

class CounterfactualSimulator:
    """
    Records suppressed signals as shadow trades in bt_signals (is_counterfactual=1).
    Shadow trades are resolved by the existing update_tracking() mechanism.
    """

    def record_suppressed(
        self,
        ticker:             str,
        direction:          str,
        entry_price:        float,
        target:             float,
        stop:               float,
        rr_ratio:           float,
        confidence:         float,
        session:            str,
        regime:             str,
        vwap_event:         str,
        rsi_zone:           str,
        entry_type:         str,
        algo_name:          str,
        suppression_reason: str,
    ) -> str:
        """
        Creates a shadow record in bt_signals with is_counterfactual=1.
        Returns the shadow signal_id, or "" on failure.
        """
        try:
            from agent.live_backtest import record_signal as bt_record_real
            # bt_record_real will do dedup — shadow trades deduplicate the same way
            sid = bt_record_real(
                ticker=ticker, direction=direction,
                entry_price=entry_price, target=target, stop=stop,
                rr_ratio=rr_ratio, confidence=confidence,
                session=session, regime=regime, vwap_event=vwap_event,
                rsi_zone=rsi_zone, entry_type=entry_type,
                algo_name=algo_name,
                is_counterfactual=1,
                suppression_reason=suppression_reason,
            )
            return sid
        except Exception as exc:
            logger.debug(f"[Counterfactual] record_suppressed error: {exc}")
            return ""

    def get_counterfactual_stats(self, lookback_days: int = 30) -> dict:
        """
        Returns {total, wins, losses, missed_win_rate} for shadow trades.
        Counterfactuals weighted at 0.25 vs real trades (weight=1.0).
        """
        try:
            from agent.db import get_conn
            from pathlib import Path
            _db = Path(__file__).parent.parent / "data" / "live_backtest.db"
            with get_conn(_db) as c:
                rows = c.execute("""
                    SELECT status FROM bt_signals
                    WHERE is_counterfactual=1
                      AND status IN ('WIN','LOSS','TIMEOUT')
                      AND fired_at >= datetime('now', ? || ' days')
                """, (f"-{lookback_days}",)).fetchall()
            total  = len(rows)
            wins   = sum(1 for r in rows if r["status"] == "WIN")
            losses = total - wins
            missed_win_rate = round(wins / total, 4) if total > 0 else 0.0
            return {"total": total, "wins": wins, "losses": losses,
                    "missed_win_rate": missed_win_rate}
        except Exception as exc:
            logger.debug(f"[Counterfactual] get_stats error: {exc}")
            return {"total": 0, "wins": 0, "losses": 0, "missed_win_rate": 0.0}

    def should_relax_filter(self, context_key: str, min_cf_trades: int = 10) -> tuple[bool, str]:
        """
        Returns (True, reason) if missed_win_rate > 0.40 with enough shadow trades,
        suggesting the filter is too aggressive for this context.
        """
        try:
            stats = self.get_counterfactual_stats()
            total = stats.get("total", 0)
            if total < min_cf_trades:
                return False, f"only {total} counterfactual trades (need {min_cf_trades})"
            mwr = stats.get("missed_win_rate", 0.0)
            if mwr > 0.40:
                return True, f"missed_win_rate={mwr:.2%} suggests filter too aggressive"
            return False, f"missed_win_rate={mwr:.2%} (threshold 40%)"
        except Exception as exc:
            return False, str(exc)


# ── 8. ModelVersionRegistry ──────────────────────────────────────────────────

class ModelVersionRegistry:
    """
    Champion/challenger pattern for model versioning.
    Persisted to data/model_registry.json.
    """

    _PATH = _DATA_DIR / "model_registry.json"

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict = {
            "champion": None,
            "challengers": {},
            "history": [],
        }

    def load(self) -> None:
        try:
            if self._PATH.exists():
                with self._lock:
                    self._data = json.loads(self._PATH.read_text())
        except Exception as exc:
            logger.warning(f"[ModelRegistry] load error: {exc}")

    def save(self) -> None:
        try:
            with self._lock:
                data = dict(self._data)
            self._PATH.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning(f"[ModelRegistry] save error: {exc}")

    def register_version(self, version_id: str, metrics_dict: dict) -> None:
        """Create a new challenger entry."""
        with self._lock:
            self._data["challengers"][version_id] = {
                "version_id": version_id,
                "metrics": metrics_dict,
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "status": "challenger",
            }
        logger.info(f"[ModelRegistry] Registered challenger {version_id}")

    def evaluate_promotion(self, version_id: str) -> tuple[bool, list[str]]:
        """
        Check all promotion gates for a challenger. Returns (can_promote, reasons).
        Gates:
          - n_trades >= 30
          - profit_factor >= 1.10
          - positive net expectancy
          - sharpe improvement >= 5% vs champion
          - max_drawdown <= 2x champion's
        """
        with self._lock:
            challenger = self._data["challengers"].get(version_id)
            champion   = self._data.get("champion")

        if not challenger:
            return False, [f"version {version_id} not found"]

        m = challenger.get("metrics", {})
        reasons: list[str] = []
        gates_pass = True

        if m.get("n_trades", 0) < 30:
            reasons.append(f"n_trades {m.get('n_trades')} < 30")
            gates_pass = False

        if m.get("profit_factor", 0) < 1.10:
            reasons.append(f"profit_factor {m.get('profit_factor'):.2f} < 1.10")
            gates_pass = False

        if m.get("expectancy", 0) <= 0:
            reasons.append(f"expectancy {m.get('expectancy'):.4f} <= 0")
            gates_pass = False

        if champion:
            cm = champion.get("metrics", {})
            champ_sharpe = float(cm.get("sharpe", 0) or 0)
            chal_sharpe  = float(m.get("sharpe", 0) or 0)
            if champ_sharpe > 0 and chal_sharpe < champ_sharpe * 1.05:
                reasons.append(f"sharpe {chal_sharpe:.2f} not 5%+ vs champion {champ_sharpe:.2f}")
                gates_pass = False

            champ_dd = float(cm.get("max_drawdown", 0) or 0)
            chal_dd  = float(m.get("max_drawdown", 0) or 0)
            if champ_dd > 0 and chal_dd > champ_dd * 2:
                reasons.append(f"max_drawdown {chal_dd:.4f} > 2x champion {champ_dd:.4f}")
                gates_pass = False

        if gates_pass:
            reasons.append("all promotion gates passed")
        return gates_pass, reasons

    def promote(self, version_id: str) -> None:
        """Promote challenger to champion; save old champion to history."""
        with self._lock:
            challenger = self._data["challengers"].get(version_id)
            if not challenger:
                return
            old_champion = self._data.get("champion")
            if old_champion:
                self._data["history"].append(old_champion)
                # Keep only last 10 in history
                self._data["history"] = self._data["history"][-10:]
            challenger["status"] = "champion"
            challenger["promoted_at"] = datetime.now(timezone.utc).isoformat()
            self._data["champion"] = challenger
            del self._data["challengers"][version_id]
        logger.info(f"[ModelRegistry] Promoted {version_id} to champion")

    def rollback(self) -> None:
        """Restore previous champion from history."""
        with self._lock:
            if not self._data["history"]:
                logger.warning("[ModelRegistry] No history to rollback to")
                return
            prev = self._data["history"].pop()
            current = self._data.get("champion")
            if current:
                self._data["challengers"][current["version_id"]] = current
            self._data["champion"] = prev
        logger.info(f"[ModelRegistry] Rolled back to {prev.get('version_id')}")

    def get_champion(self) -> dict:
        with self._lock:
            return dict(self._data.get("champion") or {})


# ── 9. RegimeTransitionHandler ───────────────────────────────────────────────

class RegimeTransitionHandler:
    """
    Monitors open trades for regime changes and produces action recommendations.
    Actions are logged via AuditLogger; no separate persistence.
    """

    def __init__(self, audit_logger: "AuditLogger"):
        self._audit = audit_logger

    def check_open_trades(self, open_trades: list[dict],
                          current_regime_by_ticker: dict[str, str]) -> list[dict]:
        """
        Compare each open trade's stored regime to the current regime.
        Returns list of action dicts.
        Actions: TIGHTEN_STOP | MOVE_TO_BREAKEVEN | EXIT_SIGNAL | SUSPEND
        """
        actions: list[dict] = []
        try:
            for trade in open_trades:
                ticker      = trade.get("ticker", "")
                trade_id    = trade.get("id", 0)
                old_regime  = trade.get("regime", "NEUTRAL")
                new_regime  = current_regime_by_ticker.get(ticker, old_regime)

                if old_regime == new_regime:
                    continue

                action: Optional[str] = None
                stop_adjustment       = 0.0
                entry_price           = float(trade.get("entry_price", 0) or 0)
                current_stop          = float(trade.get("stop", 0) or 0)

                # BULL→NEUTRAL: tighten stop by 20%
                if old_regime in ("BULL", "BULL_TREND") and new_regime == "NEUTRAL":
                    action = "TIGHTEN_STOP"
                    risk   = abs(entry_price - current_stop)
                    stop_adjustment = risk * 0.20   # reduce stop distance by 20%

                # Any→HIGH_VOL
                elif new_regime in ("HIGH_VOL",):
                    direction = trade.get("direction", "BUY")
                    current_price = float(trade.get("exit_price", entry_price) or entry_price)
                    in_profit = (direction == "BUY"  and current_price > entry_price) or \
                                (direction == "SELL" and current_price < entry_price)
                    action = "MOVE_TO_BREAKEVEN" if in_profit else "EXIT_SIGNAL"

                # Any→ABNORMAL or HALT
                elif new_regime in ("ABNORMAL", "HALT", "CIRCUIT_BREAK"):
                    action = "EXIT_SIGNAL"

                if action:
                    action_dict = {
                        "trade_id":       trade_id,
                        "ticker":         ticker,
                        "old_regime":     old_regime,
                        "new_regime":     new_regime,
                        "action":         action,
                        "stop_adjustment": round(stop_adjustment, 4),
                    }
                    actions.append(action_dict)
                    self._audit.log("REGIME_TRANSITION_ACTION", action_dict)

        except Exception as exc:
            logger.warning(f"[RegimeHandler] check_open_trades error: {exc}")

        return actions

    def apply_actions(self, actions: list[dict]) -> int:
        """
        Apply regime-transition actions. Calls paper_trading.update_trade_stop()
        for stop adjustments. Returns count of actions applied.
        """
        applied = 0
        try:
            from agent.paper_trading import update_trade_stop
            for act in actions:
                try:
                    if act["action"] in ("TIGHTEN_STOP", "MOVE_TO_BREAKEVEN"):
                        success = update_trade_stop(
                            trade_id=act["trade_id"],
                            new_stop=act.get("stop_adjustment", 0.0),
                            reason=act["action"],
                        )
                        if success:
                            applied += 1
                    elif act["action"] == "EXIT_SIGNAL":
                        # EXIT_SIGNAL — just log; actual close handled by paper_trading loop
                        self._audit.log("EXIT_SIGNAL_ISSUED", act)
                        applied += 1
                except Exception as exc:
                    logger.debug(f"[RegimeHandler] apply action error: {exc}")
        except ImportError:
            pass
        return applied


# ── 10. AuditLogger ──────────────────────────────────────────────────────────

class AuditLogger:
    """
    Append-only JSONL event log. Max 50MB (oldest trimmed).
    Persisted to data/algo_audit_log.jsonl.
    """

    _PATH        = _DATA_DIR / "algo_audit_log.jsonl"
    _MAX_BYTES   = 50 * 1024 * 1024   # 50 MB
    _lock        = threading.Lock()

    def log(self, event_type: str, data_dict: dict) -> None:
        entry = {
            "ts":         datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data":       data_dict,
        }
        try:
            with self._lock:
                self._PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(self._PATH, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                self._maybe_trim()
        except Exception as exc:
            logger.debug(f"[AuditLogger] log error: {exc}")

    def notify_operator(self, event_type: str, message: str, data: dict) -> None:
        """Log + optionally POST to ALERT_WEBHOOK_URL env var."""
        payload = {"message": message, **data}
        self.log(event_type, payload)
        webhook = os.environ.get("ALERT_WEBHOOK_URL", "")
        if webhook:
            try:
                import urllib.request
                req_data = json.dumps(payload).encode()
                req = urllib.request.Request(
                    webhook, data=req_data,
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=5)
            except Exception as exc:
                logger.debug(f"[AuditLogger] webhook post failed: {exc}")

    def _maybe_trim(self) -> None:
        """Trim oldest lines if file exceeds 50MB."""
        try:
            if self._PATH.stat().st_size > self._MAX_BYTES:
                lines = self._PATH.read_text().splitlines()
                # Keep last 80% of lines
                keep = lines[len(lines) // 5:]
                self._PATH.write_text("\n".join(keep) + "\n")
        except Exception:
            pass


# ── AlgoLearningEngine — top-level coordinator ────────────────────────────────

class AlgoLearningEngine:
    """
    Orchestrates all 10 learning components.
    Called by LearningEngine._run_cycle() via run_cycle().
    """

    def __init__(self):
        self._audit         = AuditLogger()
        self._registry      = ParameterControlRegistry()
        self._classifier    = OutcomeClassifier()
        self._loss_analyzer = LossAnalyzer()
        self._win_reinforce = WinReinforcer()
        self._selector      = AlgoSelector()
        self._adapter       = ParameterAdapter(self._registry, self._loss_analyzer)
        self._counterfact   = CounterfactualSimulator()
        self._model_reg     = ModelVersionRegistry()
        self._regime_handler = RegimeTransitionHandler(self._audit)

    def load(self) -> None:
        """Load all persistent components."""
        for comp in (self._registry, self._loss_analyzer, self._win_reinforce,
                     self._selector, self._model_reg):
            try:
                comp.load()
            except Exception as exc:
                logger.warning(f"[AlgoLearningEngine] load error ({type(comp).__name__}): {exc}")

    def save(self) -> None:
        """Save all persistent components."""
        for comp in (self._registry, self._loss_analyzer, self._win_reinforce,
                     self._selector, self._model_reg):
            try:
                comp.save()
            except Exception as exc:
                logger.warning(f"[AlgoLearningEngine] save error ({type(comp).__name__}): {exc}")

    def run_cycle(self, new_outcomes_df, cycle_num: int) -> None:
        """
        Main learning loop called by LearningEngine every cycle.

        Steps:
          1. Classify each new outcome
          2. LOSS: analyze root cause, record pattern
          3. WIN: record in WinReinforcer
          4. For each algo: run ParameterAdapter.adapt()
          5. Update AlgoSelector weights
          6. Check counterfactual stats
          7. Audit log cycle summary
        """
        import pandas as pd
        try:
            if new_outcomes_df is None or (hasattr(new_outcomes_df, 'empty') and new_outcomes_df.empty):
                return

            wins_count   = 0
            losses_count = 0
            algos_seen   = set()

            for _, row in new_outcomes_df.iterrows():
                try:
                    pnl_pct     = float(row.get("pnl_pct", 0) or 0)
                    exit_reason = str(row.get("exit_reason", "") or "")
                    status      = str(row.get("status", "") or "")
                    algo_name   = str(row.get("algo_name", "") or "")
                    regime      = str(row.get("regime", "") or "")
                    session     = str(row.get("session", "") or "")
                    vwap_event  = str(row.get("vwap_event", "") or "")
                    context_key = f"{regime}:{session}:{vwap_event}"

                    outcome = self._classifier.classify(pnl_pct / 100.0
                                                        if abs(pnl_pct) > 0.1
                                                        else pnl_pct, exit_reason)

                    # ── 2. LOSS analysis ─────────────────────────────────────
                    if outcome == "LOSS" and algo_name:
                        signal_row = dict(row)
                        # price path not available in bulk df — use empty
                        root_cause_dict = self._loss_analyzer.analyze(signal_row, [])
                        self._loss_analyzer.record_pattern(
                            algo_name, root_cause_dict["root_cause"]
                        )
                        losses_count += 1

                    # ── 3. WIN reinforcement ─────────────────────────────────
                    elif outcome == "WIN" and algo_name:
                        self._win_reinforce.record_win(algo_name, context_key)
                        wins_count += 1

                    # ── 5. AlgoSelector update ───────────────────────────────
                    if algo_name and outcome in ("WIN", "LOSS", "BREAK_EVEN"):
                        self._selector.record_outcome(
                            algo_name, context_key, won=(outcome == "WIN")
                        )

                    if algo_name:
                        algos_seen.add(algo_name)

                except Exception as row_exc:
                    logger.debug(f"[ALE] row processing error: {row_exc}")

            # ── 4. Parameter adaptation ──────────────────────────────────────
            param_changes: dict[str, dict] = {}
            for algo_name in algos_seen:
                family = _ALGO_FAMILY_MAP.get(algo_name, "")
                if family:
                    changes = self._adapter.adapt(family, algo_name, cycle_num)
                    if changes:
                        param_changes[algo_name] = changes

            # ── 6. Counterfactual check ──────────────────────────────────────
            should_relax, cf_reason = self._counterfact.should_relax_filter("", min_cf_trades=10)
            if should_relax:
                self._audit.notify_operator(
                    "COUNTERFACTUAL_FILTER_RELAX",
                    f"Counterfactual analysis suggests filter relaxation: {cf_reason}",
                    {"reason": cf_reason},
                )

            # ── 7. Audit log ──────────────────────────────────────────────────
            self._audit.log("ALGO_LEARNING_CYCLE", {
                "cycle_num":    cycle_num,
                "wins":         wins_count,
                "losses":       losses_count,
                "algos_seen":   list(algos_seen),
                "param_changes": {k: {p: list(v) for p, v in ch.items()}
                                  for k, ch in param_changes.items()},
                "cf_relax":     should_relax,
            })

            # Persist after each cycle
            self.save()

        except Exception as exc:
            logger.warning(f"[AlgoLearningEngine] run_cycle error: {exc}")

    def on_algo_signal(self, algo_name: str, context_key: str, won: bool) -> None:
        """Called when any algo trade resolves."""
        try:
            self._selector.record_outcome(algo_name, context_key, won)
            if won:
                self._win_reinforce.record_win(algo_name, context_key)
        except Exception as exc:
            logger.debug(f"[ALE] on_algo_signal error: {exc}")

    def get_algo_params(self, algo_name: str) -> dict:
        """Public API for trading_algos.py. Returns all current params for algo's family."""
        try:
            return self._registry.get_all_params(algo_name)
        except Exception:
            return {p: s["default"] for p, s in _PARAM_SPEC.items()}

    def get_algo_selector_weights(self, algo_names: list[str], context_key: str) -> dict:
        """Public API for scanner.py. Returns UCB-weighted multipliers per algo."""
        try:
            return self._selector.get_weights(algo_names, context_key)
        except Exception:
            return {a: 1.0 for a in algo_names}

    def record_suppressed(self, **kwargs) -> str:
        """Public API for scanner.py — proxy to CounterfactualSimulator."""
        try:
            return self._counterfact.record_suppressed(**kwargs)
        except Exception:
            return ""


# ── Module-level singleton ────────────────────────────────────────────────────

_engine: Optional[AlgoLearningEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> AlgoLearningEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = AlgoLearningEngine()
                _engine.load()
    return _engine


def get_algo_params(algo_name: str) -> dict:
    """Safe public accessor. Returns empty dict on any error."""
    try:
        return get_engine().get_algo_params(algo_name)
    except Exception:
        return {}


def get_selector_weights(algo_names: list, context_key: str) -> dict:
    """Safe public accessor. Returns neutral weights on any error."""
    try:
        return get_engine().get_algo_selector_weights(algo_names, context_key)
    except Exception:
        return {a: 1.0 for a in algo_names}
