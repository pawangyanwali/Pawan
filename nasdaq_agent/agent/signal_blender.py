"""
Dynamic signal blender — weights each model by its rolling directional accuracy
so that the best-performing models carry more influence in the final blended
probability, rather than using hardcoded proportions.

Architecture
------------
- DynamicBlender maintains a deque of the last WINDOW outcomes per model.
- Weights are computed as a temperature-scaled softmax over recent accuracy.
- Models with < MIN_SAMPLES outcomes receive DEFAULT_WEIGHT (equal share).
- Untrained optional models (swing, deep) are excluded from the blend and their
  weight redistributed proportionally to the active models.
- History is persisted to JSON after every SAVE_EVERY new outcomes and loaded
  on startup so accuracy learning survives process restarts.
- All state mutations are protected by a reentrant lock.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODELS: list[str] = ["scalp", "ensemble", "reversal", "swing", "deep"]
WINDOW: int = 20        # rolling window of outcomes kept per model
MIN_SAMPLES: int = 5    # minimum outcomes before a model's accuracy is trusted
DEFAULT_WEIGHT: float = 1.0 / len(MODELS)  # equal weight when no history
SOFTMAX_TEMP: float = 4.0   # temperature for softmax; >1 rewards better models
SAVE_EVERY: int = 5         # persist after this many new outcomes (avoid thrash)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ModelOutcome:
    model: str
    ticker: str
    prob: float     # predicted probability in [0, 1]
    correct: bool   # True if the model's directional call was correct
    ts: float       # unix timestamp of resolution


# ---------------------------------------------------------------------------
# Core blender
# ---------------------------------------------------------------------------

class DynamicBlender:
    """
    Tracks rolling WINDOW-outcome accuracy per model (globally, not per-ticker;
    per-ticker would require hundreds of trades per symbol to be meaningful).

    Weights are computed as a temperature-scaled softmax over recent accuracy.
    When a model has fewer than MIN_SAMPLES outcomes it receives DEFAULT_WEIGHT
    and participates in the normalization step so totals always sum to 1.0.
    """

    def __init__(self, persist_path: Optional[Path] = None) -> None:
        self._history: dict[str, deque[ModelOutcome]] = {
            m: deque(maxlen=WINDOW) for m in MODELS
        }
        self._lock = threading.RLock()
        self._persist_path: Path = (
            persist_path
            or Path(__file__).parent.parent / "data" / "blend_weights.json"
        )
        self._unsaved_count: int = 0
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        model: str,
        ticker: str,
        prob: float,
        correct: bool,
    ) -> None:
        """
        Record the resolved outcome for one model signal.  Call this after the
        lookahead period has elapsed and the direction is known.

        Parameters
        ----------
        model:   one of MODELS
        ticker:  e.g. "AAPL"
        prob:    the probability the model emitted at signal time
        correct: True if the model's directional prediction was right
        """
        if model not in MODELS:
            logger.warning("record_outcome: unknown model %r — skipping", model)
            return

        outcome = ModelOutcome(
            model=model,
            ticker=ticker,
            prob=float(prob),
            correct=bool(correct),
            ts=time.time(),
        )

        do_save = False
        with self._lock:
            self._history[model].append(outcome)
            self._unsaved_count += 1
            if self._unsaved_count >= SAVE_EVERY:
                do_save = True
                self._unsaved_count = 0
        if do_save:
            self._save()

        logger.debug(
            "Recorded outcome for %s/%s: correct=%s (queue len=%d)",
            model,
            ticker,
            correct,
            len(self._history[model]),
        )

    def get_weights(self) -> dict[str, float]:
        """
        Compute blend weights from rolling accuracy.

        Models with >= MIN_SAMPLES outcomes contribute their softmax-transformed
        accuracy.  Models with insufficient history receive DEFAULT_WEIGHT.
        All weights are normalised to sum to 1.0.

        Returns
        -------
        dict mapping each model name to its weight, e.g.
        {"scalp": 0.25, "ensemble": 0.30, "reversal": 0.15, "swing": 0.20, "deep": 0.10}
        """
        with self._lock:
            # Step 1: compute per-model accuracy or mark as unknown
            accs: dict[str, Optional[float]] = {}
            for model, hist in self._history.items():
                if len(hist) >= MIN_SAMPLES:
                    accs[model] = sum(o.correct for o in hist) / len(hist)
                else:
                    accs[model] = None

            known = {m: a for m, a in accs.items() if a is not None}
            unknown_models = [m for m, a in accs.items() if a is None]

            # Step 2: if no model has enough history yet, return equal weights
            if not known:
                return {m: DEFAULT_WEIGHT for m in MODELS}

            # Step 3: softmax over known accuracies, scaled to their share of
            # total model slots so that unknown models retain DEFAULT_WEIGHT
            exp_accs = {m: math.exp(a * SOFTMAX_TEMP) for m, a in known.items()}
            exp_total = sum(exp_accs.values())

            # Known models share (len(known) / len(MODELS)) of total weight
            known_fraction = len(known) / len(MODELS)
            known_weights = {
                m: (v / exp_total) * known_fraction
                for m, v in exp_accs.items()
            }

            # Step 4: assemble full weights dict
            unknown_weight = DEFAULT_WEIGHT  # each unknown gets 1/N share
            weights: dict[str, float] = {}
            for m in MODELS:
                weights[m] = known_weights[m] if m in known_weights else unknown_weight

            # Step 5: normalise to sum exactly to 1.0
            total_w = sum(weights.values())
            return {m: w / total_w for m, w in weights.items()}

    def blend(
        self,
        scalp_p: float,
        ensemble_p: float,
        reversal_p: float,
        swing_p: float = 0.5,
        deep_p: float = 0.5,
        swing_trained: bool = False,
        deep_trained: bool = False,
    ) -> float:
        """
        Blend model probabilities using current dynamic weights.

        Untrained models (swing, deep) are excluded from the blend and their
        weight is redistributed proportionally among the active models.

        Parameters
        ----------
        scalp_p, ensemble_p, reversal_p : float
            Model probabilities in [0, 1]; always included.
        swing_p, deep_p : float
            Model probabilities in [0, 1]; only used when the corresponding
            *_trained flag is True.
        swing_trained, deep_trained : bool
            Whether the optional models have been trained and should contribute.

        Returns
        -------
        float
            Blended probability in [0, 1].
        """
        weights = self.get_weights()

        # Build the active signal map
        signals: dict[str, float] = {
            "scalp": float(scalp_p),
            "ensemble": float(ensemble_p),
            "reversal": float(reversal_p),
        }
        if swing_trained:
            signals["swing"] = float(swing_p)
        if deep_trained:
            signals["deep"] = float(deep_p)

        # Sum weights for active models, then normalise
        active_weight_total = sum(weights[m] for m in signals)
        if active_weight_total <= 0.0:
            # Degenerate fallback: simple mean of provided signals
            return sum(signals.values()) / len(signals)

        blended = sum(
            signals[m] * (weights[m] / active_weight_total)
            for m in signals
        )

        # Clamp to [0, 1] to handle any floating-point drift
        return max(0.0, min(1.0, blended))

    def get_stats(self) -> dict:
        """
        Return current weights and per-model accuracy for UI display.

        Returns
        -------
        dict with keys:
          "weights"  : {model: weight}
          "accuracy" : {model: float | None}   # None = not enough history
          "samples"  : {model: int}             # number of outcomes in window
        """
        with self._lock:
            weights = self.get_weights()
            accuracy: dict[str, Optional[float]] = {}
            samples: dict[str, int] = {}
            for model, hist in self._history.items():
                n = len(hist)
                samples[model] = n
                if n >= MIN_SAMPLES:
                    accuracy[model] = sum(o.correct for o in hist) / n
                else:
                    accuracy[model] = None

        return {
            "weights": weights,
            "accuracy": accuracy,
            "samples": samples,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """
        Atomically persist history to JSON.

        Writes to a sibling temp file then os.rename()s it over the target so
        that a crash mid-write never leaves a corrupt file on disk.
        """
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict = {}
        for model, hist in self._history.items():
            payload[model] = [
                {
                    "model": o.model,
                    "ticker": o.ticker,
                    "prob": o.prob,
                    "correct": o.correct,
                    "ts": o.ts,
                }
                for o in hist
            ]

        tmp_path = self._persist_path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp_path, self._persist_path)
            logger.debug("Blend weights persisted to %s", self._persist_path)
        except OSError:
            logger.exception("Failed to persist blend weights to %s", self._persist_path)
            # Clean up orphaned temp file if it exists
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _load(self) -> None:
        """
        Load history from JSON on startup to preserve learning across restarts.

        Invalid or missing files are silently ignored and fresh history is used.
        Unknown model names in the file are skipped to handle schema evolution.
        """
        if not self._persist_path.exists():
            logger.debug("No existing blend history at %s — starting fresh", self._persist_path)
            return

        try:
            with open(self._persist_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)

            loaded_count = 0
            for model, entries in payload.items():
                if model not in MODELS:
                    logger.debug("_load: skipping unknown model %r in saved file", model)
                    continue
                for entry in entries:
                    outcome = ModelOutcome(
                        model=entry["model"],
                        ticker=entry["ticker"],
                        prob=float(entry["prob"]),
                        correct=bool(entry["correct"]),
                        ts=float(entry["ts"]),
                    )
                    self._history[model].append(outcome)
                    loaded_count += 1

            logger.info(
                "Loaded %d blend history entries from %s",
                loaded_count,
                self._persist_path,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning(
                "Blend history file %s is corrupt or incompatible — starting fresh",
                self._persist_path,
                exc_info=True,
            )
            # Reset to empty so a bad file doesn't poison weights
            self._history = {m: deque(maxlen=WINDOW) for m in MODELS}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_blender: Optional[DynamicBlender] = None
_singleton_lock = threading.Lock()


def get_blender() -> DynamicBlender:
    """Return the process-wide DynamicBlender instance, creating it if needed."""
    global _blender
    if _blender is None:
        with _singleton_lock:
            if _blender is None:
                _blender = DynamicBlender()
    return _blender


def blend_signals(
    scalp_p: float,
    ensemble_p: float,
    reversal_p: float,
    swing_p: float = 0.5,
    deep_p: float = 0.5,
    swing_trained: bool = False,
    deep_trained: bool = False,
) -> float:
    """
    Module-level convenience wrapper around the singleton blender.

    Drop-in replacement for the old hardcoded expression:
        ml_combined = 0.70 * meta + 0.15 * swing + 0.15 * deep
    """
    return get_blender().blend(
        scalp_p=scalp_p,
        ensemble_p=ensemble_p,
        reversal_p=reversal_p,
        swing_p=swing_p,
        deep_p=deep_p,
        swing_trained=swing_trained,
        deep_trained=deep_trained,
    )
