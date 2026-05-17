"""
Multi-model ensemble signal aggregator.

Combines predictions from StockMLModel, EnsembleMLModel (10 XGBoost),
and a lightweight LightGBM model into a single probability estimate.

The key insight: when all models agree, confidence is high.
When models disagree, the signal is weak — reduce position size or skip.

Disagreement metric: std(probabilities) across models.
  < 0.05 = very tight agreement → high confidence
  > 0.15 = wide disagreement    → skip or reduce size
"""
from __future__ import annotations

import logging
import threading
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent.parent / "data" / "models"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Disagreement thresholds
AGREE_HIGH = 0.08   # std < this = strong consensus → full confidence
AGREE_LOW  = 0.18   # std > this = too uncertain  → suppress/reduce


def ensemble_confidence_multiplier(prob_std: float) -> float:
    """
    Convert model disagreement (std of probabilities) into a confidence multiplier.
    0.05 std → 1.0x (no penalty)
    0.15 std → 0.6x
    0.20 std → 0.4x (heavy penalty)
    """
    if prob_std <= AGREE_HIGH:
        return 1.0
    if prob_std >= AGREE_LOW:
        return 0.4
    # Linear interpolation
    t = (prob_std - AGREE_HIGH) / (AGREE_LOW - AGREE_HIGH)
    return round(1.0 - t * 0.6, 3)


class MetaEnsemble:
    """
    Lightweight meta-model that takes predictions from multiple base models
    as features and outputs a calibrated final probability.

    Inputs to meta-model:
      - scalp_prob (XGBoost scalp model)
      - ensemble_prob (10-model ensemble average)
      - ensemble_agreement (fraction of ensemble models that agree)
      - daily_prob (daily model probability)
      - reversal_prob (reversal model)
      - tech_score (from technical indicators, passed in)
      - vol_score (from volume analysis)

    These 7 features train a small XGBoost meta-learner on historical outcomes.
    """

    FEATURE_NAMES = [
        "scalp_prob", "ensemble_prob", "ensemble_agreement",
        "daily_prob", "reversal_prob", "tech_score", "vol_score",
    ]

    def __init__(self, ticker: str):
        self.ticker  = ticker
        self.model   = None
        self.scaler  = StandardScaler()
        self.trained = False
        self._load()

    def _path(self) -> Path:
        return _MODEL_DIR / f"meta_{self.ticker}.joblib"

    def _save(self) -> None:
        try:
            joblib.dump({"model": self.model, "scaler": self.scaler, "trained": self.trained}, self._path())
        except Exception as e:
            logger.debug(f"[{self.ticker}] meta save: {e}")

    def _load(self) -> None:
        try:
            p = self._path()
            if p.exists():
                d = joblib.load(p)
                self.model, self.scaler, self.trained = d["model"], d["scaler"], d.get("trained", False)
        except Exception as e:
            logger.debug(f"[{self.ticker}] meta load: {e}")

    def train_from_outcomes(self, records: list[dict]) -> bool:
        """
        Train the meta-model from historical signal outcomes.

        Each record must have:
          scalp_prob, ensemble_prob, ensemble_agreement, daily_prob,
          reversal_prob, tech_score, vol_score, won (0 or 1)
        """
        if len(records) < 30:
            return False

        try:
            from xgboost import XGBClassifier
            rows = []
            labels = []
            for r in records:
                try:
                    row = [float(r.get(f, 0.5)) for f in self.FEATURE_NAMES]
                    rows.append(row)
                    labels.append(int(r["won"]))
                except (KeyError, ValueError):
                    continue

            if len(rows) < 30:
                return False

            X = np.array(rows)
            y = np.array(labels)

            class_counts = np.bincount(y)
            if len(class_counts) < 2 or (class_counts.max() / len(y)) > 0.85:
                return False

            self.scaler.fit(X)
            Xs = self.scaler.transform(X)

            base = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                  eval_metric="logloss", verbosity=0)
            self.model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.model.fit(Xs, y)
            self.trained = True
            self._save()
            logger.info(f"[{self.ticker}] MetaEnsemble trained on {len(rows)} outcomes")
            return True
        except Exception as e:
            logger.warning(f"[{self.ticker}] MetaEnsemble train failed: {e}")
            return False

    def predict(
        self,
        scalp_prob:          float = 0.5,
        ensemble_prob:       float = 0.5,
        ensemble_agreement:  float = 0.5,
        daily_prob:          float = 0.5,
        reversal_prob:       float = 0.5,
        tech_score:          float = 0.0,
        vol_score:           float = 0.0,
    ) -> tuple[float, float]:
        """
        Returns (final_probability, confidence_multiplier).
        Falls back to weighted average if meta-model not yet trained.
        """
        probs = np.array([scalp_prob, ensemble_prob, daily_prob, reversal_prob])
        prob_std = float(np.std(probs))
        mult = ensemble_confidence_multiplier(prob_std)

        if not self.trained or self.model is None:
            # Weighted fallback: ensemble gets most weight
            fallback = (
                0.25 * scalp_prob +
                0.40 * ensemble_prob +
                0.15 * daily_prob +
                0.10 * reversal_prob +
                0.10 * (0.5 + tech_score * 0.3)   # tech score mapped to [0.2, 0.8]
            )
            return round(float(np.clip(fallback, 0.0, 1.0)), 4), mult

        try:
            row = np.array([[scalp_prob, ensemble_prob, ensemble_agreement,
                             daily_prob, reversal_prob, tech_score, vol_score]])
            row_s = self.scaler.transform(row)
            prob = float(self.model.predict_proba(row_s)[0][1])
            return round(float(np.clip(prob, 0.0, 1.0)), 4), mult
        except Exception as e:
            logger.debug(f"[{self.ticker}] meta predict error: {e}")
            return round(float(np.clip(ensemble_prob, 0.0, 1.0)), 4), mult


# ── Registry ──────────────────────────────────────────────────────────────────

_meta_registry: dict[str, MetaEnsemble] = {}
_meta_lock = threading.Lock()
_MAX_META_TICKERS = 100  # cap to prevent unbounded RAM growth


def get_or_create_meta(ticker: str) -> MetaEnsemble:
    with _meta_lock:
        if ticker not in _meta_registry:
            # Evict oldest entry if at cap (simple FIFO — LRU not needed here)
            if len(_meta_registry) >= _MAX_META_TICKERS:
                oldest = next(iter(_meta_registry))
                del _meta_registry[oldest]
            _meta_registry[ticker] = MetaEnsemble(ticker)
        return _meta_registry[ticker]


def get_meta_prediction(
    ticker:             str,
    scalp_prob:         float = 0.5,
    ensemble_prob:      float = 0.5,
    ensemble_agreement: float = 0.5,
    daily_prob:         float = 0.5,
    reversal_prob:      float = 0.5,
    tech_score:         float = 0.0,
    vol_score:          float = 0.0,
) -> tuple[float, float]:
    """
    Get final probability and confidence multiplier from the meta-ensemble.
    Returns (probability_up, confidence_multiplier).
    """
    return get_or_create_meta(ticker).predict(
        scalp_prob, ensemble_prob, ensemble_agreement,
        daily_prob, reversal_prob, tech_score, vol_score
    )
