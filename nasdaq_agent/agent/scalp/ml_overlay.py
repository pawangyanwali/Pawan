"""Read-only champion inference and bounded confidence overlay."""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .ml_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    bounded_confidence_adjustment,
    expected_r,
    feature_vector,
)
from .models import ScalpSignalPlan, SignalSide

logger = logging.getLogger(__name__)
_CACHE_LOCK = threading.Lock()
_CACHE_CHECK_S = 30.0
_cache_checked_at = 0.0
_cache_version = ""
_cache_artifact: dict[str, Any] | None = None


def apply_ml_overlay(plan: ScalpSignalPlan) -> ScalpSignalPlan:
    """Populate advisory probabilities and optionally adjust confidence only."""
    from agent.config_manager import config

    shadow = bool(config.get("scalp_ml.shadow_enabled", False))
    enabled = bool(config.get("scalp_ml.overlay_enabled", False))
    if not (shadow or enabled) or not plan.valid or plan.side is SignalSide.NONE:
        return plan
    artifact = _champion_artifact()
    if not artifact:
        return plan
    try:
        vector = np.asarray([feature_vector(plan)], dtype=np.float32)
        p1 = float(artifact["tp1_model"].predict_proba(vector)[0, 1])
        p2 = min(p1, float(artifact["tp2_model"].predict_proba(vector)[0, 1]))
        expectancy = expected_r(p1, p2, plan.reward_r)
        adjustment = bounded_confidence_adjustment(expectancy)
        plan.ml_tp1_probability = round(p1, 6)
        plan.ml_tp2_probability = round(p2, 6)
        plan.ml_expected_r = round(expectancy, 6)
        plan.ml_confidence_adjustment = adjustment
        plan.ml_model_version = str(artifact["version_id"])
        plan.base_confidence = float(plan.base_confidence or plan.confidence)
        plan.ml_overlay_applied = enabled
        if enabled:
            plan.confidence = round(
                min(100.0, max(0.0, plan.base_confidence + adjustment)), 2
            )
            plan.reasons.append("ML_CONFIDENCE_OVERLAY")
        else:
            plan.reasons.append("ML_SHADOW_PREDICTION")
        _record_prediction(plan)
    except Exception as exc:
        logger.warning("[ScalpML] inference failed for %s: %s", plan.ticker, exc)
    return plan


def _champion_artifact() -> dict[str, Any] | None:
    global _cache_checked_at, _cache_version, _cache_artifact
    now = time.time()
    with _CACHE_LOCK:
        if now - _cache_checked_at <= _CACHE_CHECK_S:
            return _cache_artifact
        _cache_checked_at = now
        try:
            from .store import champion_ml_model

            metadata = champion_ml_model()
            if not metadata:
                _cache_version = ""
                _cache_artifact = None
                return None
            version = str(metadata.get("version_id") or "")
            created_at = _parse_ts(metadata.get("created_at"))
            from agent.config_manager import config
            maximum_age = timedelta(
                hours=max(1, int(config.get("scalp_ml.maximum_model_age_hours", 168)))
            )
            if datetime.now(timezone.utc) - created_at > maximum_age:
                raise ValueError("champion model is stale")
            if version == _cache_version and _cache_artifact is not None:
                return _cache_artifact
            path = Path(str(metadata.get("artifact_path") or ""))
            if not path.is_file():
                raise FileNotFoundError(f"champion artifact missing: {path}")
            expected_hash = str(metadata.get("artifact_sha256") or "")
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if not expected_hash or actual_hash != expected_hash:
                raise ValueError("champion artifact checksum mismatch")
            artifact = joblib.load(path)
            if int(artifact.get("feature_schema_version", -1)) != FEATURE_SCHEMA_VERSION:
                raise ValueError("champion feature schema is incompatible")
            if tuple(artifact.get("feature_names") or ()) != FEATURE_NAMES:
                raise ValueError("champion feature order is incompatible")
            if str(artifact.get("version_id") or "") != version:
                raise ValueError("champion registry and artifact versions differ")
            _cache_version = version
            _cache_artifact = artifact
            return artifact
        except Exception as exc:
            logger.warning("[ScalpML] champion load failed: %s", exc)
            _cache_version = ""
            _cache_artifact = None
            return None


def _record_prediction(plan: ScalpSignalPlan) -> None:
    try:
        from .store import record_ml_prediction

        record_ml_prediction(
            plan_id=plan.plan_id,
            predicted_at=datetime.now(timezone.utc).isoformat(),
            model_version=plan.ml_model_version,
            tp1_probability=plan.ml_tp1_probability,
            tp2_probability=plan.ml_tp2_probability,
            expected_r=plan.ml_expected_r,
            confidence_adjustment=plan.ml_confidence_adjustment,
            applied=plan.ml_overlay_applied,
        )
    except Exception as exc:
        logger.debug("[ScalpML] prediction audit failed: %s", exc)


def reset_model_cache() -> None:
    global _cache_checked_at, _cache_version, _cache_artifact
    with _CACHE_LOCK:
        _cache_checked_at = 0.0
        _cache_version = ""
        _cache_artifact = None


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return result.replace(tzinfo=result.tzinfo or timezone.utc).astimezone(timezone.utc)
