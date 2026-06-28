"""Chronological, economically-gated trainer for the advisory scalp overlay."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from agent.db import get_conn

from .ml_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    MIN_PLAN_SCHEMA_VERSION,
    expected_r,
    feature_vector,
)
from .store import init_scalp_tables, promote_ml_model, record_ml_evaluation

logger = logging.getLogger(__name__)
_TRAIN_LOCK = threading.Lock()
_ET = ZoneInfo("America/New_York")
_MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models" / "scalp_overlay"


@dataclass
class TrainingDataset:
    x: np.ndarray
    tp1: np.ndarray
    tp2: np.ndarray
    pnl_r: np.ndarray
    reward_r: np.ndarray
    closed_at: list[datetime]
    session_date: list[str]

    def __len__(self) -> int:
        return len(self.tp1)


def train_and_maybe_promote() -> dict[str, Any]:
    """Train one challenger and promote it only when every gate passes."""
    if not _TRAIN_LOCK.acquire(blocking=False):
        return {"status": "SKIPPED", "reason": "training already in progress"}
    try:
        return _train_locked()
    finally:
        _TRAIN_LOCK.release()


def _train_locked() -> dict[str, Any]:
    from agent.config_manager import config

    created_at = datetime.now(timezone.utc)
    version_id = f"scalp-ml-{created_at:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    dataset = load_training_dataset()
    minimum = max(50, int(config.get("scalp_ml.minimum_samples", 200)))
    if len(dataset) < minimum:
        return _reject(
            version_id, created_at, dataset,
            f"insufficient outcomes: {len(dataset)}/{minimum}",
        )

    holdout_pct = min(0.40, max(0.10, float(config.get("scalp_ml.holdout_pct", 0.25))))
    split = max(1, min(len(dataset) - 1, int(len(dataset) * (1.0 - holdout_pct))))
    train_idx = np.arange(0, split)
    holdout_idx = np.arange(split, len(dataset))
    reasons: list[str] = []
    for name, labels in (("TP1", dataset.tp1), ("TP2", dataset.tp2)):
        if len(np.unique(labels[train_idx])) < 2:
            reasons.append(f"{name} training labels contain one class")
        if len(np.unique(labels[holdout_idx])) < 2:
            reasons.append(f"{name} holdout labels contain one class")
    if reasons:
        return _reject(version_id, created_at, dataset, "; ".join(reasons), split=split)

    try:
        tp1_model = _fit_classifier(dataset.x[train_idx], dataset.tp1[train_idx])
        tp2_model = _fit_classifier(dataset.x[train_idx], dataset.tp2[train_idx])
        p1 = tp1_model.predict_proba(dataset.x[holdout_idx])[:, 1]
        p2 = np.minimum(p1, tp2_model.predict_proba(dataset.x[holdout_idx])[:, 1])
    except Exception as exc:
        logger.exception("[ScalpML] challenger fit failed")
        return _reject(version_id, created_at, dataset, f"fit failed: {exc}", split=split)

    y1 = dataset.tp1[holdout_idx]
    y2 = dataset.tp2[holdout_idx]
    train_p1 = float(np.mean(dataset.tp1[train_idx]))
    train_p2 = float(np.mean(dataset.tp2[train_idx]))
    model_expected = np.asarray([
        expected_r(a, b, reward)
        for a, b, reward in zip(p1, p2, dataset.reward_r[holdout_idx])
    ])
    threshold = float(config.get("scalp_ml.selection_expected_r", 0.0))
    selected_mask = model_expected >= threshold
    selected_indices = holdout_idx[selected_mask]
    selected_pnl = dataset.pnl_r[selected_indices]
    session_metrics = _session_metrics(dataset, selected_indices)
    metrics = {
        "tp1_auc": _auc(y1, p1),
        "tp2_auc": _auc(y2, p2),
        "tp1_brier": float(brier_score_loss(y1, p1)),
        "tp2_brier": float(brier_score_loss(y2, p2)),
        "tp1_baseline_brier": float(brier_score_loss(y1, np.full(len(y1), train_p1))),
        "tp2_baseline_brier": float(brier_score_loss(y2, np.full(len(y2), train_p2))),
        "selected_expectancy_r": float(np.mean(selected_pnl)) if len(selected_pnl) else 0.0,
        "selected_profit_factor": _profit_factor(selected_pnl),
        "selection_threshold_r": threshold,
        "session_metrics": session_metrics,
    }
    reasons = _promotion_reasons(metrics, len(selected_indices), config)
    metadata = _metadata(
        version_id, created_at, dataset, split, len(selected_indices), metrics,
        status="REJECTED" if reasons else "CHAMPION",
        rejection_reason="; ".join(reasons),
    )
    if reasons:
        record_ml_evaluation(metadata)
        logger.info("[ScalpML] challenger %s rejected: %s", version_id, metadata["rejection_reason"])
        return metadata

    artifact = {
        "version_id": version_id,
        "created_at": created_at.isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "tp1_model": tp1_model,
        "tp2_model": tp2_model,
        "metrics": metrics,
    }
    path, digest = _save_artifact(version_id, artifact)
    metadata.update(artifact_path=str(path), artifact_sha256=digest)
    promote_ml_model(metadata)
    logger.info(
        "[ScalpML] promoted %s | holdout expectancy=%+.3fR pf=%.3f selected=%d",
        version_id, metrics["selected_expectancy_r"],
        metrics["selected_profit_factor"], len(selected_indices),
    )
    return metadata


def load_training_dataset() -> TrainingDataset:
    from agent.config_manager import config

    init_scalp_tables()
    with get_conn(read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT o.closed_at, o.tp1_hit, o.tp2_hit, o.pnl_r, p.plan_json
            FROM scalp_trade_outcomes o
            JOIN scalp_signal_plans p ON p.plan_id=o.plan_id
            ORDER BY o.closed_at ASC, o.trade_id ASC
            """
        ).fetchall()
    vectors: list[list[float]] = []
    tp1: list[int] = []
    tp2: list[int] = []
    pnl: list[float] = []
    reward: list[float] = []
    timestamps: list[datetime] = []
    session_dates: list[str] = []
    cutoff = datetime.now(timezone.utc) - timedelta(
        days=max(5, int(config.get("scalp_ml.training_lookback_days", 60)))
    )
    for row in rows:
        try:
            plan = json.loads(row["plan_json"])
            if int(plan.get("schema_version") or 0) < MIN_PLAN_SCHEMA_VERSION:
                continue
            vector = feature_vector(plan)
            if len(vector) != len(FEATURE_NAMES):
                continue
            closed = _parse_ts(row["closed_at"])
            if closed < cutoff:
                continue
            vectors.append(vector)
            tp1.append(int(bool(row["tp1_hit"])))
            tp2.append(int(bool(row["tp2_hit"])))
            pnl.append(float(row["pnl_r"] or 0.0))
            reward.append(float(plan.get("reward_r") or 2.0))
            timestamps.append(closed)
            session_dates.append(closed.astimezone(_ET).date().isoformat())
        except Exception:
            continue
    x = np.asarray(vectors, dtype=np.float32)
    if not len(vectors):
        x = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return TrainingDataset(
        x=x,
        tp1=np.asarray(tp1, dtype=np.int8),
        tp2=np.asarray(tp2, dtype=np.int8),
        pnl_r=np.asarray(pnl, dtype=np.float32),
        reward_r=np.asarray(reward, dtype=np.float32),
        closed_at=timestamps,
        session_date=session_dates,
    )


def _fit_classifier(x: np.ndarray, labels: np.ndarray):
    from xgboost import XGBClassifier

    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    scale = negatives / positives if positives else 1.0
    model = XGBClassifier(
        n_estimators=160,
        max_depth=3,
        learning_rate=0.04,
        min_child_weight=5,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=2.0,
        scale_pos_weight=scale,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=42,
    )
    model.fit(x, labels, verbose=False)
    return model


def _promotion_reasons(metrics: dict[str, Any], selected: int, config) -> list[str]:
    reasons = []
    if selected < int(config.get("scalp_ml.minimum_selected_holdout", 30)):
        reasons.append("selected holdout sample floor not met")
    minimum_auc = float(config.get("scalp_ml.minimum_auc", 0.52))
    if metrics["tp1_auc"] < minimum_auc or metrics["tp2_auc"] < minimum_auc:
        reasons.append("TP1/TP2 holdout AUC below floor")
    brier_gain = float(config.get("scalp_ml.minimum_brier_improvement", 0.0))
    if metrics["tp1_baseline_brier"] - metrics["tp1_brier"] < brier_gain:
        reasons.append("TP1 probability calibration does not beat baseline")
    if metrics["tp2_baseline_brier"] - metrics["tp2_brier"] < brier_gain:
        reasons.append("TP2 probability calibration does not beat baseline")
    if metrics["selected_expectancy_r"] < float(config.get("scalp_ml.minimum_expectancy_r", 0.05)):
        reasons.append("out-of-sample expectancy below floor")
    if metrics["selected_profit_factor"] < float(config.get("scalp_ml.minimum_profit_factor", 1.10)):
        reasons.append("out-of-sample profit factor below floor")
    required_sessions = int(config.get("scalp_ml.minimum_holdout_sessions", 2))
    recent = metrics["session_metrics"][-required_sessions:]
    if len(recent) < required_sessions:
        reasons.append("insufficient distinct holdout sessions")
    else:
        min_count = int(config.get("scalp_ml.minimum_session_samples", 5))
        min_exp = float(config.get("scalp_ml.minimum_session_expectancy_r", 0.0))
        if any(row["count"] < min_count or row["expectancy_r"] < min_exp for row in recent):
            reasons.append("recent-session stability gate failed")
    return reasons


def _session_metrics(dataset: TrainingDataset, indices: np.ndarray) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for index in indices:
        grouped.setdefault(dataset.session_date[int(index)], []).append(float(dataset.pnl_r[int(index)]))
    return [
        {
            "session_date": day,
            "count": len(values),
            "expectancy_r": round(float(np.mean(values)), 6),
            "profit_factor": round(_profit_factor(np.asarray(values)), 6),
        }
        for day, values in sorted(grouped.items())
    ]


def _metadata(
    version_id: str, created_at: datetime, dataset: TrainingDataset, split: int,
    selected_count: int, metrics: dict[str, Any], *, status: str,
    rejection_reason: str = "",
) -> dict[str, Any]:
    return {
        "version_id": version_id,
        "created_at": created_at.isoformat(),
        "status": status,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "sample_count": len(dataset),
        "train_count": split,
        "holdout_count": max(0, len(dataset) - split),
        "selected_count": selected_count,
        "trained_through": dataset.closed_at[split - 1].isoformat() if split and dataset.closed_at else None,
        "evaluated_from": dataset.closed_at[split].isoformat() if split < len(dataset) else None,
        "evaluated_through": dataset.closed_at[-1].isoformat() if dataset.closed_at else None,
        "metrics": metrics,
        "rejection_reason": rejection_reason,
        "artifact_path": "",
        "artifact_sha256": "",
    }


def _reject(
    version_id: str, created_at: datetime, dataset: TrainingDataset,
    reason: str, *, split: int = 0,
) -> dict[str, Any]:
    metadata = _metadata(
        version_id, created_at, dataset, split, 0, {},
        status="REJECTED", rejection_reason=reason,
    )
    record_ml_evaluation(metadata)
    return metadata


def _save_artifact(version_id: str, artifact: dict[str, Any]) -> tuple[Path, str]:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_path = _MODEL_DIR / f"{version_id}.joblib"
    temporary = _MODEL_DIR / f".{version_id}.{uuid4().hex}.tmp"
    try:
        joblib.dump(artifact, temporary, compress=3)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, final_path)
        return final_path, digest
    finally:
        if temporary.exists():
            temporary.unlink()


def _profit_factor(values: np.ndarray) -> float:
    gains = float(np.sum(values[values > 0])) if len(values) else 0.0
    losses = abs(float(np.sum(values[values < 0]))) if len(values) else 0.0
    return gains / losses if losses > 0 else (999.0 if gains > 0 else 0.0)


def _auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, probabilities))
    except ValueError:
        return 0.0


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return result.replace(tzinfo=result.tzinfo or timezone.utc).astimezone(timezone.utc)
