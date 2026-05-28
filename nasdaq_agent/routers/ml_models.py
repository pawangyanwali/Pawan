"""
ML model routes:
  GET  /api/ml-status
  POST /api/ml-retrain
  POST /api/deep-model/train
  GET  /api/deep-model/status
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends

from auth.dependencies import require_viewer, require_admin, AuthenticatedUser
from agent.market_hours import get_market_session

router = APIRouter(tags=["ml_models"])

logger = logging.getLogger(__name__)


@router.get("/api/ml-status")
async def ml_status(_user: AuthenticatedUser = Depends(require_viewer)):
    """Aggregate status for all ML model types (includes blend weights and pipeline metrics)."""
    from agent.deep_model import get_model_info, get_training_history, is_training_active, is_trained as deep_is_trained
    from agent.ml_model import (
        _model_registry, _daily_model_registry,
        _reversal_model_registry, _ensemble_registry,
        _swing_model_registry, get_retrain_progress,
    )
    import os

    def _count(registry):
        total   = len(registry)
        trained = sum(1 for m in registry.values() if getattr(m, 'trained', False))
        return {"total": total, "trained": trained}

    blend_stats = None
    try:
        from agent.signal_blender import get_blender
        blend_stats = get_blender().get_stats()
    except Exception:
        pass

    pipeline_stats = None
    try:
        from agent.pipeline import get_pipeline
        pipeline_stats = get_pipeline().get_metrics()
    except Exception:
        pass

    cluster_status = {}
    try:
        from agent.deep_model import _cluster_trained, _CLUSTER_CONFIGS
        from config import CLUSTER_A_TICKERS, CLUSTER_B_TICKERS, CLUSTER_C_TICKERS
        cluster_tickers = {"a": CLUSTER_A_TICKERS, "b": CLUSTER_B_TICKERS, "c": CLUSTER_C_TICKERS}
        for cname in ("a", "b", "c"):
            cfg = _CLUSTER_CONFIGS.get(cname.upper(), {})
            model_path = cfg.get("path", "")
            mtime = None
            if model_path and os.path.exists(str(model_path)):
                mtime = os.path.getmtime(str(model_path))
            cluster_status[cname] = {
                "trained": bool(_cluster_trained.get(cname.upper(), False)),
                "last_trained": mtime,
                "n_tickers": len(cluster_tickers.get(cname, [])),
            }
    except Exception:
        pass

    return {
        "deep_model":        get_model_info(),
        "deep_trained":      deep_is_trained(),
        "is_training_now":   is_training_active(),
        "training_history":  get_training_history(),
        "scalp_models":      _count(_model_registry),
        "daily_models":      _count(_daily_model_registry),
        "reversal_models":   _count(_reversal_model_registry),
        "ensemble_models":   _count(_ensemble_registry),
        "swing_models":      _count(_swing_model_registry),
        "retrain_progress":  get_retrain_progress(),
        "blend_weights":     blend_stats,
        "pipeline_metrics":  pipeline_stats,
        "cluster_status":    cluster_status,
    }


@router.post("/api/ml-retrain")
async def trigger_retrain(
    background_tasks: BackgroundTasks,
    _current: AuthenticatedUser = Depends(require_admin),
):
    """
    Manually trigger a full ML retrain cycle (XGBoost + SwingML + Deep BiLSTM).
    Runs in background — check /api/ml-status for progress.
    """
    from agent.ml_model import retrain_all, _is_retraining
    from config import TRAINING_TICKERS

    if _is_retraining:
        return {"status": "already_running", "message": "Retrain already in progress."}

    try:
        session = get_market_session()
    except Exception:
        session = "UNKNOWN"
    if session != "CLOSED":
        return {
            "status": "deferred",
            "message": f"ML retrain deferred during {session}; run it in the CLOSED window.",
        }

    def _run():
        try:
            retrain_all(TRAINING_TICKERS)
        except Exception as e:
            logger.warning(f"[manual retrain] failed: {e}")

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Retrain started in background. Watch /api/ml-status for progress."}


@router.post("/api/deep-model/train")
async def trigger_deep_train(
    background_tasks: BackgroundTasks,
    _current: AuthenticatedUser = Depends(require_admin),
):
    """
    Manually trigger Deep BiLSTM training only (faster than full retrain).
    Uses cached 15-min data when available.
    """
    from agent.deep_model import is_training_active, retrain_deep_all
    from agent.data_fetcher import fetch_batch_interval
    from config import TRAINING_TICKERS

    if is_training_active():
        return {"status": "already_running", "message": "Deep model training already in progress."}

    try:
        session = get_market_session()
    except Exception:
        session = "UNKNOWN"
    if session != "CLOSED":
        return {
            "status": "deferred",
            "message": f"Deep model training deferred during {session}; run it in the CLOSED window.",
        }

    def _run():
        try:
            logger.info(f"[manual deep train] Fetching 15-min data ({len(TRAINING_TICKERS)} Tier-1)…")
            hist_15m = fetch_batch_interval(TRAINING_TICKERS, "15min", 5000, ttl=3600)
            logger.info(f"[manual deep train] Got {len(hist_15m)} tickers — starting training…")
            retrain_deep_all(hist_15m)
        except Exception as e:
            logger.warning(f"[manual deep train] failed: {e}")

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Deep BiLSTM training started. Check /api/ml-status for epoch progress."}


@router.get("/api/deep-model/status")
async def deep_model_status():
    """Deep BiLSTM model training status and architecture info."""
    from agent.deep_model import get_model_info
    return get_model_info()
