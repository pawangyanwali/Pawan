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
        _swing_model_registry, get_retrain_progress, _MODEL_DIR,
    )
    import os

    def _count(registry, prefix: str) -> dict:
        import os as _os
        files = list(_MODEL_DIR.glob(f"{prefix}_*.joblib"))
        disk_trained    = len(files)
        in_mem_trained  = sum(1 for m in registry.values() if getattr(m, 'trained', False))
        trained  = max(disk_trained, in_mem_trained)
        total    = max(len(registry), disk_trained)
        last_trained = max((_os.path.getmtime(str(f)) for f in files), default=None) if files else None
        return {"total": total, "trained": trained, "last_trained": last_trained}

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
    disk_deep_trained = False
    try:
        from agent.deep_model import _cluster_trained, _CLUSTER_CONFIGS
        from config import CLUSTER_A_TICKERS, CLUSTER_B_TICKERS, CLUSTER_C_TICKERS
        cluster_tickers = {"a": CLUSTER_A_TICKERS, "b": CLUSTER_B_TICKERS, "c": CLUSTER_C_TICKERS}
        for cname in ("a", "b", "c"):
            cfg = _CLUSTER_CONFIGS.get(cname.upper(), {})
            model_path = cfg.get("path", "")
            mtime = None
            _is_trained = bool(model_path and os.path.exists(str(model_path)))
            if _is_trained:
                mtime = os.path.getmtime(str(model_path))
                disk_deep_trained = True
            cluster_status[cname] = {
                # Read trained state from disk (shared EBS mount) not in-process dict.
                # web-api and learner containers have separate _cluster_trained copies,
                # so the in-process dict is always False in web-api.
                "trained": _is_trained,
                "last_trained": mtime,
                "n_tickers": len(cluster_tickers.get(cname, [])),
            }
    except Exception:
        pass

    try:
        _sess = get_market_session()
    except Exception:
        _sess = "UNKNOWN"
    _can_retrain = (_sess == "CLOSED")

    # Read live training state from the learner container via shared Valkey key.
    # Fallback: in-process flag (always False in web-api — only used in single-process mode).
    is_now = is_training_active()
    deep_last_error = None
    deep_last_tickers = 0
    valkey_history: list = []
    try:
        import json as _json
        from agent.valkey_client import _get_client as _vk
        _vk_raw = _vk()
        if _vk_raw:
            _ls_raw = _vk_raw.get("learner:status")
            if _ls_raw:
                _deep = _json.loads(_ls_raw).get("deep", {})
                is_now = bool(_deep.get("running", False))
                deep_last_error = _deep.get("last_error")
                deep_last_tickers = _deep.get("last_tickers", 0)
                valkey_history = _deep.get("history", []) or []
    except Exception:
        pass

    # deep_trained: prefer the disk check (shared across containers) over the
    # in-process flag, which is always False in web-api.
    _deep_trained = disk_deep_trained or deep_is_trained()

    # Loss-curve history: web-api never runs training, so read it from shared state.
    # Source-of-truth order: PostgreSQL (durable) → Valkey (live mirror) → in-process.
    _history = []
    try:
        from agent.service_state import get_state
        _row = get_state("deep:history", ignore_expiry=True)
        if _row and _row.get("history"):
            _history = _row["history"]
    except Exception:
        pass
    if not _history:
        _history = valkey_history or get_training_history()

    _model_info = get_model_info()
    _model_info["trained"] = _deep_trained

    return {
        "deep_model":        _model_info,
        "deep_trained":      _deep_trained,
        "is_training_now":   is_now,
        "deep_last_error":   deep_last_error,
        "deep_last_tickers": deep_last_tickers,
        "training_history":  _history,
        "scalp_models":      _count(_model_registry,          "scalp"),
        "daily_models":      _count(_daily_model_registry,    "daily"),
        "reversal_models":   _count(_reversal_model_registry, "reversal"),
        "ensemble_models":   _count(_ensemble_registry,       "ensemble"),
        "swing_models":      _count(_swing_model_registry,    "swing"),
        "retrain_progress":  get_retrain_progress(),
        "blend_weights":     blend_stats,
        "pipeline_metrics":  pipeline_stats,
        "cluster_status":    cluster_status,
        "market_session":    _sess,
        "can_full_retrain":  _can_retrain,
        "retrain_note": (
            f"XGBoost retrain only runs when market is CLOSED (currently: {_sess}). "
            "BiLSTM can train anytime."
            if not _can_retrain else None
        ),
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
    _current: AuthenticatedUser = Depends(require_admin),
):
    """
    Manually trigger Deep BiLSTM training.

    Writes a training request to Valkey ("deep:train:requested").  The learner
    container's _continuous_deep_loop() polls for this key every 30 s and picks
    it up regardless of market session, running the full training cycle with its
    own authenticated Schwab data fetcher.  This avoids the web-api running data
    fetching or training in its own (resource-limited, no-auth) process.
    """
    # Check whether the learner is already training via shared Valkey state
    already_running = False
    try:
        import json as _json
        from agent.valkey_client import _get_client as _vk
        _client = _vk()
        if _client:
            _ls_raw = _client.get("learner:status")
            if _ls_raw:
                already_running = bool(_json.loads(_ls_raw).get("deep", {}).get("running", False))
    except Exception:
        pass

    if already_running:
        return {"status": "already_running", "message": "Deep model training already in progress."}

    # Write the training request so the learner container picks it up
    try:
        from agent.valkey_client import _get_client as _vk
        _client = _vk()
        if _client:
            _client.setex("deep:train:requested", 600, "1")
            logger.info("[deep-model/train] Training request written to Valkey; learner will pick up within 30s")
        else:
            logger.warning("[deep-model/train] Valkey unavailable — training request not queued")
            return {"status": "error", "message": "Valkey unavailable — cannot queue training request."}
    except Exception as exc:
        logger.warning(f"[deep-model/train] Failed to write training request: {exc}")
        return {"status": "error", "message": f"Failed to queue training request: {exc}"}

    return {
        "status": "started",
        "message": (
            "Training request queued. The learner container will begin within ~30 s. "
            "Check /api/ml-status → is_training_now for live progress."
        ),
    }


@router.get("/api/deep-model/status")
async def deep_model_status():
    """Deep BiLSTM model training status and architecture info."""
    from agent.deep_model import get_model_info
    return get_model_info()
