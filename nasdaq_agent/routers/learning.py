"""
Learning / adaptive-filter routes:
  GET  /api/learning/params
  GET  /api/learning/params/full
  GET  /api/learning/params/history
  POST /api/learning/params/set
  POST /api/learning/params/reset/{family}
  GET  /api/learning-status
  GET  /api/learning/phase2
  GET  /api/learning/walk-forward-stats
  POST /api/adaptive-filter/reset
  GET  /api/learning-log
  GET  /api/weekend-learning/status
  POST /api/weekend-learning/start
  POST /api/weekend-learning/stop
  GET  /api/weekend-learning/history
  GET  /api/weekend-learning/cache-stats
"""

import json
import logging
import time

from fastapi import APIRouter, Depends, Body
from fastapi import Path as FPath

from auth.dependencies import require_viewer, require_admin, AuthenticatedUser
from routers._deps import _pt_executor

from agent.adaptive_filter import get_status as af_get_status, reset_filter as af_reset_filter
from agent.signal_tracker import get_observation_summary

router = APIRouter(tags=["learning"])

logger = logging.getLogger(__name__)

_LEARNING_PARAMS_CACHE_TTL_SECS = 2.0
_learning_params_cache: dict | None = None
_learning_params_cache_ts: float = 0.0


@router.get("/api/learning/params")
async def learning_params_status():
    """Per-family learned parameter values for dashboard display."""
    global _learning_params_cache, _learning_params_cache_ts

    try:
        from agent.algo_learning_engine import get_algo_params as _get_algo_params
        _ALE_AVAILABLE = True
    except (ImportError, Exception):
        _get_algo_params = None
        _ALE_AVAILABLE = False

    if not _ALE_AVAILABLE:
        return {"available": False, "families": {}, "defaults": {}}

    now = time.monotonic()
    if (
        _learning_params_cache is not None
        and now - _learning_params_cache_ts < _LEARNING_PARAMS_CACHE_TTL_SECS
    ):
        return _learning_params_cache

    _FAMILY_REPRESENTATIVES = {
        "ORB":         "ORB5_BULL",
        "GAP_TREND":   "GAP_AND_GO_BULL",
        "GAP_FADE":    "GAP_FADE_BULL",
        "BREAKOUT":    "PDH_BREAKOUT_BULL",
        "FLAG":        "BULL_FLAG",
        "VWAP_SCALP":  "VWAP_TOUCH_SCALP_BULL",
        "LEVEL_SCALP": "LEVEL_REJECTION_SCALP_BULL",
        "RS_REGIME":   "SPY_BETA_CATCHUP_BULL",
    }
    defaults = {
        "rvol_gate": 1.5,
        "conf_gate": 55.0,
        "target_mult": 1.5,
        "stop_mult": 1.0,
        "entry_window_bars": 3,
    }

    def _safe_params(algo_name: str) -> dict:
        try:
            return _get_algo_params(algo_name)
        except Exception:
            return {}

    results = [_safe_params(algo) for algo in _FAMILY_REPRESENTATIVES.values()]
    families = {
        family: {k: (r.get(k, defaults[k]) if isinstance(r, dict) else defaults[k]) for k in defaults}
        for family, r in zip(_FAMILY_REPRESENTATIVES.keys(), results)
    }
    payload = {"available": True, "families": families, "defaults": defaults}
    _learning_params_cache = payload
    _learning_params_cache_ts = now
    return payload


@router.get("/api/learning/params/full")
async def learning_params_full():
    """Full per-family parameter data for the Algo Params dashboard tab."""
    try:
        from agent.algo_learning_engine import get_all_families_full as _get_full
        families = _get_full()
        return {"available": True, "families": families}
    except Exception as exc:
        logger.warning("learning/params/full error: %s", exc)
        return {"available": False, "families": {}}


@router.get("/api/learning/params/history")
async def learning_params_history(family: str | None = None, limit: int = 60):
    """Recent parameter tuning history records."""
    try:
        from agent.algo_learning_engine import get_algo_tune_history as _get_hist
        rows = _get_hist(family=family, limit=limit)
        return {"success": True, "rows": rows}
    except Exception as exc:
        logger.warning("learning/params/history error: %s", exc)
        return {"success": False, "rows": []}


@router.post("/api/learning/params/set")
async def learning_params_set(
    payload: dict = Body(...),
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Manually override a single algo-family parameter value."""
    family = payload.get("family", "")
    param  = payload.get("param", "")
    value  = payload.get("value")
    if not family or not param or value is None:
        return {"success": False, "error": "family, param, and value are required"}
    try:
        value = float(value)
    except (TypeError, ValueError):
        return {"success": False, "error": "value must be numeric"}
    try:
        from agent.algo_learning_engine import set_algo_param_manual as _set_param
        ok, msg = _set_param(family, param, value)
        return {"success": ok, "message": msg}
    except Exception as exc:
        logger.warning("learning/params/set error: %s", exc)
        return {"success": False, "error": str(exc)}


@router.post("/api/learning/params/reset/{family}")
async def learning_params_reset(
    family: str = FPath(...),
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Reset all parameters for an algo family back to defaults."""
    try:
        from agent.algo_learning_engine import reset_algo_family as _reset
        ok = _reset(family)
        return {"success": ok}
    except Exception as exc:
        logger.warning("learning/params/reset error: %s", exc)
        return {"success": False, "error": str(exc)}


@router.get("/api/learning-status")
async def learning_status():
    """Adaptive filter state — blocked contexts, dynamic threshold, win rate progress."""
    import asyncio

    try:
        from agent.learning_engine import learning_engine
        _LEARNER_ENABLED = True
    except (ImportError, Exception):
        class _NoopLearner:
            def get_status(self): return {}
        learning_engine = _NoopLearner()
        _LEARNER_ENABLED = False

    try:
        from agent.algo_learning_p2 import get_phase2_engine as _get_p2_engine
        _P2_AVAILABLE = True
    except (ImportError, Exception):
        _get_p2_engine = None
        _P2_AVAILABLE = False

    loop = asyncio.get_running_loop()
    status, obs = await asyncio.gather(
        loop.run_in_executor(None, af_get_status),
        loop.run_in_executor(_pt_executor, get_observation_summary),
    )
    engine_status = learning_engine.get_status()

    if not _LEARNER_ENABLED:
        try:
            from agent.valkey_client import _get_client as _vk_client
            _vk = _vk_client()
            if _vk:
                _raw = _vk.get("learner:status")
                if _raw:
                    _d = json.loads(_raw)
                    if time.time() - _d.get("ts", 0) < 300:
                        engine_status = _d.get("engine", engine_status)
                        status        = _d.get("adaptive_filter", status)
        except Exception:
            pass

    result = {
        **status,
        "engine":       engine_status,
        "observations": obs,
    }
    if _P2_AVAILABLE:
        try:
            p2_status = await loop.run_in_executor(None, lambda: _get_p2_engine().get_status())
            result["deployment_mode"]  = p2_status.get("deployment", {}).get("current_mode", "SHADOW")
            result["drift_alerts"]     = p2_status.get("drift", {}).get("material_drifts", [])
        except Exception as _p2e:
            logger.debug("learning-status phase2 error: %s", _p2e)
    return result


@router.get("/api/learning/phase2")
async def learning_phase2_status():
    """Phase 2 status: concept drift, staged deployment, walk-forward validation, transfer tier."""
    try:
        from agent.algo_learning_p2 import get_phase2_engine as _get_p2_engine
        _P2_AVAILABLE = True
    except (ImportError, Exception):
        _get_p2_engine = None
        _P2_AVAILABLE = False

    if not _P2_AVAILABLE:
        return {"available": False}
    try:
        status = _get_p2_engine().get_status()
        return {"available": True, **status}
    except Exception as exc:
        logger.warning("phase2 status error: %s", exc)
        return {"available": True, "error": str(exc)}


@router.get("/api/learning/walk-forward-stats")
async def walk_forward_stats():
    """Walk-forward trainer last-run summary — per-family stats and param recommendations."""
    import asyncio

    try:
        from agent.walk_forward_trainer import get_walk_forward_trainer as _get_wf_trainer
        _WFT_AVAILABLE = True
    except (ImportError, Exception):
        _get_wf_trainer = None
        _WFT_AVAILABLE = False

    if not _WFT_AVAILABLE:
        return {"available": False}
    loop = asyncio.get_running_loop()
    try:
        status = await loop.run_in_executor(None, lambda: _get_wf_trainer().get_status())
        return {"available": True, **status}
    except Exception as exc:
        logger.warning("walk-forward-stats error: %s", exc)
        return {"available": True, "error": str(exc)}


@router.post("/api/adaptive-filter/reset")
async def reset_adaptive_filter(
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Reset the adaptive filter to factory defaults (threshold 60%, no blocked contexts)."""
    af_reset_filter()
    return {"ok": True, **af_get_status()}


@router.get("/api/learning-log")
async def learning_log_endpoint(limit: int = 100):
    """Last N learning engine log entries for the dashboard live feed."""
    try:
        from agent.learning_engine import learning_engine, get_learning_log
    except (ImportError, Exception):
        def get_learning_log(limit=100): return []
        class _NoopLearner:
            def get_status(self): return {}
        learning_engine = _NoopLearner()

    return {
        "log":    get_learning_log(limit=limit),
        "engine": learning_engine.get_status(),
    }


@router.get("/api/weekend-learning/status")
async def wl_status():
    """Current state of the weekend learning pipeline."""
    try:
        import agent.weekend_learner as weekend_learner
    except (ImportError, Exception):
        class _NoopWeekendLearner:
            def get_status(self): return {}
        weekend_learner = _NoopWeekendLearner()
    return weekend_learner.get_status()


@router.post("/api/weekend-learning/start")
async def wl_start(
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Manually kick off the weekend learning pipeline (admin override)."""
    try:
        import agent.weekend_learner as weekend_learner
    except (ImportError, Exception):
        class _NoopWeekendLearner:
            def start(self): return False
        weekend_learner = _NoopWeekendLearner()
    started = weekend_learner.start()
    return {
        "status":  "started" if started else "already_running",
        "message": ("Weekend learning started in background."
                    if started else "Weekend learner is already running."),
    }


@router.post("/api/weekend-learning/stop")
async def wl_stop(
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Signal the weekend learner to stop after the current phase."""
    try:
        import agent.weekend_learner as weekend_learner
        weekend_learner.stop()
    except (ImportError, Exception):
        pass
    return {"status": "stop_requested"}


@router.get("/api/weekend-learning/history")
async def wl_history():
    """Cumulative weekend learning outcomes from SQLite records store."""
    try:
        import agent.weekend_learner as weekend_learner
        return weekend_learner.historical_performance()
    except (ImportError, Exception):
        return {}


@router.get("/api/weekend-learning/cache-stats")
async def wl_cache_stats():
    """OHLCV cache stats — bars per ticker/interval stored so far."""
    from agent.historical_cache import cache_stats
    return cache_stats()
