"""
Config / position-size routes:
  GET  /api/config          — flat key-value dict, optional ?prefix=X. filter
  POST /api/config          — flat key-value dict body to update one or more keys
  GET  /api/config/{key}    — single key with value + metadata
  GET  /api/position-size   — position-size calculator
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, status

from auth.dependencies import require_analyst, require_admin, AuthenticatedUser
from config import DEFAULT_ACCOUNT_SIZE, DEFAULT_RISK_PCT, MAX_POSITION_PCT
from agent.position_sizing import calculate as calc_position

router = APIRouter(tags=["config"])


# ── /api/config GET ────────────────────────────────────────────────────────────

@router.get("/api/config")
async def get_config(
    prefix: str | None = None,
    _user: AuthenticatedUser = Depends(require_analyst),
):
    """
    Return runtime config as a flat key-value dict.

    ?prefix=paper.  → only keys starting with "paper."
    No prefix       → all keys

    Default values (from _DEFAULTS) are included for keys not yet in the DB,
    so the dashboard always shows sensible starting values.
    """
    from agent.config_manager import config, _DEFAULTS

    # Seed defaults first (covers keys not yet written to DB)
    merged: dict[str, Any] = {}
    for key, factory in _DEFAULTS.items():
        try:
            merged[key] = factory()
        except Exception:
            pass
    # DB values take precedence
    merged.update(config.all())

    if prefix:
        merged = {k: v for k, v in merged.items() if k.startswith(prefix)}

    return {"count": len(merged), "config": merged}


@router.get("/api/config-catalog")
async def get_config_catalog(
    _user: AuthenticatedUser = Depends(require_analyst),
):
    """Return every runtime key with UI metadata and its effective value."""
    from agent.config_catalog import build_catalog
    from agent.config_manager import config, _DEFAULTS

    defaults: dict[str, Any] = {}
    for key, factory in _DEFAULTS.items():
        try:
            defaults[key] = factory()
        except Exception:
            continue
    values = dict(defaults)
    values.update(config.all())
    return build_catalog(values, defaults)


# ── /api/config/{key} GET ─────────────────────────────────────────────────────

@router.get("/api/config/{key:path}")
async def get_config_key(
    key: str,
    _user: AuthenticatedUser = Depends(require_analyst),
):
    """Return a single config key with current value and metadata."""
    from agent.config_manager import config, _DEFAULTS

    current = config.all()
    if key not in current and key not in _DEFAULTS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown config key: {key!r}",
        )

    default_val = None
    if key in _DEFAULTS:
        try:
            default_val = _DEFAULTS[key]()
        except Exception:
            pass

    return {
        "key":         key,
        "value":       current.get(key, default_val),
        "default":     default_val,
        "in_db":       key in current,
        "has_default": key in _DEFAULTS,
    }


# ── /api/config POST ──────────────────────────────────────────────────────────

@router.post("/api/config")
async def update_config(
    updates: dict[str, Any] = Body(..., description="Flat {key: value} pairs to update"),
    user: AuthenticatedUser = Depends(require_admin),
):
    """
    Update one or more runtime config keys.

    Body is a flat JSON object:
      {"risk.daily_loss_halt_pct": 3.0, "scanner.scan_interval_s": 30}

    Persists to PostgreSQL and publishes to Valkey for instant hot-reload
    across all containers — no restart required.
    """
    from agent.config_manager import config, _DEFAULTS

    if not updates:
        raise HTTPException(
            status_code=422,
            detail="Request body must be a non-empty key-value object",
        )

    # Reject unknown keys
    known_keys = set(_DEFAULTS.keys()) | set(config.all().keys())
    unknown = [k for k in updates if k not in known_keys]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown config key(s): {unknown}. Must match a known _DEFAULTS key.",
        )

    current_before = config.all()
    updates = dict(updates)

    # Validate the complete scalp configuration before persisting any part of it.
    # Several fields have cross-key constraints (for example TP1 <= TP2 and
    # min_stop_pct <= max_stop_pct), so validating keys independently is unsafe.
    if any(key.startswith("scalp.") for key in updates):
        from agent.scalp.models import ScalpSignalConfig

        candidate: dict[str, Any] = {}
        for key, factory in _DEFAULTS.items():
            try:
                candidate[key] = factory()
            except Exception:
                continue
        candidate.update(current_before)
        candidate.update(updates)
        try:
            ScalpSignalConfig.from_runtime(candidate)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid scalp configuration: {exc}",
            ) from exc

    if any(key.startswith("scalp_learn.") for key in updates):
        candidate: dict[str, Any] = {}
        for key, factory in _DEFAULTS.items():
            try:
                candidate[key] = factory()
            except Exception:
                continue
        candidate.update(current_before)
        candidate.update(updates)
        try:
            _validate_scalp_learning(candidate)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid scalp learning configuration: {exc}",
            ) from exc

    use_atr_stops = bool(
        updates.get(
            "prediction.use_atr_stops",
            current_before.get("prediction.use_atr_stops", True),
        )
    )
    if use_atr_stops and (
        "prediction.min_rr" in updates or "paper.t2_r_multiple" in updates
    ):
        try:
            min_rr = float(
                updates.get(
                    "prediction.min_rr",
                    current_before.get("prediction.min_rr", _DEFAULTS["prediction.min_rr"]()),
                )
            )
            t2_mult = float(
                updates.get(
                    "paper.t2_r_multiple",
                    current_before.get("paper.t2_r_multiple", _DEFAULTS["paper.t2_r_multiple"]()),
                )
            )
            if t2_mult < min_rr:
                updates["paper.t2_r_multiple"] = min_rr
        except Exception:
            pass

    try:
        config.set_many(updates, updated_by=user.username)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Config write failed: {exc}",
        )

    current = config.all()
    return {
        "updated": len(updates),
        "keys": [
            {"key": k, "value": current.get(k, updates[k])}
            for k in sorted(updates)
        ],
    }


def _validate_scalp_learning(values: dict[str, Any]) -> None:
    window = int(values["scalp_learn.rolling_window_min"])
    min_adjust = int(values["scalp_learn.min_samples_to_adjust"])
    min_block = int(values["scalp_learn.min_samples_to_block"])
    alpha = float(values["scalp_learn.ewma_alpha"])
    reduce_r = float(values["scalp_learn.negative_reduce_r"])
    block_r = float(values["scalp_learn.negative_block_r"])
    block_wr = float(values["scalp_learn.block_win_rate"])
    confidence_wr = float(values["scalp_learn.confidence_win_rate"])
    base_floor = float(values["scalp_learn.base_confidence_floor"])
    raise_step = float(values["scalp_learn.confidence_raise_step"])
    size_mult = float(values["scalp_learn.size_reduce_mult"])
    ttl = int(values["scalp_learn.action_ttl_min"])
    if window <= 0 or min_adjust <= 0 or ttl <= 0:
        raise ValueError("window, sample floor, and action lifetime must be positive")
    if min_block < min_adjust:
        raise ValueError("min_samples_to_block cannot be below min_samples_to_adjust")
    if not 0 < alpha <= 1:
        raise ValueError("ewma_alpha must be greater than 0 and no greater than 1")
    if not block_r <= reduce_r <= 0:
        raise ValueError("negative_block_r must be no greater than negative_reduce_r, and both must be non-positive")
    if not 0 <= block_wr <= confidence_wr <= 1:
        raise ValueError("learning win-rate thresholds must be ordered between 0 and 1")
    if not 0 <= base_floor <= 100 or not 0 <= raise_step <= 100:
        raise ValueError("confidence floor and raise step must be between 0 and 100")
    if base_floor + raise_step > 100:
        raise ValueError("base confidence floor plus raise step cannot exceed 100")
    if not 0 < size_mult <= 1:
        raise ValueError("size_reduce_mult must be greater than 0 and no greater than 1")


# ── /api/position-size GET ────────────────────────────────────────────────────

@router.get("/api/position-size")
async def position_size_endpoint(
    entry:        float = 0.0,
    stop:         float = 0.0,
    account_size: float = DEFAULT_ACCOUNT_SIZE,
    risk_pct:     float = DEFAULT_RISK_PCT,
    confidence:   float = 50.0,
    direction:    str   = "BUY",
):
    """Calculate position size for given entry/stop/account parameters."""
    ps = calc_position(
        account_size=account_size,
        entry=entry,
        stop=stop,
        risk_pct=risk_pct,
        confidence=confidence,
        max_position_pct=MAX_POSITION_PCT,
    )
    return ps.to_dict()
