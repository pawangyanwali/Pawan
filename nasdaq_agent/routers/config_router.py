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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
