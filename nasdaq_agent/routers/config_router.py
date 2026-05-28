"""
Config / position-size routes:
  GET  /api/config          — return all runtime config keys with values + metadata
  POST /api/config          — update one or more keys (persists to PG, hot-reloads via Valkey)
  GET  /api/config/{key}    — return a single config key
  GET  /api/position-size   — position-size calculator
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from auth.dependencies import require_analyst, require_admin, AuthenticatedUser
from config import DEFAULT_ACCOUNT_SIZE, DEFAULT_RISK_PCT, MAX_POSITION_PCT
from agent.position_sizing import calculate as calc_position

router = APIRouter(tags=["config"])


# ── Pydantic models ────────────────────────────────────────────────────────────

class ConfigUpdateRequest(BaseModel):
    """Body for POST /api/config — supply either key+value or updates dict."""
    key:     str | None = None
    value:   Any        = None
    updates: dict[str, Any] | None = None


# ── /api/config GET ────────────────────────────────────────────────────────────

@router.get("/api/config")
async def get_config(_user: AuthenticatedUser = Depends(require_analyst)):
    """Return all runtime config keys with current values and known defaults."""
    from agent.config_manager import config, _DEFAULTS

    current = config.all()

    items: list[dict] = []
    all_keys = set(current.keys()) | set(_DEFAULTS.keys())
    for key in sorted(all_keys):
        default_val = None
        if key in _DEFAULTS:
            try:
                default_val = _DEFAULTS[key]()
            except Exception:
                pass
        items.append({
            "key":           key,
            "value":         current.get(key, default_val),
            "default":       default_val,
            "in_db":         key in current,
            "has_default":   key in _DEFAULTS,
        })

    return {"count": len(items), "config": items}


# ── /api/config/{key} GET ─────────────────────────────────────────────────────

@router.get("/api/config/{key:path}")
async def get_config_key(
    key: str,
    _user: AuthenticatedUser = Depends(require_analyst),
):
    """Return the current value for a single config key."""
    from agent.config_manager import config, _DEFAULTS

    current = config.all()
    if key not in current and key not in _DEFAULTS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown config key: {key!r}")

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
    body: ConfigUpdateRequest,
    user: AuthenticatedUser = Depends(require_admin),
):
    """
    Update one or more runtime config keys.

    Single-key form:   {"key": "risk.daily_loss_halt_pct", "value": 3.0}
    Multi-key form:    {"updates": {"risk.daily_loss_halt_pct": 3.0, "scanner.scan_interval_s": 30}}
    """
    from agent.config_manager import config, _DEFAULTS

    updates: dict[str, Any] = {}

    if body.updates:
        updates.update(body.updates)
    if body.key is not None:
        if body.value is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="'value' is required when 'key' is provided",
            )
        updates[body.key] = body.value

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide 'key'+'value' or 'updates' dict",
        )

    # Reject unknown keys — only allow keys that exist in _DEFAULTS or already in DB
    known_keys = set(_DEFAULTS.keys()) | set(config.all().keys())
    unknown = [k for k in updates if k not in known_keys]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown config key(s): {unknown}. Add to _DEFAULTS first.",
        )

    try:
        config.set_many(updates, updated_by=user.username)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Config write failed: {exc}",
        )

    # Return the updated values
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
