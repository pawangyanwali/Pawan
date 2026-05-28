"""
Config / position-size routes:
  GET  /api/config   (returns app config)
  POST /api/config   (if defined — not present in main.py, but placeholder)
  GET  /api/position-size
"""

from fastapi import APIRouter

from config import DEFAULT_ACCOUNT_SIZE, DEFAULT_RISK_PCT, MAX_POSITION_PCT
from agent.position_sizing import calculate as calc_position

router = APIRouter(tags=["config"])


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
