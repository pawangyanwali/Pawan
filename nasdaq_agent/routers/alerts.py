"""
System alert routes — surfaces operational alerts to the dashboard.
  GET  /api/alerts            — active (unresolved) alerts + severity summary
  POST /api/alerts/{id}/resolve  — manually resolve an alert (admin)
"""

from fastapi import APIRouter, Depends

from auth.dependencies import require_viewer, require_admin, AuthenticatedUser

from agent.system_alerts import (
    get_alert_summary as _alert_summary,
    get_active_alerts as _active_alerts,
    resolve_alert     as _resolve_alert,
)

router = APIRouter(tags=["alerts"])


@router.get("/api/alerts")
async def list_alerts(_user: AuthenticatedUser = Depends(require_viewer)):
    """Return active alerts grouped by severity for the dashboard banner."""
    return _alert_summary()


@router.post("/api/alerts/{alert_id}/resolve")
async def resolve_alert_route(
    alert_id: int,
    _user: AuthenticatedUser = Depends(require_admin),
):
    """Manually resolve a single alert by its id (admin only)."""
    # Resolve by alert_key of the matching row so dedup state clears cleanly.
    for a in _active_alerts(limit=200):
        if a["id"] == alert_id:
            n = _resolve_alert(alert_key=a["alert_key"], resolved_by=_user.username)
            return {"ok": True, "resolved": n}
    return {"ok": False, "error": f"Alert #{alert_id} not found or already resolved"}
