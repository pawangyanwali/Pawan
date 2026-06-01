"""
Audit-trail routes — surfaces the decision log to the dashboard.
  GET /api/audit  — recent audit events (suppressions, threshold changes, …)
                    optional filters: ?event_type=THRESHOLD_CHANGED&ticker=NVDA&limit=100
"""

from fastapi import APIRouter, Depends

from auth.dependencies import require_viewer, AuthenticatedUser

from agent.audit_log import get_recent as _audit_recent

router = APIRouter(tags=["audit"])


@router.get("/api/audit")
async def list_audit(
    event_type: str | None = None,
    ticker:     str | None = None,
    limit:      int = 100,
    _user: AuthenticatedUser = Depends(require_viewer),
):
    """Return recent audit-log entries (newest first), optionally filtered."""
    limit = max(1, min(500, int(limit)))
    events = _audit_recent(limit=limit, event_type=event_type, ticker=ticker)
    return {"count": len(events), "events": events}
