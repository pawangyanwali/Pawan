"""
Admin API endpoints for user management.

All endpoints require ADMIN role.

GET  /admin/users                    — list all users
GET  /admin/users/{user_id}          — get single user
PATCH /admin/users/{user_id}/approve — approve pending user
PATCH /admin/users/{user_id}/role    — change user role
PATCH /admin/users/{user_id}/suspend — suspend user
PATCH /admin/users/{user_id}/activate — re-activate user
DELETE /admin/users/{user_id}        — hard delete user
GET  /admin/audit-log                — recent audit events
GET  /admin/stats                    — user counts by role/status
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from agent.db import get_conn
from auth.dependencies import AuthenticatedUser, require_admin
from auth.utils import audit, invalidate_user_cache

router = APIRouter(prefix="/admin", tags=["admin"])

VALID_ROLES = {"ADMIN", "TRADER", "ANALYST", "VIEWER"}


class RoleUpdateRequest(BaseModel):
    role: str


class ApproveRequest(BaseModel):
    role: str = "VIEWER"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _user_or_404(user_id: int) -> dict:
    with get_conn() as c:
        row = c.execute(
            "SELECT id, username, email, role, status, "
            "       force_password_change, mfa_enabled, created_at, "
            "       approved_at, last_login "
            "FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return dict(row)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/users")
async def list_users(
    status: Optional[str] = Query(None, description="Filter by status"),
    role:   Optional[str] = Query(None, description="Filter by role"),
    admin:  AuthenticatedUser = Depends(require_admin),
):
    sql = (
        "SELECT id, username, email, role, status, "
        "       force_password_change, mfa_enabled, created_at, "
        "       approved_at, last_login "
        "FROM users WHERE 1=1"
    )
    params: list = []
    if status:
        sql += " AND status = %s"
        params.append(status.upper())
    if role:
        sql += " AND role = %s"
        params.append(role.upper())
    sql += " ORDER BY created_at DESC"

    with get_conn() as c:
        rows = c.execute(sql.replace("%s", "?"), params).fetchall()
    return [dict(r) for r in rows]


@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    admin: AuthenticatedUser = Depends(require_admin),
):
    return _user_or_404(user_id)


@router.patch("/users/{user_id}/approve")
async def approve_user(
    user_id: int,
    body: ApproveRequest,
    admin: AuthenticatedUser = Depends(require_admin),
):
    """Approve a PENDING user; set their role."""
    user = _user_or_404(user_id)
    if user["status"] == "ACTIVE":
        raise HTTPException(status_code=400, detail="User is already active")
    if body.role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role: {body.role}")

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    with get_conn() as c:
        c.execute(
            "UPDATE users SET status='ACTIVE', role=?, approved_at=?, approved_by=? "
            "WHERE id=?",
            (body.role, now, admin.id, user_id),
        )

    invalidate_user_cache(user_id)
    audit("user_approved", user_id=admin.id,
          detail={"target_user_id": user_id, "role": body.role})

    # Telegram notification
    try:
        from agent.notifier import send_telegram as _tg
        _tg(f"✅ User approved: {user['username']} → role={body.role}")
    except Exception:
        pass

    return {"message": f"User {user['username']} approved as {body.role}"}


@router.patch("/users/{user_id}/role")
async def update_role(
    user_id: int,
    body: RoleUpdateRequest,
    admin: AuthenticatedUser = Depends(require_admin),
):
    if body.role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role: {body.role}")

    _user_or_404(user_id)
    with get_conn() as c:
        c.execute("UPDATE users SET role=? WHERE id=?", (body.role, user_id))

    invalidate_user_cache(user_id)
    audit("role_changed", user_id=admin.id,
          detail={"target_user_id": user_id, "new_role": body.role})
    return {"message": f"Role updated to {body.role}"}


@router.patch("/users/{user_id}/suspend")
async def suspend_user(
    user_id: int,
    admin: AuthenticatedUser = Depends(require_admin),
):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot suspend yourself")
    _user_or_404(user_id)
    with get_conn() as c:
        c.execute("UPDATE users SET status='SUSPENDED' WHERE id=?", (user_id,))
    invalidate_user_cache(user_id)
    audit("user_suspended", user_id=admin.id, detail={"target_user_id": user_id})
    return {"message": "User suspended"}


@router.patch("/users/{user_id}/activate")
async def activate_user(
    user_id: int,
    admin: AuthenticatedUser = Depends(require_admin),
):
    _user_or_404(user_id)
    with get_conn() as c:
        c.execute("UPDATE users SET status='ACTIVE' WHERE id=?", (user_id,))
    invalidate_user_cache(user_id)
    audit("user_activated", user_id=admin.id, detail={"target_user_id": user_id})
    return {"message": "User activated"}


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    admin: AuthenticatedUser = Depends(require_admin),
):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    user = _user_or_404(user_id)
    with get_conn() as c:
        c.execute("DELETE FROM users WHERE id=?", (user_id,))
    audit("user_deleted", user_id=admin.id,
          detail={"target_user_id": user_id, "username": user["username"]})
    return {"message": f"User {user['username']} deleted"}


@router.get("/audit-log")
async def get_audit_log(
    limit: int = Query(100, ge=1, le=500),
    user_id: Optional[int] = Query(None),
    admin: AuthenticatedUser = Depends(require_admin),
):
    sql = (
        "SELECT al.id, al.user_id, u.username, al.action, "
        "       al.detail, al.ip_addr, al.ts "
        "FROM audit_log al LEFT JOIN users u ON u.id = al.user_id "
        "WHERE 1=1"
    )
    params: list = []
    if user_id:
        sql += " AND al.user_id = ?"
        params.append(user_id)
    sql += " ORDER BY al.ts DESC LIMIT ?"
    params.append(limit)

    with get_conn() as c:
        rows = c.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


@router.get("/stats")
async def admin_stats(admin: AuthenticatedUser = Depends(require_admin)):
    with get_conn() as c:
        rows = c.execute(
            "SELECT status, role, COUNT(*) AS cnt FROM users GROUP BY status, role"
        ).fetchall()
        total = c.execute("SELECT COUNT(*) AS cnt FROM users").fetchone()
        pending = c.execute(
            "SELECT COUNT(*) AS cnt FROM users WHERE status='PENDING'"
        ).fetchone()

    by_status: dict = {}
    by_role:   dict = {}
    for r in rows:
        by_status.setdefault(r["status"], 0)
        by_status[r["status"]] += r["cnt"]
        by_role.setdefault(r["role"], 0)
        by_role[r["role"]] += r["cnt"]

    return {
        "total":     total["cnt"] if total else 0,
        "pending":   pending["cnt"] if pending else 0,
        "by_status": by_status,
        "by_role":   by_role,
    }
