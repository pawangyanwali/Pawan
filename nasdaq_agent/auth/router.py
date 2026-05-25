"""
Auth API endpoints.

POST /auth/register          — request an account (status=PENDING)
POST /auth/login             — get access + refresh tokens
POST /auth/refresh           — exchange refresh token for new access token
POST /auth/logout            — revoke current access + refresh tokens
POST /auth/change-password   — change password (required on first login)
GET  /auth/me                — current user info
GET  /auth/ws-ticket         — 60-second one-use WebSocket ticket
POST /auth/mfa/setup         — begin MFA enrollment (admin-triggered or self)
POST /auth/mfa/verify-setup  — confirm TOTP code to activate MFA
POST /auth/mfa/verify        — submit TOTP code during login (step 2)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, field_validator

from agent.db import get_conn
from auth.dependencies import AuthenticatedUser, get_current_user
from auth.utils import (
    audit,
    blacklist_jti,
    create_access_token,
    create_refresh_token,
    create_ws_ticket,
    decode_token,
    generate_mfa_secret,
    get_qr_data_url,
    hash_password,
    invalidate_user_cache,
    verify_password,
    verify_totp,
)

router = APIRouter(prefix="/auth", tags=["auth"])

_TELEGRAM_NOTIFY = os.environ.get("TELEGRAM_NOTIFY_ADMIN_REGISTER", "1") == "1"


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

    @field_validator("username")
    @classmethod
    def username_valid(cls, v: str) -> str:
        if len(v) < 3 or len(v) > 32:
            raise ValueError("username must be 3-32 chars")
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("username may only contain letters, digits, _ and -")
        return v.lower()

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    username: str
    password: str
    totp_code: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("new_password must be at least 8 characters")
        return v


class RefreshRequest(BaseModel):
    refresh_token: str


class MFASetupVerifyRequest(BaseModel):
    totp_code: str


class MFAVerifyRequest(BaseModel):
    username: str
    totp_code: str
    # The partial token issued at step 1 of MFA login
    mfa_token: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_user_by_username(username: str) -> dict | None:
    with get_conn() as c:
        row = c.execute(
            "SELECT * FROM users WHERE username = ?", (username.lower(),)
        ).fetchone()
    return dict(row) if row else None


def _get_user_by_id(user_id: int) -> dict | None:
    with get_conn() as c:
        row = c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _notify_admin_new_user(username: str, email: str) -> None:
    if not _TELEGRAM_NOTIFY:
        return
    try:
        from agent.notifier import send_telegram as _tg
        _tg(f"🔔 New user registration pending approval: {username} ({email})")
    except Exception:
        pass


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/register", status_code=201)
async def register(body: RegisterRequest, request: Request):
    """Request a new account. Status starts as PENDING until admin approves."""
    existing = _get_user_by_username(body.username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")

    with get_conn() as c:
        dup_email = c.execute(
            "SELECT id FROM users WHERE email = ?", (body.email,)
        ).fetchone()
    if dup_email:
        raise HTTPException(status_code=409, detail="Email already registered")

    hashed = hash_password(body.password)
    with get_conn() as c:
        cur = c.execute(
            "INSERT INTO users (username, email, hashed_password, role, status) "
            "VALUES (?, ?, ?, 'VIEWER', 'PENDING') RETURNING id",
            (body.username, body.email, hashed),
        )
        row = cur.fetchone()
        user_id = row["id"] if row else None

    audit("register", user_id=user_id, ip_addr=_client_ip(request),
          detail={"username": body.username, "email": body.email})
    _notify_admin_new_user(body.username, body.email)
    return {"message": "Registration submitted. Awaiting admin approval."}


@router.post("/login")
async def login(body: LoginRequest, request: Request):
    """
    Authenticate and issue tokens.

    Returns one of:
      - {"action": "change_password"} if force_password_change is set
      - {"action": "mfa_required", "mfa_token": "..."} if MFA enabled
      - {"access_token": ..., "refresh_token": ..., "token_type": "bearer"}
    """
    user = _get_user_by_username(body.username)
    if not user or not verify_password(body.password, user["hashed_password"]):
        audit("login_failed", ip_addr=_client_ip(request),
              detail={"username": body.username})
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if user["status"] == "PENDING":
        raise HTTPException(status_code=403, detail="Account pending admin approval")
    if user["status"] == "SUSPENDED":
        raise HTTPException(status_code=403, detail="Account suspended")

    if user["force_password_change"]:
        return {"action": "change_password"}

    if user["mfa_enabled"]:
        if not body.totp_code:
            # Issue a short-lived MFA challenge token (not a full access token)
            mfa_token = create_ws_ticket(user["id"], user["username"], user["role"])
            return {"action": "mfa_required", "mfa_token": mfa_token}
        # MFA code provided inline
        if not user["mfa_secret"] or not verify_totp(user["mfa_secret"], body.totp_code):
            audit("mfa_failed", user_id=user["id"], ip_addr=_client_ip(request))
            raise HTTPException(status_code=401, detail="Invalid MFA code")

    access_token, _jti = create_access_token(user["id"], user["username"], user["role"])
    refresh_token, _   = create_refresh_token(user["id"])

    with get_conn() as c:
        c.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), user["id"]),
        )

    audit("login", user_id=user["id"], ip_addr=_client_ip(request))
    return {
        "access_token":  access_token,
        "refresh_token": refresh_token,
        "token_type":    "bearer",
        "role":          user["role"],
        "username":      user["username"],
    }


@router.post("/mfa/verify")
async def mfa_verify(body: MFAVerifyRequest, request: Request):
    """Step 2 of MFA login: submit TOTP code with the challenge token."""
    try:
        payload = decode_token(body.mfa_token)
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired MFA token")

    if payload.get("type") != "ws_ticket":
        raise HTTPException(status_code=401, detail="Invalid token type")

    user_id = int(payload["sub"])
    user = _get_user_by_id(user_id)
    if not user or user["username"] != body.username.lower():
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user["mfa_secret"] or not verify_totp(user["mfa_secret"], body.totp_code):
        audit("mfa_failed", user_id=user_id, ip_addr=_client_ip(request))
        raise HTTPException(status_code=401, detail="Invalid MFA code")

    access_token, _ = create_access_token(user["id"], user["username"], user["role"])
    refresh_token, _ = create_refresh_token(user["id"])

    with get_conn() as c:
        c.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), user["id"]),
        )

    audit("login", user_id=user["id"], ip_addr=_client_ip(request))
    return {
        "access_token":  access_token,
        "refresh_token": refresh_token,
        "token_type":    "bearer",
        "role":          user["role"],
        "username":      user["username"],
    }


@router.post("/refresh")
async def refresh_token(body: RefreshRequest):
    """Exchange a valid refresh token for a new access token."""
    try:
        payload = decode_token(body.refresh_token)
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Wrong token type")

    jti     = payload["jti"]
    user_id = int(payload["sub"])

    with get_conn() as c:
        row = c.execute(
            "SELECT id, revoked FROM refresh_tokens WHERE jti = ?", (jti,)
        ).fetchone()

    if not row or row["revoked"]:
        raise HTTPException(status_code=401, detail="Refresh token revoked")

    user = _get_user_by_id(user_id)
    if not user or user["status"] != "ACTIVE":
        raise HTTPException(status_code=403, detail="Account not active")

    access_token, _ = create_access_token(user["id"], user["username"], user["role"])
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/logout")
async def logout(
    body: RefreshRequest,
    current: AuthenticatedUser = Depends(get_current_user),
):
    """Revoke access + refresh tokens."""
    blacklist_jti(current.jti)

    try:
        payload = decode_token(body.refresh_token)
        if payload.get("type") == "refresh":
            with get_conn() as c:
                c.execute(
                    "UPDATE refresh_tokens SET revoked = TRUE WHERE jti = ?",
                    (payload["jti"],),
                )
    except Exception:
        pass

    audit("logout", user_id=current.id)
    return {"message": "Logged out"}


@router.post("/change-password")
async def change_password(body: ChangePasswordRequest, request: Request):
    """
    Change password. Accepts both authenticated users and unauthenticated
    force-change-password flows (where current_password proves identity).
    """
    # Works as unauthenticated since the old password verifies identity
    # (used on first-login force-change flow).
    raise HTTPException(status_code=501, detail="Use /auth/change-password-forced")


@router.post("/change-password-forced")
async def change_password_forced(
    body: ChangePasswordRequest,
    request: Request,
    username: str,
):
    """Force-change flow — called before a real access token is issued."""
    user = _get_user_by_username(username)
    if not user or not verify_password(body.current_password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if body.new_password == body.current_password:
        raise HTTPException(
            status_code=400, detail="New password must differ from current"
        )

    new_hash = hash_password(body.new_password)

    # Create tokens first — if this fails, the password is NOT changed yet.
    access_token, _  = create_access_token(user["id"], user["username"], user["role"])
    refresh_token, _ = create_refresh_token(user["id"])

    with get_conn() as c:
        c.execute(
            "UPDATE users SET hashed_password = ?, force_password_change = FALSE WHERE id = ?",
            (new_hash, user["id"]),
        )

    invalidate_user_cache(user["id"])
    audit("password_changed", user_id=user["id"], ip_addr=_client_ip(request))
    return {
        "access_token":  access_token,
        "refresh_token": refresh_token,
        "token_type":    "bearer",
        "role":          user["role"],
        "username":      user["username"],
    }


@router.get("/me")
async def me(current: AuthenticatedUser = Depends(get_current_user)):
    user = _get_user_by_id(current.id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id":          user["id"],
        "username":    user["username"],
        "email":       user["email"],
        "role":        user["role"],
        "mfa_enabled": user["mfa_enabled"],
        "last_login":  user["last_login"],
    }


@router.get("/ws-ticket")
async def ws_ticket(current: AuthenticatedUser = Depends(get_current_user)):
    """Issue a 60-second one-use ticket for WebSocket connection."""
    ticket = create_ws_ticket(current.id, current.username, current.role)
    return {"ticket": ticket}


@router.post("/mfa/setup")
async def mfa_setup(current: AuthenticatedUser = Depends(get_current_user)):
    """Begin MFA enrollment — generate TOTP secret and return QR code."""
    secret = generate_mfa_secret()

    with get_conn() as c:
        c.execute(
            "INSERT INTO mfa_pending (user_id, secret) VALUES (?, ?) "
            "ON CONFLICT (user_id) DO UPDATE SET secret = EXCLUDED.secret, "
            "created_at = NOW()",
            (current.id, secret),
        )

    qr = get_qr_data_url(secret, current.username)
    return {"qr_code": qr, "secret": secret}


@router.post("/mfa/verify-setup")
async def mfa_verify_setup(
    body: MFASetupVerifyRequest,
    current: AuthenticatedUser = Depends(get_current_user),
):
    """Confirm TOTP code to activate MFA on the account."""
    with get_conn() as c:
        row = c.execute(
            "SELECT secret FROM mfa_pending WHERE user_id = ?", (current.id,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=400, detail="No pending MFA setup")

    if not verify_totp(row["secret"], body.totp_code):
        raise HTTPException(status_code=400, detail="Invalid TOTP code")

    with get_conn() as c:
        c.execute(
            "UPDATE users SET mfa_enabled = TRUE, mfa_secret = ? WHERE id = ?",
            (row["secret"], current.id),
        )
        c.execute("DELETE FROM mfa_pending WHERE user_id = ?", (current.id,))

    invalidate_user_cache(current.id)
    audit("mfa_enabled", user_id=current.id)
    return {"message": "MFA enabled successfully"}
