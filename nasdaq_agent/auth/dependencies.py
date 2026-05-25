"""FastAPI dependency injection for auth: get_current_user, role guards."""
from __future__ import annotations

import time
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agent.db import get_conn
from auth.utils import (
    decode_token,
    is_blacklisted,
    get_cached_user,
    cache_user_role,
)

_bearer = HTTPBearer(auto_error=False)

# Role hierarchy: each role includes all permissions below it
_ROLE_RANK = {"ADMIN": 40, "TRADER": 30, "ANALYST": 20, "VIEWER": 10}


class AuthenticatedUser:
    __slots__ = ("id", "username", "role", "status", "jti")

    def __init__(self, id: int, username: str, role: str, status: str, jti: str):
        self.id       = id
        self.username = username
        self.role     = role
        self.status   = status
        self.jti      = jti


def _get_user_from_db(user_id: int) -> dict | None:
    with get_conn() as c:
        row = c.execute(
            "SELECT id, username, role, status FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


async def get_current_user(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> AuthenticatedUser:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if creds is None:
        raise exc

    try:
        payload = decode_token(creds.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise exc

    if payload.get("type") != "access":
        raise exc

    jti     = payload.get("jti", "")
    user_id = int(payload["sub"])

    # Fast path: Valkey blacklist check
    if is_blacklisted(jti):
        raise exc

    # Valkey hot-path for role/status (avoids DB hit on every request)
    cached = get_cached_user(user_id)
    if cached:
        role   = cached["role"]
        ust    = cached["status"]
    else:
        user = _get_user_from_db(user_id)
        if not user:
            raise exc
        role = user["role"]
        ust  = user["status"]
        cache_user_role(user_id, role, ust)

    if ust != "ACTIVE":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account suspended or pending approval",
        )

    return AuthenticatedUser(
        id=user_id,
        username=payload.get("username", ""),
        role=role,
        status=ust,
        jti=jti,
    )


def require_role(min_role: str):
    """Return a Depends that enforces a minimum role level."""
    min_rank = _ROLE_RANK.get(min_role, 0)

    async def _check(user: AuthenticatedUser = Depends(get_current_user)):
        if _ROLE_RANK.get(user.role, 0) < min_rank:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"{min_role} role required",
            )
        return user

    return _check


require_admin   = require_role("ADMIN")
require_trader  = require_role("TRADER")
require_analyst = require_role("ANALYST")
require_viewer  = require_role("VIEWER")
