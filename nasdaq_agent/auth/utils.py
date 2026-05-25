"""Auth utilities: password hashing, JWT tokens, MFA (TOTP)."""
from __future__ import annotations

import json
import os
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt as _bcrypt
import hashlib
import jwt
import pyotp

from agent.db import get_conn

# ── Config ─────────────────────────────────────────────────────────────────────

_jwt_secret_raw = os.environ.get("JWT_SECRET", "")
if not _jwt_secret_raw:
    _env = os.environ.get("APP_ENV", "development").lower()
    if _env == "production":
        raise RuntimeError(
            "JWT_SECRET env var is not set. "
            "Cannot start in production without a stable signing key — "
            "add JWT_SECRET=<64-char-hex> to your .env file."
        )
    import logging as _log_boot
    _log_boot.getLogger("auth.utils").warning(
        "JWT_SECRET not set — using a per-process random key. "
        "All sessions will be invalidated on restart. "
        "Set JWT_SECRET in .env for persistence."
    )
    _jwt_secret_raw = secrets.token_hex(32)

JWT_SECRET   = _jwt_secret_raw
JWT_ALGO     = "HS256"
ACCESS_TTL   = int(os.environ.get("JWT_ACCESS_TTL_MINUTES", "30"))   # minutes
REFRESH_TTL  = int(os.environ.get("JWT_REFRESH_TTL_DAYS",   "7"))    # days

# ── Password helpers ───────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return _bcrypt.hashpw(plain.encode(), _bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def hash_token(token: str) -> str:
    """SHA-256 hash for long tokens (JWTs). bcrypt is limited to 72 bytes."""
    return hashlib.sha256(token.encode()).hexdigest()


# ── JWT helpers ────────────────────────────────────────────────────────────────

def create_access_token(user_id: int, username: str, role: str) -> tuple[str, str]:
    """Return (token, jti)."""
    jti = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    payload = {
        "sub":      str(user_id),
        "username": username,
        "role":     role,
        "jti":      jti,
        "iat":      now,
        "exp":      now + timedelta(minutes=ACCESS_TTL),
        "type":     "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO), jti


def create_refresh_token(user_id: int) -> tuple[str, str]:
    """Return (token, jti). Persists to refresh_tokens table."""
    jti = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    exp = now + timedelta(days=REFRESH_TTL)
    payload = {
        "sub":  str(user_id),
        "jti":  jti,
        "iat":  now,
        "exp":  exp,
        "type": "refresh",
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
    token_hash = hash_token(token)

    with get_conn() as c:
        c.execute(
            "INSERT INTO refresh_tokens (user_id, token_hash, jti, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, token_hash, jti, exp.isoformat()),
        )
    return token, jti


def decode_token(token: str) -> dict:
    """Decode and verify a JWT. Raises jwt.InvalidTokenError on failure."""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])


def create_ws_ticket(user_id: int, username: str, role: str) -> str:
    """60-second single-use ticket for WebSocket auth."""
    jti = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    payload = {
        "sub":      str(user_id),
        "username": username,
        "role":     role,
        "jti":      jti,
        "iat":      now,
        "exp":      now + timedelta(seconds=60),
        "type":     "ws_ticket",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def create_mfa_challenge_token(user_id: int, username: str, role: str) -> str:
    """60-second MFA challenge token — separate type from ws_ticket."""
    jti = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    payload = {
        "sub":      str(user_id),
        "username": username,
        "role":     role,
        "jti":      jti,
        "iat":      now,
        "exp":      now + timedelta(seconds=60),
        "type":     "mfa_challenge",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


# ── Valkey blacklist ───────────────────────────────────────────────────────────

def _valkey():
    try:
        from agent.valkey_client import _get_client
        return _get_client()
    except Exception:
        return None


def blacklist_jti(jti: str, ttl_seconds: int = ACCESS_TTL * 60 + 60) -> None:
    """Add a JTI to the Valkey blacklist so it can never be reused."""
    v = _valkey()
    if v:
        try:
            v.setex(f"blacklist:{jti}", ttl_seconds, "1")
        except Exception:
            pass


def is_blacklisted(jti: str) -> bool:
    v = _valkey()
    if v:
        try:
            return bool(v.exists(f"blacklist:{jti}"))
        except Exception:
            pass
    return False


def cache_user_role(user_id: int, role: str, status: str, ttl: int = 60) -> None:
    v = _valkey()
    if v:
        try:
            v.setex(f"user:{user_id}", ttl, json.dumps({"role": role, "status": status}))
        except Exception:
            pass


def get_cached_user(user_id: int) -> dict | None:
    v = _valkey()
    if v:
        try:
            raw = v.get(f"user:{user_id}")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    return None


def invalidate_user_cache(user_id: int) -> None:
    v = _valkey()
    if v:
        try:
            v.delete(f"user:{user_id}")
        except Exception:
            pass


# ── MFA (TOTP) ────────────────────────────────────────────────────────────────

def generate_mfa_secret() -> str:
    return pyotp.random_base32()


def get_totp_uri(secret: str, username: str, issuer: str = "NASDAQ Agent") -> str:
    totp = pyotp.TOTP(secret)
    return totp.provisioning_uri(name=username, issuer_name=issuer)


def get_qr_data_url(secret: str, username: str) -> str:
    """Return a data: URL for a QR code PNG (base64-encoded)."""
    import base64
    import io
    import qrcode

    uri = get_totp_uri(secret, username)
    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def verify_totp(secret: str, code: str) -> bool:
    """Verify a 6-digit TOTP code, allowing ±1 step drift."""
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=1)


# ── Audit logging ──────────────────────────────────────────────────────────────

def audit(
    action: str,
    user_id: int | None = None,
    detail: dict | None = None,
    ip_addr: str | None = None,
) -> None:
    try:
        with get_conn() as c:
            c.execute(
                "INSERT INTO audit_log (user_id, action, detail, ip_addr) "
                "VALUES (?, ?, ?, ?)",
                (user_id, action, json.dumps(detail) if detail else None, ip_addr),
            )
    except Exception:
        pass
