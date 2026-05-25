"""
PostgreSQL schema for the auth system.

Tables:
  users           — registered users with roles, status, MFA
  refresh_tokens  — long-lived refresh token registry
  audit_log       — immutable record of auth events

Initial admin seed: pawan_gyanwali (seeded in seed.py).
"""
from __future__ import annotations

from agent.db import _get_pool

_USERS_DDL = """
CREATE TABLE IF NOT EXISTS users (
    id                    SERIAL PRIMARY KEY,
    username              TEXT UNIQUE NOT NULL,
    email                 TEXT UNIQUE NOT NULL,
    hashed_password       TEXT NOT NULL,
    role                  TEXT NOT NULL DEFAULT 'VIEWER',
    status                TEXT NOT NULL DEFAULT 'PENDING',
    force_password_change BOOLEAN NOT NULL DEFAULT FALSE,
    mfa_enabled           BOOLEAN NOT NULL DEFAULT FALSE,
    mfa_secret            TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    approved_at           TIMESTAMPTZ,
    approved_by           INTEGER REFERENCES users(id),
    last_login            TIMESTAMPTZ
)
"""

_REFRESH_TOKENS_DDL = """
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash  TEXT NOT NULL,
    jti         TEXT NOT NULL UNIQUE,
    issued_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMPTZ NOT NULL,
    revoked     BOOLEAN NOT NULL DEFAULT FALSE
)
"""

_AUDIT_LOG_DDL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id       SERIAL PRIMARY KEY,
    user_id  INTEGER REFERENCES users(id),
    action   TEXT NOT NULL,
    detail   JSONB,
    ip_addr  TEXT,
    ts       TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_MFA_PENDING_DDL = """
CREATE TABLE IF NOT EXISTS mfa_pending (
    user_id    INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    secret     TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user ON refresh_tokens(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_refresh_tokens_jti  ON refresh_tokens(jti)",
    "CREATE INDEX IF NOT EXISTS idx_audit_log_user      ON audit_log(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_log_ts        ON audit_log(ts DESC)",
]


def init_tables() -> None:
    """Create all auth tables if they don't exist."""
    pool = _get_pool()
    raw = pool.getconn()
    try:
        raw.autocommit = True
        with raw.cursor() as cur:
            for ddl in [
                _USERS_DDL,
                _REFRESH_TOKENS_DDL,
                _AUDIT_LOG_DDL,
                _MFA_PENDING_DDL,
            ]:
                cur.execute(ddl.strip())
            for idx in _INDEXES:
                cur.execute(idx)
    finally:
        pool.putconn(raw)
