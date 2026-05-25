"""
Seed the initial admin user if no users exist.

Admin: pawan_gyanwali
  - Temp password:  NasdaqAdmin@2024  (must be changed on first login)
  - Role:           ADMIN
  - Status:         ACTIVE
  - force_password_change: TRUE
  - MFA:            disabled (admin enables it after first login)
"""
from __future__ import annotations

import logging

from agent.db import get_conn
from auth.utils import hash_password

logger = logging.getLogger(__name__)

_ADMIN_USERNAME = "pawan_gyanwali"
_ADMIN_EMAIL    = "pawangyanwali@gmail.com"
_TEMP_PASSWORD  = "NasdaqAdmin@2024"   # forced change on first login


def seed_admin() -> None:
    """Insert the bootstrap admin user if it doesn't already exist."""
    with get_conn() as c:
        existing = c.execute(
            "SELECT id FROM users WHERE username = ?", (_ADMIN_USERNAME,)
        ).fetchone()

    if existing:
        logger.debug("[Auth] Admin user already exists — skipping seed")
        return

    hashed = hash_password(_TEMP_PASSWORD)
    with get_conn() as c:
        c.execute(
            "INSERT INTO users "
            "(username, email, hashed_password, role, status, force_password_change) "
            "VALUES (?, ?, ?, 'ADMIN', 'ACTIVE', TRUE)",
            (_ADMIN_USERNAME, _ADMIN_EMAIL, hashed),
        )

    logger.info(
        "[Auth] Admin user created: %s — temp password must be changed on first login",
        _ADMIN_USERNAME,
    )
