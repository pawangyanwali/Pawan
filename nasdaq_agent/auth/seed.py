"""
Seed the initial admin user if no users exist.

The bootstrap password is read from the ADMIN_TEMP_PASSWORD environment
variable. If not set, a random 16-character password is generated and
logged as a WARNING — the admin must retrieve it from the service logs
on first startup and change it immediately.
"""
from __future__ import annotations

import logging
import os
import secrets
import string

from agent.db import get_conn
from auth.utils import hash_password

logger = logging.getLogger(__name__)

_ADMIN_USERNAME = "pawan_gyanwali"
_ADMIN_EMAIL    = "pawangyanwali@gmail.com"


def _get_temp_password() -> str:
    pw = os.environ.get("ADMIN_TEMP_PASSWORD", "").strip()
    if pw:
        return pw
    # Generate a strong random password and warn loudly — admin reads it from logs
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    pw = "".join(secrets.choice(alphabet) for _ in range(20))
    logger.warning(
        "[Auth] ADMIN_TEMP_PASSWORD not set. "
        "Generated one-time bootstrap password: %s  "
        "(Change it immediately after first login — it will NOT be shown again.)",
        pw,
    )
    return pw


def seed_admin() -> None:
    """Insert the bootstrap admin user if it doesn't already exist."""
    with get_conn() as c:
        existing = c.execute(
            "SELECT id FROM users WHERE username = ?", (_ADMIN_USERNAME,)
        ).fetchone()

    if existing:
        logger.debug("[Auth] Admin user already exists — skipping seed")
        return

    temp_pw = _get_temp_password()
    hashed  = hash_password(temp_pw)
    with get_conn() as c:
        c.execute(
            "INSERT INTO users "
            "(username, email, hashed_password, role, status, force_password_change) "
            "VALUES (?, ?, ?, 'ADMIN', 'ACTIVE', TRUE)",
            (_ADMIN_USERNAME, _ADMIN_EMAIL, hashed),
        )

    logger.warning(
        "[Auth] Admin user '%s' created. "
        "Temporary password was logged above — change it on first login.",
        _ADMIN_USERNAME,
    )
