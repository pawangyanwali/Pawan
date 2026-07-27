"""Email delivery for operational alerts.

This module is intentionally small and dependency-free.  It is used by
system_alerts.py after an alert has been durably written, so email failures
must never break the alert path.
"""
from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage
from typing import Any

logger = logging.getLogger(__name__)


def _truthy(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _recipients() -> list[str]:
    raw = os.getenv("ALERT_EMAIL_TO", "")
    return [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]


def is_configured() -> bool:
    """Return True when enough SMTP settings exist to attempt email delivery."""
    if not _truthy(os.getenv("ALERT_EMAIL_ENABLED"), default=True):
        return False
    return bool(os.getenv("SMTP_HOST", "").strip() and _recipients())


def send_alert_email(alert: dict[str, Any]) -> tuple[bool, str]:
    """Send one operational alert email. Returns (ok, error_message)."""
    if not is_configured():
        return False, "SMTP_HOST and ALERT_EMAIL_TO are not configured"

    host = os.getenv("SMTP_HOST", "").strip()
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "")
    use_ssl = _truthy(os.getenv("SMTP_SSL"), default=False)
    use_starttls = _truthy(os.getenv("SMTP_STARTTLS"), default=not use_ssl)
    sender = os.getenv("ALERT_EMAIL_FROM", username or "nasdaq-agent@localhost").strip()
    subject_prefix = os.getenv("ALERT_EMAIL_SUBJECT_PREFIX", "[NASDAQ Agent]").strip()

    severity = str(alert.get("severity") or "WARNING").upper()
    title = str(alert.get("title") or "Operational alert")
    source = str(alert.get("source") or alert.get("alert_type") or "system")
    message = str(alert.get("message") or "")
    metadata = alert.get("metadata") or {}

    body_lines = [
        f"Severity: {severity}",
        f"Type: {alert.get('alert_type', '')}",
        f"Source: {source}",
        f"Title: {title}",
        "",
        message,
    ]
    if isinstance(metadata, dict) and metadata:
        body_lines.extend(["", "Metadata:"])
        for key in sorted(metadata):
            value = metadata[key]
            name = str(key).lower()
            if "token" in name or "secret" in name or "password" in name:
                value = "<redacted>"
            body_lines.append(f"- {key}: {value}")

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = ", ".join(_recipients())
    msg["Subject"] = f"{subject_prefix} {severity}: {title}"
    msg.set_content("\n".join(body_lines))

    try:
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, timeout=10) as smtp:
                if username:
                    smtp.login(username, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=10) as smtp:
                if use_starttls:
                    smtp.starttls()
                if username:
                    smtp.login(username, password)
                smtp.send_message(msg)
        return True, ""
    except Exception as exc:
        logger.warning("[alert_email] send failed: %s", exc)
        return False, str(exc)
