"""
Telegram notifier routes:
  GET  /api/notify/config
  POST /api/notify/config
  POST /api/notify/test
"""

from fastapi import APIRouter, Depends

from auth.dependencies import require_admin, AuthenticatedUser

from agent.notifier import (
    get_config as _notify_cfg,
    configure as _notify_configure,
    send_telegram as _send_telegram,
)

router = APIRouter(tags=["notify"])


@router.get("/api/notify/config")
async def notify_config():
    """Return current Telegram notifier configuration (token is never returned)."""
    return _notify_cfg()


@router.post("/api/notify/config")
async def notify_set_config(
    body: dict,
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Save Telegram bot token + chat ID. Persisted to disk across restarts."""
    token   = str(body.get("token",          "")).strip()
    chat_id = str(body.get("chat_id",        "")).strip()
    min_conf = float(body.get("min_confidence", 75.0))
    _notify_configure(token=token, chat_id=chat_id, min_confidence=min_conf)
    return {"ok": True, **_notify_cfg()}


@router.post("/api/notify/test")
async def notify_test(
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Send a test Telegram message to verify the config is working."""
    if not _notify_cfg().get("configured"):
        return {"ok": False, "error": "Not configured — set token and chat_id first"}
    ok, err = _send_telegram(
        "✅ <b>NASDAQ Agent</b> — Telegram notifications are working!\n"
        "You will receive alerts for high-confidence signals."
    )
    return {"ok": ok, "error": err if not ok else None}
