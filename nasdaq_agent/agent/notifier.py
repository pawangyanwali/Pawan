"""
Real-time signal notifier — Telegram alerts for high-confidence trading signals.

Configure via environment variables (or set once via /api/notify/config):
  TELEGRAM_BOT_TOKEN   — token from @BotFather
  TELEGRAM_CHAT_ID     — your chat/channel ID from @userinfobot
  NOTIFY_MIN_CONFIDENCE — minimum confidence to alert (default 75)

Rate limiting: at most one notification per ticker per 5 minutes so a
recurring high-confidence signal doesn't spam the channel.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_DEDUP: dict[str, float] = {}       # ticker+direction → last_notified epoch
_DEDUP_LOCK  = threading.Lock()
_DEDUP_SECS  = 300                  # 5-minute cooldown per ticker

_CFG_PATH = Path(__file__).parent.parent / "data" / "notifier_cfg.json"
_cfg_lock = threading.Lock()
_override_cfg: dict = {}            # runtime overrides (set via /api/notify/config)


def _load_cfg() -> dict:
    """Load persisted token/chat overrides (written by /api/notify/config)."""
    try:
        import json
        with open(_CFG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cfg(cfg: dict) -> None:
    _CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(_CFG_PATH, "w") as f:
        json.dump(cfg, f)


def _token() -> str:
    with _cfg_lock:
        return (_override_cfg.get("token") or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()


def _chat_id() -> str:
    with _cfg_lock:
        return (_override_cfg.get("chat_id") or os.getenv("TELEGRAM_CHAT_ID", "")).strip()


def _min_conf() -> float:
    with _cfg_lock:
        raw = _override_cfg.get("min_confidence") or os.getenv("NOTIFY_MIN_CONFIDENCE", "75")
    try:
        return float(raw)
    except Exception:
        return 75.0


def is_configured() -> bool:
    return bool(_token() and _chat_id())


def configure(token: str = "", chat_id: str = "", min_confidence: float = 75.0) -> None:
    """Set runtime config and persist to disk (survives restarts)."""
    global _override_cfg
    with _cfg_lock:
        _override_cfg = {
            "token":          token.strip(),
            "chat_id":        chat_id.strip(),
            "min_confidence": min_confidence,
        }
    _save_cfg(_override_cfg)
    logger.info(f"[Notifier] Config updated — chat_id={'***' if chat_id else 'unset'}")


def get_config() -> dict:
    return {
        "configured":      is_configured(),
        "chat_id":         _chat_id(),
        "token_set":       bool(_token()),
        "min_confidence":  _min_conf(),
    }


def send_telegram(message: str) -> tuple[bool, str]:
    """Send a message to the configured Telegram chat. Returns (ok, error_msg)."""
    token   = _token()
    chat_id = _chat_id()
    if not token or not chat_id:
        return False, "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured"
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=8,
        )
        if r.ok:
            return True, ""
        return False, r.text[:200]
    except Exception as exc:
        return False, str(exc)


def notify_signal(
    ticker:     str,
    direction:  str,
    confidence: float,
    price:      float,
    target:     float = 0.0,
    stop:       float = 0.0,
    session:    str   = "",
    regime:     str   = "",
) -> None:
    """
    Fire a Telegram alert for a new high-confidence signal.
    Rate-limited: at most once per ticker per DEDUP_SECS seconds.
    Runs in a background daemon thread so it never stalls the scan cycle.
    """
    if confidence < _min_conf():
        return
    if not is_configured():
        return

    now = time.time()
    key = f"{ticker}:{direction}"
    with _DEDUP_LOCK:
        if now - _DEDUP.get(key, 0.0) < _DEDUP_SECS:
            return
        _DEDUP[key] = now

    def _send():
        arrow = "📈" if "BUY" in direction else "📉"
        rr    = 0.0
        if stop and price and abs(price - stop) > 0:
            rr = round(abs(target - price) / abs(price - stop), 2)

        lines = [f"{arrow} <b>{ticker}  {direction}</b>  — {confidence:.0f}%"]
        lines.append(f"Price: <b>${price:.2f}</b>")
        if target and stop:
            rr_str = f"  •  R:R {rr:.1f}" if rr > 0 else ""
            lines.append(f"Target: ${target:.2f}  •  Stop: ${stop:.2f}{rr_str}")
        if session or regime:
            lines.append(f"Session: {session or '—'}  •  Regime: {regime or '—'}")

        ok, err = send_telegram("\n".join(lines))
        if not ok:
            logger.warning(f"[Notifier] Telegram send failed for {ticker}: {err}")

    threading.Thread(target=_send, daemon=True, name="notifier").start()


# Load persisted config at module import
try:
    _loaded = _load_cfg()
    if _loaded:
        with _cfg_lock:
            _override_cfg = _loaded
        logger.info("[Notifier] Loaded saved Telegram config from disk.")
except Exception:
    pass
