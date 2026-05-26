"""
Durable scan-result state via Valkey.

Pattern (mirrors valkey_client.py for prices):
  KEY   scan:latest   — full JSON snapshot; survives web-api restarts
  PUBSUB scan:notify  — lightweight notification so subscribers wake up fast

Scanner writes after every cycle.
web-api reads on startup and subscribes for live updates.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_KEY     = "scan:latest"
_CHANNEL = "scan:notify"


# ── Write (scanner → Valkey) ──────────────────────────────────────────────────

def write_latest(
    signals:       list[dict[str, Any]],
    regime:        dict[str, Any],
    session:       dict[str, Any],
    scanned_count: int,
) -> bool:
    """
    Persist the latest scan result and publish a wake-up notification.
    Returns True on success, False if Valkey is unavailable (non-fatal).
    """
    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        if client is None:
            return False

        snapshot = json.dumps({
            "ts":            time.time(),
            "signals":       signals,
            "regime":        regime,
            "session":       session,
            "scanned_count": scanned_count,
        }, separators=(",", ":"), default=str)

        pipe = client.pipeline(transaction=False)
        pipe.set(_KEY, snapshot)
        pipe.publish(_CHANNEL, "1")   # payload is just a trigger; reader fetches the key
        pipe.execute()
        return True

    except Exception as exc:
        logger.debug("[signal_snapshot] write_latest error: %s", exc)
        return False


# ── Read (web-api → Valkey) ───────────────────────────────────────────────────

def read_latest() -> Optional[dict[str, Any]]:
    """
    Return the latest scan snapshot dict, or None if not yet available.
    Called by web-api on startup so the dashboard is immediately populated
    without waiting for the next scanner cycle.
    """
    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        if client is None:
            return None
        raw = client.get(_KEY)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.debug("[signal_snapshot] read_latest error: %s", exc)
        return None


# ── Subscribe (web-api background task) ──────────────────────────────────────

def subscribe_scan_results(callback: Callable[[dict[str, Any]], None]) -> None:
    """
    Block-subscribe to scan notifications.  On each notification, reads
    scan:latest from Valkey and calls callback(snapshot_dict).

    Runs in a daemon thread — call via threading.Thread(target=..., daemon=True).
    Falls back silently if Valkey is unavailable.
    """
    import threading
    import time as _time

    def _run() -> None:
        while True:
            try:
                from agent.valkey_client import _get_client
                import redis as _redis_lib
                host_port_ssl = _get_client()
                if host_port_ssl is None:
                    _time.sleep(5)
                    continue

                from agent.valkey_client import _cfg
                host, port, ssl = _cfg()
                sub = _redis_lib.Redis(
                    host=host, port=port, ssl=ssl,
                    ssl_cert_reqs=None,
                    socket_connect_timeout=5,
                    decode_responses=False,
                )
                ps = sub.pubsub()
                ps.subscribe(_CHANNEL)
                for msg in ps.listen():
                    if msg.get("type") != "message":
                        continue
                    snap = read_latest()
                    if snap:
                        try:
                            callback(snap)
                        except Exception as _ce:
                            logger.debug("[signal_snapshot] callback error: %s", _ce)
            except Exception as exc:
                logger.warning("[signal_snapshot] subscribe error: %s — retrying in 5s", exc)
                _time.sleep(5)

    t = threading.Thread(target=_run, daemon=True, name="scan-snapshot-sub")
    t.start()
