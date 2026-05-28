"""Uniform service heartbeat publisher for container-level observability."""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Callable
from typing import Any


def _default_key(service_name: str) -> str:
    return f"service:{service_name}:heartbeat"


def publish_heartbeat(
    service_name: str,
    *,
    ttl_s: int = 120,
    key: str | None = None,
    extra: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Publish one heartbeat to PostgreSQL service_state and Valkey.

    Heartbeats are best-effort. They must never crash a service because their
    job is observability, not business logic.
    """
    payload: dict[str, Any] = {
        "ts": time.time(),
        "service": service_name,
        "pid": os.getpid(),
        "host": socket.gethostname(),
    }
    if extra:
        try:
            payload.update(extra() or {})
        except Exception as exc:
            payload["extra_error"] = str(exc)

    heartbeat_key = key or _default_key(service_name)

    try:
        from agent.service_state import set_state

        set_state(heartbeat_key, payload, ttl_s=ttl_s)
    except Exception:
        pass

    try:
        from agent.valkey_client import _get_client

        client = _get_client()
        if client:
            client.setex(heartbeat_key, ttl_s, json.dumps(payload, default=str))
    except Exception:
        pass

    return payload


def start_service_heartbeat(
    service_name: str,
    runner: object,
    *,
    interval_s: int = 30,
    ttl_s: int = 120,
    key: str | None = None,
    extra: Callable[[], dict[str, Any]] | None = None,
) -> threading.Thread:
    """Start a daemon thread that publishes service heartbeats."""

    def _loop() -> None:
        stop_event = getattr(runner, "_stop", None)
        while not bool(getattr(runner, "stopped", False)):
            publish_heartbeat(service_name, ttl_s=ttl_s, key=key, extra=extra)
            if stop_event is not None:
                if stop_event.wait(interval_s):
                    break
            else:
                time.sleep(interval_s)

    thread = threading.Thread(
        target=_loop,
        daemon=True,
        name=f"{service_name}-heartbeat",
    )
    thread.start()
    return thread
