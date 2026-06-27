#!/usr/bin/env python3
"""
Docker watchdog for the single-host production deployment.

This service watches the critical application containers through the local
Docker Engine socket and restarts a container when it is stopped or remains
unhealthy for repeated checks. It intentionally does not own business logic;
its job is to shorten recovery time when Docker's own restart policy is not
enough to handle a hung process or a failing healthcheck.
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import sys
import time
from typing import Any

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv
load_dotenv()

from services._base import ServiceRunner, configure_logging

_log = configure_logging("watchdog")
_runner = ServiceRunner("watchdog")

_SOCKET_PATH = os.getenv("WATCHDOG_DOCKER_SOCKET", "/var/run/docker.sock")
_INTERVAL_S = max(10.0, float(os.getenv("WATCHDOG_INTERVAL_S", "30")))
_UNHEALTHY_STRIKES = max(1, int(os.getenv("WATCHDOG_UNHEALTHY_STRIKES", "2")))
_RESTART_TIMEOUT_S = max(5, int(os.getenv("WATCHDOG_RESTART_TIMEOUT_S", "10")))
_TARGET_SERVICES = {
    s.strip()
    for s in os.getenv(
        "WATCHDOG_SERVICES",
        "web-api,market-data,scalp-engine,scalp-learner,scheduler,context-intel",
    ).split(",")
    if s.strip()
}


class _UnixHTTPConnection(http.client.HTTPConnection):
    """Minimal HTTP-over-Unix-socket client for Docker Engine API."""

    def __init__(self, socket_path: str, timeout: float = 10.0) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.socket_path)
        self.sock = sock


def _docker_request(method: str, path: str, body: bytes | None = None) -> tuple[int, Any]:
    conn = _UnixHTTPConnection(_SOCKET_PATH)
    headers = {"Host": "docker"}
    if body is not None:
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body))
    conn.request(method, path, body=body, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    if not data:
        return resp.status, None
    try:
        return resp.status, json.loads(data.decode("utf-8"))
    except Exception:
        return resp.status, data.decode("utf-8", errors="replace")


def _compose_service(container: dict[str, Any]) -> str:
    labels = container.get("Labels") or {}
    service = labels.get("com.docker.compose.service")
    if service:
        return str(service)
    names = container.get("Names") or []
    if names:
        return str(names[0]).lstrip("/").split("-")[0]
    return ""


def _container_name(container: dict[str, Any]) -> str:
    names = container.get("Names") or []
    return str(names[0]).lstrip("/") if names else str(container.get("Id", ""))[:12]


def _list_target_containers() -> list[dict[str, Any]]:
    status, payload = _docker_request("GET", "/containers/json?all=1")
    if status >= 300 or not isinstance(payload, list):
        raise RuntimeError(f"Docker list failed: HTTP {status} {payload!r}")
    return [c for c in payload if _compose_service(c) in _TARGET_SERVICES]


def _inspect(container_id: str) -> dict[str, Any]:
    status, payload = _docker_request("GET", f"/containers/{container_id}/json")
    if status >= 300 or not isinstance(payload, dict):
        raise RuntimeError(f"Docker inspect failed for {container_id}: HTTP {status} {payload!r}")
    return payload


def _restart(container_id: str, reason: str, name: str) -> None:
    _log.warning("Restarting %s (%s)", name, reason)
    status, payload = _docker_request(
        "POST",
        f"/containers/{container_id}/restart?t={_RESTART_TIMEOUT_S}",
    )
    if status >= 300:
        raise RuntimeError(f"Docker restart failed for {name}: HTTP {status} {payload!r}")
    _log.info("Restart requested for %s", name)


def _state_of(container: dict[str, Any]) -> tuple[str, str]:
    state = container.get("State") or ""
    status = container.get("Status") or ""
    container_id = str(container.get("Id") or "")
    try:
        detail = _inspect(container_id)
        health = ((detail.get("State") or {}).get("Health") or {}).get("Status") or ""
        if health:
            return str(state), str(health)
    except Exception as exc:
        _log.debug("Inspect failed for %s: %s", _container_name(container), exc)
    return str(state), str(status)


def _watch_loop() -> None:
    strikes: dict[str, int] = {}
    _log.info(
        "Watchdog started for services: %s",
        ", ".join(sorted(_TARGET_SERVICES)),
    )

    while not _runner.stopped:
        try:
            seen: set[str] = set()
            for container in _list_target_containers():
                container_id = str(container.get("Id") or "")
                service = _compose_service(container)
                name = _container_name(container)
                seen.add(container_id)
                state, health = _state_of(container)

                if state != "running":
                    strikes[container_id] = 0
                    _restart(container_id, f"state={state or 'unknown'}", name)
                    continue

                if health == "unhealthy":
                    strikes[container_id] = strikes.get(container_id, 0) + 1
                    _log.warning(
                        "%s unhealthy strike %d/%d",
                        service,
                        strikes[container_id],
                        _UNHEALTHY_STRIKES,
                    )
                    if strikes[container_id] >= _UNHEALTHY_STRIKES:
                        strikes[container_id] = 0
                        _restart(container_id, "health=unhealthy", name)
                    continue

                strikes[container_id] = 0

            for container_id in list(strikes):
                if container_id not in seen:
                    strikes.pop(container_id, None)
        except Exception as exc:
            _log.warning("Watchdog cycle failed: %s", exc)

        _runner._stop.wait(_INTERVAL_S)


def main() -> int:
    if not os.path.exists(_SOCKET_PATH):
        _log.error("Docker socket not found at %s", _SOCKET_PATH)
        return 1

    from agent.service_heartbeat import start_service_heartbeat

    start_service_heartbeat("watchdog", _runner)
    _runner.register_signals()
    _watch_loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
