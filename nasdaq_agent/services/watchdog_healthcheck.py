#!/usr/bin/env python3
"""Docker health check for the watchdog container."""
from __future__ import annotations

import os
import socket
import sys


_SOCKET_PATH = os.getenv("WATCHDOG_DOCKER_SOCKET", "/var/run/docker.sock")


def main() -> int:
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(_SOCKET_PATH)
        sock.sendall(b"GET /_ping HTTP/1.1\r\nHost: docker\r\nConnection: close\r\n\r\n")
        data = sock.recv(256)
        sock.close()
        if b"200 OK" not in data:
            print(f"FAIL: Docker socket ping returned {data!r}")
            return 1
    except Exception as exc:
        print(f"FAIL: Docker socket unavailable: {exc}")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
