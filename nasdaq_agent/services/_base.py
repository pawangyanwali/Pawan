"""
Shared bootstrap utilities for all service entry points.

Each service imports configure_logging() and ServiceRunner from here.
No business logic — only cross-cutting wiring that every service needs.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Callable

# Log files land in /app/logs/ which is bind-mounted to /opt/nasdaq-agent/logs
# on the host, so all service logs are accessible in one place.
_LOG_DIR = Path(os.getenv("LOG_DIR", "/app/logs"))
_LOG_MAX_BYTES  = int(os.getenv("LOG_MAX_BYTES",  str(50 * 1024 * 1024)))  # 50 MB
_LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))


def configure_logging(service_name: str) -> logging.Logger:
    """
    Configure logging for a service process.

    Writes to two destinations:
      stdout            — visible via 'docker compose logs' (captured by Docker)
      /app/logs/<name>.log — rotating file, survives container restarts,
                             bind-mounted to /opt/nasdaq-agent/logs on the host
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt   = f"%(asctime)s [%(levelname)s] {service_name} %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # Stdout handler — keeps 'docker compose logs' working
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(stdout_handler)

    # Rotating file handler — writes to bind-mounted host directory
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_DIR / f"{service_name}.log",
            maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(fmt, datefmt))
        root.addHandler(file_handler)
    except Exception as exc:
        # Non-fatal — fall back to stdout-only if log dir isn't writable
        logging.getLogger(service_name).warning(
            "Could not open log file in %s: %s — logging to stdout only", _LOG_DIR, exc
        )

    return logging.getLogger(service_name)


def ensure_sys_path() -> None:
    """Ensure the nasdaq_agent directory is importable (works from any cwd)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)


class ServiceRunner:
    """
    Minimal lifecycle manager shared by all service entry points.

    Usage:
        runner = ServiceRunner("scanner")
        runner.start(start_fn)   # start_fn() is called once
        runner.wait()            # blocks until SIGTERM / SIGINT
        runner.stop(stop_fn)     # stop_fn() is called on shutdown
    """

    def __init__(self, name: str) -> None:
        self.name  = name
        self._stop = threading.Event()

    def register_signals(self, on_stop: Callable[[], None] | None = None) -> None:
        def _handler(signum: int, frame: object) -> None:
            logging.getLogger(self.name).info(
                "Signal %d received — shutting down …", signum
            )
            self._stop.set()
            if on_stop:
                try:
                    on_stop()
                except Exception as exc:
                    logging.getLogger(self.name).warning("Stop callback error: %s", exc)

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT,  _handler)

    def wait(self) -> None:
        """Block the main thread until a stop signal is received."""
        self._stop.wait()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()



def ensure_sys_path() -> None:
    """Ensure the nasdaq_agent directory is importable (works from any cwd)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)


class ServiceRunner:
    """
    Minimal lifecycle manager shared by all service entry points.

    Usage:
        runner = ServiceRunner("scanner")
        runner.start(start_fn)   # start_fn() is called once
        runner.wait()            # blocks until SIGTERM / SIGINT
        runner.stop(stop_fn)     # stop_fn() is called on shutdown
    """

    def __init__(self, name: str) -> None:
        self.name  = name
        self._stop = threading.Event()

    def register_signals(self, on_stop: Callable[[], None] | None = None) -> None:
        def _handler(signum: int, frame: object) -> None:
            logging.getLogger(self.name).info(
                "Signal %d received — shutting down …", signum
            )
            self._stop.set()
            if on_stop:
                try:
                    on_stop()
                except Exception as exc:
                    logging.getLogger(self.name).warning("Stop callback error: %s", exc)

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT,  _handler)

    def wait(self) -> None:
        """Block the main thread until a stop signal is received."""
        self._stop.wait()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()
