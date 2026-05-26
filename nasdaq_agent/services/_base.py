"""
Shared bootstrap utilities for all service entry points.

Each service imports configure_logging() and ServiceRunner from here.
No business logic — only cross-cutting wiring that every service needs.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from typing import Callable


def configure_logging(service_name: str) -> logging.Logger:
    """Set up structured logging for a service process."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s [%(levelname)s] {service_name} %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
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
