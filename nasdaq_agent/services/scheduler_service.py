#!/usr/bin/env python3
"""
scheduler_service — time-driven tasks (EOD close, heartbeat, maintenance).

Responsibilities:
  1. EOD watchdog: force-close all paper-trade positions at 3:45 PM ET on
     market days, regardless of scanner state.
  2. Heartbeat: write a liveness key to Valkey every 30 s so external health
     monitors can detect a dead container.
  3. Extensible: add new scheduled tasks here, not in the scan loop.

Interface contract:
  WRITES  PostgreSQL  — paper-trade EOD close records
  WRITES  Valkey      — heartbeat key (scheduler:heartbeat)
  READS   nothing beyond system clock and market-hours API

Environment variables:
  LOG_LEVEL  DEBUG|INFO|WARNING  (default INFO)
  All DB / Valkey env vars inherited from environment / .env
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv
load_dotenv()

from services._base import configure_logging, ServiceRunner

_log    = configure_logging("scheduler")
_runner = ServiceRunner("scheduler")

_HEARTBEAT_KEY      = "scheduler:heartbeat"
_HEARTBEAT_INTERVAL = int(os.getenv("SCHEDULER_HEARTBEAT_S", "30"))

# EOD watchdog fires at this window: [15:45, 16:00) ET on weekdays
_EOD_HM_START = 945   # 15*60 + 45
_EOD_HM_END   = 960   # 16*60 + 00


# ── EOD watchdog ──────────────────────────────────────────────────────────────

def _eod_watchdog_loop() -> None:
    """
    Poll the clock every 30 s.  Once per calendar day, when time falls in
    [15:45, 16:00) ET on a weekday, force-close all open paper positions.
    """
    import zoneinfo
    from datetime import datetime

    fired_on: set = set()
    tz_et = zoneinfo.ZoneInfo("America/New_York")

    while not _runner.stopped:
        try:
            now_et = datetime.now(tz_et)
            hm     = now_et.hour * 60 + now_et.minute
            today  = now_et.date()

            if (now_et.weekday() < 5
                    and _EOD_HM_START <= hm < _EOD_HM_END
                    and today not in fired_on):
                fired_on.add(today)
                try:
                    from agent.paper_trading import close_all_positions_eod
                    n = close_all_positions_eod(reason="EOD_WATCHDOG_3:45PM")
                    if n:
                        _log.warning("EOD: force-closed %d position(s) at 3:45 PM ET", n)
                    else:
                        _log.info("EOD: no open positions to close")
                except Exception as exc:
                    _log.error("EOD close failed: %s", exc, exc_info=True)

            # Prune fired_on to avoid unbounded growth
            if len(fired_on) > 10:
                fired_on = set(sorted(fired_on)[-5:])

        except Exception:
            pass

        _runner._stop.wait(_HEARTBEAT_INTERVAL)


# ── Heartbeat ─────────────────────────────────────────────────────────────────

def _heartbeat_loop() -> None:
    """Write a timestamp to Valkey every SCHEDULER_HEARTBEAT_S seconds."""
    import json

    while not _runner.stopped:
        try:
            from agent.valkey_client import _get_client
            client = _get_client()
            if client:
                payload = json.dumps({"ts": time.time(), "service": "scheduler"})
                client.setex(_HEARTBEAT_KEY, _HEARTBEAT_INTERVAL * 3, payload)
        except Exception as exc:
            _log.debug("Heartbeat write failed: %s", exc)

        _runner._stop.wait(_HEARTBEAT_INTERVAL)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import threading

    _log.info("=== scheduler_service starting ===")

    threading.Thread(
        target=_eod_watchdog_loop, daemon=True, name="eod-watchdog"
    ).start()
    _log.info("EOD watchdog started — fires at 3:45 PM ET on weekdays")

    threading.Thread(
        target=_heartbeat_loop, daemon=True, name="scheduler-heartbeat"
    ).start()
    _log.info("Heartbeat loop started (interval=%ds)", _HEARTBEAT_INTERVAL)

    _log.info("Scheduler running — waiting for SIGTERM/SIGINT …")
    _runner.register_signals()
    _runner.wait()
    _log.info("=== scheduler_service stopped ===")


if __name__ == "__main__":
    main()
