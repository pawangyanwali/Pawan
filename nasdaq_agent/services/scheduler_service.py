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

# Full-session shadow diagnostics run after after-hours ends.  This is separate
# from the 15:45 EOD position close because shadow analysis should include the
# complete pre-market, regular, and after-hours scalp window.
_SHADOW_REPORT_HM = int(os.getenv("SCALP_SHADOW_REPORT_HM", str(20 * 60 + 15)))
_SHADOW_REPORT_BACKFILL_DAYS = int(os.getenv("SCALP_SHADOW_REPORT_BACKFILL_DAYS", "7"))


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
    """
    Write a liveness timestamp every SCHEDULER_HEARTBEAT_S seconds.

    Write order (both best-effort — a failure must never crash the scheduler):
      1. PostgreSQL service_state  — durable, expires_at = NOW() + 90s
      2. Valkey SETEX              — fast-path cache, TTL = 90s (backward compat)
    """
    import json

    _TTL = _HEARTBEAT_INTERVAL * 3   # 90s — matches docker-compose healthcheck

    while not _runner.stopped:
        payload = {"ts": time.time(), "service": "scheduler"}

        # ── 1. PostgreSQL (source of truth) ───────────────────────────────────
        try:
            from agent.service_state import set_state
            set_state(_HEARTBEAT_KEY, payload, ttl_s=_TTL)
        except Exception as exc:
            _log.debug("Heartbeat PostgreSQL write failed: %s", exc)

        # ── 2. Valkey (fast-path cache) ───────────────────────────────────────
        try:
            from agent.valkey_client import _get_client
            client = _get_client()
            if client:
                client.setex(_HEARTBEAT_KEY, _TTL, json.dumps(payload))
        except Exception as exc:
            _log.debug("Heartbeat Valkey write failed: %s", exc)

        _runner._stop.wait(_HEARTBEAT_INTERVAL)


def _shadow_report_loop() -> None:
    """Generate durable daily diagnostics for isolated shadow execution."""
    import zoneinfo
    from datetime import datetime

    fired_on: set = set()
    tz_et = zoneinfo.ZoneInfo("America/New_York")

    try:
        from agent.scalp.shadow_report import generate_recent_shadow_reports

        reports = generate_recent_shadow_reports(_SHADOW_REPORT_BACKFILL_DAYS)
        _log.info("Shadow diagnostics backfilled for %d day(s)", len(reports))
    except Exception as exc:
        _log.warning("Shadow diagnostics backfill failed: %s", exc, exc_info=True)

    while not _runner.stopped:
        try:
            now_et = datetime.now(tz_et)
            hm = now_et.hour * 60 + now_et.minute
            today = now_et.date()
            if now_et.weekday() < 5 and hm >= _SHADOW_REPORT_HM and today not in fired_on:
                fired_on.add(today)
                from agent.scalp.shadow_report import generate_shadow_daily_report

                report = generate_shadow_daily_report(today)
                summary = report.get("summary") or {}
                _log.info(
                    "Shadow diagnostics %s: %s | %s closed | %.3fR EV | $%.2f",
                    report.get("market_date"),
                    report.get("status"),
                    int(summary.get("closed") or 0),
                    float(summary.get("expectancy_r") or 0.0),
                    float(summary.get("pnl_dollar") or 0.0),
                )
            if len(fired_on) > 10:
                fired_on = set(sorted(fired_on)[-5:])
        except Exception as exc:
            _log.warning("Shadow diagnostics generation failed: %s", exc, exc_info=True)

        _runner._stop.wait(_HEARTBEAT_INTERVAL)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import threading
    from agent.service_heartbeat import start_service_heartbeat

    _log.info("=== scheduler_service starting ===")
    start_service_heartbeat("scheduler", _runner)

    threading.Thread(
        target=_eod_watchdog_loop, daemon=True, name="eod-watchdog"
    ).start()
    _log.info("EOD watchdog started — fires at 3:45 PM ET on weekdays")

    threading.Thread(
        target=_heartbeat_loop, daemon=True, name="scheduler-heartbeat"
    ).start()
    _log.info("Heartbeat loop started (interval=%ds)", _HEARTBEAT_INTERVAL)

    threading.Thread(
        target=_shadow_report_loop, daemon=True, name="shadow-diagnostics"
    ).start()
    _log.info(
        "Shadow diagnostics started — reports after %02d:%02d ET, backfill=%d day(s)",
        _SHADOW_REPORT_HM // 60,
        _SHADOW_REPORT_HM % 60,
        _SHADOW_REPORT_BACKFILL_DAYS,
    )

    _log.info("Scheduler running — waiting for SIGTERM/SIGINT …")
    _runner.register_signals()
    _runner.wait()
    _log.info("=== scheduler_service stopped ===")


if __name__ == "__main__":
    main()
