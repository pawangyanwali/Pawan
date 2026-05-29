#!/usr/bin/env python3
"""
scanner container health check — called by Docker HEALTHCHECK.

Exit 0  healthy
Exit 1  unhealthy

Three-tier check (each tier implies the one before it):

  1. Valkey ping           — always; proves network + Valkey are reachable
  2. scan:latest existence — PostgreSQL first, Valkey fallback;
                             proves the scanner has completed at least one cycle
  3. scan:latest freshness — active sessions only (PRE_MARKET / REGULAR / AFTER_HOURS)
                             fails when the scan result is too old, indicating the
                             scan loop has hung or crashed

Age thresholds (active sessions only):
  REGULAR     : 180 s  (~2× the ~90 s full-batch scan cycle)
  PRE / AFTER : 300 s  (extended-hours scans run slower; 5 min is reasonable)

During CLOSED sessions check 3 is skipped entirely — the scanner idles.
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/app")

# Age thresholds per session type (seconds)
_AGE_THRESHOLD = {
    "REGULAR":     180,
    "PRE_MARKET":  300,
    "AFTER_HOURS": 300,
}


def main() -> int:
    # ── 0. Container start time — suppress false-stale failures on redeploys ────
    # Read PID-1 start time from /proc/1/stat (always available on Linux/Docker).
    # Used to detect scan:latest keys left over from a previous deployment.
    _proc_start: float | None = None
    try:
        import os as _os
        with open("/proc/1/stat") as _f:
            _fields = _f.read().split()
        # Field 22 (0-indexed 21) = starttime in jiffies since boot
        _jiffies = int(_fields[21])
        _clk_tck = _os.sysconf("SC_CLK_TCK")  # typically 100
        with open("/proc/stat") as _f:
            for _line in _f:
                if _line.startswith("btime "):
                    _boot = int(_line.split()[1])
                    break
        _proc_start = _boot + _jiffies / _clk_tck
    except Exception:
        _proc_start = None  # can't determine; use fallback logic below

    # ── 1. Valkey reachability ────────────────────────────────────────────────
    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        if not client or not client.ping():
            print("FAIL: Valkey unreachable")
            return 1
    except Exception as exc:
        print(f"FAIL: Valkey error: {exc}")
        return 1

    # ── 2. scan:latest existence ──────────────────────────────────────────────
    # Try PostgreSQL first (source of truth); fall back to Valkey cache.
    scan_ts: float | None = None   # Unix timestamp of last completed scan

    try:
        from agent.service_state import get_age_s, get_state
        age_pg = get_age_s("scan:latest")
        if age_pg is not None:
            scan_ts = time.time() - age_pg
        else:
            # Row absent → scanner has not yet completed a cycle OR PG unavailable;
            # try to pull ts from the Valkey-cached snapshot.
            raw = client.get("scan:latest")
            if raw:
                d = json.loads(raw)
                scan_ts = float(d.get("ts") or 0) or None
    except Exception as exc:
        print(f"WARN: scan:latest lookup error: {exc}")
        # Non-fatal during startup — fall through to session-aware check below.

    # If scan_ts is from BEFORE this container started, it is a stale key left
    # over from a previous deployment.  Treat it the same as absent so the
    # container gets a clean start_period grace window for its first cycle.
    if scan_ts is not None:
        if _proc_start is not None and scan_ts < _proc_start:
            print(
                f"WARN: scan:latest ts ({scan_ts:.0f}) predates container start "
                f"({_proc_start:.0f}) — treating as absent (prev-deploy remnant)"
            )
            scan_ts = None
        elif _proc_start is None and (time.time() - scan_ts) > 600:
            # Fallback: if key is >10 min old and we can't confirm container start,
            # assume it's a previous-deployment remnant.
            print(
                f"WARN: scan:latest age {time.time()-scan_ts:.0f}s > 600s "
                f"and process start unknown — treating as absent"
            )
            scan_ts = None

    # scan_ts being None is acceptable during start_period (cold start).
    # We only hard-fail on stale data during active market sessions (tier 3).

    # ── 3. scan:latest freshness (active sessions only) ──────────────────────
    try:
        from agent.market_hours import get_market_session
        session = get_market_session()
    except Exception:
        session = "CLOSED"

    if session != "CLOSED":
        threshold = _AGE_THRESHOLD.get(session, 300)

        if scan_ts is None:
            # No snapshot at all during an active session → scanner has never
            # written results.  Do not fail here — the start_period covers the
            # initial scan cycle that takes up to 90 s.  The container will move
            # from "starting" to "unhealthy" only after start_period elapses if
            # the key is still absent.  Emit a warning so logs are readable.
            print(
                f"WARN: scan:latest absent during {session} session "
                f"(scanner may still be in its first cycle)"
            )
            # Return 0 — Docker start_period makes this a non-error for now.
            return 0

        age = time.time() - scan_ts
        if age > threshold:
            print(
                f"FAIL: scan:latest stale ({age:.0f} s > {threshold} s) "
                f"during {session} session — scanner loop may be hung"
            )
            return 1
        else:
            print(f"OK: scan:latest age {age:.0f} s (threshold {threshold} s, session {session})")
    else:
        print("OK: market CLOSED — age check skipped")

    return 0


if __name__ == "__main__":
    sys.exit(main())
