#!/usr/bin/env python3
"""
market-data container health check — called by Docker HEALTHCHECK.

Exit 0  healthy
Exit 1  unhealthy

Three-tier check (each tier implies the one before it):

  1. Valkey ping          — always; proves network + Valkey are reachable
  2. scanner:streamer age — always; proves the status-publisher loop is alive
                            (written every 15 s, TTL 60 s; stale after 90 s)
  3. md:prices freshness  — active sessions only (PRE_MARKET / REGULAR / AFTER_HOURS)
                            proves live quotes are actually flowing from Schwab
                            (updated_at must be < 90 s on at least one ticker)

During CLOSED sessions check 3 is skipped — there are no live quotes to flow.
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/app")


def main() -> int:
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

    # ── 2. status-publisher loop liveness ────────────────────────────────────
    # market_data_service writes scanner:streamer every 15 s.  If this key is
    # more than 90 s old, the publisher thread has died.
    try:
        raw = client.get("scanner:streamer")
        if raw:
            age = time.time() - json.loads(raw).get("ts", 0)
            if age > 90:
                print(f"FAIL: scanner:streamer stale ({age:.0f} s — publisher loop dead?)")
                return 1
        # Key absent → service just started; do not fail during start_period.
    except Exception as exc:
        print(f"WARN: scanner:streamer check error: {exc}")

    # ── 3. md:prices freshness (active sessions only) ─────────────────────────
    try:
        from agent.market_hours import get_market_session
        session = get_market_session()
    except Exception:
        session = "CLOSED"

    if session != "CLOSED":
        try:
            if client.hlen("md:prices") == 0:
                print(f"FAIL: md:prices empty during {session} session")
                return 1

            # Sample one random ticker to check updated_at freshness.
            # HRANDFIELD is available in Redis ≥ 6.2 / Valkey ≥ 7.
            try:
                ticker = client.hrandfield("md:prices")
            except Exception:
                # Fallback for older Redis: take the first key from HKEYS
                keys = client.hkeys("md:prices")
                ticker = keys[0] if keys else None

            if ticker:
                raw_q = client.hget("md:prices", ticker)
                if raw_q:
                    q   = json.loads(raw_q)
                    ts  = float(q.get("updated_at") or 0)
                    if ts > 0:
                        age = time.time() - ts
                        if age > 90:
                            print(
                                f"FAIL: md:prices stale ({age:.0f} s, ticker={ticker}) "
                                f"during {session} session"
                            )
                            return 1
        except Exception as exc:
            # Non-fatal — a Valkey read error here shouldn't hard-fail the container;
            # the ping check (step 1) already covers connectivity.
            print(f"WARN: md:prices freshness check error: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
