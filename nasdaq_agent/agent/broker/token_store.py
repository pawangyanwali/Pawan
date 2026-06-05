"""
Valkey-backed token store for Schwab OAuth tokens.

Sits on top of the existing disk + PostgreSQL (service_state) persistence.
Adds a fast Valkey key so consumers can load tokens without disk access
and survive EBS token-dir wipes.

Valkey key:  schwab:token:{app}  (JSON blob, TTL = 2 hours)
PostgreSQL:  via service_state.set_state() — already done inside _pg_save()

Only token-service writes here; all other containers read.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_VALKEY_PREFIX = "schwab:token:"
_VALKEY_TTL    = 7_200  # 2 hours — longer than the 30-min access token so we survive brief gaps


def put_token(app: str, token_data: dict) -> None:
    """Write token JSON to Valkey.  Best-effort — never raises."""
    payload = json.dumps({**token_data, "stored_at": token_data.get("stored_at", time.time())})
    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        if client:
            client.setex(f"{_VALKEY_PREFIX}{app}", _VALKEY_TTL, payload)
    except Exception as exc:
        logger.warning("[token_store] Valkey write failed for %s: %s", app, exc)


def delete_token(app: str) -> None:
    """Delete token JSON from Valkey. Best-effort - never raises."""
    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        if client:
            client.delete(f"{_VALKEY_PREFIX}{app}")
    except Exception as exc:
        logger.warning("[token_store] Valkey delete failed for %s: %s", app, exc)


def get_token(app: str) -> Optional[dict]:
    """
    Read token for `app` ('trader' or 'marketdata').

    Load order:
      1. Valkey  — fast path, written by token-service on every refresh
      2. PostgreSQL (service_state) — durable fallback, survives Valkey flush
    """
    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        if client:
            raw = client.get(f"{_VALKEY_PREFIX}{app}")
            if raw:
                data = json.loads(raw)
                if str(data.get("status", "OK")).upper() == "OK" and data.get("access_token"):
                    return data
    except Exception as exc:
        logger.warning("[token_store] Valkey read failed for %s: %s", app, exc)

    # Fallback: PostgreSQL (the _pg_load path that already exists in _TokenManager)
    try:
        from agent.service_state import get_state
        data = get_state(f"schwab:tokens:{app}", ignore_expiry=False)
        if (
            data
            and "access_token" in data
            and str(data.get("status", "OK")).upper() == "OK"
        ):
            return data
    except Exception as exc:
        logger.warning("[token_store] PG read failed for %s: %s", app, exc)

    return None
