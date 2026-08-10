"""Compact Valkey handoff for one-second dashboard quote projection."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)
_KEY = "scalp:live_indicator_states:v1"
_STATE_FIELDS = (
    "ticker", "indicator_close", "macd_hist",
    "rsi_avg_gain_14", "rsi_avg_loss_14",
    "rsi_avg_gain_7", "rsi_avg_loss_7",
    "rsi_avg_gain_2", "rsi_avg_loss_2",
    "macd_fast_ema", "macd_slow_ema", "macd_signal_ema",
)


def publish_live_indicator_states(
    plans: dict[str, Any], *, scan_ts: float, session: str
) -> bool:
    try:
        from agent.valkey_client import _get_client

        client = _get_client()
        if client is None:
            return False
        rows = []
        for ticker, plan in plans.items():
            row = {field: getattr(plan, field, None) for field in _STATE_FIELDS}
            row["ticker"] = str(ticker).upper()
            rows.append(row)
        payload = {
            "schema_version": 1,
            "scan_ts": float(scan_ts),
            "session": str(session or "UNKNOWN").upper(),
            "plans": rows,
        }
        client.setex(_KEY, 90, json.dumps(payload, separators=(",", ":"), default=str))
        return True
    except Exception as exc:
        logger.warning("[ScalpLiveFeed] publish failed: %s", exc)
        return False


def read_live_indicator_states() -> dict[str, Any] | None:
    try:
        from agent.valkey_client import _get_client

        client = _get_client()
        if client is None:
            return None
        raw = client.get(_KEY)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw) if raw else None
        if not isinstance(payload, dict):
            return None
        if time.time() - float(payload.get("scan_ts") or 0.0) > 120:
            return None
        return payload
    except Exception:
        return None
