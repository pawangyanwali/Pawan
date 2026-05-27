"""
context_snapshot.py — Valkey publish/read for per-ticker context payloads.

Key layout
----------
  ctx:latest:{TICKER}   SETEX TTL=600s  — per-ticker context JSON
  ctx:market            SETEX TTL=600s  — market-wide context (not used by scanner)

Payload schema
--------------
  {
    "ticker":             str,
    "sentiment_5m":       float,
    "sentiment_30m":      float,     ← used by scanner as `sent`
    "sentiment_2h":       float,
    "sentiment_1d":       float,
    "sentiment_velocity": float,
    "news_count_30m":     int,
    "news_shock":         bool,
    "context_risk_score": float,
    "recent_headlines":   list[str], ← used by scanner as `headlines[:5]`
    "earnings_phase":     str,       ← "" | "blackout" | "caution" | "cooldown"
    "earnings_reason":    str,
    "earnings_next_date": str,       ← "May 28, 2026" or ""
    "earnings_days_away": int,       ← 999 = unknown
    "asof_ts":            float,     ← unix timestamp of last write
    "stale_age_s":        float,     ← seconds since asof_ts (computed on read)
  }

Scanner read path
-----------------
  1. Valkey  ctx:latest:{ticker}  — expected < 10 ms
  2. PostgreSQL ticker_context_features + earnings_calendar  (fallback)
  3. Safe empty defaults  (fallback of last resort)

The scanner always receives a complete dict even when the context-intel
service has never run (missing FINNHUB_API_KEY, first boot, etc.).
In that case sentiment fields are 0.0 and earnings_phase is ""; behavior
is identical to the pre-Phase-1 stub state.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_CTX_TTL_S   = 600           # 10-minute TTL — callers trust asof_ts / stale_age_s
_KEY_PREFIX  = "ctx:latest:"
_MARKET_KEY  = "ctx:market"

# Safe defaults returned when no context data exists
_EMPTY_PAYLOAD: dict = {
    "ticker":             "",
    "sentiment_5m":       0.0,
    "sentiment_30m":      0.0,
    "sentiment_2h":       0.0,
    "sentiment_1d":       0.0,
    "sentiment_velocity": 0.0,
    "news_count_30m":     0,
    "news_shock":         False,
    "context_risk_score": 0.0,
    "recent_headlines":   [],
    "earnings_phase":     "",
    "earnings_reason":    "",
    "earnings_next_date": "",
    "earnings_days_away": 999,
    "asof_ts":            0.0,
    "stale_age_s":        9999.0,
}


# ── Valkey access ─────────────────────────────────────────────────────────────

def _client():
    """Re-use the shared singleton from valkey_client without re-importing."""
    try:
        from agent.valkey_client import _get_client
        return _get_client()
    except Exception:
        return None


# ── Write (called by context-intel service every 30 s) ───────────────────────

def publish_context_snapshot(ticker: str, payload: dict) -> bool:
    """
    Publish a ticker context snapshot to Valkey (TTL = 600 s).

    payload must contain all keys from _EMPTY_PAYLOAD above; this function
    adds "asof_ts" automatically (caller should not set it).

    Returns True on success.
    """
    c = _client()
    if c is None:
        return False
    try:
        payload = {**payload, "asof_ts": time.time()}
        key     = f"{_KEY_PREFIX}{ticker.upper()}"
        c.setex(key, _CTX_TTL_S, json.dumps(payload, separators=(",", ":")))
        return True
    except Exception as exc:
        logger.debug("[context_snapshot] publish(%s) error: %s", ticker, exc)
        return False


# ── Read (called by scanner inside analyse_ticker — target < 10 ms) ──────────

def get_context_snapshot(ticker: str) -> dict:
    """
    Return the context snapshot for *ticker*.

    Read path:
      1. Valkey   ctx:latest:{ticker}                (hot path, ~1 ms)
      2. PostgreSQL  ticker_context_features / earnings_calendar  (fallback)
      3. Safe empty defaults                         (never raises)

    Always returns a fully populated dict (all keys from _EMPTY_PAYLOAD).
    """
    # 1. Hot path
    payload = _read_valkey(ticker)
    if payload is not None:
        payload["stale_age_s"] = round(time.time() - payload.get("asof_ts", 0.0), 1)
        return payload

    # 2. PostgreSQL fallback (context-intel ran but Valkey TTL expired)
    payload = _read_pg(ticker)
    if payload is not None:
        payload["stale_age_s"] = round(time.time() - payload.get("asof_ts", 0.0), 1)
        return payload

    # 3. Empty defaults (first boot / no context-intel service)
    default = dict(_EMPTY_PAYLOAD)
    default["ticker"] = ticker.upper()
    return default


def _read_valkey(ticker: str) -> Optional[dict]:
    c = _client()
    if c is None:
        return None
    try:
        raw = c.get(f"{_KEY_PREFIX}{ticker.upper()}")
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.debug("[context_snapshot] Valkey read(%s) error: %s", ticker, exc)
        return None


def _computed_at_to_unix(features: dict) -> float:
    """
    Convert computed_at from ticker_context_features to a unix timestamp.

    psycopg2 returns a datetime object; some SQLite drivers return a string.
    Returns 0.0 if the field is missing or unparseable — the caller's
    stale_age_s will then be very large, correctly flagging the row as stale.
    """
    val = features.get("computed_at")
    if val is None:
        return 0.0
    if hasattr(val, "timestamp"):               # datetime (psycopg2 / sqlite3 row)
        try:
            return val.timestamp()
        except Exception:
            return 0.0
    # String fallback: "2026-05-27T14:23:45+00:00" or "2026-05-27 14:23:45+00:00"
    try:
        from datetime import datetime, timezone
        s = str(val).strip().replace(" ", "T")
        if s.endswith("+00"):
            s += ":00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0


def _read_pg(ticker: str) -> Optional[dict]:
    """Assemble a context payload from PostgreSQL tables."""
    try:
        from agent.context_store import get_features, get_next_earnings_from_db
        from datetime import datetime, timezone

        features = get_features(ticker)
        if features is None:
            return None

        # Use the actual time the features were computed — not time.time().
        # time.time() would make stale PG rows appear fresh to the scanner.
        asof_ts = _computed_at_to_unix(features)

        # Earnings fields
        earnings_phase     = ""
        earnings_reason    = ""
        earnings_next_date = ""
        earnings_days_away = 999

        next_dt = get_next_earnings_from_db(ticker)
        if next_dt is not None:
            now  = datetime.now(timezone.utc)
            days = (next_dt - now).days
            earnings_days_away = days
            earnings_next_date = next_dt.strftime("%b %d, %Y")
            if 0 <= days <= 3:
                earnings_phase  = "blackout"
                earnings_reason = f"Earnings in {days}d ({next_dt.strftime('%b %d')}) — blackout"
            elif -1 <= days < 0:
                earnings_phase  = "cooldown"
                earnings_reason = "Post-earnings cooldown"
            elif days <= 7:
                earnings_phase  = "caution"
                earnings_reason = f"Earnings in {days}d — reduce size"

        hl = features.get("recent_headlines", [])
        if isinstance(hl, str):
            try:
                hl = json.loads(hl)
            except Exception:
                hl = []
        elif not isinstance(hl, list):
            hl = []

        return {
            "ticker":             ticker.upper(),
            "sentiment_5m":       float(features.get("sentiment_5m", 0.0)),
            "sentiment_30m":      float(features.get("sentiment_30m", 0.0)),
            "sentiment_2h":       float(features.get("sentiment_2h", 0.0)),
            "sentiment_1d":       float(features.get("sentiment_1d", 0.0)),
            "sentiment_velocity": float(features.get("sentiment_velocity", 0.0)),
            "news_count_30m":     int(features.get("news_count_30m", 0)),
            "news_shock":         bool(features.get("news_shock", False)),
            "context_risk_score": float(features.get("context_risk_score", 0.0)),
            "recent_headlines":   hl,
            "earnings_phase":     earnings_phase,
            "earnings_reason":    earnings_reason,
            "earnings_next_date": earnings_next_date,
            "earnings_days_away": earnings_days_away,
            "asof_ts":            asof_ts,
        }
    except Exception as exc:
        logger.debug("[context_snapshot] PG fallback(%s) error: %s", ticker, exc)
        return None


def build_payload_from_features(
    ticker: str,
    features: dict,
    earnings_phase: str     = "",
    earnings_reason: str    = "",
    earnings_next_date: str = "",
    earnings_days_away: int = 999,
) -> dict:
    """
    Helper used by the context-intel service to build a full payload before
    calling publish_context_snapshot().  Keeps the field assembly in one place.
    """
    hl = features.get("recent_headlines", [])
    if isinstance(hl, str):
        try:
            hl = json.loads(hl)
        except Exception:
            hl = []
    elif not isinstance(hl, list):
        hl = []

    return {
        "ticker":             ticker.upper(),
        "sentiment_5m":       float(features.get("sentiment_5m", 0.0)),
        "sentiment_30m":      float(features.get("sentiment_30m", 0.0)),
        "sentiment_2h":       float(features.get("sentiment_2h", 0.0)),
        "sentiment_1d":       float(features.get("sentiment_1d", 0.0)),
        "sentiment_velocity": float(features.get("sentiment_velocity", 0.0)),
        "news_count_30m":     int(features.get("news_count_30m", 0)),
        "news_shock":         bool(features.get("news_shock", False)),
        "context_risk_score": float(features.get("context_risk_score", 0.0)),
        "recent_headlines":   hl,
        "earnings_phase":     earnings_phase,
        "earnings_reason":    earnings_reason,
        "earnings_next_date": earnings_next_date,
        "earnings_days_away": earnings_days_away,
        # asof_ts is injected by publish_context_snapshot()
    }
