"""
context_store.py — PostgreSQL persistence layer for context intelligence.

Tables
------
  context_events           raw scored news/events, deduped by content_hash
  ticker_context_features  rolling sentiment windows + derived flags per ticker
  earnings_calendar        normalized earnings dates from Finnhub

Pattern
-------
Follows service_state.py: lazy auto-init via _ensure_init(), init_db() is
idempotent, every public function swallows its own exceptions and returns a
safe default.

DB interface note (from agent/db.py):
  - Use agent.db.get_conn() as a context manager.
  - Placeholders are "?" (translated to "%s" by db.py automatically).
  - Do NOT call conn.cursor() — use conn.execute(sql, params) directly.
  - JSONB columns require an explicit cast: CAST(? AS jsonb).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Lazy init ─────────────────────────────────────────────────────────────────
_init_lock = threading.Lock()
_db_ready  = False


def _ensure_init() -> None:
    global _db_ready
    if _db_ready:
        return
    with _init_lock:
        if not _db_ready:
            _db_ready = init_db()


# ── DDL ───────────────────────────────────────────────────────────────────────

_DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS context_events (
        id              BIGSERIAL    PRIMARY KEY,
        ticker          TEXT         NOT NULL,
        content_hash    TEXT         NOT NULL,
        headline        TEXT         NOT NULL,
        source          TEXT         NOT NULL DEFAULT '',
        url             TEXT         NOT NULL DEFAULT '',
        published_at    TIMESTAMPTZ  NOT NULL,
        sentiment_score REAL         NOT NULL DEFAULT 0.0,
        credibility     REAL         NOT NULL DEFAULT 0.70,
        is_market_wide  BOOLEAN      NOT NULL DEFAULT FALSE,
        ingested_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        CONSTRAINT context_events_hash_uq UNIQUE (content_hash)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ticker_context_features (
        ticker                TEXT        PRIMARY KEY,
        sentiment_5m          REAL        NOT NULL DEFAULT 0.0,
        sentiment_30m         REAL        NOT NULL DEFAULT 0.0,
        sentiment_2h          REAL        NOT NULL DEFAULT 0.0,
        sentiment_1d          REAL        NOT NULL DEFAULT 0.0,
        sentiment_velocity    REAL        NOT NULL DEFAULT 0.0,
        news_count_30m        INT         NOT NULL DEFAULT 0,
        news_count_1d         INT         NOT NULL DEFAULT 0,
        news_shock            BOOLEAN     NOT NULL DEFAULT FALSE,
        context_risk_score    REAL        NOT NULL DEFAULT 0.0,
        recent_headlines      JSONB       NOT NULL DEFAULT '[]'::jsonb,
        computed_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS earnings_calendar (
        ticker          TEXT        NOT NULL,
        report_ts       TIMESTAMPTZ NOT NULL,
        hour            TEXT        NOT NULL DEFAULT '',
        eps_estimate    REAL        NULL,
        eps_actual      DOUBLE PRECISION NULL,
        rev_estimate    REAL        NULL,
        source          TEXT        NOT NULL DEFAULT 'finnhub',
        fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT earnings_calendar_pk PRIMARY KEY (ticker, report_ts)
    )
    """,
    # Migration: add eps_actual to existing deployments
    "ALTER TABLE earnings_calendar ADD COLUMN IF NOT EXISTS eps_actual DOUBLE PRECISION NULL",
    "CREATE INDEX IF NOT EXISTS idx_ctx_events_hash ON context_events (content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_ctx_events_ticker_ts ON context_events (ticker, published_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_earnings_ticker_ts ON earnings_calendar (ticker, report_ts)",
]


def init_db() -> bool:
    """
    Create context intel tables and indexes.  Idempotent.
    Returns True on success, False on DB unavailability (retried next call).
    """
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            for stmt in _DDL_STATEMENTS:
                stmt = stmt.strip()
                if stmt:
                    try:
                        conn.execute(stmt)
                    except Exception as ddl_err:
                        # Log but continue — most DDL errors are "already exists"
                        logger.debug("[context_store] DDL note: %s", ddl_err)
        logger.info("[context_store] Tables ready")
        return True
    except Exception as exc:
        logger.warning("[context_store] init_db failed (will retry): %s", exc)
        return False


# ── Scoring constants ─────────────────────────────────────────────────────────

_BULLISH_KW = {
    "beats", "beat", "exceeds", "record", "upgrade", "buy", "outperform",
    "growth", "surge", "rally", "breakout", "profit", "strong", "positive",
    "bullish", "upside", "partnership", "contract", "approval", "dividend",
    "boost", "raised", "acceleration", "acquired", "acquisition", "wins",
}
_BEARISH_KW = {
    "misses", "miss", "downgrade", "sell", "underperform", "loss", "decline",
    "fall", "drop", "weak", "negative", "bearish", "lawsuit", "recall",
    "investigation", "layoff", "cut", "warning", "risk", "concern",
    "shortage", "halt", "suspended", "fraud", "subpoena", "default",
}

# Weighted credibility by source substring (case-insensitive match)
_SOURCE_CRED: dict[str, float] = {
    "reuters":       0.95,
    "bloomberg":     0.95,
    "cnbc":          0.90,
    "marketwatch":   0.85,
    "wall street":   0.90,
    "barron":        0.85,
    "seeking alpha": 0.75,
    "motley fool":   0.70,
    "benzinga":      0.72,
}
_DEFAULT_CRED = 0.70

# Exponential decay half-life
_HALF_LIFE_MIN = 30.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_headline(text: str) -> float:
    """Score headline text → float in [-1, +1].  TextBlob + keyword boost."""
    if not text:
        return 0.0
    try:
        from textblob import TextBlob
        polarity = TextBlob(text).sentiment.polarity
    except Exception:
        polarity = 0.0
    words    = set(re.findall(r'\b\w+\b', text.lower()))
    boost    = (len(words & _BULLISH_KW) - len(words & _BEARISH_KW)) * 0.15
    return max(-1.0, min(1.0, polarity + boost))


def _source_credibility(source: str) -> float:
    src_lo = source.lower()
    for kw, cred in _SOURCE_CRED.items():
        if kw in src_lo:
            return cred
    return _DEFAULT_CRED


def _content_hash(ticker: str, headline: str, published_ts: int) -> str:
    """
    Stable dedup key.  Buckets publish timestamp into 1-hour windows to
    survive minor timestamp drift across duplicate articles.
    """
    bucket = published_ts // 3600
    raw    = f"{ticker}|{headline.lower().strip()}|{bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _event_weight(cred: float, age_minutes: float) -> float:
    """Exponential-decay weight: credibility × exp(-age / half_life)."""
    return cred * math.exp(-age_minutes / _HALF_LIFE_MIN)


# ── Event ingestion ───────────────────────────────────────────────────────────

def ingest_events(events: list[dict]) -> int:
    """
    Ingest news/event dicts into context_events (deduped by content_hash).

    Required keys per event:
      ticker       : str  — ticker symbol or "MARKET" for market-wide
      headline     : str  — article title / headline text
      published_ts : int  — unix timestamp (seconds)

    Optional keys:
      source, url, is_market_wide

    Returns count of newly inserted rows.
    """
    _ensure_init()
    if not events:
        return 0

    inserted = 0
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            for ev in events:
                headline     = (ev.get("headline") or ev.get("title") or "").strip()
                if not headline:
                    continue
                ticker       = (ev.get("ticker") or "MARKET").upper().strip()
                source       = ev.get("source", "")
                url          = ev.get("url", "")
                published_ts = int(ev.get("published_ts") or ev.get("datetime") or time.time())
                is_market    = bool(ev.get("is_market_wide", False))
                chash        = _content_hash(ticker, headline, published_ts)
                score        = _score_headline(headline)
                cred         = _source_credibility(source)
                pub_dt       = datetime.fromtimestamp(published_ts, tz=timezone.utc).isoformat()

                try:
                    conn.execute(
                        """
                        INSERT INTO context_events
                            (ticker, content_hash, headline, source, url,
                             published_at, sentiment_score, credibility, is_market_wide)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (content_hash) DO NOTHING
                        """,
                        (ticker, chash, headline, source, url,
                         pub_dt, score, cred, is_market),
                    )
                    inserted += 1
                except Exception:
                    pass   # duplicate / constraint violation — expected
    except Exception as exc:
        logger.warning("[context_store] ingest_events error: %s", exc)
    return inserted


# ── Market-news → ticker mapping ──────────────────────────────────────────────

def map_market_news_to_tickers(
    articles: list[dict],
    active_tickers: list[str],
) -> list[dict]:
    """
    Expand market-wide articles into per-ticker events by scanning headlines
    for ticker symbols.  Articles with no ticker mention are assigned to
    "MARKET" so they contribute to market-wide sentiment.

    Returns a flat list of event dicts ready for ingest_events().
    """
    ticker_set = {t.upper() for t in active_tickers}
    results: list[dict] = []

    for art in articles:
        headline = (art.get("headline") or art.get("summary") or "").upper()
        related  = (art.get("related") or "").upper()

        mentioned: set[str] = set()
        # Search headline + related field for word-boundary ticker matches
        for t in ticker_set:
            if re.search(rf'\b{re.escape(t)}\b', headline):
                mentioned.add(t)
        for t in re.findall(r'\b[A-Z]{1,5}\b', related):
            if t in ticker_set:
                mentioned.add(t)

        if mentioned:
            for t in mentioned:
                results.append({**art, "ticker": t, "is_market_wide": False})
        else:
            results.append({**art, "ticker": "MARKET", "is_market_wide": True})

    return results


# ── Rolling feature compute ───────────────────────────────────────────────────

def compute_features_for_ticker(ticker: str) -> dict:
    """
    Compute rolling context features for one ticker from recent context_events.

    Reads events for: this ticker + "MARKET" events, last 24 h.
    Returns feature dict with keys matching ticker_context_features columns.
    """
    _ensure_init()
    now_utc = datetime.now(timezone.utc)

    try:
        from agent.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT published_at, sentiment_score, credibility
                FROM context_events
                WHERE (ticker = ? OR ticker = 'MARKET')
                  AND published_at >= NOW() AT TIME ZONE 'UTC' - INTERVAL '24 hours'
                ORDER BY published_at DESC
                LIMIT 500
                """,
                (ticker,),
            ).fetchall()
    except Exception as exc:
        logger.debug("[context_store] compute_features(%s) DB error: %s", ticker, exc)
        return _empty_features(ticker)

    if not rows:
        return _empty_features(ticker)

    # Build (age_minutes, score, cred) tuples
    events: list[tuple[float, float, float]] = []
    for row in rows:
        try:
            pub = row["published_at"]
            if isinstance(pub, str):
                pub = datetime.fromisoformat(pub)
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            age_m = max(0.0, (now_utc - pub).total_seconds() / 60.0)
            events.append((age_m, float(row["sentiment_score"]), float(row["credibility"])))
        except Exception:
            continue

    if not events:
        return _empty_features(ticker)

    def _weighted_sent(max_age_m: float) -> float:
        relevant = [(a, s, c) for (a, s, c) in events if a <= max_age_m]
        if not relevant:
            return 0.0
        weights   = [_event_weight(c, a) for (a, _, c) in relevant]
        total_w   = sum(abs(w) for w in weights)
        if total_w == 0.0:
            return 0.0
        return sum(s * w for (_, s, _), w in zip(relevant, weights)) / total_w

    s_5m  = _weighted_sent(5.0)
    s_30m = _weighted_sent(30.0)
    s_2h  = _weighted_sent(120.0)
    s_1d  = _weighted_sent(1440.0)

    velocity   = s_30m - s_2h
    count_30m  = sum(1 for (a, _, _) in events if a <= 30.0)
    count_1d   = len(events)
    baseline   = max(1.0, count_1d / 48.0)          # expected 30-min rate
    news_shock = count_30m > 3 * baseline
    risk       = min(1.0, abs(s_30m) * 0.5 + (0.3 if news_shock else 0.0) + abs(velocity) * 0.2)

    # Retrieve recent headlines separately for the JSONB payload
    recent_hl: list[str] = []
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            hl_rows = conn.execute(
                """
                SELECT headline FROM context_events
                WHERE ticker = ?
                  AND published_at >= NOW() AT TIME ZONE 'UTC' - INTERVAL '24 hours'
                ORDER BY published_at DESC
                LIMIT 5
                """,
                (ticker,),
            ).fetchall()
        recent_hl = [r["headline"] for r in hl_rows]
    except Exception:
        pass

    return {
        "ticker":             ticker,
        "sentiment_5m":       round(s_5m,  4),
        "sentiment_30m":      round(s_30m, 4),
        "sentiment_2h":       round(s_2h,  4),
        "sentiment_1d":       round(s_1d,  4),
        "sentiment_velocity": round(velocity, 4),
        "news_count_30m":     count_30m,
        "news_count_1d":      count_1d,
        "news_shock":         news_shock,
        "context_risk_score": round(risk, 4),
        "recent_headlines":   json.dumps(recent_hl),
    }


def _empty_features(ticker: str) -> dict:
    return {
        "ticker":             ticker,
        "sentiment_5m":       0.0,
        "sentiment_30m":      0.0,
        "sentiment_2h":       0.0,
        "sentiment_1d":       0.0,
        "sentiment_velocity": 0.0,
        "news_count_30m":     0,
        "news_count_1d":      0,
        "news_shock":         False,
        "context_risk_score": 0.0,
        "recent_headlines":   "[]",
    }


def upsert_features(features: dict) -> None:
    """Upsert computed features into ticker_context_features."""
    _ensure_init()
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            conn.execute(
                """
                INSERT INTO ticker_context_features
                    (ticker, sentiment_5m, sentiment_30m, sentiment_2h, sentiment_1d,
                     sentiment_velocity, news_count_30m, news_count_1d,
                     news_shock, context_risk_score, recent_headlines, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS jsonb), NOW())
                ON CONFLICT (ticker) DO UPDATE SET
                    sentiment_5m       = EXCLUDED.sentiment_5m,
                    sentiment_30m      = EXCLUDED.sentiment_30m,
                    sentiment_2h       = EXCLUDED.sentiment_2h,
                    sentiment_1d       = EXCLUDED.sentiment_1d,
                    sentiment_velocity = EXCLUDED.sentiment_velocity,
                    news_count_30m     = EXCLUDED.news_count_30m,
                    news_count_1d      = EXCLUDED.news_count_1d,
                    news_shock         = EXCLUDED.news_shock,
                    context_risk_score = EXCLUDED.context_risk_score,
                    recent_headlines   = EXCLUDED.recent_headlines,
                    computed_at        = NOW()
                """,
                (
                    features["ticker"],
                    features["sentiment_5m"],
                    features["sentiment_30m"],
                    features["sentiment_2h"],
                    features["sentiment_1d"],
                    features["sentiment_velocity"],
                    features["news_count_30m"],
                    features["news_count_1d"],
                    features["news_shock"],
                    features["context_risk_score"],
                    features["recent_headlines"],
                ),
            )
    except Exception as exc:
        logger.warning("[context_store] upsert_features(%s) error: %s",
                       features.get("ticker"), exc)


def get_features(ticker: str) -> Optional[dict]:
    """
    Read ticker_context_features for a ticker.
    Returns None if the row doesn't exist or DB is unavailable.
    """
    _ensure_init()
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM ticker_context_features WHERE ticker = ?",
                (ticker,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        hl = result.get("recent_headlines")
        if isinstance(hl, str):
            try:
                result["recent_headlines"] = json.loads(hl)
            except Exception:
                result["recent_headlines"] = []
        elif hl is None:
            result["recent_headlines"] = []
        return result
    except Exception as exc:
        logger.debug("[context_store] get_features(%s) error: %s", ticker, exc)
        return None


def get_tickers_with_events(max_age_hours: int = 24) -> list[str]:
    """
    Return list of distinct tickers that have events within the last N hours.
    Excludes the synthetic "MARKET" key.
    """
    _ensure_init()
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT ticker FROM context_events
                WHERE ticker != 'MARKET'
                  AND published_at >= NOW() AT TIME ZONE 'UTC' - (? * INTERVAL '1 hour')
                ORDER BY ticker
                """,
                (max_age_hours,),
            ).fetchall()
        return [r["ticker"] for r in rows]
    except Exception as exc:
        logger.debug("[context_store] get_tickers_with_events error: %s", exc)
        return []


# ── Earnings calendar ─────────────────────────────────────────────────────────

def upsert_earnings(entries: list[dict]) -> int:
    """
    Upsert a list of earnings calendar entries.

    Each entry must have:
      ticker     : str
      report_ts  : datetime (UTC-aware) or ISO string or unix int

    Optional:
      hour           : "bmo" | "amc" | "dmh"
      eps_estimate   : float | None
      eps_actual     : float | None   ← actual EPS reported (post-earnings)
      rev_estimate   : float | None

    Returns count of rows upserted.
    """
    _ensure_init()
    if not entries:
        return 0

    count = 0
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            for e in entries:
                ts = e.get("report_ts")
                if isinstance(ts, (int, float)):
                    ts = datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
                elif isinstance(ts, datetime):
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    ts = ts.isoformat()
                # ts is now an ISO string

                ticker = (e.get("ticker") or "").upper().strip()
                if not ticker or ts is None:
                    continue

                try:
                    conn.execute(
                        """
                        INSERT INTO earnings_calendar
                            (ticker, report_ts, hour, eps_estimate, eps_actual, rev_estimate, fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?, NOW())
                        ON CONFLICT (ticker, report_ts) DO UPDATE SET
                            hour         = EXCLUDED.hour,
                            eps_estimate = EXCLUDED.eps_estimate,
                            eps_actual   = COALESCE(EXCLUDED.eps_actual, earnings_calendar.eps_actual),
                            rev_estimate = EXCLUDED.rev_estimate,
                            fetched_at   = NOW()
                        """,
                        (
                            ticker,
                            ts,
                            e.get("hour", ""),
                            e.get("eps_estimate"),
                            e.get("eps_actual"),
                            e.get("rev_estimate"),
                        ),
                    )
                    count += 1
                except Exception as row_err:
                    logger.warning("[context_store] upsert_earnings row error: %s", row_err)
    except Exception as exc:
        logger.warning("[context_store] upsert_earnings error: %s", exc)
    return count


def get_next_earnings_from_db(ticker: str) -> Optional[datetime]:
    """
    Return the next upcoming earnings datetime (UTC-aware) for ticker.
    Looks up to 90 days ahead.  Returns None when unavailable.
    """
    _ensure_init()
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            row = conn.execute(
                """
                SELECT report_ts FROM earnings_calendar
                WHERE ticker = ?
                  AND report_ts >= NOW()
                  AND report_ts <= NOW() + INTERVAL '90 days'
                ORDER BY report_ts ASC
                LIMIT 1
                """,
                (ticker,),
            ).fetchone()
        if row is None:
            return None
        ts = row["report_ts"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts
    except Exception as exc:
        logger.debug("[context_store] get_next_earnings(%s) error: %s", ticker, exc)
        return None


def get_earnings_context(ticker: str) -> dict:
    """
    Return earnings context for *ticker* from the most recent calendar row.

    Returned dict always has:
      earnings_hour      : "bmo" | "amc" | "dmh" | ""
      eps_surprise_pct   : float   (actual/estimate − 1)*100, 0.0 if unavailable
      eps_beat           : bool    True when eps_actual > eps_estimate

    Used by context-intel service to populate Valkey payload fields.
    """
    _ensure_init()
    result = {"earnings_hour": "", "eps_surprise_pct": 0.0, "eps_beat": False}
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            # Pick the most recently fetched row (covers both past and future dates)
            row = conn.execute(
                """
                SELECT hour, eps_estimate, eps_actual
                FROM earnings_calendar
                WHERE ticker = ?
                ORDER BY fetched_at DESC
                LIMIT 1
                """,
                (ticker,),
            ).fetchone()
        if row is None:
            return result
        result["earnings_hour"] = (row["hour"] or "").lower().strip()
        est = row["eps_estimate"]
        act = row["eps_actual"]
        if est is not None and act is not None and est != 0.0:
            try:
                surprise = (float(act) / float(est) - 1.0) * 100.0
                result["eps_surprise_pct"] = round(surprise, 2)
                result["eps_beat"] = float(act) > float(est)
            except (ZeroDivisionError, TypeError, ValueError):
                pass
    except Exception as exc:
        logger.debug("[context_store] get_earnings_context(%s) error: %s", ticker, exc)
    return result


def get_recent_news_for_ticker(ticker: str, max_items: int = 10) -> list[dict]:
    """
    Return the most recent context_events for a ticker, formatted for
    data_fetcher.fetch_news() compatibility:
      [{"title": str, "source": str, "url": str, "published": str}, ...]
    """
    _ensure_init()
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT headline, source, url, published_at
                FROM context_events
                WHERE ticker = ?
                ORDER BY published_at DESC
                LIMIT ?
                """,
                (ticker, max_items),
            ).fetchall()
        return [
            {
                "title":     r["headline"],
                "source":    r["source"],
                "url":       r["url"],
                "published": str(r["published_at"]),
            }
            for r in rows
        ]
    except Exception as exc:
        logger.debug("[context_store] get_recent_news(%s) error: %s", ticker, exc)
        return []
