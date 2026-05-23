"""
After-hours and pre-market monitor.

During extended sessions the scanner still fetches price data, but instead of
generating trade signals (which are suppressed) we use that data to build an
"opening bias" for each ticker: which direction is it likely to open, how
strongly, and how confident are we?

How it works
------------
1. Every scan during AFTER_HOURS or PRE_MARKET:
   - record_snapshot() is called for each ticker with its current AH price,
     the previous regular-session close, and volume metrics.
   - Snapshots accumulate across the session so we always have the *latest*
     AH picture (last write wins per ticker).

2. At market OPEN / POWER_HOUR / AVOID_ZONE:
   - get_opening_bias(ticker) returns a dict describing the expected open:
       direction:   BULLISH | BEARISH | NEUTRAL
       ah_change_pct: float (AH move from previous close, e.g. +2.3)
       magnitude:   STRONG | MODERATE | WEAK
       confidence_adj: float (±points to add to prediction confidence)
       gap_estimate_pct: float (expected gap magnitude)
   - This is read by scanner.py and added to the StockSignal for UI display
     and fed into prediction.py for confidence adjustment.

3. After each trade closes, the AH bias that was active at open is compared
   with the actual outcome and fed into the adaptive filter as the "ah_bias"
   context dimension, so the system learns which AH patterns actually predict
   winning trades.

4. Snapshots are stored in SQLite so they survive scanner restarts and can be
   reviewed in the learning dashboard.

AH magnitude thresholds
-----------------------
  STRONG   : |ah_change_pct| >= 2.0  (significant move, likely news/earnings)
  MODERATE : |ah_change_pct| >= 0.75 (notable drift)
  WEAK     : |ah_change_pct| <  0.75 (noise / mean reversion likely)

Confidence adjustment
---------------------
  STRONG  BULLISH  → +10 pts   BEARISH → +10 pts (strong signal, either dir)
  MODERATE BULLISH → +5  pts   BEARISH → +5  pts
  WEAK     NEUTRAL → 0   pts (don't adjust — too much noise)

  When AH direction confirms signal direction  → positive adjustment
  When AH direction opposes signal direction   → negative adjustment (applied in scanner)
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "ah_snapshots.db"
_lock    = threading.Lock()

# Thresholds for AH move magnitude
_STRONG_THRESH   = 2.0    # % move
_MODERATE_THRESH = 0.75   # % move

# Confidence adjustments (applied in scanner when AH confirms signal direction)
_CONF_ADJ = {
    "STRONG":   10.0,
    "MODERATE":  5.0,
    "WEAK":      0.0,
}

# AH data expires after 14 hours (one trading day)
_SNAPSHOT_TTL_HOURS = 14


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH), timeout=10)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS ah_snapshots (
                ticker          TEXT    PRIMARY KEY,
                recorded_at     TEXT    NOT NULL,
                session         TEXT    NOT NULL,
                prev_close      REAL    NOT NULL,
                ah_price        REAL    NOT NULL,
                ah_change_pct   REAL    NOT NULL,
                ah_volume       REAL    DEFAULT 0,
                avg_volume      REAL    DEFAULT 0,
                ah_volume_ratio REAL    DEFAULT 0,
                direction       TEXT    NOT NULL,
                magnitude       TEXT    NOT NULL,
                gap_estimate_pct REAL   DEFAULT 0,
                news_likely     INTEGER DEFAULT 0
            )
        """)
        c.commit()


def record_snapshot(
    ticker:      str,
    ah_price:    float,
    prev_close:  float,
    ah_volume:   float = 0.0,
    avg_volume:  float = 0.0,
    session:     str   = "AFTER_HOURS",
) -> None:
    """
    Record or update the latest AH/pre-market snapshot for a ticker.
    Called every scan cycle during AFTER_HOURS and PRE_MARKET sessions.
    """
    if prev_close <= 0 or ah_price <= 0:
        return

    ah_change_pct = round((ah_price - prev_close) / prev_close * 100, 3)
    ah_vol_ratio  = round(ah_volume / avg_volume, 2) if avg_volume > 0 else 0.0

    direction = (
        "BULLISH" if ah_change_pct > 0.1
        else "BEARISH" if ah_change_pct < -0.1
        else "NEUTRAL"
    )

    abs_chg = abs(ah_change_pct)
    if abs_chg >= _STRONG_THRESH:
        magnitude = "STRONG"
    elif abs_chg >= _MODERATE_THRESH:
        magnitude = "MODERATE"
    else:
        magnitude = "WEAK"

    # News likely when large move AND elevated volume
    news_likely = int(abs_chg >= 2.0 and ah_vol_ratio >= 1.5)

    # Gap estimate: AH move usually partially closes at open (70% realisation)
    gap_estimate = round(ah_change_pct * 0.70, 3)

    with _lock:
        with _conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO ah_snapshots
                  (ticker, recorded_at, session, prev_close, ah_price,
                   ah_change_pct, ah_volume, avg_volume, ah_volume_ratio,
                   direction, magnitude, gap_estimate_pct, news_likely)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                ticker,
                datetime.now(timezone.utc).isoformat(),
                session,
                round(prev_close, 4),
                round(ah_price, 4),
                ah_change_pct,
                round(ah_volume, 0),
                round(avg_volume, 0),
                ah_vol_ratio,
                direction,
                magnitude,
                gap_estimate,
                news_likely,
            ))
            c.commit()

    logger.debug(
        f"[AH] {ticker} {direction} {ah_change_pct:+.2f}% "
        f"({magnitude}) vol×{ah_vol_ratio:.1f}  news_likely={bool(news_likely)}"
    )


def get_opening_bias(ticker: str) -> dict:
    """
    Return the latest AH context for a ticker to be used at market open.
    Returns a dict with direction, magnitude, ah_change_pct, confidence_adj,
    gap_estimate_pct, news_likely. Returns empty dict if no fresh data.
    """
    with _lock:
        with _conn() as c:
            row = c.execute("""
                SELECT * FROM ah_snapshots WHERE ticker=?
            """, (ticker,)).fetchone()

    if row is None:
        return {}

    # Discard stale snapshots (older than TTL)
    try:
        recorded = datetime.fromisoformat(row["recorded_at"])
        age_hours = (datetime.now(timezone.utc) - recorded).total_seconds() / 3600
        if age_hours > _SNAPSHOT_TTL_HOURS:
            return {}
    except Exception:
        return {}

    direction  = row["direction"]
    magnitude  = row["magnitude"]
    base_adj   = _CONF_ADJ.get(magnitude, 0.0)

    # Confidence adjustment: will be applied as +base_adj when AH confirms signal,
    # −base_adj when AH opposes signal (scanner decides which)
    return {
        "direction":        direction,
        "magnitude":        magnitude,
        "ah_change_pct":    float(row["ah_change_pct"]),
        "ah_volume_ratio":  float(row["ah_volume_ratio"]),
        "gap_estimate_pct": float(row["gap_estimate_pct"]),
        "news_likely":      bool(row["news_likely"]),
        "confidence_adj":   base_adj,
        "session_recorded": row["session"],
        "recorded_at":      row["recorded_at"],
    }


def get_all_biases() -> list[dict]:
    """Return all current AH snapshots — used by the dashboard API."""
    init_db()
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT * FROM ah_snapshots
                ORDER BY ABS(ah_change_pct) DESC
            """).fetchall()
    return [dict(r) for r in rows]


def _magnitude_label(chg: float) -> str:
    a = abs(chg)
    if a >= _STRONG_THRESH:   return "STRONG"
    if a >= _MODERATE_THRESH: return "MODERATE"
    return "WEAK"


def get_ah_context_key(ticker: str) -> str:
    """
    Return a single context key for the adaptive filter dimension 'ah_bias'.
    e.g. 'BULLISH_STRONG', 'BEARISH_MODERATE', 'NEUTRAL_WEAK'
    Used as a learnable context: if 'ah_bias:BULLISH_STRONG' has 80% win rate
    when signal direction matches, the adaptive filter boosts confidence.
    """
    bias = get_opening_bias(ticker)
    if not bias:
        return ""
    return f"{bias['direction']}_{bias['magnitude']}"
