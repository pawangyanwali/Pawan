"""
Live backtesting engine — tracks every signal from fire to resolution in real-time.

Architecture
------------
1. Record   — when a signal fires, write it to backtest_signals with status=TRACKING
2. Update   — every scan cycle, update_tracking() is called with the current price.
              Each TRACKING signal is evaluated bar by bar:
              - WIN      : price reaches target
              - LOSS     : price hits stop
              - VWAP_LOSS: price loses VWAP (for BUY) or reclaims it (for SELL)
              - TIMEOUT  : signal open > MAX_BARS without resolution
3. Report   — stats are computed on-demand via get_performance_stats()

This differs from paper_trading.py which only tracks high-confidence trades.
Here we track EVERY BUY/SELL signal regardless of confidence, so we can measure
accuracy by confidence band, session, regime, VWAP event, entry type, etc.

R-multiple tracking
-------------------
Each bar we record how far price moved in the favorable direction as R multiples.
  R = (price - entry) / (entry - stop)   for BUY
  R = (entry - price) / (stop - entry)   for SELL
Max R achieved during the trade is stored even if the trade ultimately ends as LOSS.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DB_PATH  = Path(__file__).parent.parent / "data" / "live_backtest.db"
_lock     = threading.Lock()
MAX_BARS  = 20     # TIMEOUT after N bars if not resolved
MIN_MOVE_TO_RECORD = 0.0   # record all signals (no minimum)


# ── Database setup ─────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH), timeout=10)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS bt_signals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id     TEXT UNIQUE NOT NULL,
                fired_at      TEXT NOT NULL,
                ticker        TEXT NOT NULL,
                direction     TEXT NOT NULL,
                entry_price   REAL NOT NULL,
                target        REAL NOT NULL,
                stop          REAL NOT NULL,
                rr_ratio      REAL DEFAULT 0,
                confidence    REAL DEFAULT 0,
                -- context at signal fire time
                session       TEXT DEFAULT '',
                regime        TEXT DEFAULT '',
                vwap_event    TEXT DEFAULT '',
                rsi_zone      TEXT DEFAULT '',
                rsi_value     REAL DEFAULT 50,
                sector_etf    TEXT DEFAULT '',
                sector_trend  TEXT DEFAULT '',
                entry_type    TEXT DEFAULT 'IMMEDIATE',
                mtf_alignment TEXT DEFAULT '',
                -- resolution
                status        TEXT DEFAULT 'TRACKING',
                resolved_at   TEXT,
                exit_price    REAL,
                exit_reason   TEXT,
                bars_tracked  INTEGER DEFAULT 0,
                max_favorable_r REAL DEFAULT 0,
                pnl_pct       REAL,
                r_multiple    REAL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS bt_price_path (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id   TEXT NOT NULL,
                bar         INTEGER NOT NULL,
                price       REAL NOT NULL,
                r_val       REAL NOT NULL,
                ts          TEXT NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_bt_ticker  ON bt_signals(ticker)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_bt_status  ON bt_signals(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_bt_fired   ON bt_signals(fired_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_path_sid   ON bt_price_path(signal_id)")
        c.commit()


# ── Record a new signal ────────────────────────────────────────────────────────

def record_signal(
    ticker:       str,
    direction:    str,
    entry_price:  float,
    target:       float,
    stop:         float,
    rr_ratio:     float = 0.0,
    confidence:   float = 0.0,
    session:      str   = "",
    regime:       str   = "",
    vwap_event:   str   = "",
    rsi_zone:     str   = "",
    rsi_value:    float = 50.0,
    sector_etf:   str   = "",
    sector_trend: str   = "",
    entry_type:   str   = "IMMEDIATE",
    mtf_alignment: str  = "",
) -> str:
    """
    Record a new signal for tracking.  Returns signal_id.
    Duplicate signals for the same ticker within the same scan are ignored.
    """
    if direction not in ("BUY", "SELL"):
        return ""
    if entry_price <= 0 or target <= 0 or stop <= 0:
        return ""

    signal_id = f"{ticker}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"

    with _lock:
        with _conn() as c:
            # Deduplicate: skip if a TRACKING signal already exists for this ticker+direction
            existing = c.execute(
                "SELECT id FROM bt_signals WHERE ticker=? AND direction=? AND status='TRACKING'",
                (ticker, direction)
            ).fetchone()
            if existing:
                return ""

            c.execute("""
                INSERT INTO bt_signals
                  (signal_id, fired_at, ticker, direction, entry_price, target, stop,
                   rr_ratio, confidence, session, regime, vwap_event, rsi_zone, rsi_value,
                   sector_etf, sector_trend, entry_type, mtf_alignment)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                signal_id,
                datetime.now(timezone.utc).isoformat(),
                ticker, direction,
                round(entry_price, 4), round(target, 4), round(stop, 4),
                round(rr_ratio, 3), round(confidence, 2),
                session, regime, vwap_event, rsi_zone, round(rsi_value, 1),
                sector_etf, sector_trend, entry_type, mtf_alignment,
            ))
            c.commit()
            logger.debug(f"[BT] Tracking {direction} {ticker} @ ${entry_price:.2f}  T:${target:.2f}  S:${stop:.2f}")
    return signal_id


# ── Update all tracking signals for a ticker ──────────────────────────────────

def _compute_r(current: float, entry: float, stop: float, direction: str) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    if direction == "BUY":
        return (current - entry) / risk
    else:
        return (entry - current) / risk


def update_tracking(ticker: str, current_price: float, vwap: float = 0.0) -> list[dict]:
    """
    Update all TRACKING signals for ticker with current_price.
    Returns list of newly resolved signals (for broadcast).
    """
    resolved = []
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT signal_id, direction, entry_price, target, stop, bars_tracked, max_favorable_r
                FROM bt_signals WHERE ticker=? AND status='TRACKING'
            """, (ticker,)).fetchall()

            for row in rows:
                sid    = row["signal_id"]
                d      = row["direction"]
                entry  = row["entry_price"]
                tgt    = row["target"]
                stp    = row["stop"]
                bars   = (row["bars_tracked"] or 0) + 1
                r_val  = _compute_r(current_price, entry, stp, d)
                max_r  = max(row["max_favorable_r"] or 0, r_val)

                # Record price path
                c.execute("""
                    INSERT INTO bt_price_path (signal_id, bar, price, r_val, ts)
                    VALUES (?,?,?,?,?)
                """, (sid, bars, round(current_price, 4), round(r_val, 4),
                      datetime.now(timezone.utc).isoformat()))

                # Check resolution
                status = None
                exit_reason = None

                if d == "BUY":
                    if current_price >= tgt:
                        status = "WIN";  exit_reason = "TARGET"
                    elif current_price <= stp:
                        status = "LOSS"; exit_reason = "STOP"
                    elif vwap > 0 and current_price < vwap and entry >= vwap:
                        status = "LOSS"; exit_reason = "VWAP_LOSS"
                else:
                    if current_price <= tgt:
                        status = "WIN";  exit_reason = "TARGET"
                    elif current_price >= stp:
                        status = "LOSS"; exit_reason = "STOP"
                    elif vwap > 0 and current_price > vwap and entry <= vwap:
                        status = "LOSS"; exit_reason = "VWAP_LOSS"

                if bars >= MAX_BARS and not status:
                    status = "TIMEOUT"; exit_reason = "TIMEOUT"

                if status:
                    pnl = (current_price - entry) / entry * 100
                    if d == "SELL":
                        pnl = -pnl
                    c.execute("""
                        UPDATE bt_signals
                        SET status=?, resolved_at=?, exit_price=?, exit_reason=?,
                            bars_tracked=?, max_favorable_r=?, pnl_pct=?, r_multiple=?
                        WHERE signal_id=?
                    """, (
                        status,
                        datetime.now(timezone.utc).isoformat(),
                        round(current_price, 4), exit_reason,
                        bars, round(max_r, 3),
                        round(pnl, 3), round(r_val, 3),
                        sid,
                    ))
                    logger.info(f"[BT] {status} {d} {ticker} @ ${current_price:.2f} | R={r_val:.2f} | {exit_reason}")
                    resolved.append({
                        "ticker": ticker, "direction": d, "status": status,
                        "exit_reason": exit_reason, "r_multiple": round(r_val, 3),
                        "pnl_pct": round(pnl, 3),
                    })
                else:
                    c.execute("""
                        UPDATE bt_signals
                        SET bars_tracked=?, max_favorable_r=?
                        WHERE signal_id=?
                    """, (bars, round(max_r, 3), sid))

            c.commit()
    return resolved


# ── Query helpers ──────────────────────────────────────────────────────────────

def get_tracking_signals() -> list[dict]:
    """Return all currently-tracking signals."""
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT * FROM bt_signals WHERE status='TRACKING'
                ORDER BY fired_at DESC
            """).fetchall()
    return [dict(r) for r in rows]


def get_recent_resolved(limit: int = 100) -> list[dict]:
    """Return most recent resolved signals."""
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT * FROM bt_signals WHERE status != 'TRACKING'
                ORDER BY resolved_at DESC LIMIT ?
            """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_price_path(signal_id: str) -> list[dict]:
    """Return bar-by-bar price path for a signal (for chart replay)."""
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT bar, price, r_val, ts FROM bt_price_path
                WHERE signal_id=? ORDER BY bar ASC
            """, (signal_id,)).fetchall()
    return [dict(r) for r in rows]


def get_performance_stats(
    min_resolved: int = 1,
    lookback_days: int = 30,
) -> dict:
    """
    Compute comprehensive performance stats across all dimensions.

    Returns dict with overall stats + breakdowns by:
    direction, session, regime, vwap_event, rsi_zone, entry_type,
    confidence_band, sector_trend, mtf_alignment
    """
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT * FROM bt_signals
                WHERE status IN ('WIN','LOSS','TIMEOUT')
                AND fired_at >= datetime('now', ? || ' days')
                ORDER BY fired_at DESC
            """, (f"-{lookback_days}",)).fetchall()

    rows = [dict(r) for r in rows]
    if not rows:
        return _empty_stats()

    def _stats(subset: list[dict]) -> dict:
        n      = len(subset)
        wins   = sum(1 for r in subset if r["status"] == "WIN")
        losses = sum(1 for r in subset if r["status"] == "LOSS")
        timeouts = sum(1 for r in subset if r["status"] == "TIMEOUT")
        wr     = round(wins / n * 100, 1) if n > 0 else 0.0
        pnls   = [r["pnl_pct"]    for r in subset if r["pnl_pct"]    is not None]
        rmults = [r["r_multiple"] for r in subset if r["r_multiple"] is not None]
        max_rs = [r["max_favorable_r"] for r in subset if r["max_favorable_r"] is not None]
        avg_r  = round(sum(rmults) / len(rmults), 3) if rmults else 0.0
        avg_pnl = round(sum(pnls) / len(pnls), 3) if pnls else 0.0
        avg_max_r = round(sum(max_rs) / len(max_rs), 3) if max_rs else 0.0
        # Expectancy = win_rate * avg_win_r - loss_rate * avg_loss_r
        win_rs  = [r["r_multiple"] for r in subset if r["status"] == "WIN"  and r["r_multiple"] is not None]
        loss_rs = [abs(r["r_multiple"]) for r in subset if r["status"] in ("LOSS","TIMEOUT") and r["r_multiple"] is not None]
        avg_win_r  = round(sum(win_rs)  / len(win_rs),  2) if win_rs  else 0.0
        avg_loss_r = round(sum(loss_rs) / len(loss_rs), 2) if loss_rs else 0.0
        expectancy = round((wins / n) * avg_win_r - (1 - wins / n) * avg_loss_r, 3) if n > 0 else 0.0
        return {
            "total": n, "wins": wins, "losses": losses, "timeouts": timeouts,
            "win_rate": wr, "avg_pnl": avg_pnl, "avg_r": avg_r,
            "avg_max_r": avg_max_r, "expectancy": expectancy,
            "avg_win_r": avg_win_r, "avg_loss_r": avg_loss_r,
        }

    def _breakdown(key: str) -> dict:
        groups: dict[str, list] = {}
        for r in rows:
            v = r.get(key) or "UNKNOWN"
            groups.setdefault(v, []).append(r)
        return {k: _stats(v) for k, v in groups.items() if len(v) >= min_resolved}

    def _confidence_bands() -> dict:
        bands = {"<50": [], "50-60": [], "60-70": [], "70-80": [], "80+": []}
        for r in rows:
            c = r.get("confidence") or 0
            if c < 50:   bands["<50"].append(r)
            elif c < 60: bands["50-60"].append(r)
            elif c < 70: bands["60-70"].append(r)
            elif c < 80: bands["70-80"].append(r)
            else:        bands["80+"].append(r)
        return {k: _stats(v) for k, v in bands.items() if len(v) >= min_resolved}

    # Exit reason breakdown
    exit_reasons: dict[str, int] = {}
    for r in rows:
        er = r.get("exit_reason") or "UNKNOWN"
        exit_reasons[er] = exit_reasons.get(er, 0) + 1

    return {
        "overall":          _stats(rows),
        "tracking_count":   len(get_tracking_signals()),
        "lookback_days":    lookback_days,
        "by_direction":     _breakdown("direction"),
        "by_session":       _breakdown("session"),
        "by_regime":        _breakdown("regime"),
        "by_vwap_event":    _breakdown("vwap_event"),
        "by_rsi_zone":      _breakdown("rsi_zone"),
        "by_entry_type":    _breakdown("entry_type"),
        "by_confidence":    _confidence_bands(),
        "by_sector_trend":  _breakdown("sector_trend"),
        "by_mtf":           _breakdown("mtf_alignment"),
        "exit_reasons":     exit_reasons,
    }


def _empty_stats() -> dict:
    empty = {"total":0,"wins":0,"losses":0,"timeouts":0,"win_rate":0.0,
             "avg_pnl":0.0,"avg_r":0.0,"avg_max_r":0.0,"expectancy":0.0,
             "avg_win_r":0.0,"avg_loss_r":0.0}
    return {
        "overall": empty, "tracking_count": 0, "lookback_days": 30,
        "by_direction":{}, "by_session":{}, "by_regime":{},
        "by_vwap_event":{}, "by_rsi_zone":{}, "by_entry_type":{},
        "by_confidence":{}, "by_sector_trend":{}, "by_mtf":{},
        "exit_reasons":{},
    }


def get_outcomes_for_ml(min_count: int = 30) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame of resolved signals suitable for ML retraining feedback.

    Columns:  signal_id, ticker, direction, confidence, session, regime,
              vwap_event, rsi_zone, rsi_value, entry_type, mtf_alignment,
              rr_ratio, outcome (1=WIN, 0=LOSS/TIMEOUT)
    """
    with _lock:
        with _conn() as c:
            rows = c.execute("""
                SELECT signal_id, ticker, direction, confidence, session, regime,
                       vwap_event, rsi_zone, rsi_value, entry_type, mtf_alignment,
                       rr_ratio, status
                FROM bt_signals
                WHERE status IN ('WIN','LOSS','TIMEOUT')
                ORDER BY fired_at DESC LIMIT 500
            """).fetchall()

    if len(rows) < min_count:
        return None

    data = [dict(r) for r in rows]
    df   = pd.DataFrame(data)
    df["outcome"] = (df["status"] == "WIN").astype(int)
    return df.drop(columns=["status"])
