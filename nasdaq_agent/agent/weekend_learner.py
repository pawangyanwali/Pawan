"""
Weekend Learning Orchestrator — runs when markets are closed (Sat/Sun/holidays).

When the market is closed the agent would otherwise sit idle.  Instead, this
module uses the downtime to deepen the ML models by:

  Phase 1  FETCHING    — Pull 2 extra historical windows per ticker (going
                          further back in time) and persist to SQLite cache.
                          Each weekend adds ~64 more trading days of 5-min bars.

  Phase 2  REPLAYING   — Walk-forward signal replay on the full cached history.
                          Generates thousands of clean, labeled training records
                          (see walk_forward.py for the "clean data" guarantees).

  Phase 3  RETRAINING  — Pass the extended history to the XGBoost models for a
                          full batch retrain (not incremental) using more data
                          than the live scanner ever fetches.

  Phase 4  CALIBRATING — Derive per-context win rates from walk-forward records
                          and feed them to the adaptive filter so its suppression
                          thresholds are grounded in historical evidence.

  Phase 5  DONE        — Broadcast final summary; learner goes idle until next
                          closed-market window.

Auto-start: the scanner's main loop calls maybe_start() on every tick.
Manual start: POST /api/weekend-learning/start (admin override).
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Persistent store for walk-forward records ─────────────────────────────────
_RECORDS_DB = Path(__file__).parent.parent / "data" / "weekend_learning.db"
_RECORDS_DB.parent.mkdir(parents=True, exist_ok=True)


def _init_records_db():
    conn = sqlite3.connect(str(_RECORDS_DB), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_records (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            weekend_dt  TEXT,
            ticker      TEXT,
            bar_dt      TEXT,
            direction   TEXT,
            entry_price REAL,
            target      REAL,
            stop        REAL,
            exit_price  REAL,
            outcome     TEXT,
            pnl_r       REAL,
            bars_held   INTEGER,
            won         INTEGER
        )
    """)
    conn.commit()
    return conn


def _insert_records(records: list[dict], weekend_dt: str):
    if not records:
        return
    conn = _init_records_db()
    try:
        rows = [
            (weekend_dt, r["ticker"], r["bar_dt"], r["direction"],
             r["entry_price"], r["target"], r["stop"], r["exit_price"],
             r["outcome"], r["pnl_r"], r["bars_held"], int(r["won"]))
            for r in records
        ]
        conn.executemany(
            "INSERT INTO signal_records "
            "(weekend_dt,ticker,bar_dt,direction,entry_price,target,stop,"
            " exit_price,outcome,pnl_r,bars_held,won) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


# ── Shared state (read by /api/weekend-learning/status) ──────────────────────
_state: dict[str, Any] = {
    "phase":           "IDLE",
    "phase_label":     "Idle — waiting for weekend",
    "started_at":      None,
    "finished_at":     None,
    "weekend_dt":      None,
    "tickers_total":   0,
    "tickers_done":    0,
    "records_generated": 0,
    "new_bars_fetched":  0,
    "retrain_complete":  False,
    "filter_calibrated": False,
    "error":             None,
    "discoveries":       [],   # [{label, detail}]
    "cache_stats":       {},
}

_thread:    threading.Thread | None = None
_stop_flag: threading.Event         = threading.Event()
_broadcast: Callable | None         = None   # injected by main.py


def register_broadcast(fn: Callable) -> None:
    global _broadcast
    _broadcast = fn


def _emit(update: dict) -> None:
    """Update state and broadcast to WebSocket clients."""
    _state.update(update)
    if _broadcast:
        try:
            _broadcast({"type": "weekend_learning", "data": dict(_state)})
        except Exception:
            pass


# ── Core learning phases ──────────────────────────────────────────────────────

def _phase_fetch(tickers: list[str]) -> int:
    """Phase 1: fetch deep history for all tickers into SQLite cache."""
    from agent.historical_cache import fetch_and_store, older_window_end_dates

    _emit({"phase": "FETCHING", "phase_label": "Phase 1 — Fetching deep history"})
    total_new = 0

    def _bcast(msg: dict):
        _emit({"phase_label": f"Phase 1 — {msg.get('detail','')}"})

    # Always refresh the current window (5min + 15min + daily)
    for interval, outputsize in [("5min", 5000), ("15min", 500), ("1day", 500)]:
        if _stop_flag.is_set():
            break
        inserted = fetch_and_store(tickers, interval, outputsize,
                                   broadcast_fn=_bcast)
        total_new += sum(inserted.values())
        _emit({"new_bars_fetched": _state["new_bars_fetched"] + sum(inserted.values())})

    # Fetch OLDER windows for 5min (the key depth expansion)
    for end_date in older_window_end_dates(weeks_back=16, step_weeks=8):
        if _stop_flag.is_set():
            break
        logger.info(f"[WeekendLearner] Fetching older 5min window ending {end_date}")
        inserted = fetch_and_store(tickers, "5min", 5000, end_date=end_date,
                                   broadcast_fn=_bcast)
        total_new += sum(inserted.values())
        _emit({"new_bars_fetched": _state["new_bars_fetched"] + sum(inserted.values())})

    return total_new


def _phase_replay(tickers: list[str], weekend_dt: str) -> list[dict]:
    """Phase 2: walk-forward replay on cached history → labeled signal records."""
    from agent.historical_cache import get_bars
    from agent.walk_forward import replay_signals

    _emit({"phase": "REPLAYING",
           "phase_label": "Phase 2 — Walk-forward signal replay"})

    all_records: list[dict] = []
    n = len(tickers)

    for i, ticker in enumerate(tickers):
        if _stop_flag.is_set():
            break

        _emit({
            "phase_label":   f"Phase 2 — Replaying {ticker} ({i+1}/{n})",
            "tickers_done":  i,
        })

        df = get_bars(ticker, "5min", min_bars=100)
        if df.empty:
            continue

        records = replay_signals(ticker, df)
        if records:
            all_records.extend(records)
            _emit({"records_generated": len(all_records)})

    # Persist to SQLite for cumulative analytics
    _insert_records(all_records, weekend_dt)
    logger.info(f"[WeekendLearner] Replay done — {len(all_records)} records from {n} tickers")
    return all_records


def _phase_retrain(tickers: list[str]) -> bool:
    """Phase 3: full batch retrain using extended cached history."""
    from agent.historical_cache import get_bars
    from agent.ml_model import retrain_all

    _emit({"phase": "RETRAINING",
           "phase_label": "Phase 3 — Retraining ML models on extended history"})

    # Build pre-fetched data dicts from SQLite cache
    # (bypasses the standard 5000-bar API fetch inside retrain_all)
    hist_5m: dict  = {}
    hist_15m: dict = {}
    hist_1d: dict  = {}

    n = len(tickers)
    for i, ticker in enumerate(tickers):
        if _stop_flag.is_set():
            return False
        _emit({"phase_label": f"Phase 3 — Loading {ticker} ({i+1}/{n})"})

        df5  = get_bars(ticker, "5min",  min_bars=200)
        df15 = get_bars(ticker, "15min", min_bars=50)
        df1d = get_bars(ticker, "1day",  min_bars=50)

        if not df5.empty:  hist_5m[ticker]  = df5
        if not df15.empty: hist_15m[ticker] = df15
        if not df1d.empty: hist_1d[ticker]  = df1d

    _emit({"phase_label": f"Phase 3 — Training {len(hist_5m)} tickers (this takes a while…)"})

    try:
        retrain_all(
            tickers,
            delay      = 0.0,
            daily_data = hist_1d if hist_1d else None,
            hist_5m    = hist_5m  if hist_5m  else None,
            hist_15m   = hist_15m if hist_15m else None,
        )
        return True
    except Exception as exc:
        logger.error(f"[WeekendLearner] Retrain failed: {exc}")
        _emit({"error": f"Retrain failed: {exc}"})
        return False


def _phase_calibrate(records: list[dict]) -> bool:
    """Phase 4: derive per-context win rates and update adaptive filter."""
    from agent.walk_forward import compute_win_rate_by_context
    from agent.adaptive_filter import update_filter

    _emit({"phase": "CALIBRATING",
           "phase_label": "Phase 4 — Calibrating adaptive filter from walk-forward outcomes"})

    if not records:
        logger.info("[WeekendLearner] No records — skipping calibration")
        return False

    # Overall win rate
    total   = len(records)
    wins    = sum(1 for r in records if r["won"])
    overall_wr = wins / total if total else 0.0

    # Context-level win rates
    by_setup = compute_win_rate_by_context(records)

    stats = {
        "overall": {"win_rate": overall_wr, "count": total},
        "by_setup": by_setup,
    }

    discoveries = []

    # Tag notable suppressions / boosts
    for key, ctx in by_setup.items():
        wr  = ctx["win_rate"]
        cnt = ctx["count"]
        if cnt < 10:
            continue
        if wr < 0.35:
            discoveries.append({"label": f"Suppress: {key}", "detail": f"{wr*100:.0f}% win rate ({cnt} trades) — below threshold"})
        elif wr >= 0.72:
            discoveries.append({"label": f"Boost: {key}",    "detail": f"{wr*100:.0f}% win rate ({cnt} trades) — confidence boost"})

    try:
        update_filter(stats, source="weekend_walk_forward")
        return True
    except Exception as exc:
        logger.error(f"[WeekendLearner] Filter calibration failed: {exc}")
        return False
    finally:
        _emit({"discoveries": discoveries[:10]})   # cap at 10 for the dashboard


# ── Orchestrator thread ───────────────────────────────────────────────────────

def _run_learning(tickers: list[str]) -> None:
    """Main learning thread — runs all phases sequentially."""
    from agent.historical_cache import cache_stats

    weekend_dt = date.today().isoformat()

    try:
        _emit({
            "phase":         "FETCHING",
            "started_at":    datetime.utcnow().isoformat(),
            "weekend_dt":    weekend_dt,
            "tickers_total": len(tickers),
            "tickers_done":  0,
            "records_generated": 0,
            "new_bars_fetched":  0,
            "retrain_complete":  False,
            "filter_calibrated": False,
            "error":         None,
            "discoveries":   [],
        })

        # Phase 1 — fetch deep history
        _phase_fetch(tickers)
        if _stop_flag.is_set():
            return
        _emit({"cache_stats": cache_stats()})

        # Phase 2 — walk-forward replay
        records = _phase_replay(tickers, weekend_dt)
        if _stop_flag.is_set():
            return

        # Phase 3 — retrain with extended data
        retrained = _phase_retrain(tickers)
        _emit({"retrain_complete": retrained})
        if _stop_flag.is_set():
            return

        # Phase 4 — calibrate adaptive filter
        calibrated = _phase_calibrate(records)
        _emit({"filter_calibrated": calibrated})

        # Done
        total_wins = sum(1 for r in records if r["won"])
        overall_wr = total_wins / len(records) * 100 if records else 0.0
        _emit({
            "phase":       "DONE",
            "phase_label": (
                f"✅ Weekend learning complete — "
                f"{len(records)} signals replayed, "
                f"{overall_wr:.1f}% win rate, "
                f"{_state['new_bars_fetched']:,} new bars stored"
            ),
            "finished_at": datetime.utcnow().isoformat(),
            "tickers_done": len(tickers),
            "cache_stats": cache_stats(),
        })
        logger.info(
            f"[WeekendLearner] DONE — {len(records)} records, "
            f"{overall_wr:.1f}% WR, {_state['new_bars_fetched']:,} new bars"
        )

    except Exception as exc:
        logger.error(f"[WeekendLearner] Unhandled error: {exc}", exc_info=True)
        _emit({"phase": "ERROR", "phase_label": f"Error: {exc}", "error": str(exc)})


# ── Public API ────────────────────────────────────────────────────────────────

def is_running() -> bool:
    return _thread is not None and _thread.is_alive()


def get_status() -> dict:
    s = dict(_state)
    s["is_running"] = is_running()
    return s


def start(tickers: list[str] | None = None) -> bool:
    """
    Start the learning thread.  Returns False if already running.
    Caller may pass tickers; defaults to full active ticker list.
    """
    global _thread

    if is_running():
        logger.info("[WeekendLearner] Already running — ignoring start()")
        return False

    if tickers is None:
        from config import get_active_tickers
        tickers = get_active_tickers()

    _stop_flag.clear()
    _thread = threading.Thread(
        target=_run_learning,
        args=(tickers,),
        name="WeekendLearner",
        daemon=True,
    )
    _thread.start()
    logger.info(f"[WeekendLearner] Started — {len(tickers)} tickers")
    return True


def stop() -> None:
    """Signal the learning thread to stop after its current phase."""
    _stop_flag.set()
    logger.info("[WeekendLearner] Stop requested")


def maybe_start(tickers: list[str] | None = None) -> bool:
    """
    Auto-start when market is closed (weekend/holiday) and not already running.
    Called by the main scan loop on every tick.
    Returns True if learning was started, False if already running or not applicable.
    """
    if is_running():
        return False
    if _state.get("phase") == "DONE" and _state.get("weekend_dt") == date.today().isoformat():
        return False   # already completed this weekend

    from agent.market_hours import get_session_info
    info = get_session_info()
    if info.get("is_weekend") or info.get("is_holiday"):
        logger.info("[WeekendLearner] Market closed — auto-starting weekend learning")
        return start(tickers)
    return False


def historical_performance() -> dict:
    """
    Query cumulative weekend learning stats from the SQLite records store.
    Used by the dashboard /api/weekend-learning/history endpoint.
    """
    conn = _init_records_db()
    try:
        rows = conn.execute("""
            SELECT
                weekend_dt,
                COUNT(*)                    AS total,
                SUM(won)                    AS wins,
                ROUND(AVG(pnl_r), 3)        AS avg_pnl_r,
                ROUND(AVG(bars_held), 1)    AS avg_bars,
                COUNT(DISTINCT ticker)      AS tickers
            FROM signal_records
            GROUP BY weekend_dt
            ORDER BY weekend_dt DESC
            LIMIT 20
        """).fetchall()
        return [
            {"weekend":    r[0], "total": r[1], "wins": r[2],
             "win_rate":   round(r[2]/r[1]*100, 1) if r[1] else 0,
             "avg_pnl_r":  r[3], "avg_bars": r[4], "tickers": r[5]}
            for r in rows
        ]
    except Exception:
        return []
    finally:
        conn.close()
