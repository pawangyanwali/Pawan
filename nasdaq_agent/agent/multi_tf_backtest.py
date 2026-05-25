"""
Multi-timeframe backtester for scalping — historical OHLCV replay across
1min, 5min, 15min, 30min, and 1h bars.

Architecture
------------
Each timeframe is an independent strategy context with its own parameters
(hold duration, ATR-based target/stop multiples, RSI thresholds).  Signals
are generated rule-based (RSI + MACD + vol filter) so there is no circular
reference to the ML models being trained from these outcomes.

Results are stored in data/backtest_mtf.db for:
  1. Per-TF win rates surfaced in the dashboard
  2. Expectancy / Sharpe / max-drawdown per TF to rank strategy quality
  3. Labeled feature records fed back to XGBoost retraining
  4. Adaptive filter calibration with TF-aware context keys

Timeframe parameters (scalping-optimised)
-----------------------------------------
1min  — ultra-short scalps (spread cost heavy): tight stop 0.7×ATR, target 1.0×ATR, max 12 bars
5min  — standard scalps:                        stop 1.0×ATR, target 1.5×ATR, max 8 bars
15min — swing-scalps:                           stop 1.2×ATR, target 2.0×ATR, max 6 bars
30min — position scalps:                        stop 1.5×ATR, target 2.5×ATR, max 5 bars
1h    — intraday momentum:                      stop 2.0×ATR, target 3.0×ATR, max 4 bars
"""
from __future__ import annotations

import logging
import math
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from agent.walk_forward import replay_signals
from agent.db import get_conn, _get_pool

logger = logging.getLogger(__name__)

# ── Timeframe registry ────────────────────────────────────────────────────────

TF_CONFIGS: dict[str, dict] = {
    "1min": {
        "interval":       "1min",
        "label":          "1-Min Scalp",
        "max_bars_held":  12,
        "atr_target_mult": 1.0,
        "atr_stop_mult":   0.7,
        "rsi_buy_thresh":  35.0,
        "rsi_sell_thresh": 65.0,
        "min_vol_ratio":   1.0,
        "cooldown_bars":   12,
    },
    "5min": {
        "interval":       "5min",
        "label":          "5-Min Scalp",
        "max_bars_held":  8,
        "atr_target_mult": 1.5,
        "atr_stop_mult":   1.0,
        "rsi_buy_thresh":  35.0,
        "rsi_sell_thresh": 65.0,
        "min_vol_ratio":   1.0,
        "cooldown_bars":   8,
    },
    "15min": {
        "interval":       "15min",
        "label":          "15-Min Swing-Scalp",
        "max_bars_held":  6,
        "atr_target_mult": 2.0,
        "atr_stop_mult":   1.2,
        "rsi_buy_thresh":  38.0,
        "rsi_sell_thresh": 62.0,
        "min_vol_ratio":   0.9,
        "cooldown_bars":   6,
    },
    "30min": {
        "interval":       "30min",
        "label":          "30-Min Position Scalp",
        "max_bars_held":  5,
        "atr_target_mult": 2.5,
        "atr_stop_mult":   1.5,
        "rsi_buy_thresh":  40.0,
        "rsi_sell_thresh": 60.0,
        "min_vol_ratio":   0.8,
        "cooldown_bars":   5,
    },
    "1h": {
        "interval":       "1h",
        "label":          "1-Hr Intraday Momentum",
        "max_bars_held":  4,
        "atr_target_mult": 3.0,
        "atr_stop_mult":   2.0,
        "rsi_buy_thresh":  40.0,
        "rsi_sell_thresh": 60.0,
        "min_vol_ratio":   0.8,
        "cooldown_bars":   4,
    },
}

ALL_TIMEFRAMES = list(TF_CONFIGS.keys())

# Twelve Data interval strings (cache key → API param)
TD_INTERVAL: dict[str, str] = {
    "1min":  "1min",
    "5min":  "5min",
    "15min": "15min",
    "30min": "30min",
    "1h":    "1h",
}

# ── Database ──────────────────────────────────────────────────────────────────

_lock = threading.Lock()

_MTF_DDL = [
    """
    CREATE TABLE IF NOT EXISTS bt_mtf_trades (
        id          SERIAL PRIMARY KEY,
        run_dt      TEXT,
        timeframe   TEXT,
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
    """,
    "CREATE INDEX IF NOT EXISTS idx_mtf_tf_ticker ON bt_mtf_trades (run_dt, timeframe, ticker)",
    """
    CREATE TABLE IF NOT EXISTS bt_mtf_summary (
        run_dt      TEXT,
        timeframe   TEXT,
        ticker      TEXT,
        total       INTEGER,
        wins        INTEGER,
        win_rate    REAL,
        avg_pnl_r   REAL,
        expectancy  REAL,
        max_dd_r    REAL,
        sharpe      REAL,
        PRIMARY KEY (run_dt, timeframe, ticker)
    )
    """,
]


def init_db() -> None:
    """Create bt_mtf_trades and bt_mtf_summary tables if they don't exist."""
    pool = _get_pool()
    raw = pool.getconn()
    try:
        raw.autocommit = True
        with raw.cursor() as cur:
            for ddl in _MTF_DDL:
                cur.execute(ddl.strip())
    finally:
        pool.putconn(raw)


# ── Analytics helpers ─────────────────────────────────────────────────────────

def _sharpe(pnl_series: list[float]) -> float:
    """Annualised Sharpe (risk-free = 0) from a series of R multiples."""
    if len(pnl_series) < 4:
        return 0.0
    arr = np.array(pnl_series, dtype=float)
    mu  = arr.mean()
    sd  = arr.std(ddof=1)
    if sd == 0:
        return 0.0
    # Scale by sqrt(252) — treat each trade as one "day" for comparability
    return round(float(mu / sd * math.sqrt(252)), 3)


def _max_drawdown(pnl_series: list[float]) -> float:
    """Maximum peak-to-trough drawdown in R multiples."""
    if not pnl_series:
        return 0.0
    cumulative = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cumulative)
    drawdown    = running_max - cumulative
    return round(float(drawdown.max()), 3)


def _expectancy(records: list[dict]) -> float:
    """Expectancy = avg_win × win_rate + avg_loss × (1 - win_rate) in R."""
    wins   = [r["pnl_r"] for r in records if r.get("won")]
    losses = [r["pnl_r"] for r in records if not r.get("won")]
    if not records:
        return 0.0
    wr      = len(wins) / len(records)
    avg_win  = sum(wins)  / len(wins)  if wins   else 0.0
    avg_loss = sum(losses)/ len(losses) if losses else 0.0
    return round(avg_win * wr + avg_loss * (1 - wr), 4)


def compute_tf_stats(records: list[dict]) -> dict:
    """Compute per-timeframe performance metrics from a list of trade records."""
    if not records:
        return {"total": 0, "wins": 0, "win_rate": 0.0,
                "avg_pnl_r": 0.0, "expectancy": 0.0,
                "max_dd_r": 0.0, "sharpe": 0.0}
    wins      = [r for r in records if r.get("won")]
    pnl_series = [r["pnl_r"] for r in records]
    wr         = round(len(wins) / len(records), 4)
    return {
        "total":      len(records),
        "wins":       len(wins),
        "win_rate":   wr,
        "avg_pnl_r":  round(sum(pnl_series) / len(pnl_series), 4),
        "expectancy": _expectancy(records),
        "max_dd_r":   _max_drawdown(pnl_series),
        "sharpe":     _sharpe(pnl_series),
    }


# ── Per-ticker multi-TF backtest ──────────────────────────────────────────────

def run_ticker_mtf(
    ticker:    str,
    dfs_by_tf: dict[str, pd.DataFrame],
    run_dt:    str,
) -> dict[str, list[dict]]:
    """
    Run walk-forward replay on all available timeframes for one ticker.
    Returns {timeframe: [records]} — empty list if no data for that TF.
    """
    results: dict[str, list[dict]] = {}

    for tf, cfg in TF_CONFIGS.items():
        df = dfs_by_tf.get(tf)
        if df is None or df.empty:
            results[tf] = []
            continue

        records = replay_signals(
            ticker          = ticker,
            df              = df,
            atr_target_mult = cfg["atr_target_mult"],
            atr_stop_mult   = cfg["atr_stop_mult"],
            max_bars_held   = cfg["max_bars_held"],
            rsi_buy_thresh  = cfg["rsi_buy_thresh"],
            rsi_sell_thresh = cfg["rsi_sell_thresh"],
            min_vol_ratio   = cfg["min_vol_ratio"],
            cooldown_bars   = cfg["cooldown_bars"],
        )

        # Tag each record with its timeframe and run_dt
        for r in records:
            r["timeframe"] = tf
            r["run_dt"]    = run_dt

        results[tf] = records
        logger.debug(f"[MTF] {ticker} {tf}: {len(records)} trades")

    return results


# ── Full-universe run ─────────────────────────────────────────────────────────

def run_all_tickers(
    tickers:      list[str],
    get_df_fn:    Callable[[str, str], pd.DataFrame],
    run_dt:       str,
    broadcast_fn: Callable | None = None,
) -> dict[str, dict[str, list[dict]]]:
    """
    Run multi-TF backtest for every ticker.

    get_df_fn(ticker, timeframe) → DataFrame (from historical_cache.get_bars).
    Returns {ticker: {timeframe: [records]}}.
    """
    all_results: dict[str, dict[str, list[dict]]] = {}
    n = len(tickers)

    for i, ticker in enumerate(tickers):
        if broadcast_fn:
            broadcast_fn({
                "phase":       "BACKTESTING",
                "phase_label": f"Phase 3 — Multi-TF backtest: {ticker} ({i+1}/{n})",
                "tickers_done": i,
            })

        dfs_by_tf = {tf: get_df_fn(ticker, td) for tf, td in TD_INTERVAL.items()}
        all_results[ticker] = run_ticker_mtf(ticker, dfs_by_tf, run_dt)

    return all_results


# ── DB persistence ────────────────────────────────────────────────────────────

def store_results(all_results: dict[str, dict[str, list[dict]]], run_dt: str) -> None:
    """Persist all trade records and summary stats to SQLite."""
    trade_rows:   list[tuple] = []
    summary_rows: list[tuple] = []

    for ticker, tf_records in all_results.items():
        for tf, records in tf_records.items():
            if not records:
                continue

            for r in records:
                trade_rows.append((
                    run_dt, tf, ticker,
                    r.get("bar_dt", ""), r.get("direction", ""),
                    r.get("entry_price", 0.0), r.get("target", 0.0),
                    r.get("stop", 0.0), r.get("exit_price", 0.0),
                    r.get("outcome", ""), r.get("pnl_r", 0.0),
                    r.get("bars_held", 0), int(r.get("won", False)),
                ))

            stats = compute_tf_stats(records)
            summary_rows.append((
                run_dt, tf, ticker,
                stats["total"], stats["wins"], stats["win_rate"],
                stats["avg_pnl_r"], stats["expectancy"],
                stats["max_dd_r"], stats["sharpe"],
            ))

    with _lock:
        with get_conn() as c:
            c.executemany(
                "INSERT INTO bt_mtf_trades "
                "(run_dt,timeframe,ticker,bar_dt,direction,entry_price,"
                " target,stop,exit_price,outcome,pnl_r,bars_held,won) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                trade_rows,
            )
            c.executemany(
                "INSERT INTO bt_mtf_summary "
                "(run_dt,timeframe,ticker,total,wins,win_rate,"
                " avg_pnl_r,expectancy,max_dd_r,sharpe) "
                "VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT (run_dt,timeframe,ticker) DO UPDATE SET "
                "total=EXCLUDED.total, wins=EXCLUDED.wins, "
                "win_rate=EXCLUDED.win_rate, avg_pnl_r=EXCLUDED.avg_pnl_r, "
                "expectancy=EXCLUDED.expectancy, max_dd_r=EXCLUDED.max_dd_r, "
                "sharpe=EXCLUDED.sharpe",
                summary_rows,
            )

    logger.info(
        f"[MTF] Stored {len(trade_rows)} trades, "
        f"{len(summary_rows)} summary rows for run {run_dt}"
    )


# ── Query helpers (used by API) ───────────────────────────────────────────────

def get_summary(run_dt: str | None = None, limit_runs: int = 1) -> dict:
    """
    Return aggregated per-timeframe stats across all tickers.
    If run_dt is None, uses the most recent run.
    """
    with get_conn() as c:
        if run_dt is None:
            row = c.execute(
                "SELECT MAX(run_dt) AS max_run_dt FROM bt_mtf_summary"
            ).fetchone()
            run_dt = row["max_run_dt"] if row else None

        if not run_dt:
            return {}

        rows = c.execute("""
            SELECT timeframe,
                   SUM(total)                           AS total,
                   SUM(wins)                            AS wins,
                   ROUND(AVG(win_rate), 4)              AS win_rate,
                   ROUND(AVG(avg_pnl_r), 4)             AS avg_pnl_r,
                   ROUND(AVG(expectancy), 4)             AS expectancy,
                   ROUND(AVG(max_dd_r), 4)              AS max_dd_r,
                   ROUND(AVG(sharpe), 4)                AS sharpe,
                   COUNT(DISTINCT ticker)               AS tickers
            FROM bt_mtf_summary
            WHERE run_dt = ?
            GROUP BY timeframe
            ORDER BY timeframe
        """, (run_dt,)).fetchall()

    tf_stats: dict[str, dict] = {}
    for r in rows:
        tf    = r["timeframe"]
        total = r["total"] or 0
        wins  = r["wins"] or 0
        tf_stats[tf] = {
            "label":      TF_CONFIGS.get(tf, {}).get("label", tf),
            "total":      total,
            "wins":       wins,
            "win_rate":   round(wins / total * 100, 1) if total else 0,
            "avg_pnl_r":  r["avg_pnl_r"],
            "expectancy": r["expectancy"],
            "max_dd_r":   r["max_dd_r"],
            "sharpe":     r["sharpe"],
            "tickers":    r["tickers"],
        }

    return {"run_dt": run_dt, "by_timeframe": tf_stats}


def get_ticker_stats(ticker: str, run_dt: str | None = None) -> dict:
    """Per-TF breakdown for a single ticker."""
    with get_conn() as c:
        if run_dt is None:
            row = c.execute(
                "SELECT MAX(run_dt) AS max_run_dt FROM bt_mtf_summary"
            ).fetchone()
            run_dt = row["max_run_dt"] if row else None
        if not run_dt:
            return {}

        rows = c.execute("""
            SELECT timeframe, total, wins, win_rate, avg_pnl_r, expectancy, max_dd_r, sharpe
            FROM bt_mtf_summary
            WHERE run_dt=? AND ticker=?
            ORDER BY timeframe
        """, (run_dt, ticker)).fetchall()

    return {
        "ticker": ticker,
        "run_dt": run_dt,
        "timeframes": {
            r["timeframe"]: {
                "label":      TF_CONFIGS.get(r["timeframe"], {}).get("label", r["timeframe"]),
                "total":      r["total"], "wins": r["wins"],
                "win_rate":   round(r["wins"] / r["total"] * 100, 1) if r["total"] else 0,
                "avg_pnl_r":  r["avg_pnl_r"], "expectancy": r["expectancy"],
                "max_dd_r":   r["max_dd_r"],  "sharpe":     r["sharpe"],
            }
            for r in rows
        },
    }


def get_run_history(limit: int = 10) -> list[dict]:
    """List of past backtest run dates with overall stats."""
    with get_conn() as c:
        rows = c.execute("""
            SELECT run_dt,
                   SUM(total)                   AS total_trades,
                   ROUND(AVG(win_rate)*100, 1)  AS avg_win_rate,
                   COUNT(DISTINCT ticker)        AS tickers,
                   COUNT(DISTINCT timeframe)     AS timeframes
            FROM bt_mtf_summary
            GROUP BY run_dt
            ORDER BY run_dt DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [
        {
            "run_dt":       r["run_dt"],
            "total_trades": r["total_trades"],
            "avg_win_rate": r["avg_win_rate"],
            "tickers":      r["tickers"],
            "timeframes":   r["timeframes"],
        }
        for r in rows
    ]


def build_training_records(run_dt: str | None = None) -> pd.DataFrame:
    """
    Return all trade records from the most recent run as a DataFrame
    suitable for XGBoost retraining (outcome label + ticker/TF metadata).
    """
    with get_conn() as c:
        if run_dt is None:
            row = c.execute(
                "SELECT MAX(run_dt) AS max_run_dt FROM bt_mtf_trades"
            ).fetchone()
            run_dt = row["max_run_dt"] if row else None
        if not run_dt:
            return pd.DataFrame()

        rows = c.execute(
            "SELECT * FROM bt_mtf_trades WHERE run_dt = ?", (run_dt,)
        ).fetchall()

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def compute_filter_calibration(run_dt: str | None = None) -> dict:
    """
    Compute per-context win rates for adaptive filter calibration.
    Returns {context_key: {win_rate, count}} — same format as update_filter().
    """
    with get_conn() as c:
        if run_dt is None:
            row = c.execute(
                "SELECT MAX(run_dt) AS max_run_dt FROM bt_mtf_trades"
            ).fetchone()
            run_dt = row["max_run_dt"] if row else None
        if not run_dt:
            return {}

        by_setup: dict[str, dict] = {}

        # Win rate by timeframe
        rows = c.execute("""
            SELECT timeframe,
                   COUNT(*)  AS total,
                   SUM(won)  AS wins
            FROM bt_mtf_trades WHERE run_dt=?
            GROUP BY timeframe
            HAVING COUNT(*) >= 10
        """, (run_dt,)).fetchall()
        for r in rows:
            if r["total"]:
                by_setup[f"timeframe:{r['timeframe']}"] = {
                    "win_rate": round(r["wins"] / r["total"], 3),
                    "count":    r["total"],
                }

        # Win rate by direction
        rows = c.execute("""
            SELECT direction, COUNT(*) AS total, SUM(won) AS wins
            FROM bt_mtf_trades WHERE run_dt=?
            GROUP BY direction HAVING COUNT(*) >= 10
        """, (run_dt,)).fetchall()
        for r in rows:
            if r["total"]:
                by_setup[f"direction:{r['direction']}"] = {
                    "win_rate": round(r["wins"] / r["total"], 3),
                    "count":    r["total"],
                }

        # Win rate by timeframe × direction
        rows = c.execute("""
            SELECT timeframe, direction, COUNT(*) AS total, SUM(won) AS wins
            FROM bt_mtf_trades WHERE run_dt=?
            GROUP BY timeframe, direction HAVING COUNT(*) >= 5
        """, (run_dt,)).fetchall()
        for r in rows:
            if r["total"]:
                key = f"tf_direction:{r['timeframe']}:{r['direction']}"
                by_setup[key] = {
                    "win_rate": round(r["wins"] / r["total"], 3),
                    "count":    r["total"],
                }

        # Overall
        row = c.execute(
            "SELECT COUNT(*) AS total, SUM(won) AS wins FROM bt_mtf_trades WHERE run_dt=?",
            (run_dt,)
        ).fetchone()
        total = row["total"] or 0
        wins  = row["wins"] or 0
        overall_wr = wins / total if total else 0.0

    return {
        "overall":  {"win_rate": overall_wr, "count": total},
        "by_setup": by_setup,
    }
