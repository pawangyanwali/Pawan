"""
Vectorized backtest over historical OHLCV bars.

Signals are defined as indicator crossover conditions computed from
feature_engine.compute_features().  Each fired signal is simulated
with ATR-based T1/T2 exits matching paper_trading.py rules:
  - Stop:  entry ± 1×ATR
  - T1:    entry ± 1.5×ATR — close 50%, move stop to breakeven
  - T2:    entry ± 3.0×ATR — close remaining 50%
  - Timeout: close at market after MAX_BARS bars

Results are aggregated per signal across all tickers and reported
as win_rate / expectancy / Sharpe / max_drawdown in R-multiples.

Usage (via __main__.py):
  python -m historical --backtest
  python -m historical --backtest --interval 5min
  python -m historical --backtest --interval 1day --tickers AAPL,MSFT,NVDA
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STATUS_FILE = Path.home() / ".nasdaq_agent" / "hist_backtest_status.json"


def _write_status(state: dict) -> None:
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = dict(state)
        state["updated_at"] = time.time()
        tmp = STATUS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(STATUS_FILE)
    except Exception:
        pass


# Exit parameters — mirror paper_trading.py rules
ATR_STOP_MULT = 1.0   # stop = entry ± 1×ATR
R1_MULT       = 1.5   # T1   = entry ± 1.5×ATR (close 50%)
R2_MULT       = 3.0   # T2   = entry ± 3.0×ATR  (close 50%)
MAX_BARS      = 20    # bars before time-stop (scalp default; caller may override)


@dataclass
class TradeResult:
    ticker:    str
    interval:  str
    signal:    str
    direction: str          # "BUY" | "SELL"
    entry_ts:  pd.Timestamp
    pnl_r:     float        # P&L in R-multiples
    bars_held: int
    exit_type: str          # "t1" | "t2" | "stop" | "timeout"


@dataclass
class SignalReport:
    signal:              str
    interval:            str
    n_trades:            int   = 0
    wins:                int   = 0
    losses:              int   = 0
    win_rate:            float = 0.0
    avg_win_r:           float = 0.0
    avg_loss_r:          float = 0.0
    expectancy:          float = 0.0
    sharpe:              float = 0.0
    max_dd:              float = 0.0
    total_r:             float = 0.0
    tickers_with_trades: int   = 0


# ── Signal condition library ───────────────────────────────────────────────────

def _build_conditions(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Return {signal_name: bool_series} for all defined entry conditions.
    Every condition is fully causal (uses only past data via .shift(1)).
    """
    c     = df["Close"]
    ema9  = df.get("ema_9",      c)
    ema20 = df.get("ema_20",     c)
    rsi   = df.get("rsi_14",     pd.Series(50.0, index=df.index))
    mhst  = df.get("macd_hist",  pd.Series(0.0,  index=df.index))
    vr    = df.get("vol_ratio",  pd.Series(1.0,  index=df.index))
    bb_w  = df.get("bb_width",   pd.Series(0.0,  index=df.index))
    bb_p  = df.get("bb_pct",     pd.Series(0.5,  index=df.index))

    p_ema9  = ema9.shift(1)
    p_ema20 = ema20.shift(1)
    p_rsi   = rsi.shift(1)
    p_mhst  = mhst.shift(1)
    p_bb_w  = bb_w.shift(1)

    # BB squeeze: width in bottom 20th percentile of its own history
    bb_squeeze_thresh = bb_w.expanding(min_periods=50).quantile(0.20)

    return {
        # ── Trend crossovers ──────────────────────────────────────────────────
        "ema_cross_bull": (p_ema9 <= p_ema20) & (ema9 > ema20),
        "ema_cross_bear": (p_ema9 >= p_ema20) & (ema9 < ema20),

        # ── Momentum extremes ─────────────────────────────────────────────────
        "rsi_oversold":   (p_rsi <= 30) & (rsi > 30),
        "rsi_overbought": (p_rsi >= 70) & (rsi < 70),

        # ── MACD zero-crosses ─────────────────────────────────────────────────
        "macd_bull": (p_mhst <= 0) & (mhst > 0),
        "macd_bear": (p_mhst >= 0) & (mhst < 0),

        # ── Volume breakouts ──────────────────────────────────────────────────
        "vol_break_bull": (vr > 2.5) & (c > ema20) & (c > c.shift(1)),
        "vol_break_bear": (vr > 2.5) & (c < ema20) & (c < c.shift(1)),

        # ── Bollinger squeeze breakouts ───────────────────────────────────────
        "bb_squeeze_bull": (p_bb_w <= bb_squeeze_thresh) & (bb_p > 0.8),
        "bb_squeeze_bear": (p_bb_w <= bb_squeeze_thresh) & (bb_p < 0.2),
    }


_DIRECTION: dict[str, str] = {
    "ema_cross_bull":  "BUY",
    "ema_cross_bear":  "SELL",
    "rsi_oversold":    "BUY",
    "rsi_overbought":  "SELL",
    "macd_bull":       "BUY",
    "macd_bear":       "SELL",
    "vol_break_bull":  "BUY",
    "vol_break_bear":  "SELL",
    "bb_squeeze_bull": "BUY",
    "bb_squeeze_bear": "SELL",
}


# ── Trade simulator ────────────────────────────────────────────────────────────

def _simulate_trade(
    df:        pd.DataFrame,
    entry_idx: int,
    direction: str,
    max_bars:  int = MAX_BARS,
) -> tuple[float, int, str]:
    """
    Simulate T1/T2 exits from entry_idx bar using high/low of subsequent bars.

    Returns (pnl_r, bars_held, exit_type).
    pnl_r is expressed in R-multiples (1.0R = one ATR unit of risk).
    """
    entry_price = float(df["Close"].iloc[entry_idx])
    atr = float(df["atr_14"].iloc[entry_idx])
    if atr <= 0:
        return 0.0, 0, "skip"

    sign = 1.0 if direction == "BUY" else -1.0
    stop  = entry_price - sign * atr * ATR_STOP_MULT
    t1_px = entry_price + sign * atr * R1_MULT
    t2_px = entry_price + sign * atr * R2_MULT
    risk  = atr * ATR_STOP_MULT

    remaining = 1.0
    pnl = 0.0
    n = len(df)

    for i in range(1, min(max_bars + 1, n - entry_idx)):
        bar = df.iloc[entry_idx + i]
        hi, lo = float(bar["High"]), float(bar["Low"])

        # Stop check
        stop_hit = (direction == "BUY" and lo <= stop) or \
                   (direction == "SELL" and hi >= stop)
        if stop_hit:
            pnl += remaining * sign * (stop - entry_price) / risk
            return round(pnl, 4), i, "stop"

        # T1 check
        t1_hit = (direction == "BUY" and hi >= t1_px) or \
                 (direction == "SELL" and lo <= t1_px)
        if t1_hit and remaining > 0.5:
            pnl += 0.5 * R1_MULT          # close half at T1
            remaining = 0.5
            stop = entry_price             # move stop to breakeven

        # T2 check
        t2_hit = (direction == "BUY" and hi >= t2_px) or \
                 (direction == "SELL" and lo <= t2_px)
        if t2_hit and remaining > 0:
            pnl += remaining * R2_MULT
            return round(pnl, 4), i, "t2"

    # Time-stop: exit at last close
    last_close = float(df["Close"].iloc[min(entry_idx + max_bars, n - 1)])
    pnl += remaining * sign * (last_close - entry_price) / risk
    return round(pnl, 4), max_bars, "timeout"


# ── Per-ticker backtest ────────────────────────────────────────────────────────

def backtest_ticker(
    ticker:   str,
    interval: str,
    max_bars: int = MAX_BARS,
) -> list[TradeResult]:
    """
    Run all signal conditions against one ticker's historical bars.
    Returns a flat list of TradeResult, one per fired signal.
    """
    from agent.feature_engine import compute_features, FEATURE_COLS_V2
    from historical.store import read_ticker_bars

    df_raw = read_ticker_bars(interval, ticker)
    if len(df_raw) < 200:
        return []

    df = compute_features(df_raw, ticker)
    if df is None or df.empty:
        return []

    # Verify the core feature columns we use were actually computed
    required = {"ema_9", "ema_20", "rsi_14", "macd_hist", "vol_ratio", "atr_14"}
    if not required.issubset(df.columns):
        logger.debug("[Backtest] %s %s: missing feature columns — skip", ticker, interval)
        return []

    conditions = _build_conditions(df)
    trades: list[TradeResult] = []

    for signal_name, cond_series in conditions.items():
        direction = _DIRECTION[signal_name]
        fired_positions = [
            pos for pos, val in enumerate(cond_series.fillna(False))
            if val and pos + max_bars < len(df)
        ]

        for pos in fired_positions:
            pnl_r, bars, exit_type = _simulate_trade(df, pos, direction, max_bars)
            if exit_type == "skip":
                continue
            trades.append(TradeResult(
                ticker=ticker,
                interval=interval,
                signal=signal_name,
                direction=direction,
                entry_ts=df.index[pos],
                pnl_r=pnl_r,
                bars_held=bars,
                exit_type=exit_type,
            ))

    return trades


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate_results(
    trades:   list[TradeResult],
    interval: str,
) -> list[SignalReport]:
    """Aggregate raw trades into per-signal SignalReport objects."""
    by_signal: dict[str, list[TradeResult]] = defaultdict(list)
    for t in trades:
        by_signal[t.signal].append(t)

    reports: list[SignalReport] = []
    for signal, bucket in sorted(by_signal.items()):
        pnls   = [t.pnl_r for t in bucket]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        n      = len(pnls)
        if n == 0:
            continue

        win_rate   = len(wins) / n
        avg_win    = float(np.mean(wins))   if wins   else 0.0
        avg_loss   = float(np.mean(losses)) if losses else 0.0
        expectancy = avg_win * win_rate + avg_loss * (1 - win_rate)

        std = float(np.std(pnls))
        sharpe = float(np.mean(pnls) / std * np.sqrt(252)) if std > 0 else 0.0

        cum  = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum)
        max_dd = float((cum - peak).min()) if len(cum) else 0.0

        reports.append(SignalReport(
            signal=signal,
            interval=interval,
            n_trades=n,
            wins=len(wins),
            losses=len(losses),
            win_rate=round(win_rate, 3),
            avg_win_r=round(avg_win, 3),
            avg_loss_r=round(avg_loss, 3),
            expectancy=round(expectancy, 3),
            sharpe=round(sharpe, 3),
            max_dd=round(max_dd, 3),
            total_r=round(float(sum(pnls)), 2),
            tickers_with_trades=len({t.ticker for t in bucket}),
        ))

    return sorted(reports, key=lambda r: r.expectancy, reverse=True)


# ── Full run ───────────────────────────────────────────────────────────────────

def run_backtest(
    tickers:     list[str],
    interval:    str = "5min",
    max_bars:    int = MAX_BARS,
    progress_cb  = None,
) -> list[SignalReport]:
    """Run backtest across all tickers for one interval. Returns aggregated reports.

    Writes live progress to STATUS_FILE so the web service can poll it.
    progress_cb(done, total, ticker, n_trades_so_far) is called after each ticker.
    """
    all_trades: list[TradeResult] = []
    n = len(tickers)
    t0 = time.time()

    _write_status({
        "running": True, "done": 0, "total": n,
        "current_ticker": "", "trades_so_far": 0,
        "elapsed_s": 0.0, "started_at": t0, "results": [],
    })

    logger.info("[Backtest] %s — %d tickers, max_bars=%d", interval, n, max_bars)
    for i, ticker in enumerate(tickers, 1):
        trades = backtest_ticker(ticker, interval, max_bars)
        all_trades.extend(trades)

        _write_status({
            "running": True, "done": i, "total": n,
            "current_ticker": ticker, "trades_so_far": len(all_trades),
            "elapsed_s": round(time.time() - t0, 1),
            "started_at": t0, "results": [],
        })
        if i % 100 == 0 or i == n:
            logger.info("[Backtest] [%d/%d] total trades so far: %d", i, n, len(all_trades))
        if progress_cb:
            try:
                progress_cb(i, n, ticker, len(all_trades))
            except Exception:
                pass

    reports = aggregate_results(all_trades, interval)
    elapsed = round(time.time() - t0, 1)
    _write_status({
        "running": False, "done": n, "total": n,
        "current_ticker": "", "trades_so_far": len(all_trades),
        "elapsed_s": elapsed, "started_at": t0,
        "results": [asdict(r) for r in reports],
    })
    logger.info("[Backtest] Complete — %d trades, %d signals", len(all_trades), len(reports))
    return reports


def print_report(reports: list[SignalReport]) -> None:
    """Print a formatted summary table to stdout."""
    header = (
        f"\n{'Signal':<22} {'N':>7} {'WinRate':>8} {'AvgWin':>9} "
        f"{'AvgLoss':>9} {'Expect':>8} {'Sharpe':>8} {'MaxDD':>8} {'Tickers':>8}"
    )
    print("=" * 90)
    print(header)
    print("-" * 90)
    for r in reports:
        print(
            f"{r.signal:<22} {r.n_trades:>7,} {r.win_rate:>8.1%} "
            f"{r.avg_win_r:>8.3f}R {r.avg_loss_r:>8.3f}R "
            f"{r.expectancy:>7.3f}R {r.sharpe:>8.2f} "
            f"{r.max_dd:>7.2f}R {r.tickers_with_trades:>8}"
        )
    print("=" * 90)
    print()
