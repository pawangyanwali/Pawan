"""
Main scanning engine — optimised for Twelve Data Grow-377 plan.

3 calls per scan (50 tickers / 20 per batch = 3 batches × 0.20s ≈ 0.60s).
5M and 1H data served from in-process cache (TTL 5 min / 1 h respectively),
so additional timeframe data costs zero API calls on most scan cycles.
"""

import logging
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Callable

import numpy as np
import pandas as pd

from config import (
    NASDAQ_TICKERS,
    SCAN_INTERVAL_SECONDS,
    ML_RETRAIN_INTERVAL,
    CACHE_TTL_5M,
    CACHE_TTL_1H,
    CACHE_TTL_1D,
)
from agent.data_fetcher import (
    fetch_batch_realtime,
    fetch_batch_interval,
    fetch_ticker_info,
)
from agent.technical import compute_indicators, score_technical
from agent.volume import score_volume, relative_volume, detect_unusual_volume
from agent.ml_model import predict, predict_daily, retrain_all
from agent.sentiment import score_sentiment
from agent.prediction import generate_prediction
from agent.mtf_analysis import multi_timeframe_analysis

logger = logging.getLogger(__name__)


# ── StockSignal dataclass ─────────────────────────────────────────────────────

@dataclass
class StockSignal:
    # ── Core price data ───────────────────────────────────────────────────────
    ticker:       str
    name:         str
    price:        float
    change_pct:   float

    # ── Component scores [-1, +1] ─────────────────────────────────────────────
    technical:     float
    volume:        float
    ml_prob:       float      # 40% daily ML + 60% intraday ML
    ml_daily_prob: float      # daily model probability (separately)
    sentiment:     float
    score:         float

    # ── Legacy signal label ────────────────────────────────────────────────────
    signal:        str

    # ── Volume flags ──────────────────────────────────────────────────────────
    rel_volume:   float
    unusual_vol:  bool

    # ── Professional prediction ───────────────────────────────────────────────
    prediction:        str
    confidence:        float
    trend:             str
    trend_probability: float
    ml_trained:        bool
    target_price:      float
    stop_loss:         float
    rr_ratio:          float
    patterns:          list = field(default_factory=list)
    reasons:           list = field(default_factory=list)

    # ── Multi-timeframe analysis ───────────────────────────────────────────────
    mtf_score:      float = 0.0
    mtf_alignment:  str   = "MIXED"
    mtf_bull_count: int   = 0
    mtf_bear_count: int   = 0
    mtf_timeframes: dict  = field(default_factory=dict)

    # ── Support / Resistance ──────────────────────────────────────────────────
    supports:     list  = field(default_factory=list)
    resistances:  list  = field(default_factory=list)
    pivots:       dict  = field(default_factory=dict)
    poc:          float = 0.0

    # ── RSI zone ──────────────────────────────────────────────────────────────
    rsi_zone:         str   = "NEUTRAL"   # EXTREME_OB | OB | NEUTRAL | OS | EXTREME_OS
    rsi_value:        float = 50.0
    rsi_gated:        bool  = False       # True when RSI overrode composite direction

    # ── Exhaustion / retest / bounce entry ───────────────────────────────────
    entry_type:       str   = "IMMEDIATE"
    retest_level:     float = 0.0
    entry_zone_low:   float = 0.0
    entry_zone_high:  float = 0.0
    exhaustion_flags: list  = field(default_factory=list)
    bounce_signals:   list  = field(default_factory=list)
    rr_quality:       str   = "LOW"
    rr_qualifies:     bool  = False

    # ── Chart candles (last 80 × 1-min bars) ─────────────────────────────────
    candles:    list = field(default_factory=list)

    # ── News ──────────────────────────────────────────────────────────────────
    headlines:  list = field(default_factory=list)
    scanned_at: str  = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if hasattr(v, "item"):
                d[k] = v.item()
        d["unusual_vol"] = bool(d["unusual_vol"])
        d["ml_trained"]  = bool(d["ml_trained"])
        return d


# ── Ticker metadata cache ─────────────────────────────────────────────────────

_info_cache: dict[str, dict] = {}


def _get_info(ticker: str) -> dict:
    if ticker not in _info_cache:
        _info_cache[ticker] = fetch_ticker_info(ticker)
    return _info_cache[ticker]


def _build_candles(df: pd.DataFrame, n: int = 80) -> list:
    """Serialise the last N OHLCV bars for TradingView Lightweight Charts."""
    tail    = df.tail(n)
    candles = []
    for ts, row in tail.iterrows():
        try:
            candles.append({
                "time":   int(pd.Timestamp(ts).timestamp()),
                "open":   round(float(row["Open"]),  4),
                "high":   round(float(row["High"]),  4),
                "low":    round(float(row["Low"]),   4),
                "close":  round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
            })
        except Exception:
            pass
    return candles


# ── Single ticker analysis ────────────────────────────────────────────────────

def analyse_ticker(
    ticker: str,
    df_1m:  Optional[pd.DataFrame],
    df_5m:  pd.DataFrame,
    df_1h:  pd.DataFrame,
    df_1d:  pd.DataFrame,
) -> Optional[StockSignal]:
    try:
        if df_1m is None or df_1m.empty or len(df_1m) < 5:
            return None

        df_ind = compute_indicators(df_1m.copy())
        last   = df_ind.iloc[-1]

        price      = float(last["Close"])
        open_price = float(df_ind.iloc[0]["Open"])
        change_pct = round((price - open_price) / open_price * 100, 3) if open_price else 0.0

        tech            = score_technical(last)
        vol             = score_volume(df_ind)
        ml_scalp        = predict(ticker, df_ind)
        ml_daily_p      = predict_daily(ticker, df_1d) if not df_1d.empty else 0.5
        # Blend: 40% daily (swing context) + 60% intraday (scalp timing)
        ml_combined     = round(0.4 * ml_daily_p + 0.6 * ml_scalp, 4)
        sent, headlines = score_sentiment(ticker)
        rvol            = relative_volume(df_ind)
        uvol            = detect_unusual_volume(df_ind)

        # Multi-timeframe analysis (6 TFs: 5M, 15M, 30M, 1H, 4H, 1D)
        mtf = multi_timeframe_analysis(df_1m, df_5m, df_1h, df_1d)

        # Professional prediction
        pred = generate_prediction(
            ticker, df_ind, tech, vol, ml_combined, sent, last,
            mtf_score=mtf["mtf_score"],
        )

        score   = round(float(np.clip(pred["composite_score"], -1, 1)), 4)
        candles = _build_candles(df_1m)
        info    = _get_info(ticker)

        return StockSignal(
            ticker            = ticker,
            name              = info.get("name", ticker),
            price             = round(price, 4),
            change_pct        = change_pct,
            technical         = round(tech, 4),
            volume            = round(vol, 4),
            ml_prob           = ml_combined,
            ml_daily_prob     = round(ml_daily_p, 4),
            sentiment         = round(sent, 4),
            score             = score,
            signal            = pred["direction"],
            rel_volume        = rvol,
            unusual_vol       = uvol,
            prediction        = pred["direction"],
            confidence        = round(pred["confidence"], 1),
            trend             = pred["trend"],
            trend_probability = round(float(pred.get("trend_probability", 0.5)), 2),
            ml_trained        = bool(pred.get("ml_trained", False)),
            target_price      = pred["target_price"],
            stop_loss         = pred["stop_loss"],
            rr_ratio          = pred["rr_ratio"],
            patterns          = pred["patterns"],
            reasons           = pred["reasons"],
            supports          = pred["supports"],
            resistances       = pred["resistances"],
            pivots            = pred["pivots"],
            poc               = pred["poc"],
            mtf_score         = float(mtf["mtf_score"]),
            mtf_alignment     = mtf["alignment"],
            mtf_bull_count    = int(mtf["bull_count"]),
            mtf_bear_count    = int(mtf["bear_count"]),
            mtf_timeframes    = mtf["timeframes"],
            rsi_zone          = pred.get("rsi_zone",          "NEUTRAL"),
            rsi_value         = float(pred.get("rsi_value", 50.0)),
            rsi_gated         = bool(pred.get("rsi_gated",  False)),
            entry_type        = pred.get("entry_type",       "IMMEDIATE"),
            retest_level      = pred.get("retest_level",     0.0),
            entry_zone_low    = pred.get("entry_zone_low",   0.0),
            entry_zone_high   = pred.get("entry_zone_high",  0.0),
            exhaustion_flags  = pred.get("exhaustion_flags", []),
            bounce_signals    = pred.get("bounce_signals",   []),
            rr_quality        = pred.get("rr_quality",       "LOW"),
            rr_qualifies      = bool(pred.get("rr_qualifies", False)),
            candles           = candles,
            headlines         = headlines[:5],
        )
    except Exception as e:
        logger.warning(f"[{ticker}] analysis error: {e}", exc_info=True)
        return None


# ── Scanner ───────────────────────────────────────────────────────────────────

class Scanner:
    def __init__(self):
        self.signals:       list[StockSignal] = []
        self.last_scan:     Optional[str]     = None
        self.is_running:    bool              = False
        self._callbacks:    list[Callable]    = []
        self._last_retrain: float             = 0.0

    def register_callback(self, fn: Callable) -> None:
        self._callbacks.append(fn)

    def _notify(self, signals: list[StockSignal]) -> None:
        for fn in self._callbacks:
            try:
                fn(signals)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def _should_retrain(self) -> bool:
        if self._last_retrain == 0.0:
            return False  # startup training running in background
        return (time.time() - self._last_retrain) > ML_RETRAIN_INTERVAL

    # ── ML training ───────────────────────────────────────────────────────────

    def _train_ml_background(self) -> None:
        """
        Train both intraday (5M) and daily ML models at startup.
        Daily data served from fetch_batch_interval cache (fetched during first scan).
        We wait briefly to allow the first scan's data fetches to warm the cache.
        """
        logger.info("ML training: waiting for first scan to warm data cache…")
        time.sleep(30)   # let the first scan + cache-fill complete first

        logger.info("ML training starting (intraday + daily models)…")
        daily_data = fetch_batch_interval(NASDAQ_TICKERS, "1day", 500, ttl=CACHE_TTL_1D)
        retrain_all(NASDAQ_TICKERS, daily_data=daily_data)
        self._last_retrain = time.time()
        logger.info("ML training complete.")

    # ── Scan loop ─────────────────────────────────────────────────────────────

    def run_once(self) -> list[StockSignal]:
        t0 = time.time()
        logger.info(f"Scan starting — {len(NASDAQ_TICKERS)} tickers…")

        # Always-fresh 1M data (live signal)
        batch_1m = fetch_batch_realtime(NASDAQ_TICKERS)

        # Cached higher-TF data (only refetched when TTL expires)
        batch_5m = fetch_batch_interval(NASDAQ_TICKERS, "5min", 500,  ttl=CACHE_TTL_5M)
        batch_1h = fetch_batch_interval(NASDAQ_TICKERS, "1h",   500,  ttl=CACHE_TTL_1H)
        batch_1d = fetch_batch_interval(NASDAQ_TICKERS, "1day", 500,  ttl=CACHE_TTL_1D)

        results = []
        for ticker in NASDAQ_TICKERS:
            sig = analyse_ticker(
                ticker,
                df_1m = batch_1m.get(ticker),
                df_5m = batch_5m.get(ticker, pd.DataFrame()),
                df_1h = batch_1h.get(ticker, pd.DataFrame()),
                df_1d = batch_1d.get(ticker, pd.DataFrame()),
            )
            if sig:
                results.append(sig)

        results.sort(key=lambda s: abs(s.score), reverse=True)
        self.signals   = results
        self.last_scan = datetime.utcnow().isoformat()
        self._notify(results)
        elapsed = round(time.time() - t0, 1)
        logger.info(f"Scan complete in {elapsed}s | {len(results)}/{len(NASDAQ_TICKERS)} tickers analysed")

        if self._should_retrain():
            logger.info("Scheduled ML retrain starting…")
            daily_data = fetch_batch_interval(NASDAQ_TICKERS, "1day", 500, ttl=CACHE_TTL_1D)
            retrain_all(NASDAQ_TICKERS, daily_data=daily_data)
            self._last_retrain = time.time()
            logger.info("Scheduled ML retrain complete.")

        return results

    def _loop(self) -> None:
        self.is_running = True
        while self.is_running:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Scanner loop error: {e}", exc_info=True)
            time.sleep(SCAN_INTERVAL_SECONDS)

    def start_background(self) -> None:
        # Thread 1: ML training (waits 30s for first scan to warm cache)
        ml_thread = threading.Thread(target=self._train_ml_background, daemon=True)
        ml_thread.start()

        # Thread 2: Main scan loop (starts immediately)
        scan_thread = threading.Thread(target=self._loop, daemon=True)
        scan_thread.start()

        logger.info("Scanner started. Grow-377: 50 tickers, 1-min scan interval.")

    def stop(self) -> None:
        self.is_running = False


scanner = Scanner()
