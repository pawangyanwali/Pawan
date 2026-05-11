"""
Main scanning engine.
Runs continuously in a background thread, emitting StockSignal objects
that the FastAPI WebSocket layer broadcasts to all connected clients.
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
    DAILY_CACHE_TTL,
)
from agent.data_fetcher import (
    fetch_batch_realtime,
    fetch_batch_daily,
    fetch_ticker_info,
)
from agent.technical import compute_indicators, score_technical
from agent.volume import score_volume, relative_volume, detect_unusual_volume
from agent.ml_model import predict, predict_daily, retrain_all
from agent.sentiment import score_sentiment
from agent.prediction import generate_prediction
from agent.mtf_analysis import multi_timeframe_analysis

logger = logging.getLogger(__name__)


# ── Module-level daily OHLCV cache (shared across all analyse_ticker calls) ───

_daily_data: dict[str, pd.DataFrame] = {}
_daily_fetched_at: float = 0.0


def _get_daily(ticker: str) -> pd.DataFrame:
    return _daily_data.get(ticker, pd.DataFrame())


# ── StockSignal dataclass ─────────────────────────────────────────────────────

@dataclass
class StockSignal:
    # ── Core price data ───────────────────────────────────────────────────────
    ticker:       str
    name:         str
    price:        float
    change_pct:   float

    # ── Component scores [-1, +1] ─────────────────────────────────────────────
    technical:    float
    volume:       float
    ml_prob:      float       # combined (40% daily ML + 60% intraday ML)
    ml_daily_prob: float      # daily model probability separately
    sentiment:    float
    score:        float

    # ── Legacy signal label ────────────────────────────────────────────────────
    signal:       str

    # ── Volume flags ──────────────────────────────────────────────────────────
    rel_volume:   float
    unusual_vol:  bool

    # ── Professional prediction ───────────────────────────────────────────────
    prediction:        str    # STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL
    confidence:        float  # 0–100
    trend:             str    # UPTREND / DOWNTREND / SIDEWAYS
    trend_probability: float  # fraction of trend sub-signals that agree (0–1)
    ml_trained:        bool   # False until XGBoost intraday model has been trained
    target_price:      float
    stop_loss:         float
    rr_ratio:          float
    patterns:          list = field(default_factory=list)
    reasons:           list = field(default_factory=list)

    # ── Multi-timeframe analysis ───────────────────────────────────────────────
    mtf_score:      float = 0.0
    mtf_alignment:  str   = "MIXED"   # STRONGLY BULLISH/BULLISH/MIXED/BEARISH/STRONGLY BEARISH
    mtf_bull_count: int   = 0         # number of timeframes with UPTREND
    mtf_bear_count: int   = 0         # number of timeframes with DOWNTREND
    mtf_timeframes: dict  = field(default_factory=dict)  # per-TF breakdown

    # ── Support / Resistance ──────────────────────────────────────────────────
    supports:     list = field(default_factory=list)
    resistances:  list = field(default_factory=list)
    pivots:       dict = field(default_factory=dict)
    poc:          float = 0.0

    # ── Chart candles (last 80 × 1-min bars) ─────────────────────────────────
    candles:      list = field(default_factory=list)

    # ── News ──────────────────────────────────────────────────────────────────
    headlines:    list = field(default_factory=list)
    scanned_at:   str  = field(default_factory=lambda: datetime.utcnow().isoformat())

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
    """Serialise the last N OHLCV bars for Lightweight Charts."""
    tail    = df.tail(n)
    candles = []
    for ts, row in tail.iterrows():
        try:
            t = int(pd.Timestamp(ts).timestamp())
            candles.append({
                "time":   t,
                "open":   round(float(row["Open"]),   4),
                "high":   round(float(row["High"]),   4),
                "low":    round(float(row["Low"]),    4),
                "close":  round(float(row["Close"]),  4),
                "volume": int(row["Volume"]),
            })
        except Exception:
            pass
    return candles


# ── Single ticker analysis ────────────────────────────────────────────────────

def analyse_ticker(ticker: str, df_1m) -> Optional[StockSignal]:
    try:
        if df_1m is None or df_1m.empty or len(df_1m) < 5:
            return None

        df_daily = _get_daily(ticker)

        df_ind = compute_indicators(df_1m.copy())
        last   = df_ind.iloc[-1]

        price      = float(last["Close"])
        open_price = float(df_ind.iloc[0]["Open"])
        change_pct = round((price - open_price) / open_price * 100, 3) if open_price else 0.0

        tech             = score_technical(last)
        vol              = score_volume(df_ind)
        ml_scalp         = predict(ticker, df_ind)
        ml_daily_p       = predict_daily(ticker, df_daily) if not df_daily.empty else 0.5
        # Blend: 40% daily model (swing context) + 60% intraday model (scalp timing)
        ml_combined      = round(0.4 * ml_daily_p + 0.6 * ml_scalp, 4)
        sent, headlines  = score_sentiment(ticker)
        rvol             = relative_volume(df_ind)
        uvol             = detect_unusual_volume(df_ind)

        # Multi-timeframe analysis (resample 1M → 5M/15M/30M/1H + daily)
        mtf = multi_timeframe_analysis(df_1m, df_daily)

        # Professional prediction
        pred = generate_prediction(
            ticker, df_ind, tech, vol, ml_combined, sent, last,
            mtf_score=mtf["mtf_score"],
        )

        score   = round(float(np.clip(pred["composite_score"], -1, 1)), 4)
        signal  = pred["direction"]
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
            signal            = signal,
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
            candles           = candles,
            headlines         = headlines[:5],
        )
    except Exception as e:
        logger.warning(f"[{ticker}] analysis error: {e}", exc_info=True)
        return None


# ── Scanner loop ──────────────────────────────────────────────────────────────

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
            return False  # startup training already running in its own thread
        return (time.time() - self._last_retrain) > ML_RETRAIN_INTERVAL

    # ── Daily data refresh ────────────────────────────────────────────────────

    def _fetch_daily_data(self) -> None:
        global _daily_data, _daily_fetched_at
        logger.info("Fetching daily OHLCV for all tickers (6-month history)…")
        result = fetch_batch_daily(NASDAQ_TICKERS)
        _daily_data.update(result)
        _daily_fetched_at = time.time()
        logger.info(f"Daily data ready: {len(_daily_data)} tickers")

    def _daily_refresh_loop(self) -> None:
        """Fetch/refresh daily data at startup and every 24 hours."""
        while True:
            try:
                self._fetch_daily_data()
            except Exception as e:
                logger.error(f"Daily data fetch error: {e}", exc_info=True)
            time.sleep(DAILY_CACHE_TTL)

    # ── ML training ───────────────────────────────────────────────────────────

    def _train_ml_background(self) -> None:
        """Wait for daily data, then train both intraday and daily ML models."""
        # Give the daily data thread up to 15 min to fetch
        for _ in range(180):
            if _daily_fetched_at > 0:
                break
            time.sleep(5)

        logger.info("ML initial training starting (intraday + daily models)…")
        retrain_all(NASDAQ_TICKERS, daily_data=_daily_data)
        self._last_retrain = time.time()
        logger.info("ML initial training complete.")

    # ── Scan loop ─────────────────────────────────────────────────────────────

    def run_once(self) -> list[StockSignal]:
        logger.info(f"Starting scan of {len(NASDAQ_TICKERS)} tickers…")
        batch   = fetch_batch_realtime(NASDAQ_TICKERS)
        results = []

        for ticker in NASDAQ_TICKERS:
            sig = analyse_ticker(ticker, batch.get(ticker))
            if sig:
                results.append(sig)

        results.sort(key=lambda s: abs(s.score), reverse=True)
        self.signals   = results
        self.last_scan = datetime.utcnow().isoformat()
        self._notify(results)
        logger.info(f"Scan complete | {len(results)} stocks analysed")

        if self._should_retrain():
            logger.info("Daily ML retrain starting…")
            retrain_all(NASDAQ_TICKERS, daily_data=_daily_data)
            self._last_retrain = time.time()
            logger.info("Daily ML retrain complete.")

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
        # Thread 1: Fetch and periodically refresh daily OHLCV data
        daily_thread = threading.Thread(target=self._daily_refresh_loop, daemon=True)
        daily_thread.start()

        # Thread 2: Train ML models once daily data is ready
        ml_thread = threading.Thread(target=self._train_ml_background, daemon=True)
        ml_thread.start()

        # Thread 3: Main scan loop
        scan_thread = threading.Thread(target=self._loop, daemon=True)
        scan_thread.start()

        logger.info("Scanner started: daily-data, ML-training, and scan threads running.")

    def stop(self) -> None:
        self.is_running = False


scanner = Scanner()
