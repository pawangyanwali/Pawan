"""
Main scanning engine.
Runs continuously in a background thread, emitting StockSignal objects
that the FastAPI WebSocket layer broadcasts to all connected clients.
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Callable

import numpy as np

from config import (
    NASDAQ_TICKERS,
    SCAN_INTERVAL_SECONDS,
    ML_RETRAIN_INTERVAL,
    WEIGHT_TECHNICAL,
    WEIGHT_VOLUME,
    WEIGHT_ML,
    WEIGHT_SENTIMENT,
    STRONG_BUY_THRESHOLD,
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    STRONG_SELL_THRESHOLD,
)
from agent.data_fetcher import fetch_batch_realtime, fetch_ticker_info
from agent.technical import compute_indicators, score_technical
from agent.volume import score_volume, relative_volume, detect_unusual_volume
from agent.ml_model import predict, retrain_all
from agent.sentiment import score_sentiment

logger = logging.getLogger(__name__)


@dataclass
class StockSignal:
    ticker:        str
    name:          str
    price:         float
    change_pct:    float          # % change from prev close
    technical:     float          # [-1, +1]
    volume:        float          # [-1, +1]
    ml_prob:       float          # [0, 1] probability of going up
    sentiment:     float          # [-1, +1]
    score:         float          # [-1, +1] composite
    signal:        str            # "STRONG BUY" … "STRONG SELL"
    rel_volume:    float          # x times average volume
    unusual_vol:   bool
    headlines:     list[str]      = field(default_factory=list)
    scanned_at:    str            = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


def _label_signal(score: float) -> str:
    if score >= STRONG_BUY_THRESHOLD:
        return "STRONG BUY"
    elif score >= BUY_THRESHOLD:
        return "BUY"
    elif score <= STRONG_SELL_THRESHOLD:
        return "STRONG SELL"
    elif score <= SELL_THRESHOLD:
        return "SELL"
    return "NEUTRAL"


def _composite_score(tech: float, vol: float, ml_prob: float, sent: float) -> float:
    ml_score = (ml_prob - 0.5) * 2   # convert [0,1] → [-1,+1]
    score = (
        WEIGHT_TECHNICAL * tech +
        WEIGHT_VOLUME    * vol  +
        WEIGHT_ML        * ml_score +
        WEIGHT_SENTIMENT * sent
    )
    return round(float(np.clip(score, -1, 1)), 4)


# ── Ticker metadata cache ─────────────────────────────────────────────────────

_info_cache: dict[str, dict] = {}


def _get_info(ticker: str) -> dict:
    if ticker not in _info_cache:
        _info_cache[ticker] = fetch_ticker_info(ticker)
    return _info_cache[ticker]


# ── Single ticker analysis ────────────────────────────────────────────────────

def analyse_ticker(ticker: str, df) -> Optional[StockSignal]:
    try:
        if df is None or df.empty or len(df) < 30:
            return None

        df = compute_indicators(df.copy())
        last = df.iloc[-1]

        price      = float(last["Close"])
        open_price = float(df.iloc[0]["Open"]) if len(df) > 0 else price
        change_pct = round((price - open_price) / open_price * 100, 3) if open_price else 0.0

        tech   = score_technical(last)
        vol    = score_volume(df)
        ml     = predict(ticker, df)
        sent, headlines = score_sentiment(ticker)

        score  = _composite_score(tech, vol, ml, sent)
        signal = _label_signal(score)
        info   = _get_info(ticker)
        rvol   = relative_volume(df)
        uvol   = detect_unusual_volume(df)

        return StockSignal(
            ticker=ticker,
            name=info.get("name", ticker),
            price=round(price, 4),
            change_pct=change_pct,
            technical=round(tech, 4),
            volume=round(vol, 4),
            ml_prob=round(ml, 4),
            sentiment=round(sent, 4),
            score=score,
            signal=signal,
            rel_volume=rvol,
            unusual_vol=uvol,
            headlines=headlines[:5],
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
        return (time.time() - self._last_retrain) > ML_RETRAIN_INTERVAL

    def run_once(self) -> list[StockSignal]:
        logger.info(f"Starting scan of {len(NASDAQ_TICKERS)} tickers…")

        # Fetch price data first so dashboard gets results quickly,
        # then retrain ML models in the background afterwards.
        batch = fetch_batch_realtime(NASDAQ_TICKERS)
        results: list[StockSignal] = []

        for ticker in NASDAQ_TICKERS:
            df = batch.get(ticker)
            sig = analyse_ticker(ticker, df)
            if sig:
                results.append(sig)

        # Sort by absolute score descending (strongest signals first)
        results.sort(key=lambda s: abs(s.score), reverse=True)
        self.signals   = results
        self.last_scan = datetime.utcnow().isoformat()
        self._notify(results)
        logger.info(f"Scan complete | {len(results)} stocks analysed")

        # Retrain ML models after broadcasting results (non-blocking for dashboard)
        if self._should_retrain():
            logger.info("Starting ML retrain in background (1.2s delay between tickers)…")
            retrain_all(NASDAQ_TICKERS)
            self._last_retrain = time.time()
            logger.info("ML retrain complete.")

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
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        logger.info("Scanner background thread started.")

    def stop(self) -> None:
        self.is_running = False


# ── Singleton ─────────────────────────────────────────────────────────────────
scanner = Scanner()
