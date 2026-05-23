"""
Retrain StockMLModel for all tickers using 2-year historical 5-min bars.

Replaces the live-service retrain path (which pulls ~50 days from Schwab)
with the full historical corpus from the backfill DB — ~40× more data
produces substantially more generalised models.

The training pipeline is identical to the live service: each ticker's bars
are passed to StockMLModel.train_from_df(), which calls prepare_training_data()
(leakage-free split + ATR-adaptive labels) then fits the calibrated XGBoost.
Saved models are written to data/models/ and picked up on the next service restart.

Usage (via __main__.py):
  python -m historical --retrain
  python -m historical --retrain --tickers AAPL,MSFT,NVDA
  python -m historical --retrain --interval 5min --workers 4
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL = "5min"
_DEFAULT_WORKERS  = 4


def retrain_from_history(
    tickers:     list[str],
    interval:    str = _DEFAULT_INTERVAL,
    max_workers: int = _DEFAULT_WORKERS,
) -> dict:
    """
    Retrain StockMLModel for each ticker using stored historical bars.

    Returns summary dict: {total, trained, skipped, elapsed_s}.
    """
    from agent.ml_model import get_or_create
    from historical.store import read_ticker_bars

    t0 = time.time()
    trained = 0
    skipped = 0
    n = len(tickers)

    def _train_one(ticker: str) -> tuple[str, bool, str]:
        try:
            df = read_ticker_bars(interval, ticker)
            if df.empty or len(df) < 200:
                return ticker, False, f"only {len(df)} bars"
            model = get_or_create(ticker)
            ok = model.train_from_df(df)
            return ticker, ok, "" if ok else "train_from_df returned False"
        except Exception as exc:
            return ticker, False, str(exc)

    logger.info("[HistRetrain] Training %d tickers (%s) with %d workers",
                n, interval, max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_train_one, t): t for t in tickers}
        done_count = 0
        for fut in as_completed(futures):
            done_count += 1
            ticker, ok, reason = fut.result()
            if ok:
                trained += 1
                logger.info("[HistRetrain] [%d/%d] %s: trained", done_count, n, ticker)
            else:
                skipped += 1
                logger.warning("[HistRetrain] [%d/%d] %s: skipped — %s",
                               done_count, n, ticker, reason)

    elapsed = round(time.time() - t0, 1)
    summary = {"total": n, "trained": trained, "skipped": skipped, "elapsed_s": elapsed}
    logger.info("[HistRetrain] Done — %d trained, %d skipped in %.0fs",
                trained, skipped, elapsed)
    return summary
