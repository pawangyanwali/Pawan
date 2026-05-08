import time
import random
import yfinance as yf
import pandas as pd
import logging
from config import INTRADAY_INTERVAL, REALTIME_INTERVAL, DATA_PERIOD_DAYS

logger = logging.getLogger(__name__)

BATCH_SIZE  = 10
BATCH_DELAY = 3.0   # seconds between batches
TICKER_DELAY = 0.8  # seconds between individual ticker requests


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _history_with_retry(ticker: str, period: str, interval: str, retries: int = 3) -> pd.DataFrame:
    """
    Use Ticker.history() — more reliable than yf.download() for single tickers.
    Retries with exponential backoff on failure.
    """
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval, auto_adjust=True)
            if df is not None and not df.empty:
                df = df.dropna()
                # Normalise column names to match yf.download() output
                df.columns = [c.title() for c in df.columns]
                return df
        except Exception as e:
            logger.debug(f"[{ticker}] attempt {attempt+1} failed: {e}")
        wait = 5.0 * (2 ** attempt) + random.uniform(0, 1.5)
        logger.debug(f"[{ticker}] waiting {wait:.1f}s before retry {attempt+2}")
        time.sleep(wait)
    logger.warning(f"[{ticker}] all retries exhausted — returning empty DataFrame")
    return pd.DataFrame()


def fetch_historical(ticker: str) -> pd.DataFrame:
    """Fetch multi-day intraday OHLCV for ML training."""
    return _history_with_retry(ticker, period=f"{DATA_PERIOD_DAYS}d", interval=INTRADAY_INTERVAL)


def fetch_realtime(ticker: str) -> pd.DataFrame:
    """Fetch latest 1-day bars for a single ticker."""
    return _history_with_retry(ticker, period="1d", interval=REALTIME_INTERVAL)


def fetch_batch_realtime(tickers: list) -> dict:
    """
    Fetch 1-min intraday for many tickers.
    Uses small batches via yf.download; falls back to Ticker.history() per ticker.
    """
    result: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i: i + BATCH_SIZE]
        logger.info(f"Fetching batch {i // BATCH_SIZE + 1}/{-(-len(tickers)//BATCH_SIZE)}: {batch}")
        try:
            raw = yf.download(
                batch,
                period="1d",
                interval=REALTIME_INTERVAL,
                group_by="ticker",
                progress=False,
                auto_adjust=True,
            )
            if raw is None or raw.empty:
                failed.extend(batch)
            elif len(batch) == 1:
                df = _flatten(raw).dropna()
                if not df.empty:
                    result[batch[0]] = df
                else:
                    failed.append(batch[0])
            else:
                for t in batch:
                    try:
                        df = (raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw).dropna()
                        if not df.empty:
                            result[t] = _flatten(df)
                        else:
                            failed.append(t)
                    except Exception:
                        failed.append(t)
        except Exception as e:
            logger.warning(f"Batch download failed ({batch}): {e}")
            failed.extend(batch)

        if i + BATCH_SIZE < len(tickers):
            time.sleep(BATCH_DELAY)

    # Retry failures individually using Ticker.history()
    if failed:
        logger.info(f"Retrying {len(failed)} failed tickers individually…")
        for t in failed:
            df = _history_with_retry(t, period="1d", interval=REALTIME_INTERVAL, retries=2)
            if not df.empty:
                result[t] = df
            time.sleep(TICKER_DELAY + random.uniform(0, 0.5))

    logger.info(f"Batch fetch complete: {len(result)}/{len(tickers)} tickers OK")
    return result


def fetch_news(ticker: str) -> list:
    try:
        return (yf.Ticker(ticker).news or [])[:10]
    except Exception:
        return []


def fetch_ticker_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "market_cap": info.get("marketCap", 0),
        }
    except Exception:
        return {"name": ticker, "sector": "Unknown", "market_cap": 0}
