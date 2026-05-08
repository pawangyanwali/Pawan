import time
import random
import yfinance as yf
import pandas as pd
import logging
from config import INTRADAY_INTERVAL, REALTIME_INTERVAL, DATA_PERIOD_DAYS

logger = logging.getLogger(__name__)

# Batch size: Yahoo Finance rate-limits large bursts — keep batches small
BATCH_SIZE   = 10
BATCH_DELAY  = 2.5   # seconds between batches
RETRY_DELAY  = 5.0   # seconds before retrying a failed ticker


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns produced by multi-ticker yf.download calls."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _download_with_retry(ticker: str, period: str, interval: str, retries: int = 3) -> pd.DataFrame:
    """Download a single ticker with retry and exponential backoff."""
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                # Suppress yfinance's own error logs on retry
            )
            if df is not None and not df.empty:
                return _flatten(df.dropna())
        except Exception as e:
            logger.debug(f"[{ticker}] attempt {attempt+1} failed: {e}")
        wait = RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
        time.sleep(wait)
    return pd.DataFrame()


def fetch_historical(ticker: str) -> pd.DataFrame:
    """Fetch multi-day intraday OHLCV for ML training (up to 59 days of 5-min bars)."""
    df = _download_with_retry(ticker, period=f"{DATA_PERIOD_DAYS}d", interval=INTRADAY_INTERVAL)
    if df.empty:
        logger.warning(f"[{ticker}] no historical data returned")
    return df


def fetch_realtime(ticker: str) -> pd.DataFrame:
    """Fetch latest 1-day 1-min bars for a single ticker."""
    return _download_with_retry(ticker, period="1d", interval=REALTIME_INTERVAL)


def fetch_batch_realtime(tickers: list) -> dict:
    """
    Fetch 1-min intraday for many tickers without triggering Yahoo rate limits.
    Splits into small batches, pauses between each batch, and falls back to
    single-ticker downloads for any tickers that failed in the batch call.
    """
    result: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    # ── Batched multi-ticker download ─────────────────────────────────────────
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        logger.info(f"Fetching batch {i//BATCH_SIZE + 1}: {batch}")
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
                # Single-ticker download doesn't add an extra level
                df = _flatten(raw).dropna()
                if not df.empty:
                    result[batch[0]] = df
                else:
                    failed.append(batch[0])
            else:
                for t in batch:
                    try:
                        df = raw[t].dropna() if isinstance(raw.columns, pd.MultiIndex) else raw.dropna()
                        if not df.empty:
                            result[t] = _flatten(df)
                        else:
                            failed.append(t)
                    except Exception:
                        failed.append(t)

        except Exception as e:
            logger.warning(f"Batch download failed ({batch}): {e}")
            failed.extend(batch)

        # Pause between batches to respect Yahoo's rate limit
        if i + BATCH_SIZE < len(tickers):
            time.sleep(BATCH_DELAY)

    # ── Retry failed tickers individually ─────────────────────────────────────
    if failed:
        logger.info(f"Retrying {len(failed)} failed tickers individually…")
        for t in failed:
            df = _download_with_retry(t, period="1d", interval=REALTIME_INTERVAL)
            if not df.empty:
                result[t] = df
            time.sleep(0.5 + random.uniform(0, 0.5))

    logger.info(f"Batch fetch complete: {len(result)}/{len(tickers)} tickers retrieved")
    return result


def fetch_news(ticker: str) -> list:
    """Return latest news items for a ticker via yfinance."""
    try:
        t = yf.Ticker(ticker)
        return (t.news or [])[:10]
    except Exception:
        return []


def fetch_ticker_info(ticker: str) -> dict:
    """Return basic info (name, sector, market cap) for a ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return {
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "market_cap": info.get("marketCap", 0),
        }
    except Exception:
        return {"name": ticker, "sector": "Unknown", "market_cap": 0}
