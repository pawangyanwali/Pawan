import yfinance as yf
import pandas as pd
import logging
from config import INTRADAY_INTERVAL, REALTIME_INTERVAL, DATA_PERIOD_DAYS

logger = logging.getLogger(__name__)


def fetch_historical(ticker: str) -> pd.DataFrame:
    """Fetch multi-day intraday OHLCV for ML training (up to 59 days of 5-min bars)."""
    try:
        df = yf.download(
            ticker,
            period=f"{DATA_PERIOD_DAYS}d",
            interval=INTRADAY_INTERVAL,
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df.dropna(inplace=True)
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        logger.warning(f"[{ticker}] historical fetch error: {e}")
        return pd.DataFrame()


def fetch_realtime(ticker: str) -> pd.DataFrame:
    """Fetch latest 1-day 1-min bars for live signal generation."""
    try:
        df = yf.download(
            ticker,
            period="1d",
            interval=REALTIME_INTERVAL,
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df.dropna(inplace=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        logger.warning(f"[{ticker}] realtime fetch error: {e}")
        return pd.DataFrame()


def fetch_batch_realtime(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch 1-min intraday for multiple tickers in one yfinance call (faster)."""
    try:
        raw = yf.download(
            tickers,
            period="1d",
            interval=REALTIME_INTERVAL,
            group_by="ticker",
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        logger.error(f"Batch realtime fetch error: {e}")
        return {}

    result: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[t].dropna()
            else:
                df = raw.dropna()
            if not df.empty:
                result[t] = df
        except Exception:
            pass
    return result


def fetch_news(ticker: str) -> list[dict]:
    """Return latest news items for a ticker via yfinance."""
    try:
        info = yf.Ticker(ticker)
        news = info.news or []
        return news[:10]
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
