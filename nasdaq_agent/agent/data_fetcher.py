import time
import random
import requests
import yfinance as yf
import pandas as pd
import logging
from config import INTRADAY_INTERVAL, REALTIME_INTERVAL, DATA_PERIOD_DAYS

logger = logging.getLogger(__name__)

BATCH_SIZE   = 10
BATCH_DELAY  = 3.0
TICKER_DELAY = 1.0

# ── Browser-like session ──────────────────────────────────────────────────────
# Yahoo Finance blocks bare urllib requests; passing a session with real browser
# headers (and letting it collect cookies on first hit) fixes JSONDecodeError.

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection":      "keep-alive",
    })
    # Warm up: visit Yahoo Finance once to collect cookies
    try:
        s.get("https://finance.yahoo.com", timeout=10)
    except Exception:
        pass
    return s

_SESSION: requests.Session | None = None

def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        logger.info("Initialising Yahoo Finance session…")
        _SESSION = _make_session()
    return _SESSION

def _reset_session() -> requests.Session:
    global _SESSION
    _SESSION = None
    return _get_session()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _ticker_history(ticker: str, period: str, interval: str, session: requests.Session) -> pd.DataFrame:
    """Fetch via Ticker.history() using shared session."""
    t = yf.Ticker(ticker, session=session)
    df = t.history(period=period, interval=interval, auto_adjust=True)
    if df is not None and not df.empty:
        df = df.dropna()
        df.columns = [c.title() for c in df.columns]
        return df
    return pd.DataFrame()


def _download_with_retry(ticker: str, period: str, interval: str, retries: int = 3) -> pd.DataFrame:
    session = _get_session()
    for attempt in range(retries):
        try:
            df = _ticker_history(ticker, period, interval, session)
            if not df.empty:
                return df
        except Exception as e:
            logger.debug(f"[{ticker}] attempt {attempt+1} error: {e}")
            if attempt == 1:
                # Session may be stale — refresh it once
                session = _reset_session()
        wait = 3.0 * (2 ** attempt) + random.uniform(0, 1.5)
        time.sleep(wait)
    logger.warning(f"[{ticker}] all retries exhausted")
    return pd.DataFrame()


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_historical(ticker: str) -> pd.DataFrame:
    return _download_with_retry(ticker, period=f"{DATA_PERIOD_DAYS}d", interval=INTRADAY_INTERVAL)


def fetch_realtime(ticker: str) -> pd.DataFrame:
    return _download_with_retry(ticker, period="1d", interval=REALTIME_INTERVAL)


def fetch_batch_realtime(tickers: list) -> dict:
    """
    Fetch 1-min intraday for all tickers.
    Uses yf.download() in small batches; falls back to per-ticker Ticker.history().
    """
    result: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    session = _get_session()

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i: i + BATCH_SIZE]
        n_batches = -(-len(tickers) // BATCH_SIZE)
        logger.info(f"Fetching batch {i // BATCH_SIZE + 1}/{n_batches}: {batch}")
        try:
            raw = yf.download(
                batch,
                period="1d",
                interval=REALTIME_INTERVAL,
                group_by="ticker",
                progress=False,
                auto_adjust=True,
                session=session,
            )

            if raw is None or raw.empty:
                failed.extend(batch)
            elif len(batch) == 1:
                df = _flatten(raw).dropna()
                (result if not df.empty else {batch[0]: None})[batch[0]] = df if not df.empty else None
                if df.empty:
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
            logger.warning(f"Batch {i // BATCH_SIZE + 1} failed: {e} — will retry individually")
            failed.extend(batch)

        if i + BATCH_SIZE < len(tickers):
            time.sleep(BATCH_DELAY)

    # Retry failures individually
    if failed:
        logger.info(f"Retrying {len(failed)} failed tickers individually…")
        for t in failed:
            df = _download_with_retry(t, period="1d", interval=REALTIME_INTERVAL, retries=2)
            if not df.empty:
                result[t] = df
            time.sleep(TICKER_DELAY + random.uniform(0, 0.5))

    logger.info(f"Batch fetch complete: {len(result)}/{len(tickers)} tickers OK")
    return result


def fetch_news(ticker: str) -> list:
    try:
        return (yf.Ticker(ticker, session=_get_session()).news or [])[:10]
    except Exception:
        return []


def fetch_ticker_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker, session=_get_session()).info or {}
        return {
            "name":       info.get("shortName", ticker),
            "sector":     info.get("sector", "Unknown"),
            "market_cap": info.get("marketCap", 0),
        }
    except Exception:
        return {"name": ticker, "sector": "Unknown", "market_cap": 0}
