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
TICKER_DELAY = 1.2

# ── Yahoo Finance session with crumb auth ─────────────────────────────────────
# Yahoo Finance requires a valid "crumb" + consent cookie since 2023.
# We fetch it once on startup and reuse it for all requests.

_SESSION: requests.Session | None = None
_CRUMB:   str | None = None


def _build_session() -> tuple[requests.Session, str | None]:
    """Create a session with browser headers, accept consent, and extract crumb."""
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
        "Referer":         "https://finance.yahoo.com/",
    })

    crumb = None
    try:
        # Step 1: hit the main page to get initial cookies
        resp = s.get("https://finance.yahoo.com", timeout=15)
        logger.debug(f"Yahoo homepage status: {resp.status_code}")

        # Step 2: accept GDPR/consent if redirected
        if "consent.yahoo.com" in resp.url or resp.status_code in (302, 301):
            consent_url = "https://consent.yahoo.com/v2/collectConsent"
            s.post(consent_url, data={"agree": ["agree", "agree"], "lang": "en-US"}, timeout=10)
            s.get("https://finance.yahoo.com", timeout=10)

        # Step 3: fetch the crumb — needed for API calls since 2023
        crumb_resp = s.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=10)
        if crumb_resp.status_code == 200 and crumb_resp.text.strip():
            crumb = crumb_resp.text.strip()
            logger.info(f"Yahoo Finance crumb obtained: {crumb[:8]}…")
        else:
            logger.warning(f"Crumb fetch returned {crumb_resp.status_code}: '{crumb_resp.text[:80]}'")
    except Exception as e:
        logger.warning(f"Session build error (will continue without crumb): {e}")

    return s, crumb


def _get_session() -> requests.Session:
    global _SESSION, _CRUMB
    if _SESSION is None:
        logger.info("Initialising Yahoo Finance session with crumb…")
        _SESSION, _CRUMB = _build_session()
    return _SESSION


def _reset_session() -> requests.Session:
    global _SESSION, _CRUMB
    _SESSION = None
    _CRUMB = None
    return _get_session()


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).title() for c in df.columns]
    return df


# ── Core download with retry ──────────────────────────────────────────────────

def _ticker_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    session = _get_session()
    t = yf.Ticker(ticker, session=session)
    df = t.history(period=period, interval=interval, auto_adjust=True)
    if df is not None and not df.empty:
        return _normalise_cols(df.dropna())
    return pd.DataFrame()


def _download_with_retry(ticker: str, period: str, interval: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            df = _ticker_history(ticker, period, interval)
            if not df.empty:
                return df
            logger.debug(f"[{ticker}] empty response on attempt {attempt + 1}")
        except Exception as e:
            logger.debug(f"[{ticker}] attempt {attempt + 1} error: {e}")
            if attempt == 1:
                _reset_session()   # refresh cookies once on repeated failure
        wait = 3.0 * (2 ** attempt) + random.uniform(0, 1.5)
        time.sleep(wait)
    logger.warning(f"[{ticker}] all {retries} retries exhausted — skipping")
    return pd.DataFrame()


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_historical(ticker: str) -> pd.DataFrame:
    return _download_with_retry(
        ticker, period=f"{DATA_PERIOD_DAYS}d", interval=INTRADAY_INTERVAL
    )


def fetch_realtime(ticker: str) -> pd.DataFrame:
    return _download_with_retry(ticker, period="1d", interval=REALTIME_INTERVAL)


def fetch_batch_realtime(tickers: list) -> dict:
    """
    Fetch 1-min bars for all tickers.
    Tries yf.download() in small batches first; falls back to per-ticker
    Ticker.history() for any that fail.
    """
    result: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    session = _get_session()
    n_batches = -(-len(tickers) // BATCH_SIZE)

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i: i + BATCH_SIZE]
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
                if not df.empty:
                    result[batch[0]] = df
                else:
                    failed.append(batch[0])
            else:
                for t in batch:
                    try:
                        df = (
                            raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                        ).dropna()
                        if not df.empty:
                            result[t] = _flatten(df)
                        else:
                            failed.append(t)
                    except Exception:
                        failed.append(t)
        except Exception as e:
            logger.warning(f"Batch {i // BATCH_SIZE + 1} failed ({e}) — queuing for individual retry")
            failed.extend(batch)

        if i + BATCH_SIZE < len(tickers):
            time.sleep(BATCH_DELAY)

    # Individual retry for anything that failed
    if failed:
        logger.info(f"Retrying {len(failed)} tickers individually via Ticker.history()…")
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
