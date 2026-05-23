"""
Core fetch/resample logic for the historical backfill service.

Strategy
--------
1. Fetch 1min data per ticker in 9-calendar-day chunks (safe under Schwab limits).
   Store raw 1min bars immediately so progress survives crashes.
2. After all 1min chunks for a ticker are done, resample to 5min / 15min / 30min / 1h / 4h
   and store all derived intervals.  Resampling per-ticker avoids OHLC boundary
   artefacts that occur when resampling across chunk seams.
3. Fetch 1day separately — single request per ticker covers 2+ years.
"""

import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from agent.broker.schwab_market_data import fetch_price_history_range
from historical import progress, store
from historical.schema import RESAMPLE_FROM_1MIN

logger = logging.getLogger(__name__)


def check_auth() -> bool:
    """
    Verify that the Schwab Market Data token is valid before starting the backfill.
    Fetches one known ticker for a recent 1-day window as a live probe.
    Returns True if data comes back, False if auth is broken.
    """
    from datetime import timezone
    probe_end   = int((datetime.now(timezone.utc) - timedelta(days=2)).timestamp() * 1000)
    probe_start = int((datetime.now(timezone.utc) - timedelta(days=4)).timestamp() * 1000)
    df = fetch_price_history_range("SPY", "1min", probe_start, probe_end)
    if df.empty:
        # Try to get a more specific error from auth status
        try:
            from agent.broker.schwab_market_data import _is_authorised, _auth_headers
            authorised = _is_authorised()
            headers    = _auth_headers()
            logger.error(
                "[Backfill] Auth probe returned empty. _is_authorised=%s, headers=%s",
                authorised, "present" if headers else "MISSING",
            )
        except Exception as exc:
            logger.error("[Backfill] Auth probe failed: %s", exc)
        return False
    logger.info("[Backfill] Auth OK — SPY probe returned %d bars", len(df))
    return True


# ── Rate limiting ──────────────────────────────────────────────────────────────
# Backfill runs at 0.5 req/s by default (one request every 2s).
# This is much slower than the live scanner so both can run concurrently
# without triggering Schwab's Akamai CDN IP-level rate limiter.
_REQ_GAP = 2.0          # seconds between API calls
_last_req: float = 0.0


def _throttle() -> None:
    global _last_req
    elapsed = time.time() - _last_req
    if elapsed < _REQ_GAP:
        time.sleep(_REQ_GAP - elapsed)
    _last_req = time.time()


def _wait_if_blocked(pause_s: int = 30) -> None:
    """If Schwab's CDN backoff is active, sleep until it clears (+ pause_s extra)."""
    try:
        from agent.broker.schwab_market_data import _backoff_until
        remaining = _backoff_until - time.time()
        if remaining > 0:
            wait = remaining + pause_s
            logger.warning(
                "[Backfill] Schwab CDN block active (%.0fs remaining) — pausing %.0fs",
                remaining, wait,
            )
            time.sleep(wait)
    except Exception:
        pass


# ── Chunk generation ───────────────────────────────────────────────────────────

def date_chunks(
    start_dt: datetime,
    end_dt:   datetime,
    chunk_days: int = 9,
    newest_first: bool = True,
) -> list[tuple[int, int]]:
    """
    Return list of (start_ms, end_ms) epoch-ms pairs covering [start_dt, end_dt].
    Each window is chunk_days calendar days wide (9 days ≈ 6-7 trading days,
    safely below Schwab's apparent 10-trading-day limit for minute endpoints).

    newest_first=True (default): chunks ordered recent → old so we capture all
    available 1min history before hitting the API's lookback limit (~30 days).
    The early-abort in fetch_1min_ticker then cleanly stops when the API returns
    empty data for older dates rather than aborting before any data is fetched.
    """
    chunks = []
    cur = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end = end_dt.replace(hour=23, minute=59, second=59, microsecond=0)
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days), end)
        chunks.append((
            int(cur.timestamp() * 1000),
            int(chunk_end.timestamp() * 1000),
        ))
        cur = chunk_end + timedelta(days=1)
    if newest_first:
        chunks.reverse()
    return chunks


# ── Fetch 1min ─────────────────────────────────────────────────────────────────

def fetch_1min_ticker(
    ticker:     str,
    chunks:     list[tuple[int, int]],
    empty_limit: int = 5,
) -> int:
    """
    Fetch all 1min chunks for one ticker.  Returns total bars stored.
    Skips already-done chunks (resume-safe).
    Aborts early if Schwab returns empty data for `empty_limit` consecutive
    chunks (indicates the ticker has no history that far back).
    If the CDN is blocking, waits once then skips the ticker entirely
    rather than hammering in a retry loop.
    """
    stored = 0
    consecutive_empty = 0

    for start_ms, end_ms in chunks:
        if progress.is_chunk_done(ticker, "1min", start_ms):
            consecutive_empty = 0
            continue

        _wait_if_blocked()
        _throttle()
        df = fetch_price_history_range(ticker, "1min", start_ms, end_ms)

        if df.empty:
            # Check if this is a CDN block vs. genuinely no data for this date
            try:
                from agent.broker.schwab_market_data import _backoff_until
                if _backoff_until > time.time():
                    logger.warning(
                        "[Backfill] %s: CDN block detected — skipping ticker, will resume next run",
                        ticker,
                    )
                    return stored   # leave chunks unmarked so they retry next run
            except Exception:
                pass

            consecutive_empty += 1
            logger.debug("[Backfill] %s 1min chunk %s empty (%d)", ticker,
                         _ms_label(start_ms), consecutive_empty)
            progress.mark_chunk_done(ticker, "1min", start_ms)
            if consecutive_empty >= empty_limit:
                logger.info("[Backfill] %s: %d consecutive empty chunks — stopping early",
                            ticker, empty_limit)
                break
            continue

        consecutive_empty = 0
        n = store.upsert_bars("1min", ticker, df)
        stored += n
        progress.mark_chunk_done(ticker, "1min", start_ms)
        logger.debug("[Backfill] %s 1min chunk %s: +%d bars", ticker, _ms_label(start_ms), n)

    return stored


# ── Resample 1min → derived intervals ─────────────────────────────────────────

def resample_ticker(ticker: str) -> dict[str, int]:
    """
    Load all stored 1min bars for a ticker, resample to every derived interval,
    and upsert.  Returns dict of {interval: bars_stored}.
    """
    df_1min = store.read_ticker_bars("1min", ticker)
    if df_1min.empty:
        return {}

    results: dict[str, int] = {}
    for interval, rule in RESAMPLE_FROM_1MIN.items():
        resampled = df_1min.resample(rule).agg({
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }).dropna(subset=["Open", "Close"])
        n = store.upsert_bars(interval, ticker, resampled)
        results[interval] = n

    progress.mark_resampled(ticker)
    return results


# ── Fetch 1day ─────────────────────────────────────────────────────────────────

def fetch_daily_ticker(ticker: str, start_ms: int, end_ms: int) -> int:
    """Fetch 1day bars for one ticker covering the full backfill window."""
    if progress.is_daily_done(ticker):
        return 0
    _throttle()
    df = fetch_price_history_range(ticker, "1day", start_ms, end_ms)
    n = store.upsert_bars("1day", ticker, df) if not df.empty else 0
    progress.mark_daily_done(ticker)
    logger.debug("[Backfill] %s 1day: +%d bars", ticker, n)
    return n


# ── Main coordinator ───────────────────────────────────────────────────────────

def run(
    tickers:      list[str],
    years:        int   = 2,
    rate_s:       float = 2.0,
    resample_only: bool = False,
    daily_only:    bool = False,
) -> None:
    """
    Run the full backfill for a list of tickers.

    Phases:
      1. Fetch 1min (chunked, resumable)
      2. Resample 1min → 5min / 15min / 30min / 1h / 4h
      3. Fetch 1day
    """
    global _REQ_GAP
    _REQ_GAP = rate_s

    now_utc   = datetime.now(timezone.utc)
    start_dt  = now_utc - timedelta(days=365 * years)
    end_dt    = now_utc - timedelta(days=1)

    start_ms  = int(start_dt.timestamp() * 1000)
    end_ms    = int(end_dt.timestamp()   * 1000)

    chunks    = date_chunks(start_dt, end_dt)
    n_tickers = len(tickers)
    n_chunks  = len(chunks)

    logger.info(
        "[Backfill] %d tickers | %d years | %d chunks/ticker | ~%d total API calls",
        n_tickers, years, n_chunks, n_tickers * (n_chunks + 1),
    )

    # ── Auth probe ───────────────────────────────────────────────────────────
    if not resample_only:
        logger.info("[Backfill] Checking Schwab Market Data auth...")
        if not check_auth():
            logger.error(
                "[Backfill] ABORTING — Schwab Market Data token is missing or expired.\n"
                "  The token is owned by the running nasdaq-agent service.\n"
                "  Make sure the service is active: sudo systemctl status nasdaq-agent\n"
                "  If the service is running, wait 30s for token refresh and try again.\n"
                "  Token file location: ~/.nasdaq_agent/schwab_md_*.json"
            )
            return

    # ── Phase 1: 1min fetch ──────────────────────────────────────────────────
    if not resample_only and not daily_only:
        logger.info("[Backfill] Phase 1: fetching 1min data")
        for i, ticker in enumerate(tickers, 1):
            done_chunks = sum(
                1 for s, _ in chunks if progress.is_chunk_done(ticker, "1min", s)
            )
            if done_chunks == len(chunks):
                logger.debug("[Backfill] %s 1min already complete — skip", ticker)
                continue
            logger.info("[Backfill] [%d/%d] %s: fetching 1min (%d/%d chunks done)",
                        i, n_tickers, ticker, done_chunks, n_chunks)
            bars = fetch_1min_ticker(ticker, chunks)
            logger.info("[Backfill] [%d/%d] %s: +%d bars stored", i, n_tickers, ticker, bars)

    # ── Phase 2: resample ────────────────────────────────────────────────────
    if not daily_only:
        logger.info("[Backfill] Phase 2: resampling 1min → derived intervals")
        for i, ticker in enumerate(tickers, 1):
            if progress.is_resampled(ticker):
                continue
            result = resample_ticker(ticker)
            if result:
                logger.info("[Backfill] [%d/%d] %s resampled: %s",
                            i, n_tickers, ticker,
                            ", ".join(f"{iv}={n}" for iv, n in result.items()))

    # ── Phase 3: 1day fetch ──────────────────────────────────────────────────
    logger.info("[Backfill] Phase 3: fetching 1day data")
    for i, ticker in enumerate(tickers, 1):
        _wait_if_blocked()
        n = fetch_daily_ticker(ticker, start_ms, end_ms)
        if n:
            logger.info("[Backfill] [%d/%d] %s 1day: +%d bars", i, n_tickers, ticker, n)

    logger.info("[Backfill] All phases complete.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ms_label(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def estimate_time(tickers: list[str], years: int, rate_s: float = _REQ_GAP) -> str:
    """Human-readable time estimate for a full backfill run."""
    n_chunks   = len(date_chunks(
        datetime.now(timezone.utc) - timedelta(days=365 * years),
        datetime.now(timezone.utc) - timedelta(days=1),
    ))
    total_reqs = len(tickers) * (n_chunks + 1)   # +1 for daily
    secs       = total_reqs * rate_s
    h, rem     = divmod(int(secs), 3600)
    m          = rem // 60
    return f"~{h}h {m}m  ({total_reqs:,} API calls at {1/rate_s:.1f} req/s)"
