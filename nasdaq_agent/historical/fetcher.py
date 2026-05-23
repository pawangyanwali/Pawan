"""
Core fetch logic for the historical backfill service.

Strategy
--------
Fetch each Schwab-native interval directly to maximise history depth:
  1min  — 9-day chunks    (~50 days available from Schwab)
  5min  — 30-day chunks   (potentially 6+ months)
  15min — 90-day chunks   (potentially 1+ year)
  30min — 180-day chunks  (potentially 1+ year)
  1day  — single request  (2+ years)

After 30min is complete, resample to intervals Schwab does not provide natively:
  1h, 2h, 4h  ← resampled from 30min

All fetches are resumable: each chunk is checkpointed immediately so a crash
mid-run can be resumed without re-fetching completed work.
"""

import logging
import signal
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from agent.broker.schwab_market_data import fetch_price_history_range
from historical import progress, store
from historical.schema import CHUNK_DAYS, DIRECT_INTERVALS, RESAMPLE_FROM_30MIN

logger = logging.getLogger(__name__)

_HARD_TIMEOUT_S = 35   # SIGALRM fires after this many seconds, interrupting recv()


class _HardTimeout(Exception):
    """Raised by SIGALRM handler when an API call exceeds _HARD_TIMEOUT_S."""


def _fetch(ticker: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Call fetch_price_history_range with a SIGALRM hard deadline.

    requests/urllib3 does not reliably honour socket timeouts on pooled
    connections (the timeout is not always reset when a connection is reused).
    SIGALRM fires unconditionally after _HARD_TIMEOUT_S seconds, interrupting
    the blocking recv() syscall at the OS level so the process never stalls
    longer than that wall-clock time.

    Only called from the main thread (backfill is single-threaded);
    signal.alarm() requires the main thread.
    """
    def _handler(signum, frame):
        raise _HardTimeout()

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(_HARD_TIMEOUT_S)
    try:
        return fetch_price_history_range(ticker, interval, start_ms, end_ms)
    except _HardTimeout:
        logger.warning("[Backfill] %s %s %s: hard %ds timeout — treating as empty",
                       ticker, interval, _ms_label(start_ms), _HARD_TIMEOUT_S)
        return pd.DataFrame()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


# ── Auth probe ─────────────────────────────────────────────────────────────────

def check_auth() -> bool:
    """
    Verify the Schwab Market Data token is valid. Retries 3× with 35s gap
    to ride out transient CDN blocks before declaring failure.
    """
    probe_end   = int((datetime.now(timezone.utc) - timedelta(days=2)).timestamp() * 1000)
    probe_start = int((datetime.now(timezone.utc) - timedelta(days=4)).timestamp() * 1000)

    for attempt in range(1, 4):
        df = _fetch("SPY", "1min", probe_start, probe_end)
        if not df.empty:
            logger.info("[Backfill] Auth OK — SPY probe returned %d bars", len(df))
            return True

        try:
            from agent.broker.schwab_market_data import _is_authorised, _auth_headers, _backoff_until
            authorised  = _is_authorised()
            has_headers = bool(_auth_headers())
            blocked_for = max(0.0, _backoff_until - time.time())
        except Exception:
            authorised, has_headers, blocked_for = False, False, 0.0

        if not authorised or not has_headers:
            logger.error("[Backfill] Auth probe failed — token missing or expired.")
            return False

        wait = max(blocked_for + 5, 35)
        logger.warning("[Backfill] CDN block (attempt %d/3) — waiting %.0fs", attempt, wait)
        time.sleep(wait)

    logger.error("[Backfill] CDN block persisted after 3 retries.")
    return False


# ── Rate limiting ──────────────────────────────────────────────────────────────

_REQ_GAP   = 1.0
_last_req: float = 0.0


def _throttle() -> None:
    global _last_req
    elapsed = time.time() - _last_req
    if elapsed < _REQ_GAP:
        time.sleep(_REQ_GAP - elapsed)
    _last_req = time.time()


def _wait_if_blocked(pause_s: int = 30) -> None:
    try:
        from agent.broker.schwab_market_data import _backoff_until
        remaining = _backoff_until - time.time()
        if remaining > 0:
            wait = remaining + pause_s
            logger.warning("[Backfill] CDN block (%.0fs remaining) — pausing %.0fs", remaining, wait)
            time.sleep(wait)
    except Exception:
        pass


# ── Chunk generation ───────────────────────────────────────────────────────────

def date_chunks(
    start_dt:   datetime,
    end_dt:     datetime,
    chunk_days: int = 9,
) -> list[tuple[int, int]]:
    """
    Return (start_ms, end_ms) pairs covering [start_dt, end_dt], newest first.
    Newest-first ordering ensures we capture all available history before the
    API's lookback limit is reached; early-abort on consecutive empties then
    stops cleanly instead of wasting calls on ancient dates.
    """
    chunks = []
    cur = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end = end_dt.replace(hour=23, minute=59, second=59, microsecond=0)
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days), end)
        chunks.append((int(cur.timestamp() * 1000), int(chunk_end.timestamp() * 1000)))
        cur = chunk_end + timedelta(days=1)
    chunks.reverse()
    return chunks


# ── Generic minute-interval fetcher ───────────────────────────────────────────

def fetch_interval_ticker(
    ticker:      str,
    interval:    str,
    chunks:      list[tuple[int, int]],
    empty_limit: int = 5,
) -> int:
    """
    Fetch all chunks for one ticker/interval. Returns total bars stored.
    Skips already-done chunks (resume-safe).
    Stops early after `empty_limit` consecutive empty chunks — signals we have
    reached Schwab's lookback boundary for this interval.
    Skips the ticker entirely on a CDN block so chunks remain retryable.
    """
    stored = 0
    consecutive_empty = 0

    for start_ms, end_ms in chunks:
        if progress.is_chunk_done(ticker, interval, start_ms):
            consecutive_empty = 0
            continue

        _wait_if_blocked()
        _throttle()
        logger.info("[Backfill] %s %s %s: requesting...", ticker, interval, _ms_label(start_ms))
        df = _fetch(ticker, interval, start_ms, end_ms)
        logger.info("[Backfill] %s %s %s: got %d rows", ticker, interval, _ms_label(start_ms), len(df))

        if df.empty:
            try:
                from agent.broker.schwab_market_data import _backoff_until
                if _backoff_until > time.time():
                    logger.warning("[Backfill] %s %s: CDN block — skipping, will retry next run",
                                   ticker, interval)
                    return stored
            except Exception:
                pass

            consecutive_empty += 1
            logger.info("[Backfill] %s %s %s: empty (%d consecutive)",
                        ticker, interval, _ms_label(start_ms), consecutive_empty)
            progress.mark_chunk_done(ticker, interval, start_ms)
            if consecutive_empty >= empty_limit:
                logger.info("[Backfill] %s %s: %d consecutive empty — stopping early",
                            ticker, interval, empty_limit)
                break
            continue

        consecutive_empty = 0
        n = store.upsert_bars(interval, ticker, df)
        stored += n
        progress.mark_chunk_done(ticker, interval, start_ms)
        logger.info("[Backfill] %s %s %s: +%d bars (total: %d)",
                    ticker, interval, _ms_label(start_ms), n, stored)

    return stored


# ── Resample 30min → 1h / 2h / 4h ────────────────────────────────────────────

def resample_ticker(ticker: str) -> dict[str, int]:
    """
    Load all stored 30min bars for a ticker, resample to 1h/2h/4h, and upsert.
    Returns {interval: bars_stored}.
    """
    df_30min = store.read_ticker_bars("30min", ticker)
    if df_30min.empty:
        return {}

    results: dict[str, int] = {}
    for interval, rule in RESAMPLE_FROM_30MIN.items():
        resampled = df_30min.resample(rule).agg({
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

def fetch_daily_ticker(ticker: str, start_ms: int, end_ms: int) -> tuple[int, str]:
    """Fetch 1day bars for one ticker. Returns (bars_stored, status)."""
    if progress.is_daily_done(ticker):
        return 0, "already_done"
    _wait_if_blocked()
    _throttle()
    df = _fetch(ticker, "1day", start_ms, end_ms)
    if df.empty:
        try:
            from agent.broker.schwab_market_data import _backoff_until
            if _backoff_until > time.time():
                return 0, "cdn_block"
        except Exception:
            pass
        progress.mark_daily_done(ticker)
        return 0, "empty"
    n = store.upsert_bars("1day", ticker, df)
    progress.mark_daily_done(ticker)
    return n, "ok"


# ── Main coordinator ───────────────────────────────────────────────────────────

def run(
    tickers:       list[str],
    years:         int   = 2,
    rate_s:        float = 1.0,
    resample_only: bool  = False,
    daily_only:    bool  = False,
) -> None:
    """
    Full backfill across all intervals.

    Phases (all resumable):
      1-4. Fetch 1min / 5min / 15min / 30min directly from Schwab
      5.   Resample 30min → 1h, 2h, 4h
      6.   Fetch 1day
    """
    global _REQ_GAP
    _REQ_GAP = rate_s

    now_utc  = datetime.now(timezone.utc)
    start_dt = now_utc - timedelta(days=365 * years)
    end_dt   = now_utc - timedelta(days=1)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp()   * 1000)
    n_tickers = len(tickers)

    # ── Auth check ───────────────────────────────────────────────────────────
    if not resample_only:
        logger.info("[Backfill] Checking Schwab Market Data auth...")
        if not check_auth():
            logger.error(
                "[Backfill] ABORTING — token missing or expired.\n"
                "  Ensure nasdaq-agent service is active: sudo systemctl status nasdaq-agent\n"
                "  Token location: ~/.nasdaq_agent/schwab_md_*.json"
            )
            return

    # ── Phases 1-4: fetch each native minute interval ────────────────────────
    minute_intervals = ["1min", "5min", "15min", "30min"]

    if not resample_only and not daily_only:
        for phase_num, interval in enumerate(minute_intervals, 1):
            chunk_days = CHUNK_DAYS[interval]
            chunks     = date_chunks(start_dt, end_dt, chunk_days)
            n_chunks   = len(chunks)

            logger.info(
                "[Backfill] Phase %d: fetching %s — %d tickers, %d chunks each",
                phase_num, interval, n_tickers, n_chunks,
            )
            phase_total = 0
            for i, ticker in enumerate(tickers, 1):
                done = sum(1 for s, _ in chunks if progress.is_chunk_done(ticker, interval, s))
                if done == n_chunks:
                    logger.debug("[Backfill] [%d/%d] %s %s: all chunks done — skip",
                                 i, n_tickers, ticker, interval)
                    continue
                logger.info("[Backfill] [%d/%d] %s %s: starting (%d/%d chunks done)",
                            i, n_tickers, ticker, interval, done, n_chunks)
                bars = fetch_interval_ticker(ticker, interval, chunks)
                phase_total += bars
                logger.info("[Backfill] [%d/%d] %s %s: done +%d bars (phase total: %d)",
                            i, n_tickers, ticker, interval, bars, phase_total)

    # ── Phase 5: resample 30min → 1h, 2h, 4h ────────────────────────────────
    if not daily_only:
        logger.info("[Backfill] Phase 5: resampling 30min → 1h/2h/4h for %d tickers", n_tickers)
        for i, ticker in enumerate(tickers, 1):
            if progress.is_resampled(ticker):
                logger.debug("[Backfill] [%d/%d] %s resample: skip", i, n_tickers, ticker)
                continue
            result = resample_ticker(ticker)
            if result:
                logger.info("[Backfill] [%d/%d] %s resampled: %s",
                            i, n_tickers, ticker,
                            ", ".join(f"{iv}={n}" for iv, n in result.items()))
            else:
                logger.info("[Backfill] [%d/%d] %s resample: no 30min data yet", i, n_tickers, ticker)

    # ── Phase 6: fetch 1day ──────────────────────────────────────────────────
    logger.info("[Backfill] Phase 6: fetching 1day for %d tickers", n_tickers)
    daily_stored = 0
    for i, ticker in enumerate(tickers, 1):
        n, status = fetch_daily_ticker(ticker, start_ms, end_ms)
        daily_stored += n
        if status == "already_done":
            logger.debug("[Backfill] [%d/%d] %s 1day: skip", i, n_tickers, ticker)
        elif status == "cdn_block":
            logger.warning("[Backfill] [%d/%d] %s 1day: CDN block — will retry on resume",
                           i, n_tickers, ticker)
        elif status == "empty":
            logger.info("[Backfill] [%d/%d] %s 1day: no data", i, n_tickers, ticker)
        else:
            logger.info("[Backfill] [%d/%d] %s 1day: +%d bars (total: %d)",
                        i, n_tickers, ticker, n, daily_stored)

    logger.info("[Backfill] All phases complete.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ms_label(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def estimate_time(tickers: list[str], years: int, rate_s: float = _REQ_GAP) -> str:
    """Human-readable estimate of total API calls and wall-clock time."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=365 * years)
    end   = now - timedelta(days=1)

    total = 0
    for interval, chunk_days in CHUNK_DAYS.items():
        n_chunks = len(date_chunks(start, end, chunk_days))
        total += len(tickers) * n_chunks
    total += len(tickers)  # 1day: one call per ticker

    secs   = total * rate_s
    h, rem = divmod(int(secs), 3600)
    m      = rem // 60
    return f"~{h}h {m}m  ({total:,} API calls at {1/rate_s:.1f} req/s, worst-case all chunks fetched)"
