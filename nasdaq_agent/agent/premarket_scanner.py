"""
Pre-Market Gapper Scanner — Alpha Strike Trader PRD Section 4.1

Runs during the pre-market session (4:00–9:30 AM ET) using Twelve Data
extended-hours 1-minute bars to identify significant gap plays and build
the daily focus watchlist.

Public API
----------
run_premarket_scan(tickers)      – full scan, returns result dict
get_focus_watchlist()            – today's top-25 tickers by GQS
get_gapper_detail(ticker)        – GQS and gap details for one ticker
should_run_scan()                – True during 4:30–9:25 AM ET pre-market
get_scan_status()                – last scan time / count / watchlist summary
run_premarket_scan_background()  – launch scan in a daemon thread
"""

from __future__ import annotations

import logging
import threading
from datetime import date, datetime, time as dtime

import numpy as np
import pandas as pd
import pytz

from agent.data_fetcher import fetch_batch_interval
from config import NASDAQ_TICKERS, DEFAULT_ACCOUNT_SIZE  # noqa: F401 (DEFAULT_ACCOUNT_SIZE available for callers)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_ET = pytz.timezone("America/New_York")

# Pre-market window boundaries (ET)
_PM_START = dtime(4, 0)
_PM_END   = dtime(9, 30)

# Scan runs when we are inside 4:30–9:25 ET so there is data to evaluate
_SCAN_OPEN  = dtime(4, 30)
_SCAN_CLOSE = dtime(9, 25)

# Thresholds
_GAP_THRESHOLD_PCT   = 3.0       # ± percent to qualify as a gapper
_MIN_PM_VOLUME       = 150_000   # shares
_GQS_QUALIFY         = 65        # minimum Gap Quality Score for watchlist
_WATCHLIST_MAX       = 25        # maximum tickers in focus watchlist

# Twelve Data fetch parameters
_PM_INTERVAL    = "1min"
_PM_OUTPUTSIZE  = 60          # last 60 one-minute bars covers ~1 h of PM data
_DAILY_INTERVAL = "1day"
_DAILY_OUTPUT   = 5           # just need the most-recent close + avg volume
_DAILY_TTL      = 3_600       # 1-hour cache for daily bars

# ── Module-level state ────────────────────────────────────────────────────────

# Keyed by date.today() so stale results from yesterday are automatically ignored
_watchlist_by_date: dict[date, list[str]] = {}
_gapper_details:    dict[str, dict]       = {}   # ticker → gapper entry dict
_last_scan_result:  dict | None           = None
_scan_lock = threading.Lock()

# ── Helpers ───────────────────────────────────────────────────────────────────


def _now_et() -> datetime:
    return datetime.now(_ET)


def _is_premarket() -> bool:
    """Return True if current ET time is inside the pre-market window."""
    t = _now_et().time()
    return _PM_START <= t < _PM_END


def _is_scan_window() -> bool:
    """Return True if we are inside the active scan window (4:30–9:25 AM ET)."""
    t = _now_et().time()
    return _SCAN_OPEN <= t < _SCAN_CLOSE


def _filter_pm_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only bars whose timestamp falls in the pre-market window (4:00–9:30 ET).

    Twelve Data timestamps may be UTC or tz-aware; normalise to ET first.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx_et = idx.tz_convert(_ET)

    mask = [(t.time() >= _PM_START and t.time() < _PM_END) for t in idx_et]
    return df[mask]


def _safe_last_close(df: pd.DataFrame) -> float | None:
    """Return the last Close value from a daily DataFrame, or None on failure."""
    try:
        if df is None or df.empty:
            return None
        col = next((c for c in df.columns if c.lower() == "close"), None)
        if col is None:
            return None
        val = float(df[col].iloc[-1])
        return val if np.isfinite(val) else None
    except Exception:
        return None


def _safe_avg_volume(df: pd.DataFrame, n: int = 5) -> float | None:
    """Return the rolling average of daily Volume over the last *n* rows."""
    try:
        if df is None or df.empty:
            return None
        col = next((c for c in df.columns if c.lower() == "volume"), None)
        if col is None:
            return None
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            return None
        return float(series.tail(n).mean())
    except Exception:
        return None


# ── Gap Quality Score (GQS) ───────────────────────────────────────────────────


def _compute_gqs(
    gap_pct: float,
    pm_vol: float,
    avg_daily_vol: float | None,
    pm_bars: pd.DataFrame,
    prev_close: float,
) -> dict:
    """
    Compute the Gap Quality Score (0–100) and its four sub-components.

    Returns a dict with keys:
        gqs, gap_size_pts, volume_ratio_pts, momentum_pts, float_proxy_pts,
        pm_vol_ratio
    """

    # ── 1. Gap size (0–25) ────────────────────────────────────────────────────
    gap_size_pts = min(abs(gap_pct) / _GAP_THRESHOLD_PCT * 25.0, 25.0)

    # ── 2. Volume ratio (0–25) ────────────────────────────────────────────────
    # Compare PM volume to *expected* PM volume.
    # Proxy: typical PM volume ≈ 10 % of average daily volume.
    pm_vol_ratio = 0.0
    if avg_daily_vol and avg_daily_vol > 0:
        expected_pm_vol = avg_daily_vol * 0.10
        if expected_pm_vol > 0:
            pm_vol_ratio = pm_vol / expected_pm_vol
    # Fallback: if we cannot determine ratio, treat anything > MIN_PM_VOLUME as 1×
    if pm_vol_ratio == 0.0 and pm_vol >= _MIN_PM_VOLUME:
        pm_vol_ratio = 1.0
    volume_ratio_pts = min(pm_vol_ratio * 12.5, 25.0)

    # ── 3. Momentum strength (0–25) ───────────────────────────────────────────
    # Gap is *continuing* when the most-recent PM bar is further from prev_close
    # than the first PM bar.  Gap is *fading* when the opposite is true.
    momentum_pts = 12.5  # neutral default
    try:
        if not pm_bars.empty and len(pm_bars) >= 2:
            close_col = next((c for c in pm_bars.columns if c.lower() == "close"), None)
            if close_col:
                first_pm_price = float(pm_bars[close_col].iloc[0])
                last_pm_price  = float(pm_bars[close_col].iloc[-1])
                # Both prices relative to prev_close
                first_gap = (first_pm_price - prev_close) / prev_close * 100.0
                last_gap  = (last_pm_price  - prev_close) / prev_close * 100.0

                if abs(first_gap) > 0:
                    continuation_ratio = abs(last_gap) / abs(first_gap)
                    # >1 means gap is widening, <1 means fading
                    if continuation_ratio >= 1.2:
                        momentum_pts = 25.0   # strong continuation
                    elif continuation_ratio >= 1.0:
                        momentum_pts = 18.75  # mild continuation
                    elif continuation_ratio >= 0.8:
                        momentum_pts = 12.5   # roughly flat / slight fade
                    elif continuation_ratio >= 0.5:
                        momentum_pts = 6.25   # fading
                    else:
                        momentum_pts = 0.0    # severe fade / filling the gap
    except Exception:
        pass  # keep neutral score

    # ── 4. Float proxy (0–25) ─────────────────────────────────────────────────
    # Smaller average daily volume ≈ smaller float ≈ higher score.
    # Formula: 1 / (avg_daily_vol / 1_000_000), capped then normalised to 25.
    float_proxy_pts = 12.5  # neutral when we have no data
    try:
        if avg_daily_vol and avg_daily_vol > 0:
            raw = 1.0 / (avg_daily_vol / 1_000_000.0)
            # raw ≈ 1 for 1M avg vol (mid-cap), ≈ 10 for 100k (micro-cap)
            # Normalise: cap at 5× so micro-caps don't dominate unfairly
            capped = min(raw, 5.0)
            float_proxy_pts = (capped / 5.0) * 25.0
    except Exception:
        pass

    gqs = gap_size_pts + volume_ratio_pts + momentum_pts + float_proxy_pts

    return {
        "gqs":              round(gqs, 2),
        "gap_size_pts":     round(gap_size_pts, 2),
        "volume_ratio_pts": round(volume_ratio_pts, 2),
        "momentum_pts":     round(momentum_pts, 2),
        "float_proxy_pts":  round(float_proxy_pts, 2),
        "pm_vol_ratio":     round(pm_vol_ratio, 3),
    }


# ── VWAP reclaim detection ────────────────────────────────────────────────────


def _compute_vwap(df: pd.DataFrame) -> pd.Series | None:
    """Rolling VWAP over the supplied bars (typical-price × volume / cum-volume)."""
    try:
        high  = pd.to_numeric(df.get("High",  df.get("high")),  errors="coerce")
        low   = pd.to_numeric(df.get("Low",   df.get("low")),   errors="coerce")
        close = pd.to_numeric(df.get("Close", df.get("close")), errors="coerce")
        vol   = pd.to_numeric(df.get("Volume",df.get("volume")),errors="coerce")
        tp    = (high + low + close) / 3.0
        cum_tpv = (tp * vol).cumsum()
        cum_vol = vol.cumsum()
        return cum_tpv / cum_vol
    except Exception:
        return None


def _check_vwap_reclaim(pm_bars: pd.DataFrame, prev_close: float) -> bool:
    """
    Return True when the stock is trading ABOVE its PM VWAP but closed BELOW
    it yesterday (using prev_close as the prior-day close proxy).

    PRD Section 4.1.3: stocks trading above VWAP in pre-market but closed
    below it yesterday.
    """
    try:
        if pm_bars is None or len(pm_bars) < 5:
            return False

        vwap = _compute_vwap(pm_bars)
        if vwap is None or vwap.empty:
            return False

        close_col = next((c for c in pm_bars.columns if c.lower() == "close"), None)
        if close_col is None:
            return False

        last_pm_price = float(pm_bars[close_col].iloc[-1])
        current_vwap  = float(vwap.iloc[-1])

        above_vwap_now    = last_pm_price > current_vwap
        closed_below_vwap = prev_close    < current_vwap

        return above_vwap_now and closed_below_vwap
    except Exception:
        return False


# ── Core scan logic ───────────────────────────────────────────────────────────


def _scan_ticker(
    ticker: str,
    pm_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> dict | None:
    """
    Evaluate a single ticker.  Returns a gapper-entry dict or None if the
    ticker does not qualify (gap < threshold or volume < minimum).
    """
    try:
        # ── Previous close ────────────────────────────────────────────────────
        prev_close = _safe_last_close(daily_df)
        if prev_close is None or prev_close <= 0:
            logger.debug("%s: no valid previous close — skipping", ticker)
            return None

        # ── Filter to pre-market bars only ────────────────────────────────────
        pm_bars = _filter_pm_bars(pm_df)
        if pm_bars.empty:
            logger.debug("%s: no pre-market bars — skipping", ticker)
            return None

        # ── Resolve column names (Twelve Data may capitalise differently) ──────
        close_col  = next((c for c in pm_bars.columns if c.lower() == "close"),  None)
        high_col   = next((c for c in pm_bars.columns if c.lower() == "high"),   None)
        low_col    = next((c for c in pm_bars.columns if c.lower() == "low"),    None)
        vol_col    = next((c for c in pm_bars.columns if c.lower() == "volume"), None)

        if close_col is None:
            logger.debug("%s: no Close column in PM bars — skipping", ticker)
            return None

        # ── PM price / volume metrics ─────────────────────────────────────────
        pm_close  = float(pm_bars[close_col].iloc[-1])
        pm_volume = float(pd.to_numeric(pm_bars[vol_col], errors="coerce").sum()) \
                    if vol_col else 0.0
        pm_high   = float(pd.to_numeric(pm_bars[high_col], errors="coerce").max()) \
                    if high_col else pm_close
        pm_low    = float(pd.to_numeric(pm_bars[low_col],  errors="coerce").min()) \
                    if low_col  else pm_close

        # ── Gap calculation ───────────────────────────────────────────────────
        gap_pct = (pm_close - prev_close) / prev_close * 100.0

        if abs(gap_pct) < _GAP_THRESHOLD_PCT:
            return None  # gap too small — not a gapper

        if pm_volume < _MIN_PM_VOLUME:
            return None  # insufficient pre-market volume

        # ── Average daily volume (for GQS) ───────────────────────────────────
        avg_daily_vol = _safe_avg_volume(daily_df)

        # ── Gap Quality Score ─────────────────────────────────────────────────
        gqs_data = _compute_gqs(gap_pct, pm_volume, avg_daily_vol, pm_bars, prev_close)

        # ── VWAP reclaim flag ─────────────────────────────────────────────────
        vwap_reclaim = _check_vwap_reclaim(pm_bars, prev_close)

        return {
            "ticker":        ticker,
            "gap_pct":       round(gap_pct, 4),
            "gap_direction": "UP" if gap_pct > 0 else "DOWN",
            "pm_volume":     pm_volume,
            "gqs":           gqs_data["gqs"],
            "qualifies":     gqs_data["gqs"] >= _GQS_QUALIFY,
            "pm_high":       round(pm_high,   4),
            "pm_low":        round(pm_low,    4),
            "prev_close":    round(prev_close, 4),
            "pm_vol_ratio":  gqs_data["pm_vol_ratio"],
            "gap_size_pts":  gqs_data["gap_size_pts"],
            "volume_ratio_pts": gqs_data["volume_ratio_pts"],
            "momentum_pts":  gqs_data["momentum_pts"],
            "float_proxy_pts": gqs_data["float_proxy_pts"],
            "vwap_reclaim":  vwap_reclaim,
        }

    except Exception as exc:
        logger.warning("%s: error during scan — %s", ticker, exc, exc_info=True)
        return None


# ── Public API ────────────────────────────────────────────────────────────────


def run_premarket_scan(tickers: list | None = None) -> dict:
    """
    Run the full pre-market gapper scan.

    Parameters
    ----------
    tickers : list of ticker symbols to scan.  Defaults to NASDAQ_TICKERS.

    Returns
    -------
    dict with keys:
        gappers_up      – list of gapper-entry dicts gapping up > 3 %
        gappers_down    – list of gapper-entry dicts gapping down < -3 %
        vwap_reclaims   – subset of all gappers where vwap_reclaim is True
        focus_watchlist – top WATCHLIST_MAX gappers by GQS (direction-aware)
        scan_time       – ISO-format string of when the scan ran
        total_scanned   – number of tickers evaluated
    """
    global _last_scan_result

    if tickers is None:
        tickers = list(NASDAQ_TICKERS)

    scan_time = _now_et().isoformat()
    logger.info("Pre-market scan starting — %d tickers", len(tickers))

    # ── Fetch pre-market (1-min extended) bars ────────────────────────────────
    try:
        pm_data = fetch_batch_interval(
            tickers, _PM_INTERVAL, _PM_OUTPUTSIZE,
            extended_hours=True, ttl=0,
        )
    except Exception as exc:
        logger.error("PM bar fetch failed: %s", exc, exc_info=True)
        pm_data = {}

    # ── Fetch daily bars (prev close + avg volume) ────────────────────────────
    try:
        daily_data = fetch_batch_interval(
            tickers, _DAILY_INTERVAL, _DAILY_OUTPUT, ttl=_DAILY_TTL,
        )
    except Exception as exc:
        logger.error("Daily bar fetch failed: %s", exc, exc_info=True)
        daily_data = {}

    # ── Evaluate each ticker ──────────────────────────────────────────────────
    all_gappers: list[dict] = []

    for ticker in tickers:
        pm_df    = pm_data.get(ticker)
        daily_df = daily_data.get(ticker)

        if pm_df is None and daily_df is None:
            continue  # no data at all — skip silently

        result = _scan_ticker(ticker, pm_df, daily_df)
        if result is not None:
            all_gappers.append(result)

    # ── Split by direction ────────────────────────────────────────────────────
    gappers_up   = sorted(
        [g for g in all_gappers if g["gap_direction"] == "UP"],
        key=lambda x: x["gqs"], reverse=True,
    )
    gappers_down = sorted(
        [g for g in all_gappers if g["gap_direction"] == "DOWN"],
        key=lambda x: x["gqs"], reverse=True,
    )

    # ── VWAP reclaims ─────────────────────────────────────────────────────────
    vwap_reclaims = [g for g in all_gappers if g.get("vwap_reclaim")]

    # ── Focus watchlist (top WATCHLIST_MAX by GQS, qualifying only) ──────────
    qualified = [g for g in all_gappers if g["qualifies"]]
    qualified_sorted = sorted(qualified, key=lambda x: x["gqs"], reverse=True)
    focus_list = qualified_sorted[:_WATCHLIST_MAX]
    focus_tickers = [g["ticker"] for g in focus_list]

    # ── Persist state ─────────────────────────────────────────────────────────
    today = date.today()
    with _scan_lock:
        _watchlist_by_date[today] = focus_tickers
        _gapper_details.clear()
        for g in all_gappers:
            _gapper_details[g["ticker"]] = g

    result = {
        "gappers_up":       gappers_up,
        "gappers_down":     gappers_down,
        "vwap_reclaims":    vwap_reclaims,
        "focus_watchlist":  focus_tickers,
        "scan_time":        scan_time,
        "total_scanned":    len(tickers),
        "total_gappers":    len(all_gappers),
        "qualified_count":  len(qualified),
    }

    with _scan_lock:
        _last_scan_result = result

    logger.info(
        "Pre-market scan done — %d gappers (%d up / %d down), %d on watchlist",
        len(all_gappers), len(gappers_up), len(gappers_down), len(focus_tickers),
    )
    return result


def get_focus_watchlist() -> list[str]:
    """
    Return today's focus ticker list.

    Returns an empty list if the scan has not yet run today or if the
    market is no longer in pre-market hours.
    """
    today = date.today()
    with _scan_lock:
        return list(_watchlist_by_date.get(today, []))


def get_gapper_detail(ticker: str) -> dict:
    """
    Return the full gapper-entry dict for *ticker* from the most recent scan.

    Returns an empty dict if the ticker was not found or did not qualify.
    """
    with _scan_lock:
        return dict(_gapper_details.get(ticker, {}))


def should_run_scan() -> bool:
    """
    Return True if:
      - Current ET time is inside the scan window (4:30–9:25 AM ET), AND
      - The scan has not already completed today.
    """
    if not _is_scan_window():
        return False

    today = date.today()
    with _scan_lock:
        return today not in _watchlist_by_date


def get_scan_status() -> dict:
    """
    Return a summary of the scanner's current state.

    Keys
    ----
    last_scan_time   : ISO string or None
    total_gappers    : int
    qualified_count  : int
    watchlist_count  : int
    watchlist_tickers: list[str]
    is_premarket     : bool
    scan_window_open : bool
    scan_ran_today   : bool
    """
    today = date.today()
    with _scan_lock:
        last = _last_scan_result or {}
        watchlist = list(_watchlist_by_date.get(today, []))
        ran_today = today in _watchlist_by_date

    return {
        "last_scan_time":    last.get("scan_time"),
        "total_gappers":     last.get("total_gappers", 0),
        "qualified_count":   last.get("qualified_count", 0),
        "watchlist_count":   len(watchlist),
        "watchlist_tickers": watchlist,
        "is_premarket":      _is_premarket(),
        "scan_window_open":  _is_scan_window(),
        "scan_ran_today":    ran_today,
    }


def run_premarket_scan_background(tickers: list | None = None) -> threading.Thread:
    """
    Launch the pre-market scan in a daemon thread and return the thread object.

    Parameters
    ----------
    tickers : ticker list forwarded to run_premarket_scan.  None → NASDAQ_TICKERS.

    Usage
    -----
        t = run_premarket_scan_background()
        # continues immediately; scan runs concurrently
    """
    def _worker():
        try:
            run_premarket_scan(tickers)
        except Exception as exc:
            logger.error("Background pre-market scan failed: %s", exc, exc_info=True)

    t = threading.Thread(target=_worker, name="premarket-scanner", daemon=True)
    t.start()
    logger.info("Pre-market scan launched in background thread %s", t.name)
    return t
