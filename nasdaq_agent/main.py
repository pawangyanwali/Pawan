"""
FastAPI entry point.
Serves the static web dashboard and a WebSocket endpoint that pushes
real-time stock signals to all connected clients.
"""

import asyncio
import json
import logging
import os
import threading as _threading
import time
import numpy as np
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

# ── Service mode flags ────────────────────────────────────────────────────────
# Defined before agent imports so heavy subsystems are never loaded in containers
# that do not run them. Legacy scanner and learner default off; Release 5 owns
# those responsibilities in dedicated scalp-engine and scalp-learner services.
_SCANNER_ENABLED     = os.getenv("NASDAQ_SCANNER_ENABLED",     "0") != "0"
_MARKET_DATA_ENABLED = os.getenv("NASDAQ_MARKET_DATA_ENABLED", "1") != "0"
_LEARNER_ENABLED     = os.getenv("NASDAQ_LEARNER_ENABLED",     "0") != "0"
_SCHEDULER_ENABLED   = os.getenv("NASDAQ_SCHEDULER_ENABLED",   "1") != "0"

if _SCANNER_ENABLED:
    from agent.scanner import scanner, StockSignal
else:
    class _NoopScanner:  # type: ignore[no-redef]
        signals: list    = []
        last_scan        = None
        is_running: bool = False
        def register_callback(self, *a): pass
        def register_per_ticker_callback(self, *a): pass
        def start_background(self): pass
        def stop(self): pass
        def get_last_signals(self): return []
    scanner     = _NoopScanner()  # type: ignore[assignment]
    StockSignal = object          # type: ignore[assignment,misc]

from agent.market_hours import get_session_info, get_market_session, refresh_market_hours_cache
from agent.market_regime import get_regime
from agent.macro_calendar import check_macro_event
from agent.backtest_reporter import get_broadcast_summary
from agent.adaptive_filter import get_status as af_get_status
if _LEARNER_ENABLED:
    from agent.learning_engine import learning_engine
    import agent.weekend_learner as weekend_learner
else:
    class _NoopLearner:  # type: ignore[no-redef]
        def start(self): pass
        def stop(self): pass
        def get_status(self): return {}
    learning_engine = _NoopLearner()  # type: ignore[assignment]
    class _NoopWeekendLearner:  # type: ignore[no-redef]
        def register_broadcast(self, *a): pass
        def maybe_start(self): pass
        def get_status(self): return {}
    weekend_learner = _NoopWeekendLearner()  # type: ignore[assignment]

from agent.broker.schwab_auth import load_stored_tokens, load_stored_md_tokens
if _MARKET_DATA_ENABLED:
    from agent.broker.schwab_streamer import (
        start_streamer, start_md_poller, get_streamer_status,
        register_tick_callback, register_bulk_price_callback,
        is_md_poller_running, get_live_quotes_snapshot,
    )
else:
    def start_streamer(*a, **kw): pass          # type: ignore[misc]
    def start_md_poller(*a, **kw): pass         # type: ignore[misc]
    def get_streamer_status() -> dict: return {"connected": False, "disabled": True}  # type: ignore[misc]
    def register_tick_callback(*a): pass        # type: ignore[misc]
    def register_bulk_price_callback(*a): pass  # type: ignore[misc]
    def is_md_poller_running() -> bool: return False   # type: ignore[misc]
    def get_live_quotes_snapshot() -> dict: return {}  # type: ignore[misc]
from agent.broker.order_bridge import maybe_place_tos_order
from agent.notifier import notify_signal as _notify_signal
from config import NASDAQ_TICKERS


# ── Shared helpers and globals — defined in routers/_deps.py ─────────────────
# Imported here so main.py callback functions (_on_signals, _on_ticker, etc.)
# can use them directly, and so routers can import from routers._deps without
# depending on main.py (avoiding circular imports).
from routers._deps import (
    _pt_executor,
    _sanitize,
    _NumpyEncoder,
    _dumps,
    ConnectionManager,
    manager,
    _event_loop,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Signal cache — persists last scan across server restarts ──────────────────
# Allows /api/signals and the WS connect handler to serve stale-but-valid data
# immediately instead of waiting for the next full scan (up to 6 minutes).
_SIGNAL_CACHE_FILE = os.path.join(
    os.path.expanduser("~"), ".nasdaq_agent", "signal_cache.json"
)
_SIGNAL_CACHE_ACTIVE_MAX_AGE_SECS = 4 * 3600
_SIGNAL_CACHE_CLOSED_MAX_AGE_SECS = 5 * 24 * 3600
_last_signals_dicts: list[dict] = []   # in-memory fast path
_last_signals_ts:    str        = ""   # ISO timestamp of the cached scan


def _save_signal_cache(signals_dicts: list[dict], ts: str) -> None:
    """Atomically write signal cache to disk after every completed scan."""
    global _last_signals_dicts, _last_signals_ts
    _last_signals_dicts = signals_dicts
    _last_signals_ts    = ts
    try:
        os.makedirs(os.path.dirname(_SIGNAL_CACHE_FILE), exist_ok=True)
        tmp = _SIGNAL_CACHE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"ts": ts, "signals": signals_dicts}, f, separators=(",", ":"))
        os.replace(tmp, _SIGNAL_CACHE_FILE)
    except Exception as _ce:
        logger.debug("Signal cache write failed: %s", _ce)


def _signal_cache_max_age_seconds() -> int:
    """
    Return the allowed age for the persisted signal cache.

    During active trading sessions we keep the cache tight so stale signals do
    not masquerade as live scans. During closed/weekend/holiday windows we keep
    the last scan for several days so a restart on a long weekend still has a
    useful dashboard instead of a blank table.
    """
    try:
        info = get_session_info()
        if (
            info.get("is_weekend")
            or info.get("is_holiday")
            or info.get("session") == "CLOSED"
        ):
            return _SIGNAL_CACHE_CLOSED_MAX_AGE_SECS
    except Exception:
        pass
    return _SIGNAL_CACHE_ACTIVE_MAX_AGE_SECS


def _loaded_signal_cache_age_seconds() -> float | None:
    """Return the age of the loaded signal snapshot, preferring its scan time."""
    if _last_signals_ts:
        try:
            ts = _last_signals_ts.replace("Z", "+00:00")
            scan_dt = datetime.fromisoformat(ts)
            if scan_dt.tzinfo is None:
                scan_dt = scan_dt.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - scan_dt.astimezone(timezone.utc)).total_seconds()
        except Exception:
            pass
    try:
        return time.time() - os.path.getmtime(_SIGNAL_CACHE_FILE)
    except Exception:
        return None


def _build_observation_rows() -> tuple[list[dict], str | None]:
    """
    Build dashboard-only rows from live quote memory when no scanner signals exist.

    These rows keep the dashboard populated with current market observations
    during regular, pre-market, and after-hours quote flow. They are intentionally
    neutral and are not fed back into trading, backtesting, or model training.
    """
    try:
        session = get_session_info()
    except Exception:
        session = {}
    session_key = session.get("session", "")
    closed_window = (
        session.get("is_weekend")
        or session.get("is_holiday")
        or session_key == "CLOSED"
    )

    max_age_s = None if closed_window else 10.0

    def _fresh_quotes(raw: dict[str, dict]) -> dict[str, dict]:
        if not raw:
            return {}
        if max_age_s is None:
            return {str(sym): dict(quote) for sym, quote in raw.items() if isinstance(quote, dict)}
        now = time.time()
        fresh: dict[str, dict] = {}
        for sym, quote in raw.items():
            if not isinstance(quote, dict):
                continue
            try:
                updated_at = float(quote.get("updated_at") or 0.0)
            except Exception:
                updated_at = 0.0
            if updated_at > 0 and now - updated_at <= max_age_s:
                fresh[str(sym)] = dict(quote)
        return fresh

    quotes: dict[str, dict] = {}
    try:
        from agent.valkey_client import get_all_prices
        quotes.update(_fresh_quotes(get_all_prices()))
    except Exception:
        pass
    try:
        # In-process quote memory wins over Valkey if both have the ticker.
        quotes.update(get_live_quotes_snapshot(max_age_s=max_age_s))
    except Exception:
        pass
    if not quotes:
        return [], None

    try:
        regime = get_regime()
        regime_name = regime.regime
        regime_label = regime.label
        regime_color = regime.color
    except Exception:
        regime_name = "NEUTRAL"
        regime_label = "Neutral"
        regime_color = "#94a3b8"

    rows: list[dict] = []
    newest_ts = 0.0
    for ticker in NASDAQ_TICKERS:
        q = quotes.get(ticker)
        if not q:
            continue
        last = float(q.get("last") or 0.0)
        mark = float(q.get("mark") or 0.0)
        price = mark if session_key in {"AFTER_HOURS", "PRE_MARKET", "CLOSED"} and mark > 0 else last or mark
        if price <= 0:
            continue
        updated_at = float(q.get("updated_at") or 0.0)
        newest_ts = max(newest_ts, updated_at)
        prev_close = float(q.get("prev_close") or 0.0)
        change_pct = float(q.get("net_pct_change") or 0.0)
        if change_pct == 0.0 and prev_close > 0:
            change_pct = (price - prev_close) / prev_close * 100
        row = {
            "ticker": ticker,
            "name": ticker,
            "price": round(price, 4),
            "change_pct": round(change_pct, 3),
            "open_price": float(q.get("open") or 0.0),
            "technical": 0.0,
            "volume": float(q.get("volume") or 0.0),
            "ml_prob": 0.5,
            "ml_daily_prob": 0.5,
            "sentiment": 0.0,
            "score": 0.0,
            "signal": "NEUTRAL",
            "rel_volume": 0.0,
            "unusual_vol": False,
            "prediction": "NEUTRAL",
            "confidence": 0.0,
            "trend": "SIDEWAYS",
            "trend_probability": 0.5,
            "ml_trained": False,
            "target_price": round(price, 4),
            "stop_loss": round(price, 4),
            "rr_ratio": 0.0,
            "patterns": [],
            "reasons": ["Observation mode: live quote available, no trade signal generated."],
            "ml_swing_prob": 0.5,
            "ml_deep_prob": 0.5,
            "ml_swing_trained": False,
            "ml_deep_trained": False,
            "supports": [],
            "resistances": [],
            "pivots": {},
            "poc": 0.0,
            "mtf_score": 0.0,
            "mtf_alignment": "MIXED",
            "mtf_bull_count": 0,
            "mtf_bear_count": 0,
            "mtf_timeframes": {},
            "rsi_gated": False,
            "reversal_score": 0.0,
            "reversal_type": "NONE",
            "divergence_type": "NONE",
            "reversal_signals": [],
            "retest_level": 0.0,
            "entry_zone_low": 0.0,
            "entry_zone_high": 0.0,
            "exhaustion_flags": [],
            "bounce_signals": [],
            "session": session_key or "UNKNOWN",
            "session_label": session.get("label", ""),
            "session_color": session.get("color", "#94a3b8"),
            "session_mult": float(session.get("mult", 1.0) or 1.0),
            "session_advice": session.get("advice", ""),
            "regime": regime_name,
            "regime_label": regime_label,
            "regime_color": regime_color,
            "trading_tier": "REGULAR",
            "rs_label": "IN_LINE",
            "rs_ratio": 1.0,
            "gap_type": "FLAT",
            "gap_pct": 0.0,
            "gap_filled": False,
            "gap_fill_prob": 0.0,
            "premarket_high": 0.0,
            "premarket_low": 0.0,
            "vwap_event": "FLAT",
            "vwap_deviation": 0.0,
            "sector_etf": "",
            "sector_trend": "NEUTRAL",
            "exit_recommendation": "HOLD",
            "rsi_zone": "NEUTRAL",
            "rsi_value": 50.0,
            "entry_type": "OBSERVATION",
            "rr_quality": "LOW",
            "rr_qualifies": False,
            "is_suppressed": False,
            "suppress_reason": "",
            "earnings_blocked": False,
            "earnings_reason": "",
            "earnings_date": "",
            "earnings_days_away": 0,
            "ah_change_pct": 0.0,
            "ah_direction": "",
            "ah_magnitude": "",
            "ah_confirms_signal": False,
            "ah_news_likely": False,
            "trade_plan": {},
            "candles": [],
            "headlines": [],
            "ticker_win_rate": 0.0,
            "ticker_obs_count": 0,
            "learning_rank": 0.0,
            "is_observation": True,
            "source": "live_quote_observation",
            "scanned_at": datetime.fromtimestamp(updated_at or time.time(), timezone.utc).isoformat(),
        }
        rows.append(row)

    if newest_ts:
        return rows, datetime.fromtimestamp(newest_ts, timezone.utc).isoformat()
    return rows, None


def _merge_observation_rows(primary_rows: list[dict]) -> tuple[list[dict], str | None]:
    """Append live-quote observation rows for tickers missing from primary_rows."""
    observation_rows, observation_ts = _build_observation_rows()
    if not observation_rows:
        return primary_rows, observation_ts
    seen = {str(row.get("ticker", "")) for row in primary_rows}
    missing = [row for row in observation_rows if row.get("ticker") not in seen]
    if not missing:
        return primary_rows, observation_ts
    return primary_rows + missing, observation_ts


def _load_signal_cache() -> None:
    """Load the on-disk signal cache at startup when it is session-appropriate."""
    global _last_signals_dicts, _last_signals_ts
    try:
        if not os.path.exists(_SIGNAL_CACHE_FILE):
            return
        age = time.time() - os.path.getmtime(_SIGNAL_CACHE_FILE)
        max_age = _signal_cache_max_age_seconds()
        if age > max_age:
            logger.info(
                "[Cache] Signal cache too old (age %.0fs > max %.0fs); not loading",
                age, max_age,
            )
            return
        with open(_SIGNAL_CACHE_FILE) as f:
            data = json.load(f)
        sigs = data.get("signals", [])
        if sigs:
            _last_signals_dicts = sigs
            _last_signals_ts    = data.get("ts", "")
            logger.info(
                "[Cache] Loaded %d signals from disk (age %.0fs)", len(sigs), age
            )
    except Exception as _ce:
        logger.debug("Signal cache read failed: %s", _ce)


def _current_signal_snapshot() -> tuple[list[dict], str | None, bool]:
    """
    Return (signals, last_scan_ts, from_cache) for dashboard/API consumers.

    Source priority:
      1. In-process scanner.signals — freshest, used when scanner runs here
      2. Valkey scan:latest key    — durable cross-container snapshot
      3. Disk signal cache          — local fallback (survives Valkey outage)
      4. Live-quote observations    — last resort when no scan data exists
    """
    # Priority 1: in-process scanner (monolith / scanner-enabled container)
    if _SCANNER_ENABLED and scanner.signals:
        return [s.to_dict() for s in scanner.signals], scanner.last_scan, False

    # Priority 2: Valkey snapshot (written by scanner container after every cycle)
    try:
        from agent.signal_snapshot import read_latest as _snap_read
        snap = _snap_read()
        if snap and snap.get("signals"):
            raw_ts = snap.get("ts")
            ts_str = (
                datetime.fromtimestamp(float(raw_ts), timezone.utc).isoformat()
                if raw_ts else None
            )
            return snap["signals"], ts_str, True
    except Exception:
        pass

    # Priority 3: disk cache (survives Valkey outage or cold start)
    if _last_signals_dicts:
        age = _loaded_signal_cache_age_seconds()
        if age is None or age <= _signal_cache_max_age_seconds():
            rows, observation_ts = _merge_observation_rows(_last_signals_dicts)
            return rows, observation_ts or _last_signals_ts or scanner.last_scan, True

    # Priority 4: live-quote observations only (no scan data at all)
    observation_rows, observation_ts = _build_observation_rows()
    if observation_rows:
        return observation_rows, observation_ts or scanner.last_scan, False
    return [], _last_signals_ts or scanner.last_scan, False

# ── Schwab price → WebSocket broadcast ───────────────────────────────────────
_schwab_tick_registered: bool = False

def _on_schwab_bulk_prices(prices: dict) -> None:
    """
    Forward ALL updated Schwab quotes to WebSocket clients in ONE message per
    poll cycle.  Replaces 477 individual tick messages with a single batch,
    making browser-side updates smoother and reducing WS overhead by ~99%.

    Message format: {"type": "prices", "p": {ticker: {last, pct_change, ...}}}
    """
    if not manager.active or _event_loop is None:
        return
    if not prices:
        return
    try:
        payload = json.dumps({"type": "prices", "p": prices}, default=lambda x: None)
        asyncio.run_coroutine_threadsafe(manager.broadcast(payload), _event_loop)
    except Exception:
        pass

def _on_schwab_tick(ticker: str, quote: dict) -> None:
    """Legacy per-ticker tick — kept for WebSocket streamer path (sub-100ms)."""
    if not manager.active or _event_loop is None:
        return
    last = quote.get("last")
    if last is None:
        return
    try:
        payload = json.dumps({
            "type":           "tick",
            "ticker":         ticker,
            "last":           last,
            "bid":            quote.get("bid"),
            "ask":            quote.get("ask"),
            "volume":         quote.get("volume"),
            "high":           quote.get("high"),
            "low":            quote.get("low"),
            "net_pct_change": quote.get("net_pct_change"),
        }, default=lambda x: None)
        asyncio.run_coroutine_threadsafe(manager.broadcast(payload), _event_loop)
    except Exception:
        pass

def _on_valkey_prices(prices: dict) -> None:
    """
    Forward Valkey pub/sub price batches to WebSocket clients.
    Fires whenever the MD Poller publishes to Valkey — same cadence as
    _on_schwab_bulk_prices but sourced from the shared price bus.
    Deduplicated: if Schwab callbacks are already registered, Valkey provides
    a redundant path that is a no-op when the poller is healthy.
    """
    _on_schwab_bulk_prices(prices)


_valkey_sub_registered: bool = False


def _ensure_tick_broadcast_registered() -> None:
    """Register price→WebSocket callbacks exactly once."""
    global _schwab_tick_registered, _valkey_sub_registered
    if not _schwab_tick_registered:
        register_bulk_price_callback(_on_schwab_bulk_prices)
        register_tick_callback(_on_schwab_tick)
        _schwab_tick_registered = True

    # Also subscribe to Valkey so the WS path stays live even if the direct
    # Schwab bulk callback is replaced by a Valkey-only pipeline in a future step.
    if not _valkey_sub_registered:
        try:
            from agent.valkey_client import register_price_subscriber
            register_price_subscriber(_on_valkey_prices)
            _valkey_sub_registered = True
            logger.info("[Valkey] Price subscriber registered for WebSocket bridge.")
        except Exception as _ve:
            logger.warning(f"[Valkey] Could not register subscriber: {_ve}")


def _get_universe_total() -> int:
    try:
        from agent.ticker_universe import FULL_UNIVERSE
        return len(FULL_UNIVERSE)
    except Exception:
        from config import NASDAQ_TICKERS
        return len(NASDAQ_TICKERS)


def _on_signals(signals: list[StockSignal]) -> None:
    """Callback invoked by the scanner thread; schedule a broadcast on the main loop."""
    if _event_loop is None:
        return

    # Build alert list: high-confidence BUY/SELL signals only
    alerts = [
        {"ticker": s.ticker, "direction": s.prediction,
         "confidence": s.confidence, "price": s.price,
         "session": s.session, "regime": s.regime}
        for s in signals
        if s.prediction in ("BUY", "SELL") and s.confidence >= 70
           and s.rr_qualifies and not s.earnings_blocked
    ]

    # Market breadth — computed from current scan signals
    _above_vwap = sum(1 for s in signals if s.vwap_event in ("ABOVE","RECLAIM","EXTENDED_UP"))
    _below_vwap = sum(1 for s in signals if s.vwap_event in ("BELOW","REJECTION","EXTENDED_DOWN"))
    _bullish_signals = sum(1 for s in signals if s.prediction in ("BUY","STRONG BUY"))
    _bearish_signals = sum(1 for s in signals if s.prediction in ("SELL","STRONG SELL"))
    _total = len(signals) or 1
    breadth = {
        "above_vwap":     _above_vwap,
        "below_vwap":     _below_vwap,
        "pct_above_vwap": round(_above_vwap / _total * 100, 1),
        "bullish":        _bullish_signals,
        "bearish":        _bearish_signals,
        "bias":           "BULLISH" if _bullish_signals > _bearish_signals else "BEARISH" if _bearish_signals > _bullish_signals else "NEUTRAL",
    }

    regime  = get_regime()
    session = get_session_info()
    macro   = check_macro_event()

    try:
        bt_summary = get_broadcast_summary()
    except Exception:
        bt_summary = {}

    try:
        learn_summary = af_get_status()
        learn_compact = {
            "win_rate":          learn_summary.get("current_win_rate", 0.0),
            "target_win_rate":   learn_summary.get("target_win_rate") or 55.0,
            "dynamic_threshold": learn_summary.get("dynamic_threshold", 60.0),
            "suppressed_count":  learn_summary.get("suppressed_count", 0),
            "blocked_count":     len(learn_summary.get("blocked_contexts", {})),
            "is_learning":       learn_summary.get("is_learning", False),
        }
    except Exception:
        learn_compact = {}

    try:
        from agent.paper_trading import get_summary as _pt_summary, get_open_trades as _pt_open
        _raw_open   = _pt_open()
        # Enrich open trades with unrealized P&L using current scan prices
        _last_prices = {s.ticker: s.price for s in signals}
        for t in _raw_open:
            ep = _last_prices.get(t["ticker"])
            if ep:
                entry  = t.get("entry_price") or 0
                shares = t.get("shares_remaining") or t.get("shares") or 1
                d      = t.get("direction", "BUY")
                pnl_d  = ((ep - entry) * shares if d == "BUY" else (entry - ep) * shares)
                pnl_p  = (pnl_d / (entry * shares) * 100) if entry > 0 else 0.0
                t["current_price"]          = round(ep, 4)
                t["unrealized_pnl_dollar"]  = round(pnl_d, 2)
                t["unrealized_pnl_pct"]     = round(pnl_p, 3)
        open_trades = {t["ticker"]: t for t in _raw_open}
        pt_stats    = _pt_summary()
    except Exception:
        open_trades = {}
        pt_stats    = {}

    # ThinkorSwim auto-trade: attempt bracket orders for qualifying signals
    if _tos_auto_trade:
        _tos_results = []
        for sig in signals:
            if sig.prediction in ("BUY", "STRONG BUY", "SELL", "STRONG SELL"):
                try:
                    r = maybe_place_tos_order(sig)
                    if r.get("placed"):
                        _tos_results.append(r)
                except Exception as _te:
                    pass
        if _tos_results:
            logging.getLogger(__name__).info(
                f"TOS auto-trade: {len(_tos_results)} orders placed this cycle"
            )

    sigs_dicts = [s.to_dict() for s in signals]
    _scan_meta = {
        "universe_total": _get_universe_total(),
        "monitored_count": len(sigs_dicts),
        "active_scan_count": int(getattr(scanner, "_last_active_count", len(signals)) or len(signals)),
        "analysis_batch_count": int(getattr(scanner, "_last_analysis_batch_count", len(signals)) or len(signals)),
        "deep_analyzed_count": int(getattr(scanner, "_last_deep_analyzed_count", len(signals)) or len(signals)),
        "preserved_count": int(getattr(scanner, "_last_preserved_count", 0) or 0),
        "observation_count": int(getattr(scanner, "_last_observation_count", 0) or 0),
    }

    # Write durable snapshot to Valkey — web-api reads this on restart instead
    # of waiting for the next scan cycle (Step 3 of the containerisation plan).
    try:
        from agent.signal_snapshot import write_latest as _snap_write
        _snap_write(
            signals       = sigs_dicts,
            regime        = regime.to_dict(),
            session       = session,
            scanned_count = len(signals),
            scan_meta     = _scan_meta,
        )
    except Exception:
        pass

    # Persist to disk cache so the next page load is instantaneous
    _save_signal_cache(sigs_dicts, scanner.last_scan or "")

    payload = _dumps({
        "type":          "update",
        "signals":       sigs_dicts,
        "regime":        regime.to_dict(),
        "session":       session,
        "alerts":        alerts,
        "macro":         macro,
        "backtest":      bt_summary,
        "learning":      learn_compact,
        "open_trades":   open_trades,
        "pt_stats":      pt_stats,
        "breadth":       breadth,
        "universe_total": _scan_meta["universe_total"],
        "monitored_count": _scan_meta["monitored_count"],
        "active_scan_count": _scan_meta["active_scan_count"],
        "analysis_batch_count": _scan_meta["analysis_batch_count"],
        "deep_analyzed_count": _scan_meta["deep_analyzed_count"],
        "preserved_count": _scan_meta["preserved_count"],
        "observation_count": _scan_meta["observation_count"],
        "scanned_count": _scan_meta["active_scan_count"],
    })
    asyncio.run_coroutine_threadsafe(manager.broadcast(payload), _event_loop)

    # Telegram notifications for qualifying signals (rate-limited per ticker)
    for _s in signals:
        if (_s.prediction in ("BUY", "STRONG BUY", "SELL", "STRONG SELL")
                and getattr(_s, "rr_qualifies", False)
                and not getattr(_s, "earnings_blocked", False)):
            try:
                _notify_signal(
                    _s.ticker, _s.prediction, _s.confidence, _s.price,
                    getattr(_s, "target_price", 0.0),
                    getattr(_s, "stop_loss",    0.0),
                    getattr(_s, "session", ""),
                    getattr(_s, "regime",  ""),
                )
            except Exception:
                pass

    # Also send a standalone `prices` message with every ticker the scanner just
    # priced.  This ensures the surgical DOM update path fires even when the MD
    # poller is down (Schwab not authenticated, token expired, etc.) so prices
    # never go more than one scan cycle (~30 s) stale regardless of poller state.
    if manager.active:
        _price_patch = {
            s.ticker: {
                "last":       s.price,
                "open":       getattr(s, "open_price", 0) or 0,
                "pct_change": getattr(s, "change_pct",  0) or 0,
                "updated_at":  time.time(),
                "source":      "SCANNER",
                "source_status": "SCAN_SNAPSHOT",
                "is_live":     False,
            }
            for s in signals
            if s.price > 0
        }
        if _price_patch:
            _pp_payload = _dumps({"type": "prices", "p": _price_patch})
            asyncio.run_coroutine_threadsafe(manager.broadcast(_pp_payload), _event_loop)


def _on_valkey_scan(snap: dict) -> None:
    """Broadcast scan results from Valkey (scanner running in a separate container)."""
    if _event_loop is None:
        return
    try:
        sigs_dicts = snap.get("signals", [])
        regime     = snap.get("regime", {})
        session    = snap.get("session", {})
        _scan_meta = {
            "universe_total": int(snap.get("universe_total") or _get_universe_total()),
            "monitored_count": int(snap.get("monitored_count") or len(sigs_dicts)),
            "active_scan_count": int(snap.get("active_scan_count") or snap.get("scanned_count") or len(sigs_dicts)),
            "analysis_batch_count": int(snap.get("analysis_batch_count") or snap.get("deep_analyzed_count") or snap.get("scanned_count") or len(sigs_dicts)),
            "deep_analyzed_count": int(snap.get("deep_analyzed_count") or snap.get("scanned_count") or len(sigs_dicts)),
            "preserved_count": int(snap.get("preserved_count") or 0),
            "observation_count": int(snap.get("observation_count") or 0),
        }

        _above_vwap      = sum(1 for s in sigs_dicts if s.get("vwap_event") in ("ABOVE", "RECLAIM", "EXTENDED_UP"))
        _below_vwap      = sum(1 for s in sigs_dicts if s.get("vwap_event") in ("BELOW", "REJECTION", "EXTENDED_DOWN"))
        _bullish_signals = sum(1 for s in sigs_dicts if s.get("prediction") in ("BUY", "STRONG BUY"))
        _bearish_signals = sum(1 for s in sigs_dicts if s.get("prediction") in ("SELL", "STRONG SELL"))
        _total = len(sigs_dicts) or 1
        breadth = {
            "above_vwap":     _above_vwap,
            "below_vwap":     _below_vwap,
            "pct_above_vwap": round(_above_vwap / _total * 100, 1),
            "bullish":        _bullish_signals,
            "bearish":        _bearish_signals,
            "bias":           ("BULLISH" if _bullish_signals > _bearish_signals
                               else "BEARISH" if _bearish_signals > _bullish_signals
                               else "NEUTRAL"),
        }
        alerts = [
            {"ticker": s["ticker"], "direction": s["prediction"],
             "confidence": s.get("confidence"), "price": s.get("price"),
             "session": s.get("session"), "regime": s.get("regime")}
            for s in sigs_dicts
            if s.get("prediction") in ("BUY", "SELL")
               and (s.get("confidence") or 0) >= 70
               and s.get("rr_qualifies") and not s.get("earnings_blocked")
        ]
        try:
            macro = check_macro_event()
        except Exception:
            macro = {}
        try:
            bt_summary = get_broadcast_summary()
        except Exception:
            bt_summary = {}
        try:
            learn_summary = af_get_status()
            learn_compact = {
                "win_rate":          learn_summary.get("current_win_rate", 0.0),
                "target_win_rate":   learn_summary.get("target_win_rate") or 55.0,
                "dynamic_threshold": learn_summary.get("dynamic_threshold", 60.0),
                "suppressed_count":  learn_summary.get("suppressed_count", 0),
                "blocked_count":     len(learn_summary.get("blocked_contexts", {})),
                "is_learning":       learn_summary.get("is_learning", False),
            }
        except Exception:
            learn_compact = {}
        try:
            from agent.paper_trading import get_summary as _pt_sum2, get_open_trades as _pt_open2
            open_trades = {t["ticker"]: t for t in _pt_open2()}
            pt_stats    = _pt_sum2()
        except Exception:
            open_trades = {}
            pt_stats    = {}

        payload = _dumps({
            "type":           "update",
            "signals":        sigs_dicts,
            "regime":         regime,
            "session":        session,
            "alerts":         alerts,
            "macro":          macro,
            "backtest":       bt_summary,
            "learning":       learn_compact,
            "open_trades":    open_trades,
            "pt_stats":       pt_stats,
            "breadth":        breadth,
            "universe_total": _scan_meta["universe_total"],
            "monitored_count": _scan_meta["monitored_count"],
            "active_scan_count": _scan_meta["active_scan_count"],
            "analysis_batch_count": _scan_meta["analysis_batch_count"],
            "deep_analyzed_count": _scan_meta["deep_analyzed_count"],
            "preserved_count": _scan_meta["preserved_count"],
            "observation_count": _scan_meta["observation_count"],
            "scanned_count":  _scan_meta["active_scan_count"],
        })
        asyncio.run_coroutine_threadsafe(manager.broadcast(payload), _event_loop)

        # Price patch so the surgical DOM update fires
        if manager.active:
            _price_patch = {
                s["ticker"]: {
                    "last":       s.get("price", 0),
                    "open":       s.get("open_price", 0) or 0,
                    "pct_change": s.get("change_pct", 0) or 0,
                    "updated_at":  time.time(),
                    "source":      "SCANNER",
                    "source_status": "SCAN_SNAPSHOT",
                    "is_live":     False,
                }
                for s in sigs_dicts
                if (s.get("price") or 0) > 0
            }
            if _price_patch:
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast(_dumps({"type": "prices", "p": _price_patch})),
                    _event_loop,
                )
    except Exception as _ve:
        logging.getLogger(__name__).debug("[Valkey] scan broadcast error: %s", _ve)


# ── Ticker-update batch accumulator ───────────────────────────────────────────
# A 477-ticker scan fires _on_ticker once per result, nearly simultaneously.
# Scheduling 477 individual manager.broadcast() coroutines on the event loop
# delays keepalive pings and real-time price messages.  This coalesces updates
# into one ticker_batch message per 100ms window — typically 1-2 per scan cycle.
_ticker_batch: list[dict] = []
_ticker_batch_lock = _threading.Lock()
_ticker_batch_pending: bool = False


async def _flush_ticker_batch() -> None:
    global _ticker_batch_pending
    await asyncio.sleep(0.1)     # 100ms coalescing window; yields to event loop
    with _ticker_batch_lock:
        batch = _ticker_batch[:]
        _ticker_batch.clear()
        _ticker_batch_pending = False
    if batch and manager.active:
        await manager.broadcast(_dumps({"type": "ticker_batch", "updates": batch}))


def _on_ticker(sig: StockSignal, n_done: int, n_total: int) -> None:
    """Per-ticker callback — coalesced into batches to avoid event-loop flooding."""
    global _ticker_batch_pending
    if _event_loop is None:
        return
    with _ticker_batch_lock:
        _ticker_batch.append({"signal": sig.to_dict(), "n_done": n_done, "n_total": n_total})
        if not _ticker_batch_pending:
            _ticker_batch_pending = True
            asyncio.run_coroutine_threadsafe(_flush_ticker_batch(), _event_loop)


# ── ThinkorSwim auto-trade toggle ────────────────────────────────────────────
_tos_auto_trade: bool = os.getenv("SCHWAB_AUTO_TRADE", "false").lower() == "true"

# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    import routers._deps as _deps_mod
    _deps_mod._event_loop = asyncio.get_running_loop()
    global _event_loop
    _event_loop = _deps_mod._event_loop

    # Auth system: init tables + seed admin user — must succeed; fail hard if not.
    # Swallowing this exception would leave the app running without auth tables,
    # which means every request would 500 on the first DB hit.
    from auth.models import init_tables as _auth_init_tables
    from auth.seed import seed_admin
    from agent.after_hours_monitor import init_db as _ah_init_db
    from agent.historical_cache import init_db as _hc_init_db
    from agent.multi_tf_backtest import init_db as _mtf_init_db
    from historical.store import init_tables as _hist_init_tables
    from agent.context_store import init_db as _ctx_init_db
    from agent.system_alerts import init_db as _alerts_init_db
    from agent.audit_log import init_db as _audit_init_db
    _auth_init_tables()
    seed_admin()
    _ah_init_db()
    _hc_init_db()
    _mtf_init_db()
    _hist_init_tables()
    _ctx_init_db()  # context intel tables (context_events, ticker_context_features, earnings_calendar)
    _alerts_init_db()  # system_alerts table for dashboard alert banner
    _audit_init_db()   # audit_log table for the decision trail

    # Load persisted runtime config from DB so saved settings survive container restarts.
    # seed_defaults() only writes keys that aren't already in the DB (no overwrites).
    from agent.config_manager import config as _cfg
    _cfg.load()
    _cfg.seed_defaults()
    _cfg.start_listener()

    # Warm the market-hours cache before the first scan so get_market_session()
    # doesn't block on its first call mid-scan.  This runs in the background
    # executor so it doesn't delay startup if Schwab is temporarily unreachable.
    try:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, refresh_market_hours_cache)
    except Exception:
        pass

    _load_signal_cache()   # pre-populate cache before any scan runs
    scanner.register_callback(_on_signals)
    scanner.register_per_ticker_callback(_on_ticker)
    if _SCANNER_ENABLED:
        scanner.start_background()
    else:
        logging.getLogger(__name__).info("[Startup] Scanner disabled (NASDAQ_SCANNER_ENABLED=0)")
        try:
            from agent.signal_snapshot import subscribe_scan_results as _sub_scan
            _sub_scan(_on_valkey_scan)
            logging.getLogger(__name__).info("[Startup] Valkey scan subscription started (API-only mode)")
        except Exception as _sub_e:
            logging.getLogger(__name__).warning("[Startup] Valkey scan subscription failed: %s", _sub_e)

    if _LEARNER_ENABLED:
        learning_engine.start()
    else:
        logging.getLogger(__name__).info("[Startup] Learning engine disabled (NASDAQ_LEARNER_ENABLED=0)")

    # Weekend learner — give it a broadcast handle, then auto-start if it's a weekend
    def _wl_broadcast(payload: dict) -> None:
        if _event_loop:
            asyncio.run_coroutine_threadsafe(
                manager.broadcast(json.dumps(payload)), _event_loop
            )
    weekend_learner.register_broadcast(_wl_broadcast)

    # Paper-trade instant push — broadcast lightweight pt_update immediately on
    # every open or close so the dashboard pills (TRADES TODAY / P&L / W/L)
    # reflect the change in real-time without waiting for the next scan cycle.
    def _on_trade_event(event: str, ticker: str) -> None:
        if not _event_loop:
            return
        try:
            from agent.paper_trading import get_summary as _pt_sum, get_open_trades as _pt_open
            pt_stats   = _pt_sum()
            open_trades = {t["ticker"]: t for t in _pt_open()}
            payload = _dumps({
                "type":        "pt_update",
                "event":       event,
                "ticker":      ticker,
                "pt_stats":    pt_stats,
                "open_trades": open_trades,
            })
            asyncio.run_coroutine_threadsafe(
                manager.broadcast(payload), _event_loop
            )
        except Exception:
            pass
    from agent.paper_trading import register_trade_callback as _reg_trade_cb
    _reg_trade_cb(_on_trade_event)
    if _LEARNER_ENABLED:
        weekend_learner.maybe_start()

    # ── Data-quality startup checks (Priority 10) ─────────────────────────────
    # Non-blocking — runs after all services start so DB tables exist.
    try:
        from agent.startup_checks import run_startup_checks as _dq_checks
        import threading as _th
        _th.Thread(target=_dq_checks, name="StartupChecks", daemon=True).start()
    except Exception as _dqe:
        logging.getLogger(__name__).warning(f"[Startup] Data-quality checks skipped: {_dqe}")
    # Sweep any trades that were left open from a previous session
    try:
        from agent.paper_trading import close_stale_positions
        _stale = close_stale_positions()
        if _stale:
            logging.getLogger(__name__).warning(
                f"Startup: closed {_stale} stale open position(s) from prior session"
            )
    except Exception as _sp_e:
        logging.getLogger(__name__).warning(f"Startup stale-trade sweep failed: {_sp_e}")

    # ── EOD watchdog — fires at 3:45 PM ET independent of the scanner ────────
    # Guarantees positions are closed even if the scanner loop is stalled.
    # Gated by NASDAQ_SCHEDULER_ENABLED so it can move to the scheduler container.
    if _SCHEDULER_ENABLED:
        def _eod_watchdog():
            import time as _time
            import zoneinfo as _zi
            from datetime import datetime as _dt
            _log = logging.getLogger("eod_watchdog")
            _fired_on: set = set()
            while True:
                try:
                    now_et = _dt.now(_zi.ZoneInfo("America/New_York"))
                    hm = now_et.hour * 60 + now_et.minute
                    today = now_et.date()
                    if now_et.weekday() < 5 and 945 <= hm < 960 and today not in _fired_on:
                        _fired_on.add(today)
                        try:
                            from agent.paper_trading import close_all_positions_eod
                            n = close_all_positions_eod(reason="EOD_WATCHDOG_3:45PM")
                            if n:
                                _log.warning(f"[EOD Watchdog] Force-closed {n} position(s) at 3:45 PM ET")
                        except Exception as _e:
                            _log.error(f"[EOD Watchdog] Close failed: {_e}", exc_info=True)
                    if len(_fired_on) > 10:
                        _fired_on = set(sorted(_fired_on)[-5:])
                except Exception:
                    pass
                _time.sleep(30)

        _wd = _threading.Thread(target=_eod_watchdog, daemon=True, name="eod-watchdog")
        _wd.start()
        logging.getLogger(__name__).info("EOD watchdog started — will force-close all positions at 3:45 PM ET")
    else:
        logging.getLogger(__name__).info("[Startup] Scheduler disabled (NASDAQ_SCHEDULER_ENABLED=0) — EOD watchdog not started")

    if not _MARKET_DATA_ENABLED:
        logging.getLogger(__name__).info("[Startup] Market data disabled (NASDAQ_MARKET_DATA_ENABLED=0) — Schwab streamer/poller not started")
        # Still subscribe to Valkey md:prices so the WebSocket price bridge is
        # live when prices are published by the scanner container's streamer.
        _ensure_tick_broadcast_registered()
    from config import SCHWAB_ENABLED
    if _MARKET_DATA_ENABLED and SCHWAB_ENABLED:
        from config import NASDAQ_TICKERS as _nq_tickers
        _streamer_started = False

        # ── Attempt 1: WebSocket streamer (Accounts+Trading app) ──────────────
        # Provides sub-100ms Level 1 equities + 1-min candles + futures.
        # Requires SCHWAB_CLIENT_ID / SCHWAB_CLIENT_SECRET + prior OAuth.
        if os.getenv("SCHWAB_CLIENT_ID"):
            try:
                from agent.broker.schwab_auth import load_stored_tokens as _load_at
                if _load_at(schedule_refresh=False):
                    start_streamer(list(_nq_tickers))
                    _ensure_tick_broadcast_registered()
                    _streamer_started = True
                    logging.getLogger(__name__).info(
                        "Schwab WebSocket streamer started — real-time Level 1 data active."
                    )
                else:
                    logging.getLogger(__name__).warning(
                        "Schwab A+T tokens not found — visit /schwab/auth/at to authenticate "
                        "the WebSocket streamer."
                    )
            except Exception as _se:
                logging.getLogger(__name__).warning(f"Schwab WS streamer startup failed: {_se}")

        # ── Attempt 2: REST MDPoller (Market Data app) — fallback / supplement ─
        # Always start if MD tokens are present; runs alongside the WS streamer
        # as a fallback for gaps (token expiry, market-hours-only streaming).
        if os.getenv("SCHWAB_MD_CLIENT_ID"):
            try:
                ok_md = load_stored_md_tokens(schedule_refresh=False)
                if ok_md:
                    logging.getLogger(__name__).info(
                        "Schwab Market Data connected — starting parallel quote poller."
                    )
                    # Keep this short so the dashboard resumes second-level
                    # quote updates quickly after deploy/restart. Operators can
                    # raise it with NASDAQ_MD_STARTUP_DELAY_S if Schwab rate
                    # pressure is observed during a cold OHLCV warmup.
                    _md_startup_delay = float(os.getenv("NASDAQ_MD_STARTUP_DELAY_S", "10"))
                    start_md_poller(list(_nq_tickers), interval=1.0,
                                    parallel_batches=2, startup_delay_s=_md_startup_delay)
                    _ensure_tick_broadcast_registered()
                else:
                    logging.getLogger(__name__).warning(
                        "Schwab Market Data token not found — visit /schwab/auth/md to authenticate."
                    )
            except Exception as _be:
                logging.getLogger(__name__).warning(f"Schwab Market Data startup failed: {_be}")

        if not _streamer_started and not is_md_poller_running():
            logging.getLogger(__name__).warning(
                "No Schwab data source active — scanner will use Twelve Data / cache only."
            )

        # Subscribe to schwab:tokens_refreshed so web-api reloads the in-memory
        # token when token-service rotates it — otherwise web-api would keep the
        # stale access token after each background refresh cycle.
        def _web_api_token_reload_loop() -> None:
            import time as _t
            while True:
                try:
                    from agent.valkey_client import _get_client as _vk_get
                    _vc = _vk_get()
                    if _vc is None:
                        _t.sleep(30)
                        continue
                    _ps = _vc.pubsub()
                    _ps.subscribe("schwab:tokens_refreshed")
                    for _msg in _ps.listen():
                        if _msg and _msg.get("type") == "message":
                            from agent.broker.schwab_auth import load_stored_tokens, load_stored_md_tokens
                            load_stored_tokens(schedule_refresh=False)
                            load_stored_md_tokens(schedule_refresh=False)
                            logging.getLogger(__name__).info(
                                "[web-api] Token refresh detected — in-memory tokens reloaded."
                            )
                except Exception as _exc:
                    logging.getLogger(__name__).debug(
                        "[web-api] token reload sub error: %s — retrying in 30s", _exc
                    )
                    _t.sleep(30)

        import threading as _thr
        _thr.Thread(target=_web_api_token_reload_loop, daemon=True,
                    name="web-api-token-reload").start()

    else:
        logging.getLogger(__name__).info(
            "Schwab disabled (SCHWAB_ENABLED not set) — running on Twelve Data only."
        )

    yield
    scanner.stop()
    learning_engine.stop()


app = FastAPI(title="NASDAQ Scalping Agent", lifespan=lifespan)

# ── Bot / scanner probe blocking ──────────────────────────────────────────────
# Silently drop requests for paths that are never valid on this API server.
# Avoids noisy 404s from WordPress scanners, credential harvesters, etc.
_BOT_PATH_PREFIXES = (
    "/.env", "/.git", "/wp-", "/wp_", "/wordpress", "/admin/",
    "/phpmyadmin", "/pma", "/.aws", "/.ssh", "/config.php",
    "/xmlrpc", "/cgi-bin", "/shell", "/cmd", "/eval",
)
_BOT_EXTENSIONS = (".php", ".asp", ".aspx", ".jsp", ".cgi", ".bak", ".sql",
                   ".tar", ".gz", ".zip", ".env")


@app.middleware("http")
async def block_bot_probes(request: Request, call_next):
    path = request.url.path.lower()
    if any(path.startswith(p) for p in _BOT_PATH_PREFIXES):
        return Response(status_code=404)
    if any(path.endswith(ext) for ext in _BOT_EXTENSIONS):
        return Response(status_code=404)
    return await call_next(request)


# CORS — restrict to same-origin + known frontends
_CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:8000,http://localhost:3000,http://localhost:80"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Auth routers
from auth.router import router as auth_router
from auth.admin_router import router as admin_router
from auth.dependencies import (
    AuthenticatedUser,
    require_admin,
    require_trader,
    get_current_user,
)
app.include_router(auth_router)
app.include_router(admin_router)

# Static files (dashboard)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "web", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Domain routers ────────────────────────────────────────────────────────────

from routers.signals       import router as signals_router
from routers.system        import router as system_router
from routers.paper_trading import router as paper_trading_router
from routers.backtest      import router as backtest_router
from routers.learning      import router as learning_router
from routers.ml_models     import router as ml_models_router
from routers.broker        import router as broker_router
from routers.context       import router as context_router
from routers.notify        import router as notify_router
from routers.config_router import router as config_router
from routers.streaming     import router as streaming_router
from routers.algo          import router as algo_router
from routers.alerts        import router as alerts_router
from routers.audit         import router as audit_router
from routers.scalp         import router as scalp_router

# Re-export helpers for backward compatibility with test suite and other tooling
from routers.streaming import _safe_ws_close  # noqa: F401  (tests import this from main)

app.include_router(signals_router)
app.include_router(system_router)
app.include_router(paper_trading_router)
app.include_router(backtest_router)
app.include_router(learning_router)
app.include_router(ml_models_router)
app.include_router(broker_router)
app.include_router(context_router)
app.include_router(notify_router)
app.include_router(config_router)
app.include_router(streaming_router)
app.include_router(algo_router)
app.include_router(alerts_router)
app.include_router(audit_router)
app.include_router(scalp_router)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
