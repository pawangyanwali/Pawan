"""
FastAPI entry point.
Serves the static web dashboard and a WebSocket endpoint that pushes
real-time stock signals to all connected clients.
"""

import asyncio
import html as _html
import json
import logging
import os
import subprocess
import sys
import threading as _threading
import time
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Set

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# SSE support — gracefully degraded if sse-starlette is not installed
try:
    from sse_starlette.sse import EventSourceResponse as _EventSourceResponse
    _SSE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _EventSourceResponse = None
    _SSE_AVAILABLE = False

# ── Service mode flags ────────────────────────────────────────────────────────
# Defined before agent imports so heavy subsystems are never loaded in containers
# that don't run them.  All default to enabled — preserves single-process behaviour.
_SCANNER_ENABLED     = os.getenv("NASDAQ_SCANNER_ENABLED",     "1") != "0"
_MARKET_DATA_ENABLED = os.getenv("NASDAQ_MARKET_DATA_ENABLED", "1") != "0"
_LEARNER_ENABLED     = os.getenv("NASDAQ_LEARNER_ENABLED",     "1") != "0"
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
from agent.signal_tracker import get_stats, get_recent_signals, get_observation_summary
from agent.position_sizing import calculate as calc_position
from agent.paper_trading import get_summary as pt_summary, get_open_trades, get_closed_trades, get_daily_pnl, get_today_pnl, get_equity_curve, get_weekly_pnl, get_ticker_pnl, get_account_state, update_account_config, get_algo_performance
from agent.macro_calendar import check_macro_event, get_upcoming_events
from agent.live_backtest import get_performance_stats, get_tracking_signals, get_recent_resolved, get_price_path
from agent.backtest_reporter import get_broadcast_summary, get_full_report
from agent.adaptive_filter import get_status as af_get_status, reset_filter as af_reset_filter
from agent.after_hours_monitor import get_all_biases as ah_get_all
if _LEARNER_ENABLED:
    from agent.learning_engine import learning_engine, get_learning_log
    import agent.weekend_learner as weekend_learner
else:
    class _NoopLearner:  # type: ignore[no-redef]
        def start(self): pass
        def stop(self): pass
        def get_status(self): return {}
    learning_engine = _NoopLearner()  # type: ignore[assignment]
    def get_learning_log(limit: int = 100): return []  # type: ignore[misc]
    class _NoopWeekendLearner:  # type: ignore[no-redef]
        def register_broadcast(self, *a): pass
        def maybe_start(self): pass
        def get_status(self): return {}
    weekend_learner = _NoopWeekendLearner()  # type: ignore[assignment]
from auth.dependencies import require_viewer, require_analyst, AuthenticatedUser

try:
    if not _LEARNER_ENABLED:
        raise ImportError("learner disabled")
    from agent.algo_learning_p2 import get_phase2_engine as _get_p2_engine
    _P2_AVAILABLE = True
except (ImportError, Exception):
    _get_p2_engine = None  # type: ignore[assignment]
    _P2_AVAILABLE = False

try:
    from agent.algo_learning_engine import get_algo_params as _get_algo_params
    _ALE_AVAILABLE = True
except (ImportError, Exception):
    _get_algo_params = None  # type: ignore[assignment]
    _ALE_AVAILABLE = False

try:
    if not _LEARNER_ENABLED:
        raise ImportError("learner disabled")
    from agent.walk_forward_trainer import get_walk_forward_trainer as _get_wf_trainer
    _WFT_AVAILABLE = True
except (ImportError, Exception):
    _get_wf_trainer = None  # type: ignore[assignment]
    _WFT_AVAILABLE = False
from agent.broker.schwab_auth import (
    load_stored_tokens, load_stored_md_tokens,
    get_token_status, get_md_token_status,
    build_auth_url, exchange_auth_code,
    build_md_auth_url, exchange_md_auth_code,
)
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
from agent.broker.schwab_client import get_positions, get_account_summary, get_orders
from agent.broker.order_bridge import maybe_place_tos_order, get_daily_status
from agent.notifier import notify_signal as _notify_signal, get_config as _notify_cfg, configure as _notify_configure, send_telegram as _send_telegram
from config import (
    DEFAULT_ACCOUNT_SIZE, DEFAULT_RISK_PCT, MAX_POSITION_PCT,
    load_watchlist, save_watchlist, NASDAQ_TICKERS,
    CLUSTER_A_TICKERS, CLUSTER_B_TICKERS, CLUSTER_C_TICKERS, TICKER_CLUSTER,
)


import math

# Dedicated thread pool for paper-trading DB reads so they're never blocked by
# the default asyncio executor being saturated with scanner/ML/broker tasks.
# 2 threads is enough: get_closed_trades + get_summary + get_open_trades run
# sequentially inside (they share _lock), but we never want them to wait for
# an unrelated task to free a thread.
_pt_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="pt_db")


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None so json.dumps never crashes."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize(x) for x in obj.tolist()]
    return obj


class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalars to native Python types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _dumps(obj) -> str:
    return json.dumps(_sanitize(obj), cls=_NumpyEncoder)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        # ws is already accepted in websocket_endpoint before auth runs
        self.active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.active.discard(ws)

    async def broadcast(self, message: str) -> None:
        # Snapshot first — prevents RuntimeError if a disconnect() fires during await.
        snapshot = list(self.active)
        if not snapshot:
            return

        async def _send(ws: WebSocket) -> WebSocket | None:
            try:
                # 2s timeout: dead clients cleaned up quickly so they don't delay
                # subsequent broadcasts (keepalive pings, price messages).
                await asyncio.wait_for(ws.send_text(message), timeout=2.0)
                return None
            except Exception:
                return ws

        # Yield once before sending so keepalive pings and price messages can
        # interleave with scanner ticker_update batches on the event loop.
        await asyncio.sleep(0)
        # Send to all clients in parallel — a slow client no longer blocks fast ones.
        dead = await asyncio.gather(*[_send(ws) for ws in snapshot])
        for ws in dead:
            if ws is not None:
                self.active.discard(ws)


manager = ConnectionManager()

# Captured at startup so the scanner background thread can schedule broadcasts
_event_loop: asyncio.AbstractEventLoop | None = None

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
            "target_win_rate":   learn_summary.get("target_win_rate", 62.0),
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

    # Write durable snapshot to Valkey — web-api reads this on restart instead
    # of waiting for the next scan cycle (Step 3 of the containerisation plan).
    try:
        from agent.signal_snapshot import write_latest as _snap_write
        _snap_write(
            signals       = sigs_dicts,
            regime        = regime.to_dict(),
            session       = session,
            scanned_count = len(signals),
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
        "universe_total": _get_universe_total(),
        "scanned_count": len(signals),
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
                "target_win_rate":   learn_summary.get("target_win_rate", 62.0),
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
            "universe_total": _get_universe_total(),
            "scanned_count":  len(sigs_dicts),
        })
        asyncio.run_coroutine_threadsafe(manager.broadcast(payload), _event_loop)

        # Price patch so the surgical DOM update fires
        if manager.active:
            _price_patch = {
                s["ticker"]: {
                    "last":       s.get("price", 0),
                    "open":       s.get("open_price", 0) or 0,
                    "pct_change": s.get("change_pct", 0) or 0,
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
    global _event_loop
    _event_loop = asyncio.get_running_loop()

    # Auth system: init tables + seed admin user — must succeed; fail hard if not.
    # Swallowing this exception would leave the app running without auth tables,
    # which means every request would 500 on the first DB hit.
    from auth.models import init_tables as _auth_init_tables
    from auth.seed import seed_admin
    from agent.after_hours_monitor import init_db as _ah_init_db
    from agent.historical_cache import init_db as _hc_init_db
    from agent.multi_tf_backtest import init_db as _mtf_init_db
    from historical.store import init_tables as _hist_init_tables
    _auth_init_tables()
    seed_admin()
    _ah_init_db()
    _hc_init_db()
    _mtf_init_db()
    _hist_init_tables()

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
        # Load stored tokens into memory so /api/broker/status reflects real
        # token state even though we don't start the streamer here.
        try:
            if os.getenv("SCHWAB_CLIENT_ID"):
                load_stored_tokens()
            if os.getenv("SCHWAB_MD_CLIENT_ID"):
                load_stored_md_tokens()
        except Exception as _tl_err:
            logging.getLogger(__name__).debug("Token pre-load (status-only): %s", _tl_err)
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
                if _load_at():
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
                ok_md = load_stored_md_tokens()
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
    else:
        logging.getLogger(__name__).info(
            "Schwab disabled (SCHWAB_ENABLED not set) — running on Twelve Data only."
        )
    yield
    scanner.stop()
    learning_engine.stop()


app = FastAPI(title="NASDAQ Scalping Agent", lifespan=lifespan)

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


# ── HTTP routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/login", response_class=HTMLResponse)
@app.get("/login.html", response_class=HTMLResponse)
async def login_page():
    return FileResponse(os.path.join(STATIC_DIR, "login.html"))


@app.get("/admin.html", response_class=HTMLResponse)
async def admin_page():
    return FileResponse(os.path.join(STATIC_DIR, "admin.html"))


@app.get("/api/signals")
async def get_signals(_user: AuthenticatedUser = Depends(require_viewer)):
    """REST endpoint: returns the latest cached scan results.

    Serves live in-memory signals when a scan has completed, otherwise falls
    back to the disk cache so the page is never blank on a warm restart.
    """
    sigs, last_scan, from_cache = _current_signal_snapshot()
    return {
        "last_scan":  last_scan,
        "count":      len(sigs),
        "signals":    sigs,
        "from_cache": from_cache,
        "universe_total": _get_universe_total(),
        "empty_reason": None if sigs else "no_scanner_signal_cache_or_quote_data",
        "scanning":   scanner.is_running and not sigs,
    }


@app.get("/api/health")
async def health():
    from agent.data_fetcher import get_credit_usage
    from agent.valkey_client import health_status as vk_health
    sigs, last_scan, from_cache = _current_signal_snapshot()
    return {
        "status": "ok",
        "is_running": scanner.is_running,
        "last_scan": last_scan,
        "tickers_tracked": len(sigs),
        "from_cache": from_cache,
        "ws_clients": len(manager.active),
        "api_credits": get_credit_usage(),
        "valkey": vk_health(),
    }


def _container_health(valkey_connected: bool) -> dict:
    """
    Derive container liveness from Valkey heartbeat keys.

    Each container writes a key with a short TTL so expiry = container down:
      web-api    — always "up" (this process is answering the request)
      scanner    — scan:latest key; written after every scan cycle (can take up to ~400s)
      learner    — learner:status key; written every 60s (TTL 300s)
      scheduler  — scheduler:heartbeat key; written every 30s (TTL 90s)

    Returns a dict keyed by container name with:
      up (bool), last_seen_ago_s (float|None), detail (str)
    """
    now = time.time()
    # Scanner scan cycle can take up to ~400s on 477-ticker universe with 8 workers.
    # Allow 660s (one full cycle + 270s buffer) before marking it down.
    _SCANNER_STALE_S = 660
    result: dict = {
        "web-api": {"up": True, "last_seen_ago_s": 0.0, "detail": "serving this response"},
    }

    if not valkey_connected:
        for name in ("scanner", "learner", "scheduler"):
            result[name] = {"up": None, "last_seen_ago_s": None, "detail": "Valkey unreachable"}
        return result

    try:
        from agent.valkey_client import _get_client
        client = _get_client()
        if client is None:
            for name in ("scanner", "learner", "scheduler"):
                result[name] = {"up": None, "last_seen_ago_s": None, "detail": "no Valkey client"}
            return result

        # scanner — scan:latest written after each cycle; JSON with optional "ts" field
        try:
            raw = client.get("scan:latest")
            if raw:
                d = json.loads(raw)
                scan_ts = float(d.get("ts", 0))
                ago = round(now - scan_ts, 1) if scan_ts else None
                up = ago is not None and ago < _SCANNER_STALE_S
                result["scanner"] = {
                    "up": up,
                    "last_seen_ago_s": ago,
                    "detail": f"last scan {ago}s ago" if ago is not None else "key present, no ts",
                }
            else:
                result["scanner"] = {"up": False, "last_seen_ago_s": None, "detail": "no scan:latest key"}
        except Exception as exc:
            result["scanner"] = {"up": None, "last_seen_ago_s": None, "detail": str(exc)}

        # learner — learner:status written every 60s
        try:
            raw = client.get("learner:status")
            if raw:
                d = json.loads(raw)
                ts = float(d.get("ts", 0))
                ago = round(now - ts, 1) if ts else None
                up = ago is not None and ago < 300
                result["learner"] = {
                    "up": up,
                    "last_seen_ago_s": ago,
                    "detail": f"heartbeat {ago}s ago" if ago is not None else "key present, no ts",
                }
            else:
                result["learner"] = {"up": False, "last_seen_ago_s": None, "detail": "no learner:status key"}
        except Exception as exc:
            result["learner"] = {"up": None, "last_seen_ago_s": None, "detail": str(exc)}

        # scheduler — scheduler:heartbeat written every 30s
        try:
            raw = client.get("scheduler:heartbeat")
            if raw:
                d = json.loads(raw)
                ts = float(d.get("ts", 0))
                ago = round(now - ts, 1) if ts else None
                up = ago is not None and ago < 120
                result["scheduler"] = {
                    "up": up,
                    "last_seen_ago_s": ago,
                    "detail": f"heartbeat {ago}s ago" if ago is not None else "key present, no ts",
                }
            else:
                result["scheduler"] = {"up": False, "last_seen_ago_s": None, "detail": "no scheduler:heartbeat key"}
        except Exception as exc:
            result["scheduler"] = {"up": None, "last_seen_ago_s": None, "detail": str(exc)}

    except Exception as exc:
        for name in ("scanner", "learner", "scheduler"):
            result[name] = {"up": None, "last_seen_ago_s": None, "detail": str(exc)}

    return result


@app.get("/api/services")
async def services_status(_user: AuthenticatedUser = Depends(require_viewer)):
    """
    Aggregate health of all infrastructure services for the dashboard panel.
    Returns connectivity status for: Scanner, MD Poller, Valkey, RDS (PostgreSQL).
    """
    from agent.valkey_client import health_status as vk_health
    from agent.broker.schwab_streamer import get_streamer_status

    vk = vk_health()

    # When scanner runs in its own container, read streamer/poller status from
    # Valkey (written by scanner_service every 15s) instead of the local noop state.
    streamer = get_streamer_status()
    if not _MARKET_DATA_ENABLED and vk.get("connected"):
        try:
            from agent.valkey_client import _get_client as _vk_c
            _vc = _vk_c()
            if _vc:
                _raw = _vc.get("scanner:streamer")
                if _raw:
                    _sd = json.loads(_raw)
                    if time.time() - _sd.get("ts", 0) < 60:
                        streamer = _sd
        except Exception:
            pass

    # RDS check — lightweight: just try to get a connection from the pool
    rds_ok = False
    rds_error = None
    try:
        import psycopg2, os as _os
        conn = psycopg2.connect(
            host=_os.getenv("PGHOST", ""),
            port=int(_os.getenv("PGPORT", "5432")),
            dbname=_os.getenv("PGDATABASE", "nasdaq_agent"),
            user=_os.getenv("PGUSER", ""),
            password=_os.getenv("PGPASSWORD", ""),
            connect_timeout=3,
        )
        conn.close()
        rds_ok = True
    except Exception as _re:
        rds_error = str(_re)

    ws_st  = streamer.get("ws_streamer", {})
    md_st  = streamer.get("md_poller", {})
    sigs, last_scan, from_cache = _current_signal_snapshot()

    # When scanner runs in its own container, derive running state from the
    # Valkey scan:latest key age rather than the local scanner.is_running (always False).
    scanner_running = scanner.is_running
    if not _SCANNER_ENABLED and not scanner_running:
        try:
            from agent.valkey_client import _get_client as _vk_sc
            _vc = _vk_sc()
            if _vc:
                _raw = _vc.get("scan:latest")
                if _raw:
                    _ts = json.loads(_raw).get("ts", 0)
                    scanner_running = bool(_ts and (time.time() - float(_ts)) < 660)
        except Exception:
            pass

    return {
        "scanner": {
            "running":    scanner_running,
            "last_scan":  last_scan,
            "tickers":    len(sigs),
            "from_cache": from_cache,
            "ws_clients": len(manager.active),
        },
        "ws_streamer": {
            "running":     ws_st.get("running", False),
            "connected":   ws_st.get("connected", False),
            "live_quotes": ws_st.get("live_quotes", 0),
            "nq_bias":     ws_st.get("nq_bias", 0.0),
            "error":       ws_st.get("error"),
        },
        "md_poller": {
            "running":       md_st.get("running", False),
            "cycle":         md_st.get("cycle", 0),
            "last_ok_ago_s": md_st.get("last_ok_ago_s"),
            "live_quotes":   md_st.get("live_quotes", 0),
            "error":         md_st.get("error"),
        },
        "valkey": vk,
        "rds": {
            "connected": rds_ok,
            "error":     rds_error,
        },
        "containers": _container_health(vk.get("connected", False)),
    }


@app.get("/api/credit-usage")
async def credit_usage():
    """Schwab Market Data has no credit limits — returns zeros."""
    from agent.data_fetcher import get_credit_usage
    return get_credit_usage()


@app.get("/api/regime")
async def get_regime_endpoint():
    """Return current market regime (SPY/QQQ based)."""
    regime = get_regime()
    session = get_session_info()
    return {"regime": regime.to_dict(), "session": session}


@app.get("/api/signal-history")
async def signal_history(ticker: str = None, limit: int = 50):
    """Return recent signal history from SQLite tracker."""
    return {
        "signals": get_recent_signals(limit=limit),
        "stats":   get_stats(ticker=ticker),
    }


@app.get("/api/position-size")
async def position_size_endpoint(
    entry:        float,
    stop:         float,
    account_size: float = DEFAULT_ACCOUNT_SIZE,
    risk_pct:     float = DEFAULT_RISK_PCT,
    confidence:   float = 50.0,
    direction:    str   = "BUY",
):
    """Calculate position size for given entry/stop/account parameters."""
    ps = calc_position(
        account_size=account_size,
        entry=entry,
        stop=stop,
        risk_pct=risk_pct,
        confidence=confidence,
        max_position_pct=MAX_POSITION_PCT,
    )
    return ps.to_dict()


@app.get("/api/paper-trading")
async def paper_trading_endpoint(_user: AuthenticatedUser = Depends(require_viewer)):
    """Return paper trading summary, open and recent closed trades."""
    from datetime import date

    try:
        # Dedicated thread pool so these reads are never queued behind
        # scanner/ML/broker tasks that saturate the default asyncio executor.
        loop = asyncio.get_running_loop()
        closed, summary, open_trades = await asyncio.gather(
            loop.run_in_executor(_pt_executor, get_closed_trades, 200),
            loop.run_in_executor(_pt_executor, pt_summary),
            loop.run_in_executor(_pt_executor, get_open_trades),
        )
    except Exception as _e:
        logging.getLogger(__name__).error(f"[PT] paper-trading endpoint error: {_e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(_e), "summary": {
            "open": 0, "closed": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "avg_pnl": 0.0, "total_dollar_pnl": 0.0,
            "display_period": "all-time",
        }, "open_trades": [], "closed_trades": []})

    today_str    = date.today().isoformat()
    today_trades = [t for t in closed if (t.get("closed_at") or "")[:10] == today_str]
    all_trades   = closed

    def _stats(trades):
        total   = len(trades)
        wins    = sum(1 for t in trades if (t.get("pnl_dollar") or 0) > 0)
        dollars = [t["pnl_dollar"] for t in trades if t.get("pnl_dollar") is not None]
        pcts    = [t["pnl_pct"]    for t in trades if t.get("pnl_pct")    is not None]
        return {
            "closed":           total,
            "wins":             wins,
            "losses":           total - wins,
            "win_rate":         round(wins / total * 100, 1) if total else 0.0,
            "avg_pnl":          round(sum(pcts) / len(pcts), 3) if pcts else 0.0,
            "total_dollar_pnl": round(sum(dollars), 2) if dollars else 0.0,
        }

    today_stats = _stats(today_trades)
    all_stats   = _stats(all_trades)

    # Use TODAY stats when there are enough trades; fall back to all-time so the
    # dashboard doesn't show all zeros every morning before the first trade closes.
    display_stats  = today_stats if today_stats["closed"] >= 3 else all_stats
    display_period = "today" if today_stats["closed"] >= 3 else "all-time"
    summary.update({
        "closed":           display_stats["closed"],
        "wins":             display_stats["wins"],
        "losses":           display_stats["losses"],
        "win_rate":         display_stats["win_rate"],
        "avg_pnl":          display_stats["avg_pnl"],
        "total_pnl":        display_stats["avg_pnl"],
        "total_dollar_pnl": today_stats["total_dollar_pnl"],
        "all_time_dollar":  all_stats["total_dollar_pnl"],
        "all_time_closed":  all_stats["closed"],
        "today_closed":     today_stats["closed"],
        "display_period":   display_period,
    })
    return {
        "summary":       summary,
        "open_trades":   open_trades,
        "closed_trades": closed[:30],
        "_debug_pnl":    {
            "n_trades":      len(all_trades),
            "total_dollar":  all_stats["total_dollar_pnl"],
            "today_dollar":  today_stats["total_dollar_pnl"],
            "per_trade":     [(t["ticker"], t.get("pnl_dollar") or 0, (t.get("closed_at") or "")[:10]) for t in all_trades],
        },
    }


@app.get("/api/account-state")
async def api_account_state():
    """Full account state: capital, P&L, drawdown, config."""
    try:
        loop = asyncio.get_running_loop()
        # Pass current live prices for unrealized P&L
        try:
            from agent.broker.schwab_market_data import get_live_quotes
            prices = {t: q.get("last", 0) for t, q in get_live_quotes().items() if q.get("last")}
        except Exception:
            prices = {}
        state = await loop.run_in_executor(None, lambda: get_account_state(prices))
        return state
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/account-config")
async def api_update_account_config(
    total_budget:      float | None = None,
    max_trade_pct:     float | None = None,
    max_allocated_pct: float | None = None,
    max_open_trades:   int   | None = None,
    _current: AuthenticatedUser = Depends(require_trader),
):
    """Update paper trading budget and position limits."""
    try:
        loop = asyncio.get_running_loop()
        cfg = await loop.run_in_executor(
            None,
            lambda: update_account_config(total_budget, max_trade_pct, max_allocated_pct, max_open_trades)
        )
        return {"success": True, "config": cfg}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/deep-model/status")
async def deep_model_status():
    """Deep BiLSTM model training status and architecture info."""
    from agent.deep_model import get_model_info
    return get_model_info()


@app.get("/api/risk-status")
async def risk_status():
    """Full PRD risk engine status: circuit breaker, profit protect, portfolio heat, session."""
    from agent.risk_controls import get_risk_status
    return get_risk_status()


@app.get("/api/premarket-scan")
async def premarket_scan_endpoint():
    """Pre-market gapper scan results and today's focus watchlist."""
    try:
        from agent.premarket_scanner import get_scan_status, get_focus_watchlist
        status = get_scan_status()
        return {**status, "focus_watchlist": get_focus_watchlist()}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/premarket-scan/run")
async def trigger_premarket_scan(background_tasks: BackgroundTasks,
                                  _user: AuthenticatedUser = Depends(require_analyst)):
    """Manually trigger a pre-market gapper scan."""
    try:
        from agent.premarket_scanner import run_premarket_scan_background
        run_premarket_scan_background()
        return {"status": "started"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ml-status")
async def ml_status(_user: AuthenticatedUser = Depends(require_viewer)):
    """Aggregate status for all ML model types (includes blend weights and pipeline metrics)."""
    from agent.deep_model import get_model_info, get_training_history, is_training_active, is_trained as deep_is_trained
    from agent.ml_model import (
        _model_registry, _daily_model_registry,
        _reversal_model_registry, _ensemble_registry,
        _swing_model_registry, get_retrain_progress,
    )

    def _count(registry):
        total   = len(registry)
        trained = sum(1 for m in registry.values() if getattr(m, 'trained', False))
        return {"total": total, "trained": trained}

    # Inline blend weights
    blend_stats = None
    try:
        from agent.signal_blender import get_blender
        blend_stats = get_blender().get_stats()
    except Exception:
        pass

    # Inline pipeline metrics
    pipeline_stats = None
    try:
        from agent.pipeline import get_pipeline
        pipeline_stats = get_pipeline().get_metrics()
    except Exception:
        pass

    # Cluster model status (A/B/C BiLSTM)
    cluster_status = {}
    try:
        from agent.deep_model import _cluster_trained, _CLUSTER_CONFIGS
        from config import CLUSTER_A_TICKERS, CLUSTER_B_TICKERS, CLUSTER_C_TICKERS
        cluster_tickers = {"a": CLUSTER_A_TICKERS, "b": CLUSTER_B_TICKERS, "c": CLUSTER_C_TICKERS}
        import os
        for cname in ("a", "b", "c"):
            cfg = _CLUSTER_CONFIGS.get(cname.upper(), {})
            model_path = cfg.get("path", "")
            mtime = None
            if model_path and os.path.exists(str(model_path)):
                mtime = os.path.getmtime(str(model_path))
            cluster_status[cname] = {
                "trained": bool(_cluster_trained.get(cname.upper(), False)),
                "last_trained": mtime,
                "n_tickers": len(cluster_tickers.get(cname, [])),
            }
    except Exception:
        pass

    return {
        "deep_model":        get_model_info(),
        "deep_trained":      deep_is_trained(),
        "is_training_now":   is_training_active(),
        "training_history":  get_training_history(),
        "scalp_models":      _count(_model_registry),
        "daily_models":      _count(_daily_model_registry),
        "reversal_models":   _count(_reversal_model_registry),
        "ensemble_models":   _count(_ensemble_registry),
        "swing_models":      _count(_swing_model_registry),
        "retrain_progress":  get_retrain_progress(),
        "blend_weights":     blend_stats,
        "pipeline_metrics":  pipeline_stats,
        "cluster_status":    cluster_status,
    }


# ── Weekend Learning API ──────────────────────────────────────────────────────

@app.get("/api/weekend-learning/status")
async def wl_status():
    """Current state of the weekend learning pipeline."""
    return weekend_learner.get_status()


@app.post("/api/weekend-learning/start")
async def wl_start(
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Manually kick off the weekend learning pipeline (admin override)."""
    started = weekend_learner.start()
    return {
        "status":  "started" if started else "already_running",
        "message": ("Weekend learning started in background."
                    if started else "Weekend learner is already running."),
    }


@app.post("/api/weekend-learning/stop")
async def wl_stop(
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Signal the weekend learner to stop after the current phase."""
    weekend_learner.stop()
    return {"status": "stop_requested"}


@app.get("/api/weekend-learning/history")
async def wl_history():
    """Cumulative weekend learning outcomes from SQLite records store."""
    return weekend_learner.historical_performance()


@app.get("/api/weekend-learning/cache-stats")
async def wl_cache_stats():
    """OHLCV cache stats — bars per ticker/interval stored so far."""
    from agent.historical_cache import cache_stats
    return cache_stats()


# ── Multi-TF Backtest API ─────────────────────────────────────────────────────

@app.get("/api/backtest/mtf")
async def mtf_summary():
    """
    Aggregated multi-timeframe backtest results from the most recent weekend run.
    Returns win rate, expectancy, Sharpe, and max-drawdown for each TF.
    """
    from agent.multi_tf_backtest import get_summary
    return get_summary()


@app.get("/api/backtest/mtf/history")
async def mtf_history():
    """List of past multi-TF backtest runs with aggregate stats."""
    from agent.multi_tf_backtest import get_run_history
    return get_run_history()


@app.get("/api/backtest/mtf/{ticker}")
async def mtf_ticker(ticker: str):
    """Per-TF performance breakdown for a single ticker."""
    from agent.multi_tf_backtest import get_ticker_stats
    return get_ticker_stats(ticker.upper())


@app.post("/api/ml-retrain")
async def trigger_retrain(
    background_tasks: BackgroundTasks,
    _current: AuthenticatedUser = Depends(require_admin),
):
    """
    Manually trigger a full ML retrain cycle (XGBoost + SwingML + Deep BiLSTM).
    Runs in background — check /api/ml-status for progress.
    """
    from agent.ml_model import retrain_all, _is_retraining
    from config import TRAINING_TICKERS

    if _is_retraining:
        return {"status": "already_running", "message": "Retrain already in progress."}

    try:
        session = get_market_session()
    except Exception:
        session = "UNKNOWN"
    if session != "CLOSED":
        return {
            "status": "deferred",
            "message": f"ML retrain deferred during {session}; run it in the CLOSED window.",
        }

    def _run():
        try:
            retrain_all(TRAINING_TICKERS)
        except Exception as e:
            logger.warning(f"[manual retrain] failed: {e}")

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Retrain started in background. Watch /api/ml-status for progress."}


@app.post("/api/deep-model/train")
async def trigger_deep_train(
    background_tasks: BackgroundTasks,
    _current: AuthenticatedUser = Depends(require_admin),
):
    """
    Manually trigger Deep BiLSTM training only (faster than full retrain).
    Uses cached 15-min data when available.
    """
    from agent.deep_model import is_training_active, retrain_deep_all
    from agent.data_fetcher import fetch_batch_interval
    from config import TRAINING_TICKERS

    if is_training_active():
        return {"status": "already_running", "message": "Deep model training already in progress."}

    try:
        session = get_market_session()
    except Exception:
        session = "UNKNOWN"
    if session != "CLOSED":
        return {
            "status": "deferred",
            "message": f"Deep model training deferred during {session}; run it in the CLOSED window.",
        }

    def _run():
        try:
            logger.info(f"[manual deep train] Fetching 15-min data ({len(TRAINING_TICKERS)} Tier-1)…")
            hist_15m = fetch_batch_interval(TRAINING_TICKERS, "15min", 5000, ttl=3600)
            logger.info(f"[manual deep train] Got {len(hist_15m)} tickers — starting training…")
            retrain_deep_all(hist_15m)
        except Exception as e:
            logger.warning(f"[manual deep train] failed: {e}")

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Deep BiLSTM training started. Check /api/ml-status for epoch progress."}


# ── Historical retrain & backtest ─────────────────────────────────────────────
# These jobs run as detached subprocesses so they never block the Gunicorn
# worker or get killed by its timeout/SIGABRT.  Progress is written to a JSON
# status file by the subprocess; the API endpoints just read that file.

_PACKAGE_DIR = Path(__file__).parent          # nasdaq_agent/
_RETRAIN_STATUS = Path.home() / ".nasdaq_agent" / "hist_retrain_status.json"
_BACKTEST_STATUS = Path.home() / ".nasdaq_agent" / "hist_backtest_status.json"

_hist_retrain_proc: subprocess.Popen | None = None
_hist_backtest_proc: subprocess.Popen | None = None


def _proc_running(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


def _read_status(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


@app.post("/api/historical/retrain")
async def historical_retrain(
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Retrain all ML models using 2-year historical bars.

    Runs as a detached subprocess — never blocks the web worker.
    Poll /api/historical/retrain/status for live progress.
    """
    global _hist_retrain_proc
    if _proc_running(_hist_retrain_proc):
        return {"status": "already_running", "message": "Historical retrain already in progress."}

    cmd = [sys.executable, "-m", "historical", "--retrain", "--interval", "5min", "--workers", "4"]
    _hist_retrain_proc = subprocess.Popen(cmd, cwd=str(_PACKAGE_DIR))
    logger.info("[hist-retrain] Subprocess started (pid=%d)", _hist_retrain_proc.pid)
    return {"status": "started", "message": "Historical retrain started as background process."}


@app.get("/api/historical/retrain/status")
async def historical_retrain_status():
    """Live progress of the historical retrain subprocess (reads status file)."""
    state = _read_status(_RETRAIN_STATUS)
    state["proc_running"] = _proc_running(_hist_retrain_proc)
    # If subprocess exited but file still says running, correct it
    if not state.get("proc_running") and state.get("running"):
        state["running"] = False
    return state


@app.post("/api/historical/backtest/run")
async def historical_backtest_run(interval: str = "5min",
                                   _user: AuthenticatedUser = Depends(require_analyst)):
    """Run vectorized backtest over stored historical bars.

    Runs as a detached subprocess — never blocks the web worker.
    Poll /api/historical/backtest/results for live progress and final results.
    """
    global _hist_backtest_proc
    if _proc_running(_hist_backtest_proc):
        return {"status": "already_running", "message": "Historical backtest already in progress."}

    cmd = [sys.executable, "-m", "historical", "--backtest", "--interval", interval]
    _hist_backtest_proc = subprocess.Popen(cmd, cwd=str(_PACKAGE_DIR))
    logger.info("[hist-backtest] Subprocess started (pid=%d)", _hist_backtest_proc.pid)
    return {"status": "started", "message": f"Historical backtest started ({interval})."}


@app.get("/api/historical/backtest/results")
async def historical_backtest_results():
    """Live progress and final results of the historical backtest (reads status file)."""
    state = _read_status(_BACKTEST_STATUS)
    state["proc_running"] = _proc_running(_hist_backtest_proc)
    if not state.get("proc_running") and state.get("running"):
        state["running"] = False
    return state


@app.get("/api/paper-trading/daily")
async def paper_daily_pnl():
    """Per-day P&L summary for last 14 days."""
    return {"daily": get_daily_pnl(days=14), "today": get_today_pnl()}


@app.get("/api/paper-trading/performance")
async def paper_performance():
    """Full P&L performance dashboard data."""
    loop = asyncio.get_running_loop()
    summary, today, daily, weekly, equity, ticker = await asyncio.gather(
        loop.run_in_executor(_pt_executor, pt_summary),
        loop.run_in_executor(_pt_executor, get_today_pnl),
        loop.run_in_executor(_pt_executor, get_daily_pnl, 30),
        loop.run_in_executor(_pt_executor, get_weekly_pnl),
        loop.run_in_executor(_pt_executor, get_equity_curve, 60),
        loop.run_in_executor(_pt_executor, get_ticker_pnl),
    )
    return {
        "summary":      summary,
        "today":        today,
        "daily":        daily,
        "weekly":       weekly,
        "equity_curve": equity,
        "ticker_pnl":   ticker,
    }


@app.get("/api/algo-performance")
async def algo_performance_endpoint():
    """Per-algorithm signal fire stats and closed-trade performance."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_pt_executor, get_algo_performance)
    return result


@app.get("/api/macro-calendar")
async def macro_calendar_endpoint():
    """Return current macro event status and upcoming events."""
    return {
        "current": check_macro_event(),
        "upcoming": get_upcoming_events(days=14),
    }


@app.get("/api/watchlist")
async def get_watchlist_endpoint():
    """Return user watchlist + base tickers."""
    return {
        "base":      NASDAQ_TICKERS,
        "watchlist": load_watchlist(),
    }


@app.post("/api/watchlist/add")
async def add_to_watchlist(
    ticker: str,
    _current: AuthenticatedUser = Depends(require_trader),
):
    """Add a ticker to the watchlist."""
    ticker = ticker.upper().strip()
    wl = load_watchlist()
    if ticker not in wl and ticker not in NASDAQ_TICKERS:
        wl.append(ticker)
        save_watchlist(wl)
    return {"watchlist": load_watchlist()}


@app.post("/api/watchlist/remove")
async def remove_from_watchlist(
    ticker: str,
    _current: AuthenticatedUser = Depends(require_trader),
):
    """Remove a ticker from the user watchlist (base tickers cannot be removed)."""
    ticker = ticker.upper().strip()
    wl = [t for t in load_watchlist() if t != ticker]
    save_watchlist(wl)
    return {"watchlist": load_watchlist()}


# ── Live backtest endpoints ───────────────────────────────────────────────────

@app.get("/api/backtest/stats")
async def backtest_stats(lookback_days: int = 30):
    """Full backtest performance report with attribution breakdown."""
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, get_full_report, lookback_days)
    return JSONResponse(content=_sanitize(data))


@app.get("/api/backtest/tracking")
async def backtest_tracking():
    """Currently open (TRACKING) signals being monitored."""
    loop = asyncio.get_running_loop()
    tracking = await loop.run_in_executor(None, get_tracking_signals)
    return {"tracking": tracking}


@app.get("/api/backtest/recent")
async def backtest_recent(limit: int = 50):
    """Recently resolved backtest signals."""
    loop   = asyncio.get_running_loop()
    recent = await loop.run_in_executor(None, get_recent_resolved, limit)
    for r in recent:
        r["outcome_color"] = (
            "#00ff88" if r["status"] == "WIN" else
            "#ef4444" if r["status"] == "LOSS" else
            "#64748b"
        )
    return {"recent": recent}


@app.get("/api/backtest/path/{signal_id}")
async def backtest_path(signal_id: str):
    """Price path bars for a specific signal (for replay/chart)."""
    return {"signal_id": signal_id, "path": get_price_path(signal_id)}


_LEARNING_PARAMS_CACHE_TTL_SECS = 2.0
_learning_params_cache: dict | None = None
_learning_params_cache_ts: float = 0.0


@app.get("/api/learning/params")
async def learning_params_status():
    """Per-family learned parameter values for dashboard display."""
    global _learning_params_cache, _learning_params_cache_ts

    if not _ALE_AVAILABLE:
        return {"available": False, "families": {}, "defaults": {}}

    now = time.monotonic()
    if (
        _learning_params_cache is not None
        and now - _learning_params_cache_ts < _LEARNING_PARAMS_CACHE_TTL_SECS
    ):
        return _learning_params_cache

    # Representative algo per family — used to look up current tuned params
    _FAMILY_REPRESENTATIVES = {
        "ORB":         "ORB5_BULL",
        "GAP_TREND":   "GAP_AND_GO_BULL",
        "GAP_FADE":    "GAP_FADE_BULL",
        "BREAKOUT":    "PDH_BREAKOUT_BULL",
        "FLAG":        "BULL_FLAG",
        "VWAP_SCALP":  "VWAP_TOUCH_SCALP_BULL",
        "LEVEL_SCALP": "LEVEL_REJECTION_SCALP_BULL",
        "RS_REGIME":   "SPY_BETA_CATCHUP_BULL",
    }
    defaults = {
        "rvol_gate": 1.5,
        "conf_gate": 55.0,
        "target_mult": 1.5,
        "stop_mult": 1.0,
        "entry_window_bars": 3,
    }
    def _safe_params(algo_name: str) -> dict:
        try:
            return _get_algo_params(algo_name)
        except Exception:
            return {}

    # These are tiny in-memory reads. Keep them on the request path instead of
    # queueing eight executor jobs behind scanner/model-training work.
    results = [_safe_params(algo) for algo in _FAMILY_REPRESENTATIVES.values()]
    families = {
        family: {k: (r.get(k, defaults[k]) if isinstance(r, dict) else defaults[k]) for k in defaults}
        for family, r in zip(_FAMILY_REPRESENTATIVES.keys(), results)
    }
    payload = {"available": True, "families": families, "defaults": defaults}
    _learning_params_cache = payload
    _learning_params_cache_ts = now
    return payload


@app.get("/api/learning-status")
async def learning_status():
    """Adaptive filter state — blocked contexts, dynamic threshold, win rate progress."""
    loop = asyncio.get_running_loop()
    # af_get_status and learning_engine.get_status are pure in-memory (no DB) — run on
    # default executor. get_observation_summary does a DB read — use _pt_executor.
    status, obs = await asyncio.gather(
        loop.run_in_executor(None, af_get_status),
        loop.run_in_executor(_pt_executor, get_observation_summary),
    )
    engine_status = learning_engine.get_status()

    # When the learner runs in a separate container it publishes its state to
    # Valkey key learner:status every 60s.  Prefer that over the stale local
    # in-memory state (which never updates in the web-api container).
    if not _LEARNER_ENABLED:
        try:
            from agent.valkey_client import _get_client as _vk_client
            _vk = _vk_client()
            if _vk:
                _raw = _vk.get("learner:status")
                if _raw:
                    _d = json.loads(_raw)
                    if time.time() - _d.get("ts", 0) < 300:
                        engine_status = _d.get("engine", engine_status)
                        status        = _d.get("adaptive_filter", status)
        except Exception:
            pass

    result = {
        **status,
        "engine":       engine_status,
        "observations": obs,
    }
    if _P2_AVAILABLE:
        try:
            p2_status = await loop.run_in_executor(None, lambda: _get_p2_engine().get_status())
            result["deployment_mode"]  = p2_status.get("deployment", {}).get("current_mode", "SHADOW")
            result["drift_alerts"]     = p2_status.get("drift", {}).get("material_drifts", [])
        except Exception as _p2e:
            logger.debug("learning-status phase2 error: %s", _p2e)
    return result


@app.get("/api/learning/phase2")
async def learning_phase2_status():
    """Phase 2 status: concept drift, staged deployment, walk-forward validation, transfer tier."""
    if not _P2_AVAILABLE:
        if not _LEARNER_ENABLED:
            try:
                from agent.valkey_client import _get_client as _vk_client
                _vk = _vk_client()
                if _vk:
                    _raw = _vk.get("learner:status")
                    if _raw:
                        _d = json.loads(_raw)
                        if time.time() - _d.get("ts", 0) < 300:
                            p2 = _d.get("phase2", {})
                            if p2:
                                return {"available": True, **p2}
            except Exception:
                pass
        return {"available": False}
    try:
        status = _get_p2_engine().get_status()
        return {"available": True, **status}
    except Exception as exc:
        logger.warning("phase2 status error: %s", exc)
        return {"available": True, "error": str(exc)}


@app.get("/api/learning/walk-forward-stats")
async def walk_forward_stats():
    """Walk-forward trainer last-run summary — per-family stats and param recommendations."""
    if not _WFT_AVAILABLE:
        return {"available": False}
    loop = asyncio.get_running_loop()
    try:
        status = await loop.run_in_executor(None, lambda: _get_wf_trainer().get_status())
        return {"available": True, **status}
    except Exception as exc:
        logger.warning("walk-forward-stats error: %s", exc)
        return {"available": True, "error": str(exc)}


@app.post("/api/adaptive-filter/reset")
async def reset_adaptive_filter(
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Reset the adaptive filter to factory defaults (threshold 60%, no blocked contexts)."""
    af_reset_filter()
    return {"ok": True, **af_get_status()}


@app.get("/api/learning-log")
async def learning_log_endpoint(limit: int = 100):
    """Last N learning engine log entries for the dashboard live feed."""
    log_entries   = get_learning_log(limit=limit)
    engine_status = learning_engine.get_status()

    if not _LEARNER_ENABLED:
        try:
            from agent.valkey_client import _get_client as _vk_client
            _vk = _vk_client()
            if _vk:
                _raw = _vk.get("learner:status")
                if _raw:
                    _d = json.loads(_raw)
                    if time.time() - _d.get("ts", 0) < 300:
                        log_entries   = _d.get("log", log_entries)[:limit]
                        engine_status = _d.get("engine", engine_status)
        except Exception:
            pass

    return {"log": log_entries, "engine": engine_status}


@app.get("/api/after-hours")
async def after_hours_endpoint():
    """
    Latest after-hours / pre-market snapshot for every scanned ticker.
    Sorted by absolute AH move descending — biggest movers first.
    """
    return {"snapshots": ah_get_all()}


# ── ThinkorSwim / Schwab Broker API ──────────────────────────────────────────

def _schwab_callback_url(request: Request) -> str:
    """
    Return the public callback URL registered in the Schwab Developer Portal.
    Reads SCHWAB_CALLBACK_URL from .env first (required when behind a reverse
    proxy such as IIS, where request.base_url would return http://localhost:8000/).
    Falls back to constructing from the incoming request for local dev.
    """
    explicit = os.getenv("SCHWAB_CALLBACK_URL", "").strip().rstrip("/")
    if explicit:
        return explicit + "/schwab/callback"
    return str(request.base_url).rstrip("/") + "/schwab/callback"


@app.get("/schwab/auth")
async def schwab_web_auth(request: Request):
    """Redirect browser to Schwab Market Data OAuth login.
    The Accounts+Trading app is no longer required — Market Data only."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/schwab/auth/md")


@app.get("/schwab/callback")
async def schwab_web_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    """Schwab OAuth callback — handles Market Data app token exchange.
    Register https://scalpingstocksai.com/schwab/callback in the Schwab Developer Portal."""
    from fastapi.responses import HTMLResponse
    if error or not code:
        html = f"""<html><body style="font-family:sans-serif;padding:40px">
        <h2 style="color:#e53e3e">Schwab Auth Failed</h2>
        <p>{_html.escape(error) or 'No code received.'}</p>
        <p><a href="/schwab/auth/md">Try again</a></p></body></html>"""
        return HTMLResponse(html, status_code=400)

    redirect_uri = _schwab_callback_url(request)
    success, reason = exchange_md_auth_code(code, state, redirect_uri)
    if success:
        # Start the MD poller and wire tick broadcasts if not already running
        try:
            from agent.broker.schwab_streamer import is_streamer_ready
            if not is_streamer_ready():
                from config import NASDAQ_TICKERS
                start_md_poller(list(NASDAQ_TICKERS), interval=1.0, parallel_batches=2)
            _ensure_tick_broadcast_registered()
        except Exception:
            pass
        html = """<html><body style="font-family:sans-serif;padding:40px;background:#f0fff4">
        <h2 style="color:#276749">&#10003; Schwab Market Data Connected!</h2>
        <p>Tokens saved. REST quotes, IV, movers and price history are now live.</p>
        <p>Real-time 1-second quote poller started for all NASDAQ tickers.</p>
        <p><a href="/">&#8592; Back to Dashboard</a></p></body></html>"""
        return HTMLResponse(html)
    else:
        html = f"""<html><body style="font-family:sans-serif;padding:40px">
        <h2 style="color:#e53e3e">Token Exchange Failed</h2>
        <p><b>Reason:</b> {_html.escape(reason) if reason else 'See server logs.'}</p>
        <p><b>redirect_uri used:</b> <code>{_html.escape(redirect_uri)}</code></p>
        <p>If the redirect_uri above does not match what is registered in the Schwab
        Developer Portal, set <code>SCHWAB_CALLBACK_URL=https://scalpingstocksai.com</code>
        in your <code>.env</code> file and restart.</p>
        <p><a href="/schwab/auth/md">Try again</a></p></body></html>"""
        return HTMLResponse(html, status_code=500)


@app.get("/schwab/auth/at")
async def schwab_at_web_auth(request: Request):
    """Redirect browser to Schwab Accounts+Trading OAuth login (for WebSocket streamer)."""
    from fastapi.responses import RedirectResponse, HTMLResponse
    from agent.broker.schwab_auth import _trader, build_auth_url
    if not _trader.is_configured():
        return HTMLResponse(
            "<html><body style='font-family:sans-serif;padding:40px'>"
            "<h2>SCHWAB_CLIENT_ID not set in .env</h2>"
            "<p>Add your Accounts+Trading app credentials and restart:</p>"
            "<pre>SCHWAB_CLIENT_ID=&lt;your-app-key&gt;\n"
            "SCHWAB_CLIENT_SECRET=&lt;your-secret&gt;</pre>"
            "<p>Register <code>https://scalpingstocksai.com/schwab/callback/at</code> "
            "as the callback URL in the Schwab Developer Portal.</p>"
            "</body></html>",
            status_code=400,
        )
    redirect_uri = _schwab_callback_url(request).replace("/schwab/callback", "/schwab/callback/at")
    return RedirectResponse(url=build_auth_url(redirect_uri))


@app.get("/schwab/callback/at")
async def schwab_at_web_callback(
    request: Request, code: str = "", state: str = "", error: str = ""
):
    """OAuth callback for the Schwab Accounts+Trading app (WebSocket streamer)."""
    from fastapi.responses import HTMLResponse
    from agent.broker.schwab_auth import exchange_auth_code
    if error or not code:
        html = (f"<html><body style='font-family:sans-serif;padding:40px'>"
                f"<h2 style='color:#e53e3e'>Schwab A+T Auth Failed</h2>"
                f"<p>{_html.escape(error) or 'No code received.'}</p>"
                f"<p><a href='/schwab/auth/at'>Try again</a></p></body></html>")
        return HTMLResponse(html, status_code=400)

    redirect_uri = _schwab_callback_url(request).replace("/schwab/callback", "/schwab/callback/at")
    success, reason = exchange_auth_code(code, state, redirect_uri)
    if success:
        try:
            from config import NASDAQ_TICKERS as _nq_t
            start_streamer(list(_nq_t))
            _ensure_tick_broadcast_registered()
        except Exception:
            pass
        # Notify the scanner container to hot-reload tokens via Valkey pub/sub
        try:
            from agent.valkey_client import _get_client as _vk_c
            _vk = _vk_c()
            if _vk:
                _vk.publish("schwab:tokens_refreshed",
                            json.dumps({"ts": time.time(), "app": "at"}))
        except Exception:
            pass
        html = ("<html><body style='font-family:sans-serif;padding:40px;background:#f0fff4'>"
                "<h2 style='color:#276749'>&#10003; Schwab Accounts+Trading Connected!</h2>"
                "<p>Tokens saved. WebSocket Level 1 streamer starting now.</p>"
                "<p>You will see real-time bid/ask/last updates and 1-min candles within seconds.</p>"
                "<p><a href='/'>&#8592; Back to Dashboard</a></p></body></html>")
        return HTMLResponse(html)
    else:
        html = (f"<html><body style='font-family:sans-serif;padding:40px'>"
                f"<h2 style='color:#e53e3e'>Token Exchange Failed</h2>"
                f"<p><b>Reason:</b> {_html.escape(reason) if reason else 'See server logs.'}</p>"
                f"<p><a href='/schwab/auth/at'>Try again</a></p></body></html>")
        return HTMLResponse(html, status_code=500)


@app.get("/schwab/auth/md")
async def schwab_md_web_auth(request: Request):
    """Redirect browser to Schwab Market Data OAuth login (for REST quotes/IV/movers)."""
    from fastapi.responses import RedirectResponse, HTMLResponse
    from agent.broker.schwab_auth import _market_data
    if not _market_data.is_configured():
        return HTMLResponse(
            "<h2>SCHWAB_MD_CLIENT_ID not set in .env</h2>"
            "<p>Add the Market Data app credentials and restart.</p>",
            status_code=400,
        )
    redirect_uri = _schwab_callback_url(request)
    return RedirectResponse(url=build_md_auth_url(redirect_uri))


@app.get("/schwab/callback/md")
async def schwab_md_web_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    """Legacy callback path — redirects to the active /schwab/callback handler."""
    from fastapi.responses import RedirectResponse
    # Forward all query params to the active callback route
    params = str(request.url.query)
    target = f"/schwab/callback?{params}" if params else "/schwab/callback"
    return RedirectResponse(url=target)


@app.get("/api/broker/status")
async def broker_status(_user: AuthenticatedUser = Depends(require_viewer)):
    """Connection status for both Schwab apps, token TTLs, account info."""
    from config import SCHWAB_ENABLED
    try:
        ts    = get_token_status()
        ts_md = get_md_token_status()
        acct  = {}
        if SCHWAB_ENABLED and ts.get("connected"):
            try:
                acct = get_account_summary()
            except Exception:
                pass
        daily = get_daily_status()
        # Merge paper trading today P&L so the broker panel shows real activity
        # even when Schwab live trading is off / in paper mode.
        try:
            pt_today = get_today_pnl()
            daily = {**daily, "pnl": pt_today.get("total_pnl_dollar", daily.get("pnl", 0.0))}
        except Exception:
            pass
        streamer = get_streamer_status()
        ws_st = streamer.get("ws_streamer", {})
        md_st = streamer.get("md_poller", {})
        return {
            "connected":          ts_md.get("connected", False) or ts.get("connected", False),
            "schwab_enabled":     SCHWAB_ENABLED,
            "market_data_app":    ts_md,
            "trader_app":         ts,
            "ws_streamer": {
                "running":     ws_st.get("running", False),
                "connected":   ws_st.get("connected", False),
                "live_quotes": ws_st.get("live_quotes", 0),
                "auth_url":    "/schwab/auth/at",
                "error":       ws_st.get("error"),
            },
            "md_poller": {
                "running":     md_st.get("running", False),
                "cycle":       md_st.get("cycle", 0),
                "live_quotes": md_st.get("live_quotes", 0),
                "auth_url":    "/schwab/auth/md",
                "error":       md_st.get("error"),
            },
            "account":    acct,
            "daily":      daily,
            "auto_trade": _tos_auto_trade,
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


@app.post("/api/broker/auth")
async def broker_auth(
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Initiate Schwab OAuth flow — redirect browser to /schwab/auth instead."""
    from config import SCHWAB_ENABLED
    if not SCHWAB_ENABLED:
        return {"success": False, "error": "Schwab is disabled (SCHWAB_ENABLED=false in .env)"}
    return {"success": False, "error": "Use the browser flow: visit /schwab/auth to authorise"}


@app.get("/api/broker/positions")
async def broker_positions(
    _current: AuthenticatedUser = Depends(require_trader),
):
    """Current open positions in the ThinkorSwim paper account."""
    from config import SCHWAB_ENABLED
    if not SCHWAB_ENABLED:
        return {"positions": [], "schwab_enabled": False}
    try:
        loop = asyncio.get_running_loop()
        positions = await loop.run_in_executor(None, get_positions)
        return {"positions": positions}
    except Exception as e:
        return {"positions": [], "error": str(e)}


@app.get("/api/broker/orders")
async def broker_orders(
    _current: AuthenticatedUser = Depends(require_trader),
):
    """Recent working orders."""
    from config import SCHWAB_ENABLED
    if not SCHWAB_ENABLED:
        return {"orders": [], "schwab_enabled": False}
    try:
        loop = asyncio.get_running_loop()
        orders = await loop.run_in_executor(None, get_orders)
        return {"orders": orders}
    except Exception as e:
        return {"orders": [], "error": str(e)}


@app.post("/api/broker/auto-trade/{enabled}")
async def broker_auto_trade(
    enabled: str,
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Toggle fully-automatic order placement (true/false)."""
    global _tos_auto_trade
    _tos_auto_trade = enabled.lower() == "true"
    return {"auto_trade": _tos_auto_trade}


@app.post("/api/broker/order")
async def broker_manual_order(
    body: dict,
    _current: AuthenticatedUser = Depends(require_trader),
):
    """
    Manually trigger a bracket order for a ticker already in the signal list.
    Body: { "ticker": "NVDA" }
    """
    ticker = body.get("ticker", "").upper()
    sig = next((s for s in scanner.signals if s.ticker == ticker), None)
    if sig is None and not _SCANNER_ENABLED:
        # Scanner runs in another container — look up from Valkey snapshot
        try:
            from agent.signal_snapshot import read_latest as _snap_read2
            import types as _types
            snap2 = _snap_read2()
            if snap2:
                match = next((s for s in snap2.get("signals", []) if s.get("ticker") == ticker), None)
                if match:
                    sig = _types.SimpleNamespace(**match)
        except Exception:
            pass
    if not sig:
        return {"placed": False, "reason": f"{ticker} not in current scan"}
    result = maybe_place_tos_order(sig)
    return result


@app.get("/api/market/streamer")
async def streamer_status_endpoint():
    """WebSocket streamer health: connected, live quote count, futures bias."""
    try:
        status = get_streamer_status()
        # Include Schwab auth state so we can diagnose why streamer isn't running
        ts = get_token_status()
        status["schwab_connected"]       = ts.get("connected", False)
        status["access_token_ttl_s"]     = ts.get("access_token_ttl_s", 0)
        status["refresh_token_ttl_s"]    = ts.get("refresh_token_ttl_s", 0)
        return status
    except Exception as e:
        return {"connected": False, "error": str(e)}


@app.post("/api/market/streamer/start")
async def streamer_start_endpoint(
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Manually (re)start the Schwab WebSocket streamer."""
    import asyncio
    ts = get_token_status()
    if not ts.get("connected"):
        return {"started": False, "reason": "Schwab not authenticated — visit /schwab/auth first"}
    try:
        loop = asyncio.get_running_loop()
        from config import NASDAQ_TICKERS
        await loop.run_in_executor(None, lambda: start_streamer(list(NASDAQ_TICKERS)))
        return {"started": True, "tickers": len(NASDAQ_TICKERS)}
    except Exception as e:
        return {"started": False, "reason": str(e)}


@app.get("/api/market/movers")
async def market_movers(index: str = "$COMPX", sort: str = "PERCENT_CHANGE_UP", freq: int = 0):
    """Top movers for an index via Schwab. index: $COMPX | $SPX | $DJI"""
    try:
        from agent.broker.schwab_market_data import fetch_movers
        loop = asyncio.get_running_loop()
        movers = await loop.run_in_executor(None, lambda: fetch_movers(index, sort, freq))
        return {"movers": movers, "index": index, "sort": sort}
    except Exception as e:
        return {"movers": [], "error": str(e)}


@app.get("/api/universe")
async def universe_status():
    """Ticker universe status: total tracked, active this cycle, tier breakdown."""
    try:
        from agent.ticker_universe import get_universe_manager, TIER1, TIER2, TIER3, FULL_UNIVERSE
        mgr = get_universe_manager()
        active = mgr.get_active_tickers()
        active_set = set(active)
        return {
            "universe_total":  len(FULL_UNIVERSE),
            "active_this_cycle": len(active),
            "tier1_count":     len(TIER1),
            "tier2_count":     len(TIER2),
            "tier3_count":     len(TIER3),
            "tier1_in_active": sum(1 for t in TIER1 if t in active_set),
            "tier2_in_active": sum(1 for t in TIER2 if t in active_set),
            "tier3_in_active": sum(1 for t in TIER3 if t in active_set),
            "active_tickers":  active,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/market/hours")
async def market_hours_endpoint(market: str = "equity"):
    """Current market session status via Schwab."""
    try:
        from agent.broker.schwab_market_data import fetch_market_hours
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fetch_market_hours(market))
    except Exception as e:
        return {"is_open": None, "error": str(e)}


@app.get("/api/market/iv/{ticker}")
async def ticker_iv(ticker: str):
    """Implied volatility for a single ticker via Schwab option chains."""
    try:
        from agent.broker.schwab_market_data import fetch_iv
        loop = asyncio.get_running_loop()
        iv = await loop.run_in_executor(None, lambda: fetch_iv(ticker.upper()))
        return {"ticker": ticker.upper(), "iv": iv}
    except Exception as e:
        return {"ticker": ticker, "iv": None, "error": str(e)}


# ── SSE signal stream ────────────────────────────────────────────────────────

if _SSE_AVAILABLE:
    @app.get("/stream/signals")
    async def stream_signals(request: Request):
        """Server-Sent Events stream — pushes signal updates every 5s."""
        async def event_generator():
            while True:
                if await request.is_disconnected():
                    break
                try:
                    signals = scanner.get_last_signals()  # existing method
                    data = json.dumps([s.to_dict() for s in signals[:50]])  # top 50
                    yield {"event": "signals", "data": data}
                except Exception:
                    pass
                await asyncio.sleep(5)
        return _EventSourceResponse(event_generator())
else:
    @app.get("/stream/signals")
    async def stream_signals_fallback(request: Request):
        """
        Fallback polling endpoint (sse-starlette not installed).
        Returns the latest 50 signals as JSON.  Poll every 5 s from the client.
        """
        try:
            signals = scanner.get_last_signals()
            return JSONResponse({"signals": [s.to_dict() for s in signals[:50]]})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)


# ── Pipeline metrics ─────────────────────────────────────────────────────────

@app.get("/api/pipeline-metrics")
async def pipeline_metrics(_user: AuthenticatedUser = Depends(require_viewer)):
    """Return current pipeline throughput and worker metrics."""
    try:
        from agent.pipeline import get_pipeline
        return get_pipeline().get_metrics()
    except Exception as e:
        return {"error": str(e)}


# ── Signal blend weights ─────────────────────────────────────────────────────

@app.get("/api/blend-weights")
async def blend_weights(_user: AuthenticatedUser = Depends(require_viewer)):
    """Return current signal blender weight stats."""
    try:
        from agent.signal_blender import get_blender
        return get_blender().get_stats()
    except Exception as e:
        return {"error": str(e)}


# ── Backtester ───────────────────────────────────────────────────────────────

@app.get("/api/backtest/results")
async def backtest_results(_user: AuthenticatedUser = Depends(require_viewer)):
    """Return the latest backtester report and run status."""
    try:
        from agent.backtester import get_backtester
        bt = get_backtester()
        report = bt.get_report()
        status = bt.get_status()
        return {"status": status, "report": report.__dict__ if report else None}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/backtest/run")
async def run_backtest(background_tasks: BackgroundTasks,
                       _user: AuthenticatedUser = Depends(require_analyst)):
    """Trigger a fresh backtester run in the background."""
    try:
        from agent.backtester import get_backtester
        background_tasks.add_task(get_backtester().run_sync)
        return {"status": "started"}
    except Exception as e:
        return {"error": str(e)}


# ── Telegram notifier ────────────────────────────────────────────────────────

@app.get("/api/notify/config")
async def notify_config():
    """Return current Telegram notifier configuration (token is never returned)."""
    return _notify_cfg()


@app.post("/api/notify/config")
async def notify_set_config(
    body: dict,
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Save Telegram bot token + chat ID. Persisted to disk across restarts."""
    token   = str(body.get("token",          "")).strip()
    chat_id = str(body.get("chat_id",        "")).strip()
    min_conf = float(body.get("min_confidence", 75.0))
    _notify_configure(token=token, chat_id=chat_id, min_confidence=min_conf)
    return {"ok": True, **_notify_cfg()}


@app.post("/api/notify/test")
async def notify_test(
    _current: AuthenticatedUser = Depends(require_admin),
):
    """Send a test Telegram message to verify the config is working."""
    if not _notify_cfg().get("configured"):
        return {"ok": False, "error": "Not configured — set token and chat_id first"}
    ok, err = _send_telegram(
        "✅ <b>NASDAQ Agent</b> — Telegram notifications are working!\n"
        "You will receive alerts for high-confidence signals."
    )
    return {"ok": ok, "error": err if not ok else None}


# ── Ticker clusters ──────────────────────────────────────────────────────────

@app.get("/api/clusters")
async def ticker_clusters():
    """Return the ML cluster membership lists and per-ticker assignment map."""
    return {
        "A": CLUSTER_A_TICKERS,
        "B": CLUSTER_B_TICKERS,
        "C": CLUSTER_C_TICKERS,
        "assignments": TICKER_CLUSTER,
    }


# ── WebSocket ─────────────────────────────────────────────────────────────────

_PING_INTERVAL       = 20   # server sends a keepalive ping every N seconds
_PING_TIMEOUT        = 10   # if we can't write the ping within N seconds → presumed busy
_PING_RETRY_INTERVAL =  5   # after a timeout (event loop busy), retry after this many seconds


async def _safe_ws_close(ws: WebSocket, *, code: int, reason: str = "") -> None:
    """Best-effort WebSocket close that never escapes protocol-race errors."""
    try:
        await ws.close(code=code, reason=reason)
    except Exception as exc:
        logger.debug("WebSocket close ignored: %s", exc)


async def _ws_keepalive(ws: WebSocket) -> None:
    """
    Background task: sends a server-side ping every _PING_INTERVAL seconds.
    This resets the client's 45-second watchdog and keeps NAT/proxy sessions alive.

    Runs as a sibling asyncio.Task alongside the receive loop — completely
    separate from client messages, so there is NEVER a ping-pong feedback loop.
    When the send fails (dead socket) this task exits quietly; the receive loop
    will also error on the next read and close the connection.

    TimeoutError is caught separately and retried — the event loop can be
    briefly saturated during a scanner cycle (477 ticker_update broadcasts)
    which delays the ping write beyond _PING_TIMEOUT.  Retrying keeps the
    keepalive task alive so the next ping succeeds once the burst clears.
    """
    while True:
        try:
            await asyncio.sleep(_PING_INTERVAL)
            await asyncio.wait_for(
                ws.send_json({"type": "ping"}),
                timeout=float(_PING_TIMEOUT),
            )
        except asyncio.TimeoutError:
            # Event loop was briefly saturated (e.g. scanner broadcast storm).
            # Sleep a SHORT interval so the next ping attempt arrives well within
            # the browser's 45-second watchdog window rather than after the full
            # 20-second PING_INTERVAL (which could miss the window).
            await asyncio.sleep(_PING_RETRY_INTERVAL)
        except Exception:
            break      # socket gone — receive loop handles cleanup


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    # Accept the connection, then authenticate via either:
    #   Fast path  — valid "token" query-parameter (zero-latency, already in URL)
    #   Slow path  — {"type":"auth","token":"..."} first message within 10 seconds
    # Both paths call the same decode_token / is_blacklisted checks.
    await ws.accept()

    def _verify_token(token: str) -> bool:
        try:
            from auth.utils import decode_token, is_blacklisted
            _p = decode_token(token)
            return _p.get("type") == "access" and not is_blacklisted(_p.get("jti", ""))
        except Exception:
            return False

    _authed = False

    # Fast path: token in query param (client already attached it to the URL)
    _qtoken = ws.query_params.get("token", "")
    if _qtoken:
        _authed = _verify_token(_qtoken)

    # Slow path: wait for first-message auth (covers clients that omit the query param)
    if not _authed:
        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
            msg = json.loads(raw)
            if msg.get("type") == "auth":
                _authed = _verify_token(msg.get("token", ""))
        except (asyncio.TimeoutError, Exception):
            pass

    if not _authed:
        await _safe_ws_close(ws, code=4001)
        return

    await manager.connect(ws)
    logger.info(f"WebSocket client connected. Total: {len(manager.active)}")
    keepalive = asyncio.create_task(_ws_keepalive(ws))
    try:
        regime  = get_regime()
        session = get_session_info()

        # Choose best available signals: live > loaded persisted snapshot
        live_sigs, _last_scan, from_cache = _current_signal_snapshot()

        if live_sigs:
            # Full update so the tab is immediately usable
            await ws.send_text(_dumps({
                "type":          "update",
                "signals":       live_sigs,
                "regime":        regime.to_dict(),
                "session":       session,
                "from_cache":    from_cache,
                "scanned_count": len(live_sigs),
            }))
        else:
            # No data yet (cold start) — send a status frame so the loading
            # screen can show regime/session info rather than spinning blindly.
            await ws.send_text(_dumps({
                "type":       "scan_status",
                "scanning":   True,
                "regime":     regime.to_dict(),
                "session":    session,
                "n_total":    _get_universe_total(),
            }))

        # Drain incoming client messages.
        while True:
            await ws.receive_text()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WebSocket loop error: {e}")
    finally:
        keepalive.cancel()
        manager.disconnect(ws)
        logger.info(f"WebSocket client disconnected. Total: {len(manager.active)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
