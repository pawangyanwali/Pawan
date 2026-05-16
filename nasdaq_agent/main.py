"""
FastAPI entry point.
Serves the static web dashboard and a WebSocket endpoint that pushes
real-time stock signals to all connected clients.
"""

import asyncio
import json
import logging
import os
import numpy as np
from contextlib import asynccontextmanager
from typing import Set

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
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

from agent.scanner import scanner, StockSignal
from agent.market_hours import get_session_info
from agent.market_regime import get_regime
from agent.signal_tracker import get_stats, get_recent_signals, get_observation_summary
from agent.position_sizing import calculate as calc_position
from agent.paper_trading import get_summary as pt_summary, get_open_trades, get_closed_trades, get_daily_pnl, get_today_pnl, get_equity_curve, get_weekly_pnl, get_ticker_pnl
from agent.macro_calendar import check_macro_event, get_upcoming_events
from agent.live_backtest import get_performance_stats, get_tracking_signals, get_recent_resolved, get_price_path
from agent.backtest_reporter import get_broadcast_summary, get_full_report
from agent.adaptive_filter import get_status as af_get_status, reset_filter as af_reset_filter
from agent.after_hours_monitor import get_all_biases as ah_get_all
from agent.learning_engine import learning_engine, get_learning_log
import agent.weekend_learner as weekend_learner
from agent.broker.schwab_auth import (
    load_stored_tokens, load_stored_md_tokens,
    get_token_status, get_md_token_status,
    build_auth_url, exchange_auth_code,
    build_md_auth_url, exchange_md_auth_code,
)
from agent.broker.schwab_streamer import start_streamer, get_streamer_status
from agent.broker.schwab_client import get_positions, get_account_summary, get_orders
from agent.broker.order_bridge import maybe_place_tos_order, get_daily_status
from config import (
    DEFAULT_ACCOUNT_SIZE, DEFAULT_RISK_PCT, MAX_POSITION_PCT,
    load_watchlist, save_watchlist, NASDAQ_TICKERS,
    CLUSTER_A_TICKERS, CLUSTER_B_TICKERS, CLUSTER_C_TICKERS, TICKER_CLUSTER,
)


import math


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
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.active.discard(ws)

    async def broadcast(self, message: str) -> None:
        dead = set()
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active.discard(ws)


manager = ConnectionManager()

# Captured at startup so the scanner background thread can schedule broadcasts
_event_loop: asyncio.AbstractEventLoop | None = None


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

    payload = _dumps({
        "type":        "update",
        "signals":     [s.to_dict() for s in signals],
        "regime":      regime.to_dict(),
        "session":     session,
        "alerts":      alerts,
        "macro":       macro,
        "backtest":    bt_summary,
        "learning":    learn_compact,
        "open_trades": open_trades,
        "pt_stats":    pt_stats,
        "breadth":     breadth,
    })
    asyncio.run_coroutine_threadsafe(manager.broadcast(payload), _event_loop)


def _on_ticker(sig: StockSignal, n_done: int, n_total: int) -> None:
    """Per-ticker callback — streams each result as it completes so the dashboard
    fills progressively instead of waiting for the full scan batch."""
    if _event_loop is None:
        return
    payload = _dumps({
        "type":    "ticker_update",
        "signal":  sig.to_dict(),
        "n_done":  n_done,
        "n_total": n_total,
    })
    asyncio.run_coroutine_threadsafe(manager.broadcast(payload), _event_loop)


# ── ThinkorSwim auto-trade toggle ────────────────────────────────────────────
_tos_auto_trade: bool = os.getenv("SCHWAB_AUTO_TRADE", "false").lower() == "true"

# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _event_loop
    _event_loop = asyncio.get_running_loop()
    scanner.register_callback(_on_signals)
    scanner.register_per_ticker_callback(_on_ticker)
    scanner.start_background()
    learning_engine.start()

    # Weekend learner — give it a broadcast handle, then auto-start if it's a weekend
    def _wl_broadcast(payload: dict) -> None:
        if _event_loop:
            asyncio.run_coroutine_threadsafe(
                manager.broadcast(json.dumps(payload)), _event_loop
            )
    weekend_learner.register_broadcast(_wl_broadcast)
    weekend_learner.maybe_start()
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
    # Schwab integration — only active when SCHWAB_ENABLED=true in .env
    from config import SCHWAB_ENABLED
    if SCHWAB_ENABLED:
        try:
            if os.getenv("SCHWAB_CLIENT_ID"):
                ok = load_stored_tokens()
                if ok:
                    logging.getLogger(__name__).info("Schwab Trader app connected.")
                    from config import NASDAQ_TICKERS
                    start_streamer(list(NASDAQ_TICKERS))
        except Exception as _be:
            logging.getLogger(__name__).warning(f"Schwab Trader token load skipped: {_be}")
        try:
            if os.getenv("SCHWAB_MD_CLIENT_ID"):
                ok_md = load_stored_md_tokens()
                if ok_md:
                    logging.getLogger(__name__).info("Schwab Market Data app connected.")
        except Exception as _be:
            logging.getLogger(__name__).warning(f"Schwab MD token load skipped: {_be}")
    else:
        logging.getLogger(__name__).info("Schwab disabled (SCHWAB_ENABLED not set) — running on Twelve Data only.")
    yield
    scanner.stop()
    learning_engine.stop()


app = FastAPI(title="NASDAQ Scalping Agent", lifespan=lifespan)

# Allow IIS (port 80) and any other origin to call the FastAPI backend (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Static files (dashboard)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "web", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── HTTP routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/signals")
async def get_signals():
    """REST endpoint: returns the latest cached scan results."""
    return {
        "last_scan": scanner.last_scan,
        "count": len(scanner.signals),
        "signals": [s.to_dict() for s in scanner.signals],
    }


@app.get("/api/health")
async def health():
    from agent.data_fetcher import get_credit_usage
    return {
        "status": "ok",
        "is_running": scanner.is_running,
        "last_scan": scanner.last_scan,
        "tickers_tracked": len(scanner.signals),
        "ws_clients": len(manager.active),
        "api_credits": get_credit_usage(),
    }


@app.get("/api/credit-usage")
async def credit_usage():
    """Rolling 60-second Twelve Data credit consumption."""
    from agent.data_fetcher import get_credit_usage, CREDIT_LIMIT
    usage = get_credit_usage()
    return {**usage, "plan_limit": 377, "safe_limit": CREDIT_LIMIT}


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
async def paper_trading_endpoint():
    """Return paper trading summary, open and recent closed trades."""
    from datetime import date
    closed = get_closed_trades(limit=200)
    today_str = date.today().isoformat()   # "2026-05-15"

    # Separate today vs all-time so the two eras (100-share vs risk-based) don't mix
    today_trades   = [t for t in closed if (t.get("closed_at") or "")[:10] == today_str]
    all_trades     = closed

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

    summary = pt_summary()
    # Use TODAY stats for the primary display (consistent system, no legacy 100-share noise)
    summary.update({
        "closed":           today_stats["closed"],
        "wins":             today_stats["wins"],
        "losses":           today_stats["losses"],
        "win_rate":         today_stats["win_rate"],
        "avg_pnl":          today_stats["avg_pnl"],
        "total_pnl":        today_stats["avg_pnl"],
        "total_dollar_pnl": today_stats["total_dollar_pnl"],
        "all_time_dollar":  all_stats["total_dollar_pnl"],
        "all_time_closed":  all_stats["closed"],
    })
    return {
        "summary":       summary,
        "open_trades":   get_open_trades(),
        "closed_trades": closed[:30],
        "_debug_pnl":    {
            "n_trades":      len(all_trades),
            "total_dollar":  all_stats["total_dollar_pnl"],
            "today_dollar":  today_stats["total_dollar_pnl"],
            "per_trade":     [(t["ticker"], t.get("pnl_dollar") or 0, (t.get("closed_at") or "")[:10]) for t in all_trades],
        },
    }


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
async def trigger_premarket_scan(background_tasks: BackgroundTasks):
    """Manually trigger a pre-market gapper scan."""
    try:
        from agent.premarket_scanner import run_premarket_scan_background
        run_premarket_scan_background()
        return {"status": "started"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ml-status")
async def ml_status():
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
async def wl_start():
    """Manually kick off the weekend learning pipeline (admin override)."""
    started = weekend_learner.start()
    return {
        "status":  "started" if started else "already_running",
        "message": ("Weekend learning started in background."
                    if started else "Weekend learner is already running."),
    }


@app.post("/api/weekend-learning/stop")
async def wl_stop():
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


@app.post("/api/ml-retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Manually trigger a full ML retrain cycle (XGBoost + SwingML + Deep BiLSTM).
    Runs in background — check /api/ml-status for progress.
    """
    from agent.ml_model import retrain_all, _is_retraining
    from config import NASDAQ_TICKERS

    if _is_retraining:
        return {"status": "already_running", "message": "Retrain already in progress."}

    def _run():
        try:
            retrain_all(NASDAQ_TICKERS)
        except Exception as e:
            logger.warning(f"[manual retrain] failed: {e}")

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Retrain started in background. Watch /api/ml-status for progress."}


@app.post("/api/deep-model/train")
async def trigger_deep_train(background_tasks: BackgroundTasks):
    """
    Manually trigger Deep BiLSTM training only (faster than full retrain).
    Uses cached 15-min data when available.
    """
    from agent.deep_model import is_training_active, retrain_deep_all
    from agent.data_fetcher import fetch_batch_interval
    from config import NASDAQ_TICKERS

    if is_training_active():
        return {"status": "already_running", "message": "Deep model training already in progress."}

    def _run():
        try:
            logger.info("[manual deep train] Fetching 15-min data…")
            hist_15m = fetch_batch_interval(NASDAQ_TICKERS, "15min", 5000, ttl=3600)
            logger.info(f"[manual deep train] Got {len(hist_15m)} tickers — starting training…")
            retrain_deep_all(hist_15m)
        except Exception as e:
            logger.warning(f"[manual deep train] failed: {e}")

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Deep BiLSTM training started. Check /api/ml-status for epoch progress."}


@app.get("/api/paper-trading/daily")
async def paper_daily_pnl():
    """Per-day P&L summary for last 14 days."""
    return {"daily": get_daily_pnl(days=14), "today": get_today_pnl()}


@app.get("/api/paper-trading/performance")
async def paper_performance():
    """Full P&L performance dashboard data."""
    return {
        "summary":       pt_summary(),
        "today":         get_today_pnl(),
        "daily":         get_daily_pnl(days=30),
        "weekly":        get_weekly_pnl(),
        "equity_curve":  get_equity_curve(days=60),
        "ticker_pnl":    get_ticker_pnl(),
    }


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
async def add_to_watchlist(ticker: str):
    """Add a ticker to the watchlist."""
    ticker = ticker.upper().strip()
    wl = load_watchlist()
    if ticker not in wl and ticker not in NASDAQ_TICKERS:
        wl.append(ticker)
        save_watchlist(wl)
    return {"watchlist": load_watchlist()}


@app.post("/api/watchlist/remove")
async def remove_from_watchlist(ticker: str):
    """Remove a ticker from the user watchlist (base tickers cannot be removed)."""
    ticker = ticker.upper().strip()
    wl = [t for t in load_watchlist() if t != ticker]
    save_watchlist(wl)
    return {"watchlist": load_watchlist()}


# ── Live backtest endpoints ───────────────────────────────────────────────────

@app.get("/api/backtest/stats")
async def backtest_stats(lookback_days: int = 30):
    """Full backtest performance report with attribution breakdown."""
    return get_full_report(lookback_days=lookback_days)


@app.get("/api/backtest/tracking")
async def backtest_tracking():
    """Currently open (TRACKING) signals being monitored."""
    return {"tracking": get_tracking_signals()}


@app.get("/api/backtest/recent")
async def backtest_recent(limit: int = 50):
    """Recently resolved backtest signals."""
    recent = get_recent_resolved(limit=limit)
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


@app.get("/api/learning-status")
async def learning_status():
    """Adaptive filter state — blocked contexts, dynamic threshold, win rate progress."""
    return {
        **af_get_status(),
        "engine":       learning_engine.get_status(),
        "observations": get_observation_summary(),
    }


@app.post("/api/adaptive-filter/reset")
async def reset_adaptive_filter():
    """Reset the adaptive filter to factory defaults (threshold 60%, no blocked contexts)."""
    af_reset_filter()
    return {"ok": True, **af_get_status()}


@app.get("/api/learning-log")
async def learning_log_endpoint(limit: int = 100):
    """Last N learning engine log entries for the dashboard live feed."""
    return {
        "log":    get_learning_log(limit=limit),
        "engine": learning_engine.get_status(),
    }


@app.get("/api/after-hours")
async def after_hours_endpoint():
    """
    Latest after-hours / pre-market snapshot for every scanned ticker.
    Sorted by absolute AH move descending — biggest movers first.
    """
    return {"snapshots": ah_get_all()}


# ── ThinkorSwim / Schwab Broker API ──────────────────────────────────────────

@app.get("/schwab/auth")
async def schwab_web_auth(request: Request):
    """Redirect browser to Schwab Accounts+Trading OAuth login (for streamer + trading)."""
    from fastapi.responses import RedirectResponse
    redirect_uri = str(request.base_url).rstrip("/") + "/schwab/callback"
    return RedirectResponse(url=build_auth_url(redirect_uri))


@app.get("/schwab/callback")
async def schwab_web_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    """Schwab callback for Accounts+Trading app. Exchange code → start streamer."""
    from fastapi.responses import HTMLResponse
    if error or not code:
        html = f"""<html><body style="font-family:sans-serif;padding:40px">
        <h2 style="color:#e53e3e">Schwab Auth Failed</h2>
        <p>{error or 'No code received.'}</p>
        <p><a href="/schwab/auth">Try again</a></p></body></html>"""
        return HTMLResponse(html, status_code=400)

    redirect_uri = str(request.base_url).rstrip("/") + "/schwab/callback"
    success = exchange_auth_code(code, state, redirect_uri)
    if success:
        try:
            from config import NASDAQ_TICKERS
            start_streamer(list(NASDAQ_TICKERS))
        except Exception:
            pass
        html = """<html><body style="font-family:sans-serif;padding:40px;background:#f0fff4">
        <h2 style="color:#276749">&#10003; Schwab Connected! (Accounts &amp; Trading)</h2>
        <p>Tokens saved. Real-time WebSocket streamer started.</p>
        <p>Streaming: Level 1 quotes, 1-min candles, NASDAQ screener, NQ/ES futures.</p>
        <p>Now authorise the <b>Market Data</b> app: <a href="/schwab/auth/md">/schwab/auth/md</a></p>
        <p><a href="/">&#8592; Back to Dashboard</a></p></body></html>"""
        return HTMLResponse(html)
    else:
        html = """<html><body style="font-family:sans-serif;padding:40px">
        <h2 style="color:#e53e3e">Token Exchange Failed</h2>
        <p>Check server logs for details.</p>
        <p><a href="/schwab/auth">Try again</a></p></body></html>"""
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
    redirect_uri = str(request.base_url).rstrip("/") + "/schwab/callback/md"
    return RedirectResponse(url=build_md_auth_url(redirect_uri))


@app.get("/schwab/callback/md")
async def schwab_md_web_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    """Schwab callback for Market Data app."""
    from fastapi.responses import HTMLResponse
    if error or not code:
        html = f"""<html><body style="font-family:sans-serif;padding:40px">
        <h2 style="color:#e53e3e">Schwab Market Data Auth Failed</h2>
        <p>{error or 'No code received.'}</p>
        <p><a href="/schwab/auth/md">Try again</a></p></body></html>"""
        return HTMLResponse(html, status_code=400)

    redirect_uri = str(request.base_url).rstrip("/") + "/schwab/callback/md"
    success = exchange_md_auth_code(code, state, redirect_uri)
    if success:
        html = """<html><body style="font-family:sans-serif;padding:40px;background:#f0fff4">
        <h2 style="color:#276749">&#10003; Schwab Market Data Connected!</h2>
        <p>Tokens saved. REST quotes, IV, movers and price history are now live.</p>
        <p><a href="/">&#8592; Back to Dashboard</a></p></body></html>"""
        return HTMLResponse(html)
    else:
        html = """<html><body style="font-family:sans-serif;padding:40px">
        <h2 style="color:#e53e3e">Market Data Token Exchange Failed</h2>
        <p>Check server logs for details.</p>
        <p><a href="/schwab/auth/md">Try again</a></p></body></html>"""
        return HTMLResponse(html, status_code=500)


@app.get("/api/broker/status")
async def broker_status():
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
        return {
            **ts,
            "schwab_enabled":  SCHWAB_ENABLED,
            "market_data_app": ts_md,
            "account":         acct,
            "daily":           daily,
            "auto_trade":      _tos_auto_trade,
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


@app.post("/api/broker/auth")
async def broker_auth():
    """Initiate Schwab OAuth flow — redirect browser to /schwab/auth instead."""
    from config import SCHWAB_ENABLED
    if not SCHWAB_ENABLED:
        return {"success": False, "error": "Schwab is disabled (SCHWAB_ENABLED=false in .env)"}
    return {"success": False, "error": "Use the browser flow: visit /schwab/auth to authorise"}


@app.get("/api/broker/positions")
async def broker_positions():
    """Current open positions in the ThinkorSwim paper account."""
    from config import SCHWAB_ENABLED
    if not SCHWAB_ENABLED:
        return {"positions": [], "schwab_enabled": False}
    try:
        return {"positions": get_positions()}
    except Exception as e:
        return {"positions": [], "error": str(e)}


@app.get("/api/broker/orders")
async def broker_orders():
    """Recent working orders."""
    from config import SCHWAB_ENABLED
    if not SCHWAB_ENABLED:
        return {"orders": [], "schwab_enabled": False}
    try:
        return {"orders": get_orders()}
    except Exception as e:
        return {"orders": [], "error": str(e)}


@app.post("/api/broker/auto-trade/{enabled}")
async def broker_auto_trade(enabled: str):
    """Toggle fully-automatic order placement (true/false)."""
    global _tos_auto_trade
    _tos_auto_trade = enabled.lower() == "true"
    return {"auto_trade": _tos_auto_trade}


@app.post("/api/broker/order")
async def broker_manual_order(body: dict):
    """
    Manually trigger a bracket order for a ticker already in the signal list.
    Body: { "ticker": "NVDA" }
    """
    ticker = body.get("ticker", "").upper()
    sig = next((s for s in scanner.signals if s.ticker == ticker), None)
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
async def streamer_start_endpoint():
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
        return {"movers": fetch_movers(index, sort, freq), "index": index, "sort": sort}
    except Exception as e:
        return {"movers": [], "error": str(e)}


@app.get("/api/market/hours")
async def market_hours_endpoint(market: str = "equity"):
    """Current market session status via Schwab."""
    try:
        from agent.broker.schwab_market_data import fetch_market_hours
        return fetch_market_hours(market)
    except Exception as e:
        return {"is_open": None, "error": str(e)}


@app.get("/api/market/iv/{ticker}")
async def ticker_iv(ticker: str):
    """Implied volatility for a single ticker via Schwab option chains."""
    try:
        from agent.broker.schwab_market_data import fetch_iv
        iv = fetch_iv(ticker.upper())
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
async def pipeline_metrics():
    """Return current pipeline throughput and worker metrics."""
    try:
        from agent.pipeline import get_pipeline
        return get_pipeline().get_metrics()
    except Exception as e:
        return {"error": str(e)}


# ── Signal blend weights ─────────────────────────────────────────────────────

@app.get("/api/blend-weights")
async def blend_weights():
    """Return current signal blender weight stats."""
    try:
        from agent.signal_blender import get_blender
        return get_blender().get_stats()
    except Exception as e:
        return {"error": str(e)}


# ── Backtester ───────────────────────────────────────────────────────────────

@app.get("/api/backtest/results")
async def backtest_results():
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
async def run_backtest(background_tasks: BackgroundTasks):
    """Trigger a fresh backtester run in the background."""
    try:
        from agent.backtester import get_backtester
        background_tasks.add_task(get_backtester().run_sync)
        return {"status": "started"}
    except Exception as e:
        return {"error": str(e)}


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

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    logger.info(f"WebSocket client connected. Total: {len(manager.active)}")
    try:
        # Send current state immediately on connect
        if scanner.signals:
            payload = _dumps({
                "type": "update",
                "signals": [s.to_dict() for s in scanner.signals],
            })
            await ws.send_text(payload)

        while True:
            # Keep connection alive; scanner thread pushes updates
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
        logger.info(f"WebSocket client disconnected. Total: {len(manager.active)}")
    except Exception as e:
        manager.disconnect(ws)
        logger.warning(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
