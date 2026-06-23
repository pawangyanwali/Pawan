"""
Paper-trading routes:
  GET  /api/paper-trading
  GET  /api/paper-trading/daily
  GET  /api/paper-trading/performance
  GET  /api/account-state
  POST /api/account-config
  GET  /api/algo-performance
  GET  /api/risk-status
  GET  /api/premarket-scan
  POST /api/premarket-scan/run
"""

import asyncio
import logging
import time

from fastapi import APIRouter, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from auth.dependencies import require_viewer, require_trader, require_analyst, AuthenticatedUser
from routers._deps import _pt_executor

from agent.paper_trading import (
    get_summary as pt_summary,
    get_open_trades,
    get_closed_trades,
    get_daily_pnl,
    get_today_pnl,
    get_equity_curve,
    get_weekly_pnl,
    get_ticker_pnl,
    get_account_state,
    update_account_config,
    get_algo_performance,
    get_trade_analysis,
)

router = APIRouter(tags=["paper_trading"])


def _latest_prices_for_open_trades(open_trades: list[dict]) -> dict[str, dict]:
    """Read current open-trade prices from Valkey without calling Schwab REST."""
    tickers = {str(t.get("ticker") or "").upper() for t in open_trades}
    tickers.discard("")
    if not tickers:
        return {}
    try:
        from agent.valkey_client import get_price
        prices: dict[str, dict] = {}
        for ticker in tickers:
            quote = get_price(ticker)
            if not quote:
                continue
            last = float(quote.get("last") or quote.get("mark") or 0)
            if last <= 0:
                continue
            prices[ticker] = {
                "last": last,
                "source": quote.get("source") or quote.get("source_status") or "VALKEY",
                "updated_at": float(quote.get("updated_at") or 0),
            }
        return prices
    except Exception as exc:
        logging.getLogger(__name__).debug("[PT] price enrichment skipped: %s", exc)
        return {}


def _enrich_open_trades_with_unrealized(open_trades: list[dict]) -> tuple[list[dict], float]:
    """Add current_price and unrealized P&L fields for the paper modal."""
    prices = _latest_prices_for_open_trades(open_trades)
    now = time.time()
    total_unrealized = 0.0
    enriched: list[dict] = []
    for trade in open_trades:
        t = dict(trade)
        ticker = str(t.get("ticker") or "").upper()
        quote = prices.get(ticker)
        if quote:
            try:
                current = float(quote["last"])
                entry = float(t.get("entry_price") or 0)
                shares = int(t.get("shares_remaining") or t.get("shares") or 0)
                direction = str(t.get("direction") or "BUY").upper()
                if current > 0 and entry > 0 and shares > 0:
                    pnl = ((current - entry) * shares
                           if direction == "BUY"
                           else (entry - current) * shares)
                    denom = entry * shares
                    t["current_price"] = round(current, 4)
                    t["unrealized_pnl_dollar"] = round(pnl, 2)
                    t["unrealized_pnl_pct"] = round((pnl / denom * 100) if denom else 0.0, 3)
                    t["unrealized_price_source"] = quote.get("source")
                    if quote.get("updated_at"):
                        t["unrealized_price_age_s"] = round(max(0.0, now - quote["updated_at"]), 1)
                    total_unrealized += pnl
            except Exception:
                logging.getLogger(__name__).debug(
                    "[PT] unrealized P&L failed for %s", ticker, exc_info=True
                )
        enriched.append(t)
    return enriched, round(total_unrealized, 2)


@router.get("/api/paper-trading")
async def paper_trading_endpoint(_user: AuthenticatedUser = Depends(require_viewer)):
    """Return paper trading summary, open and recent closed trades."""
    from datetime import date

    try:
        from agent.paper_trading import get_today_pnl as _get_today_pnl
        loop = asyncio.get_running_loop()
        closed, summary, open_trades, today_db = await asyncio.gather(
            loop.run_in_executor(_pt_executor, get_closed_trades, 50),
            loop.run_in_executor(_pt_executor, pt_summary),
            loop.run_in_executor(_pt_executor, get_open_trades),
            loop.run_in_executor(_pt_executor, _get_today_pnl),
        )
    except Exception as _e:
        logging.getLogger(__name__).error(f"[PT] paper-trading endpoint error: {_e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(_e), "summary": {
            "open": 0, "closed": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "avg_pnl": 0.0, "total_dollar_pnl": 0.0,
            "display_period": "all-time",
        }, "open_trades": [], "closed_trades": []})

    open_trades, _open_unrealized = _enrich_open_trades_with_unrealized(open_trades)

    # today_db comes directly from a COUNT(*) SQL query — always accurate regardless
    # of how many trades exist. Do not derive today's count from the display list.
    _today_total     = int(today_db.get("total")  or 0)
    _today_wins      = int(today_db.get("wins")   or 0)
    _today_losses    = _today_total - _today_wins
    _today_dollar    = float(today_db.get("total_pnl_dollar") or 0)
    _today_wr        = round(_today_wins / _today_total * 100, 1) if _today_total else 0.0
    # Real portfolio % return: total P&L dollars / starting budget
    _total_pnl_pct   = round(float(today_db.get("total_pnl_pct") or 0), 3)
    # Avg per-trade % from SQL AVG(pnl_pct) across individual closed trades
    _avg_pnl_pct     = round(float(today_db.get("avg_pnl_pct") or 0), 3)
    _avg_win_dollar  = round(float(today_db.get("avg_win_dollar") or 0), 2)
    _avg_loss_dollar = round(float(today_db.get("avg_loss_dollar") or 0), 2)
    # Budget from get_summary() which reads ConfigManager — never use account_config default
    _budget          = float(summary.get("starting_balance") or 50000)

    # All-time stats from pt_summary() which queries ALL closed trades (no LIMIT cap)
    _all_time_dollar  = round(float(summary.get("total_dollar_pnl") or 0.0), 2)
    _all_time_closed  = int(summary.get("closed") or 0)
    _all_time_wins    = int(summary.get("wins")   or 0)
    _all_time_losses  = int(summary.get("losses") or 0)

    from agent.config_manager import config as _cfg_mgr
    _max_daily = int(_cfg_mgr.get("risk.max_daily_trades", 30))

    use_today      = _today_total >= 3
    display_period = "today" if use_today else "all-time"
    summary.update({
        "closed":               _today_total if use_today else _all_time_closed,
        "wins":                 _today_wins  if use_today else sum(1 for t in closed if (t.get("pnl_dollar") or 0) > 0),
        "losses":               _today_losses if use_today else sum(1 for t in closed if (t.get("pnl_dollar") or 0) <= 0),
        "win_rate":             _today_wr if use_today else (
            round(sum(1 for t in closed if (t.get("pnl_dollar") or 0) > 0) / len(closed) * 100, 1) if closed else 0.0
        ),
        "avg_pnl":              _avg_pnl_pct if use_today else (
            round(sum(t.get("pnl_pct", 0) or 0 for t in closed) / len(closed), 3) if closed else 0.0
        ),
        # total_pnl = real portfolio % return (total_dollar / budget * 100), NOT avg per trade
        "total_pnl":            _total_pnl_pct if use_today else (
            round((_all_time_dollar / _budget * 100), 3) if _budget else 0.0
        ),
        "avg_pnl_pct":          _avg_pnl_pct,
        "total_dollar_pnl":     _today_dollar,
        "avg_win_dollar":       _avg_win_dollar,
        "avg_loss_dollar":      _avg_loss_dollar,
        "all_time_dollar":      _all_time_dollar,
        "all_time_closed":      _all_time_closed,
        "all_time_wins":        _all_time_wins,
        "all_time_losses":      _all_time_losses,
        "today_closed":         _today_total,
        "today_wins":           _today_wins,
        "today_losses":         _today_losses,
        "open_unrealized_pnl":  _open_unrealized,
        "today_total_with_unrealized": round(_today_dollar + _open_unrealized, 2),
        "display_period":       display_period,
        "daily_trades_allowed": _max_daily,
        "budget":               _budget,
    })
    return {
        "summary":       summary,
        "open_trades":   open_trades,
        "closed_trades": closed[:30],
    }


@router.get("/api/account-state")
async def api_account_state():
    """Full account state: capital, P&L, drawdown, config."""
    try:
        loop = asyncio.get_running_loop()
        try:
            from agent.valkey_client import get_all_prices
            prices = {
                ticker: float(q.get("last") or q.get("mark") or 0)
                for ticker, q in get_all_prices().items()
                if float(q.get("last") or q.get("mark") or 0) > 0
            }
        except Exception:
            prices = {}
        state = await loop.run_in_executor(None, lambda: get_account_state(prices))
        return state
    except Exception as e:
        return {"error": str(e)}


@router.post("/api/account-config")
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


@router.get("/api/algo-performance")
async def algo_performance_endpoint():
    """Per-algorithm signal fire stats and closed-trade performance."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_pt_executor, get_algo_performance)
    return result


@router.get("/api/risk-status")
async def risk_status():
    """Full PRD risk engine status: circuit breaker, profit protect, portfolio heat, session."""
    from agent.risk_controls import get_risk_status
    return get_risk_status()


@router.get("/api/premarket-scan")
async def premarket_scan_endpoint():
    """Pre-market gapper scan results and today's focus watchlist."""
    try:
        from agent.premarket_scanner import get_scan_status, get_focus_watchlist
        status = get_scan_status()
        return {**status, "focus_watchlist": get_focus_watchlist()}
    except Exception as e:
        return {"error": str(e)}


@router.post("/api/premarket-scan/run")
async def trigger_premarket_scan(background_tasks: BackgroundTasks,
                                  _user: AuthenticatedUser = Depends(require_analyst)):
    """Manually trigger a pre-market gapper scan."""
    try:
        from agent.premarket_scanner import run_premarket_scan_background
        run_premarket_scan_background()
        return {"status": "started"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/paper-trading/daily")
async def paper_daily_pnl():
    """Per-day P&L summary for last 14 days."""
    return {"daily": get_daily_pnl(days=14), "today": get_today_pnl()}


@router.get("/api/execution-quality")
async def execution_quality():
    """Today's execution quality metrics: slippage, spread, strategy P&L vs actual P&L."""
    loop = asyncio.get_running_loop()
    try:
        from agent.execution.paper_broker import get_daily_execution_quality
        from agent.paper_trading import get_ticker_cooldowns
        quality = await loop.run_in_executor(_pt_executor, get_daily_execution_quality)
        cooldowns = await loop.run_in_executor(_pt_executor, get_ticker_cooldowns)
        quality["ticker_cooldowns"] = cooldowns
        return quality
    except Exception as _e:
        return {"error": str(_e)}


@router.get("/api/trade-analysis")
async def trade_analysis(date: str | None = None, _user: AuthenticatedUser = Depends(require_viewer)):
    """
    Day-end diagnostic breakdown for a given date (YYYY-MM-DD ET, defaults to today).
    Returns trade stats grouped by exit_reason, confidence, algo family, session,
    direction, T1 hit/miss, regime, and top winning/losing tickers.
    """
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(_pt_executor, get_trade_analysis, date)
        return result
    except Exception as exc:
        logging.getLogger(__name__).error("[trade-analysis] %s", exc, exc_info=True)
        return {"error": str(exc), "trades": 0}


@router.get("/api/paper-trading/performance")
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
