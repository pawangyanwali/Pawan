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
)

router = APIRouter(tags=["paper_trading"])


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

    # today_db comes directly from a COUNT(*) SQL query — always accurate regardless
    # of how many trades exist. Do not derive today's count from the display list.
    _today_total  = int(today_db.get("total")  or 0)
    _today_wins   = int(today_db.get("wins")   or 0)
    _today_losses = _today_total - _today_wins
    _today_dollar = float(today_db.get("total_pnl_dollar") or 0)
    _today_wr     = round(_today_wins / _today_total * 100, 1) if _today_total else 0.0
    # Budget from get_summary() which reads ConfigManager — never use account_config default
    _budget       = float(summary.get("starting_balance") or 50000)
    _avg_pnl_pct  = round(_today_dollar / _budget * 100 / _today_total, 3) if _today_total else 0.0

    # All-time stats from pt_summary() which queries ALL closed trades (no LIMIT cap)
    _all_time_dollar = round(float(summary.get("total_dollar_pnl") or 0.0), 2)
    _all_time_closed = int(summary.get("closed") or 0)

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
        "total_pnl":            _avg_pnl_pct,
        "total_dollar_pnl":     _today_dollar,
        "all_time_dollar":      _all_time_dollar,
        "all_time_closed":      _all_time_closed,
        "today_closed":         _today_total,
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
            from agent.broker.schwab_market_data import get_live_quotes
            prices = {t: q.get("last", 0) for t, q in get_live_quotes().items() if q.get("last")}
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
