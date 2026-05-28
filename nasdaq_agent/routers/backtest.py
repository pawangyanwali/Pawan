"""
Backtest / historical routes:
  GET  /api/backtest/stats
  GET  /api/backtest/tracking
  GET  /api/backtest/recent
  GET  /api/backtest/path/{signal_id}
  GET  /api/backtest/results
  POST /api/backtest/run
  POST /api/historical/retrain
  GET  /api/historical/retrain/status
  POST /api/historical/backtest/run
  GET  /api/historical/backtest/results
  GET  /api/backtest/mtf
  GET  /api/backtest/mtf/history
  GET  /api/backtest/mtf/{ticker}
  GET  /api/performance/*
  GET  /api/pipeline-metrics
  GET  /api/blend-weights
  GET  /api/clusters
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from auth.dependencies import require_viewer, require_analyst, require_admin, AuthenticatedUser
from routers._deps import _sanitize

from agent.live_backtest import get_performance_stats, get_tracking_signals, get_recent_resolved, get_price_path
from agent.backtest_reporter import get_broadcast_summary, get_full_report

router = APIRouter(tags=["backtest"])

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).parent.parent          # nasdaq_agent/
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


@router.get("/api/backtest/stats")
async def backtest_stats(lookback_days: int = 30):
    """Full backtest performance report with attribution breakdown."""
    import asyncio
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, get_full_report, lookback_days)
    return JSONResponse(content=_sanitize(data))


@router.get("/api/backtest/tracking")
async def backtest_tracking():
    """Currently open (TRACKING) signals being monitored."""
    import asyncio
    loop = asyncio.get_running_loop()
    tracking = await loop.run_in_executor(None, get_tracking_signals)
    return {"tracking": tracking}


@router.get("/api/backtest/recent")
async def backtest_recent(limit: int = 50):
    """Recently resolved backtest signals."""
    import asyncio
    loop   = asyncio.get_running_loop()
    recent = await loop.run_in_executor(None, get_recent_resolved, limit)
    for r in recent:
        r["outcome_color"] = (
            "#00ff88" if r["status"] == "WIN" else
            "#ef4444" if r["status"] == "LOSS" else
            "#64748b"
        )
    return {"recent": recent}


@router.get("/api/backtest/path/{signal_id}")
async def backtest_path(signal_id: str):
    """Price path bars for a specific signal (for replay/chart)."""
    return {"signal_id": signal_id, "path": get_price_path(signal_id)}


@router.get("/api/backtest/results")
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


@router.post("/api/backtest/run")
async def run_backtest(background_tasks: BackgroundTasks,
                       _user: AuthenticatedUser = Depends(require_analyst)):
    """Trigger a fresh backtester run in the background."""
    try:
        from agent.backtester import get_backtester
        background_tasks.add_task(get_backtester().run_sync)
        return {"status": "started"}
    except Exception as e:
        return {"error": str(e)}


@router.post("/api/historical/retrain")
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


@router.get("/api/historical/retrain/status")
async def historical_retrain_status():
    """Live progress of the historical retrain subprocess (reads status file)."""
    state = _read_status(_RETRAIN_STATUS)
    state["proc_running"] = _proc_running(_hist_retrain_proc)
    if not state.get("proc_running") and state.get("running"):
        state["running"] = False
    return state


@router.post("/api/historical/backtest/run")
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


@router.get("/api/historical/backtest/results")
async def historical_backtest_results():
    """Live progress and final results of the historical backtest (reads status file)."""
    state = _read_status(_BACKTEST_STATUS)
    state["proc_running"] = _proc_running(_hist_backtest_proc)
    if not state.get("proc_running") and state.get("running"):
        state["running"] = False
    return state


@router.get("/api/backtest/mtf")
async def mtf_summary():
    """
    Aggregated multi-timeframe backtest results from the most recent weekend run.
    Returns win rate, expectancy, Sharpe, and max-drawdown for each TF.
    """
    from agent.multi_tf_backtest import get_summary
    return get_summary()


@router.get("/api/backtest/mtf/history")
async def mtf_history():
    """List of past multi-TF backtest runs with aggregate stats."""
    from agent.multi_tf_backtest import get_run_history
    return get_run_history()


@router.get("/api/backtest/mtf/{ticker}")
async def mtf_ticker(ticker: str):
    """Per-TF performance breakdown for a single ticker."""
    from agent.multi_tf_backtest import get_ticker_stats
    return get_ticker_stats(ticker.upper())


@router.get("/api/pipeline-metrics")
async def pipeline_metrics(_user: AuthenticatedUser = Depends(require_viewer)):
    """Return current pipeline throughput and worker metrics."""
    try:
        from agent.pipeline import get_pipeline
        return get_pipeline().get_metrics()
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/blend-weights")
async def blend_weights(_user: AuthenticatedUser = Depends(require_viewer)):
    """Return current signal blender weight stats."""
    try:
        from agent.signal_blender import get_blender
        return get_blender().get_stats()
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/clusters")
async def ticker_clusters():
    """Return the ML cluster membership lists and per-ticker assignment map."""
    from config import CLUSTER_A_TICKERS, CLUSTER_B_TICKERS, CLUSTER_C_TICKERS, TICKER_CLUSTER
    return {
        "A": CLUSTER_A_TICKERS,
        "B": CLUSTER_B_TICKERS,
        "C": CLUSTER_C_TICKERS,
        "assignments": TICKER_CLUSTER,
    }
