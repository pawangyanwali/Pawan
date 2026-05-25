"""
Read-only data-quality checks executed on every startup.

Each check is independent and wrapped in try/except so a single failure
never blocks startup.  Results are written to the learning log so they
appear in the dashboard /api/learning-log panel.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Individual checks ─────────────────────────────────────────────────────────

def _check_confidence_scale() -> dict:
    """Flag any signals stored on the wrong 0-1 scale (value < 2.0)."""
    try:
        from agent.db import get_conn
        with get_conn() as c:
            bad = c.execute(
                "SELECT COUNT(*) AS n FROM signals WHERE confidence < 2.0 AND confidence > 0"
            ).fetchone()["n"]
            total = c.execute("SELECT COUNT(*) AS n FROM signals").fetchone()["n"]
        return {"ok": bad == 0, "bad_scale_rows": bad, "total_signals": total}
    except Exception as e:
        return {"ok": None, "error": str(e)}


def _check_live_backtest_r_zeros() -> dict:
    """Count bt_signals with r_multiple = 0 (degenerate entry == stop geometry)."""
    try:
        from agent.db import get_conn
        with get_conn() as c:
            zero_r = c.execute(
                "SELECT COUNT(*) AS n FROM bt_signals "
                "WHERE status IN ('WIN','LOSS','TIMEOUT') AND (r_multiple = 0 OR r_multiple IS NULL)"
            ).fetchone()["n"]
            total = c.execute(
                "SELECT COUNT(*) AS n FROM bt_signals WHERE status IN ('WIN','LOSS','TIMEOUT')"
            ).fetchone()["n"]
        ratio = round(zero_r / max(total, 1) * 100, 1)
        return {"ok": ratio < 10.0, "zero_r_count": zero_r, "resolved_total": total,
                "zero_r_pct": ratio}
    except Exception as e:
        return {"ok": None, "error": str(e)}


def _check_stale_tracking_signals() -> dict:
    """Count bt_signals in TRACKING status longer than MAX_BARS scans old."""
    try:
        from agent.db import get_conn
        from agent.live_backtest import MAX_BARS
        with get_conn() as c:
            stale = c.execute(
                "SELECT COUNT(*) AS n FROM bt_signals "
                "WHERE status = 'TRACKING' AND bars_tracked > ?",
                (MAX_BARS,)
            ).fetchone()["n"]
            tracking = c.execute(
                "SELECT COUNT(*) AS n FROM bt_signals WHERE status = 'TRACKING'"
            ).fetchone()["n"]
        return {"ok": stale == 0, "stale_tracking": stale, "total_tracking": tracking}
    except Exception as e:
        return {"ok": None, "error": str(e)}


def _check_adaptive_filter() -> dict:
    """Sanity-check adaptive filter state: threshold in range, WR plausible."""
    try:
        from agent.adaptive_filter import get_status, MIN_THRESHOLD, MAX_THRESHOLD
        st = get_status()
        threshold = float(st.get("dynamic_threshold", 0))
        wr         = float(st.get("current_win_rate", 0))
        blocked    = len(st.get("blocked_contexts", {}))
        ok = (MIN_THRESHOLD <= threshold <= MAX_THRESHOLD) and (0 <= wr <= 1)
        return {
            "ok": ok,
            "threshold": round(threshold, 1),
            "win_rate_pct": round(wr * 100, 1),
            "blocked_contexts": blocked,
        }
    except Exception as e:
        return {"ok": None, "error": str(e)}


def _check_open_paper_trade_geometry() -> dict:
    """Count open paper trades with inverted stop/target geometry."""
    try:
        from agent.db import get_conn
        with get_conn() as c:
            trades = c.execute(
                "SELECT direction, entry_price, stop_price, target_price "
                "FROM paper_trades WHERE status = 'OPEN' "
                "AND target_price > 0 AND stop_price > 0"
            ).fetchall()
        bad = 0
        for t in trades:
            d, e, s, tgt = t["direction"], t["entry_price"], t["stop_price"], t["target_price"]
            if d == "BUY" and (s >= e or tgt <= e):
                bad += 1
            elif d == "SELL" and (s <= e or tgt >= e):
                bad += 1
        return {"ok": bad == 0, "bad_geometry_open": bad, "total_open": len(trades)}
    except Exception as e:
        return {"ok": None, "error": str(e)}


def _check_live_backtest_expectancy() -> dict:
    """Report overall live backtest expectancy and warn if negative."""
    try:
        from agent.live_backtest import get_performance_stats
        stats = get_performance_stats(min_resolved=10)
        overall = stats.get("overall", {})
        exp = overall.get("expectancy", 0.0)
        n   = overall.get("total", 0)
        wr  = overall.get("win_rate", 0.0)
        return {
            "ok": exp >= 0 or n < 10,
            "expectancy_r": exp,
            "win_rate_pct": round(wr * 100, 1),
            "resolved": n,
        }
    except Exception as e:
        return {"ok": None, "error": str(e)}


def _check_ml_model_count() -> dict:
    """Report how many ML models are trained and loaded."""
    try:
        from agent.ml_model import (
            _model_registry, _daily_model_registry,
            _reversal_model_registry, _ensemble_registry,
            _swing_model_registry,
        )
        counts = {
            "scalp":    sum(1 for m in _model_registry.values()         if m.trained),
            "daily":    sum(1 for m in _daily_model_registry.values()   if m.trained),
            "reversal": sum(1 for m in _reversal_model_registry.values()if m.trained),
            "ensemble": sum(1 for m in _ensemble_registry.values()      if m.trained),
            "swing":    sum(1 for m in _swing_model_registry.values()   if m.trained),
        }
        return {"ok": True, "models_loaded": counts}
    except Exception as e:
        return {"ok": None, "error": str(e)}


# ── Entry point ───────────────────────────────────────────────────────────────

def run_startup_checks() -> list[dict]:
    """
    Run all data-quality checks and return results.
    Logs a summary to logger and to the learning log ring buffer.
    Never raises — all errors are captured per-check.
    """
    checks = [
        ("confidence_scale",       _check_confidence_scale),
        ("backtest_r_zeros",       _check_live_backtest_r_zeros),
        ("stale_tracking_signals", _check_stale_tracking_signals),
        ("adaptive_filter_state",  _check_adaptive_filter),
        ("open_trade_geometry",    _check_open_paper_trade_geometry),
        ("backtest_expectancy",    _check_live_backtest_expectancy),
        ("ml_model_count",         _check_ml_model_count),
    ]

    results = []
    warnings = []
    for name, fn in checks:
        t0 = time.time()
        try:
            result = fn()
        except Exception as e:
            result = {"ok": None, "error": str(e)}
        result["check"] = name
        result["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
        results.append(result)
        if result.get("ok") is False:
            warnings.append(name)

    n_fail = len(warnings)
    n_ok   = sum(1 for r in results if r.get("ok") is True)

    summary = (
        f"Startup data-quality: {n_ok}/{len(checks)} OK"
        + (f" | WARNINGS: {', '.join(warnings)}" if warnings else "")
    )
    logger.info(f"[StartupChecks] {summary}")

    # Push into learning log so dashboard shows it
    try:
        from agent.learning_engine import _log as _ll
        _ll(summary, level="WARNING" if warnings else "INFO", significant=bool(warnings))
    except Exception:
        pass

    # Log individual failing checks at WARNING level
    for r in results:
        if r.get("ok") is False:
            logger.warning(f"[StartupChecks] {r['check']}: {r}")

    return results
