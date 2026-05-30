"""
Algorithm Intelligence Center routes — /api/algo/*

Endpoints:
  GET  /api/algo/overview
  GET  /api/algo/leaderboard
  GET  /api/algo/trade-learning-timeline
  GET  /api/algo/loss-heatmap
  GET  /api/algo/params-full
  POST /api/algo/params/set
  POST /api/algo/params/reset/{family}
  GET  /api/algo/trade-attribution
  GET  /api/algo/signal-feed
  GET  /api/algo/ml-influence
  GET  /api/algo/adaptive-filter
  GET  /api/algo/export/csv
"""
from __future__ import annotations

import csv
import io
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi import Path as FPath
from fastapi import Body
from fastapi.responses import StreamingResponse

from agent.db import get_conn
from auth.dependencies import (
    require_viewer,
    require_analyst,
    require_admin,
    AuthenticatedUser,
)

router = APIRouter(tags=["algo"])
logger = logging.getLogger(__name__)

# ── Algo families and parameter spec ─────────────────────────────────────────

_ALL_FAMILIES = [
    "ORB", "GAP_TREND", "GAP_FADE", "AH_GAP_FADE", "BREAKOUT", "FLAG",
    "VWAP_SCALP", "LEVEL_SCALP", "RS_REGIME", "OFI", "VWAP_OFI",
    "EMA_PULL", "VWAP_TREND", "DONCHIAN", "ORB_ZV",
]

_PARAM_SPEC: dict[str, dict] = {
    "target_mult":       {"default": 1.5,  "min": 0.75, "max": 3.0},
    "stop_mult":         {"default": 1.0,  "min": 0.50, "max": 2.0},
    "rvol_gate":         {"default": 1.5,  "min": 1.0,  "max": 3.0},
    "conf_gate":         {"default": 55.0, "min": 45.0, "max": 75.0},
    "entry_window_bars": {"default": 3,    "min": 1,    "max": 8},
}

_ROOT_CAUSES = [
    "STOP_TOO_TIGHT",
    "WRONG_DIRECTION",
    "REGIME_MISMATCH",
    "VWAP_CONFLICT",
    "TIMING_LATE",
    "TIMEOUT_DRIFT",
    "VOLATILITY_SPIKE",
]

# Algo name → family mapping (abbreviated; covers the 15 listed families)
_ALGO_FAMILY_MAP: dict[str, str] = {  # noqa: E501
    "ORB5_BULL": "ORB", "ORB5_BEAR": "ORB",
    "ORB15_BULL": "ORB", "ORB15_BEAR": "ORB",
    "GAP_AND_GO_BULL": "GAP_TREND", "GAP_AND_GO_BEAR": "GAP_TREND",
    "GAP_FADE_BULL": "GAP_FADE", "GAP_FADE_BEAR": "GAP_FADE",
    "AH_GAP_FADE_BEAR": "AH_GAP_FADE", "AH_GAP_FADE_BULL": "AH_GAP_FADE",
    "PDH_BREAKOUT_BULL": "BREAKOUT", "PDL_BREAKDOWN_BEAR": "BREAKOUT",
    "HOD_BREAK_BULL": "BREAKOUT", "LOD_BREAK_BEAR": "BREAKOUT",
    "BULL_FLAG": "FLAG", "BEAR_FLAG": "FLAG",
    "VWAP_TOUCH_SCALP_BULL": "VWAP_SCALP", "VWAP_TOUCH_SCALP_BEAR": "VWAP_SCALP",
    "VWAP_HOD_SCALP": "VWAP_SCALP", "VWAP_LOD_SCALP": "VWAP_SCALP",
    "LEVEL_REJECTION_SCALP_BULL": "LEVEL_SCALP", "LEVEL_REJECTION_SCALP_BEAR": "LEVEL_SCALP",
    "MICRO_PULLBACK_SCALP_BULL": "LEVEL_SCALP", "MICRO_PULLBACK_SCALP_BEAR": "LEVEL_SCALP",
    "SPY_BETA_CATCHUP_BULL": "RS_REGIME", "SPY_BETA_CATCHUP_BEAR": "RS_REGIME",
    "RESIDUAL_MOMENTUM_BULL": "RS_REGIME", "RESIDUAL_REVERSION_BEAR": "RS_REGIME",
    "SECTOR_LEADER_BULL": "RS_REGIME", "REGIME_ALIGNED_LONG": "RS_REGIME",
    "OFI_IMPULSE_BULL": "OFI", "OFI_IMPULSE_BEAR": "OFI",
    "VWAP_OFI_PULL_BULL": "VWAP_OFI", "VWAP_OFI_PULL_BEAR": "VWAP_OFI",
    "EMA_SLOPE_PULL_BULL": "EMA_PULL", "EMA_SLOPE_PULL_BEAR": "EMA_PULL",
    "VWAP_TREND_BRK_BULL": "VWAP_TREND", "VWAP_TREND_BRK_BEAR": "VWAP_TREND",
    "DONCHIAN_BRK_BULL": "DONCHIAN", "DONCHIAN_BRK_BEAR": "DONCHIAN",
    "ORB_VWAP_ZV_BULL": "ORB_ZV", "ORB_VWAP_ZV_BEAR": "ORB_ZV",
    "BB_MEAN_REV_BULL": "BB_MEAN_REV", "BB_MEAN_REV_BEAR": "BB_MEAN_REV",
}

# Reverse mapping: family → list of algo_names that belong to it
_FAMILY_TO_ALGOS: dict[str, list[str]] = {}
for _a, _f in _ALGO_FAMILY_MAP.items():
    _FAMILY_TO_ALGOS.setdefault(_f, []).append(_a)


def _algo_to_family(algo_name: str) -> str:
    """Best-effort algo name → family lookup."""
    if not algo_name:
        return "UNKNOWN"
    direct = _ALGO_FAMILY_MAP.get(algo_name.upper())
    if direct:
        return direct
    upper = algo_name.upper()
    for fam in _ALL_FAMILIES:
        if fam in upper:
            return fam
    return "UNKNOWN"


# ── Range helper ──────────────────────────────────────────────────────────────

def _range_clause(col: str, range_val: str) -> tuple[str, list]:
    """
    Return (sql_fragment, params) for a WHERE range filter.
    Columns are TEXT containing ISO-8601 timestamps — cast to TIMESTAMPTZ first.
    """
    r = range_val or "today"
    if r == "today":
        return f"AND {col}::TIMESTAMPTZ::DATE = CURRENT_DATE", []
    elif r == "24h":
        return f"AND {col}::TIMESTAMPTZ >= NOW() - INTERVAL '24 hours'", []
    elif r == "7d":
        return f"AND {col}::TIMESTAMPTZ >= NOW() - INTERVAL '7 days'", []
    elif r == "30d":
        return f"AND {col}::TIMESTAMPTZ >= NOW() - INTERVAL '30 days'", []
    return f"AND {col}::TIMESTAMPTZ::DATE = CURRENT_DATE", []


def _safe_float(val, default=0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


def _safe_int(val, default=0) -> int:
    try:
        if val is None:
            return default
        return int(val)
    except Exception:
        return default


def _row_to_dict(row) -> dict:
    """Convert a psycopg2 RealDictRow to a plain Python dict."""
    if row is None:
        return {}
    try:
        return dict(row)
    except Exception:
        return {}


# ── GET /api/algo/overview ────────────────────────────────────────────────────

@router.get("/api/algo/overview")
async def algo_overview(
    _user: AuthenticatedUser = Depends(require_viewer),
):
    """High-level algorithm system health: fires, trades, tune events, filter state."""

    evals_today = 0
    fires_today = 0
    filtered_today = 0
    trades_today = 0
    fire_rate_pct = 0.0
    tune_events_today = 0
    confidence_gate = 55.0
    last_cycle_ago_s: Optional[float] = None
    feedback_loop_active = False

    # 1. Signal fires and trades-opened today
    # fires_today  = entries where trade_opened = 1  (algo actually fired a trade)
    # evals_today  = total signal evaluations logged  (pass to frontend separately)
    try:
        with get_conn() as c:
            row = c.execute(
                """
                SELECT
                    COUNT(*) AS evals,
                    COALESCE(SUM(CASE WHEN trade_opened = 1 THEN 1 ELSE 0 END), 0) AS fires,
                    COALESCE(SUM(CASE WHEN trade_opened = 0 THEN 1 ELSE 0 END), 0) AS filtered
                FROM algo_signal_log
                WHERE logged_at IS NOT NULL AND logged_at != ''
                  AND logged_at::TIMESTAMPTZ::DATE = CURRENT_DATE
                """
            ).fetchone()
        if row:
            evals_today   = _safe_int(row["evals"])
            fires_today   = _safe_int(row["fires"])
            trades_today  = fires_today            # fires == trades opened
            filtered_today = _safe_int(row["filtered"])
            if evals_today > 0:
                fire_rate_pct = round(fires_today / evals_today * 100, 1)
    except Exception as exc:
        evals_today = filtered_today = 0
        logger.debug("algo/overview signal query error: %s", exc)

    # 2. Param tune events today
    try:
        with get_conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS cnt FROM param_tune_log "
                "WHERE tuned_at IS NOT NULL AND tuned_at != '' "
                "AND tuned_at::TIMESTAMPTZ::DATE = CURRENT_DATE"
            ).fetchone()
        if row:
            tune_events_today = _safe_int(row["cnt"])
    except Exception as exc:
        logger.debug("algo/overview tune query error: %s", exc)

    # 3. Adaptive filter: confidence gate
    try:
        from agent.adaptive_filter import get_status as af_get_status
        af_status = af_get_status()
        confidence_gate = _safe_float(af_status.get("dynamic_threshold"), 55.0)
    except Exception as exc:
        logger.debug("algo/overview adaptive_filter error: %s", exc)

    # 4. Learning engine: last cycle ago + feedback loop active
    try:
        from agent.learning_engine import learning_engine
        eng_status = learning_engine.get_status()
        last_cycle_str = eng_status.get("last_cycle")
        if last_cycle_str:
            try:
                last_cycle_dt = datetime.fromisoformat(last_cycle_str)
                if last_cycle_dt.tzinfo is None:
                    last_cycle_dt = last_cycle_dt.replace(tzinfo=timezone.utc)
                last_cycle_ago_s = round(
                    (datetime.now(timezone.utc) - last_cycle_dt).total_seconds(), 1
                )
            except Exception:
                last_cycle_ago_s = None
        feedback_loop_active = bool(eng_status.get("running", False))
    except Exception as exc:
        logger.debug("algo/overview learning_engine error: %s", exc)

    return {
        "evals_today":        evals_today,
        "fires_today":        fires_today,
        "filtered_today":     filtered_today,
        "trades_today":       trades_today,
        "fire_rate_pct":      fire_rate_pct,
        "tune_events_today":  tune_events_today,
        "confidence_gate":    confidence_gate,
        "last_cycle_ago_s":   last_cycle_ago_s,
        "feedback_loop_active": feedback_loop_active,
    }


# ── GET /api/algo/leaderboard ─────────────────────────────────────────────────

@router.get("/api/algo/leaderboard")
async def algo_leaderboard(
    range: str = "today",
    _user: AuthenticatedUser = Depends(require_viewer),
):
    """Per-algo performance leaderboard with status classification."""
    try:
        return _build_leaderboard(range)
    except Exception as exc:
        logger.warning("algo/leaderboard error: %s", exc)
        return []


def _build_leaderboard(range_val: str) -> list:
    range_clause, _ = _range_clause("logged_at", range_val)

    # 1. Fires per algo from signal log
    fires_by_algo: dict[str, int] = {}
    try:
        with get_conn() as c:
            rows = c.execute(
                f"SELECT algo, COUNT(*) AS cnt FROM algo_signal_log WHERE 1=1 {range_clause} GROUP BY algo"
            ).fetchall()
        for row in rows:
            fires_by_algo[row["algo"]] = _safe_int(row["cnt"])
    except Exception as exc:
        logger.debug("leaderboard fires query error: %s", exc)

    # 2. Trades per algo from paper_trades
    trades_range_clause, _ = _range_clause("closed_at", range_val)
    trades_by_algo: dict[str, dict] = {}
    try:
        with get_conn() as c:
            rows = c.execute(
                f"""
                SELECT
                    algo_name,
                    COUNT(*) AS total,
                    SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) AS wins,
                    AVG(pnl_pct) AS avg_pnl
                FROM paper_trades
                WHERE status = 'CLOSED' AND algo_name != '' AND algo_name IS NOT NULL
                {trades_range_clause}
                GROUP BY algo_name
                """
            ).fetchall()
        for row in rows:
            trades_by_algo[row["algo_name"]] = {
                "total": _safe_int(row["total"]),
                "wins":  _safe_int(row["wins"]),
                "avg_pnl": _safe_float(row["avg_pnl"]),
            }
    except Exception as exc:
        logger.debug("leaderboard trades query error: %s", exc)

    # 3. Last tune_at per family
    last_tune_by_family: dict[str, Optional[str]] = {}
    try:
        with get_conn() as c:
            rows = c.execute(
                "SELECT family, MAX(tuned_at) AS last_tune FROM param_tune_log GROUP BY family"
            ).fetchall()
        for row in rows:
            last_tune_by_family[row["family"]] = (
                row["last_tune"].isoformat() if hasattr(row["last_tune"], "isoformat") else str(row["last_tune"])
                if row["last_tune"] else None
            )
    except Exception as exc:
        logger.debug("leaderboard tune query error: %s", exc)

    # 4. UCB weights from algo_params (use conf_gate as a proxy for quality)
    ucb_by_family: dict[str, float] = {}
    try:
        with get_conn() as c:
            rows = c.execute(
                "SELECT family, param, current_val FROM algo_params WHERE param = 'conf_gate'"
            ).fetchall()
        for row in rows:
            # Normalise: conf_gate 45..75 → ucb_weight 0..1
            spec = _PARAM_SPEC["conf_gate"]
            span = spec["max"] - spec["min"]
            norm = (_safe_float(row["current_val"]) - spec["min"]) / span if span else 0.5
            ucb_by_family[row["family"]] = round(1.0 - norm, 3)  # higher conf needed = lower weight
    except Exception as exc:
        logger.debug("leaderboard algo_params query error: %s", exc)

    # Collect all known algo identifiers
    all_algos: set[str] = set(fires_by_algo.keys()) | set(trades_by_algo.keys())
    if not all_algos:
        return []

    now = datetime.now(timezone.utc)
    result = []
    for algo in sorted(all_algos):
        family = _algo_to_family(algo)
        td = trades_by_algo.get(algo, {})
        total = td.get("total", 0)
        wins  = td.get("wins", 0)
        losses = total - wins
        win_rate = round(wins / total * 100, 1) if total > 0 else 0.0
        fires = fires_by_algo.get(algo, 0)
        avg_pnl = round(td.get("avg_pnl", 0.0), 3)
        last_tune = last_tune_by_family.get(family)
        ucb_weight = ucb_by_family.get(family, 0.5)

        # Determine status
        tuned_recently = False
        if last_tune:
            try:
                lt_dt = datetime.fromisoformat(last_tune)
                if lt_dt.tzinfo is None:
                    lt_dt = lt_dt.replace(tzinfo=timezone.utc)
                tuned_recently = (now - lt_dt).total_seconds() < 86400
            except Exception:
                pass

        if ucb_weight < 0.3:
            status = "PAUSED"
        elif tuned_recently:
            status = "TUNED"
        elif total >= 5 and win_rate < 40.0:
            status = "UNDERPERFORMING"
        elif win_rate >= 50.0:
            status = "ACTIVE"
        else:
            status = "ACTIVE"

        result.append({
            "algo": algo,
            "family": family,
            "fires": fires,
            "trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "avg_pnl_pct": avg_pnl,
            "last_tune": last_tune,
            "ucb_weight": ucb_weight,
            "status": status,
        })

    result.sort(key=lambda x: (-x["trades"], -x["win_rate"]))
    return result


# ── GET /api/algo/trade-learning-timeline ─────────────────────────────────────

@router.get("/api/algo/trade-learning-timeline")
async def algo_trade_learning_timeline(
    range: str = "today",
    page: int = 1,
    per_page: int = 50,
    _user: AuthenticatedUser = Depends(require_viewer),
):
    """Closed trades paired with any param-tune events that occurred within 10s of close."""
    range_clause, _ = _range_clause("closed_at", range)

    trades = []
    total = 0

    try:
        offset = (max(1, page) - 1) * per_page
        with get_conn() as c:
            cnt_row = c.execute(
                f"""
                SELECT COUNT(*) AS cnt FROM paper_trades
                WHERE status = 'CLOSED' AND algo_name != '' AND algo_name IS NOT NULL
                {range_clause}
                """
            ).fetchone()
            total = _safe_int(cnt_row["cnt"]) if cnt_row else 0

            rows = c.execute(
                f"""
                SELECT id, opened_at, closed_at, ticker, direction, entry_price,
                       exit_price, exit_reason, pnl_pct, pnl_dollar, shares,
                       session, regime, vwap_event, entry_type, algo_name, status,
                       confidence, rr_ratio, bars_held
                FROM paper_trades
                WHERE status = 'CLOSED' AND algo_name != '' AND algo_name IS NOT NULL
                {range_clause}
                ORDER BY closed_at DESC
                LIMIT %s OFFSET %s
                """,
                [per_page, offset],
            ).fetchall()
    except Exception as exc:
        logger.warning("trade-learning-timeline query error: %s", exc)
        return {"items": [], "total": 0, "page": page, "per_page": per_page, "error": str(exc)}

    items = []
    for row in rows:
        trade = _row_to_dict(row)
        # Serialize datetime fields
        for dt_field in ("opened_at", "closed_at"):
            if trade.get(dt_field) and hasattr(trade[dt_field], "isoformat"):
                trade[dt_field] = trade[dt_field].isoformat()

        closed_at_val = row["closed_at"]

        learning_events = []
        try:
            with get_conn() as c2:
                tune_rows = c2.execute(
                    """
                    SELECT param, old_val, new_val, reason, family,
                           EXTRACT(EPOCH FROM (tuned_at - %s)) * 1000 AS latency_ms
                    FROM param_tune_log
                    WHERE tuned_at BETWEEN %s AND %s + INTERVAL '10 seconds'
                    ORDER BY tuned_at
                    """,
                    [closed_at_val, closed_at_val, closed_at_val],
                ).fetchall()
            for tr in tune_rows:
                learning_events.append({
                    "param":      tr["param"],
                    "old_val":    _safe_float(tr["old_val"]),
                    "new_val":    _safe_float(tr["new_val"]),
                    "reason":     tr["reason"],
                    "family":     tr["family"],
                    "latency_ms": round(_safe_float(tr["latency_ms"]), 1),
                })
        except Exception as exc:
            logger.debug("trade-learning-timeline tune lookup error: %s", exc)

        items.append({
            "trade": trade,
            "learning_events": learning_events,
            "note": None if learning_events else "no_change",
        })

    return {"items": items, "total": total, "page": page, "per_page": per_page}


# ── GET /api/algo/loss-heatmap ────────────────────────────────────────────────

@router.get("/api/algo/loss-heatmap")
async def algo_loss_heatmap(
    range: str = "today",
    _user: AuthenticatedUser = Depends(require_viewer),
):
    """Loss root-cause heatmap: family × cause matrix."""
    range_clause, _ = _range_clause("tuned_at", range)
    trades_range_clause, _ = _range_clause("closed_at", range)

    # family → cause → count
    matrix: dict[str, dict[str, int]] = {}
    family_totals: dict[str, int] = {}

    # 1. Build from param_tune_log reasons
    try:
        with get_conn() as c:
            rows = c.execute(
                f"""
                SELECT family, reason, COUNT(*) AS cnt
                FROM param_tune_log
                WHERE 1=1 {range_clause}
                GROUP BY family, reason
                """
            ).fetchall()
        for row in rows:
            fam = row["family"] or "UNKNOWN"
            reason_text = (row["reason"] or "").upper()
            if fam not in matrix:
                matrix[fam] = {cause: 0 for cause in _ROOT_CAUSES}
                family_totals[fam] = 0
            matched_cause = None
            for cause in _ROOT_CAUSES:
                if cause in reason_text:
                    matched_cause = cause
                    break
            if matched_cause:
                matrix[fam][matched_cause] = matrix[fam].get(matched_cause, 0) + _safe_int(row["cnt"])
                family_totals[fam] += _safe_int(row["cnt"])
    except Exception as exc:
        logger.debug("loss-heatmap tune query error: %s", exc)

    # 2. Supplement from paper_trades exit_reason
    try:
        with get_conn() as c:
            rows = c.execute(
                f"""
                SELECT algo_name, exit_reason, COUNT(*) AS cnt
                FROM paper_trades
                WHERE status = 'CLOSED' AND algo_name != '' AND algo_name IS NOT NULL
                  AND pnl_pct < 0
                  {trades_range_clause}
                GROUP BY algo_name, exit_reason
                """
            ).fetchall()
        for row in rows:
            fam = _algo_to_family(row["algo_name"] or "")
            exit_r = (row["exit_reason"] or "").upper()
            if fam not in matrix:
                matrix[fam] = {cause: 0 for cause in _ROOT_CAUSES}
                family_totals[fam] = 0
            matched_cause = None
            for cause in _ROOT_CAUSES:
                if cause in exit_r:
                    matched_cause = cause
                    break
            if not matched_cause:
                # Map common exit reasons to root causes
                if "STOP" in exit_r:
                    matched_cause = "STOP_TOO_TIGHT"
                elif "TIMEOUT" in exit_r or "TIME" in exit_r:
                    matched_cause = "TIMEOUT_DRIFT"
                elif "VWAP" in exit_r:
                    matched_cause = "VWAP_CONFLICT"
                elif "REGIME" in exit_r:
                    matched_cause = "REGIME_MISMATCH"
            if matched_cause:
                matrix[fam][matched_cause] = matrix[fam].get(matched_cause, 0) + _safe_int(row["cnt"])
                family_totals[fam] += _safe_int(row["cnt"])
    except Exception as exc:
        logger.debug("loss-heatmap trades query error: %s", exc)

    # Convert counts to rates
    rate_matrix: dict[str, dict[str, float]] = {}
    for fam, causes in matrix.items():
        total = family_totals.get(fam, 0)
        rate_matrix[fam] = {
            cause: round(cnt / total, 4) if total > 0 else 0.0
            for cause, cnt in causes.items()
        }

    families = sorted(rate_matrix.keys())
    return {
        "matrix": rate_matrix,
        "families": families,
        "causes": _ROOT_CAUSES,
        "raw_counts": matrix,
    }


# ── GET /api/algo/params-full ─────────────────────────────────────────────────

@router.get("/api/algo/params-full")
async def algo_params_full(
    _user: AuthenticatedUser = Depends(require_viewer),
):
    """Full per-family parameter table: current, default, previous values + today's tune count."""
    # Query algo_params for all values
    params_by_family: dict[str, dict[str, dict]] = {}
    try:
        with get_conn() as c:
            rows = c.execute(
                """
                SELECT family, param, current_val, previous_val, rollback_val,
                       last_updated_cycle, last_reason, updated_at
                FROM algo_params
                ORDER BY family, param
                """
            ).fetchall()
        for row in rows:
            fam = row["family"]
            par = row["param"]
            if fam not in params_by_family:
                params_by_family[fam] = {}
            params_by_family[fam][par] = {
                "current_val":        _safe_float(row["current_val"]),
                "previous_val":       _safe_float(row["previous_val"]),
                "rollback_val":       _safe_float(row["rollback_val"]),
                "last_updated_cycle": _safe_int(row["last_updated_cycle"]),
                "last_reason":        row["last_reason"] or "",
                "updated_at":         (
                    row["updated_at"].isoformat()
                    if row["updated_at"] and hasattr(row["updated_at"], "isoformat")
                    else str(row["updated_at"] or "")
                ),
            }
    except Exception as exc:
        logger.debug("algo/params-full query error: %s", exc)

    # Tune counts today per family
    tune_count_by_family: dict[str, int] = {}
    try:
        with get_conn() as c:
            rows = c.execute(
                "SELECT family, COUNT(*) AS cnt FROM param_tune_log "
                "WHERE tuned_at IS NOT NULL AND tuned_at != '' "
                "AND tuned_at::TIMESTAMPTZ::DATE = CURRENT_DATE GROUP BY family"
            ).fetchall()
        for row in rows:
            tune_count_by_family[row["family"]] = _safe_int(row["cnt"])
    except Exception as exc:
        logger.debug("algo/params-full tune count error: %s", exc)

    # Build unified result covering all known families
    result = []
    all_families = sorted(set(_ALL_FAMILIES) | set(params_by_family.keys()))
    for fam in all_families:
        fam_params = params_by_family.get(fam, {})
        param_list = []
        for param_name, spec in _PARAM_SPEC.items():
            row_data = fam_params.get(param_name, {})
            param_list.append({
                "param":         param_name,
                "current_val":   row_data.get("current_val", spec["default"]),
                "default_val":   spec["default"],
                "previous_val":  row_data.get("previous_val", spec["default"]),
                "rollback_val":  row_data.get("rollback_val", spec["default"]),
                "min_val":       spec["min"],
                "max_val":       spec["max"],
                "last_reason":   row_data.get("last_reason", ""),
                "updated_at":    row_data.get("updated_at", ""),
            })
        result.append({
            "family":     fam,
            "params":     param_list,
            "tune_count": tune_count_by_family.get(fam, 0),
        })

    return result


# ── GET /api/algo/tune-log ────────────────────────────────────────────────────

@router.get("/api/algo/tune-log")
async def algo_tune_log(
    range:    str = Query("today"),
    page:     int = Query(1, ge=1),
    per_page: int = Query(50, le=200),
    family:   str = Query(""),
    _user: AuthenticatedUser = Depends(require_viewer),
):
    """Tune event log: every param change, why it happened, and post-tune outcome."""
    try:
        return _build_tune_log(range, page, per_page, family.strip().upper())
    except Exception as exc:
        logger.warning("algo/tune-log error: %s", exc)
        return {"items": [], "total": 0}


def _build_tune_log(range_val: str, page: int, per_page: int, family_filter: str) -> dict:
    range_clause, _ = _range_clause("tl.tuned_at", range_val)
    fam_clause = "AND tl.family = %s" if family_filter else ""
    params_list: list = [family_filter] if family_filter else []
    offset = (page - 1) * per_page

    try:
        with get_conn() as c:
            row = c.execute(
                f"SELECT COUNT(*) AS cnt FROM param_tune_log tl WHERE 1=1 {range_clause} {fam_clause}",
                params_list,
            ).fetchone()
        total = _safe_int(row["cnt"]) if row else 0

        # Fetch tune events + triggering trade via LEFT JOIN.
        # Post-tune outcome is computed in a separate query below to avoid
        # correlated subqueries against TEXT-typed closed_at column.
        with get_conn() as c:
            rows = c.execute(
                f"""
                SELECT
                    tl.id,
                    tl.family,
                    tl.param,
                    tl.old_val,
                    tl.new_val,
                    tl.reason,
                    tl.source,
                    tl.tuned_at,
                    tl.trigger_trade_id,
                    tl.trigger_ms,
                    pt.ticker       AS trigger_ticker,
                    pt.direction    AS trigger_dir,
                    pt.status       AS trigger_status,
                    pt.pnl_pct      AS trigger_pnl
                FROM param_tune_log tl
                LEFT JOIN paper_trades pt ON pt.id = tl.trigger_trade_id
                WHERE 1=1 {range_clause} {fam_clause}
                ORDER BY tl.tuned_at DESC
                LIMIT {per_page} OFFSET {offset}
                """,
                params_list,
            ).fetchall()
    except Exception as exc:
        logger.warning("algo/tune-log query error: %s", exc)
        return {"items": [], "total": 0}

    # Post-tune outcome: for each tune event, count closed trades for that family
    # in the 2 hours following the tune.  Done as a single aggregated query (not
    # per-row correlated subquery) to avoid casting TEXT closed_at inside a
    # correlated subquery where bad values can't be filtered out before the cast.
    post_tune_map: dict[int, dict] = {}
    if rows:
        try:
            # Expand each (id, family, tuned_at) into (id, algo_name, tuned_at) rows
            # using the reverse _FAMILY_TO_ALGOS map so the join matches actual algo_name
            # values in paper_trades (e.g. family "ORB" → ["ORB5_BULL","ORB5_BEAR",...])
            cte_rows: list[tuple] = []
            for r in rows:
                tid, fam, tat = r["id"], r["family"], r["tuned_at"]
                if not tat:
                    continue
                algos = _FAMILY_TO_ALGOS.get(str(fam).upper(), [str(fam)])
                for algo in algos:
                    cte_rows.append((tid, algo, tat))

            if not cte_rows:
                raise ValueError("no valid cte_rows")

            # Parameterized VALUES list — no f-string interpolation of user data
            val_placeholders = ", ".join("(%s, %s, %s::TIMESTAMPTZ)" for _ in cte_rows)
            val_params = [v for tid, algo, tat in cte_rows for v in (tid, algo, tat)]

            with get_conn() as c:
                post_rows = c.execute(
                    f"""
                    WITH tune_windows(tid, algo_name, tuned_at) AS (VALUES {val_placeholders})
                    SELECT
                        tw.tid,
                        COUNT(pt.id)                               AS post_total,
                        SUM(CASE WHEN pt.pnl_pct > 0 THEN 1 ELSE 0 END) AS post_wins
                    FROM tune_windows tw
                    LEFT JOIN paper_trades pt
                        ON  pt.algo_name = tw.algo_name
                        AND pt.status = 'CLOSED'
                        AND pt.closed_at IS NOT NULL AND pt.closed_at != ''
                        AND pt.closed_at::TIMESTAMPTZ > tw.tuned_at
                        AND pt.closed_at::TIMESTAMPTZ < tw.tuned_at + INTERVAL '2 hours'
                    GROUP BY tw.tid
                    """,
                    val_params,
                ).fetchall()
            for pr in post_rows:
                pt_total = _safe_int(pr["post_total"])
                pt_wins  = _safe_int(pr["post_wins"])
                post_tune_map[pr["tid"]] = {
                    "post_total":    pt_total,
                    "post_wins":     pt_wins,
                    "post_losses":   pt_total - pt_wins,
                    "post_win_rate": round(pt_wins / pt_total * 100, 1) if pt_total > 0 else None,
                }
        except Exception as _pt_exc:
            logger.debug("algo/tune-log post-tune stats error: %s", _pt_exc)

    items = []
    for r in rows:
        old_v  = _safe_float(r["old_val"])
        new_v  = _safe_float(r["new_val"])
        delta  = round(new_v - old_v, 4)
        direction = "up" if delta > 0 else ("down" if delta < 0 else "same")

        pt_stats    = post_tune_map.get(r["id"], {})
        post_total  = pt_stats.get("post_total", 0)
        post_wins   = pt_stats.get("post_wins", 0)
        post_losses = pt_stats.get("post_losses", 0)
        post_wr     = pt_stats.get("post_win_rate")

        trigger_pnl = None
        if r.get("trigger_pnl") is not None:
            try:
                trigger_pnl = round(float(r["trigger_pnl"]), 2)
            except (TypeError, ValueError):
                pass

        items.append({
            "id":               r["id"],
            "family":           r["family"],
            "param":            r["param"],
            "old_val":          round(old_v, 4),
            "new_val":          round(new_v, 4),
            "delta":            delta,
            "direction":        direction,
            "reason":           r["reason"] or "",
            "source":           r["source"] or "auto",
            "tuned_at":         (
                r["tuned_at"].isoformat()
                if r["tuned_at"] and hasattr(r["tuned_at"], "isoformat")
                else str(r["tuned_at"] or "")
            ),
            "trigger_trade_id": r.get("trigger_trade_id"),
            "trigger_ms":       r.get("trigger_ms"),
            "trigger_ticker":   r.get("trigger_ticker"),
            "trigger_dir":      r.get("trigger_dir"),
            "trigger_status":   r.get("trigger_status"),
            "trigger_pnl":      trigger_pnl,
            "post_total":       post_total,
            "post_wins":        post_wins,
            "post_losses":      post_losses,
            "post_win_rate":    post_wr,
        })

    return {"items": items, "total": total}


# ── POST /api/algo/params/set ─────────────────────────────────────────────────

@router.post("/api/algo/params/set")
async def algo_params_set(
    payload: dict = Body(...),
    _user: AuthenticatedUser = Depends(require_admin),
):
    """Admin override: set a single algo-family parameter value."""
    family = (payload.get("family") or "").strip().upper()
    param  = (payload.get("param") or "").strip().lower()
    value  = payload.get("value")

    if not family or not param:
        return {"success": False, "error": "family and param are required"}
    if param not in _PARAM_SPEC:
        return {
            "success": False,
            "error": f"Unknown param '{param}'. Valid params: {list(_PARAM_SPEC.keys())}",
        }
    try:
        value = float(value)
    except (TypeError, ValueError):
        return {"success": False, "error": "value must be numeric"}

    spec = _PARAM_SPEC[param]
    if value < spec["min"] or value > spec["max"]:
        return {
            "success": False,
            "error": f"value {value} out of bounds [{spec['min']}, {spec['max']}]",
        }

    old_val = spec["default"]
    try:
        with get_conn() as c:
            existing = c.execute(
                "SELECT current_val FROM algo_params WHERE family = %s AND param = %s",
                [family, param],
            ).fetchone()
            if existing:
                old_val = _safe_float(existing["current_val"])

            c.execute(
                """
                INSERT INTO algo_params (family, param, current_val, previous_val, rollback_val,
                                         last_updated_cycle, last_reason, updated_at)
                VALUES (%s, %s, %s, %s, %s, 0, 'manual_override', NOW())
                ON CONFLICT (family, param) DO UPDATE
                    SET previous_val = algo_params.current_val,
                        current_val  = EXCLUDED.current_val,
                        last_reason  = 'manual_override',
                        updated_at   = NOW()
                """,
                [family, param, value, old_val, old_val],
            )
            # Also log to param_tune_log
            c.execute(
                """
                INSERT INTO param_tune_log (family, param, old_val, new_val, reason, source, tuned_at)
                VALUES (%s, %s, %s, %s, 'manual_override', 'admin', NOW())
                """,
                [family, param, old_val, value],
            )
    except Exception as exc:
        logger.warning("algo/params/set error: %s", exc)
        return {"success": False, "error": str(exc)}

    return {
        "success": True,
        "family":  family,
        "param":   param,
        "old_val": old_val,
        "new_val": value,
    }


# ── POST /api/algo/params/reset/{family} ─────────────────────────────────────

@router.post("/api/algo/params/reset/{family}")
async def algo_params_reset(
    family: str = FPath(...),
    _user: AuthenticatedUser = Depends(require_admin),
):
    """Admin: reset all params for an algo family to defaults."""
    family = family.strip().upper()
    params_reset = []

    for param_name, spec in _PARAM_SPEC.items():
        default_val = spec["default"]
        try:
            with get_conn() as c:
                existing = c.execute(
                    "SELECT current_val FROM algo_params WHERE family = %s AND param = %s",
                    [family, param_name],
                ).fetchone()
                old_val = _safe_float(existing["current_val"]) if existing else default_val

                c.execute(
                    """
                    INSERT INTO algo_params (family, param, current_val, previous_val, rollback_val,
                                             last_updated_cycle, last_reason, updated_at)
                    VALUES (%s, %s, %s, %s, %s, 0, 'reset_to_default', NOW())
                    ON CONFLICT (family, param) DO UPDATE
                        SET previous_val = algo_params.current_val,
                            current_val  = EXCLUDED.current_val,
                            rollback_val = EXCLUDED.rollback_val,
                            last_reason  = 'reset_to_default',
                            updated_at   = NOW()
                    """,
                    [family, param_name, default_val, old_val, old_val],
                )
                c.execute(
                    """
                    INSERT INTO param_tune_log (family, param, old_val, new_val, reason, source, tuned_at)
                    VALUES (%s, %s, %s, %s, 'reset_to_default', 'admin', NOW())
                    """,
                    [family, param_name, old_val, default_val],
                )
            params_reset.append({
                "param":       param_name,
                "old_val":     old_val,
                "new_val":     default_val,
            })
        except Exception as exc:
            logger.warning("algo/params/reset/%s error for param %s: %s", family, param_name, exc)

    return {"success": True, "family": family, "params_reset": params_reset}


# ── GET /api/algo/trade-attribution ──────────────────────────────────────────

@router.get("/api/algo/trade-attribution")
async def algo_trade_attribution(
    range: str = "today",
    group_by: str = "algo",
    _user: AuthenticatedUser = Depends(require_viewer),
):
    """Trade attribution breakdown by algo, entry_type, session, or regime."""
    valid_groups = {"algo", "entry_type", "session", "regime"}
    if group_by not in valid_groups:
        group_by = "algo"

    range_clause, _ = _range_clause("closed_at", range)
    signal_range_clause, _ = _range_clause("logged_at", range)

    # Map group_by → column name in paper_trades
    col_map = {
        "algo":       "algo_name",
        "entry_type": "entry_type",
        "session":    "session",
        "regime":     "regime",
    }
    col = col_map[group_by]

    trades_by_label: dict[str, dict] = {}
    try:
        with get_conn() as c:
            rows = c.execute(
                f"""
                SELECT
                    COALESCE({col}, 'unknown') AS label,
                    COUNT(*) AS total,
                    SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN pnl_pct < 0 THEN 1 ELSE 0 END) AS losses,
                    SUM(CASE WHEN exit_reason ILIKE '%timeout%' THEN 1 ELSE 0 END) AS timeouts,
                    AVG(pnl_pct) AS avg_pnl_pct
                FROM paper_trades
                WHERE status = 'CLOSED'
                  AND closed_at IS NOT NULL
                  AND closed_at != ''
                {range_clause}
                GROUP BY {col}
                ORDER BY total DESC
                """
            ).fetchall()
        for row in rows:
            trades_by_label[row["label"] or "unknown"] = {
                "total":     _safe_int(row["total"]),
                "wins":      _safe_int(row["wins"]),
                "losses":    _safe_int(row["losses"]),
                "timeouts":  _safe_int(row["timeouts"]),
                "avg_pnl_pct": round(_safe_float(row["avg_pnl_pct"]), 3),
            }
    except Exception as exc:
        logger.warning("trade-attribution trades query error: %s", exc)
        return {"items": [], "error": str(exc)}

    # Fires per label from signal log (best effort, only for group_by=algo)
    fires_by_label: dict[str, int] = {}
    if group_by == "algo":
        try:
            with get_conn() as c:
                rows = c.execute(
                    f"""
                    SELECT algo, COUNT(*) AS cnt
                    FROM algo_signal_log
                    WHERE 1=1 {signal_range_clause}
                    GROUP BY algo
                    """
                ).fetchall()
            for row in rows:
                fires_by_label[row["algo"]] = _safe_int(row["cnt"])
        except Exception as exc:
            logger.debug("trade-attribution fires query error: %s", exc)

    result = []
    for label, td in trades_by_label.items():
        total  = td["total"]
        wins   = td["wins"]
        win_rate = round(wins / total * 100, 1) if total > 0 else 0.0
        result.append({
            "label":       label,
            "fires":       fires_by_label.get(label, 0),
            "trades":      total,
            "wins":        wins,
            "losses":      td["losses"],
            "timeouts":    td["timeouts"],
            "win_rate":    win_rate,
            "avg_pnl_pct": td["avg_pnl_pct"],
        })

    return {"items": result, "group_by": group_by, "range": range}


# ── GET /api/algo/signal-feed ─────────────────────────────────────────────────

@router.get("/api/algo/signal-feed")
async def algo_signal_feed(
    page: int = 1,
    per_page: int = 50,
    algo: str = "",
    ticker: str = "",
    result: str = "",
    _user: AuthenticatedUser = Depends(require_viewer),
):
    """Paginated algo signal log feed with optional filters."""
    where_clauses = ["1=1"]
    params: list = []

    if algo:
        where_clauses.append("algo = %s")
        params.append(algo)
    if ticker:
        where_clauses.append("ticker = %s")
        params.append(ticker.upper())
    if result == "opened":
        where_clauses.append("trade_opened = 1")
    elif result == "filtered":
        where_clauses.append("trade_opened = 0")

    where_sql = " AND ".join(where_clauses)
    offset = (max(1, page) - 1) * per_page
    total = 0
    items = []

    try:
        with get_conn() as c:
            cnt_row = c.execute(
                f"SELECT COUNT(*) AS cnt FROM algo_signal_log WHERE {where_sql}",
                params,
            ).fetchone()
            total = _safe_int(cnt_row["cnt"]) if cnt_row else 0

            rows = c.execute(
                f"""
                SELECT id, logged_at, ticker, algo, direction, confidence,
                       entry, stop, target, rr, trade_opened
                FROM algo_signal_log
                WHERE {where_sql}
                ORDER BY logged_at DESC
                LIMIT %s OFFSET %s
                """,
                params + [per_page, offset],
            ).fetchall()
        for row in rows:
            d = _row_to_dict(row)
            if d.get("logged_at") and hasattr(d["logged_at"], "isoformat"):
                d["logged_at"] = d["logged_at"].isoformat()
            # optional columns that may not exist yet
            d.setdefault("filter_reason", None)
            d.setdefault("ml_scalp_prob", None)
            items.append(d)
    except Exception as exc:
        # Try without optional columns that may not exist
        try:
            with get_conn() as c:
                cnt_row = c.execute(
                    f"SELECT COUNT(*) AS cnt FROM algo_signal_log WHERE {where_sql}",
                    params,
                ).fetchone()
                total = _safe_int(cnt_row["cnt"]) if cnt_row else 0

                rows = c.execute(
                    f"""
                    SELECT id, logged_at, ticker, algo, direction, confidence,
                           entry, stop, target, rr, trade_opened
                    FROM algo_signal_log
                    WHERE {where_sql}
                    ORDER BY logged_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    params + [per_page, offset],
                ).fetchall()
            for row in rows:
                d = _row_to_dict(row)
                if d.get("logged_at") and hasattr(d["logged_at"], "isoformat"):
                    d["logged_at"] = d["logged_at"].isoformat()
                d.setdefault("filter_reason", None)
                d.setdefault("ml_scalp_prob", None)
                items.append(d)
        except Exception as exc2:
            logger.warning("signal-feed query error: %s", exc2)
            return {"items": [], "total": 0, "page": page, "per_page": per_page, "error": str(exc2)}

    return {"items": items, "total": total, "page": page, "per_page": per_page}


# ── GET /api/algo/ml-influence ────────────────────────────────────────────────

@router.get("/api/algo/ml-influence")
async def algo_ml_influence(
    range: str = "today",
    page: int = 1,
    per_page: int = 50,
    _user: AuthenticatedUser = Depends(require_viewer),
):
    """Closed trades with ML probability scores (columns may not exist yet)."""
    range_clause, _ = _range_clause("closed_at", range)
    offset = (max(1, page) - 1) * per_page
    total = 0
    items = []

    # Try with ML columns first
    ml_cols = ", ml_scalp_prob, ml_daily_prob, ml_swing_prob, ml_deep_prob, ml_ensemble_score"
    null_guard = "AND closed_at IS NOT NULL AND closed_at != ''"
    try:
        with get_conn() as c:
            cnt_row = c.execute(
                f"""
                SELECT COUNT(*) AS cnt FROM paper_trades
                WHERE status = 'CLOSED' {null_guard} {range_clause}
                """
            ).fetchone()
            total = _safe_int(cnt_row["cnt"]) if cnt_row else 0

            rows = c.execute(
                f"""
                SELECT id, opened_at, closed_at, ticker, direction,
                       entry_price, exit_price, pnl_pct, pnl_dollar,
                       algo_name, session, regime, confidence
                       {ml_cols}
                FROM paper_trades
                WHERE status = 'CLOSED' {null_guard} {range_clause}
                ORDER BY closed_at DESC
                LIMIT %s OFFSET %s
                """,
                [per_page, offset],
            ).fetchall()

        for row in rows:
            d = _row_to_dict(row)
            for dt_field in ("opened_at", "closed_at"):
                if d.get(dt_field) and hasattr(d[dt_field], "isoformat"):
                    d[dt_field] = d[dt_field].isoformat()
            items.append(d)

    except Exception:
        # Fallback: no ML columns
        try:
            with get_conn() as c:
                cnt_row = c.execute(
                    f"""
                    SELECT COUNT(*) AS cnt FROM paper_trades
                    WHERE status = 'CLOSED' {null_guard} {range_clause}
                    """
                ).fetchone()
                total = _safe_int(cnt_row["cnt"]) if cnt_row else 0

                rows = c.execute(
                    f"""
                    SELECT id, opened_at, closed_at, ticker, direction,
                           entry_price, exit_price, pnl_pct, pnl_dollar,
                           algo_name, session, regime, confidence
                    FROM paper_trades
                    WHERE status = 'CLOSED' {null_guard} {range_clause}
                    ORDER BY closed_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    [per_page, offset],
                ).fetchall()

            for row in rows:
                d = _row_to_dict(row)
                for dt_field in ("opened_at", "closed_at"):
                    if d.get(dt_field) and hasattr(d[dt_field], "isoformat"):
                        d[dt_field] = d[dt_field].isoformat()
                # Stub out ML columns
                for ml_col in ("ml_scalp_prob", "ml_daily_prob", "ml_swing_prob",
                               "ml_deep_prob", "ml_ensemble_score"):
                    d[ml_col] = None
                items.append(d)

        except Exception as exc2:
            logger.warning("ml-influence query error: %s", exc2)
            return {"items": [], "total": 0, "page": page, "per_page": per_page, "error": str(exc2)}

    return {"items": items, "total": total, "page": page, "per_page": per_page}


# ── GET /api/algo/adaptive-filter ────────────────────────────────────────────

@router.get("/api/algo/adaptive-filter")
async def algo_adaptive_filter(
    _user: AuthenticatedUser = Depends(require_viewer),
):
    """Current adaptive filter state: confidence gate, win rate, threshold history."""
    try:
        from agent.adaptive_filter import get_status as af_get_status
        status = af_get_status()
    except Exception as exc:
        logger.warning("algo/adaptive-filter error: %s", exc)
        return {"error": str(exc)}

    # Augment with formatted fields
    dynamic_threshold = _safe_float(status.get("dynamic_threshold"), 55.0)
    current_win_rate  = _safe_float(status.get("current_win_rate"), 0.0)
    target_win_rate   = _safe_float(status.get("target_win_rate"), 55.0)
    total_resolved    = _safe_int(status.get("total_resolved"), 0)

    gap_to_target = round(target_win_rate - current_win_rate, 1)

    return {
        **status,
        "confidence_gate":    dynamic_threshold,
        "win_rate_pct":       current_win_rate,
        "target_win_rate_pct": target_win_rate,
        "gap_to_target_pct":  gap_to_target,
        "total_resolved":     total_resolved,
        "is_at_max":          dynamic_threshold >= 63.0,
        "is_at_default":      abs(dynamic_threshold - 55.0) < 0.1,
    }


# ── GET /api/algo/export/csv ──────────────────────────────────────────────────

@router.get("/api/algo/export/csv")
async def algo_export_csv(
    range: str = "today",
    type: str = "trades",
    _user: AuthenticatedUser = Depends(require_analyst),
):
    """Stream a CSV export of trades, signals, or tune events."""
    valid_types = {"trades", "signals", "tunes"}
    if type not in valid_types:
        type = "trades"

    range_clause_trades, _  = _range_clause("closed_at", range)
    range_clause_signals, _ = _range_clause("logged_at", range)
    range_clause_tunes, _   = _range_clause("tuned_at", range)

    output = io.StringIO()

    if type == "trades":
        fieldnames = [
            "id", "opened_at", "closed_at", "ticker", "direction",
            "entry_price", "exit_price", "exit_reason", "pnl_pct", "pnl_dollar",
            "shares", "session", "regime", "vwap_event", "entry_type",
            "algo_name", "status", "confidence", "rr_ratio", "bars_held",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        try:
            with get_conn() as c:
                rows = c.execute(
                    f"""
                    SELECT id, opened_at, closed_at, ticker, direction,
                           entry_price, exit_price, exit_reason, pnl_pct, pnl_dollar,
                           shares, session, regime, vwap_event, entry_type,
                           algo_name, status, confidence, rr_ratio, bars_held
                    FROM paper_trades
                    WHERE 1=1 {range_clause_trades}
                    ORDER BY closed_at DESC
                    """
                ).fetchall()
            for row in rows:
                d = _row_to_dict(row)
                for dt_field in ("opened_at", "closed_at"):
                    if d.get(dt_field) and hasattr(d[dt_field], "isoformat"):
                        d[dt_field] = d[dt_field].isoformat()
                writer.writerow(d)
        except Exception as exc:
            logger.warning("export/csv trades error: %s", exc)

    elif type == "signals":
        fieldnames = [
            "id", "logged_at", "ticker", "algo", "direction",
            "confidence", "entry", "stop", "target", "rr", "trade_opened",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        try:
            with get_conn() as c:
                rows = c.execute(
                    f"""
                    SELECT id, logged_at, ticker, algo, direction,
                           confidence, entry, stop, target, rr, trade_opened
                    FROM algo_signal_log
                    WHERE 1=1 {range_clause_signals}
                    ORDER BY logged_at DESC
                    """
                ).fetchall()
            for row in rows:
                d = _row_to_dict(row)
                if d.get("logged_at") and hasattr(d["logged_at"], "isoformat"):
                    d["logged_at"] = d["logged_at"].isoformat()
                writer.writerow(d)
        except Exception as exc:
            logger.warning("export/csv signals error: %s", exc)

    else:  # tunes
        fieldnames = [
            "id", "tuned_at", "family", "param", "old_val", "new_val", "reason", "source",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        try:
            with get_conn() as c:
                rows = c.execute(
                    f"""
                    SELECT id, tuned_at, family, param, old_val, new_val, reason, source
                    FROM param_tune_log
                    WHERE 1=1 {range_clause_tunes}
                    ORDER BY tuned_at DESC
                    """
                ).fetchall()
            for row in rows:
                d = _row_to_dict(row)
                if d.get("tuned_at") and hasattr(d["tuned_at"], "isoformat"):
                    d["tuned_at"] = d["tuned_at"].isoformat()
                writer.writerow(d)
        except Exception as exc:
            logger.warning("export/csv tunes error: %s", exc)

    output.seek(0)
    filename = f"algo_{type}_{range}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
