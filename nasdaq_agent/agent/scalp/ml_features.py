"""Leakage-safe feature contract for the advisory scalp ML overlay."""
from __future__ import annotations

import math
from typing import Any

from .models import ScalpSignalPlan

FEATURE_SCHEMA_VERSION = 3
MIN_PLAN_SCHEMA_VERSION = 2

FEATURE_NAMES = (
    "side_long", "side_short",
    "session_pre_market", "session_regular", "session_after_hours",
    "rsi_14", "rsi_7", "rsi_2",
    "rsi_extreme_os", "rsi_os", "rsi_ob", "rsi_extreme_ob",
    "macd_hist_atr", "macd_slope_atr",
    "atr_pct", "vwap_distance_atr", "rvol",
    "spread_bps", "spread_to_risk",
    "vwap_reclaim", "vwap_rejection", "vwap_above", "vwap_below",
    "path_clear", "path_blocked",
    "source_ws", "source_rest",
    "setup_score", "quote_age_s", "bar_age_min", "reward_r",
    "context_fresh", "sentiment_30m", "sentiment_velocity",
    "news_shock", "context_risk_score", "earnings_caution",
    "earnings_days_away",
    "family_reversal", "mtf_bullish", "mtf_bearish", "mtf_mixed",
    "mtf_aligned", "mtf_conflict", "rsi_14_5m",
    "macd_hist_5m_atr", "macd_slope_5m_atr", "atr_5m_pct",
    "vwap_5m_distance_atr",
)


def feature_vector(plan: ScalpSignalPlan | dict[str, Any]) -> list[float]:
    """Return a fixed-order vector using only facts available before entry."""
    read = plan.get if isinstance(plan, dict) else lambda key, default=None: getattr(plan, key, default)
    side = _text(read("side", "NONE"))
    session = _text(read("session", "UNKNOWN"))
    rsi_zone = _text(read("rsi_zone", "UNKNOWN"))
    vwap_event = _text(read("vwap_event", "UNKNOWN"))
    path = _text(read("tp2_path", "UNKNOWN"))
    source = _text(read("source", "UNKNOWN"))
    earnings_phase = _text(read("earnings_phase", ""))
    strategy_family = _text(read("strategy_family", "REVERSAL"))
    mtf_state = _text(read("mtf_state", "NO_DATA"))
    mtf_alignment = _text(read("mtf_alignment", "NO_DATA"))
    price = _number(read("price", 0.0))
    atr = _number(read("atr_14", 0.0))
    vwap = _number(read("vwap", 0.0))
    atr_denom = atr if atr > 1e-9 else 1.0
    atr_5m = _number(read("atr_14_5m", 0.0))
    vwap_5m = _number(read("vwap_5m", 0.0))
    atr_5m_denom = atr_5m if atr_5m > 1e-9 else 1.0

    values = (
        side == "LONG", side == "SHORT",
        session == "PRE_MARKET", session in {"REGULAR", "STANDARD", "PRIME", "MIDDAY"}, session == "AFTER_HOURS",
        _clip(_number(read("rsi_14", 50.0)) / 100.0, 0.0, 1.0),
        _clip(_number(read("rsi_7", 50.0)) / 100.0, 0.0, 1.0),
        _clip(_number(read("rsi_2", 50.0)) / 100.0, 0.0, 1.0),
        rsi_zone == "EXTREME_OS", rsi_zone == "OS", rsi_zone == "OB", rsi_zone == "EXTREME_OB",
        _clip(_number(read("macd_hist", 0.0)) / atr_denom, -5.0, 5.0),
        _clip(_number(read("macd_slope", 0.0)) / atr_denom, -5.0, 5.0),
        _clip((atr / price * 100.0) if price > 0 else 0.0, 0.0, 10.0),
        _clip(((price - vwap) / atr_denom) if price > 0 and vwap > 0 else 0.0, -10.0, 10.0),
        _clip(_number(read("rvol", 0.0)), 0.0, 10.0),
        _clip(_number(read("spread_bps", 0.0)), 0.0, 500.0),
        _clip(_number(read("spread_to_risk", 0.0)), 0.0, 5.0),
        "RECLAIM" in vwap_event, "REJECTION" in vwap_event,
        vwap_event in {"ABOVE", "HOLD_ABOVE"}, vwap_event in {"BELOW", "HOLD_BELOW"},
        path == "CLEAR", path.startswith("BLOCKED_BY_"),
        source in {"WS", "LIVE"}, source in {"REST", "REST_FALLBACK"},
        _clip(_number(read("setup_score", read("base_confidence", 0.0))) / 100.0, 0.0, 1.0),
        _clip(_number(read("data_age_ms", 0.0)) / 1000.0, 0.0, 60.0),
        _clip(_number(read("bar_age_ms", 0.0)) / 60000.0, 0.0, 60.0),
        _clip(_number(read("reward_r", 2.0)), 0.0, 5.0),
        bool(read("context_fresh", False)),
        _clip(_number(read("sentiment_30m", 0.0)), -1.0, 1.0),
        _clip(_number(read("sentiment_velocity", 0.0)), -2.0, 2.0),
        bool(read("news_shock", False)),
        _clip(_number(read("context_risk_score", 0.0)), 0.0, 1.0),
        earnings_phase == "CAUTION",
        _clip(_number(read("earnings_days_away", 999.0)), 0.0, 30.0) / 30.0,
        strategy_family == "REVERSAL",
        mtf_state == "BULLISH", mtf_state == "BEARISH", mtf_state == "MIXED",
        mtf_alignment == "ALIGNED", mtf_alignment == "CONFLICT",
        _clip(_number(read("rsi_14_5m", 50.0)) / 100.0, 0.0, 1.0),
        _clip(_number(read("macd_hist_5m", 0.0)) / atr_5m_denom, -5.0, 5.0),
        _clip(_number(read("macd_slope_5m", 0.0)) / atr_5m_denom, -5.0, 5.0),
        _clip((atr_5m / price * 100.0) if price > 0 else 0.0, 0.0, 20.0),
        _clip(
            ((price - vwap_5m) / atr_5m_denom)
            if price > 0 and vwap_5m > 0 else 0.0,
            -10.0,
            10.0,
        ),
    )
    return [float(value) for value in values]


def expected_r(tp1_probability: float, tp2_probability: float, reward_r: float) -> float:
    """Conservative two-stage payoff: stop=-1R, TP1-only=+0.5R."""
    p1 = _clip(_number(tp1_probability), 0.0, 1.0)
    p2 = min(p1, _clip(_number(tp2_probability), 0.0, 1.0))
    reward = _clip(_number(reward_r), 0.0, 5.0)
    return -1.0 + 1.5 * p1 + 0.5 * reward * p2


def bounded_confidence_adjustment(expectancy_r: float) -> float:
    from agent.config_manager import config

    raw = _number(expectancy_r) * max(
        0.0, float(config.get("scalp_ml.confidence_points_per_r", 5.0))
    )
    raise_cap = max(0.0, float(config.get("scalp_ml.max_confidence_raise", 5.0)))
    reduce_cap = max(0.0, float(config.get("scalp_ml.max_confidence_reduction", 15.0)))
    return round(_clip(raw, -reduce_cap, raise_cap), 3)


def _text(value: Any) -> str:
    return str(getattr(value, "value", value) or "UNKNOWN").upper()


def _number(value: Any) -> float:
    try:
        result = float(value or 0.0)
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))
