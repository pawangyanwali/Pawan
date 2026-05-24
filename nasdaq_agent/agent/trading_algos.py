"""
Intraday trading algorithm signal evaluators — Phase 1 & 2 signals.

Each eval_*() function receives a StockSignal (as a plain namespace/dict-like
object) and returns an AlgoResult or None.  evaluate_all() runs every registered
algo and returns a list of all fired AlgoResult objects.

Algorithm catalogue (this file):
  Phase 1:
    1  — 5-min ORB        (ORB5_BULL / ORB5_BEAR)
    2  — 15-min ORB       (ORB15_BULL / ORB15_BEAR)
    6  — Gap-and-Go       (GAP_AND_GO_BULL / BEAR)
    7  — Gap Fade         (GAP_FADE_BULL / BEAR)
    8  — PDH/PDL Breakout (PDH_BREAKOUT_BULL / PDL_BREAKDOWN_BEAR)
    12 — HOD/LOD Break    (HOD_BREAK_BULL / LOD_BREAK_BEAR)
    29 — Bull Flag        (BULL_FLAG)
    30 — Bear Flag        (BEAR_FLAG)
  Phase 2 scalps:
    3  — VWAP Touch Scalp      (VWAP_TOUCH_SCALP_BULL / BEAR)
    4  — VWAP HOD Scalp        (VWAP_HOD_SCALP)
    5  — VWAP LOD Scalp        (VWAP_LOD_SCALP)
    9  — Level Rejection Scalp (LEVEL_REJECTION_SCALP_BULL / BEAR)
    11 — Micro Pullback Scalp  (MICRO_PULLBACK_SCALP_BULL / BEAR)
  Phase 3 relative value:
    13 — SPY Beta Catch-Up        (SPY_BETA_CATCHUP_BULL / BEAR)
    14 — Residual Momentum        (RESIDUAL_MOMENTUM_BULL)
    15 — Residual Reversion       (RESIDUAL_REVERSION_BEAR)
    16 — Sector Leader Momentum   (SECTOR_LEADER_BULL)
    17 — Sector Laggard Catch-Up  (SECTOR_LAGGARD_CATCHUP_BULL)
    18 — Sector Counter Fade      (SECTOR_COUNTER_FADE_BEAR)
    19 — Regime Aligned Long      (REGIME_ALIGNED_LONG)
    20 — Regime Aligned Short     (REGIME_ALIGNED_SHORT)
    21 — Sector Breakout Follow   (SECTOR_BREAKOUT_BULL / BEAR)
    22 — Cross-Sectional RS Rank  (CS_RS_RANK_BULL / BEAR)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.scanner import StockSignal

logger = logging.getLogger(__name__)

# ── Adaptive parameter integration ───────────────────────────────────────────
# Imports the algo learning engine to get tuned parameter values.
# Falls back gracefully to hardcoded defaults if the engine is unavailable.
try:
    from agent.algo_learning_engine import get_algo_params as _get_algo_params
    _ALE_AVAILABLE = True
except ImportError:
    _ALE_AVAILABLE = False


def _param(algo_family: str, name: str, default: float) -> float:
    """
    Get current tuned parameter value from the learning engine, fall back to default.
    Used by all algo eval functions (proof-of-concept shown in eval_orb5;
    all other algos follow the same pattern).
    """
    if not _ALE_AVAILABLE:
        return default
    try:
        params = _get_algo_params(algo_family)
        return float(params.get(name, default))
    except Exception:
        return default


@dataclass
class AlgoResult:
    algo:       str   # e.g. "ORB5_BULL"
    direction:  str   # "BUY" | "SELL"
    confidence: float # 0–100
    entry:      float # suggested entry price
    stop:       float # suggested stop price
    target:     float # suggested target price
    rr:         float # risk/reward ratio
    reason:     str   # human-readable trigger reason

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


# ── helpers ──────────────────────────────────────────────────────────────────

def _rr(entry: float, stop: float, target: float) -> float:
    risk   = abs(entry - stop)
    reward = abs(target - entry)
    return round(reward / risk, 2) if risk > 0 else 0.0


# ── Algo 1: 5-min Opening Range Breakout ─────────────────────────────────────

def eval_orb5(sig) -> Optional[AlgoResult]:
    """
    5-min ORB (Algo 1).

    Bull trigger : close > ORH-5, RVOL ≥ 1.5, MTF gate ≠ BEAR.
    Bear trigger : close < ORL-5, RVOL ≥ 1.5, MTF gate ≠ BULL.

    Entry  = current price (market order on trigger bar close).
    Stop   = ORL (bull) / ORH (bear).
    Target = entry ± (ORH5 - ORL5) × 1.5  (1.5× range projection).
    """
    try:
        orh  = float(sig.orb5_high)
        orl  = float(sig.orb5_low)
        if orh <= 0 or orl <= 0:
            return None

        bo   = getattr(sig, "orb5_breakout", "NONE")
        rvol = float(getattr(sig, "rel_volume", 1.0))
        gate = getattr(sig, "short_tf_alignment", "MIXED")
        price = float(sig.price)
        or_range = orh - orl

        # Adaptive parameters — tuned by AlgoLearningEngine, fall back to defaults
        # Other algos follow this same _param() pattern.
        _rvol_gate   = _param("ORB", "rvol_gate",   1.5)
        _target_mult = _param("ORB", "target_mult", 1.5)

        if bo == "BULL" and rvol >= _rvol_gate and gate != "BEAR":
            entry  = price
            stop   = orl
            target = entry + or_range * _target_mult
            conf   = min(95, 60 + (rvol - 1.5) * 10 + (10 if gate == "BULL" else 0))
            reason = (
                f"ORB-5 bull breakout: price {price:.2f} > ORH {orh:.2f}; "
                f"RVOL {rvol:.1f}x (gate {_rvol_gate:.1f}x); TF gate {gate}"
            )
            return AlgoResult(
                algo="ORB5_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )

        if bo == "BEAR" and rvol >= _rvol_gate and gate != "BULL":
            entry  = price
            stop   = orh
            target = entry - or_range * _target_mult
            conf   = min(95, 60 + (rvol - 1.5) * 10 + (10 if gate == "BEAR" else 0))
            reason = (
                f"ORB-5 bear breakdown: price {price:.2f} < ORL {orl:.2f}; "
                f"RVOL {rvol:.1f}x (gate {_rvol_gate:.1f}x); TF gate {gate}"
            )
            return AlgoResult(
                algo="ORB5_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )
    except Exception as exc:
        logger.debug("eval_orb5 error: %s", exc)
    return None


# ── Algo 2: 15-min Opening Range Breakout ────────────────────────────────────

def eval_orb15(sig) -> Optional[AlgoResult]:
    """
    15-min ORB (Algo 2).

    Bull trigger : close > ORH-15, RVOL ≥ 1.5, MTF gate ≠ BEAR.
    Bear trigger : close < ORL-15, RVOL ≥ 1.5, MTF gate ≠ BULL.

    Entry  = current price.
    Stop   = ORL-15 (bull) / ORH-15 (bear).
    Target = entry ± (ORH15 - ORL15) × 1.5.
    """
    try:
        orh  = float(sig.orb15_high)
        orl  = float(sig.orb15_low)
        if orh <= 0 or orl <= 0:
            return None

        bo   = getattr(sig, "orb15_breakout", "NONE")
        rvol = float(getattr(sig, "rel_volume", 1.0))
        gate = getattr(sig, "short_tf_alignment", "MIXED")
        price = float(sig.price)
        or_range = orh - orl

        if bo == "BULL" and rvol >= 1.5 and gate != "BEAR":
            entry  = price
            stop   = orl
            target = entry + or_range * 1.5
            conf   = min(95, 60 + (rvol - 1.5) * 10 + (10 if gate == "BULL" else 0))
            reason = (
                f"ORB-15 bull breakout: price {price:.2f} > ORH {orh:.2f}; "
                f"RVOL {rvol:.1f}x; TF gate {gate}"
            )
            return AlgoResult(
                algo="ORB15_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )

        if bo == "BEAR" and rvol >= 1.5 and gate != "BULL":
            entry  = price
            stop   = orh
            target = entry - or_range * 1.5
            conf   = min(95, 60 + (rvol - 1.5) * 10 + (10 if gate == "BEAR" else 0))
            reason = (
                f"ORB-15 bear breakdown: price {price:.2f} < ORL {orl:.2f}; "
                f"RVOL {rvol:.1f}x; TF gate {gate}"
            )
            return AlgoResult(
                algo="ORB15_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )
    except Exception as exc:
        logger.debug("eval_orb15 error: %s", exc)
    return None


# ── Algo 6: Gap-and-Go ───────────────────────────────────────────────────────

def eval_gap_and_go(sig) -> Optional[AlgoResult]:
    """
    Gap-and-Go (Algo 6).

    Trigger: gap_pct ≥ +2% (or ≤ -2%), color confirms (GREEN for gap-up,
    RED for gap-down), RVOL ≥ 2x, MTF gate ≠ opposing direction.

    Entry  = current price (momentum continuation).
    Stop   = session_low (gap-up) or session_high (gap-down).
    Target = entry ± abs(gap) × 1.0  (match the gap distance again).
    """
    try:
        gap_pct  = float(getattr(sig, "gap_pct",  0.0))
        gap_type = getattr(sig, "gap_type", "FLAT")
        color    = getattr(sig, "color_vs_prev_close", "FLAT")
        rvol     = float(getattr(sig, "rel_volume", 1.0))
        gate     = getattr(sig, "short_tf_alignment", "MIXED")
        price    = float(sig.price)
        sess_low  = float(getattr(sig, "session_low",  0.0))
        sess_high = float(getattr(sig, "session_high", 0.0))
        today_open = float(getattr(sig, "today_open", price))

        abs_gap_pts = abs(gap_pct / 100.0 * today_open) if today_open > 0 else 0.0

        if gap_type == "GAP_UP" and gap_pct >= 2.0 and color == "GREEN" and rvol >= 2.0 and gate != "BEAR":
            entry  = price
            stop   = sess_low if sess_low > 0 else price * 0.98
            target = entry + abs_gap_pts
            conf   = min(95, 55 + (gap_pct - 2.0) * 5 + (rvol - 2.0) * 5 + (10 if gate == "BULL" else 0))
            reason = (
                f"Gap-and-Go bull: gap {gap_pct:+.1f}%, RVOL {rvol:.1f}x, "
                f"color {color}, TF {gate}"
            )
            return AlgoResult(
                algo="GAP_AND_GO_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )

        if gap_type == "GAP_DOWN" and gap_pct <= -2.0 and color == "RED" and rvol >= 2.0 and gate != "BULL":
            entry  = price
            stop   = sess_high if sess_high > 0 else price * 1.02
            target = entry - abs_gap_pts
            conf   = min(95, 55 + (abs(gap_pct) - 2.0) * 5 + (rvol - 2.0) * 5 + (10 if gate == "BEAR" else 0))
            reason = (
                f"Gap-and-Go bear: gap {gap_pct:+.1f}%, RVOL {rvol:.1f}x, "
                f"color {color}, TF {gate}"
            )
            return AlgoResult(
                algo="GAP_AND_GO_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )
    except Exception as exc:
        logger.debug("eval_gap_and_go error: %s", exc)
    return None


# ── Algo 7: Gap Fade ──────────────────────────────────────────────────────────

def eval_gap_fade(sig) -> Optional[AlgoResult]:
    """
    Gap Fade (Algo 7) — counter-trend gap fill trade.

    Trigger (gap-up fade): gap_pct ≥ +1.5%, price < today_open (reverting),
    RVOL ≥ 1.5x, MTF gate ≠ BULL (not strongly trending up).

    Trigger (gap-down fade): gap_pct ≤ -1.5%, price > today_open (reverting),
    RVOL ≥ 1.5x, MTF gate ≠ BEAR.

    Entry  = current price.
    Stop   = session HOD (fade-up) or session LOD (fade-down).
    Target = prev_close (full gap fill).
    """
    try:
        gap_pct     = float(getattr(sig, "gap_pct",  0.0))
        gap_type    = getattr(sig, "gap_type", "FLAT")
        rvol        = float(getattr(sig, "rel_volume", 1.0))
        gate        = getattr(sig, "short_tf_alignment", "MIXED")
        price       = float(sig.price)
        today_open  = float(getattr(sig, "today_open", price))
        prev_close  = float(getattr(sig, "prev_day_close", 0.0))
        sess_high   = float(getattr(sig, "session_high", 0.0))
        sess_low    = float(getattr(sig, "session_low",  0.0))

        if today_open <= 0 or prev_close <= 0:
            return None

        if (gap_type == "GAP_UP" and gap_pct >= 1.5 and price < today_open
                and rvol >= 1.5 and gate != "BULL"):
            entry  = price
            stop   = sess_high if sess_high > price else price * 1.01
            target = prev_close
            if target >= entry:
                return None  # price already at/below prev_close — fade already done
            conf = min(90, 50 + (gap_pct - 1.5) * 5 + (rvol - 1.5) * 5)
            reason = (
                f"Gap Fade short: gap {gap_pct:+.1f}%, price {price:.2f} < open "
                f"{today_open:.2f}, RVOL {rvol:.1f}x, TF {gate}"
            )
            return AlgoResult(
                algo="GAP_FADE_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )

        if (gap_type == "GAP_DOWN" and gap_pct <= -1.5 and price > today_open
                and rvol >= 1.5 and gate != "BEAR"):
            entry  = price
            stop   = sess_low if sess_low < price else price * 0.99
            target = prev_close
            if target <= entry:
                return None  # price already at/above prev_close
            conf = min(90, 50 + (abs(gap_pct) - 1.5) * 5 + (rvol - 1.5) * 5)
            reason = (
                f"Gap Fade long: gap {gap_pct:+.1f}%, price {price:.2f} > open "
                f"{today_open:.2f}, RVOL {rvol:.1f}x, TF {gate}"
            )
            return AlgoResult(
                algo="GAP_FADE_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )
    except Exception as exc:
        logger.debug("eval_gap_fade error: %s", exc)
    return None


# ── Algo 8: PDH / PDL Breakout ────────────────────────────────────────────────

def eval_pdh_pdl_breakout(sig) -> Optional[AlgoResult]:
    """
    Prior Day High/Low Breakout (Algo 8).

    Bull trigger : price > PDH, RVOL ≥ 1.5x, MTF gate ≠ BEAR.
    Bear trigger : price < PDL, RVOL ≥ 1.5x, MTF gate ≠ BULL.

    Entry  = current price.
    Stop   = PDH minus 0.5× ATR-proxy (bear) or PDL + 0.5× (bull).
    Target = PDH + (PDH - PDL) × 0.5 (bull) or PDL - (PDH - PDL) × 0.5 (bear).
    """
    try:
        pdh  = float(getattr(sig, "prev_day_high", 0.0))
        pdl  = float(getattr(sig, "prev_day_low",  0.0))
        rvol = float(getattr(sig, "rel_volume", 1.0))
        gate = getattr(sig, "short_tf_alignment", "MIXED")
        price = float(sig.price)

        if pdh <= 0 or pdl <= 0:
            return None
        pd_range = pdh - pdl
        if pd_range <= 0:
            return None

        atr_proxy = pd_range * 0.5

        if price > pdh and rvol >= 1.5 and gate != "BEAR":
            entry  = price
            stop   = pdh - atr_proxy
            target = pdh + pd_range * 0.5
            conf   = min(95, 55 + (rvol - 1.5) * 10 + (10 if gate == "BULL" else 0))
            reason = (
                f"PDH breakout: price {price:.2f} > PDH {pdh:.2f}; "
                f"RVOL {rvol:.1f}x; TF {gate}"
            )
            return AlgoResult(
                algo="PDH_BREAKOUT_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )

        if price < pdl and rvol >= 1.5 and gate != "BULL":
            entry  = price
            stop   = pdl + atr_proxy
            target = pdl - pd_range * 0.5
            conf   = min(95, 55 + (rvol - 1.5) * 10 + (10 if gate == "BEAR" else 0))
            reason = (
                f"PDL breakdown: price {price:.2f} < PDL {pdl:.2f}; "
                f"RVOL {rvol:.1f}x; TF {gate}"
            )
            return AlgoResult(
                algo="PDL_BREAKDOWN_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )
    except Exception as exc:
        logger.debug("eval_pdh_pdl_breakout error: %s", exc)
    return None


# ── Algo 12: HOD / LOD Momentum Break ────────────────────────────────────────

def eval_hod_lod_break(sig) -> Optional[AlgoResult]:
    """
    HOD/LOD Momentum Break (Algo 12).

    Fires when the current bar closes at a NEW session high (bull) or
    session low (bear), RVOL ≥ 1.5x, and TF gate is not opposing.

    Unlike PDH/PDL breakout this catches intraday momentum extensions that
    don't reach prior-day levels.  Requires at least 30 min of session data
    (session_high > orb_high) so we don't fire on the first few bars.

    Entry  = current price.
    Stop   = session_high minus small buffer (bull) / session_low + buffer (bear).
    Target = entry ± session_range × 0.5.
    """
    try:
        sess_high = float(getattr(sig, "session_high", 0.0))
        sess_low  = float(getattr(sig, "session_low",  0.0))
        orb_high  = float(getattr(sig, "orb15_high",   0.0)) or float(getattr(sig, "orb5_high", 0.0))
        rvol      = float(getattr(sig, "rel_volume", 1.0))
        gate      = getattr(sig, "short_tf_alignment", "MIXED")
        price     = float(sig.price)

        if sess_high <= 0 or sess_low <= 0:
            return None
        # Need meaningful session range beyond opening range
        if orb_high > 0 and sess_high <= orb_high * 1.001:
            return None  # still inside opening range, too early

        sess_range = sess_high - sess_low
        if sess_range <= 0:
            return None
        buf = sess_range * 0.05

        # Bull: price == session_high (new HOD)
        if abs(price - sess_high) <= buf and rvol >= 1.5 and gate != "BEAR":
            entry  = price
            stop   = sess_high - buf * 2
            target = entry + sess_range * 0.5
            conf   = min(90, 55 + (rvol - 1.5) * 8 + (10 if gate == "BULL" else 0))
            reason = (
                f"HOD momentum break: new HOD {sess_high:.2f}, "
                f"RVOL {rvol:.1f}x, TF {gate}"
            )
            return AlgoResult(
                algo="HOD_BREAK_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )

        # Bear: price == session_low (new LOD)
        if abs(price - sess_low) <= buf and rvol >= 1.5 and gate != "BULL":
            entry  = price
            stop   = sess_low + buf * 2
            target = entry - sess_range * 0.5
            conf   = min(90, 55 + (rvol - 1.5) * 8 + (10 if gate == "BEAR" else 0))
            reason = (
                f"LOD momentum break: new LOD {sess_low:.2f}, "
                f"RVOL {rvol:.1f}x, TF {gate}"
            )
            return AlgoResult(
                algo="LOD_BREAK_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )
    except Exception as exc:
        logger.debug("eval_hod_lod_break error: %s", exc)
    return None


# ── Flag pattern detection (used by Algos 29 & 30) ───────────────────────────

def detect_flag(df_1m) -> dict:
    """
    Detect bull/bear flag patterns in recent 1-min bars.

    Methodology:
      1. Pole  — last 5 bars before the most recent 5: total move > 0.8%, slope clear.
      2. Flag  — most recent 5 bars: range < 60% of pole range, counter-trend drift.
      3. Breakout — current close outside the flag range.

    Returns:
      bull_flag  : bool
      bear_flag  : bool
      flag_high  : float   — top of the flag consolidation zone
      flag_low   : float   — bottom of the flag consolidation zone
      pole_pct   : float   — % move of the pole (+ = up, - = down)
    """
    out = {"bull_flag": False, "bear_flag": False,
           "flag_high": 0.0, "flag_low": 0.0, "pole_pct": 0.0}
    try:
        import numpy as np
        import pandas as pd

        if df_1m is None or len(df_1m) < 12:
            return out

        closes = df_1m["Close"].values
        highs  = df_1m["High"].values
        lows   = df_1m["Low"].values
        n = len(closes)

        # Pole: bars [n-11 .. n-6] (5 bars), Flag: bars [n-5 .. n-1] (5 bars)
        pole_closes  = closes[n - 11 : n - 5]
        flag_closes  = closes[n - 6 : n]
        flag_highs   = highs[n - 6 : n]
        flag_lows    = lows[n - 6 : n]

        if len(pole_closes) < 5 or len(flag_closes) < 5:
            return out

        pole_start = float(pole_closes[0])
        pole_end   = float(pole_closes[-1])
        pole_pct   = (pole_end - pole_start) / pole_start * 100 if pole_start > 0 else 0.0
        pole_range = abs(pole_end - pole_start)

        flag_high  = float(np.max(flag_highs))
        flag_low   = float(np.min(flag_lows))
        flag_range = flag_high - flag_low
        current    = float(closes[-1])

        # Flag range must be tighter than pole; slight counter-trend or flat drift
        if pole_range <= 0 or flag_range <= 0:
            return out

        tight = flag_range < pole_range * 0.60

        # Bull flag: strong up pole, then slight downward/flat flag, breakout above flag_high
        if pole_pct >= 0.8 and tight:
            flag_slope = float(np.polyfit(range(len(flag_closes)), flag_closes, 1)[0])
            # Slope slightly negative or flat (counter-trend pullback)
            if flag_slope <= pole_range * 0.05 and current >= flag_high * 0.998:
                out["bull_flag"] = True
                out["flag_high"] = round(flag_high, 4)
                out["flag_low"]  = round(flag_low,  4)
                out["pole_pct"]  = round(pole_pct, 3)
                return out

        # Bear flag: strong down pole, then slight upward/flat flag, breakdown below flag_low
        if pole_pct <= -0.8 and tight:
            flag_slope = float(np.polyfit(range(len(flag_closes)), flag_closes, 1)[0])
            if flag_slope >= -pole_range * 0.05 and current <= flag_low * 1.002:
                out["bear_flag"] = True
                out["flag_high"] = round(flag_high, 4)
                out["flag_low"]  = round(flag_low,  4)
                out["pole_pct"]  = round(pole_pct, 3)
                return out
    except Exception as exc:
        logger.debug("detect_flag error: %s", exc)
    return out


# ── Algo 29: Bull Flag Continuation ──────────────────────────────────────────

def eval_bull_flag(sig) -> Optional[AlgoResult]:
    """
    Bull Flag Continuation (Algo 29).

    Pre-computed by detect_flag() → stored in sig.bull_flag / sig.flag_high / etc.

    Entry  = current price (breakout above flag_high confirmed on close).
    Stop   = flag_low.
    Target = entry + pole_range * 1.0 (measured move = repeat the pole).
    """
    try:
        if not bool(getattr(sig, "bull_flag", False)):
            return None
        flag_high = float(getattr(sig, "flag_high", 0.0))
        flag_low  = float(getattr(sig, "flag_low",  0.0))
        pole_pct  = float(getattr(sig, "pole_pct",  0.0))
        rvol      = float(getattr(sig, "rel_volume", 1.0))
        gate      = getattr(sig, "short_tf_alignment", "MIXED")
        price     = float(sig.price)

        if flag_high <= 0 or flag_low <= 0:
            return None
        if gate == "BEAR":
            return None

        pole_pts  = price * abs(pole_pct) / 100
        entry  = price
        stop   = flag_low
        target = entry + pole_pts
        conf   = min(90, 55 + abs(pole_pct) * 3 + (rvol - 1.0) * 5 + (10 if gate == "BULL" else 0))
        reason = (
            f"Bull flag breakout: pole {pole_pct:+.1f}%, flag {flag_low:.2f}–{flag_high:.2f}, "
            f"RVOL {rvol:.1f}x, TF {gate}"
        )
        return AlgoResult(
            algo="BULL_FLAG", direction="BUY",
            confidence=round(conf, 1),
            entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
            rr=_rr(entry, stop, target), reason=reason,
        )
    except Exception as exc:
        logger.debug("eval_bull_flag error: %s", exc)
    return None


# ── Algo 30: Bear Flag Continuation ──────────────────────────────────────────

def eval_bear_flag(sig) -> Optional[AlgoResult]:
    """
    Bear Flag Continuation (Algo 30).

    Entry  = current price (breakdown below flag_low confirmed on close).
    Stop   = flag_high.
    Target = entry - pole_range * 1.0.
    """
    try:
        if not bool(getattr(sig, "bear_flag", False)):
            return None
        flag_high = float(getattr(sig, "flag_high", 0.0))
        flag_low  = float(getattr(sig, "flag_low",  0.0))
        pole_pct  = float(getattr(sig, "pole_pct",  0.0))
        rvol      = float(getattr(sig, "rel_volume", 1.0))
        gate      = getattr(sig, "short_tf_alignment", "MIXED")
        price     = float(sig.price)

        if flag_high <= 0 or flag_low <= 0:
            return None
        if gate == "BULL":
            return None

        pole_pts  = price * abs(pole_pct) / 100
        entry  = price
        stop   = flag_high
        target = entry - pole_pts
        conf   = min(90, 55 + abs(pole_pct) * 3 + (rvol - 1.0) * 5 + (10 if gate == "BEAR" else 0))
        reason = (
            f"Bear flag breakdown: pole {pole_pct:+.1f}%, flag {flag_low:.2f}–{flag_high:.2f}, "
            f"RVOL {rvol:.1f}x, TF {gate}"
        )
        return AlgoResult(
            algo="BEAR_FLAG", direction="SELL",
            confidence=round(conf, 1),
            entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
            rr=_rr(entry, stop, target), reason=reason,
        )
    except Exception as exc:
        logger.debug("eval_bear_flag error: %s", exc)
    return None


# ── Algo 3: VWAP Touch Scalp (Reclaim / Rejection) ───────────────────────────

def eval_vwap_touch_scalp(sig) -> Optional[AlgoResult]:
    """
    VWAP Touch Scalp (Algo 3 / Phase 2.7).

    Fires on a VWAP RECLAIM (price crosses above from below) or REJECTION
    (price crosses below from above) with RVOL ≥ 1.3 confirming institutional
    participation.  Target is the first sigma band in the direction of the cross.

    Entry  = current price.
    Stop   = VWAP ∓ 40% of the σ-band half-width.
    Target = vwap_upper_1 (RECLAIM) / vwap_lower_1 (REJECTION).
    """
    try:
        event   = getattr(sig, "vwap_event", "FLAT")
        vwap    = float(getattr(sig, "vwap_price", 0.0))
        upper_1 = float(getattr(sig, "vwap_upper_1", 0.0))
        lower_1 = float(getattr(sig, "vwap_lower_1", 0.0))
        rvol    = float(getattr(sig, "rel_volume", 1.0))
        gate    = getattr(sig, "short_tf_alignment", "MIXED")
        price   = float(sig.price)

        if vwap <= 0 or rvol < 1.3:
            return None

        half_band = (upper_1 - lower_1) / 2 if upper_1 > lower_1 else vwap * 0.003
        stop_buf  = half_band * 0.4

        if event == "RECLAIM" and gate != "BEAR":
            entry  = price
            stop   = vwap - stop_buf
            target = upper_1 if upper_1 > price else price + half_band
            if stop >= entry or target <= entry:
                return None
            conf = min(88, 55 + (rvol - 1.3) * 12 + (8 if gate == "BULL" else 0))
            reason = (
                f"VWAP RECLAIM scalp: price {price:.2f} reclaimed VWAP {vwap:.2f}; "
                f"target σ1 {target:.2f}; RVOL {rvol:.1f}x"
            )
            return AlgoResult(
                algo="VWAP_TOUCH_SCALP_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )

        if event == "REJECTION" and gate != "BULL":
            entry  = price
            stop   = vwap + stop_buf
            target = lower_1 if lower_1 < price else price - half_band
            if stop <= entry or target >= entry:
                return None
            conf = min(88, 55 + (rvol - 1.3) * 12 + (8 if gate == "BEAR" else 0))
            reason = (
                f"VWAP REJECTION scalp: price {price:.2f} rejected off VWAP {vwap:.2f}; "
                f"target σ1 {target:.2f}; RVOL {rvol:.1f}x"
            )
            return AlgoResult(
                algo="VWAP_TOUCH_SCALP_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )
    except Exception as exc:
        logger.debug("eval_vwap_touch_scalp error: %s", exc)
    return None


# ── Algo 4: VWAP HOD Scalp ────────────────────────────────────────────────────

def eval_vwap_hod_scalp(sig) -> Optional[AlgoResult]:
    """
    VWAP HOD Scalp (Algo 4 / Phase 2.8) — bullish pullback-to-VWAP continuation.

    In a bullish trend (MTF gate BULL), price dips to test VWAP and bounces.
    This captures the dip-and-resume pattern, targeting a new session HOD.

    Trigger:
      - vwap_event in {ABOVE, RECLAIM, AT_1SD_UP} (price above or at VWAP)
      - z_score between −0.5 and +1.0 (near VWAP, not extended)
      - MTF gate BULL
      - RVOL ≥ 1.2

    Entry  = current price.
    Stop   = vwap_lower_1 (−1σ — structure broken if price falls through).
    Target = session_high (HOD extension).
    """
    try:
        event     = getattr(sig, "vwap_event", "FLAT")
        vwap      = float(getattr(sig, "vwap_price", 0.0))
        z_score   = float(getattr(sig, "vwap_z_score", 0.0))
        lower_1   = float(getattr(sig, "vwap_lower_1", 0.0))
        sess_high = float(getattr(sig, "session_high", 0.0))
        rvol      = float(getattr(sig, "rel_volume", 1.0))
        gate      = getattr(sig, "short_tf_alignment", "MIXED")
        price     = float(sig.price)

        if vwap <= 0 or sess_high <= 0:
            return None
        if event not in ("ABOVE", "RECLAIM", "AT_1SD_UP"):
            return None
        if not (-0.5 <= z_score <= 1.0):
            return None
        if gate != "BULL" or rvol < 1.2:
            return None
        if sess_high <= price:
            return None  # already at HOD — no room to target

        entry  = price
        stop   = lower_1 if lower_1 > 0 and lower_1 < price else vwap * 0.997
        target = sess_high
        if stop >= entry:
            return None

        conf = min(88, 50 + (rvol - 1.2) * 10 + z_score * 5)
        reason = (
            f"VWAP HOD scalp: z={z_score:.2f}, price {price:.2f} near VWAP {vwap:.2f}; "
            f"target HOD {sess_high:.2f}; RVOL {rvol:.1f}x"
        )
        return AlgoResult(
            algo="VWAP_HOD_SCALP", direction="BUY",
            confidence=round(conf, 1),
            entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
            rr=_rr(entry, stop, target), reason=reason,
        )
    except Exception as exc:
        logger.debug("eval_vwap_hod_scalp error: %s", exc)
    return None


# ── Algo 5: VWAP LOD Scalp ────────────────────────────────────────────────────

def eval_vwap_lod_scalp(sig) -> Optional[AlgoResult]:
    """
    VWAP LOD Scalp (Algo 5 / Phase 2.9) — bearish bounce-to-VWAP continuation.

    Mirror of VWAP HOD Scalp for the bear side.  Price below VWAP, bounces
    up to test VWAP from below, fails in the REJECTION zone, LOD as target.

    Trigger:
      - vwap_event in {BELOW, REJECTION, AT_1SD_DOWN}
      - z_score between −1.0 and +0.5
      - MTF gate BEAR
      - RVOL ≥ 1.2

    Entry  = current price.
    Stop   = vwap_upper_1 (+1σ).
    Target = session_low (LOD extension).
    """
    try:
        event    = getattr(sig, "vwap_event", "FLAT")
        vwap     = float(getattr(sig, "vwap_price", 0.0))
        z_score  = float(getattr(sig, "vwap_z_score", 0.0))
        upper_1  = float(getattr(sig, "vwap_upper_1", 0.0))
        sess_low = float(getattr(sig, "session_low", 0.0))
        rvol     = float(getattr(sig, "rel_volume", 1.0))
        gate     = getattr(sig, "short_tf_alignment", "MIXED")
        price    = float(sig.price)

        if vwap <= 0 or sess_low <= 0:
            return None
        if event not in ("BELOW", "REJECTION", "AT_1SD_DOWN"):
            return None
        if not (-1.0 <= z_score <= 0.5):
            return None
        if gate != "BEAR" or rvol < 1.2:
            return None
        if sess_low >= price:
            return None  # already at LOD

        entry  = price
        stop   = upper_1 if upper_1 > 0 and upper_1 > price else vwap * 1.003
        target = sess_low
        if stop <= entry or target >= entry:
            return None

        conf = min(88, 50 + (rvol - 1.2) * 10 + abs(z_score) * 5)
        reason = (
            f"VWAP LOD scalp: z={z_score:.2f}, price {price:.2f} near VWAP {vwap:.2f}; "
            f"target LOD {sess_low:.2f}; RVOL {rvol:.1f}x"
        )
        return AlgoResult(
            algo="VWAP_LOD_SCALP", direction="SELL",
            confidence=round(conf, 1),
            entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
            rr=_rr(entry, stop, target), reason=reason,
        )
    except Exception as exc:
        logger.debug("eval_vwap_lod_scalp error: %s", exc)
    return None


# ── Algo 9: Level Rejection Scalp ─────────────────────────────────────────────

def eval_level_rejection_scalp(sig) -> Optional[AlgoResult]:
    """
    Level Rejection Scalp (Algo 9 / Phase 2.10).

    Fires when price reaches the ±2σ VWAP bands (statistically extreme
    intraday extension) and the VWAP event confirms the overshoot.
    Mean-reversion trade targeting VWAP as the exit.

    Bull (AT_2SD_DOWN — oversold):
      Entry  = current price
      Stop   = lower_2 − 5% of the 2σ width (buffer)
      Target = vwap_price

    Bear (AT_2SD_UP — overbought):
      Entry  = current price
      Stop   = upper_2 + 5% buffer
      Target = vwap_price
    """
    try:
        event   = getattr(sig, "vwap_event", "FLAT")
        vwap    = float(getattr(sig, "vwap_price", 0.0))
        upper_2 = float(getattr(sig, "vwap_upper_2", 0.0))
        lower_2 = float(getattr(sig, "vwap_lower_2", 0.0))
        rvol    = float(getattr(sig, "rel_volume", 1.0))
        gate    = getattr(sig, "short_tf_alignment", "MIXED")
        price   = float(sig.price)

        if vwap <= 0 or upper_2 <= 0 or lower_2 <= 0 or rvol < 1.2:
            return None

        buf = (upper_2 - lower_2) * 0.05

        if event == "AT_2SD_DOWN" and gate != "BEAR":
            entry  = price
            stop   = lower_2 - buf
            target = vwap
            if stop >= entry or target <= entry:
                return None
            conf = min(85, 52 + (rvol - 1.2) * 10 + (8 if gate == "BULL" else 0))
            reason = (
                f"Level rejection scalp bull: price {price:.2f} at −2σ {lower_2:.2f}; "
                f"target VWAP {vwap:.2f}; RVOL {rvol:.1f}x"
            )
            return AlgoResult(
                algo="LEVEL_REJECTION_SCALP_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )

        if event == "AT_2SD_UP" and gate != "BULL":
            entry  = price
            stop   = upper_2 + buf
            target = vwap
            if stop <= entry or target >= entry:
                return None
            conf = min(85, 52 + (rvol - 1.2) * 10 + (8 if gate == "BEAR" else 0))
            reason = (
                f"Level rejection scalp bear: price {price:.2f} at +2σ {upper_2:.2f}; "
                f"target VWAP {vwap:.2f}; RVOL {rvol:.1f}x"
            )
            return AlgoResult(
                algo="LEVEL_REJECTION_SCALP_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )
    except Exception as exc:
        logger.debug("eval_level_rejection_scalp error: %s", exc)
    return None


# ── Algo 11: Micro Pullback Scalp ─────────────────────────────────────────────

def eval_micro_pullback_scalp(sig) -> Optional[AlgoResult]:
    """
    Micro Pullback Scalp (Algo 11 / Phase 2.11) — VWAP-supported trend continuation.

    In a strongly directional market (MTF gate BULL or BEAR), a small pullback
    to just above/below VWAP creates a low-risk continuation entry.

    Bull: event in {ABOVE, RECLAIM, AT_1SD_UP}, z_score 0.1–1.2, gate BULL.
      Entry  = current price
      Stop   = VWAP × 0.999 (just below VWAP — structure broken)
      Target = vwap_upper_1 (first σ extension)

    Bear: event in {BELOW, REJECTION, AT_1SD_DOWN}, z_score −1.2–(−0.1), gate BEAR.
      Entry  = current price
      Stop   = VWAP × 1.001
      Target = vwap_lower_1
    """
    try:
        event    = getattr(sig, "vwap_event", "FLAT")
        vwap     = float(getattr(sig, "vwap_price", 0.0))
        z_score  = float(getattr(sig, "vwap_z_score", 0.0))
        upper_1  = float(getattr(sig, "vwap_upper_1", 0.0))
        lower_1  = float(getattr(sig, "vwap_lower_1", 0.0))
        rvol     = float(getattr(sig, "rel_volume", 1.0))
        gate     = getattr(sig, "short_tf_alignment", "MIXED")
        mtf_gate = bool(getattr(sig, "mtf_gate_passed", False))
        price    = float(sig.price)

        if vwap <= 0 or rvol < 1.3 or gate == "MIXED":
            return None

        bull_events = {"ABOVE", "RECLAIM", "AT_1SD_UP"}
        bear_events = {"BELOW", "REJECTION", "AT_1SD_DOWN"}

        if gate == "BULL" and event in bull_events and 0.1 <= z_score <= 1.2:
            if upper_1 <= 0 or upper_1 <= price:
                return None
            entry  = price
            stop   = vwap * 0.999
            target = upper_1
            if stop >= entry or target <= entry:
                return None
            conf = min(85, 48 + (rvol - 1.3) * 10 + z_score * 6 + (10 if mtf_gate else 0))
            reason = (
                f"Micro pullback scalp bull: z={z_score:.2f}, VWAP {vwap:.2f}, "
                f"target σ1 {upper_1:.2f}; RVOL {rvol:.1f}x; TF {gate}"
            )
            return AlgoResult(
                algo="MICRO_PULLBACK_SCALP_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )

        if gate == "BEAR" and event in bear_events and -1.2 <= z_score <= -0.1:
            if lower_1 <= 0 or lower_1 >= price:
                return None
            entry  = price
            stop   = vwap * 1.001
            target = lower_1
            if stop <= entry or target >= entry:
                return None
            conf = min(85, 48 + (rvol - 1.3) * 10 + abs(z_score) * 6 + (10 if mtf_gate else 0))
            reason = (
                f"Micro pullback scalp bear: z={z_score:.2f}, VWAP {vwap:.2f}, "
                f"target σ1 {lower_1:.2f}; RVOL {rvol:.1f}x; TF {gate}"
            )
            return AlgoResult(
                algo="MICRO_PULLBACK_SCALP_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )
    except Exception as exc:
        logger.debug("eval_micro_pullback_scalp error: %s", exc)
    return None


# ── Registry ──────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Relative Value Algorithms
# ══════════════════════════════════════════════════════════════════════════════

# ── Algo 13: SPY Beta Catch-Up ────────────────────────────────────────────────

def eval_spy_beta_catchup(sig) -> Optional[AlgoResult]:
    """
    SPY Beta Catch-Up (Algo 13 / Phase 3.1).

    When the stock's sector is making a significant intraday move but the stock
    has barely moved relative to it (rs_ratio < 0.55), the stock is expected to
    catch up to its implied beta-adjusted level.

    Bull: sector_change > +0.5%, stock also positive, rs_ratio < 0.55.
    Bear: sector_change < −0.5%, stock also negative, rs_ratio < 0.55.

    Entry  = current price.
    Stop   = vwap_lower_1 (bull) / vwap_upper_1 (bear).
    Target = price × (1 + sector_change × 0.5 / 100) — half the sector move.
    """
    try:
        rs_ratio      = float(getattr(sig, "rs_ratio", 1.0))
        rs_label      = getattr(sig, "rs_label", "IN_LINE")
        sector_change = float(getattr(sig, "sector_change", 0.0))
        change_pct    = float(getattr(sig, "change_pct", 0.0))
        rvol          = float(getattr(sig, "rel_volume", 1.0))
        regime        = getattr(sig, "regime", "NEUTRAL")
        price         = float(sig.price)
        upper_1       = float(getattr(sig, "vwap_upper_1", 0.0))
        lower_1       = float(getattr(sig, "vwap_lower_1", 0.0))
        vwap          = float(getattr(sig, "vwap_price", 0.0))

        if rs_ratio <= 0 or rs_label == "COUNTER" or rvol < 1.1:
            return None
        if 0.55 <= rs_ratio <= 1.45:
            return None  # not enough lag to trade

        # Bull catch-up: market/sector up, stock lagging
        if (sector_change > 0.5 and change_pct > 0 and rs_ratio < 0.55
                and regime != "BEAR_TREND"):
            entry  = price
            stop   = lower_1 if lower_1 > 0 and lower_1 < price else vwap * 0.997
            target = price * (1 + sector_change * 0.5 / 100)
            if stop >= entry or target <= entry:
                return None
            conf = min(85, 45 + (sector_change - 0.5) * 8 + (rvol - 1.1) * 8)
            reason = (
                f"SPY beta catch-up bull: sector {sector_change:+.2f}%, "
                f"stock only {change_pct:+.2f}%, rs_ratio {rs_ratio:.2f}"
            )
            return AlgoResult(
                algo="SPY_BETA_CATCHUP_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )

        # Bear catch-up: market/sector down, stock not falling enough yet
        if (sector_change < -0.5 and change_pct < 0 and rs_ratio < 0.55
                and regime == "BEAR_TREND"):
            entry  = price
            stop   = upper_1 if upper_1 > 0 and upper_1 > price else vwap * 1.003
            target = price * (1 + sector_change * 0.5 / 100)  # negative, so price falls
            if stop <= entry or target >= entry:
                return None
            conf = min(85, 45 + (abs(sector_change) - 0.5) * 8 + (rvol - 1.1) * 8)
            reason = (
                f"SPY beta catch-up bear: sector {sector_change:+.2f}%, "
                f"stock only {change_pct:+.2f}%, rs_ratio {rs_ratio:.2f}"
            )
            return AlgoResult(
                algo="SPY_BETA_CATCHUP_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )
    except Exception as exc:
        logger.debug("eval_spy_beta_catchup error: %s", exc)
    return None


# ── Algo 14: Residual Momentum ────────────────────────────────────────────────

def eval_residual_momentum(sig) -> Optional[AlgoResult]:
    """
    Residual Momentum (Algo 14 / Phase 3.2).

    Stock strongly outperforming SPY (rs_ratio ≥ 1.5, rs_label LEADING),
    sector not a headwind, MTF gate supportive.  Ride the relative strength
    leader — positive residual return tends to persist intraday.

    Entry  = current price.
    Stop   = vwap_lower_1 (price falling back through VWAP invalidates the thesis).
    Target = entry + (rs_ratio − 1.0) × (session_high − session_low) × 0.4.
    """
    try:
        rs_ratio      = float(getattr(sig, "rs_ratio", 1.0))
        rs_label      = getattr(sig, "rs_label", "IN_LINE")
        sector_trend  = getattr(sig, "sector_trend", "NEUTRAL")
        gate          = getattr(sig, "short_tf_alignment", "MIXED")
        rvol          = float(getattr(sig, "rel_volume", 1.0))
        price         = float(sig.price)
        vwap          = float(getattr(sig, "vwap_price", 0.0))
        lower_1       = float(getattr(sig, "vwap_lower_1", 0.0))
        sess_high     = float(getattr(sig, "session_high", 0.0))
        sess_low      = float(getattr(sig, "session_low", 0.0))

        if rs_ratio < 1.5 or rs_label != "LEADING":
            return None
        if sector_trend == "BEARISH" or gate == "BEAR":
            return None
        if rvol < 1.2 or (vwap > 0 and price <= vwap):
            return None  # must be above VWAP to confirm leadership

        sess_range = (sess_high - sess_low) if sess_high > sess_low else price * 0.02
        entry  = price
        stop   = lower_1 if lower_1 > 0 and lower_1 < price else vwap * 0.997
        target = entry + (rs_ratio - 1.0) * sess_range * 0.4
        if stop >= entry or target <= entry:
            return None

        conf = min(90, 52 + (rs_ratio - 1.5) * 8 + (8 if gate == "BULL" else 0)
                   + (6 if sector_trend == "BULLISH" else 0))
        reason = (
            f"Residual momentum bull: rs_ratio={rs_ratio:.2f} ({rs_label}), "
            f"sector {sector_trend}, RVOL {rvol:.1f}x, TF {gate}"
        )
        return AlgoResult(
            algo="RESIDUAL_MOMENTUM_BULL", direction="BUY",
            confidence=round(conf, 1),
            entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
            rr=_rr(entry, stop, target), reason=reason,
        )
    except Exception as exc:
        logger.debug("eval_residual_momentum error: %s", exc)
    return None


# ── Algo 15: Residual Reversion ────────────────────────────────────────────────

def eval_residual_reversion(sig) -> Optional[AlgoResult]:
    """
    Residual Reversion (Algo 15 / Phase 3.3).

    Extreme outperformance vs SPY (rs_ratio > 2.5) combined with price well
    above VWAP (z_score > 1.5) and no genuine sector tailwind → fade the
    extreme outperformer, targeting mean reversion back to VWAP.

    Entry  = current price.
    Stop   = vwap_upper_2 + 5% buffer (above the 2σ band).
    Target = vwap_price (VWAP reversion).
    """
    try:
        rs_ratio     = float(getattr(sig, "rs_ratio", 1.0))
        rs_label     = getattr(sig, "rs_label", "IN_LINE")
        vwap_z       = float(getattr(sig, "vwap_z_score", 0.0))
        sector_trend = getattr(sig, "sector_trend", "NEUTRAL")
        vwap         = float(getattr(sig, "vwap_price", 0.0))
        upper_2      = float(getattr(sig, "vwap_upper_2", 0.0))
        rvol         = float(getattr(sig, "rel_volume", 1.0))
        gate         = getattr(sig, "short_tf_alignment", "MIXED")
        price        = float(sig.price)

        if rs_ratio <= 2.5 or rs_label != "LEADING":
            return None
        if vwap_z <= 1.5:
            return None  # not extended enough
        if sector_trend == "BULLISH":
            return None  # don't fade genuine sector momentum
        if gate == "BULL" or rvol < 1.1:
            return None
        if vwap <= 0 or price <= vwap:
            return None

        buf    = (upper_2 - vwap) * 0.05 if upper_2 > vwap else vwap * 0.005
        entry  = price
        stop   = upper_2 + buf if upper_2 > price else price * 1.008
        target = vwap
        if stop <= entry or target >= entry:
            return None

        conf = min(85, 50 + (rs_ratio - 2.5) * 5 + (vwap_z - 1.5) * 6)
        reason = (
            f"Residual reversion bear: rs_ratio={rs_ratio:.2f}, z={vwap_z:.2f}, "
            f"sector {sector_trend}, TF {gate}"
        )
        return AlgoResult(
            algo="RESIDUAL_REVERSION_BEAR", direction="SELL",
            confidence=round(conf, 1),
            entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
            rr=_rr(entry, stop, target), reason=reason,
        )
    except Exception as exc:
        logger.debug("eval_residual_reversion error: %s", exc)
    return None


# ── Algo 16: Sector Leader Momentum ───────────────────────────────────────────

def eval_sector_leader(sig) -> Optional[AlgoResult]:
    """
    Sector Leader Momentum (Algo 16 / Phase 3.4).

    Stock is outperforming its own sector ETF (stock_vs_sector == LEADING)
    in a bullish sector (sector_trend == BULLISH).  Best stock in a hot
    sector → ride sector leadership.

    Entry  = current price (above VWAP).
    Stop   = vwap_lower_1.
    Target = session_high extension (entry + session_range × 0.4).
    """
    try:
        stock_vs_sec  = getattr(sig, "stock_vs_sector", "IN_LINE")
        sector_trend  = getattr(sig, "sector_trend", "NEUTRAL")
        sector_change = float(getattr(sig, "sector_change", 0.0))
        rvol          = float(getattr(sig, "rel_volume", 1.0))
        gate          = getattr(sig, "short_tf_alignment", "MIXED")
        price         = float(sig.price)
        vwap          = float(getattr(sig, "vwap_price", 0.0))
        lower_1       = float(getattr(sig, "vwap_lower_1", 0.0))
        sess_high     = float(getattr(sig, "session_high", 0.0))
        sess_low      = float(getattr(sig, "session_low", 0.0))

        if stock_vs_sec != "LEADING" or sector_trend != "BULLISH":
            return None
        if gate == "BEAR" or rvol < 1.2:
            return None
        if vwap > 0 and price <= vwap:
            return None

        sess_range = (sess_high - sess_low) if sess_high > sess_low else price * 0.02
        entry  = price
        stop   = lower_1 if lower_1 > 0 and lower_1 < price else vwap * 0.997
        target = entry + sess_range * 0.4
        if stop >= entry or target <= entry:
            return None

        conf = min(90, 55 + sector_change * 4 + (rvol - 1.2) * 8
                   + (8 if gate == "BULL" else 0))
        reason = (
            f"Sector leader: stock LEADING {getattr(sig, 'sector_etf', 'ETF')} "
            f"(sector {sector_trend} {sector_change:+.2f}%), RVOL {rvol:.1f}x"
        )
        return AlgoResult(
            algo="SECTOR_LEADER_BULL", direction="BUY",
            confidence=round(conf, 1),
            entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
            rr=_rr(entry, stop, target), reason=reason,
        )
    except Exception as exc:
        logger.debug("eval_sector_leader error: %s", exc)
    return None


# ── Algo 17: Sector Laggard Catch-Up ──────────────────────────────────────────

def eval_sector_laggard_catchup(sig) -> Optional[AlgoResult]:
    """
    Sector Laggard Catch-Up (Algo 17 / Phase 3.5).

    Stock is lagging its sector ETF while the sector has a bullish tailwind.
    The sector drag will eventually pull the stock up — buy the laggard.

    Trigger:
      - stock_vs_sector == LAGGING, sector_trend == BULLISH
      - rs_label ≠ COUNTER (stock isn't actively falling; it's just slow)
      - price near or below VWAP (the laggard hasn't broken out yet)
      - sector_change > 0.3% (sector already moving)

    Entry  = current price.
    Stop   = vwap_lower_1.
    Target = price × (1 + sector_change × 0.7 / 100) — 70% of sector's move.
    """
    try:
        stock_vs_sec  = getattr(sig, "stock_vs_sector", "IN_LINE")
        sector_trend  = getattr(sig, "sector_trend", "NEUTRAL")
        sector_change = float(getattr(sig, "sector_change", 0.0))
        rs_label      = getattr(sig, "rs_label", "IN_LINE")
        rvol          = float(getattr(sig, "rel_volume", 1.0))
        gate          = getattr(sig, "short_tf_alignment", "MIXED")
        price         = float(sig.price)
        vwap          = float(getattr(sig, "vwap_price", 0.0))
        lower_1       = float(getattr(sig, "vwap_lower_1", 0.0))

        if stock_vs_sec != "LAGGING" or sector_trend != "BULLISH":
            return None
        if rs_label == "COUNTER":
            return None  # actively moving against market — not a laggard, a fighter
        if sector_change < 0.3 or rvol < 1.1 or gate == "BEAR":
            return None

        entry  = price
        stop   = lower_1 if lower_1 > 0 and lower_1 < price else vwap * 0.997 if vwap > 0 else price * 0.985
        target = price * (1 + sector_change * 0.7 / 100)
        if stop >= entry or target <= entry:
            return None

        conf = min(85, 45 + sector_change * 6 + (rvol - 1.1) * 8
                   + (6 if gate == "BULL" else 0))
        reason = (
            f"Sector laggard catch-up: LAGGING {getattr(sig, 'sector_etf', 'ETF')} "
            f"(sector {sector_trend} {sector_change:+.2f}%), rs {rs_label}, RVOL {rvol:.1f}x"
        )
        return AlgoResult(
            algo="SECTOR_LAGGARD_CATCHUP_BULL", direction="BUY",
            confidence=round(conf, 1),
            entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
            rr=_rr(entry, stop, target), reason=reason,
        )
    except Exception as exc:
        logger.debug("eval_sector_laggard_catchup error: %s", exc)
    return None


# ── Algo 18: Sector Counter Fade ──────────────────────────────────────────────

def eval_sector_counter_fade(sig) -> Optional[AlgoResult]:
    """
    Sector Counter Fade (Algo 18 / Phase 3.6).

    Stock is "leading" its sector by going up while the sector is in a
    BEARISH trend — the stock is fighting sector headwinds.  Historically
    sector gravity wins; fade the outlier back into the sector trend.

    Trigger:
      - stock_vs_sector == LEADING (stock outperforming sector)
      - sector_trend == BEARISH (sector falling)
      - price above VWAP (the divergence is visible above VWAP)
      - MTF gate ≠ strongly BULL

    Entry  = current price.
    Stop   = vwap_upper_2 or session_high + small buffer.
    Target = vwap_price (gravity back to VWAP).
    """
    try:
        stock_vs_sec  = getattr(sig, "stock_vs_sector", "IN_LINE")
        sector_trend  = getattr(sig, "sector_trend", "NEUTRAL")
        sector_change = float(getattr(sig, "sector_change", 0.0))
        gate          = getattr(sig, "short_tf_alignment", "MIXED")
        rvol          = float(getattr(sig, "rel_volume", 1.0))
        vwap_z        = float(getattr(sig, "vwap_z_score", 0.0))
        price         = float(sig.price)
        vwap          = float(getattr(sig, "vwap_price", 0.0))
        sess_high     = float(getattr(sig, "session_high", 0.0))
        upper_2       = float(getattr(sig, "vwap_upper_2", 0.0))

        if stock_vs_sec != "LEADING" or sector_trend != "BEARISH":
            return None
        if gate == "BULL":
            return None  # MTF is bullish — sector thesis not yet confirmed
        if sector_change > -0.3 or rvol < 1.1:
            return None  # sector isn't actually falling meaningfully
        if vwap <= 0 or price <= vwap:
            return None  # stock should be above VWAP to show the divergence

        buf    = (price - vwap) * 0.05
        entry  = price
        stop   = (upper_2 + buf) if upper_2 > price else sess_high + buf if sess_high > price else price * 1.007
        target = vwap
        if stop <= entry or target >= entry:
            return None

        conf = min(85, 48 + abs(sector_change) * 6 + vwap_z * 4)
        reason = (
            f"Sector counter fade: stock LEADING while {getattr(sig, 'sector_etf', 'ETF')} "
            f"is {sector_trend} ({sector_change:+.2f}%), z={vwap_z:.2f}"
        )
        return AlgoResult(
            algo="SECTOR_COUNTER_FADE_BEAR", direction="SELL",
            confidence=round(conf, 1),
            entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
            rr=_rr(entry, stop, target), reason=reason,
        )
    except Exception as exc:
        logger.debug("eval_sector_counter_fade error: %s", exc)
    return None


# ── Algo 19: Regime Aligned Long ──────────────────────────────────────────────

def eval_regime_aligned_long(sig) -> Optional[AlgoResult]:
    """
    Regime Aligned Long (Algo 19 / Phase 3.7).

    Maximum confluence long: all three macro layers agree —
      (1) Market regime BULL_TREND (SPY/QQQ in uptrend)
      (2) Stock is LEADING SPY (positive residual)
      (3) Sector is BULLISH (sector tailwind)
      (4) Price above VWAP (micro-structure confirms)

    Entry  = current price.
    Stop   = vwap_lower_1 (first meaningful structure below).
    Target = session_high if not reached, else entry + (session_range × 0.5).
    """
    try:
        regime       = getattr(sig, "regime", "NEUTRAL")
        rs_label     = getattr(sig, "rs_label", "IN_LINE")
        sector_trend = getattr(sig, "sector_trend", "NEUTRAL")
        vwap_event   = getattr(sig, "vwap_event", "FLAT")
        gate         = getattr(sig, "short_tf_alignment", "MIXED")
        rvol         = float(getattr(sig, "rel_volume", 1.0))
        price        = float(sig.price)
        vwap         = float(getattr(sig, "vwap_price", 0.0))
        lower_1      = float(getattr(sig, "vwap_lower_1", 0.0))
        sess_high    = float(getattr(sig, "session_high", 0.0))
        sess_low     = float(getattr(sig, "session_low", 0.0))
        rs_ratio     = float(getattr(sig, "rs_ratio", 1.0))

        if regime != "BULL_TREND":
            return None
        if rs_label not in ("LEADING",):
            return None
        if sector_trend != "BULLISH":
            return None
        if vwap_event not in ("ABOVE", "RECLAIM", "AT_1SD_UP"):
            return None
        if gate == "BEAR" or rvol < 1.2:
            return None

        sess_range = (sess_high - sess_low) if sess_high > sess_low else price * 0.02
        entry  = price
        stop   = lower_1 if lower_1 > 0 and lower_1 < price else vwap * 0.997
        target = sess_high if sess_high > price else entry + sess_range * 0.5
        if stop >= entry or target <= entry:
            return None

        conf = min(93, 62 + (rs_ratio - 1.0) * 5 + (rvol - 1.2) * 6
                   + (8 if gate == "BULL" else 0))
        reason = (
            f"Regime aligned long: regime {regime}, RS {rs_label} ({rs_ratio:.2f}x), "
            f"sector {sector_trend}, VWAP {vwap_event}, RVOL {rvol:.1f}x"
        )
        return AlgoResult(
            algo="REGIME_ALIGNED_LONG", direction="BUY",
            confidence=round(conf, 1),
            entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
            rr=_rr(entry, stop, target), reason=reason,
        )
    except Exception as exc:
        logger.debug("eval_regime_aligned_long error: %s", exc)
    return None


# ── Algo 20: Regime Aligned Short ─────────────────────────────────────────────

def eval_regime_aligned_short(sig) -> Optional[AlgoResult]:
    """
    Regime Aligned Short (Algo 20 / Phase 3.8).

    Maximum confluence short: all macro layers aligned bearish —
      (1) Market regime BEAR_TREND
      (2) Stock lagging or underperforming (stock_vs_sector == LAGGING)
      (3) Sector BEARISH (double headwind)
      (4) Price below VWAP (micro-structure confirms)

    Entry  = current price.
    Stop   = vwap_upper_1.
    Target = session_low if not yet hit, else entry − (session_range × 0.5).
    """
    try:
        regime       = getattr(sig, "regime", "NEUTRAL")
        stock_vs_sec = getattr(sig, "stock_vs_sector", "IN_LINE")
        sector_trend = getattr(sig, "sector_trend", "NEUTRAL")
        vwap_event   = getattr(sig, "vwap_event", "FLAT")
        gate         = getattr(sig, "short_tf_alignment", "MIXED")
        rvol         = float(getattr(sig, "rel_volume", 1.0))
        price        = float(sig.price)
        vwap         = float(getattr(sig, "vwap_price", 0.0))
        upper_1      = float(getattr(sig, "vwap_upper_1", 0.0))
        sess_high    = float(getattr(sig, "session_high", 0.0))
        sess_low     = float(getattr(sig, "session_low", 0.0))
        rs_ratio     = float(getattr(sig, "rs_ratio", 1.0))

        if regime != "BEAR_TREND":
            return None
        if stock_vs_sec not in ("LAGGING", "IN_LINE"):
            return None
        if sector_trend != "BEARISH":
            return None
        if vwap_event not in ("BELOW", "REJECTION", "AT_1SD_DOWN"):
            return None
        if gate == "BULL" or rvol < 1.2:
            return None

        sess_range = (sess_high - sess_low) if sess_high > sess_low else price * 0.02
        entry  = price
        stop   = upper_1 if upper_1 > 0 and upper_1 > price else vwap * 1.003
        target = sess_low if sess_low < price else entry - sess_range * 0.5
        if stop <= entry or target >= entry:
            return None

        conf = min(93, 62 + (1.0 - rs_ratio) * 5 + (rvol - 1.2) * 6
                   + (8 if gate == "BEAR" else 0))
        reason = (
            f"Regime aligned short: regime {regime}, stock {stock_vs_sec}, "
            f"sector {sector_trend}, VWAP {vwap_event}, RVOL {rvol:.1f}x"
        )
        return AlgoResult(
            algo="REGIME_ALIGNED_SHORT", direction="SELL",
            confidence=round(conf, 1),
            entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
            rr=_rr(entry, stop, target), reason=reason,
        )
    except Exception as exc:
        logger.debug("eval_regime_aligned_short error: %s", exc)
    return None


# ── Algo 21: Sector Breakout Follow ───────────────────────────────────────────

def eval_sector_breakout_follow(sig) -> Optional[AlgoResult]:
    """
    Sector Breakout Follow (Algo 21 / Phase 3.9).

    The sector ETF is making a significant intraday directional move (> ±0.8%)
    while this stock has not yet broken out proportionally (LAGGING or IN_LINE).
    The stock will follow the sector move — buy/sell the lag.

    Bull: sector_change > +0.8%, stock not COUNTER, VWAP supports.
    Bear: sector_change < −0.8%, stock not COUNTER, VWAP confirms.

    Entry  = current price.
    Stop   = vwap_lower_1 (bull) / vwap_upper_1 (bear).
    Target = price × (1 + sector_change × 0.65 / 100).
    """
    try:
        sector_change = float(getattr(sig, "sector_change", 0.0))
        stock_vs_sec  = getattr(sig, "stock_vs_sector", "IN_LINE")
        rs_label      = getattr(sig, "rs_label", "IN_LINE")
        vwap_event    = getattr(sig, "vwap_event", "FLAT")
        gate          = getattr(sig, "short_tf_alignment", "MIXED")
        rvol          = float(getattr(sig, "rel_volume", 1.0))
        price         = float(sig.price)
        vwap          = float(getattr(sig, "vwap_price", 0.0))
        upper_1       = float(getattr(sig, "vwap_upper_1", 0.0))
        lower_1       = float(getattr(sig, "vwap_lower_1", 0.0))

        if abs(sector_change) < 0.8 or rvol < 1.1:
            return None
        if stock_vs_sec == "COUNTER" or rs_label == "COUNTER":
            return None  # stock fighting the sector — different thesis

        if sector_change > 0.8 and gate != "BEAR":
            # Sector breaking out bullish
            if vwap > 0 and price < vwap * 0.995:
                return None  # stock too far below VWAP — not ready to follow
            entry  = price
            stop   = lower_1 if lower_1 > 0 and lower_1 < price else vwap * 0.997
            target = price * (1 + sector_change * 0.65 / 100)
            if stop >= entry or target <= entry:
                return None
            conf = min(87, 48 + (sector_change - 0.8) * 8 + (rvol - 1.1) * 7
                       + (8 if gate == "BULL" else 0))
            reason = (
                f"Sector breakout follow bull: {getattr(sig, 'sector_etf', 'ETF')} "
                f"+{sector_change:.2f}%, stock {stock_vs_sec}, RVOL {rvol:.1f}x"
            )
            return AlgoResult(
                algo="SECTOR_BREAKOUT_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )

        if sector_change < -0.8 and gate != "BULL":
            if vwap > 0 and price > vwap * 1.005:
                return None  # still too elevated vs VWAP
            entry  = price
            stop   = upper_1 if upper_1 > 0 and upper_1 > price else vwap * 1.003
            target = price * (1 + sector_change * 0.65 / 100)  # price × (1 − x)
            if stop <= entry or target >= entry:
                return None
            conf = min(87, 48 + (abs(sector_change) - 0.8) * 8 + (rvol - 1.1) * 7
                       + (8 if gate == "BEAR" else 0))
            reason = (
                f"Sector breakout follow bear: {getattr(sig, 'sector_etf', 'ETF')} "
                f"{sector_change:.2f}%, stock {stock_vs_sec}, RVOL {rvol:.1f}x"
            )
            return AlgoResult(
                algo="SECTOR_BREAKOUT_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )
    except Exception as exc:
        logger.debug("eval_sector_breakout_follow error: %s", exc)
    return None


# ── Algo 22: Cross-Sectional RS Rank ──────────────────────────────────────────

def eval_cross_sectional_rs(sig) -> Optional[AlgoResult]:
    """
    Cross-Sectional RS Rank (Algo 22 / Phase 3.10).

    Uses the pre-computed rs_score [-1, +1] as a proxy for cross-sectional
    relative strength ranking.  Top-tier scores (≥ 0.6) signal stocks in the
    top percentile of today's movers; bottom-tier (≤ −0.6) signal weakest.

    Combines RS rank with sector and RVOL for confirmation.

    Bull (rs_score ≥ 0.6): top-percentile RS leader + sector BULLISH + RVOL.
    Bear (rs_score ≤ −0.6): bottom-percentile RS laggard + sector BEARISH.

    Entry  = current price.
    Stop   = vwap_lower_1 (bull) / vwap_upper_1 (bear).
    Target = entry ± session_range × 0.4.
    """
    try:
        rs_score     = float(getattr(sig, "rs_score", 0.0))
        rs_label     = getattr(sig, "rs_label", "IN_LINE")
        sector_trend = getattr(sig, "sector_trend", "NEUTRAL")
        rvol         = float(getattr(sig, "rel_volume", 1.0))
        gate         = getattr(sig, "short_tf_alignment", "MIXED")
        price        = float(sig.price)
        vwap         = float(getattr(sig, "vwap_price", 0.0))
        upper_1      = float(getattr(sig, "vwap_upper_1", 0.0))
        lower_1      = float(getattr(sig, "vwap_lower_1", 0.0))
        sess_high    = float(getattr(sig, "session_high", 0.0))
        sess_low     = float(getattr(sig, "session_low", 0.0))

        if rvol < 1.2:
            return None

        sess_range = (sess_high - sess_low) if sess_high > sess_low else price * 0.02

        if rs_score >= 0.6 and sector_trend != "BEARISH" and gate != "BEAR":
            if vwap > 0 and price <= vwap:
                return None
            entry  = price
            stop   = lower_1 if lower_1 > 0 and lower_1 < price else vwap * 0.997
            target = entry + sess_range * 0.4
            if stop >= entry or target <= entry:
                return None
            conf = min(88, 50 + rs_score * 20 + (rvol - 1.2) * 8
                       + (6 if sector_trend == "BULLISH" else 0))
            reason = (
                f"CS RS rank bull: rs_score={rs_score:.2f} ({rs_label}), "
                f"sector {sector_trend}, RVOL {rvol:.1f}x"
            )
            return AlgoResult(
                algo="CS_RS_RANK_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )

        if rs_score <= -0.6 and sector_trend != "BULLISH" and gate != "BULL":
            if vwap > 0 and price >= vwap:
                return None
            entry  = price
            stop   = upper_1 if upper_1 > 0 and upper_1 > price else vwap * 1.003
            target = entry - sess_range * 0.4
            if stop <= entry or target >= entry:
                return None
            conf = min(88, 50 + abs(rs_score) * 20 + (rvol - 1.2) * 8
                       + (6 if sector_trend == "BEARISH" else 0))
            reason = (
                f"CS RS rank bear: rs_score={rs_score:.2f} ({rs_label}), "
                f"sector {sector_trend}, RVOL {rvol:.1f}x"
            )
            return AlgoResult(
                algo="CS_RS_RANK_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target), reason=reason,
            )
    except Exception as exc:
        logger.debug("eval_cross_sectional_rs error: %s", exc)
    return None


# ── Registry ──────────────────────────────────────────────────────────────────

_ALGO_REGISTRY = [
    # Phase 1 — breakout / momentum
    eval_orb5,
    eval_orb15,
    eval_gap_and_go,
    eval_gap_fade,
    eval_pdh_pdl_breakout,
    eval_hod_lod_break,
    eval_bull_flag,
    eval_bear_flag,
    # Phase 2 — VWAP scalps
    eval_vwap_touch_scalp,
    eval_vwap_hod_scalp,
    eval_vwap_lod_scalp,
    eval_level_rejection_scalp,
    eval_micro_pullback_scalp,
    # Phase 3 — relative value
    eval_spy_beta_catchup,
    eval_residual_momentum,
    eval_residual_reversion,
    eval_sector_leader,
    eval_sector_laggard_catchup,
    eval_sector_counter_fade,
    eval_regime_aligned_long,
    eval_regime_aligned_short,
    eval_sector_breakout_follow,
    eval_cross_sectional_rs,
]


def evaluate_all(sig) -> list[dict]:
    """Run every registered algo against sig; return list of fired AlgoResult dicts."""
    results = []
    for fn in _ALGO_REGISTRY:
        try:
            res = fn(sig)
            if res is not None:
                results.append(res.to_dict())
        except Exception as exc:
            logger.debug("evaluate_all %s error: %s", fn.__name__, exc)
    return results
