"""
Intraday trading algorithm signal evaluators — Phase 1 signals.

Each eval_*() function receives a StockSignal (as a plain namespace/dict-like
object) and returns an AlgoResult or None.  evaluate_all() runs every registered
algo and returns a list of all fired AlgoResult objects.

Algorithm catalogue (this file):
  1  — 5-min ORB        (ORB5_BULL / ORB5_BEAR)
  2  — 15-min ORB       (ORB15_BULL / ORB15_BEAR)
  6  — Gap-and-Go       (GAP_AND_GO_BULL / BEAR)
  7  — Gap Fade         (GAP_FADE_BULL / BEAR)
  8  — PDH/PDL Breakout (PDH_BREAKOUT_BULL / PDL_BREAKDOWN_BEAR)
  12 — HOD/LOD Break    (HOD_BREAK_BULL / LOD_BREAK_BEAR)
  29 — Bull Flag        (BULL_FLAG)
  30 — Bear Flag        (BEAR_FLAG)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.scanner import StockSignal

logger = logging.getLogger(__name__)


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

        if bo == "BULL" and rvol >= 1.5 and gate != "BEAR":
            entry  = price
            stop   = orl
            target = entry + or_range * 1.5
            conf   = min(95, 60 + (rvol - 1.5) * 10 + (10 if gate == "BULL" else 0))
            reason = (
                f"ORB-5 bull breakout: price {price:.2f} > ORH {orh:.2f}; "
                f"RVOL {rvol:.1f}x; TF gate {gate}"
            )
            return AlgoResult(
                algo="ORB5_BULL", direction="BUY",
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
                f"ORB-5 bear breakdown: price {price:.2f} < ORL {orl:.2f}; "
                f"RVOL {rvol:.1f}x; TF gate {gate}"
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


# ── Registry ──────────────────────────────────────────────────────────────────

_ALGO_REGISTRY = [
    eval_orb5,
    eval_orb15,
    eval_gap_and_go,
    eval_gap_fade,
    eval_pdh_pdl_breakout,
    eval_hod_lod_break,
    eval_bull_flag,
    eval_bear_flag,
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
