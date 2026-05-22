"""
Intraday trading algorithm signal evaluators — Phase 1 signals.

Each eval_*() function receives a StockSignal (as a plain namespace/dict-like
object) and returns an AlgoResult or None.  evaluate_all() runs every registered
algo and returns a list of all fired AlgoResult objects.

Algorithm catalogue (this file):
  1  — 5-min ORB  (ORB5_BULL / ORB5_BEAR)
  2  — 15-min ORB (ORB15_BULL / ORB15_BEAR)
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


# ── Registry ──────────────────────────────────────────────────────────────────

_ALGO_REGISTRY = [
    eval_orb5,
    eval_orb15,
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
