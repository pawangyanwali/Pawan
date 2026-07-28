"""Entry-quality assessment and temporal confirmation for scalp plans."""
from __future__ import annotations

import math
import time
from typing import Any, MutableMapping

from .models import ScalpSignalPlan, SignalSide


_EXTENDED_SESSIONS = {"PRE_MARKET", "AFTER_HOURS", "EXTENDED"}


def assess_entry_quality(plan: ScalpSignalPlan, config: Any) -> ScalpSignalPlan:
    """Score TP1 reachability from facts known before entry.

    The score is a transparent ranking, not a calibrated probability. It is
    intentionally separate from direction detection and bracket geometry.
    """
    score = 50.0
    evidence: list[str] = []

    spread = _number(plan.spread_to_risk)
    if spread <= 0.10:
        score += 10.0
        evidence.append("COST_EXCELLENT")
    elif spread < 0.20:
        score += 4.0
        evidence.append("COST_ACCEPTABLE")
    else:
        score -= 14.0
        evidence.append("COST_HEAVY")

    rvol = _number(plan.rvol)
    if rvol >= 2.0:
        score += 10.0
        evidence.append("RVOL_STRONG")
    elif rvol >= 1.0:
        score += 3.0
        evidence.append("RVOL_ADEQUATE")
    else:
        score -= 8.0
        evidence.append("RVOL_THIN")

    atr_bucket = str(plan.atr_bucket or "UNKNOWN").upper()
    if atr_bucket == "NORMAL":
        score += 6.0
        evidence.append("ATR_NORMAL")
    elif atr_bucket == "LOW":
        score -= 4.0
        evidence.append("ATR_LOW")
    else:
        score -= 2.0
        evidence.append(f"ATR_{atr_bucket}")

    vwap_event = str(plan.vwap_event or "UNKNOWN").upper()
    rsi_zone = str(plan.rsi_zone or "UNKNOWN").upper()
    if plan.side is SignalSide.LONG:
        if vwap_event == "RECLAIM":
            score += 10.0
            evidence.append("VWAP_RECLAIM_STRONG")
        elif vwap_event == "BOUNCE_SUPPORT":
            score += 8.0
            evidence.append("VWAP_SUPPORT_BOUNCE")
        else:
            score -= 12.0
            evidence.append("VWAP_LONG_LOCATION_ONLY")
        if rsi_zone == "EXTREME_OS":
            score += 8.0
            evidence.append("RSI_EXTREME_OS")
        else:
            score -= 6.0
            evidence.append("RSI_PLAIN_OS")
    elif plan.side is SignalSide.SHORT:
        if vwap_event == "REJECTION":
            score += 10.0
            evidence.append("VWAP_REJECTION_STRONG")
        elif vwap_event == "REJECT_RESISTANCE":
            score += 8.0
            evidence.append("VWAP_RESISTANCE_REJECTION")
        else:
            evidence.append("VWAP_SHORT_LOCATION_ONLY")
        if rsi_zone == "EXTREME_OB":
            score += 5.0
            evidence.append("RSI_EXTREME_OB")
        else:
            score += 2.0
            evidence.append("RSI_OB")

    score = round(max(0.0, min(100.0, score)), 1)
    session = str(plan.session or "").upper()
    threshold_key = (
        "scalp.entry_quality_min_score_extended"
        if session in _EXTENDED_SESSIONS
        else "scalp.entry_quality_min_score"
    )
    threshold_default = 70.0 if session in _EXTENDED_SESSIONS else 65.0
    threshold = max(0.0, min(100.0, _number(
        config.get(threshold_key, threshold_default), threshold_default
    )))

    plan.entry_quality_assessed = True
    plan.entry_quality_score = score
    plan.entry_quality_min_score = threshold
    plan.entry_quality_gate = "PASS" if score >= threshold else "BELOW_MINIMUM"
    plan.entry_quality_reasons = evidence
    # The former reason-count score saturated at 100 for nearly every valid
    # plan. Use the discriminating quality score as the auditable confidence.
    plan.base_confidence = score
    plan.confidence = score
    plan.reasons.extend(
        value for value in (f"ENTRY_QUALITY_{item}" for item in evidence)
        if value not in plan.reasons
    )
    return plan


def confirmation_ready(
    plan: ScalpSignalPlan,
    *,
    bar_id: int,
    pending: MutableMapping[str, dict[str, Any]],
    config: Any,
    now: float | None = None,
) -> bool:
    """Require a valid setup to remain stable across multiple runtime cycles."""
    if not bool(config.get("scalp.entry_confirmation_enabled", True)):
        pending.pop(plan.ticker, None)
        plan.entry_confirmation_state = "DISABLED"
        return True

    timestamp = time.monotonic() if now is None else float(now)
    minimum_seconds = max(
        0.0, _number(config.get("scalp.entry_confirmation_seconds", 15.0), 15.0)
    )
    minimum_observations = max(
        2, int(config.get("scalp.entry_confirmation_min_observations", 3))
    )
    max_chase_r = max(
        0.0, _number(config.get("scalp.entry_confirmation_max_chase_r", 0.25), 0.25)
    )
    ticker = str(plan.ticker or "").upper()
    identity = (plan.side.value, str(plan.setup_type or "").upper())
    state = pending.get(ticker)

    if (
        not state
        or tuple(state.get("identity") or ()) != identity
        or int(bar_id) < int(state.get("bar_id") or 0)
    ):
        pending[ticker] = {
            "identity": identity,
            "first_seen": timestamp,
            "last_seen": timestamp,
            "bar_id": int(bar_id),
            "entry": float(plan.entry),
            "risk": float(plan.risk_per_share),
            "observations": 1,
        }
        plan.entry_confirmation_state = "PENDING"
        plan.entry_confirmation_observations = 1
        plan.reasons.append("ENTRY_CONFIRMATION_PENDING")
        return False

    direction = 1.0 if plan.side is SignalSide.LONG else -1.0
    initial_entry = _number(state.get("entry"))
    initial_risk = max(_number(state.get("risk")), _number(plan.risk_per_share))
    chase_r = (
        direction * (float(plan.entry) - initial_entry) / initial_risk
        if initial_risk > 0
        else 0.0
    )
    if chase_r > max_chase_r:
        pending[ticker] = {
            "identity": identity,
            "first_seen": timestamp,
            "last_seen": timestamp,
            "bar_id": int(bar_id),
            "entry": float(plan.entry),
            "risk": float(plan.risk_per_share),
            "observations": 1,
        }
        plan.entry_confirmation_state = "RESET_CHASE"
        plan.entry_confirmation_observations = 1
        plan.reasons.append("ENTRY_CONFIRMATION_RESET_CHASE")
        return False

    state["last_seen"] = timestamp
    state["bar_id"] = max(int(state.get("bar_id") or 0), int(bar_id))
    state["observations"] = int(state.get("observations") or 1) + 1
    age = max(0.0, timestamp - _number(state.get("first_seen"), timestamp))
    observations = int(state["observations"])
    plan.entry_confirmation_age_s = round(age, 3)
    plan.entry_confirmation_observations = observations
    if age < minimum_seconds or observations < minimum_observations:
        plan.entry_confirmation_state = "PENDING"
        plan.reasons.append("ENTRY_CONFIRMATION_PENDING")
        return False

    pending.pop(ticker, None)
    plan.entry_confirmation_state = "CONFIRMED"
    plan.reasons.append("ENTRY_CONFIRMATION_CONFIRMED")
    return True


def clear_pending(
    pending: MutableMapping[str, dict[str, Any]], ticker: str
) -> None:
    pending.pop(str(ticker or "").upper(), None)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default
