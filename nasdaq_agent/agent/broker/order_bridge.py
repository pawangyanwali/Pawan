"""
Signal → ThinkorSwim order bridge.

Converts our StockSignal into a bracket order on the Schwab paper account.
Enforces risk controls before every order:

  • Max concurrent TOS positions: 5 (configurable via SCHWAB_MAX_POSITIONS)
  • Max risk per trade: 1% of account equity
  • Confidence gate: signal must meet SCHWAB_MIN_CONFIDENCE
  • Session gate: extended-hours orders use LIMIT+SEAMLESS (no market orders AH)
  • Duplicate guard: won't open a position in a ticker already held
  • Daily loss limit: halts auto-trading if daily P&L < -SCHWAB_MAX_DAILY_LOSS
"""
from __future__ import annotations

import logging
import math
import os
import threading
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_POSITIONS   = int(os.getenv("SCHWAB_MAX_POSITIONS",    "5"))
MIN_CONFIDENCE  = float(os.getenv("SCHWAB_MIN_CONFIDENCE", "68"))
RISK_PCT        = float(os.getenv("SCHWAB_RISK_PCT",       "1.0"))   # % of equity per trade
MAX_DAILY_LOSS  = float(os.getenv("SCHWAB_MAX_DAILY_LOSS", "500"))   # $ stop trading for day

# ── Daily loss guard ──────────────────────────────────────────────────────────
_daily: dict = {"date": None, "pnl": 0.0, "halted": False}
_daily_lock = threading.Lock()


def _check_daily_loss(realized_pnl_today: float) -> bool:
    """Return True if trading is allowed (daily loss not exceeded)."""
    with _daily_lock:
        today = date.today()
        if _daily["date"] != today:
            _daily.update({"date": today, "pnl": 0.0, "halted": False})
        _daily["pnl"] = realized_pnl_today
        if realized_pnl_today <= -MAX_DAILY_LOSS:
            _daily["halted"] = True
        return not _daily["halted"]


def get_daily_status() -> dict:
    with _daily_lock:
        return dict(_daily)


def _calc_qty(equity: float, entry: float, stop: float) -> int:
    """Risk-based position sizing: risk RISK_PCT of equity on the stop distance."""
    if entry <= 0 or stop <= 0 or entry == stop:
        return 0
    risk_dollars   = equity * (RISK_PCT / 100)
    risk_per_share = abs(entry - stop)
    qty = math.floor(risk_dollars / risk_per_share)
    return max(1, min(qty, 500))   # floor 1, cap 500 shares


def _is_extended_session(session: str) -> bool:
    return session in ("AFTER_HOURS", "PRE_MARKET", "CLOSED")


# ── Main entry point ──────────────────────────────────────────────────────────

def maybe_place_tos_order(signal) -> dict:
    """
    Evaluate a StockSignal and place a bracket order if all gates pass.
    Returns a result dict with 'placed', 'reason', and order details.
    """
    ticker     = signal.ticker
    direction  = signal.prediction
    confidence = signal.confidence
    entry      = signal.price
    target     = signal.target_price
    stop       = signal.stop_loss
    session    = getattr(signal, "session", "OPEN")

    # ── Gate 1: direction must be actionable ─────────────────────────────────
    if direction not in ("BUY", "STRONG BUY", "SELL", "STRONG SELL"):
        return {"placed": False, "reason": "Signal not actionable (NEUTRAL/FILTERED)"}

    # ── Gate 2: confidence threshold ─────────────────────────────────────────
    if confidence < MIN_CONFIDENCE:
        return {"placed": False, "reason": f"Confidence {confidence:.0f}% < gate {MIN_CONFIDENCE:.0f}%"}

    # R:R quality is setup context only; it does not veto an actionable signal.

    # ── Gate 4: earnings blackout ─────────────────────────────────────────────
    if getattr(signal, "earnings_blocked", False):
        return {"placed": False, "reason": f"Earnings blackout ({signal.earnings_reason})"}

    # ── Gate 5: broker connection + account info ──────────────────────────────
    try:
        from agent.broker.schwab_auth import get_access_token, get_token_status
        from agent.broker.schwab_client import (
            get_account_summary, get_positions, place_bracket_order
        )
        if not get_access_token():
            return {"placed": False, "reason": "Broker not connected — auth required"}
    except Exception as e:
        return {"placed": False, "reason": f"Broker unavailable: {e}"}

    # ── Gate 6: position limits ───────────────────────────────────────────────
    try:
        positions = get_positions()
    except Exception as e:
        return {"placed": False, "reason": f"Could not fetch positions: {e}"}

    held_tickers = {p["ticker"] for p in positions}
    if ticker in held_tickers:
        return {"placed": False, "reason": f"Already holding {ticker}"}

    if len(positions) >= MAX_POSITIONS:
        return {"placed": False, "reason": f"Max positions ({MAX_POSITIONS}) reached"}

    # ── Gate 7: daily loss limit ──────────────────────────────────────────────
    day_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
    if not _check_daily_loss(day_pnl):
        return {"placed": False, "reason": f"Daily loss limit hit (${-MAX_DAILY_LOSS:.0f}) — trading halted"}

    # ── Gate 8: account equity for sizing ────────────────────────────────────
    try:
        summary = get_account_summary()
        equity  = summary.get("equity", 0) or summary.get("buying_power", 10000)
    except Exception:
        equity  = 10000   # fallback if account fetch fails

    qty = _calc_qty(equity, entry, stop)
    if qty < 1:
        return {"placed": False, "reason": "Position size too small (entry/stop too close)"}

    # ── Gate 9: extended-hours order type ────────────────────────────────────
    tos_session = "SEAMLESS" if _is_extended_session(session) else "NORMAL"
    # Extended-hours requires LIMIT orders
    if _is_extended_session(session) and (entry <= 0 or target <= 0):
        return {"placed": False, "reason": "Extended-hours order needs valid limit prices"}

    # ── Place the bracket order ───────────────────────────────────────────────
    try:
        result = place_bracket_order(
            ticker=ticker, direction=direction, qty=qty,
            entry_price=entry, target=target, stop=stop,
            session=tos_session,
        )
        if result.get("success"):
            logger.info(
                f"TOS order placed: {direction} {qty}×{ticker} "
                f"entry={entry:.2f} tp={target:.2f} sl={stop:.2f} "
                f"session={tos_session}"
            )
            return {
                "placed":    True,
                "ticker":    ticker,
                "direction": direction,
                "qty":       qty,
                "entry":     entry,
                "target":    target,
                "stop":      stop,
                "session":   tos_session,
                "equity":    round(equity, 2),
                "risk_usd":  round(qty * abs(entry - stop), 2),
            }
        return {"placed": False, "reason": result.get("error", "Order rejected")}
    except Exception as e:
        logger.error(f"maybe_place_tos_order error: {e}")
        return {"placed": False, "reason": str(e)}
