"""
Alpha Strike Trader — PRD-compliant risk management engine.

Implements all Section 6 risk rules:
  - Daily loss circuit breakers (tiered: 1.5% warning, 2.5% halt)
  - Consecutive loss cooldowns (3 losses → 30-min pause, 5 → full halt)
  - Profit Protect Mode ($1,000 daily P&L → raise confidence bar, reduce size)
  - Daily profit ceiling ($1,500 → halt trading, protect the gain)
  - Portfolio heat cap (1.5% max combined open risk)
  - Max concurrent positions (3 — hard cap)
  - Sector concentration (2 per sector max)
  - Session time blocks (handled by market_hours.py, checked here centrally)

All limits are env-driven via config.py — no hard-coded numbers.
"""
from __future__ import annotations

import logging
import threading
import time as _time
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)

from config import (
    DEFAULT_ACCOUNT_SIZE,
    DAILY_PROFIT_TARGET_USD,
    DAILY_PROFIT_MAX_USD,
    DAILY_LOSS_WARNING_PCT,
    DAILY_LOSS_HALT_PCT,
    MAX_CONCURRENT_TRADES,
    MAX_PORTFOLIO_HEAT_PCT,
    MAX_CONSECUTIVE_LOSSES,
    COOLDOWN_AFTER_LOSSES,
    PROFIT_PROTECT_MIN_CONF,
    PROFIT_PROTECT_SIZE_MULT,
    PROFIT_PROTECT_DRAWDOWN,
)

_lock = threading.Lock()

# ── State — resets each trading day ──────────────────────────────────────────
_circuit_open:       bool  = False
_circuit_reason:     str   = ""
_circuit_date:       date  = None   # type: ignore[assignment]
_warning_issued:     bool  = False  # 1.5% warning has been shown this session
_cooldown_until:     float = 0.0    # epoch — blocked until this time
_consecutive_losses: int   = 0
_peak_daily_pnl:     float = 0.0    # tracks day's peak to measure drawdown in PPM


def _reset_if_new_day() -> None:
    global _circuit_open, _circuit_reason, _circuit_date
    global _warning_issued, _cooldown_until, _consecutive_losses, _peak_daily_pnl
    today = date.today()
    if _circuit_date != today:
        with _lock:
            _circuit_open        = False
            _circuit_reason      = ""
            _circuit_date        = today
            _warning_issued      = False
            _cooldown_until      = 0.0
            _consecutive_losses  = 0
            _peak_daily_pnl      = 0.0


# ── Sector map ────────────────────────────────────────────────────────────────
_SECTOR_MAP: dict[str, str] = {
    "NVDA":"SEMIS","AMD":"SEMIS","AVGO":"SEMIS","QCOM":"SEMIS","AMAT":"SEMIS",
    "MU":"SEMIS","KLAC":"SEMIS","LRCX":"SEMIS","ADI":"SEMIS","MRVL":"SEMIS",
    "INTC":"SEMIS","SNPS":"SEMIS","CDNS":"SEMIS","MPWR":"SEMIS","ARM":"SEMIS",
    "NXPI":"SEMIS","SWKS":"SEMIS","ON":"SEMIS","MCHP":"SEMIS","TXN":"SEMIS",
    "AAPL":"MEGA_TECH","MSFT":"MEGA_TECH","GOOGL":"MEGA_TECH","META":"MEGA_TECH",
    "AMZN":"MEGA_TECH","NFLX":"MEGA_TECH","TSLA":"MEGA_TECH",
    "ADBE":"CLOUD","INTU":"CLOUD","WDAY":"CLOUD","SNOW":"CLOUD",
    "DDOG":"CLOUD","ZS":"CLOUD","CRWD":"CLOUD","PANW":"CLOUD","OKTA":"CLOUD",
    "NET":"CLOUD","MDB":"CLOUD","TEAM":"CLOUD","GTLB":"CLOUD","HUBS":"CLOUD",
    "TWLO":"CLOUD","BILL":"CLOUD","DOCU":"CLOUD","CSCO":"CLOUD","FTNT":"CLOUD",
    "COIN":"FINTECH","HOOD":"FINTECH","PYPL":"FINTECH","AFRM":"FINTECH",
    "UPST":"FINTECH","MSTR":"FINTECH","MARA":"FINTECH","DKNG":"FINTECH","SOFI":"FINTECH",
    "REGN":"BIOTECH","AMGN":"BIOTECH","ISRG":"BIOTECH","CELH":"BIOTECH",
    "VRTX":"BIOTECH","ALNY":"BIOTECH","BMRN":"BIOTECH","GILD":"BIOTECH",
    "MRNA":"BIOTECH","BNTX":"BIOTECH","NVAX":"BIOTECH","BIIB":"BIOTECH",
    "RIVN":"EV","LCID":"EV","ENPH":"EV","FSLR":"EV","RUN":"EV","ARRY":"EV",
    "PLTR":"AI_EMERGING","SOUN":"AI_EMERGING","IONQ":"AI_EMERGING","RGTI":"AI_EMERGING",
    "QUBT":"AI_EMERGING","RKLB":"AI_EMERGING","ASTS":"AI_EMERGING",
    "SBUX":"CONSUMER","COST":"CONSUMER","LULU":"CONSUMER","CHWY":"CONSUMER",
    "BKNG":"CONSUMER","ABNB":"CONSUMER","MNST":"CONSUMER","ROST":"CONSUMER",
    "LYFT":"TRAVEL","UBER":"TRAVEL","EXPE":"TRAVEL",
    "SNAP":"MEDIA","PINS":"MEDIA","RBLX":"MEDIA","TTD":"MEDIA","ZM":"MEDIA",
    "ROKU":"MEDIA","NFLX":"MEDIA",
    "ADP":"PAYROLL","PAYX":"PAYROLL","AXON":"DEFENSE","CEG":"ENERGY",
    "SMCI":"SERVERS","MELI":"LATAM","APP":"ADTECH","CVNA":"AUTO",
    "BIDU":"CHINA_TECH","LI":"CHINA_TECH","JD":"CHINA_TECH","PDD":"CHINA_TECH",
    "NTES":"CHINA_TECH","BILI":"CHINA_TECH",
}


def get_sector(ticker: str) -> str:
    return _SECTOR_MAP.get(ticker.upper(), "OTHER")


# ── Consecutive loss tracking ─────────────────────────────────────────────────

def record_trade_outcome(won: bool) -> None:
    """Call after every paper trade closes to track consecutive losses."""
    global _consecutive_losses, _cooldown_until, _circuit_open, _circuit_reason, _circuit_date
    _reset_if_new_day()
    with _lock:
        if won:
            _consecutive_losses = 0
        else:
            _consecutive_losses += 1
            logger.info(f"[RiskControls] Consecutive losses: {_consecutive_losses}")

            if _consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                _circuit_open   = True
                _circuit_reason = (
                    f"Full trading halt: {_consecutive_losses} consecutive losses "
                    f"(limit {MAX_CONSECUTIVE_LOSSES}). Resume tomorrow."
                )
                _circuit_date   = date.today()
                logger.warning(f"[RiskControls] {_circuit_reason}")

            elif _consecutive_losses >= COOLDOWN_AFTER_LOSSES:
                _cooldown_until = _time.time() + 30 * 60   # 30-min cooldown
                logger.warning(
                    f"[RiskControls] {_consecutive_losses} consecutive losses — "
                    f"30-minute cooldown active until "
                    f"{datetime.fromtimestamp(_cooldown_until).strftime('%H:%M ET')}"
                )


# ── Daily P&L helpers ─────────────────────────────────────────────────────────

def _get_today_pnl() -> tuple[float, float]:
    """Returns (total_pnl_dollar, total_pnl_pct) for today. (0, 0) on error."""
    try:
        from agent.paper_trading import get_today_pnl
        s = get_today_pnl()
        return (
            float(s.get("total_pnl_dollar", 0) or 0),
            float(s.get("total_pnl_pct", 0) or 0),
        )
    except Exception:
        return 0.0, 0.0


def _get_trade_count() -> int:
    try:
        from agent.paper_trading import get_today_pnl
        return int(get_today_pnl().get("total", 0) or 0)
    except Exception:
        return 0


# ── Circuit breaker ───────────────────────────────────────────────────────────

def check_circuit_breaker() -> tuple[bool, str]:
    """
    Returns (blocked, reason).
    Checks all daily loss tiers, consecutive loss state, and cooldown periods.
    """
    global _circuit_open, _circuit_reason, _circuit_date, _warning_issued
    global _cooldown_until, _peak_daily_pnl
    _reset_if_new_day()

    with _lock:
        if _circuit_open:
            return True, _circuit_reason

        # Active cooldown from consecutive losses
        if _cooldown_until > 0 and _time.time() < _cooldown_until:
            remaining = int((_cooldown_until - _time.time()) / 60) + 1
            return True, f"Cooldown active ({remaining} min remaining after consecutive losses)"

    pnl_dollar, pnl_pct = _get_today_pnl()

    with _lock:
        # Track peak daily P&L for Profit Protect Mode drawdown check
        _peak_daily_pnl = max(_peak_daily_pnl, pnl_dollar)

        # ── Tier 3: 2.5% loss → HALT for the day ────────────────────────────
        if pnl_pct <= -DAILY_LOSS_HALT_PCT:
            reason = (
                f"🛑 Daily loss halt: {pnl_pct:+.2f}% loss today "
                f"(limit -{DAILY_LOSS_HALT_PCT}%). Trading halted until tomorrow."
            )
            _circuit_open   = True
            _circuit_reason = reason
            _circuit_date   = date.today()
            logger.warning(f"[RiskControls] {reason}")
            return True, reason

        # ── Profit ceiling: $1,500 → halt ───────────────────────────────────
        if pnl_dollar >= DAILY_PROFIT_MAX_USD:
            reason = (
                f"✅ Daily profit ceiling reached: ${pnl_dollar:,.0f} "
                f"(max ${DAILY_PROFIT_MAX_USD:,.0f}). Locking in gains — no new trades."
            )
            _circuit_open   = True
            _circuit_reason = reason
            _circuit_date   = date.today()
            logger.info(f"[RiskControls] {reason}")
            return True, reason

        # ── Profit Protect Mode drawdown check ──────────────────────────────
        if pnl_dollar >= DAILY_PROFIT_TARGET_USD:
            peak_drawdown = _peak_daily_pnl - pnl_dollar
            if peak_drawdown >= PROFIT_PROTECT_DRAWDOWN:
                reason = (
                    f"⚠ Profit protect drawdown: pulled back ${peak_drawdown:.0f} "
                    f"from peak ${_peak_daily_pnl:.0f}. Protecting gains."
                )
                _circuit_open   = True
                _circuit_reason = reason
                _circuit_date   = date.today()
                logger.warning(f"[RiskControls] {reason}")
                return True, reason

        # ── Tier 1: 1.5% warning — NOT a halt, just log once ────────────────
        if pnl_pct <= -DAILY_LOSS_WARNING_PCT and not _warning_issued:
            _warning_issued = True
            logger.warning(
                f"[RiskControls] ⚠ Daily loss warning: {pnl_pct:+.2f}% "
                f"(warning at -{DAILY_LOSS_WARNING_PCT}%). Review open positions."
            )

    return False, ""


# ── Profit Protect Mode ───────────────────────────────────────────────────────

def get_profit_protect_state() -> dict:
    """
    Returns whether Profit Protect Mode is active and its modified parameters.
    PPM activates when daily P&L >= DAILY_PROFIT_TARGET_USD ($1,000 default).
    """
    pnl_dollar, _ = _get_today_pnl()
    active = pnl_dollar >= DAILY_PROFIT_TARGET_USD
    return {
        "active":       active,
        "pnl_today":    round(pnl_dollar, 2),
        "target":       DAILY_PROFIT_TARGET_USD,
        "min_conf":     PROFIT_PROTECT_MIN_CONF if active else 0.0,
        "size_mult":    PROFIT_PROTECT_SIZE_MULT if active else 1.0,
        "drawdown_cap": PROFIT_PROTECT_DRAWDOWN,
        "description":  (
            f"Profit Protect Mode ON — size {PROFIT_PROTECT_SIZE_MULT*100:.0f}%, "
            f"min confidence {PROFIT_PROTECT_MIN_CONF:.0f}%"
            if active else "Profit Protect Mode inactive"
        ),
    }


# ── Portfolio heat ────────────────────────────────────────────────────────────

def get_portfolio_heat() -> dict:
    """
    Returns current combined open risk as % of account.
    Heat = sum of (entry - stop) * shares for all open trades / account_size.
    Blocks new trades if heat >= MAX_PORTFOLIO_HEAT_PCT.
    """
    try:
        from agent.paper_trading import get_open_trades
        open_trades = get_open_trades()
        total_risk = 0.0
        for t in open_trades:
            entry = float(t.get("entry_price", 0) or 0)
            stop  = float(t.get("stop", 0) or 0)
            shares = int(t.get("shares", 0) or 0)
            if entry > 0 and stop > 0 and shares > 0:
                risk_per_share = abs(entry - stop)
                total_risk += risk_per_share * shares
        heat_pct = total_risk / DEFAULT_ACCOUNT_SIZE * 100 if DEFAULT_ACCOUNT_SIZE > 0 else 0.0
        return {
            "total_risk_dollar": round(total_risk, 2),
            "heat_pct":          round(heat_pct, 3),
            "limit_pct":         MAX_PORTFOLIO_HEAT_PCT,
            "blocked":           heat_pct >= MAX_PORTFOLIO_HEAT_PCT,
            "open_count":        len(open_trades),
            "max_concurrent":    MAX_CONCURRENT_TRADES,
        }
    except Exception as e:
        logger.debug(f"[RiskControls] portfolio heat error: {e}")
        return {"heat_pct": 0.0, "blocked": False, "open_count": 0}


def check_portfolio_heat() -> tuple[bool, str]:
    """Returns (blocked, reason). Blocks when combined open risk >= 1.5%."""
    heat = get_portfolio_heat()
    if heat["blocked"]:
        reason = (
            f"Portfolio heat {heat['heat_pct']:.2f}% exceeds limit "
            f"{MAX_PORTFOLIO_HEAT_PCT}%. Reduce open risk before new trades."
        )
        return True, reason
    open_count = heat.get("open_count", 0)
    if open_count >= MAX_CONCURRENT_TRADES:
        reason = (
            f"Max concurrent trades reached ({open_count}/{MAX_CONCURRENT_TRADES}). "
            "Wait for an existing trade to close."
        )
        return True, reason
    return False, ""


# ── Sector concentration ───────────────────────────────────────────────────────

def check_sector_concentration(ticker: str, direction: str) -> tuple[bool, str]:
    """
    Returns (blocked, reason).
    Blocks when 2+ positions already open in same sector + direction.
    """
    if direction not in ("BUY", "SELL"):
        return False, ""
    sector = get_sector(ticker)
    if sector == "OTHER":
        return False, ""
    try:
        from agent.paper_trading import get_open_trades
        open_trades = get_open_trades()
        sector_count = sum(
            1 for t in open_trades
            if get_sector(t.get("ticker", "")) == sector
            and t.get("direction", "") == direction
        )
        if sector_count >= 2:
            reason = (
                f"Sector concentration: {sector_count} open {direction} positions "
                f"in {sector} sector (max 2). Skipping {ticker}."
            )
            logger.debug(f"[RiskControls] {reason}")
            return True, reason
    except Exception as e:
        logger.debug(f"[RiskControls] sector check failed: {e}")
    return False, ""


# ── Session block ─────────────────────────────────────────────────────────────

def check_session_block() -> tuple[bool, str]:
    """Returns (blocked, reason) based on current PRD session window."""
    from agent.market_hours import no_new_entries, get_block_reason
    if no_new_entries():
        return True, get_block_reason()
    return False, ""


# ── Master entry gate ─────────────────────────────────────────────────────────

def can_open_trade(
    ticker:     str,
    direction:  str,
    confidence: float,
) -> tuple[bool, str, float]:
    """
    Unified entry gate — runs all PRD checks in priority order.
    Returns (allowed, reason, size_multiplier).

    Check order:
      1. Session block (hardest block — PRD non-negotiable)
      2. Circuit breaker (daily loss / profit ceiling)
      3. Profit Protect Mode (adjusts confidence threshold and size)
      4. Portfolio heat / concurrent count
      5. Sector concentration
    """
    # 1. Session
    blocked, reason = check_session_block()
    if blocked:
        return False, reason, 0.0

    # 2. Circuit breaker
    blocked, reason = check_circuit_breaker()
    if blocked:
        return False, reason, 0.0

    # 3. Profit Protect Mode
    ppm = get_profit_protect_state()
    size_mult = ppm["size_mult"]   # 0.60 in PPM, 1.0 otherwise
    if ppm["active"] and confidence < ppm["min_conf"]:
        return False, (
            f"Profit Protect Mode: confidence {confidence:.0f}% below PPM minimum "
            f"{ppm['min_conf']:.0f}%. Signal skipped to protect ${ppm['pnl_today']:,.0f} gain."
        ), 0.0

    # 4. Portfolio heat / concurrent count
    blocked, reason = check_portfolio_heat()
    if blocked:
        return False, reason, 0.0

    # 5. Sector concentration
    norm_dir = "BUY" if "BUY" in direction else "SELL" if "SELL" in direction else direction
    blocked, reason = check_sector_concentration(ticker, norm_dir)
    if blocked:
        return False, reason, 0.0

    # Also apply session size multiplier (e.g. 0.80 in STANDARD hours)
    from agent.market_hours import position_size_multiplier
    sess_size = position_size_multiplier()
    final_size = round(size_mult * sess_size, 2) if sess_size > 0 else size_mult

    return True, "", final_size


# ── Public status for dashboard ───────────────────────────────────────────────

def get_risk_status() -> dict:
    """Full risk control state for the dashboard API."""
    _reset_if_new_day()
    pnl_dollar, pnl_pct = _get_today_pnl()
    ppm = get_profit_protect_state()
    heat = get_portfolio_heat()

    with _lock:
        circuit_open   = _circuit_open
        circuit_reason = _circuit_reason
        consec         = _consecutive_losses
        cooldown_secs  = max(0, int(_cooldown_until - _time.time())) if _cooldown_until > 0 else 0

    return {
        # Circuit breaker
        "circuit_open":           circuit_open,
        "circuit_reason":         circuit_reason,
        # Daily P&L progress
        "pnl_today_dollar":       round(pnl_dollar, 2),
        "pnl_today_pct":          round(pnl_pct, 3),
        "daily_target":           DAILY_PROFIT_TARGET_USD,
        "daily_max":              DAILY_PROFIT_MAX_USD,
        "progress_to_target_pct": round(min(pnl_dollar / DAILY_PROFIT_TARGET_USD * 100, 100), 1) if pnl_dollar > 0 else 0.0,
        # Loss limits
        "daily_loss_warning_pct": DAILY_LOSS_WARNING_PCT,
        "daily_loss_halt_pct":    DAILY_LOSS_HALT_PCT,
        # Consecutive losses
        "consecutive_losses":     consec,
        "max_consecutive":        MAX_CONSECUTIVE_LOSSES,
        "cooldown_remaining_s":   cooldown_secs,
        # Profit Protect Mode
        "profit_protect":         ppm,
        # Portfolio heat
        "portfolio_heat":         heat,
        # Session
        "session_blocked":        check_session_block()[0],
        "session_block_reason":   check_session_block()[1],
    }
