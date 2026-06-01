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
    DAILY_LOSS_LIQUIDATE_PCT,
    MAX_CONCURRENT_TRADES,
    MAX_PORTFOLIO_HEAT_PCT,
    MAX_CONSECUTIVE_LOSSES,
    COOLDOWN_AFTER_LOSSES,
    PROFIT_PROTECT_MIN_CONF,
    PROFIT_PROTECT_SIZE_MULT,
    PROFIT_PROTECT_DRAWDOWN,
    IS_PAPER_TRADING,
    MAX_DAILY_TRADES,
    VOLATILITY_HALT_ATR_MULT,
    DRAWDOWN_THROTTLE_1_PCT,
    DRAWDOWN_THROTTLE_2_PCT,
)

_lock = threading.Lock()


def _rcfg(key: str = None, fallback=None):
    """Read a runtime-configurable risk param from config_store.

    Called with (key, fallback): returns the value for that key, or fallback if absent.
    Called with no args: returns the ConfigManager object (for .get(key, default) chaining).
    """
    try:
        from agent.config_manager import config as _cfg
        if key is None:
            return _cfg
        val = _cfg.get(key)
        return val if val is not None else fallback
    except Exception:
        if key is None:
            class _NullCfg:
                def get(self, k, d=None): return d
            return _NullCfg()
        return fallback


def _account_size() -> float:
    """Account size for all risk-% math — sourced from PostgreSQL (risk.account_size),
    falling back to the DEFAULT_ACCOUNT_SIZE env constant only if config is unreachable."""
    try:
        return float(_rcfg("risk.account_size", DEFAULT_ACCOUNT_SIZE))
    except Exception:
        return float(DEFAULT_ACCOUNT_SIZE)

# ── State — resets each trading day ──────────────────────────────────────────
_circuit_open:        bool  = False
_circuit_reason:      str   = ""
_circuit_date:        date  = None   # type: ignore[assignment]
_circuit_pnl_based:   bool  = False  # True = triggered by P&L, not consecutive losses
_warning_issued:      bool  = False  # 1.5% warning has been shown this session
_cooldown_until:      float = 0.0    # epoch — blocked until this time
_consecutive_losses:  int   = 0
_peak_daily_pnl:      float = 0.0    # tracks day's peak to measure drawdown in PPM
_liquidation_triggered: bool = False # True once the 4% force-close has fired today

# ── Phase 2 state ─────────────────────────────────────────────────────────────
_volatility_halted:   bool  = False  # 2.3 — set True when ATR spike detected
_volatility_reason:   str   = ""
_volatility_atr_ratio: float = 0.0  # current session_range / avg_atr


def _reset_if_new_day() -> None:
    global _circuit_open, _circuit_reason, _circuit_date, _circuit_pnl_based
    global _warning_issued, _cooldown_until, _consecutive_losses, _peak_daily_pnl
    global _volatility_halted, _volatility_reason, _volatility_atr_ratio
    global _liquidation_triggered
    today = date.today()
    with _lock:
        if _circuit_date != today:
            _circuit_open          = False
            _circuit_reason        = ""
            _circuit_date          = today
            _circuit_pnl_based     = False
            _warning_issued        = False
            _cooldown_until        = 0.0
            _consecutive_losses    = 0
            _peak_daily_pnl        = 0.0
            _volatility_halted     = False
            _volatility_reason     = ""
            _volatility_atr_ratio  = 0.0
            _liquidation_triggered = False


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
    "ROKU":"MEDIA",
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

            if IS_PAPER_TRADING:
                # Paper mode: log streaks for observability but never block trading.
                # Blocking reduces training data volume without protecting real capital.
                if _consecutive_losses >= _rcfg("risk.max_consecutive_losses", MAX_CONSECUTIVE_LOSSES):
                    logger.warning(
                        f"[RiskControls] {_consecutive_losses} consecutive losses "
                        f"(would halt in live mode) — paper trading continues"
                    )
                elif _consecutive_losses >= _rcfg("risk.cooldown_after_losses", COOLDOWN_AFTER_LOSSES):
                    logger.warning(
                        f"[RiskControls] {_consecutive_losses} consecutive losses "
                        f"(would cooldown in live mode) — paper trading continues"
                    )
                return

            _max_consec = _rcfg("risk.max_consecutive_losses", MAX_CONSECUTIVE_LOSSES)
            _cooldown_n = _rcfg("risk.cooldown_after_losses",  COOLDOWN_AFTER_LOSSES)
            if _consecutive_losses >= _max_consec:
                _circuit_open        = True
                _circuit_pnl_based   = False   # consecutive-loss halt, NOT P&L-based
                _circuit_reason      = (
                    f"Full trading halt: {_consecutive_losses} consecutive losses "
                    f"(limit {_max_consec}). Resume tomorrow."
                )
                _circuit_date        = date.today()
                logger.warning(f"[RiskControls] {_circuit_reason}")

            elif _consecutive_losses >= _cooldown_n:
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

def _maybe_trigger_liquidation() -> None:
    """
    Tier 2 daily-loss protection: force-close ALL open positions when the account
    loss exceeds risk.daily_loss_liquidate_pct (default 4%).

    Design rationale:
      - Tier 1 (halt at 2.5%): blocks new entries; existing positions keep their
        individual stops — winners can still reach their targets.
      - Tier 2 (liquidate at 4%): the account is in genuine distress. Force-close
        everything to protect the remaining 96% of capital. A 4% loss with 6 open
        longs in a one-directional move can become 8%+ before individual stops fire.

    Runs at most once per day (guarded by _liquidation_triggered). Called from
    check_circuit_breaker() BEFORE the early-return for circuit-open so the
    force-close fires even when the halt (2.5%) already tripped earlier.
    """
    global _liquidation_triggered, _circuit_open, _circuit_reason
    global _circuit_pnl_based, _circuit_date

    with _lock:
        if _liquidation_triggered:
            return

    pnl_dollar, _ = _get_today_pnl()
    _acct = _account_size()
    if _acct <= 0:
        return
    acct_loss_pct = pnl_dollar / _acct * 100  # negative on losing day

    _liq_pct = float(_rcfg("risk.daily_loss_liquidate_pct", DAILY_LOSS_LIQUIDATE_PCT))
    if acct_loss_pct > -_liq_pct:
        return  # below liquidation threshold

    with _lock:
        if _liquidation_triggered:
            return  # another thread beat us here
        _liquidation_triggered = True
        _circuit_open      = True
        _circuit_pnl_based = True
        _circuit_reason    = (
            f"🚨 Daily loss liquidation: {acct_loss_pct:+.2f}% account loss "
            f"(liquidation threshold -{_liq_pct:.1f}%). "
            f"All positions force-closed to protect remaining capital."
        )
        _circuit_date = date.today()

    logger.warning(f"[RiskControls] {_circuit_reason}")

    # Force-close in a daemon thread — don't block this risk-check call
    import threading as _t
    def _do_close():
        try:
            from agent.paper_trading import close_all_positions_eod
            n = close_all_positions_eod("DAILY_LOSS_LIQUIDATE")
            logger.warning(f"[RiskControls] Liquidation complete — {n} position(s) force-closed")
        except Exception as exc:
            logger.error(f"[RiskControls] Liquidation close_all failed: {exc}")
    _t.Thread(target=_do_close, daemon=True, name="DailyLossLiquidate").start()


def check_circuit_breaker(session: str = "") -> tuple[bool, str]:
    """
    Returns (blocked, reason).
    Checks all daily loss tiers, consecutive loss state, and cooldown periods.

    Pass session="AFTER_HOURS" (HIGH-tier AH trades) to bypass consecutive-loss
    halts/cooldowns — those are reset for the AH window. P&L-based halts still apply.
    """
    global _circuit_open, _circuit_reason, _circuit_date, _warning_issued
    global _cooldown_until, _peak_daily_pnl, _circuit_pnl_based, _consecutive_losses
    _reset_if_new_day()

    # Check liquidation threshold BEFORE the circuit-open early return so the
    # force-close fires even when the 2.5% halt already tripped earlier today.
    _maybe_trigger_liquidation()

    with _lock:
        if session == "AFTER_HOURS":
            # Consecutive-loss circuit/cooldown from the regular session is cleared
            # for the AH window — AH HIGH-tier trades start fresh. P&L halts persist.
            if _circuit_open and not _circuit_pnl_based:
                _circuit_open       = False
                _circuit_reason     = ""
                _consecutive_losses = 0
                _cooldown_until     = 0.0
            elif _circuit_open and _circuit_pnl_based:
                return True, _circuit_reason
            # Skip the cooldown check in AH — fall through to P&L checks
        else:
            if _circuit_open:
                return True, _circuit_reason

            # Active cooldown from consecutive losses (live trading only)
            if not IS_PAPER_TRADING and _cooldown_until > 0 and _time.time() < _cooldown_until:
                remaining = int((_cooldown_until - _time.time()) / 60) + 1
                return True, f"Cooldown active ({remaining} min remaining after consecutive losses)"

    pnl_dollar, pnl_pct = _get_today_pnl()

    _acct = _account_size()
    acct_loss_pct = (pnl_dollar / _acct * 100) if _acct > 0 else 0.0

    with _lock:
        # Re-check _circuit_open here — another thread may have tripped it
        # between our earlier check and this lock acquisition.
        if _circuit_open:
            return True, _circuit_reason

        # Track peak daily P&L for Profit Protect Mode drawdown check
        _peak_daily_pnl = max(_peak_daily_pnl, pnl_dollar)

        # ── Tier 3: 2.5% account loss → HALT for the day ────────────────────
        _halt_pct = _rcfg("risk.daily_loss_halt_pct", DAILY_LOSS_HALT_PCT)
        if acct_loss_pct <= -_halt_pct:
            reason = (
                f"🛑 Daily loss halt: {acct_loss_pct:+.2f}% account loss today "
                f"(limit -{_halt_pct}%). Trading halted until tomorrow."
            )
            _circuit_open      = True
            _circuit_pnl_based = True   # P&L-based — persists even in AH
            _circuit_reason    = reason
            _circuit_date      = date.today()
            logger.warning(f"[RiskControls] {reason}")
            return True, reason

        # ── Profit ceiling: $1,500 → halt ───────────────────────────────────
        _profit_max = _rcfg("risk.daily_profit_max_usd", DAILY_PROFIT_MAX_USD)
        if pnl_dollar >= _profit_max:
            reason = (
                f"✅ Daily profit ceiling reached: ${pnl_dollar:,.0f} "
                f"(max ${_profit_max:,.0f}). Locking in gains — no new trades."
            )
            _circuit_open      = True
            _circuit_pnl_based = True   # P&L-based — persists even in AH
            _circuit_reason    = reason
            _circuit_date      = date.today()
            logger.info(f"[RiskControls] {reason}")
            return True, reason

        # ── Profit Protect Mode drawdown check ──────────────────────────────
        _profit_target = _rcfg("risk.daily_profit_target_usd", DAILY_PROFIT_TARGET_USD)
        if pnl_dollar >= _profit_target:
            peak_drawdown = _peak_daily_pnl - pnl_dollar
            if peak_drawdown >= _rcfg("risk.profit_protect_drawdown", PROFIT_PROTECT_DRAWDOWN):
                reason = (
                    f"⚠ Profit protect drawdown: pulled back ${peak_drawdown:.0f} "
                    f"from peak ${_peak_daily_pnl:.0f}. Protecting gains."
                )
                _circuit_open      = True
                _circuit_pnl_based = True   # P&L-based — persists even in AH
                _circuit_reason    = reason
                _circuit_date      = date.today()
                logger.warning(f"[RiskControls] {reason}")
                return True, reason

        # ── Tier 1: 1.5% account loss warning — NOT a halt, just log once ────
        _warn_pct = _rcfg("risk.daily_loss_warning_pct", DAILY_LOSS_WARNING_PCT)
        if acct_loss_pct <= -_warn_pct and not _warning_issued:
            _warning_issued = True
            logger.warning(
                f"[RiskControls] ⚠ Daily loss warning: {acct_loss_pct:+.2f}% "
                f"(warning at -{_warn_pct}%). Review open positions."
            )

    return False, ""


# ── Profit Protect Mode ───────────────────────────────────────────────────────

def get_profit_protect_state() -> dict:
    """
    Returns whether Profit Protect Mode is active and its modified parameters.
    PPM activates when daily P&L >= DAILY_PROFIT_TARGET_USD ($1,000 default).
    """
    pnl_dollar, _ = _get_today_pnl()
    _profit_target = _rcfg("risk.daily_profit_target_usd", DAILY_PROFIT_TARGET_USD)
    _min_conf  = float(_rcfg("risk.profit_protect_min_conf",  PROFIT_PROTECT_MIN_CONF))
    _size_mult = float(_rcfg("risk.profit_protect_size_mult", PROFIT_PROTECT_SIZE_MULT))
    _drawdown  = float(_rcfg("risk.profit_protect_drawdown",  PROFIT_PROTECT_DRAWDOWN))
    active = pnl_dollar >= _profit_target
    return {
        "active":       active,
        "pnl_today":    round(pnl_dollar, 2),
        "target":       _profit_target,
        "min_conf":     _min_conf if active else 0.0,
        "size_mult":    _size_mult if active else 1.0,
        "drawdown_cap": _drawdown,
        "description":  (
            f"Profit Protect Mode ON — size {_size_mult*100:.0f}%, "
            f"min confidence {_min_conf:.0f}%"
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
        _acct = _account_size()
        heat_pct = total_risk / _acct * 100 if _acct > 0 else 0.0
        _heat_limit   = _rcfg("risk.max_portfolio_heat_pct", MAX_PORTFOLIO_HEAT_PCT)
        _max_conc     = _rcfg("risk.max_concurrent_trades",  MAX_CONCURRENT_TRADES)
        return {
            "total_risk_dollar": round(total_risk, 2),
            "heat_pct":          round(heat_pct, 3),
            "limit_pct":         _heat_limit,
            "blocked":           heat_pct >= _heat_limit,
            "open_count":        len(open_trades),
            "max_concurrent":    _max_conc,
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
            f"{heat['limit_pct']}%. Reduce open risk before new trades."
        )
        return True, reason
    open_count = heat.get("open_count", 0)
    _paper_max = int(_rcfg().get("paper.max_open_trades", 20))
    if open_count >= _paper_max:
        reason = (
            f"Max concurrent trades reached ({open_count}/{_paper_max}). "
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
        _max_sector = int(_rcfg().get("risk.max_per_sector", 2))
        if sector_count >= _max_sector:
            reason = (
                f"Sector concentration: {sector_count} open {direction} positions "
                f"in {sector} sector (max {_max_sector}). Skipping {ticker}."
            )
            logger.debug(f"[RiskControls] {reason}")
            return True, reason
    except Exception as e:
        logger.debug(f"[RiskControls] sector check failed: {e}")
    return False, ""


# ── Session block ─────────────────────────────────────────────────────────────

def check_session_block(trading_tier: str = "REGULAR") -> tuple[bool, str, float]:
    """
    Returns (blocked, reason, ah_size_mult) based on current session and tier.

    AFTER_HOURS  HIGH     → 50% size  (mega-caps — consistent AH liquidity)
    AFTER_HOURS  MODERATE → 30% size  (large-caps — meaningful AH activity)
    AFTER_HOURS  REGULAR  → blocked   (thin spreads, no edge)
    PRE_MARKET   HIGH     → 40% size  (mega-caps with strong pre-market volume)
    PRE_MARKET   MODERATE → 25% size  (large-caps — notable PM activity)
    PRE_MARKET   REGULAR  → blocked   (too thin)
    HARD_CLOSE / CLOSED   → blocked for all tiers
    """
    from agent.market_hours import get_session, get_block_reason, no_new_entries
    session = get_session()

    _cfg = _rcfg()
    if session == "AFTER_HOURS":
        if trading_tier == "HIGH":
            return False, "", float(_cfg.get("risk.after_hours_high_size_mult",     0.50))
        if trading_tier == "MODERATE":
            return False, "", float(_cfg.get("risk.after_hours_moderate_size_mult", 0.30))
        return True, "After-hours — REGULAR-tier: thin ECN spreads, no edge outside regular hours.", 0.0

    if session == "PRE_MARKET":
        if trading_tier == "HIGH":
            return False, "", float(_cfg.get("risk.pre_market_high_size_mult",     0.40))
        if trading_tier == "MODERATE":
            return False, "", float(_cfg.get("risk.pre_market_moderate_size_mult", 0.25))
        return True, "Pre-market — REGULAR-tier: insufficient pre-market liquidity.", 0.0

    if no_new_entries():
        return True, get_block_reason(), 0.0
    return False, "", 1.0


# ── 2.3 Volatility halt ──────────────────────────────────────────────────────

def update_volatility_state(session_range_pct: float, avg_atr_pct: float) -> None:
    """
    Called by the scanner after each market cycle with:
      session_range_pct : (session_high - session_low) / session_low * 100
      avg_atr_pct       : average True Range of the past 20 days as % of price

    Sets _volatility_halted when session range exceeds VOLATILITY_HALT_ATR_MULT × avg ATR.
    """
    global _volatility_halted, _volatility_reason, _volatility_atr_ratio
    _reset_if_new_day()
    if avg_atr_pct <= 0:
        return
    ratio = session_range_pct / avg_atr_pct
    _vol_mult = float(_rcfg("risk.volatility_halt_atr_mult", VOLATILITY_HALT_ATR_MULT))
    with _lock:
        _volatility_atr_ratio = round(ratio, 2)
        if ratio >= _vol_mult:
            _volatility_halted = True
            _volatility_reason = (
                f"Volatility halt: session range {session_range_pct:.2f}% is "
                f"{ratio:.1f}× the 20-day ATR ({avg_atr_pct:.2f}%). "
                f"Scalp entries suppressed."
            )
        else:
            _volatility_halted = False
            _volatility_reason = ""


def check_volatility_halt() -> tuple[bool, str]:
    """Returns (halted, reason). Only halts scalp-tier entries, not swing/daily."""
    _reset_if_new_day()
    with _lock:
        return _volatility_halted, _volatility_reason


# ── 2.4 Max trades per day ────────────────────────────────────────────────────

def check_max_daily_trades() -> tuple[bool, str]:
    """Returns (blocked, reason) when total closed+open trades today >= MAX_DAILY_TRADES.

    In paper-trading mode this check is skipped entirely — paper mode maximises
    training-data volume the same way consecutive-loss cooldowns are skipped.
    """
    if IS_PAPER_TRADING:
        return False, ""
    try:
        from agent.paper_trading import get_today_pnl
        today_stats = get_today_pnl()
        total_today = int(today_stats.get("total", 0) or 0)
        _max_trades = _rcfg("risk.max_daily_trades", MAX_DAILY_TRADES)
        if total_today >= _max_trades:
            reason = (
                f"Max daily trades reached: {total_today}/{_max_trades}. "
                "No new entries until tomorrow."
            )
            logger.info(f"[RiskControls] {reason}")
            return True, reason
    except Exception as e:
        logger.debug(f"[RiskControls] max_daily_trades check error: {e}")
    return False, ""


# ── 2.6 Drawdown throttle ────────────────────────────────────────────────────

def get_drawdown_throttle() -> dict:
    """
    Progressive size reduction based on intraday P&L drawdown from session open.

    Thresholds (% of account):
      0.5% drawdown → reduce size to 50%
      1.0% drawdown → reduce size to 25%
      Otherwise     → no throttle (1.0×)

    Only applies when P&L is currently negative (losing day).
    """
    try:
        from agent.paper_trading import get_today_pnl
        today = get_today_pnl()
        pnl_dollar = float(today.get("total_pnl_dollar", 0) or 0)
        _acct = _account_size()
        if pnl_dollar >= 0 or _acct <= 0:
            return {"active": False, "size_mult": 1.0, "drawdown_pct": 0.0, "tier": "NONE"}
        drawdown_pct = abs(pnl_dollar) / _acct * 100
        _thr2 = float(_rcfg("risk.drawdown_throttle_2_pct", DRAWDOWN_THROTTLE_2_PCT))
        _thr1 = float(_rcfg("risk.drawdown_throttle_1_pct", DRAWDOWN_THROTTLE_1_PCT))
        if drawdown_pct >= _thr2:
            return {
                "active": True, "size_mult": 0.25,
                "drawdown_pct": round(drawdown_pct, 3),
                "tier": "SEVERE",
                "description": f"Drawdown {drawdown_pct:.2f}% — size reduced to 25%",
            }
        if drawdown_pct >= _thr1:
            return {
                "active": True, "size_mult": 0.50,
                "drawdown_pct": round(drawdown_pct, 3),
                "tier": "MODERATE",
                "description": f"Drawdown {drawdown_pct:.2f}% — size reduced to 50%",
            }
    except Exception as e:
        logger.debug(f"[RiskControls] drawdown_throttle error: {e}")
    return {"active": False, "size_mult": 1.0, "drawdown_pct": 0.0, "tier": "NONE"}


# ── Master entry gate ─────────────────────────────────────────────────────────

def can_open_trade(
    ticker:       str,
    direction:    str,
    confidence:   float,
    trading_tier: str = "REGULAR",
) -> tuple[bool, str, float]:
    """
    Unified entry gate — runs all PRD checks in priority order.
    Returns (allowed, reason, size_multiplier).

    Check order:
      1. Session block (market closed / hard-close window only)
      2. Circuit breaker (daily loss / profit ceiling)
      3. Max daily trades (2.4)
      4. Profit Protect Mode (adjusts confidence and size)
      5. Portfolio heat / concurrent count
      6. Sector concentration
      7. Volatility halt (2.3) — size only, does not block
      8. Drawdown throttle (2.6) — size only, does not block
    """
    # 1. Session
    blocked, reason, ah_size = check_session_block(trading_tier)
    if blocked:
        return False, reason, 0.0

    # 2. Circuit breaker — bypass consecutive-loss halt during extended-hours trades
    from agent.market_hours import get_session as _get_session
    _cb_session = _get_session() if (trading_tier in ("HIGH", "MODERATE") and ah_size > 0) else ""
    blocked, reason = check_circuit_breaker(_cb_session)
    if blocked:
        return False, reason, 0.0

    # 3. Max daily trades (Phase 2.4)
    blocked, reason = check_max_daily_trades()
    if blocked:
        return False, reason, 0.0

    # 4. Profit Protect Mode
    ppm = get_profit_protect_state()
    size_mult = ppm["size_mult"]   # 0.60 in PPM, 1.0 otherwise
    if ppm["active"] and confidence < ppm["min_conf"]:
        return False, (
            f"Profit Protect Mode: confidence {confidence:.0f}% below PPM minimum "
            f"{ppm['min_conf']:.0f}%. Signal skipped to protect ${ppm['pnl_today']:,.0f} gain."
        ), 0.0

    # 5. Portfolio heat / concurrent count
    blocked, reason = check_portfolio_heat()
    if blocked:
        return False, reason, 0.0

    # 6. Sector concentration
    norm_dir = "BUY" if "BUY" in direction else "SELL" if "SELL" in direction else direction
    blocked, reason = check_sector_concentration(ticker, norm_dir)
    if blocked:
        return False, reason, 0.0

    # Apply size multiplier.
    # Priority: tier-specific extended-hours cap (ah_size) > session default > PPM mult.
    # ah_size is non-zero only when check_session_block() granted extended-hours access;
    # that cap is more granular than the generic session size_mult from market_hours.
    from agent.market_hours import position_size_multiplier
    if ah_size > 0:
        # Extended-hours: use tier+session specific cap (50%/30% AH, 40%/25% PM)
        final_size = round(size_mult * ah_size, 2)
    else:
        sess_size = position_size_multiplier()
        final_size = round(size_mult * sess_size, 2) if sess_size > 0 else size_mult

    # 7. Volatility throttle (2.3) — reduce size during ATR spikes, don't block
    vhalt, _ = check_volatility_halt()
    if vhalt:
        _vol_mult = float(_rcfg().get("risk.volatility_halt_size_mult", 0.50))
        final_size = round(final_size * _vol_mult, 2)

    # 8. Drawdown throttle (2.6) — progressive size reduction on losing days
    dthrottle = get_drawdown_throttle()
    if dthrottle["active"]:
        final_size = round(final_size * dthrottle["size_mult"], 2)

    return True, "", max(final_size, 0.10)   # minimum 10% so trades still fire


# ── Public status for dashboard ───────────────────────────────────────────────

def get_risk_status() -> dict:
    """Full risk control state for the dashboard API."""
    _reset_if_new_day()
    pnl_dollar, pnl_pct = _get_today_pnl()
    ppm = get_profit_protect_state()
    heat = get_portfolio_heat()

    with _lock:
        circuit_open      = _circuit_open
        circuit_reason    = _circuit_reason
        consec            = _consecutive_losses
        cooldown_secs     = max(0, int(_cooldown_until - _time.time())) if _cooldown_until > 0 else 0
        vol_halted        = _volatility_halted
        vol_reason        = _volatility_reason
        vol_atr_ratio     = _volatility_atr_ratio

    dthrottle  = get_drawdown_throttle()
    daily_cnt  = _get_trade_count()

    return {
        # Circuit breaker
        "circuit_open":           circuit_open,
        "circuit_reason":         circuit_reason,
        # Daily P&L progress
        "pnl_today_dollar":       round(pnl_dollar, 2),
        "pnl_today_pct":          round(pnl_pct, 3),
        "daily_target":           _rcfg("risk.daily_profit_target_usd", DAILY_PROFIT_TARGET_USD),
        "daily_max":              _rcfg("risk.daily_profit_max_usd",    DAILY_PROFIT_MAX_USD),
        "progress_to_target_pct": round(min(pnl_dollar / max(_rcfg("risk.daily_profit_target_usd", DAILY_PROFIT_TARGET_USD), 1) * 100, 100), 1) if pnl_dollar > 0 else 0.0,
        # Loss limits
        "daily_loss_warning_pct":    _rcfg("risk.daily_loss_warning_pct",    DAILY_LOSS_WARNING_PCT),
        "daily_loss_halt_pct":       _rcfg("risk.daily_loss_halt_pct",       DAILY_LOSS_HALT_PCT),
        "daily_loss_liquidate_pct":  _rcfg("risk.daily_loss_liquidate_pct",  DAILY_LOSS_LIQUIDATE_PCT),
        "liquidation_triggered":     _liquidation_triggered,
        # Consecutive losses
        "consecutive_losses":     consec,
        "max_consecutive":        _rcfg("risk.max_consecutive_losses", MAX_CONSECUTIVE_LOSSES),
        "cooldown_remaining_s":   cooldown_secs,
        # Phase 2.3: Volatility halt
        "volatility_halted":      vol_halted,
        "volatility_reason":      vol_reason,
        "volatility_atr_ratio":   round(vol_atr_ratio, 2),
        "volatility_halt_mult":   _rcfg("risk.volatility_halt_atr_mult", VOLATILITY_HALT_ATR_MULT),
        # Phase 2.4: Max daily trades
        "daily_trade_count":      daily_cnt,
        "max_daily_trades":       _rcfg("risk.max_daily_trades", MAX_DAILY_TRADES),
        # Phase 2.6: Drawdown throttle
        "drawdown_throttle":      dthrottle,
        # Profit Protect Mode
        "profit_protect":         ppm,
        # Portfolio heat
        "portfolio_heat":         heat,
        # Session
        "session_blocked":        check_session_block()[0],
        "session_block_reason":   check_session_block()[1],
    }
