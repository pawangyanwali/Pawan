"""
Realistic fill price model for paper trading.

Separates "strategy was correct" from "execution was costly."

Fill formulas
─────────────
BUY  market entry   : last + half_spread + slippage
SELL market entry   : last − half_spread − slippage
LONG stop-market    : stop − slippage   (stop sell fills below trigger)
SHORT stop-market   : stop + slippage   (stop buy-to-cover fills above trigger)
T1 / T2 limit       : trigger_price     (limit orders fill at exact level)
Market / time stop  : direction-aware entry formula

Slippage components (bps of price)
────────────────────────────────────
  base_bps        session baseline  (REGULAR 3, RESTRICTED 8, PM 12, AH 18)
  volatility_bps  extra when ATR% > threshold
  liquidity_bps   extra when avg daily volume is thin
  size_bps        extra when order > 0.01% of daily dollar volume
"""
from __future__ import annotations

import collections


# ── Config helper ──────────────────────────────────────────────────────────────

def _exec_cfg(key: str, default):
    try:
        from agent.config_manager import config as _cfg
        return _cfg.get(key, default)
    except Exception:
        return default


# ── Result type ────────────────────────────────────────────────────────────────

FillResult = collections.namedtuple("FillResult", [
    "fill_price",        # actual simulated fill
    "ideal_price",       # trigger price (signal close / stop / T1 / T2)
    "slippage_bps",      # slippage component in bps
    "slippage_dollar",   # slippage cost in dollars (always positive)
    "spread_bps",        # half-spread paid in bps
    "spread_dollar",     # half-spread cost in dollars
    "liquidity_score",   # 0–1  (1 = very liquid)
    "fill_type",         # ENTRY | STOP_MARKET | T1 | T2 | TIME_STOP
    "session",
])


# ── Session baseline slippage (bps) ───────────────────────────────────────────

_SESSION_BASE_BPS: dict[str, float] = {
    "REGULAR":      3.0,
    "RESTRICTED":   8.0,
    "PRE_MARKET":  12.0,
    "AFTER_HOURS": 18.0,
    "CLOSED":       5.0,
}


def _session_base(session: str) -> float:
    return float(_exec_cfg(
        f"execution.slip_base_{session.lower()}",
        _SESSION_BASE_BPS.get(session, 5.0),
    ))


# ── Core slippage calculator ───────────────────────────────────────────────────

def _compute_slippage(
    price: float,
    atr: float,
    session: str,
    shares: int,
    avg_daily_volume: float,
    base_multiplier: float = 1.0,
) -> tuple[float, float, float]:
    """Return (slippage_bps, slippage_dollar, liquidity_score)."""
    if price <= 0:
        return 0.0, 0.0, 1.0

    base_bps = _session_base(session) * base_multiplier

    # Volatility penalty: each % ATR above the threshold adds vol_penalty_rate bps
    atr_pct = (atr / price * 100.0) if atr > 0 and price > 0 else 0.0
    vol_thresh = float(_exec_cfg("execution.slip_vol_threshold_pct", 1.0))
    vol_rate   = float(_exec_cfg("execution.slip_vol_penalty_bps",   1.5))
    volatility_bps = max(0.0, (atr_pct - vol_thresh) * vol_rate)

    # Liquidity penalty: thin stocks cost more
    liq_low = float(_exec_cfg("execution.slip_liq_low_vol",  100_000))
    liq_mid = float(_exec_cfg("execution.slip_liq_mid_vol",  500_000))
    if avg_daily_volume <= 0:
        liquidity_bps   = 5.0
        liquidity_score = 0.5
    elif avg_daily_volume < liq_low:
        liquidity_bps   = float(_exec_cfg("execution.slip_liq_low_penalty_bps",  10.0))
        liquidity_score = 0.2
    elif avg_daily_volume < liq_mid:
        liquidity_bps   = float(_exec_cfg("execution.slip_liq_mid_penalty_bps",   3.0))
        liquidity_score = 0.6
    else:
        liquidity_bps   = 0.0
        liquidity_score = 1.0

    # Size penalty: large orders relative to daily dollar volume
    position_dollar = shares * price
    daily_dollar    = avg_daily_volume * price
    if daily_dollar > 0:
        size_pct = position_dollar / daily_dollar * 100.0   # % of daily dollar vol
        size_bps = max(0.0, (size_pct - 0.01) * float(_exec_cfg("execution.slip_size_rate", 50.0)))
        size_bps = min(size_bps, float(_exec_cfg("execution.slip_size_cap_bps", 20.0)))
    else:
        size_bps = 0.0

    total_bps = min(
        base_bps + volatility_bps + liquidity_bps + size_bps,
        float(_exec_cfg("execution.slip_max_bps", 60.0)),
    )

    slippage_dollar = round(total_bps / 10_000.0 * price, 6)  # per share
    return round(total_bps, 2), round(slippage_dollar, 6), round(liquidity_score, 3)


def _compute_spread(
    price: float,
    atr: float,
    bid: float = 0.0,
    ask: float = 0.0,
) -> tuple[float, float]:
    """Return (spread_bps, half_spread_dollar_per_share)."""
    if ask > 0 and bid > 0 and ask > bid:
        spread_dollar = ask - bid
    else:
        # Synthetic: fraction of ATR (about 15% of 1-min ATR is the typical spread for liquid names)
        rate      = float(_exec_cfg("execution.spread_atr_rate", 0.15))
        min_bps   = float(_exec_cfg("execution.spread_min_bps",  1.0))
        floor_d   = price * min_bps / 10_000.0
        spread_dollar = max(floor_d, atr * rate)

    spread_bps         = spread_dollar / price * 10_000.0 if price > 0 else 0.0
    half_spread_dollar = spread_dollar / 2.0
    return round(spread_bps, 2), round(half_spread_dollar, 6)


# ── Public fill functions ──────────────────────────────────────────────────────

def compute_entry_fill(
    direction: str,
    last_price: float,
    atr: float,
    session: str,
    shares: int,
    avg_daily_volume: float = 0.0,
    bid: float = 0.0,
    ask: float = 0.0,
) -> FillResult:
    """
    Simulate a market-order entry.
    BUY: pays ask + slippage.   SELL: receives bid − slippage.
    """
    if not float(_exec_cfg("execution.fill_model_enabled", True)):
        return FillResult(last_price, last_price, 0.0, 0.0, 0.0, 0.0, 1.0, "ENTRY", session)

    slip_bps, slip_per_share, liq = _compute_slippage(
        last_price, atr, session, shares, avg_daily_volume, base_multiplier=1.0,
    )
    spread_bps, half_spread = _compute_spread(last_price, atr, bid, ask)

    if direction == "BUY":
        fill_price = last_price + half_spread + slip_per_share
    else:
        fill_price = last_price - half_spread - slip_per_share

    fill_price = round(max(fill_price, 0.01), 4)

    return FillResult(
        fill_price      = fill_price,
        ideal_price     = last_price,
        slippage_bps    = slip_bps,
        slippage_dollar = round(abs(fill_price - last_price) * shares, 2),
        spread_bps      = spread_bps,
        spread_dollar   = round(half_spread * shares, 2),
        liquidity_score = liq,
        fill_type       = "ENTRY",
        session         = session,
    )


def compute_stop_fill(
    direction: str,
    stop_price: float,
    bar_low: float,
    bar_high: float,
    atr: float,
    session: str,
    shares: int,
    avg_daily_volume: float = 0.0,
) -> FillResult:
    """
    Simulate a stop-market fill.

    Stops are 1.5× base slippage because of urgency and potential gap-through.
    When bar_low gapped well below a LONG stop (or bar_high gapped above a SHORT
    stop), a partial gap penalty is applied — the fill is not as bad as the bar
    extreme, but worse than the stop level.

    For LONG (BUY entry):  stop fires as a SELL → fills BELOW stop_price.
    For SHORT (SELL entry): stop fires as a BUY  → fills ABOVE stop_price.
    """
    if not float(_exec_cfg("execution.fill_model_enabled", True)):
        return FillResult(stop_price, stop_price, 0.0, 0.0, 0.0, 0.0, 1.0, "STOP_MARKET", session)

    slip_bps, slip_per_share, liq = _compute_slippage(
        stop_price, atr, session, shares, avg_daily_volume, base_multiplier=1.5,
    )

    # Gap-through penalty: a fraction of the gap between stop and bar extreme
    gap_factor = float(_exec_cfg("execution.stop_gap_factor", 0.30))

    if direction == "BUY":
        # LONG stop: a sell stop; fills below stop
        gap = max(0.0, stop_price - bar_low) if bar_low > 0 else 0.0
        fill_price = stop_price - slip_per_share - gap * gap_factor
    else:
        # SHORT stop: a buy stop; fills above stop
        gap = max(0.0, bar_high - stop_price) if bar_high > 0 else 0.0
        fill_price = stop_price + slip_per_share + gap * gap_factor

    fill_price = round(max(fill_price, 0.01), 4)
    actual_slip_bps = round(abs(fill_price - stop_price) / stop_price * 10_000.0, 2) if stop_price > 0 else 0.0

    return FillResult(
        fill_price      = fill_price,
        ideal_price     = stop_price,
        slippage_bps    = actual_slip_bps,
        slippage_dollar = round(abs(fill_price - stop_price) * shares, 2),
        spread_bps      = 0.0,
        spread_dollar   = 0.0,
        liquidity_score = liq,
        fill_type       = "STOP_MARKET",
        session         = session,
    )


def compute_limit_fill(
    trigger_price: float,
    fill_type: str,
    session: str,
    shares: int,
) -> FillResult:
    """
    Simulate a limit order fill (T1 or T2 target).

    Limit orders fill at the trigger price (or better in live markets).
    We don't add slippage — the broker is obligated to fill at the limit or better.
    """
    return FillResult(
        fill_price      = trigger_price,
        ideal_price     = trigger_price,
        slippage_bps    = 0.0,
        slippage_dollar = 0.0,
        spread_bps      = 0.0,
        spread_dollar   = 0.0,
        liquidity_score = 1.0,
        fill_type       = fill_type,
        session         = session,
    )
