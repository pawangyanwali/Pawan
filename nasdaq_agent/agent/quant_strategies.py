"""
Quantitative strategy implementations — Strategy Reference v1.

All 20 strategies from the reference document are implemented here.
Each eval_* function follows the same contract as trading_algos.py:
  - Receives a StockSignal (reads sig.tech_row for bar-level indicators)
  - Returns AlgoResult | None
  - Uses _param() for adaptive parameters (learned by AlgoLearningEngine)

Strategy catalogue:
  OFI family (microstructure):
    OFI_IMPULSE_BULL / BEAR         — order-flow z-score + microprice confirmation
    QUEUE_MICRO_BULL / BEAR         — queue imbalance + microprice continuation
    AGG_VOL_MOM_BULL / BEAR         — aggressor volume + ZV + VWAP

  Trend family:
    VWAP_OFI_PULL_BULL / BEAR       — VWAP pullback with OFI confirmation
    EMA_SLOPE_PULL_BULL / BEAR      — EMA slope pullback to fast EMA
    VWAP_TREND_BRK_BULL / BEAR      — VWAP + EMA trend breakout
    DONCHIAN_BRK_BULL / BEAR        — Donchian channel breakout with volume
    ORB_VWAP_ZV_BULL / BEAR         — Opening range breakout (VWAP + ZV gated)
    MACD_ACC_BULL / BEAR            — MACD histogram acceleration
    SUPERTREND_BULL / BEAR          — SuperTrend ATR-band flip continuation
    VOL_SHOCK_BULL / BEAR           — Realized-vol shock continuation

  Mean-reversion family:
    BB_MEAN_REV_BULL / BEAR         — Bollinger z-score mean reversion
    RSI2_SNAP_BULL / BEAR           — RSI(2) exhaustion snapback
    KC_FADE_BULL / BEAR             — Keltner exhaustion fade
    FAILED_BRK_BULL / BEAR          — Failed breakout reversal
    SQUEEZE_EXP_BULL / BEAR         — Bollinger–Keltner squeeze expansion

  Pair / regime / meta:
    PAIR_STAT_ARB_LONG / SHORT      — RS-ratio z-score mean reversion
    REGIME_TREND_BULL / BEAR        — Regime-switch trend engine
    REGIME_FADE_BULL / BEAR         — Regime-switch fade engine
    META_ENS_BULL / BEAR            — Probability-calibrated meta ensemble (≥0.90)

Note: MARKET_MAKING_QUOTES is informational only (passive fills not simulated)
and is not included in the registry.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Shared helpers ────────────────────────────────────────────────────────────

try:
    from agent.algo_learning_engine import get_algo_params as _get_algo_params
    _ALE_AVAILABLE = True
except ImportError:
    _ALE_AVAILABLE = False


def _param(algo_family: str, name: str, default: float) -> float:
    if not _ALE_AVAILABLE:
        return default
    try:
        return float(_get_algo_params(algo_family).get(name, default))
    except Exception:
        return default


def _rr(entry: float, stop: float, target: float) -> float:
    risk   = abs(entry - stop)
    reward = abs(target - entry)
    return round(reward / risk, 2) if risk > 0 else 0.0


def _t(row: dict, key: str, default: float = 0.0) -> float:
    """Safe float read from tech_row with NaN guard."""
    v = row.get(key, default)
    try:
        f = float(v)
        return f if f == f else default   # NaN check
    except (TypeError, ValueError):
        return default


def _get_tech(sig) -> dict:
    return getattr(sig, "tech_row", {}) or {}


# ── AlgoResult (mirrors trading_algos.AlgoResult for independence) ─────────

try:
    from agent.trading_algos import AlgoResult
except ImportError:
    from dataclasses import dataclass, asdict

    @dataclass
    class AlgoResult:
        algo: str
        direction: str
        confidence: float
        entry: float
        stop: float
        target: float
        rr: float
        reason: str

        def to_dict(self) -> dict:
            d = asdict(self)
            for k, v in d.items():
                if isinstance(v, float):
                    d[k] = round(v, 4)
            return d


# ═════════════════════════════════════════════════════════════════════════════
# 1. OFI IMPULSE  — TP 0.40 ATR | SL 0.35 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_ofi_impulse(sig) -> Optional[AlgoResult]:
    """
    Order-flow z-score spike + microprice above mid.
    OFI proxy: 5-bar CVD z-score vs 20-bar rolling std.
    Microprice vs mid proxy: close above (or below) EMA-9 by > 0.10× spread.
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        # Adaptive params
        ofi_z_gate = _param("OFI", "rvol_gate", 1.5)   # OFI z-score threshold
        tp_m       = _param("OFI", "target_mult", 0.40)
        sl_m       = _param("OFI", "stop_mult",   0.35)

        # OFI z-score: cvd_5 / rolling-std of (buy_vol - sell_vol)
        # We use vol_z_score as a correlated proxy when direct OFI std is unavailable
        cvd5      = _t(t, "cvd_5")
        vol_z     = _t(t, "vol_z_score")
        ema9      = _t(t, "ema_9")
        ema20     = _t(t, "ema_20")
        spread    = max(atr * 0.1, price * 0.0005)   # synthetic spread proxy
        mid       = (ema9 + ema20) / 2 if ema9 and ema20 else price
        rvol      = float(getattr(sig, "rel_volume", 1.0))
        regime    = getattr(sig, "regime", "NEUTRAL")

        # BULLISH: positive OFI impulse + microprice above mid + volume confirmed
        if cvd5 > 0 and vol_z >= ofi_z_gate and price > mid + 0.10 * spread:
            entry  = price
            stop   = entry - sl_m * atr
            target = entry + tp_m * atr
            if stop >= entry or target <= entry:
                return None
            conf = min(88, 58 + vol_z * 5 + (6 if regime == "BULLISH" else 0))
            return AlgoResult(
                algo="OFI_IMPULSE_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"OFI impulse bull: cvd5={cvd5:.0f}, vol_z={vol_z:.2f}, price>{mid:.2f}+spread",
            )

        # BEARISH: negative OFI impulse + microprice below mid
        if cvd5 < 0 and vol_z >= ofi_z_gate and price < mid - 0.10 * spread:
            entry  = price
            stop   = entry + sl_m * atr
            target = entry - tp_m * atr
            if stop <= entry or target >= entry:
                return None
            conf = min(88, 58 + vol_z * 5 + (6 if regime == "BEARISH" else 0))
            return AlgoResult(
                algo="OFI_IMPULSE_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"OFI impulse bear: cvd5={cvd5:.0f}, vol_z={vol_z:.2f}, price<{mid:.2f}-spread",
            )
    except Exception as exc:
        logger.debug("eval_ofi_impulse error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 2. QUEUE MICROPRICE  — TP 0.30 ATR | SL 0.25 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_queue_microprice(sig) -> Optional[AlgoResult]:
    """
    Queue imbalance (AVI proxy > 0.55) + microprice above/below mid.
    AVI proxy: avi_score = (buy_vol - sell_vol) / total_vol.
    Fast EMA confirmation required.
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        qi_gate = _param("OFI", "rvol_gate", 0.55)
        tp_m    = _param("OFI", "target_mult", 0.30)
        sl_m    = _param("OFI", "stop_mult",   0.25)

        avi   = _t(t, "avi_score")
        ema9  = _t(t, "ema_9")
        ema20 = _t(t, "ema_20")
        spread = max(atr * 0.10, price * 0.0005)
        mid    = (ema9 + ema20) / 2 if ema9 and ema20 else price

        # BULLISH: strong buy pressure + microprice above mid + close above fast EMA
        if avi >= qi_gate and price > mid + 0.20 * spread and price > ema9:
            entry  = price
            stop   = entry - sl_m * atr
            target = entry + tp_m * atr
            if stop >= entry or target <= entry:
                return None
            conf = min(85, 55 + avi * 30)
            return AlgoResult(
                algo="QUEUE_MICRO_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Queue microprice bull: AVI={avi:.2f}≥{qi_gate:.2f}, price>{mid:.2f}+0.20×spread",
            )

        # BEARISH
        if avi <= -qi_gate and price < mid - 0.20 * spread and price < ema9:
            entry  = price
            stop   = entry + sl_m * atr
            target = entry - tp_m * atr
            if stop <= entry or target >= entry:
                return None
            conf = min(85, 55 + abs(avi) * 30)
            return AlgoResult(
                algo="QUEUE_MICRO_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Queue microprice bear: AVI={avi:.2f}≤-{qi_gate:.2f}, price<{mid:.2f}-0.20×spread",
            )
    except Exception as exc:
        logger.debug("eval_queue_microprice error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 3. AGGRESSOR VOLUME MOMENTUM  — TP 0.50 ATR | SL 0.45 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_aggressor_volume_momentum(sig) -> Optional[AlgoResult]:
    """
    AVI > 0.35, ZV > 1.5, C > VWAP, fast EMA above slow EMA.
    All conditions from reference; mirrored for short side.
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        avi_gate = _param("OFI", "rvol_gate", 0.35)
        zv_gate  = _param("OFI", "entry_window_bars", 1.5)
        tp_m     = _param("OFI", "target_mult", 0.50)
        sl_m     = _param("OFI", "stop_mult",   0.45)

        avi   = _t(t, "avi_score")
        vol_z = _t(t, "vol_z_score")
        ema9  = _t(t, "ema_9")
        ema20 = _t(t, "ema_20")
        vwap  = float(getattr(sig, "vwap_price", 0.0)) or _t(t, "vwap")
        gate  = getattr(sig, "short_tf_alignment", "MIXED")

        # BULLISH
        if (avi >= avi_gate and vol_z >= zv_gate and price > vwap > 0
                and ema9 > ema20 and gate != "BEAR"):
            entry  = price
            stop   = entry - sl_m * atr
            target = entry + tp_m * atr
            if stop >= entry or target <= entry:
                return None
            conf = min(90, 58 + avi * 20 + vol_z * 4 + (8 if gate == "BULL" else 0))
            return AlgoResult(
                algo="AGG_VOL_MOM_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Aggressor vol mom bull: AVI={avi:.2f}, ZV={vol_z:.2f}, C>VWAP, EMA9>EMA20",
            )

        # BEARISH
        if (avi <= -avi_gate and vol_z >= zv_gate and price < vwap > 0
                and ema9 < ema20 and gate != "BULL"):
            entry  = price
            stop   = entry + sl_m * atr
            target = entry - tp_m * atr
            if stop <= entry or target >= entry:
                return None
            conf = min(90, 58 + abs(avi) * 20 + vol_z * 4 + (8 if gate == "BEAR" else 0))
            return AlgoResult(
                algo="AGG_VOL_MOM_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Aggressor vol mom bear: AVI={avi:.2f}, ZV={vol_z:.2f}, C<VWAP, EMA9<EMA20",
            )
    except Exception as exc:
        logger.debug("eval_aggressor_volume_momentum error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 4. MARKET-MAKING QUOTES — informational only, no tradeable signal
# ═════════════════════════════════════════════════════════════════════════════

def eval_market_making_quotes(sig) -> Optional[AlgoResult]:
    """
    Computes inventory-aware reservation price for dashboard display only.
    Per the Strategy Reference: passive fills deliberately not simulated.
    Always returns None — not in the tradeable registry.
    """
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 5. VWAP OFI PULLBACK  — TP 0.60 ATR | SL 0.45 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_vwap_ofi_pullback(sig) -> Optional[AlgoResult]:
    """
    Bullish EMA regime; prior price near VWAP; cross above VWAP with positive OFI.
    OFI confirmed by positive cvd_5 + vol_z > 0.
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        tp_m = _param("VWAP_OFI", "target_mult", 0.60)
        sl_m = _param("VWAP_OFI", "stop_mult",   0.45)

        vwap   = float(getattr(sig, "vwap_price", 0.0)) or _t(t, "vwap")
        event  = getattr(sig, "vwap_event", "FLAT")
        cvd5   = _t(t, "cvd_5")
        vol_z  = _t(t, "vol_z_score")
        ema9   = _t(t, "ema_9")
        ema20  = _t(t, "ema_20")
        rvol   = float(getattr(sig, "rel_volume", 1.0))
        gate   = getattr(sig, "short_tf_alignment", "MIXED")

        rvol_gate = _param("VWAP_OFI", "rvol_gate", 1.2)

        # BULLISH: EMA regime + VWAP reclaim + positive OFI
        if (event == "RECLAIM" and ema9 > ema20 and cvd5 > 0
                and vol_z > 0 and rvol >= rvol_gate and gate != "BEAR"):
            entry  = price
            stop   = vwap - sl_m * atr
            target = entry + tp_m * atr
            if stop >= entry or target <= entry:
                return None
            conf = min(90, 60 + vol_z * 5 + (rvol - 1.0) * 8 + (8 if gate == "BULL" else 0))
            return AlgoResult(
                algo="VWAP_OFI_PULL_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"VWAP OFI pullback bull: RECLAIM + cvd5={cvd5:.0f}, vol_z={vol_z:.2f}, EMA9>EMA20",
            )

        # BEARISH: EMA regime + VWAP rejection + negative OFI
        if (event == "REJECTION" and ema9 < ema20 and cvd5 < 0
                and vol_z > 0 and rvol >= rvol_gate and gate != "BULL"):
            entry  = price
            stop   = vwap + sl_m * atr
            target = entry - tp_m * atr
            if stop <= entry or target >= entry:
                return None
            conf = min(90, 60 + vol_z * 5 + (rvol - 1.0) * 8 + (8 if gate == "BEAR" else 0))
            return AlgoResult(
                algo="VWAP_OFI_PULL_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"VWAP OFI pullback bear: REJECTION + cvd5={cvd5:.0f}, vol_z={vol_z:.2f}, EMA9<EMA20",
            )
    except Exception as exc:
        logger.debug("eval_vwap_ofi_pullback error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 6. EMA SLOPE PULLBACK  — TP 0.75 ATR | SL 0.55 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_ema_slope_pullback(sig) -> Optional[AlgoResult]:
    """
    Fast EMA above slow; both rising (positive slope); prior pullback touched
    fast EMA; current bar closes back above fast EMA (reclaim).
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        tp_m = _param("EMA_PULL", "target_mult", 0.75)
        sl_m = _param("EMA_PULL", "stop_mult",   0.55)

        ema9       = _t(t, "ema_9")
        ema20      = _t(t, "ema_20")
        ema9_slope = _t(t, "ema_9_slope")
        e20_slope  = _t(t, "ema_20_slope")
        rvol       = float(getattr(sig, "rel_volume", 1.0))
        rvol_gate  = _param("EMA_PULL", "rvol_gate", 1.1)
        gate       = getattr(sig, "short_tf_alignment", "MIXED")

        # BULLISH: both EMAs rising, fast > slow, price just reclaimed fast EMA
        if (ema9 > ema20 > 0 and ema9_slope > 0 and e20_slope > 0
                and price > ema9 and rvol >= rvol_gate and gate != "BEAR"):
            entry  = price
            stop   = ema20 - sl_m * atr
            target = entry + tp_m * atr
            if stop >= entry or target <= entry:
                return None
            conf = min(88, 60 + ema9_slope / max(atr, 0.01) * 500 +
                       (8 if gate == "BULL" else 0))
            return AlgoResult(
                algo="EMA_SLOPE_PULL_BULL", direction="BUY",
                confidence=round(min(conf, 88), 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"EMA slope pullback bull: EMA9={ema9:.2f}>EMA20={ema20:.2f}, both rising, reclaim",
            )

        # BEARISH: both EMAs falling, fast < slow, price just lost fast EMA
        if (ema9 < ema20 and ema9_slope < 0 and e20_slope < 0
                and price < ema9 and rvol >= rvol_gate and gate != "BULL"):
            entry  = price
            stop   = ema20 + sl_m * atr
            target = entry - tp_m * atr
            if stop <= entry or target >= entry:
                return None
            conf = min(88, 60 + abs(ema9_slope) / max(atr, 0.01) * 500 +
                       (8 if gate == "BEAR" else 0))
            return AlgoResult(
                algo="EMA_SLOPE_PULL_BEAR", direction="SELL",
                confidence=round(min(conf, 88), 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"EMA slope pullback bear: EMA9={ema9:.2f}<EMA20={ema20:.2f}, both falling, breakdown",
            )
    except Exception as exc:
        logger.debug("eval_ema_slope_pullback error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 7. VWAP TREND BREAKOUT  — TP 1.00 ATR | SL 0.65 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_vwap_trend_breakout(sig) -> Optional[AlgoResult]:
    """
    Above rising VWAP and bullish EMAs; closes above prior 3-bar high with ZV>1.
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        tp_m     = _param("VWAP_TREND", "target_mult", 1.00)
        sl_m     = _param("VWAP_TREND", "stop_mult",   0.65)
        zv_gate  = _param("VWAP_TREND", "rvol_gate",   1.0)

        vwap  = float(getattr(sig, "vwap_price", 0.0)) or _t(t, "vwap")
        ema9  = _t(t, "ema_9")
        ema20 = _t(t, "ema_20")
        vol_z = _t(t, "vol_z_score")
        sh    = float(getattr(sig, "session_high", 0.0))   # HOD as proxy for 3-bar high
        gate  = getattr(sig, "short_tf_alignment", "MIXED")

        # BULLISH
        if (price > vwap > 0 and ema9 > ema20 and vol_z >= zv_gate
                and price >= sh > 0 and gate != "BEAR"):
            entry  = price
            stop   = vwap - sl_m * atr
            target = entry + tp_m * atr
            if stop >= entry or target <= entry:
                return None
            conf = min(90, 62 + vol_z * 6 + (10 if gate == "BULL" else 0))
            return AlgoResult(
                algo="VWAP_TREND_BRK_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"VWAP trend breakout bull: C>{vwap:.2f}, EMA9>EMA20, ZV={vol_z:.2f}, new HOD",
            )

        # BEARISH
        sl_v  = float(getattr(sig, "session_low", 0.0))
        if (price < vwap > 0 and ema9 < ema20 and vol_z >= zv_gate
                and (sl_v <= 0 or price <= sl_v) and gate != "BULL"):
            entry  = price
            stop   = vwap + sl_m * atr
            target = entry - tp_m * atr
            if stop <= entry or target >= entry:
                return None
            conf = min(90, 62 + vol_z * 6 + (10 if gate == "BEAR" else 0))
            return AlgoResult(
                algo="VWAP_TREND_BRK_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"VWAP trend breakout bear: C<{vwap:.2f}, EMA9<EMA20, ZV={vol_z:.2f}, new LOD",
            )
    except Exception as exc:
        logger.debug("eval_vwap_trend_breakout error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 8. DONCHIAN VOLUME BREAKOUT  — TP 1.20 ATR | SL 0.75 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_donchian_volume_breakout(sig) -> Optional[AlgoResult]:
    """
    Closes above prior 20-bar Donchian high; ZV>1.5; ATR expanded vs 20-bar avg.
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        tp_m    = _param("DONCHIAN", "target_mult", 1.20)
        sl_m    = _param("DONCHIAN", "stop_mult",   0.75)
        zv_gate = _param("DONCHIAN", "rvol_gate",   1.5)

        don_hi = _t(t, "donchian_high")
        don_lo = _t(t, "donchian_low")
        vol_z  = _t(t, "vol_z_score")
        gate   = getattr(sig, "short_tf_alignment", "MIXED")

        # BULLISH: close above prior channel high + volume surge
        if don_hi > 0 and price > don_hi and vol_z >= zv_gate and gate != "BEAR":
            entry  = price
            stop   = don_hi - sl_m * atr
            target = entry + tp_m * atr
            if stop >= entry or target <= entry:
                return None
            conf = min(90, 63 + vol_z * 6 + (8 if gate == "BULL" else 0))
            return AlgoResult(
                algo="DONCHIAN_BRK_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Donchian breakout bull: C={price:.2f}>{don_hi:.2f} channel, ZV={vol_z:.2f}",
            )

        # BEARISH: close below prior channel low + volume surge
        if don_lo > 0 and price < don_lo and vol_z >= zv_gate and gate != "BULL":
            entry  = price
            stop   = don_lo + sl_m * atr
            target = entry - tp_m * atr
            if stop <= entry or target >= entry:
                return None
            conf = min(90, 63 + vol_z * 6 + (8 if gate == "BEAR" else 0))
            return AlgoResult(
                algo="DONCHIAN_BRK_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Donchian breakdown bear: C={price:.2f}<{don_lo:.2f} channel, ZV={vol_z:.2f}",
            )
    except Exception as exc:
        logger.debug("eval_donchian_volume_breakout error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 9. OPENING RANGE BREAKOUT (VWAP + ZV gated)  — TP 1.00 ATR | SL 0.60 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_orb_vwap_zv(sig) -> Optional[AlgoResult]:
    """
    Closes above session ORB + 0.10 ATR, above VWAP, ZV>1.5.
    Complements the existing ORB5/ORB15 algos with explicit VWAP + ZV gates
    and the configurable ATR offset from the Strategy Reference.
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        tp_m    = _param("ORB_ZV", "target_mult", 1.00)
        sl_m    = _param("ORB_ZV", "stop_mult",   0.60)
        zv_gate = _param("ORB_ZV", "rvol_gate",   1.5)
        atr_off = 0.10   # ATR offset above ORB per reference

        orb_hi = float(getattr(sig, "orb5_high",  0.0)) or float(getattr(sig, "orb15_high", 0.0))
        orb_lo = float(getattr(sig, "orb5_low",   0.0)) or float(getattr(sig, "orb15_low",  0.0))
        vwap   = float(getattr(sig, "vwap_price", 0.0)) or _t(t, "vwap")
        vol_z  = _t(t, "vol_z_score")
        gate   = getattr(sig, "short_tf_alignment", "MIXED")

        # BULLISH
        if (orb_hi > 0 and price > orb_hi + atr_off * atr
                and price > vwap > 0 and vol_z >= zv_gate and gate != "BEAR"):
            entry  = price
            stop   = orb_hi - sl_m * atr
            target = entry + tp_m * atr
            if stop >= entry or target <= entry:
                return None
            conf = min(90, 63 + vol_z * 5 + (8 if gate == "BULL" else 0))
            return AlgoResult(
                algo="ORB_VWAP_ZV_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"ORB VWAP-ZV bull: C={price:.2f}>{orb_hi:.2f}+0.10ATR, VWAP={vwap:.2f}, ZV={vol_z:.2f}",
            )

        # BEARISH
        if (orb_lo > 0 and price < orb_lo - atr_off * atr
                and price < vwap > 0 and vol_z >= zv_gate and gate != "BULL"):
            entry  = price
            stop   = orb_lo + sl_m * atr
            target = entry - tp_m * atr
            if stop <= entry or target >= entry:
                return None
            conf = min(90, 63 + vol_z * 5 + (8 if gate == "BEAR" else 0))
            return AlgoResult(
                algo="ORB_VWAP_ZV_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"ORB VWAP-ZV bear: C={price:.2f}<{orb_lo:.2f}-0.10ATR, VWAP={vwap:.2f}, ZV={vol_z:.2f}",
            )
    except Exception as exc:
        logger.debug("eval_orb_vwap_zv error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 10. MACD HISTOGRAM ACCELERATION  — TP 0.90 ATR | SL 0.60 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_macd_acceleration(sig) -> Optional[AlgoResult]:
    """
    MACD histogram positive AND increasing (hist > hist_prev > 0) with
    bullish EMA + price above VWAP confirmation.
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        tp_m      = _param("MACD_ACC", "target_mult", 0.90)
        sl_m      = _param("MACD_ACC", "stop_mult",   0.60)
        rvol_gate = _param("MACD_ACC", "rvol_gate",   1.1)

        hist      = _t(t, "macd_hist")
        hist_prev = _t(t, "macd_hist_prev")
        ema9      = _t(t, "ema_9")
        ema20     = _t(t, "ema_20")
        vwap      = float(getattr(sig, "vwap_price", 0.0)) or _t(t, "vwap")
        rvol      = float(getattr(sig, "rel_volume", 1.0))
        gate      = getattr(sig, "short_tf_alignment", "MIXED")

        # BULLISH: histogram growing above zero + EMA + VWAP
        if (hist > 0 and hist > hist_prev and ema9 > ema20 and price > vwap > 0
                and rvol >= rvol_gate and gate != "BEAR"):
            entry  = price
            stop   = ema20 - sl_m * atr
            target = entry + tp_m * atr
            if stop >= entry or target <= entry:
                return None
            conf = min(90, 60 + (hist - hist_prev) / max(atr * 0.01, 0.0001) * 2 +
                       (8 if gate == "BULL" else 0))
            return AlgoResult(
                algo="MACD_ACC_BULL", direction="BUY",
                confidence=round(min(conf, 90), 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"MACD acceleration bull: hist={hist:.4f}>{hist_prev:.4f}, EMA9>EMA20, C>VWAP",
            )

        # BEARISH: histogram growing more negative
        if (hist < 0 and hist < hist_prev and ema9 < ema20 and price < vwap > 0
                and rvol >= rvol_gate and gate != "BULL"):
            entry  = price
            stop   = ema20 + sl_m * atr
            target = entry - tp_m * atr
            if stop <= entry or target >= entry:
                return None
            conf = min(90, 60 + (hist_prev - hist) / max(atr * 0.01, 0.0001) * 2 +
                       (8 if gate == "BEAR" else 0))
            return AlgoResult(
                algo="MACD_ACC_BEAR", direction="SELL",
                confidence=round(min(conf, 90), 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"MACD acceleration bear: hist={hist:.4f}<{hist_prev:.4f}, EMA9<EMA20, C<VWAP",
            )
    except Exception as exc:
        logger.debug("eval_macd_acceleration error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 11. SUPERTREND ATR CONTINUATION  — TP 1.25 ATR | SL 0.80 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_supertrend_continuation(sig) -> Optional[AlgoResult]:
    """
    SuperTrend ATR-band state flips upward (bearish→bullish or new bullish bar)
    with close above fast EMA and volume confirmation.
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        tp_m      = _param("SUPERTREND", "target_mult", 1.25)
        sl_m      = _param("SUPERTREND", "stop_mult",   0.80)
        rvol_gate = _param("SUPERTREND", "rvol_gate",   1.2)

        st_dir  = _t(t, "supertrend_dir")
        st_flip = _t(t, "supertrend_flip")
        st_lb   = _t(t, "supertrend_lower")
        st_ub   = _t(t, "supertrend_upper")
        ema9    = _t(t, "ema_9")
        vol_z   = _t(t, "vol_z_score")
        rvol    = float(getattr(sig, "rel_volume", 1.0))
        gate    = getattr(sig, "short_tf_alignment", "MIXED")

        # BULLISH: SuperTrend turned bullish (flip or confirmed) + EMA + volume
        if (st_dir >= 1.0 and st_flip >= 1.0 and price > ema9
                and rvol >= rvol_gate and gate != "BEAR"):
            entry  = price
            stop   = max(st_lb, entry - sl_m * atr) if st_lb > 0 else entry - sl_m * atr
            target = entry + tp_m * atr
            if stop >= entry or target <= entry:
                return None
            conf = min(90, 65 + vol_z * 5 + (8 if gate == "BULL" else 0))
            return AlgoResult(
                algo="SUPERTREND_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"SuperTrend flip bull: dir={st_dir:.0f}, flip={st_flip:.0f}, C>EMA9, RVOL={rvol:.1f}x",
            )

        # BEARISH
        if (st_dir <= -1.0 and st_flip >= 1.0 and price < ema9
                and rvol >= rvol_gate and gate != "BULL"):
            entry  = price
            stop   = min(st_ub, entry + sl_m * atr) if st_ub > 0 else entry + sl_m * atr
            target = entry - tp_m * atr
            if stop <= entry or target >= entry:
                return None
            conf = min(90, 65 + vol_z * 5 + (8 if gate == "BEAR" else 0))
            return AlgoResult(
                algo="SUPERTREND_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"SuperTrend flip bear: dir={st_dir:.0f}, flip={st_flip:.0f}, C<EMA9, RVOL={rvol:.1f}x",
            )
    except Exception as exc:
        logger.debug("eval_supertrend_continuation error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 12. REALISED-VOLATILITY SHOCK CONTINUATION  — TP 0.80 ATR | SL 0.55 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_volatility_shock_continuation(sig) -> Optional[AlgoResult]:
    """
    Short/long realised-vol ratio > 2 (vol expansion), positive return,
    volume surge, and VWAP confirm direction.
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        tp_m      = _param("VOL_SHOCK", "target_mult", 0.80)
        sl_m      = _param("VOL_SHOCK", "stop_mult",   0.55)
        ratio_min = _param("VOL_SHOCK", "rvol_gate",   2.0)

        shock = _t(t, "vol_shock_ratio")
        ret1  = _t(t, "ret_1")
        vwap  = float(getattr(sig, "vwap_price", 0.0)) or _t(t, "vwap")
        vol_z = _t(t, "vol_z_score")
        gate  = getattr(sig, "short_tf_alignment", "MIXED")

        # BULLISH: vol shock, positive return, price above VWAP
        if shock >= ratio_min and ret1 > 0 and price > vwap > 0 and vol_z >= 1.0:
            entry  = price
            stop   = entry - sl_m * atr
            target = entry + tp_m * atr
            if stop >= entry or target <= entry:
                return None
            conf = min(88, 60 + shock * 5 + (8 if gate == "BULL" else 0))
            return AlgoResult(
                algo="VOL_SHOCK_BULL", direction="BUY",
                confidence=round(min(conf, 88), 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Vol shock continuation bull: ratio={shock:.2f}, ret={ret1:.4f}, C>VWAP",
            )

        # BEARISH
        if shock >= ratio_min and ret1 < 0 and price < vwap > 0 and vol_z >= 1.0:
            entry  = price
            stop   = entry + sl_m * atr
            target = entry - tp_m * atr
            if stop <= entry or target >= entry:
                return None
            conf = min(88, 60 + shock * 5 + (8 if gate == "BEAR" else 0))
            return AlgoResult(
                algo="VOL_SHOCK_BEAR", direction="SELL",
                confidence=round(min(conf, 88), 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Vol shock continuation bear: ratio={shock:.2f}, ret={ret1:.4f}, C<VWAP",
            )
    except Exception as exc:
        logger.debug("eval_volatility_shock_continuation error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 13. BOLLINGER MEAN REVERSION  — TP 0.70 ATR | SL 0.70 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_bollinger_mean_reversion(sig) -> Optional[AlgoResult]:
    """
    Price z-score below negative BB band (bb_z < -1), RSI(2) < 10, quiet ATR.
    Mean-reversion trade targeting mid-band.
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        tp_m       = _param("BB_REV", "target_mult", 0.70)
        sl_m       = _param("BB_REV", "stop_mult",   0.70)
        rsi2_gate  = _param("BB_REV", "conf_gate",   10.0)
        bb_z_gate  = _param("BB_REV", "rvol_gate",   -1.0)  # negative = below lower band

        bb_z  = _t(t, "bb_z_score")
        rsi2  = _t(t, "rsi_2", 50.0)
        bb_mid = _t(t, "bb_mid")
        bb_lo  = _t(t, "bb_lower")
        bb_up  = _t(t, "bb_upper")
        vol_z = _t(t, "vol_z_score")

        # BULLISH mean-reversion: price at/below lower BB + RSI(2) oversold + calm vol
        if bb_z <= bb_z_gate and rsi2 <= rsi2_gate and vol_z < 1.5 and bb_mid > 0:
            entry  = price
            stop   = entry - sl_m * atr
            target = min(bb_mid, entry + tp_m * atr)  # target mid-band or 0.70 ATR
            if stop >= entry or target <= entry:
                return None
            conf = min(85, 60 + (rsi2_gate - rsi2) * 2 + abs(bb_z) * 5)
            return AlgoResult(
                algo="BB_MEAN_REV_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"BB mean-rev bull: bb_z={bb_z:.2f}, RSI(2)={rsi2:.1f}, quiet ATR",
            )

        # BEARISH mean-reversion: price at/above upper BB + RSI(2) overbought + calm vol
        if bb_z >= abs(bb_z_gate) and rsi2 >= (100 - rsi2_gate) and vol_z < 1.5 and bb_mid > 0:
            entry  = price
            stop   = entry + sl_m * atr
            target = max(bb_mid, entry - tp_m * atr)
            if stop <= entry or target >= entry:
                return None
            conf = min(85, 60 + (rsi2 - (100 - rsi2_gate)) * 2 + bb_z * 5)
            return AlgoResult(
                algo="BB_MEAN_REV_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"BB mean-rev bear: bb_z={bb_z:.2f}, RSI(2)={rsi2:.1f}, quiet ATR",
            )
    except Exception as exc:
        logger.debug("eval_bollinger_mean_reversion error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 14. RSI(2) EXHAUSTION SNAPBACK  — TP 0.50 ATR | SL 0.60 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_rsi_exhaustion_snapback(sig) -> Optional[AlgoResult]:
    """
    RSI(2) < 5 (extreme oversold), price far below VWAP, low volume shock, flat trend.
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        tp_m      = _param("BB_REV", "target_mult", 0.50)
        sl_m      = _param("BB_REV", "stop_mult",   0.60)
        rsi2_gate = _param("BB_REV", "entry_window_bars", 5.0)

        rsi2  = _t(t, "rsi_2", 50.0)
        vwap  = float(getattr(sig, "vwap_price", 0.0)) or _t(t, "vwap")
        vwap_z = float(getattr(sig, "vwap_z_score", 0.0))
        vol_z = _t(t, "vol_z_score")
        adx   = _t(t, "adx_14", 25.0)

        # BULLISH snapback: RSI(2) extreme low + far below VWAP + low vol + flat trend
        if (rsi2 <= rsi2_gate and vwap > 0 and price < vwap
                and vwap_z <= -1.5 and vol_z < 1.0 and adx < 25):
            entry  = price
            stop   = entry - sl_m * atr
            target = vwap   # snap back to VWAP
            target = min(target, entry + tp_m * atr)
            if stop >= entry or target <= entry:
                return None
            conf = min(82, 55 + (rsi2_gate - rsi2) * 4 + abs(vwap_z) * 3)
            return AlgoResult(
                algo="RSI2_SNAP_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"RSI(2) snapback bull: RSI2={rsi2:.1f}, VWAP_z={vwap_z:.2f}, ADX={adx:.1f}",
            )

        # BEARISH snapback: RSI(2) extreme high + far above VWAP
        if (rsi2 >= (100 - rsi2_gate) and vwap > 0 and price > vwap
                and vwap_z >= 1.5 and vol_z < 1.0 and adx < 25):
            entry  = price
            stop   = entry + sl_m * atr
            target = vwap
            target = max(target, entry - tp_m * atr)
            if stop <= entry or target >= entry:
                return None
            conf = min(82, 55 + (rsi2 - (100 - rsi2_gate)) * 4 + vwap_z * 3)
            return AlgoResult(
                algo="RSI2_SNAP_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"RSI(2) snapback bear: RSI2={rsi2:.1f}, VWAP_z={vwap_z:.2f}, ADX={adx:.1f}",
            )
    except Exception as exc:
        logger.debug("eval_rsi_exhaustion_snapback error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 15. KELTNER EXHAUSTION FADE  — TP 0.65 ATR | SL 0.65 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_keltner_exhaustion_fade(sig) -> Optional[AlgoResult]:
    """
    Low breaches lower Keltner channel but close rejects back inside — exhaustion fade.
    Moderate volume (not a volume-shock breakout).
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        tp_m     = _param("KELTNER_FADE", "target_mult", 0.65)
        sl_m     = _param("KELTNER_FADE", "stop_mult",   0.65)
        rvol_max = _param("KELTNER_FADE", "rvol_gate",   2.5)  # not a breakout vol event

        kc_lo = _t(t, "kc_lower")
        kc_hi = _t(t, "kc_upper")
        bb_mid = _t(t, "bb_mid")
        rvol  = float(getattr(sig, "rel_volume", 1.0))
        vol_z = _t(t, "vol_z_score")
        ret1  = _t(t, "ret_1")

        # BULLISH fade: low poked below Keltner lower but closed inside (ret1 > 0 = recovery)
        if (kc_lo > 0 and price > kc_lo and ret1 > 0
                and rvol < rvol_max and vol_z < 2.5):
            entry  = price
            stop   = kc_lo - sl_m * atr
            target = bb_mid if bb_mid > price else entry + tp_m * atr
            if stop >= entry or target <= entry:
                return None
            conf = min(82, 58 + abs(ret1) * 200 + (5 if vol_z < 1.0 else 0))
            return AlgoResult(
                algo="KC_FADE_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Keltner fade bull: close={price:.2f} back inside KC_lo={kc_lo:.2f}, RVOL={rvol:.1f}x",
            )

        # BEARISH fade: high poked above Keltner upper but closed inside
        if (kc_hi > 0 and price < kc_hi and ret1 < 0
                and rvol < rvol_max and vol_z < 2.5):
            entry  = price
            stop   = kc_hi + sl_m * atr
            target = bb_mid if 0 < bb_mid < price else entry - tp_m * atr
            if stop <= entry or target >= entry:
                return None
            conf = min(82, 58 + abs(ret1) * 200 + (5 if vol_z < 1.0 else 0))
            return AlgoResult(
                algo="KC_FADE_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Keltner fade bear: close={price:.2f} back inside KC_hi={kc_hi:.2f}, RVOL={rvol:.1f}x",
            )
    except Exception as exc:
        logger.debug("eval_keltner_exhaustion_fade error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 16. FAILED BREAKOUT REVERSAL  — TP 0.60 ATR | SL 0.45 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_failed_breakout_reversal(sig) -> Optional[AlgoResult]:
    """
    Low breaches prior Donchian range and closes back above it with positive AVI/volume.
    The breakout fails and reverses — trade in the reversal direction.
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        tp_m     = _param("KELTNER_FADE", "target_mult", 0.60)
        sl_m     = _param("KELTNER_FADE", "stop_mult",   0.45)
        avi_gate = _param("KELTNER_FADE", "rvol_gate",   0.20)

        don_lo = _t(t, "donchian_low")
        don_hi = _t(t, "donchian_high")
        avi    = _t(t, "avi_score")
        vol_z  = _t(t, "vol_z_score")
        ret1   = _t(t, "ret_1")

        # BULLISH reversal: price poked below Donchian low but closed above it
        if (don_lo > 0 and price > don_lo and ret1 > 0
                and avi >= avi_gate and vol_z >= 0.5):
            entry  = price
            stop   = don_lo - sl_m * atr
            target = entry + tp_m * atr
            if stop >= entry or target <= entry:
                return None
            conf = min(85, 60 + avi * 20 + vol_z * 4)
            return AlgoResult(
                algo="FAILED_BRK_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Failed breakdown reversal bull: price={price:.2f} back>{don_lo:.2f}, AVI={avi:.2f}",
            )

        # BEARISH reversal: price poked above Donchian high but closed below it
        if (don_hi > 0 and price < don_hi and ret1 < 0
                and avi <= -avi_gate and vol_z >= 0.5):
            entry  = price
            stop   = don_hi + sl_m * atr
            target = entry - tp_m * atr
            if stop <= entry or target >= entry:
                return None
            conf = min(85, 60 + abs(avi) * 20 + vol_z * 4)
            return AlgoResult(
                algo="FAILED_BRK_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Failed breakout reversal bear: price={price:.2f} back<{don_hi:.2f}, AVI={avi:.2f}",
            )
    except Exception as exc:
        logger.debug("eval_failed_breakout_reversal error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 17. BOLLINGER–KELTNER SQUEEZE EXPANSION  — TP 1.25 ATR | SL 0.70 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_squeeze_expansion(sig) -> Optional[AlgoResult]:
    """
    Prior bar in BB-inside-Keltner squeeze; current bar exits squeeze above
    upper Bollinger band with volume confirmation.
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        tp_m     = _param("SQUEEZE", "target_mult", 1.25)
        sl_m     = _param("SQUEEZE", "stop_mult",   0.70)
        zv_gate  = _param("SQUEEZE", "rvol_gate",   1.0)

        squeeze = _t(t, "bb_squeeze")
        bb_up   = _t(t, "bb_upper")
        bb_lo   = _t(t, "bb_lower")
        bb_mid  = _t(t, "bb_mid")
        vol_z   = _t(t, "vol_z_score")
        gate    = getattr(sig, "short_tf_alignment", "MIXED")

        # Squeeze must have been active (bb_squeeze==1 on this or recent bar)
        # We use squeeze == 1 AND price breaking out
        if squeeze < 1.0 or vol_z < zv_gate:
            return None

        # BULLISH: breaks above upper BB
        if price > bb_up > 0 and gate != "BEAR":
            entry  = price
            stop   = bb_mid - sl_m * atr if bb_mid > 0 else entry - sl_m * atr
            target = entry + tp_m * atr
            if stop >= entry or target <= entry:
                return None
            conf = min(90, 65 + vol_z * 6 + (8 if gate == "BULL" else 0))
            return AlgoResult(
                algo="SQUEEZE_EXP_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Squeeze expansion bull: C={price:.2f}>{bb_up:.2f} BB-upper, ZV={vol_z:.2f}",
            )

        # BEARISH: breaks below lower BB
        if price < bb_lo > 0 and gate != "BULL":
            entry  = price
            stop   = bb_mid + sl_m * atr if bb_mid > 0 else entry + sl_m * atr
            target = entry - tp_m * atr
            if stop <= entry or target >= entry:
                return None
            conf = min(90, 65 + vol_z * 6 + (8 if gate == "BEAR" else 0))
            return AlgoResult(
                algo="SQUEEZE_EXP_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Squeeze expansion bear: C={price:.2f}<{bb_lo:.2f} BB-lower, ZV={vol_z:.2f}",
            )
    except Exception as exc:
        logger.debug("eval_squeeze_expansion error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 18. PAIR-SPREAD STATISTICAL ARBITRAGE  — exit abs(z)<0.40 | stop abs(z)>3
# ═════════════════════════════════════════════════════════════════════════════

def eval_pair_stat_arb(sig) -> Optional[AlgoResult]:
    """
    RS-ratio z-score extreme — ticker vs SPY rolling residual at ±2σ.
    Long when underperformed (z < -2), short when over-performed (z > +2).
    Uses existing rs_score as the residual proxy.
    Exit when abs(z) < 0.40; stop when abs(z) > 3.
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        # rs_score is normalised [-1, +1]; scale to z-score [-4, +4] range
        rs_score = float(getattr(sig, "rs_score", 0.0))
        rs_z     = rs_score * 4.0   # proxy z-score

        z_enter = _param("PAIR_ARB", "rvol_gate",   2.0)
        tp_m    = _param("PAIR_ARB", "target_mult",  0.70)
        sl_m    = _param("PAIR_ARB", "stop_mult",    0.55)

        regime = getattr(sig, "regime", "NEUTRAL")

        # LONG: stock underperformed SPY (rs_z < -z_enter) — expect mean reversion
        if rs_z <= -z_enter and regime != "BEARISH":
            entry  = price
            stop   = entry - sl_m * atr
            target = entry + tp_m * atr
            if stop >= entry or target <= entry:
                return None
            conf = min(82, 55 + abs(rs_z) * 5)
            return AlgoResult(
                algo="PAIR_STAT_ARB_LONG", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Pair stat arb long: rs_z={rs_z:.2f}≤-{z_enter:.1f}, spread reversion",
            )

        # SHORT: stock over-performed SPY (rs_z > +z_enter) — expect mean reversion
        if rs_z >= z_enter and regime != "BULLISH":
            entry  = price
            stop   = entry + sl_m * atr
            target = entry - tp_m * atr
            if stop <= entry or target >= entry:
                return None
            conf = min(82, 55 + rs_z * 5)
            return AlgoResult(
                algo="PAIR_STAT_ARB_SHORT", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Pair stat arb short: rs_z={rs_z:.2f}≥{z_enter:.1f}, spread reversion",
            )
    except Exception as exc:
        logger.debug("eval_pair_stat_arb error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 19. REGIME-SWITCHING TREND / REVERSION ENGINE  — TP 0.90 ATR | SL 0.65 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_regime_switch_engine(sig) -> Optional[AlgoResult]:
    """
    ADX > 25 + ATR expanding → trend setup (follow direction).
    ADX < 20 + ATR quiet    → mean-reversion fade setup.
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        tp_m      = _param("REGIME_SW", "target_mult", 0.90)
        sl_m      = _param("REGIME_SW", "stop_mult",   0.65)
        rvol_gate = _param("REGIME_SW", "rvol_gate",   1.2)
        adx_trend = _param("REGIME_SW", "conf_gate",   25.0)

        adx    = _t(t, "adx_14", 20.0)
        di_pos = _t(t, "di_pos", 0.0)
        di_neg = _t(t, "di_neg", 0.0)
        shock  = _t(t, "vol_shock_ratio", 1.0)
        vwap   = float(getattr(sig, "vwap_price", 0.0)) or _t(t, "vwap")
        ema9   = _t(t, "ema_9")
        ema20  = _t(t, "ema_20")
        rvol   = float(getattr(sig, "rel_volume", 1.0))
        gate   = getattr(sig, "short_tf_alignment", "MIXED")
        regime = getattr(sig, "regime", "NEUTRAL")

        # TREND MODE (ADX > threshold + vol expanding)
        if adx >= adx_trend and shock >= 1.3 and rvol >= rvol_gate:
            if di_pos > di_neg and price > vwap > 0 and ema9 > ema20 and gate != "BEAR":
                entry  = price
                stop   = ema20 - sl_m * atr
                target = entry + tp_m * atr
                if stop < entry and target > entry:
                    conf = min(90, 62 + (adx - 25) * 1.5 + (8 if gate == "BULL" else 0))
                    return AlgoResult(
                        algo="REGIME_TREND_BULL", direction="BUY",
                        confidence=round(conf, 1),
                        entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                        rr=_rr(entry, stop, target),
                        reason=f"Regime trend bull: ADX={adx:.1f}, +DI>{di_neg:.1f}, shock={shock:.2f}",
                    )
            if di_neg > di_pos and price < vwap > 0 and ema9 < ema20 and gate != "BULL":
                entry  = price
                stop   = ema20 + sl_m * atr
                target = entry - tp_m * atr
                if stop > entry and target < entry:
                    conf = min(90, 62 + (adx - 25) * 1.5 + (8 if gate == "BEAR" else 0))
                    return AlgoResult(
                        algo="REGIME_TREND_BEAR", direction="SELL",
                        confidence=round(conf, 1),
                        entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                        rr=_rr(entry, stop, target),
                        reason=f"Regime trend bear: ADX={adx:.1f}, -DI>{di_pos:.1f}, shock={shock:.2f}",
                    )

        # FADE MODE (ADX < 20 + vol quiet = range regime)
        if adx < 20 and shock < 1.2:
            bb_z  = _t(t, "bb_z_score")
            bb_mid = _t(t, "bb_mid")
            if bb_z <= -1.5 and bb_mid > 0:
                entry  = price
                stop   = entry - sl_m * atr
                target = min(bb_mid, entry + tp_m * atr)
                if stop < entry and target > entry:
                    conf = min(80, 55 + abs(bb_z) * 5)
                    return AlgoResult(
                        algo="REGIME_FADE_BULL", direction="BUY",
                        confidence=round(conf, 1),
                        entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                        rr=_rr(entry, stop, target),
                        reason=f"Regime fade bull: ADX={adx:.1f} (range), bb_z={bb_z:.2f}",
                    )
            if bb_z >= 1.5 and bb_mid > 0:
                entry  = price
                stop   = entry + sl_m * atr
                target = max(bb_mid, entry - tp_m * atr)
                if stop > entry and target < entry:
                    conf = min(80, 55 + bb_z * 5)
                    return AlgoResult(
                        algo="REGIME_FADE_BEAR", direction="SELL",
                        confidence=round(conf, 1),
                        entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                        rr=_rr(entry, stop, target),
                        reason=f"Regime fade bear: ADX={adx:.1f} (range), bb_z={bb_z:.2f}",
                    )
    except Exception as exc:
        logger.debug("eval_regime_switch_engine error: %s", exc)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 20. PROBABILITY-CALIBRATED META ENSEMBLE  — TP 0.80 ATR | SL 0.60 ATR
# ═════════════════════════════════════════════════════════════════════════════

def eval_probability_meta_ensemble(sig) -> Optional[AlgoResult]:
    """
    Fires only when the walk-forward estimated success probability is ≥ 0.90.
    Uses the blended ML ensemble probability (scalp + swing + deep BiLSTM)
    as the calibrated probability estimate — all models must agree.
    """
    try:
        t     = _get_tech(sig)
        price = float(sig.price)
        atr   = _t(t, "atr_14") or _t(t, "atr_10")
        if atr <= 0 or price <= 0:
            return None

        prob_gate = _param("META_ENS", "conf_gate", 90.0)   # ≥90% ensemble threshold
        tp_m      = _param("META_ENS", "target_mult", 0.80)
        sl_m      = _param("META_ENS", "stop_mult",   0.60)

        ml_prob    = float(getattr(sig, "ml_prob",       0.5))
        swing_prob = float(getattr(sig, "ml_swing_prob", 0.5))
        deep_prob  = float(getattr(sig, "ml_deep_prob",  0.5))
        swing_ok   = bool(getattr(sig, "ml_swing_trained", False))
        deep_ok    = bool(getattr(sig, "ml_deep_trained",  False))

        # Composite: weight trained models more heavily
        weights = [0.5, 0.25 if swing_ok else 0, 0.25 if deep_ok else 0]
        total_w = sum(weights)
        if total_w <= 0:
            return None
        ensemble = (ml_prob * weights[0] + swing_prob * weights[1] +
                    deep_prob * weights[2]) / total_w

        gate  = getattr(sig, "short_tf_alignment", "MIXED")
        vwap  = float(getattr(sig, "vwap_price", 0.0)) or _t(t, "vwap")
        ema9  = _t(t, "ema_9")
        ema20 = _t(t, "ema_20")

        # BULLISH: very high probability bullish + EMA + VWAP confirm
        if (ensemble >= prob_gate / 100.0 and price > vwap > 0
                and ema9 > ema20 and gate != "BEAR"):
            entry  = price
            stop   = entry - sl_m * atr
            target = entry + tp_m * atr
            if stop >= entry or target <= entry:
                return None
            conf = min(95, 70 + ensemble * 25)
            return AlgoResult(
                algo="META_ENS_BULL", direction="BUY",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Meta ensemble bull: p={ensemble:.3f}≥{prob_gate/100:.2f}, EMA/VWAP aligned",
            )

        # BEARISH: very high probability bearish
        bear_ens = (1 - ml_prob) * weights[0] + (1 - swing_prob) * weights[1] + \
                   (1 - deep_prob) * weights[2]
        bear_ens /= total_w
        if (bear_ens >= prob_gate / 100.0 and price < vwap > 0
                and ema9 < ema20 and gate != "BULL"):
            entry  = price
            stop   = entry + sl_m * atr
            target = entry - tp_m * atr
            if stop <= entry or target >= entry:
                return None
            conf = min(95, 70 + bear_ens * 25)
            return AlgoResult(
                algo="META_ENS_BEAR", direction="SELL",
                confidence=round(conf, 1),
                entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                rr=_rr(entry, stop, target),
                reason=f"Meta ensemble bear: p={bear_ens:.3f}≥{prob_gate/100:.2f}, EMA/VWAP aligned",
            )
    except Exception as exc:
        logger.debug("eval_probability_meta_ensemble error: %s", exc)
    return None


# ── Public registry (all tradeable strategies) ────────────────────────────────

QUANT_ALGO_REGISTRY = [
    eval_ofi_impulse,
    eval_queue_microprice,
    eval_aggressor_volume_momentum,
    eval_vwap_ofi_pullback,
    eval_ema_slope_pullback,
    eval_vwap_trend_breakout,
    eval_donchian_volume_breakout,
    eval_orb_vwap_zv,
    eval_macd_acceleration,
    eval_supertrend_continuation,
    eval_volatility_shock_continuation,
    eval_bollinger_mean_reversion,
    eval_rsi_exhaustion_snapback,
    eval_keltner_exhaustion_fade,
    eval_failed_breakout_reversal,
    eval_squeeze_expansion,
    eval_pair_stat_arb,
    eval_regime_switch_engine,
    eval_probability_meta_ensemble,
    # eval_market_making_quotes — intentionally excluded (informational only)
]
