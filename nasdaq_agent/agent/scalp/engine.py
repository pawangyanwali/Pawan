from __future__ import annotations

from typing import Iterable

from ._utils import finite, number, positive
from .bracket import build_bracket, normalize_side
from .models import (
    BracketGeometry,
    IndicatorSnapshot,
    PathQuality,
    QuoteSnapshot,
    QuoteSource,
    ScalpSignalConfig,
    ScalpSignalPlan,
    SignalSide,
)


def create_scalp_signal_plan(
    *,
    quote: QuoteSnapshot,
    indicators: IndicatorSnapshot,
    side: SignalSide | str,
    session: str,
    config: ScalpSignalConfig | None = None,
    supports: Iterable[float] = (),
    resistances: Iterable[float] = (),
    learned_expectancy_r: float = 0.0,
    learned_win_rate: float = 0.0,
) -> ScalpSignalPlan:
    """Create one complete plan; invalid inputs produce explicit blockers."""
    cfg = config or ScalpSignalConfig()
    normalized_side = normalize_side(side)
    source = quote.normalized_source
    blockers: list[str] = []
    reasons: list[str] = []

    if normalized_side is SignalSide.NONE:
        blockers.append("NO_DIRECTIONAL_SETUP")
    if not positive(quote.last):
        blockers.append("LAST_PRICE_MISSING")
    if not positive(quote.bid):
        blockers.append("BID_MISSING")
    if not positive(quote.ask):
        blockers.append("ASK_MISSING")
    if positive(quote.bid) and positive(quote.ask) and quote.ask < quote.bid:
        blockers.append("CROSSED_QUOTE")
    if quote.data_age_ms < 0 or quote.data_age_ms > cfg.max_quote_age_ms:
        blockers.append("QUOTE_STALE")
    if source in {QuoteSource.STALE, QuoteSource.UNKNOWN}:
        blockers.append("QUOTE_SOURCE_NOT_LIVE")
    if source is QuoteSource.REST and not cfg.allow_rest_fallback_trading:
        blockers.append("REST_FALLBACK_NOT_TRADABLE")

    blockers.extend(
        f"{name}_MISSING" for name in _missing_indicator_names(indicators)
    )
    if indicators.bar_age_ms is None:
        blockers.append("INDICATOR_BAR_TIMESTAMP_MISSING")
    elif indicators.bar_age_ms < 0 or indicators.bar_age_ms > cfg.max_bar_age_ms:
        blockers.append("INDICATOR_BAR_STALE")

    entry = 0.0
    bracket: BracketGeometry | None = None
    if normalized_side is SignalSide.LONG:
        entry = quote.ask if positive(quote.ask) else quote.last
    elif normalized_side is SignalSide.SHORT:
        entry = quote.bid if positive(quote.bid) else quote.last

    if positive(entry) and positive(indicators.atr_14):
        try:
            bracket = build_bracket(
                entry=entry,
                side=normalized_side,
                atr_14=float(indicators.atr_14),
                spread=quote.spread,
                config=cfg,
            )
        except ValueError:
            blockers.append("BRACKET_GEOMETRY_INVALID")
        if bracket and bracket.risk_capped and cfg.block_when_risk_capped:
            blockers.append("REQUIRED_STOP_EXCEEDS_MAX_RISK")

    rsi_zone = _rsi_zone(indicators.rsi_14, cfg)
    vwap_event = str(indicators.vwap_event or "").upper()
    setup_type = ""
    if normalized_side is SignalSide.LONG:
        setup_type = "OVERSOLD_MACD_TURN_LONG"
        _evaluate_long_setup(indicators, rsi_zone, vwap_event, cfg, reasons, blockers)
    elif normalized_side is SignalSide.SHORT:
        setup_type = "OVERBOUGHT_MACD_TURN_SHORT"
        _evaluate_short_setup(indicators, rsi_zone, vwap_event, cfg, reasons, blockers)

    min_rvol = (
        cfg.min_rvol_extended
        if str(session).upper() in {"PRE_MARKET", "AFTER_HOURS", "EXTENDED"}
        else cfg.min_rvol_regular
    )
    if finite(indicators.rvol) and float(indicators.rvol) < min_rvol:
        blockers.append("RVOL_BELOW_SESSION_MINIMUM")
    elif finite(indicators.rvol):
        reasons.append("RVOL_CONFIRMED")

    spread_to_risk = 0.0
    path = PathQuality.UNKNOWN
    if bracket:
        spread_to_risk = (
            quote.spread / bracket.risk_per_share if bracket.risk_per_share else 0.0
        )
        if spread_to_risk > cfg.max_spread_to_risk:
            blockers.append("SPREAD_TOO_WIDE_FOR_RISK")
        else:
            reasons.append("SPREAD_ACCEPTABLE")
        path = _tp2_path(
            normalized_side,
            bracket.entry,
            bracket.tp2,
            supports=supports,
            resistances=resistances,
        )
        if cfg.block_when_path_obstructed and path in {
            PathQuality.BLOCKED_BY_RESISTANCE,
            PathQuality.BLOCKED_BY_SUPPORT,
        }:
            blockers.append(path.value)
        elif path is PathQuality.CLEAR:
            reasons.append("TP2_PATH_CLEAR")

    blockers = list(dict.fromkeys(blockers))
    reasons = list(dict.fromkeys(reasons))
    valid = not blockers and bracket is not None
    setup_score = _setup_score(reasons, blockers)

    return ScalpSignalPlan(
        ticker=quote.ticker.upper(),
        side=normalized_side,
        valid=valid,
        invalid_reason=blockers[0] if blockers else "",
        entry=bracket.entry if bracket else round(entry, 4),
        stop_loss=bracket.stop_loss if bracket else 0.0,
        tp1=bracket.tp1 if bracket else 0.0,
        tp2=bracket.tp2 if bracket else 0.0,
        risk_per_share=bracket.risk_per_share if bracket else 0.0,
        reward_r=bracket.reward_r if bracket else cfg.reward_r,
        rr_ratio=bracket.rr_ratio if bracket else 0.0,
        tp2_path=path,
        price=number(quote.last),
        bid=number(quote.bid),
        ask=number(quote.ask),
        spread_bps=(
            round((quote.spread / quote.last) * 10_000, 2)
            if positive(quote.last)
            else 0.0
        ),
        spread_to_risk=round(spread_to_risk, 4),
        data_age_ms=quote.data_age_ms,
        bar_age_ms=indicators.bar_age_ms if indicators.bar_age_ms is not None else -1,
        source=source,
        rsi_14=number(indicators.rsi_14),
        rsi_7=number(indicators.rsi_7),
        rsi_2=number(indicators.rsi_2),
        rsi_zone=rsi_zone,
        macd_hist=number(indicators.macd_hist),
        macd_hist_prev=number(indicators.macd_hist_prev),
        macd_slope=number(indicators.macd_slope),
        atr_14=number(indicators.atr_14),
        atr_bucket=_atr_bucket(indicators.atr_14, quote.last),
        vwap=number(indicators.vwap),
        vwap_event=vwap_event,
        rvol=number(indicators.rvol),
        setup_type=setup_type,
        session=str(session or "").upper(),
        setup_score=setup_score,
        confidence=setup_score,
        base_confidence=setup_score,
        learned_expectancy_r=float(learned_expectancy_r),
        learned_win_rate=float(learned_win_rate),
        reasons=reasons,
        blockers=blockers,
    )


def _atr_bucket(atr_14: float | None, price: float | None) -> str:
    if not positive(atr_14) or not positive(price):
        return "UNKNOWN"
    atr_pct = float(atr_14) / float(price) * 100.0
    if atr_pct <= 0.5:
        return "LOW"
    if atr_pct <= 1.5:
        return "NORMAL"
    return "HIGH"


def detect_scalp_signal_plan(
    *,
    quote: QuoteSnapshot,
    indicators: IndicatorSnapshot,
    session: str,
    config: ScalpSignalConfig | None = None,
    supports: Iterable[float] = (),
    resistances: Iterable[float] = (),
    learned_expectancy_r: float = 0.0,
    learned_win_rate: float = 0.0,
) -> ScalpSignalPlan:
    """Infer an oversold/overbought side without inventing neutral trades."""
    cfg = config or ScalpSignalConfig()
    zone = _rsi_zone(indicators.rsi_14, cfg)
    if zone in {"OS", "EXTREME_OS"}:
        side = SignalSide.LONG
    elif zone in {"OB", "EXTREME_OB"}:
        side = SignalSide.SHORT
    else:
        side = SignalSide.NONE
    return create_scalp_signal_plan(
        quote=quote,
        indicators=indicators,
        side=side,
        session=session,
        config=cfg,
        supports=supports,
        resistances=resistances,
        learned_expectancy_r=learned_expectancy_r,
        learned_win_rate=learned_win_rate,
    )


def _evaluate_long_setup(
    indicators: IndicatorSnapshot,
    rsi_zone: str,
    vwap_event: str,
    config: ScalpSignalConfig,
    reasons: list[str],
    blockers: list[str],
) -> None:
    if config.require_rsi_zone and rsi_zone not in {"OS", "EXTREME_OS"}:
        blockers.append("LONG_RSI_NOT_OVERSOLD")
    else:
        reasons.append(f"RSI_{rsi_zone}")
    if config.require_macd_confirm and not _long_macd_confirmed(indicators):
        blockers.append("LONG_MACD_NOT_RISING")
    else:
        reasons.append("MACD_RISING")
    if config.require_vwap_event and vwap_event not in {
        "RECLAIM", "ABOVE", "BOUNCE_SUPPORT"
    }:
        blockers.append("LONG_VWAP_RECLAIM_MISSING")
    else:
        reasons.append(f"VWAP_{vwap_event or 'CONFIRMED'}")


def _evaluate_short_setup(
    indicators: IndicatorSnapshot,
    rsi_zone: str,
    vwap_event: str,
    config: ScalpSignalConfig,
    reasons: list[str],
    blockers: list[str],
) -> None:
    if config.require_rsi_zone and rsi_zone not in {"OB", "EXTREME_OB"}:
        blockers.append("SHORT_RSI_NOT_OVERBOUGHT")
    else:
        reasons.append(f"RSI_{rsi_zone}")
    if config.require_macd_confirm and not _short_macd_confirmed(indicators):
        blockers.append("SHORT_MACD_NOT_FALLING")
    else:
        reasons.append("MACD_FALLING")
    if config.require_vwap_event and vwap_event not in {
        "REJECTION", "BELOW", "REJECT_RESISTANCE"
    }:
        blockers.append("SHORT_VWAP_REJECTION_MISSING")
    else:
        reasons.append(f"VWAP_{vwap_event or 'CONFIRMED'}")


def _missing_indicator_names(indicators: IndicatorSnapshot) -> list[str]:
    required = {
        "RSI_14": indicators.rsi_14,
        "RSI_7": indicators.rsi_7,
        "RSI_2": indicators.rsi_2,
        "MACD_HIST": indicators.macd_hist,
        "MACD_HIST_PREV": indicators.macd_hist_prev,
        "ATR_14": indicators.atr_14,
        "VWAP": indicators.vwap,
        "RVOL": indicators.rvol,
    }
    missing = [name for name, value in required.items() if not finite(value)]
    if finite(indicators.atr_14) and float(indicators.atr_14) <= 0:
        missing.append("ATR_14")
    if finite(indicators.vwap) and float(indicators.vwap) <= 0:
        missing.append("VWAP")
    return list(dict.fromkeys(missing))


def _rsi_zone(value: float | None, config: ScalpSignalConfig) -> str:
    if not finite(value):
        return "UNKNOWN"
    rsi = float(value)
    if rsi <= config.rsi_extreme_oversold:
        return "EXTREME_OS"
    if rsi <= config.rsi_oversold:
        return "OS"
    if rsi >= config.rsi_extreme_overbought:
        return "EXTREME_OB"
    if rsi >= config.rsi_overbought:
        return "OB"
    return "NEUTRAL"


def _long_macd_confirmed(indicators: IndicatorSnapshot) -> bool:
    if not finite(indicators.macd_hist) or not finite(indicators.macd_hist_prev):
        return False
    current = float(indicators.macd_hist)
    previous = float(indicators.macd_hist_prev)
    return current > previous or (previous <= 0 < current)


def _short_macd_confirmed(indicators: IndicatorSnapshot) -> bool:
    if not finite(indicators.macd_hist) or not finite(indicators.macd_hist_prev):
        return False
    current = float(indicators.macd_hist)
    previous = float(indicators.macd_hist_prev)
    return current < previous or (previous >= 0 > current)


def _tp2_path(
    side: SignalSide,
    entry: float,
    tp2: float,
    *,
    supports: Iterable[float],
    resistances: Iterable[float],
) -> PathQuality:
    support_levels = [float(v) for v in supports if positive(v)]
    resistance_levels = [float(v) for v in resistances if positive(v)]
    if side is SignalSide.LONG:
        if any(entry < level < tp2 for level in resistance_levels):
            return PathQuality.BLOCKED_BY_RESISTANCE
        return PathQuality.CLEAR if resistance_levels else PathQuality.UNKNOWN
    if side is SignalSide.SHORT:
        if any(tp2 < level < entry for level in support_levels):
            return PathQuality.BLOCKED_BY_SUPPORT
        return PathQuality.CLEAR if support_levels else PathQuality.UNKNOWN
    return PathQuality.UNKNOWN


def _setup_score(reasons: list[str], blockers: list[str]) -> float:
    if blockers:
        return max(0.0, round(50.0 - 10.0 * len(blockers), 1))
    return min(100.0, round(50.0 + 8.0 * len(reasons), 1))
