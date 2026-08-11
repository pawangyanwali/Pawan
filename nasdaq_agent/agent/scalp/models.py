from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol
from uuid import uuid4

from ._utils import finite, positive


class SignalSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class QuoteSource(str, Enum):
    WS = "WS"
    REST = "REST"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class PathQuality(str, Enum):
    CLEAR = "CLEAR"
    BLOCKED_BY_RESISTANCE = "BLOCKED_BY_RESISTANCE"
    BLOCKED_BY_SUPPORT = "BLOCKED_BY_SUPPORT"
    UNKNOWN = "UNKNOWN"


class ConfigReader(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...


@dataclass(frozen=True)
class ScalpSignalConfig:
    reward_r: float = 2.0
    tp1_r: float = 1.0
    stop_atr_multiple: float = 1.0
    min_stop_pct: float = 0.003
    max_stop_pct: float = 0.020
    spread_buffer_mult: float = 2.0
    tick_size: float = 0.01
    max_quote_age_ms: int = 5_000
    max_bar_age_ms: int = 120_000
    use_provisional_live_indicators: bool = True
    provisional_max_bar_age_ms: int = 300_000
    max_spread_to_risk: float = 0.25
    min_rvol_regular: float = 0.8
    min_rvol_extended: float = 0.4
    rsi_oversold: float = 30.0
    rsi_extreme_oversold: float = 20.0
    rsi_overbought: float = 70.0
    rsi_extreme_overbought: float = 80.0
    require_vwap_event: bool = True
    require_macd_confirm: bool = True
    require_rsi_zone: bool = True
    long_require_fast_rsi_confirmation: bool = True
    long_require_vwap_reclaim: bool = True
    long_require_mtf_not_bearish: bool = True
    long_block_bearish_market: bool = True
    short_require_fast_rsi_confirmation: bool = True
    short_premarket_require_vwap_rejection: bool = True
    short_require_mtf_not_bullish: bool = True
    short_block_bullish_market: bool = True
    allow_rest_fallback_trading: bool = False
    block_when_path_obstructed: bool = True
    block_when_risk_capped: bool = True
    mtf_enabled: bool = True
    mtf_mode: str = "SHADOW"
    mtf_max_bar_age_ms: int = 420_000
    momentum_shadow_enabled: bool = True
    reversal_shadow_enabled: bool = True
    momentum_long_rsi_min: float = 45.0
    momentum_long_rsi_max: float = 70.0
    momentum_short_rsi_min: float = 30.0
    momentum_short_rsi_max: float = 55.0

    def __post_init__(self) -> None:
        if self.reward_r <= 1.0:
            raise ValueError("reward_r must be greater than 1.0")
        if not 0 < self.tp1_r <= self.reward_r:
            raise ValueError("tp1_r must be positive and no greater than reward_r")
        if self.stop_atr_multiple <= 0:
            raise ValueError("stop_atr_multiple must be positive")
        if not 0 < self.min_stop_pct <= self.max_stop_pct:
            raise ValueError("stop percentage bounds are invalid")
        if self.spread_buffer_mult < 0:
            raise ValueError("spread_buffer_mult cannot be negative")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if self.max_quote_age_ms <= 0:
            raise ValueError("max_quote_age_ms must be positive")
        if self.max_bar_age_ms <= 0:
            raise ValueError("max_bar_age_ms must be positive")
        if self.provisional_max_bar_age_ms <= 0:
            raise ValueError("provisional_max_bar_age_ms must be positive")
        if self.max_spread_to_risk <= 0:
            raise ValueError("max_spread_to_risk must be positive")
        if self.min_rvol_regular < 0 or self.min_rvol_extended < 0:
            raise ValueError("RVOL thresholds cannot be negative")
        if str(self.mtf_mode).upper() not in {"OFF", "SHADOW"}:
            raise ValueError("mtf_mode must be OFF or SHADOW")
        if self.mtf_max_bar_age_ms <= 0:
            raise ValueError("mtf_max_bar_age_ms must be positive")
        if not 0 <= self.momentum_long_rsi_min < self.momentum_long_rsi_max <= 100:
            raise ValueError("momentum LONG RSI boundaries are invalid")
        if not 0 <= self.momentum_short_rsi_min < self.momentum_short_rsi_max <= 100:
            raise ValueError("momentum SHORT RSI boundaries are invalid")
        if not (
            0 <= self.rsi_extreme_oversold
            <= self.rsi_oversold
            < self.rsi_overbought
            <= self.rsi_extreme_overbought
            <= 100
        ):
            raise ValueError(
                "RSI boundaries must be ordered from extreme oversold to extreme overbought"
            )

    @classmethod
    def from_runtime(cls, config: ConfigReader | Mapping[str, Any]) -> "ScalpSignalConfig":
        def read(key: str, default: Any) -> Any:
            return config.get(key, default)

        return cls(
            reward_r=float(read("scalp.reward_r", 2.0)),
            tp1_r=float(read("scalp.tp1_r", 1.0)),
            stop_atr_multiple=float(read("scalp.stop_atr_multiple", 1.0)),
            min_stop_pct=float(read("scalp.min_stop_pct", 0.003)),
            max_stop_pct=float(read("scalp.max_stop_pct", 0.020)),
            spread_buffer_mult=float(read("scalp.spread_buffer_mult", 2.0)),
            tick_size=float(read("scalp.tick_size", 0.01)),
            max_quote_age_ms=int(read("scalp.max_quote_age_ms", 5_000)),
            max_bar_age_ms=int(read("scalp.max_bar_age_ms", 120_000)),
            use_provisional_live_indicators=bool(
                read("scalp.use_provisional_live_indicators", True)
            ),
            provisional_max_bar_age_ms=int(
                read("scalp.provisional_max_bar_age_ms", 300_000)
            ),
            max_spread_to_risk=float(read("scalp.max_spread_to_risk", 0.25)),
            min_rvol_regular=float(read("scalp.min_rvol_regular", 0.8)),
            min_rvol_extended=float(read("scalp.min_rvol_extended", 0.4)),
            rsi_oversold=float(read("scalp.rsi_oversold", 30.0)),
            rsi_extreme_oversold=float(read("scalp.rsi_extreme_oversold", 20.0)),
            rsi_overbought=float(read("scalp.rsi_overbought", 70.0)),
            rsi_extreme_overbought=float(read("scalp.rsi_extreme_overbought", 80.0)),
            require_vwap_event=bool(read("scalp.require_vwap_event", True)),
            require_macd_confirm=bool(read("scalp.require_macd_confirm", True)),
            require_rsi_zone=bool(read("scalp.require_rsi_zone", True)),
            long_require_fast_rsi_confirmation=bool(
                read("scalp.long_require_fast_rsi_confirmation", True)
            ),
            long_require_vwap_reclaim=bool(
                read("scalp.long_require_vwap_reclaim", True)
            ),
            long_require_mtf_not_bearish=bool(
                read("scalp.long_require_mtf_not_bearish", True)
            ),
            long_block_bearish_market=bool(
                read("scalp.long_block_bearish_market", True)
            ),
            short_require_fast_rsi_confirmation=bool(
                read("scalp.short_require_fast_rsi_confirmation", True)
            ),
            short_premarket_require_vwap_rejection=bool(
                read("scalp.short_premarket_require_vwap_rejection", True)
            ),
            short_require_mtf_not_bullish=bool(
                read("scalp.short_require_mtf_not_bullish", True)
            ),
            short_block_bullish_market=bool(
                read("scalp.short_block_bullish_market", True)
            ),
            allow_rest_fallback_trading=bool(
                read("scalp.allow_rest_fallback_trading", False)
            ),
            block_when_path_obstructed=bool(
                read("scalp.block_when_path_obstructed", True)
            ),
            block_when_risk_capped=bool(read("scalp.block_when_risk_capped", True)),
            mtf_enabled=bool(read("scalp.mtf_enabled", True)),
            mtf_mode=str(read("scalp.mtf_mode", "SHADOW")).upper(),
            mtf_max_bar_age_ms=int(read("scalp.mtf_max_bar_age_ms", 420_000)),
            momentum_shadow_enabled=bool(
                read("scalp.momentum_shadow_enabled", True)
            ),
            reversal_shadow_enabled=bool(
                read("scalp.reversal_shadow_enabled", True)
            ),
            momentum_long_rsi_min=float(
                read("scalp.momentum_long_rsi_min", 45.0)
            ),
            momentum_long_rsi_max=float(
                read("scalp.momentum_long_rsi_max", 70.0)
            ),
            momentum_short_rsi_min=float(
                read("scalp.momentum_short_rsi_min", 30.0)
            ),
            momentum_short_rsi_max=float(
                read("scalp.momentum_short_rsi_max", 55.0)
            ),
        )


@dataclass(frozen=True)
class QuoteSnapshot:
    ticker: str
    last: float
    bid: float
    ask: float
    data_age_ms: int
    source: QuoteSource | str

    @property
    def normalized_source(self) -> QuoteSource:
        if isinstance(self.source, QuoteSource):
            return self.source
        try:
            return QuoteSource(str(self.source).upper())
        except ValueError:
            return QuoteSource.UNKNOWN

    @property
    def spread(self) -> float:
        if positive(self.ask) and positive(self.bid) and self.ask >= self.bid:
            return self.ask - self.bid
        return 0.0


@dataclass(frozen=True)
class IndicatorSnapshot:
    rsi_14: float | None
    rsi_7: float | None
    rsi_2: float | None
    macd_hist: float | None
    macd_hist_prev: float | None
    atr_14: float | None
    vwap: float | None
    rvol: float | None
    vwap_event: str = ""
    bar_age_ms: int | None = 0
    indicator_close: float | None = None
    rsi_avg_gain_14: float | None = None
    rsi_avg_loss_14: float | None = None
    rsi_avg_gain_7: float | None = None
    rsi_avg_loss_7: float | None = None
    rsi_avg_gain_2: float | None = None
    rsi_avg_loss_2: float | None = None
    macd_fast_ema: float | None = None
    macd_slow_ema: float | None = None
    macd_signal_ema: float | None = None

    @property
    def macd_slope(self) -> float | None:
        if not finite(self.macd_hist) or not finite(self.macd_hist_prev):
            return None
        return float(self.macd_hist) - float(self.macd_hist_prev)


@dataclass(frozen=True)
class MultiTimeframeSnapshot:
    timeframe: str = "5m"
    state: str = "NO_DATA"
    close: float | None = None
    rsi_14: float | None = None
    macd_hist: float | None = None
    macd_hist_prev: float | None = None
    macd_slope: float | None = None
    atr_14: float | None = None
    vwap: float | None = None
    vwap_event: str = ""
    ema_fast: float | None = None
    ema_slow: float | None = None
    bar_age_ms: int | None = None
    bar_closed_at_ms: int | None = None
    completed_bars: int = 0


@dataclass(frozen=True)
class BracketGeometry:
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    risk_per_share: float
    required_risk: float
    reward_r: float
    rr_ratio: float
    risk_capped: bool


@dataclass
class ScalpSignalPlan:
    ticker: str
    side: SignalSide
    valid: bool
    invalid_reason: str
    plan_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: int = 2
    entry: float = 0.0
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    risk_per_share: float = 0.0
    reward_r: float = 0.0
    rr_ratio: float = 0.0
    tp2_path: PathQuality = PathQuality.UNKNOWN
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread_bps: float = 0.0
    spread_to_risk: float = 0.0
    data_age_ms: int = 0
    bar_age_ms: int = 0
    source: QuoteSource = QuoteSource.UNKNOWN
    execution_eligible: bool = False
    execution_blockers: list[str] = field(default_factory=list)
    execution_liquidity_qualified: bool = False
    execution_median_minute_dollar_volume: float = 0.0
    rsi_14: float = 0.0
    rsi_7: float = 0.0
    rsi_2: float = 0.0
    rsi_zone: str = "UNKNOWN"
    macd_hist: float = 0.0
    macd_hist_prev: float = 0.0
    macd_slope: float = 0.0
    indicator_close: float = 0.0
    rsi_avg_gain_14: float = 0.0
    rsi_avg_loss_14: float = 0.0
    rsi_avg_gain_7: float = 0.0
    rsi_avg_loss_7: float = 0.0
    rsi_avg_gain_2: float = 0.0
    rsi_avg_loss_2: float = 0.0
    macd_fast_ema: float = 0.0
    macd_slow_ema: float = 0.0
    macd_signal_ema: float = 0.0
    atr_14: float = 0.0
    atr_bucket: str = "UNKNOWN"
    vwap: float = 0.0
    vwap_event: str = ""
    rvol: float = 0.0
    setup_type: str = ""
    strategy_family: str = "NONE"
    session: str = ""
    setup_score: float = 0.0
    confidence: float = 0.0
    base_confidence: float = 0.0
    entry_quality_assessed: bool = False
    entry_quality_score: float = 0.0
    entry_quality_min_score: float = 0.0
    entry_quality_gate: str = "NOT_ASSESSED"
    entry_quality_reasons: list[str] = field(default_factory=list)
    entry_confirmation_state: str = "NOT_EVALUATED"
    entry_confirmation_age_s: float = 0.0
    entry_confirmation_observations: int = 0
    entry_expected_r: float = 0.0
    entry_expected_r_source: str = "NONE"
    ml_tp1_probability: float = 0.0
    ml_tp2_probability: float = 0.0
    ml_expected_r: float = 0.0
    ml_confidence_adjustment: float = 0.0
    ml_model_version: str = ""
    ml_overlay_applied: bool = False
    learned_expectancy_r: float = 0.0
    learned_win_rate: float = 0.0
    learning_sample_count: int = 0
    learning_mean_expectancy_r: float = 0.0
    learning_context_scope: str = "NONE"
    context_key: str = ""
    learning_gate: str = "ALLOW"
    learning_size_mult: float = 1.0
    learning_confidence_floor: float = 0.0
    learning_action_expires_at: str = ""
    context_fresh: bool = False
    sentiment_30m: float = 0.0
    sentiment_velocity: float = 0.0
    news_shock: bool = False
    context_risk_score: float = 0.0
    earnings_phase: str = ""
    earnings_next_date: str = ""
    earnings_days_away: int = 999
    recent_headlines: list[str] = field(default_factory=list)
    mtf_mode: str = "OFF"
    mtf_state: str = "NO_DATA"
    mtf_alignment: str = "NO_DATA"
    mtf_bar_age_ms: int = -1
    rsi_14_5m: float = 0.0
    macd_hist_5m: float = 0.0
    macd_hist_prev_5m: float = 0.0
    macd_slope_5m: float = 0.0
    atr_14_5m: float = 0.0
    vwap_5m: float = 0.0
    vwap_event_5m: str = ""
    ema_fast_5m: float = 0.0
    ema_slow_5m: float = 0.0
    shadow_strategy_family: str = ""
    shadow_side: str = "NONE"
    shadow_setup_ready: bool = False
    shadow_setup_score: float = 0.0
    shadow_reasons: list[str] = field(default_factory=list)
    shadow_blockers: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["side"] = self.side.value
        result["source"] = self.source.value
        result["tp2_path"] = self.tp2_path.value
        return result
