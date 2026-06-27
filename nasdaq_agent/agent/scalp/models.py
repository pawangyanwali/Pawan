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
    max_quote_age_ms: int = 2_000
    max_bar_age_ms: int = 120_000
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
    allow_rest_fallback_trading: bool = False
    block_when_path_obstructed: bool = True
    block_when_risk_capped: bool = True

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
        if self.max_spread_to_risk <= 0:
            raise ValueError("max_spread_to_risk must be positive")
        if self.min_rvol_regular < 0 or self.min_rvol_extended < 0:
            raise ValueError("RVOL thresholds cannot be negative")
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
            max_quote_age_ms=int(read("scalp.max_quote_age_ms", 2_000)),
            max_bar_age_ms=int(read("scalp.max_bar_age_ms", 120_000)),
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
            allow_rest_fallback_trading=bool(
                read("scalp.allow_rest_fallback_trading", False)
            ),
            block_when_path_obstructed=bool(
                read("scalp.block_when_path_obstructed", True)
            ),
            block_when_risk_capped=bool(read("scalp.block_when_risk_capped", True)),
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
    schema_version: int = 1
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
    session: str = ""
    setup_score: float = 0.0
    confidence: float = 0.0
    base_confidence: float = 0.0
    ml_tp1_probability: float = 0.0
    ml_tp2_probability: float = 0.0
    ml_expected_r: float = 0.0
    ml_confidence_adjustment: float = 0.0
    ml_model_version: str = ""
    ml_overlay_applied: bool = False
    learned_expectancy_r: float = 0.0
    learned_win_rate: float = 0.0
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
    reasons: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["side"] = self.side.value
        result["source"] = self.source.value
        result["tp2_path"] = self.tp2_path.value
        return result
