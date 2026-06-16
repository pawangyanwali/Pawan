"""
Main scanning engine — Schwab Market Data API (no per-symbol credit cost).

Universe: ~500 NASDAQ tickers tracked via Schwab /quotes bulk screening.
Active scan: Tier 1 (100 core) always + top 75 active from Tier 2/3 = ~175
per cycle.  Price history cached per ticker (TTL 300 s) → ~35 new API calls
per minute for history, well within the 90 req/min Schwab rate limit.
"""

import logging
import os
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Callable

import numpy as np
import pandas as pd

from config import (
    NASDAQ_TICKERS,
    TRAINING_TICKERS,
    SCAN_INTERVAL_SECONDS,
    ML_RETRAIN_INTERVAL,
    DEEP_FINETUNE_INTERVAL,
    CACHE_TTL_5M,
    CACHE_TTL_1H,
    CACHE_TTL_1D,
    REGIME_TICKERS,
    SECTOR_ETF_TICKERS,
    PIPELINE_WORKERS,
    get_active_tickers,
)
from agent.data_fetcher import (
    fetch_batch_realtime,
    fetch_batch_interval,
    fetch_ticker_info,
)
from agent.technical import compute_indicators, score_technical
from agent.volume import score_volume, relative_volume, detect_unusual_volume, rvol_time_of_day
from agent.ml_model import predict, predict_daily, predict_reversal, predict_ensemble, predict_swing, get_or_create_swing, retrain_all
from agent.sentiment import score_sentiment
from agent.prediction import generate_prediction
from agent.support_resistance import (
    calculate_camarilla_pivots,
    calculate_fibonacci_levels,
    calculate_value_area,
)
from agent.mtf_analysis import multi_timeframe_analysis
from agent.market_hours import get_session_info, get_market_session, confidence_multiplier
from agent.market_regime import update_regime, get_regime, apply_regime, classify_day_type
from agent.opening_range import compute_opening_range
from agent.earnings import earnings_blackout
from agent.gap_analysis import analyse_gap, detect_color_flip
from agent.relative_strength import compute_relative_strength
from agent.trade_management import build_trade_plan
from agent.signal_tracker import init_db, record_signal, resolve_pending, record_signals_batch, resolve_short_term, get_ticker_learning_scores
from agent.vwap import compute_vwap_signal
from agent.sector_etf import get_sector_context, update_etf_cache
from agent.trading_algos import evaluate_all as evaluate_trading_algos, detect_flag
from agent.exit_signals import analyse_exits
from agent.paper_trading import init_db as pt_init_db, maybe_open_trade, update_open_trades, rt_check_positions as pt_rt_check, log_algo_signals, get_execution_min_rr
from agent.macro_calendar import check_macro_event
from agent.live_backtest import (
    init_db as bt_init_db,
    record_signal as bt_record,
    update_tracking as bt_update,
    rt_check_resolution as bt_rt_check,
)
from agent.backtest_reporter import maybe_trigger_feedback_retrain, adjust_confidence
from agent.adaptive_filter import (
    get_confidence_boost,
    should_suppress,
)
from agent.ensemble_model import get_meta_prediction
from agent.deep_model import predict_deep
from agent.risk_controls import check_circuit_breaker, check_sector_concentration, check_max_daily_trades, update_volatility_state
from agent.order_flow import compute_order_flow, get_signal_strength
from agent.after_hours_monitor import (
    init_db as ah_init_db,
    record_snapshot as ah_record,
    get_opening_bias,
    get_ah_context_key,
)
from agent.trading_hours import get_trading_tier, is_signal_recommended

try:
    from agent.algo_learning_engine import (
        get_engine as _get_ale,
        get_selector_weights as _get_sel_weights,
        get_algo_params as _get_algo_params,
    )
    _ALE_AVAILABLE = True
except ImportError:
    _ALE_AVAILABLE = False

logger = logging.getLogger(__name__)


def _pipeline_workers() -> int:
    """Scan concurrency cap — the de-facto ML-inference CPU bound.

    Read from PostgreSQL (scanner.pipeline_workers) so it can be dialed down
    live on a smaller host without a redeploy; falls back to the PIPELINE_WORKERS
    constant if config is unreachable. Clamped 1–32 as a safety rail.
    """
    try:
        from agent.config_manager import config as _cfg
        n = int(_cfg.get("scanner.pipeline_workers", PIPELINE_WORKERS))
        return max(1, min(32, n))
    except Exception:
        return PIPELINE_WORKERS


def _scanner_training_enabled() -> bool:
    """
    Heavy model training must not compete with the scanner in production.

    The dedicated learner containers own retraining.  This flag exists only for
    local/legacy monolith runs that intentionally want scanner-side training.
    """
    return os.getenv("NASDAQ_SCANNER_TRAINING_ENABLED", "0").lower() in ("1", "true", "yes")

# Per-signal consecutive-fire counter for entry_window_bars staleness detection.
# Keyed by "TICKER:ALGO_NAME". Incremented each cycle the signal fires; deleted
# when the signal stops firing so the next fire starts fresh.
_sig_consec: dict[str, int] = {}
_sig_consec_lock = threading.Lock()


def _scan_interval() -> int:
    """
    Adaptive scan cadence based on market session.
      Weekend / market closed → 600s  (no trades possible, avoid 429 storms)
      09:30–11:00 ET          →  30s  (opening power hour)
      14:30–16:00 ET          →  30s  (closing power hour)
      everything else         →  60s
    """
    from datetime import datetime
    import zoneinfo
    now_et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    # Weekend: Saturday=5, Sunday=6
    if now_et.weekday() >= 5:
        return 600
    # Market holidays — poll at weekend cadence; no trades possible
    try:
        from agent.market_hours import _is_holiday
        if _is_holiday(now_et.date()):
            return 600
    except Exception:
        pass
    h, m = now_et.hour, now_et.minute
    minutes = h * 60 + m
    OPEN_RANGE_START = 9 * 60 + 30    # 09:30 ET
    OPEN_RANGE_END   = 9 * 60 + 30 + 90  # 11:00 ET
    CLOSE_START      = 14 * 60 + 30   # 14:30 ET
    CLOSE_END        = 16 * 60        # 16:00 ET
    MARKET_CLOSE     = 20 * 60        # 20:00 ET (after-hours end)
    # Outside all trading hours (overnight) — no need to poll aggressively
    if minutes < OPEN_RANGE_START - 60 or minutes >= MARKET_CLOSE:
        return 300
    if OPEN_RANGE_START <= minutes < OPEN_RANGE_END:
        return 30
    if CLOSE_START <= minutes < CLOSE_END:
        return 30
    try:
        from agent.config_manager import config as _cfg
        return max(10, min(600, int(_cfg.get("scanner.scan_interval_s", SCAN_INTERVAL_SECONDS))))
    except Exception:
        return SCAN_INTERVAL_SECONDS


# ── StockSignal dataclass ─────────────────────────────────────────────────────

@dataclass
class StockSignal:
    # ── Core price data ───────────────────────────────────────────────────────
    ticker:       str
    name:         str
    price:        float
    change_pct:   float

    # ── Component scores [-1, +1] ─────────────────────────────────────────────
    technical:     float
    volume:        float
    ml_prob:       float      # 40% daily ML + 60% intraday ML
    ml_daily_prob: float      # daily model probability (separately)
    sentiment:     float
    score:         float

    # ── Legacy signal label ────────────────────────────────────────────────────
    signal:        str

    # ── Volume flags ──────────────────────────────────────────────────────────
    rel_volume:   float
    unusual_vol:  bool

    # ── Professional prediction ───────────────────────────────────────────────
    prediction:        str
    confidence:        float
    trend:             str
    trend_probability: float
    ml_trained:        bool
    target_price:      float
    stop_loss:         float
    rr_ratio:          float
    patterns:          list = field(default_factory=list)
    reasons:           list = field(default_factory=list)

    # ── 15-min ML model probabilities ─────────────────────────────────────────
    ml_swing_prob:    float = 0.5    # SwingML (15-min XGBoost, 2h ahead)
    ml_deep_prob:     float = 0.5    # Deep BiLSTM (15-min, 1h ahead)
    ml_swing_trained: bool  = False
    ml_deep_trained:  bool  = False

    # ── Today's open price (populated from first 1-min bar; updated live via WS) ─
    open_price:   float = 0.0

    # ── Multi-timeframe analysis ───────────────────────────────────────────────
    mtf_score:           float = 0.0
    mtf_alignment:       str   = "MIXED"
    mtf_bull_count:      int   = 0
    mtf_bear_count:      int   = 0
    mtf_timeframes:      dict  = field(default_factory=dict)
    short_tf_alignment:  str   = "MIXED"  # 1M+5M+15M gate: "BULL"|"BEAR"|"MIXED"
    mtf_gate_passed:     bool  = False    # True when all short TFs agree

    # ── Support / Resistance ──────────────────────────────────────────────────
    supports:     list  = field(default_factory=list)
    resistances:  list  = field(default_factory=list)
    pivots:       dict  = field(default_factory=dict)
    poc:          float = 0.0

    # ── RSI zone ──────────────────────────────────────────────────────────────
    rsi_zone:         str   = "NEUTRAL"
    rsi_value:        float = 50.0
    rsi_gated:        bool  = False

    # ── Reversal zone ─────────────────────────────────────────────────────────
    reversal_score:   float = 0.0
    reversal_type:    str   = "NONE"     # BULLISH | BEARISH | NONE
    divergence_type:  str   = "NONE"     # BULLISH | BEARISH | NONE
    reversal_signals: list  = field(default_factory=list)

    # ── Exhaustion / retest / bounce entry ───────────────────────────────────
    entry_type:       str   = "IMMEDIATE"
    retest_level:     float = 0.0
    entry_zone_low:   float = 0.0
    entry_zone_high:  float = 0.0
    exhaustion_flags: list  = field(default_factory=list)
    bounce_signals:   list  = field(default_factory=list)
    rr_quality:       str   = "LOW"
    rr_qualifies:     bool  = False

    # ── Market session ────────────────────────────────────────────────────────
    session:        str   = "UNKNOWN"
    session_label:  str   = ""
    session_color:  str   = "#94a3b8"
    session_mult:   float = 1.0
    session_advice: str   = ""

    # ── Market regime (SPY/QQQ) ───────────────────────────────────────────────
    regime:        str   = "NEUTRAL"
    regime_label:  str   = "Neutral"
    regime_color:  str   = "#94a3b8"

    # ── Earnings blackout ─────────────────────────────────────────────────────
    earnings_blocked:  bool  = False
    earnings_reason:   str   = ""
    earnings_date:     str   = ""
    earnings_days_away: int  = 0
    earnings_phase:    str   = ""   # "" | "blackout" | "caution" | "cooldown"
    earnings_hour:     str   = ""   # "bmo" | "amc" | "dmh" | ""
    eps_surprise_pct:  float = 0.0  # (actual/estimate − 1)*100; 0 = unknown

    # ── Gap analysis ──────────────────────────────────────────────────────────
    gap_type:         str   = "FLAT"
    gap_pct:          float = 0.0
    gap_filled:       bool  = False
    gap_fill_prob:    float = 0.0
    premarket_high:   float = 0.0
    premarket_low:    float = 0.0

    gap_score:        float = 0.0   # [-1,+1] directional score from gap size/type
    today_open:       float = 0.0   # Regular-session open price (9:30 ET first bar)

    # ── PDH / PDL / ORB levels ────────────────────────────────────────────────
    prev_day_high:      float = 0.0   # Previous day high
    prev_day_low:       float = 0.0   # Previous day low
    prev_day_close:     float = 0.0   # Previous day close
    price_vs_pdh_pct:   float = 0.0   # (price - PDH) / PDH * 100 (+ve = above PDH)
    price_vs_pdl_pct:   float = 0.0   # (price - PDL) / PDL * 100 (-ve = below PDL)
    session_high:       float = 0.0   # Today's intraday HOD (regular session)
    session_low:        float = 0.0   # Today's intraday LOD (regular session)

    # ── Flag patterns (Algos 29 & 30) ────────────────────────────────────────
    bull_flag:  bool  = False  # Bull flag breakout setup detected
    bear_flag:  bool  = False  # Bear flag breakdown setup detected
    flag_high:  float = 0.0    # Top of flag consolidation zone
    flag_low:   float = 0.0    # Bottom of flag consolidation zone
    pole_pct:   float = 0.0    # % move of the flag pole (+/- direction)

    # ── Red-to-Green / Green-to-Red ───────────────────────────────────────────
    color_vs_prev_close: str  = "FLAT"  # "GREEN" | "RED" | "FLAT"
    r2g_event:           bool = False   # crossed above prev_close this bar
    g2r_event:           bool = False   # crossed below prev_close this bar
    r2g_bars_ago:        int  = -1      # bars since last R2G crossing today
    g2r_bars_ago:        int  = -1      # bars since last G2R crossing today

    orb_high:       float = 0.0   # Opening range breakout high (first 30-min)
    orb_low:        float = 0.0   # Opening range breakout low (first 30-min)
    orb_breakout:   str   = ""    # "BULL" | "BEAR" | "" — if price broke ORB
    orb5_high:      float = 0.0   # ORB-5 high (first 5-min, 9:30–9:35)
    orb5_low:       float = 0.0   # ORB-5 low (first 5-min, 9:30–9:35)
    orb5_breakout:  str   = ""    # "BULL" | "BEAR" | "NONE"
    orb5_score:     float = 0.0   # [-1, +1] directional score from ORB-5
    orb15_high:     float = 0.0   # ORB-15 high (first 15-min)
    orb15_low:      float = 0.0   # ORB-15 low (first 15-min)
    orb15_breakout: str   = ""    # "BULL" | "BEAR" | "NONE"
    orb15_score:    float = 0.0   # [-1, +1] directional score from ORB-15
    orb30_score:    float = 0.0   # [-1, +1] directional score from ORB-30

    # ── Volume profile (VAH/VAL) ──────────────────────────────────────────────
    vah:            float = 0.0   # Value Area High (70% vol rule)
    val:            float = 0.0   # Value Area Low

    # ── Day type (TREND_DAY | RANGE_DAY | UNCERTAIN) ──────────────────────────
    day_type:       str   = "UNCERTAIN"
    day_type_label: str   = "Uncertain"

    # ── Fibonacci retracement levels ─────────────────────────────────────────
    fib_levels:     dict  = field(default_factory=dict)

    # ── Relative strength vs SPY ──────────────────────────────────────────────
    rs_ratio:   float = 1.0
    rs_score:   float = 0.0
    rs_label:   str   = "IN_LINE"

    # ── VWAP signal + dynamic σ-bands ────────────────────────────────────────
    vwap_event:       str   = "FLAT"
    vwap_score:       float = 0.0
    vwap_price:       float = 0.0
    vwap_deviation:   float = 0.0
    vwap_description: str   = ""
    vwap_z_score:     float = 0.0    # z-score: (price - VWAP) / VWAP_std
    vwap_upper_1:     float = 0.0    # VWAP +1σ band
    vwap_lower_1:     float = 0.0    # VWAP −1σ band
    vwap_upper_2:     float = 0.0    # VWAP +2σ band
    vwap_lower_2:     float = 0.0    # VWAP −2σ band

    # ── Sector ETF context ────────────────────────────────────────────────────
    sector_etf:         str   = "QQQ"
    sector_trend:       str   = "NEUTRAL"
    sector_change:      float = 0.0
    stock_vs_sector:    str   = "IN_LINE"

    # ── Exit signals ──────────────────────────────────────────────────────────
    exit_recommendation: str  = "HOLD"
    exit_signals:        list = field(default_factory=list)
    exit_summary:        str  = ""

    # ── Macro calendar ────────────────────────────────────────────────────────
    macro_blocked:      bool  = False
    macro_throttled:    bool  = False
    macro_size_mult:    float = 1.0
    macro_min_confidence: float = 0.0
    macro_event:        str   = ""
    macro_description:  str   = ""

    # ── Trade management plan ─────────────────────────────────────────────────
    trade_plan: dict  = field(default_factory=dict)

    # ── Chart candles (last 80 × 1-min bars) ─────────────────────────────────
    candles:    list = field(default_factory=list)

    # ── News / context intelligence ───────────────────────────────────────────
    headlines:         list  = field(default_factory=list)
    news_shock:        bool  = False    # True when news_count_30m ≥ 3 × baseline
    sentiment_velocity: float = 0.0    # sentiment_5m − sentiment_30m (momentum)
    news_count_30m:    int   = 0        # news articles in last 30 min
    ctx_stale:         bool  = False    # True when context data > 120 s old
    scanned_at: str  = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ── Adaptive filter ───────────────────────────────────────────────────────
    is_suppressed:   bool = False   # True when adaptive filter blocked this signal
    suppress_reason: str  = ""      # Why it was suppressed
    has_open_position: bool = False # True when ticker already has an open paper trade

    # ── After-hours / pre-market context ─────────────────────────────────────
    ah_change_pct:      float = 0.0   # AH price move from previous close (%)
    ah_direction:       str   = ""    # BULLISH | BEARISH | NEUTRAL | ""
    ah_magnitude:       str   = ""    # STRONG | MODERATE | WEAK | ""
    ah_confirms_signal: bool  = False # True when AH direction aligns with prediction
    ah_news_likely:     bool  = False # True when large AH move + elevated volume
    ah_gap_estimate:    float = 0.0   # Expected gap at open (%)

    # ── Trading hours classification ──────────────────────────────────────────
    trading_tier: str = "MODERATE"    # HIGH | MODERATE | REGULAR (see trading_hours.py)

    # ── Per-ticker self-learning score ────────────────────────────────────────
    ticker_win_rate:  float = 0.0
    ticker_obs_count: int   = 0
    learning_rank:    float = 0.0

    # ── Order flow (PRD Section 3.2) ──────────────────────────────────────────
    order_flow_score:    float = 0.0   # -1.0 (sell pressure) to +1.0 (buy pressure)
    order_flow_label:    str   = "NEUTRAL"
    signal_strength:     str   = "STANDARD"   # STRONG|STANDARD|WEAK|CONFLICTED|BLOCKED|NO_SIGNAL
    signal_size_mult:    float = 1.0   # position size multiplier from arbitration

    # ── Bar-level technical row (quant strategies read this) ──────────────────
    # Populated from the last row of compute_indicators(df_1m) — all float values.
    # Enables quant_strategies.py to access any computed indicator without adding
    # individual fields to StockSignal for every new indicator.
    tech_row: dict = field(default_factory=dict)

    # ── Algorithm signals (Phase 1+ trading algos) ────────────────────────────
    algo_signals: list = field(default_factory=list)  # list of AlgoResult dicts

    def to_dict(self) -> dict:
        import math
        d = asdict(self)
        for k, v in d.items():
            if hasattr(v, "item"):
                v = v.item()
                d[k] = v
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                d[k] = None
        d["unusual_vol"] = bool(d["unusual_vol"])
        d["ml_trained"]  = bool(d["ml_trained"])
        return d


def _positive_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        if np.isfinite(v) and v > 0:
            return v
    except Exception:
        pass
    return default


def _last_positive_close(df: Optional[pd.DataFrame]) -> float:
    if df is None or df.empty or "Close" not in df.columns:
        return 0.0
    try:
        closes = df["Close"].dropna()
        closes = closes[closes > 0]
        if not closes.empty:
            return float(closes.iloc[-1])
    except Exception:
        pass
    return 0.0


def _session_open_from_1m(df: Optional[pd.DataFrame]) -> float:
    if df is None or df.empty or "Open" not in df.columns:
        return 0.0
    try:
        today = pd.Timestamp.now(tz="America/New_York").date()
        today_df = df[df.index.date == today]
        if not today_df.empty:
            return _positive_float(today_df.iloc[0].get("Open"))
    except Exception:
        pass
    try:
        return _positive_float(df.iloc[0].get("Open"))
    except Exception:
        return 0.0


def _previous_close_from_1d(df_1d: Optional[pd.DataFrame]) -> float:
    if df_1d is None or df_1d.empty or "Close" not in df_1d.columns:
        return 0.0
    try:
        today = pd.Timestamp.now(tz="America/New_York").date()
        last_date = df_1d.index[-1].date()
        if last_date >= today and len(df_1d) >= 2:
            return _positive_float(df_1d.iloc[-2].get("Close"))
        return _positive_float(df_1d.iloc[-1].get("Close"))
    except Exception:
        return _positive_float(df_1d.iloc[-1].get("Close"))


def _observation_signal(
    ticker: str,
    df_1m: Optional[pd.DataFrame],
    df_1d: Optional[pd.DataFrame],
    session: Optional[dict] = None,
    quote: Optional[dict] = None,
    reason: str = "Scan deferred this cycle",
) -> Optional[StockSignal]:
    """
    Build a lightweight dashboard row when the heavy analysis path is deferred.

    A cycle-budget timeout should not make a ticker vanish from the production
    dashboard. These rows are neutral/watch-only and are not recorded as model
    training observations; they simply preserve visibility until the next full
    analysis for that ticker completes.
    """
    if quote is None:
        try:
            from agent.valkey_client import get_all_prices
            quote = (get_all_prices() or {}).get(ticker, {}) or {}
        except Exception:
            quote = {}

    price = _last_positive_close(df_1m) or _positive_float(quote.get("last"))
    if price <= 0:
        return None

    prev_close = _previous_close_from_1d(df_1d) or _positive_float(quote.get("prev_close"))
    if prev_close <= 0:
        prev_close = price
    change_pct = float(quote.get("pct_change") or quote.get("net_pct_change") or 0.0)
    if not change_pct and prev_close > 0:
        change_pct = round((price - prev_close) / prev_close * 100.0, 3)

    open_price = (
        _session_open_from_1m(df_1m)
        or _positive_float(quote.get("open"))
        or _positive_float(quote.get("regularMarketOpen"))
    )

    sess = session or {}
    return StockSignal(
        ticker=ticker,
        name=ticker,
        price=round(price, 4),
        change_pct=round(change_pct, 3),
        open_price=round(open_price, 4),
        technical=0.0,
        volume=0.0,
        ml_prob=0.5,
        ml_daily_prob=0.5,
        sentiment=0.0,
        score=0.0,
        signal="NEUTRAL",
        rel_volume=0.0,
        unusual_vol=False,
        prediction="NEUTRAL",
        confidence=0.0,
        trend="SIDEWAYS",
        trend_probability=0.5,
        ml_trained=False,
        target_price=0.0,
        stop_loss=0.0,
        rr_ratio=float(get_execution_min_rr()),
        patterns=["OBSERVATION"],
        reasons=[reason],
        session=sess.get("session", "UNKNOWN"),
        session_label=sess.get("label", ""),
        session_color=sess.get("color", "#94a3b8"),
        session_mult=float(sess.get("size_mult", 1.0) or 1.0),
        session_advice=sess.get("advice", ""),
        signal_strength="NO_SIGNAL",
        rr_quality="OBSERVE",
        rr_qualifies=False,
    )


# ── Ticker metadata cache ─────────────────────────────────────────────────────

_info_cache:    dict[str, dict]         = {}
_spy_df_cache:  dict[str, pd.DataFrame] = {}   # SPY/QQQ 1M frames for RS/regime

# ── Per-ticker signal cooldown ────────────────────────────────────────────────
# Prevents the same setup from being recorded every 60s as independent signals.
# Key: (ticker, direction) → timestamp of last recorded signal
_SIGNAL_COOLDOWN_SECS = 15 * 60   # 15 minutes
_last_signal_ts: dict[tuple[str, str], float] = {}


def _get_info(ticker: str) -> dict:
    if ticker not in _info_cache:
        _info_cache[ticker] = fetch_ticker_info(ticker)
    return _info_cache[ticker]


def _build_candles(df: pd.DataFrame, n: int = 80) -> list:
    """
    Serialise the last N OHLCV bars for TradingView Lightweight Charts.
    Bars outside regular NYSE hours (09:30–16:00 ET) are flagged with
    ``extended=True`` so the frontend can render them with a distinct style.
    """
    import pytz
    _et = pytz.timezone("America/New_York")
    _regular_open  = pd.Timedelta(hours=9,  minutes=30)
    _regular_close = pd.Timedelta(hours=16, minutes=0)

    tail    = df.tail(n)
    candles = []
    for ts, row in tail.iterrows():
        try:
            ts_et  = pd.Timestamp(ts).tz_localize("UTC").tz_convert(_et) if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts).tz_convert(_et)
            tod    = ts_et - ts_et.normalize()
            is_ext = (tod < _regular_open) or (tod >= _regular_close)
            candles.append({
                "time":     int(pd.Timestamp(ts).timestamp()),
                "open":     round(float(row["Open"]),  4),
                "high":     round(float(row["High"]),  4),
                "low":      round(float(row["Low"]),   4),
                "close":    round(float(row["Close"]), 4),
                "volume":   int(row["Volume"]),
                "extended": bool(is_ext),
            })
        except Exception:
            pass
    return candles


def _compute_levels(df_1m: pd.DataFrame, df_1d: pd.DataFrame) -> dict:
    """Compute PDH, PDL, PDC, ORB, and today's session HOD/LOD from market data."""
    import pytz
    result = {"prev_day_high": 0.0, "prev_day_low": 0.0, "prev_day_close": 0.0,
              "orb_high": 0.0, "orb_low": 0.0, "orb_breakout": "",
              "session_high": 0.0, "session_low": 0.0}
    try:
        if not df_1d.empty and len(df_1d) >= 2:
            prev = df_1d.iloc[-2]
            result["prev_day_high"]  = round(float(prev.get("High",  0)), 4)
            result["prev_day_low"]   = round(float(prev.get("Low",   0)), 4)
            result["prev_day_close"] = round(float(prev.get("Close", 0)), 4)
    except Exception:
        pass
    try:
        if not df_1m.empty:
            et = pytz.timezone("America/New_York")
            df_et = df_1m.copy()
            df_et.index = pd.to_datetime(df_et.index)
            if df_et.index.tzinfo is None:
                df_et.index = df_et.index.tz_localize("UTC").tz_convert(et)
            else:
                df_et.index = df_et.index.tz_convert(et)
            today = df_et.index[-1].date()
            orb_mask = (
                (df_et.index.date == today) &
                (df_et.index.time >= pd.Timestamp("09:30").time()) &
                (df_et.index.time <= pd.Timestamp("10:00").time())
            )
            orb_bars = df_et[orb_mask]
            if not orb_bars.empty:
                result["orb_high"] = round(float(orb_bars["High"].max()), 4)
                result["orb_low"]  = round(float(orb_bars["Low"].min()),  4)
                last_price = float(df_et.iloc[-1]["Close"])
                if result["orb_high"] > 0:
                    if last_price > result["orb_high"]:
                        result["orb_breakout"] = "BULL"
                    elif last_price < result["orb_low"]:
                        result["orb_breakout"] = "BEAR"

            # Today's session HOD / LOD (regular session only)
            today_sess = df_et[
                (df_et.index.date == today) &
                (df_et.index.time >= pd.Timestamp("09:30").time())
            ]
            if not today_sess.empty:
                result["session_high"] = round(float(today_sess["High"].max()), 4)
                result["session_low"]  = round(float(today_sess["Low"].min()),  4)
    except Exception:
        pass
    return result


# ── ML prediction cache ───────────────────────────────────────────────────────
# Keyed by (ticker, last_bar_timestamp_nanoseconds).  If the most recent 1-min
# bar hasn't changed since the previous scan cycle, all six ML model outputs
# are identical — feature inputs are the same.  Skipping model inference saves
# ~1.5-2 s per ticker (the single largest time sink in analyse_ticker).
#
# Cache lifetime: 5 min.  A new bar closing invalidates automatically because
# the nanosecond timestamp changes.
#
# Thread safety: Python dict reads/writes are GIL-protected.  Multiple threads
# may write the same key simultaneously but always write identical values
# (same inputs → same outputs), so last-write-wins is correct.
_ML_PRED_CACHE: dict[str, tuple] = {}
_ML_PRED_CACHE_TTL_S: float = 300.0   # 5 min — covers 2× the longest scan interval


def _ml_pred_cache_get(ticker: str, bar_ts_ns: int) -> tuple | None:
    """Return cached (scalp, daily, reversal, ensemble, agree, swing, deep) or None."""
    entry = _ML_PRED_CACHE.get(ticker)
    if entry is None:
        return None
    cached_ns, *vals, cached_at = entry
    if cached_ns != bar_ts_ns or time.time() - cached_at > _ML_PRED_CACHE_TTL_S:
        return None
    return tuple(vals)


def _ml_pred_cache_put(ticker: str, bar_ts_ns: int, *vals) -> None:
    _ML_PRED_CACHE[ticker] = (bar_ts_ns, *vals, time.time())


# ── Single ticker analysis ────────────────────────────────────────────────────

def analyse_ticker(
    ticker: str,
    df_1m:  Optional[pd.DataFrame],
    df_5m:  pd.DataFrame,
    df_1h:  pd.DataFrame,
    df_1d:  pd.DataFrame,
) -> Optional[StockSignal]:
    try:
        if df_1m is None or df_1m.empty or len(df_1m) < 5:
            return None

        # ── Drop trailing incomplete bars (close=0 or NaN) ───────────────────
        # Schwab REST may return a placeholder "current bar" with close=0 when
        # called outside trading hours (CLOSED/AH sessions).  Remove it before
        # computing indicators so prediction.py doesn't receive price=0 and fall
        # back to _empty_prediction() (which has confidence=0.0).
        _df_clean = df_1m[df_1m["Close"].fillna(0) > 0]
        if _df_clean.empty or len(_df_clean) < 5:
            return None
        df_1m = _df_clean

        df_ind = compute_indicators(df_1m.copy())
        last   = df_ind.iloc[-1]

        price    = float(last["Close"])
        bar_high = float(last["High"])
        bar_low  = float(last["Low"])

        # Change % vs previous confirmed trading day's close.
        # Schwab's daily series may include a live intraday bar for today
        # (with Close = current live price), which would make change_pct = 0%.
        # Always use the LAST CONFIRMED bar: if the newest daily bar is from
        # today, use iloc[-2] (yesterday); otherwise use iloc[-1].
        _prev_close = 0.0
        if not df_1d.empty:
            try:
                import zoneinfo as _zi
                _today = pd.Timestamp.now(tz=_zi.ZoneInfo("America/New_York")).date()
                _last_bar_date = df_1d.index[-1].date()
                if _last_bar_date >= _today and len(df_1d) >= 2:
                    _prev_close = float(df_1d.iloc[-2]["Close"])
                else:
                    _prev_close = float(df_1d.iloc[-1]["Close"])
            except Exception:
                _prev_close = float(df_1d.iloc[-1]["Close"])
        if _prev_close <= 0:
            _prev_close = float(df_ind.iloc[0]["Open"]) or price
        change_pct = round((price - _prev_close) / _prev_close * 100, 3) if _prev_close else 0.0

        # Today's opening price — first 1-min bar of the current session
        _today_open = 0.0
        try:
            import zoneinfo as _zi2
            _now_et = pd.Timestamp.now(tz=_zi2.ZoneInfo("America/New_York"))
            _today_1m = df_1m[df_1m.index.date == _now_et.date()]
            if not _today_1m.empty:
                _today_open = float(_today_1m.iloc[0]["Open"])
        except Exception:
            pass

        tech            = score_technical(last)
        vol             = score_volume(df_ind)
        # ML models trained on 5-min bars — always infer on 5-min data so
        # feature distributions match training (RSI-14 on 5m = 70 min of price
        # action; on 1m it only covers 14 min, completely different signal).
        _df_ml = df_5m if (df_5m is not None and len(df_5m) >= 20) else df_ind

        # 15-min bars — resampled from 5-min (no extra API call).
        # Used by both SwingML (XGBoost on 9 months of 15-min data) and
        # the deep BiLSTM model. Both train AND infer on 15-min so there
        # is no feature-distribution mismatch.
        _df_15m = None
        try:
            from agent.data_fetcher import resample_ohlcv
            _df_15m = resample_ohlcv(_df_ml, "15min") if _df_ml is not None else None
        except Exception:
            pass
        _has_15m = _df_15m is not None and len(_df_15m) >= 20

        # ── ML prediction cache ───────────────────────────────────────────────
        # Skip 6 model calls (~1.5-2 s) when the last 1-min bar hasn't changed.
        # The bar timestamp (nanoseconds) is the uniqueness key: a new bar close
        # changes it automatically, invalidating the cache without TTL tricks.
        _bar_ts_ns = int(df_ind.index[-1].value)
        _ml_hit = _ml_pred_cache_get(ticker, _bar_ts_ns)
        try:
            _session_name_for_ml = str(get_session_info().get("session", "")).upper()
        except Exception:
            _session_name_for_ml = ""
        if _ml_hit is not None:
            (ml_scalp, ml_daily_p, ml_reversal_p,
             ml_ensemble_p, ml_agree, ml_swing_p, ml_deep_p) = _ml_hit
        else:
            ml_scalp        = predict(ticker, _df_ml)
            ml_daily_p      = predict_daily(ticker, df_1d) if not df_1d.empty else 0.5
            ml_reversal_p   = predict_reversal(ticker, _df_ml)
            ml_ensemble_p, ml_agree = predict_ensemble(ticker, _df_ml)
            ml_swing_p      = predict_swing(ticker, _df_15m) if _has_15m else 0.5
            ml_deep_p       = (
                predict_deep(ticker, _df_15m)
                if _has_15m and _session_name_for_ml != "CLOSED"
                else 0.5
            )
            _ml_pred_cache_put(ticker, _bar_ts_ns, ml_scalp, ml_daily_p,
                               ml_reversal_p, ml_ensemble_p, ml_agree,
                               ml_swing_p, ml_deep_p)

        # MetaEnsemble: calibrated fusion of all ML sources.
        # When trained (≥30 outcomes), uses a meta-XGBoost to combine signals
        # optimally; before that, falls back to a weighted average.
        ml_combined, _meta_mult    = get_meta_prediction(
            ticker             = ticker,
            scalp_prob         = ml_scalp,
            ensemble_prob      = ml_ensemble_p,
            ensemble_agreement = ml_agree,
            daily_prob         = ml_daily_p,
            reversal_prob      = ml_reversal_p,
            tech_score         = float(tech),
            vol_score          = float(vol),
        )

        # Blend 15-min models into ml_combined after MetaEnsemble.
        # SwingML adds 9-month XGBoost context; Deep BiLSTM adds sequence learning.
        # Both contribute only when trained to avoid noise from untrained models.
        from agent.deep_model import is_trained as _deep_is_trained
        from agent.signal_blender import blend_signals as _blend_signals
        _swing_trained = get_or_create_swing(ticker).trained
        _deep_trained  = _deep_is_trained()
        # Dynamic blend — weights adapt based on each model's rolling 20-trade accuracy
        ml_combined = _blend_signals(
            scalp_p       = ml_scalp,
            ensemble_p    = ml_ensemble_p,
            reversal_p    = ml_reversal_p,
            swing_p       = ml_swing_p,
            deep_p        = ml_deep_p,
            swing_trained = _swing_trained,
            deep_trained  = _deep_trained,
        )
        # ── Context snapshot (Valkey-first, < 10 ms) ─────────────────────────
        # Single read replaces two stub calls (score_sentiment + earnings_blackout).
        # Falls back: Valkey → PostgreSQL → safe defaults.
        # When FINNHUB_API_KEY is absent or context-intel is not running, all
        # fields default to 0.0 / "" / 999 — identical to the pre-Phase-1 stubs.
        from agent.context_snapshot import (
            get_context_snapshot as _get_ctx,
            get_market_context_snapshot as _get_mkt_ctx,
        )
        _ctx           = _get_ctx(ticker)
        sent           = float(_ctx.get("sentiment_30m", 0.0))
        headlines      = list(_ctx.get("recent_headlines", []))
        _earnings_hour = str(_ctx.get("earnings_hour", ""))
        _eps_surp      = float(_ctx.get("eps_surprise_pct", 0.0))

        # Build earnings blackout dict from context snapshot (matches expected keys)
        _ep = _ctx.get("earnings_phase", "")
        eb  = {
            "blocked":   _ep == "blackout" or _ep == "cooldown",
            "reason":    _ctx.get("earnings_reason",    ""),
            "next_date": _ctx.get("earnings_next_date", ""),
            "days_away": int(_ctx.get("earnings_days_away", 999)),
        }

        # ── Market-wide sentiment gate ─────────────────────────────────────────
        # Reads ctx:market from Valkey (published by context-intel service).
        # Gate: strong negative market sentiment penalises LONG signals;
        #       strong positive market sentiment penalises SHORT signals.
        _mkt_ctx       = _get_mkt_ctx()
        _mkt_sentiment = float(_mkt_ctx.get("market_sentiment", 0.0))

        rvol            = rvol_time_of_day(df_ind, df_1d)
        uvol            = detect_unusual_volume(df_ind)

        # Multi-timeframe analysis (6 TFs: 5M, 15M, 30M, 1H, 4H, 1D)
        mtf = multi_timeframe_analysis(df_1m, df_5m, df_1h, df_1d)

        # Market session — use Schwab as authoritative source when available
        # (detects unexpected early closes / circuit breakers that our local
        # time-based check would miss)
        sess_info = get_session_info()
        sess_mult = confidence_multiplier()
        # Schwab market hours cross-check skipped when SCHWAB_ENABLED=false

        # Gap analysis
        gap = analyse_gap(df_1m, df_1d)

        # Red-to-Green / Green-to-Red detection vs prior close
        # gap["prior_close"] is the same as PDC but already extracted above
        color_flip = detect_color_flip(df_1m, gap.get("prior_close", 0.0))

        # PDH / PDL / ORB levels
        levels = _compute_levels(df_1m, df_1d)

        # Relative strength vs SPY (regime spy data available via get_regime())
        regime = get_regime()
        rs = compute_relative_strength(df_1m, _spy_df_cache.get("SPY"))

        # VWAP signal (now includes z_score, upper_2/lower_2 band levels)
        vwap_sig = compute_vwap_signal(df_ind)

        # Opening Range (ORB-15 and ORB-30 levels + breakout classification)
        orb_result = compute_opening_range(df_1m)

        # Day type classification (TREND_DAY / RANGE_DAY / UNCERTAIN)
        day_type_info = classify_day_type(df_ind, df_1d)

        # Professional S/R: Camarilla pivots, Fibonacci retracements, Value Area
        try:
            _camarilla = calculate_camarilla_pivots(df_ind, df_1d)
        except Exception:
            _camarilla = {}
        try:
            _fibonacci = calculate_fibonacci_levels(df_ind)
        except Exception:
            _fibonacci = {}
        try:
            _value_area = calculate_value_area(df_ind)
        except Exception:
            _value_area = {"poc": 0.0, "vah": 0.0, "val": 0.0}

        # Sector ETF context
        sector_ctx = get_sector_context(ticker, df_1m)

        # Macro calendar blackout
        macro_ev = check_macro_event()

        # Professional prediction
        pred = generate_prediction(
            ticker, df_ind, tech, vol, ml_combined, sent, last,
            mtf_score=mtf["mtf_score"],
            ml_reversal_prob=ml_reversal_p,
            vwap_score=vwap_sig["score"],
            sector_mult=sector_ctx.score_mult,
            ensemble_prob=ml_ensemble_p,
            ensemble_agreement=ml_agree,
            df_daily=df_1d,
            eps_surprise_pct=_eps_surp,
        )

        # Blend ORB signal into composite score (15% weight when OR is established)
        # Average whichever ORB levels have formed (weight 5m most since it's fastest)
        _orb_scores = [s for s in [orb_result.or5_score, orb_result.or15_score, orb_result.or30_score] if s != 0.0]
        _orb_signal = float(np.mean(_orb_scores)) if _orb_scores else 0.0
        _orb_weight = 0.15 if (orb_result.orh_5 > 0 or orb_result.orh_15 > 0 or orb_result.orh_30 > 0) else 0.0
        _pred_comp  = float(np.clip(pred["composite_score"], -1, 1))
        _blended    = _pred_comp * (1.0 - _orb_weight) + _orb_signal * _orb_weight

        # Day type multiplier: trend signals boosted on TREND_DAY, discounted on RANGE_DAY
        if day_type_info.day_type == "TREND_DAY" and _blended > 0.1:
            _blended = float(np.clip(_blended * day_type_info.trend_boost, -1.0, 1.0))
        elif day_type_info.day_type == "RANGE_DAY" and abs(_blended) < 0.4:
            # On range days, mean-reversion signals (close to 0) are more reliable
            _blended = float(np.clip(_blended * day_type_info.mr_discount, -1.0, 1.0))

        # Apply regime multiplier to score
        raw_score = round(float(np.clip(_blended, -1, 1)), 4)
        score     = round(float(np.clip(apply_regime(raw_score, regime), -1, 1)), 4)

        # Override direction to NEUTRAL only during a true hard block.
        # Wider macro windows throttle risk instead of suppressing all trades.
        if eb["blocked"] or macro_ev["blocked"]:
            pred["direction"] = "NEUTRAL"

        # ── After-hours / CLOSED session gate ────────────────────────────────
        # Compute trading tier now so we can gate both signal direction and
        # the AH confidence boost on the same tier.
        _session_now  = sess_info.get("session", "")
        _static_tier  = get_trading_tier(ticker, 0.0)   # refined later with AH vol ratio

        if _session_now == "CLOSED":
            # Market fully closed (8pm–4am) — monitoring only
            if pred["direction"] in ("BUY", "SELL"):
                pred["direction"] = "NEUTRAL"
                pred["reasons"]   = ["⛔ Market closed (8pm–4am) — monitoring only"] + pred.get("reasons", [])

        elif _session_now == "AFTER_HOURS":
            if _static_tier == "HIGH":
                if pred["direction"] in ("BUY", "SELL"):
                    pred["reasons"] = [
                        f"🌙 AH {ticker} HIGH-tier — 50% size, extended liquidity"
                    ] + pred.get("reasons", [])
            elif _static_tier == "MODERATE":
                if pred["direction"] in ("BUY", "SELL"):
                    pred["reasons"] = [
                        f"🌙 AH {ticker} MODERATE-tier — 30% size, caution advised"
                    ] + pred.get("reasons", [])
            else:
                if pred["direction"] in ("BUY", "SELL"):
                    pred["direction"] = "NEUTRAL"
                    pred["reasons"]   = [
                        f"⛔ AH {ticker} REGULAR-tier — thin ECN spreads, monitoring only"
                    ] + pred.get("reasons", [])

        elif _session_now == "PRE_MARKET":
            if _static_tier == "HIGH":
                if pred["direction"] in ("BUY", "SELL"):
                    pred["reasons"] = [
                        f"🌅 PM {ticker} HIGH-tier — 40% size, pre-market momentum"
                    ] + pred.get("reasons", [])
            elif _static_tier == "MODERATE":
                if pred["direction"] in ("BUY", "SELL"):
                    pred["reasons"] = [
                        f"🌅 PM {ticker} MODERATE-tier — 25% size, caution advised"
                    ] + pred.get("reasons", [])
            else:
                if pred["direction"] in ("BUY", "SELL"):
                    pred["direction"] = "NEUTRAL"
                    pred["reasons"]   = [
                        f"⛔ PM {ticker} REGULAR-tier — insufficient pre-market liquidity"
                    ] + pred.get("reasons", [])

        # Exit signals (for open paper trades / active signals)
        exit_analysis = analyse_exits(
            df=df_ind,
            direction=pred["direction"],
            entry_price=price,
            target=pred["target_price"],
            stop=pred["stop_loss"],
            bars_held=0,
        )

        # Build trade plan
        tp = build_trade_plan(
            entry=price,
            stop=pred["stop_loss"],
            target=pred["target_price"],
            direction=pred["direction"],
        )

        # ── After-hours / pre-market data collection & opening bias ─────────────
        _ah_bias: dict = {}
        _current_session = sess_info.get("session", "")

        if _current_session in ("AFTER_HOURS", "PRE_MARKET"):
            # Collect AH observations — use previous day close from daily data
            _prev_close = 0.0
            if not df_1d.empty and len(df_1d) >= 2:
                _prev_close = float(df_1d.iloc[-2]["Close"]) if "Close" in df_1d.columns else 0.0
            if _prev_close <= 0 and not df_1d.empty:
                _prev_close = float(df_1d.iloc[-1]["Open"]) if "Open" in df_1d.columns else price
            _ah_vol   = float(df_ind["Volume"].sum()) if "Volume" in df_ind.columns else 0.0
            _avg_vol  = float(df_ind["Volume"].mean() * 390) if "Volume" in df_ind.columns else 0.0
            if _prev_close > 0:
                ah_record(
                    ticker     = ticker,
                    ah_price   = price,
                    prev_close = _prev_close,
                    ah_volume  = _ah_vol,
                    avg_volume = _avg_vol,
                    session    = _current_session,
                )

        else:
            # During regular session: read stored AH bias to inform confidence
            _ah_bias = get_opening_bias(ticker)

        # Apply backtest-derived confidence calibration (outcome predictor when trained,
        # simple context-win-rate table as fallback).
        pred["confidence"] = adjust_confidence(
            confidence    = pred["confidence"],
            vwap_event    = vwap_sig["event"],
            rsi_zone      = pred.get("rsi_zone", ""),
            session       = sess_info.get("session", ""),
            regime        = regime.regime,
            entry_type    = pred.get("entry_type", ""),
            direction     = pred["direction"],
            rr_ratio      = float(pred.get("rr_ratio", 0.0) or 0.0),
            mtf_alignment = float(pred.get("mtf_alignment", 0.0) or 0.0),
            rsi_value     = float(pred.get("rsi", 50.0) or 50.0),
        )

        # IMMEDIATE entries are structurally weaker (57% WR vs 62-70% for setups).
        # Apply a small penalty to raise the bar — forces IMMEDIATE to be higher conviction.
        if pred.get("entry_type") == "IMMEDIATE" and pred["direction"] in ("BUY", "SELL"):
            pred["confidence"] = round(float(max(pred["confidence"] - 5.0, 25.0)), 1)

        # Apply adaptive boost for high-win-rate contexts
        boost = get_confidence_boost(
            vwap_event = vwap_sig["event"],
            session    = sess_info.get("session", ""),
            regime     = regime.regime,
            rsi_zone   = pred.get("rsi_zone", ""),
            entry_type = pred.get("entry_type", ""),
            direction  = pred["direction"],
        )
        if boost:
            pred["confidence"] = round(
                float(min(max(pred["confidence"] + boost, 25.0), 95.0)), 1
            )

        # Apply after-hours opening bias to confidence — gated by trading tier
        _ah_confirms = False
        _ah_change   = 0.0
        _ah_dir      = ""
        _ah_mag      = ""
        _ah_news     = False
        _ah_gap_est  = 0.0
        if _ah_bias and pred["direction"] in ("BUY", "SELL"):
            _ah_dir     = _ah_bias.get("direction", "")
            _ah_mag     = _ah_bias.get("magnitude", "")
            _ah_change  = float(_ah_bias.get("ah_change_pct", 0.0))
            _ah_news    = bool(_ah_bias.get("news_likely", False))
            _ah_gap_est = float(_ah_bias.get("gap_estimate_pct", 0.0))
            _base_adj   = float(_ah_bias.get("confidence_adj", 0.0))

            # Refine tier now that we have the live AH volume ratio
            _ah_vol_ratio = float(_ah_bias.get("ah_volume_ratio", 0.0))
            _trading_tier = get_trading_tier(ticker, _ah_vol_ratio)

            # Scale the confidence adjustment by tier:
            #   HIGH     → full adjustment (AH data reliable)
            #   MODERATE → half adjustment (use with caution)
            #   REGULAR  → no adjustment (AH data is noise for thin names)
            _tier_scale = {"HIGH": 1.0, "MODERATE": 0.5, "REGULAR": 0.0}.get(_trading_tier, 0.0)
            _base_adj   = round(_base_adj * _tier_scale, 1)

            # Confirms when AH direction matches prediction
            _ah_confirms = (
                (_ah_dir == "BULLISH" and pred["direction"] == "BUY") or
                (_ah_dir == "BEARISH" and pred["direction"] == "SELL")
            )
            if _base_adj > 0:
                _conf_delta = _base_adj if _ah_confirms else -_base_adj
                pred["confidence"] = round(
                    float(min(max(pred["confidence"] + _conf_delta, 25.0), 95.0)), 1
                )
                _label = "confirms" if _ah_confirms else "opposes"
                pred["reasons"].append(
                    f"AH {_ah_dir} {_ah_change:+.1f}% ({_ah_mag}) {_label} signal "
                    f"[{_trading_tier} tier ×{_tier_scale}]"
                )
        else:
            # No AH bias data — use static tier for downstream use
            _trading_tier = _static_tier

        # ── Earnings hour gating ──────────────────────────────────────────────
        # Reduce confidence during the most volatile windows on earnings day:
        #   bmo (before-market-open):  9:30–9:45 ET — gap/spike on open
        #   amc (after-market-close):  15:45–16:00 ET — closing auction chop
        if _earnings_hour and pred["direction"] in ("BUY", "SELL", "STRONG BUY", "STRONG SELL"):
            try:
                from agent.market_hours import ET as _ET
                import datetime as _dt
                _t = _dt.datetime.now(_ET).time()
                _OPEN_START  = _dt.time(9, 30)
                _OPEN_END    = _dt.time(9, 45)
                _CLOSE_START = _dt.time(15, 45)
                _CLOSE_END   = _dt.time(16, 0)
                if _earnings_hour == "bmo" and _OPEN_START <= _t <= _OPEN_END:
                    pred["confidence"] = round(float(max(pred["confidence"] - 12.0, 25.0)), 1)
                    pred["reasons"].insert(0,
                        "⚠ BMO earnings open window (9:30–9:45 ET) — confidence reduced")
                elif _earnings_hour == "amc" and _CLOSE_START <= _t <= _CLOSE_END:
                    pred["confidence"] = round(float(max(pred["confidence"] - 12.0, 25.0)), 1)
                    pred["reasons"].insert(0,
                        "⚠ AMC earnings closing window (15:45–16:00 ET) — confidence reduced")
            except Exception:
                pass

        # ── Market-wide sentiment gate ────────────────────────────────────────
        # Penalty: bearish market hurts LONG signals; bullish market hurts SHORTs.
        # Extreme sentiment (|mkt| ≥ 0.5) blocks the contrary direction entirely.
        if _mkt_sentiment != 0.0 and pred["direction"] in ("BUY", "SELL", "STRONG BUY", "STRONG SELL"):
            _is_long = pred["direction"] in ("BUY", "STRONG BUY")
            if _mkt_sentiment < -0.5 and _is_long:
                pred["direction"] = "NEUTRAL"
                pred["reasons"].insert(0,
                    f"⛔ Market sentiment {_mkt_sentiment:.2f} — extreme negative, LONG blocked")
            elif _mkt_sentiment > 0.5 and not _is_long:
                pred["direction"] = "NEUTRAL"
                pred["reasons"].insert(0,
                    f"⛔ Market sentiment +{_mkt_sentiment:.2f} — extreme positive, SHORT blocked")
            elif _mkt_sentiment < -0.25 and _is_long:
                _mkt_penalty = round(abs(_mkt_sentiment) * 15.0, 1)   # up to −7.5 pts
                pred["confidence"] = round(float(max(pred["confidence"] - _mkt_penalty, 25.0)), 1)
                pred["reasons"].append(
                    f"Market sentiment {_mkt_sentiment:.2f} — broad negative bias, LONG confidence reduced")
            elif _mkt_sentiment > 0.25 and not _is_long:
                _mkt_penalty = round(abs(_mkt_sentiment) * 15.0, 1)
                pred["confidence"] = round(float(max(pred["confidence"] - _mkt_penalty, 25.0)), 1)
                pred["reasons"].append(
                    f"Market sentiment +{_mkt_sentiment:.2f} — broad positive bias, SHORT confidence reduced")

        _macro_size_mult = 1.0
        _macro_min_conf = 0.0
        if macro_ev.get("throttled") and pred["direction"] in ("BUY", "SELL", "STRONG BUY", "STRONG SELL"):
            _macro_size_mult = float(macro_ev.get("size_mult", 1.0) or 1.0)
            _macro_min_conf = float(macro_ev.get("min_confidence", 0.0) or 0.0)
            _macro_bump = float(macro_ev.get("confidence_bump", 0.0) or 0.0)
            _pre_macro_conf = float(pred.get("confidence", 0.0) or 0.0)
            if _pre_macro_conf < _macro_min_conf:
                pred["direction"] = "NEUTRAL"
                pred["reasons"] = [
                    f"Macro throttle: confidence {_pre_macro_conf:.1f} < {_macro_min_conf:.1f} required"
                ] + pred.get("reasons", [])
            else:
                pred["confidence"] = round(float(max(_pre_macro_conf - _macro_bump, 25.0)), 1)
                pred["reasons"].append(
                    f"Macro throttle: size x{_macro_size_mult:.2f}, confidence -{_macro_bump:.0f}"
                )

        # Check if ticker already has an open paper trade
        _has_open_position = False
        try:
            from agent.paper_trading import get_open_trades as _get_open
            _open_tickers = {t["ticker"] for t in _get_open()}
            _has_open_position = ticker in _open_tickers
        except Exception:
            pass

        # ── Order flow analysis (PRD Section 3.2) ────────────────────────────
        _of = compute_order_flow(df_ind)
        _of_score = float(_of.get("score", 0.0))
        _of_label = _of.get("label", "NEUTRAL")

        # Order flow informs SIZE but never suppresses the signal.
        # A professional trader looks at order flow context to size the position,
        # not to decide whether to show the signal.
        _arb = get_signal_strength(pred["confidence"], _of_score, regime.regime)
        _sig_strength  = _arb["strength"]
        _sig_size_mult = float(_arb.get("size_mult", 1.0))
        _sig_size_mult *= _macro_size_mult
        if pred["direction"] not in ("BUY", "SELL", "STRONG BUY", "STRONG SELL"):
            _sig_size_mult = 0.0

        # ── Collect execution-level notes (shown on dashboard; do NOT force NEUTRAL)
        # Session, risk, and order-flow constraints are execution rules, not signal
        # filters.  The signal reflects what the market is doing.  Paper trading's
        # can_open_trade() decides whether to act on it.
        from agent.market_hours import no_new_entries, get_block_reason
        _trade_blocked_reason = ""
        # Extended-hours allowed: AH HIGH(50%), AH MODERATE(30%), PM HIGH(40%), PM MODERATE(25%)
        _is_extended_trade = (
            (_session_now == "AFTER_HOURS" and _trading_tier in ("HIGH", "MODERATE")) or
            (_session_now == "PRE_MARKET"  and _trading_tier in ("HIGH", "MODERATE"))
        )
        if _is_extended_trade:
            pass  # allowed — session gate above kept BUY/SELL direction
        elif no_new_entries():
            _trade_blocked_reason = get_block_reason()
        elif pred["direction"] in ("BUY", "SELL", "STRONG BUY", "STRONG SELL"):
            _cb_blocked, _cb_msg = check_circuit_breaker()
            if _cb_blocked:
                _trade_blocked_reason = _cb_msg
            else:
                _dt_blocked, _dt_msg = check_max_daily_trades()
                if _dt_blocked:
                    _trade_blocked_reason = _dt_msg
                else:
                    _sec_blocked, _sec_msg = check_sector_concentration(
                        ticker, "BUY" if "BUY" in pred["direction"] else "SELL"
                    )
                    if _sec_blocked:
                        _trade_blocked_reason = _sec_msg

        # ── Adaptive filter context enforcement ──────────────────────────────
        # should_suppress() checks the learned blocked_contexts (low win-rate
        # patterns) AND the dynamic confidence threshold. If this signal matches
        # a blocked context or is below the learned gate, set direction NEUTRAL
        # so it's recorded as suppressed and the learner sees the outcome.
        # This enforces the filter beyond advisory-only boost/penalty.
        _is_suppressed   = False
        _suppress_reason = ""
        if pred["direction"] in ("BUY", "SELL", "STRONG BUY", "STRONG SELL"):
            _is_suppressed, _suppress_reason = should_suppress(
                vwap_event   = vwap_sig.get("event", ""),
                session      = sess_info.get("session", ""),
                regime       = regime.regime,
                rsi_zone     = pred.get("rsi_zone", ""),
                entry_type   = pred.get("entry_type", ""),
                direction    = pred["direction"],
                sector_trend = sector_ctx.sector_trend,
                confidence   = pred["confidence"],
            )
            if _is_suppressed:
                _suppressed_dir = pred["direction"]   # capture before NEUTRAL overwrite
                pred["direction"] = "NEUTRAL"
                pred["reasons"]   = [f"⛔ AF: {_suppress_reason}"] + pred.get("reasons", [])
                # Audit the suppression decision with the ML scores behind the
                # rejected signal (durable decision trail — requirement 6).
                try:
                    from agent.audit_log import audit as _audit
                    _audit(
                        "SIGNAL_SUPPRESSED",
                        f"{ticker} {pred.get('entry_type','')} {_suppressed_dir} "
                        f"suppressed: {_suppress_reason}",
                        ticker=ticker, source="scanner",
                        detail={
                            "path":        "ml_prediction",
                            "confidence":  round(float(pred["confidence"]), 2),
                            "reason":      _suppress_reason,
                            "session":     sess_info.get("session", ""),
                            "regime":      regime.regime,
                            "vwap_event":  vwap_sig.get("event", ""),
                            "rsi_zone":    pred.get("rsi_zone", ""),
                            "ml_scalp":    round(float(ml_scalp), 4),
                            "ml_daily":    round(float(ml_daily_p), 4),
                            "ml_reversal": round(float(ml_reversal_p), 4),
                            "ml_ensemble": round(float(ml_ensemble_p), 4),
                            "ml_swing":    round(float(ml_swing_p), 4),
                            "ml_deep":     round(float(ml_deep_p), 4),
                        },
                    )
                except Exception:
                    pass

        # Normalise STRONG BUY → BUY and STRONG SELL → SELL for storage.
        # These are the highest-conviction signals and must not be silently dropped.
        _raw_direction = pred["direction"]
        _norm_direction = (
            "BUY"  if _raw_direction in ("BUY",  "STRONG BUY")  else
            "SELL" if _raw_direction in ("SELL", "STRONG SELL") else
            _raw_direction
        )

        # Per-ticker cooldown: only record a new signal if 15 min have passed
        # since the same ticker+direction was last recorded. This prevents the
        # signal tracker from counting a single setup 15 times in 15 minutes.
        _cooldown_key = (ticker, _norm_direction)
        _now_ts       = time.time()
        _last_ts      = _last_signal_ts.get(_cooldown_key, 0.0)
        _cooldown_ok  = (_now_ts - _last_ts) >= _SIGNAL_COOLDOWN_SECS

        # Record ALL directional signals for learning — regardless of session,
        # circuit breaker, or order flow state.  Every prediction the model makes
        # needs a resolved outcome so the ML can learn from it.
        # Earnings/macro blackouts are the only exception (no valid prediction).
        if _norm_direction in ("BUY", "SELL") and not eb["blocked"] and not macro_ev["blocked"]:
            resolve_pending(ticker, price)
        if _norm_direction in ("BUY", "SELL") and not eb["blocked"] and not macro_ev["blocked"] and _cooldown_ok:
            _last_signal_ts[_cooldown_key] = _now_ts
            record_signal(
                ticker=ticker, direction=_norm_direction, entry=price,
                target=pred["target_price"], stop=pred["stop_loss"],
                confidence=pred["confidence"],
                session=sess_info.get("session", ""),
                regime=regime.regime,
                trading_tier=pred.get("trading_tier", "REGULAR"),
                vwap_event=vwap_sig.get("event", ""),
                rsi_zone=pred.get("rsi_zone", ""),
                rel_volume=float(rvol),
                trend=pred.get("trend", ""),
            )
            bt_record(
                ticker       = ticker,
                direction    = _norm_direction,
                entry_price  = price,
                target       = pred["target_price"],
                stop         = pred["stop_loss"],
                rr_ratio     = pred["rr_ratio"],
                confidence   = pred["confidence"],
                session      = sess_info.get("session", ""),
                regime       = regime.regime,
                vwap_event   = vwap_sig["event"],
                rsi_zone     = pred.get("rsi_zone", ""),
                rsi_value    = float(pred.get("rsi_value", 50.0)),
                sector_etf   = sector_ctx.etf,
                sector_trend = sector_ctx.sector_trend,
                entry_type   = pred.get("entry_type", "IMMEDIATE"),
                mtf_alignment = mtf["alignment"],
            )
            # Paper trade execution — maybe_open_trade() re-enforces the session
            # gate and circuit breaker at the execution layer (defense-in-depth)
            # so a mid-scan halt can never be bypassed.
            # Extended-hours size caps: AH HIGH=50%, AH MODERATE=30%, PM HIGH=40%, PM MODERATE=25%.
            _pred_exec_status: list = []
            maybe_open_trade(
                ticker            = ticker,
                direction         = _norm_direction,
                price             = price,
                target            = pred["target_price"],
                stop              = pred["stop_loss"],
                confidence        = pred["confidence"],
                rr_qualifies      = bool(pred.get("rr_qualifies", False)),
                rr_ratio          = float(pred.get("rr_ratio", 0.0)),
                rr_quality        = pred.get("rr_quality", ""),
                session           = sess_info.get("session", ""),
                regime            = regime.regime,
                vwap_event        = vwap_sig["event"],
                rsi_zone          = pred.get("rsi_zone", ""),
                entry_type        = pred.get("entry_type", "IMMEDIATE"),
                order_flow_score  = _of_score,
                size_mult         = round(_sig_size_mult, 2),
                trading_tier      = _trading_tier,
                ml_scalp_prob     = ml_scalp,
                ml_daily_prob     = ml_daily_p,
                ml_swing_prob     = ml_swing_p,
                ml_deep_prob      = ml_deep_p,
                ml_ensemble_score = int(round(ml_ensemble_p * 100)),
                atr               = float(last.get("atr_14", 0.0)),
                avg_daily_volume  = float(df_ind["Volume"].mean() * 390) if "Volume" in df_ind.columns else 0.0,
                _out_status       = _pred_exec_status,
            )

        # Update open paper trades + live backtest tracking.
        # Pass bar_high/bar_low so stop/target detection uses the full intrabar
        # range rather than just the close — catches moves that spike through a
        # level then close back inside the same 1-min candle.
        update_open_trades(ticker, df_ind, price, bar_high=bar_high, bar_low=bar_low)
        bt_update(ticker, price, vwap_sig["vwap"], bar_high=bar_high, bar_low=bar_low)

        candles = _build_candles(df_1m)
        info    = _get_info(ticker)

        # Flag pattern detection (Algos 29 & 30)
        _flag = detect_flag(df_ind)

        # Update volatility state for Phase 2.3 risk control (ATR spike detection)
        try:
            if levels["session_high"] > 0 and levels["session_low"] > 0 and len(df_1d) >= 5:
                _sess_range_pct = (levels["session_high"] - levels["session_low"]) / levels["session_low"] * 100
                _daily_ranges = (df_1d["High"] - df_1d["Low"]).iloc[-20:]
                _avg_atr_abs  = float(_daily_ranges.mean()) if len(_daily_ranges) > 0 else 0.0
                _avg_atr_pct  = (_avg_atr_abs / float(df_1d["Close"].iloc[-1]) * 100) if float(df_1d["Close"].iloc[-1]) > 0 else 0.0
                update_volatility_state(_sess_range_pct, _avg_atr_pct)
        except Exception:
            pass

        _sig = StockSignal(
            ticker            = ticker,
            name              = info.get("name", ticker),
            price             = round(price, 4),
            change_pct        = change_pct,
            open_price        = round(_today_open, 4),
            technical         = round(tech, 4),
            volume            = round(vol, 4),
            ml_prob           = ml_combined,
            ml_daily_prob     = round(ml_daily_p, 4),
            sentiment         = round(sent, 4),
            score             = score,
            signal            = pred["direction"],
            rel_volume        = rvol,
            unusual_vol       = uvol,
            prediction        = pred["direction"],
            confidence        = round(pred["confidence"], 1),
            trend             = pred["trend"],
            trend_probability = round(float(pred.get("trend_probability", 0.5)), 2),
            ml_trained        = bool(pred.get("ml_trained", False)),
            target_price      = pred["target_price"],
            stop_loss         = pred["stop_loss"],
            rr_ratio          = pred["rr_ratio"],
            patterns          = pred["patterns"],
            reasons           = pred["reasons"],
            ml_swing_prob     = round(float(ml_swing_p), 4),
            ml_deep_prob      = round(float(ml_deep_p),  4),
            ml_swing_trained  = _swing_trained,
            ml_deep_trained   = _deep_trained,
            supports          = pred["supports"],
            resistances       = pred["resistances"],
            # Merge Camarilla pivot levels into the pivots dict
            pivots            = {
                **pred.get("pivots", {}),
                **{f"cam_{k}": round(float(v), 4) for k, v in _camarilla.items()
                   if isinstance(v, (int, float)) and v > 0},
            },
            poc               = float(_value_area.get("poc", 0.0)) or pred["poc"],
            mtf_score          = float(mtf["mtf_score"]),
            mtf_alignment      = mtf["alignment"],
            mtf_bull_count     = int(mtf["bull_count"]),
            mtf_bear_count     = int(mtf["bear_count"]),
            mtf_timeframes     = mtf["timeframes"],
            short_tf_alignment = mtf.get("short_tf_alignment", "MIXED"),
            mtf_gate_passed    = bool(mtf.get("mtf_gate_passed", False)),
            rsi_zone          = pred.get("rsi_zone",          "NEUTRAL"),
            rsi_value         = float(pred.get("rsi_value",  50.0)),
            rsi_gated         = bool(pred.get("rsi_gated",   False)),
            reversal_score    = float(pred.get("reversal_score",  0.0)),
            reversal_type     = pred.get("reversal_type",    "NONE"),
            divergence_type   = pred.get("divergence_type",  "NONE"),
            reversal_signals  = pred.get("reversal_signals", []),
            entry_type        = pred.get("entry_type",       "IMMEDIATE"),
            retest_level      = pred.get("retest_level",     0.0),
            entry_zone_low    = pred.get("entry_zone_low",   0.0),
            entry_zone_high   = pred.get("entry_zone_high",  0.0),
            exhaustion_flags  = pred.get("exhaustion_flags", []),
            bounce_signals    = pred.get("bounce_signals",   []),
            rr_quality        = pred.get("rr_quality",       "LOW"),
            rr_qualifies      = bool(pred.get("rr_qualifies", False)),
            # Market session
            session           = sess_info.get("session",   "UNKNOWN"),
            session_label     = sess_info.get("label",     ""),
            session_color     = sess_info.get("color",     "#94a3b8"),
            session_mult      = float(sess_info.get("mult", 1.0)),
            session_advice    = sess_info.get("advice",    ""),
            # Market regime
            regime            = regime.regime,
            regime_label      = regime.label,
            regime_color      = regime.color,
            # Earnings + context intelligence
            earnings_blocked   = bool(eb["blocked"]),
            earnings_reason    = eb["reason"],
            earnings_date      = eb["next_date"],
            earnings_days_away = int(eb["days_away"]),
            earnings_phase     = _ep,
            earnings_hour      = _earnings_hour,
            eps_surprise_pct   = _eps_surp,
            news_shock         = bool(_ctx.get("news_shock", False)),
            sentiment_velocity = round(float(_ctx.get("sentiment_velocity", 0.0)), 4),
            news_count_30m     = int(_ctx.get("news_count_30m", 0)),
            ctx_stale          = float(_ctx.get("stale_age_s", 0.0)) > 120,
            # Gap analysis
            gap_type          = gap["gap_type"],
            gap_pct           = float(gap["gap_pct"]),
            gap_score         = float(gap["gap_score"]),
            gap_filled        = bool(gap["gap_filled"]),
            gap_fill_prob     = float(gap["fill_probability"]),
            today_open        = float(gap["today_open"]),
            premarket_high    = float(gap["premarket_high"]),
            premarket_low     = float(gap["premarket_low"]),
            # PDH / PDL / ORB levels (30-min)
            prev_day_high     = levels["prev_day_high"],
            prev_day_low      = levels["prev_day_low"],
            prev_day_close    = levels["prev_day_close"],
            price_vs_pdh_pct  = round((price - levels["prev_day_high"]) / levels["prev_day_high"] * 100, 3)
                                if levels["prev_day_high"] > 0 else 0.0,
            price_vs_pdl_pct  = round((price - levels["prev_day_low"]) / levels["prev_day_low"] * 100, 3)
                                if levels["prev_day_low"] > 0 else 0.0,
            session_high      = levels["session_high"],
            session_low       = levels["session_low"],
            # Flag patterns
            bull_flag  = bool(_flag["bull_flag"]),
            bear_flag  = bool(_flag["bear_flag"]),
            flag_high  = float(_flag["flag_high"]),
            flag_low   = float(_flag["flag_low"]),
            pole_pct   = float(_flag["pole_pct"]),
            # R2G / G2R
            color_vs_prev_close = color_flip["color"],
            r2g_event           = bool(color_flip["r2g_event"]),
            g2r_event           = bool(color_flip["g2r_event"]),
            r2g_bars_ago        = int(color_flip["r2g_bars_ago"]),
            g2r_bars_ago        = int(color_flip["g2r_bars_ago"]),
            orb_high          = levels["orb_high"],
            orb_low           = levels["orb_low"],
            orb_breakout      = levels["orb_breakout"],
            # ORB-5 (first 5-min opening range 9:30–9:35)
            orb5_high         = orb_result.orh_5,
            orb5_low          = orb_result.orl_5,
            orb5_breakout     = orb_result.breakout_5,
            orb5_score        = orb_result.or5_score,
            # ORB-15 (first 15-min opening range)
            orb15_high        = orb_result.orh_15,
            orb15_low         = orb_result.orl_15,
            orb15_breakout    = orb_result.breakout_15,
            orb15_score       = orb_result.or15_score,
            orb30_score       = orb_result.or30_score,
            # Value Area (VAH/VAL)
            vah               = float(_value_area.get("vah", 0.0)),
            val               = float(_value_area.get("val", 0.0)),
            # Day type
            day_type          = day_type_info.day_type,
            day_type_label    = day_type_info.label,
            # Fibonacci retracement levels
            fib_levels        = {
                k: round(float(v), 4) for k, v in _fibonacci.items()
                if isinstance(v, (int, float)) and v > 0
            },
            # Relative strength
            rs_ratio          = float(rs["rs_ratio"]),
            rs_score          = float(rs["rs_score"]),
            rs_label          = rs["rs_label"],
            # VWAP signal + dynamic σ-band levels
            vwap_event        = vwap_sig["event"],
            vwap_score        = float(vwap_sig["score"]),
            vwap_price        = float(vwap_sig["vwap"]),
            vwap_deviation    = float(vwap_sig["deviation"]),
            vwap_description  = vwap_sig["description"],
            vwap_z_score      = float(vwap_sig.get("z_score",  0.0)),
            vwap_upper_1      = float(vwap_sig.get("upper_1",  0.0)),
            vwap_lower_1      = float(vwap_sig.get("lower_1",  0.0)),
            vwap_upper_2      = float(vwap_sig.get("upper_2",  0.0)),
            vwap_lower_2      = float(vwap_sig.get("lower_2",  0.0)),
            # Sector
            sector_etf        = sector_ctx.etf,
            sector_trend      = sector_ctx.sector_trend,
            sector_change     = float(sector_ctx.sector_change),
            stock_vs_sector   = sector_ctx.stock_vs_sector,
            # Exit signals
            exit_recommendation = exit_analysis.recommendation,
            exit_signals      = [s.__dict__ for s in exit_analysis.signals],
            exit_summary      = exit_analysis.summary,
            # Macro
            macro_blocked     = bool(macro_ev["blocked"]),
            macro_throttled   = bool(macro_ev.get("throttled", False)),
            macro_size_mult   = float(macro_ev.get("size_mult", 1.0) or 1.0),
            macro_min_confidence = float(macro_ev.get("min_confidence", 0.0) or 0.0),
            macro_event       = macro_ev["event_name"],
            macro_description = macro_ev["description"],
            # Trade plan
            trade_plan          = tp.to_dict(),
            candles             = candles,
            headlines           = headlines[:5],
            tech_row            = {
                k: (float(v) if isinstance(v, (int, float)) and v == v else 0.0)
                for k, v in last.items()
                if not isinstance(v, (list, dict))
            },
            is_suppressed       = bool(_trade_blocked_reason),
            suppress_reason     = _trade_blocked_reason,
            has_open_position   = _has_open_position,
            ah_change_pct       = _ah_change,
            ah_direction        = _ah_dir,
            ah_magnitude        = _ah_mag,
            ah_confirms_signal  = _ah_confirms,
            ah_news_likely      = _ah_news,
            ah_gap_estimate     = _ah_gap_est,
            trading_tier        = get_trading_tier(
                ticker,
                float(_ah_bias.get("ah_volume_ratio", 0.0)) if _ah_bias else 0.0,
            ),
            order_flow_score    = _of_score,
            order_flow_label    = _of_label,
            signal_strength     = _sig_strength,
            signal_size_mult    = _sig_size_mult,
        )

        # Evaluate structured trading algorithm signals (Phase 1+)
        try:
            _sig.algo_signals = evaluate_trading_algos(_sig)
        except Exception as _ae:
            logger.debug("[%s] trading_algos error: %s", ticker, _ae)

        # Log algo fires and open algo-driven paper trades
        if _sig.algo_signals:
            # Apply AlgoSelector weights to confidence scores before processing
            if _ALE_AVAILABLE and _sig.algo_signals:
                try:
                    _context_key = f"{regime.regime}:{sess_info.get('session','')}:{vwap_sig.get('event','')}"
                    _algo_names  = [a["algo"] for a in _sig.algo_signals]
                    _sel_weights = _get_sel_weights(_algo_names, _context_key)
                    for _asig in _sig.algo_signals:
                        _asig["confidence"] = round(float(min(max(
                            _asig["confidence"] * _sel_weights.get(_asig["algo"], 1.0),
                            25.0), 95.0)), 1)
                except Exception as _ale_err:
                    logger.debug("[%s] AlgoSelector weight error: %s", ticker, _ale_err)

            _algo_trade_opened = False
            for _asig in _sig.algo_signals:
                # Attach the ML scores behind this signal so algo_signal_log
                # persists the full ML decision for EVERY signal (accepted,
                # rejected, or shadow) — completes the ML decision trail.
                _asig["ml_scalp_prob"] = round(float(ml_scalp), 4)
                _asig["ml_daily_prob"] = round(float(ml_daily_p), 4)
                _asig["ml_swing_prob"] = round(float(ml_swing_p), 4)
                _asig["ml_deep_prob"]  = round(float(ml_deep_p), 4)
                # ── Session gate for algo signals: never open when CLOSED ───────────
                # The primary prediction path neutralises pred["direction"] for CLOSED
                # sessions, but algo signals carry their own raw direction and bypass
                # that gate.  Gate them here so maybe_open_trade is not even called.
                if _session_now == "CLOSED":
                    _asig["exec_status"] = "BLOCKED_CLOSED"
                    logger.debug(
                        "[%s] %s algo signal skipped — market CLOSED", ticker, _asig["algo"]
                    )
                    continue

                if macro_ev.get("blocked"):
                    _asig["exec_status"] = "MACRO_BLOCKED"
                    logger.debug(
                        "[%s] %s algo signal skipped - macro hard block: %s",
                        ticker, _asig["algo"], macro_ev.get("event_name", ""),
                    )
                    continue
                if macro_ev.get("throttled"):
                    _algo_conf = float(_asig.get("confidence", 0.0) or 0.0)
                    _algo_min_conf = float(macro_ev.get("min_confidence", 0.0) or 0.0)
                    if _algo_conf < _algo_min_conf:
                        _asig["exec_status"] = "MACRO_THROTTLED"
                        logger.debug(
                            "[%s] %s algo skipped - macro throttle conf %.1f < %.1f",
                            ticker, _asig["algo"], _algo_conf, _algo_min_conf,
                        )
                        continue
                    _asig["confidence"] = round(float(max(
                        _algo_conf - float(macro_ev.get("confidence_bump", 0.0) or 0.0),
                        25.0,
                    )), 1)

                # Record algo signals in bt_signals for learning engine analysis
                try:
                    bt_record(
                        ticker        = ticker,
                        direction     = _asig["direction"],
                        entry_price   = float(_asig["entry"]),
                        target        = float(_asig["target"]),
                        stop          = float(_asig["stop"]),
                        rr_ratio      = float(_asig.get("rr", 0)),
                        confidence    = float(_asig["confidence"]),
                        session       = sess_info.get("session", ""),
                        regime        = regime.regime,
                        vwap_event    = vwap_sig.get("event", ""),
                        rsi_zone      = pred.get("rsi_zone", ""),
                        rsi_value     = float(pred.get("rsi_value", 50.0)),
                        sector_etf    = sector_ctx.etf,
                        sector_trend  = sector_ctx.sector_trend,
                        entry_type    = "ALGO",
                        mtf_alignment = mtf["alignment"],
                        algo_name     = _asig["algo"],
                    )
                except Exception as _bt_err:
                    logger.debug("[%s] bt_record algo error: %s", ticker, _bt_err)

                # ── Increment consecutive-fire counter (used by entry_window_bars) ───
                _sig_key = f"{ticker}:{_asig['algo']}"
                with _sig_consec_lock:
                    _consec = _sig_consec.get(_sig_key, 0) + 1
                    _sig_consec[_sig_key] = _consec

                # ── conf_gate: per-algo-family learned minimum confidence ────────────
                if _ALE_AVAILABLE:
                    try:
                        _cg = float(_get_algo_params(_asig["algo"]).get("conf_gate", 55.0))
                        if float(_asig["confidence"]) < _cg:
                            _asig["exec_status"] = "BLOCKED_CONF_GATE"
                            logger.debug(
                                "[%s] %s conf %.1f < conf_gate %.1f — suppressed",
                                ticker, _asig["algo"], _asig["confidence"], _cg,
                            )
                            continue
                    except Exception as _cge:
                        logger.debug("[%s] conf_gate error: %s", ticker, _cge)

                # ── entry_window_bars: staleness guard ───────────────────────────────
                # Signals that have been firing for too many consecutive scan cycles
                # are considered stale (breakout already digested — don't chase).
                if _ALE_AVAILABLE:
                    try:
                        _ew = int(_get_algo_params(_asig["algo"]).get("entry_window_bars", 3))
                        if _consec > _ew:
                            _asig["exec_status"] = "BLOCKED_STALE_SIGNAL"
                            logger.debug(
                                "[%s] %s stale: cycle %d > entry_window %d — trade skipped",
                                ticker, _asig["algo"], _consec, _ew,
                            )
                            continue
                    except Exception as _ewe:
                        logger.debug("[%s] entry_window error: %s", ticker, _ewe)

                # Phase 2 staged deployment: SHADOW (observe only) | PAPER | LIVE
                _routing = "PAPER"
                if _ALE_AVAILABLE:
                    try:
                        _routing = _get_ale().get_routing(_asig["algo"])
                    except Exception as _re:
                        logger.debug("[%s] routing error: %s", ticker, _re)

                _trade_id = None
                if _routing == "SHADOW":
                    _asig["exec_status"] = "SHADOW_ROUTING"
                    logger.debug(
                        "[%s] %s → SHADOW routing; signal recorded, paper trade skipped",
                        ticker, _asig["algo"],
                    )
                else:
                    # Apply adaptive filter to algo signals (checks per-family blocking)
                    _algo_suppressed, _algo_suppress_reason = should_suppress(
                        vwap_event   = vwap_sig.get("event", ""),
                        session      = sess_info.get("session", ""),
                        regime       = regime.regime,
                        rsi_zone     = pred.get("rsi_zone", ""),
                        entry_type   = "ALGO",
                        direction    = _asig["direction"],
                        sector_trend = sector_ctx.sector_trend,
                        algo_name    = _asig["algo"],
                        confidence   = float(_asig["confidence"]),
                    )
                    if _algo_suppressed:
                        _asig["exec_status"] = "FILTER_SUPPRESSED"
                        _asig["filter_reason"] = _algo_suppress_reason
                        logger.debug(
                            "[%s] %s suppressed by adaptive filter: %s",
                            ticker, _asig["algo"], _algo_suppress_reason,
                        )
                        try:
                            from agent.audit_log import audit as _audit
                            _audit(
                                "SIGNAL_SUPPRESSED",
                                f"{ticker} {_asig['algo']} {_asig['direction']} "
                                f"suppressed: {_algo_suppress_reason}",
                                ticker=ticker, source="scanner",
                                detail={
                                    "path":       "algo",
                                    "algo":       _asig["algo"],
                                    "direction":  _asig["direction"],
                                    "confidence": round(float(_asig["confidence"]), 2),
                                    "reason":     _algo_suppress_reason,
                                    "session":    sess_info.get("session", ""),
                                    "regime":     regime.regime,
                                    "ml_scalp":   round(float(ml_scalp), 4),
                                    "ml_daily":   round(float(ml_daily_p), 4),
                                    "ml_swing":   round(float(ml_swing_p), 4),
                                    "ml_deep":    round(float(ml_deep_p), 4),
                                },
                            )
                        except Exception:
                            pass
                    else:
                        _asig_status: list = []
                        _algo_rr = float(_asig.get("rr", 0))
                        _algo_target_rr = get_execution_min_rr(_asig["algo"], "ALGO")
                        _algo_rr_quality = (
                            "EXCELLENT" if _algo_rr >= 4.0 else
                            "GOOD" if _algo_rr >= 3.0 else
                            "OK" if _algo_rr > 0 else
                            "LOW"
                        )
                        _trade_id = maybe_open_trade(
                            ticker            = ticker,
                            direction         = _asig["direction"],
                            price             = float(_asig["entry"]),
                            target            = float(_asig["target"]),
                            stop              = float(_asig["stop"]),
                            confidence        = float(_asig["confidence"]),
                            rr_qualifies      = True,
                            rr_ratio          = max(_algo_rr, _algo_target_rr),
                            rr_quality        = _algo_rr_quality,
                            session           = sess_info.get("session", ""),
                            regime            = regime.regime,
                            entry_type        = "ALGO",
                            order_flow_score  = _of_score,
                            size_mult         = float(macro_ev.get("size_mult", 1.0) or 1.0),
                            trading_tier      = _trading_tier,
                            algo_name         = _asig["algo"],
                            ml_scalp_prob     = ml_scalp,
                            ml_daily_prob     = ml_daily_p,
                            ml_swing_prob     = ml_swing_p,
                            ml_deep_prob      = ml_deep_p,
                            ml_ensemble_score = int(round(ml_ensemble_p * 100)),
                            atr               = float(last.get("atr_14", 0.0)),
                            avg_daily_volume  = float(df_ind["Volume"].mean() * 390) if "Volume" in df_ind.columns else 0.0,
                            _out_status       = _asig_status,
                        )
                        _asig["exec_status"] = _asig_status[0] if _asig_status else (
                            "EXECUTED_PAPER" if _trade_id else "SHADOW_LEARN_ONLY"
                        )
                        _asig["trade_opened"] = bool(_trade_id)
                if _trade_id:
                    _algo_trade_opened = True
            try:
                log_algo_signals(ticker, _sig.algo_signals, trade_opened=_algo_trade_opened)
            except Exception as _le:
                logger.debug("[%s] algo log error: %s", ticker, _le)

        # Reset consecutive counters for signals that stopped firing this cycle
        # so the next fire is treated as fresh.
        _fired = {f"{ticker}:{s['algo']}" for s in (_sig.algo_signals or [])}
        with _sig_consec_lock:
            for _k in [k for k in list(_sig_consec) if k.startswith(f"{ticker}:")]:
                if _k not in _fired:
                    del _sig_consec[_k]

        return _sig
    except Exception as e:
        logger.warning(f"[{ticker}] analysis error: {e}", exc_info=True)
        return None


# ── Scanner ───────────────────────────────────────────────────────────────────

class Scanner:
    def __init__(self):
        self.signals:             list[StockSignal] = []
        self.last_scan:           Optional[str]     = None
        self.is_running:          bool              = False
        self._callbacks:          list[Callable]    = []
        self._per_ticker_cbs:     list[Callable]    = []
        self._last_retrain:       float             = 0.0
        self._last_deep_finetune: float             = 0.0   # hourly BiLSTM fine-tune
        self._first_scan_done:    threading.Event   = threading.Event()
        self._second_scan_done:   threading.Event   = threading.Event()
        self._scan_count:         int               = 0
        self._scan_cursor:        int               = 0
        self._last_training_defer_log: float        = 0.0

    def register_callback(self, fn: Callable) -> None:
        self._callbacks.append(fn)

    def register_per_ticker_callback(self, fn: Callable) -> None:
        """Register a callback invoked immediately as each ticker finishes analysis.
        Signature: fn(signal: StockSignal, n_done: int, n_total: int) -> None
        """
        self._per_ticker_cbs.append(fn)

    def _notify(self, signals: list[StockSignal]) -> None:
        for fn in self._callbacks:
            try:
                fn(signals)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def _notify_ticker(self, sig, n_done: int, n_total: int) -> None:
        for fn in self._per_ticker_cbs:
            try:
                fn(sig, n_done, n_total)
            except Exception as e:
                logger.debug(f"Per-ticker callback error: {e}")

    def _training_session(self) -> str:
        """Return broad market session for heavy training gates."""
        try:
            return get_market_session()
        except Exception as exc:
            logger.warning(f"[ML-Retrain] Session check failed ({exc}) — deferring training")
            return "UNKNOWN"

    def _training_allowed_now(self) -> bool:
        # Heavy model training competes with scanner/API threads. Keep it out of
        # REGULAR, PRE_MARKET, and AFTER_HOURS so the dashboard can stay live.
        return self._training_session() == "CLOSED"

    def _log_training_deferred(self, reason: str, session: str) -> None:
        now = time.time()
        if now - self._last_training_defer_log < 300:
            return
        self._last_training_defer_log = now
        logger.info(f"[ML-Retrain] {reason} deferred — session={session}; waiting for CLOSED window")

    def _should_retrain(self) -> bool:
        if self._last_retrain == 0.0:
            return False  # startup training running in background
        if (time.time() - self._last_retrain) <= ML_RETRAIN_INTERVAL:
            return False
        session = self._training_session()
        if session != "CLOSED":
            self._log_training_deferred("Scheduled retrain", session)
            return False
        return True

    def _should_finetune_deep(self) -> bool:
        """True when 1 hour has passed since last Deep BiLSTM fine-tune."""
        if self._last_deep_finetune == 0.0:
            return False  # avoid racing with startup full train
        if (time.time() - self._last_deep_finetune) <= DEEP_FINETUNE_INTERVAL:
            return False
        session = self._training_session()
        if session != "CLOSED":
            self._log_training_deferred("Deep BiLSTM fine-tune", session)
            return False
        return True

    # ── ML training ───────────────────────────────────────────────────────────

    def _train_ml_background(self) -> None:
        """
        Train both intraday (5M) and daily ML models at startup.
        Daily data served from fetch_batch_interval cache (fetched during first scan).

        Wait for the first scan to complete before training touches the API.
        First scan populates 1min, 5min, 1h, 1day SQLite caches via Schwab REST.
        Starting retrain during that window would compete for the API rate budget
        and potentially trigger 429s that slow the scan.
        """
        logger.info("ML training: waiting for first two completed scan cycles to warm data cache…")
        self._first_scan_done.wait(timeout=600)
        self._second_scan_done.wait(timeout=900)

        while not self._training_allowed_now():
            session = self._training_session()
            self._log_training_deferred("Startup retrain", session)
            time.sleep(300)
            if not self.is_running:
                logger.info("[ML-Retrain] Startup retrain cancelled — scanner stopped")
                return

        logger.info(f"ML training starting ({len(TRAINING_TICKERS)} Tier-1 tickers)…")
        daily_data = fetch_batch_interval(TRAINING_TICKERS, "1day", 500, ttl=CACHE_TTL_1D)
        retrain_all(TRAINING_TICKERS, daily_data=daily_data, skip_deep=True)
        self._last_retrain       = time.time()
        self._last_deep_finetune = time.time()   # deep phase is deferred to scheduled fine-tune
        logger.info("ML training complete; Deep BiLSTM deferred to scheduled fine-tune.")

    # ── Scan loop ─────────────────────────────────────────────────────────────

    def run_once(self) -> list[StockSignal]:
        t0 = time.time()
        active_tickers = get_active_tickers()

        # Mark scan as active so the healthcheck won't flag a slow cycle as hung.
        # Key expires in 10 min — well above any realistic cycle length.
        try:
            from agent.valkey_client import _get_client as _vk_active_c
            _vk_active = _vk_active_c()
            if _vk_active:
                _vk_active.setex("scan:active", 600, str(t0))
        except Exception:
            pass

        # ── Market-holiday guard ─────────────────────────────────────────────
        # On NYSE/NASDAQ holidays the market is CLOSED all day. Skip the full
        # 250-ticker data fetch and signal analysis to conserve Schwab API quota.
        # Return the cached signals so the dashboard stays populated.
        _sess_check = get_session_info()
        if _sess_check.get("is_holiday", False):
            logger.info("[Scanner] Market holiday — returning cached signals, skipping scan")
            return list(self.signals)

        # ── EOD Hard Close (PRD 6.4): close all positions at 3:45 PM ET ──────
        from agent.market_hours import (
            is_hard_close_window, is_closing_caution, no_new_entries,
            is_ah_eod_close_window, is_after_hours,
        )

        # ── Session-driven position management (mutually exclusive branches) ───
        if is_ah_eod_close_window():
            # 7:55 PM ET: hard close all remaining AH positions
            try:
                from agent.paper_trading import close_all_positions_eod
                closed_n = close_all_positions_eod(reason="AH_EOD_19:55", extended_hours=True)
                if closed_n:
                    logger.info(f"[Scanner] AH EOD hard close 7:55pm — {closed_n} positions closed")
            except Exception as _ahc_e:
                logger.warning(f"[Scanner] AH EOD 7:55pm close failed: {_ahc_e}")

        elif is_hard_close_window():
            # 3:45 PM ET: hard close all positions — non-overridable PRD rule
            try:
                from agent.paper_trading import close_all_positions_eod
                closed_n = close_all_positions_eod()
                if closed_n:
                    logger.info(f"[Scanner] EOD hard close triggered — {closed_n} positions closed")
            except Exception as _eod_e:
                logger.warning(f"[Scanner] EOD hard close failed: {_eod_e}")

        # ── After-hours stale-trade sweep — catches anything missed at 3:45 ──
        # Runs for no_new_entries() (HARD_CLOSE/CLOSED) AND during AFTER_HOURS.
        # close_stale_positions() is session-aware: during AH it only closes
        # regular-hour trades that leaked past 3:45 PM; intentional AH positions
        # are left open until the 7:55 PM AH EOD close.
        elif no_new_entries() or is_after_hours():
            try:
                from agent.paper_trading import close_stale_positions
                swept_n = close_stale_positions()
                if swept_n:
                    logger.info(f"[Scanner] After-hours sweep closed {swept_n} stale positions")
            except Exception as _sw_e:
                logger.warning(f"[Scanner] After-hours sweep failed: {_sw_e}")

        # ── Smart EOD pre-close (3:30–3:44 PM ET) ────────────────────────────
        elif is_closing_caution():
            try:
                from agent.paper_trading import smart_eod_review
                acted_n = smart_eod_review()
                if acted_n:
                    logger.info(f"[Scanner] Smart EOD review — acted on {acted_n} positions")
            except Exception as _eod_e:
                logger.warning(f"[Scanner] Smart EOD review failed: {_eod_e}")

        # ── Pre-market gapper scanner ─────────────────────────────────────────
        try:
            from agent.premarket_scanner import should_run_scan, run_premarket_scan_background
            if should_run_scan():
                run_premarket_scan_background()
                logger.info("[Scanner] Pre-market gapper scan launched in background")
        except Exception as _pm_e:
            logger.debug(f"[Scanner] Pre-market scan check failed: {_pm_e}")

        from config import SCHWAB_ENABLED

        # ── Schwab screener priority (only when Schwab is enabled) ────────────
        if SCHWAB_ENABLED:
            try:
                from agent.broker.schwab_streamer import get_screener_priority
                mover_symbols = get_screener_priority(tickers=set(active_tickers))
                if mover_symbols:
                    rest = [t for t in active_tickers if t not in set(mover_symbols)]
                    active_tickers = mover_symbols + rest
                    logger.debug(f"[Scanner] Screener priority: {mover_symbols[:5]}…")
            except Exception:
                pass

        # ── Prioritise today's pre-market gapper watchlist ────────────────────
        try:
            from agent.premarket_scanner import get_focus_watchlist
            _focus = get_focus_watchlist()
            if _focus:
                _rest = [t for t in active_tickers if t not in set(_focus)]
                active_tickers = _focus + _rest
                logger.debug(f"[Scanner] PM focus tickers: {_focus[:8]}")
        except Exception:
            pass

        # Cycle-budgeted scans must not starve the tail of the active universe.
        # Rotate the start point each cycle; completed rows are merged back into
        # the prior dashboard snapshot below, so high-priority names remain
        # visible while the scan cursor works through the full active set.
        _scan_cursor = 0
        if active_tickers:
            _scan_cursor = self._scan_cursor % len(active_tickers)
            if _scan_cursor:
                active_tickers = active_tickers[_scan_cursor:] + active_tickers[:_scan_cursor]

        logger.info(f"Scan starting — {len(active_tickers)} tickers…")

        # Detect session once so the data fetch uses the right mode
        _sess = get_session_info()
        # CLOSED (8pm–4am ET) still needs extended bars so the chart shows the
        # 4pm–8pm AH session candles rather than stopping at the 4pm close.
        _is_extended = _sess.get("session", "") in ("AFTER_HOURS", "PRE_MARKET", "CLOSED")

        # ── Data fetch strategy ───────────────────────────────────────────────
        # 1min fetched first (live/streaming-first with REST fallback).
        # HTF intervals (5min, 1h, 1day) fetched sequentially after 1min is done:
        # in steady state they are 100% cache hits (TTL 10min/1h/24h) and return
        # in milliseconds so sequential costs nothing.  On cold start, running
        # them in parallel caused 3×10 = 30 concurrent Schwab /pricehistory
        # calls alongside the MD poller, exceeding the 120 req/min rate limit
        # and triggering 429 storms.  Sequential keeps peak concurrency at ≤10.
        batch_1m = fetch_batch_realtime(active_tickers, extended_hours=_is_extended)
        batch_5m = fetch_batch_interval(active_tickers, "5min", 500, CACHE_TTL_5M)
        if _sess.get("session", "").upper() == "CLOSED":
            restored_from_5m: list[str] = []
            try:
                from agent.valkey_client import get_all_prices as _vk_all_prices
                _latest_prices = _vk_all_prices()
            except Exception:
                _latest_prices = {}
            for _ticker in active_tickers:
                if _ticker in batch_1m:
                    continue
                _df5 = batch_5m.get(_ticker)
                if _df5 is None or _df5.empty or len(_df5) < 5:
                    continue
                _df_proxy = _df5.copy()
                try:
                    _quote = _latest_prices.get(_ticker, {}) if isinstance(_latest_prices, dict) else {}
                    _last_px = float(_quote.get("last") or 0)
                    if _last_px > 0 and {"Close", "High", "Low"}.issubset(_df_proxy.columns):
                        _last_idx = _df_proxy.index[-1]
                        _df_proxy.loc[_last_idx, "Close"] = _last_px
                        _df_proxy.loc[_last_idx, "High"] = max(float(_df_proxy.loc[_last_idx, "High"]), _last_px)
                        _df_proxy.loc[_last_idx, "Low"] = min(float(_df_proxy.loc[_last_idx, "Low"]), _last_px)
                except Exception:
                    pass
                batch_1m[_ticker] = _df_proxy
                restored_from_5m.append(_ticker)
            if restored_from_5m:
                logger.warning(
                    "Closed-session scan using 5min cached bars as a 1min proxy "
                    "for %d tickers because live/REST 1min data is unavailable.",
                    len(restored_from_5m),
                )
        batch_1h = fetch_batch_interval(active_tickers, "1h",   500, CACHE_TTL_1H)
        # On cold start (first scan), all three prior fetches made real API calls.
        # A brief pause lets Schwab's per-minute bucket partially refill before
        # the 1day sweep, preventing the 429 storm seen in production logs.
        # On warm scans (cache hits), this branch is never reached.
        if self._scan_count == 0:
            time.sleep(8)
        batch_1d = fetch_batch_interval(active_tickers, "1day", 500, CACHE_TTL_1D)

        # Fetch SPY/QQQ + sector ETFs for regime, RS and sector context
        etf_1m = fetch_batch_realtime(SECTOR_ETF_TICKERS, extended_hours=_is_extended)
        _spy_df_cache.update(etf_1m)
        update_etf_cache(etf_1m)
        update_regime(
            df_spy=etf_1m.get("SPY"),
            df_qqq=etf_1m.get("QQQ"),
        )

        # (active_tickers already priority-ordered above — do NOT reload here)

        # Parallel scan — ThreadPoolExecutor runs analyse_ticker() concurrently
        # across all tickers. 8× faster than the old sequential for-loop.
        from agent.pipeline import get_pipeline
        _n_total = len(active_tickers)
        results = get_pipeline(n_workers=_pipeline_workers()).scan(
            active_tickers, batch_1m, batch_5m, batch_1h, batch_1d,
            on_ticker_done=lambda sig, n, t: self._notify_ticker(sig, n, _n_total),
        )

        if not results:
            self.last_scan = datetime.now(timezone.utc).isoformat()
            self._scan_count += 1
            if self.signals:
                logger.warning(
                    "Scan produced 0/%d active tickers; preserving previous "
                    "%d in-memory signals instead of clearing the dashboard.",
                    len(active_tickers),
                    len(self.signals),
                )
                self._notify(self.signals)
                return list(self.signals)
            logger.warning(
                "Scan produced 0/%d active tickers and no previous in-memory "
                "signals are available; leaving the dashboard snapshot unchanged.",
                len(active_tickers),
            )
            return []

        # Attach per-ticker learning scores and compute learning_rank.
        # Fetched once per scan cycle (single DB query for all tickers).
        try:
            _ticker_scores = get_ticker_learning_scores()
            for sig in results:
                ts = _ticker_scores.get(sig.ticker)
                if ts:
                    sig.ticker_win_rate  = ts["win_rate"]
                    sig.ticker_obs_count = ts["count"]
                    # learning_rank = signal strength × historical accuracy
                    # 0.5 baseline so new/unlearned tickers still appear
                    sig.learning_rank = round(abs(sig.score) * (0.5 + sig.ticker_win_rate), 4)
                else:
                    sig.learning_rank = round(abs(sig.score) * 0.5, 4)
        except Exception as _lr_err:
            logger.debug(f"learning_rank error: {_lr_err}")

        # A partial scan must never replace the full dashboard with only the
        # tickers that completed inside the cycle budget. Merge completed rows
        # into the previous snapshot, and create lightweight watch-only rows for
        # active tickers that have price data but no completed analysis yet.
        _actual_by_ticker = {s.ticker: s for s in results}
        _previous_by_ticker = {s.ticker: s for s in self.signals}
        _merged_by_ticker: dict[str, StockSignal] = {}
        try:
            from agent.valkey_client import get_all_prices as _get_all_prices
            _latest_quotes = _get_all_prices() or {}
        except Exception:
            _latest_quotes = {}
        _fallback_count = 0
        _preserved_count = 0
        for _ticker in active_tickers:
            if _ticker in _actual_by_ticker:
                _merged_by_ticker[_ticker] = _actual_by_ticker[_ticker]
            elif _ticker in _previous_by_ticker:
                _merged_by_ticker[_ticker] = _previous_by_ticker[_ticker]
                _preserved_count += 1
            else:
                _fallback = _observation_signal(
                    _ticker,
                    batch_1m.get(_ticker),
                    batch_1d.get(_ticker),
                    session=_sess,
                    quote=_latest_quotes.get(_ticker, {}) if isinstance(_latest_quotes, dict) else {},
                    reason="Analysis deferred by scanner cycle budget",
                )
                if _fallback is not None:
                    _merged_by_ticker[_ticker] = _fallback
                    _fallback_count += 1

        for _ticker, _sig in _previous_by_ticker.items():
            if _ticker not in _merged_by_ticker:
                _merged_by_ticker[_ticker] = _sig

        merged_results = list(_merged_by_ticker.values())
        merged_results.sort(key=lambda s: abs(s.score), reverse=True)

        self.signals   = merged_results
        self.last_scan = datetime.now(timezone.utc).isoformat()

        # Market-observation learning: record all signals with full context,
        # then check short-term price accuracy against previous scan's signals.
        try:
            resolve_short_term(results)
            record_signals_batch(results)
        except Exception as _st_err:
            logger.debug(f"signal_tracker batch error: {_st_err}")

        # Feed short-term signal accuracy directly into the adaptive filter.
        # This is the key learning loop that does NOT depend on paper trades:
        # every 90 s we measure if prices moved in the predicted direction and
        # use that to update the confidence gate.  Without this, the filter can
        # only learn from paper trade closures — which can't happen if the gate
        # is too high and no trades open (deadlock).
        try:
            from agent.signal_tracker import get_market_breakdown_stats as _st_stats
            from agent.adaptive_filter import update_filter as _af_live
            _live_stats = _st_stats()
            if _live_stats.get("overall", {}).get("total", 0) >= 5:
                _af_live(_live_stats, source="observation")
        except Exception as _af_live_err:
            logger.debug(f"live adaptive filter update error: {_af_live_err}")

        self._notify(merged_results)
        elapsed = round(time.time() - t0, 1)
        self._scan_count += 1
        if active_tickers:
            self._scan_cursor = (_scan_cursor + max(len(results), _pipeline_workers(), 1)) % len(active_tickers)
        if self._scan_count == 1:
            self._first_scan_done.set()   # unblock _train_ml_background
        if self._scan_count >= 2:
            self._second_scan_done.set()
        logger.info(
            "Scan complete in %ss | %d/%d analyzed, %d rows published "
            "(%d preserved, %d observation fallback; universe: %d)",
            elapsed,
            len(results),
            len(active_tickers),
            len(merged_results),
            _preserved_count,
            _fallback_count,
            len(NASDAQ_TICKERS),
        )

        # Clear the active-scan marker now that the cycle completed cleanly.
        try:
            from agent.valkey_client import _get_client as _vk_done_c
            _vk_done = _vk_done_c()
            if _vk_done:
                _vk_done.delete("scan:active")
        except Exception:
            pass

        # ML feedback/retraining is owned by learner services in production.
        # Keeping this out of the scanner protects price freshness and scan SLA.
        if _scanner_training_enabled():
            maybe_trigger_feedback_retrain(active_tickers)

        if _scanner_training_enabled() and self._should_retrain():
            logger.info(f"Scheduled ML retrain launching ({len(TRAINING_TICKERS)} Tier-1 tickers)…")
            self._last_retrain       = time.time()
            self._last_deep_finetune = time.time()   # full retrain counts as fine-tune too
            # Run in a daemon thread so XGBoost/sklearn training runs in the same
            # process and directly updates the in-memory _model_registry.
            # Subprocess approach (old) saved to disk but never reloaded models
            # into the main process — predictions used stale models until restart.
            def _scheduled_retrain():
                try:
                    from agent.ml_model import retrain_all as _retrain_all
                    from agent.data_fetcher import fetch_batch_interval as _fetch
                    from config import CACHE_TTL_1D as _TTL_1D
                    logger.info(f"[ML-Retrain] Thread started ({len(TRAINING_TICKERS)} tickers)")
                    daily_data = _fetch(TRAINING_TICKERS, "1day", 500, ttl=_TTL_1D)
                    _retrain_all(TRAINING_TICKERS, daily_data=daily_data, skip_deep=True)
                    logger.info("[ML-Retrain] Thread complete — in-memory models updated.")
                except Exception as _rt_e:
                    logger.warning(f"[ML-Retrain] Thread failed: {_rt_e}")

            import threading as _rt_threading
            _rt_threading.Thread(
                target=_scheduled_retrain, daemon=True, name="ml-retrain-scheduled"
            ).start()

        elif _scanner_training_enabled() and self._should_finetune_deep():
            # Hourly Deep BiLSTM fine-tune — uses cached 15-min data, no API calls.
            # Runs in a daemon thread so it doesn't block the next scan cycle.
            def _finetune():
                try:
                    from agent.deep_model import retrain_deep_all, is_training_active
                    if is_training_active():
                        return
                    logger.info("Hourly Deep BiLSTM fine-tune starting…")
                    hist_15m = fetch_batch_interval(NASDAQ_TICKERS, "15min", 5000, ttl=3600)
                    retrain_deep_all(hist_15m)
                    logger.info("Hourly Deep BiLSTM fine-tune complete.")
                except Exception as _ft_e:
                    logger.warning(f"Deep BiLSTM fine-tune failed: {_ft_e}")

            import threading as _ft_threading
            _ft_threading.Thread(target=_finetune, daemon=True, name="deep-finetune").start()
            self._last_deep_finetune = time.time()

        return results

    def _loop(self) -> None:
        self.is_running = True
        _first = True
        while self.is_running:
            try:
                # On cold start, wait up to 45 s for the WS streamer to build
                # up candles so fetch_batch_realtime can serve them from memory
                # instead of hitting the Schwab REST API for all 257 tickers.
                # This avoids the 429 cascade that makes the first scan take
                # 400+ seconds.  Skipped on warm restarts (cache already hot).
                if _first:
                    _first = False
                    try:
                        from agent.broker.schwab_streamer import (
                            get_streaming_bar_count, is_streamer_ready,
                        )
                        from config import SCHWAB_ENABLED
                        if SCHWAB_ENABLED and is_streamer_ready():
                            _deadline = time.time() + 45
                            while time.time() < _deadline:
                                # Wait until at least 10 tickers have ≥5 bars
                                ready = sum(
                                    1 for t in NASDAQ_TICKERS[:50]
                                    if get_streaming_bar_count(t) >= 5
                                )
                                if ready >= 10:
                                    logger.info(
                                        "[Scanner] WS streamer warmed (%d tickers "
                                        "with bars) — starting first scan.", ready
                                    )
                                    break
                                time.sleep(3)
                    except Exception:
                        pass  # non-Schwab path — no delay needed

                self.run_once()
            except Exception as e:
                logger.error(f"Scanner loop error: {e}", exc_info=True)
            time.sleep(_scan_interval())

    def _rt_monitor_loop(self) -> None:
        """
        Real-time price monitor — 5-second check cycle, zero API credits.
        Price source priority:
          1. Valkey hash (md:prices) — updated every ~300 ms by MD Poller
          2. Schwab WebSocket _live_quotes — fallback if Valkey unavailable
        Runs even when the Schwab streamer is disconnected, as long as Valkey has data.
        """
        try:
            from agent.valkey_client import get_price as _vk_get_price
            _valkey_available = True
        except ImportError:
            _valkey_available = False

        try:
            from agent.broker.schwab_streamer import is_streamer_ready, get_live_quote
            _streamer_available = True
        except ImportError:
            _streamer_available = False

        if not _valkey_available and not _streamer_available:
            logger.debug("[RT-Monitor] Neither Valkey nor streamer available — RT monitor inactive")
            return

        logger.info("[RT-Monitor] Real-time price monitor started (5s cycle, Valkey+streamer)")
        while self.is_running:
            try:
                from agent.paper_trading import get_open_trades as _get_open
                open_trades = _get_open()
                tickers = list({t["ticker"] for t in open_trades})
                if not tickers:
                    time.sleep(5)
                    continue

                _streamer_ready = _streamer_available and is_streamer_ready()

                for ticker in tickers:
                    last_price = 0.0

                    # 1. Try Valkey first — freshest price (~300 ms lag from MD Poller)
                    if _valkey_available:
                        try:
                            vk_quote = _vk_get_price(ticker)
                            if vk_quote:
                                last_price = float(
                                    vk_quote.get("last") or vk_quote.get("close") or 0
                                )
                        except Exception:
                            pass

                    # 2. Fallback to in-process WebSocket cache
                    if last_price <= 0 and _streamer_ready:
                        try:
                            ws_quote = get_live_quote(ticker)
                            if ws_quote:
                                last_price = float(
                                    ws_quote.get("last") or ws_quote.get("close") or 0
                                )
                        except Exception:
                            pass

                    if last_price <= 0:
                        continue

                    try:
                        resolved_bt = bt_rt_check(ticker, last_price)
                        if resolved_bt:
                            for r in resolved_bt:
                                logger.info(f"[RT-Monitor] BT {ticker}: {r}")
                    except Exception as _e:
                        logger.debug(f"[RT-Monitor] bt_rt_check {ticker}: {_e}")
                    try:
                        resolved_pt = pt_rt_check(ticker, last_price)
                        if resolved_pt:
                            for r in resolved_pt:
                                logger.info(f"[RT-Monitor] PT {ticker}: {r}")
                    except Exception as _e:
                        logger.debug(f"[RT-Monitor] pt_rt_check {ticker}: {_e}")
            except Exception as _loop_e:
                logger.debug(f"[RT-Monitor] loop error: {_loop_e}")
            time.sleep(5)

    def _bar_driven_loop(self) -> None:
        """
        Tier 3: Event-driven scan — triggered by CHART_EQUITY bar-close events
        instead of a 60-second polling clock.

        Architecture:
          1. Drain the bar-close queue in bursts (collect events for up to 2s
             so bars closing at the same minute boundary are batched together).
          2. Analyse only the tickers that received new bars — not all 175.
          3. Merge results into self.signals and notify callbacks.
          4. A 5-min watchdog re-scans ALL tickers in a background thread so
             the event loop is never blocked (reconnect gaps, missed events, etc.).

        Falls back to _loop() polling if the queue stays empty for 3 min
        (streamer disconnected or not yet authenticated).
        """
        from agent.broker.schwab_streamer import get_bar_close_queue
        bar_q = get_bar_close_queue()

        logger.info("[Scanner] Event-driven mode active — waiting for CHART_EQUITY bars…")

        _dirty:       set[str]        = set()
        _burst_start: float           = time.time()
        _last_full:   float           = 0.0
        _watchdog_running             = threading.Event()   # prevents overlapping watchdog scans
        _BURST_WINDOW  = 2.0          # collect events for 2s before analysing
        _WATCHDOG      = 300.0        # background full scan every 5 min
        _QUEUE_TIMEOUT = 180.0        # fall back to polling if no events for 3 min

        _last_event_ts = time.time()

        while self.is_running:
            # ── Drain bar-close events ────────────────────────────────────────
            try:
                ticker, _candle = bar_q.get(timeout=1.0)
                _dirty.add(ticker)
                _last_event_ts = time.time()

                # Collect more events in the burst window
                burst_deadline = time.time() + _BURST_WINDOW
                while time.time() < burst_deadline:
                    try:
                        t2, _ = bar_q.get_nowait()
                        _dirty.add(t2)
                    except Exception:
                        break

            except Exception:
                pass  # queue.Empty — no new bars, continue

            now = time.time()

            # ── Fallback: if no events for 3 min, switch to polling ───────────
            if now - _last_event_ts > _QUEUE_TIMEOUT:
                logger.info("[Scanner] No bar events for 3 min — falling back to polling loop")
                self._loop()
                return

            # ── Watchdog: full scan in background thread every 5 min ──────────
            # Never blocks the event loop — uses a guard Event to prevent overlap.
            if now - _last_full >= _WATCHDOG and not _watchdog_running.is_set():
                _last_full = now
                _watchdog_running.set()
                def _do_watchdog(guard=_watchdog_running):
                    logger.info("[Scanner] Watchdog: running background full scan…")
                    try:
                        self.run_once()
                        self._scan_count += 1
                        logger.info("[Scanner] Watchdog: full scan complete.")
                    except Exception as e:
                        logger.warning(f"[Scanner] Watchdog scan error: {e}")
                    finally:
                        guard.clear()
                threading.Thread(
                    target=_do_watchdog, daemon=True, name="scan-watchdog"
                ).start()

            # ── Burst scan: analyse only dirty tickers ────────────────────────
            if _dirty and (now - _burst_start) >= _BURST_WINDOW:
                dirty_list = list(_dirty)
                _dirty.clear()
                _burst_start = now
                try:
                    self._scan_subset(dirty_list)
                except Exception as e:
                    logger.debug(f"[Scanner] Burst scan error: {e}")

    def _scan_subset(self, tickers: list[str]) -> None:
        """
        Analyse a subset of tickers using fresh streaming data.
        Merges results into self.signals without displacing uncovered tickers.
        """
        if not tickers:
            return

        _sess = get_session_info()
        _is_extended = _sess.get("session", "") in ("AFTER_HOURS", "PRE_MARKET", "CLOSED")

        # 1min comes from streaming (instant); HTF comes from cache (instant)
        batch_1m = fetch_batch_realtime(tickers, extended_hours=_is_extended)
        batch_5m = fetch_batch_interval(tickers, "5min", 500, ttl=CACHE_TTL_5M)
        batch_1h = fetch_batch_interval(tickers, "1h",   500, ttl=CACHE_TTL_1H)
        batch_1d = fetch_batch_interval(tickers, "1day", 500, ttl=CACHE_TTL_1D)

        from agent.pipeline import get_pipeline
        n_total = len(tickers)
        new_results = get_pipeline(n_workers=min(_pipeline_workers(), n_total)).scan(
            tickers, batch_1m, batch_5m, batch_1h, batch_1d,
            on_ticker_done=lambda sig, n, t: self._notify_ticker(sig, n, n_total),
        )

        # Merge: replace existing signals for these tickers, keep others
        existing = {s.ticker: s for s in self.signals if s.ticker not in set(tickers)}
        for sig in new_results:
            existing[sig.ticker] = sig
        self.signals   = list(existing.values())
        self.last_scan = datetime.now(timezone.utc).isoformat()
        self._notify(self.signals)
        logger.debug(f"[Scanner] Burst: {len(new_results)}/{len(tickers)} tickers updated")

    def start_background(self) -> None:
        init_db()
        pt_init_db()
        bt_init_db()
        ah_init_db()

        if _scanner_training_enabled():
            ml_thread = threading.Thread(target=self._train_ml_background, daemon=True)
            ml_thread.start()
        else:
            logger.info("Scanner-side ML training disabled; learner services own retraining.")

        # Thread 2: Scan loop — polling with streaming-first data (Tier 1+2)
        # fetch_batch_realtime() already uses streaming candles when the
        # CHART_EQUITY buffer is warm (Tier A), and async REST when not (Tier B).
        # Event-driven scheduling (Tier 3) only helps in steady state after
        # the buffer is warm (~20 min); at startup it creates silent gaps.
        scan_thread = threading.Thread(target=self._loop, daemon=True, name="scan-loop")
        scan_thread.start()

        # Thread 3: Real-time position monitor
        rt_thread = threading.Thread(target=self._rt_monitor_loop, daemon=True, name="rt-monitor")
        rt_thread.start()

        logger.info(f"Scanner started. {len(NASDAQ_TICKERS)} tickers · {_pipeline_workers()} workers · streaming+polling modes.")

    def stop(self) -> None:
        self.is_running = False


scanner = Scanner()
