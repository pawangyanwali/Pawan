"""
Main scanning engine — Schwab Market Data API (no per-symbol credit cost).

Universe: ~500 NASDAQ tickers tracked via Schwab /quotes bulk screening.
Active scan: Tier 1 (100 core) always + top 75 active from Tier 2/3 = ~175
per cycle.  Price history cached per ticker (TTL 300 s) → ~35 new API calls
per minute for history, well within the 90 req/min Schwab rate limit.
"""

import logging
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
from agent.market_hours import get_session_info, confidence_multiplier
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
from agent.paper_trading import init_db as pt_init_db, maybe_open_trade, update_open_trades, rt_check_positions as pt_rt_check, log_algo_signals
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
)
from agent.ensemble_model import get_meta_prediction
from agent.deep_model import predict_deep
from agent.risk_controls import check_circuit_breaker, check_sector_concentration, update_volatility_state
from agent.order_flow import compute_order_flow, get_signal_strength
from agent.after_hours_monitor import (
    init_db as ah_init_db,
    record_snapshot as ah_record,
    get_opening_bias,
    get_ah_context_key,
)
from agent.trading_hours import get_trading_tier, is_signal_recommended

logger = logging.getLogger(__name__)


def _scan_interval() -> int:
    """
    Adaptive scan cadence based on market session.
    Power hours get 30s scans — more signals when price action is richest.
    Off-peak gets the default 60s to conserve API credits.
      09:30–11:00 ET  (opening range + momentum)  → 30s
      14:30–16:00 ET  (closing power hour)         → 30s
      everything else                              → 60s
    """
    from datetime import datetime
    import zoneinfo
    now_et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    h, m = now_et.hour, now_et.minute
    minutes = h * 60 + m
    OPEN_RANGE_END  = 9 * 60 + 30 + 90   # 11:00 ET
    OPEN_RANGE_START = 9 * 60 + 30       # 09:30 ET
    CLOSE_START = 14 * 60 + 30           # 14:30 ET (2:30pm)
    CLOSE_END   = 16 * 60                # 16:00 ET
    if OPEN_RANGE_START <= minutes < OPEN_RANGE_END:
        return 30   # opening power hour
    if CLOSE_START <= minutes < CLOSE_END:
        return 30   # closing power hour
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
    macro_event:        str   = ""
    macro_description:  str   = ""

    # ── Trade management plan ─────────────────────────────────────────────────
    trade_plan: dict  = field(default_factory=dict)

    # ── Chart candles (last 80 × 1-min bars) ─────────────────────────────────
    candles:    list = field(default_factory=list)

    # ── News ──────────────────────────────────────────────────────────────────
    headlines:  list = field(default_factory=list)
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
        ml_scalp                   = predict(ticker, _df_ml)
        ml_daily_p                 = predict_daily(ticker, df_1d) if not df_1d.empty else 0.5
        ml_reversal_p              = predict_reversal(ticker, _df_ml)
        ml_ensemble_p, ml_agree    = predict_ensemble(ticker, _df_ml)

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

        ml_swing_p = predict_swing(ticker, _df_15m) if _has_15m else 0.5
        ml_deep_p  = predict_deep(ticker, _df_15m)  if _has_15m else 0.5

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
        sent, headlines = score_sentiment(ticker)
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

        # Earnings blackout
        eb = earnings_blackout(ticker)

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

        # Override direction to NEUTRAL if earnings or macro blackout
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
                _sec_blocked, _sec_msg = check_sector_concentration(
                    ticker, "BUY" if "BUY" in pred["direction"] else "SELL"
                )
                if _sec_blocked:
                    _trade_blocked_reason = _sec_msg

        # Adaptive filter = calibration only.
        # get_confidence_boost() already applied earlier (line ~595).
        # Signal direction is NEVER modified here — the models learn from
        # all trade outcomes and improve over time.
        _is_suppressed   = False
        _suppress_reason = ""

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
            # Paper trade execution — can_open_trade() inside applies all
            # session / risk / circuit-breaker rules at the execution layer.
            # Extended-hours size caps: AH HIGH=50%, AH MODERATE=30%, PM HIGH=40%, PM MODERATE=25%.
            maybe_open_trade(
                ticker            = ticker,
                direction         = _norm_direction,
                price             = price,
                target            = pred["target_price"],
                stop              = pred["stop_loss"],
                confidence        = pred["confidence"],
                rr_qualifies      = bool(pred.get("rr_qualifies", False)),
                rr_ratio          = float(pred.get("rr_ratio", 0.0)),
                session           = sess_info.get("session", ""),
                regime            = regime.regime,
                vwap_event        = vwap_sig["event"],
                rsi_zone          = pred.get("rsi_zone", ""),
                entry_type        = pred.get("entry_type", "IMMEDIATE"),
                order_flow_score  = _of_score,
                size_mult         = round(_sig_size_mult, 2),
                trading_tier      = _trading_tier,
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
            # Earnings
            earnings_blocked   = bool(eb["blocked"]),
            earnings_reason    = eb["reason"],
            earnings_date      = eb["next_date"],
            earnings_days_away = int(eb["days_away"]),
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
            macro_event       = macro_ev["event_name"],
            macro_description = macro_ev["description"],
            # Trade plan
            trade_plan          = tp.to_dict(),
            candles             = candles,
            headlines           = headlines[:5],
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
            _algo_trade_opened = False
            for _asig in _sig.algo_signals:
                _trade_id = maybe_open_trade(
                    ticker            = ticker,
                    direction         = _asig["direction"],
                    price             = float(_asig["entry"]),
                    target            = float(_asig["target"]),
                    stop              = float(_asig["stop"]),
                    confidence        = float(_asig["confidence"]),
                    rr_qualifies      = float(_asig.get("rr", 0)) >= 1.5,
                    rr_ratio          = float(_asig.get("rr", 0)),
                    session           = sess_info.get("session", ""),
                    regime            = regime.regime,
                    entry_type        = "ALGO",
                    order_flow_score  = _of_score,
                    size_mult         = 1.0,
                    trading_tier      = _trading_tier,
                    algo_name         = _asig["algo"],
                )
                if _trade_id:
                    _algo_trade_opened = True
            try:
                log_algo_signals(ticker, _sig.algo_signals, trade_opened=_algo_trade_opened)
            except Exception as _le:
                logger.debug("[%s] algo log error: %s", ticker, _le)

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
        self._scan_count:         int               = 0

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

    def _should_retrain(self) -> bool:
        if self._last_retrain == 0.0:
            return False  # startup training running in background
        return (time.time() - self._last_retrain) > ML_RETRAIN_INTERVAL

    def _should_finetune_deep(self) -> bool:
        """True when 1 hour has passed since last Deep BiLSTM fine-tune."""
        if self._last_deep_finetune == 0.0:
            return False  # avoid racing with startup full train
        return (time.time() - self._last_deep_finetune) > DEEP_FINETUNE_INTERVAL

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
        logger.info("ML training: waiting for first two scan cycles to warm data cache…")
        # Wait for the first scan event (set at the end of run_once)
        self._first_scan_done.wait(timeout=600)
        # Extra buffer: let the second cycle finish so 5min cache is also warm
        time.sleep(90)

        logger.info(f"ML training starting ({len(TRAINING_TICKERS)} Tier-1 tickers)…")
        daily_data = fetch_batch_interval(TRAINING_TICKERS, "1day", 500, ttl=CACHE_TTL_1D)
        retrain_all(TRAINING_TICKERS, daily_data=daily_data)
        self._last_retrain       = time.time()
        self._last_deep_finetune = time.time()   # startup full train counts as fine-tune
        logger.info("ML training complete.")

    # ── Scan loop ─────────────────────────────────────────────────────────────

    def run_once(self) -> list[StockSignal]:
        t0 = time.time()
        active_tickers = get_active_tickers()

        # ── EOD Hard Close (PRD 6.4): close all positions at 3:45 PM ET ──────
        from agent.market_hours import is_hard_close_window, is_closing_caution, no_new_entries
        if is_hard_close_window():
            try:
                from agent.paper_trading import close_all_positions_eod
                closed_n = close_all_positions_eod()
                if closed_n:
                    logger.info(f"[Scanner] EOD hard close triggered — {closed_n} positions closed")
            except Exception as _eod_e:
                logger.warning(f"[Scanner] EOD hard close failed: {_eod_e}")

        # ── After-hours stale-trade sweep — catches anything missed at 3:45 ──
        elif no_new_entries():
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
        results = get_pipeline(n_workers=PIPELINE_WORKERS).scan(
            active_tickers, batch_1m, batch_5m, batch_1h, batch_1d,
            on_ticker_done=lambda sig, n, t: self._notify_ticker(sig, n, _n_total),
        )

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

        self.signals   = results
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

        self._notify(results)
        elapsed = round(time.time() - t0, 1)
        self._scan_count += 1
        if self._scan_count == 1:
            self._first_scan_done.set()   # unblock _train_ml_background
        logger.info(f"Scan complete in {elapsed}s | {len(results)}/{len(active_tickers)} active (universe: {len(NASDAQ_TICKERS)})")

        # ML feedback: retrain if enough new backtest outcomes have accumulated
        maybe_trigger_feedback_retrain(active_tickers)

        if self._should_retrain():
            logger.info(f"Scheduled ML retrain launching ({len(TRAINING_TICKERS)} Tier-1 tickers)…")
            self._last_retrain       = time.time()
            self._last_deep_finetune = time.time()   # full retrain counts as fine-tune too
            def _bg_retrain():
                try:
                    daily_data = fetch_batch_interval(TRAINING_TICKERS, "1day", 500, ttl=CACHE_TTL_1D)
                    retrain_all(TRAINING_TICKERS, daily_data=daily_data)
                    logger.info("Scheduled ML retrain complete.")
                except Exception as _re:
                    logger.warning(f"Background ML retrain failed: {_re}")
            threading.Thread(target=_bg_retrain, daemon=True, name="ml-retrain").start()

        elif self._should_finetune_deep():
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
        new_results = get_pipeline(n_workers=min(PIPELINE_WORKERS, n_total)).scan(
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

        # Thread 1: ML training (waits for first scan to warm cache)
        ml_thread = threading.Thread(target=self._train_ml_background, daemon=True)
        ml_thread.start()

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

        logger.info(f"Scanner started. {len(NASDAQ_TICKERS)} tickers · {PIPELINE_WORKERS} workers · streaming+polling modes.")

    def stop(self) -> None:
        self.is_running = False


scanner = Scanner()
