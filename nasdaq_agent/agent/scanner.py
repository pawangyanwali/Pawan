"""
Main scanning engine — optimised for Twelve Data Grow-377 plan.

3 calls per scan (50 tickers / 20 per batch = 3 batches × 0.20s ≈ 0.60s).
5M and 1H data served from in-process cache (TTL 5 min / 1 h respectively),
so additional timeframe data costs zero API calls on most scan cycles.
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
from agent.volume import score_volume, relative_volume, detect_unusual_volume
from agent.ml_model import predict, predict_daily, predict_reversal, predict_ensemble, predict_swing, get_or_create_swing, retrain_all
from agent.sentiment import score_sentiment
from agent.prediction import generate_prediction
from agent.mtf_analysis import multi_timeframe_analysis
from agent.market_hours import get_session_info, confidence_multiplier
from agent.market_regime import update_regime, get_regime, apply_regime
from agent.earnings import earnings_blackout
from agent.gap_analysis import analyse_gap
from agent.relative_strength import compute_relative_strength
from agent.trade_management import build_trade_plan
from agent.signal_tracker import init_db, record_signal, resolve_pending, record_signals_batch, resolve_short_term, get_ticker_learning_scores
from agent.vwap import compute_vwap_signal
from agent.sector_etf import get_sector_context, update_etf_cache
from agent.exit_signals import analyse_exits
from agent.paper_trading import init_db as pt_init_db, maybe_open_trade, update_open_trades
from agent.macro_calendar import check_macro_event
from agent.live_backtest import (
    init_db as bt_init_db,
    record_signal as bt_record,
    update_tracking as bt_update,
)
from agent.backtest_reporter import maybe_trigger_feedback_retrain, adjust_confidence
from agent.adaptive_filter import (
    get_confidence_boost,
)
from agent.ensemble_model import get_meta_prediction
from agent.deep_model import predict_deep
from agent.risk_controls import check_circuit_breaker, check_sector_concentration
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
    from datetime import datetime, timezone, timedelta
    now_et = datetime.now(timezone.utc) - timedelta(hours=4)  # UTC-4 for EDT
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

    # ── Multi-timeframe analysis ───────────────────────────────────────────────
    mtf_score:      float = 0.0
    mtf_alignment:  str   = "MIXED"
    mtf_bull_count: int   = 0
    mtf_bear_count: int   = 0
    mtf_timeframes: dict  = field(default_factory=dict)

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

    # ── PDH / PDL / ORB levels ────────────────────────────────────────────────
    prev_day_high:  float = 0.0   # Previous day high
    prev_day_low:   float = 0.0   # Previous day low
    prev_day_close: float = 0.0   # Previous day close
    orb_high:       float = 0.0   # Opening range breakout high (first 30-min)
    orb_low:        float = 0.0   # Opening range breakout low (first 30-min)
    orb_breakout:   str   = ""    # "BULL" | "BEAR" | "" — if price broke ORB

    # ── Relative strength vs SPY ──────────────────────────────────────────────
    rs_ratio:   float = 1.0
    rs_score:   float = 0.0
    rs_label:   str   = "IN_LINE"

    # ── VWAP signal ───────────────────────────────────────────────────────────
    vwap_event:       str   = "FLAT"
    vwap_score:       float = 0.0
    vwap_price:       float = 0.0
    vwap_deviation:   float = 0.0
    vwap_description: str   = ""

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
    """Compute PDH, PDL, PDC, and ORB (first 30-min high/low) from market data."""
    import pytz
    result = {"prev_day_high": 0.0, "prev_day_low": 0.0, "prev_day_close": 0.0,
              "orb_high": 0.0, "orb_low": 0.0, "orb_breakout": ""}
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

        price = float(last["Close"])
        # Change % vs previous daily close (works for both regular session and AH).
        # During AH, df_1d.iloc[-1] is today's close → gives true AH move.
        # During regular session, Twelve Data daily bars lag until EOD so
        # df_1d.iloc[-1] is yesterday's close → gives correct intraday change.
        _prev_close = float(df_1d.iloc[-1]["Close"]) if not df_1d.empty else 0.0
        if _prev_close <= 0:
            _prev_close = float(df_ind.iloc[0]["Open"]) or price
        change_pct = round((price - _prev_close) / _prev_close * 100, 3) if _prev_close else 0.0

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
        rvol            = relative_volume(df_ind)
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

        # PDH / PDL / ORB levels
        levels = _compute_levels(df_1m, df_1d)

        # Relative strength vs SPY (regime spy data available via get_regime())
        regime = get_regime()
        rs = compute_relative_strength(df_1m, _spy_df_cache.get("SPY"))

        # VWAP signal
        vwap_sig = compute_vwap_signal(df_ind)

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
        )

        # Apply regime multiplier to score
        raw_score = round(float(np.clip(pred["composite_score"], -1, 1)), 4)
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
            # Market fully closed (8pm–4am) — no new signals for anyone
            if pred["direction"] in ("BUY", "SELL"):
                pred["direction"] = "NEUTRAL"
                pred["reasons"]   = ["⛔ Market closed (8pm–4am) — monitoring only"] + pred.get("reasons", [])

        elif _session_now == "AFTER_HOURS":
            if _static_tier == "HIGH":
                # Mega-cap (AAPL/TSLA/NVDA/MSFT/META/AMZN/GOOGL/NFLX/AMD/AVGO) —
                # meaningful AH liquidity, allow trading at 50% size
                if pred["direction"] in ("BUY", "SELL"):
                    pred["reasons"] = [
                        f"🌙 AH trade — {ticker} HIGH-tier (50% size, extended liquidity)"
                    ] + pred.get("reasons", [])
            else:
                # MODERATE / REGULAR — thin AH spreads, no edge outside regular hours
                if pred["direction"] in ("BUY", "SELL"):
                    pred["direction"] = "NEUTRAL"
                    pred["reasons"]   = [
                        f"⛔ AH: {ticker} {_static_tier}-tier — thin liquidity, no new trades"
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

        # Apply backtest-derived confidence calibration
        pred["confidence"] = adjust_confidence(
            confidence  = pred["confidence"],
            vwap_event  = vwap_sig["event"],
            rsi_zone    = pred.get("rsi_zone", ""),
            session     = sess_info.get("session", ""),
            regime      = regime.regime,
            entry_type  = pred.get("entry_type", ""),
            direction   = pred["direction"],
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
        # HIGH-tier mega-caps (AAPL/TSLA/NVDA etc.) are allowed to trade
        # during AFTER_HOURS (4–8 PM ET) at 50% size — do NOT flag as blocked.
        _is_ah_high_tier = (_session_now == "AFTER_HOURS" and _trading_tier == "HIGH")
        if _is_ah_high_tier:
            pass  # allowed — scanner AH gate above kept BUY/SELL direction
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
            # During AFTER_HOURS, cap HIGH-tier trades to 50% position size
            _ah_size_cap = 0.5 if _session_now == "AFTER_HOURS" else 1.0
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
                size_mult         = round(_sig_size_mult * _ah_size_cap, 2),
                trading_tier      = _trading_tier,
            )

        # Update open paper trades + live backtest tracking
        update_open_trades(ticker, df_ind, price)
        bt_update(ticker, price, vwap_sig["vwap"])

        candles = _build_candles(df_1m)
        info    = _get_info(ticker)

        return StockSignal(
            ticker            = ticker,
            name              = info.get("name", ticker),
            price             = round(price, 4),
            change_pct        = change_pct,
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
            pivots            = pred["pivots"],
            poc               = pred["poc"],
            mtf_score         = float(mtf["mtf_score"]),
            mtf_alignment     = mtf["alignment"],
            mtf_bull_count    = int(mtf["bull_count"]),
            mtf_bear_count    = int(mtf["bear_count"]),
            mtf_timeframes    = mtf["timeframes"],
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
            gap_filled        = bool(gap["gap_filled"]),
            gap_fill_prob     = float(gap["fill_probability"]),
            premarket_high    = float(gap["premarket_high"]),
            premarket_low     = float(gap["premarket_low"]),
            # PDH / PDL / ORB levels
            prev_day_high     = levels["prev_day_high"],
            prev_day_low      = levels["prev_day_low"],
            prev_day_close    = levels["prev_day_close"],
            orb_high          = levels["orb_high"],
            orb_low           = levels["orb_low"],
            orb_breakout      = levels["orb_breakout"],
            # Relative strength
            rs_ratio          = float(rs["rs_ratio"]),
            rs_score          = float(rs["rs_score"]),
            rs_label          = rs["rs_label"],
            # VWAP signal
            vwap_event        = vwap_sig["event"],
            vwap_score        = float(vwap_sig["score"]),
            vwap_price        = float(vwap_sig["vwap"]),
            vwap_deviation    = float(vwap_sig["deviation"]),
            vwap_description  = vwap_sig["description"],
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

        Wait for the first scan AND its cache-fill to fully complete before
        training touches the API.  First scan populates 1min, 5min, 1h, 1day,
        and sector ETF caches — that costs ~600-700 credits and takes 2-3 min
        under the CREDIT_LIMIT=340 gate.  Starting training during that window
        used to spike to 541 credits/minute and trigger 429s.
        """
        logger.info("ML training: waiting for first two scan cycles to warm data cache…")
        # Wait for the first scan event (set at the end of run_once)
        self._first_scan_done.wait(timeout=600)
        # Extra buffer: let the second cycle finish so 5min cache is also warm
        time.sleep(90)

        logger.info("ML training starting (intraday + daily models)…")
        daily_data = fetch_batch_interval(NASDAQ_TICKERS, "1day", 500, ttl=CACHE_TTL_1D)
        retrain_all(NASDAQ_TICKERS, daily_data=daily_data)
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

        # Always-fresh 1M data — include extended-hours bars during AH/PM/CLOSED so
        # the chart reflects actual after-market price action, not stale closes.
        batch_1m = fetch_batch_realtime(active_tickers, extended_hours=_is_extended)

        # Cached higher-TF data (only refetched when TTL expires)
        batch_5m = fetch_batch_interval(active_tickers, "5min", 500,  ttl=CACHE_TTL_5M)
        batch_1h = fetch_batch_interval(active_tickers, "1h",   500,  ttl=CACHE_TTL_1H)
        batch_1d = fetch_batch_interval(active_tickers, "1day", 500,  ttl=CACHE_TTL_1D)

        # Fetch SPY/QQQ + sector ETFs for regime, RS and sector context
        etf_1m = fetch_batch_realtime(SECTOR_ETF_TICKERS, extended_hours=_is_extended)
        _spy_df_cache.update(etf_1m)
        update_etf_cache(etf_1m)
        update_regime(
            df_spy=etf_1m.get("SPY"),
            df_qqq=etf_1m.get("QQQ"),
        )

        active_tickers = get_active_tickers()

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
                _af_live(_live_stats)
        except Exception as _af_live_err:
            logger.debug(f"live adaptive filter update error: {_af_live_err}")

        self._notify(results)
        elapsed = round(time.time() - t0, 1)
        self._scan_count += 1
        if self._scan_count == 1:
            self._first_scan_done.set()   # unblock _train_ml_background
        logger.info(f"Scan complete in {elapsed}s | {len(results)}/{len(NASDAQ_TICKERS)} tickers analysed")

        # ML feedback: retrain if enough new backtest outcomes have accumulated
        maybe_trigger_feedback_retrain(active_tickers)

        if self._should_retrain():
            logger.info("Scheduled ML retrain starting…")
            daily_data = fetch_batch_interval(NASDAQ_TICKERS, "1day", 500, ttl=CACHE_TTL_1D)
            retrain_all(NASDAQ_TICKERS, daily_data=daily_data)
            self._last_retrain       = time.time()
            self._last_deep_finetune = time.time()   # full retrain counts as fine-tune too
            logger.info("Scheduled ML retrain complete.")

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
        while self.is_running:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Scanner loop error: {e}", exc_info=True)
            time.sleep(_scan_interval())

    def start_background(self) -> None:
        init_db()      # signal_history.db
        pt_init_db()   # paper_trades.db
        bt_init_db()   # live_backtest.db
        ah_init_db()   # ah_snapshots.db

        # Thread 1: ML training (waits 30s for first scan to warm cache)
        ml_thread = threading.Thread(target=self._train_ml_background, daemon=True)
        ml_thread.start()

        # Thread 2: Main scan loop (starts immediately)
        scan_thread = threading.Thread(target=self._loop, daemon=True)
        scan_thread.start()

        logger.info(f"Scanner started. {len(NASDAQ_TICKERS)} tickers · {PIPELINE_WORKERS} parallel workers · 1-min scan interval.")

    def stop(self) -> None:
        self.is_running = False


scanner = Scanner()
