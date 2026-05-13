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
from datetime import datetime
from typing import Optional, Callable

import numpy as np
import pandas as pd

from config import (
    NASDAQ_TICKERS,
    SCAN_INTERVAL_SECONDS,
    ML_RETRAIN_INTERVAL,
    CACHE_TTL_5M,
    CACHE_TTL_1H,
    CACHE_TTL_1D,
    REGIME_TICKERS,
    SECTOR_ETF_TICKERS,
    get_active_tickers,
)
from agent.data_fetcher import (
    fetch_batch_realtime,
    fetch_batch_interval,
    fetch_ticker_info,
)
from agent.technical import compute_indicators, score_technical
from agent.volume import score_volume, relative_volume, detect_unusual_volume
from agent.ml_model import predict, predict_daily, predict_reversal, retrain_all
from agent.sentiment import score_sentiment
from agent.prediction import generate_prediction
from agent.mtf_analysis import multi_timeframe_analysis
from agent.market_hours import get_session_info, confidence_multiplier
from agent.market_regime import update_regime, get_regime, apply_regime
from agent.earnings import earnings_blackout
from agent.gap_analysis import analyse_gap
from agent.relative_strength import compute_relative_strength
from agent.trade_management import build_trade_plan
from agent.signal_tracker import init_db, record_signal, resolve_pending
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
    should_suppress, get_confidence_boost, increment_suppressed,
)
from agent.after_hours_monitor import (
    init_db as ah_init_db,
    record_snapshot as ah_record,
    get_opening_bias,
    get_ah_context_key,
)
from agent.trading_hours import get_trading_tier, is_signal_recommended

logger = logging.getLogger(__name__)


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
    scanned_at: str  = field(default_factory=lambda: datetime.utcnow().isoformat())

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
        ml_scalp        = predict(ticker, df_ind)
        ml_daily_p      = predict_daily(ticker, df_1d) if not df_1d.empty else 0.5
        ml_reversal_p   = predict_reversal(ticker, df_ind)
        # Blend: 40% daily (swing context) + 60% intraday (scalp timing)
        ml_combined     = round(0.4 * ml_daily_p + 0.6 * ml_scalp, 4)
        sent, headlines = score_sentiment(ticker)
        rvol            = relative_volume(df_ind)
        uvol            = detect_unusual_volume(df_ind)

        # Multi-timeframe analysis (6 TFs: 5M, 15M, 30M, 1H, 4H, 1D)
        mtf = multi_timeframe_analysis(df_1m, df_5m, df_1h, df_1d)

        # Market session
        sess_info = get_session_info()
        sess_mult = confidence_multiplier()

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
        )

        # Apply regime multiplier to score
        raw_score = round(float(np.clip(pred["composite_score"], -1, 1)), 4)
        score     = round(float(np.clip(apply_regime(raw_score, regime), -1, 1)), 4)

        # Override direction to NEUTRAL if earnings or macro blackout
        if eb["blocked"] or macro_ev["blocked"]:
            pred["direction"] = "NEUTRAL"

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

        # Apply after-hours opening bias to confidence
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
                    f"AH {_ah_dir} {_ah_change:+.1f}% ({_ah_mag}) {_label} signal"
                )

        # Check if ticker already has an open paper trade
        _has_open_position = False
        try:
            from agent.paper_trading import get_open_trades as _get_open
            _open_tickers = {t["ticker"] for t in _get_open()}
            _has_open_position = ticker in _open_tickers
        except Exception:
            pass

        # Adaptive filter — suppress signals matching learned losing patterns
        _is_suppressed   = False
        _suppress_reason = ""
        if pred["direction"] in ("BUY", "SELL"):
            suppress, suppress_reason = should_suppress(
                vwap_event   = vwap_sig["event"],
                session      = sess_info.get("session", ""),
                regime       = regime.regime,
                rsi_zone     = pred.get("rsi_zone", ""),
                entry_type   = pred.get("entry_type", ""),
                direction    = pred["direction"],
                sector_trend = sector_ctx.sector_trend,
                confidence   = pred["confidence"],
            )
            if suppress:
                pred["direction"]  = "NEUTRAL"
                pred["reasons"]    = [f"⚡ {suppress_reason}"] + pred.get("reasons", [])
                _is_suppressed     = True
                _suppress_reason   = suppress_reason
                increment_suppressed()

        # Record signal in tracker + open paper trade (BUY/SELL only)
        if pred["direction"] in ("BUY", "SELL") and not eb["blocked"] and not macro_ev["blocked"]:
            resolve_pending(ticker, price)
            record_signal(
                ticker=ticker, direction=pred["direction"], entry=price,
                target=pred["target_price"], stop=pred["stop_loss"],
                confidence=pred["confidence"],
                session=sess_info.get("session", ""),
                regime=regime.regime,
            )
            bt_record(
                ticker       = ticker,
                direction    = pred["direction"],
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
            maybe_open_trade(
                ticker       = ticker,
                direction    = pred["direction"],
                price        = price,
                target       = pred["target_price"],
                stop         = pred["stop_loss"],
                confidence   = pred["confidence"],
                rr_qualifies = bool(pred.get("rr_qualifies", False)),
                rr_ratio     = float(pred.get("rr_ratio", 0.0)),
                session      = sess_info.get("session", ""),
                regime       = regime.regime,
                vwap_event   = vwap_sig["event"],
                rsi_zone     = pred.get("rsi_zone", ""),
                entry_type   = pred.get("entry_type", "IMMEDIATE"),
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
            is_suppressed       = _is_suppressed,
            suppress_reason     = _suppress_reason,
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
        )
    except Exception as e:
        logger.warning(f"[{ticker}] analysis error: {e}", exc_info=True)
        return None


# ── Scanner ───────────────────────────────────────────────────────────────────

class Scanner:
    def __init__(self):
        self.signals:       list[StockSignal] = []
        self.last_scan:     Optional[str]     = None
        self.is_running:    bool              = False
        self._callbacks:    list[Callable]    = []
        self._last_retrain: float             = 0.0

    def register_callback(self, fn: Callable) -> None:
        self._callbacks.append(fn)

    def _notify(self, signals: list[StockSignal]) -> None:
        for fn in self._callbacks:
            try:
                fn(signals)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def _should_retrain(self) -> bool:
        if self._last_retrain == 0.0:
            return False  # startup training running in background
        return (time.time() - self._last_retrain) > ML_RETRAIN_INTERVAL

    # ── ML training ───────────────────────────────────────────────────────────

    def _train_ml_background(self) -> None:
        """
        Train both intraday (5M) and daily ML models at startup.
        Daily data served from fetch_batch_interval cache (fetched during first scan).
        We wait briefly to allow the first scan's data fetches to warm the cache.
        """
        logger.info("ML training: waiting for first scan to warm data cache…")
        time.sleep(30)   # let the first scan + cache-fill complete first

        logger.info("ML training starting (intraday + daily models)…")
        daily_data = fetch_batch_interval(NASDAQ_TICKERS, "1day", 500, ttl=CACHE_TTL_1D)
        retrain_all(NASDAQ_TICKERS, daily_data=daily_data)
        self._last_retrain = time.time()
        logger.info("ML training complete.")

    # ── Scan loop ─────────────────────────────────────────────────────────────

    def run_once(self) -> list[StockSignal]:
        t0 = time.time()
        active_tickers = get_active_tickers()
        logger.info(f"Scan starting — {len(active_tickers)} tickers…")

        # Detect session once so the data fetch uses the right mode
        _sess = get_session_info()
        _is_extended = _sess.get("session", "") in ("AFTER_HOURS", "PRE_MARKET")

        # Always-fresh 1M data — include extended-hours bars during AH/PM so
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
        results = []
        for ticker in active_tickers:
            sig = analyse_ticker(
                ticker,
                df_1m = batch_1m.get(ticker),
                df_5m = batch_5m.get(ticker, pd.DataFrame()),
                df_1h = batch_1h.get(ticker, pd.DataFrame()),
                df_1d = batch_1d.get(ticker, pd.DataFrame()),
            )
            if sig:
                results.append(sig)

        results.sort(key=lambda s: abs(s.score), reverse=True)
        self.signals   = results
        self.last_scan = datetime.utcnow().isoformat()
        self._notify(results)
        elapsed = round(time.time() - t0, 1)
        logger.info(f"Scan complete in {elapsed}s | {len(results)}/{len(NASDAQ_TICKERS)} tickers analysed")

        # ML feedback: retrain if enough new backtest outcomes have accumulated
        maybe_trigger_feedback_retrain(active_tickers)

        if self._should_retrain():
            logger.info("Scheduled ML retrain starting…")
            daily_data = fetch_batch_interval(NASDAQ_TICKERS, "1day", 500, ttl=CACHE_TTL_1D)
            retrain_all(NASDAQ_TICKERS, daily_data=daily_data)
            self._last_retrain = time.time()
            logger.info("Scheduled ML retrain complete.")

        return results

    def _loop(self) -> None:
        self.is_running = True
        while self.is_running:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Scanner loop error: {e}", exc_info=True)
            time.sleep(SCAN_INTERVAL_SECONDS)

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

        logger.info("Scanner started. Grow-377: 50 tickers, 1-min scan interval.")

    def stop(self) -> None:
        self.is_running = False


scanner = Scanner()
