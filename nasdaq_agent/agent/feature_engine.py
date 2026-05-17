"""
feature_engine.py — Leakage-free feature engineering for the NASDAQ scalping agent.

Fixes two bugs present in the original ml_model.py pipeline:

1. DATA LEAKAGE: indicators previously computed on the full dataset before
   train/test split. Rolling-window calculations at the boundary incorporated
   future (test-period) data into training features. This module always splits
   raw OHLCV first, then computes features on each partition independently.

2. POOR LABEL QUALITY: the original binary label ``Close[t+N] > Close[t]``
   treats a 0.01% move identically to a 2% move. After spread and commission
   tiny moves are net losses. Labels here require a minimum move proportional
   to ATR, and rows that fall in the "noise band" are excluded entirely from
   training (not relabelled).

Public API
----------
FEATURE_COLS_V2 : list[str]
    The 32 feature names expected by every XGBoost model variant.

FEATURE_COLS_V3 : list[str]
    40 features = V2 + 8 new professional indicators (kc_squeeze, cvd_5,
    ema_200_dev, vwap_z, hidden_div_bull, hidden_div_bear, nr7, adx_regime).

compute_features(df, ticker) -> pd.DataFrame
    Add all 32 features to a raw OHLCV DataFrame.  Safe for inference — every
    indicator is causal (uses only past data at each row).

prepare_training_data(df_raw, ticker, lookahead_bars, atr_multiplier, test_size)
    -> tuple[X_train, y_train, X_test, y_test] | None
    Leakage-free split + feature computation + ATR-adaptive labelling.

compute_live_row(df_raw, ticker) -> np.ndarray | None
    Convenience wrapper: compute features and return only the last row as a
    (1, 32) float32 array ready for model.predict_proba().
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature column registry
# ---------------------------------------------------------------------------

# Original 23 features — names kept identical for backward compatibility with
# any saved XGBoost models trained against the old FEATURE_COLS list.
_FEATURE_COLS_V1: list[str] = [
    "rsi_14", "rsi_7",
    "macd", "macd_signal", "macd_hist",
    "bb_pct", "bb_width",
    "stoch_k", "stoch_d",
    "cci_20", "mfi_14",
    "ema_cross",
    "vol_ratio",
    "atr_14",
    "obv",
    "ret_1", "ret_3", "ret_5", "ret_10",
    "time_sin", "time_cos",
    "price_range_pos",
    "vol_trend",
]

# 9 new features appended at the end.
_FEATURE_COLS_NEW: list[str] = [
    "vwap_dev",      # (Close - VWAP) / VWAP, position relative to daily average
    "adx_14",        # Average Directional Index normalised to 0-1
    "cmf_20",        # Chaikin Money Flow -1 to +1
    "roc_5",         # 5-bar Rate of Change %
    "williams_r",    # Williams %R normalised to 0-1
    "spread_pct",    # (High - Low) / Close * 100, intrabar volatility
    "obv_slope",     # OBV z-score vs 10-bar rolling mean/std
    "ema_ribbon",    # (ema_9 - ema_50) / Close * 100, trend context
    "gap_open",      # (Open - prev_Close) / prev_Close * 100, overnight gap
]

FEATURE_COLS_V2: list[str] = _FEATURE_COLS_V1 + _FEATURE_COLS_NEW

# 8 additional features for V3 — leverage the new professional indicators
# from technical.py (KC squeeze, CVD, Chandelier, hidden divergence, NR7).
_FEATURE_COLS_V3_NEW: list[str] = [
    "kc_squeeze",      # 1 when BB inside Keltner Channel (volatility compression)
    "cvd_5",           # 5-bar Cumulative Volume Delta (buy vs sell pressure)
    "ema_200_dev",     # (Close - EMA200) / EMA200 × 100, institutional context
    "vwap_z",          # VWAP z-score: (price - VWAP) / VWAP_std, dynamic σ-bands
    "hidden_div_bull", # 1 = hidden bullish RSI divergence (uptrend continuation)
    "hidden_div_bear", # 1 = hidden bearish RSI divergence (downtrend continuation)
    "nr7",             # 1 = current bar narrowest range of last 7 (compression)
    "adx_regime",      # 1 = trend mode (ADX>0.25), 0 = mean-reversion (ADX<0.20)
]

FEATURE_COLS_V3: list[str] = FEATURE_COLS_V2 + _FEATURE_COLS_V3_NEW


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    fill: float = 0.0,
) -> pd.Series:
    """Element-wise division that replaces zero-denominator results with *fill*."""
    denom = denominator.replace(0, np.nan)
    return (numerator / denom).fillna(fill)


def _compute_session_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute session VWAP that resets at the start of each calendar day.

    For each bar the VWAP is the cumulative (typical_price × volume) divided
    by the cumulative volume since the open of that trading session.  Using a
    DatetimeIndex date as the group key makes this work correctly even when
    multiple sessions are concatenated in one DataFrame.

    Falls back to a simple cumulative VWAP when the index is not a
    DatetimeIndex (e.g. integer index in unit tests).
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    tpv = tp * df["Volume"]

    if not isinstance(df.index, pd.DatetimeIndex):
        cum_vol = df["Volume"].cumsum()
        cum_tpv = tpv.cumsum()
        return _safe_divide(cum_tpv, cum_vol, fill=df["Close"].iloc[0] if len(df) else 0.0)

    date_key = df.index.date
    vwap = pd.Series(np.nan, index=df.index, dtype=float)
    for date, grp_idx in pd.Series(date_key, index=df.index).groupby(date_key).groups.items():
        cum_vol = df.loc[grp_idx, "Volume"].cumsum()
        cum_tpv = tpv.loc[grp_idx].cumsum()
        vwap.loc[grp_idx] = (cum_tpv / cum_vol.replace(0, np.nan)).values
    return vwap


def _encode_time_of_day(index: pd.Index) -> tuple[pd.Series, pd.Series]:
    """
    Cyclical encoding of time-of-day within the 9:30–16:00 trading session.

    Returns (time_sin, time_cos) with values in [-1, 1].  Bars outside regular
    hours are clipped to the nearest boundary so they never produce NaN.
    """
    if not isinstance(index, pd.DatetimeIndex):
        zeros = pd.Series(0.0, index=index)
        return zeros, zeros

    minutes = pd.Series(index.hour * 60 + index.minute, index=index, dtype=float)
    session_start = 9 * 60 + 30   # 570
    session_len   = 390            # 9:30 → 16:00 = 390 min
    norm = ((minutes - session_start) / session_len).clip(0.0, 1.0)
    angle = 2.0 * np.pi * norm
    return np.sin(angle), np.cos(angle)


# ---------------------------------------------------------------------------
# Core feature computation
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    Add all 32 features to a raw OHLCV DataFrame and return the enriched copy.

    This function is fully **causal** — every indicator at row *t* uses only
    data available at or before *t*.  It is therefore safe to call on live /
    inference data without introducing look-ahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns Open, High, Low, Close, Volume.
        A DatetimeIndex is strongly recommended for correct VWAP and
        time-of-day encoding, but an integer index is tolerated.
    ticker : str
        Used only for log messages; does not affect computation.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with all FEATURE_COLS_V2 columns added (plus several
        intermediate columns such as ema_9, ema_50, obv, vwap that are
        needed by downstream callers).
    """
    if df is None or len(df) < 30:
        logger.debug("[%s] compute_features: too few rows (%s)", ticker, 0 if df is None else len(df))
        return df

    df = df.copy()
    # Normalise to Title-case so callers can pass either API (Title) or
    # SQLite / walk-forward (lowercase) DataFrames without KeyErrors.
    if "close" in df.columns:
        df = df.rename(columns={"open": "Open", "high": "High",
                                  "low": "Low", "close": "Close", "volume": "Volume"})
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ------------------------------------------------------------------
    # Original 23 features (V1 set — names must not change)
    # ------------------------------------------------------------------

    # -- Momentum -------------------------------------------------------
    df["rsi_14"] = ta.momentum.rsi(close, window=14).fillna(50.0)
    df["rsi_7"]  = ta.momentum.rsi(close, window=7).fillna(50.0)

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch().fillna(50.0)
    df["stoch_d"] = stoch.stoch_signal().fillna(50.0)

    # -- Trend ----------------------------------------------------------
    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd_obj.macd().fillna(0.0)
    df["macd_signal"] = macd_obj.macd_signal().fillna(0.0)
    df["macd_hist"]   = macd_obj.macd_diff().fillna(0.0)

    df["ema_9"]  = ta.trend.ema_indicator(close, window=9)
    df["ema_20"] = ta.trend.ema_indicator(close, window=20)
    df["ema_50"] = ta.trend.ema_indicator(close, window=50)

    # +1 when ema_9 > ema_20 (bullish), -1 otherwise (bearish)
    df["ema_cross"] = np.where(df["ema_9"] > df["ema_20"], 1.0, -1.0)

    df["cci_20"] = ta.trend.cci(high, low, close, window=20).fillna(0.0)

    # -- Volatility / Bands ---------------------------------------------
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pct"]   = bb.bollinger_pband().fillna(0.5)
    df["bb_width"] = bb.bollinger_wband().fillna(0.0)

    df["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)
    # ATR should never be exactly zero; replace zeros to avoid divide-by-zero later
    df["atr_14"] = df["atr_14"].replace(0, np.nan).ffill().fillna(close.pct_change().abs().mean() * close.mean())

    # -- Volume ---------------------------------------------------------
    df["mfi_14"]   = ta.volume.money_flow_index(high, low, close, vol, window=14).fillna(50.0)
    df["obv"]      = ta.volume.on_balance_volume(close, vol).fillna(0.0)
    df["vol_ratio"] = _safe_divide(vol, vol.rolling(20, min_periods=1).mean(), fill=1.0).clip(0, 20)

    # -- Returns --------------------------------------------------------
    df["ret_1"]  = close.pct_change(1).fillna(0.0)
    df["ret_3"]  = close.pct_change(3).fillna(0.0)
    df["ret_5"]  = close.pct_change(5).fillna(0.0)
    df["ret_10"] = close.pct_change(10).fillna(0.0)

    # -- Time-of-day cyclical encoding ----------------------------------
    time_sin, time_cos = _encode_time_of_day(df.index)
    df["time_sin"] = pd.Series(time_sin, index=df.index).fillna(0.0)
    df["time_cos"] = pd.Series(time_cos, index=df.index).fillna(1.0)

    # -- Intraday range position (0 = day low, 1 = day high) -----------
    # Use 78-bar rolling window ≈ 1 full 390-min session at 5-min bars
    day_high = high.rolling(78, min_periods=1).max()
    day_low  = low.rolling(78, min_periods=1).min()
    day_rng  = (day_high - day_low).replace(0, np.nan)
    df["price_range_pos"] = ((close - day_low) / day_rng).clip(0, 1).fillna(0.5)

    # -- Volume trend: 5-bar vs 20-bar rolling mean --------------------
    v5  = vol.rolling(5,  min_periods=1).mean()
    v20 = vol.rolling(20, min_periods=1).mean()
    df["vol_trend"] = _safe_divide(v5, v20, fill=1.0).clip(0, 5)

    # ------------------------------------------------------------------
    # New 9 features (V2 additions)
    # ------------------------------------------------------------------

    # -- vwap_dev: deviation from session VWAP -------------------------
    vwap = _compute_session_vwap(df)
    df["vwap"] = vwap  # keep for downstream callers (prediction.py, etc.)
    df["vwap_dev"] = _safe_divide(close - vwap, vwap, fill=0.0).clip(-0.05, 0.05)

    # -- adx_14: Average Directional Index normalised 0-1 --------------
    adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx_14"] = (adx_ind.adx().fillna(0.0) / 100.0).clip(0.0, 1.0)

    # -- cmf_20: Chaikin Money Flow ------------------------------------
    cmf_ind = ta.volume.ChaikinMoneyFlowIndicator(high, low, close, vol, window=20)
    df["cmf_20"] = cmf_ind.chaikin_money_flow().fillna(0.0).clip(-1.0, 1.0)

    # -- roc_5: 5-bar Rate of Change % ---------------------------------
    df["roc_5"] = ta.momentum.roc(close, window=5).fillna(0.0).clip(-20.0, 20.0)

    # -- williams_r: Williams %R normalised to 0-1 ---------------------
    wr_ind = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14)
    # ta returns values in [-100, 0]; we map to [0, 1] (0=most overbought, 1=most oversold)
    df["williams_r"] = ((wr_ind.williams_r().fillna(-50.0) + 100.0) / 100.0).clip(0.0, 1.0)

    # -- spread_pct: intrabar high-low spread --------------------------
    df["spread_pct"] = _safe_divide((high - low), close, fill=0.0).mul(100.0).clip(0.0, 10.0)

    # -- obv_slope: OBV z-score vs 10-bar rolling stats ---------------
    obv_roll_mean = df["obv"].rolling(10, min_periods=3).mean()
    obv_roll_std  = df["obv"].rolling(10, min_periods=3).std()
    df["obv_slope"] = (
        (df["obv"] - obv_roll_mean) / (obv_roll_std + 1e-8)
    ).fillna(0.0).clip(-5.0, 5.0)

    # -- ema_ribbon: ema_9 vs ema_50 relative to price -----------------
    df["ema_ribbon"] = _safe_divide(
        df["ema_9"] - df["ema_50"], close, fill=0.0
    ).mul(100.0).clip(-10.0, 10.0)

    # -- gap_open: overnight gap % ------------------------------------
    # (Open - prev_Close) / prev_Close * 100, clipped to ±5 %
    prev_close = close.shift(1)
    df["gap_open"] = _safe_divide(
        df["Open"] - prev_close, prev_close, fill=0.0
    ).mul(100.0).clip(-5.0, 5.0)

    # ------------------------------------------------------------------
    # V3 features — new professional indicators
    # ------------------------------------------------------------------

    # -- kc_squeeze: Bollinger inside Keltner Channel ------------------
    # Requires BB and KC already computed above; fallback to 0 if missing
    try:
        atr10 = ta.volatility.average_true_range(high, low, close, window=10).fillna(
            ta.volatility.average_true_range(high, low, close, window=14)
        )
        ema20 = ta.trend.ema_indicator(close, window=20)
        kc_upper = ema20 + 2.0 * atr10
        kc_lower = ema20 - 2.0 * atr10
        bb_obj2  = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_upper2 = bb_obj2.bollinger_hband()
        bb_lower2 = bb_obj2.bollinger_lband()
        bb_rng = bb_upper2 - bb_lower2
        kc_rng = kc_upper  - kc_lower
        df["kc_squeeze"] = (bb_rng < kc_rng).astype(float).fillna(0.0)
    except Exception:
        df["kc_squeeze"] = 0.0

    # -- cvd_5: 5-bar Cumulative Volume Delta --------------------------
    try:
        bar_range = (high - low).replace(0, np.nan)
        buy_vol   = ((close - low) / bar_range * vol).fillna(0)
        sell_vol  = vol - buy_vol
        df["cvd_5"] = (buy_vol - sell_vol).rolling(5).sum().fillna(0)
        # Normalise by average volume so it's comparable across tickers
        avg_vol = vol.rolling(20, min_periods=1).mean().replace(0, np.nan)
        df["cvd_5"] = (df["cvd_5"] / avg_vol).fillna(0.0).clip(-5.0, 5.0)
    except Exception:
        df["cvd_5"] = 0.0

    # -- ema_200_dev: deviation from EMA(200) --------------------------
    try:
        ema200 = ta.trend.ema_indicator(close, window=200)
        df["ema_200_dev"] = _safe_divide(close - ema200, ema200, fill=0.0).mul(100.0).clip(-10.0, 10.0)
    except Exception:
        df["ema_200_dev"] = 0.0

    # -- vwap_z: VWAP z-score ------------------------------------------
    # Volume-weighted σ from VWAP: z = (price - VWAP) / σ
    try:
        vwap_s = df["vwap"] if "vwap" in df.columns else vwap
        closes_arr = close.values.astype(float)
        vols_arr   = vol.values.astype(float)
        vwap_arr   = vwap_s.values.astype(float)
        z_scores   = np.zeros(len(df), dtype=float)
        for i in range(20, len(df)):
            p = closes_arr[max(0, i-20): i]
            v = vols_arr[max(0, i-20): i]
            w = vwap_arr[i]
            total = float(np.sum(v))
            if total > 0 and w > 0:
                std = float(np.sqrt(np.sum(v * (p - w) ** 2) / total))
                z_scores[i] = (closes_arr[i] - w) / std if std > 0 else 0.0
        df["vwap_z"] = pd.Series(z_scores, index=df.index).clip(-4.0, 4.0)
    except Exception:
        df["vwap_z"] = 0.0

    # -- hidden_div_bull / hidden_div_bear: hidden RSI divergence ------
    try:
        from agent.reversal import compute_reversal_features as _rev_feat, _DIV_LOOKBACK, _DIV_SWING_WINDOW, _find_swing_lows, _find_swing_highs
        prices_arr = close.values.astype(float)
        rsi_arr    = ta.momentum.rsi(close, window=14).fillna(50.0).values.astype(float)
        hb  = np.zeros(len(df), dtype=float)
        hbr = np.zeros(len(df), dtype=float)
        lookback = _DIV_LOOKBACK
        for i in range(lookback + _DIV_SWING_WINDOW + 1, len(df)):
            p_sl = prices_arr[i - lookback: i]
            r_sl = rsi_arr[i - lookback: i]
            p_lows = _find_swing_lows(p_sl)
            r_lows = _find_swing_lows(r_sl)
            if len(p_lows) >= 2 and len(r_lows) >= 2:
                if p_lows[-1][1] > p_lows[-2][1] and r_lows[-1][1] < r_lows[-2][1]:
                    hb[i] = 1.0
            p_highs = _find_swing_highs(p_sl)
            r_highs = _find_swing_highs(r_sl)
            if len(p_highs) >= 2 and len(r_highs) >= 2:
                if p_highs[-1][1] < p_highs[-2][1] and r_highs[-1][1] > r_highs[-2][1]:
                    hbr[i] = 1.0
        df["hidden_div_bull"] = hb
        df["hidden_div_bear"] = hbr
    except Exception:
        df["hidden_div_bull"] = 0.0
        df["hidden_div_bear"] = 0.0

    # -- nr7: narrowest range of last 7 bars ---------------------------
    try:
        bar_range_abs = (high - low).abs()
        df["nr7"] = (bar_range_abs == bar_range_abs.rolling(7, min_periods=1).min()).astype(float)
    except Exception:
        df["nr7"] = 0.0

    # -- adx_regime: 1 = trend (ADX>25), 0 = mean-reversion (ADX<20) --
    # adx_14 in this file is normalised to 0-1 (divided by 100)
    try:
        adx_raw = df["adx_14"] * 100.0 if "adx_14" in df.columns else \
                  ta.trend.ADXIndicator(high, low, close, window=14).adx().fillna(0.0)
        df["adx_regime"] = np.where(adx_raw > 25, 1.0, np.where(adx_raw < 20, 0.0, 0.5))
    except Exception:
        df["adx_regime"] = 0.5

    return df


# ---------------------------------------------------------------------------
# ATR-adaptive label construction
# ---------------------------------------------------------------------------

def _make_labels(
    df: pd.DataFrame,
    lookahead: int,
    atr_multiplier: float,
) -> pd.Series:
    """
    Build ATR-adaptive binary labels with a noise-exclusion zone.

    A row is labelled:
      - 1  if the forward return over *lookahead* bars >= ATR × atr_multiplier
      - 0  if the forward return <= -(ATR × atr_multiplier)
      - NaN if the move is inside the noise band (excluded from training)

    The ATR threshold uses a 20-bar rolling mean of atr_14 so that
    individual spike bars do not inflate the threshold excessively.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'Close' and 'atr_14'.
    lookahead : int
        Number of bars ahead to measure the return.
    atr_multiplier : float
        Fraction of ATR required for a clean signal.

    Returns
    -------
    pd.Series of float with values in {0.0, 1.0, NaN}.
    """
    # Express ATR as a fraction of price so it is directly comparable to fwd_ret
    # (which is also a fractional return).  Using a 20-bar rolling mean smooths
    # out individual spike bars without over-reacting to short bursts of volatility.
    atr_pct = (df["atr_14"] / df["Close"]).rolling(20, min_periods=5).mean()
    atr_threshold = atr_pct * atr_multiplier
    fwd_ret = df["Close"].shift(-lookahead) / df["Close"] - 1.0
    label = np.where(
        fwd_ret >= atr_threshold,  1.0,
        np.where(
            fwd_ret <= -atr_threshold, 0.0,
            np.nan   # noise band — exclude from training
        ),
    )
    return pd.Series(label, index=df.index)


# ---------------------------------------------------------------------------
# Public training entry-point
# ---------------------------------------------------------------------------

def prepare_training_data(
    df_raw: pd.DataFrame,
    ticker: str = "",
    lookahead_bars: int = 3,
    atr_multiplier: float = 0.3,
    test_size: float = 0.2,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Leakage-free feature computation, labelling and train/test split.

    The key difference from the original pipeline is that the raw OHLCV data
    is split *before* any indicator is computed.  Each partition (train / test)
    is feature-engineered independently, so rolling windows can never "see"
    future data from the other partition.

    Labels are ATR-adaptive: rows whose forward return falls inside ±(ATR ×
    atr_multiplier) are excluded entirely rather than being mislabelled as
    either class.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw OHLCV DataFrame, preferably with a DatetimeIndex.
    ticker : str
        Used for logging only.
    lookahead_bars : int
        Number of bars ahead used to measure the forward return for labelling.
        At 5-min resolution, 3 bars = 15 min.
    atr_multiplier : float
        Fraction of ATR that constitutes the minimum meaningful move.
        0.3 × ATR_14 is a reasonable default for 5-min scalping.
    test_size : float
        Fraction of data reserved for out-of-sample evaluation.

    Returns
    -------
    (X_train, y_train, X_test, y_test) as float32 / int numpy arrays, or
    None when there is insufficient data or class imbalance prevents training.
    """
    if df_raw is None or len(df_raw) < 200:
        logger.debug("[%s] prepare_training_data: insufficient rows", ticker)
        return None

    # ------------------------------------------------------------------ #
    # CRITICAL: split RAW data first, then compute features independently  #
    # ------------------------------------------------------------------ #
    split_idx    = int(len(df_raw) * (1.0 - test_size))
    df_train_raw = df_raw.iloc[:split_idx].copy()
    df_test_raw  = df_raw.iloc[split_idx:].copy()

    # Features computed on each partition independently — no cross-contamination
    df_train = compute_features(df_train_raw, ticker)
    df_test  = compute_features(df_test_raw,  ticker)

    # Build ATR-adaptive labels
    df_train["label"] = _make_labels(df_train, lookahead_bars, atr_multiplier)
    df_test["label"]  = _make_labels(df_test,  lookahead_bars, atr_multiplier)

    # Drop noise rows (label is NaN) and rows with any NaN feature
    required_cols = FEATURE_COLS_V2 + ["label"]
    df_train = df_train.dropna(subset=required_cols)
    df_test  = df_test.dropna(subset=required_cols)

    if len(df_train) < 60:
        logger.debug("[%s] prepare_training_data: train set too small (%d rows)", ticker, len(df_train))
        return None
    if len(df_test) < 10:
        logger.debug("[%s] prepare_training_data: test set too small (%d rows)", ticker, len(df_test))
        return None

    X_train = df_train[FEATURE_COLS_V2].values.astype(np.float32)
    y_train = df_train["label"].values.astype(int)
    X_test  = df_test[FEATURE_COLS_V2].values.astype(np.float32)
    y_test  = df_test["label"].values.astype(int)

    # Require both classes represented in training data
    if len(np.unique(y_train)) < 2:
        logger.debug("[%s] prepare_training_data: only one class in training set", ticker)
        return None

    logger.debug(
        "[%s] prepare_training_data: train=%d test=%d (class ratio train=%.2f)",
        ticker, len(X_train), len(X_test),
        y_train.mean(),
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Live / inference helper
# ---------------------------------------------------------------------------

def compute_live_row(df_raw: pd.DataFrame, ticker: str = "") -> Optional[np.ndarray]:
    """
    Compute all 32 features on the supplied OHLCV history and return only the
    most recent bar as a (1, 32) float32 array ready for model.predict_proba().

    Typically called with the last ~200 bars of 5-min data so that all
    rolling-window indicators are fully warmed up at the final row.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw OHLCV data (at least ~50 rows recommended; 200 preferred).
    ticker : str
        Used for logging only.

    Returns
    -------
    np.ndarray of shape (1, 32) or None if computation fails.
    """
    if df_raw is None or len(df_raw) < 30:
        logger.debug("[%s] compute_live_row: insufficient rows", ticker)
        return None

    try:
        df = compute_features(df_raw, ticker)
    except Exception as exc:
        logger.warning("[%s] compute_live_row: compute_features raised %s", ticker, exc)
        return None

    last = df[FEATURE_COLS_V2].iloc[[-1]]
    if last.isnull().any(axis=None):
        missing = last.columns[last.isnull().any()].tolist()
        logger.debug("[%s] compute_live_row: NaN in features %s — filling with 0", ticker, missing)
        last = last.fillna(0.0)

    return last.values.astype(np.float32)
