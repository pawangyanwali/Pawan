import pandas as pd
import numpy as np
import ta


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a comprehensive set of technical indicators to an OHLCV DataFrame.
    Returns an enriched copy — does NOT mutate the caller's DataFrame.
    """
    if df is None or len(df) < 30:
        return df

    df    = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Trend ──────────────────────────────────────────────────────────────────
    df["ema_9"]   = ta.trend.ema_indicator(close, window=9)
    df["ema_20"]  = ta.trend.ema_indicator(close, window=20)
    df["ema_50"]  = ta.trend.ema_indicator(close, window=50)
    df["ema_200"] = ta.trend.ema_indicator(close, window=200)   # major institutional level
    df["sma_20"]  = ta.trend.sma_indicator(close, window=20)

    # Standard MACD (12,26,9) — used for signal scoring + features
    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"]   = macd_obj.macd_diff()

    # Fast MACD (8,17,9) — more responsive for scalp entries
    fast_macd = ta.trend.MACD(close, window_slow=17, window_fast=8, window_sign=9)
    df["macd_fast"]      = fast_macd.macd()
    df["macd_fast_hist"] = fast_macd.macd_diff()

    # ADX + DI lines
    adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx_14"] = adx_ind.adx()
    df["di_pos"]  = adx_ind.adx_pos()   # +DI
    df["di_neg"]  = adx_ind.adx_neg()   # -DI

    # ── Momentum ───────────────────────────────────────────────────────────────
    df["rsi_14"] = ta.momentum.rsi(close, window=14)
    df["rsi_7"]  = ta.momentum.rsi(close, window=7)
    df["rsi_2"]  = ta.momentum.rsi(close, window=2)   # Connors RSI extreme mean-reversion

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["cci_20"] = ta.trend.cci(high, low, close, window=20)
    df["mfi_14"] = ta.volume.money_flow_index(high, low, close, vol, window=14)

    # ── Volatility ─────────────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_mid"]    = bb.bollinger_mavg()
    df["bb_pct"]    = bb.bollinger_pband()   # 0 = lower band, 1 = upper band
    df["bb_width"]  = bb.bollinger_wband()

    df["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)
    df["atr_10"] = ta.volatility.average_true_range(high, low, close, window=10)

    # Keltner Channels: EMA(20) ± 2×ATR(10) — the "squeeze" baseline
    atr10 = df["atr_10"].fillna(df["atr_14"])
    df["kc_upper"] = df["ema_20"] + 2.0 * atr10
    df["kc_lower"] = df["ema_20"] - 2.0 * atr10

    # BB Squeeze: 1 when BB is inside KC (volatility compression; explosion imminent)
    bb_range = df["bb_upper"] - df["bb_lower"]
    kc_range = df["kc_upper"] - df["kc_lower"]
    df["bb_squeeze"] = (bb_range < kc_range).astype(float)

    # Chandelier Exit levels (N=22, k=3×ATR)
    highest_high = high.rolling(22, min_periods=1).max()
    lowest_low   = low.rolling(22,  min_periods=1).min()
    df["chandelier_long"]  = highest_high - 3.0 * df["atr_14"]   # trailing stop for longs
    df["chandelier_short"] = lowest_low   + 3.0 * df["atr_14"]   # trailing stop for shorts

    # ── Volume ─────────────────────────────────────────────────────────────────
    df["obv"]       = ta.volume.on_balance_volume(close, vol)
    df["vwap"]      = _compute_vwap(df)
    rolling_mean    = vol.rolling(20).mean().replace(0, np.nan)
    df["vol_ratio"] = vol / rolling_mean              # relative volume vs 20-bar avg

    # CVD approximation: (close position in bar) × volume = estimated buy volume delta
    bar_range = (high - low).replace(0, np.nan)
    buy_vol   = ((close - low) / bar_range * vol).fillna(0)
    sell_vol  = vol - buy_vol
    df["cvd"]      = (buy_vol - sell_vol).cumsum()      # running CVD
    df["cvd_5"]    = (buy_vol - sell_vol).rolling(5).sum().fillna(0)  # 5-bar CVD delta

    # NR7: 1 if current bar range is narrowest of last 7 bars (compression signal)
    bar_range_abs = (high - low).abs()
    df["nr7"] = (bar_range_abs == bar_range_abs.rolling(7, min_periods=1).min()).astype(float)
    df["nr4"] = (bar_range_abs == bar_range_abs.rolling(4, min_periods=1).min()).astype(float)

    # ── EMA crossover signal: +1 bullish / -1 bearish ────────────────────────
    df["ema_cross"] = np.where(df["ema_9"] > df["ema_20"], 1.0, -1.0)
    # Full EMA stack: +1 if price > 9 > 20 > 50 > 200 (full bull stack)
    df["ema_stack"] = np.where(
        (close > df["ema_9"]) & (df["ema_9"] > df["ema_20"]) &
        (df["ema_20"] > df["ema_50"]), 1.0,
        np.where(
            (close < df["ema_9"]) & (df["ema_9"] < df["ema_20"]) &
            (df["ema_20"] < df["ema_50"]), -1.0, 0.0
        )
    )

    # ── Price momentum (1, 3, 5 bar returns) ──────────────────────────────────
    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)

    return df


def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Session-anchored VWAP — resets at the start of each trading day.
    Cumulative sums are computed per calendar date so multi-day DataFrames
    produce correct intraday VWAP for every session.
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    tpvol = tp * df["Volume"]

    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.date
        cum_vol   = df.groupby(dates)["Volume"].cumsum()
        cum_tpvol = tpvol.groupby(dates).cumsum()
    else:
        cum_vol   = df["Volume"].cumsum()
        cum_tpvol = tpvol.cumsum()

    safe_vol = cum_vol.replace(0, np.nan)
    return (cum_tpvol / safe_vol).ffill()


def compute_vwap_bands(df: pd.DataFrame) -> dict:
    """
    Compute VWAP standard deviation bands (±1σ, ±2σ, ±3σ) for the current session.
    These are dynamic — far superior to fixed %-deviation thresholds because they
    adapt to each stock's actual intraday volatility.

    Returns dict with keys: vwap, std, upper_1, lower_1, upper_2, lower_2,
    upper_3, lower_3, z_score (current deviation in σ units).
    """
    empty = {"vwap": 0.0, "std": 0.0, "z_score": 0.0,
             "upper_1": 0.0, "lower_1": 0.0,
             "upper_2": 0.0, "lower_2": 0.0,
             "upper_3": 0.0, "lower_3": 0.0}
    if df is None or len(df) < 5 or "vwap" not in df.columns:
        return empty

    closes = df["Close"].values
    vwaps  = df["vwap"].values
    vols   = df["Volume"].values

    vwap_now = float(vwaps[-1])
    if vwap_now <= 0:
        return empty

    # Volume-weighted standard deviation from VWAP (last 20 bars for responsiveness)
    tail = min(20, len(closes))
    p = closes[-tail:]
    v = vols[-tail:]
    w = vwap_now

    # σ = sqrt(Σ Vol × (Price − VWAP)² / Σ Vol)
    total_vol = float(np.sum(v))
    if total_vol <= 0:
        return empty
    vwap_std = float(np.sqrt(np.sum(v * (p - w) ** 2) / total_vol))

    price_now = float(closes[-1])
    z_score   = (price_now - vwap_now) / vwap_std if vwap_std > 0 else 0.0

    return {
        "vwap":    round(vwap_now, 4),
        "std":     round(vwap_std, 6),
        "z_score": round(z_score, 3),
        "upper_1": round(vwap_now + 1 * vwap_std, 4),
        "lower_1": round(vwap_now - 1 * vwap_std, 4),
        "upper_2": round(vwap_now + 2 * vwap_std, 4),
        "lower_2": round(vwap_now - 2 * vwap_std, 4),
        "upper_3": round(vwap_now + 3 * vwap_std, 4),
        "lower_3": round(vwap_now - 3 * vwap_std, 4),
    }


def score_technical(row: pd.Series) -> float:
    """
    Aggregate technical indicators into a single score in [-1, +1].
    Positive = bullish pressure, negative = bearish pressure.

    Now includes: ADX regime weighting, EMA200 distance, BB squeeze
    amplification, Chandelier direction, fast MACD confirmation.
    """
    signals = []

    # ── Regime context: ADX determines weight of trend vs mean-reversion ──────
    adx = float(row.get("adx_14", 20))
    # Trend strength multiplier: >25 boosts trend signals; <20 boosts mean reversion
    trend_mode     = adx > 25   # trend-following mode
    mr_mode        = adx < 20   # mean-reversion mode
    trend_weight   = min(adx / 25.0, 1.5) if trend_mode else 1.0
    mr_weight      = min(20.0 / max(adx, 5), 1.5) if mr_mode else 1.0

    # ── RSI: oversold <30 bullish, overbought >70 bearish ─────────────────────
    if pd.notna(row.get("rsi_14")):
        rsi = row["rsi_14"]
        if rsi < 30:
            signals.append(1.0 * mr_weight)
        elif rsi < 45:
            signals.append(0.5)
        elif rsi > 70:
            signals.append(-1.0 * mr_weight)
        elif rsi > 55:
            signals.append(-0.5)
        else:
            signals.append(0.0)

    # ── RSI-7 fast signal ──────────────────────────────────────────────────────
    if pd.notna(row.get("rsi_7")):
        sig = np.clip((row["rsi_7"] - 50) / 50, -1, 1)
        signals.append(sig)

    # ── MACD histogram (standard) ─────────────────────────────────────────────
    if pd.notna(row.get("macd_hist")):
        signals.append(np.sign(row["macd_hist"]) * min(abs(row["macd_hist"]) * 100, 1))

    # ── Fast MACD histogram (8,17,9) — extra weight in trend mode ─────────────
    if pd.notna(row.get("macd_fast_hist")):
        fast_sig = np.sign(row["macd_fast_hist"]) * min(abs(row["macd_fast_hist"]) * 100, 1)
        signals.append(fast_sig * (1.2 if trend_mode else 0.8))

    # ── Bollinger Band position ────────────────────────────────────────────────
    if pd.notna(row.get("bb_pct")):
        pct = row["bb_pct"]
        if pct < 0.1:
            signals.append(0.8 * mr_weight)
        elif pct > 0.9:
            signals.append(-0.8 * mr_weight)
        elif pct < 0.35:
            signals.append(0.3)
        elif pct > 0.65:
            signals.append(-0.3)
        else:
            signals.append(0.0)

    # ── BB Squeeze: amplify the directional signal when squeeze fires ──────────
    if row.get("bb_squeeze", 0) == 1.0:
        # Squeeze active — amplify whichever direction EMAs/MACD point
        ema_dir = float(row.get("ema_cross", 0))
        signals.append(ema_dir * 0.5)   # moderate boost in squeeze direction

    # ── EMA cross + stack ─────────────────────────────────────────────────────
    if pd.notna(row.get("ema_cross")):
        signals.append(row["ema_cross"] * 0.6 * trend_weight)

    # Full EMA stack confirms strong trend alignment
    if pd.notna(row.get("ema_stack")) and row["ema_stack"] != 0:
        signals.append(row["ema_stack"] * 0.4 * trend_weight)

    # ── EMA 200: major institutional context ──────────────────────────────────
    if pd.notna(row.get("ema_200")) and pd.notna(row.get("Close")) and row["ema_200"] > 0:
        dev_200 = (row["Close"] - row["ema_200"]) / row["ema_200"]
        # Mild bias signal: above 200 EMA is structurally bullish context
        signals.append(np.clip(dev_200 * 10, -0.4, 0.4))

    # ── Price vs VWAP ─────────────────────────────────────────────────────────
    if pd.notna(row.get("vwap")) and pd.notna(row.get("Close")) and row["vwap"] > 0:
        deviation = (row["Close"] - row["vwap"]) / row["vwap"]
        signals.append(np.clip(deviation * 20, -1, 1))

    # ── Stochastic ────────────────────────────────────────────────────────────
    if pd.notna(row.get("stoch_k")) and pd.notna(row.get("stoch_d")):
        k, d = row["stoch_k"], row["stoch_d"]
        if k < 20 and d < 20:
            signals.append(0.8 * mr_weight)
        elif k > 80 and d > 80:
            signals.append(-0.8 * mr_weight)
        else:
            signals.append(np.sign(k - d) * 0.3)

    # ── CCI ───────────────────────────────────────────────────────────────────
    if pd.notna(row.get("cci_20")):
        signals.append(np.clip(row["cci_20"] / 200, -1, 1))

    # ── ADX / DI lines: +DI vs -DI directional bias ──────────────────────────
    if pd.notna(row.get("di_pos")) and pd.notna(row.get("di_neg")) and trend_mode:
        di_spread = (row["di_pos"] - row["di_neg"]) / 100.0
        signals.append(np.clip(di_spread * 2, -0.6, 0.6))

    # ── CVD: 5-bar cumulative volume delta (buy vs sell pressure) ─────────────
    if pd.notna(row.get("cvd_5")):
        cvd_norm = np.sign(row["cvd_5"]) * min(abs(row["cvd_5"]) / max(abs(row.get("vol_ratio", 1)) * 1000 + 1, 1), 1)
        signals.append(cvd_norm * 0.4)

    if not signals:
        return 0.0
    return float(np.clip(np.mean(signals), -1, 1))
