import pandas as pd
import numpy as np
import ta


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a comprehensive set of technical indicators to an OHLCV DataFrame.
    All indicator columns are added in-place and the enriched DataFrame is returned.
    """
    if df is None or len(df) < 30:
        return df

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Trend ──────────────────────────────────────────────────────────────────
    df["ema_9"]  = ta.trend.ema_indicator(close, window=9)
    df["ema_20"] = ta.trend.ema_indicator(close, window=20)
    df["ema_50"] = ta.trend.ema_indicator(close, window=50)
    df["sma_20"] = ta.trend.sma_indicator(close, window=20)

    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"]   = macd_obj.macd_diff()

    # ── Momentum ───────────────────────────────────────────────────────────────
    df["rsi_14"] = ta.momentum.rsi(close, window=14)
    df["rsi_7"]  = ta.momentum.rsi(close, window=7)

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

    # ── Volume ─────────────────────────────────────────────────────────────────
    df["obv"]       = ta.volume.on_balance_volume(close, vol)
    df["vwap"]      = _compute_vwap(df)
    df["vol_ratio"] = vol / vol.rolling(20).mean()   # relative volume vs 20-bar avg

    # ── EMA crossover signal: +1 bullish / -1 bearish ────────────────────────
    df["ema_cross"] = np.where(df["ema_9"] > df["ema_20"], 1.0, -1.0)

    # ── Price momentum (1, 3, 5 bar returns) ──────────────────────────────────
    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)

    return df


def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP: cumulative (typical_price * volume) / cumulative volume."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol  = df["Volume"].cumsum()
    cum_tpvol = (tp * df["Volume"]).cumsum()
    return cum_tpvol / cum_vol


def score_technical(row: pd.Series) -> float:
    """
    Aggregate technical indicators into a single score in [-1, +1].
    Positive = bullish pressure, negative = bearish pressure.
    """
    signals = []

    # RSI: oversold <30 bullish, overbought >70 bearish
    if pd.notna(row.get("rsi_14")):
        rsi = row["rsi_14"]
        if rsi < 30:
            signals.append(1.0)
        elif rsi < 45:
            signals.append(0.5)
        elif rsi > 70:
            signals.append(-1.0)
        elif rsi > 55:
            signals.append(-0.5)
        else:
            signals.append(0.0)

    # RSI-7 fast signal
    if pd.notna(row.get("rsi_7")):
        rsi7 = row["rsi_7"]
        sig = np.clip((50 - rsi7) / 50, -1, 1)
        signals.append(-sig)  # invert: rsi7>50 means up momentum

    # MACD histogram direction
    if pd.notna(row.get("macd_hist")):
        signals.append(np.sign(row["macd_hist"]) * min(abs(row["macd_hist"]) * 100, 1))

    # Bollinger Band position
    if pd.notna(row.get("bb_pct")):
        pct = row["bb_pct"]
        if pct < 0.1:
            signals.append(0.8)   # near lower band → bounce potential
        elif pct > 0.9:
            signals.append(-0.8)  # near upper band → pullback potential
        elif pct < 0.35:
            signals.append(0.3)
        elif pct > 0.65:
            signals.append(-0.3)
        else:
            signals.append(0.0)

    # EMA cross
    if pd.notna(row.get("ema_cross")):
        signals.append(row["ema_cross"] * 0.6)

    # Price vs VWAP
    if pd.notna(row.get("vwap")) and pd.notna(row.get("Close")) and row["vwap"] > 0:
        deviation = (row["Close"] - row["vwap"]) / row["vwap"]
        signals.append(np.clip(deviation * 20, -1, 1))

    # Stochastic
    if pd.notna(row.get("stoch_k")) and pd.notna(row.get("stoch_d")):
        k, d = row["stoch_k"], row["stoch_d"]
        if k < 20 and d < 20:
            signals.append(0.8)
        elif k > 80 and d > 80:
            signals.append(-0.8)
        else:
            signals.append(np.sign(k - d) * 0.3)

    # CCI
    if pd.notna(row.get("cci_20")):
        cci = row["cci_20"]
        signals.append(np.clip(cci / 200, -1, 1))

    if not signals:
        return 0.0
    return float(np.clip(np.mean(signals), -1, 1))
