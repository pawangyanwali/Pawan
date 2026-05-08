import pandas as pd
import numpy as np


def score_volume(df: pd.DataFrame) -> float:
    """
    Analyse volume behaviour and return a score in [-1, +1].
    Positive = volume confirms bullish move; negative = volume confirms bearish move.
    Returns 0 when there is insufficient data.
    """
    if df is None or len(df) < 20:
        return 0.0

    close = df["Close"]
    vol   = df["Volume"]

    last_close   = close.iloc[-1]
    prev_close   = close.iloc[-2] if len(close) > 1 else last_close
    last_vol     = vol.iloc[-1]
    avg_vol_20   = vol.iloc[-20:].mean()
    avg_vol_5    = vol.iloc[-5:].mean()

    if avg_vol_20 == 0:
        return 0.0

    relative_vol = last_vol / avg_vol_20    # 1.0 = average, 2.0 = double average
    price_dir    = 1 if last_close >= prev_close else -1

    signals: list[float] = []

    # ── Relative volume spike ─────────────────────────────────────────────────
    if relative_vol > 3.0:
        signals.append(price_dir * 1.0)       # very unusual volume, confirms direction
    elif relative_vol > 2.0:
        signals.append(price_dir * 0.7)
    elif relative_vol > 1.5:
        signals.append(price_dir * 0.4)
    elif relative_vol < 0.5:
        signals.append(0.0)                   # low volume → no conviction

    # ── Volume trend (5-bar avg vs 20-bar avg) ────────────────────────────────
    vol_trend = avg_vol_5 / avg_vol_20
    if vol_trend > 1.5:
        signals.append(price_dir * 0.5)       # rising volume environment
    elif vol_trend < 0.7:
        signals.append(0.0)

    # ── OBV momentum: is OBV trending same direction as price? ────────────────
    if "obv" in df.columns and len(df) >= 5:
        obv_now  = df["obv"].iloc[-1]
        obv_prev = df["obv"].iloc[-5]
        obv_dir  = np.sign(obv_now - obv_prev)
        signals.append(float(obv_dir) * 0.4)

    # ── MFI (Money Flow Index) ────────────────────────────────────────────────
    if "mfi_14" in df.columns:
        mfi = df["mfi_14"].iloc[-1]
        if pd.notna(mfi):
            if mfi < 20:
                signals.append(0.8)   # oversold → bullish
            elif mfi > 80:
                signals.append(-0.8)  # overbought → bearish
            else:
                signals.append(np.clip((mfi - 50) / 50 * -1, -0.5, 0.5))

    if not signals:
        return 0.0
    return float(np.clip(np.mean(signals), -1, 1))


def detect_unusual_volume(df: pd.DataFrame, threshold: float = 2.5) -> bool:
    """Return True if the latest bar has unusually high volume."""
    if df is None or len(df) < 20:
        return False
    last_vol   = df["Volume"].iloc[-1]
    avg_vol_20 = df["Volume"].iloc[-20:].mean()
    if avg_vol_20 == 0:
        return False
    return bool((last_vol / avg_vol_20) >= threshold)


def relative_volume(df: pd.DataFrame) -> float:
    """Return latest bar's volume as a multiple of the 20-bar average."""
    if df is None or len(df) < 20:
        return 1.0
    last_vol   = df["Volume"].iloc[-1]
    avg_vol_20 = df["Volume"].iloc[-20:].mean()
    if avg_vol_20 == 0:
        return 1.0
    return round(float(last_vol / avg_vol_20), 2)
