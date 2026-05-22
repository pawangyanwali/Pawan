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
                # MFI > 50 → more money flowing in → bullish (+)
                signals.append(np.clip((mfi - 50) / 50, -0.5, 0.5))

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


# Empirical cumulative intraday volume fractions (minutes since 9:30 → cumulative %)
# Derived from broad market averages: heavy open + close, light midday (U-shape).
_INTRADAY_CUM_PROFILE = [
    (5,   0.080), (10,  0.130), (15,  0.170), (20,  0.200),
    (30,  0.240), (45,  0.278), (60,  0.313), (90,  0.380),
    (120, 0.440), (150, 0.498), (180, 0.553), (210, 0.608),
    (240, 0.660), (270, 0.718), (300, 0.783), (330, 0.858),
    (360, 0.930), (390, 1.000),
]


def _session_minutes(ts) -> int:
    """Minutes elapsed since 9:30 ET for a Timestamp (tz-aware or naive ET)."""
    t = ts.tz_convert("America/New_York") if getattr(ts, "tzinfo", None) else ts
    return (t.hour - 9) * 60 + (t.minute - 30)


def _cum_fraction_at(mins: int) -> float:
    """Interpolate cumulative volume fraction from the empirical intraday profile."""
    mins = max(0, min(mins, 390))
    profile = _INTRADAY_CUM_PROFILE
    if mins <= profile[0][0]:
        return profile[0][1] * mins / profile[0][0]
    for i in range(1, len(profile)):
        m0, f0 = profile[i - 1]
        m1, f1 = profile[i]
        if mins <= m1:
            return f0 + (f1 - f0) * (mins - m0) / (m1 - m0)
    return 1.0


def rvol_time_of_day(df_1m: pd.DataFrame, df_1d: pd.DataFrame = None) -> float:
    """
    Time-of-day adjusted RVOL.

    For each historical session in df_1m we measure how much volume had
    accumulated by the same elapsed minute — that becomes the baseline.
    Falls back to avg_daily_vol × empirical fraction when fewer than 5
    historical sessions are present in df_1m.

    Returns: current cumulative volume / expected cumulative volume (≥ 0).
    """
    if df_1m is None or len(df_1m) == 0:
        return 1.0

    try:
        idx = df_1m.index
        # Convert to ET if tz-aware
        if hasattr(idx, "tz") and idx.tz is not None:
            idx_et = idx.tz_convert("America/New_York")
        else:
            idx_et = idx

        today = idx_et[-1].date()
        today_mask = idx_et.date == today
        today_bars = df_1m[today_mask]

        if len(today_bars) == 0:
            return 1.0

        cum_vol_today = float(today_bars["Volume"].sum())
        last_ts = idx_et[today_mask][-1]
        elapsed = _session_minutes(last_ts)
        if elapsed <= 0:
            return 1.0

        # ── Method 1: historical sessions from df_1m ──────────────────────
        past_dates = sorted(set(idx_et.date) - {today})
        if len(past_dates) >= 5:
            baselines = []
            for d in past_dates[-20:]:  # up to 20 recent sessions
                mask = idx_et.date == d
                sess = df_1m[mask]
                if len(sess) == 0:
                    continue
                sess_idx_et = idx_et[mask]
                sess_elapsed = [_session_minutes(t) for t in sess_idx_et]
                # accumulate volume only up to 'elapsed' minutes
                up_to = sum(
                    float(v) for m, v in zip(sess_elapsed, sess["Volume"])
                    if m <= elapsed
                )
                if up_to > 0:
                    baselines.append(up_to)
            if baselines:
                baseline = float(np.mean(baselines))
                if baseline > 0:
                    return round(cum_vol_today / baseline, 2)

        # ── Method 2: avg daily vol × empirical fraction ──────────────────
        avg_daily = None
        if df_1d is not None and len(df_1d) >= 5:
            avg_daily = float(df_1d["Volume"].iloc[-20:].mean()) if len(df_1d) >= 20 \
                else float(df_1d["Volume"].mean())

        if avg_daily is None or avg_daily <= 0:
            # Last resort: simple ratio using today's own volume vs 20-bar 1m avg
            return relative_volume(df_1m)

        frac = _cum_fraction_at(elapsed)
        if frac <= 0:
            return 1.0
        expected = avg_daily * frac
        return round(cum_vol_today / expected, 2)

    except Exception:
        return relative_volume(df_1m)
