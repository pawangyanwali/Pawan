import pandas as pd
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────
_REG_OPEN_MINS  = 9 * 60 + 30   # 09:30 ET = 570 minutes from midnight
_REG_CLOSE_MINS = 16 * 60       # 16:00 ET = 960 minutes from midnight


def _et_minutes(ts) -> int:
    """Minutes since midnight ET for a Timestamp (tz-aware or assumed ET)."""
    t = ts.tz_convert("America/New_York") if getattr(ts, "tzinfo", None) else ts
    return t.hour * 60 + t.minute


def _detect_session(idx_et) -> str:
    """Return 'REGULAR', 'AFTER_HOURS', or 'PRE_MARKET' from the last bar's ET time."""
    last_mins = _et_minutes(idx_et[-1])
    if last_mins >= _REG_CLOSE_MINS:
        return "AFTER_HOURS"
    if last_mins < _REG_OPEN_MINS:
        return "PRE_MARKET"
    return "REGULAR"


def _session_mask_for(idx_et, session: str) -> "pd.array":
    """Boolean array selecting bars that belong to the given session."""
    mins = idx_et.hour * 60 + idx_et.minute
    if session == "AFTER_HOURS":
        return mins >= _REG_CLOSE_MINS
    if session == "PRE_MARKET":
        return mins < _REG_OPEN_MINS
    return (mins >= _REG_OPEN_MINS) & (mins < _REG_CLOSE_MINS)


def score_volume(df: pd.DataFrame) -> float:
    """
    Analyse volume behaviour and return a score in [-1, +1].
    Positive = volume confirms bullish move; negative = volume confirms bearish move.
    Returns 0 when there is insufficient data.

    In extended-hours sessions the baseline is computed from same-session bars
    only (AH vs AH, PM vs PM) so a 50K AH bar is not compared against a 5M
    regular-session average and falsely scored as ultra-low volume.
    """
    if df is None or len(df) < 5:
        return 0.0

    close = df["Close"]
    vol   = df["Volume"]

    last_close = close.iloc[-1]
    prev_close = close.iloc[-2] if len(close) > 1 else last_close
    last_vol   = vol.iloc[-1]
    price_dir  = 1 if last_close >= prev_close else -1

    # ── Session-filtered baseline ─────────────────────────────────────────────
    avg_vol_20, avg_vol_5 = _filtered_averages(df)
    if avg_vol_20 == 0:
        return 0.0

    relative_vol = last_vol / avg_vol_20
    vol_trend    = avg_vol_5 / avg_vol_20

    signals: list[float] = []

    # ── Relative volume spike ─────────────────────────────────────────────────
    if relative_vol > 3.0:
        signals.append(price_dir * 1.0)
    elif relative_vol > 2.0:
        signals.append(price_dir * 0.7)
    elif relative_vol > 1.5:
        signals.append(price_dir * 0.4)
    elif relative_vol < 0.5:
        signals.append(0.0)

    # ── Volume trend (5-bar avg vs 20-bar avg) ────────────────────────────────
    if vol_trend > 1.5:
        signals.append(price_dir * 0.5)
    elif vol_trend < 0.7:
        signals.append(0.0)

    # ── OBV momentum ─────────────────────────────────────────────────────────
    if "obv" in df.columns and len(df) >= 5:
        obv_now  = df["obv"].iloc[-1]
        obv_prev = df["obv"].iloc[-5]
        obv_dir  = np.sign(obv_now - obv_prev)
        signals.append(float(obv_dir) * 0.4)

    # ── MFI ──────────────────────────────────────────────────────────────────
    if "mfi_14" in df.columns:
        mfi = df["mfi_14"].iloc[-1]
        if pd.notna(mfi):
            if mfi < 20:
                signals.append(0.8)
            elif mfi > 80:
                signals.append(-0.8)
            else:
                signals.append(np.clip((mfi - 50) / 50, -0.5, 0.5))

    if not signals:
        return 0.0
    return float(np.clip(np.mean(signals), -1, 1))


def _filtered_averages(df: pd.DataFrame) -> tuple[float, float]:
    """Return (avg_vol_20, avg_vol_5) using same-session bars when in extended hours."""
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        idx_et = idx.tz_convert("America/New_York")
        session = _detect_session(idx_et)
        if session in ("AFTER_HOURS", "PRE_MARKET"):
            mask    = _session_mask_for(idx_et, session)
            sess_vol = df["Volume"][mask]
            if len(sess_vol) >= 5:
                avg_20 = float(sess_vol.iloc[-20:].mean())
                avg_5  = float(sess_vol.iloc[-5:].mean())
                return avg_20, avg_5
    # Regular session or insufficient extended-hours history — raw rolling window
    vol = df["Volume"]
    return float(vol.iloc[-20:].mean()), float(vol.iloc[-5:].mean())


def detect_unusual_volume(df: pd.DataFrame, threshold: float = 2.5) -> bool:
    """Return True if the latest bar has unusually high volume.

    For extended-hours sessions (AH/PM) the 20-bar baseline is computed from
    same-session bars only so thin-float stocks don't always show as unusual.
    """
    if df is None or len(df) < 5:
        return False
    last_vol   = df["Volume"].iloc[-1]
    avg_vol_20, _ = _filtered_averages(df)
    if avg_vol_20 == 0:
        return False
    return bool((last_vol / avg_vol_20) >= threshold)


def relative_volume(df: pd.DataFrame) -> float:
    """Return latest bar's volume as a multiple of the session-appropriate 20-bar average."""
    if df is None or len(df) < 5:
        return 1.0
    last_vol   = df["Volume"].iloc[-1]
    avg_vol_20, _ = _filtered_averages(df)
    if avg_vol_20 == 0:
        return 1.0
    return round(float(last_vol / avg_vol_20), 2)


# ── Regular-session intraday cumulative volume profile ────────────────────────
# Empirical: heavy open + close, light midday (U-shape). Minutes since 09:30.
_INTRADAY_CUM_PROFILE = [
    (5,   0.080), (10,  0.130), (15,  0.170), (20,  0.200),
    (30,  0.240), (45,  0.278), (60,  0.313), (90,  0.380),
    (120, 0.440), (150, 0.498), (180, 0.553), (210, 0.608),
    (240, 0.660), (270, 0.718), (300, 0.783), (330, 0.858),
    (360, 0.930), (390, 1.000),
]

# ── After-hours cumulative volume profile ─────────────────────────────────────
# 16:00–20:00 ET (240 mins). Heaviest in first 30 mins (earnings reactions,
# institutional rebalancing). Declines through the session.
# Minutes since 16:00 ET.
_AH_CUM_PROFILE = [
    (5,   0.200), (10,  0.330), (15,  0.430), (20,  0.500),
    (30,  0.600), (45,  0.690), (60,  0.750), (90,  0.830),
    (120, 0.890), (150, 0.930), (180, 0.960), (210, 0.980), (240, 1.000),
]

# ── Pre-market cumulative volume profile ──────────────────────────────────────
# 04:00–09:30 ET (330 mins). Very quiet 4–7 AM. Building from 7 AM as
# futures, news flow, and institutional orders arrive. Accelerates in the
# final 30 mins before the open.  Minutes since 04:00 ET.
_PM_CUM_PROFILE = [
    (30,  0.030), (60,  0.070), (90,  0.120), (120, 0.180),
    (150, 0.250), (180, 0.330), (210, 0.450), (240, 0.590),
    (270, 0.730), (300, 0.860), (330, 1.000),
]

# Approximate fraction of an average daily volume that trades in each
# extended session.  Used only in Method-2 fallback (≤ 3 historical sessions).
_AH_DAILY_FRACTION = 0.10   # ~10% of daily volume in AH
_PM_DAILY_FRACTION = 0.07   # ~7%  of daily volume in PM


def _session_minutes(ts) -> int:
    """Minutes elapsed since 9:30 ET (regular session reference)."""
    t = ts.tz_convert("America/New_York") if getattr(ts, "tzinfo", None) else ts
    return (t.hour - 9) * 60 + (t.minute - 30)


def _ah_elapsed(ts) -> int:
    """Minutes since 16:00 ET for an after-hours timestamp."""
    t = ts.tz_convert("America/New_York") if getattr(ts, "tzinfo", None) else ts
    return (t.hour - 16) * 60 + t.minute


def _pm_elapsed(ts) -> int:
    """Minutes since 04:00 ET for a pre-market timestamp."""
    t = ts.tz_convert("America/New_York") if getattr(ts, "tzinfo", None) else ts
    return (t.hour - 4) * 60 + t.minute


def _interp_profile(mins: int, profile: list) -> float:
    """Linearly interpolate a cumulative volume fraction from any profile list."""
    mins = max(0, min(mins, profile[-1][0]))
    if mins <= profile[0][0]:
        return profile[0][1] * mins / max(profile[0][0], 1)
    for i in range(1, len(profile)):
        m0, f0 = profile[i - 1]
        m1, f1 = profile[i]
        if mins <= m1:
            return f0 + (f1 - f0) * (mins - m0) / (m1 - m0)
    return 1.0


def _cum_fraction_at(mins: int) -> float:
    """Regular-session cumulative fraction at elapsed minutes since 09:30."""
    return _interp_profile(mins, _INTRADAY_CUM_PROFILE)


def _rvol_extended(
    df_1m: pd.DataFrame,
    idx_et,
    today,
    today_session_mask,
    elapsed_fn,
    profile: list,
    daily_fraction: float,
    df_1d: pd.DataFrame | None,
    session: str,
) -> float:
    """Shared RVOL computation for AH and PM sessions."""
    today_sess_bars = df_1m[today_session_mask]
    if len(today_sess_bars) == 0:
        return 1.0

    cum_vol_today = float(today_sess_bars["Volume"].sum())
    last_ts_et    = idx_et[today_session_mask][-1]
    elapsed       = elapsed_fn(last_ts_et)
    if elapsed <= 0:
        return 1.0

    # ── Method 1: historical same-session bars from df_1m ─────────────────────
    past_dates = sorted(set(idx_et.date) - {today})
    if len(past_dates) >= 3:   # lower bar than regular session — less ext history
        baselines = []
        for d in past_dates[-20:]:
            d_mask    = idx_et.date == d
            sess_mask = d_mask & _session_mask_for(idx_et, session)
            sess      = df_1m[sess_mask]
            if len(sess) == 0:
                continue
            sess_idx_et  = idx_et[sess_mask]
            sess_elapsed = [elapsed_fn(t) for t in sess_idx_et]
            up_to = sum(
                float(v) for m, v in zip(sess_elapsed, sess["Volume"])
                if 0 <= m <= elapsed
            )
            if up_to > 0:
                baselines.append(up_to)
        if baselines:
            baseline = float(np.mean(baselines))
            if baseline > 0:
                return round(cum_vol_today / baseline, 2)

    # ── Method 2: avg daily vol × session fraction × profile ─────────────────
    avg_daily = None
    if df_1d is not None and len(df_1d) >= 5:
        avg_daily = float(df_1d["Volume"].iloc[-20:].mean()) if len(df_1d) >= 20 \
            else float(df_1d["Volume"].mean())

    if avg_daily and avg_daily > 0:
        frac     = _interp_profile(elapsed, profile)
        expected = avg_daily * daily_fraction * frac
        if expected > 0:
            return round(cum_vol_today / expected, 2)

    return relative_volume(df_1m)


def rvol_time_of_day(df_1m: pd.DataFrame, df_1d: pd.DataFrame = None) -> float:
    """
    Time-of-day adjusted RVOL with session-aware baselines.

    Regular session (09:30–16:00 ET):
      Method 1: cumulative volume vs same-elapsed-minute baseline from past sessions.
      Method 2: avg daily vol × empirical regular-session intraday profile.

    Extended hours (PRE_MARKET 04:00–09:30, AFTER_HOURS 16:00–20:00 ET):
      Historical baseline uses ONLY same-session bars (AH vs AH history, PM vs PM).
      Fallback uses avg daily vol × session daily-fraction × session-specific profile.
      This prevents AAPL's 2M AH share count from being compared against its 50M
      regular-session daily average and appearing as near-zero RVOL.

    Returns: current cumulative session volume / expected cumulative session volume (≥ 0).
    """
    if df_1m is None or len(df_1m) == 0:
        return 1.0

    try:
        idx = df_1m.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx_et = idx.tz_convert("America/New_York")
        else:
            idx_et = idx

        today   = idx_et[-1].date()
        session = _detect_session(idx_et)

        # ── Extended-hours path ───────────────────────────────────────────────
        if session == "AFTER_HOURS":
            today_mask = (idx_et.date == today) & _session_mask_for(idx_et, "AFTER_HOURS")
            return _rvol_extended(
                df_1m, idx_et, today, today_mask,
                _ah_elapsed, _AH_CUM_PROFILE, _AH_DAILY_FRACTION,
                df_1d, "AFTER_HOURS",
            )

        if session == "PRE_MARKET":
            today_mask = (idx_et.date == today) & _session_mask_for(idx_et, "PRE_MARKET")
            return _rvol_extended(
                df_1m, idx_et, today, today_mask,
                _pm_elapsed, _PM_CUM_PROFILE, _PM_DAILY_FRACTION,
                df_1d, "PRE_MARKET",
            )

        # ── Regular session path (original logic, preserved) ──────────────────
        today_mask  = idx_et.date == today
        today_bars  = df_1m[today_mask]

        if len(today_bars) == 0:
            return 1.0

        cum_vol_today = float(today_bars["Volume"].sum())
        last_ts       = idx_et[today_mask][-1]
        elapsed       = _session_minutes(last_ts)
        if elapsed <= 0:
            return 1.0

        past_dates = sorted(set(idx_et.date) - {today})
        if len(past_dates) >= 5:
            baselines = []
            for d in past_dates[-20:]:
                mask      = idx_et.date == d
                sess      = df_1m[mask]
                if len(sess) == 0:
                    continue
                sess_idx_et  = idx_et[mask]
                sess_elapsed = [_session_minutes(t) for t in sess_idx_et]
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

        avg_daily = None
        if df_1d is not None and len(df_1d) >= 5:
            avg_daily = float(df_1d["Volume"].iloc[-20:].mean()) if len(df_1d) >= 20 \
                else float(df_1d["Volume"].mean())

        if avg_daily is None or avg_daily <= 0:
            return relative_volume(df_1m)

        frac = _cum_fraction_at(elapsed)
        if frac <= 0:
            return 1.0
        expected = avg_daily * frac
        return round(cum_vol_today / expected, 2)

    except Exception:
        return relative_volume(df_1m)
