"""
Opening Range (OR) tracker for ORB-15 and ORB-30 strategies.

The Opening Range is the high and low of the first N minutes after market open
(09:30–09:45 for ORB-15; 09:30–10:00 for ORB-30).  These levels act as the
day's primary S/R anchors.  A close outside the OR is a high-probability
directional signal for the rest of the session.

Returns
-------
OpeningRangeResult with:
  orh_15, orl_15     : ORB-15 high/low
  orh_30, orl_30     : ORB-30 high/low
  or_width_15_pct    : ORB-15 range as % of mid-price
  or_width_30_pct    : ORB-30 range as % of mid-price
  position_vs_or15   : ABOVE | BELOW | INSIDE (price vs ORB-15)
  position_vs_or30   : ABOVE | BELOW | INSIDE (price vs ORB-30)
  breakout_15        : BULL | BEAR | NONE
  breakout_30        : BULL | BEAR | NONE
  or15_score         : float [-1, +1] signal score
  or30_score         : float [-1, +1] signal score
  description        : str

Strategy rules
--------------
Bull breakout (price > ORH):
  - Enter long on first 1-min close above ORH
  - Stop = ORL (or tighter ATR stop)
  - Target = ORH + (ORH - ORL) × 1.0 (1:1 risk/reward minimum)

Bear breakdown (price < ORL):
  - Enter short on first 1-min close below ORL
  - Stop = ORH
  - Target = ORL - (ORH - ORL) × 1.0

Mean reversion inside OR (RANGE_DAY context):
  - Buy near ORL with stop below ORL; target = ORH
  - Sell near ORH with stop above ORH; target = ORL
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OR15_END = ("09:30", "09:45")   # 15-min opening range
_OR30_END = ("09:30", "10:00")   # 30-min opening range

# Breakout requires price to CLOSE above ORH / below ORL (not just wick)
# Score magnitudes
_BREAKOUT_SCORE = 0.80
_INSIDE_BULL    = 0.20    # inside OR but biased toward ORH
_INSIDE_BEAR    = -0.20   # inside OR but biased toward ORL


@dataclass
class OpeningRangeResult:
    orh_15:           float = 0.0
    orl_15:           float = 0.0
    orh_30:           float = 0.0
    orl_30:           float = 0.0
    or_width_15_pct:  float = 0.0
    or_width_30_pct:  float = 0.0
    position_vs_or15: str   = "UNKNOWN"
    position_vs_or30: str   = "UNKNOWN"
    breakout_15:      str   = "NONE"
    breakout_30:      str   = "NONE"
    or15_score:       float = 0.0
    or30_score:       float = 0.0
    description:      str   = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


def _extract_or_bars(df_et: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Return bars within the opening range window (inclusive of start, exclusive of end)."""
    try:
        t_start = pd.Timestamp(start).time()
        t_end   = pd.Timestamp(end).time()
        mask = (
            (df_et.index.time >= t_start) &
            (df_et.index.time <  t_end)
        )
        return df_et[mask]
    except Exception:
        return pd.DataFrame()


def _or_score(price: float, orh: float, orl: float) -> tuple[str, str, float]:
    """
    Classify price position relative to opening range and return a signal score.

    Returns (position, breakout_type, score).
    """
    if orh <= 0 or orl <= 0:
        return "UNKNOWN", "NONE", 0.0

    or_range = orh - orl
    if or_range <= 0:
        return "UNKNOWN", "NONE", 0.0

    if price > orh:
        # How far above ORH? Normalise by OR range
        excess = (price - orh) / or_range
        score  = float(np.clip(_BREAKOUT_SCORE + excess * 0.1, 0.0, 1.0))
        return "ABOVE", "BULL", score

    elif price < orl:
        excess = (orl - price) / or_range
        score  = float(np.clip(-_BREAKOUT_SCORE - excess * 0.1, -1.0, 0.0))
        return "BELOW", "BEAR", score

    else:
        # Inside OR: position within range → bias toward upper or lower half
        pct_in_range = (price - orl) / or_range   # 0 = at ORL, 1 = at ORH
        score = float(np.clip((pct_in_range - 0.5) * 0.6, -0.3, 0.3))
        return "INSIDE", "NONE", score


def compute_opening_range(df: pd.DataFrame) -> OpeningRangeResult:
    """
    Compute opening range levels and classify current price position.

    Parameters
    ----------
    df : pd.DataFrame
        Intraday 1-min OHLCV DataFrame (DatetimeIndex required).
        Should contain today's data plus prior session bars.

    Returns
    -------
    OpeningRangeResult with ORB-15 and ORB-30 levels and signal scores.
    """
    result = OpeningRangeResult()

    if df is None or df.empty:
        result.description = "No intraday data for opening range calculation."
        return result

    try:
        import pytz
        et = pytz.timezone("America/New_York")
        df_et = df.copy()
        df_et.index = pd.to_datetime(df_et.index)
        if df_et.index.tzinfo is None:
            df_et.index = df_et.index.tz_localize("UTC").tz_convert(et)
        else:
            df_et.index = df_et.index.tz_convert(et)

        today    = df_et.index[-1].date()
        today_df = df_et[df_et.index.date == today]
        if today_df.empty:
            result.description = "No bars for today — opening range unavailable."
            return result

        price = float(today_df["Close"].iloc[-1])

        # ── ORB-15 ────────────────────────────────────────────────────────────
        or15 = _extract_or_bars(today_df, _OR15_END[0], _OR15_END[1])
        if not or15.empty:
            result.orh_15 = round(float(or15["High"].max()), 4)
            result.orl_15 = round(float(or15["Low"].min()),  4)
            mid_15 = (result.orh_15 + result.orl_15) / 2
            result.or_width_15_pct = round(
                (result.orh_15 - result.orl_15) / mid_15 * 100, 3
            ) if mid_15 > 0 else 0.0
            pos15, bo15, s15 = _or_score(price, result.orh_15, result.orl_15)
            result.position_vs_or15 = pos15
            result.breakout_15      = bo15
            result.or15_score       = round(s15, 3)

        # ── ORB-30 ────────────────────────────────────────────────────────────
        or30 = _extract_or_bars(today_df, _OR30_END[0], _OR30_END[1])
        if not or30.empty:
            result.orh_30 = round(float(or30["High"].max()), 4)
            result.orl_30 = round(float(or30["Low"].min()),  4)
            mid_30 = (result.orh_30 + result.orl_30) / 2
            result.or_width_30_pct = round(
                (result.orh_30 - result.orl_30) / mid_30 * 100, 3
            ) if mid_30 > 0 else 0.0
            pos30, bo30, s30 = _or_score(price, result.orh_30, result.orl_30)
            result.position_vs_or30 = pos30
            result.breakout_30      = bo30
            result.or30_score       = round(s30, 3)

        # ── Human-readable description ────────────────────────────────────────
        desc_parts = []
        if result.orh_15 > 0:
            desc_parts.append(
                f"ORB-15: ${result.orl_15:.2f}–${result.orh_15:.2f} "
                f"({result.or_width_15_pct:.2f}%) — "
                f"price {result.position_vs_or15.lower()}"
                + (f" [{result.breakout_15}]" if result.breakout_15 != "NONE" else "")
            )
        if result.orh_30 > 0:
            desc_parts.append(
                f"ORB-30: ${result.orl_30:.2f}–${result.orh_30:.2f} "
                f"({result.or_width_30_pct:.2f}%) — "
                f"price {result.position_vs_or30.lower()}"
                + (f" [{result.breakout_30}]" if result.breakout_30 != "NONE" else "")
            )
        if not desc_parts:
            desc_parts = ["Opening range not yet established (pre-9:45 ET)"]

        result.description = " | ".join(desc_parts)

    except Exception as exc:
        logger.debug("compute_opening_range error: %s", exc)
        result.description = "Opening range calculation error."

    return result
