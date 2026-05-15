"""
order_flow.py — Order Flow Model (OHLCV Proxy)
===============================================
Produces a directional pressure score in [-1.0, +1.0] using only OHLCV data.

Architecture note
-----------------
The PRD calls for a Transformer reading 20 Level-2 order-book snapshots.
Because Twelve Data supplies OHLCV only, this module implements a feature-based
proxy that approximates what such a Transformer would learn from the book.
The public API (``compute_order_flow`` / ``get_signal_strength``) is designed as
a drop-in: when real Polygon.io Level-2 data becomes available, swap the feature
computation internals while keeping the same function signatures and output schema.

Proxy mapping
-------------
  L2 concept                   →  OHLCV proxy
  ─────────────────────────────────────────────────────────
  Bid-ask imbalance            →  Chaikin Money Flow (CMF)
  Aggressive buyer/seller      →  Volume-weighted bar direction
  Order-book sweep velocity    →  Price velocity vs ATR
  Block / iceberg prints       →  Volume surge (> 2× 20-bar avg)
  Stacked bids / offers        →  Consecutive directional bars w/ above-avg vol
  Divergent book vs price      →  CMF divergence vs recent high / low

Dependencies: pandas, numpy, logging (stdlib only).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NEUTRAL_RESULT: dict = {
    "score": 0.0,
    "cmf": 0.0,
    "volume_pressure": 0.0,
    "momentum_score": 0.0,
    "surge_detected": False,
    "consecutive_bars": 0,
    "divergence": "NONE",
    "interpretation": "Insufficient data — neutral stance",
    "label": "NEUTRAL",
}

_MIN_BARS = 5  # absolute minimum to compute anything meaningful


def _validate_dataframe(df: pd.DataFrame) -> Optional[str]:
    """Return an error string if *df* is unusable, else None."""
    if df is None or df.empty:
        return "DataFrame is None or empty"
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns.str.lower())
    if missing:
        return f"Missing required columns: {missing}"
    if len(df) < _MIN_BARS:
        return f"Too few rows ({len(df)} < {_MIN_BARS})"
    return None


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lower-cased column names and float dtypes."""
    df = df.copy()
    df.columns = df.columns.str.lower()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df


# ---------------------------------------------------------------------------
# Feature computations
# ---------------------------------------------------------------------------

def _compute_cmf(df: pd.DataFrame, period: int = 14) -> float:
    """
    Chaikin Money Flow over *period* bars.

    CMF = sum(MFV, period) / sum(volume, period)
    where MFV = ((2*close - high - low) / (high - low)) * volume

    A zero high-low range bar contributes zero money-flow volume (neutral).
    Result is clipped to [-1, +1].
    """
    n = min(period, len(df))
    window = df.iloc[-n:]

    hl_range = window["high"] - window["low"]
    # Avoid division by zero on doji / gap bars
    mfm = np.where(
        hl_range > 0,
        (2 * window["close"] - window["high"] - window["low"]) / hl_range,
        0.0,
    )
    mfv = mfm * window["volume"].values
    vol_sum = window["volume"].sum()

    if vol_sum == 0:
        return 0.0
    return float(np.clip(mfv.sum() / vol_sum, -1.0, 1.0))


def _compute_volume_pressure(df: pd.DataFrame, lookback: int) -> float:
    """
    Volume-weighted directional pressure.

    Each bar is classified:
      +1  if close > open  (buyers in control)
      -1  if close < open  (sellers in control)
       0  if doji (close == open)

    The classification is weighted by that bar's share of total lookback volume.
    Result is in [-1, +1].
    """
    n = min(lookback, len(df))
    window = df.iloc[-n:]

    direction = np.sign(window["close"].values - window["open"].values)  # -1, 0, +1
    vol = window["volume"].values.astype(float)
    total_vol = vol.sum()

    if total_vol == 0:
        return 0.0
    return float(np.clip((direction * vol).sum() / total_vol, -1.0, 1.0))


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range over *period* bars (returns 0 if insufficient data)."""
    n = min(period + 1, len(df))
    if n < 2:
        return 0.0
    window = df.iloc[-n:]
    prev_close = window["close"].shift(1)
    tr = pd.concat(
        [
            window["high"] - window["low"],
            (window["high"] - prev_close).abs(),
            (window["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(tr.iloc[1:].mean())  # exclude first NaN row


def _compute_momentum_score(df: pd.DataFrame, lookback: int) -> float:
    """
    Price velocity normalised by ATR.

    velocity = (close[-1] - close[-lookback]) / lookback   (per-bar change)
    momentum_score = velocity / ATR  clipped to [-1, +1]

    Interpretation: how many ATR units per bar is price moving?
    A score of +1 means price is rocketing up at >= 1 ATR/bar — extremely strong.
    """
    n = min(lookback, len(df))
    if n < 2:
        return 0.0

    close_now = float(df["close"].iloc[-1])
    close_then = float(df["close"].iloc[-n])
    velocity = (close_now - close_then) / n

    atr = _compute_atr(df, period=min(14, n))
    if atr == 0:
        return 0.0

    # Normalise: one full ATR of per-bar movement → score of ±1
    return float(np.clip(velocity / atr, -1.0, 1.0))


def _detect_volume_surge(df: pd.DataFrame, avg_period: int = 20, multiplier: float = 2.0) -> bool:
    """
    Return True if the latest bar's volume exceeds *multiplier* × the rolling
    mean over *avg_period* bars (institutional activity proxy).
    """
    if len(df) < 2:
        return False
    n = min(avg_period, len(df) - 1)
    avg_vol = float(df["volume"].iloc[-n - 1 : -1].mean())
    if avg_vol == 0:
        return False
    return float(df["volume"].iloc[-1]) >= multiplier * avg_vol


def _compute_consecutive_bars(df: pd.DataFrame, lookback: int, avg_period: int = 20) -> int:
    """
    Count the current streak of same-direction bars that also have above-average volume.

    Returns a signed integer:
      +N  →  N consecutive UP bars (close > open) with above-avg volume
      -N  →  N consecutive DOWN bars
       0  →  streak broken or no qualifying bars

    The streak counts from the most-recent bar backward.
    """
    n = min(lookback, len(df))
    if n < 2:
        return 0

    window = df.iloc[-n:]
    avg_vol = float(df["volume"].iloc[-min(avg_period, len(df)):].mean())

    directions = np.sign(window["close"].values - window["open"].values)
    volumes = window["volume"].values

    # Walk backward from latest bar
    streak = 0
    last_dir = None
    for i in range(len(window) - 1, -1, -1):
        d = int(directions[i])
        v = float(volumes[i])
        if d == 0 or v < avg_vol:
            break  # doji or below-avg volume breaks the streak
        if last_dir is None:
            last_dir = d
        if d != last_dir:
            break
        streak += 1

    return streak * (last_dir if last_dir else 0)


def _detect_divergence(
    df: pd.DataFrame,
    cmf: float,
    lookback: int,
) -> str:
    """
    Momentum divergence between price and CMF.

    BEARISH divergence: price making a new N-bar high, but CMF is declining
                        → supply absorption weakening despite price rise.
    BULLISH divergence: price making a new N-bar low, but CMF is rising
                        → demand stepping in despite price drop.
    NONE: no divergence detected.
    """
    n = min(lookback, len(df))
    if n < 4:
        return "NONE"

    window = df.iloc[-n:]
    recent_half = df.iloc[-n // 2 :]
    older_half = df.iloc[-n : -n // 2]

    price_new_high = float(window["close"].iloc[-1]) >= float(window["high"].max()) * 0.995
    price_new_low = float(window["close"].iloc[-1]) <= float(window["low"].min()) * 1.005

    if len(older_half) == 0:
        return "NONE"

    # Compare CMF of recent half vs older half as a simple trend proxy
    recent_cmf = _compute_cmf(recent_half, period=max(2, len(recent_half)))
    older_cmf = _compute_cmf(older_half, period=max(2, len(older_half)))

    if price_new_high and recent_cmf < older_cmf - 0.05:
        return "BEARISH"
    if price_new_low and recent_cmf > older_cmf + 0.05:
        return "BULLISH"
    return "NONE"


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------

def _aggregate_score(
    cmf: float,
    volume_pressure: float,
    momentum_score: float,
    surge_detected: bool,
    consecutive_bars: int,
    divergence: str,
) -> float:
    """
    Combine individual signals into a single score in [-1.0, +1.0].

    Weights are calibrated to approximate what a Level-2 Transformer would
    weight similarly across features (bid-ask imbalance ~40 %, trade direction
    ~30 %, velocity ~20 %, book depth surge ~10 %).

    A divergence penalty/bonus adjusts the final score after weighting.
    """
    # --- Weighted linear combination ---
    W_CMF = 0.40
    W_VOL_PRESSURE = 0.30
    W_MOMENTUM = 0.20
    W_CONSECUTIVE = 0.10

    # Consecutive streak normalised to [-1, +1] (cap at ±5 bars)
    consec_norm = float(np.clip(consecutive_bars / 5.0, -1.0, 1.0))

    raw = (
        W_CMF * cmf
        + W_VOL_PRESSURE * volume_pressure
        + W_MOMENTUM * momentum_score
        + W_CONSECUTIVE * consec_norm
    )

    # --- Volume surge amplifier ---
    # Surge indicates institutional print: amplify the direction we're already seeing
    if surge_detected:
        direction = np.sign(raw) if raw != 0 else 1.0
        raw = raw + direction * 0.08  # add ~8 % bias toward current direction

    # --- Divergence adjustment ---
    if divergence == "BEARISH":
        raw -= 0.10   # suppress bullish bias
    elif divergence == "BULLISH":
        raw += 0.10   # amplify bullish re-entry

    return float(np.clip(raw, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Label + interpretation
# ---------------------------------------------------------------------------

def _score_to_label(score: float) -> str:
    """Convert numeric score to categorical label per PRD thresholds."""
    if score > 0.50:
        return "STRONG_BUY"
    if score > 0.20:
        return "BUY"
    if score < -0.50:
        return "STRONG_SELL"
    if score < -0.20:
        return "SELL"
    return "NEUTRAL"


def _build_interpretation(
    label: str,
    cmf: float,
    surge_detected: bool,
    consecutive_bars: int,
    divergence: str,
) -> str:
    """Generate a concise human-readable interpretation of the order-flow state."""
    parts: list[str] = []

    if label == "STRONG_BUY":
        parts.append("Strong institutional buy pressure")
    elif label == "BUY":
        parts.append("Moderate buying pressure")
    elif label == "STRONG_SELL":
        parts.append("Strong institutional sell pressure")
    elif label == "SELL":
        parts.append("Moderate selling pressure")
    else:
        parts.append("Balanced / indecisive order flow")

    if surge_detected:
        direction = "buy" if cmf >= 0 else "sell"
        parts.append(f"volume surge detected ({direction}-side institutional print)")

    if abs(consecutive_bars) >= 3:
        side = "up" if consecutive_bars > 0 else "down"
        parts.append(f"{abs(consecutive_bars)}-bar {side}-streak with above-average volume")

    if divergence != "NONE":
        parts.append(f"{divergence.lower()} divergence (CMF vs price)")

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_order_flow(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Compute an order-flow direction score from OHLCV data.

    This function acts as an OHLCV proxy for a Level-2 order-book signal.
    When real Level-2 snapshots are available (e.g. from Polygon.io), the
    internal feature computation can be replaced while this function's
    signature and output schema remain unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns ``open``, ``high``, ``low``, ``close``,
        ``volume`` (case-insensitive).  Rows must be in chronological order
        (oldest first).  At least 5 rows are required for a non-neutral result.
    lookback : int, optional
        Number of recent bars to analyse (default 20, matching the PRD's
        "20 Level-2 snapshots" window).

    Returns
    -------
    dict
        {
            "score"           : float,   # -1.0 (strong sell) to +1.0 (strong buy)
            "cmf"             : float,   # Chaikin Money Flow [-1, +1]
            "volume_pressure" : float,   # volume-weighted bar direction [-1, +1]
            "momentum_score"  : float,   # price velocity / ATR [-1, +1]
            "surge_detected"  : bool,    # True if latest volume > 2× 20-bar avg
            "consecutive_bars": int,     # +N up-streak / -N down-streak
            "divergence"      : str,     # "BULLISH" | "BEARISH" | "NONE"
            "interpretation"  : str,     # human-readable summary
            "label"           : str,     # "STRONG_BUY" | "BUY" | "NEUTRAL" |
                                         # "SELL" | "STRONG_SELL"
        }

    Notes
    -----
    * An empty, short, or malformed DataFrame returns a neutral (score=0) result
      with a descriptive ``interpretation`` rather than raising an exception.
    * The ``lookback`` parameter is silently clamped to the available row count.
    """
    # --- Validation ---
    error = _validate_dataframe(df)
    if error:
        logger.warning("compute_order_flow: %s — returning neutral", error)
        result = dict(_NEUTRAL_RESULT)
        result["interpretation"] = f"Insufficient data ({error}) — neutral stance"
        return result

    df = _normalise_columns(df)

    # Re-validate after cleaning (rows may have dropped due to NaN)
    if len(df) < _MIN_BARS:
        result = dict(_NEUTRAL_RESULT)
        result["interpretation"] = (
            f"After cleaning, only {len(df)} usable rows — neutral stance"
        )
        return result

    lookback = max(2, min(lookback, len(df)))

    # --- Feature computation ---
    try:
        cmf = _compute_cmf(df, period=min(14, lookback))
        volume_pressure = _compute_volume_pressure(df, lookback=lookback)
        momentum_score = _compute_momentum_score(df, lookback=lookback)
        surge_detected = _detect_volume_surge(df, avg_period=min(20, len(df)))
        consecutive_bars = _compute_consecutive_bars(df, lookback=lookback)
        divergence = _detect_divergence(df, cmf=cmf, lookback=lookback)
    except Exception as exc:  # noqa: BLE001
        logger.exception("compute_order_flow: unexpected error during feature computation: %s", exc)
        result = dict(_NEUTRAL_RESULT)
        result["interpretation"] = f"Computation error ({exc}) — neutral stance"
        return result

    # --- Score aggregation ---
    score = _aggregate_score(
        cmf=cmf,
        volume_pressure=volume_pressure,
        momentum_score=momentum_score,
        surge_detected=surge_detected,
        consecutive_bars=consecutive_bars,
        divergence=divergence,
    )

    label = _score_to_label(score)
    interpretation = _build_interpretation(
        label=label,
        cmf=cmf,
        surge_detected=surge_detected,
        consecutive_bars=consecutive_bars,
        divergence=divergence,
    )

    return {
        "score": round(score, 4),
        "cmf": round(cmf, 4),
        "volume_pressure": round(volume_pressure, 4),
        "momentum_score": round(momentum_score, 4),
        "surge_detected": surge_detected,
        "consecutive_bars": int(consecutive_bars),
        "divergence": divergence,
        "interpretation": interpretation,
        "label": label,
    }


# ---------------------------------------------------------------------------
# Signal arbitration
# ---------------------------------------------------------------------------

def get_signal_strength(
    price_confidence: float,
    order_flow_score: float,
    regime: str,
) -> dict:
    """
    Combine price-model confidence, order-flow score, and market regime into
    an actionable trade signal per the PRD arbitration table.

    Parameters
    ----------
    price_confidence : float
        Confidence output of the price-direction model (0–100 scale).
        Typical threshold for acting on a signal is 72–78.
    order_flow_score : float
        Output of ``compute_order_flow``["score"] — range [-1.0, +1.0].
    regime : str
        Current market regime label.  Recognised values (case-insensitive):
        ``TRENDING_UP``, ``TRENDING``, ``TRENDING_DOWN``, ``RANGE_BOUND``,
        ``VOLATILE``, ``UNKNOWN``.
        Any unrecognised value is treated as ``UNKNOWN``.

    Returns
    -------
    dict
        {
            "strength"  : str,   # "STRONG" | "STANDARD" | "WEAK" |
                                 #  "CONFLICTED" | "BLOCKED" | "NO_SIGNAL"
            "execute"   : bool,  # whether to place the trade
            "size_mult" : float, # 1.0 (full) | 0.5 (half) | 0.0 (none)
            "reason"    : str,   # plain-English explanation
        }

    PRD Arbitration Table
    ---------------------
    STRONG BUY    : price_conf > 78 AND of_score > +0.50
                    AND regime in (TRENDING_UP, TRENDING)
    STANDARD BUY  : price_conf > 72 AND of_score > +0.20
                    AND regime not RANGE_BOUND
    WEAK BUY      : price_conf > 72 AND of_score in [-0.20, +0.20]
                    AND regime == TRENDING_UP
    CONFLICTED    : price_conf > 72 AND of_score < -0.20  → block trade
    NO SIGNAL     : price_conf <= 72
    BLOCKED_REGIME: price_conf > 72 AND regime == RANGE_BOUND

    Notes
    -----
    * ``size_mult`` of 0.0 means the trade is blocked regardless of ``execute``.
    * Rule evaluation is top-down; the first matching rule wins.
    """
    regime_upper = regime.strip().upper()

    # Normalise regime strings to known buckets
    trending_regimes = {"TRENDING_UP", "TRENDING"}
    blocked_regimes = {"RANGE_BOUND"}

    # ------------------------------------------------------------------ #
    # Rule 1 — NO SIGNAL: confidence too low
    # ------------------------------------------------------------------ #
    if price_confidence <= 72:
        return {
            "strength": "NO_SIGNAL",
            "execute": False,
            "size_mult": 0.0,
            "reason": (
                f"Price model confidence {price_confidence:.1f} is below the 72-point "
                "threshold — no trade."
            ),
        }

    # ------------------------------------------------------------------ #
    # Rule 2 — BLOCKED_REGIME: range-bound market, no directional edge
    # ------------------------------------------------------------------ #
    if regime_upper in blocked_regimes:
        return {
            "strength": "BLOCKED",
            "execute": False,
            "size_mult": 0.0,
            "reason": (
                f"Market regime is {regime} — range-bound conditions remove "
                "directional edge; trade blocked."
            ),
        }

    # ------------------------------------------------------------------ #
    # Rule 3 — CONFLICTED: price model bullish but order flow bearish
    # ------------------------------------------------------------------ #
    if order_flow_score < -0.20:
        return {
            "strength": "CONFLICTED",
            "execute": False,
            "size_mult": 0.0,
            "reason": (
                f"Order-flow score {order_flow_score:+.3f} signals selling pressure "
                f"while price model confidence is {price_confidence:.1f} — "
                "conflicted signals; trade blocked."
            ),
        }

    # ------------------------------------------------------------------ #
    # Rule 4 — STRONG BUY
    # ------------------------------------------------------------------ #
    if (
        price_confidence > 78
        and order_flow_score > 0.50
        and regime_upper in trending_regimes
    ):
        return {
            "strength": "STRONG",
            "execute": True,
            "size_mult": 1.0,
            "reason": (
                f"Strong conviction: price confidence {price_confidence:.1f} > 78, "
                f"order-flow score {order_flow_score:+.3f} > +0.50, "
                f"regime = {regime} — full position size."
            ),
        }

    # ------------------------------------------------------------------ #
    # Rule 5 — STANDARD BUY
    # ------------------------------------------------------------------ #
    if price_confidence > 72 and order_flow_score > 0.20:
        return {
            "strength": "STANDARD",
            "execute": True,
            "size_mult": 1.0,
            "reason": (
                f"Standard signal: price confidence {price_confidence:.1f} > 72, "
                f"order-flow score {order_flow_score:+.3f} > +0.20 — "
                "standard position size."
            ),
        }

    # ------------------------------------------------------------------ #
    # Rule 6 — WEAK BUY (neutral order flow but trending-up regime)
    # ------------------------------------------------------------------ #
    if (
        price_confidence > 72
        and -0.20 <= order_flow_score <= 0.20
        and regime_upper == "TRENDING_UP"
    ):
        return {
            "strength": "WEAK",
            "execute": True,
            "size_mult": 0.5,
            "reason": (
                f"Weak signal: price confidence {price_confidence:.1f} > 72 and "
                f"regime is {regime}, but order-flow score {order_flow_score:+.3f} "
                "is neutral — half position size."
            ),
        }

    # ------------------------------------------------------------------ #
    # Fallback — no rule matched (e.g. neutral OF score in non-trending regime)
    # ------------------------------------------------------------------ #
    return {
        "strength": "NO_SIGNAL",
        "execute": False,
        "size_mult": 0.0,
        "reason": (
            f"No qualifying signal: price confidence {price_confidence:.1f}, "
            f"order-flow score {order_flow_score:+.3f}, regime = {regime}."
        ),
    }
