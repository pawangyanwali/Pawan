"""
XGBoost classifier that predicts whether the next N-bar close will be
higher than the current close (label=1) or lower/flat (label=0).

Training uses the full historical 5-min dataset; inference runs on the
most recent feature row fetched during each scan cycle.
"""

import logging
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from agent.data_fetcher import fetch_historical
from agent.technical import compute_indicators
from agent.reversal import compute_reversal_features, REVERSAL_FEATURE_COLS

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "rsi_14", "rsi_7", "macd", "macd_signal", "macd_hist",
    "bb_pct", "bb_width", "stoch_k", "stoch_d", "cci_20", "mfi_14",
    "ema_cross", "vol_ratio", "atr_14", "obv",
    "ret_1", "ret_3", "ret_5",
]

LOOKAHEAD_BARS = 3   # predict direction 3×5min = 15 min ahead


class StockMLModel:
    def __init__(self, ticker: str):
        self.ticker  = ticker
        self.model   = None
        self.scaler  = StandardScaler()
        self.trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self) -> bool:
        df = fetch_historical(self.ticker)
        if df is None or len(df) < 100:
            return False

        df = compute_indicators(df)
        df = df.dropna(subset=FEATURE_COLS)

        # Label: 1 if close N bars ahead > current close
        df["label"] = (df["Close"].shift(-LOOKAHEAD_BARS) > df["Close"]).astype(int)
        df.dropna(inplace=True)

        X = df[FEATURE_COLS].values
        y = df["label"].values

        if len(X) < 60:
            return False

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        base = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            verbosity=0,
        )
        self.model = CalibratedClassifierCV(base, cv=3, method="isotonic")
        self.model.fit(X_train_s, y_train)
        self.trained = True

        acc = self.model.score(X_test_s, y_test)
        logger.info(f"[{self.ticker}] ML model trained | acc={acc:.3f} | samples={len(X_train)}")
        return True

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> float:
        """
        Return probability [0,1] that the price goes UP in the next N bars.
        Returns 0.5 (neutral) when the model is not trained or features are missing.
        """
        if not self.trained or self.model is None:
            return 0.5

        df = compute_indicators(df.copy())
        df = df.dropna(subset=FEATURE_COLS)
        if df.empty:
            return 0.5

        row   = df[FEATURE_COLS].iloc[[-1]].values
        row_s = self.scaler.transform(row)
        prob  = float(self.model.predict_proba(row_s)[0][1])
        return round(prob, 4)


# ── Global registry: one model per ticker ─────────────────────────────────────

_model_registry: dict[str, StockMLModel] = {}


def get_or_create(ticker: str) -> StockMLModel:
    """Return existing model (untrained is fine) — training happens lazily in retrain_all."""
    if ticker not in _model_registry:
        _model_registry[ticker] = StockMLModel(ticker)
    return _model_registry[ticker]


def retrain_all(tickers: list, delay: float = 1.2, daily_data: dict = None) -> None:
    """Train/retrain models one ticker at a time with a delay to avoid rate limits.

    Parameters
    ----------
    tickers    : list of ticker symbols to retrain.
    delay      : seconds to sleep between tickers to avoid API rate limits.
    daily_data : optional mapping of ticker → daily OHLCV DataFrame.  When
                 provided, each ticker's DailyMLModel is also retrained from
                 the supplied DataFrame (no extra API call required).
    """
    for t in tickers:
        try:
            m = _model_registry.get(t, StockMLModel(t))
            m.train()
            _model_registry[t] = m
        except Exception as e:
            logger.warning(f"[{t}] scalp retrain failed: {e}")

        if daily_data and t in daily_data:
            try:
                dm = get_or_create_daily(t)
                dm.train_from_df(daily_data[t])
            except Exception as e:
                logger.warning(f"[{t}] daily retrain failed: {e}")

        # Reversal model — reuses same 5M historical fetch (cached)
        try:
            rm = get_or_create_reversal(t)
            rm.train()
        except Exception as e:
            logger.warning(f"[{t}] reversal retrain failed: {e}")

        time.sleep(delay)


def predict(ticker: str, df: pd.DataFrame) -> float:
    """Return up-probability; uses 0.5 (neutral) if model not yet trained."""
    m = get_or_create(ticker)
    return m.predict_proba(df)


# ── Daily ML Model ────────────────────────────────────────────────────────────

class DailyMLModel:
    """
    XGBoost classifier trained on daily OHLCV bars.

    Predicts whether the NEXT DAY's close will be higher than today's close.
    Training data is supplied directly (no API call) via train_from_df().
    """

    def __init__(self, ticker: str):
        self.ticker  = ticker
        self.model   = None
        self.scaler  = StandardScaler()
        self.trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train_from_df(self, df_daily: pd.DataFrame) -> bool:
        """Train on the supplied daily OHLCV DataFrame.  Returns True on success."""
        if df_daily is None or len(df_daily) < 100:
            return False

        df = compute_indicators(df_daily.copy())
        df = df.dropna(subset=FEATURE_COLS)

        # Label: 1 if next-day close > today's close
        df["label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df.dropna(inplace=True)

        X = df[FEATURE_COLS].values
        y = df["label"].values

        if len(X) < 60:
            return False

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        base = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            verbosity=0,
        )
        self.model = CalibratedClassifierCV(base, cv=3, method="isotonic")
        self.model.fit(X_train_s, y_train)
        self.trained = True

        acc = self.model.score(X_test_s, y_test)
        logger.info(
            f"[{self.ticker}] DailyML trained | acc={acc:.3f} | samples={len(X_train)}"
        )
        return True

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, df_daily: pd.DataFrame) -> float:
        """
        Return probability [0, 1] that tomorrow's close will be higher.
        Returns 0.5 (neutral) when untrained or features are missing.
        """
        if not self.trained or self.model is None:
            return 0.5

        df = compute_indicators(df_daily.copy())
        df = df.dropna(subset=FEATURE_COLS)
        if df.empty:
            return 0.5

        row   = df[FEATURE_COLS].iloc[[-1]].values
        row_s = self.scaler.transform(row)
        prob  = float(self.model.predict_proba(row_s)[0][1])
        return round(prob, 4)


# ── Daily model registry ──────────────────────────────────────────────────────

_daily_model_registry: dict[str, DailyMLModel] = {}


def get_or_create_daily(ticker: str) -> DailyMLModel:
    """Return existing DailyMLModel for ticker (creates one if absent)."""
    if ticker not in _daily_model_registry:
        _daily_model_registry[ticker] = DailyMLModel(ticker)
    return _daily_model_registry[ticker]


def predict_daily(ticker: str, df_daily: pd.DataFrame) -> float:
    """Return next-day up-probability from the daily model; 0.5 if untrained."""
    dm = get_or_create_daily(ticker)
    return dm.predict_proba(df_daily)


# ── Reversal ML Model ─────────────────────────────────────────────────────────

class ReversalMLModel:
    """
    XGBoost classifier trained to detect imminent price reversals.

    Target: will price move ≥ 0.8% in the reversal direction within 5 bars?
    This is distinct from the ScalpMLModel (next-bar direction) — it looks
    for TURNING POINTS specifically, using divergence-aware features.

    Separate from ScalpMLModel so it can be trained with reversal-specific
    features (RSI divergence score, wick ratios, oscillator extremes).
    """

    # Lookahead and threshold for labelling reversals in training data
    _LOOKAHEAD  = 5     # bars ahead to check
    _MOVE_PCT   = 0.008  # 0.8% = meaningful reversal

    def __init__(self, ticker: str):
        self.ticker  = ticker
        self.model   = None
        self.scaler  = StandardScaler()
        self.trained = False

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_indicators(df.copy())
        df = compute_reversal_features(df)
        return df

    def train(self) -> bool:
        raw = fetch_historical(self.ticker)
        if raw is None or len(raw) < 150:
            return False

        df = self._prepare(raw)

        # Label: 1 if price rises ≥ 0.8% within next 5 bars (bullish reversal)
        future_high = df["Close"].rolling(self._LOOKAHEAD).max().shift(-self._LOOKAHEAD)
        df["label"] = ((future_high - df["Close"]) / df["Close"] >= self._MOVE_PCT).astype(int)
        df = df.dropna(subset=REVERSAL_FEATURE_COLS + ["label"])

        X = df[REVERSAL_FEATURE_COLS].values
        y = df["label"].values

        if len(X) < 60 or y.sum() < 10:
            return False

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        self.scaler.fit(X_train)
        X_tr = self.scaler.transform(X_train)
        X_te = self.scaler.transform(X_test)

        base = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=3,
            scale_pos_weight=float((y == 0).sum()) / max(float((y == 1).sum()), 1),
            eval_metric="logloss",
            verbosity=0,
        )
        self.model = CalibratedClassifierCV(base, cv=3, method="isotonic")
        self.model.fit(X_tr, y_train)
        self.trained = True

        acc = self.model.score(X_te, y_test)
        logger.info(
            f"[{self.ticker}] ReversalML trained | acc={acc:.3f} | "
            f"reversals={y.sum()}/{len(y)} ({y.mean()*100:.1f}%)"
        )
        return True

    def predict_proba(self, df: pd.DataFrame) -> float:
        """Return probability [0,1] of bullish reversal in next 5 bars. 0.5 if untrained."""
        if not self.trained or self.model is None:
            return 0.5
        try:
            df_feat = self._prepare(df)
            df_feat = df_feat.dropna(subset=REVERSAL_FEATURE_COLS)
            if df_feat.empty:
                return 0.5
            row   = df_feat[REVERSAL_FEATURE_COLS].iloc[[-1]].values
            row_s = self.scaler.transform(row)
            return round(float(self.model.predict_proba(row_s)[0][1]), 4)
        except Exception as e:
            logger.debug(f"[{self.ticker}] ReversalML predict error: {e}")
            return 0.5


# ── Reversal model registry ───────────────────────────────────────────────────

_reversal_model_registry: dict[str, ReversalMLModel] = {}


def get_or_create_reversal(ticker: str) -> ReversalMLModel:
    if ticker not in _reversal_model_registry:
        _reversal_model_registry[ticker] = ReversalMLModel(ticker)
    return _reversal_model_registry[ticker]


def predict_reversal(ticker: str, df: pd.DataFrame) -> float:
    """Return bullish-reversal probability from the ReversalMLModel. 0.5 if untrained."""
    return get_or_create_reversal(ticker).predict_proba(df)
