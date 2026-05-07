"""
XGBoost classifier that predicts whether the next N-bar close will be
higher than the current close (label=1) or lower/flat (label=0).

Training uses the full historical 5-min dataset; inference runs on the
most recent feature row fetched during each scan cycle.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from agent.data_fetcher import fetch_historical
from agent.technical import compute_indicators

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
            use_label_encoder=False,
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

        row   = df[FEATURE_COLS].iloc[[-1]]
        row_s = self.scaler.transform(row)
        prob  = float(self.model.predict_proba(row_s)[0][1])
        return round(prob, 4)


# ── Global registry: one model per ticker ─────────────────────────────────────

_model_registry: dict[str, StockMLModel] = {}


def get_or_train(ticker: str) -> StockMLModel:
    if ticker not in _model_registry:
        m = StockMLModel(ticker)
        m.train()
        _model_registry[ticker] = m
    return _model_registry[ticker]


def retrain_all(tickers: list[str]) -> None:
    for t in tickers:
        try:
            m = _model_registry.get(t, StockMLModel(t))
            m.train()
            _model_registry[t] = m
        except Exception as e:
            logger.warning(f"[{t}] retrain failed: {e}")


def predict(ticker: str, df: pd.DataFrame) -> float:
    """Convenience wrapper: get/train model then return up-probability."""
    m = get_or_train(ticker)
    return m.predict_proba(df)
