"""
LSTM-based market regime detector.

Trains a small LSTM on SPY/QQQ bar sequences to classify regime as:
  BULL_TREND, BEAR_TREND, CHOPPY, NEUTRAL

Advantage over rule-based detector: learns the SEQUENCE pattern of
regime transitions, not just instantaneous thresholds. A slowly grinding
bull market looks different from a gap-up spike even if the % change is identical.

Architecture:
  Input:  (batch, seq_len=20, features=6) — SPY/QQQ: ret, vol_ratio, atr_ratio, range_pct
  LSTM:   hidden=32, layers=2, dropout=0.2
  Output: 4-class softmax (BULL, BEAR, CHOPPY, NEUTRAL)

Falls back to rule-based detector if torch is not installed or model untrained.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "lstm_regime.pkl"
_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

CLASSES      = ["NEUTRAL", "BULL_TREND", "BEAR_TREND", "CHOPPY"]
SEQ_LEN      = 20   # number of 5-min bars to look back
N_FEATURES   = 6    # features per bar
HIDDEN_SIZE  = 32
N_LAYERS     = 2
MIN_TRAIN    = 200  # minimum bars to train on


def _build_features(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Build (T, N_FEATURES) feature array from OHLCV dataframe."""
    if df is None or len(df) < SEQ_LEN + 5:
        return None
    try:
        close  = df["Close"].values.astype(float)
        volume = df["Volume"].values.astype(float)
        high   = df["High"].values.astype(float)
        low    = df["Low"].values.astype(float)

        ret      = np.zeros(len(close))
        ret[1:]  = (close[1:] - close[:-1]) / (close[:-1] + 1e-9)

        vol_avg  = pd.Series(volume).rolling(20, min_periods=1).mean().values
        vol_rat  = volume / (vol_avg + 1e-9)

        tr       = np.maximum(high[1:] - low[1:],
                   np.maximum(np.abs(high[1:] - close[:-1]),
                              np.abs(low[1:]  - close[:-1])))
        tr       = np.concatenate([[tr[0]], tr])
        atr      = pd.Series(tr).rolling(14, min_periods=1).mean().values
        atr_rat  = atr / (close + 1e-9)

        range_p  = (high - low) / (close + 1e-9)
        cum_ret  = np.zeros(len(close))
        cum_ret[1:] = (close[1:] - close[0]) / (close[0] + 1e-9)

        above_vwap = np.zeros(len(close))
        vwap_num = np.cumsum((high + low + close) / 3 * volume)
        vwap_den = np.cumsum(volume)
        vwap     = vwap_num / (vwap_den + 1e-9)
        above_vwap = (close > vwap).astype(float)

        feat = np.stack([ret, vol_rat, atr_rat, range_p, cum_ret, above_vwap], axis=1)
        feat = np.clip(feat, -3.0, 3.0)
        return feat.astype(np.float32)
    except Exception as e:
        logger.debug(f"LSTM features error: {e}")
        return None


class LSTMRegimeDetector:
    """
    Wraps the PyTorch LSTM model with train/predict interface.
    Falls back gracefully if torch is unavailable.
    """

    def __init__(self):
        self.model   = None
        self.trained = False
        self._torch_ok = self._check_torch()
        if self._torch_ok:
            self._load()

    def _check_torch(self) -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            logger.info("[LSTMRegime] torch not installed — using rule-based fallback")
            return False

    def _build_model(self):
        try:
            import torch
            import torch.nn as nn

            class _LSTM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=N_FEATURES, hidden_size=HIDDEN_SIZE,
                        num_layers=N_LAYERS, dropout=0.2, batch_first=True
                    )
                    self.fc = nn.Linear(HIDDEN_SIZE, len(CLASSES))

                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])

            return _LSTM()
        except Exception as e:
            logger.debug(f"LSTM build error: {e}")
            return None

    def _save(self) -> None:
        try:
            import torch
            if self.model is not None:
                torch.save(self.model.state_dict(), str(_MODEL_PATH))
        except Exception as e:
            logger.debug(f"LSTM save error: {e}")

    def _load(self) -> None:
        try:
            if not _MODEL_PATH.exists():
                return
            import torch
            model = self._build_model()
            if model is None:
                return
            model.load_state_dict(torch.load(str(_MODEL_PATH), map_location="cpu"))
            model.eval()
            self.model   = model
            self.trained = True
            logger.info("[LSTMRegime] Model loaded from disk")
        except Exception as e:
            logger.debug(f"LSTM load error: {e}")

    def train(self, df_spy: pd.DataFrame, df_qqq: Optional[pd.DataFrame] = None,
              epochs: int = 30) -> bool:
        """
        Train on SPY (and optionally QQQ) intraday bars.
        Labels are auto-generated from return thresholds.
        """
        if not self._torch_ok:
            return False
        feat = _build_features(df_spy)
        if feat is None or len(feat) < MIN_TRAIN + SEQ_LEN:
            return False

        try:
            import torch
            import torch.nn as nn

            # Auto-label from 12-bar forward returns (60 min ahead)
            close  = df_spy["Close"].values.astype(float)
            fwd_ret = np.zeros(len(close))
            fwd_ret[:-12] = (close[12:] - close[:-12]) / (close[:-12] + 1e-9)
            atr    = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(14, min_periods=1).mean().values
            atr_n  = atr / (close + 1e-9)
            labels = np.full(len(close), 0, dtype=int)   # NEUTRAL
            labels[fwd_ret >  atr_n * 0.5] = 1   # BULL
            labels[fwd_ret < -atr_n * 0.5] = 2   # BEAR
            high   = df_spy["High"].values.astype(float)
            low    = df_spy["Low"].values.astype(float)
            range_p = (high - low) / (close + 1e-9)
            labels[range_p > np.percentile(range_p, 85)] = 3  # CHOPPY

            # Build sequences
            Xs, ys = [], []
            for i in range(SEQ_LEN, len(feat) - 12):
                Xs.append(feat[i - SEQ_LEN:i])
                ys.append(labels[i])

            if len(Xs) < 100:
                return False

            X = torch.tensor(np.array(Xs), dtype=torch.float32)
            y = torch.tensor(np.array(ys), dtype=torch.long)

            model     = self._build_model()
            if model is None:
                return False
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            model.train()
            dataset = torch.utils.data.TensorDataset(X, y)
            loader  = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

            for epoch in range(epochs):
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                preds = model(X).argmax(dim=1).numpy()
            acc = (preds == y.numpy()).mean()
            logger.info(f"[LSTMRegime] Trained | acc={acc:.3f} | samples={len(Xs)}")

            self.model   = model
            self.trained = True
            self._save()
            return True

        except Exception as e:
            logger.warning(f"[LSTMRegime] Training failed: {e}")
            return False

    def predict(self, df: pd.DataFrame) -> tuple[str, float]:
        """
        Returns (regime_label, confidence_0_to_1).
        Falls back to ("NEUTRAL", 0.0) if model not trained.
        """
        if not self.trained or self.model is None:
            return "NEUTRAL", 0.0

        feat = _build_features(df)
        if feat is None or len(feat) < SEQ_LEN:
            return "NEUTRAL", 0.0

        try:
            import torch
            seq   = torch.tensor(feat[-SEQ_LEN:][np.newaxis], dtype=torch.float32)
            with torch.no_grad():
                logits = self.model(seq)[0]
                probs  = torch.softmax(logits, dim=0).numpy()
            cls_idx = int(probs.argmax())
            return CLASSES[cls_idx], round(float(probs[cls_idx]), 4)
        except Exception as e:
            logger.debug(f"[LSTMRegime] predict error: {e}")
            return "NEUTRAL", 0.0


# ── Singleton ─────────────────────────────────────────────────────────────────

_detector = LSTMRegimeDetector()


def get_lstm_regime(df_spy: pd.DataFrame) -> tuple[str, float]:
    """Returns (regime, confidence). Falls back to NEUTRAL if untrained."""
    return _detector.predict(df_spy)


def train_lstm_regime(df_spy: pd.DataFrame) -> bool:
    """Train the LSTM regime detector on SPY data."""
    return _detector.train(df_spy)
