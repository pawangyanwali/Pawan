"""
Universal deep direction model — BiLSTM + Self-Attention.

Why universal (trained across all tickers vs. per-ticker)?
  - 80 tickers × ~1,600 sequences each = ~128,000 training samples
  - Per-ticker XGBoost only sees ~1,600 samples — a neural net needs more
  - Cross-ticker training lets the model learn market-wide patterns
  - NVDA momentum patterns look similar to AMD; the model learns that abstraction

Architecture
------------
  Input:   (batch, SEQ_LEN=20, features=23)  — 20 × 15min = 5 hours context
  Linear:  23 → 64 with LayerNorm
  BiLSTM:  2 layers, hidden=64 per direction (→ 128 total), dropout=0.3
  Attn:    4-head scaled dot-product self-attention over LSTM sequence
  MLP:     128 → 64 → GELU → Dropout(0.3) → 32 → GELU → 1 → Sigmoid

Data
----
  15-min bars: 5000 bars = ~6 months (78 × 5min × 3 = 234 × 15min → 5000/234 ≈ 21 trading days...
  Actually: 26 bars/day × 5 days × 26 weeks = 3380 bars → well under 5000 limit)
  Fetched once during retrain (TTL=3600s).

Label
-----
  1  if price rises  ≥ MIN_MOVE_PCT in the next LOOKAHEAD_BARS bars
  0  if price falls  ≥ MIN_MOVE_PCT in the next LOOKAHEAD_BARS bars
  (samples where price moves less than threshold are kept — they're labeled 0/1
   based on whether Close[t+LOOKAHEAD] > Close[t])

Backtest integration
--------------------
  Resolved live_backtest signals are used as weighted training samples:
  WIN outcomes get 2× weight, LOSS outcomes get 1× weight.
  This biases the deep model toward setups that actually worked.

Persistence
-----------
  data/models/deep_direction.pt  — state dict
  data/models/deep_scaler.pkl    — feature scaler (StandardScaler)
"""
from __future__ import annotations

import logging
import pickle
import threading
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEQ_LEN        = 20      # bars of context (20 × 15min = 5 hours)
LOOKAHEAD_BARS = 4       # predict 4 × 15min = 1 hour ahead
MIN_MOVE_PCT   = 0.002   # 0.2% move to label +1 (direction)
HIDDEN_SIZE    = 64      # BiLSTM hidden units per direction (128 total)
N_LAYERS       = 2       # LSTM stacking depth
N_HEADS        = 4       # attention heads
DROPOUT        = 0.3
EPOCHS         = 10
BATCH_SIZE     = 256
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
MIN_TRAIN_SAMPLES = 500  # skip training if fewer sequences available

_MODEL_DIR  = Path(__file__).parent.parent / "data" / "models"
_MODEL_PATH = _MODEL_DIR / "deep_direction.pt"
_SCALER_PATH= _MODEL_DIR / "deep_scaler.pkl"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)

_lock    = threading.Lock()
_trained = False

_training_history: list[dict] = []   # [{epoch, total_epochs, loss, ts, tickers}]
_is_training_now:  bool       = False
_MAX_HISTORY       = 500

# ── Feature columns (must match ml_model.FEATURE_COLS) ───────────────────────
FEATURE_COLS = [
    "rsi_14", "rsi_7", "macd", "macd_signal", "macd_hist",
    "bb_pct", "bb_width", "stoch_k", "stoch_d", "cci_20", "mfi_14",
    "ema_cross", "vol_ratio", "atr_14", "obv",
    "ret_1", "ret_3", "ret_5",
    "ret_10", "time_sin", "time_cos", "price_range_pos", "vol_trend",
]
N_FEATURES = len(FEATURE_COLS)   # 23


# ── PyTorch model definition ──────────────────────────────────────────────────

def _build_model():
    """Build the BiLSTM + Attention network. Returns None if torch unavailable."""
    try:
        import torch
        import torch.nn as nn

        class _SelfAttn(nn.Module):
            def __init__(self, d: int, heads: int):
                super().__init__()
                self.attn = nn.MultiheadAttention(d, heads, dropout=DROPOUT, batch_first=True)
                self.norm = nn.LayerNorm(d)

            def forward(self, x):
                out, _ = self.attn(x, x, x)
                return self.norm(x + out)   # residual

        class DeepDirectionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Sequential(
                    nn.Linear(N_FEATURES, HIDDEN_SIZE),
                    nn.LayerNorm(HIDDEN_SIZE),
                    nn.GELU(),
                )
                self.lstm = nn.LSTM(
                    input_size=HIDDEN_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    num_layers=N_LAYERS,
                    dropout=DROPOUT if N_LAYERS > 1 else 0.0,
                    batch_first=True,
                    bidirectional=True,
                )
                self.attn = _SelfAttn(HIDDEN_SIZE * 2, N_HEADS)
                self.head = nn.Sequential(
                    nn.Linear(HIDDEN_SIZE * 2, 64),
                    nn.GELU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(64, 32),
                    nn.GELU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                x = self.input_proj(x)
                x, _ = self.lstm(x)
                x = self.attn(x)
                x = x[:, -1, :]   # use final time-step representation
                return self.head(x).squeeze(-1)

        return DeepDirectionModel()
    except ImportError:
        return None


# ── Scaler (StandardScaler) ───────────────────────────────────────────────────

def _load_scaler():
    try:
        if _SCALER_PATH.exists():
            with open(_SCALER_PATH, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None


def _save_scaler(scaler) -> None:
    try:
        with open(_SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
    except Exception as e:
        logger.debug(f"[DeepModel] scaler save failed: {e}")


# ── Model singleton ───────────────────────────────────────────────────────────

_model  = None
_scaler = _load_scaler()


def _get_model():
    global _model, _trained
    if _model is not None:
        return _model
    _model = _build_model()
    if _model is None:
        return None
    try:
        import torch
        if _MODEL_PATH.exists():
            _model.load_state_dict(torch.load(_MODEL_PATH, map_location="cpu", weights_only=True))
            _model.eval()
            _trained = True
            logger.info("[DeepModel] Loaded from disk.")
    except Exception as e:
        logger.debug(f"[DeepModel] Load failed (will retrain): {e}")
    return _model


# ── Feature preparation ───────────────────────────────────────────────────────

def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators + live features on raw OHLCV. Returns clean DataFrame."""
    try:
        from agent.technical import compute_indicators
        from agent.ml_model import add_live_features
        df = compute_indicators(df.copy())
        df = add_live_features(df)
        df = df.dropna(subset=FEATURE_COLS)
        return df
    except Exception as e:
        logger.debug(f"[DeepModel] prepare_df error: {e}")
        return pd.DataFrame()


def _make_sequences(
    df: pd.DataFrame,
    sample_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide a window over df and produce (X, y, weights) arrays.
    X shape: (N, SEQ_LEN, N_FEATURES)
    y shape: (N,)
    weights: (N,)
    """
    X_list, y_list, w_list = [], [], []
    feat = df[FEATURE_COLS].values.astype(np.float32)
    close = df["Close"].values

    for i in range(SEQ_LEN, len(feat) - LOOKAHEAD_BARS):
        seq   = feat[i - SEQ_LEN: i]          # (SEQ_LEN, N_FEATURES)
        c_now = close[i - 1]
        c_fut = close[i - 1 + LOOKAHEAD_BARS]
        if c_now <= 0:
            continue
        move  = (c_fut - c_now) / c_now
        label = 1 if move > 0 else 0           # direction: up or down
        X_list.append(seq)
        y_list.append(label)
        w_list.append(sample_weight)

    if not X_list:
        return np.empty((0, SEQ_LEN, N_FEATURES)), np.empty(0), np.empty(0)
    return np.array(X_list), np.array(y_list, dtype=np.float32), np.array(w_list, dtype=np.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def retrain_deep_all(ticker_dfs_15m: dict[str, pd.DataFrame]) -> bool:
    """
    Train the universal deep model across all provided 15-min DataFrames.

    Parameters
    ----------
    ticker_dfs_15m : dict mapping ticker → 15-min OHLCV DataFrame

    Returns True on success.
    """
    global _model, _scaler, _trained
    global _is_training_now, _training_history
    _is_training_now = True

    try:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("[DeepModel] torch/sklearn not available — skipping deep training")
            return False

        logger.info(f"[DeepModel] Building training set from {len(ticker_dfs_15m)} tickers…")

        # ── 1. Build training sequences ───────────────────────────────────────────
        all_X, all_y, all_w = [], [], []

        # Incorporate live backtest outcomes as weighted samples
        bt_weights: dict[str, float] = {}   # ticker → weight multiplier
        try:
            from agent.live_backtest import get_outcomes_for_ml
            outcomes = get_outcomes_for_ml(min_count=1)
            if outcomes is not None and not outcomes.empty:
                for _, row in outcomes.iterrows():
                    t = str(row.get("ticker", ""))
                    won = int(row.get("won", 0))
                    bt_weights[t] = 2.0 if won else 1.0   # WIN → 2× weight
        except Exception:
            pass

        for ticker, df_raw in ticker_dfs_15m.items():
            if df_raw is None or len(df_raw) < SEQ_LEN + LOOKAHEAD_BARS + 20:
                continue
            df = _prepare_df(df_raw)
            if df.empty:
                continue
            w = bt_weights.get(ticker, 1.0)
            X, y, weights = _make_sequences(df, sample_weight=w)
            if len(X) < 10:
                continue
            all_X.append(X)
            all_y.append(y)
            all_w.append(weights)

        if not all_X:
            logger.warning("[DeepModel] No training sequences — skipping")
            return False

        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)
        w_all = np.concatenate(all_w, axis=0)

        if len(X_all) < MIN_TRAIN_SAMPLES:
            logger.warning(f"[DeepModel] Only {len(X_all)} sequences — need {MIN_TRAIN_SAMPLES}")
            return False

        logger.info(f"[DeepModel] Training on {len(X_all):,} sequences from {len(ticker_dfs_15m)} tickers")

        # ── 2. Scale features ─────────────────────────────────────────────────────
        flat = X_all.reshape(-1, N_FEATURES)
        scaler = StandardScaler()
        scaler.fit(flat)
        X_scaled = scaler.transform(flat).reshape(X_all.shape).astype(np.float32)
        _save_scaler(scaler)

        # ── 3. Build tensors + dataset ────────────────────────────────────────────
        X_t = torch.from_numpy(X_scaled)
        y_t = torch.from_numpy(y_all)
        w_t = torch.from_numpy(w_all)

        dataset = TensorDataset(X_t, y_t, w_t)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        # ── 4. Build or reset model ───────────────────────────────────────────────
        model = _build_model()
        if model is None:
            return False

        # ── 5. Class imbalance: pos_weight ────────────────────────────────────────
        n_pos = float(y_all.sum())
        n_neg = float(len(y_all) - n_pos)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])

        criterion = nn.BCELoss(reduction="none")
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        model.train()
        best_loss = float("inf")
        best_state = None

        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            n_batches  = 0
            for xb, yb, wb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                # Weighted BCE: WIN samples count twice as much
                loss = (criterion(pred, yb) * wb).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1
            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss  = avg_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info(f"[DeepModel] Epoch {epoch+1}/{EPOCHS} — loss={avg_loss:.4f}")
            import time as _time
            _training_history.append({
                "epoch":        epoch + 1,
                "total_epochs": EPOCHS,
                "loss":         round(avg_loss, 6),
                "ts":           _time.time(),
                "tickers":      len(ticker_dfs_15m),
            })
            if len(_training_history) > _MAX_HISTORY:
                _training_history = _training_history[-_MAX_HISTORY:]

        # ── 6. Save best checkpoint ───────────────────────────────────────────────
        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        torch.save(model.state_dict(), _MODEL_PATH)

        with _lock:
            _model   = model
            _scaler  = scaler
            _trained = True

        # Accuracy on training set (quick sanity check)
        with torch.no_grad():
            preds = model(X_t).numpy()
        acc = float(((preds >= 0.5).astype(int) == y_all.astype(int)).mean())
        logger.info(
            f"[DeepModel] Training complete — acc={acc:.3f}  "
            f"samples={len(X_all):,}  best_loss={best_loss:.4f}"
        )
        return True
    finally:
        _is_training_now = False


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_deep(ticker: str, df_15m: pd.DataFrame) -> float:
    """
    Return P(price up in next LOOKAHEAD_BARS × 15min) using the deep model.
    Returns 0.5 (neutral) when model is untrained or data insufficient.
    """
    global _trained

    if not _trained:
        return 0.5

    model = _get_model()
    if model is None or _scaler is None:
        return 0.5

    try:
        import torch
        df = _prepare_df(df_15m)
        if len(df) < SEQ_LEN:
            return 0.5

        seq = df[FEATURE_COLS].values[-SEQ_LEN:].astype(np.float32)
        seq_flat = seq.reshape(-1, N_FEATURES)
        seq_scaled = _scaler.transform(seq_flat).reshape(1, SEQ_LEN, N_FEATURES)

        x = torch.from_numpy(seq_scaled.astype(np.float32))
        model.eval()
        with torch.no_grad():
            prob = float(model(x).item())
        return round(float(np.clip(prob, 0.0, 1.0)), 4)
    except Exception as e:
        logger.debug(f"[DeepModel] predict error for {ticker}: {e}")
        return 0.5


def is_trained() -> bool:
    return _trained


def get_model_info() -> dict:
    return {
        "trained":      _trained,
        "seq_len":      SEQ_LEN,
        "lookahead":    LOOKAHEAD_BARS,
        "n_features":   N_FEATURES,
        "hidden_size":  HIDDEN_SIZE,
        "n_layers":     N_LAYERS,
        "n_heads":      N_HEADS,
        "epochs":       EPOCHS,
        "model_path":   str(_MODEL_PATH),
    }


def get_training_history() -> list[dict]:
    return list(_training_history)


def is_training_active() -> bool:
    return _is_training_now
