"""
Cluster-aware deep direction model — BiLSTM + Self-Attention + Ticker Embedding.

Why three cluster models?
  - Cluster A (mega-cap liquid):      learns index-correlated, low-volatility patterns
  - Cluster B (growth/SaaS):          learns earnings-driven, sector-correlated patterns
  - Cluster C (high-vol/momentum):    learns retail-driven, high-beta patterns

Architecture (per-cluster model)
---------------------------------
  Input:   (batch, SEQ_LEN=20, features=32)   — 20 × 15min = 5 hours context
  + Ticker embedding: (batch, EMB_DIM=16) broadcast across time steps
  Linear:  32+16=48 → 64 with LayerNorm
  BiLSTM:  2 layers, hidden=64 per direction (→ 128 total), dropout=0.3
  Attn:    4-head scaled dot-product self-attention over LSTM sequence
  MLP:     128 → 64 → GELU → Dropout(0.3) → 32 → GELU → 1 → Sigmoid

Data
----
  15-min bars: 5000 bars = ~6 months
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

Persistence (per cluster)
--------------------------
  data/models/deep_cluster_a.pt   / deep_scaler_a.pkl
  data/models/deep_cluster_b.pt   / deep_scaler_b.pkl
  data/models/deep_cluster_c.pt   / deep_scaler_c.pkl
"""
from __future__ import annotations

import gc
import logging
import pickle
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Optional

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
EMB_DIM        = 16      # ticker embedding dimension
EPOCHS            = 10   # full-train epochs (first time only)
FINE_TUNE_EPOCHS  = 3    # subsequent runs: only fine-tune on new data
BATCH_SIZE        = 256  # larger batches = fewer Python steps = faster CPU training
LR                = 1e-3
FINE_TUNE_LR      = 2e-4  # lower LR for fine-tuning to avoid forgetting
WEIGHT_DECAY      = 1e-4
MIN_TRAIN_SAMPLES = 500   # skip training if fewer sequences available

_MODEL_DIR  = Path(__file__).parent.parent / "data" / "models"
# Legacy path — kept for backward compat (is_trained check, get_model_info)
_MODEL_PATH = _MODEL_DIR / "deep_direction.pt"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Cluster configs ───────────────────────────────────────────────────────────
_CLUSTER_CONFIGS: dict[str, dict] = {
    "A": {
        "path":   _MODEL_DIR / "deep_cluster_a.pt",
        "scaler": _MODEL_DIR / "deep_scaler_a.pkl",
    },
    "B": {
        "path":   _MODEL_DIR / "deep_cluster_b.pt",
        "scaler": _MODEL_DIR / "deep_scaler_b.pkl",
    },
    "C": {
        "path":   _MODEL_DIR / "deep_cluster_c.pt",
        "scaler": _MODEL_DIR / "deep_scaler_c.pkl",
    },
}

# Per-cluster singletons
_cluster_models:  dict[str, Any]  = {"A": None, "B": None, "C": None}
_cluster_scalers: dict[str, Any]  = {"A": None, "B": None, "C": None}
_cluster_trained: dict[str, bool] = {"A": False, "B": False, "C": False}

_lock    = threading.Lock()
# Legacy _trained flag — kept for backward compat
_trained = False

_training_history: list[dict] = []   # [{epoch, total_epochs, loss, ts, tickers, cluster}]
_is_training_now:  bool       = False
_MAX_HISTORY       = 500

# ── Ticker → cluster-local index mapping ──────────────────────────────────────
# Built at module load time from config. Index 0 is reserved (padding_idx).

try:
    from config import (
        CLUSTER_A_TICKERS,
        CLUSTER_B_TICKERS,
        CLUSTER_C_TICKERS,
        TICKER_CLUSTER,
        NASDAQ_TICKERS,
    )
    _CLUSTER_TICKERS: dict[str, list[str]] = {
        "A": CLUSTER_A_TICKERS,
        "B": CLUSTER_B_TICKERS,
        "C": CLUSTER_C_TICKERS,
    }
    # 1-based index per cluster (0 = unknown / padding)
    _TICKER_IDX: dict[str, dict[str, int]] = {
        cluster: {t: i + 1 for i, t in enumerate(tickers)}
        for cluster, tickers in _CLUSTER_TICKERS.items()
    }
except Exception as _cfg_err:
    logger.debug(f"[DeepModel] config import failed: {_cfg_err}")
    TICKER_CLUSTER = {}
    _CLUSTER_TICKERS = {"A": [], "B": [], "C": []}
    _TICKER_IDX      = {"A": {}, "B": {}, "C": {}}

# ── Feature columns (V2 — 32 features) ───────────────────────────────────────
try:
    from agent.feature_engine import FEATURE_COLS_V2, compute_features as _compute_features
    FEATURE_COLS = FEATURE_COLS_V2
except Exception as _fe_err:
    logger.debug(f"[DeepModel] feature_engine import failed, falling back to V1: {_fe_err}")
    _compute_features = None
    FEATURE_COLS = [
        "rsi_14", "rsi_7", "macd", "macd_signal", "macd_hist",
        "bb_pct", "bb_width", "stoch_k", "stoch_d", "cci_20", "mfi_14",
        "ema_cross", "vol_ratio", "atr_14", "obv",
        "ret_1", "ret_3", "ret_5",
        "ret_10", "time_sin", "time_cos", "price_range_pos", "vol_trend",
    ]

N_FEATURES = len(FEATURE_COLS)


# ── PyTorch model definition ──────────────────────────────────────────────────

def _build_model(n_tickers: int):
    """
    Build the BiLSTM + Attention network with ticker embedding.
    n_tickers: number of tickers in the cluster (embedding size = n_tickers + 1).
    Returns None if torch is unavailable.
    """
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
            def __init__(self, n_tickers: int):
                super().__init__()
                # +1 for unknown / padding (padding_idx=0)
                self.ticker_emb = nn.Embedding(n_tickers + 1, EMB_DIM, padding_idx=0)
                self.input_proj = nn.Sequential(
                    nn.Linear(N_FEATURES + EMB_DIM, HIDDEN_SIZE),
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

            def forward(self, x, ticker_ids):
                # x: (batch, SEQ_LEN, N_FEATURES)
                # ticker_ids: (batch,) int tensor
                emb = self.ticker_emb(ticker_ids)                          # (batch, EMB_DIM)
                emb_expanded = emb.unsqueeze(1).expand(-1, x.size(1), -1) # (batch, SEQ_LEN, EMB_DIM)
                x_with_emb = torch.cat([x, emb_expanded], dim=-1)         # (batch, SEQ_LEN, N_FEATURES+EMB_DIM)
                x = self.input_proj(x_with_emb)
                x, _ = self.lstm(x)
                x = self.attn(x)
                x = x[:, -1, :]   # use final time-step representation
                return self.head(x).squeeze(-1)

        return DeepDirectionModel(n_tickers)
    except ImportError:
        return None


# ── Scaler helpers ────────────────────────────────────────────────────────────

def _load_scaler(path: Path):
    try:
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None


def _save_scaler(scaler, path: Path) -> None:
    try:
        with open(path, "wb") as f:
            pickle.dump(scaler, f)
    except Exception as e:
        logger.debug(f"[DeepModel] scaler save failed ({path.name}): {e}")


# ── Per-cluster model loader ──────────────────────────────────────────────────

def _get_cluster_model(cluster: str):
    """Load or return cached model for the given cluster."""
    global _cluster_models, _cluster_scalers, _cluster_trained

    with _lock:
        if _cluster_models[cluster] is not None:
            return _cluster_models[cluster]

        n_tickers = len(_CLUSTER_TICKERS.get(cluster, []))
        model = _build_model(n_tickers)
        if model is None:
            return None

        cfg = _CLUSTER_CONFIGS[cluster]
        try:
            import torch
            model_path = cfg["path"]
            if model_path.exists():
                model.load_state_dict(
                    torch.load(model_path, map_location="cpu", weights_only=True)
                )
                model.eval()
                _cluster_scalers[cluster] = _load_scaler(cfg["scaler"])
                _cluster_trained[cluster] = True
                logger.info(f"[DeepModel] Cluster {cluster} loaded from disk.")
        except Exception as e:
            logger.debug(f"[DeepModel] Cluster {cluster} load failed (will retrain): {e}")

        _cluster_models[cluster] = model

        return _cluster_models[cluster]


# ── Feature preparation ───────────────────────────────────────────────────────

def _prepare_df(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """Compute features on raw OHLCV using feature_engine when available."""
    try:
        if _compute_features is not None:
            df = _compute_features(df.copy(), ticker)
        else:
            from agent.technical import compute_indicators
            from agent.ml_model import add_live_features
            df = compute_indicators(df.copy())
            df = add_live_features(df)
        df = df.dropna(subset=FEATURE_COLS)
        return df
    except Exception as e:
        logger.debug(f"[DeepModel] prepare_df error ({ticker}): {e}")
        return pd.DataFrame()


def _make_sequences(
    df: pd.DataFrame,
    ticker_id: int = 0,
    sample_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide a window over df and produce (X, y, weights, ticker_ids) arrays.
    X shape:          (N, SEQ_LEN, N_FEATURES)
    y shape:          (N,)
    weights:          (N,)
    ticker_ids_array: (N,)  — filled with ticker_id for all sequences
    """
    X_list, y_list, w_list = [], [], []
    feat  = df[FEATURE_COLS].values.astype(np.float32)
    close = df["Close"].values

    for i in range(SEQ_LEN, len(feat) - LOOKAHEAD_BARS):
        seq   = feat[i - SEQ_LEN: i]          # (SEQ_LEN, N_FEATURES)
        c_now = close[i - 1]
        c_fut = close[i - 1 + LOOKAHEAD_BARS]
        if c_now <= 0:
            continue
        move  = (c_fut - c_now) / c_now
        if move >= MIN_MOVE_PCT:
            label = 1
        elif move <= -MIN_MOVE_PCT:
            label = 0
        else:
            continue   # noise zone — skip this sample
        X_list.append(seq)
        y_list.append(label)
        w_list.append(sample_weight)

    if not X_list:
        empty_ids = np.empty(0, dtype=np.int64)
        return (
            np.empty((0, SEQ_LEN, N_FEATURES)),
            np.empty(0),
            np.empty(0),
            empty_ids,
        )

    N = len(X_list)
    ticker_ids_array = np.full(N, ticker_id, dtype=np.int64)
    return (
        np.array(X_list),
        np.array(y_list, dtype=np.float32),
        np.array(w_list, dtype=np.float32),
        ticker_ids_array,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def _train_one_cluster(
    cluster_name: str,
    cluster_dfs: dict[str, pd.DataFrame],
    bt_weights: dict[str, float],
) -> bool:
    """
    Train (or fine-tune) the model for a single cluster.
    cluster_dfs: {ticker: df_15m} filtered to this cluster only.
    Returns True on success.
    """
    global _cluster_models, _cluster_scalers, _cluster_trained
    global _training_history

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning(f"[DeepModel] torch/sklearn not available — skipping cluster {cluster_name}")
        return False

    ticker_idx_map = _TICKER_IDX.get(cluster_name, {})
    n_tickers      = len(_CLUSTER_TICKERS.get(cluster_name, []))
    cfg            = _CLUSTER_CONFIGS[cluster_name]

    logger.info(
        f"[DeepModel] Cluster {cluster_name}: building sequences from "
        f"{len(cluster_dfs)} tickers…"
    )

    # Per-ticker chronological 80/20 split to avoid scaler data-leakage.
    # Sequences within each ticker are already ordered (oldest → newest) because
    # _make_sequences slides a window over a time-ordered DataFrame.
    train_X, train_y, train_w, train_ids = [], [], [], []
    val_X,   val_y,   val_w,   val_ids   = [], [], [], []

    for ticker, df_raw in cluster_dfs.items():
        if df_raw is None or len(df_raw) < SEQ_LEN + LOOKAHEAD_BARS + 20:
            continue
        df = _prepare_df(df_raw, ticker)
        if df.empty:
            continue
        w   = bt_weights.get(ticker, 1.0)
        tid = ticker_idx_map.get(ticker, 0)
        X, y, weights, ids = _make_sequences(df, ticker_id=tid, sample_weight=w)
        if len(X) < 10:
            continue
        # Chronological split — no shuffle to preserve time order
        split = max(1, int(len(X) * 0.8))
        train_X.append(X[:split]);   val_X.append(X[split:])
        train_y.append(y[:split]);   val_y.append(y[split:])
        train_w.append(weights[:split]); val_w.append(weights[split:])
        train_ids.append(ids[:split]); val_ids.append(ids[split:])

    if not train_X:
        logger.warning(f"[DeepModel] Cluster {cluster_name}: no training sequences — skipping")
        return False

    X_tr   = np.concatenate(train_X,   axis=0)
    y_tr   = np.concatenate(train_y,   axis=0)
    w_tr   = np.concatenate(train_w,   axis=0)
    id_tr  = np.concatenate(train_ids, axis=0)
    X_val  = np.concatenate(val_X,     axis=0) if val_X  else np.empty((0, SEQ_LEN, N_FEATURES))
    y_val  = np.concatenate(val_y,     axis=0) if val_y  else np.empty(0)
    w_val  = np.concatenate(val_w,     axis=0) if val_w  else np.empty(0)
    id_val = np.concatenate(val_ids,   axis=0) if val_ids else np.empty(0, dtype=np.int64)

    if len(X_tr) < MIN_TRAIN_SAMPLES:
        logger.warning(
            f"[DeepModel] Cluster {cluster_name}: only {len(X_tr)} train sequences "
            f"— need {MIN_TRAIN_SAMPLES}"
        )
        return False

    has_val = len(X_val) >= 32

    logger.info(
        f"[DeepModel] Cluster {cluster_name}: {len(X_tr):,} train / "
        f"{len(X_val):,} val sequences from {len(cluster_dfs)} tickers"
    )

    # ── Scale features (fit on TRAIN only — prevents future-data leakage) ─────
    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, N_FEATURES))
    X_tr_sc = scaler.transform(X_tr.reshape(-1, N_FEATURES)).reshape(X_tr.shape).astype(np.float32)
    if has_val:
        X_val_sc = scaler.transform(X_val.reshape(-1, N_FEATURES)).reshape(X_val.shape).astype(np.float32)
    _save_scaler(scaler, cfg["scaler"])

    # ── Build tensors + dataset ───────────────────────────────────────────────
    X_t  = torch.from_numpy(X_tr_sc)
    y_t  = torch.from_numpy(y_tr)
    w_t  = torch.from_numpy(w_tr)
    id_t = torch.from_numpy(id_tr)

    if has_val:
        X_vt  = torch.from_numpy(X_val_sc)
        y_vt  = torch.from_numpy(y_val.astype(np.float32))
        id_vt = torch.from_numpy(id_val)

    dataset = TensorDataset(X_t, y_t, w_t, id_t)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
        num_workers=0,    # 0 = use calling thread — safe inside ThreadPoolExecutor
        pin_memory=False, # no GPU pinning on CPU-only setup
    )

    # ── Load or build model ───────────────────────────────────────────────────
    is_first_train = not cfg["path"].exists()
    model = _get_cluster_model(cluster_name)
    if model is None:
        model = _build_model(n_tickers)
    if model is None:
        return False

    n_epochs   = EPOCHS if is_first_train else FINE_TUNE_EPOCHS
    lr_now     = LR     if is_first_train else FINE_TUNE_LR
    mode_label = "full training" if is_first_train else f"fine-tuning ({n_epochs} epochs)"
    logger.info(
        f"[DeepModel] Cluster {cluster_name}: starting {mode_label} "
        f"on {len(X_tr):,} train / {len(X_val):,} val sequences…"
    )

    criterion = nn.BCELoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=lr_now, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    model.train()
    best_val_loss  = float("inf")
    best_state     = None

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        for xb, yb, wb, idb in loader:
            optimizer.zero_grad()
            pred = model(xb, idb)
            loss = (criterion(pred, yb) * wb).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Evaluate on held-out validation set (no gradients, no data leakage)
        if has_val:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_vt, id_vt)
                val_loss = criterion(val_pred, y_vt).mean().item()
            track_loss = val_loss
            loss_label = f"train={avg_train_loss:.4f}  val={val_loss:.4f}"
        else:
            track_loss = avg_train_loss
            loss_label = f"loss={avg_train_loss:.4f}  (no val split)"

        if track_loss < best_val_loss:
            best_val_loss = track_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        logger.info(
            f"[DeepModel] Cluster {cluster_name} — "
            f"Epoch {epoch+1}/{n_epochs} — {loss_label}"
        )
        with _lock:
            _training_history.append({
                "epoch":        epoch + 1,
                "total_epochs": n_epochs,
                "loss":         round(avg_train_loss, 6),
                "val_loss":     round(track_loss, 6),
                "ts":           time.time(),
                "tickers":      len(cluster_dfs),
                "mode":         "full" if is_first_train else "finetune",
                "cluster":      cluster_name,
            })
            if len(_training_history) > _MAX_HISTORY:
                del _training_history[:-_MAX_HISTORY]

    # ── Restore best checkpoint (chosen by val loss, not train loss) ──────────
    if best_state:
        model.load_state_dict(best_state)

    model.eval()

    # Accuracy on held-out val split (if available) else training set
    with torch.no_grad():
        if has_val:
            eval_preds = model(X_vt, id_vt).numpy()
            eval_labels = y_val.astype(int)
            acc_label = "val"
        else:
            eval_preds = model(X_t, id_t).numpy()
            eval_labels = y_tr.astype(int)
            acc_label = "train"
    acc = float(((eval_preds >= 0.5).astype(int) == eval_labels).mean())

    # Only persist the model if it improved on the held-out split
    if best_val_loss < float("inf"):
        torch.save(model.state_dict(), cfg["path"])
        with _lock:
            _cluster_models[cluster_name]  = model
            _cluster_scalers[cluster_name] = scaler
            _cluster_trained[cluster_name] = True
        saved = True
    else:
        saved = False

    # Free large training tensors/arrays — model weights kept in _cluster_models
    del X_t, y_t, w_t, id_t, X_tr_sc, X_tr, y_tr, w_tr, id_tr
    if has_val:
        del X_vt, y_vt, id_vt, X_val_sc, X_val, y_val, w_val, id_val
    if best_state:
        del best_state
    gc.collect()

    logger.info(
        f"[DeepModel] Cluster {cluster_name} done — "
        f"{acc_label}_acc={acc:.3f}  best_val_loss={best_val_loss:.4f}"
        f"  saved={saved}"
    )
    return saved


def retrain_deep_all(ticker_dfs_15m: dict[str, pd.DataFrame]) -> bool:
    """
    Train three cluster models (A, B, C) sequentially across all provided
    15-min DataFrames.

    Parameters
    ----------
    ticker_dfs_15m : dict mapping ticker → 15-min OHLCV DataFrame

    Returns True if at least one cluster trained successfully.
    """
    global _trained, _is_training_now, _training_history

    with _lock:
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

        logger.info(
            f"[DeepModel] Starting cluster training across "
            f"{len(ticker_dfs_15m)} tickers…"
        )

        # ── Backtest-derived sample weights ──────────────────────────────────
        bt_weights: dict[str, float] = {}
        try:
            from agent.live_backtest import get_outcomes_for_ml
            outcomes = get_outcomes_for_ml(min_count=1)
            if outcomes is not None and not outcomes.empty:
                for _, row in outcomes.iterrows():
                    t   = str(row.get("ticker", ""))
                    won = int(row.get("won", 0))
                    bt_weights[t] = 2.0 if won else 1.0   # WIN → 2× weight
        except Exception:
            pass

        # ── Split tickers into per-cluster dicts ──────────────────────────────
        cluster_dfs: dict[str, dict[str, pd.DataFrame]] = {
            "A": {}, "B": {}, "C": {},
        }
        for ticker, df_raw in ticker_dfs_15m.items():
            cluster = TICKER_CLUSTER.get(ticker, "A")   # default A for unknowns
            cluster_dfs[cluster][ticker] = df_raw

        # ── Train clusters A / B / C sequentially to cap peak memory ────────────
        # Parallel training triples peak RAM (one model + tensors per cluster
        # simultaneously) — on a 4GB / no-swap host that causes OOM kills.
        # Sequential costs ~1.5× wall-clock but keeps peak usage to one cluster
        # at a time; gc.collect() between clusters releases tensors immediately.
        any_success = False
        for cluster_name in ("A", "B", "C"):
            dfs = cluster_dfs[cluster_name]
            if not dfs:
                logger.info(f"[DeepModel] Cluster {cluster_name}: no tickers — skipping")
                continue
            try:
                if _train_one_cluster(cluster_name, dfs, bt_weights):
                    any_success = True
            except Exception as exc:
                logger.warning(f"[DeepModel] Cluster {cluster_name} training raised: {exc}")
            gc.collect()  # release tensors before next cluster loads

        if any_success:
            with _lock:
                _trained = True

        return any_success

    finally:
        with _lock:
            _is_training_now = False


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_deep(ticker: str, df_15m: pd.DataFrame) -> float:
    """
    Return P(price up in next LOOKAHEAD_BARS × 15min) using the cluster model
    for the given ticker's cluster.
    Returns 0.5 (neutral) when model is untrained or data insufficient.
    """
    cluster = TICKER_CLUSTER.get(ticker, "A")   # default to A for unknown tickers

    # Read model/scaler/trained under lock to prevent using mismatched model+scaler
    # when a retrain completes between reads
    with _lock:
        model  = _cluster_models[cluster]
        scaler = _cluster_scalers[cluster]
        trained = _cluster_trained[cluster]

    # Try loading from disk if not yet in memory
    if model is None:
        model = _get_cluster_model(cluster)
        with _lock:
            scaler  = _cluster_scalers[cluster]
            trained = _cluster_trained[cluster]

    if model is None or not trained:
        return 0.5

    if scaler is None:
        return 0.5

    try:
        import torch

        df = _prepare_df(df_15m, ticker)
        if len(df) < SEQ_LEN:
            return 0.5

        seq      = df[FEATURE_COLS].values[-SEQ_LEN:].astype(np.float32)
        seq_flat = seq.reshape(-1, N_FEATURES)
        seq_scaled = scaler.transform(seq_flat).reshape(1, SEQ_LEN, N_FEATURES)

        x   = torch.from_numpy(seq_scaled.astype(np.float32))
        tid = torch.tensor(
            [_TICKER_IDX.get(cluster, {}).get(ticker, 0)],
            dtype=torch.long,
        )

        model.eval()
        with torch.no_grad():
            prob = float(model(x, tid).item())
        return round(float(np.clip(prob, 0.0, 1.0)), 4)

    except Exception as e:
        logger.debug(f"[DeepModel] predict error for {ticker}: {e}")
        return 0.5


# ── Status / utility ──────────────────────────────────────────────────────────

def is_trained() -> bool:
    """True if at least one cluster model is trained."""
    return any(_cluster_trained.values())


# Alias for backward compat
deep_is_trained = is_trained


def get_model_info() -> dict:
    trained_clusters = [c for c, t in _cluster_trained.items() if t]
    return {
        "trained":          is_trained(),
        "trained_clusters": trained_clusters,
        "seq_len":          SEQ_LEN,
        "lookahead":        LOOKAHEAD_BARS,
        "n_features":       N_FEATURES,
        "emb_dim":          EMB_DIM,
        "hidden_size":      HIDDEN_SIZE,
        "n_layers":         N_LAYERS,
        "n_heads":          N_HEADS,
        "epochs":           EPOCHS,
        "model_path":       str(_MODEL_PATH),   # legacy path for compat
        "cluster_paths": {
            c: str(cfg["path"]) for c, cfg in _CLUSTER_CONFIGS.items()
        },
    }


def get_training_history() -> list[dict]:
    return list(_training_history)


def is_training_active() -> bool:
    with _lock:
        return _is_training_now
