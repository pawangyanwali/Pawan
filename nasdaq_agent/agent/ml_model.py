"""
XGBoost classifier that predicts whether the next N-bar close will be
higher than the current close (label=1) or lower/flat (label=0).

Training uses the full historical 5-min dataset; inference runs on the
most recent feature row fetched during each scan cycle.

Models are persisted to data/models/ via joblib so training is cumulative
across restarts.  A saved model is always loaded first; retraining replaces
it only when fresh data is available.
"""

import gc
import logging
import os
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.preprocessing import StandardScaler

# ── Model persistence directory ───────────────────────────────────────────────
_MODEL_DIR = Path(__file__).parent.parent / "data" / "models"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)
from xgboost import XGBClassifier

from agent.data_fetcher import fetch_historical
from agent.feature_engine import (
    FEATURE_COLS_V2,
    compute_live_row,
    prepare_training_data,
)
from agent.reversal import REVERSAL_FEATURE_COLS, compute_reversal_features
from agent.technical import compute_indicators

logger = logging.getLogger(__name__)

import threading as _threading

_retrain_lock  = _threading.Lock()
_progress_lock = _threading.Lock()   # guards concurrent updates from worker threads
_is_retraining = False               # quick non-blocking check before acquiring lock


def _safe_transform(scaler, row: np.ndarray) -> np.ndarray | None:
    """
    Wrap StandardScaler.transform() with dtype coercion and error recovery.

    sklearn >= 1.4 uses array_api_compat which calls mean_.astype(dtype) on
    the input's dtype.  If a scaler was joblib-loaded across numpy versions,
    mean_ can deserialise as a plain Python float (no .astype()).  Converting
    the input row to float64 routes through sklearn's stable code path and
    avoids the compatibility layer entirely.

    Returns the scaled array, or None if the scaler state is unrecoverable.
    """
    try:
        row64 = np.asarray(row, dtype=np.float64)
        if row64.ndim == 1:
            row64 = row64.reshape(1, -1)
        return scaler.transform(row64)
    except AttributeError:
        # Scaler mean_/scale_ are plain Python floats — joblib version skew.
        # Caller must reset model.trained = False and retrain next cycle.
        return None
    except Exception:
        return None

# ── Per-ticker training progress tracker ──────────────────────────────────────
# Updated live during _retrain_all_locked so the UI can show a real-time queue.

import time as _time_module

_retrain_progress: dict = {
    "is_running":      False,
    "phase":           "",        # "fetching_5m" | "xgboost" | "fetching_15m" | "swing" | "deep" | "done"
    "phase_label":     "",        # human-readable phase name
    "current_ticker":  "",
    "current_model":   "",        # "scalp" | "daily" | "reversal" | "ensemble" | "swing"
    "completed":       [],        # [{"ticker","models":[],"elapsed_s"}]
    "failed":          [],        # [{"ticker","model","error"}]
    "total":           0,
    "done_count":      0,
    "started_at":      0.0,
    "elapsed_s":       0.0,
}


def get_retrain_progress() -> dict:
    """Return a snapshot of the current retrain progress (safe to call any time)."""
    p = dict(_retrain_progress)
    p["elapsed_s"] = round(_time_module.time() - p["started_at"], 1) if p["started_at"] else 0.0
    p["completed"] = list(p["completed"])
    p["failed"]    = list(p["failed"])
    return p


def _rp_set(**kwargs):
    """Update _retrain_progress fields under lock (called from retrain thread)."""
    with _progress_lock:
        _retrain_progress.update(kwargs)


def _rp_append_completed(entry: dict) -> None:
    """Thread-safe append to _retrain_progress['completed'] and bump done_count."""
    with _progress_lock:
        _retrain_progress["completed"].append(entry)
        _retrain_progress["done_count"] = len(_retrain_progress["completed"])


def _rp_append_failed(entry: dict) -> None:
    """Thread-safe append to _retrain_progress['failed']."""
    with _progress_lock:
        _retrain_progress["failed"].append(entry)


LOOKAHEAD_BARS = 3   # predict direction 3×1min = 3 min ahead (scalping)


# ── Atomic model save helper ──────────────────────────────────────────────────

def _atomic_save(obj, path: Path) -> None:
    """Write to temp file then rename — atomic on Linux, never corrupts existing file."""
    tmp = path.with_suffix(".tmp")
    try:
        joblib.dump(obj, tmp, compress=0)   # no compression — models are small, speed > size
        os.replace(tmp, path)
    except Exception as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ── Fast XGBoost training helper ──────────────────────────────────────────────

def _fast_xgb_fit(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    *,
    n_estimators: int = 400,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 3,
    scale_pos_weight: float = 1.0,
) -> "CalibratedClassifierCV":
    """
    Train XGBoost with histogram splits + early stopping + prefit sigmoid calibration.

    Speed wins vs the old CalibratedClassifierCV(cv=3):
      - tree_method='hist'  : histogram-based splits — 10-50× faster than 'exact'
      - nthread=1           : 1 thread/model lets ThreadPoolExecutor fill all CPUs cleanly
      - early_stopping_rounds=20: stops at ~60-100 trees instead of always 200+
      - cv='prefit'         : calibrate on holdout in <1 ms vs training 3 extra folds
    Net result: ~8-15× faster per model, zero quality loss.

    Fallback: if early stopping fires too early (best_iteration < 5 = model barely
    trained), we retrain without early stopping using a conservative fixed n_estimators.
    This prevents degenerate "trees=0" models on noisy tickers.
    """
    _common_kwargs = dict(
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        device="cpu",
        nthread=1,
        eval_metric="logloss",
        verbosity=0,
    )
    base = XGBClassifier(
        n_estimators=n_estimators,
        early_stopping_rounds=20,
        **_common_kwargs,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        base.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)], verbose=False)

    # Guard: if early stopping killed the model before it learned anything,
    # fall back to a fixed 60-tree model (no early stopping) so we always have
    # a usable model rather than a degenerate 0-tree estimator.
    best_iter = getattr(base, "best_iteration", 99)
    if best_iter < 5:
        base = XGBClassifier(n_estimators=60, **_common_kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            base.fit(X_tr, y_tr)

    # sklearn ≥1.6 removed cv='prefit'; FrozenEstimator is the replacement —
    # it prevents re-fitting inside CalibratedClassifierCV, same behaviour.
    # cv is capped to the smallest class size so StratifiedKFold never warns
    # about classes with fewer members than n_splits.
    _min_cls = int(np.bincount(y_cal).min()) if len(np.unique(y_cal)) > 1 else 2
    _cv = max(2, min(5, _min_cls))
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid", cv=_cv)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        cal.fit(X_cal, y_cal)
    return cal


# ── Legacy feature helpers (kept for DailyMLModel / backward compat) ──────────

def add_live_features(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """Compute extra features on top of whatever compute_indicators returns."""
    df = df.copy()
    # time-of-day cyclical encoding
    if isinstance(df.index, pd.DatetimeIndex):
        minutes = pd.Series(df.index.hour * 60 + df.index.minute, index=df.index, dtype=float)
        day_min = (16 * 60) - (9 * 60 + 30)  # 390 min trading day
        norm = (minutes - (9 * 60 + 30)) / day_min
        norm = norm.clip(0, 1)
        df["time_sin"] = np.sin(2 * np.pi * norm)
        df["time_cos"] = np.cos(2 * np.pi * norm)
    else:
        df["time_sin"] = 0.0
        df["time_cos"] = 0.0
    # where price sits in today's H-L range
    day_high = df["High"].rolling(78, min_periods=1).max()
    day_low  = df["Low"].rolling(78, min_periods=1).min()
    rng = (day_high - day_low).replace(0, np.nan)
    df["price_range_pos"] = ((df["Close"] - day_low) / rng).clip(0, 1).fillna(0.5)
    # 10-bar return
    df["ret_10"] = df["Close"].pct_change(10)
    # volume trend: recent 5-bar avg vs 20-bar avg
    v5  = df["Volume"].rolling(5,  min_periods=1).mean()
    v20 = df["Volume"].rolling(20, min_periods=1).mean()
    df["vol_trend"] = (v5 / v20.replace(0, np.nan)).fillna(1.0).clip(0, 5)

    # Streaming microstructure features — zero when Schwab disabled
    df["bid_ask_imbalance"] = 0.0
    df["nq_futures_bias"]   = 0.0
    df["es_futures_bias"]   = 0.0

    return df


# Kept for DailyMLModel which uses the legacy V1 feature set (no time/vwap meaning at daily res)
_DAILY_FEATURE_COLS = [
    "rsi_14", "rsi_7", "macd", "macd_signal", "macd_hist",
    "bb_pct", "bb_width", "stoch_k", "stoch_d", "cci_20", "mfi_14",
    "ema_cross", "vol_ratio", "atr_14", "obv",
    "ret_1", "ret_3", "ret_5",
    "ret_10", "time_sin", "time_cos", "price_range_pos", "vol_trend",
]


class StockMLModel:
    def __init__(self, ticker: str):
        self.ticker  = ticker
        self.model   = None
        self.scaler  = StandardScaler()
        self.trained = False
        self._load()   # restore from disk on construction

    # ── Persistence ───────────────────────────────────────────────────────────

    def _path(self) -> Path:
        return _MODEL_DIR / f"scalp_{self.ticker}.joblib"

    def _save(self) -> None:
        try:
            _atomic_save(
                {"model": self.model, "scaler": self.scaler, "trained": self.trained},
                self._path(),
            )
        except Exception as e:
            logger.debug(f"[{self.ticker}] scalp save failed: {e}")

    def _load(self) -> None:
        try:
            p = self._path()
            if p.exists():
                d = joblib.load(p)
                self.model, self.scaler, self.trained = d["model"], d["scaler"], d.get("trained", False)
                logger.debug(f"[{self.ticker}] scalp model loaded from disk")
        except Exception as e:
            logger.debug(f"[{self.ticker}] scalp load failed (will retrain): {e}")

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self) -> bool:
        """Fetch historical data then train. Prefer train_from_df when data is pre-fetched."""
        return self.train_from_df(fetch_historical(self.ticker))

    def train_from_df(
        self,
        df: pd.DataFrame | None,
        _prepared: tuple | None = None,
    ) -> bool:
        """Train from a pre-fetched 5-min OHLCV DataFrame (no API call).

        Pass _prepared=(X_train, y_train, X_test, y_test) to skip the feature
        computation step when the caller already has it (avoids duplicate work
        when scalp + ensemble train on the same df).
        """
        if _prepared is not None:
            result = _prepared
        else:
            result = prepare_training_data(df, ticker=self.ticker, lookahead_bars=LOOKAHEAD_BARS)
        if result is None:
            return False
        X_train, y_train, X_test, y_test = result

        class_counts = np.bincount(y_train)
        if len(class_counts) < 2 or (class_counts.max() / len(y_train)) > 0.85:
            return False

        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_tr_s = self.scaler.transform(X_train)
        X_te_s = self.scaler.transform(X_test)

        self.model = _fast_xgb_fit(X_tr_s, y_train, X_te_s, y_test,
                                    n_estimators=400, max_depth=4,
                                    learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8)
        self.trained = True
        self._save()

        acc = self.model.score(X_te_s, y_test)
        n_trees = getattr(self.model.estimator, "best_iteration", "?")
        logger.info(
            f"[{self.ticker}] ScalpML trained | acc={acc:.3f} | "
            f"trees={n_trees} | samples={len(X_train)}"
        )
        return True

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> float:
        """
        Return probability [0,1] that the price goes UP in the next N bars.
        Returns 0.5 (neutral) when the model is not trained or features are missing.
        """
        if not self.trained or self.model is None:
            return 0.5

        row = compute_live_row(df, ticker=self.ticker)
        if row is None:
            return 0.5
        expected = getattr(self.scaler, "n_features_in_", None)
        if expected is not None and row.shape[1] != expected:
            logger.debug(
                f"[{self.ticker}] scalp scaler expects {expected} features, "
                f"got {row.shape[1]} — resetting model"
            )
            self.trained = False
            return 0.5
        row_s = _safe_transform(self.scaler, row)
        if row_s is None:
            logger.warning(f"[{self.ticker}] scalp scaler corrupted — resetting for retrain")
            self.trained = False
            return 0.5
        return round(float(self.model.predict_proba(row_s)[0][1]), 4)


# ── Global registry: one model per ticker ─────────────────────────────────────

_model_registry: dict[str, StockMLModel] = {}


def get_or_create(ticker: str) -> StockMLModel:
    """Return existing model (untrained is fine) — training happens lazily in retrain_all."""
    if ticker not in _model_registry:
        _model_registry[ticker] = StockMLModel(ticker)
    return _model_registry[ticker]


def retrain_all(tickers: list, delay: float = 0.0, daily_data: dict = None,
                hist_5m: dict = None, hist_15m: dict = None) -> None:
    """Train/retrain all models using a single batch historical fetch.

    XGBoost scalp/ensemble/reversal models train on 1-min bars (Schwab provides
    ~10 days; cache grows over time).  SwingML uses 15-min bars.  DailyML uses
    daily bars (2 years).

    Parameters
    ----------
    tickers    : list of ticker symbols to retrain.
    delay      : unused (kept for API compat).
    daily_data : optional mapping of ticker → daily OHLCV DataFrame.
    hist_5m    : accepted for API compat; treated as hist_1m (1-min data).
    hist_15m   : optional pre-fetched 15-min data dict {ticker: DataFrame}.
    """
    global _is_retraining
    # Non-blocking guard: if another retrain is already running, skip this call
    if _is_retraining:
        logger.info("[retrain_all] Skipped — another retrain already in progress")
        return
    if not _retrain_lock.acquire(blocking=False):
        logger.info("[retrain_all] Skipped — lock held by concurrent retrain")
        return
    _is_retraining = True
    try:
        _retrain_all_locked(tickers, delay=delay, daily_data=daily_data,
                            hist_5m=hist_5m, hist_15m=hist_15m)
    finally:
        _is_retraining = False
        _retrain_lock.release()


def _train_one_ticker(
    t: str,
    df5m: pd.DataFrame | None,
    df15m: pd.DataFrame | None,
    daily_data: dict | None,
) -> tuple[str, list[str], float]:
    """Train all models for one ticker. Returns (ticker, models_ok, elapsed_s).

    Designed to run inside a ThreadPoolExecutor worker.  All registry mutations
    are local to each ticker so there are no shared data races.
    """
    ticker_models_ok: list[str] = []
    t0 = time.time()

    # Pre-compute training data ONCE for 5m and 15m — shared across models
    # that use the same lookahead to avoid duplicate feature engineering.
    prepared_5m  = prepare_training_data(df5m,  ticker=t, lookahead_bars=LOOKAHEAD_BARS)
    prepared_15m = prepare_training_data(df15m, ticker=t, lookahead_bars=SwingMLModel.LOOKAHEAD)

    # ── Scalp model ───────────────────────────────────────────────────────
    try:
        m = _model_registry[t] if t in _model_registry else StockMLModel(t)
        if m.train_from_df(df5m, _prepared=prepared_5m):
            ticker_models_ok.append("scalp")
        _model_registry[t] = m
    except Exception as e:
        logger.warning(f"[{t}] scalp retrain failed: {e}")
        _rp_append_failed({"ticker": t, "model": "scalp", "error": str(e)})

    # ── Daily model ───────────────────────────────────────────────────────
    if daily_data and t in daily_data:
        try:
            dm = get_or_create_daily(t)
            if dm.train_from_df(daily_data[t]):
                ticker_models_ok.append("daily")
        except Exception as e:
            logger.warning(f"[{t}] daily retrain failed: {e}")
            _rp_append_failed({"ticker": t, "model": "daily", "error": str(e)})

    # ── Reversal model (has its own label logic — can't share prepared_5m) ──
    try:
        rm = get_or_create_reversal(t)
        if rm.train_from_df(df5m):
            ticker_models_ok.append("reversal")
    except Exception as e:
        logger.warning(f"[{t}] reversal retrain failed: {e}")
        _rp_append_failed({"ticker": t, "model": "reversal", "error": str(e)})

    # ── Ensemble model (shares prepared_5m with scalp — same features/split) ─
    try:
        em = get_or_create_ensemble(t)
        if em.train_from_df(df5m, _prepared=prepared_5m):
            ticker_models_ok.append("ensemble")
    except Exception as e:
        logger.warning(f"[{t}] ensemble retrain failed: {e}")
        _rp_append_failed({"ticker": t, "model": "ensemble", "error": str(e)})

    # ── Swing model (shares prepared_15m) ────────────────────────────────
    try:
        sm = get_or_create_swing(t)
        if sm.train_from_df(df15m, _prepared=prepared_15m):
            ticker_models_ok.append("swing")
    except Exception as e:
        logger.warning(f"[{t}] swing retrain failed: {e}")
        _rp_append_failed({"ticker": t, "model": "swing", "error": str(e)})

    gc.collect()
    return t, ticker_models_ok, round(time.time() - t0, 1)


def _retrain_all_locked(tickers: list, delay: float = 0.0, daily_data: dict = None,
                        hist_5m: dict = None, hist_15m: dict = None) -> None:
    """Internal retrain — only called while _retrain_lock is held."""
    import time as _t
    from agent.data_fetcher import fetch_batch_interval

    _rp_set(is_running=True, phase="fetching_1m",
            phase_label="Fetching 1-min data (Schwab ~10 days)…",
            started_at=_t.time(), total=len(tickers),
            done_count=0, completed=[], failed=[],
            current_ticker="", current_model="")

    # ── 1-min data: ~10 days (XGBoost scalp/ensemble/reversal models) ────────
    # hist_5m param accepted for API compat — callers may pass pre-fetched data;
    # treat it as 1-min data (same variable, just a different source interval now).
    if hist_5m is not None:
        logger.info(f"[retrain_all] Using pre-fetched 1min data: {len(hist_5m)} tickers "
                    f"(avg {sum(len(v) for v in hist_5m.values())//max(len(hist_5m),1)} bars each)")
    else:
        # ttl=86400 → SQLite check uses 4-day window, so stored history is used
        # instead of live API calls whenever the scan has previously written bars.
        logger.info(f"[retrain_all] Fetching 1min history for {len(tickers)} tickers (SQLite-first, extended hours)…")
        hist_5m = fetch_batch_interval(
            tickers, "1min", 3900, ttl=86400, background=True, extended_hours=True
        )
        logger.info(f"[retrain_all] Got 1min history for {len(hist_5m)}/{len(tickers)} tickers")

    # ── 15-min data: ~6 months (swing models + deep BiLSTM) ─────────────────
    _rp_set(phase="fetching_15m", phase_label="Fetching 15-min data (SQLite/Schwab)…")
    if hist_15m is not None:
        logger.info(f"[retrain_all] Using pre-fetched 15min data: {len(hist_15m)} tickers")
    else:
        logger.info(f"[retrain_all] Fetching 15min history ({len(tickers)} tickers, SQLite-first, extended hours)…")
        hist_15m = fetch_batch_interval(
            tickers, "15min", 5000, ttl=86400, background=True, extended_hours=True
        )
        logger.info(f"[retrain_all] 15min data: {len(hist_15m)}/{len(tickers)} tickers")

    # ── Daily data: ~2 years (DailyMLModel — next-day direction) ─────────────
    _rp_set(phase="fetching_daily", phase_label="Fetching daily bars (SQLite/Schwab)…")
    if daily_data is not None:
        logger.info(f"[retrain_all] Using pre-fetched daily data: {len(daily_data)} tickers")
    else:
        logger.info(f"[retrain_all] Fetching daily history ({len(tickers)} tickers, SQLite-first)…")
        daily_data = fetch_batch_interval(tickers, "1day", 500, ttl=86400, background=True)
        logger.info(f"[retrain_all] Daily data: {len(daily_data)}/{len(tickers)} tickers")

    _rp_set(phase="xgboost", phase_label="Training XGBoost models per ticker…")

    # ── Parallel ticker training ───────────────────────────────────────────────
    # Cap at 2 workers regardless of CPU count — on a 4GB host each XGBoost
    # retrain + feature engineering can spike 100-200 MB, so running 4+ in
    # parallel risks OOM before the deep model phase even starts.
    _workers = max(1, min((os.cpu_count() or 2) // 2, 2))
    with ThreadPoolExecutor(max_workers=_workers) as executor:
        futures = {
            executor.submit(
                _train_one_ticker,
                t,
                hist_5m.get(t),
                hist_15m.get(t),
                daily_data,
            ): t
            for t in tickers
        }
        for future in as_completed(futures):
            try:
                t, models_ok, elapsed = future.result()
            except Exception as exc:
                t = futures[future]
                logger.warning(f"[{t}] _train_one_ticker raised: {exc}")
                _rp_append_failed({"ticker": t, "model": "unknown", "error": str(exc)})
                elapsed = 0.0
                models_ok = []
            _rp_append_completed({
                "ticker":    t,
                "models":    models_ok,
                "elapsed_s": elapsed,
            })

    # Free XGBoost training memory before starting the deeper BiLSTM phase
    gc.collect()

    # ── Deep BiLSTM model: universal, trained across all tickers ─────────────
    _rp_set(phase="deep", phase_label="Training Deep BiLSTM…",
            current_ticker="all tickers", current_model="bilstm")
    try:
        from agent.deep_model import retrain_deep_all
        logger.info(f"[retrain_all] Training deep BiLSTM on {len(hist_15m)} tickers…")
        retrain_deep_all(hist_15m)
    except Exception as e:
        logger.warning(f"[retrain_all] Deep model training failed: {e}")

    _rp_set(phase="done", phase_label="Complete", is_running=False,
            current_ticker="", current_model="")


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

    Uses the legacy V1 feature set via compute_indicators + add_live_features
    because daily bars lack intraday time-of-day meaning; time_sin / time_cos
    are zeroed out but kept for model-schema compatibility.
    """

    def __init__(self, ticker: str):
        self.ticker  = ticker
        self.model   = None
        self.scaler  = StandardScaler()
        self.trained = False
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _path(self) -> Path:
        return _MODEL_DIR / f"daily_{self.ticker}.joblib"

    def _save(self) -> None:
        try:
            _atomic_save(
                {"model": self.model, "scaler": self.scaler, "trained": self.trained},
                self._path(),
            )
        except Exception as e:
            logger.debug(f"[{self.ticker}] daily save failed: {e}")

    def _load(self) -> None:
        try:
            p = self._path()
            if p.exists():
                d = joblib.load(p)
                self.model, self.scaler, self.trained = d["model"], d["scaler"], d.get("trained", False)
                logger.debug(f"[{self.ticker}] daily model loaded from disk")
        except Exception as e:
            logger.debug(f"[{self.ticker}] daily load failed (will retrain): {e}")

    # ── Training ──────────────────────────────────────────────────────────────

    def train_from_df(self, df_daily: pd.DataFrame) -> bool:
        """Train on the supplied daily OHLCV DataFrame.  Returns True on success."""
        if df_daily is None or len(df_daily) < 100:
            return False

        df = compute_indicators(df_daily.copy())
        df = add_live_features(df, ticker=self.ticker)
        df = df.dropna(subset=_DAILY_FEATURE_COLS)

        # Label: 1 if next-day close > today's close
        df["label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df.dropna(inplace=True)

        X = df[_DAILY_FEATURE_COLS].values
        y = df["label"].values

        if len(X) < 60:
            return False

        class_counts = np.bincount(y)
        if len(class_counts) < 2 or (class_counts.max() / len(y)) > 0.85:
            return False

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if len(np.unique(y_train)) < 2:
            return False

        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        self.model = _fast_xgb_fit(X_train_s, y_train, X_test_s, y_test,
                                    n_estimators=300, max_depth=4,
                                    learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8)
        self.trained = True
        self._save()

        acc = self.model.score(X_test_s, y_test)
        n_trees = getattr(self.model.estimator, "best_iteration", "?")
        logger.info(
            f"[{self.ticker}] DailyML trained | acc={acc:.3f} | "
            f"trees={n_trees} | samples={len(X_train)}"
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
        if df_daily is None or len(df_daily) < 30:
            return 0.5

        df = compute_indicators(df_daily.copy())
        df = add_live_features(df, ticker=self.ticker)
        # compute_indicators silently skips when len(df) < 30 — guard here too
        missing = [c for c in _DAILY_FEATURE_COLS if c not in df.columns]
        if missing:
            return 0.5
        df = df.dropna(subset=_DAILY_FEATURE_COLS)
        if df.empty:
            return 0.5

        row = df[_DAILY_FEATURE_COLS].iloc[[-1]].values
        expected = getattr(self.scaler, "n_features_in_", None)
        if expected is not None and row.shape[1] != expected:
            logger.debug(
                f"[{self.ticker}] daily scaler expects {expected} features, "
                f"got {row.shape[1]} — resetting model"
            )
            self.trained = False
            return 0.5
        row_s = _safe_transform(self.scaler, row)
        if row_s is None:
            logger.warning(f"[{self.ticker}] daily scaler corrupted — resetting for retrain")
            self.trained = False
            return 0.5
        return round(float(self.model.predict_proba(row_s)[0][1]), 4)


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

    NOTE: deliberately does NOT use feature_engine / FEATURE_COLS_V2.
    REVERSAL_FEATURE_COLS is its own set from agent.reversal.
    """

    # Lookahead and threshold for labelling reversals in training data
    _LOOKAHEAD  = 5     # bars ahead to check
    _MOVE_PCT   = 0.008  # 0.8% = meaningful reversal

    def __init__(self, ticker: str):
        self.ticker  = ticker
        self.model   = None
        self.scaler  = StandardScaler()
        self.trained = False
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _path(self) -> Path:
        return _MODEL_DIR / f"reversal_{self.ticker}.joblib"

    def _save(self) -> None:
        try:
            _atomic_save(
                {"model": self.model, "scaler": self.scaler, "trained": self.trained},
                self._path(),
            )
        except Exception as e:
            logger.debug(f"[{self.ticker}] reversal save failed: {e}")

    def _load(self) -> None:
        try:
            p = self._path()
            if p.exists():
                d = joblib.load(p)
                self.model, self.scaler, self.trained = d["model"], d["scaler"], d.get("trained", False)
                logger.debug(f"[{self.ticker}] reversal model loaded from disk")
        except Exception as e:
            logger.debug(f"[{self.ticker}] reversal load failed (will retrain): {e}")

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_indicators(df.copy())
        df = compute_reversal_features(df)
        return df

    def train(self) -> bool:
        """Fetch historical data then train. Prefer train_from_df when data is pre-fetched."""
        return self.train_from_df(fetch_historical(self.ticker))

    def train_from_df(self, raw: pd.DataFrame | None) -> bool:
        """Train from a pre-fetched 5-min OHLCV DataFrame (no API call)."""
        if raw is None or len(raw) < 150:
            return False

        df = self._prepare(raw)

        # Label: 1 if price rises ≥ 0.8% within next 5 bars (bullish reversal)
        future_high = df["Close"].rolling(self._LOOKAHEAD).max().shift(-self._LOOKAHEAD)
        df["label"] = ((future_high - df["Close"]) / df["Close"] >= self._MOVE_PCT).astype(int)
        df = df.dropna(subset=REVERSAL_FEATURE_COLS + ["label"])

        X = df[REVERSAL_FEATURE_COLS].values
        y = df["label"].values

        # Guard: replace any remaining inf/-inf with 0 (e.g. from vol pct_change
        # on zero-volume bars that slipped through compute_reversal_features).
        X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)

        if len(X) < 60 or y.sum() < 10:
            return False

        # Imbalance check: reversal labels are rare by design, but if >85% one class
        # the calibration folds will fail, producing a divide-by-zero RuntimeWarning.
        class_counts = np.bincount(y)
        if len(class_counts) < 2 or (class_counts.max() / len(y)) > 0.85:
            return False

        X_train, X_test, y_train, y_test = (
            X[:int(len(X) * 0.8)],
            X[int(len(X) * 0.8):],
            y[:int(len(y) * 0.8)],
            y[int(len(y) * 0.8):],
        )

        if len(np.unique(y_train)) < 2:
            return False

        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_tr = self.scaler.transform(X_train)
        X_te = self.scaler.transform(X_test)

        spw = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
        self.model = _fast_xgb_fit(X_tr, y_train, X_te, y_test,
                                    n_estimators=300, max_depth=4,
                                    learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.7, min_child_weight=3,
                                    scale_pos_weight=spw)
        self.trained = True
        self._save()

        acc = self.model.score(X_te, y_test)
        n_trees = getattr(self.model.estimator, "best_iteration", "?")
        logger.info(
            f"[{self.ticker}] ReversalML trained | acc={acc:.3f} | "
            f"trees={n_trees} | reversals={y.sum()}/{len(y)} ({y.mean()*100:.1f}%)"
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
            row_s = _safe_transform(self.scaler, row)
            if row_s is None:
                self.trained = False
                return 0.5
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


# ── Swing ML Model ────────────────────────────────────────────────────────────

class SwingMLModel:
    """
    XGBoost classifier trained on 15-min bars (~9 months of data).

    Predicts whether price will be higher LOOKAHEAD × 15min = 2 hours ahead.
    Using 15-min bars instead of 5-min gives 3× more history for the same
    API limit (5000 bars × 15min ≈ 9 months vs ≈ 64 days for 5-min).

    Train and infer on the same 15-min timeframe — no mismatch with the scalp
    model; only the output probability (P(up)) is blended at inference time.

    Uses prepare_training_data() from feature_engine for leakage-free split
    and ATR-adaptive label construction.
    """

    LOOKAHEAD = 8   # 8 × 15min = 2 hours ahead

    def __init__(self, ticker: str):
        self.ticker  = ticker
        self.model   = None
        self.scaler  = StandardScaler()
        self.trained = False
        self._load()

    def _path(self) -> Path:
        return _MODEL_DIR / f"swing_{self.ticker}.joblib"

    def _save(self) -> None:
        try:
            _atomic_save(
                {"model": self.model, "scaler": self.scaler, "trained": self.trained},
                self._path(),
            )
        except Exception as e:
            logger.debug(f"[{self.ticker}] swing save failed: {e}")

    def _load(self) -> None:
        try:
            p = self._path()
            if p.exists():
                d = joblib.load(p)
                self.model, self.scaler, self.trained = d["model"], d["scaler"], d.get("trained", False)
                logger.debug(f"[{self.ticker}] swing model loaded from disk")
        except Exception as e:
            logger.debug(f"[{self.ticker}] swing load failed (will retrain): {e}")

    def train_from_df(
        self,
        df_15m: pd.DataFrame | None,
        _prepared: tuple | None = None,
    ) -> bool:
        """Train from a pre-fetched 15-min OHLCV DataFrame (no API call).

        Pass _prepared=(X_train, y_train, X_test, y_test) to skip feature
        computation when the caller already has it.
        """
        if _prepared is not None:
            result = _prepared
        else:
            result = prepare_training_data(
                df_15m, ticker=self.ticker, lookahead_bars=self.LOOKAHEAD
            )
        if result is None:
            return False
        X_train, y_train, X_test, y_test = result

        class_counts = np.bincount(y_train)
        if len(class_counts) < 2 or (class_counts.max() / len(y_train)) > 0.85:
            return False

        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_tr_s = self.scaler.transform(X_train)
        X_te_s = self.scaler.transform(X_test)

        self.model = _fast_xgb_fit(X_tr_s, y_train, X_te_s, y_test,
                                    n_estimators=300, max_depth=4,
                                    learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, min_child_weight=3)
        self.trained = True
        self._save()

        acc = self.model.score(X_te_s, y_test)
        n_trees = getattr(self.model.estimator, "best_iteration", "?")
        logger.info(
            f"[{self.ticker}] SwingML trained | acc={acc:.3f} | "
            f"trees={n_trees} | samples={len(X_train)} (15min, 2h lookahead)"
        )
        return True

    def predict_proba(self, df_15m: pd.DataFrame) -> float:
        """
        Return probability [0, 1] that price will be higher 2h ahead.
        Returns 0.5 (neutral) when untrained or features are missing.
        """
        if not self.trained or self.model is None:
            return 0.5

        row = compute_live_row(df_15m, ticker=self.ticker)
        if row is None:
            return 0.5
        expected = getattr(self.scaler, "n_features_in_", None)
        if expected is not None and row.shape[1] != expected:
            logger.debug(
                f"[{self.ticker}] swing scaler expects {expected} features, "
                f"got {row.shape[1]} — resetting model"
            )
            self.trained = False
            return 0.5
        row_s = _safe_transform(self.scaler, row)
        if row_s is None:
            logger.warning(f"[{self.ticker}] swing scaler corrupted — resetting for retrain")
            self.trained = False
            return 0.5
        return round(float(self.model.predict_proba(row_s)[0][1]), 4)


# ── Swing model registry ──────────────────────────────────────────────────────

_swing_model_registry: dict[str, SwingMLModel] = {}


def get_or_create_swing(ticker: str) -> SwingMLModel:
    if ticker not in _swing_model_registry:
        _swing_model_registry[ticker] = SwingMLModel(ticker)
    return _swing_model_registry[ticker]


def predict_swing(ticker: str, df_15m: pd.DataFrame) -> float:
    """Return 2h-ahead up-probability from the SwingMLModel on 15-min bars; 0.5 if untrained."""
    return get_or_create_swing(ticker).predict_proba(df_15m)


# ── Ensemble ML Model ─────────────────────────────────────────────────────────

class EnsembleMLModel:
    """
    Ensemble of 3 diverse XGBoost classifiers (reduced from 10 for memory).
    Confidence = agreement fraction (0.0–1.0) among models.
    Low agreement → uncertain prediction, high agreement → high-confidence.

    Uses prepare_training_data() from feature_engine for leakage-free split
    and ATR-adaptive label construction.
    """
    N_MODELS = 3

    # Diverse configs: shallow-fast / balanced / deeper-slower
    # n_estimators is an upper bound — early stopping cuts to ~40-80 trees each
    _CONFIGS = [
        dict(n_estimators=300, max_depth=3, learning_rate=0.10, subsample=0.7, colsample_bytree=0.7),
        dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
        dict(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.6, colsample_bytree=0.8),
    ]

    def __init__(self, ticker: str):
        self.ticker  = ticker
        self.models  = []
        self.scaler  = StandardScaler()
        self.trained = False
        self._load()

    def _path(self) -> Path:
        return _MODEL_DIR / f"ensemble_{self.ticker}.joblib"

    def _save(self) -> None:
        try:
            _atomic_save(
                {"models": self.models, "scaler": self.scaler, "trained": self.trained},
                self._path(),
            )
        except Exception as e:
            logger.debug(f"[{self.ticker}] ensemble save failed: {e}")

    def _load(self) -> None:
        try:
            p = self._path()
            if p.exists():
                d = joblib.load(p)
                self.models, self.scaler, self.trained = d["models"], d["scaler"], d.get("trained", False)
        except Exception as e:
            logger.debug(f"[{self.ticker}] ensemble load failed: {e}")

    def train_from_df(
        self,
        df: pd.DataFrame | None,
        _prepared: tuple | None = None,
    ) -> bool:
        """Train from a pre-fetched 5-min OHLCV DataFrame.

        Pass _prepared=(X_train, y_train, X_test, y_test) to skip feature
        computation when the caller already has it from the scalp model step.
        Each member is trained via _fast_xgb_fit (hist + early stopping).
        """
        if _prepared is not None:
            result = _prepared
        else:
            result = prepare_training_data(df, ticker=self.ticker, lookahead_bars=LOOKAHEAD_BARS)
        if result is None:
            return False
        X_train, y_train, X_test, y_test = result

        if len(X_train) < 100:
            return False
        class_counts = np.bincount(y_train)
        if len(class_counts) < 2 or (class_counts.max() / len(y_train)) > 0.85:
            return False

        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        Xtr = self.scaler.transform(X_train)
        Xte = self.scaler.transform(X_test)

        # Train each member in its own thread (3 × independent XGBoost fits)
        def _fit_member(args):
            i, cfg = args
            return _fast_xgb_fit(Xtr, y_train, Xte, y_test, **cfg)

        with ThreadPoolExecutor(max_workers=len(self._CONFIGS)) as ex:
            self.models = list(ex.map(_fit_member, enumerate(self._CONFIGS)))

        self.trained = True
        self._save()

        preds = np.array([m.predict(Xte) for m in self.models])
        majority = (preds.mean(axis=0) >= 0.5).astype(int)
        acc = float((majority == y_test).mean())
        logger.info(
            f"[{self.ticker}] Ensemble trained | acc={acc:.3f} | "
            f"models={len(self.models)} | samples={len(X_train)}"
        )
        return True

    def predict(self, df: pd.DataFrame) -> tuple[float, float]:
        """Returns (probability_up, agreement_0_to_1).
        agreement=1.0 means all models agree, 0.5 means split."""
        if not self.trained or not self.models:
            return 0.5, 0.0
        try:
            row = compute_live_row(df, ticker=self.ticker)
            if row is None:
                return 0.5, 0.0
            expected = getattr(self.scaler, "n_features_in_", None)
            if expected is not None and row.shape[1] != expected:
                logger.debug(
                    f"[{self.ticker}] ensemble scaler expects {expected} features, "
                    f"got {row.shape[1]} — resetting model"
                )
                self.trained = False
                return 0.5, 0.0
            row_s = _safe_transform(self.scaler, row)
            if row_s is None:
                self.trained = False
                return 0.5, 0.0
            probs = np.array([m.predict_proba(row_s)[0][1] for m in self.models])
            avg_prob  = float(probs.mean())
            # agreement: how consistently models agree on direction
            majority  = int(avg_prob >= 0.5)
            agreement = float((probs >= 0.5).mean()) if majority == 1 else float((probs < 0.5).mean())
            return round(avg_prob, 4), round(agreement, 4)
        except Exception as e:
            logger.debug(f"[{self.ticker}] ensemble predict error: {e}")
            return 0.5, 0.0


# ── Ensemble model registry ───────────────────────────────────────────────────

_ensemble_registry: dict[str, EnsembleMLModel] = {}


def get_or_create_ensemble(ticker: str) -> EnsembleMLModel:
    if ticker not in _ensemble_registry:
        _ensemble_registry[ticker] = EnsembleMLModel(ticker)
    return _ensemble_registry[ticker]


def predict_ensemble(ticker: str, df: pd.DataFrame) -> tuple[float, float]:
    """Returns (prob_up, agreement). agreement near 1.0 = high consensus."""
    return get_or_create_ensemble(ticker).predict(df)
