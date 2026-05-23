"""
Shared fixtures for all tests.
"""
import os
import sys
import types
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure nasdaq_agent is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── ta library stub ────────────────────────────────────────────────────────────
# Install a minimal stub when the real `ta` library is not available so that
# test_comprehensive.py can import main.py (which chains through ml_model →
# feature_engine → ta).  test_prediction.py / test_technical.py already guard
# themselves with pytest.importorskip("ta") — we tell pytest to skip those
# files when only the stub is available by using collect_ignore_glob.

_TA_IS_REAL = False
try:
    import ta as _ta_check
    _TA_IS_REAL = hasattr(getattr(_ta_check, "trend", None), "ema_indicator")
except ImportError:
    pass

if not _TA_IS_REAL and "ta" not in sys.modules:
    _ta_stub = types.ModuleType("ta")
    _ta_stub._is_stub = True
    for _sub in ("trend", "momentum", "volatility", "volume", "others"):
        _m = types.ModuleType(f"ta.{_sub}")
        _ta_stub.__dict__[_sub] = _m
        sys.modules[f"ta.{_sub}"] = _m
    sys.modules["ta"] = _ta_stub

# Skip test files that require real ta when only the stub is present
collect_ignore: list[str] = []
if not _TA_IS_REAL:
    _tests_dir = Path(__file__).parent
    collect_ignore = [
        str(_tests_dir / "test_prediction.py"),
        str(_tests_dir / "test_technical.py"),
    ]

# ── Point SQLite DBs to temp files during tests ───────────────────────────────
@pytest.fixture(autouse=True)
def tmp_db_paths(tmp_path, monkeypatch):
    """Redirect all SQLite databases to temp files so tests are isolated."""
    import agent.live_backtest as lb
    import agent.paper_trading as pt
    import agent.signal_tracker as st

    monkeypatch.setattr(lb, "_DB_PATH", tmp_path / "live_backtest.db")
    monkeypatch.setattr(pt, "_DB_PATH", tmp_path / "paper_trades.db")
    monkeypatch.setattr(st, "_DB_PATH", tmp_path / "signal_history.db")

    lb.init_db()
    pt.init_db()
    st.init_db()


# ── Synthetic OHLCV DataFrame factory ─────────────────────────────────────────
def make_ohlcv(
    n: int = 60,
    start_price: float = 100.0,
    trend: float = 0.0,       # drift per bar (e.g. 0.05 = rising)
    volatility: float = 0.5,  # std dev of random noise per bar
    volume: int = 1_000_000,
) -> pd.DataFrame:
    """Return a realistic OHLCV DataFrame with `n` 1-minute bars."""
    rng = np.random.default_rng(42)
    closes = [start_price]
    for _ in range(n - 1):
        closes.append(closes[-1] + trend + rng.normal(0, volatility))
    closes = np.array(closes)

    noise = rng.uniform(0.1, 0.5, n)
    highs  = closes + noise
    lows   = closes - noise
    opens  = np.clip(np.roll(closes, 1), lows, highs)
    opens[0] = start_price

    idx = pd.date_range("2025-01-10 09:30", periods=n, freq="1min")
    df = pd.DataFrame({
        "Open":   np.round(opens, 4),
        "High":   np.round(highs, 4),
        "Low":    np.round(lows,  4),
        "Close":  np.round(closes, 4),
        "Volume": np.full(n, volume, dtype=int),
    }, index=idx)
    # Add a basic VWAP column (cumulative typical price × volume / cumulative volume)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    df["vwap"] = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df


@pytest.fixture
def ohlcv():
    return make_ohlcv()


@pytest.fixture
def ohlcv_rising():
    return make_ohlcv(trend=0.10)


@pytest.fixture
def ohlcv_falling():
    return make_ohlcv(trend=-0.10)
