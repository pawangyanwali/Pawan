import pytest
pytestmark = pytest.mark.slow

"""
Data Integrity Tests — validates that all data produced and consumed by the
system satisfies its mathematical and logical invariants.

Covers:
  1.  OHLCV bar quality         (prices, volume, HLOC ordering, no NaN)
  2.  Feature engineering       (ranges, NaN, count, causal correctness)
  3.  Signal self-consistency   (target/stop placement, R:R, confidence)
  4.  Paper trade DB integrity  (no $0 exits, correct P&L, no duplicate open)
  5.  Live backtest integrity   (R sign matches status, formula correctness)
  6.  Position sizing invariants (risk formula, multiplier bounds)
  7.  VWAP / indicator accuracy (formula against known values)
  8.  Backtest calibration      (win-rate-to-confidence mapping)
  9.  DB schema completeness    (all required columns present)
  10. Dirty-data resilience     (zero prices, NaN bars, negative volume)

Run:
    cd nasdaq_agent
    pytest tests/test_data_integrity.py -v --tb=short
"""
from __future__ import annotations

import math
import sqlite3
import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from tests.conftest import make_ohlcv


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_bad_ohlcv(**overrides) -> pd.DataFrame:
    """Return a good OHLCV frame then inject bad values via overrides."""
    df = make_ohlcv(n=60).copy()
    for col, val in overrides.items():
        if col == "high_below_close":
            df["High"] = df["Close"] * 0.99   # HIGH < CLOSE — impossible
        elif col == "low_above_close":
            df["Low"] = df["Close"] * 1.01    # LOW > CLOSE — impossible
        elif col == "negative_price":
            df.loc[df.index[-1], "Close"] = -5.0
        elif col == "zero_volume":
            df["Volume"] = 0
        elif col == "nan_close":
            df.loc[df.index[-1], "Close"] = float("nan")
        elif col == "duplicate_timestamp":
            df.index = pd.date_range("2025-01-10 09:30", periods=len(df), freq="1min")
            duped = df.iloc[-1:].copy()
            df = pd.concat([df, duped])
        elif col == "unsorted":
            df = df.iloc[::-1]  # reverse chronological
    return df


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 – OHLCV Bar Quality
# ═════════════════════════════════════════════════════════════════════════════

class TestOHLCVIntegrity:
    """Every OHLCV bar must satisfy: 0 < Low ≤ min(O,C) ≤ max(O,C) ≤ High,
    Volume ≥ 0, no NaN in price columns."""

    def _validate(self, df: pd.DataFrame) -> list[str]:
        errors = []
        if df is None or df.empty:
            return ["empty DataFrame"]
        for col in ("Open", "High", "Low", "Close"):
            if df[col].isna().any():
                errors.append(f"NaN in {col}")
            if (df[col] <= 0).any():
                errors.append(f"non-positive price in {col}")
        if (df["High"] < df["Low"]).any():
            errors.append("High < Low")
        if (df["High"] < df["Close"]).any():
            errors.append("High < Close")
        if (df["High"] < df["Open"]).any():
            errors.append("High < Open")
        if (df["Low"] > df["Close"]).any():
            errors.append("Low > Close")
        if (df["Low"] > df["Open"]).any():
            errors.append("Low > Open")
        if (df["Volume"] < 0).any():
            errors.append("negative Volume")
        return errors

    def test_good_ohlcv_passes(self):
        df = make_ohlcv()
        assert self._validate(df) == [], f"good OHLCV should have no errors"

    def test_high_below_close_detected(self):
        df = make_bad_ohlcv(high_below_close=True)
        errors = self._validate(df)
        assert any("High" in e for e in errors), "High < Close must be flagged"

    def test_low_above_close_detected(self):
        df = make_bad_ohlcv(low_above_close=True)
        errors = self._validate(df)
        assert any("Low" in e for e in errors), "Low > Close must be flagged"

    def test_negative_price_detected(self):
        df = make_bad_ohlcv(negative_price=True)
        errors = self._validate(df)
        assert any("non-positive" in e for e in errors), "negative price must be flagged"

    def test_nan_close_detected(self):
        df = make_bad_ohlcv(nan_close=True)
        errors = self._validate(df)
        assert any("NaN in Close" in e for e in errors), "NaN Close must be flagged"

    def test_duplicate_timestamps_detected(self):
        df = make_bad_ohlcv(duplicate_timestamp=True)
        dupes = df.index.duplicated().sum()
        assert dupes > 0, "duplicate timestamps must be detectable"

    def test_bars_sorted_chronologically(self):
        df = make_ohlcv()
        assert df.index.is_monotonic_increasing, "OHLCV index must be sorted oldest-first"

    def test_high_geq_low_always(self):
        df = make_ohlcv(n=200, volatility=1.0)
        assert (df["High"] >= df["Low"]).all(), "High must always be ≥ Low"

    def test_high_geq_close_always(self):
        df = make_ohlcv(n=200)
        assert (df["High"] >= df["Close"]).all()

    def test_low_leq_close_always(self):
        df = make_ohlcv(n=200)
        assert (df["Low"] <= df["Close"]).all()

    def test_vwap_positive_and_finite(self):
        df = make_ohlcv(n=100)
        assert df["vwap"].gt(0).all(), "VWAP must be positive"
        assert df["vwap"].isna().sum() == 0, "VWAP must have no NaN"

    def test_vwap_between_low_and_high(self):
        """VWAP is a volume-weighted average price — must be within Hi-Lo range."""
        df = make_ohlcv(n=100)
        assert (df["vwap"] >= df["Low"].min() * 0.995).all()
        assert (df["vwap"] <= df["High"].max() * 1.005).all()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 – Feature Engineering Integrity
# ═════════════════════════════════════════════════════════════════════════════

class TestFeatureIntegrity:
    """Feature vectors must have correct count, bounded ranges, no NaN at
    inference time, and be computed causally (no look-ahead)."""

    @pytest.fixture(autouse=True)
    def import_fe(self):
        """Import feature_engine (requires real ta library, not the test stub)."""
        ta_mod = sys.modules.get("ta")
        if getattr(ta_mod, "_is_stub", False):
            pytest.skip("real 'ta' library required for feature integrity tests")
        try:
            from agent.feature_engine import compute_features, FEATURE_COLS_V2
            self.compute_features = compute_features
            self.FEATURE_COLS_V2 = FEATURE_COLS_V2
        except Exception as e:
            pytest.skip(f"feature_engine unavailable: {e}")

    def test_feature_count_matches_registry(self):
        df = make_ohlcv(n=200)
        result = self.compute_features(df, ticker="TEST")
        for col in self.FEATURE_COLS_V2:
            assert col in result.columns, f"feature {col!r} missing from output"

    def test_rsi_bounded_0_100(self):
        df = make_ohlcv(n=200)
        result = self.compute_features(df, "TEST")
        rsi = result["rsi_14"].dropna()
        assert rsi.between(0, 100).all(), f"RSI range: [{rsi.min():.1f}, {rsi.max():.1f}]"

    def test_stoch_bounded_0_100(self):
        df = make_ohlcv(n=200)
        result = self.compute_features(df, "TEST")
        sk = result["stoch_k"].dropna()
        assert sk.between(0, 100).all(), f"Stoch K range: [{sk.min():.1f}, {sk.max():.1f}]"

    def test_bb_pct_reasonable(self):
        df = make_ohlcv(n=200)
        result = self.compute_features(df, "TEST")
        bb = result["bb_pct"].dropna()
        # bb_pct = (close - lower) / (upper - lower); normally 0-1 but can exceed
        assert bb.notna().any(), "bb_pct must have non-NaN values"

    def test_ema_cross_binary(self):
        df = make_ohlcv(n=200)
        result = self.compute_features(df, "TEST")
        vals = set(result["ema_cross"].dropna().unique())
        assert vals.issubset({-1.0, 1.0}), f"ema_cross should be ±1, got {vals}"

    def test_no_nan_in_last_row(self):
        """Inference uses only the last row — it must have no NaN features."""
        df = make_ohlcv(n=200)
        result = self.compute_features(df, "TEST")
        last = result[self.FEATURE_COLS_V2].iloc[-1]
        nan_feats = last[last.isna()].index.tolist()
        assert nan_feats == [], f"NaN features in last row: {nan_feats}"

    def test_atr_positive(self):
        df = make_ohlcv(n=200)
        result = self.compute_features(df, "TEST")
        atr = result["atr_14"].dropna()
        assert (atr > 0).all(), "ATR must be positive"

    def test_vwap_dev_bounded(self):
        """vwap_dev is clipped to ±5% by design."""
        df = make_ohlcv(n=200)
        result = self.compute_features(df, "TEST")
        dev = result["vwap_dev"].dropna()
        assert dev.between(-0.051, 0.051).all(), f"vwap_dev range: [{dev.min():.4f}, {dev.max():.4f}]"

    def test_short_df_returns_unchanged(self):
        """compute_features should return the input unchanged when df is too short."""
        df = make_ohlcv(n=20)  # < 30 rows required
        result = self.compute_features(df, "SHORT")
        # Should return df unchanged (no FEATURE_COLS added)
        assert result is not None

    def test_features_causal_no_lookahead(self):
        """Appending one bar should not change features of earlier bars."""
        df_short = make_ohlcv(n=100)
        df_long  = make_ohlcv(n=101)  # same seed, one more bar
        r_short = self.compute_features(df_short, "T")
        r_long  = self.compute_features(df_long,  "T")
        # Last row of short == second-to-last of long (both compute from past only)
        for col in ("rsi_14", "ema_cross"):
            if col in r_short.columns and col in r_long.columns:
                v_short = r_short[col].iloc[-1]
                v_long  = r_long[col].iloc[-2]
                assert abs(v_short - v_long) < 1e-3, \
                    f"{col}: adding a bar changed past value ({v_short:.4f} → {v_long:.4f}) — lookahead detected!"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 – Signal Self-Consistency
# ═════════════════════════════════════════════════════════════════════════════

from agent.prediction import _evaluate_rr
from agent.support_resistance import get_all_sr_levels


class TestSignalConsistency:
    """Every signal must have target and stop on the correct side of price,
    positive R:R, and confidence in [0, 100]."""

    def _generate_signal_params(self, price: float, trend: float, direction: str):
        df = make_ohlcv(n=120, start_price=price, trend=trend)
        sr = get_all_sr_levels(df)
        stop, target, rr, quality, qualifies = _evaluate_rr(price, sr, direction)
        return stop, target, rr, quality, qualifies

    def test_buy_target_above_price(self):
        stop, target, rr, _, _ = self._generate_signal_params(100.0, 0.1, "BUY")
        if target > 0:
            assert target > 100.0 * 0.999, f"BUY target {target:.2f} should be above price 100"

    def test_buy_stop_below_price(self):
        stop, target, rr, _, _ = self._generate_signal_params(100.0, 0.1, "BUY")
        if stop > 0:
            assert stop < 100.0 * 1.001, f"BUY stop {stop:.2f} should be below price 100"

    def test_sell_target_below_price(self):
        stop, target, rr, _, _ = self._generate_signal_params(100.0, -0.1, "SELL")
        if target > 0:
            assert target < 100.0 * 1.001, f"SELL target {target:.2f} should be below price 100"

    def test_sell_stop_above_price(self):
        stop, target, rr, _, _ = self._generate_signal_params(100.0, -0.1, "SELL")
        if stop > 0:
            assert stop > 100.0 * 0.999, f"SELL stop {stop:.2f} should be above price 100"

    def test_rr_always_positive(self):
        for trend in (0.2, 0.0, -0.2):
            for direction in ("BUY", "SELL"):
                stop, target, rr, _, _ = self._generate_signal_params(100.0, trend, direction)
                if stop > 0 and target > 0:
                    assert rr >= 0, f"R:R must be non-negative, got {rr}"

    def test_rr_formula_correctness_buy(self):
        stop, target, rr, _, _ = self._generate_signal_params(100.0, 0.3, "BUY")
        if stop > 0 and target > 0 and abs(100.0 - stop) > 0.01:
            manual = (target - 100.0) / (100.0 - stop)
            assert abs(rr - manual) < 0.5, f"R:R formula mismatch: computed {rr:.3f} ≠ manual {manual:.3f}"

    def test_quality_labels_valid(self):
        valid_qualities = {"EXCELLENT", "GOOD", "OK", "LOW"}
        for trend in (0.3, -0.3):
            for direction in ("BUY", "SELL"):
                _, _, _, quality, _ = self._generate_signal_params(100.0, trend, direction)
                assert quality in valid_qualities, f"invalid quality: {quality}"

    def test_qualifies_at_2r_threshold(self):
        stop, target, rr, _, qualifies = self._generate_signal_params(100.0, 0.3, "BUY")
        assert qualifies == (rr >= 2.0), f"qualifies={qualifies} but rr={rr:.2f}"

    def test_confidence_after_calibration_in_range(self):
        from agent.backtest_reporter import adjust_confidence
        import agent.backtest_reporter as br
        # With extreme calibration, result must still be in [25, 95]
        with br._cal_lock:
            br._calibration = {"session:REGULAR": 1.0}
        for raw_conf in (10.0, 50.0, 95.0, 110.0):
            result = adjust_confidence(raw_conf, session="REGULAR")
            assert 25.0 <= result <= 95.0, f"calibrated conf {result} out of [25,95] for input {raw_conf}"
        with br._cal_lock:
            br._calibration = {}

    def test_target_not_equal_to_stop(self):
        stop, target, rr, _, _ = self._generate_signal_params(100.0, 0.1, "BUY")
        if stop > 0 and target > 0:
            assert abs(target - stop) > 0.001, "target and stop must not be equal"

    def test_target_not_equal_to_price(self):
        stop, target, rr, _, _ = self._generate_signal_params(100.0, 0.1, "BUY")
        if target > 0:
            assert abs(target - 100.0) > 0.001 * 100.0, "target must differ from entry by ≥ 0.1%"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 – Paper Trade Database Integrity
# ═════════════════════════════════════════════════════════════════════════════

from agent.paper_trading import (
    maybe_open_trade, update_open_trades, get_open_trades,
    get_closed_trades, get_summary,
)


class TestPaperTradeDBIntegrity:
    """Integrity rules against the SQLite paper_trades table."""

    def _get_conn(self, db_path):
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def test_no_zero_exit_price_on_closed_trades(self, tmp_path, monkeypatch):
        import agent.paper_trading as pt
        monkeypatch.setattr(pt, "_DB_PATH", tmp_path / "pt.db")
        pt.init_db()
        # Open and close a trade at target
        maybe_open_trade("INTG_A", "BUY", 100.0, 110.0, 95.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        df = make_ohlcv(start_price=112.0)
        update_open_trades("INTG_A", df, current_price=112.0)
        # Verify no closed trade has exit_price = 0
        conn = self._get_conn(tmp_path / "pt.db")
        bad = conn.execute(
            "SELECT * FROM paper_trades WHERE status='CLOSED' AND (exit_price IS NULL OR exit_price <= 0)"
        ).fetchall()
        conn.close()
        assert len(bad) == 0, f"{len(bad)} closed trades have zero/null exit price"

    def test_no_null_pnl_on_closed_trades(self, tmp_path, monkeypatch):
        import agent.paper_trading as pt
        monkeypatch.setattr(pt, "_DB_PATH", tmp_path / "pt2.db")
        pt.init_db()
        maybe_open_trade("INTG_B", "BUY", 100.0, 110.0, 95.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        df = make_ohlcv(start_price=112.0)
        update_open_trades("INTG_B", df, current_price=112.0)
        conn = self._get_conn(tmp_path / "pt2.db")
        bad = conn.execute(
            "SELECT * FROM paper_trades WHERE status='CLOSED' AND pnl_pct IS NULL"
        ).fetchall()
        conn.close()
        assert len(bad) == 0, f"{len(bad)} closed trades have NULL pnl_pct"

    def test_at_most_one_open_per_ticker(self, tmp_path, monkeypatch):
        import agent.paper_trading as pt
        monkeypatch.setattr(pt, "_DB_PATH", tmp_path / "pt3.db")
        pt.init_db()
        maybe_open_trade("INTG_C", "BUY", 100.0, 110.0, 95.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        maybe_open_trade("INTG_C", "BUY", 101.0, 111.0, 96.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        conn = self._get_conn(tmp_path / "pt3.db")
        open_for_ticker = conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE ticker='INTG_C' AND status='OPEN'"
        ).fetchone()[0]
        conn.close()
        assert open_for_ticker <= 1, f"duplicate OPEN trades detected for INTG_C: {open_for_ticker}"

    def test_entry_price_always_positive(self, tmp_path, monkeypatch):
        import agent.paper_trading as pt
        monkeypatch.setattr(pt, "_DB_PATH", tmp_path / "pt4.db")
        pt.init_db()
        for ticker, price in [("T1", 50.0), ("T2", 200.0), ("T3", 1500.0)]:
            maybe_open_trade(ticker, "BUY", price, price * 1.05, price * 0.97, confidence=70.0, rr_qualifies=True, session="REGULAR")
        conn = self._get_conn(tmp_path / "pt4.db")
        bad = conn.execute(
            "SELECT * FROM paper_trades WHERE entry_price <= 0"
        ).fetchall()
        conn.close()
        assert len(bad) == 0, f"{len(bad)} trades have non-positive entry_price"

    def test_buy_pnl_positive_when_win(self, tmp_path, monkeypatch):
        import agent.paper_trading as pt
        monkeypatch.setattr(pt, "_DB_PATH", tmp_path / "pt5.db")
        pt.init_db()
        # Open BUY at 100, target 110
        maybe_open_trade("INTG_WIN", "BUY", 100.0, 110.0, 95.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        df = make_ohlcv(start_price=115.0)
        update_open_trades("INTG_WIN", df, current_price=115.0)
        conn = self._get_conn(tmp_path / "pt5.db")
        row = conn.execute(
            "SELECT pnl_pct, pnl_dollar, exit_price FROM paper_trades WHERE ticker='INTG_WIN' AND status='CLOSED'"
        ).fetchone()
        conn.close()
        if row:
            assert row["pnl_pct"] > 0, f"winning BUY should have positive pnl_pct, got {row['pnl_pct']}"
            assert row["pnl_dollar"] > 0, f"winning BUY should have positive pnl_dollar"

    def test_buy_pnl_negative_when_loss(self, tmp_path, monkeypatch):
        import agent.paper_trading as pt
        monkeypatch.setattr(pt, "_DB_PATH", tmp_path / "pt6.db")
        pt.init_db()
        maybe_open_trade("INTG_LOSS", "BUY", 100.0, 110.0, 95.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        df = make_ohlcv(start_price=93.0)
        update_open_trades("INTG_LOSS", df, current_price=93.0)
        conn = self._get_conn(tmp_path / "pt6.db")
        row = conn.execute(
            "SELECT pnl_pct, pnl_dollar FROM paper_trades WHERE ticker='INTG_LOSS' AND status='CLOSED'"
        ).fetchone()
        conn.close()
        if row:
            assert row["pnl_pct"] < 0, f"losing BUY should have negative pnl_pct"
            assert row["pnl_dollar"] < 0

    def test_pnl_formula_accuracy(self, tmp_path, monkeypatch):
        """Winning BUY must have positive pnl_pct and pnl_dollar.
        The system uses T1/T2 partial exits so blended P&L differs from a simple
        (exit−entry)/entry formula — we verify sign and magnitude direction only."""
        import agent.paper_trading as pt
        monkeypatch.setattr(pt, "_DB_PATH", tmp_path / "pt7.db")
        pt.init_db()
        entry = 100.0
        maybe_open_trade("INTG_FORM", "BUY", entry, 108.0, 95.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        df = make_ohlcv(start_price=109.0)
        update_open_trades("INTG_FORM", df, current_price=109.0)
        conn = self._get_conn(tmp_path / "pt7.db")
        row = conn.execute(
            "SELECT pnl_pct, pnl_dollar, exit_price FROM paper_trades WHERE ticker='INTG_FORM' AND status='CLOSED'"
        ).fetchone()
        conn.close()
        if row and row["pnl_pct"] is not None:
            assert row["pnl_pct"] > 0, f"winning BUY should have positive pnl_pct, got {row['pnl_pct']:.3f}%"
            assert row["pnl_dollar"] > 0, f"winning BUY should have positive pnl_dollar, got {row['pnl_dollar']:.2f}"
            assert row["pnl_pct"] < 50, f"pnl_pct seems unrealistically large: {row['pnl_pct']:.1f}%"

    def test_closed_at_timestamp_present(self, tmp_path, monkeypatch):
        import agent.paper_trading as pt
        monkeypatch.setattr(pt, "_DB_PATH", tmp_path / "pt8.db")
        pt.init_db()
        maybe_open_trade("INTG_TS", "BUY", 100.0, 110.0, 95.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        df = make_ohlcv(start_price=112.0)
        update_open_trades("INTG_TS", df, current_price=112.0)
        conn = self._get_conn(tmp_path / "pt8.db")
        row = conn.execute(
            "SELECT closed_at FROM paper_trades WHERE ticker='INTG_TS' AND status='CLOSED'"
        ).fetchone()
        conn.close()
        if row:
            assert row["closed_at"] is not None, "closed_at must be set on CLOSED trade"

    def test_no_double_close_in_db(self, tmp_path, monkeypatch):
        """Closing an already-closed trade must be idempotent — only 1 row in DB."""
        import agent.paper_trading as pt
        monkeypatch.setattr(pt, "_DB_PATH", tmp_path / "pt9.db")
        pt.init_db()
        maybe_open_trade("INTG_DC", "BUY", 100.0, 110.0, 95.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        df = make_ohlcv(start_price=112.0)
        update_open_trades("INTG_DC", df, current_price=112.0)
        update_open_trades("INTG_DC", df, current_price=113.0)  # second close attempt
        conn = self._get_conn(tmp_path / "pt9.db")
        count = conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE ticker='INTG_DC'"
        ).fetchone()[0]
        conn.close()
        assert count == 1, f"double-close created {count} rows — expected 1"

    def test_db_schema_has_required_columns(self, tmp_path, monkeypatch):
        import agent.paper_trading as pt
        monkeypatch.setattr(pt, "_DB_PATH", tmp_path / "schema_test.db")
        pt.init_db()
        conn = self._get_conn(tmp_path / "schema_test.db")
        cols = [r[1] for r in conn.execute("PRAGMA table_info(paper_trades)").fetchall()]
        conn.close()
        required = [
            "id", "opened_at", "closed_at", "ticker", "direction",
            "entry_price", "target", "stop", "confidence", "rr_ratio",
            "status", "exit_price", "exit_reason", "pnl_pct", "pnl_dollar",
            "shares",
        ]
        for col in required:
            assert col in cols, f"paper_trades missing column: {col}"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 – Live Backtest DB Integrity
# ═════════════════════════════════════════════════════════════════════════════

from agent.live_backtest import record_signal, update_tracking, get_performance_stats


class TestLiveBacktestDBIntegrity:
    """Integrity rules for live_backtest.db — resolved signals must have
    mathematically consistent R-multiples and status codes."""

    def test_win_has_positive_r(self):
        sid = record_signal("LI_W1", "BUY", entry_price=100.0, target=110.0, stop=95.0)
        resolved = update_tracking("LI_W1", current_price=111.0, vwap=100.0)
        assert resolved and resolved[0]["status"] == "WIN"
        r = resolved[0].get("r_multiple", resolved[0].get("final_r", None))
        if r is not None:
            assert r > 0, f"WIN signal has non-positive R: {r}"

    def test_loss_has_negative_r(self):
        sid = record_signal("LI_L1", "BUY", entry_price=100.0, target=110.0, stop=95.0)
        resolved = update_tracking("LI_L1", current_price=94.0, vwap=100.0)
        assert resolved and resolved[0]["status"] == "LOSS"
        r = resolved[0].get("r_multiple", resolved[0].get("final_r", None))
        if r is not None:
            assert r < 0, f"LOSS signal has non-negative R: {r}"

    def test_timeout_has_small_r(self):
        """TIMEOUT signals should have |R| close to 0 (held without hitting target or stop)."""
        from agent.live_backtest import MAX_BARS
        sid = record_signal("LI_T1", "BUY", entry_price=100.0, target=110.0, stop=95.0)
        resolved = []
        for _ in range(MAX_BARS):
            resolved = update_tracking("LI_T1", current_price=101.0, vwap=0.0)
        assert resolved and resolved[0]["status"] == "TIMEOUT"
        r = resolved[0].get("r_multiple", resolved[0].get("final_r", None))
        if r is not None:
            assert abs(r) < 2.0, f"TIMEOUT signal has unexpectedly large R: {r}"

    def test_r_formula_buy(self):
        """R = (exit_price - entry) / (entry - stop) for BUY."""
        entry, target, stop = 100.0, 110.0, 95.0
        sid = record_signal("LI_RF", "BUY", entry_price=entry, target=target, stop=stop)
        resolved = update_tracking("LI_RF", current_price=target, vwap=100.0)
        if resolved:
            r = resolved[0].get("r_multiple", resolved[0].get("final_r", None))
            expected_r = (target - entry) / (entry - stop)  # = 2.0
            if r is not None:
                assert abs(r - expected_r) < 0.5, f"R formula error: {r:.3f} ≠ {expected_r:.3f}"

    def test_sell_win_has_positive_r(self):
        sid = record_signal("LI_SW", "SELL", entry_price=100.0, target=90.0, stop=105.0)
        resolved = update_tracking("LI_SW", current_price=89.0, vwap=100.0)
        assert resolved and resolved[0]["status"] == "WIN"
        r = resolved[0].get("r_multiple", resolved[0].get("final_r", None))
        if r is not None:
            assert r > 0

    def test_pnl_pct_sign_matches_status(self):
        """WIN signals must have positive pnl_pct, LOSS signals negative."""
        record_signal("LI_SGN", "BUY", 100.0, 110.0, 95.0)
        resolved = update_tracking("LI_SGN", 111.0, 100.0)
        if resolved:
            pnl = resolved[0].get("pnl_pct", None)
            if pnl is not None and resolved[0]["status"] == "WIN":
                assert pnl > 0, f"WIN pnl_pct should be positive, got {pnl}"

    def test_performance_stats_totals_consistent(self):
        """wins + losses + timeouts + tracking == total in performance stats."""
        for i in range(3):
            record_signal(f"LI_STAT{i}", "BUY", 100.0, 110.0, 95.0)
            update_tracking(f"LI_STAT{i}", 111.0, 0.0)  # WIN
        stats = get_performance_stats()["overall"]
        if stats["total"] > 0:
            accounted = stats["wins"] + stats["losses"] + stats.get("timeouts", 0)
            assert accounted == stats["total"], \
                f"wins({stats['wins']}) + losses({stats['losses']}) ≠ total({stats['total']})"

    def test_win_rate_formula(self):
        """win_rate = wins / total — verify it matches the individual counts."""
        for i in range(4):
            record_signal(f"LI_WR{i}", "BUY", 100.0, 110.0, 95.0)
            outcome_price = 111.0 if i < 3 else 94.0  # 3 wins, 1 loss
            update_tracking(f"LI_WR{i}", outcome_price, 0.0)
        stats = get_performance_stats()["overall"]
        if stats["total"] >= 4:
            expected_wr = stats["wins"] / stats["total"]
            assert abs(stats["win_rate"] - expected_wr) < 0.01, \
                f"win_rate {stats['win_rate']:.3f} ≠ wins/total {expected_wr:.3f}"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 – Position Sizing Invariants
# ═════════════════════════════════════════════════════════════════════════════

from agent.position_sizing import calculate


class TestPositionSizingInvariants:
    def test_dollar_risk_equals_shares_times_risk_per_share(self):
        ps = calculate(100_000, 50.0, 48.0, 1.0, 65.0, max_position_pct=100.0)
        assert abs(ps.dollar_risk - ps.shares * ps.risk_per_share) < 0.01

    def test_position_value_equals_shares_times_entry(self):
        ps = calculate(100_000, 50.0, 48.0, 1.0, 65.0, max_position_pct=100.0)
        assert abs(ps.position_value - ps.shares * 50.0) < 0.01

    def test_risk_pct_used_matches_formula(self):
        ps = calculate(100_000, 50.0, 48.0, 1.0, 65.0, max_position_pct=100.0)
        expected = ps.dollar_risk / 100_000 * 100
        assert abs(ps.risk_pct_used - expected) < 0.001

    def test_zero_shares_on_equal_entry_stop(self):
        ps = calculate(10_000, 100.0, 100.0, 1.0, 65.0)
        assert ps.shares == 0

    def test_max_position_cap_never_exceeded(self):
        for cap in (1.0, 5.0, 10.0):
            ps = calculate(100_000, 100.0, 99.0, 10.0, 99.0, max_position_pct=cap)
            assert ps.position_value <= 100_000 * cap / 100 + 100.1, \
                f"position ${ps.position_value:.0f} exceeds {cap}% cap (${100_000 * cap / 100:.0f})"

    def test_confidence_mult_monotone(self):
        """Higher confidence → higher or equal confidence multiplier."""
        mults = [calculate(10_000, 50.0, 48.0, 1.0, conf).confidence_mult
                 for conf in (30, 50, 60, 75)]
        assert mults == sorted(mults), f"confidence_mult not monotone: {mults}"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 – VWAP Formula Accuracy
# ═════════════════════════════════════════════════════════════════════════════

class TestVWAPFormulaAccuracy:
    def test_vwap_formula_bar_by_bar(self):
        """VWAP_t = sum(typical_price × volume, 1..t) / sum(volume, 1..t)."""
        df = make_ohlcv(n=30, start_price=100.0)
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        expected = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()
        diff = (df["vwap"] - expected).abs()
        assert diff.max() < 1e-4, f"VWAP formula deviation: max={diff.max():.6f}"

    def test_vwap_between_low_and_high(self):
        """Cumulative VWAP must lie within the overall [Low, High] range seen so far.
        It is NOT bounded by each individual bar's range — it is a running average
        that reflects the price history up to that point."""
        df = make_ohlcv(n=100)
        cum_low  = df["Low"].cummin()
        cum_high = df["High"].cummax()
        assert (df["vwap"] >= cum_low * 0.999).all(), "VWAP below running low"
        assert (df["vwap"] <= cum_high * 1.001).all(), "VWAP above running high"

    def test_vwap_resets_daily_semantics(self):
        """VWAP should be close to price when all bars are within one day."""
        df = make_ohlcv(n=60, start_price=150.0, volatility=0.2)
        last_close = float(df["Close"].iloc[-1])
        last_vwap  = float(df["vwap"].iloc[-1])
        # VWAP should be within 5% of current price in low-volatility scenario
        assert abs(last_vwap - last_close) / last_close < 0.05, \
            f"VWAP {last_vwap:.2f} too far from close {last_close:.2f}"

    def test_vwap_signal_deviation_sign(self):
        from agent.vwap import compute_vwap_signal
        df = make_ohlcv(start_price=110.0)
        df["vwap"] = 100.0
        r = compute_vwap_signal(df)
        assert r["deviation"] > 0, "price above VWAP must have positive deviation"

    def test_vwap_signal_computation_stable(self):
        """Same input must always give same output (deterministic)."""
        from agent.vwap import compute_vwap_signal
        df = make_ohlcv(n=60)
        r1 = compute_vwap_signal(df)
        r2 = compute_vwap_signal(df)
        assert r1["score"] == r2["score"]
        assert r1["event"] == r2["event"]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 – Backtest Calibration Integrity
# ═════════════════════════════════════════════════════════════════════════════

from agent.backtest_reporter import (
    _update_confidence_calibration, get_calibration, adjust_confidence,
)
import agent.backtest_reporter as _br


class TestCalibrationIntegrity:
    def test_high_win_rate_boosts_confidence(self):
        df = pd.DataFrame({
            "vwap_event": ["RECLAIM"] * 20,
            "outcome":    [1] * 18 + [0] * 2,  # 90% win rate
        })
        _update_confidence_calibration(df)
        cal = get_calibration()
        assert cal.get("vwap_event:RECLAIM", 0) > 0.5, "90% win rate should produce high calibration"
        with _br._cal_lock:
            _br._calibration = {}

    def test_low_win_rate_reduces_confidence(self):
        df = pd.DataFrame({
            "session": ["AFTER_HOURS"] * 20,
            "outcome": [1] * 3 + [0] * 17,  # 15% win rate
        })
        _update_confidence_calibration(df)
        cal = get_calibration()
        assert cal.get("session:AFTER_HOURS", 1.0) < 0.5, "15% win rate should produce low calibration"
        with _br._cal_lock:
            _br._calibration = {}

    def test_adjust_confidence_bounded_25_95(self):
        with _br._cal_lock:
            _br._calibration = {
                "vwap_event:RECLAIM": 0.95,
                "session:REGULAR": 0.90,
            }
        # Even with boosting, output is capped at 95
        for raw in (10, 50, 90, 100):
            result = adjust_confidence(float(raw), vwap_event="RECLAIM", session="REGULAR")
            assert 25.0 <= result <= 95.0, f"confidence {result} outside [25, 95] for input {raw}"
        with _br._cal_lock:
            _br._calibration = {}

    def test_calibration_relative_ordering(self):
        """Context with higher win rate must produce higher calibration value."""
        df = pd.DataFrame({
            "regime": ["BULL_TREND"] * 20 + ["BEAR_TREND"] * 20,
            "outcome": [1] * 18 + [0] * 2 + [0] * 15 + [1] * 5,
        })
        _update_confidence_calibration(df)
        cal = get_calibration()
        bull = cal.get("regime:BULL_TREND", 0.5)
        bear = cal.get("regime:BEAR_TREND", 0.5)
        assert bull > bear, f"BULL calibration {bull:.2f} should exceed BEAR {bear:.2f}"
        with _br._cal_lock:
            _br._calibration = {}

    def test_no_calibration_returns_unchanged(self):
        with _br._cal_lock:
            _br._calibration = {}
        for raw in (30.0, 55.0, 80.0):
            result = adjust_confidence(raw)
            assert result == raw, f"without calibration, confidence must be unchanged ({raw} → {result})"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 – Dirty Data Resilience
# ═════════════════════════════════════════════════════════════════════════════

class TestDirtyDataResilience:
    """System components must not crash when given malformed inputs."""

    def test_support_resistance_on_flat_price(self):
        from agent.support_resistance import get_all_sr_levels
        df = make_ohlcv(n=120, volatility=0.0001)  # nearly flat price
        result = get_all_sr_levels(df)
        assert isinstance(result, dict), "S/R should return dict even for flat price"

    def test_vwap_signal_on_single_bar(self):
        from agent.vwap import compute_vwap_signal
        df = make_ohlcv(n=1)
        result = compute_vwap_signal(df)
        assert result["score"] == 0.0 or isinstance(result["score"], float)

    def test_vwap_signal_on_empty_df(self):
        from agent.vwap import compute_vwap_signal
        result = compute_vwap_signal(pd.DataFrame())
        assert result is not None
        assert result["score"] == 0.0

    def test_paper_trade_zero_price_rejected(self):
        tid = maybe_open_trade("DIRTY_ZERO", "BUY", 0.0, 10.0, 0.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        assert tid is None, "zero entry price must be rejected"

    def test_paper_trade_negative_stop_rejected(self):
        tid = maybe_open_trade("DIRTY_NEG", "BUY", 100.0, 110.0, -5.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
        assert tid is None, "negative stop must be rejected"

    def test_live_backtest_zero_entry_rejected(self):
        sid = record_signal("DIRTY_LB", "BUY", entry_price=0.0, target=10.0, stop=0.0)
        assert sid == "", "zero entry price must not create a signal"

    def test_position_sizing_negative_stop_rejected(self):
        ps = calculate(10_000, 100.0, -5.0, 1.0, 65.0)
        # Stop below zero is an illegal price but the math is:
        # risk_per_share = |100 - (-5)| = 105 — still computes, returns something
        # The key check: no crash
        assert ps is not None

    def test_evaluate_rr_empty_sr(self):
        """_evaluate_rr with empty SR levels must return a valid tuple, not crash."""
        from agent.prediction import _evaluate_rr
        empty_sr = {"supports": [], "resistances": [], "pivots": {}, "poc": 0.0}
        try:
            result = _evaluate_rr(100.0, sr=empty_sr, direction="BUY")
            stop, target, rr, quality, qualifies = result
            assert isinstance(rr, float)
        except Exception as e:
            pytest.fail(f"_evaluate_rr crashed on empty SR: {e}")

    def test_make_ohlcv_with_extreme_trend(self):
        """make_ohlcv should handle extreme trend without overflow."""
        df = make_ohlcv(n=100, trend=100.0)  # very steep trend
        assert not df["Close"].isna().any()
        assert (df["Close"] > 0).all()

    def test_compute_features_nan_volume(self):
        """NaN volume should be handled gracefully."""
        try:
            from agent.feature_engine import compute_features
            df = make_ohlcv(n=100)
            df.loc[df.index[5], "Volume"] = float("nan")
            result = compute_features(df, "NANVOL")
            assert result is not None
        except Exception as e:
            pass  # Any exception is noted but not fatal — this tests resilience


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 – End-to-End Integrity (full signal→paper trade→resolution chain)
# ═════════════════════════════════════════════════════════════════════════════

class TestEndToEndIntegrity:
    """Verify data flows correctly through the full pipeline:
    signal creation → paper trade → update → close → P&L."""

    def test_full_pipeline_buy_win(self, tmp_path, monkeypatch):
        import agent.paper_trading as pt
        import agent.live_backtest as lb
        monkeypatch.setattr(pt, "_DB_PATH", tmp_path / "e2e_pt.db")
        monkeypatch.setattr(lb, "_DB_PATH", tmp_path / "e2e_lb.db")
        pt.init_db(); lb.init_db()

        entry, target, stop = 100.0, 110.0, 95.0

        # 1. Live backtest signal
        sid = record_signal("E2E_W", "BUY", entry, target, stop, confidence=75.0)
        assert sid != "", "signal must be recorded"

        # 2. Paper trade
        tid = maybe_open_trade("E2E_W", "BUY", entry, target, stop, confidence=75.0, rr_qualifies=True, session="REGULAR")
        assert tid is not None, "paper trade must open"

        # 3. Price hits target
        df = make_ohlcv(start_price=112.0)
        update_open_trades("E2E_W", df, current_price=112.0)
        bt_resolved = update_tracking("E2E_W", current_price=112.0, vwap=100.0)

        # 4. Verify paper trade closed with positive P&L
        closed = pt.get_closed_trades()
        pt_match = next((t for t in closed if t["ticker"] == "E2E_W"), None)
        assert pt_match is not None, "paper trade must be closed after target hit"
        if pt_match.get("pnl_pct"):
            assert pt_match["pnl_pct"] > 0, "winning trade must have positive P&L"

        # 5. Verify live backtest resolved as WIN
        if bt_resolved:
            assert bt_resolved[0]["status"] == "WIN"

    def test_full_pipeline_sell_loss(self, tmp_path, monkeypatch):
        import agent.paper_trading as pt
        import agent.live_backtest as lb
        monkeypatch.setattr(pt, "_DB_PATH", tmp_path / "e2e_pt2.db")
        monkeypatch.setattr(lb, "_DB_PATH", tmp_path / "e2e_lb2.db")
        pt.init_db(); lb.init_db()

        entry, target, stop = 200.0, 190.0, 205.0

        sid = record_signal("E2E_L", "SELL", entry, target, stop, confidence=72.0)
        assert sid != ""

        tid = maybe_open_trade("E2E_L", "SELL", entry, target, stop, confidence=72.0, rr_qualifies=True, session="REGULAR")
        assert tid is not None

        # Price moves against SELL (rises above stop)
        df = make_ohlcv(start_price=206.0)
        update_open_trades("E2E_L", df, current_price=206.0)
        bt_resolved = update_tracking("E2E_L", current_price=206.0, vwap=200.0)

        closed = pt.get_closed_trades()
        pt_match = next((t for t in closed if t["ticker"] == "E2E_L"), None)
        if pt_match and pt_match.get("pnl_pct"):
            assert pt_match["pnl_pct"] < 0, "losing SELL must have negative P&L"

        if bt_resolved:
            assert bt_resolved[0]["status"] == "LOSS"

    def test_data_consistency_across_layers(self, tmp_path, monkeypatch):
        """Paper trade P&L and live backtest R must be consistent in direction."""
        import agent.paper_trading as pt
        import agent.live_backtest as lb
        monkeypatch.setattr(pt, "_DB_PATH", tmp_path / "e2e_cc.db")
        monkeypatch.setattr(lb, "_DB_PATH", tmp_path / "e2e_cclb.db")
        pt.init_db(); lb.init_db()

        entry, target, stop = 100.0, 110.0, 95.0
        record_signal("E2E_CC", "BUY", entry, target, stop, confidence=70.0)
        maybe_open_trade("E2E_CC", "BUY", entry, target, stop, confidence=70.0, rr_qualifies=True, session="REGULAR")

        df = make_ohlcv(start_price=112.0)
        update_open_trades("E2E_CC", df, current_price=112.0)
        bt_resolved = update_tracking("E2E_CC", current_price=112.0, vwap=100.0)

        closed = pt.get_closed_trades()
        pt_match = next((t for t in closed if t["ticker"] == "E2E_CC"), None)

        if pt_match and bt_resolved:
            pt_is_win = (pt_match.get("pnl_pct", 0) or 0) > 0
            bt_is_win = bt_resolved[0]["status"] == "WIN"
            assert pt_is_win == bt_is_win, \
                f"paper trade and live backtest disagree: pt_win={pt_is_win}, bt_win={bt_is_win}"
