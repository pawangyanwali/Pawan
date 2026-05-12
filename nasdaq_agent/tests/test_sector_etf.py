"""
Tests for agent/sector_etf.py — sector context and ETF cache.
"""
import pytest
import pandas as pd
from agent.sector_etf import (
    get_sector_context, update_etf_cache, SectorContext, SECTOR_MAP,
)
from tests.conftest import make_ohlcv


def test_get_sector_context_returns_sector_context():
    df = make_ohlcv()
    ctx = get_sector_context("AAPL", df)
    assert isinstance(ctx, SectorContext)

def test_sector_context_has_required_fields():
    df = make_ohlcv()
    ctx = get_sector_context("NVDA", df)
    assert hasattr(ctx, "etf")
    assert hasattr(ctx, "sector_trend")
    assert hasattr(ctx, "sector_change")
    assert hasattr(ctx, "stock_vs_sector")
    assert hasattr(ctx, "score_mult")

def test_score_mult_in_range():
    df = make_ohlcv()
    ctx = get_sector_context("MSFT", df)
    assert 0.5 <= ctx.score_mult <= 2.0, f"score_mult out of range: {ctx.score_mult}"

def test_sector_map_contains_known_tickers():
    assert "AAPL" in SECTOR_MAP
    assert "NVDA" in SECTOR_MAP
    assert "MSFT" in SECTOR_MAP

def test_known_ticker_maps_to_correct_etf():
    # NVDA is a semiconductor → should map to SMH
    assert SECTOR_MAP.get("NVDA") == "SMH"

def test_unknown_ticker_uses_default():
    df = make_ohlcv()
    ctx = get_sector_context("XYZUNKNOWN", df)
    assert ctx.etf in ("QQQ", "XLK", "SMH")  # falls back to some default

def test_update_etf_cache_no_error():
    frames = {
        "QQQ": make_ohlcv(start_price=450.0),
        "SPY": make_ohlcv(start_price=500.0),
        "SMH": make_ohlcv(start_price=200.0),
    }
    update_etf_cache(frames)  # should not raise

def test_sector_context_after_etf_cache_update():
    """After populating the ETF cache, sector context should use ETF data."""
    frames = {
        "QQQ": make_ohlcv(start_price=450.0, trend=0.1),
    }
    update_etf_cache(frames)
    df = make_ohlcv(start_price=100.0, trend=0.05)
    ctx = get_sector_context("XYZUNKNOWN", df)
    assert ctx.sector_trend in ("BULLISH", "BEARISH", "NEUTRAL")

def test_or_operator_bug_fixed():
    """Regression: using `or` on DataFrame raised ValueError. Must not raise."""
    frames = {
        "SMH": make_ohlcv(start_price=200.0),
        # QQQ missing — forces fallback
    }
    update_etf_cache(frames)
    df = make_ohlcv()
    # NVDA maps to SMH which exists — should not raise
    ctx = get_sector_context("NVDA", df)
    assert ctx is not None
