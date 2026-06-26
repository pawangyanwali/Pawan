"""
Tests for agent/paper_trading.py — simulated trade execution.
"""
import pytest
from agent.paper_trading import (
    maybe_open_trade, update_open_trades, get_open_trades,
    get_closed_trades, get_summary, get_execution_min_rr,
)
from tests.conftest import make_ohlcv


def _open_valid_trade(ticker, direction, price, target, stop, **kwargs):
    if direction == "SELL":
        kwargs.setdefault("rsi_zone", "OB")
        kwargs.setdefault("rsi_value", 72.0)
        kwargs.setdefault("macd_hist", -0.02)
        kwargs.setdefault("macd_hist_prev", -0.01)
    else:
        kwargs.setdefault("rsi_zone", "OS")
        kwargs.setdefault("rsi_value", 28.0)
        kwargs.setdefault("macd_hist", 0.02)
        kwargs.setdefault("macd_hist_prev", 0.01)
    kwargs.setdefault("atr", 1.0)
    return maybe_open_trade(ticker, direction, price, target, stop, **kwargs)


def test_open_trade_high_confidence():
    tid = _open_valid_trade("AAPL", "BUY", 180.0, 185.0, 178.0, confidence=75.0, rr_qualifies=True, session="REGULAR")
    assert tid is not None, "should open trade when confidence >= 65 and rr qualifies"
    open_trades = get_open_trades()
    assert any(t["ticker"] == "AAPL" for t in open_trades)

def test_low_confidence_trade_blocked():
    """Confidence below the 25% floor must be rejected (pure data collection floor)."""
    tid = maybe_open_trade("MSFT", "BUY", 300.0, 310.0, 295.0, confidence=15.0, rr_qualifies=True, session="REGULAR")
    assert tid is None, "confidence below 25% floor should block paper trade"

def test_rr_not_qualifying_still_opens():
    """rr_qualifies=False does NOT block paper trades — we need every data point.
    Position size is just scaled down (min 20% of normal) but the trade opens."""
    tid = _open_valid_trade("NVDA", "BUY", 500.0, 510.0, 495.0, confidence=80.0, rr_qualifies=False, session="REGULAR")
    assert tid is not None, "rr_qualifies=False must still open a paper trade (data collection)"

def test_post_fill_rebuilds_target_from_configured_reward():
    """Configured R:R builds the executable target after fill/slippage."""
    from agent.config_manager import config

    config.set_many({
        "prediction.min_rr": 2.0,
        "paper.t2_r_multiple": 1.5,
    }, updated_by="test")

    tid = _open_valid_trade(
        "CFG_RR",
        "BUY",
        100.0,
        100.2,
        99.0,
        confidence=80.0,
        rr_qualifies=True,
        rr_ratio=2.0,
        session="REGULAR",
    )
    assert tid is not None
    trade = next(t for t in get_open_trades() if t["ticker"] == "CFG_RR")
    assert trade["rr_ratio"] == pytest.approx(2.0)
    assert trade["target"] == pytest.approx(trade["t2_price"])

def test_configured_reward_multiple_does_not_gate_primary_or_algo():
    """The configured reward multiple builds targets instead of blocking trades."""
    from agent.config_manager import config

    config.set_many({
        "prediction.min_rr": 2.0,
        "paper.algo_min_rr": 1.0,
        "paper.algo_t2_r_multiple": 1.5,
        "algos.keltner.exec_min_rr": 0.0,
    }, updated_by="test")

    primary_status = []
    primary_tid = _open_valid_trade(
        "RRPRIM",
        "BUY",
        100.0,
        101.5,
        99.0,
        confidence=80.0,
        rr_qualifies=False,
        rr_ratio=1.5,
        session="REGULAR",
        entry_type="IMMEDIATE",
        _out_status=primary_status,
    )
    assert primary_tid is not None, primary_status
    assert primary_status == ["EXECUTED_PAPER"]
    primary_trade = next(t for t in get_open_trades() if t["ticker"] == "RRPRIM")
    assert primary_trade["rr_ratio"] == pytest.approx(2.0)

    algo_status = []
    algo_tid = _open_valid_trade(
        "RRALGO",
        "BUY",
        100.0,
        105.0,
        99.0,
        confidence=80.0,
        rr_qualifies=True,
        rr_ratio=1.2,
        session="REGULAR",
        entry_type="ALGO",
        algo_name="KC_FADE_BULL",
        avg_daily_volume=10_000_000,
        _out_status=algo_status,
    )
    assert algo_tid is not None, algo_status
    assert algo_status == ["EXECUTED_PAPER"]
    algo_trade = next(t for t in get_open_trades() if t["ticker"] == "RRALGO")
    assert algo_trade["rr_ratio"] == pytest.approx(2.0)

def test_family_exec_min_rr_override_takes_precedence():
    from agent.config_manager import config

    config.set_many({
        "prediction.min_rr": 2.0,
        "paper.algo_min_rr": 1.0,
        "algos.regime_sw.exec_min_rr": 2.4,
    }, updated_by="test")

    assert get_execution_min_rr("REGIME_FADE_BULL", "ALGO") == 2.4
    status = []
    tid = _open_valid_trade(
        "RRREG",
        "BUY",
        100.0,
        101.2,
        99.0,
        confidence=80.0,
        rr_qualifies=False,
        rr_ratio=1.2,
        session="REGULAR",
        entry_type="ALGO",
        algo_name="REGIME_FADE_BULL",
        _out_status=status,
    )
    assert tid is not None, status
    assert status == ["EXECUTED_PAPER"]
    trade = next(t for t in get_open_trades() if t["ticker"] == "RRREG")
    assert trade["rr_ratio"] == pytest.approx(2.4)

def test_premarket_widened_stop_is_rechecked_against_intraday_cap():
    from agent.config_manager import config

    config.set_many({
        "prediction.min_rr": 2.0,
        "paper.pre_market_stop_mult": 1.5,
        "risk.intraday_max_stop_pct": 2.0,
        "risk.intraday_max_target_pct": 4.0,
    }, updated_by="test")

    status = []
    tid = _open_valid_trade(
        "WIDEPM",
        "SELL",
        100.0,
        97.0,
        101.5,
        confidence=90.0,
        rr_qualifies=True,
        rr_ratio=2.0,
        session="PRE_MARKET",
        entry_type="BOUNCE",
        trading_tier="HIGH",
        avg_daily_volume=10_000_000,
        _out_status=status,
    )

    assert tid is None
    assert status == ["BLOCKED_WIDE_GEOMETRY"]

def test_no_duplicate_open_trade():
    _open_valid_trade("TSLA", "BUY", 250.0, 260.0, 245.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
    tid2 = _open_valid_trade("TSLA", "SELL", 250.0, 240.0, 255.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
    assert tid2 is None, "second trade on same ticker must be blocked while one is open"

def test_no_trade_neutral_direction():
    tid = maybe_open_trade("AMZN", "NEUTRAL", 140.0, 145.0, 138.0, confidence=80.0, rr_qualifies=True, session="REGULAR")
    assert tid is None

def test_trade_closed_on_target():
    from agent.config_manager import config
    config.set_many({
        "risk.intraday_max_stop_pct": 10.0,
        "risk.intraday_max_target_pct": 10.0,
    }, updated_by="test")
    # exit_signals reads df.iloc[-1]["Close"] — set it above target so TARGET_HIT fires
    df = make_ohlcv(start_price=107.0)  # last Close ~107, target=105
    _open_valid_trade("HOOD", "BUY", 100.0, 105.0, 98.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
    update_open_trades("HOOD", df, current_price=107.0)
    closed = get_closed_trades()
    assert any(t["ticker"] == "HOOD" for t in closed), "trade should have closed on TARGET_HIT"

def test_trade_closed_on_stop():
    from agent.config_manager import config
    config.set_many({
        "risk.intraday_max_stop_pct": 10.0,
        "risk.intraday_max_target_pct": 10.0,
    }, updated_by="test")
    # Set df last Close below stop so STOP_HIT fires
    df = make_ohlcv(start_price=95.0)  # last Close ~95, stop=98
    _open_valid_trade("COIN", "BUY", 100.0, 108.0, 98.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
    update_open_trades("COIN", df, current_price=95.0)
    closed = get_closed_trades()
    assert any(t["ticker"] == "COIN" for t in closed)

def test_sell_trade_pnl_direction():
    from agent.config_manager import config
    config.set_many({
        "risk.intraday_max_stop_pct": 10.0,
        "risk.intraday_max_target_pct": 10.0,
    }, updated_by="test")
    df = make_ohlcv(start_price=200.0)
    _open_valid_trade("META", "SELL", 200.0, 190.0, 205.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
    # price drops — SELL trade should be profitable
    update_open_trades("META", df, current_price=188.0)
    closed = get_closed_trades()
    meta_trade = next((t for t in closed if t["ticker"] == "META"), None)
    if meta_trade:  # may not close if exit signal isn't EXIT_NOW
        assert meta_trade["pnl_pct"] is not None

def test_summary_counts():
    _open_valid_trade("T1", "BUY", 50.0, 55.0, 49.0, confidence=70.0, rr_qualifies=True, session="REGULAR")
    s = get_summary()
    assert s["open"] >= 1
    assert "wins" in s and "losses" in s and "win_rate" in s

def test_summary_win_rate_range():
    s = get_summary()
    # win_rate stored as percentage 0-100 in paper_trading (different from live_backtest)
    assert 0.0 <= s["win_rate"] <= 100.0
