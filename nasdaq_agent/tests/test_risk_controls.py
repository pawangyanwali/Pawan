"""
Comprehensive tests for agent/risk_controls.py

Covers:
  - check_circuit_breaker
  - record_trade_outcome
  - get_profit_protect_state
  - get_portfolio_heat
  - check_portfolio_heat
  - check_sector_concentration
  - check_session_block
  - update_volatility_state / check_volatility_halt
  - check_max_daily_trades
  - get_drawdown_throttle
  - can_open_trade
  - get_risk_status
  - get_sector
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import date
from unittest.mock import patch, MagicMock

import pytest

# Ensure the project root is on the path so `config` is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

import agent.risk_controls as rc
from config import (
    DAILY_LOSS_HALT_PCT,
    DAILY_PROFIT_MAX_USD,
    DAILY_PROFIT_TARGET_USD,
    PROFIT_PROTECT_DRAWDOWN,
    PROFIT_PROTECT_MIN_CONF,
    PROFIT_PROTECT_SIZE_MULT,
    MAX_CONSECUTIVE_LOSSES,
    COOLDOWN_AFTER_LOSSES,
    VOLATILITY_HALT_ATR_MULT,
    DEFAULT_ACCOUNT_SIZE,
    MAX_DAILY_TRADES,
    DRAWDOWN_THROTTLE_1_PCT,
    DRAWDOWN_THROTTLE_2_PCT,
    MAX_PORTFOLIO_HEAT_PCT,
)

# ---------------------------------------------------------------------------
# Shared autouse fixture — resets every piece of global state before each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_risk_state():
    """Reset all module-level globals before (and after) every test.

    IMPORTANT: _circuit_date is set to date.today() (not None) so that
    _reset_if_new_day() inside the module does NOT trigger a wipe of state
    that tests have carefully pre-set.
    """
    rc._circuit_open = False
    rc._circuit_reason = ""
    rc._circuit_date = date.today()   # prevent _reset_if_new_day from wiping state
    rc._circuit_pnl_based = False
    rc._warning_issued = False
    rc._cooldown_until = 0.0
    rc._consecutive_losses = 0
    rc._peak_daily_pnl = 0.0
    rc._volatility_halted = False
    rc._volatility_reason = ""
    rc._volatility_atr_ratio = 0.0
    yield
    rc._circuit_open = False
    rc._circuit_date = date.today()
    rc._circuit_pnl_based = False
    rc._volatility_halted = False


# ===========================================================================
# TestCircuitBreaker
# ===========================================================================

class TestCircuitBreaker:
    """Tests for check_circuit_breaker()."""

    def test_no_block_when_pnl_zero(self):
        """Fresh day with zero P&L → not blocked."""
        with patch("agent.risk_controls._get_today_pnl", return_value=(0.0, 0.0)):
            blocked, reason = rc.check_circuit_breaker()
        assert blocked is False
        assert reason == ""

    def test_returns_false_blank_when_clear(self):
        """Return type and value contract when all clear."""
        with patch("agent.risk_controls._get_today_pnl", return_value=(10.0, 0.1)):
            result = rc.check_circuit_breaker()
        assert isinstance(result, tuple)
        assert len(result) == 2
        blocked, reason = result
        assert blocked is False
        assert isinstance(reason, str)

    def test_blocks_when_loss_exceeds_halt_threshold(self):
        """Loss of DAILY_LOSS_HALT_PCT% or more halts trading."""
        # pnl_dollar drives acct_loss_pct = pnl_dollar / DEFAULT_ACCOUNT_SIZE * 100
        halt_dollar = -(DAILY_LOSS_HALT_PCT / 100 * DEFAULT_ACCOUNT_SIZE) - 1
        halt_pct = halt_dollar / DEFAULT_ACCOUNT_SIZE * 100
        with patch("agent.risk_controls._get_today_pnl", return_value=(halt_dollar, halt_pct)):
            blocked, reason = rc.check_circuit_breaker()
        assert blocked is True
        assert reason != ""

    def test_blocks_when_loss_exactly_at_halt_threshold(self):
        """Exactly at the loss threshold is still blocked (boundary condition)."""
        halt_dollar = -(DAILY_LOSS_HALT_PCT / 100 * DEFAULT_ACCOUNT_SIZE)
        halt_pct = halt_dollar / DEFAULT_ACCOUNT_SIZE * 100
        with patch("agent.risk_controls._get_today_pnl", return_value=(halt_dollar, halt_pct)):
            blocked, reason = rc.check_circuit_breaker()
        assert blocked is True

    def test_blocks_when_profit_ceiling_hit(self):
        """P&L >= DAILY_PROFIT_MAX_USD halts trading to lock in gains."""
        pnl = DAILY_PROFIT_MAX_USD + 100.0
        with patch("agent.risk_controls._get_today_pnl", return_value=(pnl, 3.0)):
            blocked, reason = rc.check_circuit_breaker()
        assert blocked is True
        assert reason != ""

    def test_blocks_at_exact_profit_ceiling(self):
        """Profit ceiling is inclusive (>=)."""
        with patch("agent.risk_controls._get_today_pnl", return_value=(DAILY_PROFIT_MAX_USD, 3.0)):
            blocked, _ = rc.check_circuit_breaker()
        assert blocked is True

    def test_no_block_below_profit_ceiling(self):
        """P&L just below the ceiling is still allowed."""
        pnl = DAILY_PROFIT_MAX_USD - 1.0
        with patch("agent.risk_controls._get_today_pnl", return_value=(pnl, 2.9)):
            blocked, _ = rc.check_circuit_breaker()
        assert blocked is False

    def test_blocks_when_pnl_based_flag_set(self):
        """If _circuit_pnl_based + _circuit_open are True, always blocked."""
        rc._circuit_open = True
        rc._circuit_pnl_based = True
        rc._circuit_reason = "P&L-based halt"
        rc._circuit_date = date.today()
        with patch("agent.risk_controls._get_today_pnl", return_value=(0.0, 0.0)):
            blocked, reason = rc.check_circuit_breaker()
        assert blocked is True
        assert reason == "P&L-based halt"

    def test_after_hours_clears_consecutive_loss_block(self):
        """AH session bypasses a non-pnl-based circuit (consecutive losses)."""
        rc._circuit_open = True
        rc._circuit_pnl_based = False
        rc._circuit_reason = "Consecutive loss halt"
        rc._circuit_date = date.today()
        with patch("agent.risk_controls._get_today_pnl", return_value=(0.0, 0.0)):
            blocked, _ = rc.check_circuit_breaker(session="AFTER_HOURS")
        assert blocked is False

    def test_after_hours_pnl_halt_persists(self):
        """AH session does NOT clear a P&L-based halt."""
        rc._circuit_open = True
        rc._circuit_pnl_based = True
        rc._circuit_reason = "Daily loss halt"
        rc._circuit_date = date.today()
        with patch("agent.risk_controls._get_today_pnl", return_value=(0.0, 0.0)):
            blocked, reason = rc.check_circuit_breaker(session="AFTER_HOURS")
        assert blocked is True
        assert "Daily loss halt" in reason

    def test_reason_string_nonempty_when_blocked(self):
        """Whenever blocked=True, reason is a non-empty string."""
        halt_dollar = -(DAILY_LOSS_HALT_PCT / 100 * DEFAULT_ACCOUNT_SIZE) - 50
        with patch("agent.risk_controls._get_today_pnl", return_value=(halt_dollar, -3.0)):
            blocked, reason = rc.check_circuit_breaker()
        assert blocked is True
        assert len(reason) > 0

    def test_profit_protect_drawdown_fires(self):
        """Drawdown from peak within PPM range triggers halt."""
        # Set peak above current P&L by at least PROFIT_PROTECT_DRAWDOWN
        current_pnl = DAILY_PROFIT_TARGET_USD + 10.0
        rc._peak_daily_pnl = current_pnl + PROFIT_PROTECT_DRAWDOWN + 1.0
        with patch("agent.risk_controls._get_today_pnl", return_value=(current_pnl, 2.0)):
            blocked, reason = rc.check_circuit_breaker()
        assert blocked is True
        assert "protect" in reason.lower() or "drawdown" in reason.lower()

    def test_profit_protect_drawdown_no_fire_when_below_target(self):
        """PPM drawdown check only fires when current P&L >= DAILY_PROFIT_TARGET_USD."""
        current_pnl = DAILY_PROFIT_TARGET_USD - 100.0  # below target
        rc._peak_daily_pnl = current_pnl + PROFIT_PROTECT_DRAWDOWN + 1.0
        with patch("agent.risk_controls._get_today_pnl", return_value=(current_pnl, 1.5)):
            blocked, _ = rc.check_circuit_breaker()
        assert blocked is False

    def test_pnl_based_flag_set_on_loss_halt(self):
        """After a loss halt fires, _circuit_pnl_based must be True."""
        halt_dollar = -(DAILY_LOSS_HALT_PCT / 100 * DEFAULT_ACCOUNT_SIZE) - 50
        with patch("agent.risk_controls._get_today_pnl", return_value=(halt_dollar, -3.0)):
            rc.check_circuit_breaker()
        assert rc._circuit_pnl_based is True

    def test_pnl_based_flag_set_on_profit_ceiling(self):
        """After profit ceiling fires, _circuit_pnl_based must be True."""
        with patch("agent.risk_controls._get_today_pnl", return_value=(DAILY_PROFIT_MAX_USD, 3.0)):
            rc.check_circuit_breaker()
        assert rc._circuit_pnl_based is True


# ===========================================================================
# TestRecordTradeOutcome
# ===========================================================================

class TestRecordTradeOutcome:
    """Tests for record_trade_outcome()."""

    def test_win_resets_consecutive_losses(self):
        """A win immediately resets the consecutive loss counter to 0."""
        rc._consecutive_losses = 3
        rc.record_trade_outcome(won=True)
        assert rc._consecutive_losses == 0

    def test_loss_increments_counter(self):
        """A single loss increments consecutive losses from 0 to 1."""
        rc._consecutive_losses = 0
        rc.record_trade_outcome(won=False)
        assert rc._consecutive_losses == 1

    def test_multiple_losses_increment(self):
        """Three sequential losses give consecutive_losses == 3."""
        rc._consecutive_losses = 0
        rc.record_trade_outcome(won=False)
        rc.record_trade_outcome(won=False)
        rc.record_trade_outcome(won=False)
        assert rc._consecutive_losses == 3

    def test_win_after_losses_resets(self):
        """A win after several losses resets counter fully."""
        rc._consecutive_losses = 4
        rc.record_trade_outcome(won=True)
        assert rc._consecutive_losses == 0

    def test_check_consecutive_loss_state(self):
        """Internal counter matches recorded outcomes."""
        rc._consecutive_losses = 0
        for _ in range(2):
            rc.record_trade_outcome(won=False)
        rc.record_trade_outcome(won=True)
        rc.record_trade_outcome(won=False)
        # After W then L, counter should be 1
        assert rc._consecutive_losses == 1

    def test_return_value_is_none(self):
        """record_trade_outcome returns None (no return value contract)."""
        result = rc.record_trade_outcome(won=True)
        assert result is None


# ===========================================================================
# TestProfitProtectState
# ===========================================================================

class TestProfitProtectState:
    """Tests for get_profit_protect_state()."""

    def test_inactive_when_pnl_below_target(self):
        """PPM not active when P&L < DAILY_PROFIT_TARGET_USD."""
        with patch("agent.risk_controls._get_today_pnl", return_value=(0.0, 0.0)):
            state = rc.get_profit_protect_state()
        assert state["active"] is False

    def test_active_when_pnl_at_target(self):
        """PPM activates at exactly DAILY_PROFIT_TARGET_USD."""
        with patch("agent.risk_controls._get_today_pnl", return_value=(DAILY_PROFIT_TARGET_USD, 2.0)):
            state = rc.get_profit_protect_state()
        assert state["active"] is True

    def test_active_has_correct_size_mult(self):
        """When active, size_mult equals PROFIT_PROTECT_SIZE_MULT."""
        with patch("agent.risk_controls._get_today_pnl", return_value=(DAILY_PROFIT_TARGET_USD + 1, 2.0)):
            state = rc.get_profit_protect_state()
        assert state["size_mult"] == PROFIT_PROTECT_SIZE_MULT

    def test_inactive_size_mult_is_one(self):
        """When inactive, size_mult is 1.0."""
        with patch("agent.risk_controls._get_today_pnl", return_value=(0.0, 0.0)):
            state = rc.get_profit_protect_state()
        assert state["size_mult"] == 1.0

    def test_returns_required_keys(self):
        """Dict must include all expected keys."""
        with patch("agent.risk_controls._get_today_pnl", return_value=(0.0, 0.0)):
            state = rc.get_profit_protect_state()
        for key in ("active", "pnl_today", "target", "min_conf", "size_mult", "drawdown_cap", "description"):
            assert key in state, f"Missing key: {key}"

    def test_description_reflects_active_state(self):
        """Description string changes between active/inactive."""
        with patch("agent.risk_controls._get_today_pnl", return_value=(0.0, 0.0)):
            inactive = rc.get_profit_protect_state()
        with patch("agent.risk_controls._get_today_pnl", return_value=(DAILY_PROFIT_TARGET_USD + 1, 2.0)):
            active = rc.get_profit_protect_state()
        assert inactive["description"] != active["description"]


# ===========================================================================
# TestPortfolioHeat
# ===========================================================================

class TestPortfolioHeat:
    """Tests for get_portfolio_heat() and check_portfolio_heat()."""

    def _make_trade(self, entry, stop, shares):
        return {"entry_price": entry, "stop": stop, "shares": shares}

    def test_get_returns_dict_with_required_keys(self):
        """get_portfolio_heat must include heat_pct, blocked, open_count."""
        with patch("agent.paper_trading.get_open_trades", return_value=[]):
            heat = rc.get_portfolio_heat()
        for key in ("heat_pct", "blocked", "open_count"):
            assert key in heat

    def test_zero_open_trades_gives_zero_heat(self):
        """No open trades → heat_pct == 0."""
        with patch("agent.paper_trading.get_open_trades", return_value=[]):
            heat = rc.get_portfolio_heat()
        assert heat["heat_pct"] == 0.0
        assert heat["blocked"] is False

    def test_check_portfolio_heat_returns_tuple(self):
        """check_portfolio_heat() returns a (bool, str) tuple."""
        with patch("agent.paper_trading.get_open_trades", return_value=[]):
            result = rc.check_portfolio_heat()
        assert isinstance(result, tuple)
        assert len(result) == 2
        blocked, reason = result
        assert isinstance(blocked, bool)
        assert isinstance(reason, str)

    def test_not_blocked_with_empty_positions(self):
        """No positions → not blocked."""
        with patch("agent.paper_trading.get_open_trades", return_value=[]):
            blocked, _ = rc.check_portfolio_heat()
        assert blocked is False

    def test_heat_calculation(self):
        """Risk dollar = sum(|entry-stop| * shares) for each trade."""
        trades = [
            {"entry_price": 100.0, "stop": 98.0, "shares": 10},  # risk $20
        ]
        with patch("agent.paper_trading.get_open_trades", return_value=trades):
            heat = rc.get_portfolio_heat()
        assert heat["total_risk_dollar"] == pytest.approx(20.0, abs=0.01)

    def test_open_count_matches_positions(self):
        """open_count equals the number of open trade records."""
        trades = [
            {"entry_price": 100.0, "stop": 99.0, "shares": 5},
            {"entry_price": 200.0, "stop": 198.0, "shares": 3},
        ]
        with patch("agent.paper_trading.get_open_trades", return_value=trades):
            heat = rc.get_portfolio_heat()
        assert heat["open_count"] == 2


# ===========================================================================
# TestSectorConcentration
# ===========================================================================

class TestSectorConcentration:
    """Tests for get_sector() and check_sector_concentration()."""

    def test_known_ticker_returns_sector(self):
        """NVDA is in the SEMIS sector."""
        assert rc.get_sector("NVDA") == "SEMIS"

    def test_mega_tech_ticker_returns_correct_sector(self):
        """AAPL is in MEGA_TECH."""
        assert rc.get_sector("AAPL") == "MEGA_TECH"

    def test_unknown_ticker_returns_other(self):
        """Unmapped tickers return 'OTHER'."""
        assert rc.get_sector("XYZUNKNOWN") == "OTHER"

    def test_case_insensitive_lookup(self):
        """Ticker lookup is case-insensitive."""
        assert rc.get_sector("nvda") == rc.get_sector("NVDA")

    def test_sector_concentration_check_returns_tuple(self):
        """check_sector_concentration returns (bool, str)."""
        with patch("agent.paper_trading.get_open_trades", return_value=[]):
            result = rc.check_sector_concentration("NVDA", "BUY")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_not_blocked_with_no_open_positions(self):
        """No open positions → sector check passes."""
        with patch("agent.paper_trading.get_open_trades", return_value=[]):
            blocked, _ = rc.check_sector_concentration("NVDA", "BUY")
        assert blocked is False

    def test_blocked_when_two_same_sector_direction(self):
        """2 open SEMIS BUY positions blocks a third SEMIS BUY."""
        open_trades = [
            {"ticker": "NVDA", "direction": "BUY"},
            {"ticker": "AMD", "direction": "BUY"},
        ]
        with patch("agent.paper_trading.get_open_trades", return_value=open_trades):
            blocked, reason = rc.check_sector_concentration("AVGO", "BUY")
        assert blocked is True
        assert "SEMIS" in reason or "sector" in reason.lower()

    def test_not_blocked_with_only_one_same_sector(self):
        """Only 1 open position in same sector → allowed."""
        open_trades = [{"ticker": "NVDA", "direction": "BUY"}]
        with patch("agent.paper_trading.get_open_trades", return_value=open_trades):
            blocked, _ = rc.check_sector_concentration("AMD", "BUY")
        assert blocked is False

    def test_other_sector_ticker_never_blocked(self):
        """Tickers in 'OTHER' sector are never blocked by concentration."""
        with patch("agent.paper_trading.get_open_trades", return_value=[]):
            blocked, _ = rc.check_sector_concentration("XYZUNKNOWN", "BUY")
        assert blocked is False

    def test_invalid_direction_not_blocked(self):
        """Invalid direction string → returns (False, '')."""
        with patch("agent.paper_trading.get_open_trades", return_value=[]):
            blocked, reason = rc.check_sector_concentration("NVDA", "INVALID")
        assert blocked is False
        assert reason == ""


# ===========================================================================
# TestVolatilityHalt
# ===========================================================================

class TestVolatilityHalt:
    """Tests for update_volatility_state() and check_volatility_halt()."""

    def test_normal_volatility_does_not_halt(self):
        """Session range well below ATR multiple → no halt."""
        rc.update_volatility_state(session_range_pct=1.0, avg_atr_pct=1.0)
        # ratio == 1.0, threshold is VOLATILITY_HALT_ATR_MULT (2.5) → no halt
        halted, reason = rc.check_volatility_halt()
        assert halted is False
        assert reason == ""

    def test_extreme_volatility_halts_trading(self):
        """Session range > VOLATILITY_HALT_ATR_MULT × ATR triggers halt."""
        avg_atr = 1.0
        spike = VOLATILITY_HALT_ATR_MULT * avg_atr + 0.5  # above threshold
        rc.update_volatility_state(session_range_pct=spike, avg_atr_pct=avg_atr)
        halted, reason = rc.check_volatility_halt()
        assert halted is True
        assert reason != ""

    def test_halt_clears_on_normal_conditions(self):
        """After a halt, normal volatility update clears the halt."""
        rc.update_volatility_state(session_range_pct=10.0, avg_atr_pct=1.0)
        assert rc.check_volatility_halt()[0] is True
        rc.update_volatility_state(session_range_pct=1.0, avg_atr_pct=1.0)
        halted, _ = rc.check_volatility_halt()
        assert halted is False

    def test_zero_avg_atr_skipped_safely(self):
        """avg_atr_pct=0 → function returns without error, halt unchanged."""
        rc._volatility_halted = False
        rc.update_volatility_state(session_range_pct=5.0, avg_atr_pct=0.0)
        halted, _ = rc.check_volatility_halt()
        assert halted is False  # unchanged

    def test_check_volatility_halt_returns_tuple(self):
        """Return type is (bool, str)."""
        result = rc.check_volatility_halt()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_volatility_reason_contains_detail(self):
        """Halt reason includes ATR ratio information."""
        rc.update_volatility_state(session_range_pct=5.0, avg_atr_pct=1.0)
        _, reason = rc.check_volatility_halt()
        # reason should mention volatility or ATR
        assert any(kw in reason.lower() for kw in ("volatility", "atr", "range"))

    def test_atr_ratio_stored(self):
        """_volatility_atr_ratio is set after update."""
        rc.update_volatility_state(session_range_pct=3.0, avg_atr_pct=1.5)
        assert rc._volatility_atr_ratio == pytest.approx(2.0, abs=0.01)


# ===========================================================================
# TestMaxDailyTrades
# ===========================================================================

class TestMaxDailyTrades:
    """Tests for check_max_daily_trades()."""

    def test_not_blocked_when_below_limit(self):
        """Trade count below MAX_DAILY_TRADES → not blocked."""
        mock_stats = {"total": MAX_DAILY_TRADES - 1, "total_pnl_dollar": 0, "total_pnl_pct": 0}
        with patch("agent.paper_trading.get_today_pnl", return_value=mock_stats):
            blocked, _ = rc.check_max_daily_trades()
        assert blocked is False

    def test_blocked_at_limit(self):
        """Trade count == MAX_DAILY_TRADES → blocked."""
        mock_stats = {"total": MAX_DAILY_TRADES, "total_pnl_dollar": 0, "total_pnl_pct": 0}
        with patch("agent.paper_trading.get_today_pnl", return_value=mock_stats):
            blocked, reason = rc.check_max_daily_trades()
        assert blocked is True
        assert str(MAX_DAILY_TRADES) in reason

    def test_blocked_above_limit(self):
        """Trade count above limit → blocked."""
        mock_stats = {"total": MAX_DAILY_TRADES + 5, "total_pnl_dollar": 0, "total_pnl_pct": 0}
        with patch("agent.paper_trading.get_today_pnl", return_value=mock_stats):
            blocked, _ = rc.check_max_daily_trades()
        assert blocked is True

    def test_returns_tuple(self):
        """Return type is (bool, str)."""
        mock_stats = {"total": 0, "total_pnl_dollar": 0, "total_pnl_pct": 0}
        with patch("agent.paper_trading.get_today_pnl", return_value=mock_stats):
            result = rc.check_max_daily_trades()
        assert isinstance(result, tuple)
        assert len(result) == 2


# ===========================================================================
# TestDrawdownThrottle
# ===========================================================================

class TestDrawdownThrottle:
    """Tests for get_drawdown_throttle()."""

    def test_no_throttle_on_positive_pnl(self):
        """Profitable day → no throttle."""
        mock_stats = {"total_pnl_dollar": 500.0, "total": 0}
        with patch("agent.paper_trading.get_today_pnl", return_value=mock_stats):
            result = rc.get_drawdown_throttle()
        assert result["active"] is False
        assert result["size_mult"] == 1.0

    def test_no_throttle_on_zero_pnl(self):
        """Break-even day → no throttle."""
        mock_stats = {"total_pnl_dollar": 0.0, "total": 0}
        with patch("agent.paper_trading.get_today_pnl", return_value=mock_stats):
            result = rc.get_drawdown_throttle()
        assert result["active"] is False

    def test_moderate_throttle_at_first_tier(self):
        """Drawdown >= DRAWDOWN_THROTTLE_1_PCT → 50% size."""
        loss = -(DRAWDOWN_THROTTLE_1_PCT / 100 * DEFAULT_ACCOUNT_SIZE) - 1
        mock_stats = {"total_pnl_dollar": loss, "total": 0}
        with patch("agent.paper_trading.get_today_pnl", return_value=mock_stats):
            result = rc.get_drawdown_throttle()
        assert result["active"] is True
        assert result["size_mult"] == 0.50

    def test_severe_throttle_at_second_tier(self):
        """Drawdown >= DRAWDOWN_THROTTLE_2_PCT → 25% size."""
        loss = -(DRAWDOWN_THROTTLE_2_PCT / 100 * DEFAULT_ACCOUNT_SIZE) - 1
        mock_stats = {"total_pnl_dollar": loss, "total": 0}
        with patch("agent.paper_trading.get_today_pnl", return_value=mock_stats):
            result = rc.get_drawdown_throttle()
        assert result["active"] is True
        assert result["size_mult"] == 0.25

    def test_returns_required_keys(self):
        """Dict must contain active, size_mult, drawdown_pct, tier."""
        mock_stats = {"total_pnl_dollar": 0.0, "total": 0}
        with patch("agent.paper_trading.get_today_pnl", return_value=mock_stats):
            result = rc.get_drawdown_throttle()
        for key in ("active", "size_mult", "drawdown_pct", "tier"):
            assert key in result


# ===========================================================================
# TestCheckSessionBlock
# ===========================================================================

class TestCheckSessionBlock:
    """Tests for check_session_block()."""

    def test_returns_three_tuple(self):
        """Return type must be (bool, str, float)."""
        with patch("agent.market_hours.get_session", return_value="REGULAR"), \
             patch("agent.market_hours.no_new_entries", return_value=False), \
             patch("agent.market_hours.get_block_reason", return_value=""):
            result = rc.check_session_block("REGULAR")
        assert isinstance(result, tuple)
        assert len(result) == 3
        blocked, reason, size = result
        assert isinstance(blocked, bool)
        assert isinstance(reason, str)
        assert isinstance(size, float)

    def test_after_hours_high_tier_allowed(self):
        """AH + HIGH tier → not blocked, 50% size."""
        with patch("agent.market_hours.get_session", return_value="AFTER_HOURS"), \
             patch("agent.market_hours.no_new_entries", return_value=False):
            blocked, reason, size = rc.check_session_block("HIGH")
        assert blocked is False
        assert size == 0.50

    def test_after_hours_moderate_tier_allowed(self):
        """AH + MODERATE tier → not blocked, 30% size."""
        with patch("agent.market_hours.get_session", return_value="AFTER_HOURS"), \
             patch("agent.market_hours.no_new_entries", return_value=False):
            blocked, _, size = rc.check_session_block("MODERATE")
        assert blocked is False
        assert size == 0.30

    def test_after_hours_regular_tier_blocked(self):
        """AH + REGULAR tier → blocked."""
        with patch("agent.market_hours.get_session", return_value="AFTER_HOURS"), \
             patch("agent.market_hours.no_new_entries", return_value=False):
            blocked, reason, size = rc.check_session_block("REGULAR")
        assert blocked is True
        assert size == 0.0

    def test_pre_market_high_tier_allowed(self):
        """PM + HIGH tier → not blocked, 40% size."""
        with patch("agent.market_hours.get_session", return_value="PRE_MARKET"), \
             patch("agent.market_hours.no_new_entries", return_value=False):
            blocked, _, size = rc.check_session_block("HIGH")
        assert blocked is False
        assert size == 0.40

    def test_regular_session_not_blocked(self):
        """Regular hours with no_new_entries=False → allowed."""
        with patch("agent.market_hours.get_session", return_value="REGULAR"), \
             patch("agent.market_hours.no_new_entries", return_value=False), \
             patch("agent.market_hours.get_block_reason", return_value=""):
            blocked, _, _ = rc.check_session_block("REGULAR")
        assert blocked is False

    def test_closed_session_blocked(self):
        """When no_new_entries returns True, trading is blocked."""
        with patch("agent.market_hours.get_session", return_value="REGULAR"), \
             patch("agent.market_hours.no_new_entries", return_value=True), \
             patch("agent.market_hours.get_block_reason", return_value="Market closed"):
            blocked, reason, _ = rc.check_session_block("HIGH")
        assert blocked is True


# ===========================================================================
# TestCanOpenTrade
# ===========================================================================

class TestCanOpenTrade:
    """Tests for can_open_trade()."""

    def _patch_all_clear(self):
        """Context managers for a fully clear environment."""
        return [
            patch("agent.market_hours.get_session", return_value="REGULAR"),
            patch("agent.market_hours.no_new_entries", return_value=False),
            patch("agent.market_hours.get_block_reason", return_value=""),
            patch("agent.market_hours.position_size_multiplier", return_value=1.0),
            patch("agent.risk_controls._get_today_pnl", return_value=(0.0, 0.0)),
            patch("agent.paper_trading.get_open_trades", return_value=[]),
            patch("agent.paper_trading.get_today_pnl", return_value={"total": 0, "total_pnl_dollar": 0.0}),
        ]

    def test_returns_three_tuple(self):
        """Return type is (bool, str, float)."""
        patches = self._patch_all_clear()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            result = rc.can_open_trade("NVDA", "BUY", 70.0, "REGULAR")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_returns_true_when_all_clear(self):
        """All checks passing → allowed=True, empty reason."""
        patches = self._patch_all_clear()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            allowed, reason, size = rc.can_open_trade("NVDA", "BUY", 70.0, "REGULAR")
        assert allowed is True

    def test_reason_empty_when_allowed(self):
        """When allowed, reason is empty string."""
        patches = self._patch_all_clear()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            _, reason, _ = rc.can_open_trade("NVDA", "BUY", 70.0, "REGULAR")
        assert reason == ""

    def test_returns_false_when_circuit_open(self):
        """Circuit breaker open → trade blocked."""
        rc._circuit_open = True
        rc._circuit_pnl_based = True
        rc._circuit_reason = "Loss halt"
        rc._circuit_date = date.today()
        with patch("agent.market_hours.get_session", return_value="REGULAR"), \
             patch("agent.market_hours.no_new_entries", return_value=False), \
             patch("agent.market_hours.get_block_reason", return_value=""), \
             patch("agent.market_hours.position_size_multiplier", return_value=1.0), \
             patch("agent.risk_controls._get_today_pnl", return_value=(0.0, 0.0)), \
             patch("agent.paper_trading.get_open_trades", return_value=[]), \
             patch("agent.paper_trading.get_today_pnl", return_value={"total": 0, "total_pnl_dollar": 0.0}):
            allowed, reason, _ = rc.can_open_trade("NVDA", "BUY", 70.0, "REGULAR")
        assert allowed is False
        assert reason != ""

    def test_reason_nonempty_when_blocked(self):
        """When blocked, reason must be non-empty."""
        with patch("agent.market_hours.get_session", return_value="REGULAR"), \
             patch("agent.market_hours.no_new_entries", return_value=True), \
             patch("agent.market_hours.get_block_reason", return_value="Market closed"):
            allowed, reason, _ = rc.can_open_trade("NVDA", "BUY", 70.0, "REGULAR")
        assert allowed is False
        assert len(reason) > 0

    def test_size_multiplier_is_positive(self):
        """Size multiplier from can_open_trade is positive when allowed."""
        patches = self._patch_all_clear()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            allowed, _, size = rc.can_open_trade("NVDA", "BUY", 70.0, "REGULAR")
        assert allowed is True
        assert size > 0.0

    def test_blocked_when_daily_loss_exceeded(self):
        """Circuit breaker fires when loss exceeds halt threshold."""
        halt_dollar = -(DAILY_LOSS_HALT_PCT / 100 * DEFAULT_ACCOUNT_SIZE) - 100
        halt_pct = halt_dollar / DEFAULT_ACCOUNT_SIZE * 100
        with patch("agent.market_hours.get_session", return_value="REGULAR"), \
             patch("agent.market_hours.no_new_entries", return_value=False), \
             patch("agent.market_hours.get_block_reason", return_value=""), \
             patch("agent.market_hours.position_size_multiplier", return_value=1.0), \
             patch("agent.risk_controls._get_today_pnl", return_value=(halt_dollar, halt_pct)), \
             patch("agent.paper_trading.get_open_trades", return_value=[]), \
             patch("agent.paper_trading.get_today_pnl", return_value={"total": 0, "total_pnl_dollar": halt_dollar}):
            allowed, reason, _ = rc.can_open_trade("NVDA", "BUY", 70.0, "REGULAR")
        assert allowed is False
        assert reason != ""

    def test_blocked_by_session_closed(self):
        """Hard-close / closed session blocks regardless of ticker/tier."""
        with patch("agent.market_hours.get_session", return_value="REGULAR"), \
             patch("agent.market_hours.no_new_entries", return_value=True), \
             patch("agent.market_hours.get_block_reason", return_value="Hard close"):
            allowed, _, _ = rc.can_open_trade("AAPL", "BUY", 90.0, "HIGH")
        assert allowed is False


# ===========================================================================
# TestGetRiskStatus
# ===========================================================================

class TestGetRiskStatus:
    """Tests for get_risk_status()."""

    def _mock_env(self):
        return [
            patch("agent.risk_controls._get_today_pnl", return_value=(0.0, 0.0)),
            patch("agent.paper_trading.get_open_trades", return_value=[]),
            patch("agent.paper_trading.get_today_pnl", return_value={"total": 0, "total_pnl_dollar": 0.0}),
            patch("agent.market_hours.get_session", return_value="REGULAR"),
            patch("agent.market_hours.no_new_entries", return_value=False),
            patch("agent.market_hours.get_block_reason", return_value=""),
            patch("agent.market_hours.position_size_multiplier", return_value=1.0),
        ]

    def test_returns_dict(self):
        """get_risk_status returns a dict."""
        patches = self._mock_env()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            status = rc.get_risk_status()
        assert isinstance(status, dict)

    def test_has_required_keys(self):
        """Dict must include core keys."""
        required = [
            "circuit_open", "circuit_reason",
            "consecutive_losses", "volatility_halted",
            "pnl_today_dollar", "daily_target", "daily_max",
            "profit_protect", "portfolio_heat",
        ]
        patches = self._mock_env()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            status = rc.get_risk_status()
        for key in required:
            assert key in status, f"Missing key: {key}"

    def test_circuit_open_reflected(self):
        """When circuit is manually set open, get_risk_status reflects that."""
        rc._circuit_open = True
        rc._circuit_reason = "Test halt"
        rc._circuit_date = date.today()
        patches = self._mock_env()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            status = rc.get_risk_status()
        assert status["circuit_open"] is True
        assert status["circuit_reason"] == "Test halt"

    def test_consecutive_losses_reflected(self):
        """_consecutive_losses state is exposed in get_risk_status."""
        rc._consecutive_losses = 2
        patches = self._mock_env()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            status = rc.get_risk_status()
        assert status["consecutive_losses"] == 2

    def test_volatility_halted_reflected(self):
        """Volatility halt state appears in get_risk_status."""
        rc._volatility_halted = True
        rc._volatility_reason = "Test volatility halt"
        patches = self._mock_env()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            status = rc.get_risk_status()
        assert status["volatility_halted"] is True


# ===========================================================================
# TestGetSector
# ===========================================================================

class TestGetSector:
    """Tests for get_sector()."""

    def test_semis_tickers(self):
        for ticker in ("NVDA", "AMD", "AVGO", "QCOM"):
            assert rc.get_sector(ticker) == "SEMIS"

    def test_mega_tech_tickers(self):
        for ticker in ("AAPL", "MSFT", "GOOGL", "META", "AMZN"):
            assert rc.get_sector(ticker) == "MEGA_TECH"

    def test_cloud_tickers(self):
        for ticker in ("SNOW", "DDOG", "CRWD", "ZS"):
            assert rc.get_sector(ticker) == "CLOUD"

    def test_fintech_tickers(self):
        for ticker in ("COIN", "PYPL", "HOOD"):
            assert rc.get_sector(ticker) == "FINTECH"

    def test_biotech_tickers(self):
        for ticker in ("REGN", "AMGN", "GILD"):
            assert rc.get_sector(ticker) == "BIOTECH"

    def test_unknown_returns_other(self):
        assert rc.get_sector("ZZZZZ") == "OTHER"

    def test_case_insensitive(self):
        assert rc.get_sector("aapl") == "MEGA_TECH"
        assert rc.get_sector("Nvda") == "SEMIS"

    def test_ev_sector(self):
        assert rc.get_sector("RIVN") == "EV"
        assert rc.get_sector("LCID") == "EV"

    def test_ai_emerging_sector(self):
        assert rc.get_sector("PLTR") == "AI_EMERGING"
        assert rc.get_sector("IONQ") == "AI_EMERGING"
