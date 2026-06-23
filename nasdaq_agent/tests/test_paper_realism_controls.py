from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent.paper_trading as pt
import agent.risk_controls as rc


class _Cfg:
    def __init__(self, values: dict):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


class _FakeConn:
    def __init__(self, row):
        self.row = row

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, *_args, **_kwargs):
        return self

    def fetchone(self):
        return self.row


def _reset_risk_state():
    rc._circuit_open = False
    rc._circuit_reason = ""
    rc._circuit_date = None
    rc._circuit_pnl_based = False
    rc._cooldown_until = 0.0
    rc._consecutive_losses = 0
    rc._peak_daily_pnl = 0.0


def test_realistic_paper_daily_loss_halt_blocks_execution():
    _reset_risk_state()
    cfg = {
        "paper.daily_loss_halt_usd": 300.0,
        "paper.daily_loss_halt_pct": 0.25,
    }
    try:
        with patch("agent.risk_controls._is_paper_mode", return_value=True), \
             patch("agent.risk_controls._paper_risk_enforced", return_value=True), \
             patch("agent.risk_controls._rcfg", side_effect=lambda k, d=None: cfg.get(k, d)), \
             patch("agent.risk_controls._account_size", return_value=150000.0), \
             patch("agent.risk_controls._get_today_pnl", return_value=(-350.0, -0.23)):
            blocked, reason = rc.check_circuit_breaker()

        assert blocked is True
        assert "Paper daily loss halt" in reason
    finally:
        _reset_risk_state()


def test_realistic_paper_max_daily_trades_uses_paper_cap():
    cfg = {
        "risk.max_daily_trades": 5000,
        "paper.max_daily_trades": 75,
    }
    with patch("agent.risk_controls._is_paper_mode", return_value=True), \
         patch("agent.risk_controls._paper_risk_enforced", return_value=True), \
         patch("agent.risk_controls._rcfg", side_effect=lambda k, d=None: cfg.get(k, d)), \
         patch("agent.paper_trading.get_today_pnl", return_value={"total": 75}):
        blocked, reason = rc.check_max_daily_trades()

    assert blocked is True
    assert "75/75" in reason


def test_family_damage_stop_blocks_losing_family_from_today_ledger():
    cfg = _Cfg({
        "risk.family_damage_enabled": True,
        "risk.family_damage_min_trades": 3,
        "risk.family_damage_max_losses": 3,
        "risk.family_damage_loss_usd": 100.0,
        "risk.family_damage_min_win_rate": 30.0,
        "risk.family_damage_scope_session": False,
    })
    row = {"total": 3, "wins": 0, "losses": 3, "pnl": -120.0}
    with patch("agent.config_manager.config", cfg), \
         patch("agent.db.using_postgres", return_value=False), \
         patch("agent.paper_trading._conn_ro", return_value=_FakeConn(row)):
        blocked, reason = pt.check_family_damage_stop("META_ENS_BULL", "PRIME")

    assert blocked is True
    assert "Family damage stop [meta_ens]" in reason


def test_family_open_exposure_blocks_same_family_direction_cluster():
    cfg = _Cfg({
        "risk.family_max_open_per_direction": 1,
        "risk.family_open_scope_session": False,
    })
    row = {"n": 1}
    with patch("agent.config_manager.config", cfg), \
         patch("agent.paper_trading._conn_ro", return_value=_FakeConn(row)):
        blocked, reason = pt.check_family_open_exposure("KC_FADE_BEAR", "SELL", "PRIME")

    assert blocked is True
    assert "Family exposure cap [keltner SELL]" in reason


def test_fast_family_damage_blocks_after_two_recent_losses():
    cfg = _Cfg({
        "risk.fast_family_damage_enabled": True,
        "risk.fast_family_loss_window_min": 20,
        "risk.fast_family_max_losses": 2,
        "risk.fast_family_loss_usd": 75.0,
        "risk.family_damage_scope_session": False,
    })
    row = {"total": 2, "losses": 2, "pnl": -55.0}
    with patch("agent.config_manager.config", cfg), \
         patch("agent.paper_trading._conn_ro", return_value=_FakeConn(row)):
        blocked, reason = pt.check_fast_family_damage_stop("META_ENS_BULL", "BUY", "STANDARD")

    assert blocked is True
    assert "Fast family damage [meta_ens BUY]" in reason


def test_post_auth_quarantine_blocks_immediately_after_schwab_recovery():
    cfg = _Cfg({"risk.post_auth_quarantine_min": 20})
    row = {"resolved_at": datetime.now(timezone.utc)}
    with patch("agent.config_manager.config", cfg), \
         patch("agent.paper_trading._conn_ro", return_value=_FakeConn(row)):
        blocked, reason = pt.check_post_auth_quarantine("STANDARD")

    assert blocked is True
    assert "Post-auth quarantine" in reason


def test_intraday_geometry_cap_blocks_unreachable_stop_or_target():
    cfg = _Cfg({
        "risk.intraday_max_stop_pct": 2.0,
        "risk.intraday_max_target_pct": 4.0,
    })
    with patch("agent.config_manager.config", cfg):
        blocked, reason = pt._check_intraday_geometry_cap(
            "ACLS", "SELL", price=170.0, stop=192.0, target=126.0
        )

    assert blocked is True
    assert "exceeds" in reason


def test_deep_saturation_guard_requires_ensemble_confirmation():
    cfg = _Cfg({
        "risk.deep_saturation_guard_enabled": True,
        "risk.deep_saturation_prob": 0.98,
        "risk.deep_saturation_min_ensemble": 60,
    })
    with patch("agent.config_manager.config", cfg):
        blocked, reason = pt._deep_saturation_block("AEHR", 1.0, 50)

    assert blocked is True
    assert "ensemble=50" in reason


def test_entry_spread_to_risk_blocks_expensive_fill():
    cfg = _Cfg({"risk.max_entry_spread_to_risk": 0.35})
    with patch("agent.config_manager.config", cfg):
        blocked, reason = pt._entry_spread_to_risk_block("EDIT", spread_dollar=1.12, actual_risk=2.0)

    assert blocked is True
    assert "spread is 56%" in reason
