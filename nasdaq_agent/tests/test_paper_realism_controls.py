from __future__ import annotations

import sys
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
    with patch("agent.risk_controls._is_paper_mode", return_value=True), \
         patch("agent.risk_controls._paper_risk_enforced", return_value=True), \
         patch("agent.risk_controls._rcfg", side_effect=lambda k, d=None: cfg.get(k, d)), \
         patch("agent.risk_controls._account_size", return_value=150000.0), \
         patch("agent.risk_controls._get_today_pnl", return_value=(-350.0, -0.23)):
        blocked, reason = rc.check_circuit_breaker()

    assert blocked is True
    assert "Paper daily loss halt" in reason


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

