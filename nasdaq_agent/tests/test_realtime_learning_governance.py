from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.algo_learning_engine import (
    AlgoLearningEngine,
    EconomicParameterGovernor,
)


class _Cfg:
    def __init__(self, values=None):
        self.values = values or {}

    def get(self, key, default=None):
        return self.values.get(key, default)


def _outcome(outcome_id="bt:one"):
    return pd.DataFrame([{
        "outcome_id": outcome_id,
        "pnl_pct": -0.5,
        "pnl_dollar": -20.0,
        "exit_reason": "STOP",
        "status": "LOSS",
        "algo_name": "META_ENS_BULL",
        "regime": "NEUTRAL",
        "session": "PRIME",
        "vwap_event": "BELOW",
        "direction": "BUY",
        "bars_tracked": 2,
        "max_favorable_r": 0.0,
    }])


def test_same_outcome_is_learned_once():
    engine = AlgoLearningEngine()
    engine._audit.log = MagicMock()
    engine._governor.process_family = MagicMock(return_value={"status": "HEALTHY"})
    engine.save = MagicMock()

    with patch("agent.algo_learning_p2.get_phase2_engine") as phase2:
        engine.run_cycle(_outcome(), cycle_num=1)
        engine.run_cycle(_outcome(), cycle_num=2)

    assert engine._governor.process_family.call_count == 1
    assert "bt:one" in engine._processed_set
    assert phase2.return_value.run_cycle.call_count == 1


def test_losing_family_activates_bounded_paper_canary():
    registry = MagicMock()
    registry.get.return_value = 55.0
    registry.update.return_value = True
    loss = MagicMock()
    loss.get_dominant_cause.return_value = "WRONG_DIRECTION"
    governor = EconomicParameterGovernor(registry, loss)
    baseline = {
        "n_trades": 20, "expectancy": -8.0, "profit_factor": 0.4,
        "win_rate": 0.2, "max_drawdown": 100.0,
    }

    with patch.object(governor, "_metrics", return_value=baseline), \
         patch.object(governor, "_can_activate", return_value=True), \
         patch.object(governor, "save"):
        result = governor.process_family("META_ENS", "META_ENS_BULL", 10)

    assert result["status"] == "ACTIVE"
    assert result["param"] == "conf_gate"
    registry.update.assert_called_once()


def test_bad_candidate_rolls_back_automatically():
    registry = MagicMock()
    loss = MagicMock()
    governor = EconomicParameterGovernor(registry, loss)
    governor._state["active"]["META_ENS"] = {
        "candidate_id": "c1",
        "family": "META_ENS",
        "algo_name": "META_ENS_BULL",
        "param": "conf_gate",
        "old_val": 55.0,
        "candidate_val": 56.0,
        "activated_at": "2026-06-18T10:00:00+00:00",
        "baseline": {
            "n_trades": 20, "expectancy": -2.0, "profit_factor": 0.8,
            "max_drawdown": 50.0,
        },
    }
    candidate = {
        "n_trades": 10, "expectancy": -5.0, "profit_factor": 0.5,
        "win_rate": 0.2, "max_drawdown": 80.0,
    }

    with patch.object(governor, "_metrics", return_value=candidate), \
         patch.object(governor, "save"), \
         patch("agent.config_manager.config", _Cfg()):
        result = governor.process_family("META_ENS", "META_ENS_BULL", 20)

    assert result["status"] == "ROLLED_BACK"
    registry.rollback_param.assert_called_once()
    assert "META_ENS" not in governor._state["active"]
