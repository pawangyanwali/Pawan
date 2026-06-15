from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_TRADING = ROOT / "agent" / "paper_trading.py"


def _src() -> str:
    return PAPER_TRADING.read_text(encoding="utf-8", errors="ignore")


def test_primary_prediction_trades_are_attributed_for_learning():
    src = _src()

    assert "def _effective_algo_name" in src
    assert 'return f"PRED_{entry}"' in src
    assert "algo_name = _effective_algo_name(algo_name, entry_type)" in src
    assert "check_rolling_ev_suppress(algo_name, direction, session or \"\")" in src


def test_mfe_mae_excursion_columns_are_persisted():
    src = _src()

    for column in ("mfe_dollar", "mae_dollar", "mfe_pct", "mae_pct", "mfe_r", "mae_r"):
        assert column in src
    assert "SET mfe_dollar=?, mae_dollar=?, mfe_pct=?, mae_pct=?, mfe_r=?, mae_r=?" in src
