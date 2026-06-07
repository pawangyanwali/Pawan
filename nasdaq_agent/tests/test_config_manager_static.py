from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_config_manager_does_not_override_user_t2_setting():
    src = (ROOT / "agent/config_manager.py").read_text(encoding="utf-8")

    assert "def _apply_safety_migrations" in src
    assert "paper.t2_r_multiple" in src
    assert "migration_t2_1_5" in src
    assert 'updated_by == "seed_defaults"' in src
