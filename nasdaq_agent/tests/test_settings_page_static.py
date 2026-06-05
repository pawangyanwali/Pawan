from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "web" / "static" / "index.html"
CONFIG_MANAGER = ROOT / "agent" / "config_manager.py"


def _index_text() -> str:
    return INDEX.read_text(encoding="utf-8", errors="ignore")


def test_settings_is_full_page_not_modal() -> None:
    html = _index_text()
    settings_open = '<div class="settings-page" id="settings-modal" aria-hidden="true">'
    assert settings_open in html
    assert '<div class="history-modal" id="settings-modal">' not in html
    assert "max-height:72vh;overflow-y:auto" not in html
    assert 'id="settings-field-guide"' in html
    assert "Back to Dashboard" in html


def test_duplicate_or_unwired_settings_not_exposed() -> None:
    html = _index_text()
    hidden_keys = {
        "broker.auto_trade",
        "broker.paper_trading",
        "trading.account_size",
        "trading.risk_pct",
        "trading.max_position_pct",
        "risk.pre_t1_storm_rate",
    }
    for key in hidden_keys:
        assert f'"{key}"' not in html


def test_visible_settings_have_defaults_and_runtime_consumers() -> None:
    html = _index_text()
    defaults = CONFIG_MANAGER.read_text(encoding="utf-8", errors="ignore")
    ui_keys = set(
        re.findall(
            r'"((?:paper|prediction|sizing|scanner|filter|learner|algos|sr|execution|risk|trading|broker|audit)\.[A-Za-z0-9_${}.-]+)"\s*:',
            html,
        )
    )
    default_keys = set(
        re.findall(
            r'"((?:paper|prediction|sizing|scanner|filter|learner|algos|sr|execution|risk|trading|broker|audit)\.[A-Za-z0-9_.-]+)"\s*:',
            defaults,
        )
    )
    missing_defaults = sorted(k for k in ui_keys if "${family}" not in k and k not in default_keys)
    assert missing_defaults == []

    runtime_files = [
        p
        for p in ROOT.rglob("*.py")
        if "tests" not in p.parts
        and p.name not in {"config_manager.py", "config_router.py"}
    ]
    runtime_text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in runtime_files)
    missing_consumers: list[str] = []
    for key in sorted(ui_keys):
        if "${family}" in key:
            continue
        if key.startswith("execution.slip_base_"):
            assert 'f"execution.slip_base_{session.lower()}"' in runtime_text
            continue
        if key not in runtime_text:
            missing_consumers.append(key)
    assert missing_consumers == []
