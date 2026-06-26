from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "web" / "static" / "index.html"
SETTINGS = ROOT / "web" / "static" / "settings.html"
SYSTEM_ROUTER = ROOT / "routers" / "system.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def test_settings_is_a_dedicated_page() -> None:
    dashboard = _read(INDEX)
    settings = _read(SETTINGS)
    routes = _read(SYSTEM_ROUTER)

    assert 'id="settings-modal"' not in dashboard
    assert "document.getElementById('settings-modal')" not in dashboard
    assert dashboard.count("window.location.href='/settings'") == 3
    assert '@router.get("/settings"' in routes
    assert 'FileResponse(os.path.join(STATIC_DIR, "settings.html"))' in routes
    assert "Runtime Settings" in settings
    assert 'href="/"' in settings


def test_settings_page_is_catalog_driven_and_uncluttered() -> None:
    html = _read(SETTINGS)

    assert "authFetch('/api/config-catalog')" in html
    assert "authFetch('/api/config'" in html
    assert 'id="group-nav"' in html
    assert 'id="search"' in html
    assert 'id="advanced"' in html
    assert 'id="savebar"' in html
    assert "state.original" in html
    assert "state.defaults[f.key]=f.default" in html
    assert 'class="history-modal"' not in html
    assert 'id="settings-modal"' not in html


def test_settings_page_explains_the_scalp_cutover() -> None:
    html = _read(SETTINGS)

    assert "scalp.execution_enabled" in html
    assert "valid scalp plan mandatory" in html
    assert "Reset to default" in html
