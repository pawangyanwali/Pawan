from pathlib import Path


INDEX_HTML = Path(__file__).resolve().parents[1] / "web" / "static" / "index.html"


def _html() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


def test_dashboard_auth_fetch_refreshes_and_retries_401():
    src = _html()

    assert "async function _refreshAccessToken()" in src
    assert "if (res.status !== 401) return res;" in src
    assert "const refreshed = await _refreshAccessToken();" in src
    assert "return fetch(url, retry);" in src


def test_dashboard_websocket_refreshes_rejected_access_token():
    src = _html()

    assert "async function connectWS()" in src
    assert "const _wsAt = await _ensureAccessToken();" in src
    assert "if (e.code === 4001)" in src
    assert "_refreshAccessToken().then(ok =>" in src


def test_dashboard_rest_fallback_applies_signal_payload():
    src = _html()

    assert "function _applySignalPayload(d)" in src
    assert "allSignals = d.signals;" in src
    assert "_applySignalPayload(await r.json());" in src
