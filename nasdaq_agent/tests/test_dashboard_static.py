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


def test_dashboard_does_not_reconnect_websocket_for_price_staleness_only():
    src = _html()

    assert "const wsSilent = !_wsLastMsgAt || (Date.now() - _wsLastMsgAt > _WS_WATCHDOG_MS);" in src
    assert "Backend heartbeat silent while price age is" in src
    assert "Price stale ${age}s and socket silent - forcing reconnect" not in src


def test_dashboard_target_win_rate_not_hardcoded():
    """Fail if any win-rate threshold or label is still hard-coded to 62."""
    src = _html()

    # These patterns indicate the UI is ignoring the adaptive target from the backend.
    bad_patterns = [
        ">= 62",         # comparison operators
        ">62",
        ">=62",
        "62%",           # hard-coded percent copy
        "'62% WR'",      # hard-coded label string
        '"62% WR"',
        "below 62",
        "≥62%",          # unicode ≥ in copy
        "win rate ≥62",
        "win rate >= 62",
    ]
    found = [p for p in bad_patterns if p in src]
    assert not found, (
        f"Hard-coded 62 win-rate value(s) found in index.html — use target_win_rate from backend: {found}"
    )
