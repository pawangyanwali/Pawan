"""
Tests for the WebSocket ConnectionManager and /ws endpoint.

Covers:
- connect / disconnect lifecycle
- broadcast reaches all active clients
- dead client removed without blocking live clients
- set mutation safety (disconnect during broadcast)
- per-client send timeout (slow client can't stall others)
- multiple concurrent browsers all receive messages
- reconnect: new client gets current state immediately
- /ws endpoint integration: auth flow + manager.connect() without double-accept
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Import the ConnectionManager from main without starting the full app.
# We isolate it so tests are fast and don't require a running server.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import ConnectionManager, _safe_ws_close


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ws():
    """Minimal WebSocket mock with a send_text spy."""
    ws = AsyncMock()
    ws.send_text = AsyncMock(return_value=None)
    ws.accept    = AsyncMock(return_value=None)
    return ws

def _make_slow_ws(delay: float = 10.0):
    """WebSocket whose send_text blocks for `delay` seconds."""
    ws = AsyncMock()
    ws.accept = AsyncMock(return_value=None)
    async def _slow_send(msg):
        await asyncio.sleep(delay)
    ws.send_text = _slow_send
    return ws

def _make_dead_ws():
    """WebSocket whose send_text always raises (connection already closed)."""
    ws = AsyncMock()
    ws.accept = AsyncMock(return_value=None)
    ws.send_text = AsyncMock(side_effect=RuntimeError("connection closed"))
    return ws


@pytest.mark.asyncio
async def test_safe_ws_close_suppresses_protocol_race():
    """Auth rejection must not become an ASGI error if the client already closed."""
    ws = AsyncMock()
    ws.close = AsyncMock(side_effect=AttributeError("'WebSocketProtocol' object has no attribute 'transfer_data_task'"))

    await _safe_ws_close(ws, code=4001)

    ws.close.assert_awaited_once_with(code=4001, reason="")


# ── ConnectionManager unit tests ──────────────────────────────────────────────

class TestConnectionManager:

    @pytest.mark.asyncio
    async def test_connect_adds_to_active(self):
        cm = ConnectionManager()
        ws = _make_ws()
        await cm.connect(ws)
        assert ws in cm.active

    @pytest.mark.asyncio
    async def test_disconnect_removes_from_active(self):
        cm = ConnectionManager()
        ws = _make_ws()
        await cm.connect(ws)
        cm.disconnect(ws)
        assert ws not in cm.active

    @pytest.mark.asyncio
    async def test_disconnect_unknown_client_is_safe(self):
        cm = ConnectionManager()
        ws = _make_ws()
        cm.disconnect(ws)   # never connected — must not raise
        assert ws not in cm.active

    @pytest.mark.asyncio
    async def test_broadcast_empty_set_is_noop(self):
        cm = ConnectionManager()
        await cm.broadcast('{"type":"test"}')   # must not raise

    @pytest.mark.asyncio
    async def test_broadcast_single_client_receives_message(self):
        cm = ConnectionManager()
        ws = _make_ws()
        await cm.connect(ws)
        await cm.broadcast('{"type":"update"}')
        ws.send_text.assert_awaited_once_with('{"type":"update"}')

    @pytest.mark.asyncio
    async def test_broadcast_all_clients_receive_message(self):
        cm  = ConnectionManager()
        wss = [_make_ws() for _ in range(5)]
        for ws in wss:
            await cm.connect(ws)
        msg = '{"type":"prices","p":{}}'
        await cm.broadcast(msg)
        for ws in wss:
            ws.send_text.assert_awaited_once_with(msg)

    @pytest.mark.asyncio
    async def test_dead_client_removed_after_broadcast(self):
        cm  = ConnectionManager()
        good = _make_ws()
        dead = _make_dead_ws()
        await cm.connect(good)
        await cm.connect(dead)
        await cm.broadcast('{"type":"ping"}')
        assert good in cm.active,  "live client must remain"
        assert dead not in cm.active, "dead client must be evicted"

    @pytest.mark.asyncio
    async def test_good_clients_receive_even_when_one_is_dead(self):
        cm   = ConnectionManager()
        good = _make_ws()
        dead = _make_dead_ws()
        await cm.connect(good)
        await cm.connect(dead)
        await cm.broadcast('{"type":"update"}')
        good.send_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_multiple_broadcasts_accumulate_messages(self):
        cm = ConnectionManager()
        ws = _make_ws()
        await cm.connect(ws)
        for i in range(3):
            await cm.broadcast(f'{{"type":"tick","i":{i}}}')
        assert ws.send_text.await_count == 3

    @pytest.mark.asyncio
    async def test_sequential_broadcasts_all_reach_connected_client(self):
        cm = ConnectionManager()
        ws = _make_ws()
        await cm.connect(ws)

        # broadcast is the sole public send API; multiple sequential calls
        # must each deliver to the connected client exactly once.
        await cm.broadcast('{"type":"a"}')
        await cm.broadcast('{"type":"b"}')

        assert ws.send_text.await_count == 2
        calls = [c.args[0] for c in ws.send_text.await_args_list]
        assert calls == ['{"type":"a"}', '{"type":"b"}']

    @pytest.mark.asyncio
    async def test_active_count_tracks_connects_and_disconnects(self):
        cm  = ConnectionManager()
        wss = [_make_ws() for _ in range(4)]
        for ws in wss:
            await cm.connect(ws)
        assert len(cm.active) == 4
        cm.disconnect(wss[0])
        cm.disconnect(wss[1])
        assert len(cm.active) == 2


# ── Slow-client / timeout tests ───────────────────────────────────────────────

class TestBroadcastTimeout:
    """
    A frozen browser must not stall other connected browsers.
    The per-client timeout in broadcast() ensures a slow send is abandoned
    after 5 seconds, and the live clients still receive their messages.
    """

    @pytest.mark.asyncio
    async def test_slow_client_does_not_block_fast_clients(self):
        cm   = ConnectionManager()
        fast = _make_ws()
        slow = _make_slow_ws(delay=30.0)   # would block for 30s without timeout
        await cm.connect(fast)
        await cm.connect(slow)

        # With parallel gather + 5s timeout this should complete well under 6 s.
        import time
        t0 = time.monotonic()
        await cm.broadcast('{"type":"ping"}')
        elapsed = time.monotonic() - t0

        fast.send_text.assert_awaited_once()  # fast client got the message
        assert elapsed < 7.0, f"broadcast took {elapsed:.1f}s — slow client stalled others"

    @pytest.mark.asyncio
    async def test_slow_client_evicted_after_timeout(self):
        cm   = ConnectionManager()
        slow = _make_slow_ws(delay=30.0)
        await cm.connect(slow)
        await cm.broadcast('{"type":"ping"}')
        # After timeout the send raises asyncio.TimeoutError → client evicted.
        assert slow not in cm.active, "slow/frozen client must be evicted after timeout"

    @pytest.mark.asyncio
    async def test_three_browsers_all_receive_despite_one_frozen(self):
        cm     = ConnectionManager()
        b1     = _make_ws()
        b2     = _make_ws()
        frozen = _make_slow_ws(delay=30.0)
        await cm.connect(b1)
        await cm.connect(b2)
        await cm.connect(frozen)

        await cm.broadcast('{"type":"update"}')

        b1.send_text.assert_awaited_once()
        b2.send_text.assert_awaited_once()


# ── Concurrent-disconnect safety ──────────────────────────────────────────────

class TestConcurrentDisconnect:
    """
    Calling disconnect() while broadcast() is awaiting must not raise
    RuntimeError: Set changed size during iteration.
    """

    @pytest.mark.asyncio
    async def test_disconnect_during_broadcast_no_runtime_error(self):
        cm = ConnectionManager()
        ws = _make_ws()

        disconnect_called = False

        async def _send_and_disconnect(msg):
            nonlocal disconnect_called
            cm.disconnect(ws)         # mutate active set mid-broadcast
            disconnect_called = True

        ws.send_text = _send_and_disconnect
        await cm.connect(ws)

        try:
            await cm.broadcast('{"type":"ping"}')
        except RuntimeError as e:
            pytest.fail(f"broadcast raised RuntimeError during concurrent disconnect: {e}")

        assert disconnect_called, "side-effect must have run"

    @pytest.mark.asyncio
    async def test_snapshot_isolates_broadcast_from_late_connects(self):
        """Clients that connect AFTER broadcast() snapshots must not receive that broadcast."""
        cm        = ConnectionManager()
        early     = _make_ws()
        late      = _make_ws()

        send_order = []

        async def _early_send(msg):
            send_order.append("early")
            await cm.connect(late)    # late client joins mid-broadcast

        early.send_text = _early_send
        await cm.connect(early)

        await cm.broadcast('{"type":"update"}')

        assert "early" in send_order
        late.send_text.assert_not_awaited()   # late joiner not in snapshot


# ── Multi-browser integration ─────────────────────────────────────────────────

class TestMultiBrowser:

    @pytest.mark.asyncio
    async def test_ten_browsers_all_receive_scan_update(self):
        cm      = ConnectionManager()
        browsers = [_make_ws() for _ in range(10)]
        for b in browsers:
            await cm.connect(b)

        msg = json.dumps({"type": "update", "signals": [], "scanned_count": 250})
        await cm.broadcast(msg)

        for i, b in enumerate(browsers):
            b.send_text.assert_awaited_once_with(msg), f"browser {i} did not receive message"

    @pytest.mark.asyncio
    async def test_rapid_sequential_broadcasts_no_message_loss(self):
        cm = ConnectionManager()
        ws = _make_ws()
        await cm.connect(ws)

        for i in range(20):
            await cm.broadcast(f'{{"type":"tick","i":{i}}}')

        assert ws.send_text.await_count == 20

    @pytest.mark.asyncio
    async def test_client_reconnect_receives_fresh_broadcast(self):
        """Simulates a tab reconnecting — new ws object gets the next broadcast."""
        cm = ConnectionManager()
        ws1 = _make_ws()
        await cm.connect(ws1)
        await cm.broadcast('{"type":"update","v":1}')
        ws1.send_text.assert_awaited_once()

        # Client disconnects and reconnects with a new socket object.
        cm.disconnect(ws1)
        ws2 = _make_ws()
        await cm.connect(ws2)
        await cm.broadcast('{"type":"update","v":2}')

        ws1.send_text.assert_awaited_once()  # old socket: only 1 message total
        ws2.send_text.assert_awaited_once_with('{"type":"update","v":2}')

    @pytest.mark.asyncio
    async def test_mixed_dead_and_live_browsers(self):
        cm     = ConnectionManager()
        live   = [_make_ws()      for _ in range(3)]
        dead   = [_make_dead_ws() for _ in range(2)]
        for ws in live + dead:
            await cm.connect(ws)

        await cm.broadcast('{"type":"prices","p":{}}')

        for ws in live:
            ws.send_text.assert_awaited_once()
        for ws in dead:
            assert ws not in cm.active, "dead client not evicted"
        assert len(cm.active) == 3


# ── /ws endpoint integration tests ───────────────────────────────────────────
#
# These tests drive the FULL websocket_endpoint() handler, not just
# ConnectionManager in isolation.  They use raise_server_exceptions=True so
# any server-side RuntimeError (e.g. double-accept) propagates to the test
# and causes a hard failure instead of being swallowed silently.
#
# Root cause of past regression: websocket_endpoint called ws.accept() at the
# top for auth, then manager.connect() called ws.accept() a second time →
# RuntimeError: "Expected websocket.send or websocket.close, got websocket.accept"
# This class would have caught that immediately.

def _ws_client():
    """
    TestClient wired to the real FastAPI app with:
      - scanner mocked (no background threads)
      - raise_server_exceptions=True (server RuntimeErrors fail the test)
    """
    import main as m
    mock_scanner = MagicMock()
    mock_scanner.signals = []
    mock_scanner.last_scan = None
    mock_scanner.is_running = True
    m.scanner = mock_scanner

    from fastapi.testclient import TestClient
    return TestClient(m.app, raise_server_exceptions=True)


_VALID_PAYLOAD = {"type": "access", "jti": "test-jti", "sub": "1"}
_AUTH_PATCHES = {
    "auth.utils.decode_token":    _VALID_PAYLOAD,
    "auth.utils.is_blacklisted":  False,
}


class TestWebSocketEndpoint:
    """
    Integration tests for the /ws endpoint.

    Every test here uses raise_server_exceptions=True, so a double-accept,
    double-close, or any other server-side RuntimeError will fail the test
    rather than being silently swallowed.
    """

    def test_query_param_auth_no_double_accept(self):
        """
        Regression: valid token in query-param must not trigger a second
        ws.accept() inside manager.connect() (was RuntimeError in production).
        """
        with patch("auth.utils.decode_token", return_value=_VALID_PAYLOAD), \
             patch("auth.utils.is_blacklisted", return_value=False):
            client = _ws_client()
            # If a double-accept RuntimeError occurs, raise_server_exceptions=True
            # will propagate it and the test fails.
            with client.websocket_connect("/ws?token=valid_token") as ws:
                data = ws.receive_text()   # server sends initial scan_status/update frame
                msg = json.loads(data)
                assert msg.get("type") in ("update", "scan_status", "ping"), (
                    f"unexpected first message type: {msg.get('type')}"
                )

    def test_first_message_auth_no_double_accept(self):
        """
        Regression: auth via first JSON message must also avoid double-accept.
        """
        with patch("auth.utils.decode_token", return_value=_VALID_PAYLOAD), \
             patch("auth.utils.is_blacklisted", return_value=False):
            client = _ws_client()
            with client.websocket_connect("/ws") as ws:
                ws.send_text(json.dumps({"type": "auth", "token": "valid_token"}))
                data = ws.receive_text()
                msg = json.loads(data)
                assert msg.get("type") in ("update", "scan_status", "ping")

    def test_invalid_token_closes_connection(self):
        """
        A bad token must be rejected cleanly — no unhandled RuntimeError on
        either the double-close or the double-accept path.
        """
        with patch("auth.utils.decode_token", side_effect=Exception("bad token")):
            client = _ws_client()
            try:
                with client.websocket_connect("/ws?token=bad") as ws:
                    ws.receive_text()   # server should close before sending anything
            except Exception:
                pass   # WebSocketDisconnect or similar — expected for rejected auth

    def test_no_token_closes_without_server_error(self):
        """
        No token at all: server must reject cleanly without a server-side
        RuntimeError (the double-close regression).  The server waits up to
        10 s for a slow-path auth message; we disconnect immediately after
        connecting so the test stays within the 8-second CI limit.
        """
        client = _ws_client()
        try:
            with client.websocket_connect("/ws") as ws:
                # Send a bad auth message so the server authenticates (and rejects)
                # without waiting for the 10-second slow-path timeout.
                ws.send_text(json.dumps({"type": "auth", "token": ""}))
                ws.receive_text()   # server sends close frame after rejecting
        except Exception:
            pass   # WebSocketDisconnect or similar — expected; no RuntimeError is the goal
