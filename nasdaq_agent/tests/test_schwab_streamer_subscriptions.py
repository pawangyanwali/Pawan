from __future__ import annotations

import asyncio
import json
import time

import pytest


class _FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, payload: str) -> None:
        self.messages.append(json.loads(payload))


def _commands(ws: _FakeWebSocket) -> list[tuple[str, list[str], str]]:
    result = []
    for message in ws.messages:
        request = message["requests"][0]
        result.append(
            (
                request["command"],
                request["parameters"]["keys"].split(","),
                request["requestid"],
            )
        )
    return result


def _reset(streamer) -> None:
    with streamer._lock:
        streamer._ws_desired_tickers = []
        streamer._ws_active_tickers.clear()
        streamer._ws_acknowledged_tickers.clear()
        streamer._ws_seen_at.clear()
        streamer._ws_subscription_requests.clear()
        streamer._ws_request_id = 1000


def test_subscription_reconciler_tracks_acknowledged_universe():
    from agent.broker import schwab_streamer as streamer

    _reset(streamer)
    ws = _FakeWebSocket()
    streamer.update_streamer_tickers(["aapl", "MSFT", "AAPL"])
    asyncio.run(streamer._reconcile_equity_subscriptions(
        ws,
        fields="0,1,2,3",
        customer_id="customer",
        correl_id="correl",
    ))

    commands = _commands(ws)
    assert commands == [("SUBS", ["AAPL", "MSFT"], commands[0][2])]
    request_id = commands[0][2]
    streamer._process_message(json.dumps({
        "response": [{
            "service": "LEVELONE_EQUITIES",
            "command": "SUBS",
            "requestid": request_id,
            "content": {"code": 0, "msg": "OK"},
        }]
    }))
    with streamer._lock:
        assert streamer._ws_active_tickers == {"AAPL", "MSFT"}
        assert streamer._ws_acknowledged_tickers == {"AAPL", "MSFT"}
        assert streamer._ws_subscription_requests == {}


def test_subscription_reconciler_removes_and_adds_after_hot_reload():
    from agent.broker import schwab_streamer as streamer

    _reset(streamer)
    with streamer._lock:
        streamer._ws_desired_tickers = ["AAPL", "MSFT"]
        streamer._ws_active_tickers.update({"AAPL", "MSFT"})
        streamer._ws_acknowledged_tickers.update({"AAPL", "MSFT"})

    ws = _FakeWebSocket()
    streamer.update_streamer_tickers(["AAPL", "NVDA"])
    asyncio.run(streamer._reconcile_equity_subscriptions(
        ws,
        fields="0,1,2,3",
        customer_id="customer",
        correl_id="correl",
    ))
    commands = _commands(ws)
    assert [(command, keys) for command, keys, _ in commands] == [
        ("UNSUBS", ["MSFT"]),
        ("ADD", ["NVDA"]),
    ]

    for command, _, request_id in commands:
        streamer._process_message(json.dumps({
            "response": [{
                "service": "LEVELONE_EQUITIES",
                "command": command,
                "requestid": request_id,
                "content": {"code": 0, "msg": "OK"},
            }]
        }))
    with streamer._lock:
        assert streamer._ws_active_tickers == {"AAPL", "NVDA"}
        assert streamer._ws_acknowledged_tickers == {"AAPL", "NVDA"}


def test_subscription_timeout_rolls_back_optimistic_state_for_retry():
    from agent.broker import schwab_streamer as streamer

    _reset(streamer)
    with streamer._lock:
        streamer._ws_desired_tickers = ["AAPL"]
        streamer._ws_active_tickers.add("AAPL")
        streamer._ws_subscription_requests["77"] = (
            "ADD",
            ("AAPL",),
            time.monotonic() - 11,
        )

    ws = _FakeWebSocket()
    asyncio.run(streamer._reconcile_equity_subscriptions(
        ws,
        fields="0,1,2,3",
        customer_id="customer",
        correl_id="correl",
    ))
    commands = _commands(ws)
    assert len(commands) == 1
    assert commands[0][0] == "SUBS"
    assert commands[0][1] == ["AAPL"]


def test_failed_send_does_not_leave_false_active_or_pending_state():
    from agent.broker import schwab_streamer as streamer

    class _BrokenWebSocket:
        @staticmethod
        async def send(_payload: str) -> None:
            raise ConnectionError("socket closed")

    _reset(streamer)
    streamer.update_streamer_tickers(["AAPL"])
    with pytest.raises(ConnectionError):
        asyncio.run(streamer._reconcile_equity_subscriptions(
            _BrokenWebSocket(),
            fields="0,1,2,3",
            customer_id="customer",
            correl_id="correl",
        ))
    with streamer._lock:
        assert streamer._ws_active_tickers == set()
        assert streamer._ws_subscription_requests == {}


def test_quote_activity_is_separate_from_subscription_acknowledgement(monkeypatch):
    from agent.broker import schwab_streamer as streamer

    _reset(streamer)
    now = time.time()
    with streamer._lock:
        streamer._ws_desired_tickers = ["AAPL", "MSFT"]
        streamer._ws_active_tickers.update({"AAPL", "MSFT"})
        streamer._ws_acknowledged_tickers.update({"AAPL", "MSFT"})
        streamer._ws_seen_at["AAPL"] = now
        streamer._ws_seen_at["MSFT"] = now - 30

    class _Alive:
        @staticmethod
        def is_alive() -> bool:
            return True

    monkeypatch.setattr(streamer, "_streamer_thread", _Alive())
    monkeypatch.setattr(streamer, "_ws_connected", True)
    status = streamer.get_streamer_status()["ws_streamer"]
    assert status["desired_subscriptions"] == 2
    assert status["acknowledged_subscriptions"] == 2
    assert status["subscription_coverage_pct"] == 100.0
    assert status["seen_quotes"] == 2
    assert status["active_quotes_60s"] == 2
    assert status["live_quotes"] == 1
