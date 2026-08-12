from scripts.validate_production_release import _validate_schwab_health


def _market_data(*, connected: bool, coverage: float = 100.0) -> dict:
    return {
        "ws_streamer": {
            "connected": connected,
            "desired_subscriptions": 428,
            "acknowledged_subscriptions": 428 if connected else 0,
            "subscription_coverage_pct": coverage,
            "pending_subscription_requests": 0,
        }
    }


def _tokens(*, trader: bool = True, marketdata: bool = True) -> dict:
    return {
        "trader": {
            "connected": trader,
            "access_token_ttl_s": 1200,
            "refresh_token_ttl_s": 600000,
            "_source": "market-data",
        },
        "marketdata": {
            "connected": marketdata,
            "access_token_ttl_s": 1200,
            "refresh_token_ttl_s": 600000,
            "_source": "market-data",
        },
    }


def test_closed_session_accepts_durable_tokens_without_websocket() -> None:
    details, failures = _validate_schwab_health(
        active_session=False,
        market_data=_market_data(connected=False, coverage=0.0),
        token_statuses=_tokens(),
    )

    assert failures == []
    assert details["ws"]["connected"] is False
    assert details["schwab_tokens"]["marketdata"]["connected"] is True


def test_closed_session_still_rejects_missing_durable_token() -> None:
    _, failures = _validate_schwab_health(
        active_session=False,
        market_data=_market_data(connected=False, coverage=0.0),
        token_statuses=_tokens(marketdata=False),
    )

    assert "Schwab marketdata token is not connected" in failures


def test_active_session_requires_websocket_and_ack_coverage() -> None:
    _, failures = _validate_schwab_health(
        active_session=True,
        market_data=_market_data(connected=False, coverage=0.0),
        token_statuses=_tokens(),
    )

    assert "Schwab WebSocket is not connected" in failures
    assert "Schwab subscription ACK coverage is below 95%" in failures


def test_active_session_accepts_healthy_stream_and_tokens() -> None:
    _, failures = _validate_schwab_health(
        active_session=True,
        market_data=_market_data(connected=True),
        token_statuses=_tokens(),
    )

    assert failures == []
