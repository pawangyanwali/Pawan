import gzip
import io
import json
import urllib.error
from unittest.mock import patch


def test_paper_open_trades_are_enriched_from_valkey_prices():
    from routers.paper_trading import _enrich_open_trades_with_unrealized

    trades = [
        {
            "ticker": "AAPL",
            "direction": "BUY",
            "entry_price": 100.0,
            "shares": 10,
            "shares_remaining": 10,
        },
        {
            "ticker": "MSFT",
            "direction": "SELL",
            "entry_price": 50.0,
            "shares": 4,
            "shares_remaining": 4,
        },
    ]

    def fake_get_price(ticker):
        return {
            "AAPL": {"last": 103.0, "updated_at": 1_000.0, "source": "WS"},
            "MSFT": {"last": 47.5, "updated_at": 1_000.0, "source": "REST"},
        }.get(ticker)

    with patch("agent.valkey_client.get_price", side_effect=fake_get_price), \
         patch("routers.paper_trading.time.time", return_value=1_005.0):
        enriched, total = _enrich_open_trades_with_unrealized(trades)

    assert enriched[0]["current_price"] == 103.0
    assert enriched[0]["unrealized_pnl_dollar"] == 30.0
    assert enriched[0]["unrealized_pnl_pct"] == 3.0
    assert enriched[0]["unrealized_price_age_s"] == 5.0
    assert enriched[1]["current_price"] == 47.5
    assert enriched[1]["unrealized_pnl_dollar"] == 10.0
    assert total == 40.0


def test_schwab_oauth_error_decoder_handles_gzip_body():
    from agent.broker.schwab_auth import _decode_http_error_body

    payload = {
        "error": "invalid_grant",
        "error_description": "Refresh token expired or revoked",
    }
    body = gzip.compress(json.dumps(payload).encode("utf-8"))
    headers = {"Content-Encoding": "gzip"}
    err = urllib.error.HTTPError(
        url="https://api.schwabapi.com/v1/oauth/token",
        code=400,
        msg="Bad Request",
        hdrs=headers,
        fp=io.BytesIO(body),
    )

    text, parsed = _decode_http_error_body(err)

    assert "invalid_grant" in text
    assert parsed["error"] == "invalid_grant"
    assert "expired" in parsed["error_description"]
