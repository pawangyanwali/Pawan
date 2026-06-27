from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from agent.scalp.bar_feed import _frame_from_payload
from agent.scalp.bar_hydration import (
    _normalise_frame,
    frame_is_usable,
    publish_frames_to_valkey,
)
from agent.scalp.indicators import calculate_one_minute_indicators


ROOT = Path(__file__).resolve().parents[1]


def _frame(count: int = 50, *, start: str = "2026-06-26T13:30:00Z") -> pd.DataFrame:
    index = pd.date_range(start, periods=count, freq="min")
    close = [100.0 + number * 0.01 for number in range(count)]
    return pd.DataFrame(
        {
            "Open": close,
            "High": [value + 0.05 for value in close],
            "Low": [value - 0.05 for value in close],
            "Close": close,
            "Volume": [1000.0 + number for number in range(count)],
        },
        index=index,
    )


def test_level_one_builder_produces_real_ohlcv_from_cumulative_volume():
    import agent.broker.schwab_streamer as streamer

    streamer._forming_bars.clear()
    streamer._last_cumulative_volume.clear()
    with streamer._lock:
        assert streamer._accumulate_one_minute_bar_locked(
            "AAPL", {"last": 100.0, "volume": 1000}, now=60.1
        ) is None
        assert streamer._accumulate_one_minute_bar_locked(
            "AAPL", {"last": 101.0, "volume": 1015}, now=90.0
        ) is None
        completed = streamer._accumulate_one_minute_bar_locked(
            "AAPL", {"last": 100.5, "volume": 1025}, now=120.1
        )

    assert completed == {
        "time_ms": 60_000,
        "open": 100.0,
        "high": 101.0,
        "low": 100.0,
        "close": 101.0,
        "volume": 15.0,
    }
    assert streamer._forming_bars["AAPL"]["volume"] == pytest.approx(10.0)


def test_level_one_builder_does_not_emit_weekend_quote_snapshots_as_bars():
    import agent.broker.schwab_streamer as streamer

    streamer._forming_bars.clear()
    streamer._last_cumulative_volume.clear()
    with streamer._lock:
        assert streamer._accumulate_one_minute_bar_locked(
            "AAPL", {"last": 100.0, "volume": 1000}, now=60.1
        ) is None
        completed = streamer._accumulate_one_minute_bar_locked(
            "AAPL", {"last": 100.0, "volume": 1000}, now=120.1
        )
    assert completed is None


def test_history_requires_real_volume_and_enough_bars():
    assert frame_is_usable(_frame()) is True
    no_volume = _frame()
    no_volume["Volume"] = 0.0
    assert frame_is_usable(no_volume) is False
    assert frame_is_usable(_frame(20)) is False


def test_zero_volume_polling_rows_are_removed_at_hydration_and_read_boundaries():
    frame = _frame()
    frame.loc[frame.index[-1], "Volume"] = 0.0
    assert len(_normalise_frame(frame)) == len(frame) - 1

    payload = [
        {
            "time_ms": int(index.timestamp() * 1000),
            "open": row.Open,
            "high": row.High,
            "low": row.Low,
            "close": row.Close,
            "volume": row.Volume,
        }
        for index, row in frame.iterrows()
    ]
    assert len(_frame_from_payload(payload)) == len(frame) - 1


class _Pipe:
    def __init__(self):
        self.calls = []

    def delete(self, *args): self.calls.append(("delete", args)); return self
    def rpush(self, *args): self.calls.append(("rpush", args)); return self
    def ltrim(self, *args): self.calls.append(("ltrim", args)); return self
    def expire(self, *args): self.calls.append(("expire", args)); return self
    def execute(self): return [True] * len(self.calls)


class _Client:
    def __init__(self): self.pipes = []
    def pipeline(self, transaction=False):
        pipe = _Pipe(); self.pipes.append((transaction, pipe)); return pipe


def test_hydration_replaces_valkey_history_with_authoritative_ohlcv(monkeypatch):
    import agent.valkey_client as valkey

    client = _Client()
    monkeypatch.setattr(valkey, "_get_client", lambda: client)
    assert publish_frames_to_valkey({"AAPL": _frame()}) == 1
    transaction, pipe = client.pipes[0]
    assert transaction is True
    assert pipe.calls[0] == ("delete", ("md:1m:AAPL",))
    pushed = next(args for name, args in pipe.calls if name == "rpush")
    payload = json.loads(pushed[-1])
    assert payload["volume"] > 0
    assert payload["time_ms"] > 0
    assert next(args[-1] for name, args in pipe.calls if name == "expire") >= 604800


def test_vwap_and_rvol_reset_at_regular_session_open():
    premarket = _frame(13, start="2026-06-26T13:18:00Z")
    enriched = calculate_one_minute_indicators(premarket)
    regular_open = enriched.loc[pd.Timestamp("2026-06-26T13:30:00Z")]
    source_open = premarket.loc[pd.Timestamp("2026-06-26T13:30:00Z")]
    expected_typical = (source_open["High"] + source_open["Low"] + source_open["Close"]) / 3.0
    assert regular_open["vwap"] == pytest.approx(expected_typical)
    assert pd.isna(regular_open["vol_ratio"])


def test_rvol_uses_prior_same_session_bars_across_trading_days():
    index = pd.to_datetime(
        [f"2026-06-{day:02d}T20:00:00Z" for day in range(10, 22)]
    )
    frame = pd.DataFrame(
        {
            "Open": [100.0] * 12,
            "High": [100.1] * 12,
            "Low": [99.9] * 12,
            "Close": [100.0] * 12,
            "Volume": [1000.0] * 11 + [2000.0],
        },
        index=index,
    )
    enriched = calculate_one_minute_indicators(frame)
    assert enriched.iloc[-1]["vol_ratio"] == pytest.approx(2.0)


def test_production_bar_contract_is_durable_and_self_healing():
    streamer = (ROOT / "agent" / "broker" / "schwab_streamer.py").read_text(encoding="utf-8")
    service = (ROOT / "services" / "market_data_service.py").read_text(encoding="utf-8")
    cache = (ROOT / "agent" / "historical_cache.py").read_text(encoding="utf-8")
    assert "_accumulate_one_minute_bar_locked" in streamer
    assert "7 * 24 * 60 * 60" in streamer
    assert "_bar_hydration_loop" in service
    assert "hydrate_one_minute_history" in service
    assert '"bar_hydration": dict(_bar_hydration_status)' in service
    assert "DO UPDATE SET" in cache
    assert "EXCLUDED.volume > 0" in cache
