import asyncio

import pytest
from fastapi import HTTPException

from agent.config_catalog import build_catalog
from agent.config_manager import _DEFAULTS, config
from auth.dependencies import AuthenticatedUser
from routers.config_router import update_config


def _defaults() -> dict:
    return {key: factory() for key, factory in _DEFAULTS.items()}


def _admin() -> AuthenticatedUser:
    return AuthenticatedUser(1, "test-admin", "ADMIN", "ACTIVE", "test-jti")


def test_catalog_contains_every_runtime_setting() -> None:
    defaults = _defaults()
    catalog = build_catalog(defaults, defaults)
    fields = {field["key"]: field for field in catalog["fields"]}

    assert set(fields) == set(_DEFAULTS)
    assert all(field["label"] for field in fields.values())
    assert all(field["description"] for field in fields.values())
    assert all(field["example"] for field in fields.values())
    assert fields["scalp.execution_enabled"]["default"] is False
    assert fields["scalp.reward_r"]["advanced"] is False


def test_invalid_scalp_relationship_is_rejected_before_write(monkeypatch) -> None:
    writes = []
    monkeypatch.setattr(config, "all", lambda: {})
    monkeypatch.setattr(
        config,
        "set_many",
        lambda updates, updated_by="system": writes.append((updates, updated_by)),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            update_config(
                {"scalp.tp1_r": 2.5, "scalp.reward_r": 2.0},
                _admin(),
            )
        )

    assert exc_info.value.status_code == 422
    assert "tp1_r" in str(exc_info.value.detail)
    assert writes == []


def test_valid_scalp_configuration_is_persisted(monkeypatch) -> None:
    writes = []
    monkeypatch.setattr(config, "all", lambda: {})
    monkeypatch.setattr(
        config,
        "set_many",
        lambda updates, updated_by="system": writes.append((updates, updated_by)),
    )

    result = asyncio.run(
        update_config(
            {"scalp.reward_r": 2.25, "scalp.tp1_r": 1.0},
            _admin(),
        )
    )

    assert result["updated"] == 2
    assert writes == [
        ({"scalp.reward_r": 2.25, "scalp.tp1_r": 1.0}, "test-admin")
    ]


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"scalp.min_stop_pct": 0.03, "scalp.max_stop_pct": 0.02}, "stop percentage"),
        ({"scalp.max_spread_to_risk": 0}, "max_spread_to_risk"),
        ({"scalp.min_rvol_regular": -0.1}, "RVOL"),
        (
            {"scalp.rsi_oversold": 75, "scalp.rsi_overbought": 70},
            "RSI boundaries",
        ),
    ],
)
def test_invalid_scalp_ranges_are_rejected(monkeypatch, updates, message) -> None:
    monkeypatch.setattr(config, "all", lambda: {})
    monkeypatch.setattr(config, "set_many", lambda *args, **kwargs: None)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(update_config(updates, _admin()))

    assert exc_info.value.status_code == 422
    assert message in str(exc_info.value.detail)
