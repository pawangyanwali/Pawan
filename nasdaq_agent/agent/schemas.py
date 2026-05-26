"""
Lightweight shared type definitions.

Imported by web-api without pulling in scanner/ML/broker modules.
All types here are pure Python — no pandas, numpy, sklearn, or torch.
"""
from __future__ import annotations

from typing import Any


class SignalSnapshot:
    """
    Serialised form of a completed scan cycle stored in Valkey.
    web-api reads this on startup and after each scan notification
    instead of waiting for the scanner thread to call back directly.
    """
    __slots__ = ("ts", "signals", "regime", "session", "scanned_count")

    def __init__(
        self,
        ts: float,
        signals: list[dict[str, Any]],
        regime: dict[str, Any],
        session: dict[str, Any],
        scanned_count: int,
    ) -> None:
        self.ts            = ts
        self.signals       = signals
        self.regime        = regime
        self.session       = session
        self.scanned_count = scanned_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts":            self.ts,
            "signals":       self.signals,
            "regime":        self.regime,
            "session":       self.session,
            "scanned_count": self.scanned_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SignalSnapshot":
        return cls(
            ts            = float(d.get("ts", 0)),
            signals       = d.get("signals", []),
            regime        = d.get("regime", {}),
            session       = d.get("session", {}),
            scanned_count = int(d.get("scanned_count", 0)),
        )
