"""Stable import facade for the greenfield scalping-only plan engine.

Release 1 is shadow-only: importing this module does not start services, access
storage, or alter the legacy scanner and paper execution paths.
"""

from agent.scalp import (
    BracketGeometry,
    IndicatorSnapshot,
    PathQuality,
    QuoteSnapshot,
    QuoteSource,
    ScalpSignalConfig,
    ScalpSignalPlan,
    SignalSide,
    build_bracket,
    create_scalp_signal_plan,
    detect_scalp_signal_plan,
    indicator_snapshot_from_frame,
    quote_snapshot_from_price_bus,
)

__all__ = [
    "BracketGeometry",
    "IndicatorSnapshot",
    "PathQuality",
    "QuoteSnapshot",
    "QuoteSource",
    "ScalpSignalConfig",
    "ScalpSignalPlan",
    "SignalSide",
    "build_bracket",
    "create_scalp_signal_plan",
    "detect_scalp_signal_plan",
    "indicator_snapshot_from_frame",
    "quote_snapshot_from_price_bus",
]
