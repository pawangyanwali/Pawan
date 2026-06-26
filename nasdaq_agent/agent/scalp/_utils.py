from __future__ import annotations

import math
from typing import Any


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def positive(value: Any) -> bool:
    return finite(value) and float(value) > 0


def number(value: Any) -> float:
    return float(value) if finite(value) else 0.0


def round_tick(value: float, tick_size: float) -> float:
    ticks = round(value / tick_size)
    decimals = max(0, len(f"{tick_size:.10f}".rstrip("0").split(".")[-1]))
    return round(ticks * tick_size, decimals)

