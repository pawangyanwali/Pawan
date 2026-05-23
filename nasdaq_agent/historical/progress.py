"""Checkpoint/resume for the backfill job.

Keys stored in the JSON file:
  completed_chunks  : list of "TICKER:INTERVAL:START_MS" strings
  resampled         : list of tickers where 1min→derived resample is done
  daily_done        : list of tickers where 1day fetch is done
"""

import json
import os
import time
from pathlib import Path

PROGRESS_FILE = Path.home() / ".nasdaq_agent" / "backfill_progress.json"

_state: dict = {}


def load() -> None:
    global _state
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                _state = json.load(f)
        except Exception:
            _state = {}
    _state.setdefault("completed_chunks", [])
    _state.setdefault("resampled", [])
    _state.setdefault("daily_done", [])
    _state.setdefault("started_at", time.strftime("%Y-%m-%dT%H:%M:%S"))


def _save() -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(PROGRESS_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_state, f, indent=2)
    os.replace(tmp, str(PROGRESS_FILE))


def chunk_key(ticker: str, interval: str, start_ms: int) -> str:
    return f"{ticker}:{interval}:{start_ms}"


def is_chunk_done(ticker: str, interval: str, start_ms: int) -> bool:
    return chunk_key(ticker, interval, start_ms) in _state["completed_chunks"]


def mark_chunk_done(ticker: str, interval: str, start_ms: int) -> None:
    key = chunk_key(ticker, interval, start_ms)
    if key not in _state["completed_chunks"]:
        _state["completed_chunks"].append(key)
        _save()


def is_resampled(ticker: str) -> bool:
    return ticker in _state["resampled"]


def mark_resampled(ticker: str) -> None:
    if ticker not in _state["resampled"]:
        _state["resampled"].append(ticker)
        _save()


def is_daily_done(ticker: str) -> bool:
    return ticker in _state["daily_done"]


def mark_daily_done(ticker: str) -> None:
    if ticker not in _state["daily_done"]:
        _state["daily_done"].append(ticker)
        _save()


def reset() -> None:
    """Wipe progress and start fresh."""
    global _state
    _state = {
        "completed_chunks": [],
        "resampled":        [],
        "daily_done":       [],
        "started_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save()


def summary() -> dict:
    return {
        "chunks_done":  len(_state.get("completed_chunks", [])),
        "resampled":    len(_state.get("resampled", [])),
        "daily_done":   len(_state.get("daily_done", [])),
        "started_at":   _state.get("started_at", "unknown"),
    }
