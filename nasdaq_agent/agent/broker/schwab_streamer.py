"""
Schwab WebSocket Streamer â€” real-time market data pipeline.

Why this matters (thinking like a 40-year trader):
  - REST polling gives you data that is 5â€“30s stale. At scalping timeframes
    (30sâ€“2min holds) that means you are always acting on yesterday's news.
  - Streaming gives you sub-second updates. Bid/ask imbalance, halt detection,
    and candle closes arrive the instant they happen on the exchange.
  - NQ/ES futures lead NASDAQ equities by 30â€“90 seconds. Watching futures
    drift down BEFORE the stocks follow is the edge every professional has.
  - Bid/ask size imbalance (bid_size â€“ ask_size) / (bid_size + ask_size) is the
    single most predictive real-time directional signal for scalps.

Services subscribed:
  LEVELONE_EQUITIES   â€” real-time bid/ask/volume and all-universe 1-min bars
  SCREENER_EQUITY     â€” top movers on NASDAQ (scanner priority)
  LEVELONE_FUTURES    â€” /NQ and /ES for macro direction bias

Consumed by:
  scanner.py   â†’ real-time quotes + halt detection + screener priority
  ml_model.py  â†’ bid_ask_imbalance feature, nq_futures_bias feature
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

import requests

from agent.broker.schwab_auth import get_access_token, get_token_status

logger = logging.getLogger(__name__)

TRADER_BASE = "https://api.schwabapi.com/trader/v1"

# â”€â”€ Live data stores (thread-safe via _lock) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_lock              = threading.Lock()
_live_quotes:  dict[str, dict]        = {}   # ticker â†’ quote dict
_live_candles: dict[str, deque]       = {}   # ticker â†’ rolling 1-min OHLCV
_forming_bars: dict[str, dict]        = {}   # ticker â†’ current L1-derived 1-min bar
_last_cumulative_volume: dict[str, float] = {}
_screener_up:  list[dict]             = []   # NASDAQ top % gainers (last update)
_screener_down: list[dict]            = []   # NASDAQ top % losers
_screener_vol:  list[dict]            = []   # NASDAQ top volume
_futures:      dict[str, dict]        = {}   # /NQ, /ES â†’ quote dict
_halted:       set[str]               = set()

# â”€â”€ Bar-close event bus â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import queue as _q
_bar_close_queue: _q.Queue = _q.Queue(maxsize=20000)
_bar_close_callbacks: list = []

# â”€â”€ Completed one-minute bars â†’ PostgreSQL persistence queue â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Non-blocking: WebSocket handler puts completed bars here; a daemon worker
# drains and batch-writes them to ohlcv_bars.  Sized for a full trading day
# of 1-min bars for 300 tickers (300 Ã— 390 = 117,000) with headroom.
_bar_persist_queue: _q.Queue = _q.Queue(maxsize=200_000)
_bar_persist_worker_started: bool = False

# â”€â”€ WebSocket streamer lifecycle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_streamer_thread:  Optional[threading.Thread] = None   # WS streamer thread
_streamer_start_lock = threading.Lock()
_event_loop:       Optional[asyncio.AbstractEventLoop] = None
_ws_connected:     bool = False
_ws_error:         Optional[str] = None
_ws_desired_tickers: list[str] = []
_ws_active_tickers: set[str] = set()
_ws_acknowledged_tickers: set[str] = set()
_ws_seen_at: dict[str, float] = {}
_ws_subscription_requests: dict[
    str, tuple[str, tuple[str, ...], float]
] = {}
# Keep dynamic equity request IDs away from fixed ADMIN, screener, and futures
# IDs (1, 300, 400) so a long-running socket cannot misattribute an ACK.
_ws_request_id: int = 1000

# REST and WebSocket subscriptions are deliberately independent. A previous
# shared list let a REST universe refresh make WS health claim symbols that
# had never been added to the active socket.
_mdpoller_tickers: list[str] = []

# Prices accumulated from WS stream â€” flushed to Valkey every 500 ms
_pending_ws_prices: dict[str, dict] = {}
_open_backfill_lock = threading.Lock()
_open_backfill_started: set[str] = set()

# â”€â”€ MDPoller lifecycle (separate from WS streamer) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_mdpoller_thread:    Optional[threading.Thread] = None
_mdpoller_running:   bool = False
_mdpoller_cycle:     int  = 0
_mdpoller_last_ok:   float = 0.0   # epoch of last successful cycle
_mdpoller_error:     Optional[str] = None

# â”€â”€ WS data freshness tracking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Updated every time LEVELONE_EQUITIES data arrives from Schwab WebSocket.
# MDPoller checks this to decide whether to fire a REST call or stand down.
_last_ws_data_at: float = 0.0

MAX_CANDLE_HISTORY = 2500  # several sessions for time-of-day RVOL baselines
_WS_STANDDOWN_FRESH_PCT = float(os.getenv("NASDAQ_WS_STANDDOWN_FRESH_PCT", "0.95"))
_WS_QUOTE_PRIORITY_TTL_S = max(
    1.0, float(os.getenv("NASDAQ_WS_QUOTE_PRIORITY_TTL_S", "5.0"))
)

# â”€â”€ Real-time tick callback registry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Registered functions are called on every LEVELONE_EQUITIES update.
# Throttled per-ticker to _TICK_MIN_INTERVAL seconds to avoid flooding WebSocket.
_tick_callbacks:      list          = []
_bulk_price_callbacks: list         = []   # fn(prices: dict[str, dict]) â€” one call per poll cycle
_last_tick_ts:        dict[str, float] = {}
_TICK_MIN_INTERVAL:   float         = 0.25   # max 4 price updates/s per ticker


def register_tick_callback(fn) -> None:
    """Register fn(ticker: str, quote: dict) â€” called on every throttled tick."""
    _tick_callbacks.append(fn)


def register_bulk_price_callback(fn) -> None:
    """
    Register fn(prices: dict[str, dict]) â€” called ONCE per poll cycle with ALL
    updated quotes.  Much more efficient than 477 individual tick callbacks.
    Each value dict contains: last, bid, ask, volume, high, low, pct_change.
    """
    _bulk_price_callbacks.append(fn)


# â”€â”€ User Preferences (provides streamer URL + client IDs) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _get_streamer_info() -> dict:
    """
    GET /trader/v1/userpreference â€” returns streamer credentials.
    Response includes streamerSocketUrl, schwabClientCustomerId, etc.
    """
    token = get_access_token()
    if not token:
        raise RuntimeError("No access token â€” run Schwab OAuth first")
    r = requests.get(
        f"{TRADER_BASE}/userPreference",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=15,
    )
    r.raise_for_status()
    prefs = r.json()
    info_list = prefs.get("streamerInfo", [])
    if not info_list:
        raise RuntimeError("No streamerInfo in user preferences response")
    return info_list[0]


# â”€â”€ Front-month futures contract â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _front_month(root: str) -> str:
    """
    Return the active front-month futures symbol, e.g. '/NQM26'.
    Quarterly expirations: March(H), June(M), September(U), December(Z).
    We roll ~2 weeks before expiry (3rd Friday â‰ˆ day 18-21 of month).
    """
    d = date.today()
    quarters = [(3, "H"), (6, "M"), (9, "U"), (12, "Z")]
    yy = str(d.year)[2:]
    for month, code in quarters:
        # Roll to next quarter after ~18th of expiry month
        if d.month < month or (d.month == month and d.day < 18):
            return f"/{root}{code}{yy}"
    # Past December of this year â€” go to March of next year
    return f"/{root}H{str(d.year + 1)[2:]}"


FUTURES_SYMBOLS = [_front_month("NQ"), _front_month("ES")]


# â”€â”€ Request builders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _req(service: str, command: str, reqid: int, params: dict,
         customer_id: str, correl_id: str) -> dict:
    return {
        "requestid":              str(reqid),
        "service":                service,
        "command":                command,
        "SchwabClientCustomerId": customer_id,
        "SchwabClientCorrelId":   correl_id,
        "parameters":             params,
    }


def _next_ws_request_id() -> int:
    global _ws_request_id
    with _lock:
        _ws_request_id += 1
        return _ws_request_id


async def _send_equity_subscription(
    ws,
    *,
    command: str,
    tickers: list[str],
    fields: str,
    customer_id: str,
    correl_id: str,
) -> None:
    """Send one tracked LEVELONE_EQUITIES subscription command."""
    if not tickers:
        return
    request_id = _next_ws_request_id()
    symbols = tuple(dict.fromkeys(str(t).upper() for t in tickers if t))
    with _lock:
        _ws_subscription_requests[str(request_id)] = (
            command,
            symbols,
            time.monotonic(),
        )
    try:
        await ws.send(json.dumps({"requests": [_req(
            "LEVELONE_EQUITIES",
            command,
            request_id,
            {"keys": ",".join(symbols), "fields": fields},
            customer_id,
            correl_id,
        )]}))
    except Exception:
        with _lock:
            _ws_subscription_requests.pop(str(request_id), None)
        raise
    with _lock:
        if command == "UNSUBS":
            _ws_active_tickers.difference_update(symbols)
        else:
            _ws_active_tickers.update(symbols)


async def _reconcile_equity_subscriptions(
    ws,
    *,
    fields: str,
    customer_id: str,
    correl_id: str,
) -> None:
    """Make the active socket match the latest registry-owned universe."""
    with _lock:
        expired = [
            request_id
            for request_id, (_, _, sent_at) in _ws_subscription_requests.items()
            if time.monotonic() - sent_at >= 10.0
        ]
        for request_id in expired:
            command, symbols, _ = _ws_subscription_requests.pop(request_id)
            if command == "UNSUBS":
                _ws_active_tickers.update(symbols)
            else:
                _ws_active_tickers.difference_update(symbols)
        if expired:
            logger.warning(
                "[Streamer] %d subscription request(s) timed out; retrying",
                len(expired),
            )
        desired = set(_ws_desired_tickers)
        active = set(_ws_active_tickers)
    additions = sorted(desired - active)
    removals = sorted(active - desired)
    batch_size = 100

    for index in range(0, len(removals), batch_size):
        batch = removals[index:index + batch_size]
        await _send_equity_subscription(
            ws,
            command="UNSUBS",
            tickers=batch,
            fields=fields,
            customer_id=customer_id,
            correl_id=correl_id,
        )
    for index in range(0, len(additions), batch_size):
        batch = additions[index:index + batch_size]
        await _send_equity_subscription(
            ws,
            command="ADD" if active or index else "SUBS",
            tickers=batch,
            fields=fields,
            customer_id=customer_id,
            correl_id=correl_id,
        )
# â”€â”€ Message processing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_EQUITY_FIELDS = {
    "1": "bid",   "2": "ask",   "3": "last",
    "4": "bid_size", "5": "ask_size",
    "8": "volume", "10": "high", "11": "low", "12": "prev_close",
    "18": "net_change", "19": "week52_high", "20": "week52_low",
    "32": "status", "33": "mark", "42": "net_pct_change",
    "48": "hard_to_borrow", "49": "shortable",
}

_FUTURES_FIELDS = {
    "1": "bid", "2": "ask", "3": "last",
    "8": "volume", "19": "net_change", "20": "pct_change",
    "23": "open_interest",
}

_CHART_FIELDS = {
    # Schwab CHART_EQUITY field mapping (empirically verified):
    #   "1" = sequence / chart-day counter â€” NOT the open price, omit it
    #   "2" = open, "3" = high, "4" = low, "5" = close, "7" = epoch-ms timestamp
    # Field 6 is used when Schwab supplies bar volume. LEVELONE cumulative-volume
    # deltas cover the full 477-symbol universe independently of CHART limits.
    "2": "open", "3": "high", "4": "low", "5": "close", "6": "volume",
    "7": "time_ms",
}


def _accumulate_one_minute_bar_locked(
    sym: str, quote: dict, *, now: float | None = None
) -> dict | None:
    """Update an OHLCV bar from a Level One snapshot; caller holds ``_lock``."""
    current_time = time.time() if now is None else float(now)
    minute_ms = int(current_time // 60) * 60_000
    price = float(quote.get("last") or quote.get("mark") or 0.0)
    if price <= 0:
        return None
    cumulative = max(0.0, float(quote.get("volume") or 0.0))
    previous_total = _last_cumulative_volume.get(sym)
    volume_delta = (
        cumulative - previous_total
        if cumulative > 0 and previous_total is not None and cumulative >= previous_total
        else 0.0
    )
    # A transient REST payload can omit total volume. Do not reset a valid
    # cumulative baseline to zero or the next WS tick would count the entire
    # session as one minute's volume.
    if cumulative > 0:
        _last_cumulative_volume[sym] = cumulative

    current = _forming_bars.get(sym)
    completed = None
    if current and int(current.get("time_ms") or 0) < minute_ms:
        completed = dict(current)
        current = None
    if current is None:
        current = {
            "time_ms": minute_ms,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": max(0.0, volume_delta),
        }
        _forming_bars[sym] = current
    else:
        current["high"] = max(float(current["high"]), price)
        current["low"] = min(float(current["low"]), price)
        current["close"] = price
        current["volume"] = max(
            0.0, float(current.get("volume") or 0.0) + volume_delta
        )
    # REST quote polling continues on weekends and outside equity sessions.
    # Repeated snapshots can advance the wall-clock minute without a trade;
    # those are not market bars and would poison VWAP/RVOL with zero volume.
    if complet×nûÖÚ$z{-®éÜj×Væ—fW'6R—2V×G’ ¢F–ÖRç6ÆVWƒ2¢6öçF–çVP ¢27V&Ö—BÆÂ&F6†W26–×VÇFæV÷W6Ç’à¢2V6‚v÷&¶W"'&öF67G2F†RÖöÖVçB—G2’6ÆÂ&WGW&ç2(	BæòÖW&v–ærÀÐ¢2æòv—F–ærf÷"6–&Æ–æw2âF†Rg&öçFVæB&V6V—fW2â6W&FR&–6W6 Ð¢2ÖW76vW2–â&–B7V66W76–öâæB&VæFW'2V6‚w&÷W–ÖÖVF–FVÇ’àÐ¢gWGW&W2Ò°Ð¢öfWF6…÷ööÂç7V&Ö—B…öfWF6…ö&F6…öæE÷7G&VÒÂ&F6‚Ð¢f÷"&F6‚–â7–6ÆUö&F6†W0¢ÐÐ Ð¢2v—BöæÇ’Fò¶æ÷rv†VâF†R6Æ÷vW7B&F6‚f–æ—6†W26òvR6àÐ¢26Æ7VÆFRF†R6÷'&V7B6ÆVWF–ÖRf÷"F†RæW‡B7–6ÆRàÐ¢2F–ÖV÷WBÒ–çFW'fÂ£bv—fW2'27W6†–öâ÷fW"F†RG2&WVW7@Ð¢2F–ÖV÷WB6ògWGW&W2æWfW"V"'F–ÖVB÷WB"f÷"6Æ÷rÖ'WB×fÆ–@Ð¢2…EE2&÷VæB×G&—…54Â†æG6†¶R²Æ&vR¥4ôâ–ÆöB’àÐ¢FöæRÂVæF–ærÒö6bçv—B†gWGW&W2ÂF–ÖV÷WCÖ–çFW'fÂ¢bÐ Ð¢åöö²Ò7VÒ€Ð¢f÷"b–âFöæPÐ¢–bæ÷Bbæ6æ6VÆÆVB‚’æBbæW†6WF–öâ‚’—2æöæRæBbç&W7VÇB‚Ð¢Ð¢–båöö³ ¢öÖGöÆÆW%öW'&÷"ÒæöæP¢öÖGöÆÆW%öÆ7Eöö²ÒF–ÖRçF–ÖR‚¢6öç6V7WF—fUöÖ—72Ò ¢VÇ6S Ð¢6öç6V7WF—fUöÖ—72³ÒÐ¢2öæÇ’ÆörWfW'’Ö—76W2†öæ6RW"æ&6¶öfbv–æF÷r’Fòfö–B7ÐÐ¢–b6öç6V7WF—fUöÖ—72ÓÒ÷"6öç6V7WF—fUöÖ—72RÓÒ Ð¢ÆövvW"çv&æ–ær€Ð¢b%´ÔEöÆÆW%Ò7–6ÆR¶7–6ÆWÓ¢ÆÂ¶ÆVâ†7–6ÆUö&F6†W2—Ò&F6†W2V×G’ ¢b"‡¶6öç6V7WF—fUöÖ—77Ò6öç6V7WF—fR(	B&FRÖÆ–Ö—FVB’ Ð¢Ð¢öÖGöÆÆW%öW'&÷"Òb'&FRÖÆ–Ö—FVB‡¶6öç6V7WF—fUöÖ—77Ò6öç6V7WF—fRV×G’7–6ÆW2’ Ð Ð¢–bVæF–æs Ð¢ÆövvW"çv&æ–ær†b%´ÔEöÆÆW%Ò7–6ÆR¶7–6ÆWÓ¢¶ÆVâ‡VæF–ær—Ò&F6‚†W2’F–ÖVB÷WB"Ð Ð¢2W&–öF–2†VÇF‚Æör(	BöæR”ädòW"Ö–çWFR6ò÷26â6öæf—&Ò—Bw2Æ—fPÐ¢–b7–6ÆRRcÓÒ Ð¢VÆ6VEö×2Ò–çB‚‡F–ÖRçF–ÖR‚’Òö7–6ÆU÷7F'B’¢Ð¢ÆövvW"æ–æfò€Ð¢b%´ÔEöÆÆW%Ò)É27–6ÆR¶7–6ÆWÒÂ¶åöö·Ò÷¶ÆVâ†7–6ÆUö&F6†W2—Ò&F6†W2ô² ¢b'Â¶7–6ÆUöçÒF–6¶W'2Â¶VÆ6VEö×7Ö×2 ¢Ð Ð¢W†6WBW†6WF–öâ2öS Ð¢ÆövvW"çv&æ–ær†b%´ÔEöÆÆW%ÒöÆÂW'&÷"†7–6ÆR¶7–6ÆWÒ“¢µöWÒ"Ð¢öÖGöÆÆW%öW'&÷"Ò7G"…öRÐ¢6öç6V7WF—fUöÖ—72³ÒÐ Ð¢öÖGöÆÆW%ö7–6ÆRÒ7–6ÆPÐ¢7–6ÆR³ÒÐ Ð¢24Dâ÷&FRÖÆ–Ö—B&6²Ööfc¢¶Ö’&Æö6·2Æ7B3Óc2âöæRfÆB30Ð¢2v—B÷WFÆ7G2F†R&Æö6³²F†RöÆB72ó‡2óW2ÆFFW"&WG&–VBrF–ÖW0Ð¢2–ç6–FRF†R&Æö6²v–æF÷rÂW‡FVæF–ær—BæB6W6–ærCg2²FFv2àÐ¢&6µööfbÒ–çFW'fÂ–b6öç6V7WF—fUöÖ—72ÓÒVÇ6R3ã Ð Ð¢VÆ6VBÒF–ÖRçF–ÖR‚’Òö7–6ÆU÷7F'@Ð¢&VÖ–æ–ærÒ&6µööfbÒVÆ6V@Ð¢–b&VÖ–æ–ærâ Ð¢F–ÖRç6ÆVW‡&VÖ–æ–ærÐ Ð¢öÖGöÆÆW%÷F‡&VBÒF‡&VF–æråF‡&VB€Ð¢F&vWCÕ÷öÆÅöÆö÷ÂFVÖöãÕG'VRÂæÖSÒ%66‡v$ÔEöÆÆW" Ð¢Ð¢öÖGöÆÆW%÷F‡&VBç7F'B‚Ð¢ÆövvW"æ–æfò€Ð¢b%´ÔEöÆÆW%ÒF‡&VB7F'FVB(	B¶çÒF–6¶W'2Â Ð¢b'¶ÆVâ†&F6†W2—ÒæöâÖ&Æö6¶–ær&ÆÆVÂ&F6†W2â Ð¢Ð Ð Ð¦FVb7F'E÷7G&VÖW"‡F–6¶W'3¢Æ—7E·7G%Ò’ÓâæöæS ¢"" Ð¢ÆVæ6‚F†R66‡v"vV%6ö6¶WB7G&VÖW"–â&6¶w&÷VæBFVÖöâF‡&VBàÐ¢6fRFò6ÆÂ×VÇF—ÆRF–ÖW2(	BöæÇ’7F'G2öæ6RàÐ¢6â'VâÆöæw6–FRF†RÔEöÆÆW"‡F†W’W6R6W&FRF‡&VG2’àÐ¢"" Ð¢vÆö&Â÷7G&VÖW%÷F‡&VBÂöWfVçEöÆö÷Â÷w5öFW6—&VE÷F–6¶W'0 ¢v—F‚÷7G&VÖW%÷7F'EöÆö6³ ¢–b÷7G&VÖW%÷F‡&VBæB÷7G&VÖW%÷F‡&VBæ—5öÆ—fR‚“ ¢WFFU÷7G&VÖW%÷F–6¶W'2‡F–6¶W'2¢ÆövvW"æFV'Vr‚%µ7G&VÖW%ÒÇ&VG’'Vææ–ærâ"¢&WGW&à ¢G2ÒvWE÷Fö¶Vå÷7FGW2‚¢–bæ÷BG2ævWB‚&6öææV7FVB"“ ¢GFÂÒG2ævWB‚'&Vg&W6…÷Fö¶Vå÷GFÅ÷2"Â¢ÆövvW"çv&æ–ær€¢b%µ7G&VÖW%Ò66‡v"µBæ÷B6öææV7FVB‡&Vg&W6…÷Fö¶Vå÷GFÃ×·GFÇ×2’(	B ¢b'f—6—B÷66‡v"öWF‚öBFò&RÖWF†VçF–6FRâ ¢¢vÆö&Â÷w5öW'&÷ ¢÷w5öW'&÷"Ò%66‡v"µBæ÷BWF†VçF–6FVB(	Bf—6—B÷66‡v"öWF‚öB ¢&WGW&à ¢÷w5öFW6—&VE÷F–6¶W'2ÒÆ—7B†F–7Bæg&öÖ¶W—2‡7G"‡B’çWW"‚’f÷"B–âF–6¶W'2–bB’¢öVç7W&Uö&%÷W'6—7E÷v÷&¶W"‚ ¢FVb÷'Vâ‚“ ¢vÆö&ÂöWfVçEöÆö÷ ¢Æö÷Ò7–æ6–òææWuöWfVçEöÆö÷‚¢öWfVçEöÆö÷ÒÆö÷ ¢7–æ6–òç6WEöWfVçEöÆö÷†Æö÷¢Æö÷ç'Vå÷VçF–Åö6ö×ÆWFR…÷7G&VÖW%öÖ–â‚’ ¢÷7G&VÖW%÷F‡&VBÒF‡&VF–æråF‡&VB‡F&vWCÕ÷'VâÂFVÖöãÕG'VRÂæÖSÒ%66‡v%7G&VÖW""¢÷7G&VÖW%÷F‡&VBç7F'B‚¢ÆövvW"æ–æfò†b%µ7G&VÖW%Òu27G&VÖW"7F'FVBf÷"¶ÆVâ‡F–6¶W'2—ÒF–6¶W'2â" Ð Ð¦FVb7F÷÷7G&VÖW"‚’ÓâæöæS Ð¢vÆö&ÂöWfVçEöÆö÷ Ð¢–böWfVçEöÆö÷æBæ÷BöWfVçEöÆö÷æ—5ö6Æ÷6VB‚“ Ð¢öWfVçEöÆö÷æ6ÆÅ÷6ööå÷F‡&VG6fR…öWfVçEöÆö÷ç7F÷Ð Ð Ð¦FVbvWEöÆ—fU÷V÷FR‡F–6¶W#¢7G"’ÓâF–7C Ð¢""$ÆFW7BÆWfVÂV÷FRf÷"F–6¶W"âV×G’F–7B–bæ÷B7G&VÖ–ærâ"" Ð¢v—F‚öÆö6³ Ð¢&WGW&âF–7B…öÆ—fU÷V÷FW2ævWB‡F–6¶W"Â·Ò’Ð Ð Ð¦FVbvWEöÆ—fU÷V÷FW5÷6æ6†÷B†Ö…övU÷3¢fÆöBÂæöæRÒæöæR’ÓâF–7E·7G"ÂF–7EÓ Ð¢"" Ð¢&WGW&âF‡&VB×6fR6æ6†÷Böb7W'&VçBÆWfVÂV÷FW2àÐ Ð¢Ö…övU÷2f–ÇFW'2÷WB7FÆRV÷FW2'’WFFVEöBâ76–æræöæR&WGW&ç2WfW'Ð¢V÷FR7W'&VçFÇ’†VÆB–âÖVÖ÷'’Âv†–6‚—2W6VgVÂf÷"6Æ÷6VB×6W76–öâF6†&ö&@Ð¢ö'6W'fF–öâ&÷w2v†W&RÆ7B¶æ÷vâ&–6R—27F–ÆÂ–æf÷&ÖF—fRàÐ¢"" Ð¢æ÷rÒF–ÖRçF–ÖR‚Ð¢v—F‚öÆö6³ Ð¢&WGW&â°Ð¢7–Ó¢F–7B‡V÷FRÐ¢f÷"7–ÒÂV÷FR–âöÆ—fU÷V÷FW2æ—FV×2‚Ð¢–bÖ…övU÷2—2æöæR÷"æ÷rÒfÆöB‡V÷FRævWB‚'WFFVEöB"’÷"ã’ÃÒÖ…övU÷0Ð¢ÐÐ Ð Ð¦FVbvWEö&–Eö6µö–Ö&Ææ6R‡F–6¶W#¢7G"’ÓâfÆöC Ð¢"" Ð¢&–Bö6²6—¦R–Ö&Ææ6S¢†&–E÷6—¦R(‰"6µ÷6—¦R’ò†&–E÷6—¦R²6µ÷6—¦R’àÐ¢&ævS¢(‰#ã†ÆÂ6VÆÆW'2’(i"³ã†ÆÂ'W–W'2’àÐ¢&WGW&ç2ã–bæòÆ—fRFFàÐ¢"" Ð¢v—F‚öÆö6³ Ð¢&WGW&âfÆöB…öÆ—fU÷V÷FW2ævWB‡F–6¶W"Â·Ò’ævWB‚&&–Eö6µö–Ö&Ææ6R"Âã’Ð Ð Ð¦FVbvWEöÆ—fUö6æFÆW2‡F–6¶W#¢7G"Âã¢–çBÒ3’ÓâÆ—7E¶F–7EÓ Ð¢""$Æ7Bâ6ö×ÆWFVBÖÖ–â6æFÆW2f÷"F–6¶W"g&öÒF†R7G&VÖW"â"" Ð¢v—F‚öÆö6³ Ð¢GÒöÆ—fUö6æFÆW2ævWB‡F–6¶W"Ð¢–bæ÷BG Ð¢&WGW&âµÐÐ¢—FV×2ÒÆ—7B†GÐ¢&WGW&â—FV×5²Öã¥Ò–bÆVâ†—FV×2’ââVÇ6R—FV×0Ð Ð Ð¦FVbvWE÷67&VVæW%÷&–÷&—G’‡F–6¶W'3¢6WE·7G%ÒÂæöæRÒæöæR’ÓâÆ—7E·7G%Ó Ð¢"" Ð¢&WGW&â7–Ö&öÇ2g&öÒ45$TTäU"6÷'FVB'’7F—f—G’†v–æW'2²Æ÷6W'2²föÇVÖR’àÐ¢–bF–6¶W'6—2&÷f–FVBÂöæÇ’&WGW&â7–Ö&öÇ2F†B&R–âF†B6WBàÐ¢6ÆÂF†—2Fò&RÖ÷&FW"F†R66âVWVR6ò†÷BF–6¶W'2&R66ææVBf—'7BàÐ¢"" Ð¢v—F‚öÆö6³ Ð¢6öÖ&–æVBÒ÷67&VVæW%÷W³¥Ò²÷67&VVæW%öF÷vå³¥Ò²÷67&VVæW%÷föÅ³¥ÐÐ¢6VVâÂ÷&FW&VBÒ6WB‚’ÂµÐÐ¢f÷"—FVÒ–â6öÖ&–æVC Ð¢7–ÒÒ—FVÒævWB‚'7–Ö&öÂ"Â""Ð¢–b7–ÒæB7–Òæ÷B–â6VVã Ð¢–bF–6¶W'2—2æöæR÷"7–Ò–âF–6¶W'3 Ð¢6VVâæFB‡7–ÒÐ¢÷&FW&VBæVæB‡7–ÒÐ¢&WGW&â÷&FW&V@Ð Ð Ð¦FVbvWEöçögWGW&W5ö&–2‚’ÓâfÆöC Ð¢"" Ð¢RÔÖ–æ’ä4DgWGW&W2&–3¢7Eö6†ævRæ÷&ÖÆ—6VBFò(‰#(
b³àÐ¢+ãRRÖ2Fò&÷Vv†Ç’+ã²&W–öæB+ãRR—26Æ—VBàÐ¢&WGW&ç2ã–bgWGW&W2FFæ÷Bf–Æ&ÆR–WBàÐ¢"" Ð¢7–ÒÒög&öçEöÖöçF‚‚$å"Ð¢v—F‚öÆö6³ Ð¢gÒögWGW&W2ævWB‡7–ÒÂ·ÒÐ¢7BÒfÆöB†gævWB‚'7Eö6†ævR"Â’÷"Ð¢&WGW&âÖ‚‚ÓãÂÖ–âƒãÂ7BòãR’’2ãRR(i"&–2öbã Ð Ð Ð¦FVbvWEöW5ögWGW&W5ö&–2‚’ÓâfÆöC Ð¢""%2eSRÔÖ–æ’gWGW&W2&–2‡6ÖR66ÆR2å’â"" Ð¢7–ÒÒög&öçEöÖöçF‚‚$U2"Ð¢v—F‚öÆö6³ Ð¢gÒögWGW&W2ævWB‡7–ÒÂ·ÒÐ¢7BÒfÆöB†gævWB‚'7Eö6†ævR"Â’÷"Ð¢&WGW&âÖ‚‚ÓãÂÖ–âƒãÂ7BòãR’Ð Ð Ð¦FVb—5÷F–6¶W%ö†ÇFVB‡F–6¶W#¢7G"’Óâ&ööÃ Ð¢""%G'VR–bF†RF–6¶W"w26V7W&—G’7FGW2—27W'&VçFÇ’†ÇFVBâ"" Ð¢v—F‚öÆö6³ Ð¢&WGW&âF–6¶W"–âö†ÇFV@Ð Ð Ð¦FVb&Vv—7FW%ö&%ö6Æ÷6Uö6ÆÆ&6²†fâ’ÓâæöæS Ð¢""%&Vv—7FW"fâ‡F–6¶W#¢7G"Â6æFÆS¢F–7B’(	B6ÆÆVBv†VâV6‚ÖÖ–â&"6Æ÷6W2â"" Ð¢ö&%ö6Æ÷6Uö6ÆÆ&6·2æVæB†fâÐ Ð Ð¦FVbvWEö&%ö6Æ÷6U÷VWVR‚’Óâ%÷åVWVR# ¢""%VWVRöb‡F–6¶W"Â6æFÆR’GWÆW2V&Æ—6†VBöâWfW'’öæRÖÖ–çWFR6Æ÷6Râ"" ¢&WGW&âö&%ö6Æ÷6U÷VWVPÐ Ð Ð¦FVbvWE÷7G&VÖ–æuö&%ö6÷VçB‡F–6¶W#¢7G"’Óâ–çC Ð¢""$çVÖ&W"öbÖÖ–â&'27W'&VçFÇ’'VffW&VBf÷"F–6¶W"ƒ–bæ÷B7G&VÖ–ær’â"" Ð¢v—F‚öÆö6³ Ð¢GÒöÆ—fUö6æFÆW2ævWB‡F–6¶W"Ð¢&WGW&âÆVâ†G’–bGVÇ6R Ð Ð Ð¦FVbvWEö†ÇFVE÷F–6¶W'2‚’Óâ6WE·7G%Ó Ð¢v—F‚öÆö6³ Ð¢&WGW&â6WB…ö†ÇFVBÐ Ð Ð¦FVbvWEöÆ—fUóÕöFb‡F–6¶W#¢7G"’Óâ$÷F–öæÅ¶ö&¦V7EÒ# Ð¢"" Ð¢6öçfW'BF†RÆ—fR4„%EôUT•E’6æFÆRFWVR–çFòæF2FFg&ÖRF†@Ð¢ÖF6†W2F†RGvVÇfRFFf÷&ÖBW6VB'’F†R66ææW"æBfVGW&RVæv–æRàÐ Ð¢6öÇVÖç3¢÷VâÂ†–v‚ÂÆ÷rÂ6Æ÷6RÂföÇVÖR†fÆöCcBÐ¢–æFWƒ¢FFWF–ÖT–æFW‚–âÖW&–6ôæWuõ–÷&²G¢ÂöÆFW7Bf—'7BàÐ Ð¢&WGW&ç2æöæR–bfWvW"F†âR6æFÆW2&Rf–Æ&ÆR†æ÷BVæ÷Vv‚f÷"–æF–6F÷'2’àÐ¢"" Ð¢–×÷'BæF22@Ð¢6æFÆW2ÒvWEöÆ—fUö6æFÆW2‡F–6¶W"ÂÔ…ô4äDÄUô„•5Dõ%’Ð¢–bÆVâ†6æFÆW2’ÂS Ð¢&WGW&âæöæPÐ¢FbÒBäFFg&ÖR‡°Ð¢$÷Vâ#¢¶fÆöB†2ævWB‚&÷Vâ"Â’’f÷"2–â6æFÆW5ÒÀÐ¢$†–v‚#¢¶fÆöB†2ævWB‚&†–v‚"Â’’f÷"2–â6æFÆW5ÒÀÐ¢$Æ÷r#¢¶fÆöB†2ævWB‚&Æ÷r"Â’’f÷"2–â6æFÆW5ÒÀÐ¢$6Æ÷6R#¢¶fÆöB†2ævWB‚&6Æ÷6R"Â’’f÷"2–â6æFÆW5ÒÀÐ¢%föÇVÖR#¢¶fÆöB†2ævWB‚'föÇVÖR"Â’’f÷"2–â6æFÆW5ÒÀÐ¢ÒÐ¢2'V–ÆBFFWF–ÖT–æFW‚g&öÒ66‡v"Wö6‚Ö×2F–ÖW7F×2–bf–Æ&ÆPÐ¢–b6æFÆW5³ÒævWB‚'F–ÖUö×2"“ Ð¢G2ÒBçFõöFFWF–ÖR…¶2ævWB‚'F–ÖUö×2"Â’f÷"2–â6æFÆW5ÒÀÐ¢Væ—CÒ&×2"ÂWF3ÕG'VRÐ¢Fbæ–æFW‚ÒG2çG¥ö6öçfW'B‚$ÖW&–6ôæWuõ–÷&²"Ð¢2G&÷&'2v—F‚¦W&ò6Æ÷6R†–æ6ö×ÆWFRò&BFFÐ¢FbÒFe¶Fe²$6Æ÷6R%ÒâÒæ6÷’‚Ð¢&WGW&âFb–bÆVâ†Fb’ãÒRVÇ6RæöæPÐ Ð Ð¦FVb—5÷7G&VÖW%÷&VG’‚’Óâ&ööÃ Ð¢""%G'VRv†VâF†Ru27G&VÖW"—26öææV7FVBæB†2Æ—fRV÷FRFFâ"" Ð¢&WGW&â&ööÂ€Ð¢÷7G&VÖW%÷F‡&VBæB÷7G&VÖW%÷F‡&VBæ—5öÆ—fR‚Ð¢æB÷w5ö6öææV7FVBæBöÆ—fU÷V÷FW0Ð¢Ð Ð Ð¦FVb—5÷w5öFFöÆ—fR†Ö…övU÷3¢fÆöBÒ"ã’Óâ&ööÃ Ð¢""%G'VRv†VâF†Ru27G&VÖW"†2&V6V—fVBÄUdTÄôäUôUT•D”U2FFv—F†–àÐ¢Ö…övU÷26V6öæG2âÖ÷&R&VÆ–&ÆRF†â—5÷7G&VÖW%÷&VG’‚’&V6W6R—@Ð¢6öæf—&×2FF—27GVÆÇ’fÆ÷v–ærÂæ÷B§W7BF†BF†R6ö6¶WB—2÷Vââ"" Ð¢&WGW&â&ööÂ€Ð¢÷7G&VÖW%÷F‡&VBæB÷7G&VÖW%÷F‡&VBæ—5öÆ—fR‚Ð¢æBöÆ7E÷w5öFFöBâ Ð¢æBF–ÖRçF–ÖR‚’ÒöÆ7E÷w5öFFöBÂÖ…övU÷0Ð¢Ð Ð Ð¦FVbw5ög&W6…ö6÷fW&vR†Ö…övU÷3¢fÆöBÒ"ã’ÓâfÆöC ¢""%&WGW&âg&7F–öâöbFW6—&VBF–6¶W'2v—F‚&V6VçB66‡v"u2WfVçBâ"" ¢v—F‚öÆö6³ ¢F–6¶W'2ÒÆ—7B…÷w5öFW6—&VE÷F–6¶W'2¢6VVåöBÒF–7B…÷w5÷6VVåöB¢–bæ÷BF–6¶W'3 ¢&WGW&âã ¢æ÷rÒF–ÖRçF–ÖR‚¢g&W6‚Ò7VÒ†æ÷rÒfÆöB‡6VVåöBævWB‡F–6¶W"’÷"ã’ÃÒÖ…övU÷2f÷"F–6¶W"–âF–6¶W'2¢&WGW&âg&W6‚òÆVâ‡F–6¶W'2 Ð Ð¦FVbvWE÷7G&VÖW%÷7FGW2‚’ÓâF–7C Ð¢"" Ð¢&WGW&â†VÇF‚6æ6†÷Bf÷"&÷F‚F†Ru27G&VÖW"æBF†R$U5BÔEöÆÆW"àÐ¢6ÆÆW'26â&VBw5÷7G&VÖW"â¢æBÖE÷öÆÆW"â¢–æFWVæFVçFÇ’àÐ¢"" Ð¢7–ÕöçÒög&öçEöÖöçF‚‚$å"Ð¢7–ÕöW2Òög&öçEöÖöçF‚‚$U2"Ð¢v—F‚öÆö6³ ¢çÒF–7B…ögWGW&W2ævWB‡7–ÕöçÂ·Ò’¢W2ÒF–7B…ögWGW&W2ævWB‡7–ÕöW2Â·Ò’¢FW6—&VBÒ6WB…÷w5öFW6—&VE÷F–6¶W'2¢FW6—&VEö6÷VçBÒÆVâ†FW6—&VB¢7F—fUö6÷VçBÒÆVâ…÷w5ö7F—fU÷F–6¶W'2bFW6—&VB¢6¶æ÷vÆVFvVEö6÷VçBÒÆVâ…÷w5ö6¶æ÷vÆVFvVE÷F–6¶W'2bFW6—&VB¢VæF–æuö6÷VçBÒÆVâ…÷w5÷7V'67&—F–öå÷&WVW7G2¢6VVåö6÷VçBÒ7VÒ‡F–6¶W"–â÷w5÷6VVåöBf÷"F–6¶W"–â÷w5öFW6—&VE÷F–6¶W'2¢æ÷rÒF–ÖRçF–ÖR‚¢g&W6…ö6÷VçBÒ7VÒ€¢æ÷rÒfÆöB…÷w5÷6VVåöBævWB‡F–6¶W"’÷"ã’ÃÒ"ã ¢f÷"F–6¶W"–â÷w5öFW6—&VE÷F–6¶W'0¢¢7F—fUóc5ö6÷VçBÒ7VÒ€¢æ÷rÒfÆöB…÷w5÷6VVåöBævWB‡F–6¶W"’÷"ã’ÃÒcã ¢f÷"F–6¶W"–â÷w5öFW6—&VE÷F–6¶W'0¢¢&W7Eö6÷VçBÒ7VÒ€¢7G"‚…öÆ—fU÷V÷FW2ævWB‡F–6¶W"’÷"·Ò’ævWB‚'6÷W&6U÷7FGW2"’÷"""’çWW"‚¢ÓÒ%$U5EôdÄÄ$4² ¢f÷"F–6¶W"–âöÖGöÆÆW%÷F–6¶W'0¢¢6æFÆUö6÷VçBÒÆVâ…öÆ—fUö6æFÆW2¢†ÇFVEö6÷VçBÒÆVâ…ö†ÇFVB¢Æ7E÷w5övU÷2Ò€¢&÷VæB†æ÷rÒöÆ7E÷w5öFFöBÂ2’–böÆ7E÷w5öFFöBâVÇ6RæöæP¢ Ð¢ÖGöÆÆW%övòÒ&÷VæB‡F–ÖRçF–ÖR‚’ÒöÖGöÆÆW%öÆ7Eöö²Â’–böÖGöÆÆW%öÆ7Eöö²VÇ6RæöæPÐ Ð¢&WGW&â°Ð¢2ÆVv7’fÆB¶W—2(	B¶WBf÷"&6·v&B6ö×@Ð¢&6öææV7FVB#¢÷w5ö6öææV7FVBÀÐ¢&W'&÷"#¢÷w5öW'&÷"ÀÐ¢&Æ—fU÷V÷FW2#¢g&W6…ö6÷VçBÀ¢&Æ—fUö6æFÆW2#¢6æFÆUö6÷VçBÀÐ¢&†ÇFVE÷F–6¶W'2#¢†ÇFVEö6÷VçBÀÐ¢&gWGW&W2#¢·7–Õöç¢çÂ7–ÕöW3¢W7ÒÀÐ¢&çö&–2#¢vWEöçögWGW&W5ö&–2‚’ÀÐ¢&W5ö&–2#¢vWEöW5ögWGW&W5ö&–2‚’ÀÐ¢2æWr7G'V7GW&VB¶W—2(	BW6VB'’ö’÷6W'f–6W0Ð¢'w5÷7G&VÖW"#¢°Ð¢''Vææ–ær#¢&ööÂ…÷7G&VÖW%÷F‡&VBæB÷7G&VÖW%÷F‡&VBæ—5öÆ—fR‚’’À¢&6öææV7FVB#¢÷w5ö6öææV7FVBæB&ööÂ…÷7G&VÖW%÷F‡&VBæB÷7G&VÖW%÷F‡&VBæ—5öÆ—fR‚’’À¢&FW6—&VE÷7V'67&—F–öç2#¢FW6—&VEö6÷VçBÀ¢'6VçE÷7V'67&—F–öç2#¢7F—fUö6÷VçBÀ¢&6¶æ÷vÆVFvVE÷7V'67&—F–öç2#¢6¶æ÷vÆVFvVEö6÷VçBÀ¢'VæF–æu÷7V'67&—F–öå÷&WVW7G2#¢VæF–æuö6÷VçBÀ¢'7V'67&—F–öåö6÷fW&vU÷7B#¢&÷VæB€¢6¶æ÷vÆVFvVEö6÷VçBòFW6—&VEö6÷VçB¢ãÂ¢’–bFW6—&VEö6÷VçBVÇ6RãÀ¢'6VVå÷V÷FW2#¢6VVåö6÷VçBÀ¢&7F—fU÷V÷FW5óc2#¢7F—fUóc5ö6÷VçBÀ¢&Æ—fU÷V÷FW2#¢g&W6…ö6÷VçB–b…÷7G&VÖW%÷F‡&VBæB÷7G&VÖW%÷F‡&VBæ—5öÆ—fR‚’’VÇ6RÀ¢&g&W6…ö6÷fW&vU÷7B#¢&÷VæB‡w5ög&W6…ö6÷fW&vR†Ö…övU÷3Ó"ã’¢Â’À¢&Æ7EöFFövU÷2#¢Æ7E÷w5övU÷2À¢&Æ—fUö6æFÆW2#¢6æFÆUö6÷VçBÀ¢&†ÇFVE÷F–6¶W'2#¢†ÇFVEö6÷VçBÀÐ¢&çö&–2#¢vWEöçögWGW&W5ö&–2‚’ÀÐ¢&W'&÷"#¢÷w5öW'&÷"–b…÷7G&VÖW%÷F‡&VBæB÷7G&VÖW%÷F‡&VBæ—5öÆ—fR‚’’VÇ6RæöæRÀÐ¢ÒÀÐ¢&ÖE÷öÆÆW"#¢°Ð¢''Vææ–ær#¢öÖGöÆÆW%÷'Vææ–æræB&ööÂ…öÖGöÆÆW%÷F‡&VBæBöÖGöÆÆW%÷F‡&VBæ—5öÆ—fR‚’’ÀÐ¢&7–6ÆR#¢öÖGöÆÆW%ö7–6ÆRÀÐ¢&Æ7Eööµövõ÷2#¢ÖGöÆÆW%övòÀÐ¢'Væ—fW'6U÷6—¦R#¢ÆVâ…öÖGöÆÆW%÷F–6¶W'2’À¢&fÆÆ&6µ÷V÷FW2#¢&W7Eö6÷VçBÀ¢&Æ—fU÷V÷FW2#¢&W7Eö6÷VçBÀ¢&W'&÷"#¢öÖGöÆÆW%öW'&÷"ÀÐ¢ÒÀÐ¢ÐÐ