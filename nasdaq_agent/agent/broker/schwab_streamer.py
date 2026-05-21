"""
Schwab WebSocket Streamer — real-time market data pipeline.

Why this matters (thinking like a 40-year trader):
  - REST polling gives you data that is 5–30s stale. At scalping timeframes
    (30s–2min holds) that means you are always acting on yesterday's news.
  - Streaming gives you sub-second updates. Bid/ask imbalance, halt detection,
    and candle closes arrive the instant they happen on the exchange.
  - NQ/ES futures lead NASDAQ equities by 30–90 seconds. Watching futures
    drift down BEFORE the stocks follow is the edge every professional has.
  - Bid/ask size imbalance (bid_size – ask_size) / (bid_size + ask_size) is the
    single most predictive real-time directional signal for scalps.

Services subscribed:
  LEVELONE_EQUITIES   — real-time bid/ask/volume for all 154 tickers
  CHART_EQUITY        — 1-min candles as each minute closes
  SCREENER_EQUITY     — top movers on NASDAQ (scanner priority)
  LEVELONE_FUTURES    — /NQ and /ES for macro direction bias

Consumed by:
  scanner.py   → real-time quotes + halt detection + screener priority
  ml_model.py  → bid_ask_imbalance feature, nq_futures_bias feature
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from datetime import date
from typing import Optional

import requests

from agent.broker.schwab_auth import get_access_token, get_token_status

logger = logging.getLogger(__name__)

TRADER_BASE = "https://api.schwabapi.com/trader/v1"

# ── Live data stores (thread-safe via _lock) ──────────────────────────────────
_lock              = threading.Lock()
_live_quotes:  dict[str, dict]        = {}   # ticker → quote dict
_live_candles: dict[str, deque]       = {}   # ticker → deque of last 300 1-min OHLCV
_screener_up:  list[dict]             = []   # NASDAQ top % gainers (last update)
_screener_down: list[dict]            = []   # NASDAQ top % losers
_screener_vol:  list[dict]            = []   # NASDAQ top volume
_futures:      dict[str, dict]        = {}   # /NQ, /ES → quote dict
_halted:       set[str]               = set()

# ── Bar-close event bus ───────────────────────────────────────────────────────
import queue as _q
_bar_close_queue: _q.Queue = _q.Queue(maxsize=20000)
_bar_close_callbacks: list = []

# Streamer lifecycle
_streamer_thread:  Optional[threading.Thread] = None
_event_loop:       Optional[asyncio.AbstractEventLoop] = None
_ws_connected:     bool = False
_ws_error:         Optional[str] = None
_subscribed_tickers: list[str] = []

MAX_CANDLE_HISTORY = 300   # 5 hours of 1-min bars

# ── Real-time tick callback registry ─────────────────────────────────────────
# Registered functions are called on every LEVELONE_EQUITIES update.
# Throttled per-ticker to _TICK_MIN_INTERVAL seconds to avoid flooding WebSocket.
_tick_callbacks:      list          = []
_bulk_price_callbacks: list         = []   # fn(prices: dict[str, dict]) — one call per poll cycle
_last_tick_ts:        dict[str, float] = {}
_TICK_MIN_INTERVAL:   float         = 0.25   # max 4 price updates/s per ticker


def register_tick_callback(fn) -> None:
    """Register fn(ticker: str, quote: dict) — called on every throttled tick."""
    _tick_callbacks.append(fn)


def register_bulk_price_callback(fn) -> None:
    """
    Register fn(prices: dict[str, dict]) — called ONCE per poll cycle with ALL
    updated quotes.  Much more efficient than 477 individual tick callbacks.
    Each value dict contains: last, bid, ask, volume, high, low, pct_change.
    """
    _bulk_price_callbacks.append(fn)


# ── User Preferences (provides streamer URL + client IDs) ─────────────────────

def _get_streamer_info() -> dict:
    """
    GET /trader/v1/userpreference — returns streamer credentials.
    Response includes streamerSocketUrl, schwabClientCustomerId, etc.
    """
    token = get_access_token()
    if not token:
        raise RuntimeError("No access token — run Schwab OAuth first")
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


# ── Front-month futures contract ──────────────────────────────────────────────

def _front_month(root: str) -> str:
    """
    Return the active front-month futures symbol, e.g. '/NQM26'.
    Quarterly expirations: March(H), June(M), September(U), December(Z).
    We roll ~2 weeks before expiry (3rd Friday ≈ day 18-21 of month).
    """
    d = date.today()
    quarters = [(3, "H"), (6, "M"), (9, "U"), (12, "Z")]
    yy = str(d.year)[2:]
    for month, code in quarters:
        # Roll to next quarter after ~18th of expiry month
        if d.month < month or (d.month == month and d.day < 18):
            return f"/{root}{code}{yy}"
    # Past December of this year — go to March of next year
    return f"/{root}H{str(d.year + 1)[2:]}"


FUTURES_SYMBOLS = [_front_month("NQ"), _front_month("ES")]


# ── Request builders ──────────────────────────────────────────────────────────

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


# ── Message processing ────────────────────────────────────────────────────────

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
    "1": "open", "2": "high", "3": "low", "4": "close",
    "5": "volume", "7": "time_ms",
}


def _process_levelone_equities(content: list) -> None:
    updated: list[tuple[str, dict]] = []
    with _lock:
        for item in content:
            sym = item.get("key", "")
            if not sym:
                continue
            quote = _live_quotes.setdefault(sym, {})
            for raw, name in _EQUITY_FIELDS.items():
                if raw in item:
                    quote[name] = item[raw]

            # Derived: bid/ask imbalance — the #1 real-time directional signal
            bid_sz = float(quote.get("bid_size", 0) or 0)
            ask_sz = float(quote.get("ask_size", 0) or 0)
            total  = bid_sz + ask_sz
            quote["bid_ask_imbalance"] = (bid_sz - ask_sz) / total if total > 0 else 0.0
            quote["updated_at"] = time.time()

            # Halt detection
            status = str(quote.get("status", "")).lower()
            if "halt" in status:
                _halted.add(sym)
                logger.warning(f"[Streamer] {sym} HALTED")
            else:
                _halted.discard(sym)

            updated.append((sym, dict(quote)))   # snapshot for callbacks (outside lock)

    # Fire tick callbacks outside the lock — 250ms throttle per ticker
    if updated and _tick_callbacks:
        now = time.time()
        for sym, quote in updated:
            if now - _last_tick_ts.get(sym, 0.0) >= _TICK_MIN_INTERVAL:
                _last_tick_ts[sym] = now
                for fn in _tick_callbacks:
                    try:
                        fn(sym, quote)
                    except Exception:
                        pass


def _process_levelone_futures(content: list) -> None:
    with _lock:
        for item in content:
            sym = item.get("key", "")
            if not sym:
                continue
            fq = _futures.setdefault(sym, {})
            for raw, name in _FUTURES_FIELDS.items():
                if raw in item:
                    fq[name] = item[raw]
            fq["updated_at"] = time.time()


def _process_chart_equity(content: list) -> None:
    new_bars: list[tuple[str, dict]] = []
    with _lock:
        for item in content:
            sym = item.get("key", "")
            if not sym:
                continue
            candle = {name: item[raw] for raw, name in _CHART_FIELDS.items() if raw in item}
            if not candle:
                continue
            if sym not in _live_candles:
                _live_candles[sym] = deque(maxlen=MAX_CANDLE_HISTORY)
            _live_candles[sym].append(candle)
            new_bars.append((sym, candle))

    # Fire bar-close events outside the lock so callbacks never deadlock
    for sym, candle in new_bars:
        try:
            _bar_close_queue.put_nowait((sym, candle))
        except _q.Full:
            pass
        for fn in _bar_close_callbacks:
            try:
                fn(sym, candle)
            except Exception:
                pass


def _process_screener(service: str, content: list) -> None:
    global _screener_up, _screener_down, _screener_vol
    parsed = []
    for item in content:
        items_list = item.get("4", []) or item.get("items", [])
        sort_field = item.get("2", "")
        for entry in items_list:
            parsed.append({
                "symbol":      entry.get("symbol", ""),
                "last_price":  entry.get("lastPrice", 0),
                "pct_change":  entry.get("netPercentChange", 0),
                "volume":      entry.get("totalVolume", 0),
                "trades":      entry.get("trades", 0),
            })
        with _lock:
            sf = str(sort_field).upper()
            if "UP" in sf:
                _screener_up = parsed[:]
            elif "DOWN" in sf:
                _screener_down = parsed[:]
            else:
                _screener_vol = parsed[:]


def _process_message(raw: str) -> None:
    try:
        msg = json.loads(raw)
    except Exception:
        return

    for data_block in msg.get("data", []):
        svc     = data_block.get("service", "")
        content = data_block.get("content", [])
        if svc == "LEVELONE_EQUITIES":
            _process_levelone_equities(content)
        elif svc == "LEVELONE_FUTURES":
            _process_levelone_futures(content)
        elif svc == "CHART_EQUITY":
            _process_chart_equity(content)
        elif svc in ("SCREENER_EQUITY",):
            _process_screener(svc, content)

    for resp in msg.get("response", []):
        code = resp.get("content", {}).get("code", -1)
        cmd  = resp.get("command", "")
        svc  = resp.get("service", "")
        if code == 0:
            logger.info(f"[Streamer] {svc}/{cmd} succeeded")
        elif code != -1:
            logger.warning(f"[Streamer] {svc}/{cmd} code={code}: "
                           f"{resp.get('content', {}).get('msg', '')}")


# ── WebSocket main loop ───────────────────────────────────────────────────────

async def _streamer_main(tickers: list[str]) -> None:
    global _ws_connected, _ws_error

    try:
        info = _get_streamer_info()
    except Exception as e:
        _ws_error = str(e)
        logger.error(f"[Streamer] Cannot get user preferences: {e}")
        return

    ws_url      = info.get("streamerSocketUrl", "")
    customer_id = info.get("schwabClientCustomerId", "")
    correl_id   = info.get("schwabClientCorrelId", "")
    channel     = info.get("schwabClientChannel", "IO")
    func_id     = info.get("schwabClientFunctionId", "APIAPP")

    if not ws_url:
        _ws_error = "No streamerSocketUrl in user preferences"
        return

    import websockets

    retry_delay = 5
    while True:
        try:
            logger.info(f"[Streamer] Connecting to {ws_url}…")
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=30) as ws:
                _ws_connected = True
                _ws_error = None
                retry_delay = 5   # reset on successful connect
                logger.info("[Streamer] Connected.")

                # ── 1. LOGIN ──────────────────────────────────────────────────
                token = get_access_token()
                login_msg = {"requests": [_req(
                    "ADMIN", "LOGIN", 1, {
                        "Authorization":         token,
                        "SchwabClientChannel":   channel,
                        "SchwabClientFunctionId": func_id,
                    }, customer_id, correl_id,
                )]}
                await ws.send(json.dumps(login_msg))
                resp = json.loads(await ws.recv())
                login_code = resp.get("response", [{}])[0].get("content", {}).get("code", -1)
                if login_code != 0:
                    logger.error(f"[Streamer] LOGIN failed: {resp}")
                    _ws_connected = False
                    break

                logger.info("[Streamer] Logged in.")

                # ── 2. LEVELONE_EQUITIES for all tickers ──────────────────────
                # Fields: bid, ask, last, bid_size, ask_size, volume, high, low,
                #         prev_close, net_change, 52w_high, 52w_low, status,
                #         mark, net_pct_change, hard_to_borrow, shortable
                equity_fields = "0,1,2,3,4,5,8,10,11,12,18,19,20,32,33,42,48,49"
                # Subscribe in batches of 100 (streamer symbol limit per command)
                batch_size = 100
                for i in range(0, len(tickers), batch_size):
                    batch = tickers[i: i + batch_size]
                    cmd = "SUBS" if i == 0 else "ADD"
                    subs_msg = {"requests": [_req(
                        "LEVELONE_EQUITIES", cmd, 10 + i, {
                            "keys":   ",".join(batch),
                            "fields": equity_fields,
                        }, customer_id, correl_id,
                    )]}
                    await ws.send(json.dumps(subs_msg))

                # ── 3. CHART_EQUITY — real-time 1-min candles ─────────────────
                for i in range(0, len(tickers), batch_size):
                    batch = tickers[i: i + batch_size]
                    cmd = "SUBS" if i == 0 else "ADD"
                    chart_msg = {"requests": [_req(
                        "CHART_EQUITY", cmd, 200 + i, {
                            "keys":   ",".join(batch),
                            "fields": "0,1,2,3,4,5,7",
                        }, customer_id, correl_id,
                    )]}
                    await ws.send(json.dumps(chart_msg))

                # ── 4. SCREENER_EQUITY — NASDAQ top movers ────────────────────
                screener_keys = (
                    "NASDAQ_PERCENT_CHANGE_UP_1,"
                    "NASDAQ_PERCENT_CHANGE_DOWN_1,"
                    "NASDAQ_VOLUME_0"
                )
                screener_msg = {"requests": [_req(
                    "SCREENER_EQUITY", "SUBS", 300, {
                        "keys":   screener_keys,
                        "fields": "0,1,2,3,4",
                    }, customer_id, correl_id,
                )]}
                await ws.send(json.dumps(screener_msg))

                # ── 5. LEVELONE_FUTURES — NQ + ES macro direction ─────────────
                futures_msg = {"requests": [_req(
                    "LEVELONE_FUTURES", "SUBS", 400, {
                        "keys":   ",".join(FUTURES_SYMBOLS),
                        "fields": "0,1,2,3,8,19,20,23",
                    }, customer_id, correl_id,
                )]}
                await ws.send(json.dumps(futures_msg))

                logger.info(f"[Streamer] Subscribed to {len(tickers)} equities, "
                            f"{len(FUTURES_SYMBOLS)} futures, screener.")

                # ── 6. Message loop ───────────────────────────────────────────
                async for raw in ws:
                    _process_message(raw)

        except Exception as e:
            _ws_connected = False
            _ws_error = str(e)
            logger.warning(f"[Streamer] Disconnected: {e} — retrying in {retry_delay}s")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)


# ── Public API ────────────────────────────────────────────────────────────────

def start_md_poller(tickers: list[str], interval: float = 1.0) -> None:
    """
    REST-based real-time quote poller — Market Data app only.
    No Accounts+Trading credentials required.

    Polls /marketdata/v1/quotes every `interval` seconds for all tickers,
    populates _live_quotes, and fires _tick_callbacks exactly like the
    WebSocket streamer would.  All downstream consumers (RT monitor,
    data_fetcher, dashboard tick broadcast) work unchanged.

    Trade-off vs WebSocket: ~1s latency instead of sub-100ms, which is
    more than adequate for a scalping dashboard and RT position monitor.
    """
    global _streamer_thread, _subscribed_tickers

    if _streamer_thread and _streamer_thread.is_alive():
        logger.debug("[MDPoller] Already running.")
        return

    _subscribed_tickers = list(tickers)

    def _poll_loop() -> None:
        global _ws_connected, _ws_error
        from agent.broker.schwab_market_data import fetch_full_quotes, _is_authorised

        logger.info(f"[MDPoller] Started — {len(tickers)} tickers @ {interval}s interval")
        while True:
            try:
                if not _is_authorised():
                    _ws_connected = False
                    _ws_error = "Schwab Market Data not authorized — visit /schwab/auth/md"
                    time.sleep(10)
                    continue

                quotes = fetch_full_quotes(list(tickers))
                if quotes:
                    _ws_connected = True
                    _ws_error = None
                    bulk_prices: dict[str, dict] = {}
                    updated: list[tuple[str, dict]] = []
                    with _lock:
                        for sym, q in quotes.items():
                            quote = _live_quotes.setdefault(sym, {})
                            quote["last"]           = float(q.get("last") or 0)
                            quote["bid"]            = float(q.get("bid")  or 0)
                            quote["ask"]            = float(q.get("ask")  or 0)
                            quote["volume"]         = float(q.get("volume") or 0)
                            quote["high"]           = float(q.get("high")  or 0)
                            quote["low"]            = float(q.get("low")   or 0)
                            quote["prev_close"]     = float(q.get("close") or 0)
                            quote["net_pct_change"] = float(q.get("pct_change") or 0)
                            quote["updated_at"]     = time.time()
                            if quote["last"] <= 0:
                                _halted.add(sym)
                            else:
                                _halted.discard(sym)
                            updated.append((sym, dict(quote)))
                            bulk_prices[sym] = {
                                "last":       quote["last"],
                                "bid":        quote["bid"],
                                "ask":        quote["ask"],
                                "volume":     quote["volume"],
                                "high":       quote["high"],
                                "low":        quote["low"],
                                "pct_change": quote["net_pct_change"],
                            }

                    # Bulk callback — ONE call per poll cycle with ALL prices.
                    # This replaces 477 individual WS messages with a single batch.
                    if bulk_prices and _bulk_price_callbacks:
                        for fn in _bulk_price_callbacks:
                            try:
                                fn(bulk_prices)
                            except Exception:
                                pass

                    # Individual tick callbacks (legacy — for backward compat)
                    if updated and _tick_callbacks:
                        now = time.time()
                        for sym, quote in updated:
                            if now - _last_tick_ts.get(sym, 0.0) >= _TICK_MIN_INTERVAL:
                                _last_tick_ts[sym] = now
                                for fn in _tick_callbacks:
                                    try:
                                        fn(sym, quote)
                                    except Exception:
                                        pass

            except Exception as _e:
                logger.warning(f"[MDPoller] poll error: {_e}")

            time.sleep(interval)

    _streamer_thread = threading.Thread(
        target=_poll_loop, daemon=True, name="SchwabMDPoller"
    )
    _streamer_thread.start()
    logger.info(f"[MDPoller] Thread started for {len(tickers)} tickers.")


def start_streamer(tickers: list[str]) -> None:
    """
    Launch the Schwab WebSocket streamer in a background daemon thread.
    Safe to call multiple times — only starts once.
    """
    global _streamer_thread, _event_loop, _subscribed_tickers

    if _streamer_thread and _streamer_thread.is_alive():
        logger.debug("[Streamer] Already running.")
        return

    ts = get_token_status()
    if not ts.get("connected"):
        ttl = ts.get("refresh_token_ttl_s", 0)
        logger.warning(
            f"[Streamer] Schwab not connected (refresh_token_ttl={ttl}s) — "
            f"visit /schwab/auth to re-authenticate."
        )
        global _ws_error
        _ws_error = "Schwab not authenticated — visit /schwab/auth"
        return

    _subscribed_tickers = list(tickers)

    def _run():
        global _event_loop
        loop = asyncio.new_event_loop()
        _event_loop = loop
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_streamer_main(tickers))

    _streamer_thread = threading.Thread(target=_run, daemon=True, name="SchwabStreamer")
    _streamer_thread.start()
    logger.info(f"[Streamer] Started for {len(tickers)} tickers.")


def stop_streamer() -> None:
    global _event_loop
    if _event_loop and not _event_loop.is_closed():
        _event_loop.call_soon_threadsafe(_event_loop.stop)


def get_live_quote(ticker: str) -> dict:
    """Latest Level 1 quote for a ticker. Empty dict if not streaming."""
    with _lock:
        return dict(_live_quotes.get(ticker, {}))


def get_bid_ask_imbalance(ticker: str) -> float:
    """
    Bid/ask size imbalance: (bid_size − ask_size) / (bid_size + ask_size).
    Range: −1.0 (all sellers) → +1.0 (all buyers).
    Returns 0.0 if no live data.
    """
    with _lock:
        return float(_live_quotes.get(ticker, {}).get("bid_ask_imbalance", 0.0))


def get_live_candles(ticker: str, n: int = 300) -> list[dict]:
    """Last n completed 1-min candles for a ticker from the streamer."""
    with _lock:
        dq = _live_candles.get(ticker)
        if not dq:
            return []
        items = list(dq)
        return items[-n:] if len(items) > n else items


def get_screener_priority(tickers: set[str] | None = None) -> list[str]:
    """
    Return symbols from SCREENER sorted by activity (gainers + losers + volume).
    If `tickers` is provided, only return symbols that are in that set.
    Call this to re-order the scan queue so hot tickers are scanned first.
    """
    with _lock:
        combined = _screener_up[:] + _screener_down[:] + _screener_vol[:]
    seen, ordered = set(), []
    for item in combined:
        sym = item.get("symbol", "")
        if sym and sym not in seen:
            if tickers is None or sym in tickers:
                seen.add(sym)
                ordered.append(sym)
    return ordered


def get_nq_futures_bias() -> float:
    """
    E-Mini NASDAQ 100 futures bias: pct_change normalised to −1…+1.
    ±0.5 % maps to roughly ±1.0; beyond ±0.5 % is clipped.
    Returns 0.0 if futures data not available yet.
    """
    sym = _front_month("NQ")
    with _lock:
        fq = _futures.get(sym, {})
    pct = float(fq.get("pct_change", 0) or 0)
    return max(-1.0, min(1.0, pct / 0.5))   # 0.5 % → bias of 1.0


def get_es_futures_bias() -> float:
    """S&P 500 E-Mini futures bias (same scale as NQ)."""
    sym = _front_month("ES")
    with _lock:
        fq = _futures.get(sym, {})
    pct = float(fq.get("pct_change", 0) or 0)
    return max(-1.0, min(1.0, pct / 0.5))


def is_ticker_halted(ticker: str) -> bool:
    """True if the ticker's security status is currently Halted."""
    with _lock:
        return ticker in _halted


def register_bar_close_callback(fn) -> None:
    """Register fn(ticker: str, candle: dict) — called when each 1-min bar closes."""
    _bar_close_callbacks.append(fn)


def get_bar_close_queue() -> "_q.Queue":
    """Queue of (ticker, candle) tuples published on every CHART_EQUITY bar close."""
    return _bar_close_queue


def get_streaming_bar_count(ticker: str) -> int:
    """Number of 1-min bars currently buffered for ticker (0 if not streaming)."""
    with _lock:
        dq = _live_candles.get(ticker)
        return len(dq) if dq else 0


def get_halted_tickers() -> set[str]:
    with _lock:
        return set(_halted)


def get_live_1m_df(ticker: str) -> "Optional[object]":
    """
    Convert the live CHART_EQUITY candle deque into a pandas DataFrame that
    matches the Twelve Data format used by the scanner and feature engine.

    Columns: Open, High, Low, Close, Volume  (float64)
    Index:   DatetimeIndex in America/New_York tz, oldest first.

    Returns None if fewer than 5 candles are available (not enough for indicators).
    """
    import pandas as pd
    candles = get_live_candles(ticker, MAX_CANDLE_HISTORY)
    if len(candles) < 5:
        return None
    df = pd.DataFrame({
        "Open":   [float(c.get("open",   0)) for c in candles],
        "High":   [float(c.get("high",   0)) for c in candles],
        "Low":    [float(c.get("low",    0)) for c in candles],
        "Close":  [float(c.get("close",  0)) for c in candles],
        "Volume": [float(c.get("volume", 0)) for c in candles],
    })
    # Build DatetimeIndex from Schwab epoch-ms timestamps if available
    if candles[0].get("time_ms"):
        ts = pd.to_datetime([c.get("time_ms", 0) for c in candles],
                            unit="ms", utc=True)
        df.index = ts.tz_convert("America/New_York")
    # Drop bars with zero close (incomplete / bad data)
    df = df[df["Close"] > 0].copy()
    return df if len(df) >= 5 else None


def is_streamer_ready() -> bool:
    """True when the streamer is connected and has live quote data."""
    with _lock:
        return bool(_ws_connected and _live_quotes)


def get_streamer_status() -> dict:
    """Return a health snapshot: connection state, quote/candle counts, futures bias."""
    sym_nq = _front_month("NQ")
    sym_es = _front_month("ES")
    with _lock:
        nq = dict(_futures.get(sym_nq, {}))
        es = dict(_futures.get(sym_es, {}))
        live_count = len(_live_quotes)
        candle_count = len(_live_candles)
        halted_count = len(_halted)

    return {
        "connected":      _ws_connected,
        "error":          _ws_error,
        "live_quotes":    live_count,
        "live_candles":   candle_count,
        "halted_tickers": halted_count,
        "futures": {
            sym_nq: nq,
            sym_es: es,
        },
        "nq_bias": get_nq_futures_bias(),
        "es_bias": get_es_futures_bias(),
    }
