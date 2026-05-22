"""
Schwab REST API client — order placement, position sync, account info.

Paper vs live is controlled by SCHWAB_PAPER_TRADING=true in .env.
Paper trading uses the same API endpoints but targets the paper account number.

Docs: https://developer.schwab.com/products/trader-api--individual-
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.parse
from typing import Optional

from agent.broker.schwab_auth import get_access_token

logger = logging.getLogger(__name__)

TRADER_BASE = "https://api.schwabapi.com/trader/v1"

_cached_account_hash: str = ""   # populated on first successful /accounts call


def _account_number() -> str:
    n = os.getenv("SCHWAB_ACCOUNT_NUMBER", "")
    if not n:
        raise RuntimeError("SCHWAB_ACCOUNT_NUMBER not set in .env")
    return n


def _get_account_hash() -> str:
    """
    Return the hashed account number required by Schwab's /accounts/{hash} endpoint.

    Schwab's API requires the encrypted/hashed form of the account number for all
    account-specific calls — the plain account number returns 400 Bad Request.
    We discover it once via GET /trader/v1/accounts, cache it for the session,
    and fall back to the configured SCHWAB_ACCOUNT_NUMBER if the call fails.
    """
    global _cached_account_hash
    if _cached_account_hash:
        return _cached_account_hash
    try:
        data = _get("/accounts/accountNumbers")
        # Response: [{"accountNumber": "...", "hashValue": "..."}, ...]
        if isinstance(data, list) and data:
            configured = os.getenv("SCHWAB_ACCOUNT_NUMBER", "").strip()
            # Prefer the account whose plain number matches SCHWAB_ACCOUNT_NUMBER
            for entry in data:
                if configured and str(entry.get("accountNumber", "")) == configured:
                    _cached_account_hash = entry["hashValue"]
                    logger.info(f"[Schwab] Account hash discovered for account ending ...{configured[-4:] if len(configured) >= 4 else configured}")
                    return _cached_account_hash
            # No match — use the first account
            _cached_account_hash = data[0]["hashValue"]
            logger.info(f"[Schwab] Using first account hash (no SCHWAB_ACCOUNT_NUMBER match)")
            return _cached_account_hash
    except Exception as e:
        logger.warning(f"[Schwab] Could not discover account hash: {e} — falling back to configured number")
    # Last resort: use whatever is configured (may still 400 but gives a clear error)
    return _account_number()


def _headers() -> dict:
    token = get_access_token()
    if not token:
        raise RuntimeError("Not authenticated — call /api/broker/auth first.")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }


def _is_trader_connected() -> bool:
    """True only when the Accounts+Trading token is present."""
    return bool(get_access_token())


def _get(path: str) -> dict:
    url = f"{TRADER_BASE}{path}"
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _post(path: str, body: dict) -> dict:
    url  = f"{TRADER_BASE}{path}"
    data = json.dumps(body).encode()
    req  = urllib.request.Request(url, data=data, method="POST", headers=_headers())
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read()
        return json.loads(raw) if raw else {}


def _delete(path: str) -> None:
    url = f"{TRADER_BASE}{path}"
    req = urllib.request.Request(url, method="DELETE", headers=_headers())
    with urllib.request.urlopen(req, timeout=10):
        pass


# ── Account ────────────────────────────────────────────────────────────────────

def get_account() -> dict:
    """Return account summary (balances, buying power, etc.)."""
    acct = _get_account_hash()
    try:
        data = _get(f"/accounts/{acct}?fields=positions")
        return data
    except Exception as e:
        logger.error(f"get_account error: {e}")
        return {}


def get_positions() -> list[dict]:
    """Return current open positions. Empty list if Trader app not connected."""
    if not _is_trader_connected():
        logger.debug("get_positions: Trader app not connected — skipping")
        return []
    acct = _get_account_hash()
    try:
        data = _get(f"/accounts/{acct}?fields=positions")
        positions = (
            data.get("securitiesAccount", {})
                .get("positions", [])
        )
        result = []
        for p in positions:
            instr = p.get("instrument", {})
            result.append({
                "ticker":           instr.get("symbol", ""),
                "qty":              p.get("longQuantity", 0) - p.get("shortQuantity", 0),
                "avg_price":        p.get("averagePrice", 0),
                "market_value":     p.get("marketValue", 0),
                "unrealized_pnl":   p.get("currentDayProfitLoss", 0),
                "unrealized_pnl_pct": p.get("currentDayProfitLossPercentage", 0),
            })
        return result
    except Exception as e:
        logger.error(f"get_positions error: {e}")
        return []


def get_orders(status: str = "WORKING") -> list[dict]:
    """Return open/recent orders. Empty list if Trader app not connected."""
    if not _is_trader_connected():
        logger.debug("get_orders: Trader app not connected — skipping")
        return []
    acct = _get_account_hash()
    try:
        data = _get(f"/accounts/{acct}/orders?status={status}&maxResults=50")
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"get_orders error: {e}")
        return []


def get_account_summary() -> dict:
    """Return simplified balance summary. Empty dict if Trader app not connected."""
    if not _is_trader_connected():
        logger.debug("get_account_summary: Trader app not connected — skipping")
        return {}
    try:
        data   = get_account()
        acct_d = data.get("securitiesAccount", {})
        bal    = acct_d.get("currentBalances", {})
        return {
            "account_type":    acct_d.get("type", ""),
            "account_number":  acct_d.get("accountNumber", _account_number()),
            "buying_power":    bal.get("buyingPower", 0),
            "cash_balance":    bal.get("cashBalance", 0),
            "equity":          bal.get("liquidationValue", 0),
            "day_pnl":         bal.get("dayTradingBuyingPower", 0),
            "positions_count": len(acct_d.get("positions", [])),
        }
    except Exception as e:
        logger.error(f"account_summary error: {e}")
        return {}


# ── Order placement ────────────────────────────────────────────────────────────

def place_equity_order(
    ticker:      str,
    direction:   str,   # "BUY" | "SELL"
    qty:         int,
    order_type:  str   = "MARKET",   # "MARKET" | "LIMIT"
    limit_price: float = 0.0,
    stop_price:  float = 0.0,
    duration:    str   = "DAY",      # "DAY" | "GOOD_TILL_CANCEL"
    session:     str   = "NORMAL",   # "NORMAL" | "SEAMLESS" (SEAMLESS = extended hours)
) -> dict:
    """
    Place an equity order on the Schwab paper (or live) account.

    For extended-hours orders, pass session="SEAMLESS" and order_type="LIMIT".
    Market orders are not allowed during extended hours.
    """
    acct = _get_account_hash()

    instruction = "BUY" if direction.upper() in ("BUY", "STRONG BUY") else "SELL_SHORT"

    order: dict = {
        "orderType":   order_type,
        "session":     session,
        "duration":    duration,
        "orderStrategyType": "SINGLE",
        "orderLegCollection": [{
            "instruction": instruction,
            "quantity":    qty,
            "instrument":  {
                "symbol":     ticker,
                "assetType":  "EQUITY",
            },
        }],
    }

    if order_type == "LIMIT" and limit_price > 0:
        order["price"] = round(limit_price, 2)

    if order_type == "STOP" and stop_price > 0:
        order["stopPrice"] = round(stop_price, 2)

    try:
        result = _post(f"/accounts/{acct}/orders", order)
        logger.info(f"Order placed: {direction} {qty}×{ticker} @ {order_type}")
        return {"success": True, "order": order, "response": result}
    except Exception as e:
        logger.error(f"place_order error ({ticker}): {e}")
        return {"success": False, "error": str(e)}


def place_bracket_order(
    ticker:      str,
    direction:   str,
    qty:         int,
    entry_price: float,
    target:      float,
    stop:        float,
    session:     str = "NORMAL",
) -> dict:
    """
    Place an OCO bracket order: entry + take-profit + stop-loss in one ticket.
    If entry_price ≈ current market, use MARKET entry + OCO exits.
    """
    acct = _get_account_hash()

    is_buy  = direction.upper() in ("BUY", "STRONG BUY")
    entry_instruction = "BUY"          if is_buy else "SELL_SHORT"
    tp_instruction    = "SELL"         if is_buy else "BUY_TO_COVER"
    sl_instruction    = "SELL"         if is_buy else "BUY_TO_COVER"

    order = {
        "orderStrategyType": "TRIGGER",
        "session":           session,
        "duration":          "DAY",
        "orderType":         "LIMIT",
        "price":             round(entry_price, 2),
        "orderLegCollection": [{
            "instruction": entry_instruction,
            "quantity":    qty,
            "instrument":  {"symbol": ticker, "assetType": "EQUITY"},
        }],
        "childOrderStrategies": [{
            "orderStrategyType": "OCO",
            "childOrderStrategies": [
                {   # Take profit
                    "orderStrategyType": "SINGLE",
                    "session":   session,
                    "duration":  "DAY",
                    "orderType": "LIMIT",
                    "price":     round(target, 2),
                    "orderLegCollection": [{
                        "instruction": tp_instruction,
                        "quantity":    qty,
                        "instrument":  {"symbol": ticker, "assetType": "EQUITY"},
                    }],
                },
                {   # Stop loss
                    "orderStrategyType": "SINGLE",
                    "session":   session,
                    "duration":  "DAY",
                    "orderType": "STOP",
                    "stopPrice": round(stop, 2),
                    "orderLegCollection": [{
                        "instruction": sl_instruction,
                        "quantity":    qty,
                        "instrument":  {"symbol": ticker, "assetType": "EQUITY"},
                    }],
                },
            ],
        }],
    }

    try:
        result = _post(f"/accounts/{acct}/orders", order)
        logger.info(f"Bracket order placed: {direction} {qty}×{ticker} entry={entry_price} tp={target} sl={stop}")
        return {"success": True, "ticker": ticker, "qty": qty,
                "entry": entry_price, "target": target, "stop": stop}
    except Exception as e:
        logger.error(f"bracket_order error ({ticker}): {e}")
        return {"success": False, "error": str(e)}


def cancel_order(order_id: str) -> bool:
    """Cancel an open order by ID."""
    acct = _get_account_hash()
    try:
        _delete(f"/accounts/{acct}/orders/{order_id}")
        return True
    except Exception as e:
        logger.error(f"cancel_order error ({order_id}): {e}")
        return False


def close_position(ticker: str, qty: int, is_long: bool) -> dict:
    """Market-close an existing position."""
    instruction = "SELL" if is_long else "BUY_TO_COVER"
    acct = _get_account_hash()
    order = {
        "orderType": "MARKET",
        "session":   "NORMAL",
        "duration":  "DAY",
        "orderStrategyType": "SINGLE",
        "orderLegCollection": [{
            "instruction": instruction,
            "quantity":    qty,
            "instrument":  {"symbol": ticker, "assetType": "EQUITY"},
        }],
    }
    try:
        result = _post(f"/accounts/{acct}/orders", order)
        return {"success": True, "response": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
