"""
Trading hours classification for NASDAQ equities.

All our tracked tickers are NYSE/NASDAQ-listed equities — none trade 24/7.
Extended-hours (ECN) trading runs 4:00am–8:00pm ET for most names, but
liquidity varies dramatically.  This module assigns each ticker a tier:

  HIGH      — Mega-caps with consistently tight AH spreads; AH signals reliable
  MODERATE  — Large-caps with decent AH volume; use signals with caution
  REGULAR   — Thin AH liquidity; only regular-session signals recommended

The static tier is a floor.  If the after-hours monitor reports that a ticker
had ≥10% of its average daily volume in the most recent AH session, it is
dynamically promoted to HIGH for that session.
"""
from __future__ import annotations

# ── Static tier map ───────────────────────────────────────────────────────────

# Mega-caps — consistently heavy AH + pre-market volume, tight spreads
_HIGH: frozenset[str] = frozenset({
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "GOOG", "TSLA", "AVGO", "NFLX",
    "AMD",   "QCOM", "INTC", "MU",   "SMCI",
})

# Large-caps — meaningful AH/PM activity; news-driven and crypto-adjacent names
_MODERATE: frozenset[str] = frozenset({
    # NASDAQ-100 tech
    "ADBE", "CSCO", "INTU", "AMAT", "PANW",
    "CRWD", "MRVL", "KLAC", "LRCX", "ADI",
    "SNPS", "CDNS", "ISRG", "REGN", "BKNG",
    "ADP",  "SBUX", "COST", "AMGN", "ABNB",
    "DDOG", "ZS",   "WDAY", "MELI", "ARM",
    "APP",  "COIN", "TTD",  "HOOD", "TEAM",
    "OKTA", "FTNT", "PYPL", "CEG",  "AXON",
    "SNOW", "NET",  "ANET", "MDB",  "HUBS",
    # High retail AH/PM participation
    "PLTR", "MSTR", "MARA", "RIVN", "LCID",
    "SOUN", "IONQ", "CELH", "ENPH", "RKLB",
    "ASTS", "RGTI", "QUBT", "SNAP", "RBLX",
    "UPST", "AFRM", "CVNA", "ROKU", "PINS",
    "LYFT", "DKNG", "SOFI", "HOOD", "RIOT",
    # Biotech / pharma — earnings + FDA catalyst spikes in AH/PM
    "MRNA", "BNTX", "VRTX", "ALNY", "REGN",
    "GILD", "BIIB", "AMGN", "CRSP", "BEAM",
    # China ADRs — trade in AH when Hong Kong/China market opens
    "BIDU", "PDD",  "JD",   "BABA", "NIO",
    "XPEV", "LI",   "BILI",
    # ETF-based momentum (high volume, trades continuously in AH)
    "TQQQ", "SQQQ", "SOXL",
})

# Anything not listed above defaults to REGULAR

# ── Public API ────────────────────────────────────────────────────────────────

def get_trading_tier(ticker: str, ah_volume_ratio: float = 0.0) -> str:
    """
    Return the trading-hours tier for a ticker.

    Parameters
    ----------
    ticker          : Ticker symbol.
    ah_volume_ratio : Ratio of most-recent AH session volume to average daily
                      volume (0.0 if no AH data available).

    Returns
    -------
    "HIGH" | "MODERATE" | "REGULAR"
    """
    # Dynamic promotion: heavy AH volume → upgrade to HIGH regardless of static tier
    if ah_volume_ratio >= 0.10:
        return "HIGH"
    if ticker in _HIGH:
        return "HIGH"
    if ticker in _MODERATE or ah_volume_ratio >= 0.03:
        return "MODERATE"
    return "REGULAR"


def is_signal_recommended(tier: str, session: str) -> bool:
    """Return True if acting on a signal is recommended given tier and session."""
    if session in ("PRIME", "STANDARD", "LUNCH_BLOCK", "CLOSING_CAUTION", "RESTRICTED"):
        return True                                    # all tiers OK in regular hours
    if session in ("AFTER_HOURS", "PRE_MARKET"):
        return tier in ("HIGH", "MODERATE")            # extended-hours: two tiers eligible
    return False                                       # CLOSED / HARD_CLOSE — no trades


_TIER_META: dict[str, dict] = {
    "HIGH": {
        "label":  "Extended",
        "detail": "High AH liquidity — extended-hours signals reliable",
        "color":  "#22c55e",   # green
        "icon":   "🌙",
    },
    "MODERATE": {
        "label":  "Mod-AH",
        "detail": "Moderate AH liquidity — use extended-hours signals with caution",
        "color":  "#f59e0b",   # amber
        "icon":   "🌙",
    },
    "REGULAR": {
        "label":  "Reg only",
        "detail": "Thin AH liquidity — regular session signals only recommended",
        "color":  "#64748b",   # muted
        "icon":   "🕐",
    },
}


def get_tier_meta(tier: str) -> dict:
    """Return display metadata for a tier."""
    return _TIER_META.get(tier, _TIER_META["REGULAR"])
