"""
Sector ETF context — maps each NASDAQ ticker to its primary sector ETF,
fetches that ETF's intraday data, and determines whether the stock is
leading or lagging its sector.

Why this matters
----------------
If NVDA is weak but SMH (semis ETF) is strong, the weakness is stock-specific
— the broader sector still has tailwind.  If both are weak, you have
sector-level headwind and should discount bullish signals.

Sector multipliers on composite score
--------------------------------------
Sector BULLISH  + Stock LEADING  → 1.20× (best environment)
Sector BULLISH  + Stock IN-LINE  → 1.05×
Sector BULLISH  + Stock LAGGING  → 0.90×
Sector BEARISH  + Stock LAGGING  → 1.20× (short environment)
Sector BEARISH  + Stock IN-LINE  → 1.05×
Sector BEARISH  + Stock LEADING  → 0.85×
Neutral                          → 1.00×
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Ticker → Sector ETF mapping ───────────────────────────────────────────────
SECTOR_MAP: dict[str, str] = {
    # Technology / Software
    "MSFT": "XLK", "AAPL": "XLK", "ADBE": "XLK", "CRM": "XLK",
    "ORCL": "XLK", "INTU": "XLK", "NOW": "XLK", "WDAY": "XLK",
    "DDOG": "XLK", "SNOW": "XLK", "ZS": "XLK", "OKTA": "XLK",
    "CRWD": "XLK", "PANW": "XLK", "FTNT": "XLK", "TEAM": "XLK",
    "CSCO": "XLK", "PYPL": "XLK",
    # Semiconductors
    "NVDA": "SMH", "AMD": "SMH", "INTC": "SMH", "AVGO": "SMH",
    "QCOM": "SMH", "MU": "SMH", "AMAT": "SMH", "KLAC": "SMH",
    "LRCX": "SMH", "MRVL": "SMH", "ADI": "SMH", "SNPS": "SMH",
    "CDNS": "SMH", "SMCI": "SMH", "ARM": "SMH",
    # Internet / Communication
    "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC", "AMZN": "XLC",
    "HOOD": "XLC", "TTD": "XLC",
    # Consumer Discretionary
    "TSLA": "XLY", "AMZN": "XLY", "BKNG": "XLY", "ABNB": "XLY",
    "SBUX": "XLY", "MELI": "XLY",
    # Healthcare / Biotech
    "ISRG": "XBI", "REGN": "XBI", "AMGN": "XBI",
    # Financials / Crypto
    "COIN": "XLF", "HOOD": "XLF",
    # Consumer Staples
    "COST": "XLP",
    # Utilities
    "CEG": "XLU",
    # Industrial / Defence
    "AXON": "XLI",
    # Growth / Cloud (use QQQ as default for anything not mapped)
}
_DEFAULT_SECTOR_ETF = "QQQ"

# All unique sector ETFs we need to fetch
ALL_SECTOR_ETFS = list(set(SECTOR_MAP.values())) + [_DEFAULT_SECTOR_ETF]


@dataclass
class SectorContext:
    etf:            str   = "QQQ"
    sector_change:  float = 0.0   # % intraday change of the ETF
    sector_trend:   str   = "NEUTRAL"  # BULLISH | BEARISH | NEUTRAL
    stock_rs:       float = 1.0   # stock change / sector change
    stock_vs_sector: str  = "IN_LINE"  # LEADING | LAGGING | IN_LINE | COUNTER
    score_mult:     float = 1.0
    description:    str   = ""

    def to_dict(self) -> dict:
        return {
            "etf":             self.etf,
            "sector_change":   round(self.sector_change, 3),
            "sector_trend":    self.sector_trend,
            "stock_rs":        round(self.stock_rs, 3),
            "stock_vs_sector": self.stock_vs_sector,
            "score_mult":      self.score_mult,
            "description":     self.description,
        }


def _intraday_return(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    try:
        op = float(df["Open"].iloc[0])
        cl = float(df["Close"].iloc[-1])
        return (cl - op) / op if op else 0.0
    except Exception:
        return 0.0


def analyse_sector_context(
    ticker:      str,
    df_stock:    pd.DataFrame,
    etf_frames:  dict[str, pd.DataFrame],
) -> SectorContext:
    """
    Compare stock intraday performance against its sector ETF.

    Parameters
    ----------
    ticker      : stock symbol
    df_stock    : stock's 1-min intraday DataFrame
    etf_frames  : dict {etf_symbol: DataFrame} for all fetched ETFs
    """
    ctx = SectorContext()

    etf_sym   = SECTOR_MAP.get(ticker, _DEFAULT_SECTOR_ETF)
    ctx.etf   = etf_sym
    df_etf = etf_frames.get(etf_sym)
    if df_etf is None:
        df_etf = etf_frames.get(_DEFAULT_SECTOR_ETF)

    stock_ret = _intraday_return(df_stock)
    sector_ret = _intraday_return(df_etf)

    ctx.sector_change = round(sector_ret * 100, 3)
    ctx.sector_trend  = (
        "BULLISH"  if sector_ret > 0.003  else
        "BEARISH"  if sector_ret < -0.003 else
        "NEUTRAL"
    )

    # RS of stock vs sector
    if abs(sector_ret) > 0.0005:
        rs = stock_ret / sector_ret
    else:
        rs = 1.0
    ctx.stock_rs = round(rs, 3)

    if rs >= 1.3:
        vs = "LEADING"
    elif rs <= 0.7:
        vs = "LAGGING"
    elif rs < 0.0:
        vs = "COUNTER"
    else:
        vs = "IN_LINE"
    ctx.stock_vs_sector = vs

    # Composite score multiplier
    if ctx.sector_trend == "BULLISH":
        mult = {"LEADING": 1.20, "IN_LINE": 1.05, "COUNTER": 0.80}.get(vs, 0.90)
    elif ctx.sector_trend == "BEARISH":
        mult = {"LAGGING": 1.20, "IN_LINE": 1.05, "LEADING": 0.80}.get(vs, 0.85)
    else:
        mult = 1.0
    ctx.score_mult = mult

    stock_pct  = stock_ret  * 100
    sector_pct = sector_ret * 100
    ctx.description = (
        f"{etf_sym} {sector_pct:+.2f}% ({ctx.sector_trend}) | "
        f"Stock {stock_pct:+.2f}% → {vs} (RS {rs:.2f})"
    )
    return ctx


# ── Shared ETF frame cache (updated each scan by scanner) ─────────────────────

_etf_cache: dict[str, pd.DataFrame] = {}


def update_etf_cache(frames: dict[str, pd.DataFrame]) -> None:
    _etf_cache.update(frames)


def get_sector_context(ticker: str, df_stock: pd.DataFrame) -> SectorContext:
    return analyse_sector_context(ticker, df_stock, _etf_cache)
