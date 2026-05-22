"""
NASDAQ Ticker Universe — ~500 liquid stocks in 3 tiers.

Tier 1 (100): NASDAQ-100 core — always included in every scan cycle.
Tier 2 (150): NASDAQ-200 / quality mid-caps — included when showing activity.
Tier 3 (200): High-vol / momentum / sector plays — included only when active.

Architecture
------------
Each scan cycle the UniverseManager:
  1. Calls Schwab /quotes for all ~500 tickers (1-2 API calls, cached 5 min).
  2. Scores each Tier 2/3 ticker: score = volume × (1 + abs(pct_change)).
  3. Returns Tier 1 (always) + top-N Tier 2/3 by score + user watchlist.

This means the scanner only fetches /pricehistory for ~150-175 active tickers
per cycle instead of all 500, keeping Schwab API usage well within limits.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

# ── Tier 1: NASDAQ-100 core — scan every cycle ────────────────────────────────
TIER1: list[str] = [
    # Mega-cap tech
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","NFLX",
    # Large-cap tech / semis
    "AMD","ADBE","INTU","QCOM","AMAT","MU","PANW","CRWD","MRVL","KLAC",
    "LRCX","ADI","SNPS","CDNS","ISRG","REGN","BKNG","ADP","AMGN","SBUX",
    # NASDAQ-100 completions
    "COST","CMCSA","CSCO","WDAY","MELI","ARM","TTD","SMCI","TEAM","FTNT",
    "PYPL","AXON","CEG","ABNB","DDOG","ZS","NET","ANET","OKTA","SNOW",
    "APP","ORLY","CTAS","ROP","CPRT","ROST","FAST","PCAR","PAYX","ODFL",
    "IDXX","VRSK","EA","DLTR","ALGN","ILMN","TTWO","ON","NXPI","MCHP",
    "SWKS","CTSH","NTAP","VRSN","CHKP","EXPE","DXCM","BIIB","MNST","GEHC",
    "GILD","NDAQ","INTC","PEP","LULU","MPWR","CSGP","KDP","SIRI","HON",
    # High-liquidity growth (always worth scanning)
    "COIN","HOOD","PLTR","MSTR","HUBS","DKNG","MDB","SNAP","RBLX","SOFI",
]

# ── Tier 2: NASDAQ-200 / quality mid-caps ─────────────────────────────────────
TIER2: list[str] = [
    # SaaS / cloud
    "CFLT","MNDY","NTNX","WIX","APPN","ZI","DOMO","AMPL","FRSH","FROG",
    "GTLB","BRZE","NCNO","CLBT","ALTR","PRGS","SMAR","JAMF","FSLY","SPSC",
    "QTWO","LSCC","PCTY","ESTC","PAGER","VMEO","RAMP","BIGC","FOUR","ALRM",
    # Fintech / consumer
    "AFRM","UPST","LYFT","CART","OPEN","PINS","ROKU","ZM","PTON","BILL",
    "TWLO","DOCU","CHWY","CELH","ENPH","ADSK","LOGI","MTCH","CYBR","MGNI",
    "PAGS","OPFI","PRAA","WRLD","HIMS","XPERI","OSPN","EZCORP","UPBD","GDOT",
    # Semis mid-cap
    "ACLS","ONTO","COHU","FORM","AEHR","MTSI","POWI","SITM","CRUS","AMBA",
    "SLAB","AEIS","DIOD","ALGM","SMTC","UCTT","CEVA","AXTI","ACMR","MBLY",
    "ALAB","LSCC","KLIC","CCMP","IQST","NVEC","PSEM","IPGP","IIVI","MKSI",
    # Healthcare / biotech (established)
    "HOLX","INCY","EXAS","SRPT","IONS","RARE","FOLD","NKTR","BLUE","NTLA",
    "HALO","LNTH","PRCT","TMDX","INMD","RXST","RVNC","ESTA","VERV","IMVT",
    "INSM","TGTX","PTCT","ARQT","ADMA","RCKT","VKTX","CDNA","ADPT","SNDX",
    # Clean energy / infrastructure
    "FSLR","RUN","ARRY","SEDG","NOVA","MAXN","FLNC","STEM","AMRC","SHLS",
    "HASI","SPWR","CWEN","NOVA","AY","BSIG","ITRI","REGI","PLUG","BE",
    # China ADRs (NASDAQ-listed)
    "BIDU","PDD","NTES","JD","BILI","LI","KC","DQ","NOAH","CAN",
    # Misc quality mid-cap
    "SOUN","IONQ","TENB","VRNS","QLYS","BMRN","ALNY","VRTX","ACAD","CRSP",
    "BEAM","EDIT","RXRX","ABCL","FATE","MRNA","BNTX","NVAX","UPWK","ROOT",
    "WOLF","CVNA","RELY","PSFE","PYCR","MAPS","TPVG","CLVT","EVBG","BAND",
    # Telecom / media
    "TMUS","CHTR","DISH","SIRI","LUMN","VSAT","VIAV","INFN","BAND","IRDM",
    # Consumer / retail
    "WING","TXRH","PZZA","JACK","KURA","CAKE","RRGB","PLAY","DENN","CBRL",
    # Financial / banking (NASDAQ-listed)
    "LPLA","SSNC","SEIC","HBAN","FITB","BOKF","PACW","SBGI","AGNC","LQDA",
    # Industrial / energy tech
    "PLUG","BE","ITRI","AMRC","SHLS","STEM","BSIG","HASI","CLNE","REGI",
]

# ── Tier 3: High-vol / momentum / sector plays ────────────────────────────────
TIER3: list[str] = [
    # Crypto mining / blockchain / Web3
    "MARA","RIOT","HUT","CLSK","BTBT","CIFR","BITF","SDIG","IREN","WULF",
    "MIGI","BTDR","CORZ","GRVY","BSRT","ARBK","HIVE","BFRI","BFIN","BTCS",
    # EV / clean mobility
    "RIVN","LCID","NKLA","GOEV","BLNK","CHPT","EVGO","ZEV","PTRA","AYRO",
    # Space / quantum / deep tech
    "RKLB","ASTS","RGTI","QUBT","QMCO","LUNR","SPCE","ASTR","VORB","MNTS",
    "IONQ","IQM","ARQQ","QUBT","QTWO","OQAL","QBTS","LASR","LIDR","INVZ",
    # Biotech small-cap / clinical stage
    "ROIV","GPCR","KRYS","KRTX","NRIX","PTGX","SEER","ANAB","BBIO","PRTA",
    "RAPT","AGEN","IOVA","PHAT","MGNX","XNCR","CLOV","TARS","XOMA","RLAY",
    "GRPH","GMAB","APRE","TELA","CDXS","NKTR","IMRN","CNTA","ACCD","RVMD",
    # High-vol consumer / retail
    "FIGS","BIRD","MYPS","LPSN","WISH","REAL","RDFN","COMS","MAPS","PAYO",
    # Fintech small-cap
    "DAVE","RELY","JMIA","CURO","FLGT","VNET","PAGS","OPFI","LMND","PSFE",
    # Semis / lidar / sensors
    "OUST","LAZR","MVIS","AMBA","ISSI","QRVO","FORM","SWIR","CREE","IMOS",
    # Cannabis
    "TLRY","SNDL","CGC","ACB","IIPR","CURLF","APHA","CRON","FLCX","HYFM",
    # Misc momentum / high-beta
    "VIEW","MTTR","TPVG","GAIN","PFLT","HTGC","ARCC","ORCC","GBDC","TCPC",
    # International / ADR momentum
    "XPEV","NIO","GRAB","SEA","DESP","MMYT","WB","VNET","DOYU","HUYA",
    # Special situations / SPAC legacies
    "HYLN","RIDE","COVA","ATIP","MNTV","GREE","FRGE","LWAY","AMTD","VCNX",
    # More biotech momentum
    "ALKS","PTGX","CDXS","IMRN","CNTA","RVMD","AGEN","XNCR","XOMA","TARS",
    "RCKT","ADMA","VKTX","NTGN","KROS","IMTX","PGEN","SRRA","IMCR","ABCB",
    # More tech momentum
    "ARLO","IPGP","COHR","VICR","IIIV","PLAB","FARO","MFIN","NTCT","ATEN",
    # More fintech / payments
    "IMXI","GDOT","RPAY","EVTC","CASS","CATC","PRAA","WRLD","EZCORP","QFIN",
    # ETF-adjacent NASDAQ liquid
    "SQQQ","TQQQ","QQQ","QQQM","FNGU","TECL","NAIL","LABU","LABD","SOXL",
]

# ── Deduplicated full universe ────────────────────────────────────────────────
FULL_UNIVERSE: list[str] = list(dict.fromkeys(TIER1 + TIER2 + TIER3))

_TIER1_SET = set(TIER1)
_TIER2_SET = set(TIER2) - _TIER1_SET
_TIER3_SET = set(TIER3) - _TIER1_SET - _TIER2_SET


def get_universe() -> list[str]:
    """Return all ~500 tickers in the universe (Tier 1 first)."""
    return FULL_UNIVERSE.copy()


# ── Universe Manager — smart active-ticker selection ─────────────────────────

class UniverseManager:
    """
    Maintains a cached hot list of tickers for the current scan cycle.

    Every HOT_CACHE_TTL seconds it:
      1. Calls Schwab /quotes for the full Tier 2+3 universe (1-2 API calls).
      2. Scores each ticker: volume × (1 + abs(pct_change)).
      3. Returns Tier 1 always + top-N Tier 2/3 by score + user watchlist.

    Between refreshes it returns the cached list so the scanner doesn't
    pay any extra API cost.
    """

    HOT_CACHE_TTL    = 300   # refresh bulk quotes every 5 min
    N_TIER2_ACTIVE   = 150   # how many Tier 2/3 tickers to include when active

    def __init__(self) -> None:
        self._hot:      list[str] = []
        self._hot_ts:   float     = 0.0
        self._lock                = threading.Lock()
        self._callbacks: list[Callable] = []

    def register_callback(self, fn: Callable) -> None:
        """Called with (new_hot_list) after each refresh."""
        self._callbacks.append(fn)

    def get_active_tickers(self, force_refresh: bool = False) -> list[str]:
        """
        Return the current hot list.  Refreshes from Schwab bulk quotes
        when the cache is stale.  Falls back to Tier 1 + Tier 2 if Schwab
        is not yet authorised.
        """
        with self._lock:
            if not force_refresh and self._hot and (time.time() - self._hot_ts) < self.HOT_CACHE_TTL:
                return list(self._hot)

        hot = self._build_hot_list()
        with self._lock:
            self._hot    = hot
            self._hot_ts = time.time()

        for fn in self._callbacks:
            try:
                fn(hot)
            except Exception:
                pass

        logger.info(f"[Universe] Active tickers refreshed: {len(hot)} selected from {len(FULL_UNIVERSE)} universe")
        return list(hot)

    def _build_hot_list(self) -> list[str]:
        # Tier 1 is always included
        result: list[str] = list(TIER1)
        tier1_set = set(result)

        # Bulk-screen Tier 2 + 3
        broader = [t for t in TIER2 + TIER3 if t not in tier1_set]
        try:
            from agent.broker.schwab_market_data import fetch_quotes_bulk
            quotes = fetch_quotes_bulk(broader)
        except Exception:
            quotes = {}

        if quotes:
            scored = []
            for sym in broader:
                if sym not in quotes:
                    continue
                q   = quotes[sym]
                vol = float(q.get("volume") or 0)
                pct = abs(float(q.get("pct_change") or 0))
                # weight volume more heavily; boost tickers with large % moves
                score = vol * (1.0 + pct * 2.0)
                scored.append((sym, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            result += [sym for sym, _ in scored[:self.N_TIER2_ACTIVE]]
        else:
            # Schwab not available yet — fall back to static Tier 2
            result += list(TIER2[:self.N_TIER2_ACTIVE])

        # Always include user watchlist items
        try:
            from config import load_watchlist
            for t in load_watchlist():
                if t not in set(result):
                    result.append(t)
        except Exception:
            pass

        # Also inject any live movers from the Schwab streamer
        try:
            from agent.broker.schwab_market_data import fetch_top_movers_symbols
            movers = fetch_top_movers_symbols(n=20)
            seen = set(result)
            for m in movers:
                if m not in seen:
                    result.append(m)
                    seen.add(m)
        except Exception:
            pass

        return result

    def universe_size(self) -> int:
        return len(FULL_UNIVERSE)

    def hot_size(self) -> int:
        with self._lock:
            return len(self._hot)


# ── Module-level singleton ────────────────────────────────────────────────────
_manager = UniverseManager()


def get_active_tickers_universe(force_refresh: bool = False) -> list[str]:
    """Drop-in replacement for config.get_active_tickers() with smart screening."""
    return _manager.get_active_tickers(force_refresh)


def get_universe_manager() -> UniverseManager:
    return _manager
