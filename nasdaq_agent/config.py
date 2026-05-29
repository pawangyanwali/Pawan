import os
from dotenv import load_dotenv

load_dotenv()

# Twelve Data removed — all market data now served by Schwab Market Data API.
# TWELVE_DATA_API_KEY kept as empty string so any stale imports don't hard-error.
TWELVE_DATA_API_KEY = ""

# ── Schwab Market Data (primary data source) ─────────────────────────────────
# Always enabled — Twelve Data has been removed.
SCHWAB_ENABLED = True

# ── Legacy stubs kept for import compat (no longer used functionally) ─────────
CALL_GAP    = 0.67   # Schwab rate gap: 1.5 req/s → 90 req/min (limit is 120)
BATCH_SIZE  = 1      # Schwab pricehistory is per-symbol (no multi-symbol batching)

# ── Scan / retrain cadence ────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS   = 60      # 1 min — 50 tickers / 20 per batch = 3 calls ≈ 0.6s scan
ML_RETRAIN_INTERVAL     = 21600   # Full XGBoost retrain every 6 hours
DEEP_FINETUNE_INTERVAL  = 3600    # Deep BiLSTM fine-tune every 1 hour (uses cached data, no API cost)

# ── Historical data windows ───────────────────────────────────────────────────
DATA_PERIOD_DAYS    = 180    # 6 months of 5-min data for intraday ML (~14,040 bars; cap 5000)
REALTIME_OUTPUTSIZE = 300   # 1-min bars per scan (≈5h coverage, reduced for memory)

# ── Per-interval cache TTLs (seconds) ────────────────────────────────────────
CACHE_TTL_5M  =   600   # 10 min — doubled from 300s; halves 5-min API calls per cycle
CACHE_TTL_1H  =  3600   # 1 hour
CACHE_TTL_1D  = 86400   # 24 hours  (also used as DAILY_CACHE_TTL for compatibility)
DAILY_CACHE_TTL = CACHE_TTL_1D

# ── Prediction thresholds (dashboard colouring) ───────────────────────────────
STRONG_BUY_THRESHOLD  =  0.60
BUY_THRESHOLD         =  0.30
SELL_THRESHOLD        = -0.30
STRONG_SELL_THRESHOLD = -0.60

# ── Ticker universe — ~500 liquid NASDAQ stocks across 3 tiers ───────────────
# Tiers live in agent/ticker_universe.py; imported here so the rest of the
# codebase can still use NASDAQ_TICKERS, CLUSTER_*_TICKERS from config.

from agent.ticker_universe import TIER1, TIER2, TIER3, FULL_UNIVERSE  # noqa: E402

# ML training uses the full 477-ticker universe so all tiers get per-ticker models.
TRAINING_TICKERS: list[str] = list(FULL_UNIVERSE)

# Cluster A = Tier 1 (NASDAQ-100 core, always scanned)
CLUSTER_A_TICKERS: list[str] = list(TIER1)

# Cluster B = Tier 2 (quality mid-caps, scanned when active)
CLUSTER_B_TICKERS: list[str] = list(TIER2)

# Cluster C = Tier 3 (high-vol / momentum, scanned when in movers)
CLUSTER_C_TICKERS: list[str] = list(TIER3)

# Master deduplicated list (~500 tickers)
NASDAQ_TICKERS: list[str] = list(FULL_UNIVERSE)

# Cluster assignment map
TICKER_CLUSTER: dict[str, str] = {}
for t in CLUSTER_A_TICKERS: TICKER_CLUSTER[t] = "A"
for t in CLUSTER_B_TICKERS: TICKER_CLUSTER.setdefault(t, "B")
for t in CLUSTER_C_TICKERS: TICKER_CLUSTER.setdefault(t, "C")

# Pipeline config
PIPELINE_WORKERS = 8   # ThreadPoolExecutor workers for parallel scan

# ── Company name map (avoids API calls) ──────────────────────────────────────
TICKER_NAMES = {
    "AAPL":"Apple","MSFT":"Microsoft","NVDA":"NVIDIA","AMZN":"Amazon","META":"Meta",
    "GOOGL":"Alphabet A","GOOG":"Alphabet C","TSLA":"Tesla","AVGO":"Broadcom","COST":"Costco",
    "NFLX":"Netflix","AMD":"AMD","ADBE":"Adobe","QCOM":"Qualcomm","PEP":"PepsiCo",
    "CSCO":"Cisco","INTU":"Intuit","CMCSA":"Comcast","AMAT":"Applied Materials","AMGN":"Amgen",
    "HON":"Honeywell","MU":"Micron","ISRG":"Intuitive Surgical","LRCX":"Lam Research",
    "KLAC":"KLA Corp","ADI":"Analog Devices","REGN":"Regeneron","SNPS":"Synopsys",
    "CDNS":"Cadence","MDLZ":"Mondelez","PYPL":"PayPal","GILD":"Gilead","CTAS":"Cintas",
    "MRVL":"Marvell","PANW":"Palo Alto Networks","ABNB":"Airbnb","CRWD":"CrowdStrike",
    "ORLY":"O'Reilly Auto","ADSK":"Autodesk","DXCM":"Dexcom","ADP":"ADP","MNST":"Monster Bev",
    "BKNG":"Booking Holdings","FTNT":"Fortinet","KDP":"Keurig Dr Pepper","BIIB":"Biogen",
    "MCHP":"Microchip Tech","SBUX":"Starbucks","AEP":"AEP","NXPI":"NXP Semi",
    "PCAR":"PACCAR","CPRT":"Copart","EXC":"Exelon","ROST":"Ross Stores","ODFL":"Old Dominion",
    "PAYX":"Paychex","WDAY":"Workday","FAST":"Fastenal","IDXX":"IDEXX Labs","ZS":"Zscaler",
    "TEAM":"Atlassian","DDOG":"Datadog","CEG":"Constellation Energy",
    "TTD":"Trade Desk","EA":"Electronic Arts","MRNA":"Moderna","ALGN":"Align Tech",
    "DLTR":"Dollar Tree","VRSK":"Verisk","GEHC":"GE HealthCare","ON":"ON Semi","ROP":"Roper",
    "ILMN":"Illumina","TTWO":"Take-Two","OKTA":"Okta","EXPE":"Expedia","SWKS":"Skyworks",
    "CTSH":"Cognizant","VRSN":"VeriSign","NTAP":"NetApp","HOLX":"Hologic","WBA":"Walgreens",
    "SIRI":"Sirius XM","ZM":"Zoom","DOCU":"DocuSign","PTON":"Peloton","LCID":"Lucid",
    "RIVN":"Rivian","CHKP":"Check Point","INTC":"Intel","ARM":"ARM Holdings","SMCI":"Super Micro",
    "MELI":"MercadoLibre","APP":"AppLovin","COIN":"Coinbase","HOOD":"Robinhood",
    "LULU":"Lululemon","NDAQ":"Nasdaq Inc","MPWR":"Monolithic Power","ENPH":"Enphase Energy",
    "FSLR":"First Solar","CELH":"Celsius Holdings","AXON":"Axon Enterprise",
    "SNOW":"Snowflake","MSTR":"MicroStrategy",
    # Cluster B additions (omit keys already defined above to avoid duplicate overrides)
    "PLTR":"Palantir","MARA":"Marathon Digital",
    "SOUN":"SoundHound AI","IONQ":"IonQ","NET":"Cloudflare","ANET":"Arista Networks",
    "MDB":"MongoDB","SNAP":"Snap","RBLX":"Roblox","UPST":"Upstart","AFRM":"Affirm",
    "CVNA":"Carvana","LYFT":"Lyft","ROKU":"Roku","PINS":"Pinterest",
    "BILL":"Bill.com","TWLO":"Twilio","CHWY":"Chewy",
    "GTLB":"GitLab","HUBS":"HubSpot","DKNG":"DraftKings",
    "LOGI":"Logitech","MTCH":"Match Group","CYBR":"CyberArk",
    "SOFI":"SoFi Technologies",
    # Cluster C additions
    "RGTI":"Rigetti Computing","QUBT":"Quantum Computing Inc","RKLB":"Rocket Lab",
    "ASTS":"AST SpaceMobile","WOLF":"Wolfspeed","RUN":"Sunrun","ARRY":"Array Technologies",
    "LI":"Li Auto","BIDU":"Baidu","NTES":"NetEase","JD":"JD.com","PDD":"PDD Holdings",
    "BILI":"Bilibili","NVAX":"Novavax","BNTX":"BioNTech",
    "VRTX":"Vertex Pharmaceuticals","ALNY":"Alnylam Pharmaceuticals",
    "BMRN":"BioMarin Pharmaceutical","FATE":"Fate Therapeutics","ACAD":"ACADIA Pharmaceuticals",
    "CRSP":"CRISPR Therapeutics","BEAM":"Beam Therapeutics","EDIT":"Editas Medicine",
    "RXRX":"Recursion Pharmaceuticals","ABCL":"AbCellera Biologics",
    "PTON":"Peloton","OPEN":"Opendoor Technologies","CART":"Instacart",
    "TENB":"Tenable Holdings","VRNS":"Varonis Systems","QLYS":"Qualys",
}

# ── Market regime / benchmark tickers ────────────────────────────────────────
REGIME_TICKERS = ["SPY", "QQQ"]   # fetched each scan cycle for regime detection

# ── Sector ETF universe (fetched alongside regime tickers) ───────────────────
from agent.sector_etf import ALL_SECTOR_ETFS as _SECTOR_ETFS   # noqa: E402
SECTOR_ETF_TICKERS = list(set(REGIME_TICKERS + _SECTOR_ETFS))

# ── Paper trading ─────────────────────────────────────────────────────────────
PAPER_TRADE_MIN_CONFIDENCE = 25.0   # data-collection floor — intentionally low so the
                                    # adaptive filter can observe and learn from 25–55%
                                    # confidence trades. Production live-trading gate is
                                    # the adaptive filter's dynamic_threshold (55–63%).

# ── Watchlist persistence ─────────────────────────────────────────────────────
import json as _json, pathlib as _pathlib
_WATCHLIST_PATH = _pathlib.Path(__file__).parent / "data" / "watchlist.json"

def load_watchlist() -> list:
    try:
        if _WATCHLIST_PATH.exists():
            return _json.loads(_WATCHLIST_PATH.read_text())
    except Exception:
        pass
    return []

def save_watchlist(tickers: list) -> None:
    _WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _WATCHLIST_PATH.write_text(_json.dumps(sorted(set(tickers))))

# Active ticker universe — uses smart Schwab bulk-quote screening.
# Returns Tier 1 always + top active from Tier 2/3 (ranked by volume×move)
# + any user watchlist additions.  Falls back to full list if Schwab is down.
def get_active_tickers() -> list:
    try:
        from agent.ticker_universe import get_active_tickers_universe
        return get_active_tickers_universe()
    except Exception:
        pass
    # Fallback: full static list + watchlist
    extra = [t for t in load_watchlist() if t not in NASDAQ_TICKERS]
    return NASDAQ_TICKERS + extra

# ── Earnings blackout ─────────────────────────────────────────────────────────
PRE_EARNINGS_BLACKOUT_DAYS  = 3   # suppress signals N days before earnings
POST_EARNINGS_COOLDOWN_DAYS = 1   # suppress 1 day after earnings

# ── Position sizing — driven by .env so no code change needed ────────────────
# Set these in /opt/nasdaq-agent/.env:
#   TRADING_ACCOUNT_SIZE=50000   (your actual capital)
#   TRADING_RISK_PCT=1.5         (% of account risked per trade)
#   TRADING_MAX_POSITION_PCT=10  (max single position as % of account)
DEFAULT_ACCOUNT_SIZE = float(os.getenv("TRADING_ACCOUNT_SIZE",  "50000"))
DEFAULT_RISK_PCT     = float(os.getenv("TRADING_RISK_PCT",      "1.5"))
MAX_POSITION_PCT     = float(os.getenv("TRADING_MAX_POSITION_PCT", "10.0"))

# ── Paper trading budget & capital management ─────────────────────────────────
# Total capital pool for paper trading — every open position draws from this.
# Available in .env: PAPER_BUDGET=50000
PAPER_BUDGET             = float(os.getenv("PAPER_BUDGET",           str(DEFAULT_ACCOUNT_SIZE)))
# Max % of budget a single trade can consume as position value (entry × shares)
# e.g. 5% of $50k = $2,500 max per trade
PAPER_MAX_TRADE_PCT      = float(os.getenv("PAPER_MAX_TRADE_PCT",    "5.0"))
# Max % of budget in open positions simultaneously
# e.g. 40% of $50k = $20,000 max allocated at once
PAPER_MAX_ALLOCATED_PCT  = float(os.getenv("PAPER_MAX_ALLOCATED_PCT","40.0"))
# Hard ceiling on concurrent open paper trades (overrides the 20 in paper_trading.py)
PAPER_MAX_OPEN_TRADES    = int(os.getenv(  "PAPER_MAX_OPEN_TRADES",  "10"))

# ── Alpha Strike Trader — PRD risk parameters ─────────────────────────────────
# All configurable via .env — no code changes needed.
#
# Daily P&L targets:
#   DAILY_PROFIT_TARGET=1000   ($1,000 triggers Profit Protect Mode)
#   DAILY_PROFIT_MAX=1500      ($1,500 halts trading for the day)
#
# Daily loss limits (% of account):
#   DAILY_LOSS_WARNING_PCT=1.5    (1.5% loss → 15-min pause, alert)
#   DAILY_LOSS_HALT_PCT=2.5       (2.5% loss → halt for the day)
#
# Trade limits:
#   MAX_CONCURRENT_TRADES=3       (hard cap on simultaneous open positions)
#   MAX_PORTFOLIO_HEAT_PCT=1.5    (max combined open risk as % of account)
#   MAX_CONSECUTIVE_LOSSES=5      (full halt after N consecutive losses)
#   COOLDOWN_LOSSES=3             (30-min cooldown after N consecutive losses)
#
# Profit Protect Mode (triggered at DAILY_PROFIT_TARGET):
#   PROFIT_PROTECT_CONF=80        (raise minimum confidence to 80%)
#   PROFIT_PROTECT_SIZE=0.60      (reduce position size to 60% of normal)
#   PROFIT_PROTECT_DRAWDOWN=300   (halt if drawdown from day's peak exceeds $300)

DAILY_PROFIT_TARGET_USD  = float(os.getenv("DAILY_PROFIT_TARGET",        "1000"))
DAILY_PROFIT_MAX_USD     = float(os.getenv("DAILY_PROFIT_MAX",           "1500"))
DAILY_LOSS_WARNING_PCT   = float(os.getenv("DAILY_LOSS_WARNING_PCT",     "1.5"))
DAILY_LOSS_HALT_PCT      = float(os.getenv("DAILY_LOSS_HALT_PCT",        "2.5"))
MAX_CONCURRENT_TRADES    = int(os.getenv(  "MAX_CONCURRENT_TRADES",      "3"))
MAX_PORTFOLIO_HEAT_PCT   = float(os.getenv("MAX_PORTFOLIO_HEAT_PCT",     "1.5"))
MAX_CONSECUTIVE_LOSSES   = int(os.getenv(  "MAX_CONSECUTIVE_LOSSES",     "5"))
COOLDOWN_AFTER_LOSSES    = int(os.getenv(  "COOLDOWN_LOSSES",            "3"))
PROFIT_PROTECT_MIN_CONF  = float(os.getenv("PROFIT_PROTECT_CONF",        "80.0"))
PROFIT_PROTECT_SIZE_MULT = float(os.getenv("PROFIT_PROTECT_SIZE",        "0.60"))
PROFIT_PROTECT_DRAWDOWN  = float(os.getenv("PROFIT_PROTECT_DRAWDOWN",    "300"))
# Phase 2 risk controls
MAX_DAILY_TRADES         = int(os.getenv(  "MAX_DAILY_TRADES",           "30"))   # 2.4 — total trades per day
VOLATILITY_HALT_ATR_MULT = float(os.getenv("VOLATILITY_HALT_ATR_MULT",  "2.5"))  # 2.3 — halt when range > N×ATR
DRAWDOWN_THROTTLE_1_PCT  = float(os.getenv("DRAWDOWN_THROTTLE_1_PCT",   "0.5"))  # 2.6 — 50% size at 0.5% drawdown
DRAWDOWN_THROTTLE_2_PCT  = float(os.getenv("DRAWDOWN_THROTTLE_2_PCT",   "1.0"))  # 2.6 — 25% size at 1.0% drawdown

# Set IS_PAPER_TRADING=false in .env when ready to switch to live order execution.
# While True, consecutive-loss cooldowns and circuit breakers are disabled so
# paper trades run without interruption and generate maximum training data.
IS_PAPER_TRADING = os.getenv("IS_PAPER_TRADING", "true").lower() != "false"

# ── PRD session rules ─────────────────────────────────────────────────────────
# Hard session blocks that override all signals (non-configurable per PRD)
SESSION_RESTRICTED_UNTIL = "09:45"   # no new entries until 9:45 AM ET (price discovery)
SESSION_LUNCH_START      = "11:30"   # lunch block start (thin order book)
SESSION_LUNCH_END        = "13:30"   # lunch block end
SESSION_CLOSING_CAUTION  = "15:30"   # momentum-only after this (no new scalps)
SESSION_HARD_CLOSE       = "15:45"   # all positions must be flat by this time

# ── Trade time limits (scalp mode per PRD Section 6.3) ────────────────────────
SCALP_MAX_BARS        = 20   # 20-minute hard close for scalps (20 × 1-min bars)
INTRADAY_MAX_BARS     = 90   # 90-minute hard close for intraday positions
MIN_STOP_PCT          = 0.15  # min stop distance as % of entry (prevents noise triggers)

# Legacy aliases (kept for any code that still references them)
INTRADAY_INTERVAL = "5m"
REALTIME_INTERVAL = "1m"
