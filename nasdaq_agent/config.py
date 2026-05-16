import os
from dotenv import load_dotenv

load_dotenv()

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()

if not TWELVE_DATA_API_KEY:
    raise RuntimeError(
        "\n\n  ERROR: TWELVE_DATA_API_KEY not set.\n"
        "  Create a file called .env in the nasdaq_agent folder with:\n"
        "  TWELVE_DATA_API_KEY=your_key_here\n"
    )

# ── Schwab integration gate ───────────────────────────────────────────────────
# Set SCHWAB_ENABLED=true in .env when both apps are authorised and working.
# When false (default) all Schwab code is skipped — agent runs on Twelve Data only.
SCHWAB_ENABLED = os.getenv("SCHWAB_ENABLED", "false").lower() == "true"

# ── Grow-377 plan limits ───────────────────────────────────────────────────────
# 377 credits/minute, unlimited daily credits
# 1 credit = 1 symbol in any /time_series request
# Batch up to 20 symbols per request → 20 credits per call

CALL_GAP    = 0.20   # seconds between API calls  (60 / 377 ≈ 0.16s, use 0.20s for safety)
BATCH_SIZE  = 20     # symbols per request (each symbol = 1 credit; Twelve Data supports up to 55)

# ── Scan / retrain cadence ────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS   = 60      # 1 min — 50 tickers / 20 per batch = 3 calls ≈ 0.6s scan
ML_RETRAIN_INTERVAL     = 21600   # Full XGBoost retrain every 6 hours
DEEP_FINETUNE_INTERVAL  = 3600    # Deep BiLSTM fine-tune every 1 hour (uses cached data, no API cost)

# ── Historical data windows ───────────────────────────────────────────────────
DATA_PERIOD_DAYS    = 180    # 6 months of 5-min data for intraday ML (~14,040 bars; cap 5000)
REALTIME_OUTPUTSIZE = 300   # 1-min bars per scan (≈5h coverage, reduced for memory)

# ── Per-interval cache TTLs (seconds) ────────────────────────────────────────
CACHE_TTL_5M  =   300   # 5 minutes
CACHE_TTL_1H  =  3600   # 1 hour
CACHE_TTL_1D  = 86400   # 24 hours  (also used as DAILY_CACHE_TTL for compatibility)
DAILY_CACHE_TTL = CACHE_TTL_1D

# ── Prediction thresholds (dashboard colouring) ───────────────────────────────
STRONG_BUY_THRESHOLD  =  0.60
BUY_THRESHOLD         =  0.30
SELL_THRESHOLD        = -0.30
STRONG_SELL_THRESHOLD = -0.60

# ── Ticker universe — ~165 liquid NASDAQ stocks in 3 ML clusters ──────────────

# Cluster A — Mega-cap liquid (tight spreads, highest institutional activity, index-correlated)
CLUSTER_A_TICKERS = [
    # Core mega-cap
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","NFLX","AMD",
    # Large-cap semis / enterprise
    "ADBE","QCOM","CSCO","INTU","AMAT","MU","PANW","CRWD","MRVL","KLAC",
    "LRCX","ADI","SNPS","CDNS","ISRG","REGN","BKNG","ADP","SBUX","INTC",
    # NASDAQ-100 completions
    "COST","AMGN","WDAY","MELI","ARM","TTD","SMCI","TEAM","FTNT","PYPL",
    "AXON","CEG","ABNB","DDOG","ZS","NET","ANET","OKTA","SNOW","APP",
]

# Cluster B — Growth/SaaS/Fintech (medium liquidity, earnings-driven, sector-correlated)
CLUSTER_B_TICKERS = [
    # SaaS / cloud / fintech
    "COIN","HOOD","PLTR","LULU","ENPH","CELH","SOUN","IONQ","GTLB","HUBS",
    "DKNG","BILL","DOCU","MPWR","TWLO","CHWY","MSTR","MDB","SNAP","RBLX",
    # NASDAQ-100 mid-cap
    "ODFL","PAYX","FAST","IDXX","VRSK","EA","DLTR","ALGN","ILMN",
    "TTWO","EXPE","CTSH","NTAP","SWKS","VRSN","CHKP","GEHC","ON","ROP",
    # More established
    "BIIB","GILD","MNST","PCAR","CPRT","ROST","MCHP","NXPI","CMCSA","PEP",
    "HON","CTAS","ORLY","ADSK","DXCM","LOGI","NDAQ","MTCH","CYBR","SOFI",
]

# Cluster C — High-volatility/Momentum (retail-driven, high-beta, wide spreads, meme/crypto/biotech adjacent)
CLUSTER_C_TICKERS = [
    # EV / space / quantum / deep-tech momentum
    "MARA","RIVN","LCID","RGTI","QUBT","RKLB","ASTS","WOLF","FSLR","RUN",
    "ARRY","LI","BIDU","NTES","JD","PDD","BILI","NVAX","MRNA","BNTX",
    # Biotech
    "VRTX","ALNY","BMRN","FATE","ACAD","CRSP","BEAM","EDIT","RXRX",
    "ABCL",
    # High-vol fintech / consumer / other
    "UPST","AFRM","CVNA","LYFT","ROKU","PINS","ZM","PTON","OPEN","CART",
    "TENB","VRNS","QLYS",
]

# Deduplicated master list preserving cluster order (A first, then B additions, then C additions)
NASDAQ_TICKERS: list[str] = list(dict.fromkeys(
    CLUSTER_A_TICKERS + CLUSTER_B_TICKERS + CLUSTER_C_TICKERS
))

# Cluster assignment map
TICKER_CLUSTER: dict[str, str] = {}
for t in CLUSTER_A_TICKERS: TICKER_CLUSTER[t] = "A"
for t in CLUSTER_B_TICKERS: TICKER_CLUSTER[t] = "B"
for t in CLUSTER_C_TICKERS: TICKER_CLUSTER[t] = "C"

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
PAPER_TRADE_MIN_CONFIDENCE = 65.0   # min confidence to auto-paper-trade

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

# Active ticker universe = base NASDAQ_TICKERS + any user-added watchlist items
def get_active_tickers() -> list:
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
