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

# ── Ticker universe — top 50 liquid NASDAQ stocks ────────────────────────────
# With 377 credits/min we can afford 50 tickers:
#   50 tickers / 20 per batch = 3 calls × 0.20s = 0.60s per scan
NASDAQ_TICKERS = [
    # ── Mega-cap tech (Tier 1 — always HIGH AH liquidity) ──────────────────────
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "TSLA", "AVGO", "NFLX", "AMD",

    # ── Large-cap tech + semis ────────────────────────────────────────────────
    "ADBE", "QCOM", "CSCO", "INTU", "AMAT",
    "MU",   "PANW", "CRWD", "MRVL", "KLAC",
    "LRCX", "ADI",  "SNPS", "CDNS", "ISRG",
    "REGN", "BKNG", "ADP",  "SBUX", "INTC",

    # ── Cloud / SaaS / fintech / crypto ──────────────────────────────────────
    "COST", "AMGN", "ABNB", "DDOG", "ZS",
    "WDAY", "MELI", "ARM",  "APP",  "COIN",
    "TTD",  "HOOD", "SMCI", "TEAM", "OKTA",
    "FTNT", "PYPL", "CEG",  "AXON", "SNOW",

    # ── Expansion Tier A — high-volume momentum stocks ────────────────────────
    "PLTR", "MSTR", "MARA", "RIVN", "LCID",   # AI/crypto/EV — volatile, heavy retail
    "SOUN", "IONQ", "CELH", "ENPH", "LULU",   # AI voice, quantum, energy, solar, retail
    "NET",  "ANET", "MDB",  "SNAP", "RBLX",   # cloud infra, social, gaming

]
# Expansion Tier B removed — 65 tickers keeps memory under ~800 MB on small instances.

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
    "ANSS":"ANSYS","TEAM":"Atlassian","DDOG":"Datadog","CEG":"Constellation Energy",
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
    # Expansion tickers
    "PLTR":"Palantir","MARA":"Marathon Digital","RIVN":"Rivian","LCID":"Lucid Motors",
    "SOUN":"SoundHound AI","IONQ":"IonQ","NET":"Cloudflare","ANET":"Arista Networks",
    "MDB":"MongoDB","SNAP":"Snap","RBLX":"Roblox","UPST":"Upstart","AFRM":"Affirm",
    "CVNA":"Carvana","LYFT":"Lyft","ROKU":"Roku","PINS":"Pinterest","ZM":"Zoom",
    "DOCU":"DocuSign","BILL":"Bill.com","TWLO":"Twilio","CHWY":"Chewy",
    "GTLB":"GitLab","HUBS":"HubSpot","DKNG":"DraftKings",
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

# ── Position sizing defaults ──────────────────────────────────────────────────
DEFAULT_ACCOUNT_SIZE = 10_000   # default $ account size shown in UI calculator
DEFAULT_RISK_PCT     = 1.0      # % of account risked per trade
MAX_POSITION_PCT     = 5.0      # never allocate more than this % to one trade

# Legacy aliases (kept for any code that still references them)
INTRADAY_INTERVAL = "5m"
REALTIME_INTERVAL = "1m"
