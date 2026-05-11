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
SCAN_INTERVAL_SECONDS  = 60       # 1 min — 50 tickers / 20 per batch = 3 calls ≈ 0.6s scan
ML_RETRAIN_INTERVAL    = 21600    # Retrain ML every 6 hours (was 24h on free tier)

# ── Historical data windows ───────────────────────────────────────────────────
DATA_PERIOD_DAYS    = 180    # 6 months of 5-min data for intraday ML (~14,040 bars; cap 5000)
REALTIME_OUTPUTSIZE = 500   # 1-min bars per scan (≈8h, gives 5M/15M/30M/1H resampling coverage)

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
    # Mega-cap tech (unchanged core)
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "TSLA", "AVGO", "NFLX", "AMD",
    # Large-cap tech + semi
    "ADBE", "QCOM", "CSCO", "INTU", "AMAT",
    "MU",   "PANW", "CRWD", "MRVL", "KLAC",
    # Semi / EDA / biotech
    "LRCX", "ADI",  "SNPS", "CDNS", "ISRG",
    "REGN", "BKNG", "ADP",  "SBUX", "INTC",
    # NEW — available on Grow plan
    "COST", "AMGN", "ABNB", "DDOG", "ZS",
    "WDAY", "MELI", "ARM",  "APP",  "COIN",
    "TTD",  "HOOD", "SMCI", "TEAM", "OKTA",
    "FTNT", "PYPL", "CEG",  "AXON", "SNOW",
]

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
}

# Legacy aliases (kept for any code that still references them)
INTRADAY_INTERVAL = "5m"
REALTIME_INTERVAL = "1m"
