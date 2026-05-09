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

SCAN_INTERVAL_SECONDS = 300       # 5 min — 30 tickers × 8.5s = ~4.25 min per scan
ML_RETRAIN_INTERVAL   = 86400     # Retrain ML once per day (saves API credits)
DATA_PERIOD_DAYS      = 30        # Days of 5-min history for ML (30d × 78 bars = 2340 bars)
INTRADAY_INTERVAL     = "5m"      # Candle resolution for ML features
REALTIME_INTERVAL     = "1m"      # Candle resolution for live signals

# Signal weights (must sum to 1.0)
WEIGHT_TECHNICAL = 0.35
WEIGHT_VOLUME    = 0.20
WEIGHT_ML        = 0.35
WEIGHT_SENTIMENT = 0.10

# Thresholds for dashboard colouring
STRONG_BUY_THRESHOLD  =  0.60
BUY_THRESHOLD         =  0.30
SELL_THRESHOLD        = -0.30
STRONG_SELL_THRESHOLD = -0.60

# Top 30 most liquid NASDAQ tickers — ideal for scalping on free API tier
# (30 tickers × 8.5s/request = ~4.25 min scan, fits inside 5-min interval)
NASDAQ_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "TSLA", "AVGO", "NFLX", "AMD",
    "ADBE", "QCOM", "CSCO", "INTU", "AMAT",
    "MU",   "PANW", "CRWD", "MRVL", "KLAC",
    "LRCX", "ADI",  "SNPS", "CDNS", "ISRG",
    "REGN", "BKNG", "ADP",  "SBUX", "INTC",
]

# Friendly names for the tickers (avoids extra API calls for company info)
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
}
