SCAN_INTERVAL_SECONDS = 60        # How often to rescan all tickers
ML_RETRAIN_INTERVAL = 3600        # Retrain ML model every hour
DATA_PERIOD_DAYS = 59             # Days of historical data for ML training
INTRADAY_INTERVAL = "5m"          # Candle resolution for ML features
REALTIME_INTERVAL = "1m"          # Candle resolution for live signals

# Signal weights (must sum to 1.0)
WEIGHT_TECHNICAL = 0.35
WEIGHT_VOLUME    = 0.20
WEIGHT_ML        = 0.35
WEIGHT_SENTIMENT = 0.10

# Thresholds for dashboard coloring
STRONG_BUY_THRESHOLD  =  0.60
BUY_THRESHOLD         =  0.30
SELL_THRESHOLD        = -0.30
STRONG_SELL_THRESHOLD = -0.60

# Top 100 liquid NASDAQ tickers (NASDAQ-100 + high-volume extras)
NASDAQ_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "COST",
    "NFLX", "AMD", "ADBE", "QCOM", "PEP", "CSCO", "INTU", "CMCSA", "AMAT", "AMGN",
    "HON", "MU", "ISRG", "LRCX", "KLAC", "ADI", "REGN", "SNPS", "CDNS", "MDLZ",
    "PYPL", "GILD", "CTAS", "MRVL", "PANW", "ABNB", "CRWD", "ORLY", "ADSK", "DXCM",
    "ADP", "MNST", "BKNG", "FTNT", "KDP", "BIIB", "MCHP", "SBUX", "AEP", "NXPI",
    "PCAR", "CPRT", "EXC", "ROST", "ODFL", "PAYX", "WDAY", "FAST", "IDXX", "ZS",
    "ANSS", "TEAM", "DDOG", "CEG", "TTD", "EA", "MRNA", "ALGN", "DLTR", "VRSK",
    "GEHC", "ON", "ROP", "ILMN", "TTWO", "OKTA", "EXPE", "SWKS", "CTSH", "VRSN",
    "NTAP", "HOLX", "WBA", "SIRI", "ZM", "DOCU", "PTON", "LCID", "RIVN", "CHKP",
    "INTC", "ARM", "SMCI", "MELI", "APP", "COIN", "HOOD", "RKLB", "SOFI", "PLTR",
]
