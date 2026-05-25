"""
Standalone ML retrain subprocess.

Launched by scanner via subprocess.Popen so CPU-intensive sklearn/XGBoost
training runs in a separate Python interpreter.  This keeps the gunicorn
worker's asyncio event loop (and heartbeat) completely free during training,
eliminating the WORKER TIMEOUT that previously wiped all in-process cache.

Usage (internal):
    subprocess.Popen([sys.executable, __file__], cwd=<nasdaq_agent dir>)
"""
import logging
import sys
from pathlib import Path

# Ensure the package root (nasdaq_agent/) is on the path.
_PKG = Path(__file__).parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ml_retrain")

if __name__ == "__main__":
    try:
        from config import TRAINING_TICKERS, CACHE_TTL_1D
        from agent.data_fetcher import fetch_batch_interval
        from agent.ml_model import retrain_all

        logger.info(f"[ML-Retrain] Starting ({len(TRAINING_TICKERS)} Tier-1 tickers)…")
        daily_data = fetch_batch_interval(TRAINING_TICKERS, "1day", 500, ttl=CACHE_TTL_1D)
        retrain_all(TRAINING_TICKERS, daily_data=daily_data)
        logger.info("[ML-Retrain] Complete.")
    except Exception as exc:
        logger.error(f"[ML-Retrain] Failed: {exc}", exc_info=True)
        sys.exit(1)
