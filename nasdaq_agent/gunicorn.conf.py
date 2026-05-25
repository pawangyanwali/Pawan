"""Gunicorn configuration for NASDAQ Scalping Agent."""

# ── Binding ───────────────────────────────────────────────────────────────────
bind = "0.0.0.0:8000"

# ── Worker ────────────────────────────────────────────────────────────────────
# Uvicorn worker for async FastAPI.
worker_class = "uvicorn.workers.UvicornWorker"

# Single worker: scanner, ML cache, and WebSocket manager all live in-process.
workers = 1

# ── Timeouts ──────────────────────────────────────────────────────────────────
# ML retraining NOW runs in a subprocess (agent/_ml_retrain_worker.py) so the
# worker heartbeat is never blocked by GIL-holding training threads.
# 1800 s is belt-and-suspenders for any other slow operation.
timeout = 1800
graceful_timeout = 120
keepalive = 65

# ── Logging ───────────────────────────────────────────────────────────────────
accesslog = "-"
errorlog  = "-"
loglevel  = "info"
