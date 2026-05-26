"""Gunicorn configuration for NASDAQ Scalping Agent."""

# ── Binding ───────────────────────────────────────────────────────────────────
bind = "0.0.0.0:8000"

# ── Worker ────────────────────────────────────────────────────────────────────
# Uvicorn worker for async FastAPI.
worker_class = "uvicorn.workers.UvicornWorker"

# Single worker: scanner, ML cache, and WebSocket manager all live in-process.
workers = 1

# ── Timeouts ──────────────────────────────────────────────────────────────────
# ML retraining runs in daemon threads (not subprocess) so in-memory models
# stay current.  XGBoost releases the GIL during C-level training so the
# event loop heartbeat is not blocked.  1800 s guards any other slow operation.
timeout = 1800
graceful_timeout = 120
keepalive = 65

# ── Logging ───────────────────────────────────────────────────────────────────
accesslog = "-"
errorlog  = "-"
loglevel  = "info"
