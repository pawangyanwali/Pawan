"""Gunicorn configuration for NASDAQ Scalping Agent.

Placed in the app working directory so gunicorn picks it up automatically.
Override any value via CLI flag or GUNICORN_CMD_ARGS env var.
"""

# ── Binding ───────────────────────────────────────────────────────────────────
bind = "0.0.0.0:8000"

# ── Worker ────────────────────────────────────────────────────────────────────
# Uvicorn worker gives us async FastAPI support under gunicorn process management.
worker_class = "uvicorn.workers.UvicornWorker"

# Single worker: the scanner, ML cache, and WebSocket manager all live in-process.
# Multiple workers would each hold independent state and produce conflicting scans.
workers = 1

# ── Timeouts ──────────────────────────────────────────────────────────────────
# ML retraining trains 100 Tier-1 tickers × 4 model types in a background thread.
# With 429 rate-limit back-off this can take 8-12 minutes.  The default 30 s
# timeout kills the worker mid-train and causes a cold restart that wipes all
# in-memory cache — making data disappear from the dashboard.
# 1800 s = 30 minutes gives full headroom for the worst-case retrain cycle.
timeout = 1800

# Graceful shutdown: let in-flight requests finish before killing the worker.
graceful_timeout = 120

# HTTP keep-alive for REST clients (nginx default keep-alive is 65 s).
keepalive = 65

# ── Logging ───────────────────────────────────────────────────────────────────
# Log to stdout/stderr so systemd/journald captures everything.
accesslog = "-"
errorlog  = "-"
loglevel  = "info"

# Include timestamp, PID, and level in error log.
logconfig_dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "generic": {
            "format": "%(asctime)s [%(process)d] [%(levelname)s] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "generic",
            "stream": "ext://sys.stderr",
        }
    },
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "gunicorn.error":  {"propagate": True},
        "gunicorn.access": {"propagate": True},
    },
}
