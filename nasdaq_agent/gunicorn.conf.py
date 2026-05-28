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
# Write to /app/logs/ which is bind-mounted to /opt/nasdaq-agent/logs on the host.
# Stdout ("-") is kept as a fallback for 'docker compose logs'.
import os as _os
import logging as _logging
import re as _re

_log_dir = _os.getenv("LOG_DIR", "/app/logs")
_os.makedirs(_log_dir, exist_ok=True)
accesslog = f"{_log_dir}/web-api-access.log"
errorlog  = f"{_log_dir}/web-api.log"
loglevel  = "info"


_BACKUP_PROBE_RE = _re.compile(
    r'"GET\s+/[^ ?"]+\.(?:zip|tar\.gz|tgz|tar|tar\.bz2|tar\.xz|7z|rar|gz|bz2|zst|sql(?:\.gz|\.bz2)?)\s+HTTP/1\.[01]"\s+404\b'
)
_HEALTH_OK_RE = _re.compile(r'"GET\s+/api/health\s+HTTP/1\.[01]"\s+200\b')


def _is_noise_probe_message(message: str) -> bool:
    """True for high-volume access-log noise that should not clutter web-api.log."""
    if _BACKUP_PROBE_RE.search(message):
        return True
    if _os.getenv("NASDAQ_SUPPRESS_HEALTH_ACCESS_LOGS", "1") != "0":
        return bool(_HEALTH_OK_RE.search(message))
    return False


class _AccessNoiseFilter(_logging.Filter):
    def filter(self, record: _logging.LogRecord) -> bool:
        return not _is_noise_probe_message(record.getMessage())


def _install_access_noise_filter() -> None:
    access_logger = _logging.getLogger("gunicorn.access")
    if any(isinstance(f, _AccessNoiseFilter) for f in access_logger.filters):
        return
    access_logger.addFilter(_AccessNoiseFilter())


def post_worker_init(worker):
    """Install access-log noise filtering in each Gunicorn worker process."""
    _install_access_noise_filter()
