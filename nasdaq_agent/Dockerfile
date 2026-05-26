# ── Build-time dependencies ───────────────────────────────────────────────────
# Use slim base; install only what's needed at runtime.
FROM python:3.11-slim AS base

WORKDIR /app

# System libraries required by psycopg2-binary, Pillow, and cryptography.
# libpq-dev is needed at build time for psycopg2-binary wheel compilation.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps before copying app code so Docker can cache this layer
# independently of code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU-only before the main requirements to avoid pulling the
# 2 GB CUDA build that pip selects by default on Linux.  The EC2 t3.large has
# no GPU so CPU-only (~180 MB) is sufficient for the BiLSTM regime detector.
RUN pip install --no-cache-dir \
    "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# Runtime directories that are written to at startup; create them here so the
# non-root user can write to them without host-volume ownership issues.
RUN mkdir -p /app/data/models /app/data/cache /app/logs

# Non-root user — reduces blast radius if a dependency has a vuln.
RUN useradd -m -u 1000 -s /bin/bash nasdaq \
 && chown -R nasdaq:nasdaq /app
USER nasdaq

# Health check hits the FastAPI health endpoint (web-api service only).
# Other services expose no HTTP; they are checked via the Valkey heartbeat key.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/api/health || exit 1

EXPOSE 8000

# Default CMD runs the web-api.  Services override this in docker-compose.yml.
CMD ["gunicorn", "-c", "gunicorn.conf.py", "main:app"]
