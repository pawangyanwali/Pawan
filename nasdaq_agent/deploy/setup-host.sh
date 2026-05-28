#!/usr/bin/env bash
# ── deploy/setup-host.sh ──────────────────────────────────────────────────────
# Run ONCE on the EC2 host before the first 'docker compose up'.
# Creates EBS bind-mount directories and sets correct ownership.
#
# Usage:
#   chmod +x deploy/setup-host.sh
#   sudo ./deploy/setup-host.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BASE=/opt/nasdaq-agent

echo "Creating EBS bind-mount directories under ${BASE} ..."

# These directories are bind-mounted into containers as named volumes.
# They survive image rebuilds and container restarts.
mkdir -p \
    "${BASE}/models" \
    "${BASE}/tokens" \
    "${BASE}/tokens/backup" \
    "${BASE}/cache" \
    "${BASE}/logs"

# Containers run as UID 1000 (user 'nasdaq' created in Dockerfile).
# Host directories must be writable by that UID.
chown -R 1000:1000 "${BASE}"
chmod -R 755 "${BASE}"

echo "Done. Directory layout:"
ls -la "${BASE}"

echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and fill in all values"
echo "  2. docker build -t nasdaq-agent:latest ."
echo "  3. docker compose up -d"
echo "  4. docker compose ps        # verify all containers healthy"
echo "  5. docker compose logs -f   # tail all logs"
