#!/usr/bin/env bash
# ── deploy/setup-host.sh ──────────────────────────────────────────────────────
# Run ONCE on the EC2 host before the first 'docker compose up'.
# Creates the host bind-mount directories and sets correct ownership.
#
# The docker-compose.yml mounts:
#   /opt/nasdaq-agent/nasdaq_agent/data  →  /app/data   (models, cache)
#   /opt/nasdaq-agent/tokens             →  /app/tokens  (Schwab OAuth)
#   /opt/nasdaq-agent/logs               →  /app/logs
#
# Usage:
#   chmod +x deploy/setup-host.sh
#   sudo ./deploy/setup-host.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BASE=/opt/nasdaq-agent

echo "Creating host bind-mount directories under ${BASE} ..."

mkdir -p \
    "${BASE}/nasdaq_agent/data/models" \
    "${BASE}/nasdaq_agent/data/cache" \
    "${BASE}/tokens" \
    "${BASE}/tokens/backup" \
    "${BASE}/logs"

# Containers run as UID 1000 (user 'nasdaq' created in Dockerfile).
# Host directories must be writable by that UID.
chown -R 1000:1000 "${BASE}/nasdaq_agent/data"
chown -R 1000:1000 "${BASE}/tokens"
chown -R 1000:1000 "${BASE}/logs"
chmod -R 755 "${BASE}/nasdaq_agent/data"
chmod 700 "${BASE}/tokens"

echo "Done. Directory layout:"
ls -la "${BASE}"
echo ""
ls -la "${BASE}/nasdaq_agent/data"

echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and fill in all values"
echo "  2. sudo bash nasdaq_agent/update.sh"
