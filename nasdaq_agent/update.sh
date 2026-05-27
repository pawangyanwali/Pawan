#!/usr/bin/env bash
# update.sh — pull latest code and restart all containers
# Usage: sudo bash /opt/nasdaq-agent/nasdaq_agent/update.sh

set -euo pipefail

APP_DIR="/opt/nasdaq-agent"
BRANCH="claude/nasdaq-stock-prediction-agent-kDe60"

echo "[1/5] Pulling latest code from $BRANCH ..."
git -C "$APP_DIR" fetch origin
git -C "$APP_DIR" reset --hard "origin/$BRANCH"

echo "[2/5] Ensuring data directory exists on host ..."
# This is the bind-mount source for /app/data in every container.
# Token files placed here are immediately visible to all containers.
mkdir -p "$APP_DIR/nasdaq_agent/data/models"
mkdir -p "$APP_DIR/nasdaq_agent/data/cache"
mkdir -p "$APP_DIR/logs"

echo "[3/5] Building image ..."
docker compose -f "$APP_DIR/docker-compose.yml" build

echo "[4/5] Restarting containers ..."
docker compose -f "$APP_DIR/docker-compose.yml" up -d

echo "[5/5] Checking container health ..."
sleep 6
docker compose -f "$APP_DIR/docker-compose.yml" ps --format "table {{.Name}}\t{{.Status}}"

echo ""
echo "Token files expected at: $APP_DIR/nasdaq_agent/data/schwab_tokens.json"
echo "                         $APP_DIR/nasdaq_agent/data/schwab_md_tokens.json"
if [[ -f "$APP_DIR/nasdaq_agent/data/schwab_tokens.json" ]]; then
    echo "✓ schwab_tokens.json present"
else
    echo "✗ schwab_tokens.json MISSING — visit /schwab/auth or copy from backup"
fi
if [[ -f "$APP_DIR/nasdaq_agent/data/schwab_md_tokens.json" ]]; then
    echo "✓ schwab_md_tokens.json present"
else
    echo "✗ schwab_md_tokens.json MISSING — visit /schwab/auth/md or copy from backup"
fi
