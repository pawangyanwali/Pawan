#!/usr/bin/env bash
# Manual update script — run on the server to pull latest and restart
# Usage: sudo bash /opt/nasdaq-agent/nasdaq_agent/update.sh

set -euo pipefail
APP_DIR="/opt/nasdaq-agent"
VENV="$APP_DIR/venv"
SERVICE="nasdaq-agent"

echo "[1/3] Pulling latest code..."
git -C "$APP_DIR" fetch origin
git -C "$APP_DIR" reset --hard origin/claude/nasdaq-stock-prediction-agent-kDe60

echo "[2/3] Installing dependencies..."
"$VENV/bin/pip" install -q -r "$APP_DIR/nasdaq_agent/requirements.txt"

echo "[3/3] Restarting service..."
systemctl restart "$SERVICE"
sleep 4
systemctl is-active --quiet "$SERVICE" && echo "✓ Running OK" || { echo "✗ Failed to start"; journalctl -u "$SERVICE" -n 20 --no-pager; exit 1; }

echo ""
echo "Dashboard: http://$(curl -sf http://checkip.amazonaws.com/ 2>/dev/null || echo 'YOUR_SERVER_IP')"
