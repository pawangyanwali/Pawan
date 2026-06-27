#!/usr/bin/env bash
# restore_schwab_tokens.sh
#
# Verifies Schwab token JSON files in the host bind-mounted token directory,
# fixes ownership for container writes, and restarts token-consuming services.
#
# Usage (run from /opt/nasdaq-agent):
#   sudo bash nasdaq_agent/scripts/restore_schwab_tokens.sh
#
# Token source (host):
#   /opt/nasdaq-agent/tokens/schwab_tokens.json    (trading A+T)
#   /opt/nasdaq-agent/tokens/schwab_md_tokens.json (market-data MD)
#
# Token destination (container):
#   /app/tokens/ bind-mounted from /opt/nasdaq-agent/tokens

set -euo pipefail

HOST_TOKEN_DIR="/opt/nasdaq-agent/tokens"
COMPOSE_FILE="/opt/nasdaq-agent/nasdaq_agent/docker-compose.yml"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Verify token files exist on host ──────────────────────────────────────────
TRADING_TOKEN="$HOST_TOKEN_DIR/schwab_tokens.json"
MD_TOKEN="$HOST_TOKEN_DIR/schwab_md_tokens.json"

[[ -f "$TRADING_TOKEN" ]] || error "schwab_tokens.json not found at $TRADING_TOKEN"
[[ -f "$MD_TOKEN"      ]] || error "schwab_md_tokens.json not found at $MD_TOKEN"

info "Found schwab_tokens.json    ($(wc -c < "$TRADING_TOKEN") bytes)"
info "Found schwab_md_tokens.json ($(wc -c < "$MD_TOKEN") bytes)"

# ── Fix ownership for container writes ────────────────────────────────────────
info "Ensuring token files are writable by container UID 1000 ..."
chown 1000:1000 "$TRADING_TOKEN" "$MD_TOKEN"
chmod 600 "$TRADING_TOKEN" "$MD_TOKEN"
info "Ownership fixed."

# ── Restart services that consume the tokens ──────────────────────────────────
info "Restarting token-service (sole token owner) ..."
docker compose -f "$COMPOSE_FILE" --project-directory /opt/nasdaq-agent restart token-service

info "Restarting market-data after token owner is available ..."
docker compose -f "$COMPOSE_FILE" --project-directory /opt/nasdaq-agent restart market-data

# ── Verify ────────────────────────────────────────────────────────────────────
info "Waiting 8s for services to initialise ..."
sleep 8

echo ""
info "market-data logs (last 20 lines):"
docker compose -f "$COMPOSE_FILE" --project-directory /opt/nasdaq-agent logs market-data --since 15s 2>&1 | grep -E "MD poller|streamer|token|WARN|ERROR|started" || true

echo ""
info "token-service logs (last 10 lines):"
docker compose -f "$COMPOSE_FILE" --project-directory /opt/nasdaq-agent logs token-service --since 15s 2>&1 | grep -E "Schwab|token|WARN|ERROR" | head -10 || true

echo ""
info "Done. Check above for 'MD poller started' — if still missing, re-auth via /schwab/auth/md"
