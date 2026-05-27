#!/usr/bin/env bash
# restore_schwab_tokens.sh
#
# Copies Schwab token JSON files from the host into the running containers and
# restarts the services that consume them.
#
# Usage (run from /opt/nasdaq-agent):
#   sudo bash nasdaq_agent/scripts/restore_schwab_tokens.sh
#
# Token source (host):
#   /opt/nasdaq-agent/nasdaq_agent/data/schwab_tokens.json    (trading A+T)
#   /opt/nasdaq-agent/nasdaq_agent/data/schwab_md_tokens.json (market-data MD)
#
# Token destination (container):
#   /app/data/ inside the nasdaq-data shared Docker volume

set -euo pipefail

HOST_DATA_DIR="/opt/nasdaq-agent/nasdaq_agent/data"
CONTAINER_DATA_DIR="/app/data"
COMPOSE_FILE="/opt/nasdaq-agent/docker-compose.yml"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Resolve the target container (any running service with the shared volume) ──
# Prefer web-api as it is almost always running; fall back to scanner.
_pick_container() {
    local svc
    for svc in web-api scanner market-data; do
        local name
        name=$(docker compose -f "$COMPOSE_FILE" ps -q "$svc" 2>/dev/null | head -1)
        if [[ -n "$name" ]]; then
            # Verify it is actually running
            local state
            state=$(docker inspect --format '{{.State.Status}}' "$name" 2>/dev/null)
            if [[ "$state" == "running" ]]; then
                echo "$name"
                return 0
            fi
        fi
    done
    return 1
}

# ── Verify token files exist on host ──────────────────────────────────────────
TRADING_TOKEN="$HOST_DATA_DIR/schwab_tokens.json"
MD_TOKEN="$HOST_DATA_DIR/schwab_md_tokens.json"

[[ -f "$TRADING_TOKEN" ]] || error "schwab_tokens.json not found at $TRADING_TOKEN"
[[ -f "$MD_TOKEN"      ]] || error "schwab_md_tokens.json not found at $MD_TOKEN"

info "Found schwab_tokens.json    ($(wc -c < "$TRADING_TOKEN") bytes)"
info "Found schwab_md_tokens.json ($(wc -c < "$MD_TOKEN") bytes)"

# ── Pick a running container ───────────────────────────────────────────────────
CONTAINER=$(_pick_container) || error "No running container found. Is docker compose up?"
info "Using container: $CONTAINER"

# ── Copy token files into the shared volume ───────────────────────────────────
info "Copying tokens into $CONTAINER:$CONTAINER_DATA_DIR ..."
docker cp "$TRADING_TOKEN" "$CONTAINER:$CONTAINER_DATA_DIR/schwab_tokens.json"
docker cp "$MD_TOKEN"      "$CONTAINER:$CONTAINER_DATA_DIR/schwab_md_tokens.json"
info "Copy complete."

# ── Restart services that consume the tokens ──────────────────────────────────
info "Restarting market-data (reads schwab_md_tokens.json) ..."
docker compose -f "$COMPOSE_FILE" restart market-data

info "Restarting scanner (reads schwab_tokens.json for order placement) ..."
docker compose -f "$COMPOSE_FILE" restart scanner

# ── Verify ────────────────────────────────────────────────────────────────────
info "Waiting 8s for services to initialise ..."
sleep 8

echo ""
info "market-data logs (last 20 lines):"
docker compose -f "$COMPOSE_FILE" logs market-data --since 15s 2>&1 | grep -E "MD poller|streamer|token|WARN|ERROR|started" || true

echo ""
info "scanner logs (last 10 lines):"
docker compose -f "$COMPOSE_FILE" logs scanner --since 15s 2>&1 | grep -E "Schwab|token|WARN|ERROR|scan" | head -10 || true

echo ""
info "Done. Check above for 'MD poller started' — if still missing, re-auth via /schwab/auth/md"
