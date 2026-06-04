#!/usr/bin/env bash
# update.sh — pull latest code and restart all containers
# Usage: sudo bash /opt/nasdaq-agent/nasdaq_agent/update.sh

set -euo pipefail

APP_DIR="/opt/nasdaq-agent"
DATA_DIR="$APP_DIR/nasdaq_agent/data"
BRANCH="claude/nasdaq-stock-prediction-agent-kDe60"

echo "[1/6] Pulling latest code from $BRANCH ..."
git -C "$APP_DIR" fetch origin
git -C "$APP_DIR" reset --hard "origin/$BRANCH"

echo "[2/6] Ensuring host bind-mount directories exist ..."
mkdir -p "$DATA_DIR/models"
mkdir -p "$DATA_DIR/cache"
mkdir -p "$APP_DIR/tokens"
mkdir -p "$APP_DIR/logs"
chown -R 1000:1000 "$DATA_DIR" 2>/dev/null || true
chown -R 1000:1000 "$APP_DIR/tokens" 2>/dev/null || true
chmod 700 "$APP_DIR/tokens" 2>/dev/null || true
find "$APP_DIR/tokens" -maxdepth 1 -type f -name 'schwab*_tokens.json' \
    -exec chmod 600 {} \; 2>/dev/null || true

echo "[3/6] Migrating any legacy Docker named volumes to host bind-mount paths ..."
# If a previous deploy used named volumes (nasdaq-agent_models, nasdaq-agent_cache),
# copy their contents to the host bind-mount path before recreating containers.
# This runs safely on every update — cp -u skips files already present on host.
declare -A _VOL_MAP=(
    ["nasdaq-agent_models"]="$DATA_DIR/models"
    ["nasdaq-agent_cache"]="$DATA_DIR/cache"
)
_MIGRATED=0
for _VOL in "${!_VOL_MAP[@]}"; do
    _SRC="/var/lib/docker/volumes/${_VOL}/_data"
    _DST="${_VOL_MAP[$_VOL]}"
    if [[ -d "$_SRC" ]] && [[ -n "$(ls -A "$_SRC" 2>/dev/null)" ]]; then
        echo "  Migrating named volume $_VOL → $_DST ..."
        mkdir -p "$_DST"
        cp -ru "$_SRC/." "$_DST/"
        chown -R 1000:1000 "$_DST" 2>/dev/null || true
        _COUNT=$(ls "$_DST" | wc -l)
        echo "  ✓ $_COUNT files now in $_DST"
        _MIGRATED=1
    fi
done
if [[ $_MIGRATED -eq 0 ]]; then
    echo "  No legacy named volumes found — nothing to migrate."
fi

echo "[4/6] Building image ..."
docker compose -f "$APP_DIR/docker-compose.yml" build

echo "[5/6] Recreating containers (down → up ensures bind-mount config applies) ..."
docker compose -f "$APP_DIR/docker-compose.yml" down
docker compose -f "$APP_DIR/docker-compose.yml" up -d

# Remove old named volumes now that containers are running with bind mounts.
# Safe to ignore errors if volumes are in use or don't exist.
for _VOL in nasdaq-agent_models nasdaq-agent_cache nasdaq-agent_logs; do
    if docker volume ls -q 2>/dev/null | grep -q "^${_VOL}$"; then
        docker volume rm "$_VOL" 2>/dev/null && echo "  Removed legacy named volume: $_VOL" || true
    fi
done

echo "[6/6] Checking container health ..."
sleep 6
docker compose -f "$APP_DIR/docker-compose.yml" ps --format "table {{.Name}}\t{{.Status}}"

echo ""
echo "Model files: $DATA_DIR/models/"
echo "  $(ls "$DATA_DIR/models" 2>/dev/null | wc -l) .joblib files on host"
echo ""
echo "Token files expected at: $APP_DIR/tokens/schwab_tokens.json"
echo "                         $APP_DIR/tokens/schwab_md_tokens.json"
if [[ -f "$APP_DIR/tokens/schwab_tokens.json" ]]; then
    echo "✓ schwab_tokens.json present"
else
    echo "✗ schwab_tokens.json MISSING — visit /schwab/auth or copy from backup"
fi
if [[ -f "$APP_DIR/tokens/schwab_md_tokens.json" ]]; then
    echo "✓ schwab_md_tokens.json present"
else
    echo "✗ schwab_md_tokens.json MISSING — visit /schwab/auth/md or copy from backup"
fi
