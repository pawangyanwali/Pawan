#!/usr/bin/env bash
# =============================================================================
# NASDAQ Scalping Agent — AWS / Ubuntu 22.04 deploy script
# Run this ONCE on a fresh Ubuntu 22.04 instance as root or sudo user.
#
# Usage:
#   ssh ubuntu@YOUR_SERVER_IP
#   curl -O https://raw.githubusercontent.com/pawangyanwali/Pawan/claude/nasdaq-stock-prediction-agent-kDe60/nasdaq_agent/deploy.sh
#   chmod +x deploy.sh && sudo bash deploy.sh
#
# To enable HTTPS after pointing your domain to this server:
#   sudo bash deploy.sh --domain scalpingstocksai.com
# =============================================================================

set -euo pipefail

APP_DIR="/opt/nasdaq-agent"
APP_USER="nasdaq"
REPO="https://github.com/pawangyanwali/Pawan.git"
BRANCH="claude/nasdaq-stock-prediction-agent-kDe60"
SERVICE="nasdaq-agent"
PYTHON_MIN="3.10"
DOMAIN="${DOMAIN:-}"                   # set via --domain flag or env var

# Parse flags
for arg in "$@"; do
  case $arg in
    --domain=*) DOMAIN="${arg#*=}" ;;
    --domain)   shift; DOMAIN="${1:-}" ;;
  esac
done

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
section() { echo -e "\n${GREEN}━━━━ $* ━━━━${NC}"; }

section "1 / 7 — System packages"
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git nginx curl ufw build-essential \
    libssl-dev libffi-dev sqlite3 > /dev/null
info "System packages installed."

section "2 / 7 — Create app user"
if ! id "$APP_USER" &>/dev/null; then
    useradd -r -m -d "$APP_DIR" -s /bin/bash "$APP_USER"
    info "User '$APP_USER' created."
else
    info "User '$APP_USER' already exists."
fi

section "3 / 7 — Clone / update repo"
if [ -d "$APP_DIR/.git" ]; then
    info "Repo exists — pulling latest…"
    sudo -u "$APP_USER" git -C "$APP_DIR" fetch origin
    sudo -u "$APP_USER" git -C "$APP_DIR" reset --hard "origin/$BRANCH"
else
    info "Cloning repo…"
    sudo -u "$APP_USER" git clone --branch "$BRANCH" --depth 1 "$REPO" "$APP_DIR"
fi

section "4 / 7 — Python virtual environment + dependencies"
VENV="$APP_DIR/venv"
if [ ! -d "$VENV" ]; then
    sudo -u "$APP_USER" python3 -m venv "$VENV"
fi
sudo -u "$APP_USER" "$VENV/bin/pip" install --upgrade pip -q
sudo -u "$APP_USER" "$VENV/bin/pip" install -r "$APP_DIR/nasdaq_agent/requirements.txt" -q
info "Dependencies installed."

section "5 / 7 — .env file"
ENV_FILE="$APP_DIR/nasdaq_agent/.env"
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" <<'ENVEOF'
# ── Twelve Data ───────────────────────────────────────────────────────────────
TWELVE_DATA_API_KEY=PASTE_YOUR_KEY_HERE

# ── Charles Schwab (optional — leave placeholders if not using yet) ───────────
SCHWAB_CLIENT_ID=PASTE_APP_KEY_HERE
SCHWAB_CLIENT_SECRET=PASTE_APP_SECRET_HERE
SCHWAB_ACCOUNT_NUMBER=PASTE_ACCOUNT_NUMBER_HERE
SCHWAB_PAPER_TRADING=true
SCHWAB_AUTO_TRADE=false
SCHWAB_MAX_POSITIONS=5
SCHWAB_MIN_CONFIDENCE=68
SCHWAB_RISK_PCT=1.0
SCHWAB_MAX_DAILY_LOSS=500
ENVEOF
    chown "$APP_USER:$APP_USER" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    warn ".env created at $ENV_FILE — EDIT IT NOW before starting the agent!"
else
    info ".env already exists — skipping."
fi

section "6 / 7 — systemd service"
cat > "/etc/systemd/system/${SERVICE}.service" <<SVCEOF
[Unit]
Description=NASDAQ Scalping Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${APP_DIR}/nasdaq_agent
EnvironmentFile=${ENV_FILE}
ExecStartPre=/bin/truncate -s 0 ${APP_DIR}/logs/agent.err
ExecStart=${VENV}/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --no-access-log
Restart=on-failure
RestartSec=10
StandardOutput=append:${APP_DIR}/logs/agent.log
StandardError=append:${APP_DIR}/logs/agent.err

[Install]
WantedBy=multi-user.target
SVCEOF

mkdir -p "$APP_DIR/logs"
chown -R "$APP_USER:$APP_USER" "$APP_DIR/logs"
systemctl daemon-reload
systemctl enable "$SERVICE"
info "systemd service created and enabled."

section "7 / 7 — Nginx reverse proxy"
PUBLIC_IP=$(curl -sf http://checkip.amazonaws.com/ || echo "YOUR_SERVER_IP")
SERVER_NAME="${DOMAIN:-${PUBLIC_IP} _}"

cat > "/etc/nginx/sites-available/${SERVICE}" <<NGINXEOF
server {
    listen 80;
    server_name ${SERVER_NAME};

    # WebSocket support
    location /ws {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade \$http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host \$host;
        proxy_read_timeout 3600s;
    }

    # API + dashboard
    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host \$host;
        proxy_set_header   X-Real-IP \$remote_addr;
        proxy_set_header   X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
        client_max_body_size 10M;
    }
}
NGINXEOF

ln -sf "/etc/nginx/sites-available/${SERVICE}" "/etc/nginx/sites-enabled/${SERVICE}"
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable nginx
systemctl restart nginx
info "Nginx configured."

section "Firewall"
ufw allow OpenSSH
ufw allow 'Nginx Full'   # opens both 80 and 443
ufw --force enable
info "Firewall: SSH + HTTP + HTTPS open."

# ── HTTPS via Let's Encrypt (only when --domain is provided) ─────────────────
if [ -n "$DOMAIN" ]; then
  section "SSL — Let's Encrypt for ${DOMAIN}"
  apt-get install -y --no-install-recommends certbot python3-certbot-nginx > /dev/null
  certbot --nginx \
    --non-interactive \
    --agree-tos \
    --redirect \
    --email "admin@${DOMAIN}" \
    -d "${DOMAIN}" \
    -d "www.${DOMAIN}" || warn "certbot failed — check DNS is pointing to this server"

  # Auto-renewal via systemd timer (installed by certbot package)
  systemctl enable certbot.timer 2>/dev/null || true
  info "SSL certificate installed. Auto-renewal enabled."
  info "Schwab callback URL: https://${DOMAIN}/schwab/callback"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN} Deployment complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  Next steps:"
echo -e "  1. Edit your .env file:"
echo -e "     ${YELLOW}nano $ENV_FILE${NC}"
echo -e "     (paste your TWELVE_DATA_API_KEY and Schwab credentials)"
echo ""
echo -e "  2. Start the agent:"
echo -e "     ${YELLOW}sudo systemctl start $SERVICE${NC}"
echo ""
echo -e "  3. Check it's running:"
echo -e "     ${YELLOW}sudo systemctl status $SERVICE${NC}"
echo -e "     ${YELLOW}curl http://localhost:8000/api/status${NC}"
echo ""
echo -e "  4. Open your dashboard:"
echo -e "     ${YELLOW}http://${PUBLIC_IP}${NC}"
echo ""
echo -e "  Useful commands:"
echo -e "    Logs:    ${YELLOW}tail -f $APP_DIR/logs/agent.log${NC}"
echo -e "    Errors:  ${YELLOW}tail -f $APP_DIR/logs/agent.err${NC}"
echo -e "    Restart: ${YELLOW}sudo systemctl restart $SERVICE${NC}"
echo -e "    Stop:    ${YELLOW}sudo systemctl stop $SERVICE${NC}"
echo ""
