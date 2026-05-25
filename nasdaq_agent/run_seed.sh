#!/usr/bin/env bash
# Run the admin seed script, loading .env from the same directory.
# Usage: sudo bash run_seed.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
    # Export every non-comment variable from .env into the current shell
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
else
    echo "WARNING: $ENV_FILE not found — relying on existing environment"
fi

cd "$SCRIPT_DIR"
"${VENV:-$SCRIPT_DIR/../venv}/bin/python" -c "
import sys, os
sys.path.insert(0, os.getcwd())
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from auth.models import init_tables
from auth.seed import seed_admin
print('Initialising auth tables...')
init_tables()
print('Seeding admin user...')
seed_admin()
print('Done.')
"
