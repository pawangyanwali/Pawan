#!/usr/bin/env bash
# Reset the pawan_gyanwali admin password back to the temporary password
# and force a password change on next login.
# Usage: sudo bash reset_admin_password.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
else
    echo "WARNING: $ENV_FILE not found — relying on existing environment"
fi

cd "$SCRIPT_DIR"
"${VENV:-$SCRIPT_DIR/../venv}/bin/python" - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from agent.db import get_conn
from auth.utils import hash_password

USERNAME      = "pawan_gyanwali"
TEMP_PASSWORD = "NasdaqAdmin@2024"

new_hash = hash_password(TEMP_PASSWORD)

with get_conn() as c:
    row = c.execute("SELECT id FROM users WHERE username = ?", (USERNAME,)).fetchone()
    if not row:
        # User missing — re-insert
        c.execute(
            "INSERT INTO users (username, email, hashed_password, role, status, force_password_change) "
            "VALUES (?, ?, ?, 'ADMIN', 'ACTIVE', TRUE)",
            (USERNAME, "pawangyanwali@gmail.com", new_hash),
        )
        print(f"Admin user '{USERNAME}' created with temp password.")
    else:
        c.execute(
            "UPDATE users SET hashed_password = ?, force_password_change = TRUE WHERE username = ?",
            (new_hash, USERNAME),
        )
        print(f"Admin user '{USERNAME}' password reset to temp password.")

print("Done. Login with: pawan_gyanwali / NasdaqAdmin@2024")
PYEOF
