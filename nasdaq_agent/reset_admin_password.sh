#!/usr/bin/env bash
# Reset the pawan_gyanwali admin password and force a change on next login.
#
# Usage:
#   ADMIN_TEMP_PASSWORD=<new-temp-pw> sudo -E bash reset_admin_password.sh
#
# If ADMIN_TEMP_PASSWORD is not set, a random password is generated and
# printed to stdout — record it before the terminal session closes.
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
import sys, os, secrets, string
sys.path.insert(0, os.getcwd())

from agent.db import get_conn
from auth.utils import hash_password

USERNAME = "pawan_gyanwali"

temp_pw = os.environ.get("ADMIN_TEMP_PASSWORD", "").strip()
if not temp_pw:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    temp_pw  = "".join(secrets.choice(alphabet) for _ in range(20))
    print(f"Generated temporary password: {temp_pw}")
    print("(Record this now — it will not be shown again.)")

new_hash = hash_password(temp_pw)

with get_conn() as c:
    row = c.execute("SELECT id FROM users WHERE username = ?", (USERNAME,)).fetchone()
    if not row:
        c.execute(
            "INSERT INTO users (username, email, hashed_password, role, status, force_password_change) "
            "VALUES (?, ?, ?, 'ADMIN', 'ACTIVE', TRUE)",
            (USERNAME, "pawangyanwali@gmail.com", new_hash),
        )
        print(f"Admin user '{USERNAME}' created.")
    else:
        c.execute(
            "UPDATE users SET hashed_password = ?, force_password_change = TRUE WHERE username = ?",
            (new_hash, USERNAME),
        )
        print(f"Admin user '{USERNAME}' password reset.")

print("Done. Password change will be required on next login.")
PYEOF
