"""
Schwab OAuth 2.0 authentication.

Two apps, two token managers:
  PRIMARY  (SCHWAB_CLIENT_ID / SCHWAB_CLIENT_SECRET)
      → "NASDAQ Scalping Agent" — Accounts and Trading Production
      → Used for: streamer userPreference, positions, orders
      → Route: GET /schwab/auth  →  GET /schwab/callback

  MARKET DATA (SCHWAB_MD_CLIENT_ID / SCHWAB_MD_CLIENT_SECRET)
      → "nasdaq-scalping-agent" — Market Data Production
      → Used for: /quotes, /chains, /movers, /pricehistory
      → Route: GET /schwab/auth/md  →  GET /schwab/callback/md

Environment variables (.env):
    SCHWAB_CLIENT_ID        — Accounts+Trading app key
    SCHWAB_CLIENT_SECRET    — Accounts+Trading app secret
    SCHWAB_MD_CLIENT_ID     — Market Data app key
    SCHWAB_MD_CLIENT_SECRET — Market Data app secret
    SCHWAB_ACCOUNT_NUMBER   — Paper/live account number
    SCHWAB_PAPER_TRADING    — "true" for paper, "false" for live
    SCHWAB_TOKEN_DIR        — optional token directory override
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

AUTH_URL  = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

_DEFAULT_TOKEN_DIR = Path(__file__).parent.parent.parent / "data"


def _configured_token_dir() -> Path:
    """Return the directory used for Schwab token JSON files."""
    return Path(os.getenv("SCHWAB_TOKEN_DIR", str(_DEFAULT_TOKEN_DIR))).expanduser()

# Persistent backup dir in the app user's home — writable without sudo, survives redeploys.
# Override with SCHWAB_TOKEN_BACKUP_DIR env var if a different path is preferred.
_BACKUP_DIR = Path(os.getenv("SCHWAB_TOKEN_BACKUP_DIR", Path.home() / ".nasdaq-agent"))


# ── Reusable token manager ────────────────────────────────────────────────────

class _TokenManager:
    """Manages OAuth tokens for one Schwab app (PKCE web flow)."""

    def __init__(self, name: str, client_id_env: str, client_secret_env: str,
                 token_filename: str) -> None:
        self.name             = name
        self._id_env          = client_id_env
        self._secret_env      = client_secret_env
        self._token_dir       = _configured_token_dir()
        self._token_path      = self._token_dir / token_filename
        self._tokens: dict    = {}
        self._lock            = threading.Lock()
        self._refresh_timer: Optional[threading.Timer] = None
        # Pending PKCE verifiers keyed by OAuth state param
        self._pending: dict[str, str] = {}
        self._pending_lock    = threading.Lock()

    # ── Credentials ──────────────────────────────────────────────────────────

    def client_id(self) -> str:
        v = os.getenv(self._id_env, "").strip()
        if not v:
            raise RuntimeError(f"{self._id_env} not set in .env")
        return v

    def client_secret(self) -> str:
        v = os.getenv(self._secret_env, "").strip()
        if not v:
            raise RuntimeError(f"{self._secret_env} not set in .env")
        return v

    def is_configured(self) -> bool:
        return bool(os.getenv(self._id_env, "").strip())

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        self._token_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._tokens, indent=2)
        self._token_path.write_text(payload)
        # Mirror to persistent backup so tokens survive git-pull redeploys / container restarts.
        try:
            _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            (_BACKUP_DIR / self._token_path.name).write_text(payload)
        except Exception as _e:
            logger.debug(f"[Schwab/{self.name}] Token backup write skipped: {_e}")

    def _load_from_disk(self) -> dict:
        candidates = [self._token_path]
        legacy_path = _DEFAULT_TOKEN_DIR / self._token_path.name
        if legacy_path != self._token_path:
            candidates.append(legacy_path)

        for path in candidates:
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            if path != self._token_path:
                try:
                    self._token_dir.mkdir(parents=True, exist_ok=True)
                    self._token_path.write_text(json.dumps(data, indent=2))
                    logger.info(
                        f"[Schwab/{self.name}] Tokens migrated from {path} -> {self._token_path}"
                    )
                except Exception as _e:
                    logger.warning(f"[Schwab/{self.name}] Token migration skipped: {_e}")
            return data
        # Primary path missing (fresh deploy / container restart) — try persistent backup.
        backup_path = _BACKUP_DIR / self._token_path.name
        if backup_path.exists():
            try:
                data = json.loads(backup_path.read_text())
                # Restore to primary location so normal path works from here on.
                self._token_dir.mkdir(parents=True, exist_ok=True)
                self._token_path.write_text(json.dumps(data, indent=2))
                logger.info(
                    f"[Schwab/{self.name}] Tokens restored from persistent backup → {self._token_path}"
                )
                return data
            except Exception as _e:
                logger.warning(f"[Schwab/{self.name}] Backup restore failed: {_e}")
        return {}

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _basic_auth(self) -> str:
        return base64.b64encode(
            f"{self.client_id()}:{self.client_secret()}".encode()
        ).decode()

    def _post_token(self, payload: dict) -> dict:
        data = urllib.parse.urlencode(payload).encode()
        req  = urllib.request.Request(TOKEN_URL, data=data, method="POST")
        req.add_header("Authorization", f"Basic {self._basic_auth()}")
        req.add_header("Content-Type",  "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            # Log the full response body so we can see the actual Schwab error
            # (e.g. "invalid_grant" = expired refresh token, "invalid_client" = wrong credentials)
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = "<unreadable>"
            logger.error(
                f"[Schwab/{self.name}] Token endpoint {e.code}: {body}"
            )
            raise

    # ── Token storage ─────────────────────────────────────────────────────────

    def _store(self, data: dict) -> None:
        with self._lock:
            self._tokens.clear()
            self._tokens.update(data)
            self._tokens["stored_at"] = time.time()
        self._save()
        # Reset account-hash cache so discovery retries with the new token
        if self.name == "Trader":
            try:
                import agent.broker.schwab_client as _sc
                _sc._cached_account_hash = ""
                _sc._hash_discovery_failed = False
                _sc._hash_last_attempt = 0.0
            except Exception:
                pass
        logger.info(f"[Schwab/{self.name}] Tokens saved.")

    # ── Refresh ───────────────────────────────────────────────────────────────

    def refresh(self, _retry: int = 0) -> bool:
        with self._lock:
            rt = self._tokens.get("refresh_token")
        if not rt:
            logger.warning(f"[Schwab/{self.name}] No refresh token — re-auth required.")
            return False
        try:
            data = self._post_token({"grant_type": "refresh_token", "refresh_token": rt})
            self._store(data)
            self._schedule_refresh(data.get("expires_in", 1800))
            logger.info(f"[Schwab/{self.name}] Access token refreshed.")
            return True
        except urllib.error.HTTPError as e:
            if e.code == 400:
                # 400 = invalid_grant (expired/revoked refresh token) or bad credentials.
                # Do not retry — clear tokens and require re-auth.
                logger.error(
                    f"[Schwab/{self.name}] Refresh token rejected (400) — "
                    f"tokens cleared. Re-authenticate via /schwab/auth"
                )
                with self._lock:
                    self._tokens.clear()
                if self._token_path.exists():
                    try:
                        self._token_path.unlink()
                    except Exception:
                        pass
            else:
                # 403 = Akamai WAF transient block; 5xx = Schwab outage.
                # Retry with exponential backoff (2, 4, 8, 16 … up to 30 min).
                backoff = min(120 * (2 ** _retry), 1800)
                logger.warning(
                    f"[Schwab/{self.name}] Token refresh HTTP {e.code} — "
                    f"retry #{_retry + 1} in {backoff}s"
                )
                with self._lock:
                    if self._refresh_timer:
                        self._refresh_timer.cancel()
                    t = threading.Timer(backoff, self.refresh, kwargs={"_retry": _retry + 1})
                    t.daemon = True
                    t.start()
                    self._refresh_timer = t
            return False
        except Exception as e:
            backoff = min(120 * (2 ** _retry), 1800)
            logger.warning(
                f"[Schwab/{self.name}] Token refresh error — retry #{_retry + 1} in {backoff}s: {e}"
            )
            with self._lock:
                if self._refresh_timer:
                    self._refresh_timer.cancel()
                t = threading.Timer(backoff, self.refresh, kwargs={"_retry": _retry + 1})
                t.daemon = True
                t.start()
                self._refresh_timer = t
            return False

    def _schedule_refresh(self, expires_in: int) -> None:
        if self._refresh_timer:
            self._refresh_timer.cancel()
        delay = max(60, expires_in - 300)
        self._refresh_timer = threading.Timer(delay, self.refresh)
        self._refresh_timer.daemon = True
        self._refresh_timer.start()

    # ── Load from disk (called at startup) ────────────────────────────────────

    def load_stored(self) -> bool:
        data = self._load_from_disk()
        if not data or "access_token" not in data:
            return False
        with self._lock:
            self._tokens.update(data)
        stored_at  = data.get("stored_at", 0)
        expires_in = data.get("expires_in", 1800)
        remaining  = expires_in - (time.time() - stored_at)
        if remaining < 60:
            logger.info(f"[Schwab/{self.name}] Stored token expired — refreshing…")
            return self.refresh()
        self._schedule_refresh(int(remaining))
        logger.info(f"[Schwab/{self.name}] Token loaded, valid for {int(remaining)}s.")
        return True

    # ── Web OAuth (PKCE) ──────────────────────────────────────────────────────

    def build_auth_url(self, redirect_uri: str) -> str:
        """Build the Schwab login URL. Redirect the user's browser to this URL."""
        code_verifier  = secrets.token_urlsafe(64)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).rstrip(b"=").decode()
        state = secrets.token_urlsafe(16)
        with self._pending_lock:
            self._pending[state] = code_verifier
        params = urllib.parse.urlencode({
            "response_type":         "code",
            "client_id":             self.client_id(),
            "redirect_uri":          redirect_uri,
            "code_challenge":        code_challenge,
            "code_challenge_method": "S256",
            "state":                 state,
        })
        return f"{AUTH_URL}?{params}"

    def exchange_code(self, code: str, state: str, redirect_uri: str) -> tuple[bool, str]:
        """Exchange auth code for tokens. Returns (True, "") on success or (False, reason)."""
        with self._pending_lock:
            code_verifier = self._pending.pop(state, None)
        try:
            payload = {
                "grant_type":   "authorization_code",
                "code":         code,
                "redirect_uri": redirect_uri,
            }
            if code_verifier:
                payload["code_verifier"] = code_verifier
            data = self._post_token(payload)
            # Store redirect_uri alongside tokens so refresh can reference it
            data["_redirect_uri"] = redirect_uri
            self._store(data)
            self._schedule_refresh(data.get("expires_in", 1800))
            logger.info(f"[Schwab/{self.name}] Web OAuth complete.")
            return True, ""
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = "<unreadable>"
            reason = f"HTTP {e.code}: {body}"
            logger.error(f"[Schwab/{self.name}] Code exchange failed: {reason}")
            return False, reason
        except Exception as e:
            logger.error(f"[Schwab/{self.name}] Code exchange failed: {e}")
            return False, str(e)

    # ── Public getters ────────────────────────────────────────────────────────

    def get_access_token(self) -> Optional[str]:
        with self._lock:
            return self._tokens.get("access_token")

    def get_status(self) -> dict:
        with self._lock:
            tok = dict(self._tokens)
        stored_at    = tok.get("stored_at", 0)
        expires_in   = tok.get("expires_in", 1800)
        remaining    = max(0, expires_in - (time.time() - stored_at)) if stored_at else 0
        rt_remaining = max(0, 7 * 86400 - (time.time() - stored_at)) if stored_at else 0
        return {
            "connected":           bool(tok.get("access_token")),
            "app":                 self.name,
            "access_token_ttl_s":  int(remaining),
            "refresh_token_ttl_s": int(rt_remaining),
            "refresh_token_expires": datetime.fromtimestamp(
                stored_at + 7 * 86400, tz=timezone.utc
            ).isoformat() if stored_at else None,
        }


# ── Two app instances ─────────────────────────────────────────────────────────

# Primary: Accounts and Trading → streamer + trading
_trader = _TokenManager(
    name="Trader",
    client_id_env="SCHWAB_CLIENT_ID",
    client_secret_env="SCHWAB_CLIENT_SECRET",
    token_filename="schwab_tokens.json",
)

# Market Data: REST quotes, chains, movers, price history
_market_data = _TokenManager(
    name="MarketData",
    client_id_env="SCHWAB_MD_CLIENT_ID",
    client_secret_env="SCHWAB_MD_CLIENT_SECRET",
    token_filename="schwab_md_tokens.json",
)


# ── Public API (backwards-compatible names) ───────────────────────────────────

def load_stored_tokens() -> bool:
    """Load primary (Trader) tokens from disk. Called at startup."""
    return _trader.load_stored()

def load_stored_md_tokens() -> bool:
    """Load Market Data tokens from disk. Called at startup."""
    if not _market_data.is_configured():
        return False
    return _market_data.load_stored()

def get_access_token() -> Optional[str]:
    """Primary (Accounts+Trading) access token."""
    return _trader.get_access_token()

def get_md_access_token() -> Optional[str]:
    """
    Market Data access token.
    Falls back to the primary token if MD app is not configured,
    so a single-app setup still works.
    """
    tok = _market_data.get_access_token()
    if tok:
        return tok
    return _trader.get_access_token()

def get_token_status() -> dict:
    status = _trader.get_status()
    status["paper_trading"]  = os.getenv("SCHWAB_PAPER_TRADING", "true").lower() == "true"
    status["account_number"] = os.getenv("SCHWAB_ACCOUNT_NUMBER", "")
    return status

def get_md_token_status() -> dict:
    return _market_data.get_status()

def refresh_access_token() -> bool:
    return _trader.refresh()

# Web OAuth helpers used by FastAPI routes
def build_auth_url(redirect_uri: str) -> str:
    return _trader.build_auth_url(redirect_uri)

def exchange_auth_code(code: str, state: str, redirect_uri: str) -> tuple[bool, str]:
    return _trader.exchange_code(code, state, redirect_uri)

def build_md_auth_url(redirect_uri: str) -> str:
    return _market_data.build_auth_url(redirect_uri)

def exchange_md_auth_code(code: str, state: str, redirect_uri: str) -> tuple[bool, str]:
    return _market_data.exchange_code(code, state, redirect_uri)

# Legacy aliases kept for any code that still imports these names
def get_web_auth_url() -> str:
    return _trader.build_auth_url("https://scalpingstocksai.com/schwab/callback")

def exchange_web_code(code: str) -> bool:
    return _trader.exchange_code(code, "", "https://scalpingstocksai.com/schwab/callback")

def start_auth_flow() -> dict:
    raise RuntimeError("Local browser auth not supported on server. Use GET /schwab/auth instead.")
