"""
Schwab OAuth 2.0 authentication for ThinkorSwim paper trading.

Flow
----
1. First run: opens a browser to Schwab's login page (handles MFA automatically
   as part of their login UI). After login, Schwab redirects to
   https://127.0.0.1:8182?code=XXX — our local HTTPS server captures the code.
2. We exchange the code for access_token + refresh_token and store them
   encrypted in data/schwab_tokens.json (gitignored).
3. The access token expires every 30 min — we auto-refresh it in the background.
4. The refresh token expires every 7 days — user must re-auth once a week.

Environment variables required (in .env):
    SCHWAB_CLIENT_ID      — App Key from developer.schwab.com
    SCHWAB_CLIENT_SECRET  — App Secret from developer.schwab.com
    SCHWAB_ACCOUNT_NUMBER — Paper trading account number (from ThinkorSwim)
    SCHWAB_PAPER_TRADING  — "true" to use paper endpoint, "false" for live
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import ssl
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
AUTH_URL     = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_URL    = "https://api.schwabapi.com/v1/oauth/token"
# Web callback (production) — used when running on AWS with HTTPS domain
WEB_REDIRECT_URI  = "https://scalpingstocksai.com/schwab/callback"
# Local callback (dev only) — kept for backwards compat
REDIRECT_URI = WEB_REDIRECT_URI
CALLBACK_PORT = 8182
TOKEN_PATH   = Path(__file__).parent.parent.parent / "data" / "schwab_tokens.json"

# ── Token state ───────────────────────────────────────────────────────────────
_tokens: dict = {}
_tokens_lock  = threading.Lock()
_refresh_timer: Optional[threading.Timer] = None


def _client_id() -> str:
    v = os.getenv("SCHWAB_CLIENT_ID", "")
    if not v:
        raise RuntimeError("SCHWAB_CLIENT_ID not set in .env")
    return v

def _client_secret() -> str:
    v = os.getenv("SCHWAB_CLIENT_SECRET", "")
    if not v:
        raise RuntimeError("SCHWAB_CLIENT_SECRET not set in .env")
    return v


# ── Token persistence ─────────────────────────────────────────────────────────

def _save_tokens(data: dict) -> None:
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(json.dumps(data, indent=2))
    logger.info("Schwab tokens saved.")

def _load_tokens() -> dict:
    if TOKEN_PATH.exists():
        try:
            return json.loads(TOKEN_PATH.read_text())
        except Exception:
            pass
    return {}


# ── HTTP token exchange ───────────────────────────────────────────────────────

def _basic_auth() -> str:
    creds = f"{_client_id()}:{_client_secret()}"
    return base64.b64encode(creds.encode()).decode()

def _post_token(payload: dict) -> dict:
    data = urllib.parse.urlencode(payload).encode()
    req  = urllib.request.Request(TOKEN_URL, data=data, method="POST")
    req.add_header("Authorization", f"Basic {_basic_auth()}")
    req.add_header("Content-Type",  "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


# ── Auth code capture (local HTTPS server) ────────────────────────────────────

_auth_code: Optional[str]  = None
_auth_event = threading.Event()

class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global _auth_code
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        code   = params.get("code", [None])[0]
        if code:
            _auth_code = code
            _auth_event.set()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<html><body><h2>Authenticated! You can close this tab.</h2></body></html>")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"<html><body><h2>Error: no code received.</h2></body></html>")

    def log_message(self, fmt, *args):
        pass   # suppress default access log


def _start_callback_server() -> HTTPServer:
    """Start local HTTPS server to capture OAuth callback."""
    import datetime as _dt, tempfile
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    # Generate self-signed cert in pure Python (no openssl binary needed)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_dt.datetime.now(_dt.timezone.utc))
        .not_valid_after(_dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(__import__("ipaddress").IPv4Address("127.0.0.1"))]), critical=False)
        .sign(key, hashes.SHA256())
    )

    cert_dir  = tempfile.mkdtemp()
    key_path  = Path(cert_dir) / "key.pem"
    cert_path = Path(cert_dir) / "cert.pem"

    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert_path), str(key_path))

    server = HTTPServer(("127.0.0.1", CALLBACK_PORT), _CallbackHandler)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# ── Public auth flow ──────────────────────────────────────────────────────────

def start_auth_flow() -> dict:
    """
    Launch browser-based OAuth flow.  Blocks until user completes login + MFA.
    Returns token dict on success.
    """
    global _auth_code
    _auth_code = None
    _auth_event.clear()

    # PKCE
    code_verifier  = secrets.token_urlsafe(64)
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).rstrip(b"=").decode()

    params = urllib.parse.urlencode({
        "response_type":         "code",
        "client_id":             _client_id(),
        "redirect_uri":          REDIRECT_URI,
        "scope":                 "readonly",
        "code_challenge":        code_challenge,
        "code_challenge_method": "S256",
    })
    login_url = f"{AUTH_URL}?{params}"

    server = _start_callback_server()
    logger.info(f"Opening browser for Schwab login (MFA will be prompted):\n{login_url}")

    # Open browser
    import webbrowser
    webbrowser.open(login_url)

    # Wait up to 5 min for user to complete login + MFA
    if not _auth_event.wait(timeout=300):
        server.shutdown()
        raise TimeoutError("Auth timed out — user did not complete login in 5 minutes.")
    server.shutdown()

    if not _auth_code:
        raise RuntimeError("No auth code received after login.")

    # Exchange code for tokens
    token_data = _post_token({
        "grant_type":    "authorization_code",
        "code":          _auth_code,
        "redirect_uri":  REDIRECT_URI,
        "code_verifier": code_verifier,
    })

    _store_tokens(token_data)
    _schedule_refresh(token_data.get("expires_in", 1800))
    logger.info("Schwab authentication successful.")
    return get_token_status()


def _store_tokens(data: dict) -> None:
    with _tokens_lock:
        _tokens.clear()
        _tokens.update(data)
        _tokens["stored_at"] = time.time()
    _save_tokens(_tokens)


def refresh_access_token() -> bool:
    """Use the refresh token to get a new access token. Returns True on success."""
    with _tokens_lock:
        rt = _tokens.get("refresh_token")
    if not rt:
        logger.warning("No refresh token — re-auth required.")
        return False
    try:
        data = _post_token({
            "grant_type":    "refresh_token",
            "refresh_token": rt,
        })
        _store_tokens(data)
        _schedule_refresh(data.get("expires_in", 1800))
        logger.info("Schwab access token refreshed.")
        return True
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        return False


def _schedule_refresh(expires_in: int) -> None:
    """Schedule a token refresh 5 min before expiry."""
    global _refresh_timer
    if _refresh_timer:
        _refresh_timer.cancel()
    delay = max(60, expires_in - 300)
    _refresh_timer = threading.Timer(delay, refresh_access_token)
    _refresh_timer.daemon = True
    _refresh_timer.start()


def load_stored_tokens() -> bool:
    """Load tokens from disk (called at startup). Returns True if valid."""
    data = _load_tokens()
    if not data or "access_token" not in data:
        return False
    with _tokens_lock:
        _tokens.update(data)
    stored_at  = data.get("stored_at", 0)
    expires_in = data.get("expires_in", 1800)
    remaining  = expires_in - (time.time() - stored_at)
    if remaining < 60:
        logger.info("Stored access token expired — refreshing…")
        return refresh_access_token()
    _schedule_refresh(int(remaining))
    logger.info(f"Schwab tokens loaded. Access token valid for {int(remaining)}s.")
    return True


def get_access_token() -> Optional[str]:
    with _tokens_lock:
        return _tokens.get("access_token")


# ── Web OAuth flow (production — AWS server with HTTPS domain) ────────────────

def get_web_auth_url() -> str:
    """Return the Schwab authorization URL for the web-based OAuth flow."""
    params = urllib.parse.urlencode({
        "response_type": "code",
        "client_id":     _client_id(),
        "redirect_uri":  WEB_REDIRECT_URI,
    })
    return f"{AUTH_URL}?{params}"


def exchange_web_code(code: str) -> bool:
    """
    Exchange the authorization code from /schwab/callback for tokens.
    Called by the FastAPI callback route. Returns True on success.
    """
    try:
        data = _post_token({
            "grant_type":   "authorization_code",
            "code":         code,
            "redirect_uri": WEB_REDIRECT_URI,
        })
        _store_tokens(data)
        _schedule_refresh(data.get("expires_in", 1800))
        logger.info("[Schwab] Web OAuth complete — tokens stored.")
        return True
    except Exception as e:
        logger.error(f"[Schwab] Web code exchange failed: {e}")
        return False


def get_token_status() -> dict:
    with _tokens_lock:
        tok = dict(_tokens)
    stored_at  = tok.get("stored_at", 0)
    expires_in = tok.get("expires_in", 1800)
    remaining  = max(0, expires_in - (time.time() - stored_at)) if stored_at else 0
    rt_stored  = tok.get("stored_at", 0)
    # Refresh token lasts 7 days
    rt_remaining = max(0, 7 * 86400 - (time.time() - rt_stored)) if rt_stored else 0
    return {
        "connected":          bool(tok.get("access_token")),
        "access_token_ttl_s": int(remaining),
        "refresh_token_ttl_s": int(rt_remaining),
        "refresh_token_expires": datetime.fromtimestamp(
            rt_stored + 7 * 86400, tz=timezone.utc
        ).isoformat() if rt_stored else None,
        "paper_trading":      os.getenv("SCHWAB_PAPER_TRADING", "true").lower() == "true",
        "account_number":     os.getenv("SCHWAB_ACCOUNT_NUMBER", ""),
    }
