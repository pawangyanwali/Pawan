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
import gzip
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import urllib.parse
import urllib.request
import uuid
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

_SCHWAB_TOKENS_DDL = """
    CREATE TABLE IF NOT EXISTS schwab_tokens (
        app           TEXT PRIMARY KEY,
        access_token  TEXT,
        refresh_token TEXT,
        expires_in    INTEGER NOT NULL DEFAULT 1800,
        stored_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
        refreshed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        generation    BIGINT NOT NULL DEFAULT 0,
        status        TEXT NOT NULL DEFAULT 'OK',
        last_error    TEXT
    )
"""


def _ensure_schwab_token_table(conn) -> None:
    """Create or migrate the durable PostgreSQL Schwab token table."""
    conn.execute(_SCHWAB_TOKENS_DDL)
    conn.execute("ALTER TABLE schwab_tokens ADD COLUMN IF NOT EXISTS generation BIGINT NOT NULL DEFAULT 0")
    conn.execute("ALTER TABLE schwab_tokens ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'OK'")
    conn.execute("ALTER TABLE schwab_tokens ADD COLUMN IF NOT EXISTS last_error TEXT")


def _row_value(row, key: str, index: int, default=None):
    if row is None:
        return default
    if hasattr(row, "get"):
        return row.get(key, default)
    try:
        return row[index]
    except Exception:
        return default


def _decode_http_error_body(err: urllib.error.HTTPError) -> tuple[str, dict]:
    """Return a readable Schwab error body even when the response is gzip encoded."""
    try:
        raw = err.read() or b""
    except Exception:
        return "<unreadable>", {}
    if isinstance(raw, str):
        text = raw
    else:
        try:
            encoding = ""
            try:
                encoding = (err.headers.get("Content-Encoding") or "").lower()
            except Exception:
                encoding = ""
            if encoding == "gzip" or raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = repr(raw[:500])
    parsed = {}
    try:
        parsed = json.loads(text) if text else {}
    except Exception:
        parsed = {}
    return text[:2000], parsed if isinstance(parsed, dict) else {}


def init_schwab_token_store() -> bool:
    """Ensure the durable PostgreSQL Schwab token store exists."""
    try:
        from agent.db import get_conn
        with get_conn() as conn:
            _ensure_schwab_token_table(conn)
        return True
    except Exception as exc:
        logger.warning("[Schwab] schwab_tokens table init failed: %s", exc)
        return False


# ── Distributed refresh-lock helper ──────────────────────────────────────────

def _release_refresh_lock(vk_client, lock_key: str, lock_token: str, held: bool) -> None:
    """Release the Valkey SETNX refresh lock if we hold it.

    Uses a Lua compare-and-delete to avoid accidentally releasing a lock that
    was re-acquired by another container after our TTL expired.
    """
    if not held or vk_client is None:
        return
    try:
        # Lua atomic compare-and-delete: only DEL if the stored value matches
        # our UUID, so we never accidentally delete another container's lock.
        _LUA_RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end"""
        vk_client.eval(_LUA_RELEASE, 1, lock_key, lock_token)
    except Exception:
        pass  # best-effort — TTL will expire the lock in ≤30s regardless


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
        # Atomic write: write to a sibling tmp file then rename so concurrent
        # readers in the other container never see a partial JSON blob.
        _tmp = self._token_path.with_suffix(".json.tmp")
        _tmp.write_text(payload)
        _tmp.rename(self._token_path)
        # Mirror to persistent backup so tokens survive git-pull redeploys / container restarts.
        try:
            _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            _btmp = _BACKUP_DIR / (_tmp.name)
            _btmp.write_text(payload)
            _btmp.rename(_BACKUP_DIR / self._token_path.name)
        except Exception as _e:
            logger.debug(f"[Schwab/{self.name}] Token backup write skipped: {_e}")

    def _load_from_disk(self) -> dict:
        try:
            from agent.broker.token_store import get_token as _vk_get_token
            cached = _vk_get_token(self.name.lower())
            if cached and cached.get("access_token"):
                return cached
        except Exception:
            pass

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
        # Last resort: PostgreSQL (survives both file-dir wipes and container restarts)
        data = self._pg_load()
        if data:
            try:
                self._token_dir.mkdir(parents=True, exist_ok=True)
                self._token_path.write_text(json.dumps(data, indent=2))
                logger.info(f"[Schwab/{self.name}] Tokens restored from PostgreSQL → {self._token_path}")
            except Exception as _e:
                logger.warning(f"[Schwab/{self.name}] Could not write restored tokens to file: {_e}")
            return data
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
            # Log the full response body so we can see the actual Schwab error.
            # Schwab sometimes gzip-encodes OAuth errors even when urllib leaves
            # the body compressed, which made prior incident logs unreadable.
            body, parsed = _decode_http_error_body(e)
            try:
                setattr(e, "schwab_error_body", body)
                setattr(e, "schwab_error_json", parsed)
            except Exception:
                pass
            logger.error(
                f"[Schwab/{self.name}] Token endpoint {e.code}: {body}"
            )
            raise

    # ── Token storage ─────────────────────────────────────────────────────────

    def _store(self, data: dict) -> None:
        with self._lock:
            # Preserve the refresh_token if Schwab's response omits it (e.g. rotating
            # token implementations that only return a new access_token on refresh).
            existing_rt = self._tokens.get("refresh_token")
            self._tokens.clear()
            self._tokens.update(data)
            if "refresh_token" not in self._tokens and existing_rt:
                self._tokens["refresh_token"] = existing_rt
            self._tokens["stored_at"] = time.time()
            self._tokens["generation"] = int(self._tokens["stored_at"] * 1000)
            self._tokens["status"] = "OK"
            snapshot = dict(self._tokens)
        self._save()
        # Persist to PostgreSQL as a durable fallback — survives token-dir wipes.
        self._pg_save(snapshot)
        # Reset account-hash cache so discovery retries with the new token
        if self.name == "Trader":
            try:
                import agent.broker.schwab_client as _sc
                _sc._cached_account_hash = ""
                _sc._hash_discovery_failed = False
                _sc._hash_last_attempt = 0.0
            except Exception:
                pass
        # Mirror to Valkey so token-service consumers can read without disk access.
        try:
            from agent.broker.token_store import put_token as _vk_put
            _vk_put(self.name.lower(), snapshot)
        except Exception:
            pass
        logger.info(f"[Schwab/{self.name}] Tokens saved.")

    def _publish_token_event(self, event: str, snapshot: dict | None = None) -> None:
        """Publish an app-specific token event plus the legacy aggregate event."""
        try:
            from agent.valkey_client import _get_client as _vk_get
            client = _vk_get()
            if not client:
                return
            snap = snapshot or {}
            payload = {
                "ts": time.time(),
                "app": self.name.lower(),
                "event": event,
                "generation": int(snap.get("generation") or 0),
                "access_expires_at": (
                    float(snap.get("stored_at") or 0.0)
                    + float(snap.get("expires_in") or 1800)
                ),
            }
            client.publish(f"schwab:{event}:{self.name.lower()}", json.dumps(payload))
            client.publish("schwab:tokens_refreshed", json.dumps(payload))
        except Exception:
            pass

    def _invalidate_all_stores(self, reason: str = "auth_required") -> None:
        """Clear every token store so invalid refresh tokens cannot be resurrected."""
        with self._lock:
            self._tokens.clear()
        for path in (self._token_path, _BACKUP_DIR / self._token_path.name):
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
        try:
            from agent.broker.token_store import delete_token as _vk_delete
            _vk_delete(self.name.lower())
        except Exception:
            pass
        try:
            from agent.db import get_conn
            with get_conn() as conn:
                _ensure_schwab_token_table(conn)
                conn.execute("""
                    INSERT INTO schwab_tokens
                        (app, access_token, refresh_token, expires_in, stored_at,
                         refreshed_at, generation, status, last_error)
                    VALUES (%s, NULL, NULL, 1800, now(), now(), 0, 'AUTH_REQUIRED', %s)
                    ON CONFLICT (app) DO UPDATE SET
                        access_token = NULL,
                        refresh_token = NULL,
                        refreshed_at = now(),
                        status = 'AUTH_REQUIRED',
                        last_error = EXCLUDED.last_error
                """, (self.name.lower(), reason))
        except Exception as exc:
            logger.debug(f"[Schwab/{self.name}] schwab_tokens invalidation failed: {exc}")
        try:
            from agent.service_state import set_state
            set_state(
                f"schwab:tokens:{self.name.lower()}",
                {"status": "AUTH_REQUIRED", "reason": reason, "stored_at": time.time()},
                ttl_s=8 * 86400,
            )
        except Exception:
            pass
        try:
            from agent.valkey_client import _get_client as _vk_get
            client = _vk_get()
            if client:
                status_payload = {
                    "connected": False,
                    "app": self.name,
                    "access_token_ttl_s": 0,
                    "refresh_token_ttl_s": 0,
                    "refresh_token_expires": None,
                    "status": "AUTH_REQUIRED",
                    "reason": reason,
                    "updated_at": time.time(),
                }
                client.setex(
                    f"schwab:token_status:{self.name.lower()}",
                    8 * 86400,
                    json.dumps(status_payload, separators=(",", ":")),
                )
                payload = {
                    "ts": time.time(),
                    "app": self.name.lower(),
                    "event": "auth_required",
                    "reason": reason,
                }
                client.publish(f"schwab:auth_required:{self.name.lower()}", json.dumps(payload))
        except Exception:
            pass

    def _pg_save(self, tokens: dict) -> None:
        """Mirror tokens to PostgreSQL — dedicated schwab_tokens table + service_state fallback."""
        # ── 1. Dedicated schwab_tokens table (primary durable store) ──────────
        try:
            from agent.db import get_conn
            with get_conn() as conn:
                _ensure_schwab_token_table(conn)
                conn.execute("""
                    INSERT INTO schwab_tokens
                        (app, access_token, refresh_token, expires_in, stored_at,
                         refreshed_at, generation, status, last_error)
                    VALUES (%s, %s, %s, %s, to_timestamp(%s), now(), %s, 'OK', NULL)
                    ON CONFLICT (app) DO UPDATE SET
                        access_token  = EXCLUDED.access_token,
                        refresh_token = EXCLUDED.refresh_token,
                        expires_in    = EXCLUDED.expires_in,
                        stored_at     = EXCLUDED.stored_at,
                        refreshed_at  = now(),
                        generation    = EXCLUDED.generation,
                        status        = 'OK',
                        last_error    = NULL
                """, (
                    self.name.lower(),
                    tokens.get("access_token"),
                    tokens.get("refresh_token"),
                    tokens.get("expires_in", 1800),
                    tokens.get("stored_at", time.time()),
                    int(tokens.get("generation") or int(tokens.get("stored_at", time.time()) * 1000)),
                ))
        except Exception as exc:
            logger.debug(f"[Schwab/{self.name}] schwab_tokens PG write failed: {exc}")
        # ── 2. service_state fallback (backward compat — keeps _pg_load working) ─
        try:
            from agent.service_state import set_state
            set_state(f"schwab:tokens:{self.name.lower()}", tokens, ttl_s=8 * 86400)
        except Exception as exc:
            logger.debug(f"[Schwab/{self.name}] service_state PG backup failed: {exc}")

    def _pg_load(self) -> dict:
        """Load tokens from PostgreSQL — schwab_tokens table first, then service_state."""
        # ── 1. Dedicated table ────────────────────────────────────────────────
        try:
            from agent.db import get_conn
            with get_conn() as conn:
                _ensure_schwab_token_table(conn)
                row = conn.execute(
                    """SELECT access_token, refresh_token, expires_in,
                              EXTRACT(EPOCH FROM stored_at)::double precision AS stored_at,
                              COALESCE(generation, 0) AS generation,
                              COALESCE(status, 'OK') AS status
                       FROM schwab_tokens WHERE app = %s""",
                    (self.name.lower(),)
                ).fetchone()
                access_token = _row_value(row, "access_token", 0)
                status = _row_value(row, "status", 5, "OK")
                if row and access_token and str(status).upper() == "OK":
                    return {
                        "access_token":  access_token,
                        "refresh_token": _row_value(row, "refresh_token", 1),
                        "expires_in":    _row_value(row, "expires_in", 2, 1800),
                        "stored_at":     float(_row_value(row, "stored_at", 3, 0.0)),
                        "generation":    int(_row_value(row, "generation", 4, 0) or 0),
                        "status":        status,
                    }
        except Exception as exc:
            logger.debug(f"[Schwab/{self.name}] schwab_tokens PG read failed: {exc}")
        # ── 2. service_state fallback ─────────────────────────────────────────
        try:
            from agent.service_state import get_state
            data = get_state(f"schwab:tokens:{self.name.lower()}", ignore_expiry=False)
            if data and "access_token" in data:
                return data
        except Exception as exc:
            logger.debug(f"[Schwab/{self.name}] service_state PG restore failed: {exc}")
        return {}

    # ── Refresh ───────────────────────────────────────────────────────────────

    def refresh(self, _retry: int = 0) -> bool:
        # Credentials must be present to refresh — fail fast rather than looping forever.
        if not self.is_configured():
            logger.warning(
                f"[Schwab/{self.name}] {self._id_env} not set — token refresh skipped. "
                f"Set the environment variable and restart to re-enable live data."
            )
            try:
                from agent.system_alerts import raise_alert
                raise_alert(
                    alert_type="SCHWAB_AUTH",
                    severity="CRITICAL",
                    source=self.name.lower(),
                    title=f"Schwab {self.name} credentials not configured",
                    message=(
                        f"{self._id_env} is not set in the environment. "
                        f"Token refresh is disabled — live market data and trading are "
                        f"unavailable until credentials are configured and the server restarted."
                    ),
                    metadata={"missing_env": self._id_env, "app": self.name.lower()},
                )
            except Exception:
                pass
            return False
        with self._lock:
            rt = self._tokens.get("refresh_token")
            old_access_token = self._tokens.get("access_token", "")
        if not rt:
            logger.warning(f"[Schwab/{self.name}] No refresh token — re-auth required.")
            return False

        # ── Distributed refresh lock ───────────────────────────────────────────
        # Both web-api and market-data containers share the same token file.
        # Both schedule independent refresh timers for the same _TokenManager,
        # so they can race: both call _post_token with the same refresh_token.
        # Schwab rotates refresh tokens on use — the second caller gets 400
        # invalid_grant with the now-consumed old token.
        #
        # Fix: SET NX EX 30 in Valkey; loser waits 5s, re-reads the fresh token
        # written by the winner, and skips calling Schwab entirely.
        _lock_key   = f"schwab:refresh_lock:{self.name.lower()}"
        _lock_token = str(uuid.uuid4())
        _vk_client  = None
        _lock_held  = False
        try:
            from agent.valkey_client import _get_client as _vk_get
            _vk_client = _vk_get()
        except Exception:
            pass

        if _vk_client is not None:
            try:
                _lock_held = bool(_vk_client.set(_lock_key, _lock_token, nx=True, ex=30))
            except Exception:
                _lock_held = True   # Valkey error — proceed without lock (best-effort)

            if not _lock_held:
                # Another container is already refreshing; wait for it to finish,
                # then reload the newly-written token instead of calling Schwab again.
                logger.info(
                    f"[Schwab/{self.name}] Refresh lock held by peer — "
                    f"waiting 5 s then reloading"
                )
                time.sleep(5)
                fresh = self._load_from_disk()
                if fresh and fresh.get("access_token"):
                    remaining = fresh.get("expires_in", 1800) - (
                        time.time() - fresh.get("stored_at", 0)
                    )
                    if remaining > 60:
                        with self._lock:
                            self._tokens.update(fresh)
                        self._schedule_refresh(int(remaining))
                        logger.info(
                            f"[Schwab/{self.name}] Adopted token from peer refresh "
                            f"(valid for {int(remaining)}s)."
                        )
                        return True
                # Token still stale after waiting — fall through and try anyway.
                logger.warning(
                    f"[Schwab/{self.name}] Peer-written token still stale — "
                    f"attempting own refresh."
                )
        else:
            _lock_held = True   # No Valkey — proceed without distributed lock

        # ── Post-lock disk check ───────────────────────────────────────────────
        # Even after winning the lock, the peer may have finished its own
        # refresh between when we read rt from memory and now (e.g. it held
        # the lock, refreshed, released it, and we acquired it immediately
        # after).  If the disk file is newer than our in-memory stored_at, the
        # peer already refreshed — adopt the fresh token without calling Schwab.
        with self._lock:
            _mem_stored_at = self._tokens.get("stored_at", 0)
        _disk_check = self._load_from_disk()
        if _disk_check and _disk_check.get("access_token"):
            _disk_stored_at = _disk_check.get("stored_at", 0)
            if _disk_stored_at > _mem_stored_at + 5:  # disk is 5+ seconds newer
                _disk_remaining = _disk_check.get("expires_in", 1800) - (
                    time.time() - _disk_stored_at
                )
                if _disk_remaining > 60:
                    with self._lock:
                        self._tokens.update(_disk_check)
                    self._schedule_refresh(int(_disk_remaining))
                    _release_refresh_lock(_vk_client, _lock_key, _lock_token, _lock_held)
                    logger.info(
                        f"[Schwab/{self.name}] Post-lock: peer already refreshed "
                        f"(disk stored_at={_disk_stored_at:.0f} > mem={_mem_stored_at:.0f}) "
                        f"— adopted without Schwab call."
                    )
                    return True

        try:
            data = self._post_token({"grant_type": "refresh_token", "refresh_token": rt})
            self._store(data)
            self._schedule_refresh(data.get("expires_in", 1800))
            logger.info(f"[Schwab/{self.name}] Access token refreshed.")
            # Release the distributed lock now that the new token is on disk.
            _release_refresh_lock(_vk_client, _lock_key, _lock_token, _lock_held)
            # Clear any outstanding auth alert now that refresh succeeded.
            try:
                from agent.system_alerts import resolve_alert
                resolve_alert(alert_key=f"SCHWAB_AUTH:{self.name.lower()}")
            except Exception:
                pass
            # Only publish a token-rotation event when the access token
            # actually rotated. New consumers use app-specific channels and
            # generation ids so a MarketData refresh cannot restart the WS.
            new_access_token = data.get("access_token", "")
            if new_access_token and new_access_token != old_access_token:
                with self._lock:
                    snapshot = dict(self._tokens)
                self._publish_token_event("token_rotated", snapshot)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 400:
                # 400 = invalid_grant (expired/revoked refresh token) or bad credentials.
                # Do not retry — clear tokens and require re-auth.
                _auth_url = "/schwab/auth/md" if self.name.lower() == "marketdata" else "/schwab/auth/at"
                _body = str(getattr(e, "schwab_error_body", "") or "")
                _parsed = getattr(e, "schwab_error_json", {}) or {}
                _err = str(_parsed.get("error") or "invalid_grant")
                _desc = str(_parsed.get("error_description") or _body or "refresh token rejected")
                _reason = f"{_err}: {_desc}"[:500]
                logger.error(
                    f"[Schwab/{self.name}] Refresh token rejected (400) — "
                    f"tokens cleared. Re-authenticate via {_auth_url}. Schwab said: {_reason}"
                )
                # CRITICAL: re-auth required — surface to dashboard, not just logs.
                # This blocks live quotes/trading until an operator re-authenticates.
                try:
                    from agent.system_alerts import raise_alert
                    raise_alert(
                        alert_type="SCHWAB_AUTH",
                        severity="CRITICAL",
                        source=self.name.lower(),
                        title=f"Schwab {self.name} re-authentication required",
                        message=(
                            f"Refresh token was rejected (HTTP 400 {_err}). "
                            f"Tokens cleared — re-authenticate via {_auth_url} to "
                            f"restore live market data and trading. Schwab response: {_desc[:300]}"
                        ),
                        metadata={
                            "http_code": 400,
                            "app": self.name.lower(),
                            "schwab_error": _err,
                            "schwab_error_description": _desc[:1000],
                        },
                    )
                except Exception:
                    pass
                # Release lock before clearing state — the peer container that
                # lost the race will wake up from its 5-s sleep and find our
                # lock already gone; it will then re-read the (now-deleted) file
                # and skip a second Schwab call.
                _release_refresh_lock(_vk_client, _lock_key, _lock_token, _lock_held)
                self._invalidate_all_stores(reason=_reason)
            else:
                # 403 = Akamai WAF transient block; 5xx = Schwab outage.
                # Retry with exponential backoff (2, 4, 8, 16 … up to 30 min).
                backoff = min(120 * (2 ** _retry), 1800)
                logger.warning(
                    f"[Schwab/{self.name}] Token refresh HTTP {e.code} — "
                    f"retry #{_retry + 1} in {backoff}s"
                )
                # WARNING after repeated transient failures — auto-resolves once a
                # retry succeeds. Dedup collapses the retry loop into one banner
                # with a rising occurrence count.
                if _retry >= 2:
                    try:
                        from agent.system_alerts import raise_alert
                        raise_alert(
                            alert_type="SCHWAB_AUTH",
                            severity="WARNING",
                            source=self.name.lower(),
                            title=f"Schwab {self.name} token refresh failing",
                            message=(
                                f"Token refresh returned HTTP {e.code} on retry "
                                f"#{_retry + 1}; retrying in {backoff}s. Live data "
                                f"may be degraded until it recovers."
                            ),
                            metadata={"http_code": e.code, "retry": _retry + 1,
                                      "app": self.name.lower()},
                        )
                    except Exception:
                        pass
                # Release lock so the retry timer can re-acquire it when it fires.
                _release_refresh_lock(_vk_client, _lock_key, _lock_token, _lock_held)
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
            if _retry >= 2:
                try:
                    from agent.system_alerts import raise_alert
                    raise_alert(
                        alert_type="SCHWAB_AUTH",
                        severity="WARNING",
                        source=self.name.lower(),
                        title=f"Schwab {self.name} token refresh erroring",
                        message=(
                            f"Token refresh raised an error on retry #{_retry + 1}; "
                            f"retrying in {backoff}s. Live data may be degraded: {e}"
                        ),
                        metadata={"error": str(e), "retry": _retry + 1,
                                  "app": self.name.lower()},
                    )
                except Exception:
                    pass
            _release_refresh_lock(_vk_client, _lock_key, _lock_token, _lock_held)
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

    def load_stored(self, schedule_refresh: bool = False) -> bool:
        """
        Load tokens from disk (or PG/Valkey fallback).

        schedule_refresh=True  — token-service only; schedules the background timer.
        schedule_refresh=False — all other containers; load token into memory for
                                 in-process use but leave refresh scheduling to
                                 token-service so there is exactly one refresh timer
                                 per app across the entire fleet.
        """
        data = self._load_from_disk()
        if not data or "access_token" not in data:
            return False
        with self._lock:
            self._tokens.update(data)
        # If credentials are absent, token is read-only (no refresh possible).
        # Tokens may still be valid for the remaining TTL, but we can't renew them.
        if not self.is_configured():
            logger.warning(
                f"[Schwab/{self.name}] Tokens loaded from disk but {self._id_env} is not set "
                f"— refresh scheduling disabled. Live data will stop when the access token expires."
            )
            return True
        stored_at  = data.get("stored_at", 0)
        expires_in = data.get("expires_in", 1800)
        remaining  = expires_in - (time.time() - stored_at)
        if not schedule_refresh:
            logger.info(
                f"[Schwab/{self.name}] Token loaded (valid for {max(0, int(remaining))}s) "
                f"— refresh timer owned by token-service."
            )
            return True
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
            data["_redirect_uri"] = redirect_uri
            self._store(data)
            # Resolve outstanding re-auth alert immediately — tokens are fresh.
            try:
                from agent.system_alerts import resolve_alert
                resolve_alert(alert_key=f"SCHWAB_AUTH:{self.name.lower()}")
            except Exception:
                pass
            # token-service is the sole owner of refresh timers — notify it and
            # all consumers via pub/sub.  Do NOT call _schedule_refresh() here:
            # that would create a second timer competing with token-service's,
            # and both would use the same refresh_token → 400 invalid_grant.
            try:
                from agent.valkey_client import _get_client as _vk_get
                _vk = _vk_get()
                if _vk:
                    import json as _json
                    _ts = time.time()
                    with self._lock:
                        snapshot = dict(self._tokens)
                    payload = {
                        "ts": _ts,
                        "app": self.name.lower(),
                        "event": "new_auth",
                        "generation": int(snapshot.get("generation") or 0),
                    }
                    _vk.publish(
                        f"schwab:new_auth:{self.name.lower()}",
                        _json.dumps(payload),
                    )
                    self._publish_token_event("token_rotated", snapshot)
            except Exception:
                pass
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

def load_stored_tokens(schedule_refresh: bool = False) -> bool:
    """Load primary (Trader) tokens from disk. Called at startup.

    Pass schedule_refresh=False in all containers except token-service.
    """
    return _trader.load_stored(schedule_refresh=schedule_refresh)

def load_stored_md_tokens(schedule_refresh: bool = False) -> bool:
    """Load Market Data tokens from disk. Called at startup.

    Pass schedule_refresh=False in all containers except token-service.
    """
    if not _market_data.is_configured():
        return False
    return _market_data.load_stored(schedule_refresh=schedule_refresh)

def get_access_token() -> Optional[str]:
    """Primary (Accounts+Trading) access token."""
    return _trader.get_access_token()

def get_md_access_token() -> Optional[str]:
    """
    Dedicated Market Data access token.

    Do not fall back to the Trader app token. The two-app production setup must
    keep WS/trading auth and REST market-data auth independent so one failure
    cannot mask the other.
    """
    return _market_data.get_access_token()

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
