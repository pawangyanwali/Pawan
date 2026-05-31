"""
Runtime configuration store — Phase 2.

Stores runtime-tunable settings in a PostgreSQL `config_store` table.
Values are JSON-encoded TEXT (supports float, int, bool, str, list).
Changes propagate to all processes via Valkey pub/sub hot-reload without
requiring a container restart.

Usage:
    from agent.config_manager import config

    value = config.get("paper.budget", 50000.0)
    config.set("paper.budget", 75000.0, updated_by="dashboard")
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    # ── Paper trading ──────────────────────────────────────────────────────────
    "paper.budget":                        lambda: float(os.getenv("PAPER_BUDGET", "50000")),
    "paper.max_trade_pct":                 lambda: float(os.getenv("PAPER_MAX_TRADE_PCT", "5.0")),
    "paper.max_allocated_pct":             lambda: float(os.getenv("PAPER_MAX_ALLOCATED_PCT", "40.0")),
    "paper.max_open_trades":               lambda: int(os.getenv("PAPER_MAX_OPEN_TRADES", "10")),
    "paper.min_confidence":                lambda: float(os.getenv("PAPER_TRADE_MIN_CONFIDENCE", "25.0")),
    # Extended-hours gates
    "paper.ext_hours_high_min_conf":       lambda: 70.0,   # min confidence for HIGH-tier in PM/AH
    "paper.ext_hours_moderate_min_conf":   lambda: 60.0,   # min confidence for MODERATE-tier in PM/AH
    "paper.pre_market_stop_mult":          lambda: 1.5,    # widen stops 1.5× in pre-market
    "paper.after_hours_stop_mult":         lambda: 2.0,    # widen stops 2× in after-hours
    # Position sizing
    "paper.rr_size_mult_min":              lambda: 0.20,   # minimum size multiplier from R:R calculation
    "paper.rr_denominator":                lambda: 2.0,    # divisor in rr_ratio / N → size_mult
    "paper.breakeven_stop_offset":         lambda: 0.02,   # $ offset above entry for T1 breakeven stop
    # EOD management
    "paper.eod_trail_stop_pct":            lambda: 0.003,  # 0.3% trailing stop for EOD winners
    "paper.eod_recovery_stop_pct":         lambda: 0.002,  # 0.2% recovery stop for EOD losers with momentum
    "paper.eod_strong_winner_pct":         lambda: 0.5,    # P&L% threshold for "strong winner" EOD path
    "paper.eod_small_winner_pct":          lambda: 0.1,    # P&L% threshold for "small winner" EOD path
    "paper.eod_loss_threshold_pct":        lambda: -0.3,   # P&L% below which position is a "meaningful loss"
    # Adaptive filter feedback
    "paper.filter_feedback_min_trades":    lambda: 5,      # min closed trades before feeding back to filter
    # ── Trading / account sizing ───────────────────────────────────────────────
    # Seeded from env on first deploy; live-editable via /api/config thereafter.
    "trading.account_size":                lambda: float(os.getenv("TRADING_ACCOUNT_SIZE", "50000")),
    "trading.risk_pct":                    lambda: float(os.getenv("TRADING_RISK_PCT", "1.5")),
    "trading.max_position_pct":            lambda: float(os.getenv("TRADING_MAX_POSITION_PCT", "10.0")),
    "trading.is_paper":                    lambda: os.getenv("IS_PAPER_TRADING", "true").lower() != "false",
    # ── Broker flags ──────────────────────────────────────────────────────────
    "broker.auto_trade":                   lambda: os.getenv("SCHWAB_AUTO_TRADE", "false").lower() == "true",
    "broker.paper_trading":                lambda: os.getenv("SCHWAB_PAPER_TRADING", "true").lower() == "true",
    # ── Scanner cadence ────────────────────────────────────────────────────────
    # scan_interval_s can be reduced from 60→30 during REGULAR session without restart.
    "scanner.scan_interval_s":             lambda: int(os.getenv("SCAN_INTERVAL_SECONDS", "60")),
    "scanner.pre_earnings_blackout_days":  lambda: 3,
    "scanner.post_earnings_cooldown_days": lambda: 1,
    # ── Risk controls ──────────────────────────────────────────────────────────
    # Account size used for all risk-% math (daily-loss halt, portfolio heat,
    # drawdown throttle). Lives in PostgreSQL so it can be changed live and stays
    # the single source of truth for risk percentages.
    "risk.account_size":                   lambda: float(os.getenv("TRADING_ACCOUNT_SIZE", "50000")),
    "risk.daily_loss_warning_pct":         lambda: float(os.getenv("DAILY_LOSS_WARNING_PCT", "1.5")),
    "risk.daily_loss_halt_pct":            lambda: float(os.getenv("DAILY_LOSS_HALT_PCT", "2.5")),
    # Tier 2 loss circuit: force-close ALL open positions at this loss %.
    # Separate from halt (2.5%) so winners can run to their stops while the
    # halt blocks new entries. At this level the account is in genuine distress.
    "risk.daily_loss_liquidate_pct":       lambda: float(os.getenv("DAILY_LOSS_LIQUIDATE_PCT", "4.0")),
    "risk.daily_profit_target_usd":        lambda: float(os.getenv("DAILY_PROFIT_TARGET", "1000")),
    "risk.daily_profit_max_usd":           lambda: float(os.getenv("DAILY_PROFIT_MAX", "1500")),
    "risk.max_concurrent_trades":          lambda: int(os.getenv("MAX_CONCURRENT_TRADES", "3")),
    "risk.max_portfolio_heat_pct":         lambda: float(os.getenv("MAX_PORTFOLIO_HEAT_PCT", "1.5")),
    "risk.max_consecutive_losses":         lambda: int(os.getenv("MAX_CONSECUTIVE_LOSSES", "5")),
    "risk.cooldown_after_losses":          lambda: int(os.getenv("COOLDOWN_LOSSES", "3")),
    "risk.max_daily_trades":               lambda: int(os.getenv("MAX_DAILY_TRADES", "30")),
    "risk.max_per_sector":                 lambda: 2,      # max concurrent positions in same sector
    "risk.volatility_halt_size_mult":      lambda: 0.50,   # size reduction when ATR volatility is elevated
    # Extended-hours position size caps by tier
    "risk.pre_market_high_size_mult":      lambda: 0.40,   # 40% full size for HIGH tier in pre-market
    "risk.pre_market_moderate_size_mult":  lambda: 0.25,   # 25% full size for MODERATE tier in pre-market
    "risk.after_hours_high_size_mult":     lambda: 0.50,   # 50% full size for HIGH tier in after-hours
    "risk.after_hours_moderate_size_mult": lambda: 0.30,   # 30% full size for MODERATE tier in after-hours
    # ── Profit Protect Mode ────────────────────────────────────────────────────
    "risk.profit_protect_min_conf":        lambda: float(os.getenv("PROFIT_PROTECT_CONF", "80.0")),
    "risk.profit_protect_size_mult":       lambda: float(os.getenv("PROFIT_PROTECT_SIZE", "0.60")),
    "risk.profit_protect_drawdown":        lambda: float(os.getenv("PROFIT_PROTECT_DRAWDOWN", "300")),
    # ── Volatility / drawdown circuit breakers ─────────────────────────────────
    "risk.volatility_halt_atr_mult":       lambda: float(os.getenv("VOLATILITY_HALT_ATR_MULT", "2.5")),
    "risk.drawdown_throttle_1_pct":        lambda: float(os.getenv("DRAWDOWN_THROTTLE_1_PCT", "0.5")),
    "risk.drawdown_throttle_2_pct":        lambda: float(os.getenv("DRAWDOWN_THROTTLE_2_PCT", "1.0")),
    # ── Position sizing — confidence multipliers ───────────────────────────────
    "sizing.conf_high_threshold":          lambda: 75.0,   # confidence ≥ this → high multiplier
    "sizing.conf_high_mult":               lambda: 1.25,   # size multiplier when confidence is high
    "sizing.conf_medium_threshold":        lambda: 60.0,   # confidence ≥ this → medium multiplier
    "sizing.conf_medium_mult":             lambda: 1.0,    # size multiplier when confidence is medium
    "sizing.conf_low_threshold":           lambda: 50.0,   # confidence < this → low multiplier
    "sizing.conf_low_mult":                lambda: 0.5,    # size multiplier when confidence is low
    "sizing.conf_default_mult":            lambda: 0.75,   # size multiplier between low and medium thresholds
    # ── Learner ────────────────────────────────────────────────────────────────
    "learner.deep_enabled":                lambda: os.getenv("LEARNER_DEEP_ENABLED", "1").lower() in ("1", "true", "yes"),
    "learner.deep_interval_s":             lambda: max(300, int(os.getenv("LEARNER_DEEP_INTERVAL_S", "3600"))),
    # Market-hours fine-tune cadence. During REGULAR/PRE_MARKET/AFTER_HOURS the
    # learner runs a lightweight 3-epoch fine-tune every N seconds so the BiLSTM
    # keeps learning intraday. 0 disables market-hours training (CLOSED-only).
    # Isolation: the learner container is capped at 1.5 CPU so this never starves
    # the scanner (2.5 CPU) on the 4-vCPU host.
    "learner.deep_market_interval_s":      lambda: max(0, int(os.getenv("LEARNER_DEEP_MARKET_INTERVAL_S", "1800"))),
    "learner.deep_ticker_limit":           lambda: max(1, int(os.getenv("LEARNER_DEEP_TICKER_LIMIT", "100"))),
    # ── Adaptive filter ────────────────────────────────────────────────────────
    "filter.throttle_start_wr":            lambda: 0.50,
    "filter.max_penalty_pts":              lambda: 35,
    # ── Quant strategy reference — Phase 5 configurable defaults ───────────────
    # Each value is the reference-document default; the learning engine will tune
    # these automatically from live trade outcomes (target_mult, stop_mult, etc.)
    "algos.ofi.ofi_z_gate":               lambda: 1.5,    # OFI z-score threshold
    "algos.ofi.avi_gate":                 lambda: 0.35,   # AVI threshold for momentum
    "algos.ofi.qi_gate":                  lambda: 0.55,   # queue imbalance gate
    "algos.vwap_ofi.rvol_gate":           lambda: 1.2,    # VWAP-OFI RVOL gate
    "algos.ema_pull.rvol_gate":           lambda: 1.1,    # EMA pullback RVOL gate
    "algos.vwap_trend.zv_gate":           lambda: 1.0,    # VWAP trend ZV gate
    "algos.donchian.zv_gate":             lambda: 1.5,    # Donchian breakout ZV gate
    "algos.orb_zv.zv_gate":              lambda: 1.5,    # ORB VWAP-ZV gate
    "algos.macd_acc.rvol_gate":           lambda: 1.1,    # MACD acc RVOL gate
    "algos.supertrend.rvol_gate":         lambda: 1.2,    # SuperTrend RVOL gate
    "algos.vol_shock.ratio_min":          lambda: 2.0,    # short/long vol ratio threshold
    "algos.bb_rev.rsi2_oversold":         lambda: 10.0,   # RSI(2) oversold gate
    "algos.bb_rev.bb_z_gate":            lambda: -1.0,   # BB z-score gate (negative)
    "algos.rsi2_snap.rsi2_extreme":       lambda: 5.0,    # RSI(2) extreme gate
    "algos.keltner_fade.rvol_max":        lambda: 2.5,    # Keltner fade max RVOL
    "algos.squeeze.zv_gate":             lambda: 1.0,    # squeeze expansion ZV gate
    "algos.pair_arb.z_enter":            lambda: 2.0,    # pair arb entry z-score
    "algos.regime_sw.adx_trend":         lambda: 25.0,   # ADX trend threshold
    "algos.meta_ens.prob_gate":          lambda: 0.90,   # meta ensemble probability gate
}

# Legacy column map: config_store key → account_config column name
_LEGACY_COLUMN_MAP: dict[str, str] = {
    "paper.budget":           "total_budget",
    "paper.max_trade_pct":    "max_trade_pct",
    "paper.max_allocated_pct": "max_allocated_pct",
    "paper.max_open_trades":  "max_open_trades",
}

_VALKEY_CHANNEL = "config:changed"

_DDL = """
CREATE TABLE IF NOT EXISTS config_store (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    updated_by TEXT NOT NULL DEFAULT 'system'
)
"""

_UPSERT = """
INSERT INTO config_store (key, value, updated_at, updated_by)
VALUES (?, ?, ?, ?)
ON CONFLICT (key) DO UPDATE SET
    value      = EXCLUDED.value,
    updated_at = EXCLUDED.updated_at,
    updated_by = EXCLUDED.updated_by
"""


# ── ConfigManager ──────────────────────────────────────────────────────────────

class ConfigManager:
    """
    Thread-safe singleton config store backed by PostgreSQL.

    All reads come from an in-memory cache; writes go to DB then cache.
    Valkey pub/sub propagates changes across all processes in real time.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._lock  = threading.RLock()
        self._listener_started = False

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _ensure_table(self) -> None:
        """Create config_store table if it does not exist."""
        try:
            from agent.db import get_conn
            with get_conn() as c:
                c.execute(_DDL)
        except Exception as exc:
            logger.warning("[ConfigManager] _ensure_table failed: %s", exc)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Read all rows from config_store into the in-memory cache.
        Also ensures the table exists.
        """
        self._ensure_table()
        try:
            from agent.db import get_conn
            with get_conn() as c:
                rows = c.execute(
                    "SELECT key, value FROM config_store"
                ).fetchall()
            with self._lock:
                for row in rows:
                    try:
                        self._cache[row["key"]] = json.loads(row["value"])
                    except Exception:
                        self._cache[row["key"]] = row["value"]
            logger.info("[ConfigManager] Loaded %d keys from config_store", len(rows))
        except Exception as exc:
            logger.warning("[ConfigManager] load() failed: %s", exc)

    def seed_defaults(self) -> None:
        """
        For each key in _DEFAULTS not yet in cache:
          1. Try to read from legacy account_config table (paper.* keys).
          2. Fall back to the default factory lambda.

        This preserves user-tuned settings across the migration.
        """
        for key, factory in _DEFAULTS.items():
            with self._lock:
                if key in self._cache:
                    continue

            # Try legacy account_config migration for paper.* keys
            value = None
            legacy_col = _LEGACY_COLUMN_MAP.get(key)
            if legacy_col:
                try:
                    from agent.db import get_conn
                    with get_conn() as c:
                        row = c.execute(
                            f"SELECT {legacy_col} FROM account_config WHERE id=1"
                        ).fetchone()
                    if row and row[legacy_col] is not None:
                        value = row[legacy_col]
                        logger.info(
                            "[ConfigManager] Migrated %s from account_config.%s = %r",
                            key, legacy_col, value,
                        )
                except Exception as exc:
                    logger.debug("[ConfigManager] Legacy read for %s failed: %s", key, exc)

            # Fall back to default factory
            if value is None:
                try:
                    value = factory()
                except Exception as exc:
                    logger.warning("[ConfigManager] Default factory for %s failed: %s", key, exc)
                    continue

            # Persist to DB (only if still absent — race guard)
            with self._lock:
                if key in self._cache:
                    continue

            try:
                self.set(key, value, updated_by="system")
            except Exception as exc:
                logger.warning("[ConfigManager] seed_defaults set(%s) failed: %s", key, exc)

    def get(self, key: str, default: Any = None) -> Any:
        """Thread-safe read from in-memory cache."""
        with self._lock:
            return self._cache.get(key, default)

    def set(self, key: str, value: Any, updated_by: str = "system") -> None:
        """Write value to DB, update cache, publish to Valkey."""
        encoded = json.dumps(value)
        ts      = self._now_iso()
        try:
            from agent.db import get_conn
            with get_conn() as c:
                c.execute(_UPSERT, (key, encoded, ts, updated_by))
        except Exception as exc:
            logger.error("[ConfigManager] set(%s) DB error: %s", key, exc)
            raise

        with self._lock:
            self._cache[key] = value

        # Publish to Valkey for hot-reload across processes
        try:
            from agent.valkey_client import _get_client
            client = _get_client()
            if client is not None:
                payload = json.dumps({"key": key, "value": value})
                client.publish(_VALKEY_CHANNEL, payload)
        except Exception as exc:
            logger.debug("[ConfigManager] Valkey publish failed for %s: %s", key, exc)

    def set_many(self, updates: dict[str, Any], updated_by: str = "system") -> None:
        """Atomic multi-key write — one DB transaction, then publish each key."""
        if not updates:
            return
        ts = self._now_iso()
        rows = [(k, json.dumps(v), ts, updated_by) for k, v in updates.items()]
        try:
            from agent.db import get_conn
            with get_conn() as c:
                for row in rows:
                    c.execute(_UPSERT, row)
        except Exception as exc:
            logger.error("[ConfigManager] set_many DB error: %s", exc)
            raise

        with self._lock:
            for k, v in updates.items():
                self._cache[k] = v

        # Publish each changed key
        try:
            from agent.valkey_client import _get_client
            client = _get_client()
            if client is not None:
                pipe = client.pipeline(transaction=False)
                for k, v in updates.items():
                    payload = json.dumps({"key": k, "value": v})
                    pipe.publish(_VALKEY_CHANNEL, payload)
                pipe.execute()
        except Exception as exc:
            logger.debug("[ConfigManager] Valkey publish (set_many) failed: %s", exc)

    def all(self) -> dict[str, Any]:
        """Return a copy of the full in-memory cache."""
        with self._lock:
            return dict(self._cache)

    def _reload_key(self, key: str) -> None:
        """Re-read a single key from DB into cache (called on hot-reload)."""
        try:
            from agent.db import get_conn
            with get_conn() as c:
                row = c.execute(
                    "SELECT value FROM config_store WHERE key = ?", (key,)
                ).fetchone()
            if row:
                with self._lock:
                    try:
                        self._cache[key] = json.loads(row["value"])
                    except Exception:
                        self._cache[key] = row["value"]
                logger.debug("[ConfigManager] Hot-reloaded key: %s", key)
        except Exception as exc:
            logger.debug("[ConfigManager] _reload_key(%s) failed: %s", key, exc)

    def start_listener(self, runner=None) -> None:
        """
        Start a daemon thread that subscribes to Valkey `config:changed` channel.
        On message: parses JSON, extracts `key`, calls `_reload_key(key)`.
        """
        if self._listener_started:
            return
        self._listener_started = True
        t = threading.Thread(
            target=self._listener_loop,
            daemon=True,
            name="ConfigManagerListener",
        )
        t.start()
        logger.info("[ConfigManager] Valkey listener thread started.")

    def _listener_loop(self) -> None:
        """Background daemon: subscribe to config:changed and hot-reload keys."""
        retry_delay = 2.0
        while True:
            try:
                from agent.valkey_client import _cfg as _vcfg
                import redis as _redis_lib
                host, port, ssl = _vcfg()
                sub_client = _redis_lib.Redis(
                    host=host,
                    port=port,
                    ssl=ssl,
                    ssl_cert_reqs=None,
                    socket_connect_timeout=5,
                    socket_timeout=60,
                    decode_responses=False,
                )
                pubsub = sub_client.pubsub()
                pubsub.subscribe(_VALKEY_CHANNEL)
                logger.info("[ConfigManager] Subscribed to channel '%s'", _VALKEY_CHANNEL)
                retry_delay = 2.0  # reset on successful connect

                for message in pubsub.listen():
                    if message.get("type") != "message":
                        continue
                    data = message.get("data", b"")
                    try:
                        payload = json.loads(data)
                        key = payload.get("key")
                        if key:
                            self._reload_key(key)
                    except Exception:
                        pass

            except Exception as exc:
                logger.warning(
                    "[ConfigManager] Listener error: %s — retrying in %.0fs",
                    exc, retry_delay,
                )
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)


# ── Module-level singleton ─────────────────────────────────────────────────────

config = ConfigManager()
