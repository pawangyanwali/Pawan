# NASDAQ Paper-Trading Agent — System Design Document

**Version**: 2.0 · **Updated**: 2026-05-30

---

## 1. Purpose

An ML-driven paper-trading system that continuously scans ~500 NASDAQ stocks, generates entry signals using an ensemble of machine learning models and rule-based algorithms, executes simulated trades with realistic sizing and risk management, and self-improves by learning from every trade outcome. The system is designed to eventually transition from paper to live trading via Schwab's API.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   EC2 t3a.xlarge (4 vCPU, 16 GB RAM)                   │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │   web-api    │  │   scanner    │  │ market-data  │  │  learner   │  │
│  │  FastAPI     │  │  ML engine   │  │ Schwab WS    │  │ Adaptive   │  │
│  │  port 8000   │  │  8 workers   │  │ + REST poll  │  │ + BiLSTM   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  │
│         │                 │                  │                │          │
│  ┌──────┴─────────────────┴──────────────────┴────────────────┴──────┐  │
│  │            Docker Bridge Network  (nasdaq-net)                     │  │
│  └──────┬─────────────────┬──────────────────┬────────────────┬──────┘  │
│         │                 │                  │                │          │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴──────┐  ┌─────┴──────┐  │
│  │  scheduler   │  │context-intel │  │  watchdog   │  │   (EBS)    │  │
│  │  EOD tasks   │  │ news/earnings│  │ auto-restart│  │  models/   │  │
│  └──────────────┘  └──────────────┘  └─────────────┘  │  tokens/   │  │
│                                                         │  cache/    │  │
└─────────────────────────────────────────────────────────────────────────┘
                    │                           │
         ┌──────────┴───────────┐   ┌──────────┴───────────┐
         │   AWS RDS PostgreSQL │   │  AWS ElastiCache      │
         │   (source of truth)  │   │  Valkey (hot cache +  │
         │   trades, signals,   │   │  pub/sub, <100ms)     │
         │   learning state     │   │                       │
         └──────────────────────┘   └───────────────────────┘
                    │
         ┌──────────┴───────────┐
         │  Schwab API          │   + Finnhub (news/earnings)
         │  WebSocket + REST    │
         │  OAuth 2.0           │
         └──────────────────────┘
```

---

## 3. Containers

| Container | CPU | RAM | Purpose |
|-----------|-----|-----|---------|
| `web-api` | 1.0 | 768 MB | FastAPI REST + WebSocket, dashboard, auth, OAuth |
| `scanner` | 2.5 | 3 GB | ML inference, signal generation, paper trading |
| `market-data` | 0.5 | 512 MB | Schwab WebSocket streamer + REST price poller |
| `learner` | 1.5 | 3.5 GB | Adaptive filter, BiLSTM retraining |
| `scheduler` | 0.25 | 256 MB | EOD heartbeat, time-based tasks |
| `context-intel` | 0.5 | 512 MB | Finnhub news/earnings poller |
| `watchdog` | 0.1 | 128 MB | Container health monitor, auto-restart |

### Container flags (env vars)
Each container has service flags that disable subsystems it doesn't own:
- `NASDAQ_MARKET_DATA_ENABLED` — enables Schwab streamer (only in `market-data`)
- `NASDAQ_LEARNER_ENABLED` — enables learning engine (in `scanner` + `learner`)
- `NASDAQ_SCANNER_ENABLED` — enables scan loop (only in `scanner`)

---

## 4. Data Flow

### 4.1 Market Data Pipeline
```
Schwab WebSocket
    ↓ CHART_EQUITY (1-min OHLC, no volume)
    ↓ QUOTE (last price, ~3.3 Hz)
market-data container
    → Valkey md:prices HASH (latest quote per ticker)
    → Valkey md:1m:{ticker} LIST (2h TTL, candles)
    → Pub/sub: md:prices channel (fan-out to scanner + web-api)
```

**Volume note**: WebSocket streaming candles omit volume. The scanner injects `Volume=0.0` when reading from Valkey (Tier A½ path) so that technical indicators don't crash.

### 4.2 Scan Cycle (every 60s, adaptive)
```
1. Fetch ~500 quotes from md:prices (Valkey) or Schwab /quotes REST
2. Filter to active tickers (Tier 1 always, Tier 2/3 top movers)
3. Parallel inference (8 workers):
   a. Fetch OHLCV bars (1m from Valkey, 5m/15m/1d from cache)
   b. compute_indicators() → MACD, RSI, BB, ATR, VWAP, etc.
   c. predict() → XGBoost scalp probability
   d. predict_daily() → XGBoost daily probability
   e. predict_ensemble() → meta-XGBoost blend
   f. predict_swing() → 15-min XGBoost
   g. predict_deep() → BiLSTM regime
   h. get_meta_prediction() → calibrated fusion
4. Rule-based algo evaluation (20+ algos: ORB, GAP, VWAP, FLAG, etc.)
5. Adaptive filter (confidence gate, currently advisory mode)
6. Risk checks (sector limits, heat, drawdown, daily trade cap)
7. maybe_open_trade() → paper trade if gates pass
8. update_open_trades() → T1/T2/stop/EOD exits per ticker
9. Write scan:latest → Valkey + PostgreSQL
10. Publish scan:notify → Valkey pub/sub (wake dashboard WebSocket)
```

### 4.3 Trade Lifecycle
```
Signal fires
  → maybe_open_trade() checks all gates
  → INSERT paper_trades (status=OPEN)
  → Stores ML scores: ml_scalp_prob, ml_daily_prob, ml_swing_prob,
    ml_deep_prob, ml_ensemble_score
  ↓
Trade monitoring (every scan + every 5s RT check)
  → T1 at 1R: close 50% shares, set stop → breakeven
  → T2 at 2R: close remaining 50% shares
  → Stop hit: close all remaining shares
  → Time stop: 20 bars (scalp) or 90 bars (intraday)
  → EOD: hard close ALL at 3:45 PM ET (non-negotiable)
  ↓
_record_close() writes:
  → status=CLOSED, exit_price, exit_reason
  → pnl_dollar = (exit - entry) × close_shares + partial_pnl
  → pnl_pct = pnl_dollar / (entry × total_shares) × 100
  ↓
Publish trade:closed → Valkey (triggers feedback loop)
```

### 4.4 Learning Feedback Loop
```
trade:closed published to Valkey
  ↓ LearningFeedback thread (daemon, subscribed 24/7)
  → _run_trade_feedback(trade_data)
  → LearningEngine.run_cycle() for the specific algo family
  → param_tune_log updated with changes + reason
  → algo_params updated with new values
  ← Confirmed in <5s from trade close
```

---

## 5. Database Schema

### Core Trading Tables

**`paper_trades`**
```
id, opened_at (TEXT ISO-8601), closed_at (TEXT),
ticker, direction (BUY/SELL), entry_price, target, stop,
confidence, rr_ratio, rr_qualifies,
shares, shares_remaining, t1_hit, t1_price, t2_price,
partial_pnl_dollar, breakeven_set,
session, regime, vwap_event, rsi_zone, entry_type,
order_flow_score, size_mult, cost_basis, algo_name,
ml_scalp_prob, ml_daily_prob, ml_swing_prob, ml_deep_prob,
ml_ensemble_score, feedback_triggered_at,
bars_held, status (OPEN/CLOSED), exit_price, exit_reason,
pnl_pct, pnl_dollar
```

⚠️ **Important**: `opened_at` and `closed_at` are TEXT, not TIMESTAMPTZ.
Always guard before casting: `col IS NOT NULL AND col != '' AND col::TIMESTAMPTZ`

**`algo_signal_log`**
```
id, logged_at (TEXT), ticker, algo, direction,
confidence, entry, stop, target, rr,
trade_opened (INTEGER 0/1),
ml_scalp_prob, ml_daily_prob, ml_swing_prob, ml_deep_prob,
filter_reason
```

**`param_tune_log`**
```
id, tuned_at (TEXT), algo, family, param_name,
old_value, new_value, reason, performance,
trigger_trade_id, trigger_ms
```

**`algo_params`** — current tunable values (5 params per family)
```
id, algo, family, param_name, value,
is_tuning, min_val, max_val, updated_at
```

**`service_state`** — durable container heartbeats (PostgreSQL JSONB)
```
key (PK), value (JSONB), updated_at (TIMESTAMPTZ), expires_at (TIMESTAMPTZ)
Keys: scan:latest, learner:status, scheduler:heartbeat, scanner:streamer
```

⚠️ **JSONB gotcha**: `json.dumps` serializes `float('nan')` as `NaN` (invalid JSON).
Always call `_sanitize_for_jsonb(value)` before writing to this table.

### Auth Tables
`users`, `refresh_tokens`, `audit_log`, `mfa_pending`

### Context Tables
`earnings_calendar`, `context_events`, `ticker_context_features`

### Learning Tables
`balance_snapshots`, `account_config`, `config_store`, `backtest_*`

---

## 6. Valkey (Redis) Layout

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `md:prices` | HASH | none | Latest L1 quote per ticker |
| `md:1m:{ticker}` | LIST | 2h | 1-min candle OHLC (no volume) |
| `scan:latest` | STRING | none | Full scan snapshot (also in PG) |
| `learner:status` | STRING | 300s | Learning engine state |
| `scheduler:heartbeat` | STRING | 90s | Scheduler alive indicator |
| `scanner:streamer` | STRING | 60s | Streamer connection state |

**Pub/sub channels** (no persistence):
- `md:prices` — live price fan-out
- `scan:notify` — lightweight wake for WebSocket
- `trade:closed` — triggers immediate feedback loop
- `schwab:tokens_refreshed` — OAuth token rotation
- `config:updated` — hot-reload config changes

---

## 7. ML Model Stack

| Model | Type | Timeframe | Purpose |
|-------|------|-----------|---------|
| Scalp (per-ticker) | XGBoost | 1-min | Short-term direction probability |
| Daily | XGBoost | Daily | Multi-day trend probability |
| Reversal | XGBoost | 1-min | Mean-reversion detection |
| Swing | XGBoost | 15-min | 2h-ahead probability |
| Deep BiLSTM | Seq2Seq LSTM | 15-min | Regime classification |
| MetaEnsemble | XGBoost | — | Calibrated fusion of all above |
| SignalBlender | Weighted avg | — | Dynamic weight mixing (rolling 20-trade accuracy) |

**Retraining**:
- Scalp/Daily/Reversal/Swing: triggered when 15+ new trade outcomes accumulate
- BiLSTM: weekend learner only, gated behind `is_market_hours() == False`
- MetaEnsemble: retrains with scalp/daily (minimum 30 outcomes)

**Untrained fallback**: Each model returns 0.5 until trained. `get_meta_prediction()` uses weighted average fallback when MetaEnsemble is untrained, never returns 0.0 or fails silently.

---

## 8. Trading Algorithms (20+ strategies)

Organized into families with tunable parameters:

| Family | Examples | Description |
|--------|----------|-------------|
| `ORB` | ORB5_BULL, ORB15_BULL | Opening range breakout |
| `GAP_TREND` | GAP_AND_GO_BULL | Gap-and-go trend follow |
| `GAP_FADE` | GAP_FADE_BULL | Fade overextended gap opens |
| `BREAKOUT` | PDH_BREAKOUT, HOD_BREAK | Prior day high/low breakouts |
| `FLAG` | BULL_FLAG, BEAR_FLAG | Flag pattern continuation |
| `VWAP_SCALP` | VWAP_TOUCH_SCALP | VWAP bounce/rejection scalps |
| `LEVEL_SCALP` | LEVEL_REJECTION, MICRO_PULLBACK | Key level scalps |
| `RS_REGIME` | SPY_BETA_CATCHUP, SECTOR_LEADER | Relative strength plays |
| `OFI` | OFI_IMPULSE | Order flow imbalance |
| `EMA_PULL` | EMA_SLOPE_PULL | EMA pullback continuation |
| `BB_MEAN_REV` | BB_MEAN_REV_BULL/BEAR | Bollinger Band mean reversion |

**Tunable parameters per family** (stored in `algo_params`):
- `conf_gate` — minimum confidence to fire
- `stop_mult` — stop-loss multiplier
- `rr_min` — minimum risk/reward ratio
- `size_mult` — position size multiplier
- `cooldown_bars` — bars between signals

---

## 9. Risk Management

### Position-Level
- Stop at user-defined price (adjusted by ATR-based `stop_mult`)
- T1 partial exit at 1R → locks 50% profit, stop moves to breakeven ± $0.02
- T2 full exit at 2R
- Time stop: 20 bars (scalp entries), 90 bars (intraday entries)
- Hard EOD close at 3:45 PM ET

### Portfolio-Level
- Max 5% budget per trade (`paper.max_trade_pct`)
- Max 40% total allocated capital (`paper.max_allocated_pct`)
- Max 10 concurrent open trades (`paper.max_open_trades`)
- Max 30 trades per day (`MAX_DAILY_TRADES`)
- Max 5 consecutive losses → circuit breaker (`MAX_CONSECUTIVE_LOSSES`)
- Daily loss halt at 2.5% drawdown (`DAILY_LOSS_HALT_PCT`)
- Sector concentration limits

### Session Rules
| Session | New Trades | Existing Trades |
|---------|-----------|-----------------|
| PRE_MARKET | HIGH/MODERATE tier only, 1.5× stop | Monitor only |
| REGULAR | All tiers | Normal management |
| CLOSING_CAUTION (3:30–3:44) | No new scalps | Smart exit (lock winners, cut losers) |
| HARD_CLOSE (3:45–4:00) | None | Force-close ALL |
| AFTER_HOURS | HIGH/MODERATE tier only, 2× stop | AH trades stay open |
| AH_EOD (7:55 PM) | None | Force-close ALL remaining |

---

## 10. API Surface

13 FastAPI routers. Key endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /api/signals` | Current scan results (dashboard) |
| `GET /api/paper-trading` | Open + closed trades |
| `GET /api/account-state` | Equity, P&L, heat |
| `GET /api/learning-status` | Win-rate, gate, context blocks |
| `GET /api/algo/overview` | KPI strip (evals, fires, tunes, gate) |
| `GET /api/algo/leaderboard` | Per-algo performance ranking |
| `GET /api/algo/tune-log` | Parameter tuning history with post-tune WR |
| `GET /api/algo/ml-influence` | ML scores per closed trade |
| `GET /api/algo/trade-attribution` | Trade breakdown by algo/session/regime |
| `GET /api/algo/loss-heatmap` | algo-family × root-cause matrix |
| `POST /api/config` | Hot-reload any runtime parameter |
| `GET /stream/signals` | WebSocket real-time feed |

**Auth**: JWT (HS256) + refresh tokens. Roles: `admin > trader > analyst > viewer`.

---

## 11. Configuration System

**Two-tier design:**

1. **Secrets** (`.env`, never in DB):
   - `JWT_SECRET`, `SCHWAB_*`, `PGPASSWORD`, `FINNHUB_API_KEY`

2. **Runtime config** (`config_store` PostgreSQL table, hot-reloadable):
   - Paper trading limits, risk thresholds, scanner cadence, learner intervals
   - Changed via `POST /api/config` → written to PG → published to `config:updated` Valkey channel → all containers reload within 1s (no restart required)

---

## 12. Monitoring & Observability

**Container health**: Docker HEALTHCHECK on each container + watchdog auto-restart.

**Service-level health** (`GET /api/services`):
- Reads `service_state` table keys for each container
- Flags: HEALTHY (fresh heartbeat), STALE (old), STARTING, STOPPED

**Scanner health** (`services/scanner_healthcheck.py`):
- Checks `scan:latest` age against session-aware thresholds (180s REGULAR, 300s AH/PM)
- Valkey fallback if PostgreSQL `scan:latest` is stale
- Guards against pre-deployment data flagging as current

**Dashboards**:
- `index.html` — main trading dashboard (signals, trades, P&L, regime, ML, infra)
- `algo.html` — Algorithm Intelligence Center (10 panels: leaderboard, tune log, loss heatmap, params, ML influence, signal feed, adaptive filter, attribution)
- `admin.html` — user management, config editor, audit log

---

## 13. Known Design Decisions & Tradeoffs

### Adaptive Filter in "Observe" Mode
`ADAPTIVE_FILTER_ENFORCEMENT_MODE` defaults to `"observe"`. The filter computes and stores dynamic thresholds but **does not block trades**. This is intentional during the data-collection phase — blocking low-confidence signals would starve the learner. The gate value shown in the dashboard (currently 55%) is advisory. To enforce it, set `ADAPTIVE_FILTER_ENFORCEMENT_MODE=enforce`.

### TEXT Timestamps
`opened_at`, `closed_at`, `logged_at`, `tuned_at` are stored as ISO-8601 TEXT strings (not native TIMESTAMPTZ). This was chosen for SQLite compatibility during early development. **Always add `IS NOT NULL AND != ''` guards before any `::TIMESTAMPTZ` cast** to avoid 500 errors on malformed rows.

### SQLite `?` Placeholders
`paper_trading.py` uses SQLite-style `?` placeholders. The `_PgConnection` wrapper transparently translates `?` → `%s`. This is a legacy artifact from the SQLite-to-PostgreSQL migration. All new code should use `%s` directly.

### Valkey Candles Missing Volume
Schwab's `CHART_EQUITY` WebSocket messages do not include volume in the streaming candle data. The scanner injects `Volume=0.0` when building DataFrames from Valkey candles. Technical indicators that use volume (OBV, VWAP) are computed correctly using the REST-fetched historical data; only the live streaming candles are volume-less.

### Paper Trading as Data Collection Layer
The confidence floor is intentionally low (25%) so the learning system sees a wide range of outcomes. The adaptive filter's dynamic threshold (55%) governs what the system *would* recommend for live trading, but paper trades are opened at 25%+ to collect training data across confidence bands.

---

## 14. Bug Registry (Closed)

All issues found in the 2026-05-30 audit are fixed in commit `c70ef1f`:

| # | Severity | Component | Issue | Status |
|---|----------|-----------|-------|--------|
| C-1 | CRITICAL | adaptive_filter.py | NameError on `new_blocked`/`new_boosted` — filter never persisted, threshold frozen | Fixed |
| H-1 | HIGH | paper_trading.py | `rt_check_positions` swapped `close_shares`/`total_shares` — P&L overstated | Fixed |
| H-2 | HIGH | paper_trading.py | `close_stale_positions` built fake IDs from `range(n)` — latent | Fixed |
| H-3 | HIGH | paper_trading.py | `closed_at::date` cast without NULL guard in P&L queries | Fixed |
| H-4 | HIGH | routers/algo.py | SQL injection in `_build_tune_log` CTE VALUES f-string | Fixed |
| H-5 | HIGH | routers/algo.py | `logged_at`/`tuned_at` casts without NULL guard in overview | Fixed |
| M-1 | MEDIUM | scanner.py | `if/if/elif` allowed AH_EOD + stale sweep to double-fire | Fixed |
| M-2 | MEDIUM | learning_engine.py | `feedback_loop_active` checked `_running` not thread liveness | Fixed |
| M-5 | MEDIUM | routers/algo.py | Post-tune CTE joined `algo_name = family` — always 0 matches | Fixed |

Open (design decisions, not bugs):
- M-3: SQLite `?` placeholders throughout paper_trading.py (transparent layer handles it)
- M-4: Adaptive filter defaults to `observe` mode (intentional — see §13)
- L-1: `_cycle_count` unguarded across threads (cosmetic, GIL-safe)
- L-2: `ensemble_agreement` excluded from `prob_std` fallback (marginal accuracy impact)
