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
    "paper.algo_min_rr":                   lambda: 1.0,    # legacy; R:R no longer hard-gates algo-family paper trades
    "paper.breakeven_stop_offset":         lambda: 0.02,   # $ offset above entry for T1 breakeven stop
    # EOD management
    "paper.eod_trail_stop_pct":            lambda: 0.003,  # 0.3% trailing stop for EOD winners
    "paper.eod_recovery_stop_pct":         lambda: 0.002,  # 0.2% recovery stop for EOD losers with momentum
    "paper.eod_strong_winner_pct":         lambda: 0.5,    # P&L% threshold for "strong winner" EOD path
    "paper.eod_small_winner_pct":          lambda: 0.1,    # P&L% threshold for "small winner" EOD path
    "paper.eod_loss_threshold_pct":        lambda: -0.3,   # P&L% below which position is a "meaningful loss"
    # Adaptive filter feedback
    "paper.filter_feedback_min_trades":    lambda: 5,      # min closed trades before feeding back to filter
    # Realistic paper execution guardrails. Paper mode still logs blocked signals
    # for learning, but simulated capital must obey the same brakes live trading
    # would use.
    "paper.enforce_risk_controls":         lambda: True,
    "paper.daily_loss_halt_usd":           lambda: 300.0,
    "paper.daily_loss_halt_pct":           lambda: 0.25,
    "paper.max_daily_trades":              lambda: 75,
    # Greenfield scalping engine (Release 1 shadow mode). These settings stay
    # isolated from legacy execution settings until the controlled cutover.
    "scalp.shadow_enabled":                lambda: True,
    "scalp.execution_enabled":             lambda: False,
    "scalp.reward_r":                      lambda: 2.0,
    "scalp.tp1_r":                         lambda: 1.0,
    "scalp.stop_atr_multiple":             lambda: 1.0,
    "scalp.min_stop_pct":                  lambda: 0.003,
    "scalp.max_stop_pct":                  lambda: 0.020,
    "scalp.spread_buffer_mult":            lambda: 2.0,
    "scalp.tick_size":                     lambda: 0.01,
    "scalp.max_quote_age_ms":              lambda: 2000,
    "scalp.max_bar_age_ms":                lambda: 120000,
    "scalp.max_spread_to_risk":            lambda: 0.25,
    "scalp.min_rvol_regular":              lambda: 0.8,
    "scalp.min_rvol_extended":             lambda: 0.4,
    "scalp.rsi_oversold":                  lambda: 30.0,
    "scalp.rsi_extreme_oversold":          lambda: 20.0,
    "scalp.rsi_overbought":                lambda: 70.0,
    "scalp.rsi_extreme_overbought":        lambda: 80.0,
    "scalp.require_vwap_event":            lambda: True,
    "scalp.require_macd_confirm":          lambda: True,
    "scalp.require_rsi_zone":              lambda: True,
    "scalp.allow_rest_fallback_trading":   lambda: False,
    "scalp.block_when_path_obstructed":    lambda: True,
    "scalp.block_when_risk_capped":        lambda: True,
    "scalp.mtf_enabled":                   lambda: True,
    "scalp.mtf_mode":                      lambda: "SHADOW",
    "scalp.mtf_max_bar_age_ms":            lambda: 420000,
    "scalp.momentum_shadow_enabled":       lambda: True,
    "scalp.reversal_shadow_enabled":       lambda: True,
    "scalp.momentum_long_rsi_min":         lambda: 45.0,
    "scalp.momentum_long_rsi_max":         lambda: 70.0,
    "scalp.momentum_short_rsi_min":        lambda: 30.0,
    "scalp.momentum_short_rsi_max":        lambda: 55.0,
    "scalp_runtime.cycle_interval_s":       lambda: 5.0,
    "scalp_runtime.workers":                lambda: 8,
    "scalp_runtime.bar_lookback":           lambda: 500,
    "scalp_runtime.blocked_sessions":       lambda: [
        "CLOSED", "RESTRICTED", "CLOSING_CAUTION", "HARD_CLOSE"
    ],
    # Shared execution preflight. Plans remain visible for research, while only
    # policy-approved candidates may enter shadow or canonical paper execution.
    "scalp_runtime.execution_policy_enabled": lambda: True,
    "scalp_runtime.execution_blocked_sessions": lambda: [
        "CLOSED", "RESTRICTED", "LUNCH_BLOCK", "CLOSING_CAUTION", "HARD_CLOSE"
    ],
    "scalp_runtime.require_live_execution_data": lambda: True,
    "scalp_runtime.pre_market_size_mult":   lambda: 0.35,
    "scalp_runtime.restricted_size_mult":   lambda: 0.0,
    "scalp_runtime.prime_size_mult":        lambda: 1.0,
    "scalp_runtime.lunch_size_mult":        lambda: 0.0,
    "scalp_runtime.standard_size_mult":     lambda: 0.80,
    "scalp_runtime.closing_size_mult":      lambda: 0.0,
    "scalp_runtime.after_hours_size_mult":  lambda: 0.30,
    "scalp_runtime.hard_close_size_mult":   lambda: 0.0,
    "scalp_runtime.closed_size_mult":       lambda: 0.0,
    "scalp_runtime.position_max_quote_age_ms": lambda: 5_000,
    "scalp_runtime.require_context_data":   lambda: True,
    "scalp_runtime.max_context_age_s":      lambda: 180.0,
    "scalp_runtime.max_context_risk_score": lambda: 0.8,
    "scalp_runtime.adverse_news_sentiment": lambda: 0.25,
    # Bounded scalp outcome learning. Actions only tighten execution and expire.
    "scalp_learn.enabled":                  lambda: True,
    "scalp_learn.shadow_outcomes_enabled":  lambda: True,
    "scalp_learn.rolling_window_min":       lambda: 120,
    "scalp_learn.min_samples_to_adjust":    lambda: 3,
    "scalp_learn.min_samples_to_block":     lambda: 12,
    "scalp_learn.ewma_alpha":               lambda: 0.25,
    "scalp_learn.negative_reduce_r":        lambda: -0.05,
    "scalp_learn.negative_block_r":         lambda: -0.20,
    "scalp_learn.block_win_rate":           lambda: 0.40,
    "scalp_learn.confidence_win_rate":      lambda: 0.48,
    "scalp_learn.base_confidence_floor":    lambda: 60.0,
    "scalp_learn.confidence_raise_step":    lambda: 10.0,
    "scalp_learn.size_reduce_mult":         lambda: 0.50,
    "scalp_learn.fast_stop_circuit_enabled": lambda: True,
    "scalp_learn.fast_stop_window_min":     lambda: 10,
    "scalp_learn.fast_stop_count":          lambda: 3,
    "scalp_learn.fast_stop_size_mult":      lambda: 0.25,
    "scalp_learn.action_ttl_min":           lambda: 60,
    # Release 4 advisory ML. Auto-arm waits for the sample floor, then begins
    # isolated challenger training without requiring a later operator action.
    "scalp_ml.training_enabled":             lambda: False,
    "scalp_ml.auto_train_when_ready":         lambda: True,
    "scalp_ml.shadow_enabled":               lambda: False,
    "scalp_ml.overlay_enabled":              lambda: False,
    "scalp_ml.training_interval_min":        lambda: 60,
    "scalp_ml.training_lookback_days":       lambda: 60,
    "scalp_ml.maximum_model_age_hours":      lambda: 168,
    "scalp_ml.minimum_samples":              lambda: 200,
    "scalp_ml.holdout_pct":                  lambda: 0.25,
    "scalp_ml.minimum_selected_holdout":     lambda: 30,
    "scalp_ml.minimum_holdout_sessions":     lambda: 2,
    "scalp_ml.minimum_session_samples":      lambda: 5,
    "scalp_ml.minimum_expectancy_r":         lambda: 0.05,
    "scalp_ml.minimum_profit_factor":        lambda: 1.10,
    "scalp_ml.minimum_session_expectancy_r": lambda: 0.0,
    "scalp_ml.minimum_auc":                  lambda: 0.52,
    "scalp_ml.minimum_brier_improvement":    lambda: 0.0,
    "scalp_ml.selection_expected_r":         lambda: 0.0,
    "scalp_ml.confidence_points_per_r":      lambda: 5.0,
    "scalp_ml.max_confidence_raise":         lambda: 5.0,
    "scalp_ml.max_confidence_reduction":     lambda: 15.0,
    # ── Prediction / R:R engine ────────────────────────────────────────────────
    # These control the trade-entry quality filter. All hot-reload — no restart needed.
    "prediction.min_rr":                   lambda: 1.5,    # target reward multiple (e.g. 2.0 = risk 1, reward 2); not a signal gate
    "prediction.min_stop_dist_pct":        lambda: 0.004,  # stop must be ≥ this % from entry (avoids noise stops)
    "prediction.max_risk_pct":             lambda: 0.020,  # cap risk at this % of stock price per scalp
    "prediction.min_target_pct":           lambda: 0.003,  # target must be ≥ this % from entry
    # ATR-based stop mode (recommended for automated scalping — eliminates T2/target gap)
    # When true: stop = ATR × stop_atr_multiple, target = entry + t2_r_multiple × risk,
    #            structure used only as filter (trade rejected if resistance blocks path).
    # When false: legacy structural mode (stop at support, target at resistance).
    "prediction.use_atr_stops":            lambda: True,   # true = fixed R:R (recommended); false = structure-based
    "prediction.stop_atr_multiple":        lambda: 1.0,    # stop distance = N × ATR(14). 1.0 = 1×ATR is standard scalp stop
    "prediction.min_stop_daily_atr_pct":   lambda: 0.05,   # floor: stop ≥ 5% of daily ATR(14) — prevents 6-cent pre-market stops
    # T1/T2 exit multipliers — both expressed as multiples of the initial risk distance.
    # T1 is the partial-exit level (take 50% off, move stop to breakeven).
    # T2 is the full-exit target. Setting t2_r_multiple = prediction.min_rr makes
    # T2 exactly equal to the minimum R:R target — no gap between filter and exit.
    "paper.t1_r_multiple":                 lambda: 1.0,    # T1 = entry + 1× risk_dist
    "paper.t2_r_multiple":                 lambda: 1.5,    # T2 = 1.5R — more achievable in choppy sessions
    "paper.algo_t2_r_multiple":            lambda: 1.5,    # T2 for named algo-family paper trades; primary predictions use paper.t2_r_multiple
    # Time stops: hard-close positions after N bars if still open
    "paper.max_bars_scalp":                lambda: 20,     # 20-min hard close for scalp trades
    "paper.max_bars_intraday":             lambda: 90,     # 90-min hard close for intraday trades
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
    # Concurrency cap for the parallel scan ThreadPoolExecutor. This is the de-facto
    # ML-inference CPU bound: at most N tickers are analysed at once, and each runs
    # its 6 models single-threaded (XGBoost nthread=1, torch 1+1 threads). Lives in
    # PostgreSQL so it can be dialed down live on a smaller host without a redeploy.
    # Clamped 1–32 to prevent a typo from oversubscribing the scanner's 2.5 vCPU.
    "scanner.pipeline_workers":            lambda: max(1, min(32, int(os.getenv("PIPELINE_WORKERS", "8")))),
    "scanner.analysis_batch_size":         lambda: max(1, min(600, int(os.getenv("NASDAQ_SCAN_ANALYSIS_BATCH_SIZE", "96")))),
    "scanner.ticker_timeout_s":            lambda: float(os.getenv("NASDAQ_SCAN_TICKER_TIMEOUT_S", "45")),
    "scanner.cycle_budget_s":              lambda: float(os.getenv("NASDAQ_SCAN_CYCLE_BUDGET_S", "20")),
    "scanner.slow_ticker_cooldown_s":      lambda: float(os.getenv("NASDAQ_SCAN_SLOW_TICKER_COOLDOWN_S", "300")),
    "scanner.pre_earnings_blackout_days":  lambda: 3,
    "scanner.post_earnings_cooldown_days": lambda: 1,
    "macro.enabled":                       lambda: True,
    "macro.high_hard_block_minutes":       lambda: 30.0,
    "macro.high_throttle_hours":           lambda: 4.0,
    "macro.high_throttle_size_mult":       lambda: 0.35,
    "macro.high_throttle_conf_bump":       lambda: 15.0,
    "macro.high_throttle_min_conf":        lambda: 72.0,
    "macro.medium_throttle_hours":         lambda: 2.0,
    "macro.medium_throttle_size_mult":     lambda: 0.65,
    "macro.medium_throttle_conf_bump":     lambda: 7.0,
    "macro.medium_throttle_min_conf":      lambda: 62.0,
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
    "risk.max_daily_trades":               lambda: int(os.getenv("MAX_DAILY_TRADES", "500")),
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
    # ── Execution safety gates ─────────────────────────────────────────────────
    "paper.block_restricted_session":       lambda: True,   # block paper exec during 9:30-9:44 ET price discovery
    # ── Post-T1 profit-lock and trailing stop ──────────────────────────────────
    "paper.t1_profit_lock_r":              lambda: 0.20,   # after T1: stop = entry + 0.20R (locks small profit)
    "paper.post_t1_trail_r":               lambda: 0.40,   # trail stop = water_mark − 0.40R after T1
    # ── Pre-T1 stop-hit storm circuit ──────────────────────────────────────────
    "risk.pre_t1_storm_enabled":            lambda: True,
    "risk.pre_t1_storm_window_min":         lambda: 30,     # rolling window in minutes
    "risk.pre_t1_storm_max_hits":           lambda: 8,      # max pre-T1 stop hits before circuit trips
    "risk.pre_t1_storm_loss_usd":           lambda: 250.0,  # max rolling pre-T1 dollar loss before circuit trips
    "risk.pre_t1_storm_rate":              lambda: 0.60,    # max pre-T1 hit rate (60%) before circuit trips
    # ── Rolling EV adaptive confidence floor ──────────────────────────────────
    "risk.rolling_ev_enabled":              lambda: True,
    "risk.rolling_ev_window_min":           lambda: 120,    # rolling window in minutes
    "risk.rolling_ev_min_trades":           lambda: 5,      # min trades before suppression kicks in
    "risk.rolling_ev_suppress_threshold":   lambda: -2.0,   # avg $/trade below which to suppress
    "risk.rolling_ev_conf_bump":            lambda: 15.0,   # additional confidence pts required
    "risk.rolling_ev_hard_block":           lambda: True,   # negative EV blocks, not just confidence-bumps
    # Family-level damage stop: DB-backed intraday kill switch. Signals continue
    # to be logged for learning, but new paper executions stop once a family proves
    # it is hurting the session.
    "risk.family_damage_enabled":           lambda: True,
    "risk.family_damage_min_trades":        lambda: 3,
    "risk.family_damage_max_losses":        lambda: 3,
    "risk.family_damage_loss_usd":          lambda: 100.0,
    "risk.family_damage_min_win_rate":      lambda: 30.0,
    "risk.family_damage_scope_session":     lambda: False,
    "risk.fast_family_damage_enabled":      lambda: True,
    "risk.fast_family_loss_window_min":     lambda: 20,
    "risk.fast_family_max_losses":          lambda: 2,
    "risk.fast_family_loss_usd":            lambda: 75.0,
    "risk.family_max_open_per_direction":   lambda: 1,
    "risk.family_open_scope_session":       lambda: False,
    "risk.post_auth_quarantine_min":        lambda: 20,
    "risk.intraday_max_stop_pct":           lambda: 2.0,
    "risk.intraday_max_target_pct":         lambda: 4.0,
    "risk.first_loss_probation_enabled":     lambda: True,
    "risk.first_loss_probation_window_min":  lambda: 60,
    "risk.first_loss_probation_sessions":    lambda: "PRE_MARKET,LUNCH_BLOCK",
    "risk.first_loss_probation_min_ensemble": lambda: 55,
    "risk.first_loss_probation_conf_bump":   lambda: 8.0,
    "risk.first_loss_probation_size_mult":   lambda: 0.50,
    "risk.kc_fade_bear_lunch_block":         lambda: True,
    "risk.pred_immediate_premarket_min_ensemble": lambda: 55,
    "risk.pred_immediate_premarket_min_conf": lambda: 80.0,
    "risk.technical_entry_gate_enabled":     lambda: True,
    "risk.technical_entry_gate_missing_data_block": lambda: True,
    "risk.technical_entry_gate_require_atr": lambda: True,
    "risk.technical_entry_gate_require_rsi": lambda: True,
    "risk.technical_entry_gate_require_macd": lambda: True,
    "risk.technical_entry_gate_buy_rsi_zones": lambda: "OS,EXTREME_OS",
    "risk.technical_entry_gate_sell_rsi_zones": lambda: "OB,EXTREME_OB",
    "risk.late_day_min_minutes_to_eod":     lambda: 35,
    "risk.target_reach_atr_fraction":       lambda: 0.35,
    "risk.deep_saturation_guard_enabled":   lambda: True,
    "risk.deep_saturation_prob":            lambda: 0.98,
    "risk.deep_saturation_min_ensemble":    lambda: 60,
    "risk.max_entry_spread_to_risk":        lambda: 0.35,
    # ── Flash-stop guard (sub-60-second stop hits) ────────────────────────────
    "risk.flash_stop_enabled":              lambda: True,
    "risk.flash_stop_seconds":              lambda: 60,     # stop within N seconds = flash stop
    "risk.flash_stop_max_per_family":       lambda: 3,      # max flash stops per window before gate trips
    "risk.flash_stop_window_min":           lambda: 30,     # rolling window for flash stop count
    # ── Profit-aware daily loss cap ────────────────────────────────────────────
    "risk.daily_loss_trailing_days":        lambda: 5,      # trailing days to measure profit cushion
    "risk.daily_loss_profit_fraction":      lambda: 0.50,   # max daily loss = min(halt_pct, this × trailing_profit)
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
    "learner.feedback_retrain_min_new":     lambda: 15,
    "learner.feedback_retrain_cooldown_s":  lambda: 600,
    # Outcome-driven parameter canaries. A bounded candidate is tested in paper
    # execution, then promoted or rolled back using actual post-activation P&L.
    "learner.param_canary_enabled":         lambda: True,
    "learner.param_canary_min_baseline":    lambda: 15,
    "learner.param_canary_min_outcomes":    lambda: 10,
    "learner.param_canary_min_pf":          lambda: 1.05,
    "learner.param_canary_min_expectancy":  lambda: 0.0,
    "learner.param_canary_min_improvement": lambda: 0.05,
    "learner.param_canary_max_drawdown_mult": lambda: 1.25,
    # Model champion/challenger economic gates.
    "learner.model_min_trades":             lambda: 30,
    "learner.model_min_profit_factor":      lambda: 1.10,
    "learner.model_min_expectancy":         lambda: 0.0,
    "learner.model_min_sharpe_improvement": lambda: 0.05,
    "learner.model_max_drawdown_mult":      lambda: 2.0,
    # ── Adaptive filter ────────────────────────────────────────────────────────
    "filter.throttle_start_wr":            lambda: 0.50,
    "filter.max_penalty_pts":              lambda: 35,
    # ── Audit log (decision trail) ──────────────────────────────────────────────
    # Append-only PostgreSQL audit_log: suppression decisions, threshold changes.
    "audit.enabled":                       lambda: os.getenv("AUDIT_ENABLED", "1").lower() in ("1", "true", "yes"),
    "audit.retention_days":                lambda: max(0, int(os.getenv("AUDIT_RETENTION_DAYS", "14"))),
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
    # ── Per-family paper execution controls ────────────────────────────────────
    # exec_enabled: allow paper trades from this family (signals still logged for learning)
    # exec_size_mult: family-level size multiplier applied on top of global sizing
    # exec_min_conf: family override for min confidence (0 = use global paper.min_confidence)
    # exec_min_rr: family target-R override (0 = use global prediction.min_rr)
    # exec_block_sessions: comma-separated sessions to block (e.g. "RESTRICTED,PRE_MARKET")
    "algos.bb_rev.exec_enabled":            lambda: True,
    "algos.bb_rev.exec_size_mult":          lambda: 0.15,   # de-risked: 15% size until positive expectancy proven
    "algos.bb_rev.exec_min_conf":           lambda: 88.0,   # higher bar than global (de-risked)
    "algos.bb_rev.exec_min_rr":             lambda: 1.8,    # higher target-R than global when configured lower
    "algos.bb_rev.exec_block_sessions":     lambda: "RESTRICTED",
    "algos.macd_acc.exec_enabled":          lambda: True,
    "algos.macd_acc.exec_size_mult":        lambda: 1.0,
    "algos.macd_acc.exec_min_conf":         lambda: 0.0,
    "algos.macd_acc.exec_min_rr":           lambda: 0.0,
    "algos.macd_acc.exec_block_sessions":   lambda: "",
    "algos.supertrend.exec_enabled":        lambda: True,
    "algos.supertrend.exec_size_mult":      lambda: 1.0,
    "algos.supertrend.exec_min_conf":       lambda: 0.0,
    "algos.supertrend.exec_min_rr":         lambda: 0.0,
    "algos.supertrend.exec_block_sessions": lambda: "",
    "algos.orb_zv.exec_enabled":            lambda: True,
    "algos.orb_zv.exec_size_mult":          lambda: 1.0,
    "algos.orb_zv.exec_min_conf":           lambda: 0.0,
    "algos.orb_zv.exec_min_rr":             lambda: 0.0,
    "algos.orb_zv.exec_block_sessions":     lambda: "",
    "algos.rsi2_snap.exec_enabled":         lambda: True,
    "algos.rsi2_snap.exec_size_mult":       lambda: 1.0,
    "algos.rsi2_snap.exec_min_conf":        lambda: 0.0,
    "algos.rsi2_snap.exec_min_rr":          lambda: 0.0,
    "algos.rsi2_snap.exec_block_sessions":  lambda: "",
    "algos.ema_pull.exec_enabled":          lambda: True,
    "algos.ema_pull.exec_size_mult":        lambda: 1.0,
    "algos.ema_pull.exec_min_conf":         lambda: 0.0,
    "algos.ema_pull.exec_min_rr":           lambda: 0.0,
    "algos.ema_pull.exec_block_sessions":   lambda: "",
    "algos.vwap_trend.exec_enabled":        lambda: True,
    "algos.vwap_trend.exec_size_mult":      lambda: 1.0,
    "algos.vwap_trend.exec_min_conf":       lambda: 0.0,
    "algos.vwap_trend.exec_min_rr":         lambda: 0.0,
    "algos.vwap_trend.exec_block_sessions": lambda: "",
    "algos.donchian.exec_enabled":          lambda: True,
    "algos.donchian.exec_size_mult":        lambda: 1.0,
    "algos.donchian.exec_min_conf":         lambda: 0.0,
    "algos.donchian.exec_min_rr":           lambda: 0.0,
    "algos.donchian.exec_block_sessions":   lambda: "",
    "algos.vol_shock.exec_enabled":         lambda: True,
    "algos.vol_shock.exec_size_mult":       lambda: 1.0,
    "algos.vol_shock.exec_min_conf":        lambda: 0.0,
    "algos.vol_shock.exec_min_rr":          lambda: 0.0,
    "algos.vol_shock.exec_block_sessions":  lambda: "",
    "algos.keltner.exec_enabled":           lambda: True,
    "algos.keltner.exec_size_mult":         lambda: 1.0,
    "algos.keltner.exec_min_conf":          lambda: 0.0,
    "algos.keltner.exec_min_rr":            lambda: 0.0,
    "algos.keltner.exec_block_sessions":    lambda: "",
    "algos.meta_ens.exec_enabled":          lambda: True,
    "algos.meta_ens.exec_size_mult":        lambda: 1.0,
    "algos.meta_ens.exec_min_conf":         lambda: 0.0,
    "algos.meta_ens.exec_min_rr":           lambda: 0.0,
    "algos.meta_ens.exec_block_sessions":   lambda: "",
    "algos.regime_sw.exec_enabled":         lambda: True,
    "algos.regime_sw.exec_size_mult":       lambda: 1.0,
    "algos.regime_sw.exec_min_conf":        lambda: 0.0,
    "algos.regime_sw.exec_min_rr":          lambda: 0.0,
    "algos.regime_sw.exec_block_sessions":  lambda: "",
    # ── Execution realism (Phase 6) ────────────────────────────────────────────
    # fill_model_enabled: set False to revert to ideal-price fills (for comparison)
    "execution.fill_model_enabled":         lambda: True,
    # Session baseline slippage (bps) — controls how much worse than ideal each fill is
    "execution.slip_base_regular":          lambda: 3.0,
    "execution.slip_base_restricted":       lambda: 8.0,
    "execution.slip_base_pre_market":       lambda: 12.0,
    "execution.slip_base_after_hours":      lambda: 18.0,
    # Volatility penalty — adds N bps per % ATR above threshold
    "execution.slip_vol_threshold_pct":     lambda: 1.0,    # ATR% above which penalty starts
    "execution.slip_vol_penalty_bps":       lambda: 1.5,    # bps per % ATR over threshold
    # Liquidity penalty based on avg daily volume
    "execution.slip_liq_low_vol":           lambda: 100_000, # shares/day below = low liquidity
    "execution.slip_liq_mid_vol":           lambda: 500_000, # shares/day below = mid liquidity
    "execution.slip_liq_low_penalty_bps":   lambda: 10.0,
    "execution.slip_liq_mid_penalty_bps":   lambda: 3.0,
    # Position size penalty — large orders relative to daily volume cost more
    "execution.slip_size_rate":             lambda: 50.0,   # bps per % of daily dollar vol
    "execution.slip_size_cap_bps":          lambda: 20.0,   # cap on size penalty
    "execution.slip_max_bps":               lambda: 60.0,   # hard ceiling on total slippage
    # Stop-market specific — gap penalty when bar low is well below stop
    "execution.stop_gap_factor":            lambda: 0.30,   # fraction of gap below stop added to slippage
    # Spread estimation from ATR when bid/ask unavailable
    "execution.spread_atr_rate":            lambda: 0.15,   # synthetic spread = 15% of ATR
    "execution.spread_min_bps":             lambda: 1.0,    # floor spread in bps
    # ── Ticker-level damage control ─────────────────────────────────────────────
    "risk.ticker_loss_cooldown_usd":        lambda: 50.0,   # $ loss in window to trip cooldown
    "risk.ticker_loss_window_min":          lambda: 30,     # rolling window (minutes)
    "risk.ticker_pre_t1_stops_max":         lambda: 2,      # pre-T1 stops in window before cooldown
    "risk.ticker_pre_t1_window_min":        lambda: 30,     # rolling window for pre-T1 stops
    "risk.ticker_cooldown_min":             lambda: 60,     # how long to block the ticker
    # ── Support & Resistance tuning ────────────────────────────────────────────
    # All constants used in support_resistance.py are hot-reloadable here.
    "sr.cluster_tolerance_pct":             lambda: 0.40,   # % — merge levels within this distance (0.4% default)
    "sr.min_rows_pivot":                    lambda: 3,      # min bars required to compute pivot points
    "sr.min_rows_swing":                    lambda: 5,      # min bars required for swing high/low scan
    "sr.swing_window_bars":                 lambda: 10,     # bars on each side to confirm a swing high/low
    "sr.swing_max_levels":                  lambda: 5,      # max S/R levels returned per side
    "sr.poc_buckets":                       lambda: 50,     # histogram buckets for volume POC / value area
    "sr.fibonacci_lookback_bars":           lambda: 50,     # bars scanned for Fibonacci swing high/low
    "sr.fallback_support_pct":              lambda: 0.98,   # price × this when no support level found
    "sr.fallback_resistance_pct":           lambda: 1.02,   # price × this when no resistance level found
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
        """Write all _DEFAULTS to config_store using INSERT … ON CONFLICT DO NOTHING.

        Existing user-set values are never overwritten — only absent keys get the
        Python default.  Also migrates legacy account_config values for the four
        paper.* keys that were previously stored there.

        Called at startup by web-api, scalp-engine, and scalp-learner after
        load() so that every config key appears in the DB and the Settings UI.
        """
        self._ensure_table()
        now = self._now_iso()

        # Legacy migration: read from account_config if present
        legacy: dict[str, Any] = {}
        try:
            from agent.db import get_conn
            with get_conn() as c:
                row = c.execute(
                    "SELECT total_budget, max_trade_pct, max_allocated_pct, max_open_trades"
                    " FROM account_config WHERE id=1"
                ).fetchone()
            if row:
                mapping = {
                    "paper.budget":            ("total_budget",      float),
                    "paper.max_trade_pct":     ("max_trade_pct",     float),
                    "paper.max_allocated_pct": ("max_allocated_pct", float),
                    "paper.max_open_trades":   ("max_open_trades",   int),
                }
                for cfg_key, (col, cast) in mapping.items():
                    if row[col] is not None:
                        legacy[cfg_key] = cast(row[col])
        except Exception:
            pass  # account_config may not exist on fresh deploys

        inserted = 0
        try:
            from agent.db import get_conn
            with get_conn() as c:
                for key, factory in _DEFAULTS.items():
                    try:
                        value = legacy.get(key)
                        if value is None:
                            value = factory()
                        val_json = json.dumps(value)
                        c.execute(
                            """INSERT INTO config_store (key, value, updated_at, updated_by)
                               VALUES (?, ?, ?, 'seed_defaults')
                               ON CONFLICT (key) DO NOTHING""",
                            (key, val_json, now),
                        )
                        inserted += 1
                    except Exception as exc:
                        logger.warning("[ConfigManager] seed_defaults: skip %s: %s", key, exc)
        except Exception as exc:
            logger.warning("[ConfigManager] seed_defaults failed: %s", exc)
            return

        logger.info("[ConfigManager] seed_defaults: seeded %d keys (ON CONFLICT DO NOTHING)", inserted)
        self._apply_safety_migrations(now)
        # Reload cache so newly-inserted defaults are visible immediately in this process
        self.load()

    def _apply_safety_migrations(self, now: str) -> None:
        """Apply narrow one-time config repairs for unsafe legacy defaults."""
        try:
            from agent.db import get_conn
            with get_conn() as c:
                row = c.execute(
                    "SELECT value, updated_by FROM config_store WHERE key = ?",
                    ("paper.t2_r_multiple",),
                ).fetchone()
                if not row:
                    return
                try:
                    current_t2 = float(json.loads(row["value"]))
                except Exception:
                    return
                updated_by = str(row["updated_by"] or "")
                if abs(current_t2 - 2.0) < 1e-9 and updated_by == "seed_defaults":
                    c.execute(
                        _UPSERT,
                        (
                            "paper.t2_r_multiple",
                            json.dumps(1.5),
                            now,
                            "migration_t2_1_5",
                        ),
                    )
                    logger.info(
                        "[ConfigManager] Migrated paper.t2_r_multiple from legacy 2.0R to 1.5R"
                    )

                row = c.execute(
                    "SELECT value, updated_by FROM config_store WHERE key = ?",
                    ("paper.t1_r_multiple",),
                ).fetchone()
                if not row:
                    return
                try:
                    current_t1 = float(json.loads(row["value"]))
                except Exception:
                    return
                updated_by = str(row["updated_by"] or "")
                if abs(current_t1 - 1.5) < 1e-9:
                    c.execute(
                        _UPSERT,
                        (
                            "paper.t1_r_multiple",
                            json.dumps(1.0),
                            now,
                            "migration_t1_1_0",
                        ),
                    )
                    logger.info(
                        "[ConfigManager] Migrated paper.t1_r_multiple from legacy 1.5R to 1.0R"
                    )

                row = c.execute(
                    "SELECT value FROM config_store WHERE key = ?",
                    ("risk.technical_entry_gate_missing_data_block",),
                ).fetchone()
                if row:
                    try:
                        current_missing_block = bool(json.loads(row["value"]))
                    except Exception:
                        current_missing_block = False
                    if not current_missing_block:
                        c.execute(
                            _UPSERT,
                            (
                                "risk.technical_entry_gate_missing_data_block",
                                json.dumps(True),
                                now,
                                "migration_technical_missing_block",
                            ),
                        )
                        logger.info(
                            "[ConfigManager] Enabled technical missing-data blocking"
                        )

                row = c.execute(
                    "SELECT value FROM config_store WHERE key = ?",
                    ("paper.max_bars_scalp",),
                ).fetchone()
                if row:
                    try:
                        current_max_bars = int(json.loads(row["value"]))
                    except Exception:
                        current_max_bars = 20
                    if current_max_bars >= 300:
                        c.execute(
                            _UPSERT,
                            (
                                "paper.max_bars_scalp",
                                json.dumps(20),
                                now,
                                "migration_scalp_horizon_20",
                            ),
                        )
                        logger.info(
                            "[ConfigManager] Restored paper.max_bars_scalp from %d to 20",
                            current_max_bars,
                        )
        except Exception as exc:
            logger.warning("[ConfigManager] safety migration failed: %s", exc)

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

_SENTINEL = object()


# Patch get() to fall through to _DEFAULTS when key is absent from cache and
# no explicit default was supplied.  This makes config_manager the single source
# of truth: callers can write config.get("paper.budget") with no hardcoded
# fallback and still get the env-var-seeded default on a cold start.
_orig_get = ConfigManager.get


def _get_with_defaults(self, key: str, default: Any = _SENTINEL) -> Any:
    with self._lock:
        if key in self._cache:
            return self._cache[key]
    if default is not _SENTINEL:
        return default
    factory = _DEFAULTS.get(key)
    if factory is not None:
        try:
            return factory()
        except Exception:
            pass
    return None


ConfigManager.get = _get_with_defaults  # type: ignore[method-assign]

config = ConfigManager()
