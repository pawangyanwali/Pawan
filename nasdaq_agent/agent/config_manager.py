"""
Runtime configuration store â€” Phase 2.

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

# â”€â”€ Defaults â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_DEFAULTS: dict[str, Any] = {
    # â”€â”€ Paper trading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "paper.budget":                        lambda: float(os.getenv("PAPER_BUDGET", "50000")),
    "paper.max_trade_pct":                 lambda: float(os.getenv("PAPER_MAX_TRADE_PCT", "5.0")),
    "paper.max_allocated_pct":             lambda: float(os.getenv("PAPER_MAX_ALLOCATED_PCT", "40.0")),
    "paper.max_open_trades":               lambda: int(os.getenv("PAPER_MAX_OPEN_TRADES", "10")),
    "paper.min_confidence":                lambda: float(os.getenv("PAPER_TRADE_MIN_CONFIDENCE", "25.0")),
    # Extended-hours gates
    "paper.ext_hours_high_min_conf":       lambda: 70.0,   # min confidence for HIGH-tier in PM/AH
    "paper.ext_hours_moderate_min_conf":   lambda: 60.0,   # min confidence for MODERATE-tier in PM/AH
    "paper.pre_market_stop_mult":          lambda: 1.5,    # widen stops 1.5Ã— in pre-market
    "paper.after_hours_stop_mult":         lambda: 2.0,    # widen stops 2Ã— in after-hours
    # Position sizing
    "paper.rr_size_mult_min":              lambda: 0.20,   # minimum size multiplier from R:R calculation
    "paper.rr_denominator":                lambda: 2.0,    # divisor in rr_ratio / N â†’ size_mult
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
    "scalp.candidate_tracking_enabled":    lambda: True,
    "scalp.candidate_retention_days":      lambda: 30,
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
    "scalp.use_provisional_live_indicators": lambda: True,
    "scalp.provisional_max_bar_age_ms":    lambda: 300000,
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
    "scalp.long_require_fast_rsi_confirmation": lambda: True,
    "scalp.long_require_vwap_reclaim":     lambda: True,
    "scalp.long_require_mtf_not_bearish":  lambda: True,
    "scalp.long_block_bearish_market":     lambda: True,
    "scalp.short_require_fast_rsi_confirmation": lambda: True,
    "scalp.short_premarket_require_vwap_rejection": lambda: True,
    "scalp.short_require_mtf_not_bullish": lambda: True,
    "scalp.short_block_bullish_market":    lambda: True,
    "scalp.allow_rest_fallback_trading":   lambda: False,
    "scalp.block_when_path_obstructed":    lambda: True,
    "scalp.block_when_risk_capped":        lambda: True,
    "scalp.shadow_fixed_risk_enabled":     lambda: True,
    "scalp.shadow_risk_per_trade_usd":     lambda: 25.0,
    "scalp.shadow_min_shares":             lambda: 1,
    "scalp.shadow_max_shares":             lambda: 500,
    # Entry approval remains independent from signal direction and bracket math.
    # These gates answer whether a valid setup is executable now.
    "scalp.entry_quality_gate_enabled":     lambda: True,
    "scalp.entry_quality_min_score":        lambda: 65.0,
    "scalp.entry_quality_min_score_extended": lambda: 70.0,
    "scalp.entry_quality_require_positive_ml_ev": lambda: True,
    "scalp.entry_quality_min_ml_expected_r": lambda: 0.05,
    "scalp.entry_quality_empirical_gate_enabled": lambda: True,
    "scalp.entry_quality_empirical_min_samples": lambda: 10,
    "scalp.entry_quality_min_empirical_expectancy_r": lambda: 0.0,
    "scalp.entry_quality_shadow_probe_size_mult": lambda: 0.10,
    "scalp.entry_confirmation_enabled":     lambda: True,
    "scalp.entry_confirmation_seconds":     lambda: 15.0,
    "scalp.entry_confirmation_min_observations": lambda: 3,
    "scalp.entry_confirmation_max_chase_r": lambda: 0.25,
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
    "scalp_runtime.bar_lookback":           lambda: 2500,
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
    "scalp_runtime.context_cluster_throttle_enabled": lambda: True,
    "scalp_runtime.context_cluster_window_min": lambda: 10,
    "scalp_runtime.context_cluster_max_entries": lambda: 3,
    "scalp_runtime.context_cluster_use_setup_session": lambda: True,
    "scalp_runtime.setup_session_cluster_max_entries": lambda: 2,
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
    "scalp_learn.dollar_guard_enabled":     lambda: True,
    "scalp_learn.negative_reduce_dollar":   lambda: -25.0,
    "scalp_learn.negative_block_dollar":    lambda: -100.0,
    "scalp_learn.fast_stop_circuit_enabled": lambda: True,
    "scalp_learn.fast_stop_window_min":     lambda: 10,
    "scalp_learn.fast_stop_count":          lambda: 3,
    "scalp_learn.fast_stop_size_mult":      lambda: 0.25,
    "scalp_learn.setup_session_fast_stop_block_enabled": lambda: True,
    "scalp_learn.setup_session_fast_stop_block_count": lambda: 2,
    "scalp_learn.pre_tp1_failure_circuit_enabled": lambda: True,
    "scalp_learn.pre_tp1_failure_window_min": lambda: 120,
    "scalp_learn.pre_tp1_failure_count":    lambda: 2,
    "scalp_learn.pre_tp1_failure_size_mult": lambda: 0.25,
    "scalp_learn.setup_session_pre_tp1_block_enabled": lambda: True,
    "scalp_learn.setup_session_gate_enabled": lambda: True,
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
    "scalp_ml.bootstrap_minimum_samples":    lambda: 75,
    "scalp_ml.bootstrap_training_enabled":   lambda: True,
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
    # â”€â”€ Prediction / R:R engine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # These control the trade-entry quality filter. All hot-reload â€” no restart needed.
    "prediction.min_rr":                   lambda: 1.5,    # target reward multiple (e.g. 2.0 = risk 1, reward 2); not a signal gate
    "prediction.min_stop_dist_pct":        lambda: 0.004,  # stop must be â‰¥ this % from entry (avoids noise stops)
    "prediction.max_risk_pct":             lambda: 0.020,  # cap risk at this % of stock price per scalp
    "prediction.min_target_pct":           lambda: 0.003,  # target must be â‰¥ this % from entry
    # ATR-based stop mode (recommended for automated scalping â€” eliminates T2/target gap)
    # When true: stop = ATR Ã— stop_atr_multiple, target = entry + t2_r_multiple Ã— risk,
    #            structure used only as filter (trade rejected if resistance blocks path).
    # When false: legacy structural mode (stop at support, target at resistance).
    "prediction.use_atr_stops":            lambda: True,   # true = fixed R:R (recommended); false = structure-based
    "prediction.stop_atr_multiple":        lambda: 1.0,    # stop distance = N Ã— ATR(14). 1.0 = 1Ã—ATR is standard scalp stop
    "prediction.min_stop_daily_atr_pct":   lambda: 0.05,   # floor: stop â‰¥ 5% of daily ATR(14) â€” prevents 6-cent pre-market stops
    # T1/T2 exit multipliers â€” both expressed as multiples of the initial risk distance.
    # T1 is the partial-exit level (take 50% off, move stop to breakeven).
    # T2 is the full-exit target. Setting t2_r_multiple = prediction.min_rr makes
    # T2 exactly equal to the minimum R:R target â€” no gap between filter and exit.
    "paper.t1_r_multiple":                 lambda: 1.0,    # T1 = entry + 1Ã— risk_dist
    "paper.t2_r_multiple":                 lambda: 1.5,    # T2 = 1.5R â€” more achievable in choppy sessions
    "paper.algo_t2_r_multiple":            lambda: 1.5,    # T2 for named algo-family paper trades; primary predictions use paper.t2_r_multiple
    # Time stops: hard-close positions after N bars if still open
    "paper.max_bars_scalp":                lambda: 20,     # 20-min hard close for scalp trades÷}{¶‰ËkºwµçJRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠR 4(4(€€€‘•˜±½…¡Í•±˜¤€´ø9½¹”è4(€€€€€€€€ˆˆˆ4(€€€€€€€I•……±°É½İÌ™É½´½¹™¥}ÍÑ½É”¥¹Ñ¼Ñ¡”¥¸µµ•µ½Éä…¡”¸4(€€€€€€€±Í¼•¹ÍÕÉ•ÌÑ¡”Ñ…‰±”•á¥ÍÑÌ¸4(€€€€€€€€ˆˆˆ4(€€€€€€€Í•±˜¹}•¹ÍÕÉ•}Ñ…‰±” ¤4(€€€€€€€ÑÉäè4(€€€€€€€€€€€™É½´…•¹Ğ¹‘ˆ¥µÁ½ÉĞ•Ñ}½¹¸4(€€€€€€€€€€€İ¥Ñ •Ñ}½¹¸ ¤…ÌŒè4(€€€€€€€€€€€€€€€É½İÌ€ôŒ¹•á•ÕÑ” 4(€€€€€€€€€€€€€€€€€€€€‰M1P­•ä°Ù…±Õ”I=4½¹™¥}ÍÑ½É”ˆ4(€€€€€€€€€€€€€€€€¤¹™•Ñ¡…±° ¤4(€€€€€€€€€€€İ¥Ñ Í•±˜¹}±½¬è4(€€€€€€€€€€€€€€€™½ÈÉ½Ü¥¸É½İÌè4(€€€€€€€€€€€€€€€€€€€ÑÉäè4(€€€€€€€€€€€€€€€€€€€€€€€Í•±˜¹}…¡•mÉ½İl‰­•ä‰ut€ô©Í½¸¹±½…‘Ì¡É½İl‰Ù…±Õ”‰t¤4(€€€€€€€€€€€€€€€€€€€•á•ÁĞá•ÁÑ¥½¸è4(€€€€€€€€€€€€€€€€€€€€€€€Í•±˜¹}…¡•mÉ½İl‰­•ä‰ut€ôÉ½İl‰Ù…±Õ”‰t4(€€€€€€€€€€€±½•È¹¥¹™¼ ‰m½¹™¥5…¹…•Ét1½…‘•€•­•åÌ™É½´½¹™¥}ÍÑ½É”ˆ°±•¸¡É½İÌ¤¤4(€€€€€€€•á•ÁĞá•ÁÑ¥½¸…Ì•áŒè4(€€€€€€€€€€€±½•È¹İ…É¹¥¹œ ‰m½¹™¥5…¹…•Ét±½… ¤™…¥±•è€•Ìˆ°•áŒ¤4(4(€€€‘•˜Í••‘}‘•™…Õ±ÑÌ¡Í•±˜¤€´ø9½¹”è4(€€€€€€€€ˆˆ‰]É¥Ñ”…±°}U1QLÑ¼½¹™¥}ÍÑ½É”ÕÍ¥¹œ%9MIPƒŠ˜=8=91%P<9=Q!%9¸4(4(€€€€€€€á¥ÍÑ¥¹œÕÍ•ÈµÍ•ĞÙ…±Õ•Ì…É”¹•Ù•È½Ù•ÉİÉ¥ÑÑ•¸ƒŠP½¹±ä…‰Í•¹Ğ­•åÌ•ĞÑ¡”4(€€€€€€€AåÑ¡½¸‘•™…Õ±Ğ¸€±Í¼µ¥É…Ñ•Ì±•…ä…½Õ¹Ñ}½¹™¥œÙ…±Õ•Ì™½ÈÑ¡”™½ÕÈ4(€€€€€€€Á…Á•È¸¨­•åÌÑ¡…Ğİ•É”ÁÉ•Ù¥½ÕÍ±äÍÑ½É•Ñ¡•É”¸4(4(€€€€€€€…±±•…ĞÍÑ…ÉÑÕÀ‰äİ•ˆµ…Á¤°Í…±Àµ•¹¥¹”°…¹Í…±Àµ±•…É¹•È…™Ñ•È(€€€€€€€±½… ¤Í¼Ñ¡…Ğ•Ù•Éä½¹™¥œ­•ä…ÁÁ•…ÉÌ¥¸Ñ¡”…¹Ñ¡”M•ÑÑ¥¹ÌU$¸4(€€€€€€€€ˆˆˆ4(€€€€€€€Í•±˜¹}•¹ÍÕÉ•}Ñ…‰±” ¤4(€€€€€€€¹½Ü€ôÍ•±˜¹}¹½İ}¥Í¼ ¤4(4(€€€€€€€€Œ1•…äµ¥É…Ñ¥½¸èÉ•…™É½´…½Õ¹Ñ}½¹™¥œ¥˜ÁÉ•Í•¹Ğ4(€€€€€€€±•…äè‘¥ÑmÍÑÈ°¹åt€ôíô4(€€€€€€€ÑÉäè4(€€€€€€€€€€€™É½´…•¹Ğ¹‘ˆ¥µÁ½ÉĞ•Ñ}½¹¸4(€€€€€€€€€€€İ¥Ñ •Ñ}½¹¸ ¤…ÌŒè4(€€€€€€€€€€€€€€€É½Ü€ôŒ¹•á•ÕÑ” 4(€€€€€€€€€€€€€€€€€€€€‰M1PÑ½Ñ…±}‰Õ‘•Ğ°µ…á}ÑÉ…‘•}ÁĞ°µ…á}…±±½…Ñ•‘}ÁĞ°µ…á}½Á•¹}ÑÉ…‘•Ìˆ4(€€€€€€€€€€€€€€€€€€€€ˆI=4…½Õ¹Ñ}½¹™¥œ]!I¥ôÄˆ4(€€€€€€€€€€€€€€€€¤¹™•Ñ¡½¹” ¤4(€€€€€€€€€€€¥˜É½Üè4(€€€€€€€€€€€€€€€µ…ÁÁ¥¹œ€ôì4(€€€€€€€€€€€€€€€€€€€€‰Á…Á•È¹‰Õ‘•Ğˆè€€€€€€€€€€€€ ‰Ñ½Ñ…±}‰Õ‘•Ğˆ°€€€€€™±½…Ğ¤°4(€€€€€€€€€€€€€€€€€€€€‰Á…Á•È¹µ…á}ÑÉ…‘•}ÁĞˆè€€€€€ ‰µ…á}ÑÉ…‘•}ÁĞˆ°€€€€™±½…Ğ¤°4(€€€€€€€€€€€€€€€€€€€€‰Á…Á•È¹µ…á}…±±½…Ñ•‘}ÁĞˆè€ ‰µ…á}…±±½…Ñ•‘}ÁĞˆ°™±½…Ğ¤°4(€€€€€€€€€€€€€€€€€€€€‰Á…Á•È¹µ…á}½Á•¹}ÑÉ…‘•Ìˆè€€€ ‰µ…á}½Á•¹}ÑÉ…‘•Ìˆ°€€¥¹Ğ¤°4(€€€€€€€€€€€€€€€ô4(€€€€€€€€€€€€€€€™½È™}­•ä°€¡½°°…ÍĞ¤¥¸µ…ÁÁ¥¹œ¹¥Ñ•µÌ ¤è4(€€€€€€€€€€€€€€€€€€€¥˜É½İm½±t¥Ì¹½Ğ9½¹”è4(€€€€€€€€€€€€€€€€€€€€€€€±•…åm™}­•åt€ô…ÍĞ¡É½İm½±t¤4(€€€€€€€•á•ÁĞá•ÁÑ¥½¸è4(€€€€€€€€€€€Á…ÍÌ€€Œ…½Õ¹Ñ}½¹™¥œµ…ä¹½Ğ•á¥ÍĞ½¸™É•Í ‘•Á±½åÌ4(4(€€€€€€€¥¹Í•ÉÑ•€ô€À4(€€€€€€€ÑÉäè4(€€€€€€€€€€€™É½´…•¹Ğ¹‘ˆ¥µÁ½ÉĞ•Ñ}½¹¸4(€€€€€€€€€€€İ¥Ñ •Ñ}½¹¸ ¤…ÌŒè4(€€€€€€€€€€€€€€€™½È­•ä°™…Ñ½Éä¥¸}U1QL¹¥Ñ•µÌ ¤è4(€€€€€€€€€€€€€€€€€€€ÑÉäè4(€€€€€€€€€€€€€€€€€€€€€€€Ù…±Õ”€ô±•…ä¹•Ğ¡­•ä¤4(€€€€€€€€€€€€€€€€€€€€€€€¥˜Ù…±Õ”¥Ì9½¹”è4(€€€€€€€€€€€€€€€€€€€€€€€€€€€Ù…±Õ”€ô™…Ñ½Éä ¤4(€€€€€€€€€€€€€€€€€€€€€€€Ù…±}©Í½¸€ô©Í½¸¹‘ÕµÁÌ¡Ù…±Õ”¤4(€€€€€€€€€€€€€€€€€€€€€€€Œ¹•á•ÕÑ” 4(€€€€€€€€€€€€€€€€€€€€€€€€€€€€ˆˆ‰%9MIP%9Q<½¹™¥}ÍÑ½É”€¡­•ä°Ù…±Õ”°ÕÁ‘…Ñ•‘}…Ğ°ÕÁ‘…Ñ•‘}‰ä¤4(€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Y1UL€ ü°€ü°€ü°€Í••‘}‘•™…Õ±ÑÌœ¤4(€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€=8=91%P€¡­•ä¤<9=Q!%9ˆˆˆ°4(€€€€€€€€€€€€€€€€€€€€€€€€€€€€¡­•ä°Ù…±}©Í½¸°¹½Ü¤°4(€€€€€€€€€€€€€€€€€€€€€€€€¤4(€€€€€€€€€€€€€€€€€€€€€€€¥¹Í•ÉÑ•€¬ô€Ä4(€€€€€€€€€€€€€€€€€€€•á•ÁĞá•ÁÑ¥½¸…Ì•áŒè4(€€€€€€€€€€€€€€€€€€€€€€€±½•È¹İ…É¹¥¹œ ‰m½¹™¥5…¹…•ÉtÍ••‘}‘•™…Õ±ÑÌèÍ­¥À€•Ìè€•Ìˆ°­•ä°•áŒ¤4(€€€€€€€•á•ÁĞá•ÁÑ¥½¸…Ì•áŒè4(€€€€€€€€€€€±½•È¹İ…É¹¥¹œ ‰m½¹™¥5…¹…•ÉtÍ••‘}‘•™…Õ±ÑÌ™…¥±•è€•Ìˆ°•áŒ¤4(€€€€€€€€€€€É•ÑÕÉ¸4(4(€€€€€€€±½•È¹¥¹™¼ ‰m½¹™¥5…¹…•ÉtÍ••‘}‘•™…Õ±ÑÌèÍ••‘•€•­•åÌ€¡=8=91%P<9=Q!%9¤ˆ°¥¹Í•ÉÑ•¤(€€€€€€€Í•±˜¹}…ÁÁ±å}Í…™•Ñå}µ¥É…Ñ¥½¹Ì¡¹½Ü¤(€€€€€€€€ŒI•±½……¡”Í¼¹•İ±äµ¥¹Í•ÉÑ•‘•™…Õ±ÑÌ…É”Ù¥Í¥‰±”¥µµ•‘¥…Ñ•±ä¥¸Ñ¡¥ÌÁÉ½•ÍÌ(€€€€€€€Í•±˜¹±½… ¤((€€€‘•˜}…ÁÁ±å}Í…™•Ñå}µ¥É…Ñ¥½¹Ì¡Í•±˜°¹½ÜèÍÑÈ¤€´ø9½¹”è(€€€€€€€€ˆˆ‰ÁÁ±ä¹…ÉÉ½Ü½¹”µÑ¥µ”½¹™¥œÉ•Á…¥ÉÌ™½ÈÕ¹Í…™”±•…ä‘•™…Õ±ÑÌ¸ˆˆˆ(€€€€€€€ÑÉäè(€€€€€€€€€€€™É½´…•¹Ğ¹‘ˆ¥µÁ½ÉĞ•Ñ}½¹¸(€€€€€€€€€€€İ¥Ñ •Ñ}½¹¸ ¤…ÌŒè(€€€€€€€€€€€€€€€É½Ü€ôŒ¹•á•ÕÑ” (€€€€€€€€€€€€€€€€€€€€‰M1PÙ…±Õ”°ÕÁ‘…Ñ•‘}‰äI=4½¹™¥}ÍÑ½É”]!I­•ä€ô€üˆ°(€€€€€€€€€€€€€€€€€€€€ ‰Á…Á•È¹ĞÉ}É}µÕ±Ñ¥Á±”ˆ°¤°(€€€€€€€€€€€€€€€€¤¹™•Ñ¡½¹” ¤(€€€€€€€€€€€€€€€¥˜¹½ĞÉ½Üè(€€€€€€€€€€€€€€€€€€€É•ÑÕÉ¸(€€€€€€€€€€€€€€€ÑÉäè(€€€€€€€€€€€€€€€€€€€ÕÉÉ•¹Ñ}ĞÈ€ô™±½…Ğ¡©Í½¸¹±½…‘Ì¡É½İl‰Ù…±Õ”‰t¤¤(€€€€€€€€€€€€€€€•á•ÁĞá•ÁÑ¥½¸è(€€€€€€€€€€€€€€€€€€€É•ÑÕÉ¸(€€€€€€€€€€€€€€€ÕÁ‘…Ñ•‘}‰ä€ôÍÑÈ¡É½İl‰ÕÁ‘…Ñ•‘}‰ä‰t½È€ˆˆ¤(€€€€€€€€€€€€€€€¥˜…‰Ì¡ÕÉÉ•¹Ñ}ĞÈ€´€È¸À¤€ğ€Å”´ä…¹ÕÁ‘…Ñ•‘}‰ä€ôô€‰Í••‘}‘•™…Õ±ÑÌˆè(€€€€€€€€€€€€€€€€€€€Œ¹•á•ÕÑ” (€€€€€€€€€€€€€€€€€€€€€€€}UAMIP°(€€€€€€€€€€€€€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€€€€€€€€€€€€€‰Á…Á•È¹ĞÉ}É}µÕ±Ñ¥Á±”ˆ°(€€€€€€€€€€€€€€€€€€€€€€€€€€€©Í½¸¹‘ÕµÁÌ Ä¸Ô¤°(€€€€€€€€€€€€€€€€€€€€€€€€€€€¹½Ü°(€€€€€€€€€€€€€€€€€€€€€€€€€€€€‰µ¥É…Ñ¥½¹}ĞÉ|Å|Ôˆ°(€€€€€€€€€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€±½•È¹¥¹™¼ (€€€€€€€€€€€€€€€€€€€€€€€€‰m½¹™¥5…¹…•Ét5¥É…Ñ•Á…Á•È¹ĞÉ}É}µÕ±Ñ¥Á±”™É½´±•…ä€È¸ÁHÑ¼€Ä¸ÕHˆ(€€€€€€€€€€€€€€€€€€€€¤((€€€€€€€€€€€€€€€É½Ü€ôŒ¹•á•ÕÑ” (€€€€€€€€€€€€€€€€€€€€‰M1PÙ…±Õ”°ÕÁ‘…Ñ•‘}‰äI=4½¹™¥}ÍÑ½É”]!I­•ä€ô€üˆ°(€€€€€€€€€€€€€€€€€€€€ ‰Á…Á•È¹ĞÅ}É}µÕ±Ñ¥Á±”ˆ°¤°(€€€€€€€€€€€€€€€€¤¹™•Ñ¡½¹” ¤(€€€€€€€€€€€€€€€¥˜¹½ĞÉ½Üè(€€€€€€€€€€€€€€€€€€€É•ÑÕÉ¸(€€€€€€€€€€€€€€€ÑÉäè(€€€€€€€€€€€€€€€€€€€ÕÉÉ•¹Ñ}ĞÄ€ô™±½…Ğ¡©Í½¸¹±½…‘Ì¡É½İl‰Ù…±Õ”‰t¤¤(€€€€€€€€€€€€€€€•á•ÁĞá•ÁÑ¥½¸è(€€€€€€€€€€€€€€€€€€€É•ÑÕÉ¸(€€€€€€€€€€€€€€€ÕÁ‘…Ñ•‘}‰ä€ôÍÑÈ¡É½İl‰ÕÁ‘…Ñ•‘}‰ä‰t½È€ˆˆ¤(€€€€€€€€€€€€€€€¥˜…‰Ì¡ÕÉÉ•¹Ñ}ĞÄ€´€Ä¸Ô¤€ğ€Å”´äè(€€€€€€€€€€€€€€€€€€€Œ¹•á•ÕÑ” (€€€€€€€€€€€€€€€€€€€€€€€}UAMIP°(€€€€€€€€€€€€€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€€€€€€€€€€€€€‰Á…Á•È¹ĞÅ}É}µÕ±Ñ¥Á±”ˆ°(€€€€€€€€€€€€€€€€€€€€€€€€€€€©Í½¸¹‘ÕµÁÌ Ä¸À¤°(€€€€€€€€€€€€€€€€€€€€€€€€€€€¹½Ü°(€€€€€€€€€€€€€€€€€€€€€€€€€€€€‰µ¥É…Ñ¥½¹}ĞÅ|Å|Àˆ°(€€€€€€€€€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€±½•È¹¥¹™¼ (€€€€€€€€€€€€€€€€€€€€€€€€‰m½¹™¥5…¹…•Ét5¥É…Ñ•Á…Á•È¹ĞÅ}É}µÕ±Ñ¥Á±”™É½´±•…ä€Ä¸ÕHÑ¼€Ä¸ÁHˆ(€€€€€€€€€€€€€€€€€€€€¤((€€€€€€€€€€€€€€€É½Ü€ôŒ¹•á•ÕÑ” (€€€€€€€€€€€€€€€€€€€€‰M1PÙ…±Õ”I=4½¹™¥}ÍÑ½É”]!I­•ä€ô€üˆ°(€€€€€€€€€€€€€€€€€€€€ ‰É¥Í¬¹Ñ•¡¹¥…±}•¹ÑÉå}…Ñ•}µ¥ÍÍ¥¹}‘…Ñ…}‰±½¬ˆ°¤°(€€€€€€€€€€€€€€€€¤¹™•Ñ¡½¹” ¤(€€€€€€€€€€€€€€€¥˜É½Üè(€€€€€€€€€€€€€€€€€€€ÑÉäè(€€€€€€€€€€€€€€€€€€€€€€€ÕÉÉ•¹Ñ}µ¥ÍÍ¥¹}‰±½¬€ô‰½½°¡©Í½¸¹±½…‘Ì¡É½İl‰Ù…±Õ”‰t¤¤(€€€€€€€€€€€€€€€€€€€•á•ÁĞá•ÁÑ¥½¸è(€€€€€€€€€€€€€€€€€€€€€€€ÕÉÉ•¹Ñ}µ¥ÍÍ¥¹}‰±½¬€ô…±Í”(€€€€€€€€€€€€€€€€€€€¥˜¹½ĞÕÉÉ•¹Ñ}µ¥ÍÍ¥¹}‰±½¬è(€€€€€€€€€€€€€€€€€€€€€€€Œ¹•á•ÕÑ” (€€€€€€€€€€€€€€€€€€€€€€€€€€€}UAMIP°(€€€€€€€€€€€€€€€€€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€‰É¥Í¬¹Ñ•¡¹¥…±}•¹ÑÉå}…Ñ•}µ¥ÍÍ¥¹}‘…Ñ…}‰±½¬ˆ°(€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€©Í½¸¹‘ÕµÁÌ¡QÉÕ”¤°(€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€¹½Ü°(€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€‰µ¥É…Ñ¥½¹}Ñ•¡¹¥…±}µ¥ÍÍ¥¹}‰±½¬ˆ°(€€€€€€€€€€€€€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€€€€€±½•È¹¥¹™¼ (€€€€€€€€€€€€€€€€€€€€€€€€€€€€‰m½¹™¥5…¹…•Ét¹…‰±•Ñ•¡¹¥…°µ¥ÍÍ¥¹œµ‘…Ñ„‰±½­¥¹œˆ(€€€€€€€€€€€€€€€€€€€€€€€€¤((€€€€€€€€€€€€€€€É½Ü€ôŒ¹•á•ÕÑ” (€€€€€€€€€€€€€€€€€€€€‰M1PÙ…±Õ”I=4½¹™¥}ÍÑ½É”]!I­•ä€ô€üˆ°(€€€€€€€€€€€€€€€€€€€€ ‰Á…Á•È¹µ…á}‰…ÉÍ}Í…±Àˆ°¤°(€€€€€€€€€€€€€€€€¤¹™•Ñ¡½¹” ¤(€€€€€€€€€€€€€€€¥˜É½Üè(€€€€€€€€€€€€€€€€€€€ÑÉäè(€€€€€€€€€€€€€€€€€€€€€€€ÕÉÉ•¹Ñ}µ…á}‰…ÉÌ€ô¥¹Ğ¡©Í½¸¹±½…‘Ì¡É½İl‰Ù…±Õ”‰t¤¤(€€€€€€€€€€€€€€€€€€€•á•ÁĞá•ÁÑ¥½¸è(€€€€€€€€€€€€€€€€€€€€€€€ÕÉÉ•¹Ñ}µ…á}‰…ÉÌ€ô€ÈÀ(€€€€€€€€€€€€€€€€€€€¥˜ÕÉÉ•¹Ñ}µ…á}‰…ÉÌ€øô€ÌÀÀè(€€€€€€€€€€€€€€€€€€€€€€€Œ¹•á•ÕÑ” (€€€€€€€€€€€€€€€€€€€€€€€€€€€}UAMIP°(€€€€€€€€€€€€€€€€€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€‰Á…Á•È¹µ…á}‰…ÉÍ}Í…±Àˆ°(€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€©Í½¸¹‘ÕµÁÌ ÈÀ¤°(€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€¹½Ü°(€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€‰µ¥É…Ñ¥½¹}Í…±Á}¡½É¥é½¹|ÈÀˆ°(€€€€€€€€€€€€€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€€€€€±½•È¹¥¹™¼ (€€€€€€€€€€€€€€€€€€€€€€€€€€€€‰m½¹™¥5…¹…•ÉtI•ÍÑ½É•Á…Á•È¹µ…á}‰…ÉÍ}Í…±À™É½´€•Ñ¼€ÈÀˆ°(€€€€€€€€€€€€€€€€€€€€€€€€€€€ÕÉÉ•¹Ñ}µ…á}‰…ÉÌ°(€€€€€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€•á•ÁĞá•ÁÑ¥½¸…Ì•áŒè(€€€€€€€€€€€±½•È¹İ…É¹¥¹œ ‰m½¹™¥5…¹…•ÉtÍ…™•Ñäµ¥É…Ñ¥½¸™…¥±•è€•Ìˆ°•áŒ¤((€€€‘•˜•Ğ¡Í•±˜°­•äèÍÑÈ°‘•™…Õ±Ğè¹ä€ô9½¹”¤€´ø¹äè(€€€€€€€€ˆˆ‰Q¡É•…µÍ…™”É•…™É½´¥¸µµ•µ½Éä…¡”¸ˆˆˆ(€€€€€€€İ¥Ñ Í•±˜¹}±½¬è(€€€€€€€€€€€É•ÑÕÉ¸Í•±˜¹}…¡”¹•Ğ¡­•ä°‘•™…Õ±Ğ¤(4(€€€‘•˜Í•Ğ¡Í•±˜°­•äèÍÑÈ°Ù…±Õ”è¹ä°ÕÁ‘…Ñ•‘}‰äèÍÑÈ€ô€‰ÍåÍÑ•´ˆ¤€´ø9½¹”è4(€€€€€€€€ˆˆ‰]É¥Ñ”Ù…±Õ”Ñ¼°ÕÁ‘…Ñ”…¡”°ÁÕ‰±¥Í Ñ¼Y…±­•ä¸ˆˆˆ4(€€€€€€€•¹½‘•€ô©Í½¸¹‘ÕµÁÌ¡Ù…±Õ”¤4(€€€€€€€ÑÌ€€€€€€ôÍ•±˜¹}¹½İ}¥Í¼ ¤4(€€€€€€€ÑÉäè4(€€€€€€€€€€€™É½´…•¹Ğ¹‘ˆ¥µÁ½ÉĞ•Ñ}½¹¸4(€€€€€€€€€€€İ¥Ñ •Ñ}½¹¸ ¤…ÌŒè4(€€€€€€€€€€€€€€€Œ¹•á•ÕÑ”¡}UAMIP°€¡­•ä°•¹½‘•°ÑÌ°ÕÁ‘…Ñ•‘}‰ä¤¤4(€€€€€€€•á•ÁĞá•ÁÑ¥½¸…Ì•áŒè4(€€€€€€€€€€€±½•È¹•ÉÉ½È ‰m½¹™¥5…¹…•ÉtÍ•Ğ •Ì¤•ÉÉ½Èè€•Ìˆ°­•ä°•áŒ¤4(€€€€€€€€€€€É…¥Í”4(4(€€€€€€€İ¥Ñ Í•±˜¹}±½¬è4(€€€€€€€€€€€Í•±˜¹}…¡•m­•åt€ôÙ…±Õ”4(4(€€€€€€€€ŒAÕ‰±¥Í Ñ¼Y…±­•ä™½È¡½ĞµÉ•±½……É½ÍÌÁÉ½•ÍÍ•Ì4(€€€€€€€ÑÉäè4(€€€€€€€€€€€™É½´…•¹Ğ¹Ù…±­•å}±¥•¹Ğ¥µÁ½ÉĞ}•Ñ}±¥•¹Ğ4(€€€€€€€€€€€±¥•¹Ğ€ô}•Ñ}±¥•¹Ğ ¤4(€€€€€€€€€€€¥˜±¥•¹Ğ¥Ì¹½Ğ9½¹”è4(€€€€€€€€€€€€€€€Á…å±½…€ô©Í½¸¹‘ÕµÁÌ¡ì‰­•äˆè­•ä°€‰Ù…±Õ”ˆèÙ…±Õ•ô¤4(€€€€€€€€€€€€€€€±¥•¹Ğ¹ÁÕ‰±¥Í ¡}Y1-e}!990°Á…å±½…¤4(€€€€€€€•á•ÁĞá•ÁÑ¥½¸…Ì•áŒè4(€€€€€€€€€€€±½•È¹‘•‰Õœ ‰m½¹™¥5…¹…•ÉtY…±­•äÁÕ‰±¥Í ™…¥±•™½È€•Ìè€•Ìˆ°­•ä°•áŒ¤4(4(€€€‘•˜Í•Ñ}µ…¹ä¡Í•±˜°ÕÁ‘…Ñ•Ìè‘¥ÑmÍÑÈ°¹åt°ÕÁ‘…Ñ•‘}‰äèÍÑÈ€ô€‰ÍåÍÑ•´ˆ¤€´ø9½¹”è4(€€€€€€€€ˆˆ‰Ñ½µ¥ŒµÕ±Ñ¤µ­•äİÉ¥Ñ”ƒŠP½¹”ÑÉ…¹Í…Ñ¥½¸°Ñ¡•¸ÁÕ‰±¥Í •… ­•ä¸ˆˆˆ4(€€€€€€€¥˜¹½ĞÕÁ‘…Ñ•Ìè4(€€€€€€€€€€€É•ÑÕÉ¸4(€€€€€€€ÑÌ€ôÍ•±˜¹}¹½İ}¥Í¼ ¤4(€€€€€€€É½İÌ€ôl¡¬°©Í½¸¹‘ÕµÁÌ¡Ø¤°ÑÌ°ÕÁ‘…Ñ•‘}‰ä¤™½È¬°Ø¥¸ÕÁ‘…Ñ•Ì¹¥Ñ•µÌ ¥t4(€€€€€€€ÑÉäè4(€€€€€€€€€€€™É½´…•¹Ğ¹‘ˆ¥µÁ½ÉĞ•Ñ}½¹¸4(€€€€€€€€€€€İ¥Ñ •Ñ}½¹¸ ¤…ÌŒè4(€€€€€€€€€€€€€€€™½ÈÉ½Ü¥¸É½İÌè4(€€€€€€€€€€€€€€€€€€€Œ¹•á•ÕÑ”¡}UAMIP°É½Ü¤4(€€€€€€€•á•ÁĞá•ÁÑ¥½¸…Ì•áŒè4(€€€€€€€€€€€±½•È¹•ÉÉ½È ‰m½¹™¥5…¹…•ÉtÍ•Ñ}µ…¹ä•ÉÉ½Èè€•Ìˆ°•áŒ¤4(€€€€€€€€€€€É…¥Í”4(4(€€€€€€€İ¥Ñ Í•±˜¹}±½¬è4(€€€€€€€€€€€™½È¬°Ø¥¸ÕÁ‘…Ñ•Ì¹¥Ñ•µÌ ¤è4(€€€€€€€€€€€€€€€Í•±˜¹}…¡•m­t€ôØ4(4(€€€€€€€€ŒAÕ‰±¥Í •… ¡…¹•­•ä4(€€€€€€€ÑÉäè4(€€€€€€€€€€€™É½´…•¹Ğ¹Ù…±­•å}±¥•¹Ğ¥µÁ½ÉĞ}•Ñ}±¥•¹Ğ4(€€€€€€€€€€€±¥•¹Ğ€ô}•Ñ}±¥•¹Ğ ¤4(€€€€€€€€€€€¥˜±¥•¹Ğ¥Ì¹½Ğ9½¹”è4(€€€€€€€€€€€€€€€Á¥Á”€ô±¥•¹Ğ¹Á¥Á•±¥¹”¡ÑÉ…¹Í…Ñ¥½¸õ…±Í”¤4(€€€€€€€€€€€€€€€™½È¬°Ø¥¸ÕÁ‘…Ñ•Ì¹¥Ñ•µÌ ¤è4(€€€€€€€€€€€€€€€€€€€Á…å±½…€ô©Í½¸¹‘ÕµÁÌ¡ì‰­•äˆè¬°€‰Ù…±Õ”ˆèÙô¤4(€€€€€€€€€€€€€€€€€€€Á¥Á”¹ÁÕ‰±¥Í ¡}Y1-e}!990°Á…å±½…¤4(€€€€€€€€€€€€€€€Á¥Á”¹•á•ÕÑ” ¤4(€€€€€€€•á•ÁĞá•ÁÑ¥½¸…Ì•áŒè4(€€€€€€€€€€€±½•È¹‘•‰Õœ ‰m½¹™¥5…¹…•ÉtY…±­•äÁÕ‰±¥Í €¡Í•Ñ}µ…¹ä¤™…¥±•è€•Ìˆ°•áŒ¤4(4(€€€‘•˜…±°¡Í•±˜¤€´ø‘¥ÑmÍÑÈ°¹åtè4(€€€€€€€€ˆˆ‰I•ÑÕÉ¸„½Áä½˜Ñ¡”™Õ±°¥¸µµ•µ½Éä…¡”¸ˆˆˆ4(€€€€€€€İ¥Ñ Í•±˜¹}±½¬è4(€€€€€€€€€€€É•ÑÕÉ¸‘¥Ğ¡Í•±˜¹}…¡”¤4(4(€€€‘•˜}É•±½…‘}­•ä¡Í•±˜°­•äèÍÑÈ¤€´ø9½¹”è4(€€€€€€€€ˆˆ‰I”µÉ•…„Í¥¹±”­•ä™É½´¥¹Ñ¼…¡”€¡…±±•½¸¡½ĞµÉ•±½…¤¸ˆˆˆ4(€€€€€€€ÑÉäè4(€€€€€€€€€€€™É½´…•¹Ğ¹‘ˆ¥µÁ½ÉĞ•Ñ}½¹¸4(€€€€€€€€€€€İ¥Ñ •Ñ}½¹¸ ¤…ÌŒè4(€€€€€€€€€€€€€€€É½Ü€ôŒ¹•á•ÕÑ” 4(€€€€€€€€€€€€€€€€€€€€‰M1PÙ…±Õ”I=4½¹™¥}ÍÑ½É”]!I­•ä€ô€üˆ°€¡­•ä°¤4(€€€€€€€€€€€€€€€€¤¹™•Ñ¡½¹” ¤4(€€€€€€€€€€€¥˜É½Üè4(€€€€€€€€€€€€€€€İ¥Ñ Í•±˜¹}±½¬è4(€€€€€€€€€€€€€€€€€€€ÑÉäè4(€€€€€€€€€€€€€€€€€€€€€€€Í•±˜¹}…¡•m­•åt€ô©Í½¸¹±½…‘Ì¡É½İl‰Ù…±Õ”‰t¤4(€€€€€€€€€€€€€€€€€€€•á•ÁĞá•ÁÑ¥½¸è4(€€€€€€€€€€€€€€€€€€€€€€€Í•±˜¹}…¡•m­•åt€ôÉ½İl‰Ù…±Õ”‰t4(€€€€€€€€€€€€€€€±½•È¹‘•‰Õœ ‰m½¹™¥5…¹…•Ét!½ĞµÉ•±½…‘•­•äè€•Ìˆ°­•ä¤4(€€€€€€€•á•ÁĞá•ÁÑ¥½¸…Ì•áŒè4(€€€€€€€€€€€±½•È¹‘•‰Õœ ‰m½¹™¥5…¹…•Ét}É•±½…‘}­•ä •Ì¤™…¥±•è€•Ìˆ°­•ä°•áŒ¤4(4(€€€‘•˜ÍÑ…ÉÑ}±¥ÍÑ•¹•È¡Í•±˜°ÉÕ¹¹•Èõ9½¹”¤€´ø9½¹”è4(€€€€€€€€ˆˆˆ4(€€€€€€€MÑ…ÉĞ„‘…•µ½¸Ñ¡É•…Ñ¡…ĞÍÕ‰ÍÉ¥‰•ÌÑ¼Y…±­•ä½¹™¥œé¡…¹•‘€¡…¹¹•°¸4(€€€€€€€=¸µ•ÍÍ…”èÁ…ÉÍ•Ì)M=8°•áÑÉ…ÑÌ­•å€°…±±Ì}É•±½…‘}­•ä¡­•ä¥€¸4(€€€€€€€€ˆˆˆ4(€€€€€€€¥˜Í•±˜¹}±¥ÍÑ•¹•É}ÍÑ…ÉÑ•è4(€€€€€€€€€€€É•ÑÕÉ¸4(€€€€€€€Í•±˜¹}±¥ÍÑ•¹•É}ÍÑ…ÉÑ•€ôQÉÕ”4(€€€€€€€Ğ€ôÑ¡É•…‘¥¹œ¹Q¡É•… 4(€€€€€€€€€€€Ñ…É•ĞõÍ•±˜¹}±¥ÍÑ•¹•É}±½½À°4(€€€€€€€€€€€‘…•µ½¸õQÉÕ”°4(€€€€€€€€€€€¹…µ”ô‰½¹™¥5…¹…•É1¥ÍÑ•¹•Èˆ°4(€€€€€€€€¤4(€€€€€€€Ğ¹ÍÑ…ÉĞ ¤4(€€€€€€€±½•È¹¥¹™¼ ‰m½¹™¥5…¹…•ÉtY…±­•ä±¥ÍÑ•¹•ÈÑ¡É•…ÍÑ…ÉÑ•¸ˆ¤4(4(€€€‘•˜}±¥ÍÑ•¹•É}±½½À¡Í•±˜¤€´ø9½¹”è4(€€€€€€€€ˆˆ‰	…­É½Õ¹‘…•µ½¸èÍÕ‰ÍÉ¥‰”Ñ¼½¹™¥œé¡…¹•…¹¡½ĞµÉ•±½…­•åÌ¸ˆˆˆ4(€€€€€€€É•ÑÉå}‘•±…ä€ô€È¸À4(€€€€€€€İ¡¥±”QÉÕ”è4(€€€€€€€€€€€ÑÉäè4(€€€€€€€€€€€€€€€™É½´…•¹Ğ¹Ù…±­•å}±¥•¹Ğ¥µÁ½ÉĞ}™œ…Ì}Ù™œ4(€€€€€€€€€€€€€€€¥µÁ½ÉĞÉ•‘¥Ì…Ì}É•‘¥Í}±¥ˆ4(€€€€€€€€€€€€€€€¡½ÍĞ°Á½ÉĞ°ÍÍ°€ô}Ù™œ ¤4(€€€€€€€€€€€€€€€ÍÕ‰}±¥•¹Ğ€ô}É•‘¥Í}±¥ˆ¹I•‘¥Ì 4(€€€€€€€€€€€€€€€€€€€¡½ÍĞõ¡½ÍĞ°4(€€€€€€€€€€€€€€€€€€€Á½ÉĞõÁ½ÉĞ°4(€€€€€€€€€€€€€€€€€€€ÍÍ°õÍÍ°°4(€€€€€€€€€€€€€€€€€€€ÍÍ±}•ÉÑ}É•ÅÌõ9½¹”°4(€€€€€€€€€€€€€€€€€€€Í½­•Ñ}½¹¹•Ñ}Ñ¥µ•½ÕĞôÔ°4(€€€€€€€€€€€€€€€€€€€Í½­•Ñ}Ñ¥µ•½ÕĞôØÀ°4(€€€€€€€€€€€€€€€€€€€‘•½‘•}É•ÍÁ½¹Í•Ìõ…±Í”°4(€€€€€€€€€€€€€€€€¤4(€€€€€€€€€€€€€€€ÁÕ‰ÍÕˆ€ôÍÕ‰}±¥•¹Ğ¹ÁÕ‰ÍÕˆ ¤4(€€€€€€€€€€€€€€€ÁÕ‰ÍÕˆ¹ÍÕ‰ÍÉ¥‰”¡}Y1-e}!990¤4(€€€€€€€€€€€€€€€±½•È¹¥¹™¼ ‰m½¹™¥5…¹…•ÉtMÕ‰ÍÉ¥‰•Ñ¼¡…¹¹•°€œ•Ìœˆ°}Y1-e}!990¤4(€€€€€€€€€€€€€€€É•ÑÉå}‘•±…ä€ô€È¸À€€ŒÉ•Í•Ğ½¸ÍÕ•ÍÍ™Õ°½¹¹•Ğ4(4(€€€€€€€€€€€€€€€™½Èµ•ÍÍ…”¥¸ÁÕ‰ÍÕˆ¹±¥ÍÑ•¸ ¤è4(€€€€€€€€€€€€€€€€€€€¥˜µ•ÍÍ…”¹•Ğ ‰ÑåÁ”ˆ¤€„ô€‰µ•ÍÍ…”ˆè4(€€€€€€€€€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”4(€€€€€€€€€€€€€€€€€€€‘…Ñ„€ôµ•ÍÍ…”¹•Ğ ‰‘…Ñ„ˆ°ˆˆˆ¤4(€€€€€€€€€€€€€€€€€€€ÑÉäè4(€€€€€€€€€€€€€€€€€€€€€€€Á…å±½…€ô©Í½¸¹±½…‘Ì¡‘…Ñ„¤4(€€€€€€€€€€€€€€€€€€€€€€€­•ä€ôÁ…å±½…¹•Ğ ‰­•äˆ¤4(€€€€€€€€€€€€€€€€€€€€€€€¥˜­•äè4(€€€€€€€€€€€€€€€€€€€€€€€€€€€Í•±˜¹}É•±½…‘}­•ä¡­•ä¤4(€€€€€€€€€€€€€€€€€€€•á•ÁĞá•ÁÑ¥½¸è4(€€€€€€€€€€€€€€€€€€€€€€€Á…ÍÌ4(4(€€€€€€€€€€€•á•ÁĞá•ÁÑ¥½¸…Ì•áŒè4(€€€€€€€€€€€€€€€±½•È¹İ…É¹¥¹œ 4(€€€€€€€€€€€€€€€€€€€€‰m½¹™¥5…¹…•Ét1¥ÍÑ•¹•È•ÉÉ½Èè€•ÌƒŠPÉ•ÑÉå¥¹œ¥¸€”¸Á™Ìˆ°4(€€€€€€€€€€€€€€€€€€€•áŒ°É•ÑÉå}‘•±…ä°4(€€€€€€€€€€€€€€€€¤4(€€€€€€€€€€€€€€€Ñ¥µ”¹Í±••À¡É•ÑÉå}‘•±…ä¤4(€€€€€€€€€€€€€€€É•ÑÉå}‘•±…ä€ôµ¥¸¡É•ÑÉå}‘•±…ä€¨€È°€ÌÀ¤4(4(4(ŒƒŠRŠR 5½‘Õ±”µ±•Ù•°Í¥¹±•Ñ½¸ƒŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠR 4(4)}M9Q%90€ô½‰©•Ğ ¤4(4(4(ŒA…Ñ •Ğ ¤Ñ¼™…±°Ñ¡É½Õ Ñ¼}U1QLİ¡•¸­•ä¥Ì…‰Í•¹Ğ™É½´…¡”…¹4(Œ¹¼•áÁ±¥¥Ğ‘•™…Õ±Ğİ…ÌÍÕÁÁ±¥•¸€Q¡¥Ìµ…­•Ì½¹™¥}µ…¹…•ÈÑ¡”Í¥¹±”Í½ÕÉ”4(Œ½˜ÑÉÕÑ è…±±•ÉÌ…¸İÉ¥Ñ”½¹™¥œ¹•Ğ ‰Á…Á•È¹‰Õ‘•Ğˆ¤İ¥Ñ ¹¼¡…É‘½‘•4(Œ™…±±‰…¬…¹ÍÑ¥±°•ĞÑ¡”•¹ØµÙ…ÈµÍ••‘•‘•™…Õ±Ğ½¸„½±ÍÑ…ÉĞ¸4)}½É¥}•Ğ€ô½¹™¥5…¹…•È¹•Ğ4(4(4)‘•˜}•Ñ}İ¥Ñ¡}‘•™…Õ±ÑÌ¡Í•±˜°­•äèÍÑÈ°‘•™…Õ±Ğè¹ä€ô}M9Q%90¤€´ø¹äè4(€€€İ¥Ñ Í•±˜¹}±½¬è4(€€€€€€€¥˜­•ä¥¸Í•±˜¹}…¡”è4(€€€€€€€€€€€É•ÑÕÉ¸Í•±˜¹}…¡•m­•åt4(€€€¥˜‘•™…Õ±Ğ¥Ì¹½Ğ}M9Q%90è4(€€€€€€€É•ÑÕÉ¸‘•™…Õ±Ğ4(€€€™…Ñ½Éä€ô}U1QL¹•Ğ¡­•ä¤4(€€€¥˜™…Ñ½Éä¥Ì¹½Ğ9½¹”è4(€€€€€€€ÑÉäè4(€€€€€€€€€€€É•ÑÕÉ¸™…Ñ½Éä ¤4(€€€€€€€•á•ÁĞá•ÁÑ¥½¸è4(€€€€€€€€€€€Á…ÍÌ4(€€€É•ÑÕÉ¸9½¹”4(4(4)½¹™¥5…¹…•È¹•Ğ€ô}•Ñ}İ¥Ñ¡}‘•™…Õ±ÑÌ€€ŒÑåÁ”è¥¹½É•mµ•Ñ¡½µ…ÍÍ¥¹t4(4)½¹™¥œ€ô½¹™¥5…¹…•È ¤4(