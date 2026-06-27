"""UI metadata for the complete runtime configuration catalog."""
from __future__ import annotations

import re
from typing import Any


GROUPS = [
    ("scalp", "Scalping Engine", "Signal validity, real-time inputs, and deterministic bracket construction."),
    ("scalp_runtime", "Scalp Runtime", "Universe coverage, canonical plan cadence, concurrency, and session ownership."),
    ("scalp_learn", "Scalp Learning", "Bounded outcome learning, context gates, and expiring automatic actions."),
    ("scalp_ml", "Scalp ML Overlay", "Advisory TP1/TP2 probability models, economic promotion gates, and bounded confidence adjustment."),
    ("paper", "Paper Trading", "Simulated account, position limits, exits, and live-like execution behavior."),
    ("risk", "Risk Controls", "Account protection, exposure, cooldowns, and adaptive execution brakes."),
    ("execution", "Execution Model", "Spread, slippage, liquidity, and stop-fill simulation."),
    ("prediction", "Legacy Prediction", "Legacy composite prediction controls retained during scalp cutover."),
    ("sizing", "Position Sizing", "Confidence and risk-based share sizing."),
    ("scanner", "Scanner", "Universe coverage, cadence, concurrency, and timeout behavior."),
    ("learner", "Learning", "Continuous model training and bounded adaptive learning cadence."),
    ("filter", "Adaptive Filter", "Context performance thresholds and confidence adjustments."),
    ("algos", "Algorithm Families", "Per-family execution, confidence, size, and session overrides."),
    ("sr", "Support and Resistance", "Swing, pivot, Fibonacci, value-area, and clustering sensitivity."),
    ("macro", "Market Context", "Economic event throttles and context risk controls."),
    ("trading", "Trading Account", "Live/paper mode and account-level sizing values."),
    ("broker", "Broker", "Broker execution mode and order-routing controls."),
    ("audit", "Audit and Retention", "Decision audit behavior and retention windows."),
    ("system", "System", "Advanced runtime controls not owned by another domain."),
]

_GROUP_BY_PREFIX = {prefix: (group, description) for prefix, group, description in GROUPS}

_SCALP_DETAILS: dict[str, tuple[str, str, str]] = {
    "scalp.shadow_enabled": (
        "Shadow evaluation",
        "Build and display new scalp plans without allowing them to open paper trades. Keep this enabled during comparison and rollout.",
        "Enabled with execution disabled lets you compare plans safely against the legacy engine.",
    ),
    "scalp.execution_enabled": (
        "Scalp execution cutover",
        "Makes a valid ScalpSignalPlan mandatory for every new paper entry. Legacy prediction and algorithm calls cannot open trades while enabled.",
        "Enable only after shadow plans show fresh quotes, complete indicators, and expected bracket geometry.",
    ),
    "scalp.reward_r": (
        "TP2 reward multiple",
        "Sets TP2 as a multiple of initial per-share risk. This changes bracket geometry; it is not a signal-direction gate.",
        "2.0 means a $1.00 stop distance produces TP2 $2.00 from entry.",
    ),
    "scalp.tp1_r": (
        "TP1 reward multiple",
        "Sets the partial-profit level as a multiple of initial risk. It must be positive and cannot exceed TP2 reward.",
        "1.0 takes the first partial exit after price moves one initial risk unit.",
    ),
    "scalp.stop_atr_multiple": (
        "ATR stop multiple",
        "Sets the volatility component of initial stop distance using real-time one-minute ATR-14.",
        "1.2 with ATR $0.50 contributes a $0.60 minimum stop distance.",
    ),
    "scalp.min_stop_pct": (
        "Minimum stop percent",
        "Prevents stops from sitting inside ordinary price noise when ATR or spread is unusually small.",
        "0.003 means the stop cannot be closer than 0.30% of entry.",
    ),
    "scalp.max_stop_pct": (
        "Maximum stop percent",
        "Caps per-share stop distance. Plans that require a wider volatility stop are blocked instead of silently using unsafe geometry.",
        "0.02 caps initial risk distance at 2% of entry.",
    ),
    "scalp.spread_buffer_mult": (
        "Spread risk buffer",
        "Requires stop distance to cover a configurable multiple of the current bid/ask spread.",
        "2.0 with a $0.05 spread contributes at least $0.10 risk distance.",
    ),
    "scalp.tick_size": (
        "Price tick size",
        "Rounds entries, stops, and targets to a tradable price increment.",
        "0.01 is appropriate for standard US equities quoted in cents.",
    ),
    "scalp.max_quote_age_ms": (
        "Maximum quote age",
        "Blocks plans when the Level 1 quote is older than this threshold, even if the service connection still reports healthy.",
        "2000 blocks quotes older than two seconds.",
    ),
    "scalp.max_bar_age_ms": (
        "Maximum indicator-bar age",
        "Blocks a live quote from being paired with stale RSI, MACD, ATR, VWAP, or RVOL calculations.",
        "120000 permits the most recently closed one-minute bar for up to two minutes.",
    ),
    "scalp.max_spread_to_risk": (
        "Maximum spread-to-risk",
        "Limits transaction cost relative to the planned stop distance.",
        "0.25 blocks a $0.30 spread when initial risk is only $1.00.",
    ),
    "scalp.min_rvol_regular": (
        "Regular-session minimum RVOL",
        "Requires sufficient time-of-day relative volume before a regular-session scalp is actionable.",
        "0.8 requires at least 80% of the expected volume pace.",
    ),
    "scalp.min_rvol_extended": (
        "Extended-session minimum RVOL",
        "Applies the relative-volume floor during pre-market and after-hours sessions.",
        "0.4 permits lower extended-hours volume while still rejecting inactive symbols.",
    ),
    "scalp.rsi_oversold": (
        "RSI oversold boundary",
        "Defines the highest RSI-14 value eligible for an oversold LONG reversal candidate.",
        "30 marks RSI-14 values at or below 30 as oversold.",
    ),
    "scalp.rsi_extreme_oversold": (
        "RSI extreme-oversold boundary",
        "Separates extreme downside momentum from the ordinary oversold zone for explanation and learning context.",
        "20 classifies RSI-14 values at or below 20 as EXTREME_OS.",
    ),
    "scalp.rsi_overbought": (
        "RSI overbought boundary",
        "Defines the lowest RSI-14 value eligible for an overbought SHORT reversal candidate.",
        "70 marks RSI-14 values at or above 70 as overbought.",
    ),
    "scalp.rsi_extreme_overbought": (
        "RSI extreme-overbought boundary",
        "Separates extreme upside momentum from the ordinary overbought zone for explanation and learning context.",
        "80 classifies RSI-14 values at or above 80 as EXTREME_OB.",
    ),
    "scalp.require_vwap_event": (
        "Require VWAP confirmation",
        "Requires LONG candidates to reclaim/hold above VWAP and SHORT candidates to reject/hold below VWAP.",
        "Disable only for controlled experiments because direction otherwise lacks price-location confirmation.",
    ),
    "scalp.require_macd_confirm": (
        "Require MACD turn",
        "Requires the real-time MACD histogram to improve for LONG or deteriorate for SHORT before entry.",
        "A LONG passes when the current histogram is greater than the prior histogram or crosses above zero.",
    ),
    "scalp.require_rsi_zone": (
        "Require RSI extreme",
        "Requires oversold RSI for LONG and overbought RSI for SHORT in the initial reversal setup family.",
        "Keep enabled until additional deterministic continuation families are implemented.",
    ),
    "scalp.allow_rest_fallback_trading": (
        "Allow REST fallback entries",
        "Allows paper execution from fresh REST quotes when WebSocket data is unavailable. Source remains visible on every plan.",
        "Keep disabled for strict live-data parity; enable only to test degraded-mode behavior.",
    ),
    "scalp.block_when_path_obstructed": (
        "Block obstructed TP2 path",
        "Blocks a LONG when resistance lies before TP2, or a SHORT when support lies before TP2. Structure never rewrites TP2.",
        "A LONG entry at $100 with TP2 $102 is blocked by resistance at $101.40.",
    ),
    "scalp.block_when_risk_capped": (
        "Block capped volatility risk",
        "Rejects a plan when ATR/spread requires a stop wider than the configured maximum rather than squeezing the stop artificially.",
        "If ATR requires 2.5% but max stop is 2%, the plan is watch-only.",
    ),
}

_SCALP_LEARN_DETAILS: dict[str, tuple[str, str, str]] = {
    "scalp_learn.enabled": ("Immediate outcome learning", "Updates context statistics whenever a SCALP_PLAN_V1 paper trade closes. Disabling it preserves outcomes but stops automatic gate changes.", "Keep enabled in paper mode to measure same-session adaptation."),
    "scalp_learn.rolling_window_min": ("Rolling context window", "Limits decisions to outcomes closed within this many minutes for the exact setup context.", "120 evaluates the most recent two hours."),
    "scalp_learn.min_samples_to_adjust": ("Samples before adjustment", "Minimum matching outcomes required before confidence or size may be tightened.", "5 prevents one isolated loss from changing execution."),
    "scalp_learn.min_samples_to_block": ("Samples before blocking", "Minimum matching outcomes required before a context may be temporarily blocked.", "12 requires a broader failure cluster than a size reduction."),
    "scalp_learn.ewma_alpha": ("Expectancy EWMA alpha", "Weight assigned to the newest outcome when calculating rolling R expectancy.", "0.25 gives the newest trade 25% weight."),
    "scalp_learn.negative_reduce_r": ("Size-reduction expectancy", "Triggers reduced size when EWMA expectancy falls to this R value after the adjustment sample floor.", "-0.05R reduces exposure before a hard block."),
    "scalp_learn.negative_block_r": ("Context-block expectancy", "Allows a temporary block when EWMA expectancy and posterior win rate are both weak.", "-0.20R requires materially negative recent expectancy."),
    "scalp_learn.block_win_rate": ("Block posterior win rate", "Bayesian posterior win-rate ceiling required together with negative expectancy for a hard block.", "0.40 requires the smoothed win rate to be below 40%."),
    "scalp_learn.confidence_win_rate": ("Confidence-raise win rate", "Raises the confidence floor when posterior win rate is weak but hard-block conditions are not met.", "0.48 tightens contexts below a 48% smoothed win rate."),
    "scalp_learn.base_confidence_floor": ("Base learned confidence floor", "Starting confidence requirement used for a CONFIDENCE_RAISE action.", "60 plus a 10-point step creates a 70% floor."),
    "scalp_learn.confidence_raise_step": ("Confidence raise step", "Percentage points added to the learned confidence floor for a weak context.", "10 raises a 60% floor to 70%."),
    "scalp_learn.size_reduce_mult": ("Learned size multiplier", "Multiplier applied while SIZE_REDUCE is active. Learning may reduce, but never increase, size.", "0.50 halves the planned position."),
    "scalp_learn.action_ttl_min": ("Learning action lifetime", "Minutes before an automatic action expires unless refreshed by another outcome.", "60 makes every automatic action reversible within one hour."),
}

_SCALP_RUNTIME_DETAILS: dict[str, tuple[str, str, str]] = {
    "scalp_runtime.cycle_interval_s": (
        "Canonical plan interval",
        "Seconds between full-universe plan refreshes. The browser still marks prices and positions every second from the live price bus.",
        "5 refreshes all 477 canonical plans every five seconds.",
    ),
    "scalp_runtime.workers": (
        "Analysis workers",
        "Maximum worker threads used for deterministic per-ticker calculation. Broker calls never run inside these workers.",
        "8 leaves capacity for market data and the API on the four-vCPU host.",
    ),
    "scalp_runtime.bar_lookback": (
        "One-minute bar lookback",
        "Closed one-minute bars used for RSI, MACD, ATR, VWAP, RVOL, and local structure.",
        "120 supplies two hours of recent intraday evidence.",
    ),
    "scalp_runtime.blocked_sessions": (
        "Blocked entry sessions",
        "Sessions in which plans remain visible but cannot become actionable or open a paper position.",
        "CLOSED, RESTRICTED, CLOSING_CAUTION, and HARD_CLOSE blocks unsafe entry windows.",
    ),
    "scalp_runtime.require_context_data": (
        "Require fresh market context",
        "Blocks new entries when earnings and news context is missing or stale while keeping the plan visible.",
        "Enabled prevents a technical scalp from trading without a current context-intel snapshot.",
    ),
    "scalp_runtime.max_context_age_s": (
        "Maximum context age",
        "Maximum age in seconds for the ticker earnings and news snapshot.",
        "180 allows three minutes before context is treated as stale.",
    ),
    "scalp_runtime.max_context_risk_score": (
        "Maximum context risk",
        "Blocks a plan when the normalized earnings, macro, halt, or news risk score reaches this value.",
        "0.8 blocks high-risk event contexts on a zero-to-one scale.",
    ),
    "scalp_runtime.adverse_news_sentiment": (
        "Adverse news-shock threshold",
        "Blocks LONG during a negative news shock and SHORT during a positive news shock.",
        "0.25 requires absolute 30-minute sentiment of at least 0.25 together with news_shock=true.",
    ),
}

_SCALP_ML_DETAILS: dict[str, tuple[str, str, str]] = {
    "scalp_ml.training_enabled": ("Enable ML training", "Allows the isolated learner service to train TP1-before-stop and TP2-before-stop challengers from closed SCALP_PLAN_V1 outcomes. It never trains in scanner or web-api.", "Keep disabled until enough canonical outcomes exist; enabling does not activate execution."),
    "scalp_ml.shadow_enabled": ("Enable shadow predictions", "Loads the promoted champion and records probabilities on valid scalp plans without changing confidence or execution.", "Use shadow mode first to compare predictions with realized outcomes."),
    "scalp_ml.overlay_enabled": ("Apply confidence overlay", "Applies the promoted model's bounded confidence adjustment. It cannot create a setup, change bracket levels, alter size, or bypass blockers.", "Enable only after shadow calibration is economically validated."),
    "scalp_ml.training_interval_min": ("Training interval", "Minutes between challenger training attempts inside the learner container.", "60 evaluates a fresh challenger at most once per hour."),
    "scalp_ml.training_lookback_days": ("Training lookback", "Maximum age of canonical outcomes included in a challenger dataset so obsolete regimes cannot dominate current scalping behavior.", "60 trains only from the most recent sixty calendar days."),
    "scalp_ml.maximum_model_age_hours": ("Maximum champion age", "Rejects inference from a champion older than this many hours. Missing or stale ML always fails open to deterministic confidence.", "168 expires a champion after seven days without successful revalidation."),
    "scalp_ml.minimum_samples": ("Minimum training outcomes", "Minimum closed canonical outcomes required before a challenger may be fitted.", "200 prevents promotion from a tiny sample."),
    "scalp_ml.holdout_pct": ("Chronological holdout fraction", "Newest fraction of outcomes reserved strictly for out-of-sample promotion testing. No random shuffle or scaler fit touches it.", "0.25 reserves the newest 25% for validation."),
    "scalp_ml.minimum_selected_holdout": ("Minimum evaluated holdout trades", "Minimum holdout rows whose model expected-R clears the selection threshold before economic metrics are trusted.", "30 requires at least thirty independently evaluated opportunities."),
    "scalp_ml.minimum_holdout_sessions": ("Minimum validation sessions", "Number of recent market dates that must independently satisfy session stability checks.", "2 prevents one unusually strong day from promoting a model."),
    "scalp_ml.minimum_session_samples": ("Minimum trades per session", "Minimum selected holdout trades required on each recent validation session.", "5 requires meaningful coverage on both recent days."),
    "scalp_ml.minimum_expectancy_r": ("Promotion expectancy floor", "Minimum realized mean R on selected chronological holdout rows.", "0.05 requires at least +0.05R per selected trade."),
    "scalp_ml.minimum_profit_factor": ("Promotion profit-factor floor", "Minimum gross holdout wins divided by absolute gross holdout losses.", "1.10 requires ten percent more gross profit than gross loss."),
    "scalp_ml.minimum_session_expectancy_r": ("Per-session expectancy floor", "Minimum realized expectancy required independently on each recent validation session.", "0.0 prevents promotion when either recent session is negative."),
    "scalp_ml.minimum_auc": ("Minimum holdout AUC", "Minimum out-of-sample discrimination required independently for both TP1 and TP2 classifiers.", "0.52 requires each model to rank outcomes better than chance."),
    "scalp_ml.minimum_brier_improvement": ("Minimum Brier improvement", "Required probability-calibration improvement over a constant training-prevalence baseline on chronological holdout data.", "0.0 rejects a model whose probabilities are worse than the naive baseline."),
    "scalp_ml.selection_expected_r": ("Prediction selection threshold", "Minimum model-implied expected R used to include a holdout row in economic promotion evaluation.", "0.0 evaluates only model-positive opportunities."),
    "scalp_ml.confidence_points_per_r": ("Confidence sensitivity", "Percentage-point adjustment produced per one unit of model-implied expected R before caps.", "5 adds 2.5 points for +0.50 expected R."),
    "scalp_ml.max_confidence_raise": ("Maximum confidence increase", "Hard cap on positive model influence. The plan must already be valid before this can matter.", "5 prevents ML from adding more than five confidence points."),
    "scalp_ml.max_confidence_reduction": ("Maximum confidence reduction", "Hard cap on negative model influence. Negative evidence may tighten more strongly than positive evidence can relax.", "15 permits up to a fifteen-point reduction."),
}


def build_catalog(values: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    groups = [
        {"id": prefix, "label": label, "description": description}
        for prefix, label, description in GROUPS
    ]
    fields = []
    for key in sorted(set(defaults) | set(values)):
        value = values.get(key, defaults.get(key))
        default = defaults.get(key)
        prefix = key.split(".", 1)[0]
        group = prefix if prefix in _GROUP_BY_PREFIX else "system"
        label, description, example = _detail(key, value, group)
        fields.append(
            {
                "key": key,
                "group": group,
                "label": label,
                "description": description,
                "example": example,
                "value": value,
                "default": default,
                "type": _value_type(value),
                "advanced": key not in _SCALP_DETAILS and key not in _SCALP_RUNTIME_DETAILS and key not in _SCALP_LEARN_DETAILS and key not in _SCALP_ML_DETAILS,
            }
        )
    return {"schema_version": 1, "groups": groups, "fields": fields}


def _detail(key: str, value: Any, group: str) -> tuple[str, str, str]:
    if key in _SCALP_DETAILS:
        return _SCALP_DETAILS[key]
    if key in _SCALP_RUNTIME_DETAILS:
        return _SCALP_RUNTIME_DETAILS[key]
    if key in _SCALP_LEARN_DETAILS:
        return _SCALP_LEARN_DETAILS[key]
    if key in _SCALP_ML_DETAILS:
        return _SCALP_ML_DETAILS[key]
    suffix = key.split(".", 1)[-1]
    label = re.sub(r"\s+", " ", suffix.replace("_", " ")).strip().title()
    group_label = _GROUP_BY_PREFIX.get(group, ("System", ""))[0]
    description = (
        f"Runtime {group_label.lower()} setting `{key}`. It is persisted in PostgreSQL "
        "and hot-reloaded by the owning service without a container restart."
    )
    if isinstance(value, bool):
        example = "Enabled applies the behavior immediately; disabled leaves the owning feature inactive."
    elif isinstance(value, (int, float)):
        example = f"Current/default reference value: {value}. Change gradually and verify the owning service metrics."
    elif isinstance(value, list):
        example = "Enter a JSON list. Each item is preserved in order."
    else:
        example = f"Current/default reference value: {value!s}."
    return label, description, example


def _value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    return "string"
