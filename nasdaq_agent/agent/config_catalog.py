"""UI metadata for the complete runtime configuration catalog."""
from __future__ import annotations

import re
from typing import Any


GROUPS = [
    ("scalp", "Scalping Engine", "Signal validity, real-time inputs, and deterministic bracket construction."),
    ("scalp_runtime", "Scalp Runtime", "Universe coverage, canonical plan cadence, concurrency, and session ownership."),
    ("scalp_learn", "Scalp Learning", "Bounded outcome learning, context gates, and expiring automatic actions."),
    ("scalp_ml", "Scalp ML Overlay", "Advisory TP1/TP2 probability models, economic promotion gates, and bounded confidence adjustment."),
    ("scalp_activation", "Production Activation", "Evidence required before canonical paper execution can be enabled."),
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
        "Opens valid plans in an isolated hypothetical ledger, marks them from executable bid/ask prices, and resolves TP1, TP2, stop, trail, session, and time exits without changing account P&L or learning controls.",
        "Enable this while execution is disabled to measure win rate, expectancy in R, and profit factor safely.",
    ),
    "scalp.candidate_tracking_enabled": (
        "Counterfactual candidate tracking",
        "Observes valid pre-confirmation plans and MTF-ready alternatives through their immutable stop, TP1, TP2, trail, session, and time outcomes. Candidate trials never open positions, consume risk budget, or train a model.",
        "Keep enabled to compare admitted trades with setups rejected by confirmation or policy before changing any production gate.",
    ),
    "scalp.candidate_retention_days": (
        "Candidate evidence retention",
        "Keeps resolved counterfactual setup episodes for this many calendar days. Cleanup runs at most once daily outside the one-second marking transaction.",
        "Use at least 30 days for a meaningful gate comparison while bounding PostgreSQL growth.",
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
        "5000 matches the trusted price-bus SLA and blocks quotes older than five seconds.",
    ),
    "scalp.max_bar_age_ms": (
        "Maximum indicator-bar age",
        "Blocks a live quote from being paired with stale RSI, MACD, ATR, VWAP, or RVOL calculations.",
        "120000 permits the most recently closed one-minute bar for up to two minutes.",
    ),
    "scalp.use_provisional_live_indicators": (
        "Use live provisional indicators",
        "Projects the current live quote onto the latest closed one-minute RSI/MACD state so the scanner can evaluate the current scalp instead of waiting for the next finalized bar.",
        "Enabled keeps dashboard RSI/MACD responsive while still blocking truly stale bars.",
    ),
    "scalp.provisional_max_bar_age_ms": (
        "Maximum provisional base-bar age",
        "Upper limit for using a closed bar as the base state for provisional live RSI/MACD. Older bars remain blocked as stale data.",
        "300000 allows a live quote to refresh indicators only when the base one-minute bar is no more than five minutes old.",
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
    "scalp.long_require_fast_rsi_confirmation": (
        "Require fast RSI long turn",
        "Requires RSI-2 to rise above RSI-7 before an oversold LONG reversal can become actionable, reducing falling-knife entries.",
        "Enabled means RSI-14 can be oversold, but the fast RSI must show an actual bounce.",
    ),
    "scalp.long_require_vwap_reclaim": (
        "Require long VWAP reclaim",
        "Requires LONG reversals to show a VWAP reclaim or support bounce instead of accepting a generic above-VWAP state.",
        "Enabled prevents buying a weak oversold bounce just because price is above VWAP.",
    ),
    "scalp.long_require_mtf_not_bearish": (
        "Block bearish 5-minute longs",
        "Blocks LONG reversal entries when the completed five-minute context is bearish or directly conflicts with the long setup.",
        "Enabled prevents a one-minute oversold signal from buying into a larger bearish tape.",
    ),
    "scalp.long_block_bearish_market": (
        "Block longs in bearish market",
        "Blocks LONG entries when QQQ/SPY market context is bearish, using five-minute state and one-minute VWAP/MACD evidence from the same scan.",
        "Enabled keeps individual oversold bounces observational while the broader market is selling off.",
    ),
    "scalp.short_require_fast_rsi_confirmation": (
        "Require fast RSI short rollover",
        "Requires RSI-2 to fall below RSI-7 before an overbought SHORT reversal can become actionable, reducing early shorts while price is still squeezing upward.",
        "Enabled means RSI-14 can be overbought, but the fast RSI must show an actual rollover.",
    ),
    "scalp.short_premarket_require_vwap_rejection": (
        "Require pre-market short VWAP rejection",
        "During pre-market only, requires a true VWAP rejection or resistance rejection for SHORT entries instead of accepting a generic below-VWAP state.",
        "Enabled prevents clustered pre-market shorts from firing just because price is below VWAP.",
    ),
    "scalp.short_require_mtf_not_bullish": (
        "Block bullish 5-minute shorts",
        "Blocks SHORT reversal entries when the completed five-minute context is bullish or directly conflicts with the short setup.",
        "Enabled prevents a one-minute overbought signal from shorting into a larger bullish tape.",
    ),
    "scalp.short_block_bullish_market": (
        "Block shorts in bullish market",
        "Blocks SHORT entries when QQQ/SPY market context is bullish, using five-minute state and one-minute VWAP/MACD evidence from the same scan.",
        "Enabled keeps individual overbought shorts observational while the broader market is squeezing higher.",
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
    "scalp.shadow_fixed_risk_enabled": (
        "Fixed-risk shadow sizing",
        "Sizes shadow trades from a fixed dollar risk budget per trade before applying session and learning multipliers.",
        "Enabled makes reports compare strategy quality instead of different dollar risk per ticker.",
    ),
    "scalp.shadow_risk_per_trade_usd": (
        "Shadow risk dollars",
        "Dollar amount the shadow ledger may risk before policy, session, and learning reductions.",
        "25 with a $1.00 stop opens about 25 shares; with a $5.00 stop it opens about 5 shares.",
    ),
    "scalp.shadow_min_shares": (
        "Shadow minimum shares",
        "Smallest share count allowed after fixed-risk sizing and reductions.",
        "1 keeps very wide-stop trades observable when the risk budget can support at least one share.",
    ),
    "scalp.shadow_max_shares": (
        "Shadow maximum shares",
        "Hard cap on share count after fixed-risk sizing so very tight stops cannot create unrealistic share counts.",
        "500 prevents penny-wide stops from opening thousands of simulated shares.",
    ),
    "scalp.entry_quality_gate_enabled": (
        "TP1 reachability gate",
        "Requires a valid directional setup to pass a separate execution-quality assessment using spread-to-risk, RVOL, ATR regime, RSI extremity, and an actual VWAP reclaim or rejection. It never changes side, stop, TP1, or TP2.",
        "Enabled rejects technically valid setups whose evidence is too weak to justify risking one stop unit before TP1.",
    ),
    "scalp.entry_quality_min_score": (
        "Regular-session reachability score",
        "Minimum transparent TP1-reachability score required for PRIME and STANDARD entries. The score ranks execution quality; it is not presented as a calibrated probability.",
        "65 was selected from a 30-day shadow audit and must be forward-validated before canonical execution.",
    ),
    "scalp.entry_quality_min_score_extended": (
        "Extended-session reachability score",
        "Minimum reachability score for PRE_MARKET and AFTER_HOURS entries, where spreads and quote depth are less reliable.",
        "70 is deliberately stricter than regular hours because extended-session execution has greater liquidity risk.",
    ),
    "scalp.entry_quality_require_positive_ml_ev": (
        "Require positive champion EV",
        "When a validated ML champion exists, requires its pre-entry expected-R estimate to clear the configured floor. No unpromoted or missing model can block a trade through this rule.",
        "Enabled makes ML a downside gate only after chronological economic promotion succeeds.",
    ),
    "scalp.entry_quality_min_ml_expected_r": (
        "Minimum champion expected R",
        "Minimum model-implied expected return after the champion estimates TP1-before-stop and TP2-before-stop probabilities.",
        "0.05 requires at least +0.05R modeled expectancy after the configured two-stage payoff.",
    ),
    "scalp.entry_quality_empirical_gate_enabled": (
        "Empirical context EV guard",
        "Uses closed outcomes from the matching context or broader setup + side + session to prevent repeatedly funding a context with negative observed expectancy.",
        "In shadow mode a negative mature context receives a small probe allocation; canonical paper execution is blocked.",
    ),
    "scalp.entry_quality_empirical_min_samples": (
        "Empirical EV sample floor",
        "Minimum matching closed outcomes required before observed mean expectancy may control execution.",
        "10 avoids treating one or two random outcomes as a durable economic conclusion.",
    ),
    "scalp.entry_quality_min_empirical_expectancy_r": (
        "Minimum empirical expected R",
        "Lowest accepted mean R expectancy for a mature learned context.",
        "0.0 requires the context to be non-negative; shadow probes continue collecting recovery evidence below the floor.",
    ),
    "scalp.entry_quality_shadow_probe_size_mult": (
        "Negative-context probe size",
        "Shadow-only size multiplier used when a mature context is empirically negative, preserving forward evidence while sharply reducing simulated dollar exposure.",
        "0.10 observes the context at ten percent of normal shadow risk; canonical paper remains blocked.",
    ),
    "scalp.entry_confirmation_enabled": (
        "Stable-entry confirmation",
        "Requires an eligible plan to remain valid across several scalp-engine cycles before execution. It filters transient provisional RSI, MACD, and VWAP flips.",
        "Enabled confirms inside the current scalp minute instead of waiting a full bar and entering late.",
    ),
    "scalp.entry_confirmation_seconds": (
        "Confirmation dwell seconds",
        "Minimum continuous time the same ticker, side, and setup must remain eligible before entry.",
        "15 seconds spans roughly three five-second runtime observations.",
    ),
    "scalp.entry_confirmation_min_observations": (
        "Confirmation observations",
        "Minimum number of independent runtime observations required during the dwell window.",
        "3 prevents a delayed second cycle from counting as stable evidence by itself.",
    ),
    "scalp.entry_confirmation_max_chase_r": (
        "Maximum confirmation chase",
        "Maximum favorable price movement, measured in initial R, allowed between first detection and confirmed entry. Larger moves reset confirmation rather than chasing.",
        "0.25 prevents entering after price has already consumed more than one-quarter of the route to TP1.",
    ),
    "scalp.mtf_enabled": (
        "Five-minute scalp context",
        "Calculates context from fully closed five-minute bars while live quotes and one-minute bars remain the only execution clock.",
        "Enabled adds five-minute trend, RSI, MACD, ATR, and VWAP evidence without changing entries in SHADOW mode.",
    ),
    "scalp.mtf_mode": (
        "Multi-timeframe rollout mode",
        "OFF disables the assessment. SHADOW records and displays reversal versus momentum-pullback evidence but cannot change validity, confidence, size, stops, or targets.",
        "Keep SHADOW until chronological net-of-cost validation proves positive expectancy.",
    ),
    "scalp.mtf_max_bar_age_ms": (
        "Maximum closed 5m bar age",
        "Marks five-minute context unavailable when its last fully closed bar is older than this limit. The current incomplete five-minute bucket is never used.",
        "420000 allows seven minutes from the close of the last completed five-minute bar.",
    ),
    "scalp.momentum_shadow_enabled": (
        "Evaluate momentum pullbacks",
        "Builds shadow continuation candidates from aligned closed-five-minute trend plus one-minute RSI pullback, MACD reacceleration, VWAP direction, and RVOL.",
        "Enabled measures continuation opportunities that the original reversal-only engine cannot detect.",
    ),
    "scalp.reversal_shadow_enabled": (
        "Evaluate scalp reversals",
        "Builds shadow reversal candidates from one-minute RSI extremes, MACD turn, VWAP reaction, and five-minute conflict detection.",
        "Enabled keeps the original reversal thesis measurable beside momentum pullbacks.",
    ),
    "scalp.momentum_long_rsi_min": (
        "Momentum LONG RSI floor",
        "Lowest closed one-minute RSI-14 accepted as a pullback inside a bullish five-minute scalp context.",
        "45 avoids treating a deeply oversold breakdown as an ordinary continuation pullback.",
    ),
    "scalp.momentum_long_rsi_max": (
        "Momentum LONG RSI ceiling",
        "Highest closed one-minute RSI-14 accepted before a bullish continuation entry is considered too extended.",
        "70 rejects chasing an already overbought one-minute move.",
    ),
    "scalp.momentum_short_rsi_min": (
        "Momentum SHORT RSI floor",
        "Lowest closed one-minute RSI-14 accepted before a bearish continuation entry is considered too extended.",
        "30 rejects chasing an already oversold one-minute move lower.",
    ),
    "scalp.momentum_short_rsi_max": (
        "Momentum SHORT RSI ceiling",
        "Highest closed one-minute RSI-14 accepted as a bounce inside a bearish five-minute scalp context.",
        "55 limits short continuation candidates to modest countertrend bounces.",
    ),
}

_SCALP_LEARN_DETAILS: dict[str, tuple[str, str, str]] = {
    "scalp_learn.enabled": ("Immediate outcome learning", "Updates context statistics whenever a SCALP_PLAN_V1 paper or enabled shadow trade closes. Disabling it preserves outcomes but stops automatic gate changes.", "Keep enabled in paper/shadow mode to measure same-session adaptation."),
    "scalp_learn.shadow_outcomes_enabled": ("Learn from shadow mistakes", "Turns closed shadow trades into negative-ID learning outcomes. This never changes paper P&L, but it lets the next matching setup reduce size, raise confidence, or block automatically.", "Enabled means a bad simulated LONG RECLAIM setup can tighten the next similar setup without human intervention."),
    "scalp_learn.rolling_window_min": ("Rolling context window", "Limits decisions to outcomes closed within this many minutes for the exact setup context.", "120 evaluates the most recent two hours."),
    "scalp_learn.min_samples_to_adjust": ("Samples before adjustment", "Minimum matching outcomes required before confidence or size may be tightened by the normal rolling-expectancy path.", "3 reacts within the same failure cluster while still ignoring one isolated bad print."),
    "scalp_learn.min_samples_to_block": ("Samples before blocking", "Minimum matching outcomes required before a context may be temporarily blocked.", "12 requires a broader failure cluster than a size reduction."),
    "scalp_learn.ewma_alpha": ("Expectancy EWMA alpha", "Weight assigned to the newest outcome when calculating rolling R expectancy.", "0.25 gives the newest trade 25% weight."),
    "scalp_learn.negative_reduce_r": ("Size-reduction expectancy", "Triggers reduced size when EWMA expectancy falls to this R value after the adjustment sample floor.", "-0.05R reduces exposure before a hard block."),
    "scalp_learn.negative_block_r": ("Context-block expectancy", "Allows a temporary block when EWMA expectancy and posterior win rate are both weak.", "-0.20R requires materially negative recent expectancy."),
    "scalp_learn.block_win_rate": ("Block posterior win rate", "Bayesian posterior win-rate ceiling required together with negative expectancy for a hard block.", "0.40 requires the smoothed win rate to be below 40%."),
    "scalp_learn.confidence_win_rate": ("Confidence-raise win rate", "Raises the confidence floor when posterior win rate is weak but hard-block conditions are not met.", "0.48 tightens contexts below a 48% smoothed win rate."),
    "scalp_learn.base_confidence_floor": ("Base learned confidence floor", "Starting confidence requirement used for a CONFIDENCE_RAISE action.", "60 plus a 10-point step creates a 70% floor."),
    "scalp_learn.confidence_raise_step": ("Confidence raise step", "Percentage points added to the learned confidence floor for a weak context.", "10 raises a 60% floor to 70%."),
    "scalp_learn.size_reduce_mult": ("Learned size multiplier", "Multiplier applied while SIZE_REDUCE is active. Learning may reduce, but never increase, size.", "0.50 halves the planned position."),
    "scalp_learn.dollar_guard_enabled": ("Dollar-aware learning guard", "Lets the context gate react when recent trades are profitable in R but losing real dollars because of size, fills, spread, or slippage.", "Enabled catches a +0.15R day that is still -$50 and reduces the next matching setup."),
    "scalp_learn.negative_reduce_dollar": ("Dollar size-reduction threshold", "Recent same-context dollar P&L at or below this value triggers a temporary size reduction after the adjustment sample floor.", "-25 reduces size when the rolling context has lost at least twenty-five dollars."),
    "scalp_learn.negative_block_dollar": ("Dollar block threshold", "Recent same-context dollar P&L at or below this value can temporarily block the setup once the block sample floor is met and win-rate evidence is weak.", "-100 blocks a context that repeatedly loses real dollars even if a few partial exits look positive in R."),
    "scalp_learn.fast_stop_circuit_enabled": ("Fast stop circuit", "Immediately tightens a setup context when clustered stop exits occur inside the fast-stop window, even before the normal rolling sample floor would react.", "Enabled catches three near-back-to-back STOP exits in the same context during a noisy tape."),
    "scalp_learn.fast_stop_window_min": ("Fast stop window", "Minutes used to count clustered stop exits for the fast circuit.", "10 means only stops from the last ten minutes count toward the circuit."),
    "scalp_learn.fast_stop_count": ("Fast stop count", "Number of losing stop exits in the fast-stop window required to trigger immediate size reduction.", "3 turns three same-context stop exits into an automatic risk tightening action."),
    "scalp_learn.fast_stop_size_mult": ("Fast stop size multiplier", "Temporary position-size multiplier applied by the fast stop circuit. It can only reduce exposure.", "0.25 means the next matching setup trades at one-quarter size until the action expires or context recovers."),
    "scalp_learn.setup_session_fast_stop_block_enabled": ("Setup-session fast stop block", "Escalates the broad setup + side + session gate to a temporary block when repeated stop exits cluster inside the fast-stop window.", "Enabled blocks the next PRE_MARKET short cluster after two rapid stop exits instead of only reducing size after three."),
    "scalp_learn.setup_session_fast_stop_block_count": ("Setup-session stop block count", "Number of rapid stop exits in the same setup + side + session required to trigger a temporary cooldown block.", "2 blocks the third matching setup/session attempt inside the fast-stop window."),
    "scalp_learn.pre_tp1_failure_circuit_enabled": ("Pre-TP1 failure circuit", "Counts every losing trade that failed to reach TP1, including TIME_STOP as well as STOP. This catches weak follow-through that the stop-only circuit misses.", "Enabled lets two non-follow-through outcomes tighten the next matching setup immediately."),
    "scalp_learn.pre_tp1_failure_window_min": ("Pre-TP1 failure window", "Rolling minutes used to count losing outcomes that never reached TP1.", "120 captures recurring non-follow-through across the current two-hour market context."),
    "scalp_learn.pre_tp1_failure_count": ("Pre-TP1 failure count", "Number of losing no-TP1 outcomes required to trigger the immediate circuit.", "2 reacts to a repeated failure while still ignoring one isolated loss."),
    "scalp_learn.pre_tp1_failure_size_mult": ("Pre-TP1 probe size", "Temporary multiplier for the exact learned context after the pre-TP1 failure count is reached.", "0.25 keeps exact-context observation at one-quarter size."),
    "scalp_learn.setup_session_pre_tp1_block_enabled": ("Block repeated setup-session failures", "Escalates repeated pre-TP1 failures across the broader setup + side + session into a temporary block.", "Enabled prevents a third same-session reversal after two different tickers both fail before TP1."),
    "scalp_learn.setup_session_gate_enabled": ("Broad setup-session learning", "Maintains an additional setup + side + session gate so clustered failures tighten even when RSI/VWAP buckets differ slightly.", "Enabled lets losing OVERSOLD_MACD_TURN_LONG + LONG + STANDARD trades reduce the next similar long in the same session."),
    "scalp_learn.action_ttl_min": ("Learning action lifetime", "Minutes before an automatic action expires unless refreshed by another outcome.", "60 makes every automatic action reversible within one hour."),
}

_SCALP_RUNTIME_DETAILS: dict[str, tuple[str, str, str]] = {
    "scalp_activation.required_market_days": (
        "Consecutive compliant market days",
        "Number of recent active market days that must all pass scanner latency, quote coverage, and data-completeness gates before canonical paper execution is allowed.",
        "5 requires one complete trading week of operational evidence; weekends are naturally excluded.",
    ),
    "scalp.execution_max_spread_bps": (
        "Execution-universe maximum spread",
        "Keeps every ticker visible and analyzed, but prevents shadow or canonical execution when the live bid/ask spread indicates insufficient liquidity.",
        "30 permits a spread up to 0.30% of price; the stricter spread-to-risk rule still applies to each bracket.",
    ),
    "scalp.execution_min_median_minute_dollar_volume": (
        "Execution-universe minimum minute liquidity",
        "Minimum median traded dollar value per positive-volume one-minute bar across the recent session history. This separates the monitored universe from the executable universe without hiding symbols.",
        "25000 requires a typical traded minute to represent at least $25,000 of notional volume.",
    ),
    "scalp.candidate_episode_cooldown_min": (
        "Independent candidate cooldown",
        "Minutes after a candidate episode resolves before the same ticker, side, candidate type, and strategy family can start another statistical trial.",
        "15 prevents consecutive one-minute observations of one setup from being counted as independent evidence.",
    ),
    "scalp_activation.min_cycles_per_day": (
        "Minimum measured cycles per day",
        "Prevents a short healthy window from representing an entire market day in the activation decision.",
        "300 requires at least five hours of one-per-minute persisted active-session observations per day.",
    ),
    "scalp_activation.max_cycle_p95_ms": (
        "Maximum scanner p95 latency",
        "The 95th percentile full-universe cycle duration allowed on every qualifying market day.",
        "8000 means at least 95% of measured cycles complete within eight seconds.",
    ),
    "scalp_activation.max_data_gap_pct": (
        "Maximum plan data-gap rate",
        "Largest average share of canonical plans with missing or stale required market inputs on a qualifying day.",
        "5 allows no more than five data-gap plans per hundred plans evaluated.",
    ),
    "scalp_activation.min_quote_coverage_pct": (
        "Minimum trusted quote coverage",
        "Minimum average share of the eligible universe with fresh WebSocket or explicitly identified REST-fallback quotes.",
        "95 requires trusted quotes for at least 95 percent of ticker observations.",
    ),
    "scalp_activation.min_canonical_trials": (
        "Minimum resolved canonical trials",
        "Resolved CANONICAL_VALID counterfactual trials required before execution can rely on statistical evidence.",
        "100 avoids activating from a handful of unusually favorable setups.",
    ),
    "scalp_activation.min_expectancy_r": (
        "Minimum canonical expectancy",
        "Average realized R per resolved canonical trial must be strictly above this value.",
        "0 requires positive out-of-sample expectancy after losses and partial exits.",
    ),
    "scalp_activation.min_profit_factor": (
        "Minimum canonical profit factor",
        "Gross winning R divided by gross losing R required across resolved canonical trials.",
        "1.10 requires ten percent more gross winning R than gross losing R.",
    ),
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
        "Closed one-minute OHLCV bars retained for RSI, MACD, ATR, session-reset VWAP, RVOL, and local structure. Values below 390 are rejected because they cannot represent a complete regular session.",
        "2500 preserves several sessions so RVOL can compare the same minute of day instead of unrelated bars.",
    ),
    "scalp_runtime.mtf_bar_lookback": (
        "Five-minute context source bars",
        "One-minute bars used to build completed five-minute context. This is intentionally bounded because multi-session RVOL profiling is owned by the one-minute tier.",
        "500 covers a complete regular session plus warm-up without reprocessing all 2500 profile bars.",
    ),
    "scalp_runtime.blocked_sessions": (
        "Blocked entry sessions",
        "Sessions in which plans remain visible but cannot become actionable or open a paper position.",
        "CLOSED, RESTRICTED, CLOSING_CAUTION, and HARD_CLOSE blocks unsafe entry windows.",
    ),
    "scalp_runtime.execution_policy_enabled": (
        "Shared execution preflight",
        "Applies the same session, market-data, daily-loss, trade-count, and concurrency checks before a valid plan may enter shadow or canonical paper execution.",
        "Keep enabled so shadow results represent trades the production risk contract would actually allow.",
    ),
    "scalp_runtime.execution_blocked_sessions": (
        "Execution-only blocked sessions",
        "Keeps technically valid plans visible as candidates while preventing them from consuming simulated capital in unsafe sessions.",
        "LUNCH_BLOCK remains observable but does not enter executable shadow until it proves positive expectancy.",
    ),
    "scalp_runtime.require_live_execution_data": (
        "Require trusted execution data",
        "Blocks new positions when Schwab requires authorization, the price bus is stale, the quote source is not tradable, or the ticker quote exceeds the configured age.",
        "Existing positions continue to be monitored, but no new risk is added during degraded data.",
    ),
    "scalp_runtime.pre_market_size_mult": ("Pre-market execution size", "Multiplier applied after risk-based sizing for pre-market entries.", "0.35 uses 35% of the normal risk-sized position."),
    "scalp_runtime.restricted_size_mult": ("Opening-price-discovery size", "Execution multiplier for the restricted opening window. Zero disables entries while plans remain visible.", "0 blocks 9:30-9:44 ET execution."),
    "scalp_runtime.prime_size_mult": ("Prime-session execution size", "Multiplier for the highest-quality regular-session window.", "1.0 uses the full risk-sized position."),
    "scalp_runtime.lunch_size_mult": ("Midday execution size", "Multiplier for the low-volume midday window. A blocked session remains blocked even when this value is nonzero.", "0 keeps lunch candidates observational only."),
    "scalp_runtime.standard_size_mult": ("Standard-session execution size", "Multiplier for post-lunch regular-session entries.", "0.80 uses 80% of normal size."),
    "scalp_runtime.closing_size_mult": ("Closing-caution execution size", "Multiplier during the closing-caution window.", "0 prevents new scalps near forced liquidation."),
    "scalp_runtime.after_hours_size_mult": ("After-hours execution size", "Multiplier for eligible after-hours plans that pass spread and live-data checks.", "0.30 uses 30% of normal size."),
    "scalp_runtime.hard_close_size_mult": ("Hard-close execution size", "Multiplier inside the mandatory liquidation window.", "0 prevents all new entries."),
    "scalp_runtime.closed_size_mult": ("Closed-market execution size", "Multiplier while the market is closed.", "0 prevents all new entries."),
    "scalp_runtime.position_max_quote_age_ms": (
        "Maximum position-mark quote age",
        "Prevents stale or snapshot quotes from triggering TP1, TP2, trailing stops, or stop losses on an open simulated position.",
        "5000 requires a trusted position mark no older than five seconds.",
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
    "scalp_runtime.context_cluster_throttle_enabled": (
        "Context cluster throttle",
        "Prevents the same setup/side/session/context from opening too many shadow or paper entries inside a short window.",
        "Enabled stops a fourth identical context entry after three already opened in ten minutes.",
    ),
    "scalp_runtime.context_cluster_window_min": (
        "Context cluster window",
        "Minutes used by the execution policy to count recent same-context entries.",
        "10 means the throttle only considers entries opened in the last ten minutes.",
    ),
    "scalp_runtime.context_cluster_max_entries": (
        "Max same-context entries",
        "Maximum entries allowed for the same learned context inside the cluster window before new entries are blocked.",
        "3 lets the system test a setup cluster but prevents a full-universe stampede.",
    ),
    "scalp_runtime.context_cluster_use_setup_session": (
        "Throttle setup-session clusters",
        "Counts setup + side + session entries together instead of only exact RSI/VWAP context matches, catching correlated ticker clusters earlier.",
        "Enabled treats QQQ, TQQQ, SOXL, NVDA, and MU pre-market shorts as the same execution idea.",
    ),
    "scalp_runtime.setup_session_cluster_max_entries": (
        "Max setup-session entries",
        "Maximum entries allowed for the same setup + side + session inside the cluster window before new entries are blocked.",
        "2 lets the engine test one or two pre-market shorts, then waits for outcomes before adding more.",
    ),
}

_SCALP_ML_DETAILS: dict[str, tuple[str, str, str]] = {
    "scalp_ml.training_enabled": ("Enable ML training", "Allows the isolated learner service to train TP1-before-stop and TP2-before-stop challengers from closed SCALP_PLAN_V1 outcomes. It never trains in scanner or web-api.", "Keep disabled until enough canonical outcomes exist; enabling does not activate execution."),
    "scalp_ml.auto_train_when_ready": ("Auto-start ML at sample gate", "Arms the isolated learner now and automatically starts challenger training once the minimum canonical outcome count is reached. Before the gate, it observes without creating empty challenger evaluations.", "Enabled with a 200-outcome floor means training starts automatically at outcome 200."),
    "scalp_ml.shadow_enabled": ("Enable shadow predictions", "Loads the promoted champion and records probabilities on valid scalp plans without changing confidence or execution.", "Use shadow mode first to compare predictions with realized outcomes."),
    "scalp_ml.overlay_enabled": ("Apply confidence overlay", "Applies the promoted model's bounded confidence adjustment. It cannot create a setup, change bracket levels, alter size, or bypass blockers.", "Enable only after shadow calibration is economically validated."),
    "scalp_ml.training_interval_min": ("Training interval", "Minutes between challenger training attempts inside the learner container.", "60 evaluates a fresh challenger at most once per hour."),
    "scalp_ml.training_lookback_days": ("Training lookback", "Maximum age of canonical outcomes included in a challenger dataset so obsolete regimes cannot dominate current scalping behavior.", "60 trains only from the most recent sixty calendar days."),
    "scalp_ml.maximum_model_age_hours": ("Maximum champion age", "Rejects inference from a champion older than this many hours. Missing or stale ML always fails open to deterministic confidence.", "168 expires a champion after seven days without successful revalidation."),
    "scalp_ml.minimum_samples": ("Minimum training outcomes", "Minimum closed canonical outcomes required before a challenger may be fitted.", "200 prevents promotion from a tiny sample."),
    "scalp_ml.bootstrap_minimum_samples": ("Bootstrap evaluation outcomes", "Smaller sample floor that lets the learner fit and reject early challengers for visibility before promotion is allowed.", "75 starts useful diagnostics earlier; promotion still requires the main minimum training outcomes gate."),
    "scalp_ml.bootstrap_training_enabled": ("Enable bootstrap evaluation", "Allows early challenger fitting from the bootstrap sample floor while retaining full promotion gates.", "Enabled means the learner can explain why a model is not ready instead of only saying waiting for samples."),
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
