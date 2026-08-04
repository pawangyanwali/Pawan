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
        "2000 blocks quotes older than two seconds.",
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
        "Extended-session reachability scoã¿{¶‰Ëkºwµçq…Ñ¥½¸¸	É½­•È…±±Ì¹•Ù•ÈÉÕ¸¥¹Í¥‘”Ñ¡•Í”İ½É­•ÉÌ¸ˆ°(€€€€€€€€ˆà±•…Ù•Ì…Á…¥Ñä™½Èµ…É­•Ğ‘…Ñ„…¹Ñ¡”A$½¸Ñ¡”™½ÕÈµÙAT¡½ÍĞ¸ˆ°(€€€€¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹‰…É}±½½­‰…¬ˆè€ (€€€€€€€€‰=¹”µµ¥¹ÕÑ”‰…È±½½­‰…¬ˆ°(€€€€€€€€‰±½Í•½¹”µµ¥¹ÕÑ”=!1X‰…ÉÌÉ•Ñ…¥¹•™½ÈIM$°5°QH°Í•ÍÍ¥½¸µÉ•Í•ĞY]@°IY=0°…¹±½…°ÍÑÉÕÑÕÉ”¸Y…±Õ•Ì‰•±½Ü€ÌäÀ…É”É•©•Ñ•‰•…ÕÍ”Ñ¡•ä…¹¹½ĞÉ•ÁÉ•Í•¹Ğ„½µÁ±•Ñ”É•Õ±…ÈÍ•ÍÍ¥½¸¸ˆ°(€€€€€€€€ˆÈÔÀÀÁÉ•Í•ÉÙ•ÌÍ•Ù•É…°Í•ÍÍ¥½¹ÌÍ¼IY=0…¸½µÁ…É”Ñ¡”Í…µ”µ¥¹ÕÑ”½˜‘…ä¥¹ÍÑ•…½˜Õ¹É•±…Ñ•‰…ÉÌ¸ˆ°(€€€€¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹‰±½­•‘}Í•ÍÍ¥½¹Ìˆè€ (€€€€€€€€‰	±½­••¹ÑÉäÍ•ÍÍ¥½¹Ìˆ°(€€€€€€€€‰M•ÍÍ¥½¹Ì¥¸İ¡¥ Á±…¹ÌÉ•µ…¥¸Ù¥Í¥‰±”‰ÕĞ…¹¹½Ğ‰•½µ”…Ñ¥½¹…‰±”½È½Á•¸„Á…Á•ÈÁ½Í¥Ñ¥½¸¸ˆ°(€€€€€€€€‰1=M°IMQI%Q°1=M%9}UQ%=8°…¹!I}1=M‰±½­ÌÕ¹Í…™”•¹ÑÉäİ¥¹‘½İÌ¸ˆ°(€€€€¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹•á•ÕÑ¥½¹}Á½±¥å}•¹…‰±•ˆè€ (€€€€€€€€‰M¡…É••á•ÕÑ¥½¸ÁÉ•™±¥¡Ğˆ°(€€€€€€€€‰ÁÁ±¥•ÌÑ¡”Í…µ”Í•ÍÍ¥½¸°µ…É­•Ğµ‘…Ñ„°‘…¥±äµ±½ÍÌ°ÑÉ…‘”µ½Õ¹Ğ°…¹½¹ÕÉÉ•¹ä¡•­Ì‰•™½É”„Ù…±¥Á±…¸µ…ä•¹Ñ•ÈÍ¡…‘½Ü½È…¹½¹¥…°Á…Á•È•á•ÕÑ¥½¸¸ˆ°(€€€€€€€€‰-••À•¹…‰±•Í¼Í¡…‘½ÜÉ•ÍÕ±ÑÌÉ•ÁÉ•Í•¹ĞÑÉ…‘•ÌÑ¡”ÁÉ½‘ÕÑ¥½¸É¥Í¬½¹ÑÉ…Ğİ½Õ±…ÑÕ…±±ä…±±½Ü¸ˆ°(€€€€¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹•á•ÕÑ¥½¹}‰±½­•‘}Í•ÍÍ¥½¹Ìˆè€ (€€€€€€€€‰á•ÕÑ¥½¸µ½¹±ä‰±½­•Í•ÍÍ¥½¹Ìˆ°(€€€€€€€€‰-••ÁÌÑ•¡¹¥…±±äÙ…±¥Á±…¹ÌÙ¥Í¥‰±”…Ì…¹‘¥‘…Ñ•Ìİ¡¥±”ÁÉ•Ù•¹Ñ¥¹œÑ¡•´™É½´½¹ÍÕµ¥¹œÍ¥µÕ±…Ñ•…Á¥Ñ…°¥¸Õ¹Í…™”Í•ÍÍ¥½¹Ì¸ˆ°(€€€€€€€€‰1U9!}	1=,É•µ…¥¹Ì½‰Í•ÉÙ…‰±”‰ÕĞ‘½•Ì¹½Ğ•¹Ñ•È•á•ÕÑ…‰±”Í¡…‘½ÜÕ¹Ñ¥°¥ĞÁÉ½Ù•ÌÁ½Í¥Ñ¥Ù”•áÁ•Ñ…¹ä¸ˆ°(€€€€¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹É•ÅÕ¥É•}±¥Ù•}•á•ÕÑ¥½¹}‘…Ñ„ˆè€ (€€€€€€€€‰I•ÅÕ¥É”ÑÉÕÍÑ••á•ÕÑ¥½¸‘…Ñ„ˆ°(€€€€€€€€‰	±½­Ì¹•ÜÁ½Í¥Ñ¥½¹Ìİ¡•¸M¡İ…ˆÉ•ÅÕ¥É•Ì…ÕÑ¡½É¥é…Ñ¥½¸°Ñ¡”ÁÉ¥”‰ÕÌ¥ÌÍÑ…±”°Ñ¡”ÅÕ½Ñ”Í½ÕÉ”¥Ì¹½ĞÑÉ…‘…‰±”°½ÈÑ¡”Ñ¥­•ÈÅÕ½Ñ”•á••‘ÌÑ¡”½¹™¥ÕÉ•…”¸ˆ°(€€€€€€€€‰á¥ÍÑ¥¹œÁ½Í¥Ñ¥½¹Ì½¹Ñ¥¹Õ”Ñ¼‰”µ½¹¥Ñ½É•°‰ÕĞ¹¼¹•ÜÉ¥Í¬¥Ì…‘‘•‘ÕÉ¥¹œ‘•É…‘•‘…Ñ„¸ˆ°(€€€€¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹ÁÉ•}µ…É­•Ñ}Í¥é•}µÕ±Ğˆè€ ‰AÉ”µµ…É­•Ğ•á•ÕÑ¥½¸Í¥é”ˆ°€‰5Õ±Ñ¥Á±¥•È…ÁÁ±¥•…™Ñ•ÈÉ¥Í¬µ‰…Í•Í¥é¥¹œ™½ÈÁÉ”µµ…É­•Ğ•¹ÑÉ¥•Ì¸ˆ°€ˆÀ¸ÌÔÕÍ•Ì€ÌÔ”½˜Ñ¡”¹½Éµ…°É¥Í¬µÍ¥é•Á½Í¥Ñ¥½¸¸ˆ¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹É•ÍÑÉ¥Ñ•‘}Í¥é•}µÕ±Ğˆè€ ‰=Á•¹¥¹œµÁÉ¥”µ‘¥Í½Ù•ÉäÍ¥é”ˆ°€‰á•ÕÑ¥½¸µÕ±Ñ¥Á±¥•È™½ÈÑ¡”É•ÍÑÉ¥Ñ•½Á•¹¥¹œİ¥¹‘½Ü¸i•É¼‘¥Í…‰±•Ì•¹ÑÉ¥•Ìİ¡¥±”Á±…¹ÌÉ•µ…¥¸Ù¥Í¥‰±”¸ˆ°€ˆÀ‰±½­Ì€äèÌÀ´äèĞĞP•á•ÕÑ¥½¸¸ˆ¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹ÁÉ¥µ•}Í¥é•}µÕ±Ğˆè€ ‰AÉ¥µ”µÍ•ÍÍ¥½¸•á•ÕÑ¥½¸Í¥é”ˆ°€‰5Õ±Ñ¥Á±¥•È™½ÈÑ¡”¡¥¡•ÍĞµÅÕ…±¥ÑäÉ•Õ±…ÈµÍ•ÍÍ¥½¸İ¥¹‘½Ü¸ˆ°€ˆÄ¸ÀÕÍ•ÌÑ¡”™Õ±°É¥Í¬µÍ¥é•Á½Í¥Ñ¥½¸¸ˆ¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹±Õ¹¡}Í¥é•}µÕ±Ğˆè€ ‰5¥‘‘…ä•á•ÕÑ¥½¸Í¥é”ˆ°€‰5Õ±Ñ¥Á±¥•È™½ÈÑ¡”±½ÜµÙ½±Õµ”µ¥‘‘…äİ¥¹‘½Ü¸‰±½­•Í•ÍÍ¥½¸É•µ…¥¹Ì‰±½­••Ù•¸İ¡•¸Ñ¡¥ÌÙ…±Õ”¥Ì¹½¹é•É¼¸ˆ°€ˆÀ­••ÁÌ±Õ¹ …¹‘¥‘…Ñ•Ì½‰Í•ÉÙ…Ñ¥½¹…°½¹±ä¸ˆ¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹ÍÑ…¹‘…É‘}Í¥é•}µÕ±Ğˆè€ ‰MÑ…¹‘…ÉµÍ•ÍÍ¥½¸•á•ÕÑ¥½¸Í¥é”ˆ°€‰5Õ±Ñ¥Á±¥•È™½ÈÁ½ÍĞµ±Õ¹ É•Õ±…ÈµÍ•ÍÍ¥½¸•¹ÑÉ¥•Ì¸ˆ°€ˆÀ¸àÀÕÍ•Ì€àÀ”½˜¹½Éµ…°Í¥é”¸ˆ¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹±½Í¥¹}Í¥é•}µÕ±Ğˆè€ ‰±½Í¥¹œµ…ÕÑ¥½¸•á•ÕÑ¥½¸Í¥é”ˆ°€‰5Õ±Ñ¥Á±¥•È‘ÕÉ¥¹œÑ¡”±½Í¥¹œµ…ÕÑ¥½¸İ¥¹‘½Ü¸ˆ°€ˆÀÁÉ•Ù•¹ÑÌ¹•ÜÍ…±ÁÌ¹•…È™½É•±¥ÅÕ¥‘…Ñ¥½¸¸ˆ¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹…™Ñ•É}¡½ÕÉÍ}Í¥é•}µÕ±Ğˆè€ ‰™Ñ•Èµ¡½ÕÉÌ•á•ÕÑ¥½¸Í¥é”ˆ°€‰5Õ±Ñ¥Á±¥•È™½È•±¥¥‰±”…™Ñ•Èµ¡½ÕÉÌÁ±…¹ÌÑ¡…ĞÁ…ÍÌÍÁÉ•……¹±¥Ù”µ‘…Ñ„¡•­Ì¸ˆ°€ˆÀ¸ÌÀÕÍ•Ì€ÌÀ”½˜¹½Éµ…°Í¥é”¸ˆ¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹¡…É‘}±½Í•}Í¥é•}µÕ±Ğˆè€ ‰!…Éµ±½Í”•á•ÕÑ¥½¸Í¥é”ˆ°€‰5Õ±Ñ¥Á±¥•È¥¹Í¥‘”Ñ¡”µ…¹‘…Ñ½Éä±¥ÅÕ¥‘…Ñ¥½¸İ¥¹‘½Ü¸ˆ°€ˆÀÁÉ•Ù•¹ÑÌ…±°¹•Ü•¹ÑÉ¥•Ì¸ˆ¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹±½Í•‘}Í¥é•}µÕ±Ğˆè€ ‰±½Í•µµ…É­•Ğ•á•ÕÑ¥½¸Í¥é”ˆ°€‰5Õ±Ñ¥Á±¥•Èİ¡¥±”Ñ¡”µ…É­•Ğ¥Ì±½Í•¸ˆ°€ˆÀÁÉ•Ù•¹ÑÌ…±°¹•Ü•¹ÑÉ¥•Ì¸ˆ¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹Á½Í¥Ñ¥½¹}µ…á}ÅÕ½Ñ•}…•}µÌˆè€ (€€€€€€€€‰5…á¥µÕ´Á½Í¥Ñ¥½¸µµ…É¬ÅÕ½Ñ”…”ˆ°(€€€€€€€€‰AÉ•Ù•¹ÑÌÍÑ…±”½ÈÍ¹…ÁÍ¡½ĞÅÕ½Ñ•Ì™É½´ÑÉ¥•É¥¹œQ@Ä°Q@È°ÑÉ…¥±¥¹œÍÑ½ÁÌ°½ÈÍÑ½À±½ÍÍ•Ì½¸…¸½Á•¸Í¥µÕ±…Ñ•Á½Í¥Ñ¥½¸¸ˆ°(€€€€€€€€ˆÔÀÀÀÉ•ÅÕ¥É•Ì„ÑÉÕÍÑ•Á½Í¥Ñ¥½¸µ…É¬¹¼½±‘•ÈÑ¡…¸™¥Ù”Í•½¹‘Ì¸ˆ°(€€€€¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹É•ÅÕ¥É•}½¹Ñ•áÑ}‘…Ñ„ˆè€ (€€€€€€€€‰I•ÅÕ¥É”™É•Í µ…É­•Ğ½¹Ñ•áĞˆ°(€€€€€€€€‰	±½­Ì¹•Ü•¹ÑÉ¥•Ìİ¡•¸•…É¹¥¹Ì…¹¹•İÌ½¹Ñ•áĞ¥Ìµ¥ÍÍ¥¹œ½ÈÍÑ…±”İ¡¥±”­••Á¥¹œÑ¡”Á±…¸Ù¥Í¥‰±”¸ˆ°(€€€€€€€€‰¹…‰±•ÁÉ•Ù•¹ÑÌ„Ñ•¡¹¥…°Í…±À™É½´ÑÉ…‘¥¹œİ¥Ñ¡½ÕĞ„ÕÉÉ•¹Ğ½¹Ñ•áĞµ¥¹Ñ•°Í¹…ÁÍ¡½Ğ¸ˆ°(€€€€¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹µ…á}½¹Ñ•áÑ}…•}Ìˆè€ (€€€€€€€€‰5…á¥µÕ´½¹Ñ•áĞ…”ˆ°(€€€€€€€€‰5…á¥µÕ´…”¥¸Í•½¹‘Ì™½ÈÑ¡”Ñ¥­•È•…É¹¥¹Ì…¹¹•İÌÍ¹…ÁÍ¡½Ğ¸ˆ°(€€€€€€€€ˆÄàÀ…±±½İÌÑ¡É•”µ¥¹ÕÑ•Ì‰•™½É”½¹Ñ•áĞ¥ÌÑÉ•…Ñ•…ÌÍÑ…±”¸ˆ°(€€€€¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹µ…á}½¹Ñ•áÑ}É¥Í­}Í½É”ˆè€ (€€€€€€€€‰5…á¥µÕ´½¹Ñ•áĞÉ¥Í¬ˆ°(€€€€€€€€‰	±½­Ì„Á±…¸İ¡•¸Ñ¡”¹½Éµ…±¥é••…É¹¥¹Ì°µ…É¼°¡…±Ğ°½È¹•İÌÉ¥Í¬Í½É”É•…¡•ÌÑ¡¥ÌÙ…±Õ”¸ˆ°(€€€€€€€€ˆÀ¸à‰±½­Ì¡¥ µÉ¥Í¬•Ù•¹Ğ½¹Ñ•áÑÌ½¸„é•É¼µÑ¼µ½¹”Í…±”¸ˆ°(€€€€¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹…‘Ù•ÉÍ•}¹•İÍ}Í•¹Ñ¥µ•¹Ğˆè€ (€€€€€€€€‰‘Ù•ÉÍ”¹•İÌµÍ¡½¬Ñ¡É•Í¡½±ˆ°(€€€€€€€€‰	±½­Ì1=9‘ÕÉ¥¹œ„¹•…Ñ¥Ù”¹•İÌÍ¡½¬…¹M!=IP‘ÕÉ¥¹œ„Á½Í¥Ñ¥Ù”¹•İÌÍ¡½¬¸ˆ°(€€€€€€€€ˆÀ¸ÈÔÉ•ÅÕ¥É•Ì…‰Í½±ÕÑ”€ÌÀµµ¥¹ÕÑ”Í•¹Ñ¥µ•¹Ğ½˜…Ğ±•…ÍĞ€À¸ÈÔÑ½•Ñ¡•Èİ¥Ñ ¹•İÍ}Í¡½¬õÑÉÕ”¸ˆ°(€€€€¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹½¹Ñ•áÑ}±ÕÍÑ•É}Ñ¡É½ÑÑ±•}•¹…‰±•ˆè€ (€€€€€€€€‰½¹Ñ•áĞ±ÕÍÑ•ÈÑ¡É½ÑÑ±”ˆ°(€€€€€€€€‰AÉ•Ù•¹ÑÌÑ¡”Í…µ”Í•ÑÕÀ½Í¥‘”½Í•ÍÍ¥½¸½½¹Ñ•áĞ™É½´½Á•¹¥¹œÑ½¼µ…¹äÍ¡…‘½Ü½ÈÁ…Á•È•¹ÑÉ¥•Ì¥¹Í¥‘”„Í¡½ÉĞİ¥¹‘½Ü¸ˆ°(€€€€€€€€‰¹…‰±•ÍÑ½ÁÌ„™½ÕÉÑ ¥‘•¹Ñ¥…°½¹Ñ•áĞ•¹ÑÉä…™Ñ•ÈÑ¡É•”…±É•…‘ä½Á•¹•¥¸Ñ•¸µ¥¹ÕÑ•Ì¸ˆ°(€€€€¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹½¹Ñ•áÑ}±ÕÍÑ•É}İ¥¹‘½İ}µ¥¸ˆè€ (€€€€€€€€‰½¹Ñ•áĞ±ÕÍÑ•Èİ¥¹‘½Üˆ°(€€€€€€€€‰5¥¹ÕÑ•ÌÕÍ•‰äÑ¡”•á•ÕÑ¥½¸Á½±¥äÑ¼½Õ¹ĞÉ••¹ĞÍ…µ”µ½¹Ñ•áĞ•¹ÑÉ¥•Ì¸ˆ°(€€€€€€€€ˆÄÀµ•…¹ÌÑ¡”Ñ¡É½ÑÑ±”½¹±ä½¹Í¥‘•ÉÌ•¹ÑÉ¥•Ì½Á•¹•¥¸Ñ¡”±…ÍĞÑ•¸µ¥¹ÕÑ•Ì¸ˆ°(€€€€¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹½¹Ñ•áÑ}±ÕÍÑ•É}µ…á}•¹ÑÉ¥•Ìˆè€ (€€€€€€€€‰5…àÍ…µ”µ½¹Ñ•áĞ•¹ÑÉ¥•Ìˆ°(€€€€€€€€‰5…á¥µÕ´•¹ÑÉ¥•Ì…±±½İ•™½ÈÑ¡”Í…µ”±•…É¹•½¹Ñ•áĞ¥¹Í¥‘”Ñ¡”±ÕÍÑ•Èİ¥¹‘½Ü‰•™½É”¹•Ü•¹ÑÉ¥•Ì…É”‰±½­•¸ˆ°(€€€€€€€€ˆÌ±•ÑÌÑ¡”ÍåÍÑ•´Ñ•ÍĞ„Í•ÑÕÀ±ÕÍÑ•È‰ÕĞÁÉ•Ù•¹ÑÌ„™Õ±°µÕ¹¥Ù•ÉÍ”ÍÑ…µÁ•‘”¸ˆ°(€€€€¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹½¹Ñ•áÑ}±ÕÍÑ•É}ÕÍ•}Í•ÑÕÁ}Í•ÍÍ¥½¸ˆè€ (€€€€€€€€‰Q¡É½ÑÑ±”Í•ÑÕÀµÍ•ÍÍ¥½¸±ÕÍÑ•ÉÌˆ°(€€€€€€€€‰½Õ¹ÑÌÍ•ÑÕÀ€¬Í¥‘”€¬Í•ÍÍ¥½¸•¹ÑÉ¥•ÌÑ½•Ñ¡•È¥¹ÍÑ•…½˜½¹±ä•á…ĞIM$½Y]@½¹Ñ•áĞµ…Ñ¡•Ì°…Ñ¡¥¹œ½ÉÉ•±…Ñ•Ñ¥­•È±ÕÍÑ•ÉÌ•…É±¥•È¸ˆ°(€€€€€€€€‰¹…‰±•ÑÉ•…ÑÌEED°QEED°M=a0°9Y°…¹5TÁÉ”µµ…É­•ĞÍ¡½ÉÑÌ…ÌÑ¡”Í…µ”•á•ÕÑ¥½¸¥‘•„¸ˆ°(€€€€¤°(€€€€‰Í…±Á}ÉÕ¹Ñ¥µ”¹Í•ÑÕÁ}Í•ÍÍ¥½¹}±ÕÍÑ•É}µ…á}•¹ÑÉ¥•Ìˆè€ (€€€€€€€€‰5…àÍ•ÑÕÀµÍ•ÍÍ¥½¸•¹ÑÉ¥•Ìˆ°(€€€€€€€€‰5…á¥µÕ´•¹ÑÉ¥•Ì…±±½İ•™½ÈÑ¡”Í…µ”Í•ÑÕÀ€¬Í¥‘”€¬Í•ÍÍ¥½¸¥¹Í¥‘”Ñ¡”±ÕÍÑ•Èİ¥¹‘½Ü‰•™½É”¹•Ü•¹ÑÉ¥•Ì…É”‰±½­•¸ˆ°(€€€€€€€€ˆÈ±•ÑÌÑ¡”•¹¥¹”Ñ•ÍĞ½¹”½ÈÑİ¼ÁÉ”µµ…É­•ĞÍ¡½ÉÑÌ°Ñ¡•¸İ…¥ÑÌ™½È½ÕÑ½µ•Ì‰•™½É”…‘‘¥¹œµ½É”¸ˆ°(€€€€¤°)ô()}M1A}51}Q%1Lè‘¥ÑmÍÑÈ°ÑÕÁ±•mÍÑÈ°ÍÑÈ°ÍÑÉut€ôì(€€€€‰Í…±Á}µ°¹ÑÉ…¥¹¥¹}•¹…‰±•ˆè€ ‰¹…‰±”50ÑÉ…¥¹¥¹œˆ°€‰±±½İÌÑ¡”¥Í½±…Ñ•±•…É¹•ÈÍ•ÉÙ¥”Ñ¼ÑÉ…¥¸Q@Äµ‰•™½É”µÍÑ½À…¹Q@Èµ‰•™½É”µÍÑ½À¡…±±•¹•ÉÌ™É½´±½Í•M1A}A19}XÄ½ÕÑ½µ•Ì¸%Ğ¹•Ù•ÈÑÉ…¥¹Ì¥¸Í…¹¹•È½Èİ•ˆµ…Á¤¸ˆ°€‰-••À‘¥Í…‰±•Õ¹Ñ¥°•¹½Õ …¹½¹¥…°½ÕÑ½µ•Ì•á¥ÍĞì•¹…‰±¥¹œ‘½•Ì¹½Ğ…Ñ¥Ù…Ñ”•á•ÕÑ¥½¸¸ˆ¤°(€€€€‰Í…±Á}µ°¹…ÕÑ½}ÑÉ…¥¹}İ¡•¹}É•…‘äˆè€ ‰ÕÑ¼µÍÑ…ÉĞ50…ĞÍ…µÁ±”…Ñ”ˆ°€‰ÉµÌÑ¡”¥Í½±…Ñ•±•…É¹•È¹½Ü…¹…ÕÑ½µ…Ñ¥…±±äÍÑ…ÉÑÌ¡…±±•¹•ÈÑÉ…¥¹¥¹œ½¹”Ñ¡”µ¥¹¥µÕ´…¹½¹¥…°½ÕÑ½µ”½Õ¹Ğ¥ÌÉ•…¡•¸	•™½É”Ñ¡”…Ñ”°¥Ğ½‰Í•ÉÙ•Ìİ¥Ñ¡½ÕĞÉ•…Ñ¥¹œ•µÁÑä¡…±±•¹•È•Ù…±Õ…Ñ¥½¹Ì¸ˆ°€‰¹…‰±•İ¥Ñ „€ÈÀÀµ½ÕÑ½µ”™±½½Èµ•…¹ÌÑÉ…¥¹¥¹œÍÑ…ÉÑÌ…ÕÑ½µ…Ñ¥…±±ä…Ğ½ÕÑ½µ”€ÈÀÀ¸ˆ¤°(€€€€‰Í…±Á}µ°¹Í¡…‘½İ}•¹…‰±•ˆè€ ‰¹…‰±”Í¡…‘½ÜÁÉ•‘¥Ñ¥½¹Ìˆ°€‰1½…‘ÌÑ¡”ÁÉ½µ½Ñ•¡…µÁ¥½¸…¹É•½É‘ÌÁÉ½‰…‰¥±¥Ñ¥•Ì½¸Ù…±¥Í…±ÀÁ±…¹Ìİ¥Ñ¡½ÕĞ¡…¹¥¹œ½¹™¥‘•¹”½È•á•ÕÑ¥½¸¸ˆ°€‰UÍ”Í¡…‘½Üµ½‘”™¥ÉÍĞÑ¼½µÁ…É”ÁÉ•‘¥Ñ¥½¹Ìİ¥Ñ É•…±¥é•½ÕÑ½µ•Ì¸ˆ¤°(€€€€‰Í…±Á}µ°¹½Ù•É±…å}•¹…‰±•ˆè€ ‰ÁÁ±ä½¹™¥‘•¹”½Ù•É±…äˆ°€‰ÁÁ±¥•ÌÑ¡”ÁÉ½µ½Ñ•µ½‘•°Ì‰½Õ¹‘•½¹™¥‘•¹”…‘©ÕÍÑµ•¹Ğ¸%Ğ…¹¹½ĞÉ•…Ñ”„Í•ÑÕÀ°¡…¹”‰É…­•Ğ±•Ù•±Ì°…±Ñ•ÈÍ¥é”°½È‰åÁ…ÍÌ‰±½­•ÉÌ¸ˆ°€‰¹…‰±”½¹±ä…™Ñ•ÈÍ¡…‘½Ü…±¥‰É…Ñ¥½¸¥Ì•½¹½µ¥…±±äÙ…±¥‘…Ñ•¸ˆ¤°(€€€€‰Í…±Á}µ°¹ÑÉ…¥¹¥¹}¥¹Ñ•ÉÙ…±}µ¥¸ˆè€ ‰QÉ…¥¹¥¹œ¥¹Ñ•ÉÙ…°ˆ°€‰5¥¹ÕÑ•Ì‰•Ñİ••¸¡…±±•¹•ÈÑÉ…¥¹¥¹œ…ÑÑ•µÁÑÌ¥¹Í¥‘”Ñ¡”±•…É¹•È½¹Ñ…¥¹•È¸ˆ°€ˆØÀ•Ù…±Õ…Ñ•Ì„™É•Í ¡…±±•¹•È…Ğµ½ÍĞ½¹”Á•È¡½ÕÈ¸ˆ¤°(€€€€‰Í…±Á}µ°¹ÑÉ…¥¹¥¹}±½½­‰…­}‘…åÌˆè€ ‰QÉ…¥¹¥¹œ±½½­‰…¬ˆ°€‰5…á¥µÕ´…”½˜…¹½¹¥…°½ÕÑ½µ•Ì¥¹±Õ‘•¥¸„¡…±±•¹•È‘…Ñ…Í•ĞÍ¼½‰Í½±•Ñ”É•¥µ•Ì…¹¹½Ğ‘½µ¥¹…Ñ”ÕÉÉ•¹ĞÍ…±Á¥¹œ‰•¡…Ù¥½È¸ˆ°€ˆØÀÑÉ…¥¹Ì½¹±ä™É½´Ñ¡”µ½ÍĞÉ••¹ĞÍ¥áÑä…±•¹‘…È‘…åÌ¸ˆ¤°(€€€€‰Í…±Á}µ°¹µ…á¥µÕµ}µ½‘•±}…•}¡½ÕÉÌˆè€ ‰5…á¥µÕ´¡…µÁ¥½¸…”ˆ°€‰I•©•ÑÌ¥¹™•É•¹”™É½´„¡…µÁ¥½¸½±‘•ÈÑ¡…¸Ñ¡¥Ìµ…¹ä¡½ÕÉÌ¸5¥ÍÍ¥¹œ½ÈÍÑ…±”50…±İ…åÌ™…¥±Ì½Á•¸Ñ¼‘•Ñ•Éµ¥¹¥ÍÑ¥Œ½¹™¥‘•¹”¸ˆ°€ˆÄØà•áÁ¥É•Ì„¡…µÁ¥½¸…™Ñ•ÈÍ•Ù•¸‘…åÌİ¥Ñ¡½ÕĞÍÕ•ÍÍ™Õ°É•Ù…±¥‘…Ñ¥½¸¸ˆ¤°(€€€€‰Í…±Á}µ°¹µ¥¹¥µÕµ}Í…µÁ±•Ìˆè€ ‰5¥¹¥µÕ´ÑÉ…¥¹¥¹œ½ÕÑ½µ•Ìˆ°€‰5¥¹¥µÕ´±½Í•…¹½¹¥…°½ÕÑ½µ•ÌÉ•ÅÕ¥É•‰•™½É”„¡…±±•¹•Èµ…ä‰”™¥ÑÑ•¸ˆ°€ˆÈÀÀÁÉ•Ù•¹ÑÌÁÉ½µ½Ñ¥½¸™É½´„Ñ¥¹äÍ…µÁ±”¸ˆ¤°(€€€€‰Í…±Á}µ°¹‰½½ÑÍÑÉ…Á}µ¥¹¥µÕµ}Í…µÁ±•Ìˆè€ ‰	½½ÑÍÑÉ…À•Ù…±Õ…Ñ¥½¸½ÕÑ½µ•Ìˆ°€‰Mµ…±±•ÈÍ…µÁ±”™±½½ÈÑ¡…Ğ±•ÑÌÑ¡”±•…É¹•È™¥Ğ…¹É•©•Ğ•…É±ä¡…±±•¹•ÉÌ™½ÈÙ¥Í¥‰¥±¥Ñä‰•™½É”ÁÉ½µ½Ñ¥½¸¥Ì…±±½İ•¸ˆ°€ˆÜÔÍÑ…ÉÑÌÕÍ•™Õ°‘¥…¹½ÍÑ¥Ì•…É±¥•ÈìÁÉ½µ½Ñ¥½¸ÍÑ¥±°É•ÅÕ¥É•ÌÑ¡”µ…¥¸µ¥¹¥µÕ´ÑÉ…¥¹¥¹œ½ÕÑ½µ•Ì…Ñ”¸ˆ¤°(€€€€‰Í…±Á}µ°¹‰½½ÑÍÑÉ…Á}ÑÉ…¥¹¥¹}•¹…‰±•ˆè€ ‰¹…‰±”‰½½ÑÍÑÉ…À•Ù…±Õ…Ñ¥½¸ˆ°€‰±±½İÌ•…É±ä¡…±±•¹•È™¥ÑÑ¥¹œ™É½´Ñ¡”‰½½ÑÍÑÉ…ÀÍ…µÁ±”™±½½Èİ¡¥±”É•Ñ…¥¹¥¹œ™Õ±°ÁÉ½µ½Ñ¥½¸…Ñ•Ì¸ˆ°€‰¹…‰±•µ•…¹ÌÑ¡”±•…É¹•È…¸•áÁ±…¥¸İ¡ä„µ½‘•°¥Ì¹½ĞÉ•…‘ä¥¹ÍÑ•…½˜½¹±äÍ…å¥¹œİ…¥Ñ¥¹œ™½ÈÍ…µÁ±•Ì¸ˆ¤°(€€€€‰Í…±Á}µ°¹¡½±‘½ÕÑ}ÁĞˆè€ ‰¡É½¹½±½¥…°¡½±‘½ÕĞ™É…Ñ¥½¸ˆ°€‰9•İ•ÍĞ™É…Ñ¥½¸½˜½ÕÑ½µ•ÌÉ•Í•ÉÙ•ÍÑÉ¥Ñ±ä™½È½ÕĞµ½˜µÍ…µÁ±”ÁÉ½µ½Ñ¥½¸Ñ•ÍÑ¥¹œ¸9¼É…¹‘½´Í¡Õ™™±”½ÈÍ…±•È™¥ĞÑ½Õ¡•Ì¥Ğ¸ˆ°€ˆÀ¸ÈÔÉ•Í•ÉÙ•ÌÑ¡”¹•İ•ÍĞ€ÈÔ”™½ÈÙ…±¥‘…Ñ¥½¸¸ˆ¤°(€€€€‰Í…±Á}µ°¹µ¥¹¥µÕµ}Í•±•Ñ•‘}¡½±‘½ÕĞˆè€ ‰5¥¹¥µÕ´•Ù…±Õ…Ñ•¡½±‘½ÕĞÑÉ…‘•Ìˆ°€‰5¥¹¥µÕ´¡½±‘½ÕĞÉ½İÌİ¡½Í”µ½‘•°•áÁ•Ñ•µH±•…ÉÌÑ¡”Í•±•Ñ¥½¸Ñ¡É•Í¡½±‰•™½É”•½¹½µ¥Œµ•ÑÉ¥Ì…É”ÑÉÕÍÑ•¸ˆ°€ˆÌÀÉ•ÅÕ¥É•Ì…Ğ±•…ÍĞÑ¡¥ÉÑä¥¹‘•Á•¹‘•¹Ñ±ä•Ù…±Õ…Ñ•½ÁÁ½ÉÑÕ¹¥Ñ¥•Ì¸ˆ¤°(€€€€‰Í…±Á}µ°¹µ¥¹¥µÕµ}¡½±‘½ÕÑ}Í•ÍÍ¥½¹Ìˆè€ ‰5¥¹¥µÕ´Ù…±¥‘…Ñ¥½¸Í•ÍÍ¥½¹Ìˆ°€‰9Õµ‰•È½˜É••¹Ğµ…É­•Ğ‘…Ñ•ÌÑ¡…ĞµÕÍĞ¥¹‘•Á•¹‘•¹Ñ±äÍ…Ñ¥Í™äÍ•ÍÍ¥½¸ÍÑ…‰¥±¥Ñä¡•­Ì¸ˆ°€ˆÈÁÉ•Ù•¹ÑÌ½¹”Õ¹ÕÍÕ…±±äÍÑÉ½¹œ‘…ä™É½´ÁÉ½µ½Ñ¥¹œ„µ½‘•°¸ˆ¤°(€€€€‰Í…±Á}µ°¹µ¥¹¥µÕµ}Í•ÍÍ¥½¹}Í…µÁ±•Ìˆè€ ‰5¥¹¥µÕ´ÑÉ…‘•ÌÁ•ÈÍ•ÍÍ¥½¸ˆ°€‰5¥¹¥µÕ´Í•±•Ñ•¡½±‘½ÕĞÑÉ…‘•ÌÉ•ÅÕ¥É•½¸•… É••¹ĞÙ…±¥‘…Ñ¥½¸Í•ÍÍ¥½¸¸ˆ°€ˆÔÉ•ÅÕ¥É•Ìµ•…¹¥¹™Õ°½Ù•É…”½¸‰½Ñ É••¹Ğ‘…åÌ¸ˆ¤°(€€€€‰Í…±Á}µ°¹µ¥¹¥µÕµ}•áÁ•Ñ…¹å}Èˆè€ ‰AÉ½µ½Ñ¥½¸•áÁ•Ñ…¹ä™±½½Èˆ°€‰5¥¹¥µÕ´É•…±¥é•µ•…¸H½¸Í•±•Ñ•¡É½¹½±½¥…°¡½±‘½ÕĞÉ½İÌ¸ˆ°€ˆÀ¸ÀÔÉ•ÅÕ¥É•Ì…Ğ±•…ÍĞ€¬À¸ÀÕHÁ•ÈÍ•±•Ñ•ÑÉ…‘”¸ˆ¤°(€€€€‰Í…±Á}µ°¹µ¥¹¥µÕµ}ÁÉ½™¥Ñ}™…Ñ½Èˆè€ ‰AÉ½µ½Ñ¥½¸ÁÉ½™¥Ğµ™…Ñ½È™±½½Èˆ°€‰5¥¹¥µÕ´É½ÍÌ¡½±‘½ÕĞİ¥¹Ì‘¥Ù¥‘•‰ä…‰Í½±ÕÑ”É½ÍÌ¡½±‘½ÕĞ±½ÍÍ•Ì¸ˆ°€ˆÄ¸ÄÀÉ•ÅÕ¥É•ÌÑ•¸Á•É•¹Ğµ½É”É½ÍÌÁÉ½™¥ĞÑ¡…¸É½ÍÌ±½ÍÌ¸ˆ¤°(€€€€‰Í…±Á}µ°¹µ¥¹¥µÕµ}Í•ÍÍ¥½¹}•áÁ•Ñ…¹å}Èˆè€ ‰A•ÈµÍ•ÍÍ¥½¸•áÁ•Ñ…¹ä™±½½Èˆ°€‰5¥¹¥µÕ´É•…±¥é••áÁ•Ñ…¹äÉ•ÅÕ¥É•¥¹‘•Á•¹‘•¹Ñ±ä½¸•… É••¹ĞÙ…±¥‘…Ñ¥½¸Í•ÍÍ¥½¸¸ˆ°€ˆÀ¸ÀÁÉ•Ù•¹ÑÌÁÉ½µ½Ñ¥½¸İ¡•¸•¥Ñ¡•ÈÉ••¹ĞÍ•ÍÍ¥½¸¥Ì¹•…Ñ¥Ù”¸ˆ¤°(€€€€‰Í…±Á}µ°¹µ¥¹¥µÕµ}…ÕŒˆè€ ‰5¥¹¥µÕ´¡½±‘½ÕĞUˆ°€‰5¥¹¥µÕ´½ÕĞµ½˜µÍ…µÁ±”‘¥ÍÉ¥µ¥¹…Ñ¥½¸É•ÅÕ¥É•¥¹‘•Á•¹‘•¹Ñ±ä™½È‰½Ñ Q@Ä…¹Q@È±…ÍÍ¥™¥•ÉÌ¸ˆ°€ˆÀ¸ÔÈÉ•ÅÕ¥É•Ì•… µ½‘•°Ñ¼É…¹¬½ÕÑ½µ•Ì‰•ÑÑ•ÈÑ¡…¸¡…¹”¸ˆ¤°(€€€€‰Í…±Á}µ°¹µ¥¹¥µÕµ}‰É¥•É}¥µÁÉ½Ù•µ•¹Ğˆè€ ‰5¥¹¥µÕ´	É¥•È¥µÁÉ½Ù•µ•¹Ğˆ°€‰I•ÅÕ¥É•ÁÉ½‰…‰¥±¥Ñäµ…±¥‰É…Ñ¥½¸¥µÁÉ½Ù•µ•¹Ğ½Ù•È„½¹ÍÑ…¹ĞÑÉ…¥¹¥¹œµÁÉ•Ù…±•¹”‰…Í•±¥¹”½¸¡É½¹½±½¥…°¡½±‘½ÕĞ‘…Ñ„¸ˆ°€ˆÀ¸ÀÉ•©•ÑÌ„µ½‘•°İ¡½Í”ÁÉ½‰…‰¥±¥Ñ¥•Ì…É”İ½ÉÍ”Ñ¡…¸Ñ¡”¹…¥Ù”‰…Í•±¥¹”¸ˆ¤°(€€€€‰Í…±Á}µ°¹Í•±•Ñ¥½¹}•áÁ•Ñ•‘}Èˆè€ ‰AÉ•‘¥Ñ¥½¸Í•±•Ñ¥½¸Ñ¡É•Í¡½±ˆ°€‰5¥¹¥µÕ´µ½‘•°µ¥µÁ±¥••áÁ•Ñ•HÕÍ•Ñ¼¥¹±Õ‘”„¡½±‘½ÕĞÉ½Ü¥¸•½¹½µ¥ŒÁÉ½µ½Ñ¥½¸•Ù…±Õ…Ñ¥½¸¸ˆ°€ˆÀ¸À•Ù…±Õ…Ñ•Ì½¹±äµ½‘•°µÁ½Í¥Ñ¥Ù”½ÁÁ½ÉÑÕ¹¥Ñ¥•Ì¸ˆ¤°(€€€€‰Í…±Á}µ°¹½¹™¥‘•¹•}Á½¥¹ÑÍ}Á•É}Èˆè€ ‰½¹™¥‘•¹”Í•¹Í¥Ñ¥Ù¥Ñäˆ°€‰A•É•¹Ñ…”µÁ½¥¹Ğ…‘©ÕÍÑµ•¹ĞÁÉ½‘Õ•Á•È½¹”Õ¹¥Ğ½˜µ½‘•°µ¥µÁ±¥••áÁ•Ñ•H‰•™½É”…ÁÌ¸ˆ°€ˆÔ…‘‘Ì€È¸ÔÁ½¥¹ÑÌ™½È€¬À¸ÔÀ•áÁ•Ñ•H¸ˆ¤°(€€€€‰Í…±Á}µ°¹µ…á}½¹™¥‘•¹•}É…¥Í”ˆè€ ‰5…á¥µÕ´½¹™¥‘•¹”¥¹É•…Í”ˆ°€‰!…É…À½¸Á½Í¥Ñ¥Ù”µ½‘•°¥¹™±Õ•¹”¸Q¡”Á±…¸µÕÍĞ…±É•…‘ä‰”Ù…±¥‰•™½É”Ñ¡¥Ì…¸µ…ÑÑ•È¸ˆ°€ˆÔÁÉ•Ù•¹ÑÌ50™É½´…‘‘¥¹œµ½É”Ñ¡…¸™¥Ù”½¹™¥‘•¹”Á½¥¹ÑÌ¸ˆ¤°(€€€€‰Í…±Á}µ°¹µ…á}½¹™¥‘•¹•}É•‘ÕÑ¥½¸ˆè€ ‰5…á¥µÕ´½¹™¥‘•¹”É•‘ÕÑ¥½¸ˆ°€‰!…É…À½¸¹•…Ñ¥Ù”µ½‘•°¥¹™±Õ•¹”¸9•…Ñ¥Ù”•Ù¥‘•¹”µ…äÑ¥¡Ñ•¸µ½É”ÍÑÉ½¹±äÑ¡…¸Á½Í¥Ñ¥Ù”•Ù¥‘•¹”…¸É•±…à¸ˆ°€ˆÄÔÁ•Éµ¥ÑÌÕÀÑ¼„™¥™Ñ••¸µÁ½¥¹ĞÉ•‘ÕÑ¥½¸¸ˆ¤°)ô(()‘•˜‰Õ¥±‘}…Ñ…±½œ¡Ù…±Õ•Ìè‘¥ÑmÍÑÈ°¹åt°‘•™…Õ±ÑÌè‘¥ÑmÍÑÈ°¹åt¤€´ø‘¥ÑmÍÑÈ°¹åtè(€€€É½ÕÁÌ€ôl(€€€€€€€ì‰¥ˆèÁÉ•™¥à°€‰±…‰•°ˆè±…‰•°°€‰‘•ÍÉ¥ÁÑ¥½¸ˆè‘•ÍÉ¥ÁÑ¥½¹ô(€€€€€€€™½ÈÁÉ•™¥à°±…‰•°°‘•ÍÉ¥ÁÑ¥½¸¥¸I=UAL(€€€t(€€€™¥•±‘Ì€ômt(€€€™½È­•ä¥¸Í½ÉÑ•¡Í•Ğ¡‘•™…Õ±ÑÌ¤ğÍ•Ğ¡Ù…±Õ•Ì¤¤è(€€€€€€€Ù…±Õ”€ôÙ…±Õ•Ì¹•Ğ¡­•ä°‘•™…Õ±ÑÌ¹•Ğ¡­•ä¤¤(€€€€€€€‘•™…Õ±Ğ€ô‘•™…Õ±ÑÌ¹•Ğ¡­•ä¤(€€€€€€€ÁÉ•™¥à€ô­•ä¹ÍÁ±¥Ğ ˆ¸ˆ°€Ä¥lÁt(€€€€€€€É½ÕÀ€ôÁÉ•™¥à¥˜ÁÉ•™¥à¥¸}I=UA}	e}AI%`•±Í”€‰ÍåÍÑ•´ˆ(€€€€€€€±…‰•°°‘•ÍÉ¥ÁÑ¥½¸°•á…µÁ±”€ô}‘•Ñ…¥°¡­•ä°Ù…±Õ”°É½ÕÀ¤(€€€€€€€™¥•±‘Ì¹…ÁÁ•¹ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰­•äˆè­•ä°(€€€€€€€€€€€€€€€€‰É½ÕÀˆèÉ½ÕÀ°(€€€€€€€€€€€€€€€€‰±…‰•°ˆè±…‰•°°(€€€€€€€€€€€€€€€€‰‘•ÍÉ¥ÁÑ¥½¸ˆè‘•ÍÉ¥ÁÑ¥½¸°(€€€€€€€€€€€€€€€€‰•á…µÁ±”ˆè•á…µÁ±”°(€€€€€€€€€€€€€€€€‰Ù…±Õ”ˆèÙ…±Õ”°(€€€€€€€€€€€€€€€€‰‘•™…Õ±Ğˆè‘•™…Õ±Ğ°(€€€€€€€€€€€€€€€€‰ÑåÁ”ˆè}Ù…±Õ•}ÑåÁ”¡Ù…±Õ”¤°(€€€€€€€€€€€€€€€€‰…‘Ù…¹•ˆè­•ä¹½Ğ¥¸}M1A}Q%1L…¹­•ä¹½Ğ¥¸}M1A}IU9Q%5}Q%1L…¹­•ä¹½Ğ¥¸}M1A}1I9}Q%1L…¹­•ä¹½Ğ¥¸}M1A}51}Q%1L°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€É•ÑÕÉ¸ì‰Í¡•µ…}Ù•ÉÍ¥½¸ˆè€Ä°€‰É½ÕÁÌˆèÉ½ÕÁÌ°€‰™¥•±‘Ìˆè™¥•±‘Íô(()‘•˜}‘•Ñ…¥°¡­•äèÍÑÈ°Ù…±Õ”è¹ä°É½ÕÀèÍÑÈ¤€´øÑÕÁ±•mÍÑÈ°ÍÑÈ°ÍÑÉtè(€€€¥˜­•ä¥¸}M1A}Q%1Lè(€€€€€€€É•ÑÕÉ¸}M1A}Q%1Mm­•åt(€€€¥˜­•ä¥¸}M1A}IU9Q%5}Q%1Lè(€€€€€€€É•ÑÕÉ¸}M1A}IU9Q%5}Q%1Mm­•åt(€€€¥˜­•ä¥¸}M1A}1I9}Q%1Lè(€€€€€€€É•ÑÕÉ¸}M1A}1I9}Q%1Mm­•åt(€€€¥˜­•ä¥¸}M1A}51}Q%1Lè(€€€€€€€É•ÑÕÉ¸}M1A}51}Q%1Mm­•åt(€€€ÍÕ™™¥à€ô­•ä¹ÍÁ±¥Ğ ˆ¸ˆ°€Ä¥l´Åt(€€€±…‰•°€ôÉ”¹ÍÕˆ¡È‰qÌ¬ˆ°€ˆ€ˆ°ÍÕ™™¥à¹É•Á±…” ‰|ˆ°€ˆ€ˆ¤¤¹ÍÑÉ¥À ¤¹Ñ¥Ñ±” ¤(€€€É½ÕÁ}±…‰•°€ô}I=UA}	e}AI%`¹•Ğ¡É½ÕÀ°€ ‰MåÍÑ•´ˆ°€ˆˆ¤¥lÁt(€€€‘•ÍÉ¥ÁÑ¥½¸€ô€ (€€€€€€€˜‰IÕ¹Ñ¥µ”íÉ½ÕÁ}±…‰•°¹±½İ•È ¥ôÍ•ÑÑ¥¹œí­•åõ€¸%Ğ¥ÌÁ•ÉÍ¥ÍÑ•¥¸A½ÍÑÉ•ME0€ˆ(€€€€€€€€‰…¹¡½ĞµÉ•±½…‘•‰äÑ¡”½İ¹¥¹œÍ•ÉÙ¥”İ¥Ñ¡½ÕĞ„½¹Ñ…¥¹•ÈÉ•ÍÑ…ÉĞ¸ˆ(€€€€¤(€€€¥˜¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°‰½½°¤è(€€€€€€€•á…µÁ±”€ô€‰¹…‰±•…ÁÁ±¥•ÌÑ¡”‰•¡…Ù¥½È¥µµ•‘¥…Ñ•±äì‘¥Í…‰±•±•…Ù•ÌÑ¡”½İ¹¥¹œ™•…ÑÕÉ”¥¹…Ñ¥Ù”¸ˆ(€€€•±¥˜¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°€¡¥¹Ğ°™±½…Ğ¤¤è(€€€€€€€•á…µÁ±”€ô˜‰ÕÉÉ•¹Ğ½‘•™…Õ±ĞÉ•™•É•¹”Ù…±Õ”èíÙ…±Õ•ô¸¡…¹”É…‘Õ…±±ä…¹Ù•É¥™äÑ¡”½İ¹¥¹œÍ•ÉÙ¥”µ•ÑÉ¥Ì¸ˆ(€€€•±¥˜¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°±¥ÍĞ¤è(€€€€€€€•á…µÁ±”€ô€‰¹Ñ•È„)M=8±¥ÍĞ¸… ¥Ñ•´¥ÌÁÉ•Í•ÉÙ•¥¸½É‘•È¸ˆ(€€€•±Í”è(€€€€€€€•á…µÁ±”€ô˜‰ÕÉÉ•¹Ğ½‘•™…Õ±ĞÉ•™•É•¹”Ù…±Õ”èíÙ…±Õ”…Íô¸ˆ(€€€É•ÑÕÉ¸±…‰•°°‘•ÍÉ¥ÁÑ¥½¸°•á…µÁ±”(()‘•˜}Ù…±Õ•}ÑåÁ”¡Ù…±Õ”è¹ä¤€´øÍÑÈè(€€€¥˜¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°‰½½°¤è(€€€€€€€É•ÑÕÉ¸€‰‰½½±•…¸ˆ(€€€¥˜¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°¥¹Ğ¤…¹¹½Ğ¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°‰½½°¤è(€€€€€€€É•ÑÕÉ¸€‰¥¹Ñ••Èˆ(€€€¥˜¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°™±½…Ğ¤è(€€€€€€€É•ÑÕÉ¸€‰¹Õµ‰•Èˆ(€€€¥˜¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°±¥ÍĞ¤è(€€€€€€€É•ÑÕÉ¸€‰…ÉÉ…äˆ(€€€É•ÑÕÉ¸€‰ÍÑÉ¥¹œˆ