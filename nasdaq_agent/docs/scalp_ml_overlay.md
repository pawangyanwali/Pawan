# Scalp ML Overlay - Release 4

## Purpose

Release 4 answers two narrow questions for a plan that is already valid:

1. What is the probability that TP1 is reached before the initial stop?
2. What is the probability that TP2 is reached before the initial stop?

It does not create direction, signal validity, brackets, size, or risk policy.

## Ownership

| Responsibility | Owner |
|---|---|
| Build deterministic plan | scanner / `agent.scalp.engine` |
| Train challenger models | learner / `agent.scalp.ml_trainer` |
| Persist champion metadata | PostgreSQL |
| Persist versioned model artifacts | shared `models` Docker volume |
| Load champion and infer | scanner / `agent.scalp.ml_overlay` |
| Apply risk gates after inference | `agent.scalp.learning` and paper broker |
| Display probabilities and promotion evidence | `/scalp` command center |

The learner is the only writer of model artifacts. Scanner processes are read-only consumers.

## Safety Defaults

```text
scalp_ml.training_enabled = false
scalp_ml.shadow_enabled   = false
scalp_ml.overlay_enabled  = false
scalp.execution_enabled   = false
```

Deployment therefore installs the capability without training a model, changing confidence, or opening a canonical scalp trade.

## Labels

Only closed `SCALP_PLAN_V1` trades are eligible.

```text
TP1 label = scalp_trade_outcomes.tp1_hit
TP2 label = scalp_trade_outcomes.tp2_hit
```

The training row joins the outcome back to the exact persisted `plan_json`. No current quote, revised indicator, or post-trade field is consulted.

## Feature Contract

Feature schema version 1 contains only pre-entry facts:

- LONG/SHORT side.
- PRE_MARKET/REGULAR/AFTER_HOURS session.
- RSI-14, RSI-7, RSI-2, and RSI zone.
- MACD histogram and slope normalized by ATR.
- ATR as percent of price.
- price-to-VWAP distance normalized by ATR.
- RVOL.
- spread in basis points and spread-to-risk.
- VWAP event.
- TP2 path quality.
- WS/REST source.
- deterministic setup score.
- quote age and bar age.
- canonical reward multiple.

Explicitly excluded:

- TP1/TP2/stop outcomes.
- P&L, MFE, MAE, or exit reason.
- any price observed after the plan timestamp.
- ticker identity.
- legacy ML scores.
- learned context statistics that were updated after the trade.

Tree models are used without a scaler, eliminating scaler leakage.

## Chronological Training

1. Load outcomes from the configurable recent lookback window.
2. Sort by close timestamp and trade id.
3. Reserve the newest `holdout_pct` as an untouched holdout.
4. Fit separate TP1 and TP2 XGBoost classifiers on the older rows only.
5. Evaluate probabilities and economics on the newer rows only.
6. Reject any split where either label has only one class in train or holdout.

Rows are never shuffled.

## Expected-R Formula

TP2 probability is capped at TP1 probability because TP2 cannot occur before TP1.

For a 50/50 TP1/TP2 exit:

```text
stop before TP1       = -1.0R
TP1 but not TP2       = +0.5R
TP2 reached           = +0.5R + 0.5 * reward_r

expected_r = -1 + 1.5 * P(TP1) + 0.5 * reward_r * P(TP2)
```

This estimate is used for model selection and confidence adjustment. Promotion economics always use realized holdout `pnl_r`, including simulated spread and slippage.

## Promotion Gates

A challenger is promoted only when all conditions pass:

- minimum total canonical outcomes.
- TP1 and TP2 holdout AUC exceed the configured floor.
- TP1 and TP2 Brier scores beat constant training-prevalence baselines.
- minimum number of selected holdout opportunities.
- selected holdout realized expectancy is positive and above the configured floor.
- selected holdout realized profit factor exceeds the configured floor.
- at least two recent holdout sessions meet independent sample and expectancy floors.

Accuracy alone cannot promote a model.

Every challenger, including rejected challengers, is written to `scalp_ml_models` with metrics and rejection reason.

## Artifact Integrity

- Artifacts are written to a temporary file and atomically renamed.
- SHA-256 is persisted with the champion metadata.
- Scanner verifies checksum, feature schema, feature order, and version before loading.
- A champion older than `maximum_model_age_hours` is rejected.
- Missing, corrupt, incompatible, or stale artifacts fail open to deterministic confidence.
- Artifacts are versioned; promotion never overwrites the previous model file.

## Inference Boundary

Inference runs only when shadow or overlay mode is explicitly enabled and the deterministic plan is already valid.

Allowed mutation:

```text
final_confidence = clamp(base_confidence + bounded_ml_adjustment, 0, 100)
```

Forbidden mutations:

- `valid`.
- `side`.
- `entry`.
- `stop_loss`.
- `tp1`.
- `tp2`.
- `risk_per_share`.
- position size.
- learning or broker blockers.

Positive confidence influence is capped more tightly than negative influence. Online context gates run after ML and retain final authority to tighten or block.

## Rollout

1. Accumulate canonical paper outcomes with all ML controls disabled.
2. Enable training only; inspect rejected/promoted evidence.
3. Enable shadow predictions; compare calibration and realized outcomes.
4. Enable confidence overlay only after operator review.
5. Keep canonical paper execution disabled until the broader platform cutover is approved.

## Database Tables

### `scalp_ml_models`

Stores challenger/champion state, chronological split boundaries, metrics, rejection reasons, artifact location, and checksum.

### `scalp_ml_predictions`

Stores plan id, champion version, TP1/TP2 probability, expected R, confidence adjustment, and whether the adjustment was applied or shadow-only.
