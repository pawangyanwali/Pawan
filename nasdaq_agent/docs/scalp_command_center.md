# Scalp Command Center

## Purpose

The new UI is an operational scalping workspace, not a reformatted version of
the legacy scanner table. It answers four questions in order:

1. Is the data live enough to trade?
2. Which setups are actionable now?
3. What is the exact bracket and why is it valid or blocked?
4. What exposure and learning action is currently active?

The existing dashboard remains available during shadow validation. The command
center becomes the default only after browser-to-API-to-database parity tests
pass for every displayed value.

## Desktop Layout

```text
+--------------------------------------------------------------------------+
| NASDAQ SCALP | Session | WS 451 | REST 26 | STALE 0 | age 180ms | Risk OK |
+----------------------+--------------------------------+------------------+
| Opportunities        | Selected Signal Plan           | Account / Risk   |
|                      |                                |                  |
| LONG  4  SHORT  2    | NVDA  LONG  VWAP RECLAIM      | Open risk 0.6%   |
| BLOCKED 31  GAP 0    | Entry  Stop  TP1  TP2  R:R    | Daily P&L        |
|                      | RSI  MACD  ATR  VWAP  RVOL     | Daily loss room  |
| ticker/setup cards   | Reasons and blockers           | Active positions |
| sorted by readiness  | Compact live chart             | Learning actions |
+----------------------+--------------------------------+------------------+
| Activity: plan -> gate -> order -> fills -> outcome -> learning action      |
+--------------------------------------------------------------------------+
```

No page section is rendered as a decorative card. Framed surfaces are reserved
for repeated opportunity items, a selected signal plan, and active positions.

## Navigation

Primary views:

- `Trade`: opportunities, selected plan, positions, and risk.
- `Market`: all tickers grouped by actionable, watch, blocked, and data gap.
- `Learning`: context expectancy, throttles, pauses, and recent actions.
- `Review`: outcomes, execution quality, and attribution.
- `Settings`: runtime configuration with ownership and examples.

Operational status remains visible in every view. It is never hidden in a
tooltip or modal.

## Market Data Strip

Always-visible fields:

- Session and market clock.
- WebSocket ticker count.
- REST fallback ticker count.
- Stale ticker count.
- Oldest quote age.
- Last completed 1-minute bar age.
- Schwab token owner health.
- Engine and execution health.

Status language:

- `LIVE`: required universe coverage and freshness meet SLA.
- `HYBRID`: REST is supplementing WebSocket; counts are explicit.
- `DEGRADED`: coverage or latency is below the trading SLA.
- `NOT TRADABLE`: execution is blocked; monitoring may continue.

## Opportunities Panel

The default list contains only:

- Valid LONG plans.
- Valid SHORT plans.
- Watch-only candidates one confirmation away.

Each item shows ticker, side, setup, price, confidence, R:R, data source, quote
age, and one-line state. Sorting defaults to validity, learned expectancy,
confidence, then freshness.

Filters use segmented controls:

- `Actionable`, `Watch`, `Blocked`, `Data gaps`.
- `LONG`, `SHORT`, `Both`.
- Session and setup family.

The complete 477-ticker universe is available in `Market`; it is not forced
into the primary trading view.

## Selected Signal Plan

The selected plan is the canonical `ScalpSignalPlan`. The UI must not recompute
or infer any value in JavaScript.

Required display:

- Entry, stop, TP1, TP2, risk per share, and configured reward multiple.
- TP2 path quality and nearest blocking structure.
- RSI-14/7/2 with zone.
- Current and prior MACD histogram plus slope.
- ATR-14, VWAP event, RVOL, spread in bps, and spread/risk.
- Data source and age.
- Deterministic setup reasons.
- All blockers, in execution order.
- Learned expectancy and sample count.

Blocked plans use plain language followed by the machine reason code. Example:

```text
Blocked: quote is 3.2 seconds old; maximum is 2.0 seconds.
QUOTE_STALE
```

## Risk Panel

Required fields:

- Buying power and paper/live mode.
- Current open risk in dollars and account percent.
- Daily realized and unrealized P&L.
- Daily loss warning/halt/liquidation thresholds.
- Concurrent positions and remaining capacity.
- Sector and directional concentration.
- Active risk brake with reason and expiry.

The UI may not show `Risk OK` when any required value is missing.

## Responsive Design

At narrow widths:

1. Health strip remains first.
2. Opportunity list becomes the primary view.
3. Selecting a ticker opens a full-height plan sheet.
4. Risk is a dedicated tab, not squeezed beside the plan.
5. Tables become stacked labeled values; horizontal scrolling is avoided.

## Data Contract

The browser receives versioned payloads:

```json
{
  "schema_version": 1,
  "asof_ts": "2026-06-26T14:30:00Z",
  "market_data_health": {},
  "plans": [],
  "positions": [],
  "risk": {},
  "learning_actions": []
}
```

Every payload has server time, source time, and freshness. WebSocket messages
carry deltas; a REST snapshot restores state after reconnect. The browser must
discard older sequence numbers and display reconnect/degraded state explicitly.

## UI Delivery

### UI Release 1: Design shell

- Add a dedicated command-center route and reusable design tokens.
- Render market health and empty/loading/error states from versioned fixtures.
- Keep it behind a feature flag.

### UI Release 2: Shadow plans

- Stream real `ScalpSignalPlan` records.
- Add opportunities and why-no-trade views.
- Compare displayed values against API and PostgreSQL facts.

### UI Release 3: Execution and learning

- Add order lifecycle, fills, positions, risk, and learning actions.
- Run desktop and mobile browser regression tests.

### UI Release 4: Default cutover

- Make command center the default.
- Keep legacy dashboard read-only for one release.
- Remove legacy markup, scripts, endpoints, and tests after parity sign-off.

## Acceptance Criteria

- One-second updates do not reflow the layout.
- No value is derived differently in the browser and backend.
- Every blocked candidate has a visible reason.
- Every plan level matches persisted facts exactly.
- A stale or disconnected state is impossible to present as live.
- Desktop and mobile screenshots have no overlap, clipping, or hidden controls.

