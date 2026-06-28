# Scalp Multi-Timeframe Strategy Contract

## Scope

This is a scalp-only design. Expected holding time is seconds to minutes.

- Live Level 1 quotes own executable price and spread.
- Fully closed one-minute bars own entry timing, ATR bracket geometry, RSI,
  MACD turn, VWAP event, and RVOL.
- Fully closed five-minute bars provide context only.
- The current incomplete five-minute bucket is never used.
- Daily, hourly, and swing-style trend logic are outside this contract.

## Setup families

### Reversal

A reversal candidate requires:

- one-minute RSI in the configured oversold LONG or overbought SHORT zone;
- one-minute MACD histogram turning in the intended direction;
- one-minute VWAP reclaim/hold for LONG or rejection/hold for SHORT; and
- no strong opposing five-minute context in the shadow strategy.

Oversold alone is not a LONG signal and overbought alone is not a SHORT
signal. The turn and price-location evidence are mandatory.

### Momentum pullback

A momentum-pullback candidate requires:

- aligned five-minute EMA/VWAP/MACD-slope context;
- one-minute RSI inside the configured pullback band, not at an exhausted
  extreme;
- one-minute MACD reacceleration in the context direction;
- one-minute VWAP direction confirmation; and
- session-appropriate one-minute RVOL.

This family detects continuation moves that the original reversal-only engine
cannot represent.

## Risk and execution ownership

Multi-timeframe context cannot rewrite entry, stop, TP1, TP2, size, or source
freshness. The existing deterministic bracket remains:

- stop distance from one-minute ATR, minimum price distance, and spread;
- TP1 and TP2 from configured R multiples;
- spread-to-risk and structural-path checks before execution; and
- live price/position marks refreshed independently of plan calculation.

## Rollout

Phase 1 is `SHADOW` only. Shadow fields are displayed and included as
pre-entry ML features, but they cannot alter canonical trade validity.

Activation requires chronological out-of-sample comparison, separately for
family, side, session, liquidity bucket, and market regime. Evaluation must
include simulated spread, slippage, stop-gap penalties, and rejected/expired
opportunities. Minimum evidence:

1. At least 200 resolved shadow opportunities per family and 50 per promoted
   family/session cell.
2. Positive expectancy after modeled execution costs.
3. Profit factor above 1.10 and no negative recent-session stability cell.
4. Stable maximum adverse excursion and stop-hit distribution.
5. No material increase in quote/bar staleness or full-universe cycle time.

No threshold may be promoted from same-session P&L alone. Changes progress
through shadow, paper canary, and only then a separately approved live mode.

## Research basis

- Gao et al., *Market Intraday Momentum*, Journal of Financial Economics:
  <https://www.sciencedirect.com/science/article/pii/S0304405X18301351>
- Chordia et al., short-run microstructure and longer-horizon momentum:
  <https://conference.nber.org/confer/2008/mms08/korajczyk.pdf>
- Short-term reversals after extreme one-minute Nasdaq-100 moves:
  <https://www.sciencedirect.com/science/article/pii/S1062976921000922>
- Intraday reversal profitability after bid-ask costs:
  <https://www.sciencedirect.com/science/article/pii/S0378426604000949>
- Sullivan, Timmermann, and White, data-snooping controls for technical rules:
  <https://eprints.lse.ac.uk/119144/1/dp303.pdf>
- Bailey et al., probability of backtest overfitting:
  <https://escholarship.org/uc/item/4w1110bb>
