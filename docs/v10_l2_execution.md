# V10 L2 execution research foundation

`src.backtest.v10_l2` is an offline library with no data loading, prediction,
strategy selection, exchange connection, or order placement. It does not change
the existing next-bar-open backtester or any frozen research protocol. Adoption
by an experiment requires a separate preregistration of the execution semantics.
Consumed V9 evidence is not fresh validation data.

## Caller contract

Supply one isolated session of strictly availability-ordered `Quote` records and
strictly decision-ordered `Decision` records to `replay`. All timestamps must be
aware, all prices, quantities, cash, and cost rates finite `Decimal` values.
The caller is responsible for dataset eligibility, session boundaries and frozen
input provenance; this library does not certify eligibility from a session label.
Each replay starts flat and must end flat. There is no automatic terminal exit.

`ExecutionRules` requires fee bps per side, additional adverse slippage bps per
side, fixed latency and maximum quote age; there are no chosen cost defaults.
The base scenario is 10/2 bps per side and stress 20/4 bps per side. Those imply
24/48 bps nominal round trips **before separately charged spread**, with actual
fees calculated on each fill notional rather than a flat return subtraction.

Execution occurs exactly at decision time plus latency. The latest quote with
availability <= execution is used, including a quote available at that instant.
This is explicit causal carry-forward, bounded by execution minus source time;
source timestamps are preserved in every fill. No future quote is substituted.
Missing, crossed, stale, or nonpositive bid/ask fails the replay. Source times
cannot regress. Decisions cannot precede signal availability. Each subsequent
decision must occur strictly after the preceding fill; pending orders cannot
overlap. Latency applies equally to entry and exit.

## Position and accounting

Only one fully funded LONG position is allowed. ENTER supplies base quantity;
EXIT liquidates the entire quantity. Both fees are paid in quote currency.
Entry buys ask multiplied by (1 + slippage rate); exit sells bid multiplied by
(1 - slippage rate). Cash pays actual notional and fee. No borrowing or shorting.
The model assumes the caller's size fills at top of book plus configured
slippage; depth capacity, queueing, partial fills, tick/lot rounding and exchange
fee-asset discounts are not modeled. It is not a production execution simulator.

For quantity q, entry/exit midpoint me/mx, ask ae, bid bx, and fills pe/px:

- Mid-price move (quote-currency PnL): q * (mx - me).
- Executable gross PnL before extra slippage: q * (bx - ae).
- Spread cost: mid-price move minus executable gross PnL.
- Filled gross PnL before fees: q * (px - pe).
- Slippage cost: executable gross PnL minus filled gross PnL.
- Fees: entry notional * fee rate + exit notional * fee rate.
- Net PnL: mid-price move - spread - slippage - fees.

All reported returns use the same denominator, actual entry fill notional, so
attribution is additive. They are simple fractional returns, not log returns or
returns on initial cash. `gross_return` includes spread and slippage, before fees;
`executable_gross_return` includes spread only. `mid_return` is the mid-price PnL
normalized by entry fill notional. Holding time is exit fill minus entry fill.
Fill records retain decisions, source/availability/execution timestamps, quote,
quantity, actual price, notional and per-side fee.

Aggregate expectancy is the arithmetic mean of trade returns; profit factor is
positive net PnL / absolute negative net PnL. With no losses it is undefined
(`null`), with numerator/denominator reported separately. Win rate counts strictly
positive net PnL. Turnover is both-side actual notional, also divided by initial
cash. Costs and average cost are in quote currency. Empty-sample rates are null.
Arithmetic uses an isolated 50-digit Decimal context. JSON uses exact decimal
strings and UTC timestamps; hashes bind rules, initial cash, quotes, decisions,
session identity and results. Replaying identical inputs produces identical JSON.

## Event compression

`EventRules` requires trigger threshold, rearm threshold (<= trigger), consecutive
one-second sample count and cooldown. These must come from an external protocol.
`EventCompressor` begins unarmed. An observed value <= rearm arms it; a strict
value > trigger starts persistence. N consecutive qualifying observations emit
one `SignalEvent` at the Nth observation, retaining the crossing time. No event is
backdated to the first qualifying second. Nonqualifying observations reset the
count; gaps also disarm and require a fresh observed rearm. Time reversal fails.

An emitted event reserves the compressor until the caller calls `release` at
the actual exit (or explicit cancellation) timestamp. Busy observations never
queue events. Cooldown begins at release. It must expire before a new observed
rearm and crossing; rearm values observed during cooldown do not count. Equality
with cooldown expiry is allowed. Observation timestamps must follow release.
Create a fresh compressor per session. The compressor schedules neither exits
nor holds and has no knowledge of prices or profitability. Use emitted decision
and signal-availability timestamps to construct caller-authorized decisions;
the execution replay independently rejects any overlapping trade schedule.

Only synthetic tests exercise this foundation. No V9 consumed dataset or blind
holdout is loaded, and no new hypothesis or strategy is evaluated.
