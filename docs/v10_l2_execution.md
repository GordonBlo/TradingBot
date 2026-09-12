# V10 depth-aware L2 execution research foundation

`src.backtest.v10_l2` is an offline library. It has no prediction, strategy
selection, dataset loading, exchange connection or order placement. It changes no
frozen experiment. A future experiment must preregister its inputs and every
caller-supplied rule. Consumed V9 evidence is not fresh validation evidence.

## Causal depth contract

Each immutable `DepthSnapshot` has a symbol, source timestamp, availability
timestamp, nonnegative sequence ID, and nonempty bid and ask levels. Its symbol
must match the frozen exchange constraints. Bids must descend and
asks ascend strictly by price. Prices and available base quantities are positive,
finite `Decimal` values. Duplicate levels, conflicting duplicate quantities,
crossed or locked books, regressing source times, regressing sequences, and
non-increasing availability times fail explicitly.

Replay executes each decision at `decision.at + latency`. It selects the latest
snapshot whose availability is at or before that exact instant. Snapshot age is
measured from its preserved source timestamp and bounded by `max_quote_age`; no
future snapshot is substituted. A signal must be available by its decision time,
and the next decision must occur after the previous execution. Each replay starts
flat and must finish flat.

## Depth execution and position state

ENTER is a LONG market buy and walks asks from best price upward. EXIT is a market
sell and walks bids from best price downward. The fill records requested and
filled base quantity, fill ratio, levels consumed, top quote, unadjusted book
VWAP, final VWAP after configured adverse slippage, worst book and adjusted fill
prices, quote notional, fee, and cost attribution.

The caller must choose `FULL_FILL_OR_REJECT` or `ALLOW_PARTIAL_FILL`. Full mode
rejects insufficient causal depth. Partial mode consumes only recorded liquidity.
A partial entry opens only its actual filled quantity. A partial exit preserves
the exact residual and requires a later explicit EXIT. Each partial fill and the
remaining quantity must satisfy the frozen quantity filters; a replay with a
residual or otherwise open position fails. Liquidity is not invented.

One fully funded LONG position is allowed. ENTER supplies base quantity; EXIT has
no quantity and acts on the exact remaining position. Fees are paid in quote
currency. Cash pays actual fill notional plus entry fee and receives exit notional
less exit fee. Borrowing, shorting, overlapping positions, pending overlapping
decisions and implicit terminal liquidation are prohibited.

## Binance Spot constraints

`ExchangeConstraints` holds caller-frozen `LOT_SIZE`, optional
`MARKET_LOT_SIZE`, and applicable `MIN_NOTIONAL`/`NOTIONAL` rules. Both quantity
filters apply to a market order when `MARKET_LOT_SIZE` is present. Zero-valued
exchangeInfo fields are represented as disabled constraints. Quantity minimum,
maximum and step checks use exact Decimal remainder arithmetic. Active market
notional minimum and maximum checks use the actual simulated fill notional.

`exchange_constraints_from_exchange_info` translates an already acquired public
Binance Spot exchangeInfo payload. `load_public_exchange_constraints` accepts the
repository's unauthenticated `PublicMarketDataClient` interface and invokes only
`get_exchange_info`. Recognized filters must be unique and structurally complete.
Numeric filter values must be exact decimal strings, integers or Decimals; floats
are refused. `avgPriceMins` and market applicability flags remain hash-bound.

`PRICE_FILTER` is deliberately ignored for fills. Observed reconstructed market
prices are never rounded to tick size. The adapter does not hardcode current
BTCUSDC filter values. Constraints are embedded in replay JSON and hashes so an
experiment must preserve the exchangeInfo-derived model it actually used.

Binance's live market-order notional acceptance may reference a weighted average
over `avgPriceMins`. This offline engine applies the active constraints to actual
simulated fill notional. A future preregistration requiring exact admission parity
must provide and freeze a causal weighted-average-price source and its semantics.

## Additive accounting

For every buy or sell fill, midpoint-to-final execution cost decomposes into:

- spread cost from midpoint to the same-side top quote;
- depth slippage from that top quote to the book VWAP;
- configured additional adverse slippage from book VWAP to final VWAP.

For a completed position, exit contributions are quantity-weighted across all
partial exit fills. The trade identities are:

```text
mid-price move - spread cost                    = top-of-book gross PnL
top-of-book gross PnL - depth slippage          = depth-executable gross PnL
depth-executable gross PnL - adverse slippage   = filled gross PnL
filled gross PnL - fees                         = final net PnL
```

Every return uses actual entry fill notional as its denominator. Expectancy is the
arithmetic mean of trade returns. Profit factor uses positive net PnL divided by
absolute negative net PnL and is null when there are no losses. Win rate counts
strictly positive net PnL. Turnover is both-side actual notional. Aggregate costs
separate spread, depth slippage, additional slippage and fees.

Arithmetic runs in an isolated 50-digit Decimal context. Canonical JSON uses exact
decimal strings and UTC timestamps. The input hash binds session identity, depth
snapshots, decisions, execution rules, exchange constraints, fill mode and initial
cash. The replay hash also binds all fills, trades and aggregate results. Identical
inputs produce byte-identical JSON and hashes.

## Event compression

`EventCompressor` remains strategy-free. Its caller supplies threshold, rearm,
consecutive one-second persistence and cooldown. It begins unarmed, requires an
observed rearm value, emits at the final persistence observation, reserves one
event until explicit release, never queues while busy, and starts cooldown at
release. Gaps disarm it and require another observed rearm. It schedules no exits,
quantities or holding periods.

The model still assumes displayed depth remains executable at the selected
snapshot and applies one uniform extra slippage rate after walking the book. It
does not model queue depletion between snapshot and arrival, hidden liquidity,
market impact beyond supplied levels, partial fill timing within an order, tick or
lot rounding of observed fills, exchange fee-asset discounts, or live rejection
codes. Only synthetic tests exercise this foundation.
