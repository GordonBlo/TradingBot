# V12 Phase 1: conditional maker quantity bound

Execution-identifiability engineering only. This is not an economic
preregistration, collection authorization, strategy, fee amendment, or empirical
fill validation. BTCUSDC Spot, LONG LIMIT_MAKER only; no credentials or orders.
V9/V10 evidence remains CONSUMED; V10 CLOSED; V11 DEFERRED / CLOSED_FOR_NOW.
The blind holdout remains LOCKED. No economic conclusion follows from this work.

## Observation model and knowledge at a local time

At local monotonic time n, knowledge consists only of records whose receipt
monotonic timestamp is <= n, whose UTC receipt is <= the local as-of time, and
whose integrity/provenance has been checked. Processing time is not receipt
time. Exchange timestamp order must never replace receipt order to make a
decision use a message that had not arrived.

| Category | Meaning |
| --- | --- |
| OBSERVED | Persisted public payloads: reconstructed displayed depth; depth U/u/E; individual trade ID/p/q/T/E/m; aggTrade a/f/l/p/q/T/E/m; bookTicker u/b/B/a/A; source units; local UTC/monotonic receipt; ingest ordinal; connection/session identities; public filters/status; explicit snapshot, resync, gap/outage boundaries. Reconstructed state retains source and availability times. |
| ASSUMED | Counterfactual order acceptance and continuous eligibility, arrival ask, external-flow invariance, ordinary price-priority matching, absence of STP, fixed submission/cancellation latencies, a valid exchange/local clock mapping. These are named scenario inputs, not findings from public data. |
| UNOBSERVABLE | Actual acknowledgement/working/cancel times, individual queue positions, cancellation identities, hidden reserve amounts, actual counterfactual fills, and behavioral response to our displayed order. None is replaced by zero. |
| INDETERMINATE | Unknown or invalid coverage, arrival/cancel timestamp overlap, missing identifiers, stale reconstruction, unknown symbol status, clock discontinuity, receipt/source contradiction, disconnected intervals. A timing-overlap event gets zero additional credit; lost history latches the order INDETERMINATE. |

Depth values are absolute replacement quantities. Depth decreases are not
cancellations and cannot be used twice with reported executions. bookTicker
contains a depth update ID, not a matching-engine timestamp in the documented
JSON payload. Exchange E is publication/event time; T is trade time. Neither
gives an order acknowledgement or a cross-stream total matching order.

Current V10 records separate depth/aggTrade sockets, application UTC receipt,
per-file indices, snapshots and depth resync boundaries. It lacks individual
trades, best-quote stream, explicit aggTrade outage intervals, monotonic receipt,
shared ingest ordinal and clock/backlog provenance. Its deterministic receipt
merge is not a matching-engine ordering. V10 recording/results are unchanged.

## State machine

DECIDED -> SUBMISSION_PENDING at the decision's local instant. The exogenous
submission latency sets an assumed activation instant; exchange-clock bounds
turn it into a closed interval [a_min, a_max]. It is not measured order latency.

At activation:

- Assumed arrival ask <= L: REJECTED_ASSUMED, no taker conversion or fill.
- A simultaneous assertion of acceptance AND a crossing arrival is contradictory
  and gives INDETERMINATE, not a bound over an empty compatible-world set.
- Unknown acceptance/ask or missing matching assumptions: INDETERMINATE.
- All explicit assumptions asserted and ask > L: CONDITIONALLY_WORKING.

Positive credit gives PARTIALLY_FILLED; cumulative Q gives FILLED. A local cancel
request from a working/partial state gives CANCEL_PENDING. Fixed cancellation
latency and clock bounds give [c_min, c_max]. At the assumed local cancel effect
the state is CANCELLED; partial credited inventory is retained. Fully credited
quantity wins a cancel race. Late receipts of executions certainly before
c_min can still increase the historical bound, while retaining CANCELLED for
partial quantity. No claim of an actual cancel acknowledgement is made.

Any missing coverage, gap, reconnect, resync, new bootstrap snapshot, invalid
book, stale state or uncertainty in trading eligibility latches INDETERMINATE.
Prior valid credit is preserved as a bound, never increased after that boundary.
A later valid snapshot cannot repair this order's missing path.

## Theorem, limits and counterexamples

Let H be a public historical execution path WITHOUT our order. Insert one
hypothetical quantity-Q bid at L. Define a compatible counterfactual world as
one satisfying ALL of the following, independently of outcomes:

1. The bid is accepted, resting and eligible throughout the declared working
   interval, with valid quantity/price filters, continuous trading and no STP,
   routing exception, unsolicited expiry or rejection. Arrival ask is an
   explicit assumption; a public local quote cannot prove it.
2. Matching is ordinary price/time priority, with atomic aggressive-order
   matching actions. A resting eligible bid cannot be skipped for a lower bid.
   Iceberg refresh follows the same venue priority mechanism in both worlds.
3. External order instructions (submissions, quantities, limits and identified
   cancellations) and their matching-action ordering/times are unchanged by
   insertion. This is a fixed-order-flow counterfactual, NOT a guarantee of
   zero market impact or invariant behavior in an actual market.
4. Recorded individual trade timestamps conservatively bound those matching
   actions, not merely delayed publication. The order must be eligible for the
   atomic matching action itself; if timestamps cannot establish that under the
   declared timing model, the action is ambiguous. Clock bounds, timestamp
   resolution and stream coverage are valid;
   unresolved activation/cancel ties do not establish eligibility.
5. Only ordinary on-book aggressive sells are eligible. Off-book/block trades
   and unresolved execution-mode exceptions are not eligible evidence.

Let V(t) sum quantities of distinct ordinary individual trade prints that:

- have arrived by local t;
- have buyer-is-maker=true and price p < L;
- have their whole trade-time interval strictly after a_max and strictly
  before c_min (if a cancellation was requested);
- occur within continuously verified history before any integrity boundary.

Then the queue-independent conditional credit is

    C(t) = min(Q, V(t)).

### Proof

Compare the original and inserted-order matching paths under the same external
instructions. Before our order is full, it cannot displace execution of bids
strictly above L or of eligible same-price orders ahead of it. Those orders
have priority in both paths. External same-price orders submitted after our
insertion are behind it. Any lower-price volume displaced by our order leaves
lower-priority liquidity behind our still-working bid. Fixed cancellations
may remove ahead liquidity but cannot insert it or move a behind order ahead.
The same argument holds for hidden reserve: reserve with priority ahead must
be consumed before our order in both paths; refreshed reserve behind cannot
receive volume in preference to our eligible bid.

Consequently historical aggressive-sell volume executed strictly below L
cannot bypass our remaining eligible quantity in the inserted path. It is
absorbed by our bid until Q is exhausted. Summing distinct qualifying volume
and capping at Q proves actual counterfactual fills >= C(t). Discarding ambiguous
events only weakens the bound. The argument concerns displaced external sell
volume, not equality of observed depth between the two paths.

This is the strongest general bound from strict-below volume alone without
queue knowledge: compatible worlds can put all observed executions at L ahead
of us and give us exactly min(Q,V) of displaced below-L volume. Particular
histories can imply a stronger bound; this engine does not search for them.
The unconditional public-data lower bound remains ZERO when acceptance or
these matching assumptions are not asserted. Hidden liquidity is unknown,
not zero. Public data cannot empirically verify the counterfactual assumptions.

### Counterexamples to broader rules

- Q=2, three units ahead, one unit trades at L: actual fill=0 despite touch.
- Q=2, historical below-L volume=.25: full-fill-on-trade-through credits 2
  even though a compatible inserted world fills only .25.
- Trades below L before assumed activation: actual fill can be 0.
- Unknown post-only acceptance: rejection gives actual fill=0 despite any
  subsequent trade-through.
- Unknown cancel effect overlapping a trade: a compatible early cancellation
  gives 0 additional fills.
- Cancellation before/after our position can give different actual fills with
  identical public depth and trades. Displayed depletion is not a fill proof.
- External traders can react to our displayed bid by withdrawing or changing
  sell instructions. Without flow invariance the historical through volume is
  not an actual-world lower bound.

The credit is quantity, not a conservative PnL bound. Omitted executions may
lose money; credited fills must not be selected using subsequent markouts.

## Adversarial compatible sets

All quantities below are fabricated; Q=2. Sets are exact for the specified
synthetic witnesses, or intervals allowed by explicitly missing information.
An integer witness set does not exclude intermediate quantities in more
general worlds. The proof above covers quantities beyond the finite oracle.

| Case | Compatible actual fill quantities | Credited quantity |
| --- | --- | --- |
| Same-price volume entirely ahead | {0} | 0 |
| Hidden ahead versus refreshed behind, same public history | {0,2} | 0 |
| Cancellation ahead versus behind, same public history | {0,2} | 0 |
| Known cancellation ahead followed by sell 1 below L | {1} | 1 |
| Additions behind, trades only at L | includes 0 | 0 |
| Replenishment at L | includes 0 | 0 |
| Working trade-through, historical below volume 2 | {2} | 2 |
| Before activation, including delayed receipt | includes 0 | 0 |
| Partial trade-through, historical below volume 1 | {1} | 1 |
| Multiple price levels, below volume 1 | {1} | 1 |
| Depth before/after same exchange trade, reordered receipt | minimum 1 under the same declared working assumptions | 1 |
| Regressing exchange T despite increasing receipt | [0,2] for unresolved history | 0; INDETERMINATE |
| Dropped depth or individual trade | [0,2] for unresolved history | 0; INDETERMINATE |
| Reconnect/resync, then later valid snapshot | [0,2] for unresolved history | 0; INDETERMINATE |
| Cancel pending, certain pre-effect volume 1, later boundary tie | [1,2] | 1 |
| Price moves away without aggressive sell | {0} in specified path | 0 |
| Post-only would cross at assumed arrival | {0} | 0; REJECTED_ASSUMED |
| Truncated/unknown coverage or stale observations | [0,2] | 0; INDETERMINATE |
| Same timestamp overlapping activation/cancel interval | includes 0 | 0 |
| Same receipt timestamps with ordinal tie-break | model minimum unchanged | no queue inference |
| aggTrade overlaps individual trade IDs | unchanged | individual volume credited once |

Tests also exhaustively compare 4,096 independently matched synthetic worlds,
varying ahead/better/additional liquidity, two sell sizes and cancellation
quantity. Each world compares the engine credit against its actual inserted
order fills. Separate identical-observation witnesses establish why touch and
depletion cannot identify actual fills. This is deterministic adversarial
verification, not a statistical calibration, forecast or economic evaluation.

## Engine contract

`src/research/v12_maker_bound.py` handles one hypothetical order and one
normalized public tape. It reuses existing Decimal/timestamp integrity helpers;
it does not duplicate the existing order-book reconstructor. The eventual
upstream adapter must validate raw hashes, depth contents, filters/status,
coverage, freshness, clock bounds and ordinary on-book trade provenance.
Sequence continuity alone is not proof of valid book contents.

The engine checks global ingestion continuity, per-stream connection identity,
depth U/u continuity, individual trade continuity, local clocks, exchange trade
chronology, receipt causality, duplicate conflicts and interval eligibility.
Each individual trade can credit volume once. Aggregate trades and bookTicker
cannot credit any volume. Fill price is L, never the historical lower price.
Arithmetic is exact within a declared bounded Decimal domain and independent
of caller decimal precision. No real-data loader, network, PnL, commission,
profitability classifier or strategy integration exists.

The state and ledger are conditional scenario outputs. No actual exchange fill
or acceptance is claimed. The engine cannot detect an integrity event withheld
by its caller. Missing source provenance must be an upstream boundary, not a
silent assertion of healthy coverage. Initial IDs/coverage are explicit inputs.

## Exact future public collection specification

New schema family: `V12_PUBLIC_EXECUTION_OBSERVATION_1`; not implemented here.
Append-only raw envelopes, immutable closed-session manifest and closure hashes.
No records sourced from credentials or private account channels.

Every envelope persists:

- schema_version, record_type, symbol=BTCUSDC, venue=BINANCE_SPOT, session_id;
- connection_id, stream_name, per-connection stream_record_index;
- shared ingestion_ordinal allocated at application message receipt, before
  decoding/dispatch; UTC receipt ISO text and integer UTC epoch microseconds;
- receipt_monotonic_ns, monotonic_clock_id/host_boot_id, clock_mapping_id;
- exchange_timestamp_unit (`MICROSECOND`, `MILLISECOND`, or `NONE`), raw E/T
  when present, normalized exchange intervals retaining original precision;
- exact source message bytes (base64 or UTF-8 with encoding declared),
  raw_payload_sha256 over those exact bytes, decoded canonical payload and
  canonical_payload_sha256 over explicitly UTF-8 canonical JSON;
- decode_start/end_monotonic_ns, dispatch_monotonic_ns,
  persistence_complete_monotonic_ns, ingress_queue_depth, backlog high-water,
  observed overflow/drop counts and websocket library/version/settings.

Receipt is the earliest application delivery point, not a kernel receive time
or engine time. Parsing/persistence diagnostics cannot replace receipt times.
Clock records persist UTC/monotonic samples, adjustment/discontinuity events,
clock-sync source, uncertainty policy and public server-time HTTP request/response
intervals. RTT bounds clock comparisons; it does NOT measure order latency.
Unsupportable clock bounds invalidate eligibility. Preserve explicit unknowns.

| Record type | Required payload/provenance and integrity |
| --- | --- |
| TRADE | Exact @trade e/E/s/t/p/q/T/m; M retained but ignored. Decimal p/q >0, Boolean m, source timestamp units, IDs unique/contiguous across eligible coverage; exact duplicates retained but no duplicate volume, conflicting duplicates/regressions/gaps invalidate coverage. Distinguish/exclude any non-on-book execution class. |
| AGGTRADE | e/E/s/a/p/q/f/l/T/m, M ignored. Contiguous aggregate and underlying ranges; reconcile underlying IDs and Decimal volume with individual trades. Never add aggregate volume to individual volume. |
| BOOK_TICKER | u/s/b/B/a/A; valid uncrossed bid/ask and nonnegative quantities. E/T absent => explicit null, never synthesized. IDs relate to depth updates but need not be consecutive: not every update changes the best quote. ID regressions/conflicts and consistency discrepancies are explicit. |
| DIFF_DEPTH | e/E/s/U/u/b/a, exact replacement quantities, zero deletion, Decimal validity, update continuity against prior u, original source/receipt times. Reuse deterministic reconstruction with snapshot bridging. |
| REST_SNAPSHOT | lastUpdateId/bids/asks, configured snapshot limit and retained-level bound, endpoint/query, HTTP request-start/response-receipt UTC and monotonic times, status and exact response hash. No invented exchange snapshot time. Track known price coverage; missing outside it is unknown. |
| CONNECTION | CONNECT_START, CONNECTED, DISCONNECTED, RECONNECT_START, CONNECT_FAILED, CLOSE; stream, connection/session identity, UTC/monotonic times, reason/exception class, last observed IDs and outage-open/close records. Healthy empty intervals differ from disconnected intervals. |
| RESYNC | START/INVALIDATE/SNAPSHOT_REQUEST/SNAPSHOT_RECEIVED/BRIDGE_COMPLETE/FAIL; reason, old/new IDs, affected coverage, buffer counts/overflow. New snapshot cannot restore past order eligibility. |
| CLOCK | Clock mapping identity, local UTC/monotonic pair, offset interval and provenance, server-time request/response bounds, drift/discontinuity flag. Fixed submission/cancel scenarios are separate model inputs. |
| SYMBOL_RULES | Exact public exchangeInfo/executionRules response bytes/hash, retrieval interval, status, supported order types, price/lot/notional/order filters and validity interval. Unknown or stale applicability disables the hypothetical order; never query account order/commission APIs. |
| SESSION_CLOSE | Stream hashes, counts, first/last IDs and receipt times, lifecycle/outage inventory, raw closed flag, source/runtime/config identities, explicit integrity failures and deterministic replay hashes. Eligibility requires closed immutable files. |

Request `timeUnit=MICROSECOND` where supported; retain explicit source precision
and version parsers accordingly. The old V10 millisecond parser is unchanged.
Record all lifecycle transitions even when no market messages arrive.

Replay uses ingestion_ordinal within one clock/session epoch; monotonic and UTC
receipt consistency are checked, not used to reshuffle. Lifecycle records use
the same ordinal allocator. Same receipt timestamps are permitted with distinct
ordinals. Separate clock epochs/sessions require explicit boundaries, not a
synthetic global engine ordering. Individual IDs provide within-trade continuity;
depth IDs provide within-depth continuity. E/T, bookTicker u and aggTrade IDs
must never be combined to assert a cross-stream matching-engine sequence.
Late/out-of-order records are preserved and flagged, not hindsight-sorted into
eligibility. Backfill, if recorded later, is a separate provenance class and
cannot create past causal credit or heal an outstanding order's missing history.

Primary documentation:

- [Binance public WebSocket streams](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md)
- [REST order and market-data semantics](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md)
- [Same-price priority and iceberg amendments](https://github.com/binance/binance-spot-api-docs/blob/master/faqs/order_amend_keep_priority.md)
- [Private execution reports, outside V12 scope](https://github.com/binance/binance-spot-api-docs/blob/master/user-data-stream.md)

## Decision boundary

CONDITIONAL_MAKER_BOUND_READY requires the proof's assumptions to remain named,
adversarial credit <= minimum-compatible-fill invariants, and passing focused
and full deterministic suites. It is readiness of an isolated mathematical
bound, not readiness to collect, validate a strategy or claim actual fill rates.
The next prerequisite is a separately authorized public collector/provenance
implementation and synthetic replay verification against this specification.
If that pipeline cannot establish useful eligible intervals, stop rather than
relax assumptions or manufacture fills. Actual empirical fill calibration would
require separately authorized submitted-order feedback, outside public-only V12.
