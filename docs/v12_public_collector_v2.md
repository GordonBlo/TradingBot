# V12 Phase 2: public observation and engineering replay

This is infrastructure certification, not acquisition authorization, a frozen
experiment, a maker economics result, or evidence of profitability. Phase 1
remains authoritative and unchanged. V9/V10 are consumed; V10 is closed; V11
is deferred. The blind holdout remains locked. No credentials or order interface
are used. The bounded CLI requires `--execute` and permits at most 30 seconds
of engineering capture, with bounded connection/HTTP cleanup afterwards.

## Binding and public surface

The collector verifies all three Phase-1 hashes recorded in its verification
artifact and binds that artifact's own hash. The session manifest binds these
four hashes, collector/replay/V10/order-book source hashes, HEAD, Python,
websockets version, explicit engineering guards, and the public URL.

Reuse consists of V10 canonical serialization, SHA/source helpers, its existing
public REST base, and the existing deterministic L2 reconstructor. The new raw
envelope is separate because the old collector does not preserve individual
trades or the common ingestion provenance required here. Historical code and
artifacts are not changed.

One concurrent combined public WebSocket subscribes to BTCUSDC depth100ms,
individual trade, aggTrade, and bookTicker. Market-data-only REST GET paths are
allowlisted to `/api/v3/time`, `/api/v3/exchangeInfo`, and `/api/v3/depth`.
HTTP requests use the MICROSECOND header; the stream URL uses
`timeUnit=MICROSECOND`. Exact HTTP response and WebSocket message bytes survive
as base64 with raw and canonical hashes. No invented E/T is assigned to
bookTicker or depth snapshots. Individual and aggregate trade E/T, depth E,
U/u, aggregate a/f/l and absolute depth replacements retain original payloads.

Public references:
[streams and snapshot bridging](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md),
[market-only endpoints](https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md),
[REST time units and symbol filters](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md).

## Provenance and lifecycle

One event-loop ingress assigns an ordinal and UTC/monotonic reading before
decoding. This boundary is the application receive boundary, not socket arrival
or exchange matching order. Ordinals are never retrospectively sorted by E/T.
Each record binds session, connection, reconnect epoch, per-stream index,
clock namespace, raw timestamp units/intervals, processing start/end, dispatch
availability and delay. Raw and acknowledgement journals are append-only with
flush/fsync. The separate acknowledgement hashes the exact event line and
records completion after fsync; the event cannot contain its own future
persistence completion. A lost acknowledgement makes the session ineligible.

The monotonic namespace is process-scoped; host boot ID is explicitly unknown.
Windows receipts use precise FILETIME UTC and the monotonic QPC-backed
perf_counter_ns; coarse GetTickCount64/datetime clock ticks cannot falsely
masquerade as sub-millisecond observations. Clock implementation is recorded.
Clock sampling span is persisted. UTC jumps/regressions, namespace changes,
index discontinuities and excess application processing delay invalidate replay.
Transport/kernel/WebSocket queue residence is **unknown**, not zero. Known
application drops are distinct from unobservable upstream drops. Sequence
continuity is independently tested. No exact arrival/submission latency claim
can be inferred from these receipts.

CONNECT_START/CONNECTED and RECONNECT_START/RECONNECTED bind epochs.
DISCONNECTED preserves outage start; reconnect preserves its end ordinal,
UTC and monotonic time. Snapshot request/response records contain worker HTTP
start/response readings plus separate ingestion availability. BOOK_SYNC_ESTABLISHED,
GAP_DETECTED, RESYNC_START/COMPLETE and SESSION_CLOSE are explicit. Later resync
can reconstruct current engineering state but never clears a historical failure.

Public serverTime yields an RTT-bracket offset diagnostic. This assumes a stable
local UTC clock and that serverTime was sampled within that request. It does not
measure matching-engine offset or order latency. Those remain exogenous,
explicit Phase-1 conditional scenario assumptions; observed diagnostics must
be compatible, but cannot establish them. Guards in Policy are conservative
engineering rejection inputs, not optimized execution parameters.

## Dated metadata and limitations

Exact exchangeInfo supplies symbol status, Spot permission, LIMIT_MAKER support,
PRICE_FILTER, LOT_SIZE, notional filters and all other original fields. Fixed
tick/step/min/max/notional constraints are checked for hypothetical orders.
Dynamic percent-price/reference-average limits, account order limits, changing
status between snapshots, exchange execution exceptions and actual acceptance
remain additional conditional assumptions. They are not claimed verified from
one metadata snapshot. Metadata/clock/book freshness guards prevent unrestricted
carry-forward. Commission is UNAVAILABLE_PUBLIC_ONLY: no authenticated
commission request and no assumed zero fee. No separate executionRules endpoint
is assumed supported by the documented market-only allowlist.

## Replay and reconciliation

Replay verifies exact-byte hashes, decoded originals, units, all local availability
times and acknowledgement hashes. It requires snapshots before reconstruction
and an overlapping U/u bridge; stale diffs do not advance state. A bounded
pre-snapshot buffer becomes usable only when the snapshot has been processed.
Decimal replacements are strict: malformed values/floats/nonfinite constants,
duplicate JSON keys/prices, crossed books or gaps are explicit failures.
Book hashes cover retained levels. Trimming shrinks known coverage; absent levels
outside it are not zero liquidity. bookTicker quotes can be compared only at
the exact reconstructed u. Intermediate u remains explicitly unverifiable;
no future nearest-neighbor matching is used or supplied to order decisions.

Distinct individual trade IDs must be consecutive after the first observed
high-water mark. Identical duplicates are deduplicated; conflicting duplicates,
regressions, incompatible timestamps and dropped IDs fail. Aggregates require
consecutive a IDs and disjoint contiguous f-l ranges. Pending missing individual
IDs are explicit. Once present, exact Decimal quantity, price and maker side
are checked. Different individual trade times inside an aggregate are preserved
as diagnostics, not forced equal. Every recorded individual must reconcile;
capture-edge partial ranges are ineligible rather than silently trimmed.
Pending reconciliation is indexed by missing ID, avoiding per-event history scans.

Certification replays twice and compares canonical bytes (ordering, book hashes,
inputs, failures and transitions). Closed raw/manifest/ack hashes are recorded
and rechecked on later loading. ENGINEERING_REPLAY_CERTIFIED still has
research_eligible=false and acquisition_authorized=false.

## Adapter: conditional quantity, never PnL

An explicit MakerPlan binds a processed decision prefix, order/session, fixed
exogenous submission/cancellation latency, clock intervals and Phase-1
assumptions. Prefix inputs must have been processed by both decision clocks;
no received-but-unprocessed observation is used. Fresh, synchronized, known
coverage, pre-decision trade high-water ID, compatible clock bracket and fixed
symbol filters are required. Every subsequent envelope reaches the unchanged
Phase-1 state machine; only individual aggressive sells strictly below the
limit credit min(remaining quantity, distinct qualifying quantity). Touch,
same-price, cancellations, displayed depletion, aggTrade and bookTicker never
credit volume. Explicit cancel clocks must follow working start; uncertain
arrival/cancel timestamp buckets yield no credit.

Gaps, resync, reconnect, stale/incompatible metadata/clock/book or unknown price
coverage latch INDETERMINATE. The Phase-1 provisional ledger stays immutable.
If final reconciliation contradicts it or a session is incomplete, the adapter
returns certified_conditional_credit=null; it never retroactively invents fills.
Public unconditional credit is always zero, including ambiguous acceptance.
There is no fees, PnL, strategy or private execution integration.

## Future eligibility before any acquisition

A separate future acquisition protocol must specify cutoff, exact stream/clock
policy, boundary handling, completeness, raw closure/seal and eligibility. Require
clean close, successful bridge, continuous depth and individual trades,
no unresolved gap/outage/resync, valid clocks/provenance, bounded processing,
metadata sufficient for the intended conditional scenario, complete trade/agg
reconciliation, reproducible book/maker replay, and immutable raw/source hashes.
Unknown upstream backlog must be acknowledged; matching latency/acceptance is
still conditional. Orders may only occupy covered post-sync intervals after
pre-decision high-water IDs are available. A partial first/last aggregate or
uncovered trade rejects the current whole-session engineering certificate;
any future explicit bounded-prefix policy must be defined before collection.
Maker-bound certification requires an explicit predeclared hypothetical plan;
network/schema certification alone does not certify a fill experiment.

Synthetic adversarial fixtures cover all requested cases. Optional public smoke
tests only schema/network compatibility and remains engineering data. No V12
acquisition or economic preregistration is created by this phase.
