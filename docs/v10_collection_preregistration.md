# V10 public microstructure acquisition protocol

The immutable acquisition manifest lives in
`research/v10_public_microstructure_collection/<preregistration-id>/manifest.json`.
Its content-derived ID and full definition SHA-256 bind the frozen rules. The
loader verifies the complete definition and never creates or rewrites it.
Changes require a separately frozen protocol; this protocol has no recovery
or predictive evaluation mode.

```text
python -m src.cli.run_v10_collection --mode PLAN
python -m src.cli.run_v10_collection --mode READINESS
```

Both commands are read-only and print JSON. PLAN reads only the acquisition
manifest. READINESS defaults to `data/microstructure/v10`; an explicit
`--data-root` may select another isolated V10 artifact directory. V9 roots and
their ancestors are forbidden. Exit codes: 0 for PLAN/READY, 2 for NOT_READY,
1 for protocol or scan-wide integrity failure. Individual ineligible sessions
remain listed with their rejection reason and contribute no hours.

## Frozen eligibility

Only shared BTCUSDC public Spot depth@100ms and aggTrade sessions starting
strictly after the cutoff may qualify. The cutoff is the first full UTC
15-minute boundary strictly after manifest creation, including when creation
falls exactly on a boundary. Equality, straddling, active/incomplete and
interrupted sessions are rejected.

The scanner verifies normal closure, exact artifact and stream identities,
UTC session boundaries, positive exact trade quantities/prices, maker booleans,
raw hashes and counts, live/replay accounting and two complete deterministic
replays. It also computes a hash of every causal trade context and rehashes
artifacts after verification. Source commits must be locally resolvable and
contain the frozen collector code. Local replay code must match the same
SHA-256 hashes (source line endings alone are normalized to LF).

V1 conservatively rejects **all depth sequence gaps**, even subsequently
resynchronized ones, and all trade-ID gaps, regressions, conflicting duplicates,
invalid records and invalid/crossed books. Exact consecutive trade duplicates
remain in the raw file and count once in derived contexts. Older regressing IDs
fail rather than being silently deduplicated.

Depth resync markers remain raw source truth and must reconcile with counters;
replacement snapshots need a preceding boundary. The current collector records
reconnect counts without complete raw reconnect boundaries for both sources.
Consequently, **any reconnect is ineligible under V1**. A future protocol and
collector version would be needed to admit such sessions.

## Causality and accounting

Order is receive/availability UTC, then depth before aggTrade, then source record
index. Buffered updates become available no earlier than snapshot receipt.
Trade contexts use the latest valid book available at or before trade receipt.
Earlier missing contexts remain missing, are counted, and are allowed; the
scanner never interpolates or repairs them with future observations. Exchange
timestamps later than receipt fail; no clock correction is inferred.

Readiness requires at least **8 eligible closed sessions, 24 eligible hours,
and 3 distinct UTC session-start dates**. Hours are the smaller of requested
duration and actual closed interval. Identical full artifact copies count once;
conflicting copies invalidate every copy of that ID. Overlapping sessions are
rejected, with touching endpoints allowed. Elapsed acquisition hours do not
claim continuous usable L2 coverage or a minimum number of future research
samples. Missing contexts remain excluded or handled only as specified by a
future hypothesis-specific protocol.

These thresholds establish acquisition sufficiency only. Sessions may support
later authorized V10 discovery. They are **not independent confirmatory strategy
evidence** unless that strategy hypothesis was frozen before session generation
and its separately frozen eligibility requirements pass. Consumed V9 evidence
is never fresh evidence. The blind holdout remains LOCKED.

## Immutability boundary

The collector uses exclusive creation and append-only logs, and readiness checks
content hashes, source identity and changes during the scan. These are local
integrity guarantees, not cryptographic authentication of Binance messages or
hardware-enforced write protection. Preserve the frozen manifest in version
control before future collection. No commit or live collection is performed by
PLAN or READINESS, and neither can make a trading or predictive decision.
