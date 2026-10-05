# V12 — DRAFT / NOT FROZEN / ACQUISITION NOT AUTHORIZED

No final protocol identity, definition hash or cutoff is assigned. `DRAFT.json`
contains the exact operational proposal. No prospective collection or maker-plan
evaluation is authorized. This draft is not a frozen manifest.

## Resume evidence

HEAD `ce59f7f68c10b1b20239051f1b1dbc1dc48f0492`, branch `v3.3-manual`,
matches the hardening tag. Phase-1/2, historical tags, original defect and
hardening verification are preserved. The shutdown defect is fixed; offline
failure/recovery tests pass without another demonstrated correctness defect.

One public soak preserved the unchanged **30-second engineering guard**. It
recorded 456 events: 203 depth, 27 individual trades, 13 aggregates, 201 tickers
and 12 REST/control records. All eight raw-replay certification checks passed;
ordinals were continuous, reconciliation complete, gaps/reconnects/clock failures
zero. Files, WebSocket and sampler closed. No maker plan, credited quantity,
score, return, adverse selection or economics was evaluated. This session is
permanently ENGINEERING_ONLY, never prospective evidence.

Persistence averaged 69,975 bytes/s: approximately 252 MB/hour and 6.05 GB/day
if this rate continued. Public payload throughput was 13,586 bytes/s. There
were 912 fsync calls (29.40/s); capture including raw/ack fsync had p50/95/99
2.245/4.272/5.084ms. Envelope-only delay was 32.5/135.2/223.1us and excludes
persistence/live replay. CPU including certification was 14.78% of one core;
observed RSS peaked at 104.6 MB, process high-water 106.7 MB. Logical process
writes were 64.4 kB/s; physical disk throughput is unknown. Observed WS frame
occupancy reached 35; transport/kernel/upstream residence remains **UNKNOWN**.
SESSION_CLOSE-to-journal-close was 3.677ms, excluding earlier socket/worker
drain and later certification. The metrics record explicitly preserves these
measurement scopes; this short sample is not a forecast or sustained-load proof.

## Decision and prerequisites

**NEEDS_OPERATIONAL_HARDENING.** Current code accepts only engineering captures
<=30s. Prospective 15-minute capture, bounded-memory behavior and refresh are
unverified. Backlog occupancy is not persisted/guarded; current processing
delay excludes fsync and replay. Prospective authorization, fixed scheduler,
disk reserve, failure-aware eligibility and atomic session/dataset seals need
implementation and certification before freeze. No guard was bypassed here.

## Proposed coverage and endpoint

Propose 15-minute sessions at 00:15 and 12:15 UTC, after a future cutoff.
Readiness requires six eligible sessions, one eligible synchronized hour,
three UTC start dates and at most 40% of all eligible coverage from one date.
Count every passing session; no selective removal. Two independent boots on
each of three dates is a minimal repeatability check; six windows give 90
minutes gross, with the one-hour floor rejecting mostly-unsynchronized capture.
This establishes no statistical power, maker-event adequacy or regime coverage.
It does not reuse V10's 24h/8/4 discovery targets.

The fixed maximum is 72 hours after cutoff. Launch every slot strictly after
cutoff whose full window plus 30s cleanup allowance fits before expiry. Require
bootstrap within 10s. Stop at first closed-session readiness PASS, otherwise
seal INSUFFICIENT_ACQUISITION. Preserve failed/missed slots; no retry in the
same slot, catch-up, discretionary extension or outcome-selected extra drain.

## Cutoff, classification and eligibility

Complete operational prerequisites, obtain separate freeze authorization,
commit/tag the final protocol and source/config/runtime bindings, then select
and bind a future UTC cutoff strictly after freeze. Obtain separate acquisition
authorization. First control/receipt must be strictly post-cutoff; no bootstrap
straddle. No actual timestamp is chosen. Every smoke/soak/engineering session
remains excluded forever, regardless of certification or timestamp.

Precedence: FAILED, then INELIGIBLE, then ELIGIBLE. ACTIVE is PENDING, never
research input. Interruption, Ctrl+C, truncation, missing ack/seal, malformed
hash/provenance/replay or resource-close failure is FAILED. Closed structurally
verifiable sessions failing operational eligibility are INELIGIBLE. A failure
sidecar/controller latch cannot be overridden by raw replay. Restart only under
new identity/directory; never repair, trim, rewrite, delete or promote history.

ELIGIBLE requires prospective authorization/scope, CLOSED, successful resource
cleanup, immutable seal, exact raw/ack/manifest/metadata bindings, every ack,
two byte-identical replays, valid snapshot bridge, continuous depth/trade/agg
IDs and ranges, complete reconciliation, continuous ordinals/stream indices/
epochs, valid UTC/monotonic namespace, required dated metadata, no historical
gap/outage/resync/drop/latch and post-cutoff start. Proposed guards to implement:
full receipt-through-fsync-and-live-replay availability <=1s and persisted
local WS frame occupancy <=128. Refresh clock/metadata at most every 30s;
retain existing 60s freshness and 10ms clock-discontinuity limits. Unknown
upstream delay remains unknown; these requirements do not claim current support.

First trade ID proves no earlier history. Partial first/last aggregates and
unmatched trades reject the whole session. No trimming, selective prefix,
backfill or drain selected to improve reconciliation. Count synchronized
coverage from first BOOK_SYNC_ESTABLISHED to expected endpoint DISCONNECTED;
retain all bootstrap bytes. Fixed 900s endpoint cannot be extended by outcomes.

## Storage and seals

Rotate paired raw/ack journals by immutable session directory. For each event,
flush/fsync raw before its SHA-bound ack, then flush/fsync ack. No batching.
Budget at four times measured throughput: about 252 MB per 15-minute session.
This is operational headroom, not a confidence bound or guarantee. Preflight
requires 10GiB reserve plus one session budget; 10GiB exceeds 40 such budgets.
Use a separate probe for local same-volume write/fsync/close and atomic
publication. Check reserve before every append; breach stops acquisition,
attempts FAILED closure and preserves evidence. Never auto-delete raw data.

Close owned tasks/files, flush/fsync, hash exact raw/ack/manifest/metadata bytes
and replay twice. Write/fsync an exclusive temporary seal on the same volume;
publish final marker atomically without overwrite, then verify bytes. Record
actual filesystem durability capability; do not invent directory-fsync or
power-loss guarantees. A missing/torn marker is FAILED. Optional compression
is post-seal and lossless, with archive and original hashes; decompress to
identical bytes and retain originals.

Future dataset seal binds protocol ID/SHA/commit/tag, cutoff record/hash, every
session and missed slot including failed/ineligible history, exact UTC and
monotonic clocks, raw/ack/manifest/session-seal/replay/metadata hashes,
collector/config/source/runtime identities and readiness totals/booleans.
Absent artifacts have explicit null hashes/reasons, never omitted history.
Body hash excludes itself; later evaluation preregistration binds final seal
bytes. Publish immutably before maker-plan evaluation. No dataset seal is
created here.

## Outcome firewall

During acquisition expose only session/classification counts, eligible hours,
UTC dates, event/gap counts, clock/provenance/reconciliation/replay health,
collector and disk health. No credited quantities, fill rates, scores, returns,
adverse selection or PnL. Acquisition must have no maker/score/economic entry
point and no exploratory raw-data readout.

Completion does not authorize evaluation. A separate preregistration before
opening outcomes must freeze limit-price rule, quantity, submission/cancellation
latency assumptions, expiry, conditional acceptance assumptions, maker-bound
application, horizon, fees/costs, estimands, statistical method and success/
failure gates. No parameter may be chosen from prospective outcomes. Public
data does not establish unconditional hypothetical fills or profitability.

V9/V10 remain consumed, V10 closed, V11 deferred/closed for now, holdout locked.
No prospective acquisition, private data, orders, final freeze, commit or
paper/live advancement occurred in this task.
