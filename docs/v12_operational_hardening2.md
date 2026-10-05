# V12 operational hardening #2

**ENGINEERING CERTIFICATION != ACQUISITION AUTHORIZATION.**
No final acquisition manifest, cutoff, prospective creation interface, maker
plan, prediction, forward return, fill rate, economics or order interface is
introduced. Phase-1/2, Hardening #1 and Phase-3 draft/evidence remain historical.
Only the existing collector receives opt-in hooks; the frozen maker/replay
implementation is unchanged.

## Explicit bounded entry points

The original `v12_public_smoke --execute --seconds` remains <=30s. New
`v12_engineering_soak --engineering-soak --duration-seconds` accepts 1–900s.
The API also requires `engineering_authorized=True`; no prospective mode exists.
Capture deadline is fixed. Cleanup and offline certification follow without
extending market observation. HTTP workers are shielded and drained; independent
resource cleanup preserves primary and secondary failures. Clock and dated
exchangeInfo refresh every 30s; existing 60s freshness guards are enforced on
market observations, with receipt-time checks and explicit historical latches.

## Completion provenance

Raw envelopes retain their original pre-persistence dispatch semantics. Each
raw line is fsynced before its SHA-bound ack. Raw fsync completion is the existing
ack timestamp. Ack fsync completion is sampled after that fsync. The raw event
then enters the unchanged live ReplayMachine; a separate `availability.jsonl`
row records receipt, raw/ack completion, live completion, full local delay,
live trace hash, timestamped queue observations, guard violations and high-water
values. This avoids writing future completion inside its own raw/ack record.

The total metric includes raw/ack persistence and live processing, and must be
<=1s. It excludes the completion journal's own subsequent persistence, which
cannot truthfully describe itself before completion. That journal is separately
flushed/fsynced per row and sealed. Any missing/malformed completion rejects
the session. A future consumer must join this completion provenance, rather
than treating old raw dispatch as full pipeline availability. No execution
latency or exchange-arrival claim is made; upstream residence stays UNKNOWN.

## Queue semantics and bounds

The websockets assembler's queue holds **data frames**, not necessarily complete
messages. Every enqueue is observed, including bursts that overshoot its soft
backpressure mark; the 128-frame guard is a separate rejection ceiling, not a
claim that the library hard-caps occupancy. Receipt samples observe current
occupancy as well. Observations carry their own UTC/monotonic timestamps and
are persisted in completion rows. Kernel, transport and upstream queues remain
unknown. An incompatible queue implementation fails explicitly.

Writer and replay run synchronously under one ingress owner; there are no
asynchronous writer/replay queues. Their pending counts include the current
item, maximum one; excess pending work latches rejection. The captured event
is retained and processed before controlled termination. Other limits bound
session persistence by its preflight budget, raw record count to 100,000 and
aggregate reconciliation references to 1,000,000. The first exceeding event
and shutdown evidence are retained; these are operational resource rejection
limits, not changes to accepted replay/maker semantics. No silent drops.

## Disk and identity protection

The parent volume must have 10GiB reserve plus the full planned session budget.
Sizing uses ceil(4 × Phase-3 measured 69,975.4422 bytes/s), or 279,902 bytes/s.
That is operational headroom, not a forecast. Before raw/ack/completion and
finalization writes, available space is checked against reserve and pending
evidence. Low space stops without deleting evidence; a failed shutdown/record
is FAILED. Tests inject free-space doubles and never fill the real disk.

Every session uses a new directory and an exclusive fsynced identity reservation
in the workspace engineering identity registry. Existing directories or duplicate
IDs are rejected without touching prior sessions. Registry reservations remain
after failure. Tests may inject an isolated temporary registry. Historical
sessions are never retroactively registered, repaired or rewritten.

## Atomic seal and classification

Owned tasks and all three journals must close successfully after final fsync.
Metadata is finalized, exact hashes computed and the unchanged raw replay
certified twice byte-for-byte. The seal binds session ID, engineering scope,
classification, start/end clocks and namespace, raw/ack/completion/manifest/
metadata/terminal hashes, source/config/runtime identities, replay digest,
lifecycle checks, guard metrics, violations and failure reasons. Its body
digest excludes its own digest field. Exclusive `seal.pending.json` is flushed/
fsynced on the same volume and atomically renamed without overwrite using the
existing repository publisher. Final bytes are rechecked. Temp files are not
deleted or reused after failure. File fsync and no-replace publication do not
establish Windows directory durability or a power-loss guarantee.

Classification precedence is FAILED > INELIGIBLE > ELIGIBLE. Journal/ack/seal/
shutdown/corruption failure is FAILED; operationally closed continuity,
reconciliation, clock, backlog, availability or metadata rejection is INELIGIBLE.
Clean engineering capture is also research-INELIGIBLE with research status
ENGINEERING_ONLY and research_eligible=false. A pure classifier function prepares
future prospective mechanics, but no prospective session can be created here.

ACTIVE is reported only by the process that currently owns the capture; external
readers without a seal fail closed. A stale marker/PID cannot establish ACTIVE.
Read-only classification verifies seals and completion provenance repeatedly
without rewriting anything. A failure sidecar takes precedence over a later
replay or existing seal; no automatic promotion/recovery exists.

## Certification boundary

Offline tests precede exactly one authorized public engineering soak, preferred
900s. Raw replay and seal verification are operational checks only. A shorter,
failed or operationally rejected soak cannot support sustained certification.
The Phase-3 acquisition draft is unchanged: it remains NOT FROZEN and acquisition
unauthorized. V9/V10 remain consumed, V10 closed, V11 deferred, blind holdout
locked. No profitable strategy has been validated.

## Hardening #2 engineering run result

The single authorized 900s attempt ended after 65.276s of recorded lifecycle
when a data-frame burst reached 203 buffered frames, exceeding the unchanged
128-frame guard. The burst was observed on every enqueue, persisted and latched;
the capture stopped without retry, threshold relaxation or evidence repair.
The seal is valid, research classification INELIGIBLE / ENGINEERING_ONLY.
All resources closed. Raw sequence checks and reconciliation passed, but the
clean-end check correctly failed after controlled guard termination.

Full availability p50/95/99/max was 2.634/14.669/16.855/102.927ms, below 1s.
The reserve remained above 391GB. The exact guard-trigger-to-final-journal-close
duration was not instrumented: the clean-end shutdown timer is null, not zero.
This limitation is retained in verification. No 15-minute sustained certification
is claimed. Decision: **SUSTAINED_COLLECTION_UNRELIABLE** under the implemented
guards. A separately authorized engineering review is required before another
network test or any acquisition-protocol freeze.
