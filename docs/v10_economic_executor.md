# V10 economic discovery executor — implementation only

Authoritative preregistration: `a296e5ed304640d7`, definition SHA-256
`a296e5ed304640d706916cc8e562cedc6febf561b20b0c8a99d59ac30b275a33`.
The acquisition manifest, discovery manifest, reference implementations and
historical protocol documentation are unchanged.

This implementation is **not authorization to execute the real diagnostic**.
The existing CLI still exposes only non-predictive `PLAN` and hash-only `VERIFY`.
No real authorization record, reservation, target, prediction or result is created
by installing/importing these modules or running their synthetic tests.

## Components

- `v10_economic_features`: observes the frozen acquisition reconstruction machine;
  streams the causal depth-before-trade merge into bounded rolling flow windows
  and one-second/100ms-offset quote samples. Buffered diffs become usable only at
  their bridge receipt. It preserves last-diff source timestamps, rejects stale
  endpoints, applies a common four-horizon mask, and never crosses session ends.
- `v10_economic_discovery`: the exact paired 13/40-feature Ridge models, alpha 1,
  training-only median/population scaling, seven expanding whole-session folds,
  identical baseline rows, and 1,000 paired circular-shift refits using seed
  20260916. Feature-only preprocessing and Gram inverses may be cached; fitted
  coefficients and training-target means are never reused across permutations.
- `v10_economic_reporting`: every predefined training-quantile bin, positive tail
  and economic threshold; Decimal quote-cost response, run persistence, fixed
  five-minute clustering, turnover demand and non-overlapping diagnostic anchors.
  Only the isolated 300s primary mapping enters classification. No threshold
  selection, position simulation or trading-profitability conclusion exists.
- `v10_economic_execution`: a dormant, fixed-path runner with preregistration,
  fresh acquisition readiness, complete hash rebinding, executor/runtime identity,
  exclusive reservation and exclusive-create, hash-sealed reports.

## Separate future authorization

A later task must explicitly authorize **one real execution** for the ID above.
Only that future task may create the separate authorization record at
`research/v10_economic_execution_authorization/a296e5ed304640d7/authorization.json`.
There is deliberately no authorization writer or execution CLI in this change.
The record must identify the user's separate authorization and bind the exact
definition, complete executor source hashes, source commit and Python 3.12 runtime
identity. Its strict schema is enforced by `validate_authorization`.

The dormant runner verifies that record and all frozen bindings, reruns acquisition
READINESS, and compares the resulting complete dataset binding before reservation.
Both before sampling and after computation it verifies source/runtime and data
hashes again. An identity change fails closed. No real inputs are replaceable
through command-line arguments, alternate roots or reduced permutation settings.

`execute_once()` is the sole production orchestrator and accepts no arguments.
The former injectable `reserved_execution()` helper has been removed, with no
alias. Sampling, diagnostic execution and result publication are owned by the
guarded function. The reusable `reserve()` primitive only creates a directory
and writes its ledger; it cannot load data or run/publish a diagnostic. Synthetic
orchestration tests patch dependencies in temporary workspaces, not production
callback parameters. These controls do not protect against arbitrary Python
monkeypatching or modification of the source by an administrator.

## Permanent reservation and crash behavior

The fixed output directory is
`reports/v10_microstructure_economic_discovery/a296e5ed304640d7/`.
Its exclusive atomic creation is the reservation. **Any existing directory or
result refuses execution**, even an empty directory left before ledger publication.
The immutable `reservation.json` is flushed/fsynced before constructing targets.
The dataset is conservatively marked consumed in the reservation; a crash does not
restore an unused opportunity. The implementation never removes this directory.

An exception, process termination, failed integrity check, disk-write failure or
partial result leaves the reservation intact. There is no force/reset/retry/
overwrite mechanism. Do not delete, restore away or relocate the reservation to
permit another execution. This is a local filesystem/process-crash guarantee,
not protection against administrative deletion, disk loss or rollback from backup.
File writes are exclusive and fsynced; directory entries are also fsynced where
the operating system supports directory fsync.

No classification is published until post-computation binding checks pass.
Artifacts are `samples.jsonl` (features, targets, economics and causal endpoint
provenance), `predictions.json`, `null.json`, `report.json`, and `seal.json`.
The report binds the input dataset, authorization, executor/runtime and auxiliary
artifact hashes; the final seal hashes the report. Missing seal means incomplete,
not permission to rerun. Decimal financial values are strings; undefined
descriptive metrics remain JSON null; nonfinite or undefined required primary
statistics stop classification. Serialization adds no variable wall-clock fields.

Every report states **DISCOVERY ONLY** and:

> Positive classification does not establish executable profitability.

Quote response is an infinitesimal-liquidity economic bound, not a sized fill.
Depth capacity remains NOT_ESTIMATED. The blind holdout remains locked; V9 data
is not an input. Later strategy validation requires separately frozen execution
semantics and genuinely new prospective sessions.
