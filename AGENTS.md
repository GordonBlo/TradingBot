# TradingBot Agent Instructions

## Project

Research-grade crypto trading and execution system.

Primary traded market:
- BTCUSDC Spot
- 15m
- LONG only

Core requirements:
- deterministic behavior
- causal correctness
- reproducible research
- explicit integrity failures
- no silent assumptions

Prefer existing project infrastructure over parallel implementations.

---

## Instruction Precedence

For a specific frozen experiment:

1. Frozen preregistration manifest
2. Frozen dataset manifest
3. This AGENTS.md
4. Current task prompt

A frozen experiment manifest is authoritative for that experiment.

Never silently modify a frozen rule, threshold, horizon, gate, seed,
window, dataset, or execution semantic.

If instructions conflict:
STOP and report the conflict instead of guessing.

Historical research artifacts must not be rewritten merely to make them
look cleaner or internally consistent.

---

## Blind Holdout

Authoritative blind holdout:

2025-08-01T00:00:00+00:00
to
2026-02-01T00:00:00+00:00

During research it must remain:

- LOCKED
- NOT DOWNLOADED when avoidable
- NOT LOADED
- NOT REVEALED
- NOT CONSUMED
- NOT EVALUATED

Never access it unless a future task explicitly authorizes final holdout
evaluation.

Previously inspected outcome data is CONSUMED research data and must
never be described as fresh OOS / independent validation.

---

## Causality

Never use future information.

At decision/signal time T:

source_timestamp <= T

must hold for every source value used.

Rules:
- no future fill
- no future nearest-neighbor matching
- no interpolation using future observations
- missing data remains missing unless a causal carry-forward rule is
  explicitly defined
- preserve original source timestamps when carrying values
- forming candles/intervals must not be treated as completed data

Any future-data violation is an integrity failure.

---

## Frozen Execution Semantics

Unless an explicitly preregistered experiment changes them:

- BTCUSDC Spot
- 15m
- LONG only
- next-bar-open execution
- STOP_FIRST ambiguous-bar policy
- no leverage
- no margin
- no shorting

Base costs:
- fee = 10 bps per side
- slippage = 2 bps per side

Doubled-cost stress:
- fee = 20 bps per side
- slippage = 4 bps per side

Never silently change execution, accounting, risk, cost, stop, target,
hold, or cooldown semantics.

When filtering a frozen candidate schedule:
- use the frozen candidate schedule directly
- a filter may remove candidates only unless explicitly specified
- never re-evaluate the parent strategy against divergent filtered
  account state

---

## Research Method

Preferred sequence:

1. data/source validation
2. integrity validation
3. diagnostic/discovery
4. preregistration
5. commit/tag frozen preregistration
6. replay / independent validation
7. forensic audit if needed
8. synthesis
9. CLOSE or CONTINUE decision

Do not:
- reverse failed rules post hoc
- search alternative thresholds after seeing outcomes
- change horizons after seeing results
- mine features and present them as confirmatory evidence
- reuse consumed outcomes as independent validation
- optimize until something becomes profitable

Discovery evidence and independent validation evidence must remain
explicitly separated.

Independent validation must be genuinely outcome-unseen.

---

## Current Research State

V6:
- small positive frictionless edge
- negative after realistic costs

V7 Spot aggTrades:
- exhausted / NOT_SUPPORTED

V8 derivatives context:
- H0 funding NOT_SUPPORTED
- H1 open-interest expansion NOT_SUPPORTED
- discovery WEAK_OR_UNSTABLE_SIGNAL
- strongest Mark/Index premium feature failed independent validation
- derivatives branch CLOSED

V9:
- prospective BTCUSDC Spot L2 information discovery completed
- classification: STABLE_L2_INFORMATION
- V9 data is CONSUMED research evidence, not fresh independent validation
- public REST snapshot + WebSocket diff-depth collector validated
- deterministic local order-book reconstruction validated
- deterministic L2 feature extraction validated
- standalone short-horizon taker economics were insufficient to justify a trading strategy
- no validated profitable trading strategy

V10:
- cost-aware and depth-aware execution research foundation exists
- public Spot L2 + aggTrade collector exists
- combined L2 + aggTrade 300s economic discovery completed and SEALED
- preregistration: a296e5ed304640d7; classification: NO_STABLE_COMBINED_SIGNAL
- forensic audit PASS; no INVALIDATING_ISSUE identified
- all 7 information gates failed; combined signal did not improve on L2-only
- tail and anchor coverage passed; tail base-net and anchor base-net failed
- this economic discovery hypothesis / research branch is CLOSED
- acquisition protocol 8326791b411c27b5 supplied 8 sessions / 24 hours / 4 UTC start dates
- V10 discovery data is CONSUMED and must never be reused as fresh validation
- blind holdout remains LOCKED
- no advancement of this hypothesis to strategy validation, paper trading, or live trading
- no profitable strategy has been validated
- immutable closure: research/v10_microstructure_economic_discovery/a296e5ed304640d7/synthesis.json

Do not reopen a CLOSED research branch without explicit justification.

---

## V9 L2 Data Rules

Use Binance PUBLIC Spot market data only.

Collector requirements:
- REST depth snapshot bootstrap
- WebSocket diff-depth updates
- correct snapshot/update bridging
- sequence-gap detection
- resync after integrity failure
- deterministic reconstruction
- append-only raw persistence
- UTC timestamps
- Decimal-safe price/quantity handling

Never silently continue an invalid local order book.

Raw events are the source of truth.

Derived features must be reproducible from persisted raw events.

---

## V9 Session Eligibility

An ACTIVE / currently-writing L2 session must never be used as research
input.

Research data must come only from CLOSED immutable sessions.

Before a session becomes research-eligible, require:
- raw file closed
- deterministic replay succeeds
- sequence gaps = 0 unless explicitly classified otherwise
- invalid events = 0
- crossed/invalid reconstructed book states = 0
- feature reconstruction integrity passes
- deterministic hashes recorded

Interrupted sessions may be used only after successful offline recovery
and explicit PARTIAL/RECOVERED classification.

For prospective experiments:
- honor the preregistered cutoff exactly
- pre-cutoff sessions are engineering/discovery-only when specified
- never allow a session that violates the frozen eligibility rule

---

## V9 Feature Semantics

Current L2 feature foundation may include:

- best bid / ask
- mid price
- spread / spread bps
- top-N bid/ask depth
- depth imbalance
- microprice
- microprice displacement
- depth concentration
- bid/ask depth added
- bid/ask depth removed
- update intensity

Do not call removed depth "cancellation" unless execution-vs-cancel
causality is actually identifiable.

Feature buckets must use exact causal UTC boundaries.

Partial buckets must remain explicitly marked partial.

---

## Live Trading / Orders

Research code must not place real orders unless a future dedicated live
execution milestone explicitly authorizes it.

Never:
- enable withdrawals
- expose secrets
- commit API keys
- place real orders during research
- mix research logic directly with exchange credentials

Future live execution must use separate components for:
- exchange adapter
- execution state machine
- order reconciliation
- position reconciliation
- persistent state recovery
- risk engine
- stale-data guard
- duplicate-order guard
- kill switch

Shadow/paper mode must precede real-money execution.

---

## Repository Hygiene

Do not commit:

- data/
- reports/
- .venv/
- pytest cache/runtime directories
- raw market archives
- generated ZIP files
- local databases unless explicitly versioned
- .env files
- API keys/secrets

Prefer surgical changes.

Do not use `git add .` when generated/raw files may exist.

Do not commit unless explicitly instructed.

---

## Testing

Run focused tests first.

Run the full suite when appropriate.

On Windows use a fresh pytest basetemp and disable cacheprovider:

$run = ".pytest_runtime\run_$((Get-Date).ToString('yyyyMMdd_HHmmss'))"
New-Item -ItemType Directory -Force $run | Out-Null
.\.venv\Scripts\python.exe -m pytest -ra --basetemp="$run\basetemp" -p no:cacheprovider

Tests must not depend on live network access unless the task explicitly
requests a bounded integration/smoke validation.

Prefer structural/deterministic correctness tests over fragile
wall-clock performance assertions.

---

## Data / Numeric Integrity

Prefer:
- Decimal for financial quantities where existing architecture uses it
- timezone-aware UTC timestamps
- deterministic serialization
- deterministic hashes/manifests
- explicit missing values
- explicit source timestamps

Do not silently coerce malformed source data into valid observations.

Checksums and source integrity must be verified when official archives
provide them.

---

## Performance

Avoid repeated full-history scans inside per-event/per-candle loops.

Prefer:
- streaming
- monotonic cursors
- indexed lookup
- bisect / merge-asof style causal lookup
- bounded memory
- reusable immutable snapshots where semantics permit

Optimization must preserve exact reference semantics.

---

## Codex Work Style

Before editing:
- inspect relevant existing code
- reuse existing helpers/models
- understand frozen manifests and invariants

During implementation:
- keep changes minimal
- preserve determinism
- preserve causal semantics
- avoid unnecessary abstractions
- avoid unrelated refactors

If an assumption is uncertain:
inspect the code/data/manifests instead of guessing.

Final response should be concise and include only:
- what changed
- important integrity/result metrics
- focused/full test result
- any limitation or unresolved issue

Do not produce long narrative summaries unless explicitly requested.
