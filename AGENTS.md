# TradingBot Agent Instructions

## Project
Research-grade crypto trading system.

Primary traded market:
- BTCUSDC Spot
- 15m
- LONG only

Research and execution code must remain deterministic and causally valid.

## Critical Research Invariants

- Never access the locked blind holdout during research.
- Blind holdout:
  2025-08-01T00:00:00+00:00
  to
  2026-02-01T00:00:00+00:00

Holdout status must remain:
- LOCKED
- NOT LOADED
- NOT REVEALED
- NOT CONSUMED
- NOT EVALUATED

- Previously inspected research data is consumed data, never fresh OOS.
- Never use future information.
- Source timestamps used at signal time must be <= signal cutoff.
- Missing data stays missing unless an explicitly causal carry-forward rule exists.
- Never silently interpolate, future-fill, or nearest-match future observations.

## Frozen Execution Semantics

Unless a task explicitly introduces a new preregistered architecture:

- next-bar-open execution
- STOP_FIRST ambiguous-bar policy
- LONG only
- Spot only
- no leverage
- no margin
- no shorting

Base cost model:
- fee: 10 bps per side
- slippage: 2 bps per side

Doubled-cost stress:
- fee: 20 bps per side
- slippage: 4 bps per side

Do not change execution/risk/cost semantics silently.

## Research Method

Prefer:
1. data/integrity validation
2. diagnostic/discovery
3. preregistration
4. commit/tag preregistration
5. replay/validation
6. synthesis/close-or-continue decision

Do not:
- reverse failed rules post hoc
- mine thresholds after inspecting outcomes
- feature-mine consumed results without explicitly labeling discovery
- claim independent validation on previously outcome-inspected data

Independent validation must be genuinely outcome-unseen.

## Current Research State

V6:
- small frictionless edge
- negative after costs

V7 Spot aggTrades:
- exhausted / not supported

V8 derivatives context:
- funding H0 NOT_SUPPORTED
- OI H1 NOT_SUPPORTED
- discovery WEAK_OR_UNSTABLE_SIGNAL
- strongest mark/index premium feature failed independent H2 validation
- V8 derivatives branch CLOSED

V9:
- prospective BTCUSDC Spot L2 order-book research
- public REST snapshot + WebSocket diff depth
- own future historical L2 dataset
- no strategy signal yet

## V9 L2 Rules

- Binance PUBLIC market data only.
- No authentication required for market-data collector.
- Raw events must be persisted before derived reconstruction/features.
- Maintain correct snapshot + diff-depth sequence semantics.
- Detect sequence gaps.
- Resync on integrity failure.
- Never silently continue an invalid local order book.
- Preserve deterministic replay capability.

## Orders / Live Trading

Current research tasks must not place real orders unless explicitly requested in a future dedicated execution milestone.

Never:
- enable withdrawals
- expose API secrets
- commit API keys
- place real orders as part of research/testing

Live execution must eventually use a separate exchange adapter, risk engine, state reconciliation and kill switch.

## Repository Hygiene

Do not commit:
- data/
- reports/
- .venv/
- pytest runtime/cache
- raw market archives
- generated ZIPs
- API secrets / .env files

Prefer surgical changes over broad refactors.

Do not modify historical research artifacts merely to make them look cleaner.

## Tests

Windows pytest runs should use a fresh basetemp and disable cacheprovider.

Example:

$run = ".pytest_runtime\run_$((Get-Date).ToString('yyyyMMdd_HHmmss'))"
New-Item -ItemType Directory -Force $run | Out-Null
.\.venv\Scripts\python.exe -m pytest -ra --basetemp="$run\basetemp" -p no:cacheprovider

Run focused tests first, then full suite when appropriate.

## Codex Work Style

- Inspect existing infrastructure before creating parallel systems.
- Reuse existing models/helpers where appropriate.
- Keep changes minimal.
- Preserve determinism.
- Preserve causal semantics.
- Do not commit unless explicitly instructed.
- Report concise results, not long narratives.