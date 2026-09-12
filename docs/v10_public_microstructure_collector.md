# V10 public microstructure collector

This infrastructure records Binance public BTCUSDC Spot `depth@100ms` and
`aggTrade` streams concurrently. It does not authenticate, place orders,
compute signals, evaluate outcomes, or determine research eligibility.

## Bounded collection

Short smoke run:

```text
python -m src.cli.record_v10_microstructure --validate-seconds 120 --output-root data/ci/v10_microstructure_smoke --snapshot-limit 5000 --max-levels 5000
```

Multi-hour prospective run:

```text
python -m src.cli.record_v10_microstructure --duration-seconds 10800 --output-root data/microstructure/v10 --snapshot-limit 5000 --max-levels 5000
```

Both modes are explicitly bounded to at most 86,400 seconds, which makes a
3-hour invocation suitable for a GitHub-hosted job with a fresh runner and an
artifact upload step. Generated data remains under ignored `data/` paths.

## Session artifacts

Each session has one directory under
`<output-root>/BTCUSDC/YYYY/MM/DD/<session-id>/` containing:

- `depth.jsonl`: append-only REST snapshot, diff-depth, and resync-boundary raw records;
- `aggtrades.jsonl`: append-only public aggregate-trade raw records;
- `session.manifest.json`: immutable session identity, stream definitions, collector version, and source commit;
- `closure.summary.json`: immutable normal-closure status, raw hashes, live/replay counters, and deterministic replay result.

Absence of `closure.summary.json` classifies a stopped session as
`INTERRUPTED`. A normal closure is either `CLOSED_INTEGRITY_PASSED` or
`CLOSED_INTEGRITY_FAILED`; neither classification makes the data research
eligible. Eligibility remains explicitly `UNASSESSED` pending a separately
frozen V10 preregistration and readiness process.

## Causal synchronization

Offline replay orders records by the preserved local UTC receive timestamp.
When timestamps are equal, depth precedes aggTrade, then each source's
append-only record index breaks ties. Every trade context therefore receives
only the latest reconstructed book whose causal availability is no later than
that trade's receive timestamp. Missing earlier book state remains missing;
later snapshots and diffs never repair an earlier trade context.
