# V10 GitHub artifact import

From the repository root, with Python 3.12 and the project requirements installed:

```powershell
.\.venv\Scripts\python.exe -m src.cli.import_v10_github_artifacts "C:\Downloads\v10-microstructure-overnight-123-attempt-1-20260913T120000000000Z.zip" "C:\Downloads\v10-microstructure-overnight-123-attempt-1-20260913T150000000000Z.zip"
```

Supply one or more downloaded overnight ZIP paths. The importer accepts the five
flat GitHub artifact files, or one canonical session-directory wrapper. It rejects
extra files, multiple sessions, traversal, links, encrypted entries, case collisions,
CI markers and CI_ONLY metadata before publishing anything.

Each archive must contain `depth.jsonl`, `aggtrades.jsonl`,
`session.manifest.json`, `closure.summary.json`, and `acquisition.validation.json`.
The frozen preregistration is `8326791b411c27b5`; start must be strictly after
`2026-09-12T08:30:00Z`. The original 10800-second overnight settings are required.
Raw identity/hashes, closure, provenance, integrity counters, causal contexts and
deterministic dual-stream replay are checked by the existing frozen acquisition
scanner. Stored validation/readiness must match fresh validation, ignoring only
the transported machine-local manifest path. Stored remote paths are never followed.

The collector's source commit must already be available in local Git history so
the frozen source hashes can be verified. The importer does not fetch anything.
Hashes check integrity, not publisher authentication: use ZIPs downloaded from
your trusted GitHub workflow; unsigned metadata is not a cryptographic attestation.

Validated bytes are published, unchanged, to:

```text
data/microstructure/v10/BTCUSDC/YYYY/MM/DD/<session_id>/
```

Staging is outside readiness discovery on the same filesystem. An exclusive
importer lock and atomic no-replace directory rename publish all five files at
once. Windows and Linux are supported; other platforms fail closed. A duplicate
is skipped only when the canonical local copy contains exactly the same five
files with byte-identical contents. Different or incomplete local files are never
overwritten. Each ZIP succeeds or fails independently; later failures do not undo
earlier successful imports. A leftover `.v10-import.lock` after interruption must
be inspected manually before retrying; the importer does not delete someone else's
lock or auto-recover interrupted staging. Staging requires space for one expanded
archive (maximum 64 GiB; metadata maximum 16 MiB per JSON file).

After processing all ZIPs, the CLI automatically runs only:

```text
python -m src.cli.run_v10_collection --mode READINESS
```

READINESS validates independent sessions with a spawned process pool, using at most
four logical CPUs and no more workers than sessions. Both CLIs accept `--workers N`;
`--workers 1` uses the original serial path. Explicit counts are capped by logical
CPUs and session count. The Python scanner API defaults to serial; pass `workers=None`
for automatic selection. Each worker runs the same raw checks, two replays, causal
context validation and artifact hashing. Results merge in sorted manifest-path order;
duplicate/overlap checks, final rehashing and readiness gates remain serial.
Scripts using the parallel Python API need a guarded `if __name__ == "__main__":`
entry point for process spawning; both CLIs already provide one.

Archives are still inspected, staged, validated and atomically published sequentially.
The optimized path is used by the final dataset READINESS scan. Progress such as
`[1/3] inspecting`, `[1/3] replaying`, `[1/3] validated` and `FINAL READINESS NOT_READY`
goes to stderr; stdout remains deterministic JSON. A process-pool failure aborts the
scan without publishing a readiness result. More workers increase memory use; use
`--workers 1` for constrained machines or serial debugging.

JSON output reports imported IDs, identical duplicates, rejected paths with reasons,
eligible sessions/hours/UTC start dates, and readiness status. Exit code 1 means an
artifact was rejected or readiness could not complete; 0 means the import completed
without rejection. `NOT_READY` is not an import failure: frozen data sufficiency
still requires 8 eligible sessions, 24 eligible hours and 3 UTC session-start dates.
The final canonical scan remains authoritative, including overlap detection across
separately imported sessions. Imported does not mean the aggregate dataset is READY.

No collection, predictive evaluation, strategy research, blind-holdout access or
V9-data reuse occurs. Acquisition readiness is not confirmatory strategy evidence.
