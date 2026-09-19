# V10 microstructure economic discovery preregistration

Frozen ID: `a296e5ed304640d7`.
Definition SHA-256: `a296e5ed304640d706916cc8e562cedc6febf561b20b0c8a99d59ac30b275a33`.
The bound READY dataset has eight sessions, 24 hours and four UTC start dates
(2026-09-13 through 2026-09-16). This is a protocol record, not a diagnostic result.

This protocol freezes a **DISCOVERY ONLY** diagnostic before its first outcome
analysis. It does not authorize execution. The immutable content-addressed manifest
under `research/v10_microstructure_economic_discovery/<id>/manifest.json` is
authoritative. It binds the eight current acquisition-eligible sessions, all five
files per session, the non-predictive READY report, acquisition preregistration
`8326791b411c27b5`, source commit and reference-source hashes. No future session may
be substituted or added. At authorized analysis start these sessions become
CONSUMED discovery evidence, never independent strategy confirmation.

## Frozen comparison

Use 13 L2 features versus the same 13 plus 21 trade-flow features and six causal
interactions. Levels are 1/5/10/20; flow windows are 1/5/30 seconds. Exact
buyer-is-maker semantics determine aggressor sign. Duplicate aggregate records
do not create new flow. Counts refer to aggregate records, not their underlying
trade count. Removed depth is not labelled cancellation.
Absolute book-depth sums are intermediate inputs to imbalance/concentration,
not extra standalone predictors outside the requested families.

Use a causal one-second grid with a 30-second warmup, bounded stale-book age and
no cross-session carry. Fixed horizons are 5/30/60/300 seconds; **300 seconds alone
classifies**. Both feature sets use Ridge alpha 1.0, training-only preprocessing,
identical rows and seven expanding chronological whole-session test folds.
The constant training-target-mean baseline uses the same test rows.

Freeze 1,000 paired session-wise circular-shift null replicates, seed 20260916,
with circular offsets more than 600 rows from zero. Both models are refit for each
replicate. No IID significance test, model search, threshold search, sign reversal
or secondary-horizon rescue is permitted.

## Economic meaning and limits

Report simple mid returns and LONG ask-entry/bid-exit price response with fixed
100ms decision-to-quote latency. Base fee/slippage is 10/2 bps per side; stress is
20/4. Exact Decimal accounting separates mid movement, spread, adverse slippage,
fees and net return; 24/48 bps are reference cost scales, not extra deductions.

These are **quote-price diagnostics, not sized fills or trades**. Historical
exchangeInfo/admission inputs are not in this dataset. Do not invent quantity
filters, liquidity, impact or order admission. Depth capacity is NOT_ESTIMATED.
Even a positive classification cannot establish profitability. Future execution
research must separately freeze sizing, depth sweeps, exchange constraints,
inventory/risk and realistic fill assumptions using genuinely new sessions.

Quantiles are 50/75/90/95/97.5/99 percent. Bucket/tail boundaries come only from
training predictions; pooled OOS quantiles describe scale, never choose rules.
Report all predefined bins, conditional responses, monotonicity, persistence,
five-minute event clustering and turnover-demand proxies. Fixed economic
exceedance levels are 12/24/36/48 bps. Missing/negative results remain in the report.

## Primary classification, not a result

The information gate requires positive combined and incremental held-out
Spearman, combined MSE below both L2-only and constant baseline, positive combined
and incremental direction in at least 5/7 held-out sessions, and both circular-shift
p-values below 0.05.

Only the positive combined 300s **training-q95 tail** can pass the economic gate:
at least 100 pooled rows and 20 rows in each of at least five test sessions;
positive pooled and at least 5/7 per-session mean base-net response; and at least
20 fixed non-overlapping diagnostic anchors across at least four sessions with
positive pooled base-net response. Anchors address overlapping intervals, not
strategy execution. Stress and monotonicity are mandatory descriptive outputs,
not alternative classification routes.

- Information gate fails: `NO_STABLE_COMBINED_SIGNAL`.
- Information passes, economic gate fails: `INFORMATION_PRESENT_BUT_TOO_SMALL`.
- Both pass: `ECONOMIC_SIGNAL_PRESENT`.

The middle label includes insufficient prespecified economic coverage, not proof
that every possible signal is small. Any integrity failure stops the diagnostic
without a research classification. Seven test sessions and overlapping targets
limit inference; report per-session as well as pooled metrics. This preregistration
occurs after acquisition and cannot retroactively create confirmatory evidence.

## Safe modes only

```text
python -m src.cli.run_v10_economic_discovery --mode PLAN
python -m src.cli.run_v10_economic_discovery --mode VERIFY
```

PLAN reads only the manifest. VERIFY rechecks reference-source and artifact hashes;
it does not recompute predictive features, targets or outcomes. Neither creates or
rewrites a manifest. No EVALUATE/RUN mode or authorization bypass exists.

A later task must explicitly authorize this ID and implement/test the frozen
diagnostic. Before opening outcomes it must pass fresh acquisition readiness and
binding checks, reserve an exclusive persistent one-time run ledger, and freeze
the executor/runtime identity. Existing reservation/result means stop, including
after interruption. Reports are exclusive-create, ignored artifacts. No re-runs,
post-hoc amendments or changed settings are authorized by this record.
