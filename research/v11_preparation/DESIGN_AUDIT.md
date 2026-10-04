# V11 preparation — not a preregistration

Status: PREPARATION ONLY / NOT READY TO PREREGISTER. No collection or V11
outcome evaluation is authorized by this artifact. Prepared 2026-09-24.

## Boundaries and provenance

BTCUSDC public Spot, LONG only; no leverage, margin, shorts or orders. V6 H0
`e1eef7bdd0c37ad4` is a failed economic benchmark, not a validated strategy.
V9 V2 `853870051677af08` is CONSUMED information-discovery evidence. V10
`a296e5ed304640d7` remains CLOSED / NO_STABLE_COMBINED_SIGNAL; its sealed and
closure artifacts are unchanged. The authoritative blind holdout stays LOCKED.
No V9/V10 observations can count as fresh V11 validation.

## Static score preparation

The eight original V9 V2 session identities/hashes come exclusively from its
sealed confirmatory_evaluation.json (SHA-256
298a57f5a5fc01f1fd896ebe338906d6c121baf499d04bca73c06cda9f5bc9c2).
The dedicated v9_static_score directory reserves one real fit before loading
training rows. An existing directory refuses another preparation. A failed
reservation is retained; no automatic retry is implemented.

Use all 86,127 original eligible primary rows, in original canonical session /
timestamp order. Reverify raw, closure and feature hashes and exact session
boundaries with the frozen loader. It also constructs the old sample structure's
auxiliary target fields, but only the original exact 30s target enters training;
no secondary model, metric, permutation, sign or threshold is evaluated.

The 15 ordered features, training medians, imputed means, population standard
deviations (zero scale -> zero), Ridge alpha=1, inverse solver, and unpenalized
training-target-mean intercept reuse the original implementation. An isolated
all-missing dummy reference row satisfies its nonempty held-out API without
entering training; there is no evaluation of real held-out rows. The exported
bundle binds coefficients, preprocessing, training identities and hashes,
target definition, source byte hashes, Git HEAD and Python/runtime identity.
SHA-256 binds both model payload and complete bundle bytes.

Preparation completed: one real fit, 86,127 training rows, eight bound sessions.
Bundle: v9_static_score/score_bundle.json. Complete-file SHA-256:
`c2fb47fb459a97024ebedc12af0354d6c28792e2dce87c1207eba2ba5e4664e2`.
The deterministic identity test pins these bytes. No real fit is repeated by tests.

The sole prospective rule remains prediction > 0. A missing usable score is an
explicit operational rejection. No recalibration or post-outcome sign reversal.
Synthetic reference tests require exact float equality with the frozen solver,
including JSON round-trip, missing values and zero-variance columns.

## Research execution foundation

src/backtest/v11_entry_filter.py is a pure in-memory adapter; it has no collector,
network client, production executor, research classifier, or order interface.
It reuses V10's generic depth/Decimal accounting helpers, not V10 discovery.

Input is a pre-existing V6-derived parent schedule, not all unconstrained raw
signals. This module does not claim to implement the future parent materializer
or prove a supplied candidate satisfied every EMA condition. The future
materializer must bind the unchanged V6 generator/config, complete causal OHLCV,
quantity/risk/cash policy and unfiltered inventory state before collection.
Changing fills can change that reference inventory path; never describe the new
path as a byte-identical historical V6 replay.

Each candidate supplies its finalized aligned 15m candle, receipt time, causal
completed-4h ATR and immutable quantity. Receipt must be within 1s after close.
Entry uses the earliest received depth observation at/after receipt +100ms,
with at most 1s additional wait and 1s source age. It must remain in the next
15m bar. These are explicit proposed engineering bounds, not inferred alpha.
This latency-aware fill rule requires a future V11 execution amendment; it
does not modify frozen V6's NEXT_BAR_OPEN convention.

Full requested quantity must be executable against asks; no resizing, partial
fills or silent omission. Exchange quantity/notional constraints and available
unfiltered cash are checked. Geometry is max(ATR14_4h, base entry fill * .0096),
stop -1R, target +2R, maximum 96 entry-bar intervals. A fully executable bid
VWAP triggers protection; fees/additional slippage are applied at fills.
After a trigger, liquidation is at the first eligible observation >= trigger
+100ms. Adverse movement is realized even after a target trigger. Stress
reprices those same observations/quantities and does not regenerate stops or
candidate schedules. It is sensitivity accounting, not a second strategy run.

No pre-entry candle extremes are used. Ordered post-entry observations decide
the first trigger; a later stop never overwrites an earlier target. Unordered
or tied observation times fail rather than fabricating an ambiguous OHLC path.
Thus this adapter has no unresolved stop/target ambiguity to which STOP_FIRST
needs to be applied. A future coarse-bar fallback would need explicit separate
semantics and tests and is not implemented. Missing protective observations
for >1s fail. A collector must certify gap-free reconstruction upstream; sequence
IDs alone cannot prove this because a depth diff can span several update IDs.

V6's adapter observes an exit at its candle close, sets bars_since_exit=0, blocks
the next four closes and permits the fifth later close. The new reference checks
that identical bar-index cooldown: earliest next signal close is the exit bar's
open +6*15m. A rejected trade still occupies the parent position/cooldown.

Base fee/slippage is 10/2 bps per side, stress 20/4. LONG buys ask depth, sells
bid depth. Net PnL is exit proceeds minus entry cost minus both fees; embedded
spread and depth impact are attributed, never subtracted a second time.

Filtering binds the reference ledger hash and requires one decision per parent
in exactly the same order. Accepted rows share the original trade/quantity;
rejections contribute zero and remain in the denominator. There is no callback
to a strategy or filtered-account sizing; released cash cannot create a trade.
No parent/subset changes are permitted after outcomes. This module provides no
statistical significance or successful-V11 classification.

## Candidate-rate feasibility (not power)

Only consumed schedule metadata is used: 443 candidates / 990 evaluation days,
372 distinct candidate UTC dates, 11 windows with 25–63 candidates per window,
maximum three candidates per UTC date. No candidate-return/score relationship
was evaluated. Machine-readable counts are in
reports/v11_preparation/feasibility_metadata.json.

| Fixed evaluation days | Candidates at historical mean rate |
| --- | ---: |
| 180 | 80.5 |
| 270 | 120.8 |
| 365 | 163.3 |
| 448 | 200.5 |
| 560 | 250.6 |
| 728 | 325.8 |

These are arithmetic projections, not forecasts or confidence bounds. For 100
accepted candidates, 50% retention needs 200 parent candidates (~447 days);
40% retention needs 250 (~559 days). Neither retention nor future date coverage
is known. The earlier 200-parent/100-accepted requirements already imply at
least 50% retention when N=200; their interaction with a 40% floor was obscured.
The historical rate of candidate dates is 372/990, but filtering can concentrate
acceptance across dates, so it cannot guarantee the 60 accepted-date floor.

728 days is not a power calculation. With the mean-rate projection, each 28-day
block contains only ~12.5 parent candidates and perhaps 5–6 accepted candidates.
Counts varied substantially across historical windows. Sixteen positive blocks
is neither a calibrated sign test nor a proven regime-diversity requirement.
Calendar diversity alone cannot guarantee distinct market regimes.

Conclusion: NO shorter months-long confirmatory design is established as
defensible from these metadata. Nor is the 728-day design established as
adequately powered. Keeping the proposed 200/100 coverage floors makes a
six-to-nine-month completion implausible at the observed mean rate. A 365-day
interval also falls below 200 expected parents. Do not lower sample floors,
switch to unconstrained V6 signals, or choose another parent simply to finish.

There is no identifiable "shortest defensible" duration without a justified
minimum important effect, dependence/variance assumptions and an explicit
acceptable probability of insufficient coverage. A future power exercise may
use prespecified synthetic sensitivity scenarios (not V11 outcomes) and must
report its assumptions. No power or guaranteed-duration claim is made here.

## Success-gate audit before freezing

Let r_i be base net R on the reference initial risk, a_i the fixed acceptance,
N all parents, K accepted, EP=sum(r_i)/N, EA=sum(a_i*r_i)/K, and
Delta=sum((a_i-1)*r_i)/N. Delta is the primary opportunity-level improvement;
abstention is zero return and never removes a parent from its denominator.

| Earlier proposal | Assessment / disposition |
| --- | --- |
| Retention 40–80% | Anti-abstention principle is sound; exact limits lack a cost/power rationale. The upper bound would reject a useful filter retaining >80% without an economic reason. Do not freeze these limits. Require absolute accepted/rejected coverage and a justified minimum exposure floor later; no new numeric optimum selected. |
| EA > 0 | Retain: selecting a still-negative schedule is not useful after costs. A positive point estimate alone is not evidence of established profitability. |
| EA-EP >= .05R | No documented minimum-important-effect basis. Remove from proposed primary success criteria pending an ex ante rationale; report descriptively. |
| Delta >= .02R | Same problem. Positive opportunity improvement is essential; .02R has not been economically justified. No replacement effect size is invented. |
| One-sided 95% lower bound for Delta > 0 | Retain the directional evidence principle. Seven-day blocks, PCG64 seed and 10,000 draws make computation reproducible, not inference valid. Block length and sparse-count behavior need prespecified synthetic calibration; the current bootstrap is not implemented/endorsed as a valid final test. |
| Positive Delta in >=16/26 blocks | Arbitrary and noisy with sparse candidates. Report the fixed block/date profile; do not freeze the 16/26 cutoff. A justified consistency gate remains an unresolved prerequisite. |
| Filtered maximum drawdown <= parent | Useful descriptive risk comparison, but path/position/exposure conventions matter and reduced exposure alone reduces drawdown. A zero-tolerance gate is not justified; define a risk budget and continuous mark-to-market accounting before proposing a binding risk gate. |

Additional inference issue: Delta=-sum(rejected r_i)/N. A positive Delta can
reflect avoiding losses in a generally losing parent, even if the score does
not identify economically good entries. EA>0, substantial accepted coverage,
date coverage and exposure reporting therefore remain necessary. A matched-
retention randomized diagnostic, if ever desired, would require advance
specification and must not become a rescue analysis; none is run here.

## Proposed next V11 status

Keep exactly one fixed V6 parent and one static V9 positive-score rule. Do not
freeze a duration or numerical success gate now. Before any future cutoff,
resolve materializer/collector integration, effect-size rationale, exposure
floor, temporal consistency, risk accounting and statistical calibration using
engineering/synthetic inputs. If months-only completion is mandatory, this
parent is presently infeasible for the proposed coverage requirements; stop
instead of weakening the scientific claim.

A future approved protocol must fix its endpoint before collection, start all
new sessions strictly after its future cutoff, include post-cutoff warmup and
terminal exit observation, use only closed immutable sessions, and evaluate
once. At the fixed endpoint insufficient count/date/integrity coverage means
INSUFFICIENT_PROSPECTIVE_EVIDENCE (or INTEGRITY_FAILURE), never a discretionary
extension. No paper/live advancement is authorized by this preparation or by
discovery evidence alone. Passive/maker research remains separate because queue
position, fills, partial fills and cancellation/latency uncertainty are not
identified by aggregate depth snapshots.
