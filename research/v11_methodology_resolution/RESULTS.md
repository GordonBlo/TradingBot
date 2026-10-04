# V11 methodology resolution result

**METHODOLOGY_UNRESOLVED. No primary method selected; no V11 preregistration.**

This completed synthetic-only study preserves the earlier engineering foundation and readiness result. It does not evaluate a trading filter, establish economic performance, or use consumed candidate returns to choose a method. The sole prospective rule remains prediction > 0; fabricated score distributions vary retention as sensitivity assumptions, not alternative filter thresholds.

## Design and acceptance

The pre-execution [specification](STUDY_SPEC.md) binds exactly three candidates and eleven null families, seven rate/retention profiles, and 8,192 independent replications per cell. At 728 days this is 630,784 fabricated datasets and 1,892,352 method evaluations across 231 cells. All methods use paired calendar-date contributions and counts, including zero dates. All four Monte Carlo batches completed; no methods, scenarios, seeds, tolerances or bandwidths were added after results.

Nominal error is 5%. Entire simultaneous exact Clopper-Pearson intervals must fall inside 3.5%–6.5% error and 93.5%–96.5% coverage; failure upper limits must be at most 1%. Bonferroni allocation covers all 5,544 potential metric intervals across six durations. Bootstrap split-draw disagreement upper limits must be at most 2%. The ±1.5 percentage-point allowance is a declared calibration tolerance, not proof of exact 5% validity. Failures contribute neither rejection nor coverage, with the full replication denominator retained.

## Results

| Method | Failed scenario cells | Maximum null rejection | Minimum coverage | Maximum failure |
|---|---:|---:|---:|---:|
| CBB_LONG | 77/77 | 17.41% | 82.52% | 2.36% |
| HAC_LONG | 76/77 | 16.82% | 83.18% | 2.36% |
| SELF_NORMALIZED | 54/77 | 10.28% | 89.72% | 2.36% |

All maximum rejection rates occur under AR56 at the low rate 25/90, retention .50. Simultaneous Monte Carlo intervals for those rates are:

- CBB_LONG: 15.60%–19.33%; four batch rates 18.12%, 17.09%, 17.48%, 16.94%.
- HAC_LONG: 15.04%–18.72%; four batch rates 17.14%, 16.99%, 16.89%, 16.26%.
- SELF_NORMALIZED: 8.85%–11.84%; four batch rates 10.45%, 10.30%, 10.30%, 10.06%.

These worst distortions exceed the allowed ceiling even at the lower simultaneous confidence limits. They are not explained by Monte Carlo noise. A failed closeness screen elsewhere can instead mean insufficient calibration precision or conservatism; it is not automatically proof of inflation in that particular cell.

At base rate 443/990 and retention .50:

| Null family | CBB_LONG rejection | HAC_LONG rejection | SELF_NORMALIZED rejection |
|---|---:|---:|---:|
| iid_candidate | 7.75% | 7.90% | 4.77% |
| date_shared | 7.59% | 7.80% | 4.70% |
| clustered_iid | 8.33% | 8.69% | 5.29% |
| ar7 | 9.57% | 9.51% | 5.83% |
| ar28 | 12.56% | 12.32% | 7.04% |
| skew_date | 6.20% | 6.41% | 3.86% |
| t3_date | 7.70% | 7.79% | 4.81% |
| ar56 | 16.32% | 15.87% | 9.40% |
| ar28_skew | 11.57% | 11.60% | 6.65% |
| ar28_t3 | 12.57% | 12.34% | 7.79% |
| ar28_persistent_filter | 10.83% | 11.46% | 6.20% |

Full per-cell coverage, exact ordinary/simultaneous Monte Carlo intervals, failures, batch stability, and count/date quantiles are in [anchor_results.json](../../reports/v11_methodology_resolution/calibration/anchor_results.json). Hash-bound per-replication bounds and coverage counts were verified independently against all 231 aggregates without resimulating.

## Sparsity, retention, dependence and numerical stability

- Under AR28, self-normalized rejection ranges from 6.41% to 7.85% across base-rate retention .25/.40/.50/.75/.90. The low-rate .50-retention case is 8.04%. Higher retention does not certify validity, and no retention floor is selected.
- Under AR56, self-normalized rejection is 9.40% at base-rate .50 retention, versus 10.28% at the low rate. The low-rate scenario averages 201.94 parents on 169.65 dates and 100.93 accepted parents on 92.48 dates. These are synthetic sensitivities, not rate forecasts or sufficient sample floors.
- Persistent acceptance at .90 retention produces 193/8,192 unavailable runs (2.36%) for every method. All 193 have every parent accepted, hence zero paired improvement and a degenerate normalizer; these are explicit failures, not discarded observations or successful economic evidence.
- Maximum CBB split-draw decision disagreement is 1.34%; its simultaneous upper bound is 1.9971%, within the declared 2% diagnostic limit. Primary decisions always use all 1,999 draws. This numerical check does not rescue its size failure.

The SELF_NORMALIZED signed limiting critical value is 5.3356399234576815. Its 262,144-path quantile Monte Carlo interval is [5.303268250471789, 5.368318172102008]. Paired 512/2,048-grid difference is 0.00311%; half-sample difference is 0.31321%. Both pass the predeclared 1% numerical check. Thus the observed serial-dependence failures cannot be dismissed as a failed critical-value computation.

## Methodological assessment

CBB_LONG and HAC_LONG use L(T)=ceil(112*(T/728)^(1/3)), motivated before simulation by four AR28 e-folding lengths. At the tested endpoint, the long bandwidth also consumes a substantial fraction of the available calendar interval. Neither a long block nor a normal HAC cutoff solves finite-sample distortion. SELF_NORMALIZED reduces the distortion substantially but fails the required persistent and sparse panels. These are rejections of these exact finite-sample candidates under the declared standard, not a theorem that all dependent-data inference is impossible.

The self-normalized influence construction has a signed Brownian bridge limit under appropriate stationary functional-CLT conditions; it does not assume independent date signs. See [Shao (2010)](https://arxiv.org/pdf/1005.2137) and [Kiefer–Vogelsang (2005)](https://www.ssc.wisc.edu/~bhansen/718/KieferVogelsang.pdf). The ratio uses estimated joint count/contribution influence, not an active-date average. Stationary bootstrap and dependent wild approaches remain bandwidth-dependent alternatives; independent signs or permutation of deterministic filter labels lack the required symmetry/exchangeability. These exclusions were documented before calibration, and none was added afterward.

The panel covers stationary finite-variance nulls, not every filter/outcome dependency, structural break, or long-memory process. Student-t(3) has finite variance; no claim extends to infinite-variance returns. No power, economic effect, return forecast, or strategy profitability was estimated.

## Duration and gate consequences

No method passed the required 728-day anchor. Per the predeclared conditional design, the 180/270/365/448/560-day extension was NOT_TRIGGERED. This does not assert empirical invalidity at unrun durations, nor permit choosing one to escape the failed anchor. No prospective duration is selected.

No primary Delta inference gate can be frozen from this study. Positive base-net accepted expectancy, positive opportunity improvement, meaningful accepted exposure, date/temporal diversity and risk reporting remain requirements for any future defensible protocol, but their numerical floors, retention constraints, duration and effect thresholds remain unfrozen. No advance to preregistration, collection, strategy validation, paper trading or live trading follows from this result. Further methodology would require a separate prespecified study, not an additional method or relaxed gate in this completed comparison.

## Tests and integrity

Focused: 41 passed. Full pytest: 1,228 passed; one existing cache_dir warning when cacheprovider is disabled. Both use fresh Windows basetemp directories and -p no:cacheprovider. The first sandboxed focused run hit Windows temporary-directory access restrictions; the authorized external-sandbox run completed successfully. Ruff passed.

All 387 pre-existing protected files are byte-identical, including engineering source, frozen manifests, V9 bundle, V9/V10 artifacts, authorization and V10 closure. All 81 calibration artifacts in the hash manifest and all 231 saved-bound aggregates verify. Git HEAD and index are unchanged; no tracked changes or staging. V9/V10 remain CONSUMED, V10 CLOSED, holdout LOCKED and unaccessed by this task. No real V11 evaluation, preregistration, collection, orders or commit. These statements describe this task and its commands, not an OS-wide access-log audit.

- [Machine-readable result](RESULT.json)
- [Calibration binding](../../reports/v11_methodology_resolution/calibration/study_binding.json)
- [Calibration artifact hashes](../../reports/v11_methodology_resolution/calibration/artifact_hashes.json)
- [Integrity verification](../../reports/v11_methodology_resolution/integrity_after.json)
