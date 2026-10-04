# V11 readiness study specification — synthetic only, NOT a preregistration

Specified before calibration execution, 2026-09-24. No real candidate outcomes,
prospective observations, V9 refitting, V10 execution, or parent selection.
Authoritative engineering inputs: research/v11_preparation and its immutable
V9 score bundle c2fb47fb459a97024ebedc12af0354d6c28792e2dce87c1207eba2ba5e4664e2.

## Synthetic families and counts

Durations: 180, 270, 365, 448, 560, 728 complete UTC dates. Metadata only:
443/990 parents/day, 372/990 active dates/day, maximum observed three/day.
Rate sensitivity: 25/90, 443/990, 63/90 per day, from existing count metadata;
these are scenarios, not rate confidence limits. Active dates have 1/2/3 parents
with P(3)=.02, P(2)=443/372-1-.04, remainder P(1). Active probability is
rate/(443/372). Date activity is independent or a stationary binary Markov chain
with .85 persistence (refresh from Bernoulli active probability otherwise).
This does not simulate actual V6 prices/triggers or guarantee its future rate.

Feasibility: 4,000 replications per duration/rate/clustering combination;
retention .25/.40/.50/.75/.90, candidate Bernoulli decisions independent of counts.
Joint insufficient-coverage rule under study is the earlier provisional floor:
N>=200, K>=100, >=120 parent dates, >=60 accepted dates, retention>=.40.
This is a sensitivity benchmark, not a claim those numbers ensure power.
Report each marginal failure, means, joint failure and Wilson 95% Monte Carlo
intervals. No population-forecast or power inference from these frequencies.

## Inference methods specified in advance

Unit: full UTC date vector (sum paired improvement, parent count); include all
inactive dates and preserve all same-date candidates. Statistic=sum improvement
/sum parent count. Methods: independent date bootstrap (length 1), circular
moving blocks of exactly 7 or 28 UTC dates. Draw ceil(D/L) starts uniformly from
all D dates, concatenate and truncate to D dates, including a correctly sized
last block. All candidates, acceptance and parent count travel together.
999 resamples, one-sided 95% basic lower limit = 2*observed - bootstrap 95th
percentile, NumPy linear quantile. Seed 20260925, PCG64 with named SeedSequence
components for duration/family/replicate/method; independent streams. No retries
or dropping draws: zero parent denominator or fewer than two active dates,
nonfinite input, or degenerate resampling distribution => INFERENCE_UNAVAILABLE.

400 Monte Carlo replications per duration/family/mean; base rate, retention .5.
Families: independent Gaussian candidate noise; shared Gaussian date noise;
stationary Gaussian date AR(1) with phi=exp(-1/7); same with phi=exp(-1/28);
independent centered exponential date noise (skewed). AR families also cluster
candidate dates; the others use independent date activity. Variance of the
Gaussian state is one; initialize at its stationary N(0,1) distribution.
Rejected parent R=-(mean+noise), so paired improvement=(1-accept)*(mean+noise).
Accepted parent R=.1+independent unit Gaussian noise; it cancels in Delta.
Means -0.1/0/+0.1 imply true opportunity improvement .5*mean. These are synthetic
effect sensitivities, not economically important effect thresholds or forecasts.
Report false rejection of Delta<=0, one-sided coverage, failure rates, CI width
and positive-effect rejection sensitivity. Report Monte Carlo intervals.

Diagnostic admissibility screen per duration/method: across every null family,
Wilson upper 95% bound on false rejection <=.08, Wilson lower 95% bound on
one-sided coverage >=.92, and no inference failures. These tolerances detect
material miscalibration with finite Monte Carlo precision; passing is NOT proof
of exact 5% size. A method can be selected only with a separate defensible
dependence argument for future data. No block-length search or extra methods
after seeing results. If assumptions cannot be justified, report C even if a
simulation screen passes. No endpoint is selected on positive-effect sensitivity.

Date-level sign flips are NOT a general test of mean Delta=0: the deterministic
filter does not randomize labels and skewed/dependent date contributions need
not have independent sign symmetry. No purported exact permutation result will
be produced. References: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
and https://www.stat.cmu.edu/~cshalizi/dst/20/lectures/16/lecture-16.html .

## Gate principles and endpoint decision

Keep integrity, positive base-net EA, positive Delta and calibrated directional
evidence. Meaningful exposure requires both absolute counts and an ex ante
retention/exposure floor; 40% in this study is a declared provisional operational
constraint, not a calibrated scientific threshold. Preserve candidate/date
diversity, report stress, mark-to-market downside and exposure. Do not restore
unsupported .05R/.02R/16-of-26 gates. Risk comparison is reported, not optimized.
No risk-return claim or minimum-important-effect threshold is manufactured.

A requires an honest complete protocol and one exact endpoint. B requires
otherwise resolved methodology plus demonstrably impractical duration. C takes
precedence if statistical assumptions or risk/accounting remain unresolved.
If C or B, no future V11 inference method or protocol is frozen by this study.

## Resume amendment, 2026-09-28 — before the first calibration run

The continuation explicitly requests fat-tailed sensitivity. Add one sixth
family, t3_date: independent date-shared Student t with 3 degrees of freedom,
divided by sqrt(3) for unit variance. Mean exists; fourth moment does not.
It uses independent candidate-date activity and the same three mean shifts.
All original scenarios, methods, seeds, draws and decision rules remain intact.
No calibration result existed when this amendment was made. Accepted outcomes
cancel algebraically and are not generated; the code studies Delta only, not
the power or Type-I error of the full conjunction of future success gates.
