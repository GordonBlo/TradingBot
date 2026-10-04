# V11 statistical methodology resolution — specification before simulation

2026-09-28. SYNTHETIC METHODOLOGY STUDY ONLY; NOT A V11 PREREGISTRATION.
The previous readiness verdict remains METHODOLOGY_UNRESOLVED until this study
is complete. Preserve all previous artifacts, V6, the V9 score, prediction > 0,
V10 closure, and the locked holdout. No real candidate returns are inputs.

## Estimand and assumptions

For parent opportunity i, d_i = (a_i - 1) R_i, where a_i is the fixed filter's
0/1 acceptance and R_i is the parent's base-net return in reference risk units.
Thus accepted contributions cancel and rejected parents contribute -R_i.
Delta = E(Y_t)/E(N_t), estimated by sum(Y_t)/sum(N_t); Y_t is the sum of d_i
on UTC date t and N_t is the parent count. Retain every calendar date, including
zeros. Do not average active-date means or resample acceptance labels.

All three candidates require a stationary, short-memory joint date process,
positive finite intensity, finite variance, and a nondegenerate functional CLT
for Y_t - Delta N_t. None is distribution-free under arbitrary serial dependence,
nonstationary regimes, or an infinite-variance return process. A deterministic
filter is compatible with joint-process inference but supplies no randomization.
The estimated ratio influence is z_t = Y_t - Delta_hat N_t. Count randomness
must enter uncertainty estimation. A passing simulation is necessary, not proof
that the unknown future process satisfies these assumptions.

## Exactly three candidate methods, no additions after results

1. CBB_LONG: paired circular moving-block bootstrap of (Y_t,N_t), 1,999 draws.
   L(T) = ceil(112*(T/728)^(1/3)). At 728 days L=112: four times the prespecified
   28-day e-folding scale, with exp(-4) residual AR correlation. This motivates
   a stress candidate, not a claim that dependence ends at day 112. The growing,
   sublinear rule satisfies the usual block-growth direction asymptotically.
   Draw uniform starts, concatenate complete blocks and an independently drawn
   truncated last block to exactly T dates, wrapping circularly. One-sided 95%
   basic bound = 2*Delta_hat - linear_quantile(bootstrap ratios, .95). No redraws.
2. HAC_LONG: date-level Bartlett/Newey-West variance on z_t, bandwidth L(T),
   weights 1-k/L for k=1,...,L-1, autocovariances divided by T. Bound is
   Delta_hat - Phi^-1(.95)*sqrt(T*LRV)/sum(N). No degrees-of-freedom correction
   or data-chosen bandwidth. Its normal critical value is precisely what must
   be calibrated; long bandwidth alone is not presumed to repair size.
3. SELF_NORMALIZED: plug-in ratio influence cumulative sums S_t=sum_{j<=t}z_j;
   scale = sqrt(sum S_t^2 / T)/sum(N). Bound = Delta_hat - c*.scale, where c is
   the .95 quantile of B(1)/sqrt(integral_0^1 [B(r)-r B(1)]^2 dr).
   This is the influence-function self-normalized / full-bandwidth fixed-b
   construction. It uses a signed one-sided pivot, NOT a normal critical value
   or a two-sided 95% squared-pivot cutoff. No fitted dependence bandwidth.
   Ratio linearization needs a stable positive count mean. Finite sparse samples
   and high persistence can still distort this asymptotic approximation.

Critical-value quadrature: 262,144 independent Brownian paths, 2,048 equal grid
increments, seed stream (20260928,0), PCG64. Right-endpoint bridge-square Riemann
sum. Record paired 512-grid diagnostic (every fourth point), half-sample
quantiles, and distribution-free binomial order-statistic 95% quantile bounds.
Use the full 2,048-grid quantile irrespective of results. Relative grid and
half-sample differences must each be <=1%; otherwise SELF_NORMALIZED cannot be
selected. Do not increase paths, grids, or adjust a critical value in response
to scenario calibration. This simulates the analytical limiting law, not a
critical value fitted to synthetic return families.

Alternatives assessed but not added to the comparison:
- Stationary bootstrap: geometrically distributed block lengths are defensible
  under mixing, but still need a growing mean length and can attenuate persistent
  covariance. It does not remove the present assumption problem. CBB is the one
  representative block-bootstrap candidate; no mean-length search is permitted.
- Fixed-b Bartlett at b=1 is proportional to the mean influence self-normalizer;
  counting it again would duplicate a candidate. Normal critical values are not
  appropriate for that fixed-b statistic.
- Subsampling needs a prespecified growing window and enough approximately
  informative windows. It is not added as a window-search fallback.
- Independent date wild signs, exact sign tests, and label permutations are
  rejected here: independence/sign symmetry/exchangeability or randomized labels
  are absent, especially under skew and AR dependence. Dependent wild bootstrap
  can be valid with a dependence kernel/bandwidth, but is not a tuning-free or
  exact solution and is outside this small candidate set.

## Required fabricated null families (true Delta=0)

Reuse the previous count generator, not historical outcomes: mean base rate
443/990 per day, active-date multiplicity 1/2/3 with P(3)=.02 and
P(2)=443/372-1-.04. Active probability = rate/(443/372). Clustered activity
retains its previous state with probability .85 and otherwise refreshes from
that stationary Bernoulli distribution. This is sensitivity modeling, not a
V6 rate forecast. Synthetic latent U scores yield ACCEPT exactly when U is below
retention; equivalently prediction = retention-U > 0. This fabricates scores,
does not apply/refit the real frozen score, and does not randomize observed labels.

Required families: iid_candidate (independent timing and Gaussian candidate
contributions); date_shared (independent timing, common Gaussian date shock);
clustered_iid (clustered timing, independent Gaussian candidate shocks);
ar7 and ar28 (clustered timing, common stationary Gaussian date AR(1),
phi=exp(-1/7) or exp(-1/28)); skew_date (independent timing, Exp(1)-1 date shock);
t3_date (independent timing, Student-t(3)/sqrt(3) date shock).

Additional stresses specified NOW: ar56 (clustered, phi=exp(-1/56));
ar28_skew and ar28_t3 (same AR28 but centered-exponential or standardized-t3
innovations); ar28_persistent_filter (Gaussian AR28 outcome, plus independent
Gaussian AR28 latent score per date, threshold Phi^-1(retention), shared by that
date's candidates). Non-Gaussian AR starts use 2,048 discarded warmup dates,
then unit-variance innovation scaling sqrt(1-phi^2). These are stationary to
negligible initialization error, not a claim of exact finite burn-in stationarity.
All latent score/count innovations are independent of outcome innovations;
each candidate has d_i = rejected_i * zero-mean shock. This ensures the null
analytically even with clustered scores. It does not cover every possible
conditional filter/outcome dependency or nonstationary market regime.

Seven prespecified profiles for EVERY family: base rate with retentions
.25,.40,.50,.75,.90; low rate 25/90 and high rate 63/90 each at retention .50.
The seven profiles are a stress panel, not a full factorial. No exposure floors
are imposed or certified. Record parent/accepted/rejected counts and active dates.

## Replications, acceptance, and duration interaction

Stage 1: T=728, all 11 families x 7 profiles x 3 methods = 231 cells.
8,192 independent Monte Carlo replications per cell in four batches of 2,048.
All methods share each fabricated input; bootstrap streams are separate per
replication. PCG64 SeedSequence keys include seed 20260928, stage, duration,
family, profile, batch, replicate, and purpose; generator loop order is fixed.
No early abandonment of a scenario after failure. No repeated calibration seeds.

Stage 2 executes ONLY if at least one candidate passes stage 1: all stage-1
survivors, all families/profiles, at 180,270,365,448,560 days, same replications.
A method must pass EVERY duration to qualify; never select a favorable duration.
If none survives stage 1, record stage 2 NOT_TRIGGERED; shorter duration cannot
rescue a method rejected at the required 728-day anchor. This is not a statement
that every unrun short-duration method necessarily fails.

Nominal Type-I error .05; numerical equivalence band [.035,.065] declared before
results. One-sided coverage band [.935,.965]. This +/-1.5 percentage-point
tolerance bounds allowed size distortion to 30% of nominal; it is an explicit
engineering calibration tolerance, not proof of exact size .05. Conservative
near-never-reject methods also fail closeness. Failures <=.01. Require all entire
simultaneous exact Clopper-Pearson Monte Carlo intervals to lie inside the error
and coverage bands, and failure upper bounds <=.01. For multiplicity, allocate
.05/(6*77*3*4) to each two-sided interval, covering all potential durations,
methods, profiles/families, and four metrics (rejection, coverage, failure,
bootstrap split-draw decision disagreement). Bonferroni remains valid with
shared synthetic inputs. 8,192 gives ordinary nominal SE about .0024; the more
stringent simultaneous intervals distinguish the prior large distortions from
Monte Carlo noise without treating point estimates as truth.

At the null, coverage and rejection are complementary when inference exists;
these are not independent confirmations. Failures count as no rejection AND no
coverage, are never removed from the denominator, and have a separate gate.
Conditional-on-valid coverage is also descriptive. Inference is unavailable for
nonfinite values, malformed counts, fewer than two active dates, nonpositive
total count, degenerate/nonpositive normalization, or ANY empty/degenerate CBB
resample. No imputation, redraw, winsorization, or repaired covariance.

Report four batch rates and ordinary exact 95% intervals as Monte Carlo stability
diagnostics. CBB also compares first 999 vs last 1,000 draw decisions; its
simultaneous disagreement upper bound must be <=.02 for selection. The full
1,999 draws always determine its primary bound. Other methods have no bootstrap
draw noise. Persist per-replication bounds/failure flags and hashes so the report
can be audited without rerunning simulations.

Selection priority fixed now: SELF_NORMALIZED, CBB_LONG, HAC_LONG, conditional
on passing every required cell and methodological assumptions being defensible.
If all fail, selected_method=null and METHODOLOGY_UNRESOLVED. No new methods,
bandwidths, confidence levels, or relaxed tolerances in this study. No effect or
power simulations, duration selection, exposure floor, or V11 gate is frozen.

## Primary methodological references

- Shao (2010), corrected paper, equations (3)-(4), ratio/smooth-function and
  fixed-b discussion: https://arxiv.org/pdf/1005.2137
- Kiefer & Vogelsang (2005), bandwidth-dependent limits:
  https://www.ssc.wisc.edu/~bhansen/718/KieferVogelsang.pdf
- Politis & Romano (1994), stationary bootstrap construction and conditions:
  https://users.ssc.wisc.edu/~behansen/718/Politis%20Romano.pdf

Engineering, score, and research-state integrity are checked by byte hashes
against reports/v11_methodology_resolution/integrity_before.json. Only this new
study's source, tests, specification and synthetic outputs may be added.
