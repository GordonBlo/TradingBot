# Synthetic calibration results — no real V11 outcomes

All scenarios/methods were specified before the first study run. NumPy 2.2.6, PCG64 seed 20260925. Feasibility: 4,000 replications per cell; inference: 400 replications, 999 resamples. Means and Monte Carlo frequencies below are sensitivity results, not forecasts or validated power.

## Base-rate / 50% retention coverage

| Days | Parent mean IID / clustered | Accepted mean IID / clustered | Parent dates IID / clustered | Accepted dates IID / clustered | Insufficient % IID / clustered |
|---:|---:|---:|---:|---:|---:|
| 180 | 80.48 / 80.79 | 40.20 / 40.45 | 67.60 / 67.88 | 36.79 / 37.02 | 100.00 / 100.00 |
| 270 | 120.85 / 120.11 | 60.31 / 60.01 | 101.43 / 100.96 | 55.25 / 54.96 | 100.00 / 99.15 |
| 365 | 163.29 / 164.16 | 81.61 / 82.23 | 137.17 / 137.92 | 74.70 / 75.30 | 99.78 / 84.88 |
| 448 | 200.84 / 198.47 | 100.44 / 99.30 | 168.52 / 166.71 | 91.98 / 90.93 | 60.02 / 56.23 |
| 560 | 250.99 / 250.43 | 125.54 / 125.22 | 210.69 / 210.20 | 114.94 / 114.66 | 0.80 / 17.93 |
| 728 | 325.45 / 326.97 | 162.69 / 163.30 | 273.30 / 274.70 | 149.00 / 149.57 | 0.00 / 1.20 |

Insufficiency uses the previously proposed provisional floors: 200 parents, 100 accepted, 120 parent dates, 60 accepted dates, >=40% retention. It does not establish the scientific sufficiency of these floors. All marginal failure rates and Wilson 95% Monte Carlo intervals are in feasibility.json. Zero simulated failures is not proof of zero probability.

## Retention sensitivity — base rate, clustered dates

| Days | 25% retained: insufficient % | 40% | 50% | 75% | 90% |
|---:|---:|---:|---:|---:|---:|
| 180 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 270 | 100.00 | 99.98 | 99.15 | 98.92 | 98.92 |
| 365 | 100.00 | 97.67 | 84.88 | 82.42 | 82.42 |
| 448 | 100.00 | 89.08 | 56.23 | 51.73 | 51.73 |
| 560 | 100.00 | 69.67 | 17.93 | 14.45 | 14.45 |
| 728 | 100.00 | 51.78 | 1.20 | 0.60 | 0.60 |

The 25% scenario deliberately tests anti-abstention failure. At true 40% retention, a hard realized >=40% floor alone produces approximately 50% failure even with ample N; this boundary behavior must not be mistaken for weak economics. Full low/base/high rate and independent/clustered timing sensitivities: 180 cells in feasibility.json.

## Null inference robustness

| Days | Block days | Worst null family | False-positive % | Monte Carlo 95% interval % | Minimum one-sided coverage % across families | Maximum failure % |
|---:|---:|---|---:|---:|---:|---:|
| 180 | 1 | ar28 | 34.75 | 30.25–39.54 | 65.25 | 0.25 |
| 180 | 7 | ar28 | 23.00 | 19.14–27.37 | 74.75 | 2.50 |
| 180 | 28 | ar28 | 19.75 | 16.14–23.93 | 71.50 | 8.75 |
| 270 | 1 | ar28 | 34.25 | 29.77–39.03 | 65.75 | 0.00 |
| 270 | 7 | ar28 | 25.75 | 21.71–30.25 | 73.75 | 0.50 |
| 270 | 28 | ar28 | 19.25 | 15.69–23.40 | 79.25 | 1.50 |
| 365 | 1 | ar28 | 34.25 | 29.77–39.03 | 65.75 | 0.00 |
| 365 | 7 | ar28 | 24.50 | 20.54–28.94 | 75.50 | 0.00 |
| 365 | 28 | ar28 | 17.75 | 14.32–21.80 | 82.25 | 0.00 |
| 448 | 1 | ar28 | 33.50 | 29.05–38.26 | 66.50 | 0.00 |
| 448 | 7 | ar28 | 24.50 | 20.54–28.94 | 75.50 | 0.00 |
| 448 | 28 | ar28 | 17.50 | 14.09–21.53 | 82.50 | 0.00 |
| 560 | 1 | ar28 | 38.75 | 34.10–43.61 | 61.25 | 0.00 |
| 560 | 7 | ar28 | 26.25 | 22.18–30.77 | 73.75 | 0.00 |
| 560 | 28 | ar28 | 18.00 | 14.55–22.06 | 82.00 | 0.00 |
| 728 | 1 | ar28 | 30.00 | 25.72–34.66 | 70.00 | 0.00 |
| 728 | 7 | ar28 | 21.25 | 17.52–25.52 | 78.75 | 0.00 |
| 728 | 28 | ar28 | 14.75 | 11.61–18.56 | 85.25 | 0.00 |

All 18 duration/method screens failed. Strong serial dependence remains materially anti-conservative even at 728 dates. IID-candidate null false-positive rates at 728 days were 5.75%, 5.50%, 5.75% for 1/7/28-day blocks: performance under independence does not justify use under dependent date observations.

Short clustered samples sometimes generated an empty-parent bootstrap sample. The entire inference then failed; no draws or replications were silently dropped. Coverage counts failures as noncoverage. One-sided coverage at zero equals one minus rejection minus failure; it is not an independent diagnostic.

Maximum split-draw rejection disagreement across all cells: 7.25%. This compares the first 499 and last 500 draws and measures Monte Carlo instability, not temporal robustness.

Full inference.json contains all 324 cells, including Gaussian, shared-date, AR7, AR28, centered exponential and t3 date families; negative, zero and positive fabricated means; rejection rates, coverage, lower-bound margins, failure rates and Monte Carlo intervals. Effect sensitivities are not optimized effect sizes. The full joint success gate was not simulated.

## Decision

**METHODOLOGY_UNRESOLVED.** No candidate method is selected. No prospective inference method, endpoint or V11 preregistration is frozen. Date sign flips are not generally valid for a deterministic filter with asymmetric or serially dependent paired contributions. A future method needs a defensible dependence argument and calibration; neither favorable IID simulations nor a larger calendar sample resolves the demonstrated failures.
