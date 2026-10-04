"""Prespecified synthetic sensitivity study; never loads market data or models."""
from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

DURATIONS = (180, 270, 365, 448, 560, 728)
RATES = (25/90, 443/990, 63/90)
RETENTIONS = (.25, .4, .5, .75, .9)
FAMILIES = ('iid_candidate', 'date_shared', 'ar7', 'ar28', 'skew_date', 't3_date')
LENGTHS = (1, 7, 28)
MEANS = (-.1, 0., .1)
SEED = 20260925
DRAWS = 999
REPLICATIONS = 400
FEASIBILITY_REPLICATIONS = 4000


def stream(*keys):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, *keys])))


def wilson(successes, total):
    if total <= 0:
        raise ValueError('positive Monte Carlo denominator required')
    p, z = successes/total, 1.959963984540054
    den = 1 + z*z/total
    center = (p + z*z/(2*total))/den
    radius = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [max(0., center-radius), min(1., center+radius)]


def counts(rng, replicas, days, rate, clustered):
    q = rate / (443/372)
    if not 0 < q < 1:
        raise ValueError('unsupported synthetic rate')
    active = np.empty((replicas, days), dtype=bool)
    active[:, 0] = rng.random(replicas) < q
    for t in range(1, days):
        refresh = rng.random(replicas) < q
        active[:, t] = np.where(rng.random(replicas) < .85, active[:, t-1], refresh) if clustered else refresh
    u = rng.random((replicas, days))
    p2 = 443/372 - 1 - .04
    multiplicity = np.where(u < .02, 3, np.where(u < .02 + p2, 2, 1))
    return active * multiplicity


def circular_sums(values, length):
    values = np.asarray(values)
    if not 1 <= length <= len(values):
        raise ValueError('invalid block length')
    extended = np.concatenate((values, values[:length-1]))
    cumulative = np.concatenate(([0.], np.cumsum(extended)))
    return cumulative[length:] - cumulative[:-length]


def resample_totals(vectors, length, rng, draws=DRAWS):
    """Paired whole-date vectors; truncated final block has its own exact sums."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or not np.all(np.isfinite(vectors)):
        raise ValueError('finite date vectors required')
    days = vectors.shape[1]
    if not 1 <= length <= days or draws < 2:
        raise ValueError('invalid bootstrap design')
    complete, remaining = divmod(days, length)
    starts = rng.integers(days, size=(draws, complete))
    totals = np.stack([circular_sums(row, length)[starts].sum(axis=1) for row in vectors])
    if remaining:
        last = rng.integers(days, size=draws)
        totals += np.stack([circular_sums(row, remaining)[last] for row in vectors])
    return totals


def lower_bound(observed, draws):
    if not np.isfinite(observed) or not np.all(np.isfinite(draws)) or np.ptp(draws) == 0:
        raise ValueError('INFERENCE_UNAVAILABLE: nonfinite or degenerate distribution')
    return float(2*observed - np.quantile(draws, .95, method='linear'))


def paired_lower_bound(improvement, parent_counts, *, length, seed=SEED, draws=DRAWS):
    """Diagnostic candidate method only. No scientifically selected V11 method."""
    y, n = np.asarray(improvement, dtype=float), np.asarray(parent_counts, dtype=float)
    if (y.shape != n.shape or y.ndim != 1 or np.any(n < 0) or np.any(n != np.floor(n))
            or np.count_nonzero(n) < 2 or np.any(y[n == 0] != 0)):
        raise ValueError('INFERENCE_UNAVAILABLE: invalid or sparse paired dates')
    totals = resample_totals([y, n], length, np.random.Generator(np.random.PCG64(seed)), draws)
    if np.any(totals[1] == 0):
        raise ValueError('INFERENCE_UNAVAILABLE: empty resample; no redraw')
    return lower_bound(y.sum()/n.sum(), totals[0]/totals[1])


def feasibility():
    output = []
    for di, days in enumerate(DURATIONS):
        for ri, rate in enumerate(RATES):
            for clustered in (False, True):
                rng = stream(1, di, ri, int(clustered))
                n = counts(rng, FEASIBILITY_REPLICATIONS, days, rate, clustered)
                total, dates = n.sum(axis=1), (n > 0).sum(axis=1)
                for retention in RETENTIONS:
                    accepted = rng.binomial(n, retention)
                    k, kd = accepted.sum(axis=1), (accepted > 0).sum(axis=1)
                    failures = {'parent_count': total < 200, 'accepted_count': k < 100,
                                'parent_dates': dates < 120, 'accepted_dates': kd < 60,
                                'retention': k < .4*total}
                    failed = np.logical_or.reduce(list(failures.values()))
                    output.append({
                        'days': days, 'rate': rate, 'clustered_dates': clustered,
                        'retention': retention, 'replications': FEASIBILITY_REPLICATIONS,
                        'mean_parents': float(total.mean()), 'mean_accepted': float(k.mean()),
                        'mean_parent_dates': float(dates.mean()), 'mean_accepted_dates': float(kd.mean()),
                        'insufficient_probability': float(failed.mean()),
                        'insufficient_mc95': wilson(int(failed.sum()), len(failed)),
                        'marginal_failure_probability': {key: float(value.mean()) for key, value in failures.items()},
                    })
    return output


def synthetic(rng, days, family):
    n = counts(rng, REPLICATIONS, days, RATES[1], family.startswith('ar'))
    exists = np.arange(3)[None, None, :] < n[:, :, None]
    rejected = exists & (rng.random(exists.shape) >= .5)
    if family == 'iid_candidate':
        noise = rng.normal(size=exists.shape)
    elif family == 'skew_date':
        noise = (rng.exponential(size=(REPLICATIONS, days, 1)) - 1)
    elif family == 't3_date':
        noise = rng.standard_t(3, size=(REPLICATIONS, days, 1))/math.sqrt(3)
    else:
        state = rng.normal(size=(REPLICATIONS, days))
        if family.startswith('ar'):
            phi = math.exp(-1/int(family[2:]))
            for t in range(1, days):
                state[:, t] = phi*state[:, t-1] + math.sqrt(1-phi*phi)*state[:, t]
        noise = state[:, :, None]
    # Accepted net R cancels from Delta. It need not be simulated or evaluated.
    return n, rejected.sum(axis=2), (rejected*noise).sum(axis=2)


def calibrate(progress=print):
    output = []
    for di, days in enumerate(DURATIONS):
        for fi, family in enumerate(FAMILIES):
            n, rejected, y = synthetic(stream(2, di, fi), days, family)
            for length in LENGTHS:
                stats = {mean: {'reject': 0, 'covered': 0, 'failed': 0, 'margins': [], 'split_disagree': 0}
                         for mean in MEANS}
                for rep in range(REPLICATIONS):
                    totals = resample_totals([y[rep], rejected[rep], n[rep]], length,
                                             stream(3, di, fi, rep, length))
                    for mean, acc in stats.items():
                        if np.count_nonzero(n[rep]) < 2 or np.any(totals[2] <= 0):
                            acc['failed'] += 1
                            continue
                        observed = (y[rep].sum() + mean*rejected[rep].sum())/n[rep].sum()
                        samples = (totals[0] + mean*totals[1])/totals[2]
                        try:
                            lower = lower_bound(observed, samples)
                        except ValueError:
                            acc['failed'] += 1
                            continue
                        acc['reject'] += lower > 0
                        acc['covered'] += lower <= .5*mean
                        acc['margins'].append(observed-lower)
                        split = len(samples)//2
                        acc['split_disagree'] += ((lower_bound(observed, samples[:split]) > 0)
                                                 != (lower_bound(observed, samples[split:]) > 0))
                for mean, acc in stats.items():
                    valid = REPLICATIONS - acc['failed']
                    output.append({
                        'days': days, 'family': family, 'block_days': length,
                        'rejected_mean': mean, 'true_delta': .5*mean,
                        'replications': REPLICATIONS, 'valid': valid,
                        'failure_probability': acc['failed']/REPLICATIONS,
                        'reject_probability': float(acc['reject']/REPLICATIONS),
                        'reject_mc95': wilson(int(acc['reject']), REPLICATIONS),
                        # Failures count as noncoverage, never silently discarded.
                        'coverage': float(acc['covered']/REPLICATIONS),
                        'coverage_mc95': wilson(int(acc['covered']), REPLICATIONS),
                        'mean_lower_margin': float(np.mean(acc['margins'])) if valid else None,
                        'split_draw_rejection_disagreement': float(acc['split_disagree']/REPLICATIONS),
                    })
        progress(f'Synthetic calibration complete: {days} days', flush=True)
    return output


def run_study(output=Path('reports/v11_readiness_study/calibration')):
    output.mkdir(parents=True, exist_ok=False)
    spec = Path('research/v11_readiness_study/STUDY_SPEC.md')
    binding = {'status': 'SYNTHETIC_ONLY_NOT_V11_OUTCOMES', 'seed': SEED,
               'draws': DRAWS, 'replications': REPLICATIONS, 'numpy': np.__version__,
               'python': platform.python_version(), 'spec_sha256': hashlib.sha256(spec.read_bytes()).hexdigest(),
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'study_binding.json').write_text(json.dumps(binding, indent=2)+'\n')
    results = feasibility()
    (output/'feasibility.json').write_text(json.dumps(results, indent=2)+'\n')
    results = calibrate()
    (output/'inference.json').write_text(json.dumps(results, indent=2)+'\n')
    screens = []
    for days in DURATIONS:
        for length in LENGTHS:
            rows = [r for r in results if r['days']==days and r['block_days']==length
                    and r['rejected_mean']==0]
            screens.append({'days': days, 'block_days': length,
                            'diagnostic_screen_pass': all(r['reject_mc95'][1] <= .08
                            and r['coverage_mc95'][0] >= .92 and r['failure_probability']==0 for r in rows)})
    (output/'method_screens.json').write_text(json.dumps(screens, indent=2)+'\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    (output/'artifact_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    return screens


if __name__ == '__main__':
    print(json.dumps(run_study(), indent=2))
