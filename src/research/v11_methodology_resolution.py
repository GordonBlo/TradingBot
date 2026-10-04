"""Synthetic-only method calibration. No market-data/model/collector imports."""
from __future__ import annotations

import hashlib
import json
import math
import platform
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

import numpy as np

from src.research.v11_synthetic_calibration import counts, resample_totals

SEED = 20260928
REPLICATIONS = 8192
BATCH = 2048
DRAWS = 1999
BROWNIAN_PATHS = 262144
BROWNIAN_GRID = 2048
METHODS = ('CBB_LONG', 'HAC_LONG', 'SELF_NORMALIZED')
PRIORITY = ('SELF_NORMALIZED', 'CBB_LONG', 'HAC_LONG')
FAMILIES = ('iid_candidate', 'date_shared', 'clustered_iid', 'ar7', 'ar28',
            'skew_date', 't3_date', 'ar56', 'ar28_skew', 'ar28_t3',
            'ar28_persistent_filter')
PROFILES = tuple((443/990, p) for p in (.25, .4, .5, .75, .9)) + ((25/90, .5), (63/90, .5))
EXTENSION_DAYS = (180, 270, 365, 448, 560)
MC_ALPHA = .05 / (6 * 77 * 3 * 4)
SPEC = Path('research/v11_methodology_resolution/STUDY_SPEC.md')


def stream(*keys):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, *keys])))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')


def block_length(days):
    if days < 2:
        raise ValueError('at least two calendar dates required')
    length = math.ceil(112 * (days/728)**(1/3))
    if length >= days:
        raise ValueError('INFERENCE_UNAVAILABLE: block spans whole interval')
    return length


@lru_cache(maxsize=12)
def _log_choose(n):
    k = np.arange(1, n+1, dtype=float)
    return np.concatenate(([0.], np.cumsum(np.log(n-k+1)-np.log(k))))


def binomial_cdf(k, n, p):
    """Finite binomial sum, with complement for the shorter upper tail."""
    if not 0 <= p <= 1 or n < 0:
        raise ValueError('invalid binomial inputs')
    if k < 0:
        return 0.
    if k >= n:
        return 1.
    if p == 0:
        return 1.
    if p == 1:
        return 0.
    if k > n//2:
        return 1-binomial_cdf(n-k-1, n, 1-p)
    i = np.arange(k+1)
    terms = _log_choose(n)[:k+1] + i*math.log(p) + (n-i)*math.log1p(-p)
    return float(np.clip(np.exp(terms).sum(), 0., 1.))


def clopper_pearson(k, n, alpha=.05):
    """Exact binomial interval (floating-point finite-sum inversion)."""
    if not isinstance(k, (int, np.integer)) or not 0 <= k <= n or n <= 0 or not 0 < alpha < 1:
        raise ValueError('invalid Monte Carlo count')
    if k > n//2:
        lo, hi = clopper_pearson(n-k, n, alpha)
        return [1-hi, 1-lo]
    if k == 0:
        return [0., -math.expm1(math.log(alpha/2)/n)]

    def solve(index, target):
        low, high = 0., 1.
        for _ in range(52):
            mid = (low+high)/2
            if binomial_cdf(index, n, mid) > target:
                low = mid
            else:
                high = mid
        return (low+high)/2

    return [solve(k-1, 1-alpha/2), solve(k, alpha/2)]


def quantile_mc_interval(values, probability=.95):
    """Order-statistic CI for a continuous-distribution quantile."""
    values = np.sort(values)
    n = len(values)

    def rank(tail):
        lo, hi = 0, n
        while lo < hi:
            mid = (lo+hi)//2
            if binomial_cdf(mid, n, probability) < tail:
                lo = mid+1
            else:
                hi = mid
        return lo

    return [float(values[max(0, rank(.025)-1)]), float(values[min(n-1, rank(.975))])]


def brownian_critical():
    rng = stream(0)
    fine, coarse = [], []
    time = np.arange(1, BROWNIAN_GRID+1)/BROWNIAN_GRID
    for _ in range(BROWNIAN_PATHS//1024):
        path = np.cumsum(rng.normal(size=(1024, BROWNIAN_GRID)), axis=1)
        endpoint = path[:, -1].copy()
        path -= endpoint[:, None]*time
        fine.extend(endpoint / np.sqrt(np.mean(path*path, axis=1)))
        coarse.extend(endpoint / np.sqrt(np.mean(path[:, 3::4]**2, axis=1)))
    fine, coarse = np.asarray(fine), np.asarray(coarse)
    critical = float(np.quantile(fine, .95, method='linear'))
    coarse_q = float(np.quantile(coarse, .95, method='linear'))
    half_q = [float(np.quantile(part, .95, method='linear')) for part in np.split(fine, 2)]
    grid_difference = abs(coarse_q/critical-1)
    split_difference = abs(half_q[0]-half_q[1])/critical
    return {'critical': critical, 'paths': BROWNIAN_PATHS, 'grid': BROWNIAN_GRID,
            'coarse_grid': BROWNIAN_GRID//4, 'coarse_critical': coarse_q,
            'half_criticals': half_q, 'quantile_mc95': quantile_mc_interval(fine),
            'relative_grid_difference': grid_difference,
            'relative_half_difference': split_difference,
            'numerical_screen_pass': grid_difference <= .01 and split_difference <= .01}


def validate_dates(y, n):
    y, n = np.asarray(y, dtype=float), np.asarray(n, dtype=float)
    if (y.shape != n.shape or y.ndim not in (1, 2) or y.shape[-1] < 2
            or not np.isfinite(y).all() or not np.isfinite(n).all()
            or np.any(n < 0) or np.any(n != np.floor(n)) or np.any(y[n == 0] != 0)):
        raise ValueError('INFERENCE_UNAVAILABLE: malformed paired dates')
    return y, n


def analytic_bounds(y, n, critical):
    """Batch bounds; NaN explicitly means unavailable, never rejection."""
    y, n = validate_dates(y, n)
    y, n = np.atleast_2d(y), np.atleast_2d(n)
    days = y.shape[1]
    length = block_length(days)
    total = n.sum(axis=1)
    available = ((n > 0).sum(axis=1) >= 2) & (total > 0)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        delta = y.sum(axis=1)/total
        z = y-delta[:, None]*n
        cumulative = np.cumsum(z, axis=1)
        sn_scale = np.sqrt(np.sum(cumulative*cumulative, axis=1)/days)/total
        # Zero-padded rolling-sum identity equals the noncircular Bartlett HAC
        # covariance sum, including boundary windows and divisor T.
        padded = np.pad(z, ((0, 0), (length-1, length-1)))
        cs = np.pad(np.cumsum(padded, axis=1), ((0, 0), (1, 0)))
        rolling = cs[:, length:]-cs[:, :-length]
        hac_scale = np.sqrt(np.sum(rolling*rolling, axis=1)/length)/total
        bounds = {'HAC_LONG': delta-NormalDist().inv_cdf(.95)*hac_scale,
                  'SELF_NORMALIZED': delta-critical*sn_scale}
    for name, scale in [('HAC_LONG', hac_scale), ('SELF_NORMALIZED', sn_scale)]:
        valid = available & np.isfinite(bounds[name]) & (scale > 0) & np.isfinite(scale)
        bounds[name][~valid] = np.nan
    return bounds


def cbb_bound(y, n, rng, draws=DRAWS):
    y, n = validate_dates(y, n)
    if y.ndim != 1:
        raise ValueError('one replicate required')
    if np.count_nonzero(n) < 2:
        return math.nan, False
    totals = resample_totals([y, n], block_length(len(y)), rng, draws)
    if np.any(totals[1] <= 0):
        return math.nan, False
    samples = totals[0]/totals[1]
    if not np.isfinite(samples).all() or np.ptp(samples) == 0:
        return math.nan, False
    delta = y.sum()/n.sum()
    bound = 2*delta-float(np.quantile(samples, .95, method='linear'))
    first, last = samples[:draws//2], samples[draws//2:]
    disagreement = ((2*delta-np.quantile(first, .95, method='linear') > 0)
                    != (2*delta-np.quantile(last, .95, method='linear') > 0))
    return bound, bool(disagreement)


def ar_state(rng, replicas, days, tau, innovation='normal'):
    phi = math.exp(-1/tau)
    factor = math.sqrt(1-phi*phi)
    if innovation == 'normal':
        out = rng.normal(size=(replicas, days))
        for t in range(1, days):
            out[:, t] = phi*out[:, t-1]+factor*out[:, t]
        return out
    size = (replicas, days+2048)
    noise = (rng.exponential(size=size)-1 if innovation == 'skew'
             else rng.standard_t(3, size=size)/math.sqrt(3))
    state = np.zeros(replicas)
    out = np.empty((replicas, days))
    for t in range(days+2048):
        state = phi*state+factor*noise[:, t]
        if t >= 2048:
            out[:, t-2048] = state
    return out


def synthetic(rng, replicas, days, family, rate, retention):
    if family not in FAMILIES or not 0 < retention < 1:
        raise ValueError('outside prespecified scenarios')
    clustered = family.startswith('ar') or family == 'clustered_iid'
    n = counts(rng, replicas, days, rate, clustered)
    exists = np.arange(3)[None, None, :] < n[:, :, None]
    if family == 'ar28_persistent_filter':
        latent = ar_state(rng, replicas, days, 28)
        accept = exists & (latent[:, :, None] < NormalDist().inv_cdf(retention))
    else:
        accept = exists & (rng.random(exists.shape) < retention)
    rejected = exists & ~accept
    if family in ('iid_candidate', 'clustered_iid'):
        noise = rng.normal(size=exists.shape)
    elif family == 'skew_date':
        noise = (rng.exponential(size=(replicas, days))-1)[:, :, None]
    elif family == 't3_date':
        noise = (rng.standard_t(3, size=(replicas, days))/math.sqrt(3))[:, :, None]
    elif family.startswith('ar'):
        tau = 7 if family == 'ar7' else 56 if family == 'ar56' else 28
        innovation = 'skew' if family.endswith('_skew') else 't3' if family.endswith('_t3') else 'normal'
        noise = ar_state(rng, replicas, days, tau, innovation)[:, :, None]
    else:
        noise = rng.normal(size=(replicas, days, 1))
    return n, accept.sum(axis=2), (rejected*noise).sum(axis=2)


def summarize(lower, split, observed, parent_n, accepted_n, parent_dates, accepted_dates):
    valid = np.isfinite(lower)
    rejected = valid & (lower > 0)
    covered = valid & (lower <= 0)
    failed = ~valid
    metrics = {}
    for name, flags in [('rejection', rejected), ('coverage', covered),
                        ('failure', failed), ('split_disagreement', split)]:
        k, total = int(flags.sum()), len(flags)
        metrics[name] = {'count': k, 'rate': k/total,
                         'mc95': clopper_pearson(k, total),
                         'simultaneous_mc': clopper_pearson(k, total, MC_ALPHA)}
    metrics['conditional_coverage'] = float(covered.sum()/valid.sum()) if valid.any() else None
    metrics['batch_rejection_rates'] = [float(x.mean()) for x in np.split(rejected, 4)]
    metrics['batch_rejection_mc95'] = [clopper_pearson(int(x.sum()), len(x)) for x in np.split(rejected, 4)]
    metrics['batch_failure_rates'] = [float(x.mean()) for x in np.split(failed, 4)]
    metrics['mean_lower_margin'] = float(np.mean(observed[valid]-lower[valid])) if valid.any() else None
    metrics['counts'] = {key: {'mean': float(value.mean()),
                              'q05_q50_q95': np.quantile(value, [.05, .5, .95]).tolist()}
                         for key, value in [('parent', parent_n), ('accepted', accepted_n),
                                            ('parent_dates', parent_dates), ('accepted_dates', accepted_dates)]}
    rejection_ci = metrics['rejection']['simultaneous_mc']
    coverage_ci = metrics['coverage']['simultaneous_mc']
    metrics['screen'] = {
        'size_close': rejection_ci[0] >= .035 and rejection_ci[1] <= .065,
        'coverage_close': coverage_ci[0] >= .935 and coverage_ci[1] <= .965,
        'failure_low': metrics['failure']['simultaneous_mc'][1] <= .01,
        'draw_stability': metrics['split_disagreement']['simultaneous_mc'][1] <= .02}
    metrics['passes'] = all(metrics['screen'].values())
    return metrics


def calibrate_duration(output, days, methods, critical, progress=print):
    rows = []
    stage = int(days != 728)+1
    for fi, family in enumerate(FAMILIES):
        for pi, (rate, retention) in enumerate(PROFILES):
            arrays = {name: [] for name in methods}
            splits, observed, ns, ks, nds, kds = [], [], [], [], [], []
            for bi in range(REPLICATIONS//BATCH):
                n, accepted, y = synthetic(stream(stage, days, fi, pi, bi, 0), BATCH, days,
                                           family, rate, retention)
                bounds = analytic_bounds(y, n, critical)
                delta = np.divide(y.sum(axis=1), n.sum(axis=1), out=np.full(BATCH, np.nan), where=n.sum(axis=1) > 0)
                observed.extend(delta)
                ns.extend(n.sum(axis=1))
                ks.extend(accepted.sum(axis=1))
                nds.extend((n > 0).sum(axis=1))
                kds.extend((accepted > 0).sum(axis=1))
                for name in methods:
                    if name != 'CBB_LONG':
                        arrays[name].extend(bounds[name])
                if 'CBB_LONG' in methods:
                    for rep in range(BATCH):
                        lower, disagree = cbb_bound(y[rep], n[rep], stream(stage, days, fi, pi, bi, rep, 1))
                        arrays['CBB_LONG'].append(lower)
                        splits.append(disagree)
            observed, ns, ks, nds, kds = map(np.asarray, (observed, ns, ks, nds, kds))
            arrays = {name: np.asarray(values) for name, values in arrays.items()}
            split = np.asarray(splits, dtype=bool) if splits else np.zeros(REPLICATIONS, dtype=bool)
            filename = f'{days}_{family}_p{pi}.npz'
            # Output directory is exclusive, and every filename is single-use.
            with (output/filename).open('xb') as handle:
                np.savez_compressed(handle, **arrays, split_disagreement=split, delta=observed,
                                    parent_n=ns, accepted_n=ks, parent_dates=nds, accepted_dates=kds)
            for name in methods:
                summary = summarize(arrays[name], split if name == 'CBB_LONG' else np.zeros_like(split),
                                    observed, ns, ks, nds, kds)
                rows.append({'days': days, 'family': family, 'profile': pi, 'rate': rate,
                             'retention': retention, 'method': name, 'true_delta': 0.,
                             'replications': REPLICATIONS, 'replicate_artifact': filename, **summary})
            # Progress intentionally excludes outcome metrics until the full run is complete.
            progress(f'completed T={days} family={family} profile={pi}', flush=True)
    return rows


def run_study(output=Path('reports/v11_methodology_resolution/calibration')):
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    binding = {'kind': 'SYNTHETIC_METHODOLOGY_ONLY_NOT_V11_PREREGISTRATION',
               'spec_sha256': digest(SPEC), 'source_sha256': digest(__file__),
               'reused_source_sha256': digest('src/research/v11_synthetic_calibration.py'),
               'seed': SEED, 'replications': REPLICATIONS, 'draws': DRAWS,
               'numpy': np.__version__, 'python': platform.python_version(),
               'methods': METHODS, 'families': FAMILIES, 'profiles': PROFILES,
               'mc_alpha_per_interval': MC_ALPHA}
    write_json(output/'study_binding.json', binding)
    critical = brownian_critical()
    write_json(output/'limiting_critical.json', critical)
    print('limiting critical computation complete; starting full anchor panel', flush=True)
    rows = calibrate_duration(output, 728, METHODS, critical['critical'])
    write_json(output/'anchor_results.json', rows)
    survivors = [m for m in PRIORITY if all(r['passes'] for r in rows if r['method'] == m)
                 and (m != 'SELF_NORMALIZED' or critical['numerical_screen_pass'])]
    anchor_survivors = list(survivors)
    if survivors:
        for days in EXTENSION_DAYS:
            rows.extend(calibrate_duration(output, days, anchor_survivors, critical['critical']))
        write_json(output/'all_results.json', rows)
        survivors = [m for m in anchor_survivors if all(r['passes'] for r in rows if r['method'] == m)]
    # Passing synthetic calibration alone does not establish future-process assumptions.
    decision = {'anchor_survivors': anchor_survivors, 'all_duration_survivors': survivors,
                'duration_extension': 'COMPLETED' if anchor_survivors else 'NOT_TRIGGERED',
                'selected_method': None,
                'verdict': 'METHODOLOGY_UNRESOLVED',
                'reason': ('NO_METHOD_PASSED_ALL_REQUIRED_NULL_CELLS' if not survivors
                           else 'CALIBRATION_PASS_REQUIRES_SEPARATE_ASSUMPTION_REVIEW'),
                'failed_cells_by_method': {m: sum(not r['passes'] for r in rows if r['method'] == m) for m in METHODS},
                'result_cells': len(rows)}
    write_json(output/'decision.json', decision)
    if digest(SPEC) != binding['spec_sha256'] or digest(__file__) != binding['source_sha256']:
        raise RuntimeError('STUDY_INTEGRITY_FAILURE: code/spec changed during execution')
    write_json(output/'artifact_hashes.json', {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == '__main__':
    run_study()
