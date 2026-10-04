import math
from statistics import NormalDist

import numpy as np
import pytest

from src.research import v11_methodology_resolution as study


def reference_dates():
    n = np.tile([1, 0, 2, 1, 0, 3], 30)
    y = np.tile([.7, 0, -1.2, .9, 0, -.1], 30)
    return y, n


def test_long_bandwidth_is_ex_ante_and_sublinear():
    assert study.block_length(728) == 112
    assert [study.block_length(t) for t in (180, 270, 365, 448, 560)] == [71, 81, 89, 96, 103]
    with pytest.raises(ValueError):
        study.block_length(10)


def test_hac_matches_direct_non_circular_covariance_formula():
    y, n = reference_dates()
    t = len(y)
    delta = y.sum()/n.sum()
    z = y-delta*n
    length = study.block_length(t)
    lrv = np.dot(z, z)/t
    for k in range(1, length):
        lrv += 2*(1-k/length)*np.dot(z[k:], z[:-k])/t
    expected = delta-NormalDist().inv_cdf(.95)*math.sqrt(t*lrv)/n.sum()
    assert study.analytic_bounds(y, n, 5.3)['HAC_LONG'][0] == pytest.approx(expected, abs=1e-12)


def test_self_normalization_ratio_influence_and_full_bandwidth_identity():
    y, n = reference_dates()
    t = len(y)
    delta = y.sum()/n.sum()
    z = y-delta*n
    normalizer = np.sum(np.cumsum(z)**2)/t**2
    full_hac = np.dot(z, z)/t
    for k in range(1, t):
        full_hac += 2*(1-k/t)*np.dot(z[k:], z[:-k])/t
    assert full_hac == pytest.approx(2*normalizer, abs=1e-12)
    expected = delta-5.3*math.sqrt(t*normalizer)/n.sum()
    assert study.analytic_bounds(y, n, 5.3)['SELF_NORMALIZED'][0] == pytest.approx(expected)


@pytest.mark.parametrize('method', ['HAC_LONG', 'SELF_NORMALIZED'])
def test_ratio_bound_translation_scale_and_batch_invariance(method):
    y, n = reference_dates()
    baseline = study.analytic_bounds(y, n, 5.3)[method][0]
    assert study.analytic_bounds(y+.25*n, n, 5.3)[method][0] == pytest.approx(baseline+.25)
    assert study.analytic_bounds(3*y, n, 5.3)[method][0] == pytest.approx(3*baseline)
    batched = study.analytic_bounds(np.stack([y, 3*y]), np.stack([n, n]), 5.3)[method]
    assert batched.tolist() == pytest.approx([baseline, 3*baseline])


def test_zero_dates_are_preserved_and_not_active_date_means():
    y, n = reference_dates()
    # Misweighting multiple-parent dates produces a different estimand.
    active_date_mean = np.mean(y[n > 0]/n[n > 0])
    assert active_date_mean != pytest.approx(y.sum()/n.sum())
    y_with_gap, n_with_gap = np.concatenate([y, np.zeros(180)]), np.concatenate([n, np.zeros(180)])
    assert study.analytic_bounds(y, n, 5.3)['SELF_NORMALIZED'][0] != pytest.approx(
        study.analytic_bounds(y_with_gap, n_with_gap, 5.3)['SELF_NORMALIZED'][0])


@pytest.mark.parametrize('case', ['nan', 'negative', 'fractional', 'empty_date_value', 'shape'])
def test_malformed_dates_fail_explicitly(case):
    y, n = (value.astype(float) for value in reference_dates())
    if case == 'nan':
        y[0] = np.nan
    elif case == 'negative':
        n[0] = -1
    elif case == 'fractional':
        n[0] = .5
    elif case == 'empty_date_value':
        y[1] = 1
    else:
        n = n[:-1]
    with pytest.raises(ValueError, match='INFERENCE_UNAVAILABLE'):
        study.analytic_bounds(y, n, 5.3)
    with pytest.raises(ValueError):
        study.cbb_bound(y, n, study.stream(99), draws=19)


def test_degenerate_and_insufficient_dates_never_reject():
    for n in (np.zeros(180), np.ones(180), np.r_[1., np.zeros(179)]):
        results = study.analytic_bounds(np.zeros(180), n, 5.3)
        assert all(np.isnan(value[0]) for value in results.values())
        lower, disagreement = study.cbb_bound(np.zeros(180), n, study.stream(98), draws=19)
        assert math.isnan(lower) and not disagreement


def test_cbb_matches_existing_frozen_study_mechanics_and_has_no_redraw(monkeypatch):
    y, n = reference_dates()
    totals = study.resample_totals([y, n], study.block_length(len(y)), study.stream(97), 99)
    expected = 2*y.sum()/n.sum()-np.quantile(totals[0]/totals[1], .95, method='linear')
    actual = study.cbb_bound(y, n, study.stream(97), draws=99)
    assert actual[0] == pytest.approx(expected)
    assert actual == study.cbb_bound(y, n, study.stream(97), draws=99)
    calls = []

    def empty_draw(*args):
        calls.append(1)
        return np.array([[1., 0., 2.], [1., 0., 1.]])

    monkeypatch.setattr(study, 'resample_totals', empty_draw)
    assert math.isnan(study.cbb_bound(y, n, study.stream(97))[0])
    assert len(calls) == 1


def test_exact_binomial_intervals_and_complement_symmetry():
    assert study.binomial_cdf(2, 4, .5) == pytest.approx(11/16)
    assert study.clopper_pearson(0, 10) == pytest.approx([0, 1-.025**.1])
    assert study.clopper_pearson(5, 10) == pytest.approx([.1870860284473985, .8129139715526015])
    lo, hi = study.clopper_pearson(1, 20, study.MC_ALPHA)
    inverse = study.clopper_pearson(19, 20, study.MC_ALPHA)
    assert inverse == pytest.approx([1-hi, 1-lo])
    assert study.binomial_cdf(1, 20, hi) == pytest.approx(study.MC_ALPHA/2, abs=1e-12)
    assert study.binomial_cdf(0, 20, lo) == pytest.approx(1-study.MC_ALPHA/2, abs=1e-12)


@pytest.mark.parametrize('family', study.FAMILIES)
def test_synthetic_families_repeatable_subset_only_and_no_empty_date_values(family):
    left = study.synthetic(study.stream(96), 8, 40, family, 443/990, .5)
    right = study.synthetic(study.stream(96), 8, 40, family, 443/990, .5)
    for a, b in zip(left, right):
        assert np.array_equal(a, b)
    n, accepted, y = left
    assert set(np.unique(n)).issubset({0, 1, 2, 3})
    assert np.all(accepted <= n)
    assert np.all(y[n == 0] == 0)
    assert np.all(y[accepted == n] == 0)
    assert np.isfinite(y).all()


def test_failure_is_no_rejection_and_no_coverage_not_dropped():
    lower = np.array([-.1, .1, np.nan, -.2]*2)
    ones = np.ones(8)
    row = study.summarize(lower, np.zeros(8, dtype=bool), ones, ones, ones, ones, ones)
    assert row['rejection']['rate'] == .25
    assert row['coverage']['rate'] == .5
    assert row['failure']['rate'] == .25
    assert row['conditional_coverage'] == pytest.approx(2/3)
    assert not row['passes']


def test_brownian_signed_critical_deterministic_and_diagnostics_bound(monkeypatch):
    monkeypatch.setattr(study, 'BROWNIAN_PATHS', 2048)
    monkeypatch.setattr(study, 'BROWNIAN_GRID', 64)
    first = study.brownian_critical()
    assert first == study.brownian_critical()
    assert first['critical'] > NormalDist().inv_cdf(.95)
    assert first['quantile_mc95'][0] < first['critical'] < first['quantile_mc95'][1]


def test_rejected_anchor_does_not_trigger_duration_or_promote_method(tmp_path, monkeypatch):
    called = []

    def reject(output, days, methods, critical):
        called.append(days)
        return [{'method': m, 'passes': False} for m in methods]

    monkeypatch.setattr(study, 'calibrate_duration', reject)
    monkeypatch.setattr(study, 'brownian_critical', lambda: {'critical': 5.3, 'numerical_screen_pass': True})
    output = tmp_path/'synthetic'
    study.run_study(output)
    import json
    decision = json.loads((output/'decision.json').read_text())
    assert called == [728]
    assert decision['selected_method'] is None
    assert decision['verdict'] == 'METHODOLOGY_UNRESOLVED'
    assert decision['duration_extension'] == 'NOT_TRIGGERED'
    with pytest.raises(FileExistsError):
        study.run_study(output)


def test_study_has_no_market_model_collector_or_order_imports():
    import ast
    tree = ast.parse(study.Path(study.__file__).read_text())
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert [module for module in modules if module.startswith('src.')] == [
        'src.research.v11_synthetic_calibration']
    assert study.METHODS == ('CBB_LONG', 'HAC_LONG', 'SELF_NORMALIZED')
    assert len(study.FAMILIES) == 11 and len(study.PROFILES) == 7
