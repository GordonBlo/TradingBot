import numpy as np
import pytest

from src.research.v11_synthetic_calibration import (
    circular_sums,
    counts,
    lower_bound,
    paired_lower_bound,
    resample_totals,
    stream,
    synthetic,
    wilson,
)


def test_circular_blocks_and_truncated_last_block_preserve_dates():
    assert circular_sums([1, 2, 3, 4], 3).tolist() == [6, 9, 8, 7]
    totals = resample_totals([np.ones(10), np.ones(10)*2], 7, stream(99), draws=20)
    assert np.all(totals[0] == 10)
    assert np.all(totals[1] == 20)


def test_paired_counts_resample_with_improvement():
    totals = resample_totals([[2, 0, 4, 2], [1, 0, 2, 1]], 2, stream(98), draws=30)
    assert np.array_equal(totals[0], 2*totals[1])


def test_one_sided_basic_interval_formula():
    draws = np.arange(100)/100
    assert lower_bound(.5, draws) == pytest.approx(1-np.quantile(draws, .95))


@pytest.mark.parametrize('y,n', [([0, 0], [0, 0]), ([0, 1], [1, 0]),
                               ([1, 2], [-1, 2]), ([1, 2], [.5, 2]),
                               ([0, 0], [1, 1]), ([float('nan'), 1], [1, 1])])
def test_invalid_sparse_or_degenerate_inputs_never_produce_success(y, n):
    with pytest.raises(ValueError):
        paired_lower_bound(y, n, length=1, draws=30)


def test_determinism_and_count_family_metadata():
    left = counts(stream(97), 1000, 100, 443/990, False)
    right = counts(stream(97), 1000, 100, 443/990, False)
    assert np.array_equal(left, right)
    assert set(np.unique(left)) == {0, 1, 2, 3}
    assert abs(left.mean()-443/990) < .01
    assert abs((left > 0).mean()-372/990) < .01
    assert wilson(0, 400)[1] > 0  # Monte Carlo uncertainty is not suppressed.


def test_repeatable_candidate_interval():
    y = [1, -1, 0, 1, 0, -2, 1, 0]*10
    n = [1, 1, 0, 1, 0, 2, 1, 0]*10
    assert paired_lower_bound(y, n, length=7, draws=100) == paired_lower_bound(y, n, length=7, draws=100)


def test_fat_tailed_family_is_finite_and_keeps_inactive_dates_zero():
    n, rejected, y = synthetic(stream(96), 60, 't3_date')
    assert np.all(np.isfinite(y))
    assert np.all(y[n == 0] == 0)
    assert np.all(rejected <= n)
