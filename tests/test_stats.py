import math

import numpy as np

from maskrcnn_modules.shared.stats import bootstrap_ci, wilcoxon_paired


def test_bootstrap_ci_point_estimate_matches_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    point, lo, hi = bootstrap_ci(values, n_bootstrap=500, seed=0)

    assert math.isclose(point, 3.0)
    assert lo <= point <= hi


def test_bootstrap_ci_drops_nans_silently():
    values = [1.0, 2.0, float("nan"), 3.0]

    point, lo, hi = bootstrap_ci(values, n_bootstrap=200, seed=0)

    assert math.isclose(point, 2.0)
    assert lo <= point <= hi


def test_bootstrap_ci_returns_nan_with_fewer_than_two_finite_values():
    point, lo, hi = bootstrap_ci([1.0, float("nan")])

    assert math.isnan(point)
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_wilcoxon_paired_detects_consistent_difference():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [2.0, 3.0, 4.0, 5.0, 6.0]  # b is uniformly a+1 -> should be significant

    result = wilcoxon_paired(a, b)

    assert result["n_pairs"] == 5
    assert result["pvalue"] < 0.1


def test_wilcoxon_paired_drops_non_finite_pairs():
    a = [1.0, 2.0, float("nan"), 4.0]
    b = [1.5, 2.5, 3.5, float("nan")]

    result = wilcoxon_paired(a, b)

    assert result["n_pairs"] == 2


def test_wilcoxon_paired_identical_arrays_returns_nan():
    a = [1.0, 2.0, 3.0]

    result = wilcoxon_paired(a, a)

    assert math.isnan(result["statistic"])
    assert math.isnan(result["pvalue"])


def test_wilcoxon_paired_mismatched_length_raises():
    try:
        wilcoxon_paired([1.0, 2.0], [1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched lengths")


def test_bootstrap_ci_accepts_numpy_statistic():
    values = [1.0, 1.0, 1.0, 10.0]

    point, _lo, _hi = bootstrap_ci(values, statistic=np.median, n_bootstrap=200, seed=1)

    assert math.isclose(point, 1.0)
