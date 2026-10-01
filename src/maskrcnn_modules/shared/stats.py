# Vendored from SuperDefectExperiments (src/anomaly_modules/shared/stats.py, commit 11c1c57).
# Do not edit here: re-copy from the source repo so the two do not drift.
# Docstring references to ``anomaly_modules`` refer to that repo.
"""Post-hoc statistical layer: bootstrap CIs and paired significance tests.

These are **not** in-loop torchmetrics -- they run in the reporting stage
over per-image (or per-unit) scores, so we can bootstrap without re-running
the model. Consumed by ``scripts/compare_experiments.py`` over the
per-image CSVs written by :func:`anomaly_modules.shared.reporting.compute_per_image_metrics`.
"""

from typing import Callable, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return ``(point_estimate, lo, hi)`` for a statistic over ``values``.

    Non-parametric percentile bootstrap. NaNs in ``values`` are dropped
    silently (per-image metrics often contain NaNs on empty-defect images).
    If fewer than 2 finite values remain, returns ``(nan, nan, nan)``.
    """
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (float("nan"), float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    point = float(statistic(x))
    n = x.size
    draws = rng.integers(0, n, size=(n_bootstrap, n))
    boots = np.asarray([statistic(x[d]) for d in draws], dtype=np.float64)
    alpha = 1.0 - ci
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return (point, float(lo), float(hi))


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> dict:
    """Wilcoxon signed-rank paired test.

    Pairs are dropped where either side is non-finite. Returns
    ``{"statistic": float, "pvalue": float, "n_pairs": int}``. Uses
    ``scipy.stats.wilcoxon``.
    """
    from scipy.stats import wilcoxon

    a_arr = np.asarray(list(a), dtype=np.float64)
    b_arr = np.asarray(list(b), dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError("wilcoxon_paired: a and b must be same length")
    keep = np.isfinite(a_arr) & np.isfinite(b_arr)
    a_arr, b_arr = a_arr[keep], b_arr[keep]
    if a_arr.size < 2 or np.all(a_arr == b_arr):
        return {"statistic": float("nan"), "pvalue": float("nan"), "n_pairs": int(a_arr.size)}
    stat, p = wilcoxon(a_arr, b_arr, zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(stat), "pvalue": float(p), "n_pairs": int(a_arr.size)}
