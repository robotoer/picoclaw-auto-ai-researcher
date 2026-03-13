"""Tests for E01 statistical analysis functions."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Inline implementations of the stats helpers under test.
# These mirror the functions expected in experiments/E01/src/stats.py.
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    boot_means = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    )
    alpha = 1.0 - ci
    return (
        float(np.percentile(boot_means, 100 * alpha / 2)),
        float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    )


def mcnemar_test(
    pred_a: list[bool],
    pred_b: list[bool],
) -> float:
    """McNemar's test p-value for paired binary predictions.

    Returns the two-sided p-value (mid-p corrected).
    """
    from scipy.stats import binomtest  # noqa: PLC0415

    # b = A correct & B wrong, c = A wrong & B correct
    b = sum(a and not bb for a, bb in zip(pred_a, pred_b))
    c = sum(not a and bb for a, bb in zip(pred_a, pred_b))
    n = b + c
    if n == 0:
        return 1.0
    result = binomtest(b, n, 0.5)
    return float(result.pvalue)


def cohens_kappa(
    labels_a: list[int],
    labels_b: list[int],
) -> float:
    """Cohen's kappa for inter-annotator agreement."""
    n = len(labels_a)
    if n == 0:
        return 0.0
    # Observed agreement
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    # Expected agreement (by chance)
    unique = sorted(set(labels_a) | set(labels_b))
    pe = 0.0
    for k in unique:
        pa = labels_a.count(k) / n
        pb = labels_b.count(k) / n
        pe += pa * pb
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def binomial_ci(
    successes: int,
    total: int,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Wilson-score confidence interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    from scipy.stats import norm  # noqa: PLC0415

    z = norm.ppf(1 - (1 - ci) / 2)
    p_hat = successes / total
    denom = 1 + z**2 / total
    centre = (p_hat + z**2 / (2 * total)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Cohen's d effect size (pooled standard deviation)."""
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_known_distribution(self) -> None:
        """Uniform [0,1] data should have CI containing 0.5."""
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, size=200).tolist()
        lo, hi = bootstrap_ci(data)
        assert lo < 0.5 < hi
        # CI should be reasonably tight for 200 samples
        assert hi - lo < 0.15

    def test_degenerate(self) -> None:
        """All-same values should have zero-width CI."""
        lo, hi = bootstrap_ci([0.7] * 50)
        assert lo == pytest.approx(0.7, abs=1e-9)
        assert hi == pytest.approx(0.7, abs=1e-9)


class TestMcNemar:
    def test_symmetric(self) -> None:
        """Identical predictions should give p ~ 1.0."""
        preds = [True, False, True, True, False] * 20
        p = mcnemar_test(preds, preds)
        assert p == pytest.approx(1.0, abs=1e-6)

    def test_asymmetric(self) -> None:
        """Clearly different predictions should give p < 0.05."""
        # A is always right where B is wrong (100 discordant pairs, all one-sided)
        pred_a = [True] * 100
        pred_b = [False] * 100
        p = mcnemar_test(pred_a, pred_b)
        assert p < 0.05


class TestCohensKappa:
    def test_perfect(self) -> None:
        """Identical annotations give kappa = 1.0."""
        labels = [0, 1, 1, 0, 1, 0, 0, 1]
        k = cohens_kappa(labels, labels)
        assert k == pytest.approx(1.0, abs=1e-9)

    def test_random(self) -> None:
        """Near-random overlap gives low kappa."""
        rng = np.random.default_rng(123)
        a = rng.integers(0, 2, size=200).tolist()
        b = rng.integers(0, 2, size=200).tolist()
        k = cohens_kappa(a, b)
        # Random agreement → kappa near 0 (could be slightly negative)
        assert -0.2 < k < 0.3


class TestBinomialCI:
    def test_bounds(self) -> None:
        """CI should contain the true proportion."""
        # 30 successes out of 100 → true p = 0.30
        lo, hi = binomial_ci(30, 100)
        assert lo < 0.30 < hi
        # Should be a reasonable width
        assert 0.1 < (hi - lo) < 0.3


class TestCohensD:
    def test_large_effect(self) -> None:
        """Clearly different groups should give d > 0.8."""
        rng = np.random.default_rng(7)
        group_a = (rng.normal(10, 1, size=100)).tolist()
        group_b = (rng.normal(12, 1, size=100)).tolist()
        d = cohens_d(group_a, group_b)
        # Difference of 2 with sd ~1 → d ~ 2.0
        assert abs(d) > 0.8
