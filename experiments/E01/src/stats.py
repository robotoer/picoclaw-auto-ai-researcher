"""Statistical analysis for E01 experiment."""

from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        values: Observed values.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for 95% CI).

    Returns:
        Tuple of ``(mean, lower_bound, upper_bound)``.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return (0.0, 0.0, 0.0)

    rng = np.random.default_rng(seed=42)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    mean = float(np.mean(arr))

    return (mean, lower, upper)


def mcnemar_test(
    model_a_correct: list[bool],
    model_b_correct: list[bool],
) -> dict[str, float | bool]:
    """McNemar's test for paired nominal data.

    Compares whether two models have the same error rate on the same set of
    items.

    Args:
        model_a_correct: Per-item correctness for model A.
        model_b_correct: Per-item correctness for model B.

    Returns:
        Dict with ``statistic``, ``p_value``, and ``significant`` (at alpha=0.05).
    """
    a = np.asarray(model_a_correct, dtype=bool)
    b = np.asarray(model_b_correct, dtype=bool)

    if len(a) != len(b):
        raise ValueError("Both lists must have the same length.")

    # Contingency counts:
    # b = number where A correct & B wrong
    # c = number where A wrong & B correct
    b_count = int(np.sum(a & ~b))
    c_count = int(np.sum(~a & b))

    # McNemar's test with continuity correction.
    if b_count + c_count == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    statistic = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
    p_value = float(stats.chi2.sf(statistic, df=1))

    return {
        "statistic": round(float(statistic), 6),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
    }


def cohens_kappa(
    annotations_a: list[set[int]],
    annotations_b: list[set[int]],
) -> float:
    """Compute Cohen's kappa for inter-annotator agreement.

    Each element in the lists is a set of item indices that the annotator
    marked as positive.  The universe of items is the union of all indices
    across both annotators.

    Args:
        annotations_a: Per-document sets of positive item indices for annotator A.
        annotations_b: Per-document sets of positive item indices for annotator B.

    Returns:
        Cohen's kappa coefficient.
    """
    if len(annotations_a) != len(annotations_b):
        raise ValueError("Annotation lists must have the same length.")

    # Flatten into binary vectors over the item universe.
    all_items: set[int] = set()
    for sa, sb in zip(annotations_a, annotations_b):
        all_items |= sa | sb

    if not all_items:
        return 1.0  # Perfect agreement on empty sets.

    items = sorted(all_items)
    a_binary = []
    b_binary = []
    for sa, sb in zip(annotations_a, annotations_b):
        for item in items:
            a_binary.append(item in sa)
            b_binary.append(item in sb)

    a_arr = np.asarray(a_binary, dtype=int)
    b_arr = np.asarray(b_binary, dtype=int)
    n = len(a_arr)

    # Observed agreement.
    p_o = float(np.sum(a_arr == b_arr)) / n

    # Expected agreement by chance.
    a_pos = np.sum(a_arr) / n
    b_pos = np.sum(b_arr) / n
    p_e = float(a_pos * b_pos + (1 - a_pos) * (1 - b_pos))

    if p_e == 1.0:
        return 1.0

    return round((p_o - p_e) / (1.0 - p_e), 6)


def binomial_ci(
    successes: int,
    total: int,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial confidence interval.

    Args:
        successes: Number of successes (e.g. hallucinated claims).
        total: Total number of trials.
        ci: Confidence level.

    Returns:
        Tuple of ``(lower_bound, upper_bound)``.
    """
    if total == 0:
        return (0.0, 1.0)

    alpha = 1.0 - ci
    lower = float(stats.beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = float(stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes))

    # Clamp to [0, 1].
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    return (round(lower, 6), round(upper, 6))


def cohens_d(
    group_a: list[float],
    group_b: list[float],
) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses the pooled standard deviation as the denominator.

    Args:
        group_a: Observations for group A.
        group_b: Observations for group B.

    Returns:
        Cohen's d (positive means group A has a higher mean).
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)

    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))

    # Pooled standard deviation.
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std == 0:
        return 0.0

    return round((mean_a - mean_b) / pooled_std, 6)


def run_full_analysis(metrics: dict) -> dict:
    """Run all statistical tests on experiment metrics.

    Expects ``metrics`` to be keyed by model name, each containing
    ``per_paper`` with lists of dicts that have ``precision``, ``recall``,
    ``f1`` fields.

    Args:
        metrics: Aggregated experiment metrics from
            :func:`matcher.compute_experiment_metrics`.

    Returns:
        Comprehensive results dict with per-model CIs, pairwise comparisons,
        and effect sizes.
    """
    model_names = sorted(metrics.keys())
    results: dict = {"per_model": {}, "pairwise_comparisons": []}

    # Per-model bootstrap CIs.
    for model_name in model_names:
        per_paper = metrics[model_name].get("per_paper", [])
        precisions = [p["precision"] for p in per_paper]
        recalls = [p["recall"] for p in per_paper]
        f1s = [p["f1"] for p in per_paper]

        results["per_model"][model_name] = {
            "precision_ci": bootstrap_ci(precisions),
            "recall_ci": bootstrap_ci(recalls),
            "f1_ci": bootstrap_ci(f1s),
            "n_papers": len(per_paper),
        }

    # Pairwise comparisons between models.
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i + 1 :]:
            per_a = metrics[model_a].get("per_paper", [])
            per_b = metrics[model_b].get("per_paper", [])

            f1_a = [p["f1"] for p in per_a]
            f1_b = [p["f1"] for p in per_b]

            # Effect size on F1.
            d = cohens_d(f1_a, f1_b)

            # McNemar's on per-claim correctness (using F1 > 0 as proxy).
            min_len = min(len(f1_a), len(f1_b))
            a_correct = [f > 0.5 for f in f1_a[:min_len]]
            b_correct = [f > 0.5 for f in f1_b[:min_len]]
            mcnemar = mcnemar_test(a_correct, b_correct) if min_len > 0 else {
                "statistic": 0.0, "p_value": 1.0, "significant": False
            }

            results["pairwise_comparisons"].append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "cohens_d_f1": d,
                    "mcnemar": mcnemar,
                }
            )

    return results
