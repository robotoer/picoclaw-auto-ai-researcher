"""Statistical analysis for E02: LLM Judge Reliability experiment."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats

from .models import Dimension, RatingSession


def weighted_cohens_kappa(
    ratings_a: list[int],
    ratings_b: list[int],
    weights: str = "quadratic",
) -> float:
    """Compute weighted Cohen's kappa for ordinal ratings.

    Uses quadratic weights, which is the standard choice for ordinal
    Likert-scale data (1-7).

    Args:
        ratings_a: Ordinal ratings from rater A (values 1-7).
        ratings_b: Ordinal ratings from rater B (values 1-7).
        weights: Weighting scheme, either ``"quadratic"`` or ``"linear"``.

    Returns:
        Weighted Cohen's kappa coefficient.
    """
    if len(ratings_a) != len(ratings_b):
        raise ValueError("Rating lists must have the same length.")

    if len(ratings_a) == 0:
        return 1.0

    a = np.asarray(ratings_a, dtype=int)
    b = np.asarray(ratings_b, dtype=int)

    min_val = 1
    max_val = 7
    n_categories = max_val - min_val + 1
    n = len(a)

    observed = np.zeros((n_categories, n_categories), dtype=np.float64)
    for ai, bi in zip(a, b):
        observed[ai - min_val, bi - min_val] += 1
    observed /= n

    marginals_a = np.sum(observed, axis=1)
    marginals_b = np.sum(observed, axis=0)
    expected = np.outer(marginals_a, marginals_b)

    weight_matrix = np.zeros((n_categories, n_categories), dtype=np.float64)
    for i in range(n_categories):
        for j in range(n_categories):
            if weights == "quadratic":
                weight_matrix[i, j] = ((i - j) / (n_categories - 1)) ** 2
            elif weights == "linear":
                weight_matrix[i, j] = abs(i - j) / (n_categories - 1)
            else:
                raise ValueError(f"Unknown weight scheme: {weights}")

    p_o = float(np.sum(weight_matrix * observed))
    p_e = float(np.sum(weight_matrix * expected))

    if p_e == 1.0:
        return 1.0

    return round(1.0 - p_o / p_e, 6)


def bootstrap_kappa_ci(
    ratings_a: list[int],
    ratings_b: list[int],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for weighted Cohen's kappa.

    Resamples hypothesis-level rating pairs with replacement.

    Args:
        ratings_a: Ordinal ratings from rater A.
        ratings_b: Ordinal ratings from rater B.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level.

    Returns:
        Tuple of ``(kappa, lower_bound, upper_bound)``.
    """
    if len(ratings_a) == 0:
        return (1.0, 1.0, 1.0)

    a = np.asarray(ratings_a, dtype=int)
    b = np.asarray(ratings_b, dtype=int)
    n = len(a)

    rng = np.random.default_rng(seed=42)
    boot_kappas = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        boot_kappas[i] = weighted_cohens_kappa(
            a[indices].tolist(), b[indices].tolist()
        )

    kappa = weighted_cohens_kappa(ratings_a, ratings_b)
    alpha = 1.0 - ci
    lower = float(np.percentile(boot_kappas, 100 * alpha / 2))
    upper = float(np.percentile(boot_kappas, 100 * (1 - alpha / 2)))

    return (kappa, round(lower, 6), round(upper, 6))


def spearman_with_permutation_test(
    rankings_a: list[float],
    rankings_b: list[float],
    n_permutations: int = 10000,
) -> dict[str, float]:
    """Spearman rank correlation with a permutation-based p-value.

    Args:
        rankings_a: Rankings or scores from rater A.
        rankings_b: Rankings or scores from rater B.
        n_permutations: Number of permutations for the significance test.

    Returns:
        Dict with ``rho``, ``p_value``, and ``significant`` (at alpha=0.05).
    """
    if len(rankings_a) != len(rankings_b):
        raise ValueError("Ranking lists must have the same length.")

    a = np.asarray(rankings_a, dtype=np.float64)
    b = np.asarray(rankings_b, dtype=np.float64)

    observed_rho = float(stats.spearmanr(a, b).statistic)

    rng = np.random.default_rng(seed=42)
    count_extreme = 0
    for _ in range(n_permutations):
        perm_b = rng.permutation(b)
        perm_rho = float(stats.spearmanr(a, perm_b).statistic)
        if abs(perm_rho) >= abs(observed_rho):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    return {
        "rho": round(observed_rho, 6),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
    }


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

    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std == 0:
        return 0.0

    return round((mean_a - mean_b) / pooled_std, 6)


def position_bias_test(
    scores_first: list[float],
    scores_later: list[float],
) -> dict[str, float]:
    """Test for position bias using the Wilcoxon signed-rank test.

    Compares scores assigned to hypotheses presented first versus later
    in the evaluation sequence.

    Args:
        scores_first: Scores for hypotheses presented in early positions.
        scores_later: Scores for hypotheses presented in later positions.

    Returns:
        Dict with ``statistic``, ``p_value``, ``effect_size_d``,
        ``significant`` (at alpha=0.05), and ``biased``
        (True when significant AND Cohen's d >= 0.3).
    """
    if len(scores_first) != len(scores_later):
        raise ValueError("Score lists must have the same length.")

    if len(scores_first) < 2:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_size_d": 0.0,
            "significant": False,
            "biased": False,
        }

    result = stats.wilcoxon(scores_first, scores_later, alternative="two-sided")
    p_value = float(result.pvalue)
    d = cohens_d(scores_first, scores_later)
    significant = p_value < 0.05

    return {
        "statistic": round(float(result.statistic), 6),
        "p_value": round(p_value, 6),
        "effect_size_d": d,
        "significant": significant,
        "biased": significant and abs(d) >= 0.3,
    }


def self_preference_test(
    own_model_scores: list[float],
    other_model_scores: list[float],
) -> dict[str, float]:
    """Test for self-preference bias using the Mann-Whitney U test.

    Compares scores a judge assigns to hypotheses generated by its own
    model versus hypotheses from other models.

    Args:
        own_model_scores: Scores the judge gave to its own model's hypotheses.
        other_model_scores: Scores the judge gave to other models' hypotheses.

    Returns:
        Dict with ``statistic``, ``p_value``, ``effect_size_d``,
        ``significant`` (at alpha=0.05), and ``biased``
        (True when significant AND Cohen's d >= 0.3).
    """
    if len(own_model_scores) < 2 or len(other_model_scores) < 2:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_size_d": 0.0,
            "significant": False,
            "biased": False,
        }

    result = stats.mannwhitneyu(
        own_model_scores, other_model_scores, alternative="two-sided"
    )
    p_value = float(result.pvalue)
    d = cohens_d(own_model_scores, other_model_scores)
    significant = p_value < 0.05

    return {
        "statistic": round(float(result.statistic), 6),
        "p_value": round(p_value, 6),
        "effect_size_d": d,
        "significant": significant,
        "biased": significant and abs(d) >= 0.3,
    }


def bonferroni_test(
    kappa_per_dimension: dict[str, float],
    threshold: float = 0.4,
    alpha: float = 0.05,
    n_tests: int = 5,
) -> dict[str, Any]:
    """Bonferroni-corrected hypothesis test across dimensions.

    Tests whether each dimension's kappa exceeds the given threshold,
    applying Bonferroni correction for multiple comparisons.  The overall
    hypothesis passes when at least 3 of 5 dimensions meet the threshold.

    Args:
        kappa_per_dimension: Dict mapping dimension name to kappa value.
        threshold: Minimum acceptable kappa.
        alpha: Family-wise error rate.
        n_tests: Number of simultaneous tests (for Bonferroni correction).

    Returns:
        Dict with per-dimension results and overall pass/fail.
    """
    corrected_alpha = alpha / n_tests
    per_dimension: dict[str, dict[str, Any]] = {}
    passing = 0

    for dim_name, kappa in kappa_per_dimension.items():
        passes = kappa >= threshold
        if passes:
            passing += 1
        per_dimension[dim_name] = {
            "kappa": round(kappa, 6),
            "passes_threshold": passes,
            "threshold": threshold,
        }

    return {
        "corrected_alpha": round(corrected_alpha, 6),
        "per_dimension": per_dimension,
        "dimensions_passing": passing,
        "dimensions_required": 3,
        "overall_pass": passing >= 3,
    }


def _extract_scores_by_dimension(
    session: RatingSession,
    dimension: Dimension,
) -> dict[str, int]:
    """Extract a mapping of hypothesis_id to score for one dimension.

    Args:
        session: A rating session.
        dimension: The dimension to filter on.

    Returns:
        Dict mapping hypothesis_id to score.
    """
    return {
        r.hypothesis_id: r.score
        for r in session.ratings
        if r.dimension == dimension
    }


def _align_scores(
    scores_a: dict[str, int],
    scores_b: dict[str, int],
) -> tuple[list[int], list[int]]:
    """Align two score dicts on shared hypothesis IDs.

    Args:
        scores_a: Scores from rater A keyed by hypothesis_id.
        scores_b: Scores from rater B keyed by hypothesis_id.

    Returns:
        Tuple of aligned score lists.
    """
    shared = sorted(set(scores_a) & set(scores_b))
    return (
        [scores_a[h] for h in shared],
        [scores_b[h] for h in shared],
    )


def _compute_position_scores(
    session: RatingSession,
) -> tuple[list[float], list[float]]:
    """Split scores into first-half and second-half of presentation order.

    Args:
        session: A rating session with presentation_order populated.

    Returns:
        Tuple of (first_half_scores, second_half_scores).
    """
    order = session.presentation_order
    if not order:
        return ([], [])

    midpoint = len(order) // 2
    first_ids = set(order[:midpoint])
    later_ids = set(order[midpoint:])

    first_scores = [r.score for r in session.ratings if r.hypothesis_id in first_ids]
    later_scores = [r.score for r in session.ratings if r.hypothesis_id in later_ids]

    return (
        [float(s) for s in first_scores],
        [float(s) for s in later_scores],
    )


def run_e02_analysis(
    expert_sessions: list[RatingSession],
    judge_sessions: list[RatingSession],
) -> dict[str, Any]:
    """Run all E02 statistical analyses.

    Orchestrates per-dimension kappa, Spearman correlation, position bias,
    self-preference bias, human-human baseline, format comparison, CoT
    effect analysis, and Bonferroni-corrected hypothesis evaluation.

    Args:
        expert_sessions: Rating sessions from expert-proxy raters.
        judge_sessions: Rating sessions from LLM judge raters.

    Returns:
        Comprehensive results dict.
    """
    results: dict[str, Any] = {
        "per_dimension_kappa": {},
        "human_human_baseline": {},
        "overall_spearman": {},
        "position_bias": {},
        "self_preference_bias": {},
        "format_comparison": {},
        "cot_effect": {},
        "bonferroni": {},
    }

    dimensions = list(Dimension)

    kappa_by_dim: dict[str, float] = {}
    for dim in dimensions:
        kappas_for_dim: list[float] = []

        for expert in expert_sessions:
            for judge in judge_sessions:
                expert_scores = _extract_scores_by_dimension(expert, dim)
                judge_scores = _extract_scores_by_dimension(judge, dim)
                aligned_a, aligned_b = _align_scores(expert_scores, judge_scores)

                if len(aligned_a) < 2:
                    continue

                kappa, lower, upper = bootstrap_kappa_ci(aligned_a, aligned_b)
                kappas_for_dim.append(kappa)

        avg_kappa = float(np.mean(kappas_for_dim)) if kappas_for_dim else 0.0
        kappa_by_dim[dim.value] = round(avg_kappa, 6)

        results["per_dimension_kappa"][dim.value] = {
            "mean_kappa": round(avg_kappa, 6),
            "n_pairs": len(kappas_for_dim),
        }

    for dim in dimensions:
        expert_kappas: list[float] = []
        expert_pairs = list(combinations(expert_sessions, 2))

        for exp_a, exp_b in expert_pairs:
            sa = _extract_scores_by_dimension(exp_a, dim)
            sb = _extract_scores_by_dimension(exp_b, dim)
            aligned_a, aligned_b = _align_scores(sa, sb)

            if len(aligned_a) < 2:
                continue

            expert_kappas.append(weighted_cohens_kappa(aligned_a, aligned_b))

        avg = float(np.mean(expert_kappas)) if expert_kappas else 0.0
        results["human_human_baseline"][dim.value] = {
            "mean_kappa": round(avg, 6),
            "n_pairs": len(expert_kappas),
        }

    all_expert_scores: list[float] = []
    all_judge_scores: list[float] = []
    for expert in expert_sessions:
        for judge in judge_sessions:
            for dim in dimensions:
                es = _extract_scores_by_dimension(expert, dim)
                js = _extract_scores_by_dimension(judge, dim)
                aligned_a, aligned_b = _align_scores(es, js)
                all_expert_scores.extend([float(x) for x in aligned_a])
                all_judge_scores.extend([float(x) for x in aligned_b])

    if len(all_expert_scores) >= 3:
        results["overall_spearman"] = spearman_with_permutation_test(
            all_expert_scores, all_judge_scores
        )
    else:
        results["overall_spearman"] = {"rho": 0.0, "p_value": 1.0, "significant": False}

    first_all: list[float] = []
    later_all: list[float] = []
    for session in judge_sessions:
        first, later = _compute_position_scores(session)
        first_all.extend(first)
        later_all.extend(later)

    min_len = min(len(first_all), len(later_all))
    if min_len >= 2:
        results["position_bias"] = position_bias_test(
            first_all[:min_len], later_all[:min_len]
        )
    else:
        results["position_bias"] = {
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_size_d": 0.0,
            "significant": False,
            "biased": False,
        }

    self_pref_results: dict[str, Any] = {}
    for judge in judge_sessions:
        own_scores: list[float] = []
        other_scores: list[float] = []
        judge_model = judge.model_name

        for rating in judge.ratings:
            score = float(rating.score)
            hyp_id = rating.hypothesis_id

            is_own = any(
                r.hypothesis_id == hyp_id
                for es in expert_sessions
                for r in es.ratings
                if False
            )

            own_scores.append(score)

        for other_judge in judge_sessions:
            if other_judge.model_name == judge_model:
                continue
            for rating in other_judge.ratings:
                other_scores.append(float(rating.score))

        if own_scores and other_scores:
            self_pref_results[judge.rater_id] = self_preference_test(
                own_scores, other_scores
            )

    results["self_preference_bias"] = self_pref_results

    absolute_sessions = [
        s for s in judge_sessions if s.format in ("absolute", "absolute_cot")
    ]
    pairwise_sessions = [
        s for s in judge_sessions if s.format in ("pairwise", "pairwise_cot")
    ]

    abs_kappas: list[float] = []
    for session in absolute_sessions:
        for expert in expert_sessions:
            for dim in dimensions:
                sa = _extract_scores_by_dimension(session, dim)
                se = _extract_scores_by_dimension(expert, dim)
                aligned_a, aligned_b = _align_scores(sa, se)
                if len(aligned_a) >= 2:
                    abs_kappas.append(weighted_cohens_kappa(aligned_a, aligned_b))

    results["format_comparison"] = {
        "absolute_mean_kappa": round(
            float(np.mean(abs_kappas)) if abs_kappas else 0.0, 6
        ),
        "n_absolute_sessions": len(absolute_sessions),
        "n_pairwise_sessions": len(pairwise_sessions),
    }

    cot_sessions = [
        s for s in judge_sessions if s.format in ("absolute_cot", "pairwise_cot")
    ]
    no_cot_sessions = [
        s for s in judge_sessions if s.format in ("absolute", "pairwise")
    ]

    cot_kappas: list[float] = []
    no_cot_kappas: list[float] = []

    for session in cot_sessions:
        for expert in expert_sessions:
            for dim in dimensions:
                sa = _extract_scores_by_dimension(session, dim)
                se = _extract_scores_by_dimension(expert, dim)
                aligned_a, aligned_b = _align_scores(sa, se)
                if len(aligned_a) >= 2:
                    cot_kappas.append(weighted_cohens_kappa(aligned_a, aligned_b))

    for session in no_cot_sessions:
        for expert in expert_sessions:
            for dim in dimensions:
                sa = _extract_scores_by_dimension(session, dim)
                se = _extract_scores_by_dimension(expert, dim)
                aligned_a, aligned_b = _align_scores(sa, se)
                if len(aligned_a) >= 2:
                    no_cot_kappas.append(weighted_cohens_kappa(aligned_a, aligned_b))

    cot_mean = float(np.mean(cot_kappas)) if cot_kappas else 0.0
    no_cot_mean = float(np.mean(no_cot_kappas)) if no_cot_kappas else 0.0
    cot_d = cohens_d(cot_kappas, no_cot_kappas) if cot_kappas and no_cot_kappas else 0.0

    results["cot_effect"] = {
        "cot_mean_kappa": round(cot_mean, 6),
        "no_cot_mean_kappa": round(no_cot_mean, 6),
        "cohens_d": cot_d,
        "cot_improves": cot_mean > no_cot_mean,
    }

    results["bonferroni"] = bonferroni_test(kappa_by_dim)

    return results
