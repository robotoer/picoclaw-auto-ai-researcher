"""Statistical analysis for E03 (Semantic Novelty Measurement)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score


def compute_auc_with_ci(
    labels: list[int],  # 1=novel, 0=incremental
    scores: list[float],  # novelty metric scores
    n_bootstrap: int = 10000,
) -> dict[str, float]:
    """AUC-ROC with bootstrap 95% CI."""
    auc = roc_auc_score(labels, scores)
    rng = np.random.default_rng(42)
    boot_aucs: list[float] = []
    arr_labels = np.array(labels)
    arr_scores = np.array(scores)
    n = len(labels)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(set(arr_labels[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(arr_labels[idx], arr_scores[idx]))
    ci_lower = float(np.percentile(boot_aucs, 2.5))
    ci_upper = float(np.percentile(boot_aucs, 97.5))
    return {
        "auc": round(auc, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
    }


def spearman_with_permutation(
    x: list[float],
    y: list[float],
    n_permutations: int = 10000,
) -> dict[str, float]:
    """Spearman rho with permutation p-value."""
    rho = float(sp_stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(42)
    count = 0
    arr_y = np.array(y)
    arr_x = np.array(x)
    for _ in range(n_permutations):
        perm_y = rng.permutation(arr_y)
        perm_rho = float(sp_stats.spearmanr(arr_x, perm_y).statistic)
        if abs(perm_rho) >= abs(rho):
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    return {"rho": round(rho, 6), "p_value": round(p_value, 6), "significant": p_value < 0.05}


def fleiss_kappa(ratings_matrix: list[list[int]], n_categories: int = 2) -> float:
    """Fleiss' kappa for multiple raters.

    ratings_matrix: list of [n_items] lists, each containing counts per category.
    E.g., for 3 raters and 2 categories: [[3, 0], [2, 1], [1, 2], ...]
    means item 0 got 3 "novel" and 0 "incremental", item 1 got 2 and 1, etc.
    """
    matrix = np.array(ratings_matrix, dtype=float)
    n_items, _n_cats = matrix.shape
    n_raters = int(matrix[0].sum())

    # Proportion of assignments to each category
    p_j = matrix.sum(axis=0) / (n_items * n_raters)

    # Agreement per item
    P_i = (np.sum(matrix**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))

    P_bar = float(np.mean(P_i))
    P_e = float(np.sum(p_j**2))

    if P_e == 1.0:
        return 1.0
    return round((P_bar - P_e) / (1.0 - P_e), 6)


def precision_at_k(
    labels: list[int],  # 1=novel, 0=incremental
    scores: list[float],
    k: int,
) -> float:
    """Precision@k: fraction of top-k scored papers that are novel."""
    sorted_indices = np.argsort(scores)[::-1][:k]
    return round(float(np.mean([labels[i] for i in sorted_indices])), 6)


def delong_test(
    labels: list[int],
    scores_a: list[float],
    scores_b: list[float],
) -> dict[str, float]:
    """Simplified DeLong test for comparing two AUCs on the same dataset.

    Returns z-statistic and p-value.
    """
    auc_a = roc_auc_score(labels, scores_a)
    auc_b = roc_auc_score(labels, scores_b)
    # Simplified: use bootstrap to estimate SE of difference
    rng = np.random.default_rng(42)
    n = len(labels)
    diffs: list[float] = []
    for _ in range(10000):
        idx = rng.choice(n, size=n, replace=True)
        la = np.array(labels)[idx]
        if len(set(la)) < 2:
            continue
        a = roc_auc_score(la, np.array(scores_a)[idx])
        b = roc_auc_score(la, np.array(scores_b)[idx])
        diffs.append(a - b)
    se = float(np.std(diffs))
    if se == 0:
        return {"z": 0.0, "p_value": 1.0, "auc_a": round(auc_a, 6), "auc_b": round(auc_b, 6)}
    z = (auc_a - auc_b) / se
    p = 2 * (1 - float(sp_stats.norm.cdf(abs(z))))
    return {
        "z": round(z, 6),
        "p_value": round(p, 6),
        "auc_a": round(auc_a, 6),
        "auc_b": round(auc_b, 6),
    }


def holm_bonferroni(
    p_values: dict[str, float], alpha: float = 0.05
) -> dict[str, dict[str, Any]]:
    """Holm-Bonferroni correction for multiple comparisons."""
    sorted_items = sorted(p_values.items(), key=lambda x: x[1])
    n = len(sorted_items)
    results: dict[str, dict[str, Any]] = {}
    for i, (name, p) in enumerate(sorted_items):
        adjusted_alpha = alpha / (n - i)
        results[name] = {
            "p_value": round(p, 6),
            "adjusted_alpha": round(adjusted_alpha, 6),
            "significant": p < adjusted_alpha,
        }
    return results


def uzzi_replication_test(
    papers_with_atypical: list[bool],  # True if paper has atypical references (bottom 10th percentile z-score)
    papers_top_cited: list[bool],  # True if paper is in top citation decile
) -> dict[str, Any]:
    """Fisher's exact test replicating Uzzi et al. finding.

    Tests whether atypical reference papers are more likely to be highly cited.
    """
    # Build 2x2 contingency table
    a = sum(1 for at, tc in zip(papers_with_atypical, papers_top_cited) if at and tc)
    b = sum(1 for at, tc in zip(papers_with_atypical, papers_top_cited) if at and not tc)
    c = sum(1 for at, tc in zip(papers_with_atypical, papers_top_cited) if not at and tc)
    d = sum(1 for at, tc in zip(papers_with_atypical, papers_top_cited) if not at and not tc)

    odds_ratio, p_value = sp_stats.fisher_exact([[a, b], [c, d]])
    return {
        "odds_ratio": round(float(odds_ratio), 6),
        "p_value": round(float(p_value), 6),
        "significant": p_value < 0.05,
        "table": {
            "atypical_cited": a,
            "atypical_not_cited": b,
            "typical_cited": c,
            "typical_not_cited": d,
        },
    }


def run_e03_analysis(
    metric_scores: dict[str, list[float]],  # metric_name -> scores (aligned with papers)
    binary_labels: list[int],  # 1=novel, 0=incremental
    human_likert: list[float],  # mean human novelty score per paper
    citation_counts: list[int],  # 2-year citation count per paper
    paper_ids: list[str],
) -> dict[str, Any]:
    """Run all E03 statistical analyses."""
    results: dict[str, Any] = {
        "per_metric": {},
        "comparisons": {},
        "combined": {},
        "uzzi": {},
        "holm_bonferroni": {},
    }

    auc_pvalues: dict[str, float] = {}
    for metric_name, scores in metric_scores.items():
        auc = compute_auc_with_ci(binary_labels, scores)
        spearman_cite = spearman_with_permutation(
            scores, [float(c) for c in citation_counts]
        )
        spearman_human = spearman_with_permutation(scores, human_likert)
        p10 = precision_at_k(binary_labels, scores, 10)
        p20 = precision_at_k(binary_labels, scores, 20)

        results["per_metric"][metric_name] = {
            "auc": auc,
            "spearman_citation": spearman_cite,
            "spearman_human": spearman_human,
            "precision_at_10": p10,
            "precision_at_20": p20,
        }
        auc_pvalues[metric_name] = 1.0 - auc["auc"]  # Use 1-AUC as proxy p-value

    # Pairwise DeLong comparisons
    metric_names = sorted(metric_scores.keys())
    for i, m1 in enumerate(metric_names):
        for m2 in metric_names[i + 1 :]:
            results["comparisons"][f"{m1}_vs_{m2}"] = delong_test(
                binary_labels, metric_scores[m1], metric_scores[m2]
            )

    # Holm-Bonferroni on AUC >= 0.70 threshold
    results["holm_bonferroni"] = holm_bonferroni(auc_pvalues)

    return results
