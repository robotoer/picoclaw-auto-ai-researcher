"""Statistical analysis for E04: Knowledge Graph Consistency."""

from __future__ import annotations

import numpy as np
from scipy import stats


def chi_squared_test(condition_data: list[dict]) -> dict:
    """Chi-squared test comparing contradiction rates across 4 conditions.

    Args:
        condition_data: list of {"condition": str, "n_contradictions": int, "n_total": int}

    Returns:
        Dict with chi2, p_value, significant bool.
    """
    observed = np.array(
        [[d["n_contradictions"], d["n_total"] - d["n_contradictions"]] for d in condition_data]
    )
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "expected": expected.tolist(),
        "significant": bool(p_value < 0.05),
    }


def mcnemar_test(condition_a: list[bool], condition_b: list[bool]) -> dict:
    """McNemar's test for paired binary outcomes (same papers, different conditions).

    Args:
        condition_a: list of True/False for whether each paper's claims had contradictions.
        condition_b: list of True/False for whether each paper's claims had contradictions.

    Returns:
        Dict with chi2, p_value, significant.
    """
    if len(condition_a) != len(condition_b):
        raise ValueError("Condition lists must have equal length (paired observations)")

    a = np.array(condition_a)
    b = np.array(condition_b)

    # Discordant pairs
    b_pos = int(np.sum(~a & b))  # a=0, b=1
    c_pos = int(np.sum(a & ~b))  # a=1, b=0

    n_discordant = b_pos + c_pos
    if n_discordant == 0:
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "significant": False,
            "b": b_pos,
            "c": c_pos,
        }

    # McNemar's chi-squared with continuity correction
    chi2 = (abs(b_pos - c_pos) - 1) ** 2 / (b_pos + c_pos)
    p_value = float(stats.chi2.sf(chi2, df=1))

    return {
        "chi2": float(chi2),
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
        "b": b_pos,
        "c": c_pos,
    }


def exact_binomial_ci(n_successes: int, n_trials: int, alpha: float = 0.05) -> dict:
    """Clopper-Pearson exact binomial 95% CI.

    Args:
        n_successes: Number of successes.
        n_trials: Number of trials.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Dict with rate, ci_lower, ci_upper.
    """
    if n_trials == 0:
        return {"rate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    rate = n_successes / n_trials

    # Clopper-Pearson exact interval
    if n_successes == 0:
        ci_lower = 0.0
    else:
        ci_lower = float(stats.beta.ppf(alpha / 2, n_successes, n_trials - n_successes + 1))

    if n_successes == n_trials:
        ci_upper = 1.0
    else:
        ci_upper = float(stats.beta.ppf(1 - alpha / 2, n_successes + 1, n_trials - n_successes))

    return {
        "rate": float(rate),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def growth_curve_regression(checkpoints: list[dict]) -> dict:
    """Linear regression on contradiction rate vs papers ingested.

    Args:
        checkpoints: list of {"n_papers": int, "contradiction_rate": float}

    Returns:
        Dict with slope, intercept, r_squared, p_value, slope_positive bool.
    """
    x = np.array([cp["n_papers"] for cp in checkpoints], dtype=float)
    y = np.array([cp["contradiction_rate"] for cp in checkpoints], dtype=float)

    if len(x) < 2:
        return {
            "slope": 0.0,
            "intercept": float(y[0]) if len(y) > 0 else 0.0,
            "r_squared": 0.0,
            "p_value": 1.0,
            "slope_positive": False,
        }

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value**2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "slope_positive": bool(slope > 0),
    }


def tetrachoric_correlation(errors_a: list[bool], errors_b: list[bool]) -> dict:
    """Tetrachoric correlation for binary extractor errors.

    Approximates using Pearson phi coefficient (since tetrachoric requires R).

    Args:
        errors_a: Binary error indicators for extractor A.
        errors_b: Binary error indicators for extractor B.

    Returns:
        Dict with phi, interpretation.
    """
    if len(errors_a) != len(errors_b):
        raise ValueError("Error lists must have equal length")

    a = np.array(errors_a, dtype=int)
    b = np.array(errors_b, dtype=int)

    # 2x2 contingency table
    n11 = int(np.sum(a & b))
    n10 = int(np.sum(a & ~b.astype(bool)))
    n01 = int(np.sum(~a.astype(bool) & b))
    n00 = int(np.sum(~a.astype(bool) & ~b.astype(bool)))

    n = n11 + n10 + n01 + n00
    if n == 0:
        return {"phi": 0.0, "interpretation": "no data"}

    # Phi coefficient
    denom = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if denom == 0:
        phi = 0.0
    else:
        phi = (n11 * n00 - n10 * n01) / denom

    # Interpret
    abs_phi = abs(phi)
    if abs_phi < 0.1:
        interpretation = "negligible"
    elif abs_phi < 0.3:
        interpretation = "weak"
    elif abs_phi < 0.5:
        interpretation = "moderate"
    else:
        interpretation = "strong"

    return {
        "phi": float(phi),
        "interpretation": interpretation,
        "contingency_table": {"n11": n11, "n10": n10, "n01": n01, "n00": n00},
    }


def extractor_independence_analysis(
    extractor_results: dict[str, list[bool]],
) -> dict:
    """Analyze whether extractor errors are independent.

    Compute pairwise correlations, observed vs theoretical consensus error rate.

    Args:
        extractor_results: model name -> list of error indicators per paper.

    Returns:
        Dict with correlations, observed_rate, theoretical_rate, effective_n_extractors.
    """
    models = sorted(extractor_results.keys())
    n_models = len(models)

    if n_models < 2:
        return {
            "correlations": {},
            "observed_rate": 0.0,
            "theoretical_rate": 0.0,
            "effective_n_extractors": n_models,
        }

    # Per-model error rates
    error_rates: dict[str, float] = {}
    for model in models:
        errors = extractor_results[model]
        error_rates[model] = sum(errors) / len(errors) if errors else 0.0

    # Pairwise phi correlations
    correlations: dict[str, float] = {}
    phi_values: list[float] = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            pair_key = f"{models[i]}_vs_{models[j]}"
            result = tetrachoric_correlation(
                extractor_results[models[i]], extractor_results[models[j]]
            )
            correlations[pair_key] = result["phi"]
            phi_values.append(result["phi"])

    mean_phi = float(np.mean(phi_values)) if phi_values else 0.0

    # Observed consensus error rate (majority vote)
    n_papers = len(next(iter(extractor_results.values())))
    consensus_errors = 0
    for idx in range(n_papers):
        votes = sum(1 for model in models if extractor_results[model][idx])
        if votes > n_models / 2:
            consensus_errors += 1
    observed_rate = consensus_errors / n_papers if n_papers > 0 else 0.0

    # Theoretical rate assuming independence
    mean_error_rate = float(np.mean(list(error_rates.values())))
    threshold = n_models // 2 + 1
    # P(majority errors) under independence: sum of binomial probabilities
    theoretical_rate = float(
        sum(
            stats.binom.pmf(k, n_models, mean_error_rate)
            for k in range(threshold, n_models + 1)
        )
    )

    # Effective number of independent extractors (Spearman-Brown-like estimate)
    if mean_phi >= 1.0:
        effective_n = 1.0
    elif mean_phi <= 0.0:
        effective_n = float(n_models)
    else:
        effective_n = n_models / (1 + (n_models - 1) * mean_phi)

    return {
        "correlations": correlations,
        "error_rates": error_rates,
        "mean_phi": mean_phi,
        "observed_rate": observed_rate,
        "theoretical_rate": theoretical_rate,
        "effective_n_extractors": float(effective_n),
        "n_models": n_models,
    }


def run_e04_analysis(
    condition_results: list[dict],
    extractor_errors: dict[str, list[bool]] | None = None,
) -> dict:
    """Run all E04 statistical analyses.

    Args:
        condition_results: one dict per condition with keys:
            - condition: str (condition name)
            - n_contradictions: int
            - n_total: int
            - paper_contradictions: list[bool] (per-paper contradiction indicator)
            - checkpoints: list[{"n_papers": int, "contradiction_rate": float}]
            - n_hallucinations: int (spot-check hallucination count)
            - n_spot_checks: int (total spot checks)
            - provenance_completeness: float (0-1)
        extractor_errors: optional model -> list of error indicators per paper.

    Returns:
        Comprehensive analysis dict.
    """
    analysis: dict = {}

    # Normalize field names: accept both n_total and n_claims_total
    for r in condition_results:
        if "n_total" not in r and "n_claims_total" in r:
            r["n_total"] = r["n_claims_total"]
        if "checkpoints" not in r and "growth_checkpoints" in r:
            r["checkpoints"] = r["growth_checkpoints"]

    # 1. Chi-squared test across conditions
    chi2_data = [
        {
            "condition": r["condition"],
            "n_contradictions": r["n_contradictions"],
            "n_total": r["n_total"],
        }
        for r in condition_results
    ]
    analysis["chi_squared"] = chi_squared_test(chi2_data)

    # 2. Contradiction rates with CIs
    analysis["contradiction_rates"] = []
    for r in condition_results:
        ci = exact_binomial_ci(r["n_contradictions"], r["n_total"])
        analysis["contradiction_rates"].append(
            {
                "condition": r["condition"],
                **ci,
            }
        )

    # 3. Pairwise McNemar tests
    analysis["pairwise_mcnemar"] = {}
    for i in range(len(condition_results)):
        for j in range(i + 1, len(condition_results)):
            a = condition_results[i]
            b = condition_results[j]
            pair_key = f"{a['condition']}_vs_{b['condition']}"
            if "paper_contradictions" in a and "paper_contradictions" in b:
                analysis["pairwise_mcnemar"][pair_key] = mcnemar_test(
                    a["paper_contradictions"], b["paper_contradictions"]
                )

    # 4. Growth curve regressions
    analysis["growth_curves"] = {}
    for r in condition_results:
        if "checkpoints" in r and r["checkpoints"]:
            analysis["growth_curves"][r["condition"]] = growth_curve_regression(r["checkpoints"])

    # 5. Hallucination rates with CIs
    analysis["hallucination_rates"] = []
    for r in condition_results:
        n_halluc = r.get("n_hallucinations", 0)
        n_spot = r.get("n_spot_checks", 50)  # default 50 spot-checks
        ci = exact_binomial_ci(n_halluc, n_spot)
        analysis["hallucination_rates"].append(
            {
                "condition": r["condition"],
                **ci,
            }
        )

    # 6. Provenance completeness
    analysis["provenance_completeness"] = []
    for r in condition_results:
        if "provenance_completeness" in r:
            analysis["provenance_completeness"].append(
                {
                    "condition": r["condition"],
                    "completeness": r["provenance_completeness"],
                }
            )

    # 7. Extractor independence analysis
    if extractor_errors is not None:
        analysis["extractor_independence"] = extractor_independence_analysis(extractor_errors)

    return analysis
