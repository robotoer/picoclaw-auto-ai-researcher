"""Visualization for E01 experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "font.family": "sans-serif",
    }
)


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save a figure as both PDF and PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf", format="pdf")
    fig.savefig(output_dir / f"{name}.png", format="png")
    plt.close(fig)


def _bootstrap_ci(
    values: list[float],
    n_boot: int = 10_000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if not values:
        return (0.0, 0.0)
    arr = np.array(values)
    rng = np.random.default_rng(42)
    boot_means = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    )
    alpha = 1.0 - ci
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def _binomial_ci(
    successes: int,
    total: int,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Compute Wilson score binomial confidence interval."""
    if total == 0:
        return (0.0, 0.0)
    from scipy import stats as scipy_stats  # noqa: PLC0415

    z = scipy_stats.norm.ppf(1 - (1 - ci) / 2)
    p_hat = successes / total
    denom = 1 + z**2 / total
    centre = (p_hat + z**2 / (2 * total)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def plot_precision_recall_by_model(
    metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    """Grouped bar chart of precision, recall, and F1 per model with bootstrap CI error bars."""
    model_results: dict[str, dict[str, Any]] = metrics.get("per_model", {})
    if not model_results:
        return

    model_names = list(model_results.keys())
    metric_names = ["precision", "recall", "f1"]
    n_models = len(model_names)
    n_metrics = len(metric_names)

    fig, ax = plt.subplots(figsize=(max(8, n_models * 2.5), 5))
    x = np.arange(n_models)
    width = 0.25
    colors = ["#2176AE", "#57B8FF", "#B7E0FF"]

    for i, metric_name in enumerate(metric_names):
        means: list[float] = []
        ci_low: list[float] = []
        ci_high: list[float] = []
        for model in model_names:
            per_paper_values: list[float] = model_results[model].get(
                f"per_paper_{metric_name}", []
            )
            if per_paper_values:
                mean_val = float(np.mean(per_paper_values))
                lo, hi = _bootstrap_ci(per_paper_values)
            else:
                mean_val = model_results[model].get(metric_name, 0.0)
                lo, hi = mean_val, mean_val
            means.append(mean_val)
            ci_low.append(mean_val - lo)
            ci_high.append(hi - mean_val)

        offset = (i - (n_metrics - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=[ci_low, ci_high],
            label=metric_name.capitalize(),
            color=colors[i],
            capsize=3,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_dir, "precision_recall_f1_by_model")


def plot_claim_type_breakdown(
    metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    """Stacked bar chart of extraction performance by claim type per model."""
    per_type: dict[str, dict[str, Any]] = metrics.get("per_type", {})
    if not per_type:
        return

    claim_types = sorted(per_type.keys())
    # Gather the set of models across all claim types
    all_models: set[str] = set()
    for ct_data in per_type.values():
        all_models.update(ct_data.keys())
    model_names = sorted(all_models)

    if not model_names or not claim_types:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 2.5), 5))
    x = np.arange(len(model_names))
    n_types = len(claim_types)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_types, 3)))  # type: ignore[attr-defined]

    bottoms = np.zeros(len(model_names))
    for i, ct in enumerate(claim_types):
        values: list[float] = []
        for model in model_names:
            values.append(per_type[ct].get(model, {}).get("f1", 0.0))
        arr = np.array(values)
        ax.bar(
            x,
            arr,
            bottom=bottoms,
            label=ct.capitalize(),
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms += arr

    ax.set_ylabel("Cumulative F1")
    ax.set_title("Extraction Performance by Claim Type")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_figure(fig, output_dir, "claim_type_breakdown")


def plot_hallucination_rates(
    metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    """Bar chart of hallucination rate per model with binomial CI error bars."""
    model_results: dict[str, dict[str, Any]] = metrics.get("per_model", {})
    if not model_results:
        return

    model_names: list[str] = []
    rates: list[float] = []
    ci_low: list[float] = []
    ci_high: list[float] = []

    for model, data in model_results.items():
        fp = int(data.get("false_positives", 0))
        total_extracted = int(data.get("total_extracted", 0))
        if total_extracted == 0:
            continue
        rate = fp / total_extracted
        lo, hi = _binomial_ci(fp, total_extracted)
        model_names.append(model)
        rates.append(rate)
        ci_low.append(rate - lo)
        ci_high.append(hi - rate)

    if not model_names:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 2), 5))
    x = np.arange(len(model_names))
    ax.bar(
        x,
        rates,
        yerr=[ci_low, ci_high],
        color="#E63946",
        capsize=4,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_ylabel("Hallucination Rate")
    ax.set_title("Hallucination Rate by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, min(1.05, max(rates) * 1.5 + 0.05) if rates else 1.05)
    fig.tight_layout()
    _save_figure(fig, output_dir, "hallucination_rates")


def plot_extraction_stability(
    metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    """Box plot of F1 across 3 runs per model (shows variance)."""
    stability: dict[str, list[list[float]]] = metrics.get("stability", {})
    if not stability:
        return

    model_names: list[str] = []
    all_f1s: list[list[float]] = []

    for model, runs in stability.items():
        # runs is a list of per-paper F1 lists, one per run
        flat: list[float] = []
        for run_f1s in runs:
            if isinstance(run_f1s, list):
                flat.extend(run_f1s)
            else:
                flat.append(float(run_f1s))
        if flat:
            model_names.append(model)
            all_f1s.append(flat)

    if not model_names:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 2), 5))
    bp = ax.boxplot(
        all_f1s,
        labels=model_names,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    colors = ["#2176AE", "#57B8FF", "#B7E0FF", "#A8DADC", "#457B9D"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    ax.set_ylabel("F1 Score")
    ax.set_title("Extraction Stability Across Runs")
    ax.set_ylim(0, 1.05)
    if len(model_names) > 3:
        ax.set_xticklabels(model_names, rotation=15, ha="right")
    fig.tight_layout()
    _save_figure(fig, output_dir, "extraction_stability")


def plot_granularity_comparison(
    metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    """Histogram of claim length (tokens) for human vs each model."""
    granularity: dict[str, list[int]] = metrics.get("claim_lengths", {})
    if not granularity:
        return

    sources = list(granularity.keys())
    if not sources:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]
    all_lengths: list[int] = []
    for lengths in granularity.values():
        all_lengths.extend(lengths)

    if not all_lengths:
        plt.close(fig)
        return

    max_len = max(all_lengths)
    bins = np.linspace(0, max_len + 1, min(30, max_len + 1))

    for i, source in enumerate(sources):
        lengths = granularity[source]
        if lengths:
            ax.hist(
                lengths,
                bins=bins,
                alpha=0.6,
                label=source,
                color=colors[i % len(colors)],
                edgecolor="white",
                linewidth=0.5,
            )

    ax.set_xlabel("Claim Length (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("Claim Granularity: Token Length Distribution")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_dir, "granularity_comparison")


def generate_all_plots(
    metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    """Run all visualization functions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_fns = [
        plot_precision_recall_by_model,
        plot_claim_type_breakdown,
        plot_hallucination_rates,
        plot_extraction_stability,
        plot_granularity_comparison,
    ]

    for fn in plot_fns:
        try:
            fn(metrics, output_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: {fn.__name__} failed: {exc}")
