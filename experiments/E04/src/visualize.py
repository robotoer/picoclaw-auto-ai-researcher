"""Visualization for E04: Knowledge Graph Consistency."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Consistent color scheme for the 4 conditions
CONDITION_COLORS = {
    "No Filter": "#2196F3",
    "Layer 1": "#4CAF50",
    "Layer 1+2": "#FF9800",
    "Layer 1+2+3": "#F44336",
}
CONDITION_ORDER = ["No Filter", "Layer 1", "Layer 1+2", "Layer 1+2+3"]


def _get_color(condition: str) -> str:
    """Get color for a condition, falling back to gray."""
    return CONDITION_COLORS.get(condition, "#9E9E9E")


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save figure as both PNG and PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(analysis: dict, output_dir: Path) -> None:
    """Generate all E04 plots.

    Args:
        analysis: Output from run_e04_analysis.
        output_dir: Directory to save plots.
    """
    if "contradiction_rates" in analysis and analysis["contradiction_rates"]:
        _plot_contradiction_rates(analysis, output_dir)

    if "growth_curves" in analysis and analysis["growth_curves"]:
        _plot_growth_curves(analysis, output_dir)

    if "hallucination_rates" in analysis and analysis["hallucination_rates"]:
        _plot_hallucination_rates(analysis, output_dir)

    if "extractor_independence" in analysis:
        _plot_extractor_correlation(analysis, output_dir)

    if "provenance_completeness" in analysis and analysis["provenance_completeness"]:
        _plot_provenance_completeness(analysis, output_dir)


def _plot_contradiction_rates(analysis: dict, output_dir: Path) -> None:
    """Bar chart: contradiction rate per condition with CI error bars."""
    rates_data = analysis["contradiction_rates"]

    # Sort by condition order
    sorted_data = sorted(
        rates_data,
        key=lambda d: CONDITION_ORDER.index(d["condition"])
        if d["condition"] in CONDITION_ORDER
        else len(CONDITION_ORDER),
    )

    conditions = [d["condition"] for d in sorted_data]
    rates = [d["rate"] for d in sorted_data]
    ci_lower = [d["rate"] - d["ci_lower"] for d in sorted_data]
    ci_upper = [d["ci_upper"] - d["rate"] for d in sorted_data]
    colors = [_get_color(c) for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(conditions))
    ax.bar(x, rates, color=colors, edgecolor="white", linewidth=0.5)
    ax.errorbar(
        x,
        rates,
        yerr=[ci_lower, ci_upper],
        fmt="none",
        ecolor="black",
        capsize=5,
        linewidth=1.5,
    )

    ax.set_xlabel("Condition")
    ax.set_ylabel("Contradiction Rate")
    ax.set_title("Contradiction Rate by Verification Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add significance annotation from chi-squared test
    if "chi_squared" in analysis:
        chi2_result = analysis["chi_squared"]
        sig_text = (
            f"$\\chi^2$ = {chi2_result['chi2']:.2f}, "
            f"p = {chi2_result['p_value']:.4f}"
        )
        if chi2_result["significant"]:
            sig_text += " *"
        ax.text(
            0.98,
            0.95,
            sig_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            style="italic",
        )

    fig.tight_layout()
    _save_figure(fig, output_dir, "contradiction_rates")


def _plot_growth_curves(analysis: dict, output_dir: Path) -> None:
    """Line plot: contradiction rate vs papers ingested for each condition.

    X-axis: 100, 200, 300, 400, 500 papers. Lines for each condition.
    """
    growth_data = analysis["growth_curves"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for condition in CONDITION_ORDER:
        if condition not in growth_data:
            continue
        regression = growth_data[condition]
        color = _get_color(condition)

        # Plot regression line
        x_range = np.array([100, 200, 300, 400, 500])
        y_pred = regression["slope"] * x_range + regression["intercept"]
        ax.plot(
            x_range,
            y_pred,
            color=color,
            linewidth=2,
            label=f"{condition} (R²={regression['r_squared']:.3f})",
        )

    # Also plot raw checkpoint data if available in the analysis
    if "checkpoint_raw" in analysis:
        for condition, checkpoints in analysis["checkpoint_raw"].items():
            color = _get_color(condition)
            x_pts = [cp["n_papers"] for cp in checkpoints]
            y_pts = [cp["contradiction_rate"] for cp in checkpoints]
            ax.scatter(x_pts, y_pts, color=color, s=40, zorder=5)

    ax.set_xlabel("Papers Ingested")
    ax.set_ylabel("Contradiction Rate")
    ax.set_title("Contradiction Rate Growth as Knowledge Graph Expands")
    ax.set_xticks([100, 200, 300, 400, 500])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_figure(fig, output_dir, "growth_curves")


def _plot_hallucination_rates(analysis: dict, output_dir: Path) -> None:
    """Bar chart: hallucination spot-check rate per condition with CI."""
    hall_data = analysis["hallucination_rates"]

    sorted_data = sorted(
        hall_data,
        key=lambda d: CONDITION_ORDER.index(d["condition"])
        if d["condition"] in CONDITION_ORDER
        else len(CONDITION_ORDER),
    )

    conditions = [d["condition"] for d in sorted_data]
    rates = [d["rate"] for d in sorted_data]
    ci_lower = [d["rate"] - d["ci_lower"] for d in sorted_data]
    ci_upper = [d["ci_upper"] - d["rate"] for d in sorted_data]
    colors = [_get_color(c) for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(conditions))
    ax.bar(x, rates, color=colors, edgecolor="white", linewidth=0.5)
    ax.errorbar(
        x,
        rates,
        yerr=[ci_lower, ci_upper],
        fmt="none",
        ecolor="black",
        capsize=5,
        linewidth=1.5,
    )

    ax.set_xlabel("Condition")
    ax.set_ylabel("Hallucination Rate")
    ax.set_title("Hallucination Rate by Verification Condition (Spot-Check)")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_figure(fig, output_dir, "hallucination_rates")


def _plot_extractor_correlation(analysis: dict, output_dir: Path) -> None:
    """Heatmap of pairwise extractor error correlations."""
    independence = analysis["extractor_independence"]
    correlations = independence.get("correlations", {})

    if not correlations:
        return

    # Extract unique model names from pair keys
    models: list[str] = []
    for key in correlations:
        parts = key.split("_vs_")
        for part in parts:
            if part not in models:
                models.append(part)
    models.sort()

    n = len(models)
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, 1.0)

    for key, phi in correlations.items():
        parts = key.split("_vs_")
        if len(parts) == 2:
            i = models.index(parts[0])
            j = models.index(parts[1])
            matrix[i, j] = phi
            matrix[j, i] = phi

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(models, fontsize=9)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(matrix[i, j]) > 0.5 else "black",
            )

    ax.set_title("Pairwise Extractor Error Correlations (Phi Coefficient)")
    fig.colorbar(im, ax=ax, label="Phi Coefficient")

    # Add effective N annotation
    eff_n = independence.get("effective_n_extractors", 0)
    ax.text(
        0.02,
        -0.12,
        f"Effective independent extractors: {eff_n:.1f}",
        transform=ax.transAxes,
        fontsize=9,
        style="italic",
    )

    fig.tight_layout()
    _save_figure(fig, output_dir, "extractor_correlation")


def _plot_provenance_completeness(analysis: dict, output_dir: Path) -> None:
    """Bar chart: provenance completeness per condition."""
    prov_data = analysis["provenance_completeness"]

    sorted_data = sorted(
        prov_data,
        key=lambda d: CONDITION_ORDER.index(d["condition"])
        if d["condition"] in CONDITION_ORDER
        else len(CONDITION_ORDER),
    )

    conditions = [d["condition"] for d in sorted_data]
    completeness = [d["completeness"] for d in sorted_data]
    colors = [_get_color(c) for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(conditions))
    bars = ax.bar(x, completeness, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, completeness):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Condition")
    ax.set_ylabel("Provenance Completeness")
    ax.set_title("Provenance Completeness by Verification Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_figure(fig, output_dir, "provenance_completeness")
