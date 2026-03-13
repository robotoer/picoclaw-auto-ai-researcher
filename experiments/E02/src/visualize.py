"""Visualization for E02: LLM Judge Reliability experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

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


# ---------------------------------------------------------------------------
# Plot 1: Heatmap — Kappa by dimension and model
# ---------------------------------------------------------------------------


def plot_kappa_heatmap(
    analysis: dict[str, Any],
    output_dir: Path,
) -> None:
    """Heatmap of Cohen's kappa values: dimensions (rows) x judge models (columns)."""
    per_model_kappa: dict[str, dict[str, float]] = analysis.get("per_model_kappa", {})
    if not per_model_kappa:
        return

    dimensions = ["novelty", "feasibility", "importance", "clarity", "specificity"]
    models = list(per_model_kappa.keys())

    if not models:
        return

    # Build the data matrix: rows = dimensions, cols = models.
    data = np.zeros((len(dimensions), len(models)))
    for j, model in enumerate(models):
        dim_kappas = per_model_kappa[model]
        for i, dim in enumerate(dimensions):
            data[i, j] = dim_kappas.get(dim, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 2.2), 5))

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-0.2, vmax=1.0)

    # Label axes.
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(dimensions)))
    ax.set_xticklabels([m.split("/")[-1] for m in models], rotation=25, ha="right")
    ax.set_yticklabels([d.capitalize() for d in dimensions])

    # Annotate cells with kappa values.
    for i in range(len(dimensions)):
        for j in range(len(models)):
            value = data[i, j]
            text_color = "white" if value < 0.3 or value > 0.8 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color,
                    fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cohen's Kappa")

    ax.set_title("Inter-Rater Agreement (Kappa) by Dimension and Model")
    # Restore spines for the heatmap.
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout()
    _save_figure(fig, output_dir, "kappa_heatmap")


# ---------------------------------------------------------------------------
# Plot 2: Bar chart — Overall Spearman rho per model
# ---------------------------------------------------------------------------


def plot_spearman_rho(
    analysis: dict[str, Any],
    output_dir: Path,
) -> None:
    """Bar chart of overall Spearman rho per judge model, with error bars if available."""
    spearman_rho: dict[str, Any] = analysis.get("spearman_rho", {})
    if not spearman_rho:
        return

    models: list[str] = []
    rhos: list[float] = []
    ci_low: list[float] = []
    ci_high: list[float] = []

    for model, rho_data in spearman_rho.items():
        models.append(model)
        if isinstance(rho_data, dict):
            rho = rho_data.get("rho", 0.0)
            lo = rho_data.get("ci_lower", rho)
            hi = rho_data.get("ci_upper", rho)
            rhos.append(rho)
            ci_low.append(rho - lo)
            ci_high.append(hi - rho)
        else:
            rhos.append(float(rho_data))
            ci_low.append(0.0)
            ci_high.append(0.0)

    if not models:
        return

    colors = ["#2176AE", "#57B8FF", "#B7E0FF", "#A8DADC", "#457B9D"]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 2), 5))
    x = np.arange(len(models))

    has_errors = any(lo > 0 or hi > 0 for lo, hi in zip(ci_low, ci_high))
    yerr = [ci_low, ci_high] if has_errors else None

    bars = ax.bar(
        x,
        rhos,
        yerr=yerr,
        color=[colors[i % len(colors)] for i in range(len(models))],
        capsize=4,
        edgecolor="white",
        linewidth=0.5,
    )

    # Add value labels on bars.
    for bar, rho in zip(bars, rhos):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rho:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Spearman's Rho")
    ax.set_title("Overall Rank Correlation with Expert Proxy")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("/")[-1] for m in models], rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.7, color="#E63946", linestyle="--", linewidth=1, alpha=0.7,
               label="Threshold (0.70)")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_dir, "spearman_rho_by_model")


# ---------------------------------------------------------------------------
# Plot 3: Grouped bar chart — Format comparison
# ---------------------------------------------------------------------------


def plot_format_comparison(
    analysis: dict[str, Any],
    output_dir: Path,
) -> None:
    """Grouped bar chart comparing absolute vs pairwise, with/without CoT, showing kappa."""
    format_comparison: dict[str, Any] = analysis.get("format_comparison", {})
    if not format_comparison:
        return

    # Expected formats: absolute, absolute_cot, pairwise, pairwise_cot.
    format_labels = ["absolute", "absolute_cot", "pairwise", "pairwise_cot"]
    display_labels = ["Absolute", "Absolute + CoT", "Pairwise", "Pairwise + CoT"]

    # Group by base format for side-by-side display.
    present_formats = [f for f in format_labels if f in format_comparison]
    if not present_formats:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(present_formats) * 1.8), 5))

    # If per-model data exists, show grouped bars; otherwise show aggregate.
    # Check if format_comparison values are dicts with per-model data.
    sample_val = format_comparison[present_formats[0]]
    has_per_model = isinstance(sample_val, dict) and "per_model" in sample_val

    if has_per_model:
        # Grouped bars: one group per format, bars per model.
        all_models: set[str] = set()
        for fmt in present_formats:
            per_model = format_comparison[fmt].get("per_model", {})
            all_models.update(per_model.keys())
        model_list = sorted(all_models)

        x = np.arange(len(present_formats))
        n_models = len(model_list)
        width = 0.8 / max(n_models, 1)
        colors = ["#2176AE", "#57B8FF", "#E9C46A", "#E76F51", "#264653"]

        for m_idx, model in enumerate(model_list):
            kappas: list[float] = []
            for fmt in present_formats:
                per_model = format_comparison[fmt].get("per_model", {})
                kappas.append(per_model.get(model, {}).get("kappa", 0.0))

            offset = (m_idx - (n_models - 1) / 2) * width
            ax.bar(
                x + offset,
                kappas,
                width,
                label=model.split("/")[-1],
                color=colors[m_idx % len(colors)],
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [display_labels[format_labels.index(f)] for f in present_formats],
            rotation=15,
            ha="right",
        )
        ax.legend()
    else:
        # Simple bars: one per format.
        x = np.arange(len(present_formats))
        kappas = []
        for fmt in present_formats:
            val = format_comparison[fmt]
            if isinstance(val, dict):
                kappas.append(val.get("mean_kappa", 0.0))
            else:
                kappas.append(float(val))

        bar_colors = ["#2176AE", "#57B8FF", "#E9C46A", "#E76F51"]
        bars = ax.bar(
            x,
            kappas,
            color=[bar_colors[i % len(bar_colors)] for i in range(len(x))],
            edgecolor="white",
            linewidth=0.5,
        )

        for bar, kappa in zip(bars, kappas):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{kappa:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [display_labels[format_labels.index(f)] for f in present_formats],
            rotation=15,
            ha="right",
        )

    ax.set_ylabel("Cohen's Kappa")
    ax.set_title("Rating Format Comparison")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    _save_figure(fig, output_dir, "format_comparison")


# ---------------------------------------------------------------------------
# Plot 4: Bias visualization
# ---------------------------------------------------------------------------


def plot_bias_effects(
    analysis: dict[str, Any],
    output_dir: Path,
) -> None:
    """Grouped bar chart showing position bias and self-preference bias effect sizes per model."""
    bias_tests: dict[str, Any] = analysis.get("bias_tests", {})
    if not bias_tests:
        return

    position_bias: dict[str, Any] = bias_tests.get("position_bias", {})
    self_pref_bias: dict[str, Any] = bias_tests.get("self_preference", {})

    if not position_bias and not self_pref_bias:
        return

    # Collect all models across both bias types.
    all_models: set[str] = set()
    all_models.update(position_bias.keys())
    all_models.update(self_pref_bias.keys())
    models = sorted(all_models)

    if not models:
        return

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 2.5), 5))
    x = np.arange(len(models))
    width = 0.35

    # Position bias effect sizes.
    pos_effects: list[float] = []
    pos_sig: list[bool] = []
    for model in models:
        data = position_bias.get(model, {})
        if isinstance(data, dict):
            pos_effects.append(data.get("effect_size", 0.0))
            pos_sig.append(data.get("significant", False))
        else:
            pos_effects.append(0.0)
            pos_sig.append(False)

    # Self-preference bias effect sizes.
    self_effects: list[float] = []
    self_sig: list[bool] = []
    for model in models:
        data = self_pref_bias.get(model, {})
        if isinstance(data, dict):
            self_effects.append(data.get("effect_size", 0.0))
            self_sig.append(data.get("significant", False))
        else:
            self_effects.append(0.0)
            self_sig.append(False)

    bars_pos = ax.bar(
        x - width / 2,
        pos_effects,
        width,
        label="Position Bias",
        color="#2176AE",
        edgecolor="white",
        linewidth=0.5,
    )
    bars_self = ax.bar(
        x + width / 2,
        self_effects,
        width,
        label="Self-Preference Bias",
        color="#E76F51",
        edgecolor="white",
        linewidth=0.5,
    )

    # Mark significant effects with an asterisk.
    for i, (bar, sig) in enumerate(zip(bars_pos, pos_sig)):
        if sig:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                "*",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color="#2176AE",
            )

    for i, (bar, sig) in enumerate(zip(bars_self, self_sig)):
        if sig:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                "*",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color="#E76F51",
            )

    ax.set_ylabel("Effect Size (Cohen's d)")
    ax.set_title("Bias Effect Sizes by Model")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("/")[-1] for m in models], rotation=15, ha="right")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.axhline(y=0.2, color="#E63946", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Small effect (d=0.2)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_figure(fig, output_dir, "bias_effects")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_all_plots(
    analysis: dict[str, Any],
    output_dir: Path,
) -> None:
    """Run all E02 visualization functions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_fns = [
        plot_kappa_heatmap,
        plot_spearman_rho,
        plot_format_comparison,
        plot_bias_effects,
    ]

    for fn in plot_fns:
        try:
            fn(analysis, output_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: {fn.__name__} failed: {exc}")
