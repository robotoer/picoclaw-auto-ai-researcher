"""Visualization module for E03 (Semantic Novelty Measurement) results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_all_plots(analysis: dict[str, Any], output_dir: Path) -> None:
    """Generate all E03 plots and save to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_auc_comparison(analysis, output_dir)
    _plot_spearman_bars(analysis, output_dir)
    _plot_precision_at_k(analysis, output_dir)


def _plot_auc_comparison(analysis: dict[str, Any], output_dir: Path) -> None:
    """Bar chart of AUC-ROC per metric with confidence intervals."""
    per_metric = analysis.get("per_metric", {})
    if not per_metric:
        return

    metrics = sorted(per_metric.keys())
    aucs = [per_metric[m]["auc"]["auc"] for m in metrics]
    ci_lower = [per_metric[m]["auc"]["ci_lower"] for m in metrics]
    ci_upper = [per_metric[m]["auc"]["ci_upper"] for m in metrics]

    errors = [
        [a - lo for a, lo in zip(aucs, ci_lower)],
        [u - a for a, u in zip(aucs, ci_upper)],
    ]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    ax.bar(x, aucs, yerr=errors, capsize=5, color=colors[: len(metrics)])
    ax.axhline(y=0.7, color="red", linestyle="--", label="Threshold (AUC=0.70)")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random (AUC=0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("E03: Novelty Metric AUC-ROC Comparison")
    ax.legend()
    ax.set_ylim(0, 1.05)

    for i, v in enumerate(aucs):
        ax.text(i, v + errors[1][i] + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "auc_comparison.png", dpi=150)
    fig.savefig(output_dir / "auc_comparison.pdf")
    plt.close(fig)


def _plot_spearman_bars(analysis: dict[str, Any], output_dir: Path) -> None:
    """Bar chart of Spearman rho with citation and human scores."""
    per_metric = analysis.get("per_metric", {})
    if not per_metric:
        return

    metrics = sorted(per_metric.keys())
    rho_cite = [per_metric[m]["spearman_citation"]["rho"] for m in metrics]
    rho_human = [per_metric[m]["spearman_human"]["rho"] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, rho_cite, width, label="vs Citation Count", color="#2196F3")
    ax.bar(x + width / 2, rho_human, width, label="vs Human Score", color="#4CAF50")
    ax.axhline(
        y=0.3, color="red", linestyle="--", label="Citation Threshold (\u03c1=0.30)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.set_ylabel("Spearman \u03c1")
    ax.set_title("E03: Correlation with Citation Count and Human Novelty Score")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "spearman_comparison.png", dpi=150)
    fig.savefig(output_dir / "spearman_comparison.pdf")
    plt.close(fig)


def _plot_precision_at_k(analysis: dict[str, Any], output_dir: Path) -> None:
    """Bar chart of precision@10 and precision@20."""
    per_metric = analysis.get("per_metric", {})
    if not per_metric:
        return

    metrics = sorted(per_metric.keys())
    p10 = [per_metric[m]["precision_at_10"] for m in metrics]
    p20 = [per_metric[m]["precision_at_20"] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, p10, width, label="Precision@10", color="#FF9800")
    ax.bar(x + width / 2, p20, width, label="Precision@20", color="#9C27B0")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.set_ylabel("Precision")
    ax.set_title("E03: Precision@k for Retrieving Novel Papers")
    ax.legend()
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_dir / "precision_at_k.png", dpi=150)
    fig.savefig(output_dir / "precision_at_k.pdf")
    plt.close(fig)
