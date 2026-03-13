"""Main runner for E01: Claim Extraction Accuracy experiment.

Usage:
    python experiments/E01/run_experiment.py fetch
    python experiments/E01/run_experiment.py annotate
    python experiments/E01/run_experiment.py extract
    python experiments/E01/run_experiment.py analyze
    python experiments/E01/run_experiment.py visualize
    python experiments/E01/run_experiment.py all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — add the experiment's src to the Python path so its modules
# can be imported regardless of where the script is invoked from.
# ---------------------------------------------------------------------------

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

DATA_DIR = EXPERIMENT_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
RESULTS_DIR = DATA_DIR / "results"
EXTRACTIONS_DIR = RESULTS_DIR / "extractions"
FIGURES_DIR = RESULTS_DIR / "figures"

# Default OpenRouter base URL.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models to test via OpenRouter.  All use the same OPENROUTER_API_KEY.
MODELS: list[dict[str, str]] = [
    {"name": "anthropic/claude-sonnet-4"},
    {"name": "openai/gpt-4o"},
    {"name": "anthropic/claude-haiku-4"},
]

# Model used for ground-truth annotation (dual-annotator protocol).
# Use Opus 4.6 as the "expert" annotator for highest quality ground truth.
ANNOTATION_MODEL = "anthropic/claude-opus-4"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    for d in (PAPERS_DIR, ANNOTATIONS_DIR, RESULTS_DIR, EXTRACTIONS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [saved] {path}")


def _load_env() -> None:
    """Load .env file from the project root if it exists."""
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def _get_api_key() -> str:
    """Get the OpenRouter API key from the environment."""
    _load_env()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print("[error] OPENROUTER_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)
    return key


def _available_models() -> list[dict[str, str]]:
    api_key = _get_api_key()
    return [
        {**m, "api_key": api_key, "base_url": OPENROUTER_BASE_URL}
        for m in MODELS
    ]


# ---------------------------------------------------------------------------
# Step 1: Fetch papers
# ---------------------------------------------------------------------------


def cmd_fetch() -> None:
    """Download papers from ArXiv."""
    _ensure_dirs()
    from src.paper_fetcher import download_paper_texts, select_papers

    print("Selecting papers (5 per type × 4 types = 20) ...")
    papers = select_papers(n_per_type=5)
    enriched = download_paper_texts(papers, PAPERS_DIR)
    print(f"  [done] Fetched {len(enriched)} papers to {PAPERS_DIR}")


# ---------------------------------------------------------------------------
# Step 2: Generate ground truth annotations
# ---------------------------------------------------------------------------


async def cmd_annotate() -> None:
    """Generate ground-truth annotations via dual-annotator protocol."""
    _ensure_dirs()
    api_key = _get_api_key()

    from src.annotator import generate_ground_truth

    print(f"Running dual-annotator ground truth generation (model: {ANNOTATION_MODEL}) ...")
    await generate_ground_truth(
        PAPERS_DIR, ANNOTATIONS_DIR, api_key,
        model=ANNOTATION_MODEL, base_url=OPENROUTER_BASE_URL,
    )

    n = len(list(ANNOTATIONS_DIR.glob("*_annotations.json")))
    print(f"  [done] Annotated {n} papers")


# ---------------------------------------------------------------------------
# Step 3: Run claim extraction across models
# ---------------------------------------------------------------------------


async def cmd_extract() -> None:
    """Run claim extraction across all available models."""
    _ensure_dirs()
    models = _available_models()
    if not models:
        print("[error] No API keys found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
        sys.exit(1)

    from src.claim_extractor_e01 import run_extraction

    print(f"Running extraction with {len(models)} models, 3 runs each ...")
    await run_extraction(
        papers_dir=PAPERS_DIR,
        output_dir=EXTRACTIONS_DIR,
        models=models,
        n_runs=3,
    )
    print("  [done] Extraction complete")


# ---------------------------------------------------------------------------
# Step 4: Analyze results
# ---------------------------------------------------------------------------


def cmd_analyze() -> None:
    """Compute metrics, run statistical tests, write summary."""
    _ensure_dirs()

    from src.matcher import compute_experiment_metrics
    from src.stats import binomial_ci, bootstrap_ci, cohens_d, mcnemar_test

    api_key = _get_api_key()
    print("Computing metrics ...")
    per_model = compute_experiment_metrics(EXTRACTIONS_DIR, ANNOTATIONS_DIR, api_key=api_key)

    # --- Bootstrap CIs ---
    print("Running bootstrap CIs ...")
    for model_name, data in per_model.items():
        papers = data.get("per_paper", [])
        precisions = [p["precision"] for p in papers]
        recalls = [p["recall"] for p in papers]
        f1s = [p["f1"] for p in papers]

        if precisions:
            mean_p, lo_p, hi_p = bootstrap_ci(precisions)
            mean_r, lo_r, hi_r = bootstrap_ci(recalls)
            mean_f, lo_f, hi_f = bootstrap_ci(f1s)
            data["precision_ci"] = {"mean": mean_p, "lower": lo_p, "upper": hi_p}
            data["recall_ci"] = {"mean": mean_r, "lower": lo_r, "upper": hi_r}
            data["f1_ci"] = {"mean": mean_f, "lower": lo_f, "upper": hi_f}

        # Hallucination rate (false positives / total extracted).
        total_fp = sum(len(p.get("false_positives", [])) for p in papers)
        total_ext = sum(p.get("n_extracted", 0) for p in papers)
        data["total_extracted"] = total_ext
        data["total_false_positives"] = total_fp
        if total_ext > 0:
            lo_h, hi_h = binomial_ci(total_fp, total_ext)
            data["hallucination_rate"] = round(total_fp / total_ext, 4)
            data["hallucination_ci"] = {"lower": lo_h, "upper": hi_h}
        else:
            data["hallucination_rate"] = 0.0

    # --- Pairwise McNemar tests ---
    model_names = list(per_model.keys())
    pairwise_tests: dict[str, Any] = {}
    if len(model_names) >= 2:
        print("Running pairwise McNemar tests ...")
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                ma, mb = model_names[i], model_names[j]
                papers_a = per_model[ma].get("per_paper", [])
                papers_b = per_model[mb].get("per_paper", [])
                # Build per-paper correct/incorrect lists (F1 > 0.5 = "correct")
                correct_a = [p["f1"] > 0.5 for p in papers_a]
                correct_b = [p["f1"] > 0.5 for p in papers_b]
                min_len = min(len(correct_a), len(correct_b))
                if min_len > 0:
                    result = mcnemar_test(correct_a[:min_len], correct_b[:min_len])
                    pairwise_tests[f"{ma}_vs_{mb}"] = result

        # Cohen's d for F1
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                ma, mb = model_names[i], model_names[j]
                f1_a = [p["f1"] for p in per_model[ma].get("per_paper", [])]
                f1_b = [p["f1"] for p in per_model[mb].get("per_paper", [])]
                if f1_a and f1_b:
                    d = cohens_d(f1_a, f1_b)
                    pairwise_tests[f"{ma}_vs_{mb}_cohens_d"] = round(d, 4)

    # --- Assemble final analysis ---
    analysis = {
        "timestamp": datetime.now(UTC).isoformat(),
        "per_model": per_model,
        "statistical_tests": pairwise_tests,
        "n_papers": len(list(ANNOTATIONS_DIR.glob("*_annotations.json"))),
        "n_models": len(model_names),
    }

    _save_json(analysis, RESULTS_DIR / "analysis.json")
    _write_summary(analysis, RESULTS_DIR / "summary.md")
    print("  [done] Analysis complete")


def _write_summary(analysis: dict[str, Any], path: Path) -> None:
    lines: list[str] = [
        "# E01: Claim Extraction Accuracy — Results Summary",
        "",
        f"Generated: {analysis.get('timestamp', 'unknown')}",
        f"Papers: {analysis.get('n_papers', 0)}  |  Models: {analysis.get('n_models', 0)}",
        "",
        "## Per-Model Results",
        "",
        "| Model | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Halluc. Rate |",
        "|-------|--------------------|-----------------|-------------|--------------|",
    ]

    per_model: dict[str, Any] = analysis.get("per_model", {})
    for model, data in per_model.items():
        p_ci = data.get("precision_ci", {})
        r_ci = data.get("recall_ci", {})
        f_ci = data.get("f1_ci", {})
        halluc = data.get("hallucination_rate", 0)

        p_str = f"{p_ci.get('mean', 0):.3f} [{p_ci.get('lower', 0):.3f}, {p_ci.get('upper', 0):.3f}]"
        r_str = f"{r_ci.get('mean', 0):.3f} [{r_ci.get('lower', 0):.3f}, {r_ci.get('upper', 0):.3f}]"
        f_str = f"{f_ci.get('mean', 0):.3f} [{f_ci.get('lower', 0):.3f}, {f_ci.get('upper', 0):.3f}]"
        lines.append(f"| {model} | {p_str} | {r_str} | {f_str} | {halluc:.3f} |")

    # Hypothesis verdict
    lines.extend(["", "## Hypothesis Evaluation", ""])
    best_f1 = 0.0
    best_model = ""
    best_precision = 0.0
    best_recall = 0.0
    best_halluc = 1.0
    for model, data in per_model.items():
        f1 = data.get("mean_f1", 0)
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_precision = data.get("mean_precision", 0)
            best_recall = data.get("mean_recall", 0)
            best_halluc = data.get("hallucination_rate", 1)

    precision_pass = best_precision >= 0.80
    recall_pass = best_recall >= 0.70
    halluc_pass = best_halluc <= 0.10
    hypothesis_supported = precision_pass and recall_pass and halluc_pass

    lines.append(f"- Best model: **{best_model}** (F1={best_f1:.3f})")
    lines.append(f"- Precision >= 0.80: **{'PASS' if precision_pass else 'FAIL'}** ({best_precision:.3f})")
    lines.append(f"- Recall >= 0.70: **{'PASS' if recall_pass else 'FAIL'}** ({best_recall:.3f})")
    lines.append(f"- Hallucination rate <= 0.10: **{'PASS' if halluc_pass else 'FAIL'}** ({best_halluc:.3f})")
    lines.append(f"- **Hypothesis {'SUPPORTED' if hypothesis_supported else 'NOT SUPPORTED'}**")

    # Statistical tests
    lines.extend(["", "## Statistical Tests", ""])
    stat_tests = analysis.get("statistical_tests", {})
    if stat_tests:
        for name, value in stat_tests.items():
            lines.append(f"- {name}: {value}")
    else:
        lines.append("_No pairwise tests (fewer than 2 models)._")

    lines.append("")
    path.write_text("\n".join(lines))
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Step 5: Visualize
# ---------------------------------------------------------------------------


def cmd_visualize() -> None:
    """Generate plots from analysis results."""
    analysis_path = RESULTS_DIR / "analysis.json"
    if not analysis_path.exists():
        print(f"[error] No analysis at {analysis_path}. Run 'analyze' first.")
        sys.exit(1)

    with open(analysis_path) as f:
        analysis = json.load(f)

    from src.visualize import generate_all_plots

    generate_all_plots(analysis, FIGURES_DIR)
    print(f"  [done] Figures saved to {FIGURES_DIR}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


async def cmd_all() -> None:
    """Run the full experiment pipeline end-to-end."""
    steps: list[tuple[str, Any]] = [
        ("fetch", lambda: cmd_fetch()),
        ("annotate", lambda: cmd_annotate()),
        ("extract", lambda: cmd_extract()),
        ("analyze", lambda: cmd_analyze()),
        ("visualize", lambda: cmd_visualize()),
    ]
    for name, fn in steps:
        print(f"\n{'#' * 60}")
        print(f"# Step: {name}")
        print(f"{'#' * 60}\n")
        result = fn()
        if asyncio.iscoroutine(result):
            await result

    print("\n[done] Full E01 pipeline complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E01: Claim Extraction Accuracy")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")
    subparsers.add_parser("fetch", help="Download papers from ArXiv")
    subparsers.add_parser("annotate", help="Generate ground-truth annotations")
    subparsers.add_parser("extract", help="Run claim extraction across models")
    subparsers.add_parser("analyze", help="Compute metrics and statistical tests")
    subparsers.add_parser("visualize", help="Generate plots")
    subparsers.add_parser("all", help="Run the full pipeline end-to-end")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    sync_commands = {"fetch": cmd_fetch, "analyze": cmd_analyze, "visualize": cmd_visualize}
    async_commands = {"annotate": cmd_annotate, "extract": cmd_extract, "all": cmd_all}

    if args.command in sync_commands:
        sync_commands[args.command]()
    elif args.command in async_commands:
        asyncio.run(async_commands[args.command]())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
