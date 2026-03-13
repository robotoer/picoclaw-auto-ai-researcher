"""Main runner for E02: LLM Judge Reliability experiment.

Usage:
    python experiments/E02/run_experiment.py generate
    python experiments/E02/run_experiment.py evaluate
    python experiments/E02/run_experiment.py analyze
    python experiments/E02/run_experiment.py visualize
    python experiments/E02/run_experiment.py all
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
HYPOTHESES_DIR = DATA_DIR / "hypotheses"
RATINGS_DIR = DATA_DIR / "ratings"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Default OpenRouter base URL.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# LLM judge models to evaluate.
JUDGE_MODELS = [
    "anthropic/claude-opus-4-6",
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-4o",
    "google/gemini-2.0-flash-001",
]

# Expert proxy model (used as ground-truth reference).
EXPERT_PROXY_MODEL = "anthropic/claude-opus-4"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    for d in (HYPOTHESES_DIR, RATINGS_DIR, RESULTS_DIR, FIGURES_DIR):
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


def _load_hypotheses() -> list[Any]:
    """Load generated hypotheses from disk as Hypothesis objects."""
    from src.models import Hypothesis

    path = HYPOTHESES_DIR / "hypotheses.json"
    if not path.exists():
        print(f"[error] No hypotheses found at {path}. Run 'generate' first.")
        sys.exit(1)
    with open(path) as f:
        raw = json.load(f)
    return [Hypothesis(**h) if isinstance(h, dict) else h for h in raw]


def _load_all_ratings() -> list[dict[str, Any]]:
    """Load all rating session files from the ratings directory."""
    sessions: list[dict[str, Any]] = []
    for path in sorted(RATINGS_DIR.glob("*.json")):
        with open(path) as f:
            sessions.append(json.load(f))
    if not sessions:
        print(f"[error] No ratings found in {RATINGS_DIR}. Run 'evaluate' first.")
        sys.exit(1)
    return sessions


# ---------------------------------------------------------------------------
# Step 1: Generate hypotheses
# ---------------------------------------------------------------------------


async def cmd_generate() -> None:
    """Generate 50 research hypotheses across quality tiers."""
    _ensure_dirs()
    api_key = _get_api_key()

    from src.hypothesis_generator import generate_hypotheses

    print("Generating 50 hypotheses across quality tiers ...")
    hypotheses = await generate_hypotheses(
        output_dir=HYPOTHESES_DIR,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )
    print(f"  [done] Generated {len(hypotheses)} hypotheses to {HYPOTHESES_DIR}")


# ---------------------------------------------------------------------------
# Step 2: Run evaluations
# ---------------------------------------------------------------------------


async def cmd_evaluate() -> None:
    """Run all LLM judge and expert proxy evaluations."""
    _ensure_dirs()
    api_key = _get_api_key()
    hypotheses = _load_hypotheses()

    from src import judge

    # Run expert proxy ratings (ground truth reference).
    print(f"Running expert proxy ratings (model: {EXPERT_PROXY_MODEL}) ...")
    expert_sessions = await judge.run_expert_proxy_ratings(
        hypotheses=hypotheses,
        model=EXPERT_PROXY_MODEL,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )
    for session in expert_sessions:
        session_data = (
            session.model_dump(mode="json")
            if hasattr(session, "model_dump")
            else session
        )
        filename = f"expert_{session_data.get('format', 'absolute')}_{session_data.get('rater_id', 'proxy')}.json"
        _save_json(session_data, RATINGS_DIR / filename)

    print(f"  [done] Expert proxy ratings saved ({len(expert_sessions)} sessions)")

    # Select 20 hypothesis pairs spanning the quality range for pairwise comparison.
    pairs = judge.select_pairwise_pairs(hypotheses, n_pairs=20)
    print(f"Selected {len(pairs)} hypothesis pairs for pairwise comparison")

    # Run LLM judge ratings (all models, all formats).
    print(f"Running LLM judge ratings ({len(JUDGE_MODELS)} models) ...")
    judge_sessions = await judge.run_llm_judge_ratings(
        hypotheses=hypotheses,
        pairs=pairs,
        models=JUDGE_MODELS,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )
    for session in judge_sessions:
        session_data = (
            session.model_dump(mode="json")
            if hasattr(session, "model_dump")
            else session
        )
        rater_slug = session_data.get("rater_id", "judge").replace("/", "_")
        fmt = session_data.get("format", "absolute")
        filename = f"judge_{rater_slug}_{fmt}.json"
        _save_json(session_data, RATINGS_DIR / filename)

    print(f"  [done] LLM judge ratings saved ({len(judge_sessions)} sessions)")


# ---------------------------------------------------------------------------
# Step 3: Analyze results
# ---------------------------------------------------------------------------


def cmd_analyze() -> None:
    """Compute metrics, run statistical tests, write summary."""
    _ensure_dirs()

    from src import stats
    from src.models import RatingSession

    raw_ratings = _load_all_ratings()
    hypotheses = _load_hypotheses()

    all_sessions = [RatingSession(**r) if isinstance(r, dict) else r for r in raw_ratings]
    expert_sessions = [s for s in all_sessions if s.rater_type == "expert_proxy"]
    judge_sessions = [s for s in all_sessions if s.rater_type == "llm_judge"]

    print(f"Running E02 analysis ({len(expert_sessions)} expert, {len(judge_sessions)} judge sessions) ...")
    analysis = stats.run_e02_analysis(
        expert_sessions=expert_sessions,
        judge_sessions=judge_sessions,
    )

    analysis["timestamp"] = datetime.now(UTC).isoformat()
    analysis["n_hypotheses"] = len(hypotheses)
    analysis["n_judge_models"] = len(JUDGE_MODELS)
    analysis["judge_models"] = JUDGE_MODELS

    _save_json(analysis, RESULTS_DIR / "analysis.json")
    _write_summary(analysis, RESULTS_DIR / "summary.md")
    print("  [done] Analysis complete")


def _write_summary(analysis: dict[str, Any], path: Path) -> None:
    """Write a markdown summary of the E02 analysis results."""
    lines: list[str] = [
        "# E02: LLM Judge Reliability — Results Summary",
        "",
        f"Generated: {analysis.get('timestamp', 'unknown')}",
        f"Hypotheses: {analysis.get('n_hypotheses', 0)}  |  "
        f"Judge Models: {analysis.get('n_judge_models', 0)}",
        "",
    ]

    # --- Per-model kappa on each dimension ---
    lines.extend(["## Inter-Rater Agreement: Cohen's Kappa by Dimension", ""])

    dimensions = ["novelty", "feasibility", "importance", "clarity", "specificity"]
    per_model_kappa: dict[str, dict[str, Any]] = analysis.get("per_model_kappa", {})

    if per_model_kappa:
        header = "| Model | " + " | ".join(d.capitalize() for d in dimensions) + " |"
        sep = "|-------" + "|--------" * len(dimensions) + "|"
        lines.append(header)
        lines.append(sep)

        for model, dim_kappas in per_model_kappa.items():
            values = [f"{dim_kappas.get(d, 0.0):.3f}" for d in dimensions]
            lines.append(f"| {model} | " + " | ".join(values) + " |")
    else:
        lines.append("_No kappa data available._")

    # --- Overall Spearman rho ---
    lines.extend(["", "## Overall Correlation: Spearman's Rho", ""])

    spearman_rho: dict[str, Any] = analysis.get("spearman_rho", {})
    if spearman_rho:
        lines.append("| Model | Spearman Rho | p-value |")
        lines.append("|-------|-------------|---------|")
        for model, rho_data in spearman_rho.items():
            if isinstance(rho_data, dict):
                rho = rho_data.get("rho", 0.0)
                p_val = rho_data.get("p_value", 1.0)
                lines.append(f"| {model} | {rho:.3f} | {p_val:.4f} |")
            else:
                lines.append(f"| {model} | {rho_data:.3f} | — |")
    else:
        lines.append("_No Spearman rho data available._")

    # --- Hypothesis evaluation ---
    lines.extend(["", "## Hypothesis Evaluation", ""])

    criteria = analysis.get("hypothesis_evaluation", {})
    if criteria:
        # H1: At least one LLM judge achieves kappa >= 0.6 on all dimensions
        kappa_pass = criteria.get("kappa_threshold_pass", False)
        best_kappa_model = criteria.get("best_kappa_model", "unknown")
        min_kappa = criteria.get("best_model_min_kappa", 0.0)
        lines.append(
            f"- **H1** Kappa >= 0.60 on all dimensions (best model: {best_kappa_model}): "
            f"**{'PASS' if kappa_pass else 'FAIL'}** (min kappa = {min_kappa:.3f})"
        )

        # H2: Spearman rho >= 0.7 for at least one model
        rho_pass = criteria.get("rho_threshold_pass", False)
        best_rho_model = criteria.get("best_rho_model", "unknown")
        best_rho = criteria.get("best_rho_value", 0.0)
        lines.append(
            f"- **H2** Spearman rho >= 0.70 (best model: {best_rho_model}): "
            f"**{'PASS' if rho_pass else 'FAIL'}** (rho = {best_rho:.3f})"
        )

        # H3: CoT improves agreement
        cot_pass = criteria.get("cot_improves_agreement", False)
        cot_delta = criteria.get("cot_kappa_delta", 0.0)
        lines.append(
            f"- **H3** CoT improves agreement: "
            f"**{'PASS' if cot_pass else 'FAIL'}** (delta kappa = {cot_delta:+.3f})"
        )

        # Overall
        all_pass = kappa_pass and rho_pass
        lines.append(f"- **Overall: {'SUPPORTED' if all_pass else 'NOT SUPPORTED'}**")
    else:
        lines.append("_No hypothesis evaluation data available._")

    # --- Bias test results ---
    lines.extend(["", "## Bias Tests", ""])

    bias_tests: dict[str, Any] = analysis.get("bias_tests", {})
    if bias_tests:
        # Position bias
        position_bias = bias_tests.get("position_bias", {})
        if position_bias:
            lines.append("### Position Bias")
            lines.append("")
            lines.append("| Model | Effect Size | Significant |")
            lines.append("|-------|------------|-------------|")
            for model, bias_data in position_bias.items():
                if isinstance(bias_data, dict):
                    effect = bias_data.get("effect_size", 0.0)
                    sig = bias_data.get("significant", False)
                    lines.append(
                        f"| {model} | {effect:.3f} | {'Yes' if sig else 'No'} |"
                    )

        # Self-preference bias
        self_pref_bias = bias_tests.get("self_preference", {})
        if self_pref_bias:
            lines.append("")
            lines.append("### Self-Preference Bias")
            lines.append("")
            lines.append("| Model | Effect Size | Significant |")
            lines.append("|-------|------------|-------------|")
            for model, bias_data in self_pref_bias.items():
                if isinstance(bias_data, dict):
                    effect = bias_data.get("effect_size", 0.0)
                    sig = bias_data.get("significant", False)
                    lines.append(
                        f"| {model} | {effect:.3f} | {'Yes' if sig else 'No'} |"
                    )
    else:
        lines.append("_No bias test data available._")

    # --- Format comparison ---
    format_comparison: dict[str, Any] = analysis.get("format_comparison", {})
    if format_comparison:
        lines.extend(["", "## Format Comparison (Absolute vs Pairwise)", ""])
        lines.append("| Format | Mean Kappa |")
        lines.append("|--------|-----------|")
        for fmt, data in format_comparison.items():
            if isinstance(data, dict):
                kappa = data.get("mean_kappa", 0.0)
                lines.append(f"| {fmt} | {kappa:.3f} |")
            else:
                lines.append(f"| {fmt} | {data:.3f} |")

    lines.append("")
    path.write_text("\n".join(lines))
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Step 4: Visualize
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
        ("generate", lambda: cmd_generate()),
        ("evaluate", lambda: cmd_evaluate()),
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

    print("\n[done] Full E02 pipeline complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E02: LLM Judge Reliability")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")
    subparsers.add_parser("generate", help="Generate research hypotheses")
    subparsers.add_parser("evaluate", help="Run LLM judge and expert evaluations")
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

    sync_commands = {"analyze": cmd_analyze, "visualize": cmd_visualize}
    async_commands = {
        "generate": cmd_generate,
        "evaluate": cmd_evaluate,
        "all": cmd_all,
    }

    if args.command in sync_commands:
        sync_commands[args.command]()
    elif args.command in async_commands:
        asyncio.run(async_commands[args.command]())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
