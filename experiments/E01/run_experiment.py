"""Main runner for E01: Claim Extraction Accuracy experiment."""

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
# Path setup
# ---------------------------------------------------------------------------

EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
RESULTS_DIR = DATA_DIR / "results"

MODELS: list[dict[str, str]] = [
    {
        "name": "claude-sonnet-4-20250514",
        "provider": "anthropic",
        "env_key": "ANTHROPIC_API_KEY",
    },
    {
        "name": "gpt-4o",
        "provider": "openai",
        "env_key": "OPENAI_API_KEY",
    },
    {
        "name": "claude-haiku-4-5-20251001",
        "provider": "anthropic",
        "env_key": "ANTHROPIC_API_KEY",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _available_models() -> list[dict[str, str]]:
    """Return only models whose API key is present in the environment."""
    available: list[dict[str, str]] = []
    for model in MODELS:
        key = model["env_key"]
        if os.environ.get(key):
            available.append(model)
        else:
            print(f"[skip] {model['name']} — {key} not set")
    return available


def _ensure_dirs() -> None:
    """Create required directories if they don't exist."""
    for d in (PAPERS_DIR, ANNOTATIONS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Any:
    """Load a JSON file, returning None if it doesn't exist."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_json(data: Any, path: Path) -> None:
    """Write *data* as indented JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[saved] {path}")


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


async def cmd_fetch(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Download papers from ArXiv."""
    _ensure_dirs()
    # Lazy import so the CLI loads fast even without all deps installed
    try:
        from experiments.E01.src import fetch_papers  # type: ignore[import-untyped]
    except ImportError:
        print(
            "[error] Could not import fetch_papers module. "
            "Ensure experiments/E01/src/fetch_papers.py exists."
        )
        sys.exit(1)

    papers = await fetch_papers.fetch_all(output_dir=PAPERS_DIR)
    print(f"[done] Fetched {len(papers)} papers to {PAPERS_DIR}")


async def cmd_annotate(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Generate ground-truth annotations via dual-annotator protocol."""
    _ensure_dirs()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[error] ANTHROPIC_API_KEY required for annotation.")
        sys.exit(1)

    try:
        from experiments.E01.src import annotate  # type: ignore[import-untyped]
    except ImportError:
        print(
            "[error] Could not import annotate module. "
            "Ensure experiments/E01/src/annotate.py exists."
        )
        sys.exit(1)

    result = await annotate.run(papers_dir=PAPERS_DIR, output_dir=ANNOTATIONS_DIR)
    print(f"[done] Annotated {result.get('paper_count', '?')} papers")


async def cmd_extract(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Run claim extraction across all available models."""
    _ensure_dirs()
    models = _available_models()
    if not models:
        print("[error] No API keys found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
        sys.exit(1)

    try:
        from experiments.E01.src import extract  # type: ignore[import-untyped]
    except ImportError:
        print(
            "[error] Could not import extract module. "
            "Ensure experiments/E01/src/extract.py exists."
        )
        sys.exit(1)

    for model_cfg in models:
        print(f"\n{'=' * 60}")
        print(f"Extracting with {model_cfg['name']} ({model_cfg['provider']})")
        print("=" * 60)
        await extract.run(
            model_name=model_cfg["name"],
            provider=model_cfg["provider"],
            papers_dir=PAPERS_DIR,
            output_dir=RESULTS_DIR / "extractions" / model_cfg["name"],
            num_runs=3,
        )

    print("\n[done] Extraction complete for all available models")


async def cmd_analyze(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Compute metrics and run statistical tests."""
    _ensure_dirs()
    try:
        from experiments.E01.src import analyze  # type: ignore[import-untyped]
    except ImportError:
        print(
            "[error] Could not import analyze module. "
            "Ensure experiments/E01/src/analyze.py exists."
        )
        sys.exit(1)

    analysis = await analyze.run(
        annotations_dir=ANNOTATIONS_DIR,
        results_dir=RESULTS_DIR,
    )

    analysis_path = RESULTS_DIR / "analysis.json"
    _save_json(analysis, analysis_path)

    # Write human-readable summary
    summary_path = RESULTS_DIR / "summary.md"
    _write_summary(analysis, summary_path)
    print(f"\n[done] Analysis saved to {analysis_path}")


def _write_summary(analysis: dict[str, Any], path: Path) -> None:
    """Write a Markdown summary of the analysis results."""
    lines: list[str] = [
        "# E01: Claim Extraction Accuracy — Summary",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Per-Model Results",
        "",
        "| Model | Precision | Recall | F1 | Hallucination Rate |",
        "|-------|-----------|--------|----|--------------------|",
    ]

    per_model: dict[str, Any] = analysis.get("per_model", {})
    for model, data in per_model.items():
        p = data.get("precision", 0.0)
        r = data.get("recall", 0.0)
        f1 = data.get("f1", 0.0)
        total_ext = data.get("total_extracted", 1)
        fp = data.get("false_positives", 0)
        halluc = fp / total_ext if total_ext else 0.0
        lines.append(f"| {model} | {p:.3f} | {r:.3f} | {f1:.3f} | {halluc:.3f} |")

    lines.extend(
        [
            "",
            "## Statistical Tests",
            "",
        ]
    )

    stat_tests: dict[str, Any] = analysis.get("statistical_tests", {})
    if stat_tests:
        for test_name, result in stat_tests.items():
            lines.append(f"- **{test_name}**: {result}")
    else:
        lines.append("_No statistical tests recorded._")

    lines.append("")
    path.write_text("\n".join(lines))
    print(f"[saved] {path}")


async def cmd_visualize(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Generate plots from analysis results."""
    analysis_path = RESULTS_DIR / "analysis.json"
    analysis = _load_json(analysis_path)
    if analysis is None:
        print(f"[error] No analysis found at {analysis_path}. Run 'analyze' first.")
        sys.exit(1)

    from experiments.E01.src.visualize import generate_all_plots

    figures_dir = RESULTS_DIR / "figures"
    generate_all_plots(analysis, figures_dir)
    print(f"[done] Figures saved to {figures_dir}")


async def cmd_all(args: argparse.Namespace) -> None:
    """Run the full pipeline end-to-end."""
    steps: list[tuple[str, Any]] = [
        ("fetch", cmd_fetch),
        ("annotate", cmd_annotate),
        ("extract", cmd_extract),
        ("analyze", cmd_analyze),
        ("visualize", cmd_visualize),
    ]
    for name, fn in steps:
        print(f"\n{'#' * 60}")
        print(f"# Step: {name}")
        print(f"{'#' * 60}\n")
        await fn(args)

    print("\n[done] Full pipeline complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="E01: Claim Extraction Accuracy experiment runner.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    subparsers.add_parser("fetch", help="Download papers from ArXiv")
    subparsers.add_parser("annotate", help="Generate ground-truth annotations")
    subparsers.add_parser("extract", help="Run claim extraction across models")
    subparsers.add_parser("analyze", help="Compute metrics and statistical tests")
    subparsers.add_parser("visualize", help="Generate plots")
    subparsers.add_parser("all", help="Run the full pipeline end-to-end")

    return parser


_COMMANDS: dict[str, Any] = {
    "fetch": cmd_fetch,
    "annotate": cmd_annotate,
    "extract": cmd_extract,
    "analyze": cmd_analyze,
    "visualize": cmd_visualize,
    "all": cmd_all,
}


def main() -> None:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    asyncio.run(handler(args))


if __name__ == "__main__":
    main()
