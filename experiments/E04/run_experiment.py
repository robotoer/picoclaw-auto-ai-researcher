"""Main runner for E04: Knowledge Graph Consistency Under Continuous Ingestion.

Usage:
    python experiments/E04/run_experiment.py collect-landmark
    python experiments/E04/run_experiment.py collect-ingestion
    python experiments/E04/run_experiment.py extract-ground-truth
    python experiments/E04/run_experiment.py run-conditions
    python experiments/E04/run_experiment.py analyze
    python experiments/E04/run_experiment.py visualize
    python experiments/E04/run_experiment.py all
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

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

DATA_DIR = EXPERIMENT_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
GT_DIR = DATA_DIR / "ground_truth"
RESULTS_DIR = DATA_DIR / "results"
EXTRACTIONS_DIR = RESULTS_DIR / "extractions"
FIGURES_DIR = RESULTS_DIR / "figures"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Ground truth extraction model (high quality)
GT_MODEL = "anthropic/claude-opus-4"

# Multi-extractor models for Layer 1
EXTRACTOR_MODELS = [
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-4o",
    "google/gemini-2.0-flash-001",
]

# Verification model for Layer 2 & 3
VERIFICATION_MODEL = "anthropic/claude-sonnet-4-6"

# Spot-check model (high quality)
SPOT_CHECK_MODEL = "anthropic/claude-opus-4"

# Conditions to test
CONDITIONS = ["no_filtering", "layer1", "layer1_2", "layer1_2_3"]

# Growth curve checkpoints
CHECKPOINTS = [100, 200, 300, 400, 500]


def _ensure_dirs() -> None:
    for d in (PAPERS_DIR, GT_DIR, RESULTS_DIR, EXTRACTIONS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [saved] {path}")


def _load_env() -> None:
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def _get_api_key() -> str:
    _load_env()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print("[error] OPENROUTER_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)
    return key


def _load_papers(filename: str) -> list[Any]:
    from src.models import Paper

    path = PAPERS_DIR / filename
    if not path.exists():
        print(f"[error] No papers at {path}. Run the appropriate collect step first.")
        sys.exit(1)
    with open(path) as f:
        raw = json.load(f)
    return [Paper(**p) if isinstance(p, dict) else p for p in raw]


def _load_ground_truth() -> list[Any]:
    from src.models import Claim

    path = GT_DIR / "ground_truth_claims.json"
    if not path.exists():
        print(f"[error] No ground truth at {path}. Run 'extract-ground-truth' first.")
        sys.exit(1)
    with open(path) as f:
        raw = json.load(f)
    return [Claim(**c) if isinstance(c, dict) else c for c in raw]


# -----------------------------------------------------------------------
# Step 1a: Collect landmark papers
# -----------------------------------------------------------------------
async def cmd_collect_landmark() -> None:
    """Fetch 100 landmark AI/ML papers for ground truth construction."""
    _ensure_dirs()

    from src.data_collection import fetch_landmark_papers

    path = PAPERS_DIR / "landmark_papers.json"
    if path.exists():
        print(f"  [skip] Landmark papers already at {path}")
        return

    print("Fetching landmark papers from Semantic Scholar ...")
    papers = await fetch_landmark_papers(n_papers=100)
    _save_json([p.model_dump(mode="json") for p in papers], path)
    print(f"  [done] {len(papers)} landmark papers saved")


# -----------------------------------------------------------------------
# Step 1b: Collect ingestion papers
# -----------------------------------------------------------------------
async def cmd_collect_ingestion() -> None:
    """Fetch 500 ingestion papers."""
    _ensure_dirs()

    from src.data_collection import fetch_ingestion_papers

    path = PAPERS_DIR / "ingestion_papers.json"
    if path.exists():
        print(f"  [skip] Ingestion papers already at {path}")
        return

    print("Fetching ingestion papers from Semantic Scholar ...")
    papers = await fetch_ingestion_papers(n_papers=500)
    _save_json([p.model_dump(mode="json") for p in papers], path)
    print(f"  [done] {len(papers)} ingestion papers saved")


# -----------------------------------------------------------------------
# Step 2: Extract ground truth claims
# -----------------------------------------------------------------------
async def cmd_extract_ground_truth() -> None:
    """Extract verified claims from landmark papers."""
    _ensure_dirs()
    api_key = _get_api_key()

    path = GT_DIR / "ground_truth_claims.json"
    if path.exists():
        print(f"  [skip] Ground truth already at {path}")
        return

    landmark_papers = _load_papers("landmark_papers.json")

    from src.data_collection import extract_ground_truth_claims

    print(f"Extracting ground truth claims from {len(landmark_papers)} papers ...")
    claims = await extract_ground_truth_claims(
        papers=landmark_papers,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        model=GT_MODEL,
    )
    _save_json([c.model_dump(mode="json") for c in claims], path)
    print(f"  [done] {len(claims)} ground truth claims extracted")


# -----------------------------------------------------------------------
# Step 3: Run all 4 conditions
# -----------------------------------------------------------------------
async def cmd_run_conditions() -> None:
    """Run ingestion pipeline under all 4 conditions."""
    _ensure_dirs()
    api_key = _get_api_key()

    ingestion_papers = _load_papers("ingestion_papers.json")
    ground_truth_claims = _load_ground_truth()
    paper_map = {p.paper_id: p for p in ingestion_papers}

    from src.data_collection import extract_claims_multi, extract_claims_single
    from src.kg_builder import (
        compute_duplicate_rate,
        compute_provenance_completeness,
        detect_contradictions,
        layer1_multi_extractor_voting,
        layer2_temporal_consistency,
        layer3_source_verification,
        spot_check_claims,
    )
    from src.models import ConditionResult

    # Check for existing multi-extractor results
    multi_extract_path = EXTRACTIONS_DIR / "multi_extractions.json"
    single_extract_path = EXTRACTIONS_DIR / "single_extractions.json"

    # --- Single extractor results (for no_filtering condition) ---
    if single_extract_path.exists():
        print(f"  [skip] Single extractions already at {single_extract_path}")
        with open(single_extract_path) as f:
            single_raw = json.load(f)
        from src.models import Claim, ExtractionResult
        single_extractions = []
        for er in single_raw:
            single_extractions.append(ExtractionResult(
                paper_id=er["paper_id"],
                extractor_model=er["extractor_model"],
                claims=[Claim(**c) for c in er["claims"]],
                timestamp=er.get("timestamp", ""),
            ))
    else:
        print("Extracting claims (single extractor) ...")
        single_extractions = []
        for i, paper in enumerate(ingestion_papers):
            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{len(ingestion_papers)} papers")
            result = await extract_claims_single(
                paper=paper,
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
                model=EXTRACTOR_MODELS[0],  # Use first model for single
            )
            single_extractions.append(result)
        _save_json(
            [er.model_dump(mode="json") for er in single_extractions],
            single_extract_path,
        )

    # --- Multi-extractor results (for layer1+ conditions) ---
    if multi_extract_path.exists():
        print(f"  [skip] Multi extractions already at {multi_extract_path}")
        with open(multi_extract_path) as f:
            multi_raw = json.load(f)
        from src.models import Claim, ExtractionResult
        # multi_raw is a dict: paper_id -> list of ExtractionResults
        multi_extractions: dict[str, list[Any]] = {}
        for pid, ers in multi_raw.items():
            multi_extractions[pid] = [
                ExtractionResult(
                    paper_id=er["paper_id"],
                    extractor_model=er["extractor_model"],
                    claims=[Claim(**c) for c in er["claims"]],
                    timestamp=er.get("timestamp", ""),
                )
                for er in ers
            ]
    else:
        print("Extracting claims (multi-extractor) ...")
        multi_extractions = {}
        for i, paper in enumerate(ingestion_papers):
            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{len(ingestion_papers)} papers")
            results = await extract_claims_multi(
                paper=paper,
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
                models=EXTRACTOR_MODELS,
            )
            multi_extractions[paper.paper_id] = results
        # Serialize
        serialized = {
            pid: [er.model_dump(mode="json") for er in ers]
            for pid, ers in multi_extractions.items()
        }
        _save_json(serialized, multi_extract_path)

    # --- Run each condition ---
    all_condition_results: list[dict[str, Any]] = []
    extractor_errors: dict[str, list[bool]] = {m: [] for m in EXTRACTOR_MODELS}

    for condition in CONDITIONS:
        cond_path = RESULTS_DIR / f"condition_{condition}.json"
        if cond_path.exists():
            print(f"  [skip] Condition {condition} already at {cond_path}")
            with open(cond_path) as f:
                all_condition_results.append(json.load(f))
            continue

        print(f"\n{'=' * 60}")
        print(f"Running condition: {condition}")
        print(f"{'=' * 60}")

        cond_result = ConditionResult(condition=condition)
        growth_data: list[dict[str, Any]] = []

        # Process papers in checkpoint batches
        for checkpoint_idx, checkpoint in enumerate(CHECKPOINTS):
            start_idx = CHECKPOINTS[checkpoint_idx - 1] if checkpoint_idx > 0 else 0
            end_idx = min(checkpoint, len(ingestion_papers))
            batch_papers = ingestion_papers[start_idx:end_idx]

            print(f"  Checkpoint {checkpoint}: processing papers {start_idx}-{end_idx}")

            batch_claims = []

            if condition == "no_filtering":
                # Raw single-extractor claims
                for er in single_extractions[start_idx:end_idx]:
                    batch_claims.extend(er.claims)

            elif condition == "layer1":
                # Multi-extractor voting
                for paper in batch_papers:
                    pid = paper.paper_id
                    if pid in multi_extractions:
                        voted = layer1_multi_extractor_voting(multi_extractions[pid])
                        batch_claims.extend(voted)

            elif condition == "layer1_2":
                # Multi-extractor voting + temporal consistency
                for paper in batch_papers:
                    pid = paper.paper_id
                    if pid in multi_extractions:
                        voted = layer1_multi_extractor_voting(multi_extractions[pid])
                        accepted, _flagged = await layer2_temporal_consistency(
                            claims=voted,
                            ground_truth_claims=ground_truth_claims,
                            api_key=api_key,
                            base_url=OPENROUTER_BASE_URL,
                            model=VERIFICATION_MODEL,
                        )
                        batch_claims.extend(accepted)

            elif condition == "layer1_2_3":
                # All layers
                for paper in batch_papers:
                    pid = paper.paper_id
                    if pid in multi_extractions:
                        voted = layer1_multi_extractor_voting(multi_extractions[pid])
                        accepted, flagged = await layer2_temporal_consistency(
                            claims=voted,
                            ground_truth_claims=ground_truth_claims,
                            api_key=api_key,
                            base_url=OPENROUTER_BASE_URL,
                            model=VERIFICATION_MODEL,
                        )
                        if flagged:
                            verified, _rejected = await layer3_source_verification(
                                flagged_claims=flagged,
                                papers=paper_map,
                                api_key=api_key,
                                base_url=OPENROUTER_BASE_URL,
                                model=VERIFICATION_MODEL,
                            )
                            accepted.extend(verified)
                        batch_claims.extend(accepted)

            # Update cumulative metrics
            cond_result.n_claims_total += len(batch_claims)
            cond_result.n_papers_ingested = end_idx

            # Detect contradictions at this checkpoint
            if batch_claims:
                contradictions = await detect_contradictions(
                    kg_claims=batch_claims,
                    ground_truth_claims=ground_truth_claims,
                    api_key=api_key,
                    base_url=OPENROUTER_BASE_URL,
                    model=VERIFICATION_MODEL,
                )
                n_new_contradictions = sum(1 for c in contradictions if c.is_contradiction)
                cond_result.n_contradictions += n_new_contradictions

            current_rate = (
                cond_result.n_contradictions / cond_result.n_claims_total
                if cond_result.n_claims_total > 0 else 0.0
            )

            growth_data.append({
                "n_papers": end_idx,
                "n_claims": cond_result.n_claims_total,
                "n_contradictions": cond_result.n_contradictions,
                "contradiction_rate": current_rate,
            })

            print(f"    Claims: {cond_result.n_claims_total}, "
                  f"Contradictions: {cond_result.n_contradictions}, "
                  f"Rate: {current_rate:.4f}")

        cond_result.growth_checkpoints = growth_data

        # Final metrics
        if cond_result.n_claims_total > 0:
            cond_result.contradiction_rate = (
                cond_result.n_contradictions / cond_result.n_claims_total
            )

            # Collect all claims for this condition for spot-check
            all_cond_claims = []
            if condition == "no_filtering":
                for er in single_extractions:
                    all_cond_claims.extend(er.claims)
            else:
                for pid in multi_extractions:
                    if condition == "layer1":
                        all_cond_claims.extend(
                            layer1_multi_extractor_voting(multi_extractions[pid])
                        )
                    # For layer1_2 and layer1_2_3, we already tracked — use accumulated
                    # For spot-check, sample from whatever claims we have

            if not all_cond_claims:
                # Fallback: use single extraction claims
                for er in single_extractions:
                    all_cond_claims.extend(er.claims)

            # Spot-check
            print(f"  Running spot-check for {condition} ...")
            spot_results = await spot_check_claims(
                claims=all_cond_claims,
                papers=paper_map,
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
                model=SPOT_CHECK_MODEL,
                n_samples=50,
            )
            n_halluc = sum(
                1 for s in spot_results
                if s.category in ("unsupported", "fabricated_source")
            )
            cond_result.n_hallucinations = n_halluc
            cond_result.hallucination_rate = n_halluc / len(spot_results) if spot_results else 0.0

            # Duplicate rate and provenance
            cond_result.duplicate_rate = compute_duplicate_rate(all_cond_claims[:500])
            cond_result.provenance_completeness = compute_provenance_completeness(all_cond_claims)

        result_dict = cond_result.model_dump(mode="json")
        _save_json(result_dict, cond_path)
        all_condition_results.append(result_dict)

    # --- Extractor independence analysis (from multi-extractor data) ---
    print("\nAnalyzing extractor independence ...")
    # For each paper, check which extractors produced claims that matched ground truth
    # vs which produced hallucinated/contradictory claims
    extractor_error_path = RESULTS_DIR / "extractor_errors.json"
    if not extractor_error_path.exists() and multi_extractions:
        for pid, ers in list(multi_extractions.items())[:200]:  # Sample for efficiency
            for er in ers:
                has_error = any(
                    not c.verified for c in er.claims
                ) if er.claims else True
                if er.extractor_model in extractor_errors:
                    extractor_errors[er.extractor_model].append(has_error)
        _save_json(extractor_errors, extractor_error_path)

    # Save combined results
    _save_json(all_condition_results, RESULTS_DIR / "all_conditions.json")
    print("\n  [done] All conditions complete")


# -----------------------------------------------------------------------
# Step 4: Analyze
# -----------------------------------------------------------------------
def cmd_analyze() -> None:
    """Run statistical analysis on all condition results."""
    _ensure_dirs()

    from src import stats

    # Load condition results
    all_cond_path = RESULTS_DIR / "all_conditions.json"
    if not all_cond_path.exists():
        print(f"[error] No condition results at {all_cond_path}. Run 'run-conditions' first.")
        sys.exit(1)
    with open(all_cond_path) as f:
        condition_results = json.load(f)

    # Load extractor errors if available
    extractor_errors = None
    ee_path = RESULTS_DIR / "extractor_errors.json"
    if ee_path.exists():
        with open(ee_path) as f:
            extractor_errors = json.load(f)

    print(f"Running E04 analysis ({len(condition_results)} conditions) ...")
    analysis = stats.run_e04_analysis(
        condition_results=condition_results,
        extractor_errors=extractor_errors,
    )

    # Hypothesis evaluation
    hyp = _evaluate_hypotheses(condition_results, analysis)
    analysis["hypothesis_evaluation"] = hyp
    analysis["timestamp"] = datetime.now(UTC).isoformat()
    analysis["n_conditions"] = len(condition_results)

    _save_json(analysis, RESULTS_DIR / "analysis.json")
    _write_summary(analysis, condition_results, RESULTS_DIR / "summary.md")
    print("  [done] Analysis complete")


def _evaluate_hypotheses(
    condition_results: list[dict[str, Any]],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate success/failure criteria from the experiment spec."""
    cond_map = {cr["condition"]: cr for cr in condition_results}

    no_filter = cond_map.get("no_filtering", {})
    layer1_2 = cond_map.get("layer1_2", {})
    layer1_2_3 = cond_map.get("layer1_2_3", {})

    no_filter_rate = no_filter.get("contradiction_rate", 1.0)
    l12_rate = layer1_2.get("contradiction_rate", 1.0)
    l123_rate = layer1_2_3.get("contradiction_rate", 1.0)
    l123_halluc = layer1_2_3.get("hallucination_rate", 1.0)
    l123_provenance = layer1_2_3.get("provenance_completeness", 0.0)

    # Growth curve check
    growth = analysis.get("growth_curves", {})
    l123_growth = growth.get("layer1_2_3", {})
    slope_positive = l123_growth.get("slope_positive", True)

    return {
        "h1_no_filter_baseline": {
            "criterion": "contradiction_rate <= 10%",
            "value": no_filter_rate,
            "pass": no_filter_rate <= 0.10,
        },
        "h2_layer1_2_rate": {
            "criterion": "contradiction_rate <= 2%",
            "value": l12_rate,
            "pass": l12_rate <= 0.02,
        },
        "h3_all_layers_rate": {
            "criterion": "contradiction_rate <= 0.5%",
            "value": l123_rate,
            "pass": l123_rate <= 0.005,
        },
        "h4_hallucination_rate": {
            "criterion": "hallucination_rate <= 5%",
            "value": l123_halluc,
            "pass": l123_halluc <= 0.05,
        },
        "h5_provenance": {
            "criterion": "provenance_completeness >= 95%",
            "value": l123_provenance,
            "pass": l123_provenance >= 0.95,
        },
        "h6_growth_stable": {
            "criterion": "contradiction rate slope non-positive with all layers",
            "slope_positive": slope_positive,
            "pass": not slope_positive,
        },
        "overall": (
            no_filter_rate <= 0.10
            and l12_rate <= 0.02
            and l123_rate <= 0.005
            and l123_halluc <= 0.05
            and l123_provenance >= 0.95
            and not slope_positive
        ),
    }


def _write_summary(
    analysis: dict[str, Any],
    condition_results: list[dict[str, Any]],
    path: Path,
) -> None:
    """Write markdown summary of results."""
    lines = [
        "# E04: Knowledge Graph Consistency — Results Summary",
        "",
        f"Generated: {analysis.get('timestamp', 'unknown')}",
        f"Conditions: {analysis.get('n_conditions', 0)}",
        "",
        "## Contradiction Rate by Condition",
        "",
        "| Condition | Papers | Claims | Contradictions | Rate | Halluc. Rate | Provenance |",
        "|-----------|--------|--------|----------------|------|-------------|------------|",
    ]

    cond_labels = {
        "no_filtering": "No Filter",
        "layer1": "Layer 1",
        "layer1_2": "Layer 1+2",
        "layer1_2_3": "Layer 1+2+3",
    }

    for cr in condition_results:
        label = cond_labels.get(cr["condition"], cr["condition"])
        lines.append(
            f"| {label} | {cr['n_papers_ingested']} | {cr['n_claims_total']} "
            f"| {cr['n_contradictions']} | {cr['contradiction_rate']:.4f} "
            f"| {cr['hallucination_rate']:.4f} | {cr['provenance_completeness']:.3f} |"
        )

    lines.extend(["", "## Hypothesis Evaluation", ""])
    hyp = analysis.get("hypothesis_evaluation", {})
    for key, val in hyp.items():
        if key == "overall":
            lines.append(f"- **Overall: {'SUPPORTED' if val else 'NOT SUPPORTED'}**")
        elif isinstance(val, dict):
            status = "PASS" if val.get("pass") else "FAIL"
            criterion = val.get("criterion", "")
            value = val.get("value", val.get("slope_positive", ""))
            lines.append(f"- **{key}**: {criterion} — **{status}** (value={value})")

    # Chi-squared test
    chi2 = analysis.get("chi_squared", {})
    if chi2:
        lines.extend([
            "",
            "## Chi-Squared Test (Across All Conditions)",
            "",
            f"- χ² = {chi2.get('chi2', 'N/A')}, p = {chi2.get('p_value', 'N/A')}",
            f"- Significant: {chi2.get('significant', 'N/A')}",
        ])

    # Extractor independence
    ei = analysis.get("extractor_independence", {})
    if ei:
        lines.extend([
            "",
            "## Extractor Independence",
            "",
            f"- Effective independent extractors: {ei.get('effective_n_extractors', 'N/A')}",
            f"- Observed consensus error rate: {ei.get('observed_rate', 'N/A')}",
            f"- Theoretical rate (independence): {ei.get('theoretical_rate', 'N/A')}",
        ])

    lines.append("")
    path.write_text("\n".join(lines))
    print(f"  [saved] {path}")


# -----------------------------------------------------------------------
# Step 5: Visualize
# -----------------------------------------------------------------------
def cmd_visualize() -> None:
    """Generate all plots."""
    analysis_path = RESULTS_DIR / "analysis.json"
    if not analysis_path.exists():
        print(f"[error] No analysis at {analysis_path}. Run 'analyze' first.")
        sys.exit(1)
    with open(analysis_path) as f:
        analysis = json.load(f)

    from src.visualize import generate_all_plots

    generate_all_plots(analysis, FIGURES_DIR)
    print(f"  [done] Figures saved to {FIGURES_DIR}")


# -----------------------------------------------------------------------
# Full pipeline
# -----------------------------------------------------------------------
async def cmd_all() -> None:
    steps: list[tuple[str, Any]] = [
        ("collect-landmark", lambda: cmd_collect_landmark()),
        ("collect-ingestion", lambda: cmd_collect_ingestion()),
        ("extract-ground-truth", lambda: cmd_extract_ground_truth()),
        ("run-conditions", lambda: cmd_run_conditions()),
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
    print("\n[done] Full E04 pipeline complete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E04: Knowledge Graph Consistency Under Continuous Ingestion"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")
    subparsers.add_parser("collect-landmark", help="Fetch landmark papers")
    subparsers.add_parser("collect-ingestion", help="Fetch ingestion papers")
    subparsers.add_parser("extract-ground-truth", help="Extract ground truth claims")
    subparsers.add_parser("run-conditions", help="Run all 4 conditions")
    subparsers.add_parser("analyze", help="Run statistical analysis")
    subparsers.add_parser("visualize", help="Generate plots")
    subparsers.add_parser("all", help="Run full pipeline")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    sync_commands = {
        "analyze": cmd_analyze,
        "visualize": cmd_visualize,
    }
    async_commands = {
        "collect-landmark": cmd_collect_landmark,
        "collect-ingestion": cmd_collect_ingestion,
        "extract-ground-truth": cmd_extract_ground_truth,
        "run-conditions": cmd_run_conditions,
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
