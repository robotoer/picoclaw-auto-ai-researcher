"""Main runner for E03: Semantic Novelty Measurement experiment.

Usage:
    python experiments/E03/run_experiment.py collect
    python experiments/E03/run_experiment.py annotate
    python experiments/E03/run_experiment.py compute
    python experiments/E03/run_experiment.py analyze
    python experiments/E03/run_experiment.py visualize
    python experiments/E03/run_experiment.py all
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
ANNOTATIONS_DIR = DATA_DIR / "annotations"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Expert proxy model for novelty annotation
EXPERT_PROXY_MODEL = "anthropic/claude-opus-4"

# LLM judge models for direct novelty scoring
LLM_JUDGE_MODELS = [
    "anthropic/claude-opus-4-6",
    "anthropic/claude-sonnet-4-6",
]


def _ensure_dirs() -> None:
    for d in (PAPERS_DIR, ANNOTATIONS_DIR, RESULTS_DIR, FIGURES_DIR):
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


def _load_papers() -> list[Any]:
    from src.models import Paper

    path = PAPERS_DIR / "papers.json"
    if not path.exists():
        print(f"[error] No papers at {path}. Run 'collect' first.")
        sys.exit(1)
    with open(path) as f:
        raw = json.load(f)
    return [Paper(**p) if isinstance(p, dict) else p for p in raw]


def _load_annotations() -> list[Any]:
    from src.models import AnnotationSession

    sessions = []
    for path in sorted(ANNOTATIONS_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        sessions.append(AnnotationSession(**data) if isinstance(data, dict) else data)
    if not sessions:
        print(f"[error] No annotations in {ANNOTATIONS_DIR}. Run 'annotate' first.")
        sys.exit(1)
    return sessions


# Step 1: Collect papers
async def cmd_collect() -> None:
    """Fetch 100 papers from Semantic Scholar and compute embeddings."""
    _ensure_dirs()

    from src.data_collection import compute_tfidf_embeddings, fetch_papers

    papers_path = PAPERS_DIR / "papers.json"
    if papers_path.exists():
        print(f"  [skip] Papers already collected at {papers_path}")
        return

    print("Fetching papers from Semantic Scholar ...")
    papers = await fetch_papers(n_papers=100)
    print(f"  Fetched {len(papers)} papers")

    print("Computing TF-IDF embeddings ...")
    papers = compute_tfidf_embeddings(papers)

    _save_json([p.model_dump(mode="json") for p in papers], papers_path)
    print(f"  [done] {len(papers)} papers saved with embeddings")


# Step 2: Annotate
async def cmd_annotate() -> None:
    """Generate expert proxy novelty annotations."""
    _ensure_dirs()
    api_key = _get_api_key()
    papers = _load_papers()

    from src.data_collection import annotate_papers_novelty, get_llm_novelty_scores

    # Expert proxy annotations
    existing_expert = list(ANNOTATIONS_DIR.glob("expert_*.json"))
    if existing_expert:
        print(f"  [skip] Found {len(existing_expert)} existing expert annotation files")
    else:
        print(f"Running expert proxy annotations (model: {EXPERT_PROXY_MODEL}) ...")
        sessions = await annotate_papers_novelty(
            papers=papers,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            model=EXPERT_PROXY_MODEL,
        )
        for session in sessions:
            data = session.model_dump(mode="json")
            filename = f"expert_{session.rater_id}.json"
            _save_json(data, ANNOTATIONS_DIR / filename)
        print(f"  [done] Expert annotations saved ({len(sessions)} sessions)")

    # LLM judge novelty scores
    existing_judge = list(ANNOTATIONS_DIR.glob("llm_judge_*.json"))
    if existing_judge:
        print(f"  [skip] Found {len(existing_judge)} existing LLM judge files")
    else:
        print(f"Running LLM judge novelty scoring ({len(LLM_JUDGE_MODELS)} models) ...")
        llm_scores = await get_llm_novelty_scores(
            papers=papers,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            models=LLM_JUDGE_MODELS,
        )
        _save_json(llm_scores, ANNOTATIONS_DIR / "llm_judge_scores.json")
        print(f"  [done] LLM judge scores saved ({len(llm_scores)} scores)")


# Step 3: Compute novelty metrics
def cmd_compute() -> None:
    """Compute all 4 novelty metrics."""
    _ensure_dirs()
    papers = _load_papers()

    from src import metrics

    print("Computing novelty metrics ...")

    # 1. Embedding distance
    emb_scores = metrics.embedding_distance_from_centroid(papers)
    print(f"  Embedding distance: {len(emb_scores)} scores")

    # 2. Atypical references
    ref_scores = metrics.atypical_reference_combinations(papers)
    print(f"  Atypical references: {len(ref_scores)} scores")

    # 3. Topic cluster distance
    topic_scores = metrics.topic_cluster_distance(papers)
    print(f"  Topic distance: {len(topic_scores)} scores")

    # 4. LLM novelty (from judge annotations)
    llm_path = ANNOTATIONS_DIR / "llm_judge_scores.json"
    if llm_path.exists():
        with open(llm_path) as f:
            raw_llm = json.load(f)
        llm_scores = metrics.llm_novelty_scores(raw_llm)
        print(f"  LLM judgment: {len(llm_scores)} scores")
    else:
        llm_scores = []
        print("  [warn] No LLM judge scores found, skipping LLM metric")

    # 5. Combined metric (needs labels)
    annotations = _load_annotations()
    from src.models import NoveltyLabel

    # Majority vote for binary labels
    paper_votes: dict[str, list[int]] = {}
    paper_likert: dict[str, list[float]] = {}
    for session in annotations:
        for ann in session.annotations:
            paper_votes.setdefault(ann.paper_id, []).append(
                1 if ann.binary_label == NoveltyLabel.novel else 0
            )
            paper_likert.setdefault(ann.paper_id, []).append(float(ann.likert_score))

    labels = {
        pid: int(sum(votes) > len(votes) / 2) for pid, votes in paper_votes.items()
    }

    all_metric_scores = {
        "embedding_distance": emb_scores,
        "atypical_references": ref_scores,
        "topic_distance": topic_scores,
    }
    if llm_scores:
        all_metric_scores["llm_judgment"] = llm_scores

    combined_scores = metrics.combined_metric(all_metric_scores, labels)
    print(f"  Combined: {len(combined_scores)} scores")

    # Save all scores
    all_scores = {
        name: [s.model_dump(mode="json") for s in scores]
        for name, scores in all_metric_scores.items()
    }
    all_scores["combined"] = [s.model_dump(mode="json") for s in combined_scores]
    _save_json(all_scores, RESULTS_DIR / "metric_scores.json")
    _save_json(labels, RESULTS_DIR / "binary_labels.json")
    _save_json(
        {pid: float(sum(v) / len(v)) for pid, v in paper_likert.items()},
        RESULTS_DIR / "human_likert.json",
    )
    print("  [done] All metric scores saved")


# Step 4: Analyze
def cmd_analyze() -> None:
    """Run statistical analysis."""
    _ensure_dirs()
    papers = _load_papers()

    from src import stats

    # Load metric scores
    with open(RESULTS_DIR / "metric_scores.json") as f:
        raw_scores = json.load(f)
    with open(RESULTS_DIR / "binary_labels.json") as f:
        labels_dict = json.load(f)
    with open(RESULTS_DIR / "human_likert.json") as f:
        likert_dict = json.load(f)

    # Align all data by paper_id
    paper_ids = [p.paper_id for p in papers]
    binary_labels = [labels_dict.get(pid, 0) for pid in paper_ids]
    human_likert = [likert_dict.get(pid, 4.0) for pid in paper_ids]
    citation_counts = [p.citation_count_2yr for p in papers]

    # Build metric score vectors aligned by paper_id
    metric_scores: dict[str, list[float]] = {}
    for metric_name, score_list in raw_scores.items():
        pid_to_score = {s["paper_id"]: s["score"] for s in score_list}
        metric_scores[metric_name] = [pid_to_score.get(pid, 0.0) for pid in paper_ids]

    print(f"Running E03 analysis ({len(metric_scores)} metrics, {len(papers)} papers) ...")
    analysis = stats.run_e03_analysis(
        metric_scores=metric_scores,
        binary_labels=binary_labels,
        human_likert=human_likert,
        citation_counts=citation_counts,
        paper_ids=paper_ids,
    )

    # Add inter-rater agreement
    annotations = _load_annotations()
    from src.models import NoveltyLabel

    # Build ratings matrix for Fleiss' kappa
    rater_labels: dict[str, dict[str, int]] = {}  # rater_id -> paper_id -> label
    for session in annotations:
        for ann in session.annotations:
            rater_labels.setdefault(session.rater_id, {})[ann.paper_id] = (
                1 if ann.binary_label == NoveltyLabel.novel else 0
            )

    if len(rater_labels) >= 2:
        raters = sorted(rater_labels.keys())
        ratings_matrix = []
        for pid in paper_ids:
            votes = [rater_labels.get(r, {}).get(pid, 0) for r in raters]
            novel_count = sum(votes)
            incr_count = len(votes) - novel_count
            ratings_matrix.append([novel_count, incr_count])

        fk = stats.fleiss_kappa(ratings_matrix)
        analysis["inter_rater_agreement"] = {
            "fleiss_kappa": fk,
            "n_raters": len(raters),
        }

    # Hypothesis evaluation
    best_auc = max(
        (m["auc"]["auc"] for m in analysis["per_metric"].values()),
        default=0.0,
    )
    best_spearman_cite = max(
        (m["spearman_citation"]["rho"] for m in analysis["per_metric"].values()),
        default=0.0,
    )
    analysis["hypothesis_evaluation"] = {
        "h1_auc_pass": best_auc >= 0.70,
        "best_auc": best_auc,
        "h2_spearman_pass": best_spearman_cite >= 0.30,
        "best_spearman_citation": best_spearman_cite,
        "overall": best_auc >= 0.70 and best_spearman_cite >= 0.30,
    }

    analysis["timestamp"] = datetime.now(UTC).isoformat()
    analysis["n_papers"] = len(papers)
    analysis["n_metrics"] = len(metric_scores)

    _save_json(analysis, RESULTS_DIR / "analysis.json")
    _write_summary(analysis, RESULTS_DIR / "summary.md")
    print("  [done] Analysis complete")


def _write_summary(analysis: dict[str, Any], path: Path) -> None:
    """Write markdown summary."""
    lines = [
        "# E03: Semantic Novelty Measurement — Results Summary",
        "",
        f"Generated: {analysis.get('timestamp', 'unknown')}",
        f"Papers: {analysis.get('n_papers', 0)}  |  Metrics: {analysis.get('n_metrics', 0)}",
        "",
        "## AUC-ROC per Metric",
        "",
        "| Metric | AUC | 95% CI | Spearman ρ (citation) | Spearman ρ (human) | P@10 | P@20 |",
        "|--------|-----|--------|----------------------|-------------------|------|------|",
    ]

    for name, data in sorted(analysis.get("per_metric", {}).items()):
        auc = data["auc"]
        sc = data["spearman_citation"]
        sh = data["spearman_human"]
        lines.append(
            f"| {name} | {auc['auc']:.3f} | [{auc['ci_lower']:.3f}, {auc['ci_upper']:.3f}] "
            f"| {sc['rho']:.3f} (p={sc['p_value']:.4f}) "
            f"| {sh['rho']:.3f} (p={sh['p_value']:.4f}) "
            f"| {data['precision_at_10']:.3f} | {data['precision_at_20']:.3f} |"
        )

    lines.extend(["", "## Hypothesis Evaluation", ""])
    hyp = analysis.get("hypothesis_evaluation", {})
    if hyp:
        lines.append(
            f"- **H1** AUC >= 0.70: **{'PASS' if hyp['h1_auc_pass'] else 'FAIL'}** "
            f"(best AUC = {hyp['best_auc']:.3f})"
        )
        lines.append(
            f"- **H2** Spearman ρ >= 0.30 with citation: "
            f"**{'PASS' if hyp['h2_spearman_pass'] else 'FAIL'}** "
            f"(best ρ = {hyp['best_spearman_citation']:.3f})"
        )
        lines.append(
            f"- **Overall: {'SUPPORTED' if hyp['overall'] else 'NOT SUPPORTED'}**"
        )

    ira = analysis.get("inter_rater_agreement", {})
    if ira:
        lines.extend(["", "## Inter-Rater Agreement", ""])
        lines.append(
            f"Fleiss' kappa: {ira['fleiss_kappa']:.3f} ({ira['n_raters']} raters)"
        )

    lines.append("")
    path.write_text("\n".join(lines))
    print(f"  [saved] {path}")


# Step 5: Visualize
def cmd_visualize() -> None:
    """Generate plots."""
    analysis_path = RESULTS_DIR / "analysis.json"
    if not analysis_path.exists():
        print(f"[error] No analysis at {analysis_path}. Run 'analyze' first.")
        sys.exit(1)
    with open(analysis_path) as f:
        analysis = json.load(f)
    from src.visualize import generate_all_plots

    generate_all_plots(analysis, FIGURES_DIR)
    print(f"  [done] Figures saved to {FIGURES_DIR}")


# Full pipeline
async def cmd_all() -> None:
    steps: list[tuple[str, Any]] = [
        ("collect", lambda: cmd_collect()),
        ("annotate", lambda: cmd_annotate()),
        ("compute", lambda: cmd_compute()),
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
    print("\n[done] Full E03 pipeline complete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E03: Semantic Novelty Measurement")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")
    subparsers.add_parser("collect", help="Fetch papers from Semantic Scholar")
    subparsers.add_parser("annotate", help="Generate novelty annotations")
    subparsers.add_parser("compute", help="Compute novelty metrics")
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
        "compute": cmd_compute,
        "analyze": cmd_analyze,
        "visualize": cmd_visualize,
    }
    async_commands = {
        "collect": cmd_collect,
        "annotate": cmd_annotate,
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
