"""Claim matching using sentence embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


def encode_claims(
    claims: list[str],
    model_name: str = _DEFAULT_MODEL,
) -> np.ndarray:
    """Encode a list of claim strings into dense embeddings.

    Args:
        claims: List of natural-language claim strings.
        model_name: Sentence-transformer model identifier.

    Returns:
        NumPy array of shape ``(len(claims), embedding_dim)``.
    """
    if not claims:
        return np.empty((0, 384))
    model = SentenceTransformer(model_name)
    return model.encode(claims, convert_to_numpy=True)


def match_claims(
    extracted: list[str],
    ground_truth: list[str],
    threshold: float = 0.85,
) -> dict:
    """Match extracted claims against ground truth using cosine similarity.

    Each extracted claim is matched to its closest ground-truth claim.
    Matches with cosine similarity >= ``threshold`` count as true positives.

    Args:
        extracted: List of extracted claim texts.
        ground_truth: List of gold-standard claim texts.
        threshold: Cosine similarity threshold for a match.

    Returns:
        Dict with ``matches``, ``true_positives``, ``false_positives``,
        ``false_negatives``, ``precision``, ``recall``, and ``f1``.
    """
    if not extracted and not ground_truth:
        return {
            "matches": [],
            "true_positives": 0,
            "false_positives": [],
            "false_negatives": [],
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }

    if not extracted:
        return {
            "matches": [],
            "true_positives": 0,
            "false_positives": [],
            "false_negatives": list(range(len(ground_truth))),
            "precision": 1.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    if not ground_truth:
        return {
            "matches": [],
            "true_positives": 0,
            "false_positives": list(range(len(extracted))),
            "false_negatives": [],
            "precision": 0.0,
            "recall": 1.0,
            "f1": 0.0,
        }

    model = SentenceTransformer(_DEFAULT_MODEL)
    ext_embs = model.encode(extracted, convert_to_numpy=True)
    gt_embs = model.encode(ground_truth, convert_to_numpy=True)

    # Cosine similarity matrix.
    ext_norm = ext_embs / (np.linalg.norm(ext_embs, axis=1, keepdims=True) + 1e-9)
    gt_norm = gt_embs / (np.linalg.norm(gt_embs, axis=1, keepdims=True) + 1e-9)
    sim_matrix = ext_norm @ gt_norm.T  # shape: (n_ext, n_gt)

    # Greedy one-to-one matching.
    matched_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    false_positives: list[int] = []

    # Sort extracted claims by their best match score (descending) for greedy assignment.
    ext_best = [(i, int(np.argmax(sim_matrix[i])), float(np.max(sim_matrix[i])))
                for i in range(len(extracted))]
    ext_best.sort(key=lambda x: x[2], reverse=True)

    for ext_idx, best_gt, best_sim in ext_best:
        if best_sim >= threshold and best_gt not in matched_gt:
            matches.append((ext_idx, best_gt, round(best_sim, 4)))
            matched_gt.add(best_gt)
        else:
            false_positives.append(ext_idx)

    false_negatives = [j for j in range(len(ground_truth)) if j not in matched_gt]

    tp = len(matches)
    precision = tp / len(extracted) if extracted else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "matches": matches,
        "true_positives": tp,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_paper_metrics(
    paper_id: str,
    extraction_result: dict,
    gold_standard: list[dict],
    threshold: float = 0.85,
) -> dict:
    """Compute matching metrics for one paper's extraction result.

    Args:
        paper_id: ArXiv ID of the paper.
        extraction_result: Parsed extraction result dict (must contain ``claims``).
        gold_standard: List of gold-standard claim dicts (must contain ``text``).
        threshold: Cosine similarity threshold.

    Returns:
        Dict with paper_id, model_name, run_number, and all matching metrics.
    """
    extracted_texts = [c["text"] for c in extraction_result.get("claims", [])]
    gt_texts = [c["text"] for c in gold_standard]

    metrics = match_claims(extracted_texts, gt_texts, threshold=threshold)
    metrics["paper_id"] = paper_id
    metrics["model_name"] = extraction_result.get("model_name", "unknown")
    metrics["run_number"] = extraction_result.get("run_number", 0)
    metrics["n_extracted"] = len(extracted_texts)
    metrics["n_gold"] = len(gt_texts)

    return metrics


def compute_experiment_metrics(
    results_dir: Path,
    annotations_dir: Path,
) -> dict:
    """Aggregate metrics across all papers and models.

    Args:
        results_dir: Directory containing extraction result JSON files.
        annotations_dir: Directory containing annotation JSON files.

    Returns:
        Dict keyed by model name, each containing aggregated precision,
        recall, F1, and per-paper breakdowns.
    """
    # Load gold standards.
    gold_by_paper: dict[str, list[dict]] = {}
    for ann_file in sorted(annotations_dir.glob("*_annotations.json")):
        ann = json.loads(ann_file.read_text(encoding="utf-8"))
        paper_id = ann.get("paper_id", ann_file.stem.replace("_annotations", ""))
        gold_by_paper[paper_id] = ann.get("gold_standard", [])

    # Process each extraction result.
    model_metrics: dict[str, list[dict]] = {}
    for result_file in sorted(results_dir.glob("*.json")):
        result = json.loads(result_file.read_text(encoding="utf-8"))
        paper_id = result.get("paper_id", "")
        model_name = result.get("model_name", "unknown")

        gold = gold_by_paper.get(paper_id)
        if gold is None:
            continue

        metrics = compute_paper_metrics(paper_id, result, gold)
        model_metrics.setdefault(model_name, []).append(metrics)

    # Aggregate per model.
    aggregated: dict[str, dict] = {}
    for model_name, paper_metrics_list in model_metrics.items():
        precisions = [m["precision"] for m in paper_metrics_list]
        recalls = [m["recall"] for m in paper_metrics_list]
        f1s = [m["f1"] for m in paper_metrics_list]

        aggregated[model_name] = {
            "mean_precision": round(float(np.mean(precisions)), 4),
            "mean_recall": round(float(np.mean(recalls)), 4),
            "mean_f1": round(float(np.mean(f1s)), 4),
            "std_precision": round(float(np.std(precisions)), 4),
            "std_recall": round(float(np.std(recalls)), 4),
            "std_f1": round(float(np.std(f1s)), 4),
            "n_papers": len(paper_metrics_list),
            "per_paper": paper_metrics_list,
        }

    return aggregated
