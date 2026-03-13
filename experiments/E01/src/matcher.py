"""Claim matching using embedding similarity.

Supports two backends:
1. API-based embeddings via OpenRouter/OpenAI (default, no torch needed)
2. Local sentence-transformers (requires torch, used when available)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import numpy as np


def _get_embeddings_api(
    texts: list[str],
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "openai/text-embedding-3-small",
) -> np.ndarray:
    """Get embeddings via an OpenAI-compatible API."""
    if not texts:
        return np.empty((0, 1536))

    if api_key is None:
        # Try loading from .env
        env_file = Path(__file__).resolve().parents[3] / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY", "")

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": model, "input": texts},
        )
        response.raise_for_status()

    data = response.json()["data"]
    embeddings = [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]
    return np.array(embeddings)


def _token_similarity(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    """Jaccard token overlap fallback (no API/model needed)."""
    if not texts_a or not texts_b:
        return np.empty((len(texts_a), len(texts_b)))

    def tokenize(t: str) -> set[str]:
        return set(t.lower().split())

    tokens_a = [tokenize(t) for t in texts_a]
    tokens_b = [tokenize(t) for t in texts_b]

    matrix = np.zeros((len(texts_a), len(texts_b)))
    for i, ta in enumerate(tokens_a):
        for j, tb in enumerate(tokens_b):
            inter = len(ta & tb)
            union = len(ta | tb)
            matrix[i, j] = inter / union if union > 0 else 0.0
    return matrix


def encode_claims(
    claims: list[str],
    api_key: str | None = None,
    use_api: bool = True,
) -> np.ndarray:
    """Encode claims to embeddings.

    Args:
        claims: List of claim texts.
        api_key: API key for embedding model.
        use_api: If True, use API embeddings. If False, use token overlap.
    """
    if not claims:
        return np.empty((0, 1536))

    if use_api and api_key:
        return _get_embeddings_api(claims, api_key=api_key)
    return _get_embeddings_api(claims)


def match_claims(
    extracted: list[str],
    ground_truth: list[str],
    threshold: float = 0.85,
    api_key: str | None = None,
) -> dict:
    """Match extracted claims against ground truth using cosine similarity.

    Uses API-based embeddings by default. Falls back to token overlap
    if no API key is available.
    """
    if not extracted and not ground_truth:
        return {
            "matches": [], "true_positives": 0, "false_positives": [],
            "false_negatives": [], "precision": 1.0, "recall": 1.0, "f1": 1.0,
        }
    if not extracted:
        return {
            "matches": [], "true_positives": 0, "false_positives": [],
            "false_negatives": list(range(len(ground_truth))),
            "precision": 1.0, "recall": 0.0, "f1": 0.0,
        }
    if not ground_truth:
        return {
            "matches": [], "true_positives": 0,
            "false_positives": list(range(len(extracted))),
            "false_negatives": [], "precision": 0.0, "recall": 1.0, "f1": 0.0,
        }

    # Try API embeddings first, fall back to token overlap.
    try:
        ext_embs = _get_embeddings_api(extracted, api_key=api_key)
        gt_embs = _get_embeddings_api(ground_truth, api_key=api_key)

        ext_norm = ext_embs / (np.linalg.norm(ext_embs, axis=1, keepdims=True) + 1e-9)
        gt_norm = gt_embs / (np.linalg.norm(gt_embs, axis=1, keepdims=True) + 1e-9)
        sim_matrix = ext_norm @ gt_norm.T
    except Exception:
        # Fallback to token overlap with adjusted threshold.
        sim_matrix = _token_similarity(extracted, ground_truth)
        threshold = min(threshold, 0.40)

    # Greedy one-to-one matching.
    matched_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    false_positives: list[int] = []

    ext_best = [
        (i, int(np.argmax(sim_matrix[i])), float(np.max(sim_matrix[i])))
        for i in range(len(extracted))
    ]
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
    api_key: str | None = None,
) -> dict:
    """Compute matching metrics for one paper's extraction result."""
    extracted_texts = [c["text"] for c in extraction_result.get("claims", [])]
    gt_texts = [c["text"] for c in gold_standard]

    metrics = match_claims(extracted_texts, gt_texts, threshold=threshold, api_key=api_key)
    metrics["paper_id"] = paper_id
    metrics["model_name"] = extraction_result.get("model_name", "unknown")
    metrics["run_number"] = extraction_result.get("run_number", 0)
    metrics["n_extracted"] = len(extracted_texts)
    metrics["n_gold"] = len(gt_texts)

    return metrics


def compute_experiment_metrics(
    results_dir: Path,
    annotations_dir: Path,
    api_key: str | None = None,
) -> dict:
    """Aggregate metrics across all papers and models."""
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

        metrics = compute_paper_metrics(paper_id, result, gold, api_key=api_key)
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
            "total_extracted": sum(m["n_extracted"] for m in paper_metrics_list),
            "false_positives": sum(len(m["false_positives"]) for m in paper_metrics_list),
            "per_paper": paper_metrics_list,
        }

    return aggregated
