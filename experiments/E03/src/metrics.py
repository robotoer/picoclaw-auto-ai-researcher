"""Novelty metrics for E03: Semantic Novelty Measurement.

Four computational metrics plus a combined logistic-regression ensemble.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from .models import NoveltyScore, Paper


def embedding_distance_from_centroid(papers: list[Paper]) -> list[NoveltyScore]:
    """Cosine distance from the leave-one-out centroid of all other papers.

    For each paper, the centroid is the mean embedding of all *other* papers.
    Score = 1 - cosine_similarity(paper, centroid).  Higher = more novel.
    """
    embeddings = np.array([p.embedding for p in papers])
    scores: list[NoveltyScore] = []

    for i, paper in enumerate(papers):
        mask = np.ones(len(papers), dtype=bool)
        mask[i] = False
        centroid = embeddings[mask].mean(axis=0, keepdims=True)
        sim = float(cosine_similarity(embeddings[i : i + 1], centroid)[0, 0])
        scores.append(
            NoveltyScore(
                paper_id=paper.paper_id,
                metric_name="embedding_distance",
                score=1.0 - sim,
                details={"cosine_similarity": sim},
            )
        )

    return scores


def atypical_reference_combinations(papers: list[Paper]) -> list[NoveltyScore]:
    """Uzzi et al. (2013) atypicality: z-scored reference pair co-occurrence.

    For each paper the novelty score is the negated 10th-percentile z-score
    across all of its reference pairs.  More negative z (more unusual combo)
    yields a higher novelty score.
    """
    pair_counts: Counter[tuple[str, str]] = Counter()
    ref_counts: Counter[str] = Counter()
    total_papers = len(papers)

    for paper in papers:
        refs = paper.references
        for ref in refs:
            ref_counts[ref] += 1
        for r1, r2 in combinations(sorted(refs), 2):
            pair_counts[(r1, r2)] += 1

    scores: list[NoveltyScore] = []

    for paper in papers:
        refs = sorted(paper.references)
        if len(refs) < 2:
            scores.append(
                NoveltyScore(
                    paper_id=paper.paper_id,
                    metric_name="atypical_references",
                    score=0.0,
                    details={"n_refs": len(refs), "reason": "too_few_refs"},
                )
            )
            continue

        z_scores: list[float] = []
        for r1, r2 in combinations(refs, 2):
            observed = pair_counts.get((r1, r2), 0)
            expected = (ref_counts[r1] * ref_counts[r2]) / total_papers
            std = max(np.sqrt(expected * (1 - expected / total_papers)), 1e-6)
            z = (observed - expected) / std
            z_scores.append(float(z))

        if z_scores:
            z_10th = float(np.percentile(z_scores, 10))
            novelty = -z_10th
        else:
            z_10th = 0.0
            novelty = 0.0

        scores.append(
            NoveltyScore(
                paper_id=paper.paper_id,
                metric_name="atypical_references",
                score=novelty,
                details={"z_10th_percentile": z_10th, "n_ref_pairs": len(z_scores)},
            )
        )

    return scores


def topic_cluster_distance(
    papers: list[Paper], n_clusters: int = 20
) -> list[NoveltyScore]:
    """Minimum cosine distance to the nearest KMeans cluster centroid.

    Higher distance means the paper does not fit well into any existing
    topical cluster and is therefore more novel.
    """
    embeddings = np.array([p.embedding for p in papers])
    effective_k = min(n_clusters, len(papers) - 1)

    kmeans = MiniBatchKMeans(n_clusters=effective_k, random_state=42, n_init=3)
    kmeans.fit(embeddings)

    centroids = kmeans.cluster_centers_
    scores: list[NoveltyScore] = []

    for i, paper in enumerate(papers):
        sims = cosine_similarity(embeddings[i : i + 1], centroids)[0]
        max_sim = float(np.max(sims))
        min_dist = 1.0 - max_sim
        scores.append(
            NoveltyScore(
                paper_id=paper.paper_id,
                metric_name="topic_distance",
                score=min_dist,
                details={
                    "nearest_cluster": int(np.argmax(sims)),
                    "max_similarity": max_sim,
                },
            )
        )

    return scores


def llm_novelty_scores(llm_scores: list[dict[str, object]]) -> list[NoveltyScore]:
    """Convert raw LLM novelty judgments to NoveltyScore objects.

    Each dict in *llm_scores* must contain ``paper_id`` (str), ``model``
    (str), and ``score`` (int, 1-7).  Scores are averaged across models and
    normalized to [0, 1].
    """
    paper_scores: defaultdict[str, list[float]] = defaultdict(list)
    for s in llm_scores:
        paper_scores[str(s["paper_id"])].append(float(s["score"]))  # type: ignore[arg-type]

    scores: list[NoveltyScore] = []
    for paper_id, score_list in paper_scores.items():
        avg = float(np.mean(score_list))
        normalized = (avg - 1.0) / 6.0  # Map 1-7 to 0-1
        scores.append(
            NoveltyScore(
                paper_id=paper_id,
                metric_name="llm_judgment",
                score=normalized,
                details={"raw_avg": avg, "n_models": len(score_list)},
            )
        )

    return scores


def combined_metric(
    all_metric_scores: dict[str, list[NoveltyScore]],
    labels: dict[str, int],
) -> list[NoveltyScore]:
    """Combine metrics via logistic regression with leave-one-out CV.

    Parameters
    ----------
    all_metric_scores:
        Mapping from metric name to its list of :class:`NoveltyScore`.
    labels:
        Mapping from paper_id to binary label (1 = novel, 0 = incremental).

    Returns
    -------
    list[NoveltyScore]
        LOO-CV predicted probabilities as combined novelty scores.
    """
    metric_names = sorted(all_metric_scores.keys())

    # Build paper_id -> metric_name -> score mapping
    paper_metrics: dict[str, dict[str, float]] = {}
    for mname, mscores in all_metric_scores.items():
        for s in mscores:
            paper_metrics.setdefault(s.paper_id, {})[mname] = s.score

    paper_ids = sorted(paper_metrics.keys())
    X = np.array(
        [[paper_metrics[pid].get(m, 0.0) for m in metric_names] for pid in paper_ids]
    )
    y = np.array([labels.get(pid, 0) for pid in paper_ids])

    loo = LeaveOneOut()
    predictions = np.zeros(len(paper_ids))
    scaler = StandardScaler()

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train_s, y_train)
        predictions[test_idx] = clf.predict_proba(X_test_s)[:, 1]

    return [
        NoveltyScore(
            paper_id=pid,
            metric_name="combined",
            score=float(predictions[i]),
        )
        for i, pid in enumerate(paper_ids)
    ]
