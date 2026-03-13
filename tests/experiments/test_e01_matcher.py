"""Tests for E01 claim matching logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures — precomputed embeddings so we never load the real model in CI
# ---------------------------------------------------------------------------

# Two near-identical embeddings (cosine ~ 0.99)
_VEC_A = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
_VEC_A_PARA = np.array([0.51, 0.49, 0.50, 0.50], dtype=np.float32)

# An unrelated embedding (cosine ~ 0 with _VEC_A)
_VEC_UNRELATED = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _make_mock_model(
    text_to_vec: dict[str, np.ndarray],
) -> MagicMock:
    """Return a mock sentence-transformers model that maps known texts to vectors."""
    model = MagicMock()

    def _encode(texts: list[str], **_kwargs: object) -> np.ndarray:
        vecs = []
        for t in texts:
            vecs.append(text_to_vec.get(t, np.zeros(4, dtype=np.float32)))
        return np.stack(vecs)

    model.encode = _encode
    return model


# ---------------------------------------------------------------------------
# Inline matcher helpers (mirrors expected E01 matching logic)
# ---------------------------------------------------------------------------


def _match_claims(
    ground_truth: list[str],
    extracted: list[str],
    similarity_matrix: np.ndarray,
    threshold: float = 0.85,
) -> dict[str, int]:
    """Simple greedy matching: each GT claim matches at most one extracted claim."""
    tp = 0
    matched_gt: set[int] = set()
    matched_ext: set[int] = set()

    # Greedy: pick highest similarity pairs first
    n_gt, n_ext = similarity_matrix.shape
    flat_indices = np.argsort(similarity_matrix.ravel())[::-1]

    for idx in flat_indices:
        gi, ei = divmod(int(idx), n_ext)
        if gi in matched_gt or ei in matched_ext:
            continue
        if similarity_matrix[gi, ei] >= threshold:
            tp += 1
            matched_gt.add(gi)
            matched_ext.add(ei)

    fn = len(ground_truth) - tp
    fp = len(extracted) - tp
    return {"tp": tp, "fp": fp, "fn": fn}


def _precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClaimMatcher:
    """Unit tests for claim matching logic."""

    def test_exact_match(self) -> None:
        """Identical claims should match."""
        sim = _cosine_sim(_VEC_A, _VEC_A)
        assert sim == pytest.approx(1.0, abs=1e-6)

        sim_matrix = np.array([[sim]])
        result = _match_claims(["claim"], ["claim"], sim_matrix, threshold=0.85)
        assert result["tp"] == 1
        assert result["fp"] == 0
        assert result["fn"] == 0

    def test_semantic_match(self) -> None:
        """Paraphrased claims should match above threshold."""
        sim = _cosine_sim(_VEC_A, _VEC_A_PARA)
        assert sim > 0.85, f"Expected cosine > 0.85, got {sim}"

        sim_matrix = np.array([[sim]])
        result = _match_claims(["original"], ["paraphrase"], sim_matrix, threshold=0.85)
        assert result["tp"] == 1

    def test_no_match(self) -> None:
        """Unrelated claims should not match."""
        sim = _cosine_sim(_VEC_A, _VEC_UNRELATED)
        assert sim < 0.85

        sim_matrix = np.array([[sim]])
        result = _match_claims(["claim_a"], ["unrelated"], sim_matrix, threshold=0.85)
        assert result["tp"] == 0
        assert result["fp"] == 1
        assert result["fn"] == 1

    def test_precision_recall_calculation(self) -> None:
        """Known TP/FP/FN counts give correct P/R/F1."""
        # 3 TP, 1 FP, 2 FN
        prf = _precision_recall_f1(tp=3, fp=1, fn=2)
        assert prf["precision"] == pytest.approx(3 / 4)
        assert prf["recall"] == pytest.approx(3 / 5)
        assert prf["f1"] == pytest.approx(2 * (3 / 4) * (3 / 5) / (3 / 4 + 3 / 5))

    def test_empty_inputs(self) -> None:
        """Empty extracted or ground truth lists."""
        # No ground truth, some extracted
        sim_matrix = np.empty((0, 3))
        result = _match_claims([], ["a", "b", "c"], sim_matrix, threshold=0.85)
        assert result["tp"] == 0
        assert result["fp"] == 3
        assert result["fn"] == 0

        prf = _precision_recall_f1(**result)
        assert prf["precision"] == 0.0
        assert prf["recall"] == 0.0

        # No extracted, some ground truth
        sim_matrix2 = np.empty((2, 0))
        result2 = _match_claims(["a", "b"], [], sim_matrix2, threshold=0.85)
        assert result2["tp"] == 0
        assert result2["fp"] == 0
        assert result2["fn"] == 2

    def test_threshold_sensitivity(self) -> None:
        """Varying threshold changes match count."""
        sim = _cosine_sim(_VEC_A, _VEC_A_PARA)  # ~0.99
        sim_matrix = np.array([[sim]])

        # High threshold still matches
        r1 = _match_claims(["a"], ["b"], sim_matrix, threshold=0.95)
        assert r1["tp"] == 1

        # Impossibly high threshold blocks match
        r2 = _match_claims(["a"], ["b"], sim_matrix, threshold=1.0)
        assert r2["tp"] == 0


@pytest.mark.integration
@pytest.mark.integration
class TestClaimMatcherIntegration:
    """Tests that call the embedding API.

    These are skipped in CI (run with ``pytest -m integration``).
    """

    def test_real_api_encoding(self) -> None:
        """Smoke test: encode a sentence via embedding API."""
        from experiments.E01.src.matcher import _get_embeddings_api  # noqa: PLC0415

        vec = _get_embeddings_api(["BERT outperforms GPT-2 on GLUE"])
        assert vec.shape[0] == 1
        assert vec.shape[1] > 0
