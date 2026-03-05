"""Tests for KnowledgeConsolidator."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from auto_researcher.config import ConsolidationConfig
from auto_researcher.learning.consolidation import ConsolidationReport, KnowledgeConsolidator
from auto_researcher.models.claim import Claim, ClaimRelation, ClaimStatus
from auto_researcher.models.hypothesis import Hypothesis, HypothesisStatus


@pytest.fixture
def config() -> ConsolidationConfig:
    return ConsolidationConfig(
        dedup_similarity_threshold=0.92,
        confidence_decay_rate=0.01,
        min_confidence_threshold=0.1,
        stale_hypothesis_days=90,
    )


@pytest.fixture
def consolidator(config: ConsolidationConfig) -> KnowledgeConsolidator:
    return KnowledgeConsolidator(config)


def _make_claim(
    id: str,
    confidence: float = 0.8,
    status: ClaimStatus = ClaimStatus.EXTRACTED,
    extracted_at: datetime | None = None,
    source_paper_ids: list[str] | None = None,
    supporting_claim_ids: list[str] | None = None,
    half_life_days: int = 365,
) -> Claim:
    return Claim(
        id=id,
        entity_1="A",
        relation=ClaimRelation.OUTPERFORMS,
        entity_2="B",
        confidence=confidence,
        status=status,
        extracted_at=extracted_at or datetime.utcnow(),
        source_paper_ids=source_paper_ids or [],
        supporting_claim_ids=supporting_claim_ids or [],
        half_life_days=half_life_days,
    )


def _make_hypothesis(
    id: str,
    status: HypothesisStatus = HypothesisStatus.GENERATED,
    confidence: float = 0.5,
    last_updated: datetime | None = None,
) -> Hypothesis:
    return Hypothesis(
        id=id,
        entity_1="X",
        relation="outperforms",
        entity_2="Y",
        confidence=confidence,
        status=status,
        last_updated=last_updated or datetime.utcnow(),
    )


# ── Deduplication ─────────────────────────────────────────────────


class TestDeduplication:
    def test_no_embeddings(self, consolidator: KnowledgeConsolidator) -> None:
        claims = [_make_claim("c1"), _make_claim("c2")]
        result = consolidator._deduplicate_claims(claims, None)
        assert result["merged_count"] == 0

    def test_single_claim(self, consolidator: KnowledgeConsolidator) -> None:
        claims = [_make_claim("c1")]
        result = consolidator._deduplicate_claims(claims, {"c1": [1.0, 0.0]})
        assert result["merged_count"] == 0

    def test_merge_similar_claims(self, consolidator: KnowledgeConsolidator) -> None:
        claims = [
            _make_claim("c1", confidence=0.9, source_paper_ids=["p1"]),
            _make_claim("c2", confidence=0.7, source_paper_ids=["p2"]),
        ]
        # Nearly identical embeddings -> should merge
        embeddings = {
            "c1": [1.0, 0.0, 0.0],
            "c2": [0.99, 0.05, 0.0],
        }
        result = consolidator._deduplicate_claims(claims, embeddings)
        assert result["merged_count"] == 1
        # c1 has higher confidence, so c2 sources merge into c1
        assert "p2" in claims[0].source_paper_ids

    def test_no_merge_dissimilar(self, consolidator: KnowledgeConsolidator) -> None:
        claims = [
            _make_claim("c1", source_paper_ids=["p1"]),
            _make_claim("c2", source_paper_ids=["p2"]),
        ]
        # Very different embeddings
        embeddings = {
            "c1": [1.0, 0.0, 0.0],
            "c2": [0.0, 1.0, 0.0],
        }
        result = consolidator._deduplicate_claims(claims, embeddings)
        assert result["merged_count"] == 0

    def test_merge_keeps_higher_confidence(self, consolidator: KnowledgeConsolidator) -> None:
        claims = [
            _make_claim("c1", confidence=0.5, source_paper_ids=["p1"]),
            _make_claim("c2", confidence=0.9, source_paper_ids=["p2"]),
        ]
        embeddings = {
            "c1": [1.0, 0.0],
            "c2": [1.0, 0.0],  # identical -> merge
        }
        consolidator._deduplicate_claims(claims, embeddings)
        # c2 had higher confidence, so c1 merges into c2
        assert "p1" in claims[1].source_paper_ids
        assert claims[1].confidence == 0.9

    def test_merge_supporting_claim_ids(self, consolidator: KnowledgeConsolidator) -> None:
        claims = [
            _make_claim("c1", confidence=0.9, supporting_claim_ids=["s1"]),
            _make_claim("c2", confidence=0.5, supporting_claim_ids=["s2"]),
        ]
        embeddings = {"c1": [1.0], "c2": [1.0]}
        consolidator._deduplicate_claims(claims, embeddings)
        assert "s2" in claims[0].supporting_claim_ids

    def test_missing_embedding_skipped(self, consolidator: KnowledgeConsolidator) -> None:
        claims = [_make_claim("c1"), _make_claim("c2")]
        embeddings = {"c1": [1.0, 0.0]}  # c2 missing
        result = consolidator._deduplicate_claims(claims, embeddings)
        assert result["merged_count"] == 0


# ── Confidence Decay ──────────────────────────────────────────────


class TestConfidenceDecay:
    def test_fresh_claims_not_decayed(self, consolidator: KnowledgeConsolidator) -> None:
        now = datetime.utcnow()
        claims = [_make_claim("c1", confidence=0.5, extracted_at=now)]
        decayed = consolidator._apply_confidence_decay(claims, now)
        assert len(decayed) == 0
        assert claims[0].status == ClaimStatus.EXTRACTED

    def test_old_low_confidence_claim_becomes_stale(
        self, consolidator: KnowledgeConsolidator
    ) -> None:
        now = datetime.utcnow()
        old = now - timedelta(days=3650)  # 10 years ago
        claims = [_make_claim("c1", confidence=0.15, extracted_at=old, half_life_days=365)]
        decayed = consolidator._apply_confidence_decay(claims, now)
        assert len(decayed) == 1
        assert claims[0].status == ClaimStatus.STALE

    def test_already_stale_not_re_added(self, consolidator: KnowledgeConsolidator) -> None:
        now = datetime.utcnow()
        old = now - timedelta(days=3650)
        claims = [_make_claim("c1", confidence=0.15, extracted_at=old,
                              status=ClaimStatus.STALE, half_life_days=365)]
        decayed = consolidator._apply_confidence_decay(claims, now)
        assert len(decayed) == 0

    def test_high_confidence_survives_decay(self, consolidator: KnowledgeConsolidator) -> None:
        now = datetime.utcnow()
        old = now - timedelta(days=365)
        claims = [_make_claim("c1", confidence=0.9, extracted_at=old, half_life_days=365)]
        decayed = consolidator._apply_confidence_decay(claims, now)
        assert len(decayed) == 0


# ── Hypothesis Promotion/Demotion ─────────────────────────────────


class TestHypothesisStatus:
    def test_promote_confirmed_high_confidence(
        self, consolidator: KnowledgeConsolidator
    ) -> None:
        now = datetime.utcnow()
        h = _make_hypothesis("h1", status=HypothesisStatus.CONFIRMED, confidence=0.9)
        promoted, demoted = consolidator._update_hypothesis_status([h], now)
        assert len(promoted) == 1
        assert h.status == HypothesisStatus.PROMOTED

    def test_no_promote_confirmed_low_confidence(
        self, consolidator: KnowledgeConsolidator
    ) -> None:
        now = datetime.utcnow()
        h = _make_hypothesis("h1", status=HypothesisStatus.CONFIRMED, confidence=0.5)
        promoted, demoted = consolidator._update_hypothesis_status([h], now)
        assert len(promoted) == 0
        assert h.status == HypothesisStatus.CONFIRMED

    def test_demote_refuted(self, consolidator: KnowledgeConsolidator) -> None:
        now = datetime.utcnow()
        h = _make_hypothesis("h1", status=HypothesisStatus.REFUTED)
        promoted, demoted = consolidator._update_hypothesis_status([h], now)
        assert len(demoted) == 1
        assert h.last_updated == now

    def test_generated_not_promoted_or_demoted(
        self, consolidator: KnowledgeConsolidator
    ) -> None:
        now = datetime.utcnow()
        h = _make_hypothesis("h1", status=HypothesisStatus.GENERATED)
        promoted, demoted = consolidator._update_hypothesis_status([h], now)
        assert len(promoted) == 0
        assert len(demoted) == 0


# ── Stale Hypothesis Flagging ─────────────────────────────────────


class TestStaleHypotheses:
    def test_flag_stale_old_hypothesis(self, consolidator: KnowledgeConsolidator) -> None:
        now = datetime.utcnow()
        old = now - timedelta(days=100)
        h = _make_hypothesis("h1", status=HypothesisStatus.GENERATED, last_updated=old)
        stale = consolidator._flag_stale_hypotheses([h], now)
        assert len(stale) == 1
        assert h.status == HypothesisStatus.STALE

    def test_recent_hypothesis_not_stale(self, consolidator: KnowledgeConsolidator) -> None:
        now = datetime.utcnow()
        recent = now - timedelta(days=10)
        h = _make_hypothesis("h1", status=HypothesisStatus.GENERATED, last_updated=recent)
        stale = consolidator._flag_stale_hypotheses([h], now)
        assert len(stale) == 0

    def test_promoted_never_flagged_stale(self, consolidator: KnowledgeConsolidator) -> None:
        now = datetime.utcnow()
        old = now - timedelta(days=200)
        h = _make_hypothesis("h1", status=HypothesisStatus.PROMOTED, last_updated=old)
        stale = consolidator._flag_stale_hypotheses([h], now)
        assert len(stale) == 0

    def test_refuted_never_flagged_stale(self, consolidator: KnowledgeConsolidator) -> None:
        now = datetime.utcnow()
        old = now - timedelta(days=200)
        h = _make_hypothesis("h1", status=HypothesisStatus.REFUTED, last_updated=old)
        stale = consolidator._flag_stale_hypotheses([h], now)
        assert len(stale) == 0


# ── Cosine Similarity ────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert KnowledgeConsolidator._cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert KnowledgeConsolidator._cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert KnowledgeConsolidator._cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_empty_vectors(self) -> None:
        assert KnowledgeConsolidator._cosine_similarity([], []) == 0.0

    def test_zero_vector(self) -> None:
        assert KnowledgeConsolidator._cosine_similarity([0, 0], [1, 0]) == 0.0

    def test_mismatched_lengths(self) -> None:
        assert KnowledgeConsolidator._cosine_similarity([1, 0], [1, 0, 0]) == 0.0


# ── Full Consolidation Pipeline ───────────────────────────────────


class TestFullConsolidation:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, consolidator: KnowledgeConsolidator) -> None:
        now = datetime.utcnow()
        old = now - timedelta(days=3650)
        stale_hyp_date = now - timedelta(days=100)

        claims = [
            _make_claim("c1", confidence=0.9, extracted_at=now, source_paper_ids=["p1"]),
            _make_claim("c2", confidence=0.9, extracted_at=now, source_paper_ids=["p2"]),
            _make_claim("c3", confidence=0.15, extracted_at=old, half_life_days=365),
        ]

        hypotheses = [
            _make_hypothesis("h1", status=HypothesisStatus.CONFIRMED, confidence=0.9),
            _make_hypothesis("h2", status=HypothesisStatus.REFUTED),
            _make_hypothesis("h3", status=HypothesisStatus.GENERATED, last_updated=stale_hyp_date),
        ]

        # c1 and c2 are similar
        embeddings = {
            "c1": [1.0, 0.0],
            "c2": [1.0, 0.0],
            "c3": [0.0, 1.0],
        }

        report = await consolidator.run_consolidation(claims, hypotheses, embeddings)

        assert isinstance(report, ConsolidationReport)
        assert report.duplicates_merged == 1
        assert report.claims_decayed >= 1  # c3 should decay to stale
        assert report.hypotheses_promoted == 1  # h1
        assert report.hypotheses_demoted == 1  # h2
        assert report.hypotheses_flagged_stale == 1  # h3

    @pytest.mark.asyncio
    async def test_empty_inputs(self, consolidator: KnowledgeConsolidator) -> None:
        report = await consolidator.run_consolidation([], [], None)
        assert report.duplicates_merged == 0
        assert report.claims_decayed == 0
        assert report.hypotheses_promoted == 0


class TestConsolidationReport:
    def test_defaults(self) -> None:
        report = ConsolidationReport()
        assert report.duplicates_merged == 0
        assert report.claims_decayed == 0
        assert report.run_at is not None
