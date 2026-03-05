"""Comprehensive tests for all data models."""

import math
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from auto_researcher.models.paper import Paper, PaperMetadata, ProcessingLevel
from auto_researcher.models.claim import Claim, ClaimRelation, ClaimStatus
from auto_researcher.models.hypothesis import (
    Hypothesis,
    HypothesisStatus,
    FalsificationCriteria,
)
from auto_researcher.models.gap import Gap, GapType, GapStatus, GapNode, GapEdge
from auto_researcher.models.research_thread import (
    ResearchThread,
    ThreadStatus,
    ExperimentDesign,
    ExperimentResult,
)
from auto_researcher.models.reward import (
    SUNFIREScore,
    IWPGReward,
    PeerReviewResult,
    ReviewDecision,
    ReviewComment,
)
from auto_researcher.models.memory import (
    EpisodicEntry,
    MemoryType,
    MetaMemoryEntry,
    ProceduralEntry,
)
from auto_researcher.models.agent import AgentMessage, AgentRole, AgentState


# ── Helpers ──────────────────────────────────────────────────────────


def _make_paper_metadata(**overrides) -> PaperMetadata:
    defaults = dict(
        arxiv_id="2401.00001",
        title="Test Paper",
        authors=["Alice", "Bob"],
        abstract="An abstract.",
        categories=["cs.AI"],
        published=datetime(2024, 1, 1),
    )
    defaults.update(overrides)
    return PaperMetadata(**defaults)


# ── ProcessingLevel ──────────────────────────────────────────────────


class TestProcessingLevel:
    def test_values(self):
        assert ProcessingLevel.UNPROCESSED == "unprocessed"
        assert ProcessingLevel.ABSTRACT_ONLY == "abstract_only"
        assert ProcessingLevel.FULL_TEXT == "full_text"


# ── PaperMetadata ────────────────────────────────────────────────────


class TestPaperMetadata:
    def test_required_fields(self):
        pm = _make_paper_metadata()
        assert pm.arxiv_id == "2401.00001"
        assert pm.title == "Test Paper"
        assert pm.authors == ["Alice", "Bob"]

    def test_optional_fields_default_none(self):
        pm = _make_paper_metadata()
        assert pm.updated is None
        assert pm.doi is None
        assert pm.journal_ref is None
        assert pm.pdf_url is None

    def test_serialization_round_trip(self):
        pm = _make_paper_metadata(doi="10.1234/test")
        data = pm.model_dump()
        restored = PaperMetadata(**data)
        assert restored == pm


# ── Paper ────────────────────────────────────────────────────────────


class TestPaper:
    def test_defaults(self):
        p = Paper(metadata=_make_paper_metadata())
        assert p.processing_level == ProcessingLevel.UNPROCESSED
        assert p.relevance_score == 0.0
        assert p.full_text is None
        assert p.sections == {}
        assert p.tables == []
        assert p.references == []

    def test_properties(self):
        p = Paper(metadata=_make_paper_metadata())
        assert p.arxiv_id == "2401.00001"
        assert p.title == "Test Paper"

    def test_serialization(self):
        p = Paper(metadata=_make_paper_metadata(), relevance_score=0.8)
        data = p.model_dump()
        assert data["relevance_score"] == 0.8
        restored = Paper(**data)
        assert restored.relevance_score == 0.8


# ── ClaimRelation / ClaimStatus ──────────────────────────────────────


class TestClaimEnums:
    def test_claim_relation_values(self):
        assert ClaimRelation.OUTPERFORMS == "outperforms"
        assert ClaimRelation.CONTRADICTS == "contradicts"

    def test_claim_status_values(self):
        assert ClaimStatus.EXTRACTED == "extracted"
        assert ClaimStatus.STALE == "stale"


# ── Claim ────────────────────────────────────────────────────────────


class TestClaim:
    def test_creation(self):
        c = Claim(
            entity_1="ModelA",
            relation=ClaimRelation.OUTPERFORMS,
            entity_2="ModelB",
            confidence=0.9,
        )
        assert c.entity_1 == "ModelA"
        assert c.status == ClaimStatus.EXTRACTED

    def test_confidence_validation(self):
        with pytest.raises(ValidationError):
            Claim(
                entity_1="A",
                relation=ClaimRelation.SUPPORTS,
                entity_2="B",
                confidence=1.5,
            )
        with pytest.raises(ValidationError):
            Claim(
                entity_1="A",
                relation=ClaimRelation.SUPPORTS,
                entity_2="B",
                confidence=-0.1,
            )

    def test_decayed_confidence_no_decay(self):
        now = datetime(2024, 6, 1)
        c = Claim(
            entity_1="A",
            relation=ClaimRelation.SUPPORTS,
            entity_2="B",
            confidence=1.0,
            extracted_at=now,
        )
        assert c.decayed_confidence(now) == pytest.approx(1.0)

    def test_decayed_confidence_half_life(self):
        extracted = datetime(2024, 1, 1)
        c = Claim(
            entity_1="A",
            relation=ClaimRelation.SUPPORTS,
            entity_2="B",
            confidence=1.0,
            half_life_days=365,
            extracted_at=extracted,
        )
        one_half_life_later = extracted + timedelta(days=365)
        decayed = c.decayed_confidence(one_half_life_later)
        assert decayed == pytest.approx(0.5, abs=0.01)

    def test_decayed_confidence_custom_half_life(self):
        extracted = datetime(2024, 1, 1)
        c = Claim(
            entity_1="A",
            relation=ClaimRelation.SUPPORTS,
            entity_2="B",
            confidence=0.8,
            half_life_days=30,
            extracted_at=extracted,
        )
        later = extracted + timedelta(days=60)  # 2 half-lives
        decayed = c.decayed_confidence(later)
        assert decayed == pytest.approx(0.8 * 0.25, abs=0.02)

    def test_defaults(self):
        c = Claim(
            entity_1="A",
            relation=ClaimRelation.SUPPORTS,
            entity_2="B",
            confidence=0.5,
        )
        assert c.half_life_days == 365
        assert c.source_paper_ids == []
        assert c.contradicting_claim_ids == []
        assert c.supporting_claim_ids == []


# ── HypothesisStatus / FalsificationCriteria ─────────────────────────


class TestHypothesisEnums:
    def test_status_values(self):
        assert HypothesisStatus.GENERATED == "generated"
        assert HypothesisStatus.PROMOTED == "promoted"


class TestFalsificationCriteria:
    def test_creation(self):
        fc = FalsificationCriteria(
            description="Test if X holds",
            test_method="Run benchmark Y",
            expected_outcome_if_true="Score > 0.9",
            expected_outcome_if_false="Score < 0.5",
        )
        assert fc.test_method == "Run benchmark Y"


# ── Hypothesis ───────────────────────────────────────────────────────


class TestHypothesis:
    def test_creation_defaults(self):
        h = Hypothesis(
            entity_1="MethodA",
            relation="improves",
            entity_2="TaskB",
            confidence=0.7,
        )
        assert h.status == HypothesisStatus.GENERATED
        assert h.novelty_score == 0.0
        assert h.granularity == "medium"
        assert h.falsification_criteria == []

    def test_confidence_validation(self):
        with pytest.raises(ValidationError):
            Hypothesis(
                entity_1="A",
                relation="r",
                entity_2="B",
                confidence=2.0,
            )

    def test_serialization(self):
        h = Hypothesis(
            entity_1="A",
            relation="r",
            entity_2="B",
            falsification_criteria=[
                FalsificationCriteria(
                    description="d",
                    test_method="t",
                    expected_outcome_if_true="y",
                    expected_outcome_if_false="n",
                )
            ],
        )
        data = h.model_dump()
        restored = Hypothesis(**data)
        assert len(restored.falsification_criteria) == 1


# ── GapType / GapStatus ─────────────────────────────────────────────


class TestGapEnums:
    def test_gap_type(self):
        assert GapType.EMPIRICAL == "empirical"
        assert GapType.REPLICATION == "replication"

    def test_gap_status(self):
        assert GapStatus.OPEN == "open"
        assert GapStatus.FILLED == "filled"


# ── Gap ──────────────────────────────────────────────────────────────


class TestGap:
    def test_creation_defaults(self):
        g = Gap(gap_type=GapType.EMPIRICAL, description="Need more data")
        assert g.importance == 0.5
        assert g.tractability == 0.5
        assert g.status == GapStatus.OPEN

    def test_priority_score_formula(self):
        g = Gap(
            gap_type=GapType.EMPIRICAL,
            description="test",
            importance=0.8,
            tractability=0.6,
            novelty=0.9,
            timeliness=0.5,
        )
        expected = (0.8 * 0.6 * 0.9) / (1.0 + 0.5)
        assert g.priority_score() == pytest.approx(expected)

    def test_priority_score_zero_urgency(self):
        g = Gap(
            gap_type=GapType.EMPIRICAL,
            description="test",
            importance=1.0,
            tractability=1.0,
            novelty=1.0,
            timeliness=0.0,
        )
        assert g.priority_score() == pytest.approx(1.0)

    def test_validation_bounds(self):
        with pytest.raises(ValidationError):
            Gap(gap_type=GapType.EMPIRICAL, description="test", importance=1.5)


# ── GapNode ──────────────────────────────────────────────────────────


class TestGapNode:
    def test_creation(self):
        n = GapNode(id="n1", node_type="concept", label="Transformers")
        assert n.coverage_score == 0.0
        assert n.paper_count == 0
        assert n.adjacent_node_ids == []
        assert n.gaps == []

    def test_coverage_validation(self):
        with pytest.raises(ValidationError):
            GapNode(id="n1", node_type="concept", label="X", coverage_score=2.0)


# ── GapEdge ──────────────────────────────────────────────────────────


class TestGapEdge:
    def test_creation(self):
        e = GapEdge(source_id="a", target_id="b", edge_type="builds_on")
        assert e.weight == 1.0
        assert e.is_negative is False
        assert e.conditions == ""

    def test_negative_edge(self):
        e = GapEdge(
            source_id="a",
            target_id="b",
            edge_type="should_connect_but_doesnt",
            is_negative=True,
        )
        assert e.is_negative is True


# ── ResearchThread / ExperimentDesign / ExperimentResult ─────────────


class TestResearchThread:
    def test_defaults(self):
        rt = ResearchThread(gap_id="g1", title="Thread 1")
        assert rt.status == ThreadStatus.INITIALIZED
        assert rt.compute_budget == 100.0
        assert rt.compute_used == 0.0
        assert rt.revision_count == 0
        assert rt.published_at is None

    def test_serialization(self):
        rt = ResearchThread(
            gap_id="g1",
            title="Thread 1",
            hypothesis_ids=["h1"],
            draft_sections={"intro": "text"},
        )
        data = rt.model_dump()
        restored = ResearchThread(**data)
        assert restored.hypothesis_ids == ["h1"]
        assert restored.draft_sections["intro"] == "text"


class TestExperimentDesign:
    def test_creation(self):
        ed = ExperimentDesign(
            hypothesis_id="h1",
            description="Run benchmark",
            methodology="Standard eval",
            datasets=["MMLU"],
            metrics=["accuracy"],
        )
        assert ed.estimated_compute_hours == 0.0
        assert ed.datasets == ["MMLU"]


class TestExperimentResult:
    def test_creation(self):
        er = ExperimentResult(
            experiment_id="e1",
            hypothesis_id="h1",
            outcome="confirmed",
            metrics={"accuracy": 0.95},
        )
        assert er.metrics["accuracy"] == 0.95
        assert er.limitations == []


# ── ThreadStatus ─────────────────────────────────────────────────────


class TestThreadStatus:
    def test_values(self):
        assert ThreadStatus.INITIALIZED == "initialized"
        assert ThreadStatus.PUBLISHED == "published"
        assert ThreadStatus.ABANDONED == "abandoned"


# ── SUNFIREScore ─────────────────────────────────────────────────────


class TestSUNFIREScore:
    def test_defaults(self):
        s = SUNFIREScore()
        assert s.surprise == 0.0
        assert s.composite() == 0.0

    def test_composite_all_ones(self):
        s = SUNFIREScore(
            surprise=1.0,
            usefulness=1.0,
            novelty=1.0,
            feasibility=1.0,
            impact_breadth=1.0,
            rigor=1.0,
            elegance=1.0,
        )
        assert s.composite() == pytest.approx(1.0)

    def test_composite_custom_weights(self):
        s = SUNFIREScore(surprise=1.0)
        result = s.composite(w_s=1.0, w_u=0.0, w_n=0.0, w_f=0.0, w_i=0.0, w_r=0.0, w_e=0.0)
        assert result == pytest.approx(1.0)

    def test_composite_weighted_sum(self):
        s = SUNFIREScore(
            surprise=0.5,
            usefulness=0.8,
            novelty=0.6,
            feasibility=0.7,
            impact_breadth=0.4,
            rigor=0.9,
            elegance=0.3,
        )
        expected = (
            0.15 * 0.5
            + 0.20 * 0.8
            + 0.20 * 0.6
            + 0.10 * 0.7
            + 0.15 * 0.4
            + 0.10 * 0.9
            + 0.10 * 0.3
        )
        assert s.composite() == pytest.approx(expected)

    def test_validation(self):
        with pytest.raises(ValidationError):
            SUNFIREScore(surprise=1.5)


# ── IWPGReward ───────────────────────────────────────────────────────


class TestIWPGReward:
    def test_defaults(self):
        r = IWPGReward()
        assert r.total() == 0.0

    def test_total_positive_only(self):
        r = IWPGReward(novelty=1.0, surprise=1.0, utility=1.0, reproducibility=1.0)
        expected = 0.25 * 1.0 + 0.15 * 1.0 + 0.20 * 1.0 + 0.15 * 1.0
        assert r.total() == pytest.approx(expected)

    def test_total_with_penalties(self):
        r = IWPGReward(redundancy=1.0, complexity_cost=1.0)
        expected = -0.15 * 1.0 - 0.10 * 1.0
        assert r.total() == pytest.approx(expected)

    def test_total_custom_weights(self):
        r = IWPGReward(novelty=0.5, redundancy=0.3)
        result = r.total(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, epsilon=1.0, zeta=0.0)
        assert result == pytest.approx(0.5 - 0.3)

    def test_validation(self):
        with pytest.raises(ValidationError):
            IWPGReward(novelty=-0.1)


# ── PeerReviewResult ─────────────────────────────────────────────────


class TestPeerReviewResult:
    def test_creation(self):
        pr = PeerReviewResult(
            thread_id="t1",
            decision=ReviewDecision.ACCEPT,
            overall_score=0.85,
        )
        assert pr.decision == ReviewDecision.ACCEPT
        assert pr.reviews == []
        assert pr.round_number == 1

    def test_with_reviews(self):
        rc = ReviewComment(
            reviewer_id="r1",
            aspect="methodology",
            comment="Good approach",
        )
        pr = PeerReviewResult(
            thread_id="t1",
            decision=ReviewDecision.REVISE,
            overall_score=0.6,
            reviews=[rc],
        )
        assert len(pr.reviews) == 1
        assert pr.reviews[0].severity == "minor"


class TestReviewDecision:
    def test_values(self):
        assert ReviewDecision.ACCEPT == "accept"
        assert ReviewDecision.REVISE == "revise"
        assert ReviewDecision.REJECT == "reject"


# ── MemoryType ───────────────────────────────────────────────────────


class TestMemoryType:
    def test_values(self):
        assert MemoryType.EPISODIC == "episodic"
        assert MemoryType.PROCEDURAL == "procedural"
        assert MemoryType.META == "meta"


# ── EpisodicEntry ────────────────────────────────────────────────────


class TestEpisodicEntry:
    def test_creation_defaults(self):
        e = EpisodicEntry(content="Something happened")
        assert e.memory_type == MemoryType.EPISODIC
        assert e.importance == 0.5
        assert e.access_count == 0
        assert e.tags == []
        assert e.embedding is None

    def test_validation(self):
        with pytest.raises(ValidationError):
            EpisodicEntry(content="test", importance=2.0)

    def test_serialization(self):
        e = EpisodicEntry(
            content="test",
            tags=["ml", "nlp"],
            context={"agent": "critic"},
        )
        data = e.model_dump()
        restored = EpisodicEntry(**data)
        assert restored.tags == ["ml", "nlp"]
        assert restored.context["agent"] == "critic"


# ── MetaMemoryEntry ──────────────────────────────────────────────────


class TestMetaMemoryEntry:
    def test_creation(self):
        m = MetaMemoryEntry(
            topic="transformers",
            competence_level=0.7,
            confidence=0.6,
        )
        assert m.blind_spots == []
        assert m.knowledge_sources == []

    def test_validation(self):
        with pytest.raises(ValidationError):
            MetaMemoryEntry(topic="x", competence_level=1.5)


# ── ProceduralEntry ──────────────────────────────────────────────────


class TestProceduralEntry:
    def test_creation(self):
        p = ProceduralEntry(
            name="search_papers",
            description="Search arxiv",
            tool_sequence=["arxiv_search", "filter", "rank"],
        )
        assert p.success_rate == 0.5
        assert p.use_count == 0
        assert p.last_used is None

    def test_validation(self):
        with pytest.raises(ValidationError):
            ProceduralEntry(name="x", description="y", success_rate=1.5)


# ── AgentRole / AgentState ───────────────────────────────────────────


class TestAgentRole:
    def test_values(self):
        assert AgentRole.ORCHESTRATOR == "orchestrator"
        assert AgentRole.CRITIC == "critic"
        assert AgentRole.STATISTICIAN == "statistician"


class TestAgentState:
    def test_values(self):
        assert AgentState.IDLE == "idle"
        assert AgentState.ERROR == "error"


# ── AgentMessage ─────────────────────────────────────────────────────


class TestAgentMessage:
    def test_creation_defaults(self):
        m = AgentMessage(
            sender=AgentRole.ORCHESTRATOR,
            task_type="analyze_paper",
        )
        assert m.receiver is None
        assert m.priority == 5
        assert m.expected_output_format == "json"
        assert m.payload == {}

    def test_priority_validation(self):
        with pytest.raises(ValidationError):
            AgentMessage(
                sender=AgentRole.ORCHESTRATOR,
                task_type="test",
                priority=11,
            )
        with pytest.raises(ValidationError):
            AgentMessage(
                sender=AgentRole.ORCHESTRATOR,
                task_type="test",
                priority=-1,
            )

    def test_with_receiver(self):
        m = AgentMessage(
            sender=AgentRole.ORCHESTRATOR,
            receiver=AgentRole.CRITIC,
            task_type="critique_hypothesis",
            payload={"hypothesis_id": "h1"},
        )
        assert m.receiver == AgentRole.CRITIC

    def test_serialization(self):
        m = AgentMessage(
            sender=AgentRole.HYPOTHESIS_GENERATOR,
            receiver=AgentRole.EXPERIMENT_DESIGNER,
            task_type="design_experiment",
            priority=8,
        )
        data = m.model_dump()
        restored = AgentMessage(**data)
        assert restored.sender == AgentRole.HYPOTHESIS_GENERATOR
        assert restored.priority == 8
