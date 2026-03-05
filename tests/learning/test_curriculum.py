"""Tests for CurriculumPlanner and TopicCandidate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from auto_researcher.config import CurriculumConfig
from auto_researcher.learning.curriculum_planner import CurriculumPlanner, TopicCandidate
from auto_researcher.models.memory import MetaMemoryEntry
from auto_researcher.utils.llm import LLMClient


@pytest.fixture
def config() -> CurriculumConfig:
    return CurriculumConfig(
        zpd_min_overlap=0.4,
        zpd_max_overlap=0.8,
        field_momentum_weight=0.3,
        gap_density_weight=0.4,
        strategic_value_weight=0.3,
    )


@pytest.fixture
def mock_llm() -> LLMClient:
    llm = MagicMock(spec=LLMClient)
    llm.generate_structured = AsyncMock(return_value={
        "prerequisite_overlap": 0.6,
        "prerequisites": [],
        "difficulty": 0.5,
        "estimated_learning_effort": 0.5,
    })
    return llm


@pytest.fixture
def planner(mock_llm: LLMClient, config: CurriculumConfig) -> CurriculumPlanner:
    return CurriculumPlanner(mock_llm, config)


# ── TopicCandidate ZPD Scoring ────────────────────────────────────


class TestTopicCandidateZPD:
    def test_zpd_in_sweet_spot_center(self) -> None:
        """Midpoint of ZPD range should score close to 1.0."""
        tc = TopicCandidate("topic", prerequisite_overlap=0.6)
        score = tc.zpd_score(0.4, 0.8)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_zpd_below_min(self) -> None:
        """Below min overlap: score should be < 0.5."""
        tc = TopicCandidate("topic", prerequisite_overlap=0.1)
        score = tc.zpd_score(0.4, 0.8)
        assert score < 0.5
        assert score >= 0.0

    def test_zpd_above_max(self) -> None:
        """Above max overlap: score decreases."""
        tc = TopicCandidate("topic", prerequisite_overlap=0.95)
        score = tc.zpd_score(0.4, 0.8)
        assert score < 1.0
        assert score >= 0.0

    def test_zpd_at_min_boundary(self) -> None:
        tc = TopicCandidate("topic", prerequisite_overlap=0.4)
        score = tc.zpd_score(0.4, 0.8)
        # At boundary, should be in the sweet spot
        assert score > 0.5

    def test_zpd_at_max_boundary(self) -> None:
        tc = TopicCandidate("topic", prerequisite_overlap=0.8)
        score = tc.zpd_score(0.4, 0.8)
        assert score > 0.5

    def test_zpd_zero_overlap(self) -> None:
        tc = TopicCandidate("topic", prerequisite_overlap=0.0)
        score = tc.zpd_score(0.4, 0.8)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_zpd_full_overlap(self) -> None:
        tc = TopicCandidate("topic", prerequisite_overlap=1.0)
        score = tc.zpd_score(0.4, 0.8)
        assert score == pytest.approx(0.0, abs=0.01)


# ── CurriculumPlanner ────────────────────────────────────────────


class TestCurriculumPlanner:
    def test_load_competencies(self, planner: CurriculumPlanner) -> None:
        entries = [
            MetaMemoryEntry(topic="NLP", competence_level=0.7),
            MetaMemoryEntry(topic="CV", competence_level=0.3),
        ]
        planner.load_competencies(entries)
        assert planner.get_competence("NLP") == pytest.approx(0.7)
        assert planner.get_competence("CV") == pytest.approx(0.3)
        assert planner.get_competence("RL") == 0.0

    @pytest.mark.asyncio
    async def test_select_next_topics_basic(self, planner: CurriculumPlanner) -> None:
        candidates = [
            TopicCandidate("A", prerequisite_overlap=0.6, field_momentum=0.8,
                           gap_density=0.7, strategic_value=0.9),
            TopicCandidate("B", prerequisite_overlap=0.2, field_momentum=0.3,
                           gap_density=0.2, strategic_value=0.1),
            TopicCandidate("C", prerequisite_overlap=0.7, field_momentum=0.5,
                           gap_density=0.5, strategic_value=0.5),
        ]
        selected = await planner.select_next_topics(candidates, n=2)
        assert len(selected) == 2
        # A should rank higher due to being in ZPD sweet spot + high momentum/strategic value
        assert selected[0].topic == "A"

    @pytest.mark.asyncio
    async def test_select_next_topics_uses_llm_for_default_overlap(
        self, planner: CurriculumPlanner, mock_llm: LLMClient
    ) -> None:
        """When overlap is default 0.5 and competencies exist, LLM is called."""
        planner.load_competencies([
            MetaMemoryEntry(topic="NLP", competence_level=0.7),
        ])

        candidates = [
            TopicCandidate("X", prerequisite_overlap=0.5),  # default -> triggers LLM
        ]
        await planner.select_next_topics(candidates, n=1)
        mock_llm.generate_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_next_topics_no_llm_when_overlap_set(
        self, planner: CurriculumPlanner, mock_llm: LLMClient
    ) -> None:
        """When overlap is explicitly set (not 0.5), LLM should not be called."""
        planner.load_competencies([
            MetaMemoryEntry(topic="NLP", competence_level=0.7),
        ])

        candidates = [
            TopicCandidate("X", prerequisite_overlap=0.6),  # explicitly set
        ]
        await planner.select_next_topics(candidates, n=1)
        mock_llm.generate_structured.assert_not_called()

    @pytest.mark.asyncio
    async def test_select_next_topics_llm_failure_defaults(
        self, planner: CurriculumPlanner, mock_llm: LLMClient
    ) -> None:
        """If LLM fails, overlap should default to 0.5."""
        planner.load_competencies([
            MetaMemoryEntry(topic="NLP", competence_level=0.7),
        ])
        mock_llm.generate_structured = AsyncMock(side_effect=RuntimeError("LLM down"))

        candidates = [
            TopicCandidate("X", prerequisite_overlap=0.5),
        ]
        selected = await planner.select_next_topics(candidates, n=1)
        assert len(selected) == 1
        assert selected[0].prerequisite_overlap == 0.5  # fallback

    def test_record_learning_outcome_new_topic(self, planner: CurriculumPlanner) -> None:
        planner.record_learning_outcome("new_topic", 0.8)
        assert planner.get_competence("new_topic") == pytest.approx(0.8 * 0.3)

    def test_record_learning_outcome_existing_topic(self, planner: CurriculumPlanner) -> None:
        planner.load_competencies([
            MetaMemoryEntry(topic="NLP", competence_level=0.5),
        ])
        planner.record_learning_outcome("NLP", 0.6)
        assert planner.get_competence("NLP") == pytest.approx(0.5 + 0.6 * 0.3)

    def test_record_learning_outcome_caps_at_1(self, planner: CurriculumPlanner) -> None:
        planner.load_competencies([
            MetaMemoryEntry(topic="NLP", competence_level=0.95),
        ])
        planner.record_learning_outcome("NLP", 1.0)
        assert planner.get_competence("NLP") == 1.0

    def test_get_knowledge_frontier(self, planner: CurriculumPlanner) -> None:
        planner.load_competencies([
            MetaMemoryEntry(topic="A", competence_level=0.1),  # too low
            MetaMemoryEntry(topic="B", competence_level=0.5),  # in range
            MetaMemoryEntry(topic="C", competence_level=0.7),  # in range
            MetaMemoryEntry(topic="D", competence_level=0.9),  # too high
        ])
        frontier = planner.get_knowledge_frontier(threshold=0.3)
        assert set(frontier) == {"B", "C"}

    def test_suggest_review_topics(self, planner: CurriculumPlanner) -> None:
        now = datetime.now(UTC)
        old_date = now - timedelta(days=60)
        recent_date = now - timedelta(days=5)

        planner.load_competencies([
            MetaMemoryEntry(topic="old_known", competence_level=0.6, last_assessed=old_date),
            MetaMemoryEntry(topic="recent_known", competence_level=0.6, last_assessed=recent_date),
            MetaMemoryEntry(topic="old_low", competence_level=0.1, last_assessed=old_date),
        ])

        review = planner.suggest_review_topics(decay_days=30)
        assert "old_known" in review
        assert "recent_known" not in review
        assert "old_low" not in review  # competence too low

    @pytest.mark.asyncio
    async def test_field_momentum_weighting(self, planner: CurriculumPlanner) -> None:
        """Topics with higher field momentum should score higher."""
        candidates = [
            TopicCandidate("low_momentum", prerequisite_overlap=0.6,
                           field_momentum=0.1, gap_density=0.5, strategic_value=0.5),
            TopicCandidate("high_momentum", prerequisite_overlap=0.6,
                           field_momentum=0.9, gap_density=0.5, strategic_value=0.5),
        ]
        selected = await planner.select_next_topics(candidates, n=2)
        assert selected[0].topic == "high_momentum"
