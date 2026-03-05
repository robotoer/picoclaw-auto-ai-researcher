"""Tests for SUNFIREEvaluator."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from auto_researcher.config import SUNFIREWeights
from auto_researcher.evaluation.sunfire import (
    COMMUNITY_GAMING_MIN_VARIANCE,
    NOVELTY_HACKING_THRESHOLD,
    SURPRISE_MANIPULATION_THRESHOLD,
    SUNFIREEvaluator,
)
from auto_researcher.models.research_thread import ResearchThread
from auto_researcher.models.reward import SUNFIREScore
from auto_researcher.utils.llm import LLMClient


def make_thread(**kwargs) -> ResearchThread:
    defaults = dict(
        id="thread-1",
        gap_id="gap-1",
        title="Test Research Thread",
        hypothesis_ids=["h1", "h2"],
        experiment_ids=["e1"],
        result_ids=["r1"],
        draft_sections={
            "abstract": "This paper studies X.",
            "introduction": "We introduce a novel approach.",
            "methodology": "Our method works as follows.",
            "results": "We observe significant improvements.",
            "conclusion": "In conclusion, X is effective.",
        },
        compute_budget=100.0,
        compute_used=30.0,
    )
    defaults.update(kwargs)
    return ResearchThread(**defaults)


@pytest.fixture
def mock_llm():
    llm = AsyncMock(spec=LLMClient)
    llm.generate_structured = AsyncMock(return_value={"score": 0.7, "justification": "Good"})
    return llm


@pytest.fixture
def weights():
    return SUNFIREWeights()


@pytest.fixture
def evaluator(mock_llm, weights):
    return SUNFIREEvaluator(llm=mock_llm, weights=weights)


class TestHeuristicScore:
    def test_rigor_all_present(self, evaluator):
        thread = make_thread()
        score = evaluator._heuristic_score("rigor", thread)
        assert score == 1.0

    def test_rigor_none_present(self, evaluator):
        thread = make_thread(experiment_ids=[], result_ids=[], draft_sections={})
        score = evaluator._heuristic_score("rigor", thread)
        assert score == 0.0

    def test_feasibility_low_compute(self, evaluator):
        thread = make_thread(compute_used=10.0, compute_budget=100.0)
        score = evaluator._heuristic_score("feasibility", thread)
        assert score == pytest.approx(0.9)

    def test_feasibility_high_compute(self, evaluator):
        thread = make_thread(compute_used=100.0, compute_budget=100.0)
        score = evaluator._heuristic_score("feasibility", thread)
        assert score == pytest.approx(0.0)

    def test_novelty_multiple_hypotheses(self, evaluator):
        thread = make_thread(hypothesis_ids=["h1", "h2", "h3", "h4", "h5"])
        score = evaluator._heuristic_score("novelty", thread)
        assert score == 1.0

    def test_novelty_no_hypotheses(self, evaluator):
        thread = make_thread(hypothesis_ids=[])
        score = evaluator._heuristic_score("novelty", thread)
        assert score == 0.0

    def test_unknown_dimension_returns_neutral(self, evaluator):
        thread = make_thread()
        score = evaluator._heuristic_score("surprise", thread)
        assert score == 0.5


class TestAntiGaming:
    def test_no_adjustment_normal_scores(self, evaluator):
        score = SUNFIREScore(
            surprise=0.6, usefulness=0.7, novelty=0.5,
            feasibility=0.6, impact_breadth=0.5, rigor=0.7, elegance=0.6,
        )
        adjusted = evaluator._anti_gaming_adjustment(score)
        assert adjusted == score

    def test_novelty_hacking_detected(self, evaluator):
        score = SUNFIREScore(
            surprise=0.5, usefulness=0.5, novelty=0.98,
            feasibility=0.5, impact_breadth=0.5, rigor=0.2, elegance=0.5,
        )
        adjusted = evaluator._anti_gaming_adjustment(score)
        assert adjusted.novelty < score.novelty
        assert adjusted.novelty == pytest.approx(0.98 * 0.7)

    def test_surprise_manipulation_detected(self, evaluator):
        score = SUNFIREScore(
            surprise=0.98, usefulness=0.1, novelty=0.5,
            feasibility=0.5, impact_breadth=0.5, rigor=0.5, elegance=0.5,
        )
        adjusted = evaluator._anti_gaming_adjustment(score)
        assert adjusted.surprise < score.surprise
        assert adjusted.surprise == pytest.approx(0.98 * 0.7)

    def test_community_gaming_all_high_similar(self, evaluator):
        score = SUNFIREScore(
            surprise=0.9, usefulness=0.9, novelty=0.9,
            feasibility=0.9, impact_breadth=0.9, rigor=0.9, elegance=0.9,
        )
        adjusted = evaluator._anti_gaming_adjustment(score)
        # All dimensions should be penalized by 0.85
        assert adjusted.surprise == pytest.approx(0.9 * 0.85)
        assert adjusted.usefulness == pytest.approx(0.9 * 0.85)

    def test_no_community_gaming_varied_scores(self, evaluator):
        score = SUNFIREScore(
            surprise=0.9, usefulness=0.3, novelty=0.8,
            feasibility=0.5, impact_breadth=0.7, rigor=0.9, elegance=0.4,
        )
        adjusted = evaluator._anti_gaming_adjustment(score)
        # High variance, should not trigger community gaming
        assert adjusted.surprise == score.surprise

    def test_low_scores_not_community_gaming(self, evaluator):
        # All similar but low mean - should NOT trigger (mean > 0.8 required)
        score = SUNFIREScore(
            surprise=0.3, usefulness=0.3, novelty=0.3,
            feasibility=0.3, impact_breadth=0.3, rigor=0.3, elegance=0.3,
        )
        adjusted = evaluator._anti_gaming_adjustment(score)
        assert adjusted == score


class TestCompositeScore:
    def test_default_weights(self, evaluator):
        score = SUNFIREScore(
            surprise=1.0, usefulness=1.0, novelty=1.0,
            feasibility=1.0, impact_breadth=1.0, rigor=1.0, elegance=1.0,
        )
        composite = evaluator.composite_score(score)
        assert composite == pytest.approx(1.0)

    def test_zero_scores(self, evaluator):
        score = SUNFIREScore()
        composite = evaluator.composite_score(score)
        assert composite == pytest.approx(0.0)

    def test_partial_scores(self, evaluator):
        score = SUNFIREScore(surprise=1.0)
        composite = evaluator.composite_score(score)
        assert composite == pytest.approx(0.15)  # default surprise weight


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_evaluate_calls_llm_for_each_dimension(self, evaluator, mock_llm):
        thread = make_thread()
        result = await evaluator.evaluate(thread)
        # 7 dimensions
        assert mock_llm.generate_structured.call_count == 7
        assert isinstance(result, SUNFIREScore)

    @pytest.mark.asyncio
    async def test_evaluate_blends_llm_and_heuristic(self, evaluator, mock_llm):
        mock_llm.generate_structured.return_value = {"score": 0.8, "justification": "Good"}
        thread = make_thread()
        result = await evaluator.evaluate(thread)
        # For rigor: 0.7 * 0.8 + 0.3 * 1.0 (all heuristic components present) = 0.86
        assert result.rigor == pytest.approx(0.86)

    @pytest.mark.asyncio
    async def test_evaluate_llm_failure_uses_default(self, evaluator, mock_llm):
        mock_llm.generate_structured.side_effect = Exception("LLM failed")
        thread = make_thread()
        result = await evaluator.evaluate(thread)
        # Falls back to 0.5 for LLM, then blends with heuristic
        assert isinstance(result, SUNFIREScore)

    @pytest.mark.asyncio
    async def test_evaluate_appends_to_history(self, evaluator, mock_llm):
        thread = make_thread()
        assert len(evaluator._history) == 0
        await evaluator.evaluate(thread)
        assert len(evaluator._history) == 1

    @pytest.mark.asyncio
    async def test_evaluate_clamps_scores(self, evaluator, mock_llm):
        mock_llm.generate_structured.return_value = {"score": 1.5, "justification": "Over"}
        thread = make_thread()
        result = await evaluator.evaluate(thread)
        # LLM score clamped to 1.0, blended: 0.7*1.0 + 0.3*heuristic
        for dim in ["surprise", "usefulness", "novelty", "feasibility", "impact_breadth", "rigor", "elegance"]:
            assert 0.0 <= getattr(result, dim) <= 1.0


class TestExtractContent:
    def test_includes_title_and_sections(self):
        thread = make_thread()
        content = SUNFIREEvaluator._extract_content(thread)
        assert "Test Research Thread" in content
        assert "Introduction" in content
        assert "Methodology" in content

    def test_missing_sections(self):
        thread = make_thread(draft_sections={})
        content = SUNFIREEvaluator._extract_content(thread)
        assert "Test Research Thread" in content
