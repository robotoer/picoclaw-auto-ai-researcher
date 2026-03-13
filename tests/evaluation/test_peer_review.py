"""Tests for SimulatedPeerReview."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from auto_researcher.config import PeerReviewConfig
from auto_researcher.evaluation.peer_review import SimulatedPeerReview
from auto_researcher.models.research_thread import ResearchThread
from auto_researcher.models.reward import PeerReviewResult, ReviewComment, ReviewDecision
from auto_researcher.utils.llm import LLMClient, LLMResponse


def make_thread(**kwargs) -> ResearchThread:
    defaults = dict(
        id="thread-1",
        gap_id="gap-1",
        title="Test Research Paper",
        hypothesis_ids=["h1"],
        experiment_ids=["e1"],
        result_ids=["r1"],
        draft_sections={
            "abstract": "We present a novel approach.",
            "introduction": "Background and motivation.",
            "methodology": "Our method works as follows.",
            "results": "We observe improvements.",
        },
        compute_budget=100.0,
        compute_used=30.0,
    )
    defaults.update(kwargs)
    return ResearchThread(**defaults)


def make_reviewer_response(score=0.7, recommendation="accept"):
    return {
        "scores": {
            "methodology": 0.7,
            "novelty": 0.8,
            "clarity": 0.7,
            "significance": 0.6,
            "reproducibility": 0.7,
        },
        "comments": [
            {
                "aspect": "methodology",
                "comment": "Well-designed experiments.",
                "severity": "minor",
                "suggestion": "Add ablation study.",
            }
        ],
        "overall_score": score,
        "recommendation": recommendation,
        "summary": "Solid work overall.",
    }


def make_area_chair_response(decision="accept", score=0.8):
    return {
        "decision": decision,
        "meta_review": "Paper is ready for publication.",
        "overall_score": score,
        "key_concerns": [],
    }


def make_author_response():
    return {
        "responses": [
            {"comment_index": 0, "response": "Will add ablation.", "action": "will_fix"}
        ],
        "revision_plan": "Add ablation study and clarify methods.",
    }


@pytest.fixture
def mock_llm():
    llm = AsyncMock(spec=LLMClient)
    return llm


@pytest.fixture
def config():
    return PeerReviewConfig(num_reviewers=3, max_revision_rounds=3)


@pytest.fixture
def reviewer(mock_llm, config):
    return SimulatedPeerReview(llm=mock_llm, config=config)


class TestQualityGateCheck:
    @pytest.mark.asyncio
    async def test_passes_with_all_required(self, reviewer):
        thread = make_thread()
        assert await reviewer.quality_gate_check(thread) is True

    @pytest.mark.asyncio
    async def test_fails_missing_abstract(self, reviewer):
        thread = make_thread(draft_sections={"introduction": "X", "methodology": "Y"})
        assert await reviewer.quality_gate_check(thread) is False

    @pytest.mark.asyncio
    async def test_fails_missing_methodology(self, reviewer):
        thread = make_thread(draft_sections={"abstract": "X", "introduction": "Y"})
        assert await reviewer.quality_gate_check(thread) is False

    @pytest.mark.asyncio
    async def test_fails_no_hypotheses(self, reviewer):
        thread = make_thread(hypothesis_ids=[])
        assert await reviewer.quality_gate_check(thread) is False


class TestRunReviewRound:
    @pytest.mark.asyncio
    async def test_collects_reviews_from_all_reviewers(self, reviewer, mock_llm):
        mock_llm.generate_structured = AsyncMock(side_effect=[
            make_reviewer_response(),  # reviewer 1
            make_reviewer_response(),  # reviewer 2
            make_reviewer_response(),  # reviewer 3
            make_author_response(),    # author response
            make_area_chair_response(),  # area chair
        ])
        thread = make_thread()
        result = await reviewer._run_review_round(thread, round_number=1)
        assert isinstance(result, PeerReviewResult)
        # 3 reviewers x 1 comment each = 3 comments
        assert len(result.reviews) == 3

    @pytest.mark.asyncio
    async def test_area_chair_makes_decision(self, reviewer, mock_llm):
        mock_llm.generate_structured = AsyncMock(side_effect=[
            make_reviewer_response(),
            make_reviewer_response(),
            make_reviewer_response(),
            make_author_response(),
            make_area_chair_response(decision="reject", score=0.3),
        ])
        thread = make_thread()
        result = await reviewer._run_review_round(thread, round_number=1)
        assert result.decision == ReviewDecision.REJECT
        assert result.overall_score == 0.3

    @pytest.mark.asyncio
    async def test_invalid_decision_defaults_to_revise(self, reviewer, mock_llm):
        mock_llm.generate_structured = AsyncMock(side_effect=[
            make_reviewer_response(),
            make_reviewer_response(),
            make_reviewer_response(),
            make_author_response(),
            {"decision": "maybe", "meta_review": "Unsure", "overall_score": 0.5},
        ])
        thread = make_thread()
        result = await reviewer._run_review_round(thread, round_number=1)
        assert result.decision == ReviewDecision.REVISE


class TestReview:
    @pytest.mark.asyncio
    async def test_accept_on_first_round(self, reviewer, mock_llm):
        mock_llm.generate_structured = AsyncMock(side_effect=[
            make_reviewer_response(),
            make_reviewer_response(),
            make_reviewer_response(),
            make_author_response(),
            make_area_chair_response(decision="accept"),
        ])
        thread = make_thread()
        result = await reviewer.review(thread)
        assert result.decision == ReviewDecision.ACCEPT
        assert len(thread.review_history) == 1

    @pytest.mark.asyncio
    async def test_reject_stops_loop(self, reviewer, mock_llm):
        mock_llm.generate_structured = AsyncMock(side_effect=[
            make_reviewer_response(),
            make_reviewer_response(),
            make_reviewer_response(),
            make_author_response(),
            make_area_chair_response(decision="reject", score=0.2),
        ])
        thread = make_thread()
        result = await reviewer.review(thread)
        assert result.decision == ReviewDecision.REJECT

    @pytest.mark.asyncio
    async def test_revise_loops_then_accepts(self, reviewer, mock_llm):
        mock_llm.generate = AsyncMock(return_value=LLMResponse(content="Revised text.", model="test", usage={}))
        mock_llm.generate_structured = AsyncMock(side_effect=[
            # Round 1: revise
            make_reviewer_response(recommendation="revise"),
            make_reviewer_response(recommendation="revise"),
            make_reviewer_response(recommendation="revise"),
            make_author_response(),
            make_area_chair_response(decision="revise", score=0.5),
            # Round 2: accept
            make_reviewer_response(recommendation="accept"),
            make_reviewer_response(recommendation="accept"),
            make_reviewer_response(recommendation="accept"),
            make_author_response(),
            make_area_chair_response(decision="accept", score=0.8),
        ])
        thread = make_thread()
        result = await reviewer.review(thread)
        assert result.decision == ReviewDecision.ACCEPT
        assert thread.revision_count == 1
        assert len(thread.review_history) == 2

    @pytest.mark.asyncio
    async def test_max_revision_rounds(self, mock_llm):
        config = PeerReviewConfig(num_reviewers=3, max_revision_rounds=2)
        reviewer = SimulatedPeerReview(llm=mock_llm, config=config)
        mock_llm.generate = AsyncMock(return_value=LLMResponse(content="Revised.", model="test", usage={}))

        # All rounds return revise
        revise_responses = [
            make_reviewer_response(recommendation="revise"),
            make_reviewer_response(recommendation="revise"),
            make_reviewer_response(recommendation="revise"),
            make_author_response(),
            make_area_chair_response(decision="revise", score=0.5),
        ]
        mock_llm.generate_structured = AsyncMock(side_effect=revise_responses * 2)

        thread = make_thread()
        result = await reviewer.review(thread)
        # After max rounds with revise, last result returned
        assert result.decision == ReviewDecision.REVISE


class TestReviewerFeedbackFailure:
    @pytest.mark.asyncio
    async def test_reviewer_failure_returns_default(self, reviewer, mock_llm):
        mock_llm.generate_structured = AsyncMock(side_effect=[
            Exception("LLM failed"),  # reviewer 1 fails
            make_reviewer_response(),  # reviewer 2
            make_reviewer_response(),  # reviewer 3
            make_author_response(),
            make_area_chair_response(),
        ])
        thread = make_thread()
        result = await reviewer._run_review_round(thread, round_number=1)
        # Still produces a result despite one reviewer failing
        assert isinstance(result, PeerReviewResult)


class TestFormatPaper:
    def test_includes_all_sections(self):
        thread = make_thread()
        text = SimulatedPeerReview._format_paper(thread)
        assert "Abstract" in text
        assert "Introduction" in text
        assert "Methodology" in text
        assert "Results" in text

    def test_missing_sections_uses_title(self):
        thread = make_thread(draft_sections={})
        text = SimulatedPeerReview._format_paper(thread)
        assert "Test Research Paper" in text

    def test_partial_sections(self):
        thread = make_thread(draft_sections={"abstract": "Just an abstract."})
        text = SimulatedPeerReview._format_paper(thread)
        assert "Abstract" in text
        assert "Introduction" not in text


class TestGenerateRevision:
    @pytest.mark.asyncio
    async def test_no_major_issues_skips(self, reviewer, mock_llm):
        review_result = PeerReviewResult(
            thread_id="t1",
            decision=ReviewDecision.REVISE,
            overall_score=0.5,
            reviews=[
                ReviewComment(reviewer_id="r1", aspect="clarity", comment="Minor typo", severity="minor"),
            ],
        )
        thread = make_thread()
        original_sections = dict(thread.draft_sections)
        await reviewer._generate_revision(thread, review_result)
        assert thread.draft_sections == original_sections

    @pytest.mark.asyncio
    async def test_major_issues_revises_sections(self, reviewer, mock_llm):
        mock_llm.generate = AsyncMock(return_value=LLMResponse(content="Revised content.", model="test", usage={}))
        review_result = PeerReviewResult(
            thread_id="t1",
            decision=ReviewDecision.REVISE,
            overall_score=0.5,
            reviews=[
                ReviewComment(
                    reviewer_id="r1", aspect="methodology",
                    comment="Missing controls", severity="major",
                    suggestion="Add control experiments.",
                ),
            ],
        )
        thread = make_thread()
        await reviewer._generate_revision(thread, review_result)
        # Each draft section should have been revised
        assert mock_llm.generate.call_count == len(thread.draft_sections)
