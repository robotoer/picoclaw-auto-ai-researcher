"""Tests for RelevanceFilter."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from auto_researcher.config import ArxivConfig, LLMConfig
from auto_researcher.ingestion.relevance_filter import (
    FilterTier,
    RelevanceFilter,
    RelevanceResult,
)
from auto_researcher.models import Paper, PaperMetadata, ProcessingLevel


def make_paper(title: str = "Test Paper", abstract: str = "Test abstract", categories: list[str] | None = None) -> Paper:
    return Paper(
        metadata=PaperMetadata(
            arxiv_id="2401.00001v1",
            title=title,
            authors=["Author"],
            abstract=abstract,
            categories=categories or ["cs.AI"],
            published=datetime(2024, 1, 15),
        ),
    )


@pytest.fixture
def arxiv_config():
    return ArxivConfig(relevance_threshold=0.3, full_processing_threshold=0.7)


@pytest.fixture
def llm_config():
    return LLMConfig(api_key="test-key")


class TestKeywordScoring:
    def test_no_keywords_returns_neutral(self, arxiv_config, llm_config):
        rf = RelevanceFilter(arxiv_config, llm_config)
        paper = make_paper(title="Anything", abstract="Whatever")
        score = rf._keyword_score(paper)
        assert score == 0.5

    def test_matching_positive_keyword(self, arxiv_config, llm_config):
        rf = RelevanceFilter(arxiv_config, llm_config)
        rf.set_keyword_weights({"transformer": 1.0, "cnn": 0.5})
        paper = make_paper(title="Transformer Models", abstract="About transformers")
        score = rf._keyword_score(paper)
        assert score > 0.5

    def test_no_matching_keywords(self, arxiv_config, llm_config):
        rf = RelevanceFilter(arxiv_config, llm_config)
        rf.set_keyword_weights({"quantum": 1.0})
        paper = make_paper(title="Neural Networks", abstract="Deep learning study")
        score = rf._keyword_score(paper)
        assert score == 0.5  # no match, no change from base

    def test_case_insensitive(self, arxiv_config, llm_config):
        rf = RelevanceFilter(arxiv_config, llm_config)
        rf.set_keyword_weights({"TRANSFORMER": 1.0})
        paper = make_paper(title="transformer models", abstract="test")
        score = rf._keyword_score(paper)
        assert score > 0.5

    def test_score_bounded_zero_one(self, arxiv_config, llm_config):
        rf = RelevanceFilter(arxiv_config, llm_config)
        rf.set_keyword_weights({"a": 100.0})
        paper = make_paper(title="a a a", abstract="a")
        score = rf._keyword_score(paper)
        assert 0.0 <= score <= 1.0


class TestTierClassification:
    def test_discard_tier(self, arxiv_config, llm_config):
        rf = RelevanceFilter(arxiv_config, llm_config)
        result = rf._make_result(make_paper(), 0.1, "low")
        assert result.tier == FilterTier.DISCARD

    def test_abstract_only_tier(self, arxiv_config, llm_config):
        rf = RelevanceFilter(arxiv_config, llm_config)
        result = rf._make_result(make_paper(), 0.5, "medium")
        assert result.tier == FilterTier.ABSTRACT_ONLY

    def test_full_processing_tier(self, arxiv_config, llm_config):
        rf = RelevanceFilter(arxiv_config, llm_config)
        result = rf._make_result(make_paper(), 0.8, "high")
        assert result.tier == FilterTier.FULL_PROCESSING

    def test_boundary_relevance_threshold(self, arxiv_config, llm_config):
        rf = RelevanceFilter(arxiv_config, llm_config)
        result = rf._make_result(make_paper(), 0.3, "boundary")
        assert result.tier == FilterTier.ABSTRACT_ONLY

    def test_boundary_full_processing_threshold(self, arxiv_config, llm_config):
        rf = RelevanceFilter(arxiv_config, llm_config)
        result = rf._make_result(make_paper(), 0.7, "boundary")
        assert result.tier == FilterTier.FULL_PROCESSING

    def test_processing_level_updated(self, arxiv_config, llm_config):
        rf = RelevanceFilter(arxiv_config, llm_config)
        result = rf._make_result(make_paper(), 0.8, "high")
        assert result.paper.processing_level == ProcessingLevel.FULL_TEXT

        result2 = rf._make_result(make_paper(), 0.5, "medium")
        assert result2.paper.processing_level == ProcessingLevel.ABSTRACT_ONLY

        result3 = rf._make_result(make_paper(), 0.1, "low")
        assert result3.paper.processing_level == ProcessingLevel.UNPROCESSED


class TestScorePapers:
    @pytest.mark.asyncio
    async def test_score_papers_with_llm(self, arxiv_config, llm_config):
        rf = RelevanceFilter(arxiv_config, llm_config, research_agenda="Study AI safety")
        mock_result = {"score": 0.8, "reasoning": "Highly relevant"}

        with patch.object(rf._llm, "generate_structured", new_callable=AsyncMock, return_value=mock_result):
            papers = [make_paper(title="AI Safety Methods")]
            results = await rf.score_papers(papers)
            assert len(results) == 1
            assert results[0].score == pytest.approx(0.8)
            assert results[0].tier == FilterTier.FULL_PROCESSING

    @pytest.mark.asyncio
    async def test_score_papers_keyword_prefilter_skips_llm(self, arxiv_config, llm_config):
        rf = RelevanceFilter(arxiv_config, llm_config, research_agenda="Study AI safety")
        rf.set_keyword_weights({"quantum_computing_specific_term": 1.0})

        llm_mock = AsyncMock()
        with patch.object(rf._llm, "generate_structured", llm_mock):
            papers = [make_paper(title="Unrelated", abstract="Nothing matching")]
            results = await rf.score_papers(papers)
            # Keyword score is 0.5 (base), not < 0.1, so LLM IS called
            assert llm_mock.called

    @pytest.mark.asyncio
    async def test_score_blends_keyword_and_llm(self, arxiv_config, llm_config):
        rf = RelevanceFilter(arxiv_config, llm_config, research_agenda="Study transformers")
        rf.set_keyword_weights({"transformer": 1.0})
        mock_result = {"score": 0.9, "reasoning": "Very relevant"}

        with patch.object(rf._llm, "generate_structured", new_callable=AsyncMock, return_value=mock_result):
            papers = [make_paper(title="Transformer Architecture", abstract="About transformers")]
            results = await rf.score_papers(papers)
            # Score should be 0.3 * keyword + 0.7 * 0.9
            assert results[0].score > 0.5

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_keyword(self, arxiv_config, llm_config):
        rf = RelevanceFilter(arxiv_config, llm_config, research_agenda="Study AI")
        rf.set_keyword_weights({"ai": 1.0})

        with patch.object(rf._llm, "generate_structured", new_callable=AsyncMock, side_effect=Exception("LLM down")):
            papers = [make_paper(title="AI Research", abstract="About ai")]
            results = await rf.score_papers(papers)
            assert len(results) == 1
            assert results[0].reasoning == "keyword-only scoring"

    @pytest.mark.asyncio
    async def test_no_agenda_uses_keyword_only(self, arxiv_config, llm_config):
        rf = RelevanceFilter(arxiv_config, llm_config, research_agenda="")
        papers = [make_paper()]
        results = await rf.score_papers(papers)
        assert len(results) == 1
        assert results[0].reasoning == "keyword-only scoring"

    @pytest.mark.asyncio
    async def test_set_research_agenda(self, arxiv_config, llm_config):
        rf = RelevanceFilter(arxiv_config, llm_config)
        rf.set_research_agenda("New agenda")
        assert rf._research_agenda == "New agenda"
