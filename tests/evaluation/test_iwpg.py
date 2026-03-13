"""Tests for IWPGScorer."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from auto_researcher.config import IWPGWeights
from auto_researcher.evaluation.iwpg import IWPGScorer
from auto_researcher.models.research_thread import ResearchThread
from auto_researcher.models.reward import IWPGReward
from auto_researcher.utils.llm import LLMClient


def make_thread(**kwargs) -> ResearchThread:
    defaults = dict(
        id="thread-1",
        gap_id="gap-1",
        title="Test Thread",
        hypothesis_ids=["h1"],
        experiment_ids=["e1"],
        result_ids=["r1"],
        draft_sections={"abstract": "Abstract text", "methodology": "Method text"},
        compute_budget=100.0,
        compute_used=30.0,
        revision_count=2,
    )
    defaults.update(kwargs)
    return ResearchThread(**defaults)


@pytest.fixture
def mock_llm():
    llm = AsyncMock(spec=LLMClient)
    llm.generate_structured = AsyncMock(return_value={"utility": 0.7, "reasoning": "Useful"})
    return llm


@pytest.fixture
def weights():
    return IWPGWeights()


@pytest.fixture
def scorer(mock_llm, weights):
    return IWPGScorer(llm=mock_llm, weights=weights)


class TestComputeNovelty:
    def test_no_embedding_returns_default(self, scorer):
        assert scorer._compute_novelty(None) == 0.5

    def test_no_literature_returns_default(self, scorer):
        assert scorer._compute_novelty([1.0, 0.0]) == 0.5

    def test_high_similarity_low_novelty(self, scorer):
        scorer.load_literature_embeddings([[1.0, 0.0, 0.0]])
        novelty = scorer._compute_novelty([1.0, 0.0, 0.0])
        assert novelty == pytest.approx(0.0, abs=0.01)

    def test_low_similarity_high_novelty(self, scorer):
        scorer.load_literature_embeddings([[1.0, 0.0, 0.0]])
        novelty = scorer._compute_novelty([0.0, 1.0, 0.0])
        assert novelty == pytest.approx(1.0, abs=0.01)

    def test_knn_averaging(self, scorer):
        scorer.load_literature_embeddings([
            [1.0, 0.0], [0.0, 1.0], [0.7, 0.7],
        ])
        novelty = scorer._compute_novelty([1.0, 0.0])
        # Top-3 similarities averaged
        assert 0.0 < novelty < 1.0


class TestComputeSurprise:
    def test_no_results_default(self, scorer):
        thread = make_thread(result_ids=[], revision_count=0)
        assert scorer._compute_surprise(thread) == 0.5

    def test_more_results_higher_surprise(self, scorer):
        thread1 = make_thread(result_ids=["r1"], revision_count=0)
        thread2 = make_thread(result_ids=["r1", "r2", "r3"], revision_count=0)
        s1 = scorer._compute_surprise(thread1)
        s2 = scorer._compute_surprise(thread2)
        assert s2 > s1

    def test_revisions_increase_surprise(self, scorer):
        thread1 = make_thread(result_ids=["r1"], revision_count=0)
        thread2 = make_thread(result_ids=["r1"], revision_count=3)
        assert scorer._compute_surprise(thread2) > scorer._compute_surprise(thread1)


class TestComputeReproducibility:
    def test_full_thread(self, scorer):
        thread = make_thread()
        score = scorer._compute_reproducibility(thread)
        assert score == pytest.approx(1.0)  # 0.3 base + 0.3 methodology + 0.2 experiments + 0.2 results

    def test_minimal_thread(self, scorer):
        thread = make_thread(draft_sections={}, experiment_ids=[], result_ids=[])
        score = scorer._compute_reproducibility(thread)
        assert score == pytest.approx(0.3)  # base only


class TestComputeRedundancy:
    def test_no_embedding(self, scorer):
        assert scorer._compute_redundancy(None) == 0.0

    def test_no_literature(self, scorer):
        assert scorer._compute_redundancy([1.0, 0.0]) == 0.0

    def test_high_similarity_penalized(self, scorer):
        scorer.load_literature_embeddings([[1.0, 0.0]])
        penalty = scorer._compute_redundancy([1.0, 0.0])
        assert penalty > 0

    def test_low_similarity_no_penalty(self, scorer):
        scorer.load_literature_embeddings([[1.0, 0.0]])
        penalty = scorer._compute_redundancy([0.0, 1.0])
        assert penalty == 0.0


class TestComputeComplexityCost:
    def test_zero_budget(self, scorer):
        thread = make_thread(compute_budget=0, compute_used=10)
        assert scorer._compute_complexity_cost(thread) == 1.0

    def test_half_used(self, scorer):
        thread = make_thread(compute_budget=100, compute_used=50)
        assert scorer._compute_complexity_cost(thread) == pytest.approx(0.5)

    def test_over_budget(self, scorer):
        thread = make_thread(compute_budget=100, compute_used=200)
        assert scorer._compute_complexity_cost(thread) == 1.0


class TestComputeReward:
    @pytest.mark.asyncio
    async def test_compute_full_reward(self, scorer, mock_llm):
        thread = make_thread()
        reward = await scorer.compute_reward(thread)
        assert isinstance(reward, IWPGReward)
        assert 0.0 <= reward.utility <= 1.0
        mock_llm.generate_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_reward_appended_to_history(self, scorer, mock_llm):
        thread = make_thread()
        await scorer.compute_reward(thread)
        assert len(scorer._reward_history) == 1

    @pytest.mark.asyncio
    async def test_llm_failure_default_utility(self, scorer, mock_llm):
        mock_llm.generate_structured.side_effect = Exception("LLM down")
        thread = make_thread()
        reward = await scorer.compute_reward(thread)
        assert reward.utility == 0.5


class TestTotalReward:
    def test_total_with_defaults(self, scorer):
        reward = IWPGReward(
            novelty=1.0, surprise=1.0, utility=1.0,
            reproducibility=1.0, redundancy=0.0, complexity_cost=0.0,
        )
        total = scorer.total_reward(reward)
        expected = 0.25 + 0.15 + 0.20 + 0.15  # alpha+beta+gamma+delta
        assert total == pytest.approx(expected)

    def test_penalties_reduce_total(self, scorer):
        reward_no_penalty = IWPGReward(novelty=0.5, redundancy=0.0, complexity_cost=0.0)
        reward_with_penalty = IWPGReward(novelty=0.5, redundancy=1.0, complexity_cost=1.0)
        assert scorer.total_reward(reward_no_penalty) > scorer.total_reward(reward_with_penalty)


class TestSurrogateRewards:
    def test_returns_all_timescales(self, scorer):
        reward = IWPGReward(
            novelty=0.8, surprise=0.6, utility=0.7,
            reproducibility=0.5, redundancy=0.1, complexity_cost=0.2,
        )
        surrogates = scorer.surrogate_rewards(reward)
        assert "immediate" in surrogates
        assert "short_term" in surrogates
        assert "medium_term" in surrogates
        assert "long_term" in surrogates


class TestWeightUpdate:
    def test_update_weights(self, scorer):
        new_weights = IWPGWeights(novelty=0.30, surprise=0.10, utility=0.25)
        scorer.update_weights(new_weights)
        assert scorer._weights.novelty == 0.30
        assert len(scorer._weight_history) == 2

    def test_meta_rl_too_few_entries(self, scorer):
        result = scorer.meta_rl_weight_update([0.5, 0.6])
        assert result == scorer._weights  # no change

    @pytest.mark.asyncio
    async def test_meta_rl_adjusts_weights(self, scorer, mock_llm):
        # Build up reward history
        for i in range(5):
            thread = make_thread(id=f"t{i}")
            await scorer.compute_reward(thread)

        feedback = [0.9, 0.8, 0.7, 0.6, 0.5]
        new_weights = scorer.meta_rl_weight_update(feedback)
        assert isinstance(new_weights, IWPGWeights)
        # Weights should be bounded
        for field_name in ["novelty", "surprise", "utility", "reproducibility", "redundancy_penalty", "complexity_cost"]:
            w = getattr(new_weights, field_name)
            assert 0.05 <= w <= 0.40


class TestCosineSimlarity:
    def test_identical_vectors(self):
        assert IWPGScorer._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert IWPGScorer._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert IWPGScorer._cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert IWPGScorer._cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        assert IWPGScorer._cosine_similarity([1.0], [1.0, 0.0]) == 0.0

    def test_zero_vector(self):
        assert IWPGScorer._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
