"""Tests for MetaRLTrainer, ThompsonSampler, and UCBSelector."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from auto_researcher.config import CurriculumConfig
from auto_researcher.learning.curriculum_planner import (
    CurriculumPlanner,
    MetaRLEpisode,
    MetaRLTrainer,
    ThompsonSampler,
    TopicCandidate,
    UCBSelector,
)
from auto_researcher.utils.llm import LLMClient


@pytest.fixture
def config():
    return CurriculumConfig()


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.generate_structured = AsyncMock(return_value={
        "prerequisite_overlap": 0.6,
        "prerequisites": [],
        "difficulty": 0.5,
        "estimated_learning_effort": 0.5,
    })
    return llm


@pytest.fixture
def planner(mock_llm, config):
    return CurriculumPlanner(mock_llm, config)


@pytest.fixture
def trainer(planner):
    return MetaRLTrainer(planner)


# ── ThompsonSampler ──────────────────────────────────────────────


class TestThompsonSampler:
    def test_initial_posterior(self):
        ts = ThompsonSampler()
        alpha, beta = ts.get_posterior("unknown")
        assert alpha == 1.0
        assert beta == 1.0

    def test_update_increases_alpha(self):
        ts = ThompsonSampler()
        ts.update("t1", 1.0)
        alpha, beta = ts.get_posterior("t1")
        assert alpha == 2.0
        assert beta == 1.0

    def test_update_increases_beta_on_failure(self):
        ts = ThompsonSampler()
        ts.update("t1", 0.0)
        alpha, beta = ts.get_posterior("t1")
        assert alpha == 1.0
        assert beta == 2.0

    def test_sample_returns_valid_topic(self):
        ts = ThompsonSampler()
        ts.update("a", 0.9)
        ts.update("b", 0.1)
        topics = ["a", "b"]
        result = ts.sample(topics)
        assert result in topics

    def test_high_reward_topic_sampled_more(self):
        ts = ThompsonSampler()
        for _ in range(20):
            ts.update("good", 0.9)
            ts.update("bad", 0.1)

        counts = {"good": 0, "bad": 0}
        for _ in range(100):
            t = ts.sample(["good", "bad"])
            counts[t] += 1
        assert counts["good"] > counts["bad"]

    def test_partial_reward(self):
        ts = ThompsonSampler()
        ts.update("t1", 0.5)
        alpha, beta = ts.get_posterior("t1")
        assert alpha == pytest.approx(1.5)
        assert beta == pytest.approx(1.5)


# ── UCBSelector ──────────────────────────────────────────────────


class TestUCBSelector:
    def test_unexplored_topic_selected_first(self):
        ucb = UCBSelector()
        ucb.update("explored", 0.5)
        result = ucb.select(["explored", "new_topic"])
        assert result == "new_topic"

    def test_ucb_score_none_for_unexplored(self):
        ucb = UCBSelector()
        assert ucb.get_ucb_score("unknown") is None

    def test_ucb_score_computed(self):
        ucb = UCBSelector()
        ucb.update("t1", 0.7)
        ucb.update("t1", 0.8)
        score = ucb.get_ucb_score("t1")
        assert score is not None
        assert score > 0.7  # mean + exploration bonus

    def test_exploration_bonus_decreases_with_more_observations(self):
        ucb = UCBSelector()
        # Add two topics so total_selections grows faster than individual counts
        ucb.update("t1", 0.5)
        ucb.update("t2", 0.5)
        ucb.update("t1", 0.5)
        score_few = ucb.get_ucb_score("t1")
        for _ in range(20):
            ucb.update("t1", 0.5)
        score_many = ucb.get_ucb_score("t1")
        # With many more observations of t1, the exploration bonus should be smaller
        assert score_many < score_few

    def test_select_prefers_higher_mean(self):
        ucb = UCBSelector()
        for _ in range(20):
            ucb.update("good", 0.9)
            ucb.update("bad", 0.1)
        result = ucb.select(["good", "bad"])
        assert result == "good"


# ── MetaRLTrainer ────────────────────────────────────────────────


class TestMetaRLTrainer:
    def test_initial_state(self, trainer):
        assert trainer.episode_count == 0
        assert "field_momentum_weight" in trainer.policy_params

    @pytest.mark.asyncio
    async def test_run_single_episode(self, trainer):
        candidates = [
            TopicCandidate("A", prerequisite_overlap=0.6),
            TopicCandidate("B", prerequisite_overlap=0.7),
            TopicCandidate("C", prerequisite_overlap=0.5),
        ]
        learn_fn = AsyncMock(return_value=0.7)
        eval_fn = AsyncMock(return_value=0.8)

        episode = await trainer.run_episode(candidates, learn_fn, eval_fn, steps=2)

        assert isinstance(episode, MetaRLEpisode)
        assert len(episode.topics) == 2
        assert episode.eval_score == 0.8
        assert trainer.episode_count == 1

    @pytest.mark.asyncio
    async def test_policy_updated_after_two_episodes(self, trainer):
        candidates = [
            TopicCandidate("A", prerequisite_overlap=0.6),
            TopicCandidate("B", prerequisite_overlap=0.7),
        ]
        initial_params = dict(trainer.policy_params)

        learn_fn = AsyncMock(return_value=0.5)
        eval_fn = AsyncMock(return_value=0.5)
        await trainer.run_episode(candidates, learn_fn, eval_fn, steps=1)

        eval_fn_improved = AsyncMock(return_value=0.9)
        await trainer.run_episode(candidates, learn_fn, eval_fn_improved, steps=1)

        # Policy should have been updated after second episode
        assert trainer.episode_count == 2
        # With improvement, weights should increase
        for key in initial_params:
            assert trainer.policy_params[key] >= initial_params[key]

    @pytest.mark.asyncio
    async def test_training_history(self, trainer):
        candidates = [TopicCandidate("A", prerequisite_overlap=0.6)]
        learn_fn = AsyncMock(return_value=0.7)
        eval_fn = AsyncMock(return_value=0.8)

        await trainer.run_episode(candidates, learn_fn, eval_fn, steps=1)
        history = trainer.training_history
        assert len(history) == 1
        assert history[0]["episode"] == 1
        assert history[0]["eval_score"] == 0.8

    @pytest.mark.asyncio
    async def test_thompson_sampler_updated_during_episode(self, trainer):
        candidates = [TopicCandidate("A", prerequisite_overlap=0.6)]
        learn_fn = AsyncMock(return_value=0.9)
        eval_fn = AsyncMock(return_value=0.8)

        await trainer.run_episode(candidates, learn_fn, eval_fn, steps=1)

        alpha, beta = trainer.thompson_sampler.get_posterior("A")
        assert alpha > 1.0  # Updated with reward

    @pytest.mark.asyncio
    async def test_ucb_updated_during_episode(self, trainer):
        candidates = [TopicCandidate("A", prerequisite_overlap=0.6)]
        learn_fn = AsyncMock(return_value=0.9)
        eval_fn = AsyncMock(return_value=0.8)

        await trainer.run_episode(candidates, learn_fn, eval_fn, steps=1)

        score = trainer.ucb_selector.get_ucb_score("A")
        assert score is not None

    @pytest.mark.asyncio
    async def test_policy_params_clamped(self, trainer):
        """Policy params should stay in [0.01, 1.0] range."""
        candidates = [TopicCandidate("A", prerequisite_overlap=0.6)]
        learn_fn = AsyncMock(return_value=0.5)

        # Run many episodes with declining eval to push params down
        for score in [0.9, 0.1, 0.01, 0.001]:
            eval_fn = AsyncMock(return_value=score)
            await trainer.run_episode(candidates, learn_fn, eval_fn, steps=1)

        for v in trainer.policy_params.values():
            assert 0.01 <= v <= 1.0


class TestMetaRLEpisode:
    def test_episode_creation(self):
        ep = MetaRLEpisode(topics=["A", "B"], rewards=[0.7, 0.8], eval_score=0.75)
        assert ep.topics == ["A", "B"]
        assert ep.rewards == [0.7, 0.8]
        assert ep.eval_score == 0.75
