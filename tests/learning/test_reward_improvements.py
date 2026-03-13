"""Tests for DriftDetector and EnsembleRewardModel."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from auto_researcher.learning.reward_model import (
    DriftDetector,
    EnsembleRewardModel,
    RewardModel,
    RewardSample,
)
from auto_researcher.utils.llm import LLMClient


def make_mock_reward_model(score: float = 0.5) -> RewardModel:
    llm = AsyncMock(spec=LLMClient)
    llm.generate_structured = AsyncMock(return_value={"score": score, "reasoning": "ok"})
    return RewardModel(llm)


# ── DriftDetector ────────────────────────────────────────────────


class TestDriftDetector:
    def test_no_drift_with_insufficient_data(self):
        dd = DriftDetector(window_size=10)
        for i in range(5):
            dd.add_score(0.5)
        result = dd.check_drift()
        assert result["drifted"] is False
        assert result["n"] == 5

    def test_no_drift_with_stable_distribution(self):
        dd = DriftDetector(window_size=10)
        for _ in range(20):
            dd.add_score(0.5)
        result = dd.check_drift()
        assert result["drifted"] is False
        assert result["kl_divergence"] == pytest.approx(0.0, abs=0.01)

    def test_drift_detected_with_distribution_shift(self):
        dd = DriftDetector(window_size=10, kl_threshold=0.1)
        # Old distribution: low scores
        for _ in range(10):
            dd.add_score(0.1)
        # New distribution: high scores
        for _ in range(10):
            dd.add_score(0.9)
        result = dd.check_drift()
        assert result["drifted"] is True
        assert result["kl_divergence"] > 0.1

    def test_kl_divergence_symmetric_stable(self):
        dd = DriftDetector(window_size=10)
        # Same distribution both halves
        for _ in range(20):
            dd.add_score(0.5)
        result = dd.check_drift()
        assert result["kl_divergence"] == pytest.approx(0.0, abs=0.01)

    def test_window_size_respected(self):
        dd = DriftDetector(window_size=5)
        # Add more than 2*window scores; deque should cap at 10
        for i in range(20):
            dd.add_score(0.5)
        assert len(dd._scores) == 10


# ── EnsembleRewardModel ──────────────────────────────────────────


class TestEnsembleRewardModel:
    def test_requires_at_least_one_model(self):
        with pytest.raises(ValueError, match="At least one"):
            EnsembleRewardModel([])

    def test_model_count(self):
        models = [make_mock_reward_model(0.5) for _ in range(3)]
        ensemble = EnsembleRewardModel(models)
        assert ensemble.model_count == 3

    @pytest.mark.asyncio
    async def test_score_returns_mean_and_std(self):
        models = [
            make_mock_reward_model(0.4),
            make_mock_reward_model(0.6),
            make_mock_reward_model(0.5),
        ]
        ensemble = EnsembleRewardModel(models)
        result = await ensemble.score("test output")
        assert "mean" in result
        assert "std" in result
        assert "uncertainty" in result
        assert result["mean"] == pytest.approx(0.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_score_uncertainty_with_agreement(self):
        models = [make_mock_reward_model(0.5) for _ in range(3)]
        ensemble = EnsembleRewardModel(models)
        result = await ensemble.score("test output")
        assert result["std"] == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_score_uncertainty_with_disagreement(self):
        models = [
            make_mock_reward_model(0.2),
            make_mock_reward_model(0.8),
        ]
        ensemble = EnsembleRewardModel(models)
        result = await ensemble.score("test output")
        assert result["std"] > 0.2

    def test_add_training_sample_propagates(self):
        models = [make_mock_reward_model(0.5) for _ in range(3)]
        ensemble = EnsembleRewardModel(models)
        sample = RewardSample(output_text="test", score=0.7)
        ensemble.add_training_sample(sample)
        for model in models:
            assert model.sample_count == 1

    @pytest.mark.asyncio
    async def test_validate_against_holdout(self):
        models = [make_mock_reward_model(0.5) for _ in range(2)]
        ensemble = EnsembleRewardModel(models)
        holdout = [
            RewardSample(output_text="test1", score=0.7),
            RewardSample(output_text="test2", score=0.3),
        ]
        result = await ensemble.validate_against_holdout(holdout)
        assert "mean_error" in result
        assert result["n"] == 2
        # Models return 0.5, actual are 0.7 and 0.3
        assert result["mean_error"] == pytest.approx(0.2, abs=0.01)

    @pytest.mark.asyncio
    async def test_validate_empty_holdout(self):
        models = [make_mock_reward_model(0.5)]
        ensemble = EnsembleRewardModel(models)
        result = await ensemble.validate_against_holdout([])
        assert result["n"] == 0
        assert result["mean_error"] == 0.0
