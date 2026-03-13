"""Tests for SUNFIRE calibration functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from auto_researcher.config import SUNFIREWeights
from auto_researcher.evaluation.sunfire import (
    CalibrationData,
    CalibrationSample,
    SUNFIREEvaluator,
)
from auto_researcher.utils.llm import LLMClient


@pytest.fixture
def mock_llm():
    llm = AsyncMock(spec=LLMClient)
    llm.generate_structured = AsyncMock(return_value={"score": 0.7, "justification": "Good"})
    return llm


@pytest.fixture
def evaluator(mock_llm):
    return SUNFIREEvaluator(llm=mock_llm, weights=SUNFIREWeights())


class TestCalibrationData:
    def test_empty_calibration(self):
        cal = CalibrationData()
        assert len(cal.samples) == 0
        assert cal.last_calibrated is None

    def test_calibration_sample_creation(self):
        sample = CalibrationSample(
            sunfire_scores={"surprise": 0.8, "novelty": 0.6},
            review_score=0.7,
            review_confidence=0.9,
            review_aspects={"surprise": 0.75, "novelty": 0.65},
        )
        assert sample.review_score == 0.7
        assert sample.review_confidence == 0.9

    def test_calibration_data_serialization(self):
        cal = CalibrationData(
            samples=[
                CalibrationSample(
                    sunfire_scores={"surprise": 0.8},
                    review_score=0.7,
                )
            ]
        )
        json_str = cal.model_dump_json()
        restored = CalibrationData.model_validate_json(json_str)
        assert len(restored.samples) == 1
        assert restored.samples[0].review_score == 0.7


class TestCalibrateFromReviews:
    def test_single_review_no_adjustments(self, evaluator):
        reviews = [
            {
                "sunfire_scores": {"surprise": 0.8, "novelty": 0.6},
                "score": 0.7,
                "confidence": 0.9,
                "aspects": {"surprise": 0.75, "novelty": 0.65},
            }
        ]
        adjustments = evaluator.calibrate_from_reviews(reviews)
        # Need at least 2 samples
        assert adjustments == {}

    def test_multiple_reviews_produce_adjustments(self, evaluator):
        reviews = [
            {
                "sunfire_scores": {"surprise": 0.8, "novelty": 0.6},
                "score": 0.7,
                "confidence": 1.0,
                "aspects": {"surprise": 0.6, "novelty": 0.7},
            },
            {
                "sunfire_scores": {"surprise": 0.7, "novelty": 0.5},
                "score": 0.6,
                "confidence": 1.0,
                "aspects": {"surprise": 0.5, "novelty": 0.6},
            },
        ]
        adjustments = evaluator.calibrate_from_reviews(reviews)
        assert "surprise" in adjustments
        assert "novelty" in adjustments
        # SUNFIRE overestimates surprise: 0.8 vs 0.6 and 0.7 vs 0.5 -> negative adjustment
        assert adjustments["surprise"] < 0
        # SUNFIRE underestimates novelty: 0.6 vs 0.7 and 0.5 vs 0.6 -> positive adjustment
        assert adjustments["novelty"] > 0

    def test_calibration_updates_state(self, evaluator):
        reviews = [
            {
                "sunfire_scores": {"surprise": 0.8},
                "score": 0.7,
                "confidence": 1.0,
                "aspects": {"surprise": 0.6},
            },
            {
                "sunfire_scores": {"surprise": 0.7},
                "score": 0.6,
                "confidence": 1.0,
                "aspects": {"surprise": 0.5},
            },
        ]
        evaluator.calibrate_from_reviews(reviews)
        assert evaluator._calibration.last_calibrated is not None
        assert len(evaluator._calibration.samples) == 2

    def test_confidence_weighting(self, evaluator):
        reviews = [
            {
                "sunfire_scores": {"surprise": 0.8},
                "score": 0.7,
                "confidence": 1.0,
                "aspects": {"surprise": 0.6},
            },
            {
                "sunfire_scores": {"surprise": 0.3},
                "score": 0.5,
                "confidence": 0.1,  # very low confidence
                "aspects": {"surprise": 0.9},
            },
        ]
        adjustments = evaluator.calibrate_from_reviews(reviews)
        # The high-confidence review (overestimate) should dominate
        assert adjustments["surprise"] < 0


class TestCorrelation:
    def test_correlation_computed(self, evaluator):
        reviews = [
            {
                "sunfire_scores": {
                    "surprise": 0.8, "usefulness": 0.7, "novelty": 0.6,
                    "feasibility": 0.5, "impact_breadth": 0.5, "rigor": 0.7, "elegance": 0.6,
                },
                "score": 0.7,
                "confidence": 1.0,
                "aspects": {},
            },
            {
                "sunfire_scores": {
                    "surprise": 0.3, "usefulness": 0.2, "novelty": 0.1,
                    "feasibility": 0.4, "impact_breadth": 0.3, "rigor": 0.2, "elegance": 0.1,
                },
                "score": 0.2,
                "confidence": 1.0,
                "aspects": {},
            },
        ]
        evaluator.calibrate_from_reviews(reviews)
        assert len(evaluator._calibration.correlation_history) == 1
        corr = evaluator._calibration.correlation_history[0]
        assert "pearson_r" in corr
        # High SUNFIRE -> high review score, so positive correlation
        assert corr["pearson_r"] > 0

    def test_correlation_insufficient_data(self, evaluator):
        corr = evaluator._compute_correlation()
        assert corr["pearson_r"] == 0.0


class TestCalibrationPersistence:
    def test_save_and_load(self, evaluator):
        reviews = [
            {
                "sunfire_scores": {"surprise": 0.8},
                "score": 0.7,
                "confidence": 1.0,
                "aspects": {"surprise": 0.6},
            },
            {
                "sunfire_scores": {"surprise": 0.7},
                "score": 0.6,
                "confidence": 1.0,
                "aspects": {"surprise": 0.5},
            },
        ]
        evaluator.calibrate_from_reviews(reviews)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.json"
            evaluator.save_calibration(path)
            assert path.exists()

            new_evaluator = SUNFIREEvaluator(
                llm=AsyncMock(spec=LLMClient), weights=SUNFIREWeights()
            )
            new_evaluator.load_calibration(path)
            assert len(new_evaluator._calibration.samples) == 2
            assert new_evaluator._calibration.last_calibrated is not None

    def test_load_nonexistent_file(self, evaluator):
        evaluator.load_calibration(Path("/nonexistent/path.json"))
        assert len(evaluator._calibration.samples) == 0
