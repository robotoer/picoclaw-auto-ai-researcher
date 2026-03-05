"""Reward model training and management for RLHF/RLAIF."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from auto_researcher.utils.llm import LLMClient
from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)

REWARD_SCORING_PROMPT = """\
You are a reward model for research quality. Score the following research output \
on a scale from 0.0 to 1.0 based on overall quality, novelty, and rigor.

Research output:
{output_text}

Previous scores for calibration (output_hash: score):
{calibration_examples}

Respond with JSON: {{"score": <float 0-1>, "reasoning": "<string>"}}
"""


class RewardSample:
    """A single (output, score) pair for reward model training."""

    def __init__(
        self,
        output_text: str,
        score: float,
        source: str = "community",
        timestamp: datetime | None = None,
    ) -> None:
        self.output_text = output_text
        self.score = score
        self.source = source  # "community", "citation", "review", "ai_feedback"
        self.timestamp = timestamp or datetime.utcnow()


class RewardModel:
    """Manages reward model training from community feedback.

    Uses LLM-as-judge with calibration from historical (output, score) pairs.
    Supports RLHF (human feedback) and RLAIF (AI feedback) signals.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._training_data: list[RewardSample] = []
        self._score_history: list[tuple[str, float, float]] = []  # (id, predicted, actual)
        self._last_retrain: datetime | None = None
        self._drift_threshold: float = 0.15

    def add_training_sample(self, sample: RewardSample) -> None:
        """Add a new (output, score) pair for training."""
        self._training_data.append(sample)
        logger.info(
            "reward_sample_added",
            source=sample.source,
            score=sample.score,
            total_samples=len(self._training_data),
        )

    def add_samples_batch(self, samples: list[RewardSample]) -> None:
        self._training_data.extend(samples)

    async def score(self, output_text: str) -> float:
        """Score a research output using the reward model."""
        calibration = self._get_calibration_examples()
        prompt = REWARD_SCORING_PROMPT.format(
            output_text=output_text[:4000],
            calibration_examples=calibration,
        )
        try:
            result = await self._llm.generate_structured(
                prompt=prompt,
                system="You are a calibrated reward model for research quality assessment.",
                temperature=0.2,
            )
            score = max(0.0, min(1.0, float(result.get("score", 0.5))))
            return score
        except Exception:
            logger.exception("reward_scoring_failed")
            return 0.5

    async def retrain(self) -> dict[str, Any]:
        """Retrain the reward model using accumulated samples.

        Since we use LLM-as-judge, retraining means updating calibration examples
        and checking for drift.
        """
        if len(self._training_data) < 5:
            return {"status": "insufficient_data", "sample_count": len(self._training_data)}

        # Check calibration quality on recent samples
        recent = self._training_data[-10:]
        errors: list[float] = []
        for sample in recent:
            predicted = await self.score(sample.output_text)
            error = abs(predicted - sample.score)
            errors.append(error)
            self._score_history.append(("retrain_check", predicted, sample.score))

        mean_error = sum(errors) / len(errors) if errors else 0.0
        self._last_retrain = datetime.utcnow()

        result = {
            "status": "retrained",
            "sample_count": len(self._training_data),
            "mean_calibration_error": mean_error,
            "drift_detected": mean_error > self._drift_threshold,
        }

        if mean_error > self._drift_threshold:
            logger.warning("reward_model_drift", mean_error=mean_error)

        logger.info("reward_model_retrained", **result)
        return result

    def detect_drift(self) -> bool:
        """Check if the reward model has drifted from ground truth."""
        if len(self._score_history) < 5:
            return False

        recent = self._score_history[-10:]
        errors = [abs(pred - actual) for _, pred, actual in recent]
        mean_error = sum(errors) / len(errors)
        return mean_error > self._drift_threshold

    def record_prediction(self, output_id: str, predicted: float, actual: float) -> None:
        """Record a prediction-vs-actual pair for drift detection."""
        self._score_history.append((output_id, predicted, actual))

    def _get_calibration_examples(self, n: int = 5) -> str:
        """Get recent training samples as calibration examples."""
        if not self._training_data:
            return "No calibration examples available."

        recent = self._training_data[-n:]
        lines = []
        for i, sample in enumerate(recent):
            text_preview = sample.output_text[:100].replace("\n", " ")
            lines.append(f"{i+1}. \"{text_preview}...\" -> score: {sample.score:.2f} (source: {sample.source})")
        return "\n".join(lines)

    @property
    def sample_count(self) -> int:
        return len(self._training_data)

    @property
    def needs_retrain(self) -> bool:
        """Whether the model should be retrained based on new data or drift."""
        if self._last_retrain is None and len(self._training_data) >= 5:
            return True
        if self._last_retrain is not None:
            samples_since = sum(
                1 for s in self._training_data if s.timestamp > self._last_retrain
            )
            if samples_since >= 10:
                return True
        return self.detect_drift()
