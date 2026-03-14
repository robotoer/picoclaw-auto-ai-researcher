"""SUNFIRE framework: Surprise, Usefulness, Novelty, Feasibility, Impact breadth, Rigor, Elegance."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from auto_researcher.config import SUNFIREWeights
from auto_researcher.models.reward import SUNFIREScore
from auto_researcher.models.research_thread import ResearchThread
from auto_researcher.utils.llm import LLMClient
from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)

DIMENSION_PROMPT_TEMPLATE = """\
You are an expert research evaluator. Score the following research output on the dimension "{dimension}" \
from 0.0 to 1.0 with a brief justification.

Dimension definition: {definition}

Research title: {title}
Research content:
{content}

Respond with JSON: {{"score": <float 0-1>, "justification": "<string>"}}
"""

DIMENSION_DEFINITIONS = {
    "surprise": "How much does this contradict or extend current expectations? High surprise = defies conventional wisdom or reveals unexpected relationships.",
    "usefulness": "How practically applicable is this? Can it be directly used by practitioners to improve systems, processes, or understanding?",
    "novelty": "How new is this contribution? Does it introduce genuinely new concepts, methods, or combinations, or is it incremental?",
    "feasibility": "How realistic is it to implement, reproduce, or build upon this work with reasonable resources?",
    "impact_breadth": "How many fields, subfields, or application areas could benefit from this work?",
    "rigor": "How methodologically sound is the work? Are experiments well-controlled, results statistically significant, and claims well-supported?",
    "elegance": "How simple and clean is the core insight relative to its explanatory/predictive power? Occam-style parsimony.",
}

# Anti-gaming thresholds
NOVELTY_HACKING_THRESHOLD = 0.95
SURPRISE_MANIPULATION_THRESHOLD = 0.95
COMMUNITY_GAMING_MIN_VARIANCE = 0.05

DIMENSION_NAMES = list(DIMENSION_DEFINITIONS.keys())


class CalibrationSample(BaseModel):
    """A single calibration sample from a human review."""

    sunfire_scores: dict[str, float] = Field(default_factory=dict)
    review_score: float = Field(ge=0.0, le=1.0)
    review_confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    review_aspects: dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CalibrationData(BaseModel):
    """Stores calibration samples and learned weight adjustments."""

    samples: list[CalibrationSample] = Field(default_factory=list)
    weight_adjustments: dict[str, float] = Field(default_factory=dict)
    correlation_history: list[dict[str, float | int | str]] = Field(default_factory=list)
    last_calibrated: datetime | None = None


class SUNFIREEvaluator:
    """Evaluates research outputs using the SUNFIRE framework."""

    def __init__(self, llm: LLMClient, weights: SUNFIREWeights) -> None:
        self._llm = llm
        self._weights = weights
        self._history: list[SUNFIREScore] = []
        self._calibration = CalibrationData()

    async def evaluate(self, thread: ResearchThread) -> SUNFIREScore:
        """Compute all 7 SUNFIRE dimensions for a research thread."""
        content = self._extract_content(thread)
        title = thread.title

        scores: dict[str, float] = {}
        for dimension, definition in DIMENSION_DEFINITIONS.items():
            llm_score = await self._score_dimension(dimension, definition, title, content)
            heuristic_score = self._heuristic_score(dimension, thread)
            # Blend LLM and heuristic scores (70/30)
            scores[dimension] = 0.7 * llm_score + 0.3 * heuristic_score

        raw_score = SUNFIREScore(
            surprise=scores["surprise"],
            usefulness=scores["usefulness"],
            novelty=scores["novelty"],
            feasibility=scores["feasibility"],
            impact_breadth=scores["impact_breadth"],
            rigor=scores["rigor"],
            elegance=scores["elegance"],
        )

        adjusted = self._anti_gaming_adjustment(raw_score)
        self._history.append(adjusted)
        logger.info(
            "sunfire_evaluated",
            thread_id=thread.id,
            composite=adjusted.composite(
                self._weights.surprise,
                self._weights.usefulness,
                self._weights.novelty,
                self._weights.feasibility,
                self._weights.impact_breadth,
                self._weights.rigor,
                self._weights.elegance,
            ),
        )
        return adjusted

    def composite_score(self, score: SUNFIREScore) -> float:
        return score.composite(
            self._weights.surprise,
            self._weights.usefulness,
            self._weights.novelty,
            self._weights.feasibility,
            self._weights.impact_breadth,
            self._weights.rigor,
            self._weights.elegance,
        )

    async def _score_dimension(
        self, dimension: str, definition: str, title: str, content: str
    ) -> float:
        prompt = DIMENSION_PROMPT_TEMPLATE.format(
            dimension=dimension, definition=definition, title=title, content=content[:4000]
        )
        try:
            result = await self._llm.generate_structured(
                prompt=prompt,
                system="You are a rigorous research evaluator. Return only valid JSON.",
                temperature=0.3,
            )
            score = float(result.get("score", 0.5))
            return max(0.0, min(1.0, score))
        except Exception:
            logger.exception("sunfire_dimension_score_failed", dimension=dimension)
            return 0.5

    def _heuristic_score(self, dimension: str, thread: ResearchThread) -> float:
        """Compute heuristic-based score as a complement to LLM scoring."""
        if dimension == "rigor":
            has_experiments = len(thread.experiment_ids) > 0
            has_results = len(thread.result_ids) > 0
            has_methodology = "methodology" in thread.draft_sections
            return sum([has_experiments, has_results, has_methodology]) / 3.0

        if dimension == "feasibility":
            budget_ratio = 1.0 - (thread.compute_used / max(thread.compute_budget, 1.0))
            return max(0.0, min(1.0, budget_ratio))

        if dimension == "novelty":
            return min(1.0, len(thread.hypothesis_ids) * 0.2)

        return 0.5

    def _anti_gaming_adjustment(self, score: SUNFIREScore) -> SUNFIREScore:
        """Detect and penalize potential gaming of the scoring system."""
        adjustments: dict[str, float] = {}

        # Novelty hacking: suspiciously high novelty with low rigor
        if score.novelty > NOVELTY_HACKING_THRESHOLD and score.rigor < 0.3:
            logger.warning("novelty_hacking_detected", novelty=score.novelty, rigor=score.rigor)
            adjustments["novelty"] = score.novelty * 0.7

        # Surprise manipulation: extreme surprise without substance
        if score.surprise > SURPRISE_MANIPULATION_THRESHOLD and score.usefulness < 0.2:
            logger.warning("surprise_manipulation_detected", surprise=score.surprise, usefulness=score.usefulness)
            adjustments["surprise"] = score.surprise * 0.7

        # Community gaming: all dimensions suspiciously similar (low variance)
        values = [score.surprise, score.usefulness, score.novelty, score.feasibility,
                  score.impact_breadth, score.rigor, score.elegance]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        if variance < COMMUNITY_GAMING_MIN_VARIANCE and mean > 0.8:
            logger.warning("community_gaming_detected", variance=variance, mean=mean)
            penalty = 0.85
            adjustments = {
                "surprise": score.surprise * penalty,
                "usefulness": score.usefulness * penalty,
                "novelty": score.novelty * penalty,
                "feasibility": score.feasibility * penalty,
                "impact_breadth": score.impact_breadth * penalty,
                "rigor": score.rigor * penalty,
                "elegance": score.elegance * penalty,
            }

        if not adjustments:
            return score

        return SUNFIREScore(
            surprise=adjustments.get("surprise", score.surprise),
            usefulness=adjustments.get("usefulness", score.usefulness),
            novelty=adjustments.get("novelty", score.novelty),
            feasibility=adjustments.get("feasibility", score.feasibility),
            impact_breadth=adjustments.get("impact_breadth", score.impact_breadth),
            rigor=adjustments.get("rigor", score.rigor),
            elegance=adjustments.get("elegance", score.elegance),
        )

    def calibrate_from_reviews(self, reviews: list[dict[str, Any]]) -> dict[str, float]:
        """Adjust heuristic scoring weights to match human reviewer assessments.

        Each review dict should have:
        - "score": float 0-1 (overall review score)
        - "confidence": float 0-1 (reviewer confidence)
        - "aspects": dict[str, float] mapping dimension names to reviewer scores
        - "sunfire_scores": dict[str, float] (the SUNFIRE scores that were given)
        """
        for review in reviews:
            sample = CalibrationSample(
                sunfire_scores=review.get("sunfire_scores", {}),
                review_score=review.get("score", 0.5),
                review_confidence=review.get("confidence", 1.0),
                review_aspects=review.get("aspects", {}),
            )
            self._calibration.samples.append(sample)

        if len(self._calibration.samples) < 2:
            return {}

        adjustments = self._compute_weight_adjustments()
        self._calibration.weight_adjustments = adjustments
        self._calibration.last_calibrated = datetime.now(UTC)

        correlation = self._compute_correlation()
        self._calibration.correlation_history.append({
            "timestamp_iso": datetime.now(UTC).isoformat(),
            **correlation,
        })

        logger.info(
            "sunfire_calibrated",
            num_samples=len(self._calibration.samples),
            adjustments=adjustments,
            correlation=correlation,
        )
        return adjustments

    def _compute_weight_adjustments(self) -> dict[str, float]:
        """Compute per-dimension weight adjustments from calibration data."""
        adjustments: dict[str, float] = {}
        for dim in DIMENSION_NAMES:
            pairs = []
            for s in self._calibration.samples:
                if dim in s.sunfire_scores and dim in s.review_aspects:
                    pairs.append((s.sunfire_scores[dim], s.review_aspects[dim], s.review_confidence))
            if not pairs:
                continue
            total_weight = sum(c for _, _, c in pairs)
            if total_weight == 0:
                continue
            weighted_error = sum(c * (review - sunfire) for sunfire, review, c in pairs)
            adjustments[dim] = weighted_error / total_weight
        return adjustments

    def _compute_correlation(self) -> dict[str, float]:
        """Compute Pearson correlation between SUNFIRE composite and review scores."""
        samples = self._calibration.samples
        if len(samples) < 2:
            return {"pearson_r": 0.0, "n": len(samples)}

        composites = []
        review_scores = []
        for s in samples:
            if s.sunfire_scores:
                score = SUNFIREScore.model_validate(
                    {d: s.sunfire_scores.get(d, 0.0) for d in DIMENSION_NAMES}
                )
                composites.append(self.composite_score(score))
                review_scores.append(s.review_score)

        if len(composites) < 2:
            return {"pearson_r": 0.0, "n": len(composites)}

        n = len(composites)
        mean_c = sum(composites) / n
        mean_r = sum(review_scores) / n
        cov = sum((c - mean_c) * (r - mean_r) for c, r in zip(composites, review_scores)) / n
        std_c = math.sqrt(sum((c - mean_c) ** 2 for c in composites) / n)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in review_scores) / n)
        if std_c == 0 or std_r == 0:
            return {"pearson_r": 0.0, "n": n}
        return {"pearson_r": cov / (std_c * std_r), "n": n}

    def save_calibration(self, path: Path) -> None:
        """Save calibration state to a JSON file."""
        path.write_text(self._calibration.model_dump_json(indent=2))

    def load_calibration(self, path: Path) -> None:
        """Load calibration state from a JSON file."""
        if path.exists():
            self._calibration = CalibrationData.model_validate_json(path.read_text())

    @staticmethod
    def _extract_content(thread: ResearchThread) -> str:
        """Extract readable content from a research thread for evaluation."""
        parts = [f"Title: {thread.title}"]
        for section_name in ["abstract", "introduction", "methodology", "results", "conclusion"]:
            if section_name in thread.draft_sections:
                parts.append(f"\n## {section_name.title()}\n{thread.draft_sections[section_name]}")
        return "\n".join(parts)
