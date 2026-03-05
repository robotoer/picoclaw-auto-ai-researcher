"""SUNFIRE framework: Surprise, Usefulness, Novelty, Feasibility, Impact breadth, Rigor, Elegance."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

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


class SUNFIREEvaluator:
    """Evaluates research outputs using the SUNFIRE framework."""

    def __init__(self, llm: LLMClient, weights: SUNFIREWeights) -> None:
        self._llm = llm
        self._weights = weights
        self._history: list[SUNFIREScore] = []

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

    @staticmethod
    def _extract_content(thread: ResearchThread) -> str:
        """Extract readable content from a research thread for evaluation."""
        parts = [f"Title: {thread.title}"]
        for section_name in ["abstract", "introduction", "methodology", "results", "conclusion"]:
            if section_name in thread.draft_sections:
                parts.append(f"\n## {section_name.title()}\n{thread.draft_sections[section_name]}")
        return "\n".join(parts)
