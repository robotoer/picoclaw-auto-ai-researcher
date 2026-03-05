"""Research impact prediction: citations, community uptake, cross-field breadth."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from auto_researcher.models.research_thread import ResearchThread
from auto_researcher.models.reward import SUNFIREScore
from auto_researcher.utils.llm import LLMClient
from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)

IMPACT_PROMPT = """\
Estimate the research impact of the following work. Consider citation potential, \
community adoption likelihood, and cross-field applicability.

Title: {title}
Abstract: {abstract}
Fields touched: {fields}

Respond with JSON:
{{
    "citation_estimate_1yr": <int>,
    "citation_estimate_5yr": <int>,
    "community_uptake": <float 0-1>,
    "cross_field_breadth": <float 0-1>,
    "applicable_fields": ["<string>"],
    "reasoning": "<string>"
}}
"""


class ImpactPrediction(BaseModel):
    thread_id: str
    citation_estimate_1yr: int = 0
    citation_estimate_5yr: int = 0
    community_uptake: float = 0.0
    cross_field_breadth: float = 0.0
    applicable_fields: list[str] = Field(default_factory=list)
    predicted_at: datetime = Field(default_factory=datetime.utcnow)
    actual_citations: int | None = None
    accuracy_score: float | None = None


class ImpactPredictor:
    """Predicts research impact and tracks prediction accuracy over time."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._predictions: dict[str, ImpactPrediction] = {}
        self._accuracy_history: list[float] = []

    async def predict(
        self,
        thread: ResearchThread,
        sunfire_score: SUNFIREScore | None = None,
    ) -> ImpactPrediction:
        """Predict the impact of a research thread."""
        abstract = thread.draft_sections.get("abstract", thread.title)
        fields = ", ".join(thread.literature_context[:5]) if thread.literature_context else "AI/ML"

        llm_prediction = await self._llm_predict(thread.title, abstract, fields)

        # Blend LLM prediction with heuristics from SUNFIRE if available
        if sunfire_score is not None:
            llm_prediction["community_uptake"] = (
                0.6 * llm_prediction.get("community_uptake", 0.5)
                + 0.4 * sunfire_score.usefulness
            )
            llm_prediction["cross_field_breadth"] = (
                0.6 * llm_prediction.get("cross_field_breadth", 0.5)
                + 0.4 * sunfire_score.impact_breadth
            )

        prediction = ImpactPrediction(
            thread_id=thread.id,
            citation_estimate_1yr=llm_prediction.get("citation_estimate_1yr", 5),
            citation_estimate_5yr=llm_prediction.get("citation_estimate_5yr", 20),
            community_uptake=llm_prediction.get("community_uptake", 0.5),
            cross_field_breadth=llm_prediction.get("cross_field_breadth", 0.3),
            applicable_fields=llm_prediction.get("applicable_fields", []),
        )

        self._predictions[thread.id] = prediction
        logger.info(
            "impact_predicted",
            thread_id=thread.id,
            citations_1yr=prediction.citation_estimate_1yr,
            uptake=prediction.community_uptake,
        )
        return prediction

    async def _llm_predict(self, title: str, abstract: str, fields: str) -> dict[str, Any]:
        prompt = IMPACT_PROMPT.format(title=title, abstract=abstract[:3000], fields=fields)
        try:
            return await self._llm.generate_structured(
                prompt=prompt,
                system="You are a research impact analyst with deep expertise in AI/ML research trends.",
                temperature=0.4,
            )
        except Exception:
            logger.exception("impact_prediction_failed")
            return {}

    def record_actual_citations(self, thread_id: str, actual_citations: int) -> float | None:
        """Record actual citation count and compute prediction accuracy."""
        prediction = self._predictions.get(thread_id)
        if prediction is None:
            return None

        prediction.actual_citations = actual_citations
        if prediction.citation_estimate_1yr > 0:
            ratio = actual_citations / prediction.citation_estimate_1yr
            accuracy = max(0.0, 1.0 - abs(1.0 - ratio))
        else:
            accuracy = 1.0 if actual_citations == 0 else 0.0

        prediction.accuracy_score = accuracy
        self._accuracy_history.append(accuracy)
        logger.info(
            "impact_accuracy_recorded",
            thread_id=thread_id,
            predicted=prediction.citation_estimate_1yr,
            actual=actual_citations,
            accuracy=accuracy,
        )
        return accuracy

    def average_accuracy(self) -> float:
        """Average prediction accuracy across all tracked predictions."""
        if not self._accuracy_history:
            return 0.0
        return sum(self._accuracy_history) / len(self._accuracy_history)

    def get_prediction(self, thread_id: str) -> ImpactPrediction | None:
        return self._predictions.get(thread_id)
