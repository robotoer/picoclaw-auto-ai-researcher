"""Shared data models for E02: LLM Judge Reliability experiment."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class QualityTier(str, Enum):
    TRIVIAL = "trivial"
    INCREMENTAL = "incremental"
    AMBITIOUS = "ambitious"
    FLAWED = "flawed"
    MIXED = "mixed"


class Dimension(str, Enum):
    NOVELTY = "novelty"
    FEASIBILITY = "feasibility"
    IMPORTANCE = "importance"
    CLARITY = "clarity"
    SPECIFICITY = "specificity"


class Hypothesis(BaseModel):
    """A research hypothesis to be evaluated."""

    id: str = Field(description="Unique identifier, e.g. H01")
    text: str = Field(description="The hypothesis statement")
    rationale: str = Field(description="2-3 sentence rationale")
    quality_tier: QualityTier = Field(description="Intended quality level")
    source_model: str = Field(description="Model that generated this hypothesis")


class AbsoluteRating(BaseModel):
    """A single absolute rating (1-7 Likert) on one dimension."""

    hypothesis_id: str
    rater_id: str
    rater_type: Literal["llm_judge", "expert_proxy"]
    dimension: Dimension
    score: int = Field(ge=1, le=7)
    reasoning: str = Field(default="", description="Optional reasoning (for CoT)")


class PairwiseRating(BaseModel):
    """A pairwise comparison between two hypotheses on one dimension."""

    hypothesis_a_id: str
    hypothesis_b_id: str
    rater_id: str
    dimension: Dimension
    winner: Literal["a", "b", "tie"]
    reasoning: str = Field(default="")


class RatingSession(BaseModel):
    """Complete set of ratings from one rater in one format."""

    rater_id: str
    rater_type: Literal["llm_judge", "expert_proxy"]
    model_name: str
    format: Literal["absolute", "absolute_cot", "pairwise", "pairwise_cot"]
    ratings: list[AbsoluteRating] = Field(default_factory=list)
    pairwise_ratings: list[PairwiseRating] = Field(default_factory=list)
    presentation_order: list[str] = Field(
        default_factory=list, description="Order hypothesis IDs were presented"
    )
