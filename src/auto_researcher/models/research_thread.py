"""Research thread data models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class ThreadStatus(str, Enum):
    INITIALIZED = "initialized"
    LITERATURE_REVIEW = "literature_review"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    CRITIQUE = "critique"
    EXPERIMENT_DESIGN = "experiment_design"
    EXPERIMENT_EXECUTION = "experiment_execution"
    RESULT_INTERPRETATION = "result_interpretation"
    SYNTHESIS = "synthesis"
    PEER_REVIEW = "peer_review"
    REVISION = "revision"
    PUBLISHED = "published"
    ABANDONED = "abandoned"


class ExperimentDesign(BaseModel):
    """A designed experiment to test a hypothesis."""

    id: str = ""
    hypothesis_id: str
    description: str
    methodology: str
    datasets: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=list)
    confounds: list[str] = Field(default_factory=list)
    estimated_compute_hours: float = 0.0
    expected_outcomes: dict[str, str] = Field(default_factory=dict)
    statistical_power: float | None = None
    code: str | None = None


class ExperimentResult(BaseModel):
    """Results from running an experiment."""

    id: str = ""
    experiment_id: str
    hypothesis_id: str
    outcome: str  # "confirmed", "refuted", "inconclusive"
    metrics: dict[str, float] = Field(default_factory=dict)
    statistical_significance: float | None = None
    interpretation: str = ""
    limitations: list[str] = Field(default_factory=list)
    new_questions: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    compute_hours_used: float = 0.0
    executed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ResearchThread(BaseModel):
    """A complete research thread from gap identification to publication."""

    id: str = ""
    gap_id: str
    title: str
    status: ThreadStatus = ThreadStatus.INITIALIZED
    hypothesis_ids: list[str] = Field(default_factory=list)
    experiment_ids: list[str] = Field(default_factory=list)
    result_ids: list[str] = Field(default_factory=list)
    literature_context: list[str] = Field(default_factory=list)
    draft_sections: dict[str, str] = Field(default_factory=dict)
    review_history: list[dict[str, str]] = Field(default_factory=list)
    revision_count: int = 0
    sunfire_score: float | None = None
    iwpg_reward: float | None = None
    compute_budget: float = 100.0
    compute_used: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    published_at: datetime | None = None
