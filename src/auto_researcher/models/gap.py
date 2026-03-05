"""Gap map data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class GapType(str, Enum):
    EMPIRICAL = "empirical"
    POPULATION = "population"
    METHODOLOGICAL = "methodological"
    CONTRADICTORY = "contradictory"
    NEGATIVE_RESULTS = "negative_results"
    COMBINATION = "combination"
    OPERATIONALIZATION = "operationalization"
    REPLICATION = "replication"
    SCALE = "scale"


class GapStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    FILLED = "filled"
    DEPRIORITIZED = "deprioritized"


class Gap(BaseModel):
    """A research gap identified in the intelligence space."""

    id: str = ""
    gap_type: GapType
    description: str
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    tractability: float = Field(ge=0.0, le=1.0, default=0.5)
    novelty: float = Field(ge=0.0, le=1.0, default=0.5)
    timeliness: float = Field(ge=0.0, le=1.0, default=0.5)
    foundational_impact: float = Field(ge=0.0, le=1.0, default=0.5)
    status: GapStatus = GapStatus.OPEN
    adjacent_concepts: list[str] = Field(default_factory=list)
    related_paper_ids: list[str] = Field(default_factory=list)
    blocking_gaps: list[str] = Field(default_factory=list)
    unlocked_by: list[str] = Field(default_factory=list)
    identified_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    def priority_score(self) -> float:
        """Compute gap priority using the gap prioritization algorithm."""
        urgency = self.timeliness
        return (self.importance * self.tractability * self.novelty) / (1.0 + urgency)


class GapNode(BaseModel):
    """A node in the gap map representing a concept or technique combination."""

    id: str
    node_type: str  # "concept", "technique", "dataset", "benchmark", "research_question"
    label: str
    coverage_score: float = Field(ge=0.0, le=1.0, default=0.0)
    adjacent_node_ids: list[str] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    paper_count: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class GapEdge(BaseModel):
    """An edge in the gap map."""

    source_id: str
    target_id: str
    edge_type: str  # "builds_on", "improves_upon", "evaluated_on", "requires", "enables",
    # "should_connect_but_doesnt", "tried_and_failed", "theoretically_related"
    weight: float = 1.0
    is_negative: bool = False  # True for missing/failed connections
    conditions: str = ""
    source_paper_ids: list[str] = Field(default_factory=list)
