"""Data models for E03: Semantic Novelty Measurement."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CitationTier(str, Enum):
    """Citation impact tier based on 2-year citation percentile."""

    high_impact = "high_impact"  # top 10% by 2-year citation
    average = "average"  # 25th-75th percentile


class NoveltyLabel(str, Enum):
    """Binary novelty classification."""

    novel = "novel"
    incremental = "incremental"


class Paper(BaseModel):
    """An AI/ML paper with metadata and optional embedding."""

    paper_id: str  # Semantic Scholar paper ID
    title: str
    abstract: str
    year: int
    venue: str = ""
    citation_count_2yr: int = 0
    citation_tier: CitationTier = CitationTier.average
    authors: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)  # list of paper IDs
    fields_of_study: list[str] = Field(default_factory=list)
    source_url: str = ""
    embedding: list[float] = Field(default_factory=list)  # computed embedding vector


class NoveltyAnnotation(BaseModel):
    """A single rater's novelty annotation for a paper."""

    paper_id: str
    rater_id: str
    rater_type: str  # "expert_proxy" or "llm_judge"
    binary_label: NoveltyLabel
    likert_score: int = Field(ge=1, le=7)  # 1-7 novelty score
    reasoning: str = ""


class AnnotationSession(BaseModel):
    """A batch of annotations from one rater."""

    rater_id: str
    rater_type: str  # "expert_proxy"
    model_name: str
    annotations: list[NoveltyAnnotation] = Field(default_factory=list)


class NoveltyScore(BaseModel):
    """A single novelty metric score for a paper."""

    paper_id: str
    metric_name: str  # "embedding_distance", "atypical_references", "topic_distance", "llm_judgment"
    score: float
    details: dict[str, object] = Field(default_factory=dict)  # metric-specific details


class MetricResults(BaseModel):
    """Results for one novelty metric across all papers."""

    metric_name: str
    scores: list[NoveltyScore] = Field(default_factory=list)
    auc_roc: float = 0.0
    auc_ci_lower: float = 0.0
    auc_ci_upper: float = 0.0
    spearman_citation: float = 0.0
    spearman_citation_p: float = 1.0
    spearman_human: float = 0.0
    spearman_human_p: float = 1.0
    precision_at_10: float = 0.0
    precision_at_20: float = 0.0
