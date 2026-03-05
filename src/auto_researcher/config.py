"""Configuration management for the autonomous research system."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7
    api_key: str = ""


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"


class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "research_papers"
    embedding_dim: int = 768


class ArxivConfig(BaseModel):
    categories: list[str] = Field(default_factory=lambda: ["cs.AI", "cs.LG", "cs.CL", "stat.ML"])
    max_results_per_query: int = 100
    relevance_threshold: float = 0.3
    full_processing_threshold: float = 0.7
    poll_interval_hours: int = 6


class GapMapConfig(BaseModel):
    frontier_threshold: float = 0.3
    min_importance: float = 0.2
    combination_search_depth: int = 3
    temporal_decay_default_halflife_days: int = 365


class SUNFIREWeights(BaseModel):
    surprise: float = 0.15
    usefulness: float = 0.20
    novelty: float = 0.20
    feasibility: float = 0.10
    impact_breadth: float = 0.15
    rigor: float = 0.10
    elegance: float = 0.10


class IWPGWeights(BaseModel):
    novelty: float = 0.25
    surprise: float = 0.15
    utility: float = 0.20
    reproducibility: float = 0.15
    redundancy_penalty: float = 0.15
    complexity_cost: float = 0.10


class CurriculumConfig(BaseModel):
    zpd_min_overlap: float = 0.4
    zpd_max_overlap: float = 0.8
    field_momentum_weight: float = 0.3
    gap_density_weight: float = 0.4
    strategic_value_weight: float = 0.3


class PeerReviewConfig(BaseModel):
    num_reviewers: int = 3
    max_revision_rounds: int = 3
    acceptance_threshold: float = 0.7


class OrchestratorConfig(BaseModel):
    outer_loop_interval_days: int = 7
    middle_loop_interval_hours: int = 24
    inner_loop_interval_minutes: int = 30
    max_concurrent_threads: int = 5
    compute_budget_per_thread: float = 100.0


class ConsolidationConfig(BaseModel):
    run_interval_hours: int = 24
    dedup_similarity_threshold: float = 0.92
    confidence_decay_rate: float = 0.01
    min_confidence_threshold: float = 0.1
    stale_hypothesis_days: int = 90


class ResearchConfig(BaseModel):
    """Top-level configuration for the entire research system."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    arxiv: ArxivConfig = Field(default_factory=ArxivConfig)
    gap_map: GapMapConfig = Field(default_factory=GapMapConfig)
    sunfire: SUNFIREWeights = Field(default_factory=SUNFIREWeights)
    iwpg: IWPGWeights = Field(default_factory=IWPGWeights)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    peer_review: PeerReviewConfig = Field(default_factory=PeerReviewConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")
