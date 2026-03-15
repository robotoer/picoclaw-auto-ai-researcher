"""Pydantic data models for E04: Knowledge Graph Consistency Under Continuous Ingestion.

Tests whether multi-layer hallucination prevention maintains KG consistency
when ingesting 500+ papers across 4 conditions:
  - no_filtering: baseline with no hallucination prevention
  - layer1: multi-extractor voting
  - layer1_2: add temporal consistency checking
  - layer1_2_3: add source verification
"""

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """A factual claim extracted from a paper."""

    claim_id: str
    paper_id: str
    text: str
    section: str = ""
    quote: str = ""
    confidence: float = 1.0
    extractor: str = ""
    verified: bool = False


class ExtractionResult(BaseModel):
    """Result from one extractor for one paper."""

    paper_id: str
    extractor_model: str
    claims: list[Claim] = Field(default_factory=list)
    timestamp: str = ""


class KGNode(BaseModel):
    """A node in the knowledge graph (represents a concept or entity)."""

    node_id: str
    label: str
    node_type: str


class KGEdge(BaseModel):
    """An edge in the knowledge graph (represents a relationship/claim)."""

    edge_id: str
    source_node: str
    target_node: str
    relation: str
    claim_id: str
    paper_id: str
    confidence: float = 1.0


class KnowledgeGraph(BaseModel):
    """A knowledge graph built from extracted claims."""

    nodes: dict[str, KGNode] = Field(default_factory=dict)
    edges: dict[str, KGEdge] = Field(default_factory=dict)
    claims: dict[str, Claim] = Field(default_factory=dict)


class ContradictionResult(BaseModel):
    """Result of checking a claim against ground truth."""

    claim_id: str
    ground_truth_claim_id: str
    entailment_score: float
    contradiction_score: float
    is_contradiction: bool
    reasoning: str = ""


class SpotCheckResult(BaseModel):
    """Manual verification of a claim against source paper."""

    claim_id: str
    paper_id: str
    category: str
    reasoning: str = ""


class ConditionResult(BaseModel):
    """Results for one experimental condition."""

    condition: str
    n_papers_ingested: int = 0
    n_claims_total: int = 0
    n_contradictions: int = 0
    contradiction_rate: float = 0.0
    n_hallucinations: int = 0
    hallucination_rate: float = 0.0
    n_duplicates: int = 0
    duplicate_rate: float = 0.0
    provenance_completeness: float = 0.0
    growth_checkpoints: list[dict] = Field(default_factory=list)


class Paper(BaseModel):
    """A paper from Semantic Scholar."""

    paper_id: str
    title: str
    abstract: str
    year: int
    venue: str = ""
    citation_count: int = 0
    authors: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    fields_of_study: list[str] = Field(default_factory=list)
    source_url: str = ""
    is_landmark: bool = False
