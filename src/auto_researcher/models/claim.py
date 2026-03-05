"""Claim data models for knowledge graph entries."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ClaimRelation(str, Enum):
    OUTPERFORMS = "outperforms"
    IS_VARIANT_OF = "is_variant_of"
    REFUTES = "refutes"
    SUPPORTS = "supports"
    REQUIRES = "requires"
    ENABLES = "enables"
    IS_APPLIED_TO = "is_applied_to"
    IMPROVES = "improves"
    EXTENDS = "extends"
    CONTRADICTS = "contradicts"
    IS_EQUIVALENT_TO = "is_equivalent_to"


class ClaimStatus(str, Enum):
    EXTRACTED = "extracted"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    HYPOTHESIS = "hypothesis"
    STALE = "stale"


class Claim(BaseModel):
    """A structured claim extracted from a paper or generated as a hypothesis.

    Format: (entity_1, relation, entity_2, conditions, confidence)
    """

    id: str = Field(default_factory=lambda: "")
    entity_1: str
    relation: ClaimRelation
    entity_2: str
    conditions: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    status: ClaimStatus = ClaimStatus.EXTRACTED
    source_paper_ids: list[str] = Field(default_factory=list)
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    last_verified: datetime | None = None
    half_life_days: int = 365
    contradicting_claim_ids: list[str] = Field(default_factory=list)
    supporting_claim_ids: list[str] = Field(default_factory=list)

    def decayed_confidence(self, now: datetime | None = None) -> float:
        """Compute confidence with temporal decay."""
        import math

        if now is None:
            now = datetime.utcnow()
        age_days = (now - self.extracted_at).days
        decay = math.exp(-0.693 * age_days / self.half_life_days)
        return self.confidence * decay
