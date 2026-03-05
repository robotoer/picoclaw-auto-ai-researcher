"""Reward and evaluation data models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class ReviewDecision(str, Enum):
    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"


class SUNFIREScore(BaseModel):
    """SUNFIRE framework score for research interestingness.

    Surprise, Usefulness, Novelty, Feasibility, Impact breadth, Rigor, Elegance.
    """

    surprise: float = Field(ge=0.0, le=1.0, default=0.0)
    usefulness: float = Field(ge=0.0, le=1.0, default=0.0)
    novelty: float = Field(ge=0.0, le=1.0, default=0.0)
    feasibility: float = Field(ge=0.0, le=1.0, default=0.0)
    impact_breadth: float = Field(ge=0.0, le=1.0, default=0.0)
    rigor: float = Field(ge=0.0, le=1.0, default=0.0)
    elegance: float = Field(ge=0.0, le=1.0, default=0.0)
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def composite(
        self,
        w_s: float = 0.15,
        w_u: float = 0.20,
        w_n: float = 0.20,
        w_f: float = 0.10,
        w_i: float = 0.15,
        w_r: float = 0.10,
        w_e: float = 0.10,
    ) -> float:
        return (
            w_s * self.surprise
            + w_u * self.usefulness
            + w_n * self.novelty
            + w_f * self.feasibility
            + w_i * self.impact_breadth
            + w_r * self.rigor
            + w_e * self.elegance
        )


class IWPGReward(BaseModel):
    """Interest-Weighted Policy Gradient reward signal."""

    novelty: float = Field(ge=0.0, le=1.0, default=0.0)
    surprise: float = Field(ge=0.0, le=1.0, default=0.0)
    utility: float = Field(ge=0.0, le=1.0, default=0.0)
    reproducibility: float = Field(ge=0.0, le=1.0, default=0.0)
    redundancy: float = Field(ge=0.0, le=1.0, default=0.0)
    complexity_cost: float = Field(ge=0.0, le=1.0, default=0.0)
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def total(
        self,
        alpha: float = 0.25,
        beta: float = 0.15,
        gamma: float = 0.20,
        delta: float = 0.15,
        epsilon: float = 0.15,
        zeta: float = 0.10,
    ) -> float:
        return (
            alpha * self.novelty
            + beta * self.surprise
            + gamma * self.utility
            + delta * self.reproducibility
            - epsilon * self.redundancy
            - zeta * self.complexity_cost
        )


class ReviewComment(BaseModel):
    reviewer_id: str
    aspect: str  # "methodology", "novelty", "clarity", "significance", "reproducibility"
    comment: str
    severity: str = "minor"  # "minor", "major", "critical"
    suggestion: str = ""


class PeerReviewResult(BaseModel):
    """Result of simulated peer review."""

    thread_id: str
    decision: ReviewDecision
    overall_score: float = Field(ge=0.0, le=1.0)
    reviews: list[ReviewComment] = Field(default_factory=list)
    meta_review: str = ""
    round_number: int = 1
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
