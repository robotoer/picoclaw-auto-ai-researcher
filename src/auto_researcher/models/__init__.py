"""Data models for the autonomous research system."""

from auto_researcher.models.paper import Paper, PaperMetadata, ProcessingLevel
from auto_researcher.models.claim import Claim, ClaimRelation, ClaimStatus
from auto_researcher.models.hypothesis import (
    Hypothesis,
    HypothesisStatus,
    FalsificationCriteria,
)
from auto_researcher.models.gap import Gap, GapType, GapNode, GapEdge, GapStatus
from auto_researcher.models.research_thread import (
    ResearchThread,
    ThreadStatus,
    ExperimentResult,
    ExperimentDesign,
)
from auto_researcher.models.reward import (
    SUNFIREScore,
    IWPGReward,
    PeerReviewResult,
    ReviewDecision,
)
from auto_researcher.models.memory import (
    EpisodicEntry,
    MemoryType,
    MetaMemoryEntry,
    ProceduralEntry,
)
from auto_researcher.models.agent import AgentMessage, AgentRole, AgentState

__all__ = [
    "Paper",
    "PaperMetadata",
    "ProcessingLevel",
    "Claim",
    "ClaimRelation",
    "ClaimStatus",
    "Hypothesis",
    "HypothesisStatus",
    "FalsificationCriteria",
    "Gap",
    "GapType",
    "GapNode",
    "GapEdge",
    "GapStatus",
    "ResearchThread",
    "ThreadStatus",
    "ExperimentResult",
    "ExperimentDesign",
    "SUNFIREScore",
    "IWPGReward",
    "PeerReviewResult",
    "ReviewDecision",
    "EpisodicEntry",
    "MemoryType",
    "MetaMemoryEntry",
    "ProceduralEntry",
    "AgentMessage",
    "AgentRole",
    "AgentState",
]
