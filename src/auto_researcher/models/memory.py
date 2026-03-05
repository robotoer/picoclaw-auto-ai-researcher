"""Memory system data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    META = "meta"


class EpisodicEntry(BaseModel):
    """An entry in episodic memory — records a specific event or interaction."""

    id: str = ""
    memory_type: MemoryType = MemoryType.EPISODIC
    content: str
    context: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    source: str = ""  # agent or pipeline that created it
    embedding: list[float] | None = None
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_at: datetime | None = None
    access_count: int = 0


class MetaMemoryEntry(BaseModel):
    """Meta-memory: what the system knows about its own knowledge state."""

    topic: str
    competence_level: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    last_assessed: datetime = Field(default_factory=datetime.utcnow)
    knowledge_sources: list[str] = Field(default_factory=list)
    blind_spots: list[str] = Field(default_factory=list)
    related_topics: list[str] = Field(default_factory=list)


class ProceduralEntry(BaseModel):
    """Procedural memory: reusable research subroutines."""

    id: str = ""
    name: str
    description: str
    code: str | None = None
    tool_sequence: list[str] = Field(default_factory=list)
    success_rate: float = Field(ge=0.0, le=1.0, default=0.5)
    use_count: int = 0
    last_used: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
