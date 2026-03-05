"""Paper data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ProcessingLevel(str, Enum):
    UNPROCESSED = "unprocessed"
    ABSTRACT_ONLY = "abstract_only"
    FULL_TEXT = "full_text"


class PaperMetadata(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: datetime
    updated: datetime | None = None
    doi: str | None = None
    journal_ref: str | None = None
    pdf_url: str | None = None
    source_url: str | None = None
    comment: str | None = None


class Paper(BaseModel):
    """A research paper with extracted content and metadata."""

    metadata: PaperMetadata
    processing_level: ProcessingLevel = ProcessingLevel.UNPROCESSED
    relevance_score: float = 0.0
    full_text: str | None = None
    sections: dict[str, str] = Field(default_factory=dict)
    tables: list[dict[str, str]] = Field(default_factory=list)
    figures: list[dict[str, str]] = Field(default_factory=list)
    equations: list[str] = Field(default_factory=list)
    code_snippets: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    citation_count: int | None = None
    embedding: list[float] | None = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def arxiv_id(self) -> str:
        return self.metadata.arxiv_id

    @property
    def title(self) -> str:
        return self.metadata.title
