"""Ingestion pipeline: ArXiv monitoring, extraction, and knowledge integration."""

from auto_researcher.ingestion.arxiv_monitor import ArxivMonitor
from auto_researcher.ingestion.claim_extractor import ClaimExtractor
from auto_researcher.ingestion.kg_updater import ConflictReport, KGUpdater
from auto_researcher.ingestion.pdf_extractor import ExtractedContent, PDFExtractor
from auto_researcher.ingestion.pipeline import IngestionPipeline, PipelineStats
from auto_researcher.ingestion.relevance_filter import (
    FilterTier,
    RelevanceFilter,
    RelevanceResult,
)
from auto_researcher.ingestion.trend_detector import (
    TopicTrend,
    TrendDetector,
    TrendReport,
)

__all__ = [
    "ArxivMonitor",
    "ClaimExtractor",
    "ConflictReport",
    "ExtractedContent",
    "FilterTier",
    "IngestionPipeline",
    "KGUpdater",
    "PDFExtractor",
    "PipelineStats",
    "RelevanceFilter",
    "RelevanceResult",
    "TopicTrend",
    "TrendDetector",
    "TrendReport",
]
