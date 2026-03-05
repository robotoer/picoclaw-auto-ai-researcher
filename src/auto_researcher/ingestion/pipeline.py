"""Full ingestion pipeline orchestration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

from auto_researcher.config import ResearchConfig
from auto_researcher.models import Claim, Paper, ProcessingLevel
from auto_researcher.utils.logging import get_logger

from auto_researcher.ingestion.arxiv_monitor import ArxivMonitor
from auto_researcher.ingestion.claim_extractor import ClaimExtractor
from auto_researcher.ingestion.kg_updater import ConflictReport, KGUpdater
from auto_researcher.ingestion.pdf_extractor import PDFExtractor
from auto_researcher.ingestion.relevance_filter import FilterTier, RelevanceFilter
from auto_researcher.ingestion.trend_detector import TrendDetector, TrendReport

logger = get_logger(__name__)

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BATCH_SIZE = 10


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""

    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    papers_fetched: int = 0
    papers_discarded: int = 0
    papers_abstract_only: int = 0
    papers_full_processed: int = 0
    claims_extracted: int = 0
    conflicts_detected: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float | None:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()


class IngestionPipeline:
    """Orchestrates the full ingestion flow.

    Flow: fetch -> filter -> extract PDF -> extract claims -> KG update -> trend analysis -> memory write
    """

    def __init__(
        self,
        config: ResearchConfig,
        knowledge_graph=None,
        vector_store=None,
        gap_map=None,
        episodic_memory=None,
        research_agenda: str = "",
    ) -> None:
        self._config = config
        self._monitor = ArxivMonitor(config.arxiv)
        self._filter = RelevanceFilter(config.arxiv, config.llm, research_agenda)
        self._pdf_extractor = PDFExtractor(config.data_dir / "pdfs")
        self._claim_extractor = ClaimExtractor(config.llm)
        self._trend_detector = TrendDetector()
        self._kg_updater = KGUpdater(knowledge_graph) if knowledge_graph else None
        self._vector_store = vector_store
        self._gap_map = gap_map
        self._episodic_memory = episodic_memory
        self._batch_size = _DEFAULT_BATCH_SIZE
        self._max_retries = _DEFAULT_MAX_RETRIES

    async def run(self) -> PipelineStats:
        """Execute a single pass of the ingestion pipeline."""
        stats = PipelineStats()

        # Step 1: Fetch papers from ArXiv
        papers = await self._fetch_with_retry()
        stats.papers_fetched = len(papers)
        if not papers:
            logger.info("no_new_papers")
            stats.completed_at = datetime.utcnow()
            return stats

        # Step 2: Relevance filtering
        relevance_results = await self._filter.score_papers(papers)
        full_process = []
        abstract_only = []
        for result in relevance_results:
            if result.tier == FilterTier.FULL_PROCESSING:
                full_process.append(result.paper)
            elif result.tier == FilterTier.ABSTRACT_ONLY:
                abstract_only.append(result.paper)
            else:
                stats.papers_discarded += 1

        stats.papers_abstract_only = len(abstract_only)
        stats.papers_full_processed = len(full_process)

        # Step 3: PDF extraction for full-process papers (in batches)
        extracted_papers: list[Paper] = []
        for batch in self._batched(full_process, self._batch_size):
            batch_results = await asyncio.gather(
                *[self._extract_pdf_with_retry(p) for p in batch],
                return_exceptions=True,
            )
            for result in batch_results:
                if isinstance(result, Exception):
                    stats.errors.append(f"pdf_extraction: {result}")
                else:
                    extracted_papers.append(result)

        # Combine: extracted full papers + abstract-only papers
        all_processed = extracted_papers + abstract_only

        # Step 4: Claim extraction
        all_claims: list[Claim] = []
        all_conflicts: list[ConflictReport] = []
        for paper in all_processed:
            try:
                claims = await self._claim_extractor.extract_claims(paper)
                all_claims.extend(claims)

                # Step 5: KG update
                if self._kg_updater and claims:
                    added, conflicts = await self._kg_updater.update_from_paper(
                        paper, claims
                    )
                    all_conflicts.extend(conflicts)
            except Exception as e:
                stats.errors.append(f"claim_extraction({paper.arxiv_id}): {e}")
                logger.exception("claim_processing_failed", arxiv_id=paper.arxiv_id)

        stats.claims_extracted = len(all_claims)
        stats.conflicts_detected = len(all_conflicts)

        # Step 6: Vector store indexing
        if self._vector_store:
            for paper in all_processed:
                try:
                    await self._vector_store.index_paper(paper)
                except Exception as e:
                    stats.errors.append(f"vector_indexing({paper.arxiv_id}): {e}")

        # Step 7: Trend analysis
        try:
            trend_report = await self._trend_detector.analyze(all_processed)
            if self._gap_map and trend_report.trending_topics:
                await self._update_gap_map(trend_report)
        except Exception as e:
            stats.errors.append(f"trend_analysis: {e}")
            logger.exception("trend_analysis_failed")

        # Step 8: Memory write
        if self._episodic_memory:
            await self._write_memory(stats, all_processed, all_conflicts)

        # Mark papers as processed
        self._monitor.mark_processed([p.arxiv_id for p in all_processed])

        stats.completed_at = datetime.utcnow()
        logger.info(
            "pipeline_run_complete",
            fetched=stats.papers_fetched,
            processed=stats.papers_full_processed,
            claims=stats.claims_extracted,
            conflicts=stats.conflicts_detected,
            errors=len(stats.errors),
            duration_s=stats.duration_seconds,
        )
        return stats

    async def _fetch_with_retry(self) -> list[Paper]:
        """Fetch papers with retry logic."""
        for attempt in range(self._max_retries):
            try:
                return await self._monitor.fetch_recent_papers()
            except Exception:
                logger.warning("fetch_retry", attempt=attempt + 1)
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(2**attempt)
        logger.error("fetch_exhausted_retries")
        return []

    async def _extract_pdf_with_retry(self, paper: Paper) -> Paper:
        """Extract PDF content with retry logic."""
        for attempt in range(self._max_retries):
            try:
                return await self._pdf_extractor.extract_from_paper(paper)
            except Exception:
                logger.warning(
                    "pdf_retry", arxiv_id=paper.arxiv_id, attempt=attempt + 1
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(2**attempt)
        return paper  # Return unextracted paper on failure

    async def _update_gap_map(self, report: TrendReport) -> None:
        """Update the gap map based on trend analysis."""
        for topic in report.trending_topics:
            try:
                await self._gap_map.update_topic_coverage(
                    keywords=topic.keywords,
                    paper_count=topic.paper_count,
                    growth_rate=topic.growth_rate,
                )
            except Exception:
                logger.exception("gap_map_update_failed", topic_id=topic.topic_id)

        for combo in report.emerging_combinations:
            try:
                await self._gap_map.flag_emerging_combination(
                    categories=combo["categories"],
                    example_titles=combo.get("example_titles", []),
                )
            except Exception:
                logger.exception("gap_map_combo_failed")

    async def _write_memory(
        self,
        stats: PipelineStats,
        papers: list[Paper],
        conflicts: list[ConflictReport],
    ) -> None:
        """Write pipeline run summary to episodic memory."""
        summary = (
            f"Ingestion run: {stats.papers_fetched} fetched, "
            f"{stats.papers_full_processed} fully processed, "
            f"{stats.claims_extracted} claims extracted, "
            f"{stats.conflicts_detected} conflicts detected."
        )
        if conflicts:
            conflict_details = "; ".join(
                f"{c.conflict_type}: {c.new_claim.entity_1} {c.new_claim.relation.value} {c.new_claim.entity_2}"
                for c in conflicts[:5]
            )
            summary += f" Notable conflicts: {conflict_details}"

        try:
            await self._episodic_memory.add_entry(
                content=summary,
                source="ingestion_pipeline",
                tags=["ingestion", "pipeline_run"],
                importance=min(0.3 + 0.1 * len(conflicts), 1.0),
            )
        except Exception:
            logger.exception("memory_write_failed")

    @staticmethod
    def _batched(items: list, size: int):
        """Yield successive batches from a list."""
        for i in range(0, len(items), size):
            yield items[i : i + size]

    async def close(self) -> None:
        """Clean up all resources."""
        await asyncio.gather(
            self._monitor.close(),
            self._pdf_extractor.close(),
            self._claim_extractor.close(),
            self._filter.close(),
            return_exceptions=True,
        )
