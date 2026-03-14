"""ArXiv API monitoring and paper fetching."""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from collections.abc import Callable, Awaitable
from datetime import datetime

import httpx

from auto_researcher.config import ArxivConfig
from auto_researcher.models import Paper, PaperMetadata, ProcessingLevel
from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)

ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"


class ArxivMonitor:
    """Monitors ArXiv for new papers in configured categories."""

    def __init__(self, config: ArxivConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(timeout=60.0)
        self._processed_ids: set[str] = set()

    async def fetch_recent_papers(self) -> list[Paper]:
        """Fetch recent papers from ArXiv across all configured categories."""
        all_papers: list[Paper] = []
        for category in self.config.categories:
            try:
                papers = await self._fetch_category(category)
                new_papers = [p for p in papers if p.arxiv_id not in self._processed_ids]
                all_papers.extend(new_papers)
                logger.info(
                    "fetched_papers",
                    category=category,
                    total=len(papers),
                    new=len(new_papers),
                )
            except Exception:
                logger.exception("fetch_failed", category=category)
        # Deduplicate by arxiv_id (papers can appear in multiple categories)
        seen: set[str] = set()
        unique: list[Paper] = []
        for paper in all_papers:
            if paper.arxiv_id not in seen:
                seen.add(paper.arxiv_id)
                unique.append(paper)
        return unique

    async def _fetch_category(self, category: str) -> list[Paper]:
        """Fetch papers for a single ArXiv category."""
        params: dict[str, str | int] = {
            "search_query": f"cat:{category}",
            "start": 0,
            "max_results": self.config.max_results_per_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = await self._client.get(ARXIV_API_BASE, params=params)
        resp.raise_for_status()
        return self._parse_feed(resp.text)

    def _parse_feed(self, xml_text: str) -> list[Paper]:
        """Parse ArXiv Atom feed XML into Paper objects."""
        root = ET.fromstring(xml_text)
        papers: list[Paper] = []
        for entry in root.findall(f"{{{ATOM_NS}}}entry"):
            try:
                paper = self._parse_entry(entry)
                if paper is not None:
                    papers.append(paper)
            except Exception:
                logger.exception("parse_entry_failed")
        return papers

    def _parse_entry(self, entry: ET.Element) -> Paper | None:
        """Parse a single Atom entry into a Paper."""
        arxiv_id_raw = entry.findtext(f"{{{ATOM_NS}}}id", default="")
        if not arxiv_id_raw:
            return None
        # ArXiv ID is the last path segment of the URL
        arxiv_id = arxiv_id_raw.split("/abs/")[-1]

        title = entry.findtext(f"{{{ATOM_NS}}}title", default="").strip()
        title = " ".join(title.split())  # normalize whitespace

        abstract = entry.findtext(f"{{{ATOM_NS}}}summary", default="").strip()
        abstract = " ".join(abstract.split())

        authors = [
            author.findtext(f"{{{ATOM_NS}}}name", default="")
            for author in entry.findall(f"{{{ATOM_NS}}}author")
        ]

        categories = [
            cat.get("term", "")
            for cat in entry.findall(f"{{{ARXIV_NS}}}primary_category")
        ]
        categories += [
            cat.get("term", "")
            for cat in entry.findall(f"{{{ATOM_NS}}}category")
            if cat.get("term", "") not in categories
        ]

        published_str = entry.findtext(f"{{{ATOM_NS}}}published", default="")
        updated_str = entry.findtext(f"{{{ATOM_NS}}}updated", default="")
        published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        updated = (
            datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
            if updated_str
            else None
        )

        pdf_url = None
        for link in entry.findall(f"{{{ATOM_NS}}}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break

        doi_el = entry.find(f"{{{ARXIV_NS}}}doi")
        journal_ref_el = entry.find(f"{{{ARXIV_NS}}}journal_ref")
        comment_el = entry.find(f"{{{ARXIV_NS}}}comment")

        metadata = PaperMetadata(
            arxiv_id=arxiv_id,
            title=title,
            authors=authors,
            abstract=abstract,
            categories=categories,
            published=published,
            updated=updated,
            pdf_url=pdf_url,
            doi=doi_el.text if doi_el is not None else None,
            journal_ref=journal_ref_el.text if journal_ref_el is not None else None,
            comment=comment_el.text if comment_el is not None else None,
            source_url=arxiv_id_raw,
        )

        return Paper(metadata=metadata, processing_level=ProcessingLevel.UNPROCESSED)

    def mark_processed(self, arxiv_ids: list[str]) -> None:
        """Mark paper IDs as already processed to avoid duplicates."""
        self._processed_ids.update(arxiv_ids)

    def load_processed_ids(self, ids: set[str]) -> None:
        """Load a set of previously processed IDs (e.g. from persistent storage)."""
        self._processed_ids = ids

    async def poll_loop(self, callback: Callable[[list[Paper]], Awaitable[None]]) -> None:
        """Continuously poll ArXiv at the configured interval.

        Args:
            callback: Async callable that receives a list of new Papers.
        """
        interval_seconds = self.config.poll_interval_hours * 3600
        while True:
            try:
                papers = await self.fetch_recent_papers()
                if papers:
                    await callback(papers)
                    self.mark_processed([p.arxiv_id for p in papers])
            except Exception:
                logger.exception("poll_loop_error")
            await asyncio.sleep(interval_seconds)

    async def close(self) -> None:
        await self._client.aclose()
