"""Fetch and select papers for E01 experiment."""

from __future__ import annotations

import json
from pathlib import Path

import arxiv
import structlog

logger = structlog.get_logger(__name__)

# Queries designed to find specific paper types in relevant AI/ML categories.
PAPER_TYPE_QUERIES: dict[str, dict[str, str]] = {
    "empirical": {
        "category": "cs.LG",
        "query": "benchmark evaluation experimental results dataset",
    },
    "theoretical": {
        "category": "cs.LG",
        "query": "theoretical analysis convergence proof bounds",
    },
    "survey": {
        "category": "cs.AI",
        "query": "survey review overview taxonomy comprehensive",
    },
    "methods": {
        "category": "cs.CL",
        "query": "novel method architecture algorithm framework",
    },
}


def fetch_candidate_papers(
    category: str, query: str, max_results: int = 50
) -> list[dict]:
    """Search ArXiv for candidate papers matching a category and query.

    Args:
        category: ArXiv category (e.g. 'cs.LG').
        query: Free-text search query.
        max_results: Maximum number of results to fetch.

    Returns:
        List of paper metadata dicts.
    """
    search_query = f"cat:{category} AND ({query})"
    logger.info("fetching_candidates", query=search_query, max_results=max_results)

    client = arxiv.Client()
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: list[dict] = []
    for result in client.results(search):
        papers.append(
            {
                "arxiv_id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "categories": result.categories,
                "published": result.published.isoformat(),
                "pdf_url": result.pdf_url,
            }
        )

    logger.info("candidates_fetched", count=len(papers))
    return papers


def select_papers(n_per_type: int = 5) -> list[dict]:
    """Select papers stratified by type (empirical, theoretical, survey, methods).

    Fetches candidates for each type and selects the top ``n_per_type``
    from each category to produce a balanced corpus.

    Args:
        n_per_type: Number of papers to select per paper type.

    Returns:
        List of selected paper dicts with a ``paper_type`` field attached.
    """
    selected: list[dict] = []

    for paper_type, params in PAPER_TYPE_QUERIES.items():
        logger.info("selecting_papers", paper_type=paper_type)
        candidates = fetch_candidate_papers(
            category=params["category"],
            query=params["query"],
            max_results=50,
        )

        for paper in candidates[:n_per_type]:
            paper["paper_type"] = paper_type
            selected.append(paper)

    logger.info("papers_selected", total=len(selected))
    return selected


def download_paper_texts(papers: list[dict], output_dir: Path) -> list[dict]:
    """Download paper texts and save each as a JSON file.

    For practical reasons, we use the abstract as the text source rather than
    attempting PDF parsing, which is complex and unreliable.  This is a known
    simplification — the experiment measures extraction quality on whatever text
    is provided, so using abstracts keeps the pipeline deterministic and
    reproducible while still exercising the full extraction workflow.

    Args:
        papers: List of paper metadata dicts (from :func:`select_papers`).
        output_dir: Directory to write per-paper JSON files.

    Returns:
        The enriched list of paper dicts (with ``full_text`` added).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    enriched: list[dict] = []
    for paper in papers:
        # Use abstract as full_text (see docstring for rationale).
        paper_record = {
            "arxiv_id": paper["arxiv_id"],
            "title": paper["title"],
            "authors": paper["authors"],
            "abstract": paper["abstract"],
            "categories": paper["categories"],
            "published": paper["published"],
            "paper_type": paper["paper_type"],
            "full_text": paper["abstract"],
        }

        out_path = output_dir / f"{paper['arxiv_id']}.json"
        out_path.write_text(json.dumps(paper_record, indent=2), encoding="utf-8")
        logger.info("paper_saved", arxiv_id=paper["arxiv_id"], path=str(out_path))
        enriched.append(paper_record)

    return enriched


def main() -> None:
    """Run the full paper-fetching pipeline and save to the E01 data directory."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )

    data_dir = Path(__file__).resolve().parent.parent / "data" / "papers"
    logger.info("starting_paper_fetch", output_dir=str(data_dir))

    papers = select_papers(n_per_type=5)
    enriched = download_paper_texts(papers, data_dir)

    logger.info("pipeline_complete", total_papers=len(enriched))


if __name__ == "__main__":
    main()
