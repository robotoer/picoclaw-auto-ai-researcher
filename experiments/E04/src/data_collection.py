"""Data collection and claim extraction for E04: Knowledge Graph Consistency.

Fetches landmark papers (highly-cited, 2018-2023) and ingestion papers
(recent cs.AI/cs.LG, 2022-2024) from Semantic Scholar. Extracts ground truth
claims from landmark papers via LLM, and runs multi-extractor claim extraction
on ingestion papers with configurable layers.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .models import Claim, ExtractionResult, Paper

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Semantic Scholar helpers
# ---------------------------------------------------------------------------

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = (
    "paperId,title,abstract,year,venue,citationCount,"
    "authors,references,fieldsOfStudy,externalIds"
)

_LANDMARK_QUERIES = [
    "deep learning",
    "transformer",
    "reinforcement learning",
    "language model",
    "computer vision",
]

_INGESTION_QUERIES = [
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "neural network",
    "language model",
    "reinforcement learning",
    "computer vision",
    "generative model",
]

_MULTI_EXTRACTOR_MODELS = [
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-4o",
    "google/gemini-2.0-flash-001",
]


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ReadTimeout)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
)
async def _search_papers(
    client: httpx.AsyncClient,
    query: str,
    *,
    offset: int = 0,
    limit: int = 100,
    year: str = "2022-2024",
) -> dict[str, Any]:
    """Search Semantic Scholar for papers matching *query*."""
    resp = await client.get(
        f"{_S2_BASE}/paper/search",
        params={
            "query": query,
            "offset": offset,
            "limit": limit,
            "fields": _S2_FIELDS,
            "fieldsOfStudy": "Computer Science",
            "year": year,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def _raw_to_paper(raw: dict[str, Any], *, is_landmark: bool = False) -> Paper | None:
    """Convert a Semantic Scholar result dict to a Paper, or None if invalid."""
    abstract = raw.get("abstract")
    citation_count = raw.get("citationCount") or 0
    if not abstract:
        return None

    authors = [a.get("name", "") for a in (raw.get("authors") or [])]
    references = [
        r.get("paperId", "")
        for r in (raw.get("references") or [])
        if r.get("paperId")
    ]

    external_ids = raw.get("externalIds") or {}
    doi = external_ids.get("DOI", "")
    source_url = f"https://doi.org/{doi}" if doi else ""

    return Paper(
        paper_id=raw["paperId"],
        title=raw.get("title", ""),
        abstract=abstract,
        year=raw.get("year", 0),
        venue=raw.get("venue", "") or "",
        citation_count=citation_count,
        authors=authors,
        references=references,
        fields_of_study=raw.get("fieldsOfStudy") or [],
        source_url=source_url,
        is_landmark=is_landmark,
    )


# ---------------------------------------------------------------------------
# Fetch landmark papers
# ---------------------------------------------------------------------------


async def fetch_landmark_papers(n_papers: int = 100) -> list[Paper]:
    """Fetch highly-cited AI/ML papers (>=100 citations, 2018-2023) from Semantic Scholar.

    Searches across multiple AI/ML topic queries, filters for papers with at
    least 100 citations, deduplicates, and samples *n_papers* from the top.
    """
    import random

    log = logger.bind(n_papers=n_papers)
    log.info("fetch_landmark_papers.start")

    all_papers: list[Paper] = []

    async with httpx.AsyncClient() as client:
        for query in _LANDMARK_QUERIES:
            for offset in range(0, 400, 100):
                log.info(
                    "fetch_landmark_papers.search",
                    query=query,
                    offset=offset,
                )
                try:
                    data = await _search_papers(
                        client,
                        query,
                        offset=offset,
                        limit=100,
                        year="2018-2023",
                    )
                except Exception:
                    log.warning(
                        "fetch_landmark_papers.search_failed",
                        query=query,
                        offset=offset,
                        exc_info=True,
                    )
                    continue

                for raw in data.get("data") or []:
                    paper = _raw_to_paper(raw, is_landmark=True)
                    if paper is not None and paper.citation_count >= 100:
                        all_papers.append(paper)

                # Rate limiting: 1 second between requests
                await asyncio.sleep(1.0)

    # Deduplicate by paper_id
    seen: set[str] = set()
    unique: list[Paper] = []
    for p in all_papers:
        if p.paper_id not in seen:
            seen.add(p.paper_id)
            unique.append(p)

    log.info("fetch_landmark_papers.collected", total_unique=len(unique))

    # Sort by citation count descending and sample
    unique.sort(key=lambda p: p.citation_count, reverse=True)

    if len(unique) <= n_papers:
        log.warning(
            "fetch_landmark_papers.insufficient_papers",
            requested=n_papers,
            available=len(unique),
        )
        return unique

    rng = random.Random(42)
    sampled = rng.sample(unique[:n_papers * 3], min(n_papers, len(unique)))
    sampled.sort(key=lambda p: p.citation_count, reverse=True)

    log.info("fetch_landmark_papers.done", n_sampled=len(sampled))
    return sampled


# ---------------------------------------------------------------------------
# Fetch ingestion papers
# ---------------------------------------------------------------------------


async def fetch_ingestion_papers(n_papers: int = 500) -> list[Paper]:
    """Fetch recent AI/ML papers (2022-2024) from Semantic Scholar.

    Uses broader queries and a lower citation threshold (>=5) to collect a
    diverse set of recent papers for ingestion testing.
    """
    import random

    log = logger.bind(n_papers=n_papers)
    log.info("fetch_ingestion_papers.start")

    all_papers: list[Paper] = []

    async with httpx.AsyncClient() as client:
        for query in _INGESTION_QUERIES:
            for offset in range(0, 400, 100):
                log.info(
                    "fetch_ingestion_papers.search",
                    query=query,
                    offset=offset,
                )
                try:
                    data = await _search_papers(
                        client,
                        query,
                        offset=offset,
                        limit=100,
                        year="2022-2024",
                    )
                except Exception:
                    log.warning(
                        "fetch_ingestion_papers.search_failed",
                        query=query,
                        offset=offset,
                        exc_info=True,
                    )
                    continue

                for raw in data.get("data") or []:
                    paper = _raw_to_paper(raw, is_landmark=False)
                    if paper is not None and paper.citation_count >= 5:
                        all_papers.append(paper)

                # Rate limiting: 1 second between requests
                await asyncio.sleep(1.0)

    # Deduplicate by paper_id
    seen: set[str] = set()
    unique: list[Paper] = []
    for p in all_papers:
        if p.paper_id not in seen:
            seen.add(p.paper_id)
            unique.append(p)

    log.info("fetch_ingestion_papers.collected", total_unique=len(unique))

    if len(unique) <= n_papers:
        log.warning(
            "fetch_ingestion_papers.insufficient_papers",
            requested=n_papers,
            available=len(unique),
        )
        return unique

    rng = random.Random(42)
    sampled = rng.sample(unique, n_papers)

    log.info("fetch_ingestion_papers.done", n_sampled=len(sampled))
    return sampled


# ---------------------------------------------------------------------------
# OpenRouter LLM helpers
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ReadTimeout)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
async def _call_openrouter(
    client: httpx.AsyncClient,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.0,
) -> str:
    """Send a chat completion request to OpenRouter and return the content."""
    resp = await client.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)
    return text


# ---------------------------------------------------------------------------
# Ground truth claim extraction from landmark papers
# ---------------------------------------------------------------------------

_GROUND_TRUTH_PROMPT = """\
You are a meticulous AI research analyst. Extract well-established, verifiable \
factual claims from the following paper's abstract. Only include claims that are \
clearly stated and could be independently verified.

Title: {title}
Abstract: {abstract}

Provide your response as a JSON array (and nothing else):
[
  {{
    "text": "the factual claim in a clear, standalone sentence",
    "section": "abstract",
    "quote": "the exact phrase from the abstract supporting this claim",
    "confidence": <float 0.0-1.0, how confident you are this is a verifiable fact>
  }},
  ...
]

Guidelines:
- Extract only factual claims, not opinions or vague statements
- Each claim should be self-contained and understandable without the paper
- Include quantitative results when stated (e.g., "achieves 95% accuracy on X")
- Include methodological claims (e.g., "uses attention mechanism for Y")
- Confidence should reflect how clearly the claim is stated and how verifiable it is
- Aim for 3-8 claims per paper"""


async def extract_ground_truth_claims(
    papers: list[Paper],
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "anthropic/claude-opus-4",
) -> list[Claim]:
    """Extract verified claims from landmark papers via LLM.

    Uses a careful prompt asking the LLM to extract only well-established,
    verifiable factual claims from each paper's abstract.
    Returns claims with full provenance (paper_id, section, quote).
    """
    log = logger.bind(n_papers=len(papers), model=model)
    log.info("extract_ground_truth_claims.start")

    all_claims: list[Claim] = []

    async with httpx.AsyncClient() as client:
        for paper in papers:
            prompt = _GROUND_TRUTH_PROMPT.format(
                title=paper.title,
                abstract=paper.abstract,
            )

            try:
                raw_response = await _call_openrouter(
                    client,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                )
            except Exception:
                log.warning(
                    "extract_ground_truth_claims.call_failed",
                    paper_id=paper.paper_id,
                    exc_info=True,
                )
                continue

            claims = _parse_claims_response(
                raw_response,
                paper_id=paper.paper_id,
                extractor=model,
                verified=True,
            )
            all_claims.extend(claims)
            log.info(
                "extract_ground_truth_claims.paper_done",
                paper_id=paper.paper_id,
                n_claims=len(claims),
            )

            # Rate limiting
            await asyncio.sleep(0.5)

    log.info(
        "extract_ground_truth_claims.done",
        total_claims=len(all_claims),
    )
    return all_claims


def _parse_claims_response(
    raw: str,
    paper_id: str,
    extractor: str,
    *,
    verified: bool = False,
) -> list[Claim]:
    """Parse a JSON array of claims from an LLM response."""
    log = logger.bind(paper_id=paper_id, extractor=extractor)
    text = _strip_code_fences(raw)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        log.warning("parse_claims.json_failed", raw=raw[:200])
        return []

    if not isinstance(parsed, list):
        log.warning("parse_claims.not_a_list", type=type(parsed).__name__)
        return []

    claims: list[Claim] = []
    for item in parsed:
        if not isinstance(item, dict) or "text" not in item:
            continue

        claim_id = f"{paper_id}_{extractor}_{uuid.uuid4().hex[:8]}"
        confidence = item.get("confidence", 1.0)
        if not isinstance(confidence, (int, float)):
            confidence = 1.0

        claims.append(
            Claim(
                claim_id=claim_id,
                paper_id=paper_id,
                text=item["text"],
                section=item.get("section", ""),
                quote=item.get("quote", ""),
                confidence=float(confidence),
                extractor=extractor,
                verified=verified,
            )
        )

    return claims


# ---------------------------------------------------------------------------
# Single-extractor claim extraction
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are an AI research claim extractor. Extract all factual claims from the \
following paper's abstract. Include both methodological claims and result claims.

Title: {title}
Abstract: {abstract}

Provide your response as a JSON array (and nothing else):
[
  {{
    "text": "the factual claim in a clear, standalone sentence",
    "section": "abstract",
    "quote": "the exact phrase from the abstract supporting this claim",
    "confidence": <float 0.0-1.0, how confident you are in this extraction>
  }},
  ...
]

Guidelines:
- Extract all factual claims, including methods, results, and contributions
- Each claim should be self-contained and understandable without the paper
- Include quantitative results when stated
- Confidence reflects extraction quality, not claim truth
- Aim for 3-10 claims per paper"""


async def extract_claims_single(
    paper: Paper,
    api_key: str,
    base_url: str,
    model: str,
) -> ExtractionResult:
    """Extract claims from a single paper using one model."""
    log = logger.bind(paper_id=paper.paper_id, model=model)
    log.info("extract_claims_single.start")

    prompt = _EXTRACTION_PROMPT.format(
        title=paper.title,
        abstract=paper.abstract,
    )

    async with httpx.AsyncClient() as client:
        try:
            raw_response = await _call_openrouter(
                client,
                api_key=api_key,
                base_url=base_url,
                model=model,
                prompt=prompt,
                temperature=0.0,
            )
        except Exception:
            log.warning(
                "extract_claims_single.call_failed",
                exc_info=True,
            )
            return ExtractionResult(
                paper_id=paper.paper_id,
                extractor_model=model,
            )

    claims = _parse_claims_response(
        raw_response,
        paper_id=paper.paper_id,
        extractor=model,
        verified=False,
    )

    log.info("extract_claims_single.done", n_claims=len(claims))
    return ExtractionResult(
        paper_id=paper.paper_id,
        extractor_model=model,
        claims=claims,
    )


# ---------------------------------------------------------------------------
# Multi-extractor claim extraction (Layer 1)
# ---------------------------------------------------------------------------


async def extract_claims_multi(
    paper: Paper,
    api_key: str,
    base_url: str,
    models: list[str] | None = None,
) -> list[ExtractionResult]:
    """Extract claims from a paper using multiple models (for Layer 1 voting).

    Runs extraction with each model independently and returns all results.
    The caller can then apply voting/consensus logic across extractors.
    """
    models = models or _MULTI_EXTRACTOR_MODELS
    log = logger.bind(paper_id=paper.paper_id, n_models=len(models))
    log.info("extract_claims_multi.start")

    results: list[ExtractionResult] = []

    async with httpx.AsyncClient() as client:
        for model in models:
            prompt = _EXTRACTION_PROMPT.format(
                title=paper.title,
                abstract=paper.abstract,
            )

            try:
                raw_response = await _call_openrouter(
                    client,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                )
            except Exception:
                log.warning(
                    "extract_claims_multi.call_failed",
                    model=model,
                    exc_info=True,
                )
                results.append(
                    ExtractionResult(
                        paper_id=paper.paper_id,
                        extractor_model=model,
                    )
                )
                continue

            claims = _parse_claims_response(
                raw_response,
                paper_id=paper.paper_id,
                extractor=model,
                verified=False,
            )

            results.append(
                ExtractionResult(
                    paper_id=paper.paper_id,
                    extractor_model=model,
                    claims=claims,
                )
            )

            log.info(
                "extract_claims_multi.model_done",
                model=model,
                n_claims=len(claims),
            )

            # Rate limiting between models
            await asyncio.sleep(0.5)

    log.info(
        "extract_claims_multi.done",
        n_results=len(results),
        total_claims=sum(len(r.claims) for r in results),
    )
    return results
