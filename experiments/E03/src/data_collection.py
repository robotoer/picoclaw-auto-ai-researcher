"""Data collection and novelty annotation for E03: Semantic Novelty Measurement.

Collects AI/ML papers from Semantic Scholar and generates expert proxy
novelty annotations via LLMs through OpenRouter.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .models import (
    AnnotationSession,
    CitationTier,
    NoveltyAnnotation,
    NoveltyLabel,
    Paper,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Semantic Scholar helpers
# ---------------------------------------------------------------------------

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = (
    "paperId,title,abstract,year,venue,citationCount,"
    "authors,references,fieldsOfStudy,externalIds"
)


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ReadTimeout)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
)
async def _search_papers(
    client: httpx.AsyncClient,
    query: str,
    offset: int = 0,
    limit: int = 100,
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
            "year": "2022-2023",
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def _raw_to_paper(raw: dict[str, Any]) -> Paper | None:
    """Convert a Semantic Scholar result dict to a Paper, or None if invalid."""
    abstract = raw.get("abstract")
    citation_count = raw.get("citationCount") or 0
    if not abstract or citation_count < 1:
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
        citation_count_2yr=citation_count,
        authors=authors,
        references=references,
        fields_of_study=raw.get("fieldsOfStudy") or [],
        source_url=source_url,
    )


async def fetch_papers(n_papers: int = 100) -> list[Paper]:
    """Fetch AI/ML papers from Semantic Scholar and sample by citation tier.

    Collects ~500-1000 candidate papers, then samples *n_papers* total:
    half high-impact (top 10% by citation) and half average (25th-75th
    percentile).
    """
    log = logger.bind(n_papers=n_papers)
    log.info("fetch_papers.start")

    all_papers: list[Paper] = []
    queries = ["machine learning", "deep learning", "artificial intelligence"]

    async with httpx.AsyncClient() as client:
        for query in queries:
            for offset in range(0, 400, 100):
                log.info(
                    "fetch_papers.search",
                    query=query,
                    offset=offset,
                )
                try:
                    data = await _search_papers(
                        client, query=query, offset=offset, limit=100
                    )
                except Exception:
                    log.warning(
                        "fetch_papers.search_failed",
                        query=query,
                        offset=offset,
                        exc_info=True,
                    )
                    continue

                for raw in data.get("data") or []:
                    paper = _raw_to_paper(raw)
                    if paper is not None:
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

    log.info("fetch_papers.collected", total_unique=len(unique))

    # Sort by citation count descending
    unique.sort(key=lambda p: p.citation_count_2yr, reverse=True)

    if len(unique) < n_papers:
        log.warning(
            "fetch_papers.insufficient_papers",
            requested=n_papers,
            available=len(unique),
        )
        for p in unique:
            p.citation_tier = CitationTier.average
        return unique

    # Determine percentile boundaries
    n_total = len(unique)
    top_10_idx = max(1, int(n_total * 0.10))
    p25_idx = int(n_total * 0.25)
    p75_idx = int(n_total * 0.75)

    high_impact_pool = unique[:top_10_idx]
    average_pool = unique[p25_idx:p75_idx]

    half = n_papers // 2

    import random

    rng = random.Random(42)
    sampled_high = rng.sample(high_impact_pool, min(half, len(high_impact_pool)))
    sampled_avg = rng.sample(average_pool, min(half, len(average_pool)))

    for p in sampled_high:
        p.citation_tier = CitationTier.high_impact
    for p in sampled_avg:
        p.citation_tier = CitationTier.average

    result = sampled_high + sampled_avg
    log.info(
        "fetch_papers.done",
        high_impact=len(sampled_high),
        average=len(sampled_avg),
    )
    return result


# ---------------------------------------------------------------------------
# TF-IDF embeddings
# ---------------------------------------------------------------------------


def compute_tfidf_embeddings(papers: list[Paper]) -> list[Paper]:
    """Compute TF-IDF embeddings for papers using abstract text."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    log = logger.bind(n_papers=len(papers))
    log.info("compute_tfidf_embeddings.start")

    texts = [p.abstract for p in papers]
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    updated: list[Paper] = []
    for i, paper in enumerate(papers):
        updated_paper = paper.model_copy(
            update={"embedding": tfidf_matrix[i].toarray()[0].tolist()}
        )
        updated.append(updated_paper)

    log.info("compute_tfidf_embeddings.done", embedding_dim=tfidf_matrix.shape[1])
    return updated


# ---------------------------------------------------------------------------
# OpenRouter LLM helpers
# ---------------------------------------------------------------------------

_PERSONA_DESCRIPTIONS: dict[str, str] = {
    "expert_theorist": (
        "a senior ML researcher with 15+ years of experience, focused on "
        "theoretical contributions and mathematical novelty in machine learning. "
        "You value papers that introduce fundamentally new formalisms, proofs, "
        "or theoretical frameworks."
    ),
    "expert_empiricist": (
        "an applied ML researcher focused on methodological novelty and "
        "experimental rigor. You value papers that introduce new methods, "
        "architectures, or training procedures that demonstrably advance "
        "the state of the art."
    ),
    "expert_reviewer": (
        "an experienced peer reviewer for top ML venues (NeurIPS, ICML, ICLR) "
        "focused on overall contribution significance. You evaluate whether a "
        "paper makes a meaningful advance over prior work and whether its "
        "claims are well-supported."
    ),
}

_ANNOTATION_PROMPT = """\
You are {persona_description}

Rate the novelty of the following research paper based on its abstract.

Title: {title}
Abstract: {abstract}

Provide your assessment as JSON (and nothing else):
{{
  "binary_label": "novel" or "incremental",
  "likert_score": <integer 1-7>,
  "reasoning": "brief explanation"
}}

Guidelines:
- "novel": Introduces a substantially new idea, method, or finding not previously established
- "incremental": Extends or applies existing work without a qualitatively new contribution
- Score 1-3: incremental, 4: borderline, 5-7: novel"""


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


def _parse_annotation_response(raw: str, paper_id: str, rater_id: str, rater_type: str) -> NoveltyAnnotation | None:
    """Parse a JSON annotation response from the LLM."""
    log = logger.bind(paper_id=paper_id, rater_id=rater_id)

    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        log.warning("parse_annotation.json_failed", raw=raw[:200])
        return None

    binary_label_raw = parsed.get("binary_label", "").lower().strip()
    try:
        label = NoveltyLabel(binary_label_raw)
    except ValueError:
        log.warning("parse_annotation.invalid_label", label=binary_label_raw)
        return None

    likert = parsed.get("likert_score")
    if not isinstance(likert, int) or not (1 <= likert <= 7):
        log.warning("parse_annotation.invalid_likert", likert=likert)
        return None

    return NoveltyAnnotation(
        paper_id=paper_id,
        rater_id=rater_id,
        rater_type=rater_type,
        binary_label=label,
        likert_score=likert,
        reasoning=parsed.get("reasoning", ""),
    )


async def annotate_papers_novelty(
    papers: list[Paper],
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "anthropic/claude-opus-4",
) -> list[AnnotationSession]:
    """Generate expert proxy novelty annotations for papers.

    Creates three annotation sessions (one per persona), each rating every
    paper independently via the specified LLM model.
    """
    log = logger.bind(n_papers=len(papers), model=model)
    log.info("annotate_papers_novelty.start")

    sessions: list[AnnotationSession] = []

    async with httpx.AsyncClient() as client:
        for persona_id, persona_desc in _PERSONA_DESCRIPTIONS.items():
            log.info("annotate_papers_novelty.persona_start", persona=persona_id)
            session = AnnotationSession(
                rater_id=persona_id,
                rater_type="expert_proxy",
                model_name=model,
            )

            for paper in papers:
                prompt = _ANNOTATION_PROMPT.format(
                    persona_description=persona_desc,
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
                        "annotate_papers_novelty.call_failed",
                        paper_id=paper.paper_id,
                        persona=persona_id,
                        exc_info=True,
                    )
                    continue

                annotation = _parse_annotation_response(
                    raw_response,
                    paper_id=paper.paper_id,
                    rater_id=persona_id,
                    rater_type="expert_proxy",
                )
                if annotation is not None:
                    session.annotations.append(annotation)

                # Rate limiting
                await asyncio.sleep(0.5)

            sessions.append(session)
            log.info(
                "annotate_papers_novelty.persona_done",
                persona=persona_id,
                n_annotations=len(session.annotations),
            )

    log.info("annotate_papers_novelty.done", n_sessions=len(sessions))
    return sessions


# ---------------------------------------------------------------------------
# LLM judge novelty scores
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an AI research expert. Evaluate the novelty of the following paper.

Title: {title}
Abstract: {abstract}

Provide your assessment as JSON (and nothing else):
{{
  "binary_label": "novel" or "incremental",
  "likert_score": <integer 1-7>,
  "reasoning": "brief explanation"
}}

Guidelines:
- "novel": Introduces a substantially new idea, method, or finding not previously established
- "incremental": Extends or applies existing work without a qualitatively new contribution
- Score 1-3: incremental, 4: borderline, 5-7: novel"""

_JUDGE_MODELS = [
    "anthropic/claude-opus-4-6",
    "anthropic/claude-sonnet-4-6",
]


async def get_llm_novelty_scores(
    papers: list[Paper],
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    models: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Get LLM novelty judgments from specified models.

    Returns a list of dicts, each with keys: paper_id, model, score, label.
    """
    models = models or _JUDGE_MODELS
    log = logger.bind(n_papers=len(papers), models=models)
    log.info("get_llm_novelty_scores.start")

    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient() as client:
        for model in models:
            log.info("get_llm_novelty_scores.model_start", model=model)

            for paper in papers:
                prompt = _JUDGE_PROMPT.format(
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
                        "get_llm_novelty_scores.call_failed",
                        paper_id=paper.paper_id,
                        model=model,
                        exc_info=True,
                    )
                    continue

                annotation = _parse_annotation_response(
                    raw_response,
                    paper_id=paper.paper_id,
                    rater_id=f"judge_{model}",
                    rater_type="llm_judge",
                )
                if annotation is not None:
                    results.append({
                        "paper_id": paper.paper_id,
                        "model": model,
                        "score": annotation.likert_score,
                        "label": annotation.binary_label.value,
                    })

                # Rate limiting
                await asyncio.sleep(0.5)

            log.info(
                "get_llm_novelty_scores.model_done",
                model=model,
                n_results=sum(1 for r in results if r["model"] == model),
            )

    log.info("get_llm_novelty_scores.done", total_results=len(results))
    return results
