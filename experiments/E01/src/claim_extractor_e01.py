"""Claim extraction for E01 experiment using multiple LLM providers."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import httpx
import structlog
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ClaimType(str, Enum):
    QUANTITATIVE = "quantitative"
    METHODOLOGICAL = "methodological"
    COMPARATIVE = "comparative"
    EXISTENCE = "existence"
    CAUSAL = "causal"


class ExtractedClaim(BaseModel):
    """A single free-text claim extracted from a paper."""

    text: str = Field(description="Natural-language falsifiable proposition")
    claim_type: ClaimType = Field(description="Category of the claim")
    confidence: float = Field(ge=0.0, le=1.0, description="Model's confidence")
    source_quote: str = Field(description="Verbatim text the claim is based on")


class ClaimExtractionResult(BaseModel):
    """Full result of one extraction run on one paper by one model."""

    paper_id: str
    model_name: str
    run_number: int
    claims: list[ExtractedClaim] = Field(default_factory=list)
    raw_response: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = 0.0
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


# ---------------------------------------------------------------------------
# Shared prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert scientific claim extractor. Your task is to identify every \
discrete, falsifiable factual claim in a research paper and express each as a \
single natural-language proposition.

Guidelines:
- Extract all factual claims as single falsifiable propositions.
- Classify each claim as one of: quantitative, methodological, comparative, \
existence, causal.
- Provide your confidence estimate (0.0-1.0) for each claim.
- Quote the exact source text the claim is derived from.
- Do NOT include opinions, future work suggestions, or hedged speculation.
- Decompose compound claims into individual atomic propositions.
- Each claim must be understandable in isolation, without needing the paper context.
"""

EXTRACTION_PROMPT = """\
Extract all factual claims from the following paper.

Paper title: {title}

Paper text:
{text}

Respond with a JSON array of claim objects. Each object must have:
- "text": a single falsifiable natural-language proposition
- "claim_type": one of ["quantitative", "methodological", "comparative", "existence", "causal"]
- "confidence": float 0.0-1.0
- "source_quote": the verbatim passage from the paper supporting this claim

Example:
[
  {{
    "text": "BERT achieves 84.6% accuracy on the GLUE benchmark.",
    "claim_type": "quantitative",
    "confidence": 0.95,
    "source_quote": "Our model achieves 84.6% on GLUE..."
  }}
]
"""


# ---------------------------------------------------------------------------
# Provider-specific extraction functions
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
async def extract_claims_anthropic(
    paper_text: str,
    paper_title: str,
    model: str,
    api_key: str,
) -> ClaimExtractionResult:
    """Extract claims using the Anthropic Messages API.

    Args:
        paper_text: The paper's full text (or abstract).
        paper_title: Title of the paper.
        model: Anthropic model identifier (e.g. ``claude-sonnet-4-20250514``).
        api_key: Anthropic API key.

    Returns:
        A :class:`ClaimExtractionResult` with parsed claims.
    """
    prompt = EXTRACTION_PROMPT.format(title=paper_title, text=paper_text)

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 4096,
                "temperature": 0,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
    latency_ms = (time.perf_counter() - start) * 1000

    body = response.json()
    raw_text = body["content"][0]["text"]
    usage = {
        "input_tokens": body.get("usage", {}).get("input_tokens", 0),
        "output_tokens": body.get("usage", {}).get("output_tokens", 0),
    }

    claims = _parse_claims(raw_text)

    return ClaimExtractionResult(
        paper_id="",  # filled in by caller
        model_name=model,
        run_number=0,
        claims=claims,
        raw_response=raw_text,
        usage=usage,
        latency_ms=latency_ms,
    )


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
async def extract_claims_openai(
    paper_text: str,
    paper_title: str,
    model: str,
    api_key: str,
) -> ClaimExtractionResult:
    """Extract claims using the OpenAI Chat Completions API.

    Args:
        paper_text: The paper's full text (or abstract).
        paper_title: Title of the paper.
        model: OpenAI model identifier (e.g. ``gpt-4o``).
        api_key: OpenAI API key.

    Returns:
        A :class:`ClaimExtractionResult` with parsed claims.
    """
    prompt = EXTRACTION_PROMPT.format(title=paper_title, text=paper_text)

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        response.raise_for_status()
    latency_ms = (time.perf_counter() - start) * 1000

    body = response.json()
    raw_text = body["choices"][0]["message"]["content"]
    usage_data = body.get("usage", {})
    usage = {
        "input_tokens": usage_data.get("prompt_tokens", 0),
        "output_tokens": usage_data.get("completion_tokens", 0),
    }

    claims = _parse_claims(raw_text)

    return ClaimExtractionResult(
        paper_id="",
        model_name=model,
        run_number=0,
        claims=claims,
        raw_response=raw_text,
        usage=usage,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROVIDER_FUNCS = {
    "anthropic": extract_claims_anthropic,
    "openai": extract_claims_openai,
}


def _parse_claims(raw_text: str) -> list[ExtractedClaim]:
    """Parse a JSON array of claim dicts from raw LLM output."""
    # Try to find JSON array in the response (models sometimes wrap in markdown).
    text = raw_text.strip()
    start_idx = text.find("[")
    end_idx = text.rfind("]")
    if start_idx == -1 or end_idx == -1:
        logger.warning("no_json_array_in_response", raw=text[:200])
        return []

    try:
        items = json.loads(text[start_idx : end_idx + 1])
    except json.JSONDecodeError:
        logger.warning("json_parse_failed", raw=text[:200])
        return []

    claims: list[ExtractedClaim] = []
    for item in items:
        try:
            claims.append(ExtractedClaim(**item))
        except Exception:
            logger.warning("claim_parse_error", item=str(item)[:200])
    return claims


async def run_extraction(
    papers_dir: Path,
    output_dir: Path,
    models: list[dict[str, str]],
    n_runs: int = 3,
) -> None:
    """Run claim extraction across all papers, models, and repetitions.

    Args:
        papers_dir: Directory containing per-paper JSON files.
        output_dir: Directory to write extraction results.
        models: List of dicts with keys ``name``, ``provider``, ``api_key``.
            Provider must be ``"anthropic"`` or ``"openai"``.
        n_runs: Number of independent runs per (paper, model) pair.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_files = sorted(papers_dir.glob("*.json"))
    logger.info("starting_extraction", n_papers=len(paper_files), n_models=len(models))

    for paper_file in paper_files:
        paper = json.loads(paper_file.read_text(encoding="utf-8"))
        paper_id = paper["arxiv_id"]
        paper_text = paper.get("full_text") or paper.get("abstract", "")
        paper_title = paper["title"]

        for model_cfg in models:
            provider = model_cfg["provider"]
            model_name = model_cfg["name"]
            api_key = model_cfg["api_key"]

            extract_fn = _PROVIDER_FUNCS.get(provider)
            if extract_fn is None:
                logger.error("unknown_provider", provider=provider)
                continue

            for run_num in range(1, n_runs + 1):
                logger.info(
                    "extracting",
                    paper_id=paper_id,
                    model=model_name,
                    run=run_num,
                )
                try:
                    result = await extract_fn(
                        paper_text=paper_text,
                        paper_title=paper_title,
                        model=model_name,
                        api_key=api_key,
                    )
                    result.paper_id = paper_id
                    result.run_number = run_num

                    out_path = output_dir / f"{paper_id}_{model_name}_run{run_num}.json"
                    out_path.write_text(
                        result.model_dump_json(indent=2), encoding="utf-8"
                    )
                    logger.info(
                        "extraction_saved",
                        path=str(out_path),
                        n_claims=len(result.claims),
                    )
                except Exception:
                    logger.exception(
                        "extraction_failed",
                        paper_id=paper_id,
                        model=model_name,
                        run=run_num,
                    )

                # Small delay to be polite to APIs.
                await asyncio.sleep(1.0)
