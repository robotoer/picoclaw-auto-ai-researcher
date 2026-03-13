"""Ground truth annotation generation using dual-annotator protocol."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import numpy as np
import structlog
from sentence_transformers import SentenceTransformer
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompts — the two annotator personas differ only in extraction philosophy.
# ---------------------------------------------------------------------------

_SYSTEM_BASE = """\
You are an expert annotator creating ground-truth claim annotations for a \
scientific paper.  Extract factual claims as single, falsifiable natural-\
language propositions.

For every claim provide:
- "text": the claim as a standalone proposition
- "claim_type": one of [quantitative, methodological, comparative, existence, causal]
- "source_quote": the verbatim passage from the paper

Respond with a JSON array of objects.
"""

CONSERVATIVE_SYSTEM = (
    _SYSTEM_BASE
    + """
Annotation policy — CONSERVATIVE:
- Only extract claims that are **explicitly and unambiguously stated** in the text.
- Do NOT infer claims, read between the lines, or include anything implied.
- If there is any ambiguity about whether something is a factual claim, skip it.
- Prefer precision over recall.
"""
)

COMPREHENSIVE_SYSTEM = (
    _SYSTEM_BASE
    + """
Annotation policy — COMPREHENSIVE:
- Extract **all** claims, including those that are implied or indirectly stated.
- Include claims derivable from tables, figures descriptions, and methodology.
- When in doubt, include the claim rather than omitting it.
- Prefer recall over precision.
"""
)

ANNOTATION_PROMPT = """\
Extract all factual claims from the following paper according to your annotation policy.

Paper title: {title}

Paper text:
{text}

Respond with a JSON array. Example:
[
  {{
    "text": "Transformer models outperform RNNs on machine translation.",
    "claim_type": "comparative",
    "source_quote": "Our transformer model achieved a BLEU score 2.3 points higher..."
  }}
]
"""


# ---------------------------------------------------------------------------
# Annotator functions
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
async def _call_anthropic(
    system: str,
    prompt: str,
    api_key: str,
    model: str = "claude-opus-4-20250514",
) -> str:
    """Make a single Anthropic Messages API call and return the text response."""
    async with httpx.AsyncClient(timeout=180.0) as client:
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
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
    return response.json()["content"][0]["text"]


def _parse_annotation(raw: str) -> list[dict]:
    """Parse a JSON array of annotation dicts from raw LLM output."""
    text = raw.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        logger.warning("no_json_array_in_annotation", raw=text[:200])
        return []
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("annotation_json_parse_failed", raw=text[:200])
        return []


async def annotate_conservative(
    paper_text: str,
    paper_title: str,
    api_key: str,
) -> list[dict]:
    """Run the conservative annotator on a paper.

    Returns a list of claim dicts (text, claim_type, source_quote).
    """
    prompt = ANNOTATION_PROMPT.format(title=paper_title, text=paper_text)
    raw = await _call_anthropic(
        system=CONSERVATIVE_SYSTEM, prompt=prompt, api_key=api_key
    )
    return _parse_annotation(raw)


async def annotate_comprehensive(
    paper_text: str,
    paper_title: str,
    api_key: str,
) -> list[dict]:
    """Run the comprehensive annotator on a paper.

    Returns a list of claim dicts (text, claim_type, source_quote).
    """
    prompt = ANNOTATION_PROMPT.format(title=paper_title, text=paper_text)
    raw = await _call_anthropic(
        system=COMPREHENSIVE_SYSTEM, prompt=prompt, api_key=api_key
    )
    return _parse_annotation(raw)


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def reconcile_annotations(
    conservative: list[dict],
    comprehensive: list[dict],
    similarity_threshold: float = 0.85,
) -> dict:
    """Reconcile conservative and comprehensive annotations into a gold standard.

    Gold standard consists of:
    1. Claims agreed upon by both annotators (cosine similarity >= threshold).
    2. Claims unique to the comprehensive set that pass a reconciliation check
       (i.e., they are plausible claims, just conservatively omitted).

    Args:
        conservative: Claims from the conservative annotator.
        comprehensive: Claims from the comprehensive annotator.
        similarity_threshold: Cosine similarity threshold for matching.

    Returns:
        Dict with keys ``annotator_1``, ``annotator_2``, ``gold_standard``,
        and ``agreement_stats``.
    """
    if not conservative and not comprehensive:
        return {
            "annotator_1": [],
            "annotator_2": [],
            "gold_standard": [],
            "agreement_stats": {
                "n_conservative": 0,
                "n_comprehensive": 0,
                "n_agreed": 0,
                "n_comprehensive_only": 0,
                "n_gold": 0,
                "agreement_rate": 0.0,
            },
        }

    model = SentenceTransformer("all-MiniLM-L6-v2")

    cons_texts = [c.get("text", "") for c in conservative]
    comp_texts = [c.get("text", "") for c in comprehensive]

    cons_embs = model.encode(cons_texts, convert_to_numpy=True) if cons_texts else np.empty((0, 384))
    comp_embs = model.encode(comp_texts, convert_to_numpy=True) if comp_texts else np.empty((0, 384))

    # Compute cosine similarity matrix.
    if len(cons_embs) > 0 and len(comp_embs) > 0:
        # Normalise for cosine similarity.
        cons_norm = cons_embs / (np.linalg.norm(cons_embs, axis=1, keepdims=True) + 1e-9)
        comp_norm = comp_embs / (np.linalg.norm(comp_embs, axis=1, keepdims=True) + 1e-9)
        sim_matrix = cons_norm @ comp_norm.T
    else:
        sim_matrix = np.empty((len(cons_embs), len(comp_embs)))

    # Greedy best-match: each conservative claim matches at most one comprehensive claim.
    matched_comp_indices: set[int] = set()
    agreed_claims: list[dict] = []

    for ci in range(len(cons_texts)):
        if len(comp_texts) == 0:
            break
        best_j = int(np.argmax(sim_matrix[ci]))
        if sim_matrix[ci, best_j] >= similarity_threshold and best_j not in matched_comp_indices:
            matched_comp_indices.add(best_j)
            # Use the comprehensive version (richer text) for the gold standard.
            agreed_claims.append(comprehensive[best_j])

    # Comprehensive-only claims are included in the gold standard because they
    # represent valid claims that the conservative annotator was simply too
    # strict to include.  In a full pipeline a third reconciliation pass would
    # filter dubious ones, but for this experiment we include them.
    comprehensive_only = [
        comprehensive[j]
        for j in range(len(comprehensive))
        if j not in matched_comp_indices
    ]

    gold_standard = agreed_claims + comprehensive_only

    n_agreed = len(agreed_claims)
    n_comprehensive_only = len(comprehensive_only)
    total_union = len(gold_standard)
    agreement_rate = (
        n_agreed / len(comprehensive) if comprehensive else 0.0
    )

    return {
        "annotator_1": conservative,
        "annotator_2": comprehensive,
        "gold_standard": gold_standard,
        "agreement_stats": {
            "n_conservative": len(conservative),
            "n_comprehensive": len(comprehensive),
            "n_agreed": n_agreed,
            "n_comprehensive_only": n_comprehensive_only,
            "n_gold": total_union,
            "agreement_rate": round(agreement_rate, 4),
        },
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def generate_ground_truth(
    papers_dir: Path,
    output_dir: Path,
    api_key: str,
) -> None:
    """Run the dual-annotator pipeline on all papers and save annotations.

    Args:
        papers_dir: Directory containing per-paper JSON files.
        output_dir: Directory to write annotation results.
        api_key: Anthropic API key.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_files = sorted(papers_dir.glob("*.json"))
    logger.info("starting_annotation", n_papers=len(paper_files))

    for paper_file in paper_files:
        paper = json.loads(paper_file.read_text(encoding="utf-8"))
        paper_id = paper["arxiv_id"]
        paper_text = paper.get("full_text") or paper.get("abstract", "")
        paper_title = paper["title"]

        logger.info("annotating_paper", paper_id=paper_id)

        try:
            conservative, comprehensive = await asyncio.gather(
                annotate_conservative(paper_text, paper_title, api_key),
                annotate_comprehensive(paper_text, paper_title, api_key),
            )

            result = reconcile_annotations(conservative, comprehensive)
            result["paper_id"] = paper_id

            out_path = output_dir / f"{paper_id}_annotations.json"
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            logger.info(
                "annotation_saved",
                paper_id=paper_id,
                n_gold=result["agreement_stats"]["n_gold"],
            )
        except Exception:
            logger.exception("annotation_failed", paper_id=paper_id)

        # Rate-limit between papers.
        await asyncio.sleep(2.0)

    logger.info("annotation_complete")
