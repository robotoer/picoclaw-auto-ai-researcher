"""Structured claim extraction from paper text using LLM."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from auto_researcher.config import LLMConfig
from auto_researcher.models import (
    Claim,
    ClaimRelation,
    ClaimStatus,
    Paper,
)
from auto_researcher.utils.llm import LLMClient
from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)

CLAIM_EXTRACTION_SYSTEM = """\
You are a precise scientific claim extractor. Given a research paper's text, \
extract structured claims as (entity_1, relation, entity_2, conditions, confidence) tuples.

Entities can be: models, methods, architectures, datasets, metrics, concepts, techniques, or authors.

Relations must be one of: outperforms, is_variant_of, refutes, supports, requires, enables, \
is_applied_to, improves, extends, contradicts, is_equivalent_to.

Conditions describe the specific experimental setup or constraints under which the claim holds.

Confidence is your estimate (0.0-1.0) of how strongly the paper supports this claim, based on:
- Strength of evidence presented
- Whether it is a main finding vs. tangential observation
- Reproducibility signals (code available, detailed methodology)
"""

CLAIM_EXTRACTION_PROMPT = """\
Extract all key claims from the following paper text. For each claim, provide:
- entity_1: The subject entity
- relation: One of [outperforms, is_variant_of, refutes, supports, requires, enables, \
is_applied_to, improves, extends, contradicts, is_equivalent_to]
- entity_2: The object entity
- conditions: Experimental conditions or constraints
- confidence: Your confidence in this claim (0.0-1.0)

Paper title: {title}
Paper text:
{text}

Respond with a JSON array of claim objects. Example:
[
  {{
    "entity_1": "GPT-4",
    "relation": "outperforms",
    "entity_2": "GPT-3.5",
    "conditions": "on MMLU benchmark, zero-shot",
    "confidence": 0.95
  }}
]
"""

# Maximum text length to send to LLM (chars). Longer papers are truncated.
_MAX_TEXT_LENGTH = 30000


class ClaimExtractor:
    """Extracts structured claims from papers using an LLM."""

    def __init__(self, llm_config: LLMConfig) -> None:
        self._llm = LLMClient(llm_config)

    async def extract_claims(self, paper: Paper) -> list[Claim]:
        """Extract structured claims from a paper.

        Uses the paper's full text if available, otherwise falls back to abstract.
        """
        text = paper.full_text or paper.metadata.abstract
        if not text:
            logger.warning("no_text_for_extraction", arxiv_id=paper.arxiv_id)
            return []

        # Truncate very long texts
        if len(text) > _MAX_TEXT_LENGTH:
            text = text[:_MAX_TEXT_LENGTH] + "\n[... truncated]"

        prompt = CLAIM_EXTRACTION_PROMPT.format(
            title=paper.metadata.title,
            text=text,
        )

        try:
            result = await self._llm.generate_structured(
                prompt=prompt,
                system=CLAIM_EXTRACTION_SYSTEM,
                temperature=0.2,
            )
        except Exception:
            logger.exception("claim_extraction_failed", arxiv_id=paper.arxiv_id)
            return []

        return self._parse_claims(result, paper.arxiv_id)

    def _parse_claims(self, raw: dict | list, source_paper_id: str) -> list[Claim]:
        """Parse LLM output into Claim objects."""
        items = raw if isinstance(raw, list) else raw.get("claims", [])
        claims: list[Claim] = []

        for item in items:
            try:
                relation_str = item.get("relation", "").lower()
                try:
                    relation = ClaimRelation(relation_str)
                except ValueError:
                    logger.warning("unknown_relation", relation=relation_str)
                    continue

                confidence = float(item.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))

                claim = Claim(
                    id=str(uuid.uuid4()),
                    entity_1=str(item.get("entity_1", "")),
                    relation=relation,
                    entity_2=str(item.get("entity_2", "")),
                    conditions=str(item.get("conditions", "")),
                    confidence=confidence,
                    status=ClaimStatus.EXTRACTED,
                    source_paper_ids=[source_paper_id],
                    extracted_at=datetime.now(UTC),
                )
                claims.append(claim)
            except Exception:
                logger.exception("claim_parse_error", item=str(item))

        logger.info(
            "claims_extracted",
            paper_id=source_paper_id,
            count=len(claims),
        )
        return claims

    async def close(self) -> None:
        await self._llm.close()
