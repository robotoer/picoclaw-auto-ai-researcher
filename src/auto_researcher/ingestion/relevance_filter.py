"""Adaptive relevance scoring for incoming papers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from auto_researcher.config import ArxivConfig, LLMConfig
from auto_researcher.models import Paper, ProcessingLevel
from auto_researcher.utils.llm import LLMClient
from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)

RELEVANCE_SYSTEM = """\
You are a research relevance assessor. Given a paper's title and abstract, \
and a description of the current research agenda, score the paper's relevance \
on a scale of 0.0 to 1.0.

Consider:
- Direct relevance to current research questions
- Methodological novelty that could be applied
- Foundational contributions to related fields
- Potential for cross-pollination of ideas
"""

RELEVANCE_PROMPT = """\
Research agenda:
{agenda}

Paper title: {title}
Paper abstract: {abstract}
Paper categories: {categories}

Score this paper's relevance to the research agenda on a 0.0 to 1.0 scale.
Respond with JSON: {{"score": <float>, "reasoning": "<brief explanation>"}}
"""


class FilterTier(str, Enum):
    DISCARD = "discard"
    ABSTRACT_ONLY = "abstract_only"
    FULL_PROCESSING = "full_processing"


@dataclass
class RelevanceResult:
    """Result of relevance scoring for a paper."""

    paper: Paper
    score: float
    tier: FilterTier
    reasoning: str = ""


class RelevanceFilter:
    """Scores papers on relevance to the current research agenda.

    Three tiers:
    - discard: score < relevance_threshold (default 0.3)
    - abstract_only: relevance_threshold <= score < full_processing_threshold (default 0.7)
    - full_processing: score >= full_processing_threshold
    """

    def __init__(
        self,
        arxiv_config: ArxivConfig,
        llm_config: LLMConfig,
        research_agenda: str = "",
    ) -> None:
        self._config = arxiv_config
        self._llm = LLMClient(llm_config)
        self._research_agenda = research_agenda
        self._keyword_weights: dict[str, float] = {}

    def set_research_agenda(self, agenda: str) -> None:
        """Update the research agenda used for relevance scoring."""
        self._research_agenda = agenda

    def set_keyword_weights(self, weights: dict[str, float]) -> None:
        """Set keyword-based scoring weights for fast pre-filtering."""
        self._keyword_weights = weights

    async def score_papers(self, papers: list[Paper]) -> list[RelevanceResult]:
        """Score a batch of papers and classify into processing tiers."""
        results: list[RelevanceResult] = []
        for paper in papers:
            result = await self._score_single(paper)
            results.append(result)

        # Log tier distribution
        tier_counts = {t: 0 for t in FilterTier}
        for r in results:
            tier_counts[r.tier] += 1
        logger.info(
            "relevance_scoring_complete",
            total=len(results),
            full=tier_counts[FilterTier.FULL_PROCESSING],
            abstract=tier_counts[FilterTier.ABSTRACT_ONLY],
            discard=tier_counts[FilterTier.DISCARD],
        )
        return results

    async def _score_single(self, paper: Paper) -> RelevanceResult:
        """Score a single paper's relevance."""
        # Fast keyword pre-filter
        keyword_score = self._keyword_score(paper)

        # If keyword score is very low, skip LLM scoring
        if keyword_score < 0.1 and self._keyword_weights:
            return self._make_result(paper, keyword_score, "low keyword relevance")

        # LLM-based scoring if we have an agenda
        if self._research_agenda:
            try:
                llm_score, reasoning = await self._llm_score(paper)
                # Blend keyword and LLM scores
                if self._keyword_weights:
                    final_score = 0.3 * keyword_score + 0.7 * llm_score
                else:
                    final_score = llm_score
                return self._make_result(paper, final_score, reasoning)
            except Exception:
                logger.exception("llm_scoring_failed", arxiv_id=paper.arxiv_id)

        # Fallback to keyword score only
        return self._make_result(paper, keyword_score, "keyword-only scoring")

    def _keyword_score(self, paper: Paper) -> float:
        """Fast keyword-based relevance score."""
        if not self._keyword_weights:
            return 0.5  # neutral default

        text = f"{paper.metadata.title} {paper.metadata.abstract}".lower()
        score = 0.0
        max_possible = sum(abs(w) for w in self._keyword_weights.values())

        for keyword, weight in self._keyword_weights.items():
            if keyword.lower() in text:
                score += weight

        if max_possible > 0:
            return max(0.0, min(1.0, 0.5 + score / (2 * max_possible)))
        return 0.5

    async def _llm_score(self, paper: Paper) -> tuple[float, str]:
        """Use LLM to score relevance."""
        prompt = RELEVANCE_PROMPT.format(
            agenda=self._research_agenda,
            title=paper.metadata.title,
            abstract=paper.metadata.abstract,
            categories=", ".join(paper.metadata.categories),
        )
        result = await self._llm.generate_structured(
            prompt=prompt,
            system=RELEVANCE_SYSTEM,
            temperature=0.1,
        )
        score = float(result.get("score", 0.5))
        score = max(0.0, min(1.0, score))
        reasoning = str(result.get("reasoning", ""))
        return score, reasoning

    def _make_result(
        self, paper: Paper, score: float, reasoning: str
    ) -> RelevanceResult:
        """Create a RelevanceResult with the appropriate tier."""
        if score >= self._config.full_processing_threshold:
            tier = FilterTier.FULL_PROCESSING
        elif score >= self._config.relevance_threshold:
            tier = FilterTier.ABSTRACT_ONLY
        else:
            tier = FilterTier.DISCARD

        updated_paper = paper.model_copy(
            update={
                "relevance_score": score,
                "processing_level": (
                    ProcessingLevel.FULL_TEXT
                    if tier == FilterTier.FULL_PROCESSING
                    else ProcessingLevel.ABSTRACT_ONLY
                    if tier == FilterTier.ABSTRACT_ONLY
                    else ProcessingLevel.UNPROCESSED
                ),
            }
        )
        return RelevanceResult(
            paper=updated_paper, score=score, tier=tier, reasoning=reasoning
        )

    async def close(self) -> None:
        await self._llm.close()
