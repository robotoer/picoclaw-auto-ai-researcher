"""Nightly knowledge consolidation: dedup, decay, promote/demote hypotheses."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from auto_researcher.config import ConsolidationConfig
from auto_researcher.models.claim import Claim, ClaimStatus
from auto_researcher.models.hypothesis import Hypothesis, HypothesisStatus
from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)


class KnowledgeConsolidator:
    """Performs nightly knowledge graph maintenance and consolidation."""

    def __init__(self, config: ConsolidationConfig) -> None:
        self._config = config

    async def run_consolidation(
        self,
        claims: list[Claim],
        hypotheses: list[Hypothesis],
        embeddings: dict[str, list[float]] | None = None,
    ) -> ConsolidationReport:
        """Run the full nightly consolidation pipeline."""
        now = datetime.now(UTC)

        # 1. Deduplicate semantically similar claims
        dedup_result = self._deduplicate_claims(claims, embeddings)

        # 2. Apply confidence decay to stale facts
        decayed = self._apply_confidence_decay(claims, now)

        # 3. Promote/demote hypotheses
        promoted, demoted = self._update_hypothesis_status(hypotheses, now)

        # 4. Flag stale hypotheses
        stale = self._flag_stale_hypotheses(hypotheses, now)

        report = ConsolidationReport(
            duplicates_merged=dedup_result["merged_count"],
            claims_decayed=len(decayed),
            hypotheses_promoted=len(promoted),
            hypotheses_demoted=len(demoted),
            hypotheses_flagged_stale=len(stale),
            below_confidence_threshold=len([
                c for c in claims
                if c.decayed_confidence(now) < self._config.min_confidence_threshold
            ]),
            run_at=now,
        )

        logger.info(
            "consolidation_complete",
            merged=report.duplicates_merged,
            decayed=report.claims_decayed,
            promoted=report.hypotheses_promoted,
            demoted=report.hypotheses_demoted,
            stale=report.hypotheses_flagged_stale,
        )
        return report

    def _deduplicate_claims(
        self,
        claims: list[Claim],
        embeddings: dict[str, list[float]] | None,
    ) -> dict[str, int]:
        """Cluster semantically similar claims and mark duplicates."""
        if not embeddings or len(claims) < 2:
            return {"merged_count": 0}

        merged_count = 0
        processed: set[str] = set()

        for i, claim_a in enumerate(claims):
            if claim_a.id in processed:
                continue
            emb_a = embeddings.get(claim_a.id)
            if emb_a is None:
                continue

            for j in range(i + 1, len(claims)):
                claim_b = claims[j]
                if claim_b.id in processed:
                    continue
                emb_b = embeddings.get(claim_b.id)
                if emb_b is None:
                    continue

                similarity = self._cosine_similarity(emb_a, emb_b)
                if similarity >= self._config.dedup_similarity_threshold:
                    # Keep the claim with higher confidence, merge sources
                    if claim_a.confidence >= claim_b.confidence:
                        self._merge_claim_into(claim_a, claim_b)
                        processed.add(claim_b.id)
                    else:
                        self._merge_claim_into(claim_b, claim_a)
                        processed.add(claim_a.id)
                    merged_count += 1

        return {"merged_count": merged_count}

    @staticmethod
    def _merge_claim_into(target: Claim, source: Claim) -> None:
        """Merge source claim into target: combine sources and supporting evidence."""
        for paper_id in source.source_paper_ids:
            if paper_id not in target.source_paper_ids:
                target.source_paper_ids.append(paper_id)
        for sid in source.supporting_claim_ids:
            if sid not in target.supporting_claim_ids:
                target.supporting_claim_ids.append(sid)
        target.confidence = max(target.confidence, source.confidence)

    def _apply_confidence_decay(self, claims: list[Claim], now: datetime) -> list[Claim]:
        """Apply temporal confidence decay and return claims that crossed threshold."""
        decayed: list[Claim] = []
        for claim in claims:
            new_conf = claim.decayed_confidence(now)
            if new_conf < self._config.min_confidence_threshold and claim.status != ClaimStatus.STALE:
                claim.status = ClaimStatus.STALE
                decayed.append(claim)
        return decayed

    def _update_hypothesis_status(
        self, hypotheses: list[Hypothesis], now: datetime
    ) -> tuple[list[Hypothesis], list[Hypothesis]]:
        """Promote confirmed hypotheses to established facts, demote refuted ones."""
        promoted: list[Hypothesis] = []
        demoted: list[Hypothesis] = []

        for h in hypotheses:
            if h.status == HypothesisStatus.CONFIRMED and h.confidence >= 0.8:
                h.status = HypothesisStatus.PROMOTED
                h.last_updated = now
                promoted.append(h)
            elif h.status == HypothesisStatus.REFUTED:
                h.last_updated = now
                demoted.append(h)

        return promoted, demoted

    def _flag_stale_hypotheses(
        self, hypotheses: list[Hypothesis], now: datetime
    ) -> list[Hypothesis]:
        """Flag hypotheses that haven't been updated within the staleness window."""
        threshold = timedelta(days=self._config.stale_hypothesis_days)
        stale: list[Hypothesis] = []
        for h in hypotheses:
            if h.status not in (HypothesisStatus.PROMOTED, HypothesisStatus.REFUTED):
                if (now - h.last_updated) > threshold:
                    h.status = HypothesisStatus.STALE
                    h.last_updated = now
                    stale.append(h)
        return stale

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class ConsolidationReport:
    """Summary of a consolidation run."""

    def __init__(
        self,
        duplicates_merged: int = 0,
        claims_decayed: int = 0,
        hypotheses_promoted: int = 0,
        hypotheses_demoted: int = 0,
        hypotheses_flagged_stale: int = 0,
        below_confidence_threshold: int = 0,
        run_at: datetime | None = None,
    ) -> None:
        self.duplicates_merged = duplicates_merged
        self.claims_decayed = claims_decayed
        self.hypotheses_promoted = hypotheses_promoted
        self.hypotheses_demoted = hypotheses_demoted
        self.hypotheses_flagged_stale = hypotheses_flagged_stale
        self.below_confidence_threshold = below_confidence_threshold
        self.run_at = run_at or datetime.now(UTC)
