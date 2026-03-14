"""Knowledge graph update logic for integrating extracted claims."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from auto_researcher.models import (
    Claim,
    ClaimRelation,
    ClaimStatus,
    Paper,
)
from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)

# Relations that are semantically opposite
_CONTRADICTING_RELATIONS: dict[ClaimRelation, set[ClaimRelation]] = {
    ClaimRelation.OUTPERFORMS: {ClaimRelation.CONTRADICTS},
    ClaimRelation.SUPPORTS: {ClaimRelation.REFUTES, ClaimRelation.CONTRADICTS},
    ClaimRelation.REFUTES: {ClaimRelation.SUPPORTS},
    ClaimRelation.CONTRADICTS: {ClaimRelation.SUPPORTS, ClaimRelation.IS_EQUIVALENT_TO},
    ClaimRelation.IS_EQUIVALENT_TO: {ClaimRelation.CONTRADICTS},
}


class ConflictReport:
    """A detected conflict between claims in the knowledge graph."""

    def __init__(
        self,
        new_claim: Claim,
        existing_claim: Claim,
        conflict_type: str,
    ) -> None:
        self.new_claim = new_claim
        self.existing_claim = existing_claim
        self.conflict_type = conflict_type
        self.detected_at = datetime.now(UTC)

    def __repr__(self) -> str:
        return (
            f"ConflictReport({self.conflict_type}: "
            f"'{self.new_claim.entity_1} {self.new_claim.relation.value} {self.new_claim.entity_2}' "
            f"vs '{self.existing_claim.entity_1} {self.existing_claim.relation.value} {self.existing_claim.entity_2}')"
        )


class KGUpdater:
    """Updates the knowledge graph with extracted claims.

    Accepts a KnowledgeGraph instance (from infrastructure layer) and manages
    the logic of adding claims, detecting contradictions, and updating confidence.
    """

    def __init__(self, knowledge_graph: Any) -> None:
        """Initialize with a KnowledgeGraph instance.

        Args:
            knowledge_graph: An instance from auto_researcher.infrastructure.KnowledgeGraph.
                             Must support add_claim, get_claims_for_entity, update_claim methods.
        """
        self._kg = knowledge_graph

    async def update_from_paper(
        self, paper: Paper, claims: list[Claim]
    ) -> tuple[list[Claim], list[ConflictReport]]:
        """Integrate claims from a paper into the knowledge graph.

        Returns:
            Tuple of (added_claims, conflicts_detected).
        """
        added: list[Claim] = []
        conflicts: list[ConflictReport] = []

        for claim in claims:
            # Check for contradictions with existing claims
            claim_conflicts = await self._detect_contradictions(claim)
            conflicts.extend(claim_conflicts)

            if claim_conflicts:
                claim = claim.model_copy(
                    update={
                        "contradicting_claim_ids": [
                            c.existing_claim.id for c in claim_conflicts
                        ]
                    }
                )
                # Flag existing contradicting claims
                for conflict in claim_conflicts:
                    await self._flag_contradiction(conflict)

            # Check for corroborating evidence
            corroboration = await self._find_corroboration(claim)
            if corroboration:
                claim = claim.model_copy(
                    update={
                        "supporting_claim_ids": [c.id for c in corroboration],
                        "confidence": self._boost_confidence(
                            claim.confidence, len(corroboration)
                        ),
                    }
                )
                # Also update the existing claims' confidence
                for existing in corroboration:
                    boosted = self._boost_confidence(
                        existing.confidence, 1
                    )
                    await self._kg.update_claim(
                        existing.id,
                        {
                            "confidence": boosted,
                            "supporting_claim_ids": existing.supporting_claim_ids
                            + [claim.id],
                        },
                    )

            # Add nodes for entities if they don't exist
            await self._ensure_entity_node(claim.entity_1)
            await self._ensure_entity_node(claim.entity_2)

            # Add the claim as an edge
            await self._kg.add_claim(claim)
            added.append(claim)

        logger.info(
            "kg_updated",
            paper_id=paper.arxiv_id,
            claims_added=len(added),
            conflicts_found=len(conflicts),
        )
        return added, conflicts

    async def _detect_contradictions(self, claim: Claim) -> list[ConflictReport]:
        """Detect contradictions between a new claim and existing claims."""
        conflicts: list[ConflictReport] = []

        existing_claims = await self._kg.get_claims_for_entity(claim.entity_1)

        for existing in existing_claims:
            # Same entities, contradicting relations
            if (
                existing.entity_2 == claim.entity_2
                and self._are_contradicting(claim.relation, existing.relation)
            ):
                conflicts.append(
                    ConflictReport(claim, existing, "contradicting_relation")
                )

            # Reversed direction with same relation (A outperforms B vs B outperforms A)
            if (
                existing.entity_1 == claim.entity_2
                and existing.entity_2 == claim.entity_1
                and existing.relation == claim.relation
                and claim.relation
                in {ClaimRelation.OUTPERFORMS, ClaimRelation.IMPROVES}
            ):
                conflicts.append(
                    ConflictReport(claim, existing, "reversed_direction")
                )

        return conflicts

    def _are_contradicting(
        self, rel_a: ClaimRelation, rel_b: ClaimRelation
    ) -> bool:
        """Check if two relations are semantically contradictory."""
        contradicts_set = _CONTRADICTING_RELATIONS.get(rel_a, set())
        return rel_b in contradicts_set

    async def _flag_contradiction(self, conflict: ConflictReport) -> None:
        """Flag an existing claim as contradicted."""
        existing = conflict.existing_claim
        update: dict[str, Any] = {
            "contradicting_claim_ids": existing.contradicting_claim_ids
            + [conflict.new_claim.id],
        }
        # Only change status if the new claim has higher confidence
        if conflict.new_claim.confidence > existing.confidence:
            update["status"] = ClaimStatus.CONTRADICTED.value
        await self._kg.update_claim(existing.id, update)

    async def _find_corroboration(self, claim: Claim) -> list[Claim]:
        """Find existing claims that corroborate the new claim."""
        existing_claims = await self._kg.get_claims_for_entity(claim.entity_1)
        corroborating: list[Claim] = []
        for existing in existing_claims:
            if (
                existing.entity_2 == claim.entity_2
                and existing.relation == claim.relation
                and existing.id != claim.id
            ):
                corroborating.append(existing)
        return corroborating

    def _boost_confidence(self, base: float, n_corroborations: int) -> float:
        """Boost confidence based on corroborating evidence.

        Uses a diminishing-returns formula: each additional source adds less.
        """
        boost = sum(0.05 / (i + 1) for i in range(n_corroborations))
        return min(1.0, base + boost)

    async def _ensure_entity_node(self, entity_name: str) -> None:
        """Ensure an entity node exists in the knowledge graph."""
        try:
            await self._kg.ensure_node(entity_name)
        except Exception:
            logger.debug("ensure_node_noop", entity=entity_name)
