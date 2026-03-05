"""Provenance tracking for claims in the knowledge graph."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from auto_researcher.utils.logging import get_logger
from auto_researcher.verification.claim_verifier import VerificationStatus

logger = get_logger(__name__)


class ProvenanceRecord(BaseModel):
    """Full provenance record for a single claim."""

    claim_id: str
    source_paper_id: str
    extraction_method: str = "llm"
    extractor_model: str = ""
    extraction_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    verification_history: list[dict[str, Any]] = Field(default_factory=list)
    downstream_dependents: list[str] = Field(default_factory=list)
    upstream_premises: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ProvenanceTracker:
    """Track provenance, dependencies, and verification history for claims.

    Supports forward/backward tracing through the claim dependency graph
    and cascade quarantine operations.
    """

    def __init__(self) -> None:
        self._records: dict[str, ProvenanceRecord] = {}

    def record(
        self,
        claim_id: str,
        source_paper_id: str,
        extraction_method: str = "llm",
        extractor_model: str = "",
    ) -> ProvenanceRecord:
        """Create a provenance record for a newly extracted claim.

        Args:
            claim_id: Unique identifier for the claim.
            source_paper_id: ID of the paper the claim was extracted from.
            extraction_method: Method used for extraction (e.g., "llm", "rule").
            extractor_model: Model identifier used for extraction.

        Returns:
            The created ProvenanceRecord.
        """
        rec = ProvenanceRecord(
            claim_id=claim_id,
            source_paper_id=source_paper_id,
            extraction_method=extraction_method,
            extractor_model=extractor_model,
        )
        self._records[claim_id] = rec

        logger.info(
            "provenance_recorded",
            claim_id=claim_id,
            source_paper_id=source_paper_id,
            extraction_method=extraction_method,
        )

        return rec

    def add_dependency(self, dependent_id: str, premise_id: str) -> None:
        """Link a derived claim to one of its premises.

        Args:
            dependent_id: The claim that depends on the premise.
            premise_id: The premise claim.
        """
        dep_rec = self._records.get(dependent_id)
        prem_rec = self._records.get(premise_id)

        if dep_rec is None:
            logger.warning("dependency_missing_dependent", dependent_id=dependent_id)
            return
        if prem_rec is None:
            logger.warning("dependency_missing_premise", premise_id=premise_id)
            return

        if premise_id not in dep_rec.upstream_premises:
            dep_rec.upstream_premises.append(premise_id)
        if dependent_id not in prem_rec.downstream_dependents:
            prem_rec.downstream_dependents.append(dependent_id)

        logger.debug(
            "dependency_added",
            dependent_id=dependent_id,
            premise_id=premise_id,
        )

    def get_record(self, claim_id: str) -> ProvenanceRecord | None:
        """Retrieve the provenance record for a claim.

        Args:
            claim_id: The claim identifier.

        Returns:
            The ProvenanceRecord, or None if not found.
        """
        return self._records.get(claim_id)

    def get_downstream(self, claim_id: str) -> list[str]:
        """Get all transitive downstream dependents of a claim (recursive).

        Args:
            claim_id: The root claim identifier.

        Returns:
            List of all downstream dependent claim IDs.
        """
        visited: set[str] = set()
        result: list[str] = []
        stack = [claim_id]

        while stack:
            current = stack.pop()
            rec = self._records.get(current)
            if rec is None:
                continue
            for dep_id in rec.downstream_dependents:
                if dep_id not in visited:
                    visited.add(dep_id)
                    result.append(dep_id)
                    stack.append(dep_id)

        return result

    def cascade_impact(self, claim_id: str) -> int:
        """Count all transitive downstream dependents of a claim.

        Args:
            claim_id: The root claim identifier.

        Returns:
            Number of downstream dependents (transitive).
        """
        return len(self.get_downstream(claim_id))

    def quarantine(self, claim_id: str) -> list[str]:
        """Quarantine a claim and all its downstream dependents.

        Args:
            claim_id: The claim to quarantine.

        Returns:
            List of all quarantined claim IDs (including the root).
        """
        quarantined: list[str] = []

        # Quarantine the root claim
        root_rec = self._records.get(claim_id)
        if root_rec is not None:
            root_rec.verification_status = VerificationStatus.QUARANTINED
            root_rec.verification_history.append(
                {
                    "status": VerificationStatus.QUARANTINED.value,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "reason": "direct_quarantine",
                }
            )
            quarantined.append(claim_id)

        # Quarantine all downstream
        downstream = self.get_downstream(claim_id)
        for dep_id in downstream:
            rec = self._records.get(dep_id)
            if rec is not None:
                rec.verification_status = VerificationStatus.QUARANTINED
                rec.verification_history.append(
                    {
                        "status": VerificationStatus.QUARANTINED.value,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "reason": f"cascade_from_{claim_id}",
                    }
                )
                quarantined.append(dep_id)

        logger.warning(
            "claims_quarantined",
            root_claim_id=claim_id,
            total_quarantined=len(quarantined),
        )

        return quarantined

    def update_status(
        self,
        claim_id: str,
        status: VerificationStatus,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Update the verification status of a claim.

        Args:
            claim_id: The claim identifier.
            status: The new verification status.
            details: Optional details about the status change.
        """
        rec = self._records.get(claim_id)
        if rec is None:
            logger.warning("update_status_missing_record", claim_id=claim_id)
            return

        rec.verification_status = status
        history_entry: dict[str, Any] = {
            "status": status.value,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if details:
            history_entry["details"] = details
        rec.verification_history.append(history_entry)

        logger.info(
            "status_updated",
            claim_id=claim_id,
            new_status=status.value,
        )
