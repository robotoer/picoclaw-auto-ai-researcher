"""Multi-layer claim verification pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from auto_researcher.models import Claim, ClaimRelation
from auto_researcher.utils.logging import get_logger
from auto_researcher.verification.confidence_propagation import ConfidencePropagator
from auto_researcher.verification.entropy_monitor import BatchVerdict, EntropyMonitor

logger = get_logger(__name__)


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    PROVISIONAL = "provisional"
    VERIFIED = "verified"
    SUSPICIOUS = "suspicious"
    QUARANTINED = "quarantined"


class VerificationResult(BaseModel):
    """Result of verifying a single claim."""

    claim_id: str
    status: VerificationStatus
    checks_passed: list[str] = Field(default_factory=list)
    checks_failed: list[str] = Field(default_factory=list)
    confidence_adjustment: float = 0.0  # 0 = no change, negative = reduce
    details: dict[str, Any] = Field(default_factory=dict)
    verified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ClaimVerifier:
    """Multi-layer verification pipeline for extracted claims.

    Layer 1: Structural sanity checks on individual claims.
    Layer 2: Batch-level entropy anomaly detection.
    Layer 3: Usage-level confidence and corroboration checks.
    """

    def __init__(
        self,
        entropy_monitor: EntropyMonitor,
        confidence_propagator: ConfidencePropagator,
    ) -> None:
        self.entropy_monitor = entropy_monitor
        self.confidence_propagator = confidence_propagator

    def verify_extraction(
        self, claim: Claim, source_text: str = ""
    ) -> VerificationResult:
        """Layer 1: Structural sanity checks on a single extracted claim.

        Checks:
        - Non-empty entity_1 and entity_2
        - Valid relation (member of ClaimRelation)
        - Confidence in [0, 1]

        Args:
            claim: The claim to verify.
            source_text: Optional source text for future cross-reference checks.

        Returns:
            VerificationResult with passed/failed checks.
        """
        passed: list[str] = []
        failed: list[str] = []
        details: dict[str, Any] = {}

        # Check non-empty entities
        if claim.entity_1 and claim.entity_1.strip():
            passed.append("entity_1_non_empty")
        else:
            failed.append("entity_1_non_empty")
            details["entity_1"] = "Entity 1 is empty or whitespace"

        if claim.entity_2 and claim.entity_2.strip():
            passed.append("entity_2_non_empty")
        else:
            failed.append("entity_2_non_empty")
            details["entity_2"] = "Entity 2 is empty or whitespace"

        # Check valid relation
        try:
            ClaimRelation(claim.relation)
            passed.append("valid_relation")
        except ValueError:
            failed.append("valid_relation")
            details["relation"] = f"Invalid relation: {claim.relation}"

        # Check confidence in range
        if 0.0 <= claim.confidence <= 1.0:
            passed.append("confidence_in_range")
        else:
            failed.append("confidence_in_range")
            details["confidence"] = f"Confidence {claim.confidence} out of [0, 1]"

        # Determine status
        if failed:
            status = VerificationStatus.SUSPICIOUS
            confidence_adj = -0.2 * len(failed)
        else:
            status = VerificationStatus.PROVISIONAL
            confidence_adj = 0.0

        logger.info(
            "extraction_verified",
            claim_id=claim.id,
            status=status.value,
            checks_passed=len(passed),
            checks_failed=len(failed),
        )

        return VerificationResult(
            claim_id=claim.id,
            status=status,
            checks_passed=passed,
            checks_failed=failed,
            confidence_adjustment=confidence_adj,
            details=details,
        )

    def verify_batch(
        self, claims: list[Claim]
    ) -> tuple[list[VerificationResult], BatchVerdict]:
        """Layer 2: Batch-level verification with entropy monitoring.

        Runs individual extraction checks on each claim, then checks the
        batch for entropy/KL anomalies.

        Args:
            claims: List of claims in the batch.

        Returns:
            Tuple of (individual results, batch verdict).
        """
        # Run Layer 1 on each claim
        results = [self.verify_extraction(claim) for claim in claims]

        # Run entropy monitoring
        batch_verdict = self.entropy_monitor.record_batch(claims)

        # If batch is anomalous, adjust individual results
        if batch_verdict.recommendation == "quarantine":
            for result in results:
                result.status = VerificationStatus.QUARANTINED
                result.confidence_adjustment = min(result.confidence_adjustment, -0.5)
                result.details["batch_anomaly"] = batch_verdict.anomaly_type
        elif batch_verdict.recommendation == "flag":
            for result in results:
                if result.status == VerificationStatus.PROVISIONAL:
                    result.status = VerificationStatus.SUSPICIOUS
                result.confidence_adjustment = min(result.confidence_adjustment, -0.1)
                result.details["batch_anomaly"] = batch_verdict.anomaly_type

        logger.info(
            "batch_verified",
            batch_size=len(claims),
            anomalous=batch_verdict.is_anomalous,
            recommendation=batch_verdict.recommendation,
        )

        return results, batch_verdict

    def verify_for_use(
        self,
        claim: Claim,
        usage: str,
        supporting_claims: list[Claim] | None = None,
    ) -> VerificationResult:
        """Layer 3: Verify a claim meets requirements for a specific usage level.

        Checks:
        - Confidence threshold for the usage level
        - Corroboration count (supporting claims)

        Args:
            claim: The claim to verify for use.
            usage: Usage level ("storage", "hypothesis", "experiment", "publication").
            supporting_claims: Optional list of corroborating claims.

        Returns:
            VerificationResult indicating whether the claim is usable.
        """
        passed: list[str] = []
        failed: list[str] = []
        details: dict[str, Any] = {"usage_level": usage}

        # Check confidence threshold
        meets_threshold = self.confidence_propagator.check_usage_threshold(
            claim.confidence, usage
        )
        if meets_threshold:
            passed.append(f"confidence_threshold_{usage}")
        else:
            failed.append(f"confidence_threshold_{usage}")
            details["confidence"] = (
                f"Confidence {claim.confidence:.3f} below threshold for {usage}"
            )

        # Check corroboration for higher usage levels
        corroboration_requirements: dict[str, int] = {
            "storage": 0,
            "hypothesis": 0,
            "experiment": 1,
            "publication": 2,
        }
        required_support = corroboration_requirements.get(usage, 0)
        actual_support = len(supporting_claims) if supporting_claims else 0

        if actual_support >= required_support:
            passed.append("corroboration_count")
        else:
            failed.append("corroboration_count")
            details["corroboration"] = (
                f"Has {actual_support} supporting claims, needs {required_support} "
                f"for {usage}"
            )

        # Determine status
        if failed:
            status = VerificationStatus.SUSPICIOUS
            confidence_adj = -0.1
        else:
            status = VerificationStatus.VERIFIED
            confidence_adj = 0.0

        logger.info(
            "usage_verified",
            claim_id=claim.id,
            usage=usage,
            status=status.value,
            meets_threshold=meets_threshold,
            support_count=actual_support,
        )

        return VerificationResult(
            claim_id=claim.id,
            status=status,
            checks_passed=passed,
            checks_failed=failed,
            confidence_adjustment=confidence_adj,
            details=details,
        )
