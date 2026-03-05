"""Information-theoretic monitoring for hallucination detection."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from pydantic import BaseModel, Field

from auto_researcher.models import Claim
from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)


class BatchVerdict(BaseModel):
    """Result of entropy-based anomaly detection on a claim batch."""

    is_anomalous: bool = False
    entropy: float = 0.0
    entropy_z_score: float = 0.0
    kl_divergence: float = 0.0
    anomaly_type: str = Field(default="none")  # none, low_entropy, high_entropy, kl_drift
    recommendation: str = Field(default="accept")  # accept, flag, quarantine
    details: str = ""


class EntropyMonitor:
    """Monitor claim distributions for anomalous entropy shifts.

    Uses Shannon entropy and KL divergence to detect hallucination
    patterns in ingested claim batches.
    """

    def __init__(
        self,
        window_size: int = 50,
        entropy_threshold: float = 2.5,
        kl_alert_threshold: float = 0.1,
        kl_quarantine_threshold: float = 0.3,
    ) -> None:
        self.window_size = window_size
        self.entropy_threshold = entropy_threshold
        self.kl_alert_threshold = kl_alert_threshold
        self.kl_quarantine_threshold = kl_quarantine_threshold

        self._entropy_history: deque[float] = deque(maxlen=window_size)
        self._baseline_distribution: dict[str, float] = {}
        self._total_claims_seen: int = 0
        self._category_counts: dict[str, int] = {}

    def compute_entropy(self, distribution: dict[str, int]) -> float:
        """Compute Shannon entropy of a claim category distribution.

        Args:
            distribution: Mapping of category -> count.

        Returns:
            Shannon entropy in bits.
        """
        total = sum(distribution.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in distribution.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    def compute_kl_divergence(
        self, p: dict[str, float], q: dict[str, float]
    ) -> float:
        """Compute KL(P||Q) with Laplace smoothing.

        Args:
            p: The observed distribution (probability values).
            q: The reference/baseline distribution (probability values).

        Returns:
            KL divergence in bits.
        """
        all_keys = set(p.keys()) | set(q.keys())
        n = len(all_keys)
        if n == 0:
            return 0.0

        # Laplace smoothing constant
        epsilon = 1e-10

        kl = 0.0
        for key in all_keys:
            p_val = p.get(key, 0.0) + epsilon
            q_val = q.get(key, 0.0) + epsilon
            # Re-normalize isn't strictly necessary with epsilon, but keep it clean
            kl += p_val * math.log2(p_val / q_val)
        return max(kl, 0.0)

    def _distribution_from_claims(self, claims: list[Claim]) -> dict[str, int]:
        """Build a category count distribution from claims using relation values."""
        dist: dict[str, int] = {}
        for claim in claims:
            category = claim.relation.value
            dist[category] = dist.get(category, 0) + 1
        return dist

    def _to_probability(self, counts: dict[str, int]) -> dict[str, float]:
        """Convert count distribution to probability distribution."""
        total = sum(counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in counts.items()}

    def _update_baseline(self, batch_counts: dict[str, int]) -> None:
        """Update the running baseline category counts."""
        for category, count in batch_counts.items():
            self._category_counts[category] = (
                self._category_counts.get(category, 0) + count
            )
        self._total_claims_seen += sum(batch_counts.values())
        # Recompute baseline as probability distribution
        if self._total_claims_seen > 0:
            self._baseline_distribution = self._to_probability(self._category_counts)

    def record_batch(self, claims: list[Claim]) -> BatchVerdict:
        """Record a batch of claims, update history, and return verdict.

        Args:
            claims: List of claims in the ingestion batch.

        Returns:
            BatchVerdict with entropy stats and anomaly assessment.
        """
        if not claims:
            return BatchVerdict(details="Empty batch")

        batch_counts = self._distribution_from_claims(claims)
        entropy = self.compute_entropy(batch_counts)

        # Store entropy in rolling window
        self._entropy_history.append(entropy)

        # Compute KL divergence against baseline
        kl_div = 0.0
        if self._baseline_distribution:
            batch_probs = self._to_probability(batch_counts)
            kl_div = self.compute_kl_divergence(batch_probs, self._baseline_distribution)

        # Compute z-score if we have enough history
        z_score = 0.0
        if len(self._entropy_history) >= 3:
            values = list(self._entropy_history)
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance) if variance > 0 else 0.0
            z_score = (entropy - mean) / std if std > 0 else 0.0

        # Update baseline after computing divergence
        self._update_baseline(batch_counts)

        # Determine anomaly
        verdict = self._assess_anomaly(entropy, z_score, kl_div)

        logger.info(
            "batch_entropy_recorded",
            entropy=round(entropy, 4),
            z_score=round(z_score, 4),
            kl_divergence=round(kl_div, 4),
            anomaly_type=verdict.anomaly_type,
            recommendation=verdict.recommendation,
            batch_size=len(claims),
        )

        return verdict

    def _assess_anomaly(
        self, entropy: float, z_score: float, kl_div: float
    ) -> BatchVerdict:
        """Assess whether the batch is anomalous based on entropy and KL stats."""
        anomaly_type = "none"
        recommendation = "accept"
        is_anomalous = False
        details_parts: list[str] = []

        # Check entropy z-score
        if abs(z_score) > self.entropy_threshold:
            is_anomalous = True
            if z_score < -self.entropy_threshold:
                anomaly_type = "low_entropy"
                details_parts.append(
                    f"Entropy z-score {z_score:.2f} below -{self.entropy_threshold}: "
                    "claims concentrated in narrow category range"
                )
            else:
                anomaly_type = "high_entropy"
                details_parts.append(
                    f"Entropy z-score {z_score:.2f} above {self.entropy_threshold}: "
                    "unusual spread across categories"
                )
            recommendation = "flag"

        # Check KL divergence
        if kl_div > self.kl_quarantine_threshold:
            is_anomalous = True
            anomaly_type = "kl_drift"
            recommendation = "quarantine"
            details_parts.append(
                f"KL divergence {kl_div:.4f} exceeds quarantine threshold "
                f"{self.kl_quarantine_threshold}"
            )
        elif kl_div > self.kl_alert_threshold:
            is_anomalous = True
            if anomaly_type == "none":
                anomaly_type = "kl_drift"
            if recommendation == "accept":
                recommendation = "flag"
            details_parts.append(
                f"KL divergence {kl_div:.4f} exceeds alert threshold "
                f"{self.kl_alert_threshold}"
            )

        details = "; ".join(details_parts) if details_parts else "No anomalies detected"

        return BatchVerdict(
            is_anomalous=is_anomalous,
            entropy=entropy,
            entropy_z_score=z_score,
            kl_divergence=kl_div,
            anomaly_type=anomaly_type,
            recommendation=recommendation,
            details=details,
        )

    def check_batch_anomaly(self, claims: list[Claim]) -> BatchVerdict:
        """Check if a batch has entropy/KL anomalies without updating baseline.

        This is a read-only check — it does not modify internal state.

        Args:
            claims: List of claims to check.

        Returns:
            BatchVerdict with anomaly assessment.
        """
        if not claims:
            return BatchVerdict(details="Empty batch")

        batch_counts = self._distribution_from_claims(claims)
        entropy = self.compute_entropy(batch_counts)

        # Compute z-score against current history
        z_score = 0.0
        if len(self._entropy_history) >= 3:
            values = list(self._entropy_history)
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance) if variance > 0 else 0.0
            z_score = (entropy - mean) / std if std > 0 else 0.0

        # Compute KL divergence against baseline
        kl_div = 0.0
        if self._baseline_distribution:
            batch_probs = self._to_probability(batch_counts)
            kl_div = self.compute_kl_divergence(batch_probs, self._baseline_distribution)

        return self._assess_anomaly(entropy, z_score, kl_div)
