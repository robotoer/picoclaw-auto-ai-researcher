"""Confidence propagation framework for derived claims."""

from __future__ import annotations

from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)


class ConfidencePropagator:
    """Propagate and attenuate confidence scores through claim dependency chains.

    Implements the five confidence propagation rules:
    1. Conjunction decay: derived confidence <= min(premises) with penalty for count
    2. Chain attenuation: confidence decays with derivation depth
    3. Corroboration boost: independent evidence increases confidence
    4. Hard ceiling: no claim can exceed HARD_CEILING
    5. Usage thresholds: minimum confidence required for each usage level
    """

    HARD_CEILING: float = 0.95
    CHAIN_ATTENUATION: float = 0.95  # gamma
    CONJUNCTION_LAMBDA: float = 0.1
    MIN_CONFIDENCE_FOR_HYPOTHESIS: float = 0.50
    MIN_CONFIDENCE_FOR_EXPERIMENT: float = 0.70
    MIN_CONFIDENCE_FOR_PUBLICATION: float = 0.85

    def conjunction_decay(self, premise_confidences: list[float]) -> float:
        """Rule 1: Conjunction decay.

        Derived confidence is min(premises) * 1/(1 + lambda*(n-1)).
        More premises means more decay.

        Args:
            premise_confidences: Confidence values of all premises.

        Returns:
            Decayed confidence value.
        """
        if not premise_confidences:
            return 0.0
        n = len(premise_confidences)
        min_conf = min(premise_confidences)
        decay_factor = 1.0 / (1.0 + self.CONJUNCTION_LAMBDA * (n - 1))
        return min_conf * decay_factor

    def chain_attenuation(self, source_confidence: float, depth: int) -> float:
        """Rule 2: Chain attenuation.

        Confidence decays exponentially with derivation depth.

        Args:
            source_confidence: Confidence at the source.
            depth: Number of derivation steps from source.

        Returns:
            Attenuated confidence value.
        """
        return source_confidence * (self.CHAIN_ATTENUATION ** depth)

    def corroboration_boost(self, evidence: list[tuple[float, float]]) -> float:
        """Rule 3: Corroboration boost from independent evidence.

        Combined confidence: 1 - prod(1 - c_i * ind_i).

        Args:
            evidence: List of (confidence, independence_score) tuples.

        Returns:
            Boosted confidence value.
        """
        if not evidence:
            return 0.0
        product = 1.0
        for confidence, independence in evidence:
            product *= 1.0 - confidence * independence
        return min(1.0 - product, self.HARD_CEILING)

    def compute_derived_confidence(
        self, premise_confidences: list[float], depth: int = 0
    ) -> float:
        """Combine conjunction decay, chain attenuation, and hard ceiling.

        Args:
            premise_confidences: Confidence values of all premises.
            depth: Derivation depth from original sources.

        Returns:
            Final derived confidence, capped at HARD_CEILING.
        """
        if not premise_confidences:
            return 0.0
        # Rule 1: conjunction decay
        conf = self.conjunction_decay(premise_confidences)
        # Rule 2: chain attenuation
        conf = self.chain_attenuation(conf, depth)
        # Rule 4: hard ceiling
        return min(conf, self.HARD_CEILING)

    def check_usage_threshold(self, confidence: float, usage: str) -> bool:
        """Rule 5: Check if confidence meets the threshold for the given usage level.

        Args:
            confidence: The confidence value to check.
            usage: One of "storage", "hypothesis", "experiment", "publication".

        Returns:
            True if confidence meets the minimum for the usage level.
        """
        thresholds: dict[str, float] = {
            "storage": 0.0,
            "hypothesis": self.MIN_CONFIDENCE_FOR_HYPOTHESIS,
            "experiment": self.MIN_CONFIDENCE_FOR_EXPERIMENT,
            "publication": self.MIN_CONFIDENCE_FOR_PUBLICATION,
        }
        threshold = thresholds.get(usage)
        if threshold is None:
            logger.warning("unknown_usage_level", usage=usage)
            return False
        return confidence >= threshold

    def propagate(
        self, claims: dict[str, float], dependencies: dict[str, list[str]]
    ) -> dict[str, float]:
        """Topological sort and recompute all confidences through the dependency graph.

        Args:
            claims: Mapping of claim_id -> base confidence.
            dependencies: Mapping of claim_id -> list of premise claim_ids.

        Returns:
            Mapping of claim_id -> recomputed confidence.
        """
        # Build adjacency for topological sort
        in_degree: dict[str, int] = {cid: 0 for cid in claims}
        dependents: dict[str, list[str]] = {cid: [] for cid in claims}

        for cid, premises in dependencies.items():
            if cid not in in_degree:
                in_degree[cid] = 0
            for premise_id in premises:
                if premise_id in claims:
                    dependents[premise_id].append(cid)
                    in_degree[cid] = in_degree.get(cid, 0) + 1

        # Kahn's algorithm for topological sort
        queue: list[str] = [cid for cid, deg in in_degree.items() if deg == 0]
        sorted_order: list[str] = []

        while queue:
            node = queue.pop(0)
            sorted_order.append(node)
            for dependent in dependents.get(node, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Compute depths
        depths: dict[str, int] = {}
        for cid in sorted_order:
            premises = dependencies.get(cid, [])
            if not premises:
                depths[cid] = 0
            else:
                depths[cid] = max(depths.get(p, 0) for p in premises if p in claims) + 1

        # Propagate confidences in topological order
        result: dict[str, float] = {}
        for cid in sorted_order:
            premises = dependencies.get(cid, [])
            valid_premises = [p for p in premises if p in result]

            if not valid_premises:
                # Root claim: apply ceiling only
                result[cid] = min(claims[cid], self.HARD_CEILING)
            else:
                premise_confs = [result[p] for p in valid_premises]
                result[cid] = self.compute_derived_confidence(
                    premise_confs, depth=depths[cid]
                )

        # Handle any claims not reached by topological sort (cycles or disconnected)
        for cid in claims:
            if cid not in result:
                logger.warning("claim_not_in_topo_sort", claim_id=cid)
                result[cid] = min(claims[cid], self.HARD_CEILING)

        return result
