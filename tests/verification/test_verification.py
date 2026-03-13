"""Comprehensive tests for the verification package.

Tests cover entropy monitoring, confidence propagation,
claim verification, and provenance tracking.
"""

import math

import pytest

from auto_researcher.models.claim import Claim, ClaimRelation
from auto_researcher.verification.entropy_monitor import EntropyMonitor, BatchVerdict
from auto_researcher.verification.confidence_propagation import ConfidencePropagator
from auto_researcher.verification.claim_verifier import (
    ClaimVerifier,
    VerificationResult,
    VerificationStatus,
)
from auto_researcher.verification.provenance_tracker import (
    ProvenanceTracker,
    ProvenanceRecord,
)


# ── Helpers ──────────────────────────────────────────────────────────


def make_claim(
    entity_1="ModelA",
    relation=ClaimRelation.OUTPERFORMS,
    entity_2="ModelB",
    confidence=0.8,
    claim_id="claim-1",
):
    return Claim(
        id=claim_id,
        entity_1=entity_1,
        relation=relation,
        entity_2=entity_2,
        confidence=confidence,
        source_paper_ids=["paper-1"],
    )


def make_claims_uniform(n=50):
    """Create a batch of claims with roughly uniform relation distribution."""
    relations = list(ClaimRelation)
    claims = []
    for i in range(n):
        rel = relations[i % len(relations)]
        claims.append(
            make_claim(
                entity_1=f"Entity{i}",
                entity_2=f"Entity{i+100}",
                relation=rel,
                confidence=0.7,
                claim_id=f"claim-uniform-{i}",
            )
        )
    return claims


def make_claims_single_relation(n=50, relation=ClaimRelation.OUTPERFORMS):
    """Create a batch of claims that all share the same relation."""
    return [
        make_claim(
            entity_1=f"Entity{i}",
            entity_2=f"Entity{i+100}",
            relation=relation,
            confidence=0.7,
            claim_id=f"claim-mono-{i}",
        )
        for i in range(n)
    ]


# ══════════════════════════════════════════════════════════════════════
# EntropyMonitor
# ══════════════════════════════════════════════════════════════════════


class TestEntropyMonitorPureFunctions:
    """Tests for the entropy computations."""

    def test_entropy_uniform_distribution(self):
        """Uniform distribution should yield maximum entropy (log2)."""
        monitor = EntropyMonitor()
        n = 5
        # compute_entropy takes dict[str, int] (counts)
        dist = {f"cat_{i}": 10 for i in range(n)}
        entropy = monitor.compute_entropy(dist)
        expected = math.log2(n)
        assert entropy == pytest.approx(expected, abs=1e-6)

    def test_entropy_single_category(self):
        """A single-category distribution has entropy 0."""
        monitor = EntropyMonitor()
        dist = {"only": 10}
        entropy = monitor.compute_entropy(dist)
        assert entropy == pytest.approx(0.0, abs=1e-9)

    def test_entropy_two_equal_categories(self):
        monitor = EntropyMonitor()
        dist = {"a": 5, "b": 5}
        entropy = monitor.compute_entropy(dist)
        assert entropy == pytest.approx(math.log2(2), abs=1e-6)

    def test_entropy_skewed_distribution(self):
        """A heavily skewed distribution should have low (but positive) entropy."""
        monitor = EntropyMonitor()
        dist = {"dominant": 99, "rare": 1}
        entropy = monitor.compute_entropy(dist)
        assert 0.0 < entropy < math.log2(2)

    def test_entropy_empty_distribution(self):
        monitor = EntropyMonitor()
        entropy = monitor.compute_entropy({})
        assert entropy == 0.0

    def test_kl_divergence_identical(self):
        """KL divergence of identical distributions is ~0."""
        monitor = EntropyMonitor()
        p = {"a": 0.5, "b": 0.3, "c": 0.2}
        q = {"a": 0.5, "b": 0.3, "c": 0.2}
        kl = monitor.compute_kl_divergence(p, q)
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_kl_divergence_very_different(self):
        """Very different distributions produce a large positive KL divergence."""
        monitor = EntropyMonitor()
        p = {"a": 0.99, "b": 0.005, "c": 0.005}
        q = {"a": 0.01, "b": 0.495, "c": 0.495}
        kl = monitor.compute_kl_divergence(p, q)
        assert kl > 1.0

    def test_kl_divergence_is_non_negative(self):
        monitor = EntropyMonitor()
        p = {"a": 0.7, "b": 0.3}
        q = {"a": 0.4, "b": 0.6}
        kl = monitor.compute_kl_divergence(p, q)
        assert kl >= 0.0

    def test_laplace_smoothing_prevents_division_by_zero(self):
        """Smoothing ensures we never compute log(0) even when a
        category is absent from one distribution."""
        monitor = EntropyMonitor()
        p = {"a": 1.0}
        q = {"a": 0.5, "b": 0.5}
        kl = monitor.compute_kl_divergence(p, q)
        assert math.isfinite(kl)


class TestEntropyMonitorBatches:
    """Tests for recording batches and detecting anomalies."""

    def test_record_batch_normal(self):
        """A single normal batch should not flag any anomaly."""
        monitor = EntropyMonitor()
        claims = make_claims_uniform(50)
        verdict = monitor.record_batch(claims)
        assert isinstance(verdict, BatchVerdict)
        # First batch has no history to compare against, shouldn't be anomalous
        assert verdict.is_anomalous is False

    def test_batch_verdict_fields_populated(self):
        """All BatchVerdict fields should be filled after recording."""
        monitor = EntropyMonitor()
        claims = make_claims_uniform(30)
        verdict = monitor.record_batch(claims)
        assert verdict.entropy is not None
        assert isinstance(verdict.entropy, float)
        assert verdict.entropy > 0

    def test_kl_drift_detected_after_shift(self):
        """After several normal batches, a sudden shift to a single
        relation type should trigger KL drift detection."""
        monitor = EntropyMonitor()
        # Build up baseline with uniform batches
        for _ in range(5):
            monitor.record_batch(make_claims_uniform(50))
        # Introduce a drastic shift
        shifted = make_claims_single_relation(50, ClaimRelation.REFUTES)
        verdict = monitor.record_batch(shifted)
        assert verdict.is_anomalous is True

    def test_single_relation_batch_low_entropy(self):
        """A batch where all claims have the same relation should
        have very low entropy."""
        monitor = EntropyMonitor()
        # Build baseline first
        for _ in range(5):
            monitor.record_batch(make_claims_uniform(50))
        mono = make_claims_single_relation(50)
        verdict = monitor.record_batch(mono)
        assert verdict.entropy < 0.5

    def test_empty_batch_handling(self):
        """An empty batch should not crash and should return a verdict."""
        monitor = EntropyMonitor()
        verdict = monitor.record_batch([])
        assert isinstance(verdict, BatchVerdict)

    def test_no_anomaly_when_insufficient_history(self):
        """First batch should never flag anomaly even if monotone."""
        monitor = EntropyMonitor()
        mono = make_claims_single_relation(50)
        verdict = monitor.record_batch(mono)
        # Not enough history for z-score to be meaningful
        assert verdict.is_anomalous is False

    def test_gradual_shift_does_not_flag(self):
        """Very slow distribution changes should not trigger the
        anomaly detector with a relaxed KL threshold."""
        # Use a higher KL threshold so minor drift doesn't trigger
        monitor = EntropyMonitor(kl_alert_threshold=0.5, kl_quarantine_threshold=1.0)
        relations = list(ClaimRelation)
        for step in range(10):
            claims = []
            for i in range(50):
                if i < step:
                    rel = relations[0]
                else:
                    rel = relations[i % len(relations)]
                claims.append(
                    make_claim(
                        entity_1=f"E{step}_{i}",
                        entity_2=f"F{step}_{i}",
                        relation=rel,
                        claim_id=f"grad-{step}-{i}",
                    )
                )
            verdict = monitor.record_batch(claims)
        # After gentle drift across 10 batches with relaxed threshold
        assert verdict.is_anomalous is False


# ══════════════════════════════════════════════════════════════════════
# ConfidencePropagator
# ══════════════════════════════════════════════════════════════════════


class TestConjunctionDecay:
    """conjunction_decay: models AND-chain weakening."""

    def test_single_premise_no_penalty(self):
        prop = ConfidencePropagator()
        result = prop.conjunction_decay([0.9])
        # decay(1) = 1/(1+0.1*0) = 1.0 -> 0.9 * 1.0 = 0.9
        assert result == pytest.approx(0.9, abs=1e-6)

    def test_two_premises(self):
        prop = ConfidencePropagator()
        result = prop.conjunction_decay([0.9, 0.8])
        # decay(2) = 1/(1+0.1*1) = 1/1.1 ≈ 0.909
        # min=0.8, result ≈ 0.8 * 0.909 = 0.727
        expected = 0.8 * (1.0 / 1.1)
        assert result == pytest.approx(expected, abs=1e-3)

    def test_five_premises(self):
        prop = ConfidencePropagator()
        confs = [0.9, 0.85, 0.8, 0.75, 0.7]
        result = prop.conjunction_decay(confs)
        # decay(5) = 1/(1+0.1*4) = 1/1.4 ≈ 0.714
        # min=0.7, result ≈ 0.7 * 0.714 = 0.5
        expected = 0.7 * (1.0 / 1.4)
        assert result == pytest.approx(expected, abs=1e-3)

    def test_decay_monotonically_decreases_with_premises(self):
        prop = ConfidencePropagator()
        vals = [prop.conjunction_decay([0.9] * n) for n in range(1, 8)]
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1]

    def test_empty_premises_returns_zero(self):
        prop = ConfidencePropagator()
        assert prop.conjunction_decay([]) == 0.0


class TestChainAttenuation:
    """chain_attenuation: models citation-chain depth decay."""

    def test_depth_zero_unchanged(self):
        prop = ConfidencePropagator()
        result = prop.chain_attenuation(0.9, depth=0)
        assert result == pytest.approx(0.9, abs=1e-6)

    def test_depth_three(self):
        prop = ConfidencePropagator()
        result = prop.chain_attenuation(0.9, depth=3)
        # 0.9 * 0.95^3 ≈ 0.771
        expected = 0.9 * (0.95 ** 3)
        assert result == pytest.approx(expected, abs=1e-3)

    def test_depth_ten(self):
        prop = ConfidencePropagator()
        result = prop.chain_attenuation(0.9, depth=10)
        # 0.9 * 0.95^10 ≈ 0.540
        expected = 0.9 * (0.95 ** 10)
        assert result == pytest.approx(expected, abs=1e-3)

    def test_attenuation_is_monotone_in_depth(self):
        prop = ConfidencePropagator()
        prev = 0.9
        for d in range(1, 15):
            cur = prop.chain_attenuation(0.9, depth=d)
            assert cur <= prev
            prev = cur


class TestCorroborationBoost:
    """corroboration_boost: Bayesian-style multi-source agreement."""

    def test_two_independent_sources(self):
        prop = ConfidencePropagator()
        # evidence is list of (confidence, independence_score) tuples
        result = prop.corroboration_boost([(0.7, 1.0), (0.7, 1.0)])
        # 1 - (1-0.7)^2 = 1 - 0.09 = 0.91
        assert result == pytest.approx(0.91, abs=0.02)

    def test_three_independent_sources(self):
        prop = ConfidencePropagator()
        result = prop.corroboration_boost([(0.6, 1.0), (0.6, 1.0), (0.6, 1.0)])
        # 1 - (1-0.6)^3 = 1 - 0.064 = 0.936
        assert result == pytest.approx(0.936, abs=0.02)

    def test_zero_independence_no_boost(self):
        """With independence=0 the boost should be minimal."""
        prop = ConfidencePropagator()
        result = prop.corroboration_boost([(0.7, 0.0), (0.7, 0.0)])
        # 1 - (1-0)^2 = 0
        assert result == pytest.approx(0.0, abs=0.05)

    def test_single_source(self):
        prop = ConfidencePropagator()
        result = prop.corroboration_boost([(0.8, 1.0)])
        # 1 - (1-0.8) = 0.8
        assert result == pytest.approx(0.8, abs=0.02)

    def test_empty_evidence_returns_zero(self):
        prop = ConfidencePropagator()
        assert prop.corroboration_boost([]) == 0.0


class TestDerivedConfidence:
    """compute_derived_confidence: combines all propagation rules."""

    def test_hard_ceiling(self):
        """No derived confidence may exceed 0.95."""
        prop = ConfidencePropagator()
        result = prop.compute_derived_confidence([0.99, 0.99], depth=0)
        assert result <= 0.95

    def test_with_depth(self):
        prop = ConfidencePropagator()
        result = prop.compute_derived_confidence([0.9], depth=3)
        # conjunction_decay([0.9]) = 0.9
        # chain_attenuation(0.9, 3) = 0.9 * 0.95^3 ≈ 0.771
        expected = 0.9 * (0.95 ** 3)
        assert result == pytest.approx(expected, abs=1e-3)

    def test_empty_returns_zero(self):
        prop = ConfidencePropagator()
        assert prop.compute_derived_confidence([]) == 0.0


class TestUsageThresholds:
    """check_usage_threshold: gate confidence for different use-cases."""

    @pytest.mark.parametrize(
        "usage, threshold",
        [
            ("storage", 0.0),
            ("hypothesis", 0.50),
            ("experiment", 0.70),
            ("publication", 0.85),
        ],
    )
    def test_threshold_just_above_passes(self, usage, threshold):
        prop = ConfidencePropagator()
        assert prop.check_usage_threshold(threshold + 0.01, usage) is True

    @pytest.mark.parametrize(
        "usage, threshold",
        [
            ("hypothesis", 0.50),
            ("experiment", 0.70),
            ("publication", 0.85),
        ],
    )
    def test_threshold_just_below_fails(self, usage, threshold):
        prop = ConfidencePropagator()
        assert prop.check_usage_threshold(threshold - 0.01, usage) is False

    def test_unknown_usage_returns_false(self):
        prop = ConfidencePropagator()
        assert prop.check_usage_threshold(0.99, "unknown_level") is False


class TestConfidencePropagatorDAG:
    """propagate: recompute confidence through a DAG of claims."""

    def test_propagate_updates_downstream(self):
        """Changing a root claim's confidence should lower
        all downstream derived claims after propagation."""
        prop = ConfidencePropagator()

        claims = {"root": 0.9, "child": 0.85, "grandchild": 0.80}
        dependencies = {
            "root": [],
            "child": ["root"],
            "grandchild": ["child"],
        }

        updated = prop.propagate(claims, dependencies)
        # Grandchild should be attenuated more than child
        assert updated["grandchild"] <= updated["child"]
        # All should be at most 0.95 (ceiling)
        for conf in updated.values():
            assert conf <= 0.95

    def test_propagate_root_unchanged(self):
        prop = ConfidencePropagator()
        claims = {"a": 0.8}
        deps = {"a": []}
        result = prop.propagate(claims, deps)
        assert result["a"] == pytest.approx(0.8, abs=1e-6)

    def test_propagate_long_chain(self):
        prop = ConfidencePropagator()
        n = 10
        claims = {f"c{i}": 0.9 for i in range(n)}
        deps = {f"c{i}": [f"c{i-1}"] if i > 0 else [] for i in range(n)}
        result = prop.propagate(claims, deps)
        # Each step attenuates, so final should be lower than first
        assert result[f"c{n-1}"] < result["c0"]


# ══════════════════════════════════════════════════════════════════════
# ClaimVerifier
# ══════════════════════════════════════════════════════════════════════


class TestClaimVerifierExtraction:
    """verify_extraction: structural validity checks on claims."""

    def _make_verifier(self):
        return ClaimVerifier(
            entropy_monitor=EntropyMonitor(),
            confidence_propagator=ConfidencePropagator(),
        )

    def test_valid_claim_passes(self):
        verifier = self._make_verifier()
        claim = make_claim()
        result = verifier.verify_extraction(claim)
        assert isinstance(result, VerificationResult)
        assert result.status == VerificationStatus.PROVISIONAL

    def test_empty_entity_1_fails(self):
        verifier = self._make_verifier()
        claim = make_claim(entity_1="")
        result = verifier.verify_extraction(claim)
        assert result.status == VerificationStatus.SUSPICIOUS

    def test_empty_entity_2_fails(self):
        verifier = self._make_verifier()
        claim = make_claim(entity_2="")
        result = verifier.verify_extraction(claim)
        assert result.status == VerificationStatus.SUSPICIOUS

    def test_whitespace_only_entity_fails(self):
        verifier = self._make_verifier()
        claim = make_claim(entity_1="   ")
        result = verifier.verify_extraction(claim)
        assert result.status == VerificationStatus.SUSPICIOUS

    def test_checks_passed_list_populated(self):
        verifier = self._make_verifier()
        claim = make_claim()
        result = verifier.verify_extraction(claim)
        assert len(result.checks_passed) > 0


class TestClaimVerifierBatch:
    """verify_batch: batch processing that returns individual results and BatchVerdict."""

    def _make_verifier(self):
        return ClaimVerifier(
            entropy_monitor=EntropyMonitor(),
            confidence_propagator=ConfidencePropagator(),
        )

    def test_batch_returns_results_and_verdict(self):
        verifier = self._make_verifier()
        claims = [make_claim(claim_id=f"c-{i}") for i in range(5)]
        results, verdict = verifier.verify_batch(claims)
        assert isinstance(verdict, BatchVerdict)
        assert len(results) == 5

    def test_batch_mixed_valid_invalid(self):
        verifier = self._make_verifier()
        good = [make_claim(claim_id=f"g-{i}") for i in range(3)]
        bad = [make_claim(entity_1="", claim_id=f"b-{i}") for i in range(2)]
        results, verdict = verifier.verify_batch(good + bad)
        suspicious = [r for r in results if r.status == VerificationStatus.SUSPICIOUS]
        assert len(suspicious) >= 2


class TestClaimVerifierForUse:
    """verify_for_use: confidence-gated usage verification."""

    def _make_verifier(self):
        return ClaimVerifier(
            entropy_monitor=EntropyMonitor(),
            confidence_propagator=ConfidencePropagator(),
        )

    def test_sufficient_confidence_passes(self):
        verifier = self._make_verifier()
        claim = make_claim(confidence=0.9)
        result = verifier.verify_for_use(claim, usage="publication", supporting_claims=[make_claim(), make_claim(claim_id="c2")])
        assert result.status == VerificationStatus.VERIFIED

    def test_insufficient_confidence_fails(self):
        verifier = self._make_verifier()
        claim = make_claim(confidence=0.3)
        result = verifier.verify_for_use(claim, usage="publication")
        assert result.status == VerificationStatus.SUSPICIOUS

    def test_experiment_level_passes_with_confidence(self):
        verifier = self._make_verifier()
        claim = make_claim(confidence=0.75)
        result = verifier.verify_for_use(claim, usage="experiment", supporting_claims=[make_claim()])
        assert result.status == VerificationStatus.VERIFIED

    def test_experiment_level_needs_corroboration(self):
        verifier = self._make_verifier()
        claim = make_claim(confidence=0.75)
        # No supporting claims -> fails corroboration check
        result = verifier.verify_for_use(claim, usage="experiment")
        assert result.status == VerificationStatus.SUSPICIOUS

    def test_storage_always_passes(self):
        verifier = self._make_verifier()
        claim = make_claim(confidence=0.01)
        result = verifier.verify_for_use(claim, usage="storage")
        assert result.status == VerificationStatus.VERIFIED


# ══════════════════════════════════════════════════════════════════════
# ProvenanceTracker
# ══════════════════════════════════════════════════════════════════════


class TestProvenanceTrackerBasics:
    """Core record-keeping operations."""

    def test_record_creates_provenance(self):
        tracker = ProvenanceTracker()
        record = tracker.record("prov-1", source_paper_id="paper-1")
        assert isinstance(record, ProvenanceRecord)
        assert record.claim_id == "prov-1"

    def test_get_nonexistent_returns_none(self):
        tracker = ProvenanceTracker()
        assert tracker.get_record("does-not-exist") is None

    def test_get_existing_returns_record(self):
        tracker = ProvenanceTracker()
        tracker.record("prov-2", source_paper_id="paper-1")
        record = tracker.get_record("prov-2")
        assert record is not None
        assert record.claim_id == "prov-2"

    def test_update_status(self):
        tracker = ProvenanceTracker()
        tracker.record("prov-3", source_paper_id="paper-1")
        tracker.update_status("prov-3", VerificationStatus.VERIFIED)
        record = tracker.get_record("prov-3")
        assert record.verification_status == VerificationStatus.VERIFIED

    def test_update_status_records_history(self):
        tracker = ProvenanceTracker()
        tracker.record("prov-4", source_paper_id="paper-1")
        tracker.update_status("prov-4", VerificationStatus.VERIFIED)
        tracker.update_status("prov-4", VerificationStatus.SUSPICIOUS)
        record = tracker.get_record("prov-4")
        assert len(record.verification_history) >= 2

    def test_record_with_extraction_method(self):
        tracker = ProvenanceTracker()
        record = tracker.record("prov-5", source_paper_id="paper-1", extraction_method="rule")
        assert record.extraction_method == "rule"


class TestProvenanceTrackerDependencies:
    """Dependency tracking and cascade impact."""

    def test_add_dependency(self):
        tracker = ProvenanceTracker()
        tracker.record("parent", source_paper_id="p1")
        tracker.record("child", source_paper_id="p1")
        tracker.add_dependency("child", "parent")
        record = tracker.get_record("child")
        assert "parent" in record.upstream_premises

    def test_get_downstream_transitive(self):
        tracker = ProvenanceTracker()
        for cid in ["a", "b", "c"]:
            tracker.record(cid, source_paper_id="p1")
        tracker.add_dependency("b", "a")
        tracker.add_dependency("c", "b")
        downstream = tracker.get_downstream("a")
        assert "b" in downstream
        assert "c" in downstream

    def test_cascade_impact_counts_all(self):
        tracker = ProvenanceTracker()
        for cid in ["root", "d1", "d2", "d3"]:
            tracker.record(cid, source_paper_id="p1")
        tracker.add_dependency("d1", "root")
        tracker.add_dependency("d2", "root")
        tracker.add_dependency("d3", "d1")
        count = tracker.cascade_impact("root")
        assert count == 3  # d1, d2, d3

    def test_empty_tracker_zero_impact(self):
        tracker = ProvenanceTracker()
        assert tracker.cascade_impact("anything") == 0

    def test_quarantine_marks_all_dependents(self):
        tracker = ProvenanceTracker()
        for cid in ["root", "mid", "leaf"]:
            tracker.record(cid, source_paper_id="p1")
        tracker.add_dependency("mid", "root")
        tracker.add_dependency("leaf", "mid")
        quarantined = tracker.quarantine("root")
        assert len(quarantined) == 3
        for cid in ["root", "mid", "leaf"]:
            record = tracker.get_record(cid)
            assert record.verification_status == VerificationStatus.QUARANTINED

    def test_multiple_dependency_levels(self):
        """Build a deeper DAG and verify downstream discovery."""
        tracker = ProvenanceTracker()
        ids = [f"level-{i}" for i in range(5)]
        for cid in ids:
            tracker.record(cid, source_paper_id="p1")
        for i in range(len(ids) - 1):
            tracker.add_dependency(ids[i + 1], ids[i])
        downstream = tracker.get_downstream(ids[0])
        assert len(downstream) == 4  # all except root

    def test_quarantine_does_not_affect_unrelated(self):
        tracker = ProvenanceTracker()
        for cid in ["a", "b", "unrelated"]:
            tracker.record(cid, source_paper_id="p1")
        tracker.add_dependency("b", "a")
        tracker.quarantine("a")
        unrelated = tracker.get_record("unrelated")
        assert unrelated.verification_status != VerificationStatus.QUARANTINED
