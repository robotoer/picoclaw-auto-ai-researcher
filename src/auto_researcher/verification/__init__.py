"""Hallucination cascade prevention and claim verification."""

from auto_researcher.verification.confidence_propagation import ConfidencePropagator
from auto_researcher.verification.entropy_monitor import EntropyMonitor
from auto_researcher.verification.claim_verifier import ClaimVerifier
from auto_researcher.verification.provenance_tracker import ProvenanceTracker

__all__ = [
    "ConfidencePropagator",
    "EntropyMonitor",
    "ClaimVerifier",
    "ProvenanceTracker",
]
