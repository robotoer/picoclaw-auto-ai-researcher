"""Evaluation: SUNFIRE scoring, IWPG rewards, peer review simulation."""

from auto_researcher.evaluation.sunfire import SUNFIREEvaluator
from auto_researcher.evaluation.iwpg import IWPGScorer
from auto_researcher.evaluation.peer_review import SimulatedPeerReview
from auto_researcher.evaluation.impact_predictor import ImpactPredictor

__all__ = [
    "SUNFIREEvaluator",
    "IWPGScorer",
    "SimulatedPeerReview",
    "ImpactPredictor",
]
