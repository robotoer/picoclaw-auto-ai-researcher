"""Research agents: specialized AI agents for autonomous research."""

from auto_researcher.agents.base import BaseAgent
from auto_researcher.agents.critic import Critic
from auto_researcher.agents.experiment_designer import ExperimentDesigner
from auto_researcher.agents.hypothesis_generator import HypothesisGenerator
from auto_researcher.agents.literature_analyst import LiteratureAnalyst
from auto_researcher.agents.science_communicator import ScienceCommunicator
from auto_researcher.agents.statistician import Statistician
from auto_researcher.agents.synthesizer import Synthesizer

__all__ = [
    "BaseAgent",
    "Critic",
    "ExperimentDesigner",
    "HypothesisGenerator",
    "LiteratureAnalyst",
    "ScienceCommunicator",
    "Statistician",
    "Synthesizer",
]
