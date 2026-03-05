"""Experiment Designer agent for designing rigorous experiments."""

from __future__ import annotations

import uuid
from typing import Any

from auto_researcher.agents.base import BaseAgent
from auto_researcher.config import ResearchConfig
from auto_researcher.models.agent import AgentMessage, AgentRole, AgentState
from auto_researcher.models.research_thread import ExperimentDesign
from auto_researcher.utils.llm import LLMClient

SYSTEM_PROMPT = (
    "You are an Experiment Designer in an autonomous AI research system. "
    "You design rigorous experiments that maximally discriminate between "
    "competing hypotheses. You apply Bayesian reasoning for expected "
    "information gain, specify all necessary controls and confounds, "
    "perform statistical power analysis, and estimate compute costs. "
    "Your designs are precise, reproducible, and efficient."
)


class ExperimentDesigner(BaseAgent):
    """Agent for designing experiments that test hypotheses."""

    role = AgentRole.EXPERIMENT_DESIGNER

    def __init__(self, config: ResearchConfig, llm: LLMClient) -> None:
        super().__init__(config, llm)
        self._designs: list[ExperimentDesign] = []

    async def execute(self, task: AgentMessage) -> AgentMessage:
        self.set_state(AgentState.WORKING)

        handlers = {
            "design_experiment": self._design_experiment,
            "power_analysis": self._power_analysis,
            "simulate_outcomes": self._simulate_outcomes,
            "estimate_compute": self._estimate_compute,
            "generate_code": self._generate_code,
            "information_gain": self._information_gain,
        }

        handler = handlers.get(task.task_type)
        if handler is None:
            return self.create_message(
                receiver=task.sender,
                task_type="error",
                payload={"error": f"Unknown task type: {task.task_type}"},
                in_reply_to=task.message_id,
            )

        result = await handler(task.payload)
        return self.create_message(
            receiver=task.sender,
            task_type=f"{task.task_type}_result",
            payload=result,
            in_reply_to=task.message_id,
        )

    async def _design_experiment(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Design a complete experiment for testing a hypothesis."""
        hypothesis = payload.get("hypothesis", "")
        hypothesis_id = payload.get("hypothesis_id", "")
        competing_hypotheses = payload.get("competing_hypotheses", [])
        constraints = payload.get("constraints", {})
        available_datasets = payload.get("available_datasets", [])
        available_models = payload.get("available_models", [])

        prompt = (
            f"Design a rigorous experiment to test this hypothesis:\n\n"
            f"Primary Hypothesis: {hypothesis}\n\n"
        )
        if competing_hypotheses:
            prompt += (
                "Competing hypotheses to discriminate from:\n" +
                "\n".join(f"- {h}" for h in competing_hypotheses) + "\n\n"
            )
        if available_datasets:
            prompt += f"Available datasets: {available_datasets}\n"
        if available_models:
            prompt += f"Available models: {available_models}\n"
        if constraints:
            prompt += f"Constraints: {constraints}\n"

        prompt += (
            "\nDesign a complete experiment as JSON with:\n"
            "- description: what the experiment does and why\n"
            "- methodology: detailed step-by-step methodology\n"
            "- datasets: list of datasets to use and why\n"
            "- models: list of models/baselines to compare\n"
            "- metrics: list of metrics with justification\n"
            "- controls: list of control conditions\n"
            "- confounds: list of potential confounds and how to address them\n"
            "- independent_variables: what is manipulated\n"
            "- dependent_variables: what is measured\n"
            "- sample_size_rationale: why this sample size\n"
            "- expected_outcomes: dict mapping hypothesis outcomes to expected results\n"
            "- statistical_tests: which tests to use\n"
            "- estimated_compute_hours: rough estimate\n"
            "- risk_assessment: what could go wrong"
        )
        result = await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.4)

        design = ExperimentDesign(
            id=str(uuid.uuid4()),
            hypothesis_id=hypothesis_id,
            description=result.get("description", ""),
            methodology=result.get("methodology", ""),
            datasets=result.get("datasets", []),
            models=result.get("models", []),
            metrics=result.get("metrics", []),
            controls=result.get("controls", []),
            confounds=result.get("confounds", []),
            estimated_compute_hours=float(result.get("estimated_compute_hours", 0)),
            expected_outcomes=result.get("expected_outcomes", {}),
            statistical_power=result.get("statistical_power"),
        )
        self._designs.append(design)

        self.write_episodic(
            f"Designed experiment for hypothesis {hypothesis_id}",
            tags=["experiment_design", hypothesis_id],
            importance=0.8,
        )

        return {
            "design": design.model_dump(mode="json"),
            "full_analysis": result,
        }

    async def _power_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Perform statistical power analysis for an experiment."""
        effect_size = payload.get("effect_size", "")
        significance_level = payload.get("significance_level", 0.05)
        desired_power = payload.get("desired_power", 0.8)
        test_type = payload.get("test_type", "")
        groups = int(payload.get("groups", 2))

        prompt = (
            f"Perform a statistical power analysis.\n\n"
            f"Expected effect size: {effect_size}\n"
            f"Significance level (alpha): {significance_level}\n"
            f"Desired power (1-beta): {desired_power}\n"
            f"Statistical test: {test_type}\n"
            f"Number of groups: {groups}\n\n"
            "Provide JSON with:\n"
            "- required_sample_size: minimum per group\n"
            "- total_sample_size: across all groups\n"
            "- actual_power: power at recommended sample size\n"
            "- sensitivity_analysis: how power changes with sample size "
            "(list of {n, power} pairs)\n"
            "- recommendations: practical advice\n"
            "- assumptions: list of assumptions made"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.2)

    async def _simulate_outcomes(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Pre-experiment simulation to estimate outcome distributions."""
        experiment_description = payload.get("experiment_description", "")
        hypothesis = payload.get("hypothesis", "")
        prior_results = payload.get("prior_results", [])

        prompt = (
            f"Simulate likely outcomes of this experiment before running it.\n\n"
            f"Experiment: {experiment_description}\n"
            f"Hypothesis: {hypothesis}\n\n"
            f"Prior results from similar work:\n" +
            "\n".join(f"- {r}" for r in prior_results) + "\n\n"
            "Provide JSON with:\n"
            "- scenarios: list of {name, probability, expected_metrics, interpretation}\n"
            "  Include at minimum: hypothesis_confirmed, hypothesis_refuted, inconclusive\n"
            "- prior_probability_true: P(hypothesis is true) based on priors\n"
            "- expected_information_gain: how much we learn regardless of outcome\n"
            "- worst_case: what happens if nothing works\n"
            "- best_case: ideal outcome and its implications\n"
            "- recommendation: should we proceed, modify, or abandon"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.5)

    async def _estimate_compute(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Estimate compute costs for an experiment."""
        models = payload.get("models", [])
        datasets = payload.get("datasets", [])
        num_runs = int(payload.get("num_runs", 1))
        methodology = payload.get("methodology", "")

        prompt = (
            f"Estimate compute resources needed for this experiment.\n\n"
            f"Models: {models}\n"
            f"Datasets: {datasets}\n"
            f"Number of runs: {num_runs}\n"
            f"Methodology: {methodology}\n\n"
            "Provide JSON with:\n"
            "- total_gpu_hours: estimated GPU hours\n"
            "- gpu_type_recommended: e.g., A100, H100\n"
            "- estimated_wall_time_hours: real-time hours\n"
            "- estimated_cost_usd: rough dollar cost\n"
            "- breakdown: dict of {component: hours} for each step\n"
            "- memory_requirements_gb: peak GPU memory needed\n"
            "- storage_requirements_gb: disk space for data and outputs\n"
            "- optimization_suggestions: ways to reduce cost"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.3)

    async def _generate_code(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Generate executable experiment code."""
        design = payload.get("design", {})
        framework = payload.get("framework", "pytorch")
        language = payload.get("language", "python")

        prompt = (
            f"Generate executable {language} code for this experiment using {framework}.\n\n"
            f"Experiment Design:\n{design}\n\n"
            "Generate clean, well-documented code that:\n"
            "1. Sets up the experiment (data loading, model init)\n"
            "2. Implements the methodology\n"
            "3. Runs the experiment with proper controls\n"
            "4. Collects and saves all metrics\n"
            "5. Performs statistical tests\n"
            "6. Generates summary output\n\n"
            "Return JSON: {\"code\": \"...\", \"requirements\": [...], "
            "\"usage_instructions\": \"...\", \"expected_outputs\": [...]}"
        )
        result = await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.3)

        if "code" in result and self._designs:
            self._designs[-1].code = result["code"]

        return result

    async def _information_gain(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Estimate expected information gain from an experiment."""
        experiments = payload.get("experiments", [])
        hypotheses = payload.get("hypotheses", [])

        prompt = (
            "Estimate the expected information gain for each experiment-hypothesis pair.\n\n"
            "Experiments:\n" +
            "\n".join(f"{i+1}. {e}" for i, e in enumerate(experiments)) + "\n\n"
            "Hypotheses:\n" +
            "\n".join(f"{i+1}. {h}" for i, h in enumerate(hypotheses)) + "\n\n"
            "For each experiment, estimate:\n"
            "- Which hypotheses does it discriminate between?\n"
            "- What is the expected information gain (in bits)?\n"
            "- What is the cost-to-information ratio?\n\n"
            "Return JSON: {\"experiments\": [{\"index\": <int>, "
            "\"discriminates_between\": [<hypothesis_indices>], "
            "\"expected_info_gain_bits\": <float>, "
            "\"cost_info_ratio\": <float>, "
            "\"recommendation\": \"...\"}], "
            "\"optimal_ordering\": [<experiment_indices_in_order>], "
            "\"rationale\": \"...\"}"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.3)

    def get_designs(self) -> list[ExperimentDesign]:
        return list(self._designs)
