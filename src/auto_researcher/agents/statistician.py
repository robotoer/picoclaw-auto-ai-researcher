"""Statistician agent for result interpretation and statistical testing."""

from __future__ import annotations

from typing import Any

from auto_researcher.agents.base import BaseAgent
from auto_researcher.config import ResearchConfig
from auto_researcher.models.agent import AgentMessage, AgentRole, AgentState
from auto_researcher.utils.llm import LLMClient

SYSTEM_PROMPT = (
    "You are a Statistician agent in an autonomous AI research system. "
    "You specialize in rigorous statistical analysis: hypothesis testing, "
    "power analysis, effect size estimation, confidence intervals, and "
    "detecting common statistical issues. You are meticulous about "
    "assumptions, multiple comparison corrections, and proper interpretation "
    "of p-values and effect sizes. You flag statistical errors others miss."
)


class Statistician(BaseAgent):
    """Agent for statistical analysis and result interpretation."""

    role = AgentRole.STATISTICIAN

    def __init__(self, config: ResearchConfig, llm: LLMClient) -> None:
        super().__init__(config, llm)
        self._analyses: list[dict[str, Any]] = []

    async def execute(self, task: AgentMessage) -> AgentMessage:
        self.set_state(AgentState.WORKING)

        handlers = {
            "interpret_results": self._interpret_results,
            "power_analysis": self._power_analysis,
            "effect_size": self._effect_size,
            "multiple_comparisons": self._multiple_comparisons,
            "confidence_intervals": self._confidence_intervals,
            "detect_issues": self._detect_issues,
            "recommend_test": self._recommend_test,
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
        self._analyses.append({"task_type": task.task_type, "result": result})
        return self.create_message(
            receiver=task.sender,
            task_type=f"{task.task_type}_result",
            payload=result,
            in_reply_to=task.message_id,
        )

    async def _interpret_results(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Interpret experimental results with statistical rigor."""
        metrics = payload.get("metrics", {})
        hypothesis = payload.get("hypothesis", "")
        experiment_description = payload.get("experiment_description", "")
        sample_sizes = payload.get("sample_sizes", {})
        baselines = payload.get("baselines", {})

        prompt = (
            f"Interpret these experimental results with statistical rigor.\n\n"
            f"Hypothesis: {hypothesis}\n"
            f"Experiment: {experiment_description}\n\n"
            f"Metrics:\n{metrics}\n\n"
            f"Sample sizes: {sample_sizes}\n"
            f"Baselines: {baselines}\n\n"
            "Provide a thorough interpretation:\n"
            "1. Do the results support or refute the hypothesis?\n"
            "2. How large are the effects in practical terms?\n"
            "3. Are the results statistically significant?\n"
            "4. Are there any concerning patterns in the data?\n"
            "5. What can and cannot be concluded?\n\n"
            "Return JSON with:\n"
            "- interpretation: detailed narrative interpretation\n"
            "- conclusion: confirmed/refuted/inconclusive\n"
            "- confidence_in_conclusion: 0.0-1.0\n"
            "- effect_sizes: dict of {metric: {size, practical_significance}}\n"
            "- caveats: list of important caveats\n"
            "- additional_analyses_needed: list of follow-up analyses\n"
            "- presentation_recommendations: how to report these results"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.2)

    async def _power_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Perform statistical power analysis."""
        test_type = payload.get("test_type", "")
        effect_size = payload.get("effect_size", "")
        alpha = float(payload.get("alpha", 0.05))
        power = float(payload.get("power", 0.8))
        groups = int(payload.get("groups", 2))
        design = payload.get("design", "between_subjects")

        prompt = (
            f"Perform a detailed power analysis.\n\n"
            f"Statistical test: {test_type}\n"
            f"Expected effect size: {effect_size}\n"
            f"Significance level (alpha): {alpha}\n"
            f"Desired power: {power}\n"
            f"Number of groups: {groups}\n"
            f"Design: {design}\n\n"
            "Calculate and provide JSON with:\n"
            "- required_n_per_group: minimum sample size per group\n"
            "- total_n: total required sample size\n"
            "- achieved_power: power at recommended N\n"
            "- power_curve: list of {n, power} showing how power changes with N\n"
            "- effect_size_detectable: minimum detectable effect at given N and power\n"
            "- assumptions: list of assumptions in this analysis\n"
            "- recommendations: practical advice on sample sizing\n"
            "- formula_used: the power formula or method applied"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.2)

    async def _effect_size(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Estimate and interpret effect sizes."""
        metric_name = payload.get("metric_name", "")
        treatment_values = payload.get("treatment_values", [])
        control_values = payload.get("control_values", [])
        context = payload.get("context", "")

        prompt = (
            f"Estimate and interpret effect sizes.\n\n"
            f"Metric: {metric_name}\n"
            f"Treatment values: {treatment_values}\n"
            f"Control values: {control_values}\n"
            f"Context: {context}\n\n"
            "Compute multiple effect size measures and interpret them.\n\n"
            "Return JSON with:\n"
            "- cohens_d: estimated Cohen's d (or equivalent)\n"
            "- cohens_d_interpretation: small/medium/large based on conventions\n"
            "- practical_significance: is this meaningful in context?\n"
            "- confidence_interval: {lower, upper, confidence_level}\n"
            "- other_measures: dict of other relevant effect size measures\n"
            "- comparison_to_field: how does this compare to typical effects "
            "in this area\n"
            "- interpretation: narrative interpretation"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.2)

    async def _multiple_comparisons(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply and recommend multiple comparison corrections."""
        comparisons = payload.get("comparisons", [])
        family_wise_alpha = float(payload.get("family_wise_alpha", 0.05))

        prompt = (
            f"Apply multiple comparison corrections to these tests.\n\n"
            f"Comparisons:\n" +
            "\n".join(f"{i+1}. {c}" for i, c in enumerate(comparisons)) + "\n\n"
            f"Desired family-wise error rate: {family_wise_alpha}\n\n"
            "Apply multiple correction methods and compare.\n\n"
            "Return JSON with:\n"
            "- bonferroni: {adjusted_alpha, results: list of {comparison, "
            "p_value, adjusted_p, significant}}\n"
            "- holm: {results: list of {comparison, p_value, adjusted_p, significant}}\n"
            "- fdr_bh: {results: list of {comparison, q_value, significant}}\n"
            "- recommendation: which method to use and why\n"
            "- num_comparisons: total number of comparisons\n"
            "- concerns: any issues with the multiple testing setup"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.2)

    async def _confidence_intervals(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Compute and interpret confidence intervals."""
        estimate = payload.get("estimate", "")
        data_summary = payload.get("data_summary", {})
        confidence_level = float(payload.get("confidence_level", 0.95))
        method = payload.get("method", "")

        prompt = (
            f"Compute and interpret confidence intervals.\n\n"
            f"Point estimate: {estimate}\n"
            f"Data summary: {data_summary}\n"
            f"Confidence level: {confidence_level}\n"
            f"Method preference: {method}\n\n"
            "Return JSON with:\n"
            "- ci_lower: lower bound\n"
            "- ci_upper: upper bound\n"
            "- point_estimate: best estimate\n"
            "- method_used: which CI method was applied\n"
            "- interpretation: what the CI means in context\n"
            "- width_assessment: is the CI narrow enough to be useful?\n"
            "- assumptions: list of assumptions\n"
            "- alternative_methods: other CI methods that could be used"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.2)

    async def _detect_issues(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Detect statistical issues in results or methodology."""
        results = payload.get("results", {})
        methodology = payload.get("methodology", "")
        claims = payload.get("claims", [])

        prompt = (
            f"Detect statistical issues in these results and methodology.\n\n"
            f"Methodology:\n{methodology}\n\n"
            f"Results:\n{results}\n\n"
            f"Claims being made:\n" +
            "\n".join(f"- {c}" for c in claims) + "\n\n"
            "Check for these common issues:\n"
            "1. P-hacking or selective reporting\n"
            "2. Multiple comparisons without correction\n"
            "3. Underpowered studies\n"
            "4. Inappropriate statistical tests\n"
            "5. Violated assumptions (normality, independence, etc.)\n"
            "6. Confusing statistical and practical significance\n"
            "7. Base rate fallacy\n"
            "8. Simpson's paradox potential\n"
            "9. Overfitting or data leakage\n"
            "10. Inappropriate causal claims from correlational data\n\n"
            "Return JSON with:\n"
            "- issues: list of {issue, severity, description, recommendation}\n"
            "  severity: minor/major/critical\n"
            "- overall_statistical_quality: poor/acceptable/good/excellent\n"
            "- claims_supported: list of claims well-supported by the stats\n"
            "- claims_unsupported: list of claims not supported or overclaimed\n"
            "- missing_analyses: list of analyses that should have been done"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.3)

    async def _recommend_test(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Recommend appropriate statistical tests for a given scenario."""
        research_question = payload.get("research_question", "")
        data_type = payload.get("data_type", "")
        sample_size = payload.get("sample_size", "")
        groups = payload.get("groups", "")
        design = payload.get("design", "")

        prompt = (
            f"Recommend the most appropriate statistical test(s).\n\n"
            f"Research question: {research_question}\n"
            f"Data type: {data_type}\n"
            f"Sample size: {sample_size}\n"
            f"Number of groups: {groups}\n"
            f"Study design: {design}\n\n"
            "Return JSON with:\n"
            "- recommended_test: primary recommended test\n"
            "- alternatives: list of alternative tests with trade-offs\n"
            "- assumptions_to_check: list of assumptions that must hold\n"
            "- non_parametric_fallback: what to use if assumptions are violated\n"
            "- implementation_notes: practical advice for running the test\n"
            "- common_mistakes: pitfalls to avoid with this test"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.2)

    def get_analyses(self) -> list[dict[str, Any]]:
        return list(self._analyses)
