"""Synthesizer agent for cross-domain connection and meta-analysis."""

from __future__ import annotations

from typing import Any

from auto_researcher.agents.base import BaseAgent
from auto_researcher.config import ResearchConfig
from auto_researcher.models.agent import AgentMessage, AgentRole, AgentState
from auto_researcher.utils.llm import LLMClient

SYSTEM_PROMPT = (
    "You are a Synthesizer agent in an autonomous AI research system. "
    "You operate at a higher abstraction level than other agents, "
    "identifying cross-domain connections, structural analogies, and "
    "emergent patterns across research. You maintain an analogy database "
    "and write comprehensive surveys and synthesis documents. You think "
    "about the big picture and identify themes that others miss."
)


class Synthesizer(BaseAgent):
    """Agent for cross-domain synthesis and meta-analysis."""

    role = AgentRole.SYNTHESIZER

    def __init__(self, config: ResearchConfig, llm: LLMClient) -> None:
        super().__init__(config, llm)
        self._analogy_database: list[dict[str, Any]] = []
        self._synthesis_cache: dict[str, dict[str, Any]] = {}

    async def execute(self, task: AgentMessage) -> AgentMessage:
        self.set_state(AgentState.WORKING)

        handlers = {
            "find_connections": self._find_connections,
            "detect_analogies": self._detect_analogies,
            "meta_analysis": self._meta_analysis,
            "write_survey": self._write_survey,
            "identify_patterns": self._identify_patterns,
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

    async def _find_connections(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Identify cross-domain connections between research areas."""
        domain_a = payload.get("domain_a", "")
        domain_b = payload.get("domain_b", "")
        concepts_a = payload.get("concepts_a", [])
        concepts_b = payload.get("concepts_b", [])

        prompt = (
            f"Find deep cross-domain connections between these research areas.\n\n"
            f"Domain A: {domain_a}\n"
            f"Key concepts in A: {concepts_a}\n\n"
            f"Domain B: {domain_b}\n"
            f"Key concepts in B: {concepts_b}\n\n"
            "Look for:\n"
            "1. Shared mathematical structures\n"
            "2. Isomorphic problems under different names\n"
            "3. Techniques from one domain applicable to the other\n"
            "4. Shared failure modes or limitations\n"
            "5. Complementary strengths\n"
            "6. Historical connections or divergence points\n\n"
            "Return JSON with:\n"
            "- connections: list of {concept_a, concept_b, connection_type, "
            "strength, description, potential_for_transfer}\n"
            "- shared_structures: list of mathematical/conceptual structures "
            "common to both\n"
            "- transfer_opportunities: list of specific ideas to transfer\n"
            "- novel_combinations: list of new ideas from combining both domains\n"
            "- confidence: 0.0-1.0 in the identified connections"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.6)

    async def _detect_analogies(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Detect structural analogies between research findings."""
        findings = payload.get("findings", [])
        focus_domain = payload.get("focus_domain", "")

        prompt = (
            "Detect structural analogies among these research findings.\n\n"
            "Findings:\n" +
            "\n".join(f"{i+1}. {f}" for i, f in enumerate(findings)) + "\n\n"
        )
        if focus_domain:
            prompt += f"Focus domain: {focus_domain}\n\n"

        prompt += (
            "A structural analogy means two findings share the same abstract "
            "pattern even though they involve different specific entities.\n\n"
            "For example: 'dropout in neural networks' is structurally analogous "
            "to 'random subspace methods in ensemble learning' -- both use "
            "random subset selection to improve generalization.\n\n"
            "Return JSON with:\n"
            "- analogies: list of {finding_a_index, finding_b_index, "
            "shared_structure, abstraction_level, strength, "
            "implications_for_unification}\n"
            "- meta_patterns: list of higher-order patterns across analogies\n"
            "- suggested_abstractions: list of useful abstract frameworks\n"
            "- novel_predictions: predictions made by extending analogies"
        )
        result = await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.6)

        for analogy in result.get("analogies", []):
            self._analogy_database.append(analogy)

        return result

    async def _meta_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Perform meta-analysis across multiple results."""
        results = payload.get("results", [])
        research_question = payload.get("research_question", "")

        prompt = (
            f"Perform a meta-analysis of these research results.\n\n"
            f"Research question: {research_question}\n\n"
            f"Individual results:\n" +
            "\n".join(f"{i+1}. {r}" for i, r in enumerate(results)) + "\n\n"
            "Conduct a thorough meta-analysis:\n"
            "1. Overall effect direction and magnitude\n"
            "2. Heterogeneity across studies\n"
            "3. Potential moderators of the effect\n"
            "4. Publication bias assessment\n"
            "5. Robustness of conclusions\n\n"
            "Return JSON with:\n"
            "- overall_finding: summary of the combined evidence\n"
            "- effect_direction: positive/negative/null/mixed\n"
            "- effect_consistency: consistent/inconsistent/highly_variable\n"
            "- moderators: list of {variable, effect, evidence}\n"
            "- confidence_in_conclusion: 0.0-1.0\n"
            "- limitations: list of meta-analytic limitations\n"
            "- gaps_identified: list of gaps in the evidence base\n"
            "- recommendations: list of next steps"
        )
        result = await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.4)

        cache_key = research_question or str(len(self._synthesis_cache))
        self._synthesis_cache[cache_key] = result
        return result

    async def _write_survey(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Write a comprehensive survey or synthesis document."""
        topic = payload.get("topic", "")
        papers = payload.get("papers", [])
        claims = payload.get("claims", [])
        target_length = payload.get("target_length", "medium")
        audience = payload.get("audience", "researchers")

        length_guide = {
            "short": "2-3 pages (about 1500 words)",
            "medium": "5-8 pages (about 4000 words)",
            "long": "10-15 pages (about 8000 words)",
        }

        prompt = (
            f"Write a comprehensive survey on: {topic}\n\n"
            f"Source papers ({len(papers)} total):\n" +
            "\n".join(f"- {p}" for p in papers[:30]) + "\n\n"
            "Key claims in the field:\n" +
            "\n".join(f"- {c}" for c in claims[:30]) + "\n\n"
            f"Target length: {length_guide.get(target_length, target_length)}\n"
            f"Target audience: {audience}\n\n"
            "The survey should:\n"
            "1. Provide a clear taxonomy of the field\n"
            "2. Identify major trends and shifts\n"
            "3. Highlight open problems and controversies\n"
            "4. Make connections between subfields\n"
            "5. Suggest future directions\n\n"
            "Return JSON with:\n"
            "- title: survey title\n"
            "- abstract: abstract text\n"
            "- sections: list of {title, content}\n"
            "- taxonomy: hierarchical categorization of the field\n"
            "- open_problems: list of important open questions\n"
            "- future_directions: list of promising research directions"
        )
        return await self._ask_llm_structured(
            prompt, system=SYSTEM_PROMPT, temperature=0.5,
        )

    async def _identify_patterns(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Identify emergent patterns across research threads."""
        threads = payload.get("threads", [])
        timeframe = payload.get("timeframe", "")

        prompt = (
            "Identify emergent patterns across these research threads.\n\n"
            "Research threads:\n" +
            "\n".join(f"{i+1}. {t}" for i, t in enumerate(threads)) + "\n\n"
        )
        if timeframe:
            prompt += f"Timeframe: {timeframe}\n\n"

        prompt += (
            "Look for:\n"
            "1. Recurring themes across threads\n"
            "2. Convergence of different approaches\n"
            "3. Divergence or fragmentation trends\n"
            "4. Accelerating or decelerating areas\n"
            "5. Unexpected connections between threads\n"
            "6. Paradigm shifts in progress\n\n"
            "Return JSON with:\n"
            "- patterns: list of {name, description, evidence, significance, "
            "trend_direction}\n"
            "- convergences: list of areas coming together\n"
            "- divergences: list of areas fragmenting\n"
            "- paradigm_shifts: list of potential paradigm changes\n"
            "- strategic_implications: what these patterns mean for research direction"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.5)

    def get_analogy_database(self) -> list[dict[str, Any]]:
        return list(self._analogy_database)

    def get_synthesis_cache(self) -> dict[str, dict[str, Any]]:
        return dict(self._synthesis_cache)
