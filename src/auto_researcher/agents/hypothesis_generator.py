"""Hypothesis Generator agent for creating novel, testable hypotheses."""

from __future__ import annotations

import uuid
from typing import Any

from auto_researcher.agents.base import BaseAgent
from auto_researcher.config import ResearchConfig
from auto_researcher.models.agent import AgentMessage, AgentRole, AgentState
from auto_researcher.models.hypothesis import (
    FalsificationCriteria,
    Hypothesis,
)
from auto_researcher.utils.llm import LLMClient

SYSTEM_PROMPT = (
    "You are a Hypothesis Generator in an autonomous AI research system. "
    "You create novel, testable, well-scoped hypotheses from knowledge graphs "
    "and gap maps. You think creatively but rigorously, always providing "
    "falsification criteria and rationale. You reason about counterfactuals "
    "and consider multiple granularities of risk and reward."
)


class HypothesisGenerator(BaseAgent):
    """Agent for generating novel, testable research hypotheses."""

    role = AgentRole.HYPOTHESIS_GENERATOR

    def __init__(self, config: ResearchConfig, llm: LLMClient) -> None:
        super().__init__(config, llm)
        self._generated: list[Hypothesis] = []

    async def execute(self, task: AgentMessage) -> AgentMessage:
        self.set_state(AgentState.WORKING)

        handlers = {
            "generate_hypotheses": self._generate_hypotheses,
            "counterfactual_reasoning": self._counterfactual_reasoning,
            "score_novelty": self._score_novelty,
            "rank_hypotheses": self._rank_hypotheses,
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

    async def _generate_hypotheses(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Generate hypotheses from knowledge graph and gap map."""
        gaps = payload.get("gaps", [])
        claims = payload.get("claims", [])
        granularity = payload.get("granularity", "all")
        num_hypotheses = int(payload.get("num_hypotheses", 5))

        granularity_instructions = ""
        if granularity == "all":
            granularity_instructions = (
                "Generate hypotheses at multiple granularities:\n"
                "- high_risk/high_reward: bold, potentially transformative\n"
                "- medium: reasonable extensions with good expected value\n"
                "- incremental: safe, high-confidence improvements"
            )
        else:
            granularity_instructions = f"Generate {granularity} granularity hypotheses."

        prompt = (
            f"Generate {num_hypotheses} novel, testable research hypotheses.\n\n"
            f"Known gaps:\n{gaps}\n\n"
            f"Known claims:\n{claims}\n\n"
            f"{granularity_instructions}\n\n"
            "For each hypothesis, provide JSON with:\n"
            "- entity_1: the subject\n"
            "- relation: the predicted relationship\n"
            "- entity_2: the object\n"
            "- conditions: conditions under which this should hold\n"
            "- confidence: your prior confidence 0.0-1.0\n"
            "- rationale: why you think this is worth investigating\n"
            "- supporting_evidence: list of evidence supporting this\n"
            "- counter_evidence: list of evidence against this\n"
            "- falsification_criteria: list of {description, test_method, "
            "expected_outcome_if_true, expected_outcome_if_false}\n"
            "- granularity: high_risk/medium/incremental\n"
            "- novelty_reasoning: why this is novel\n\n"
            "Return JSON: {\"hypotheses\": [...]}"
        )
        result = await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.7)

        hypotheses = self._parse_hypotheses(result.get("hypotheses", []))
        self._generated.extend(hypotheses)

        self.write_episodic(
            f"Generated {len(hypotheses)} hypotheses",
            tags=["hypothesis_generation"],
            importance=0.8,
        )

        return {
            "hypotheses": [h.model_dump(mode="json") for h in hypotheses],
            "count": len(hypotheses),
        }

    async def _counterfactual_reasoning(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Perform counterfactual reasoning on a hypothesis."""
        hypothesis_text = payload.get("hypothesis", "")
        known_facts = payload.get("known_facts", [])

        prompt = (
            f"Perform counterfactual reasoning on this hypothesis:\n\n"
            f"Hypothesis: {hypothesis_text}\n\n"
            f"Known facts:\n" + "\n".join(f"- {f}" for f in known_facts) + "\n\n"
            "Reason about:\n"
            "1. If this hypothesis is TRUE, what else must be true? List implications.\n"
            "2. If this hypothesis is FALSE, what does that imply? List implications.\n"
            "3. What are the most surprising implications if true?\n"
            "4. What existing beliefs would need to be revised?\n"
            "5. What new hypotheses emerge from this reasoning?\n\n"
            "Return JSON with keys: implications_if_true, implications_if_false, "
            "surprising_implications, beliefs_to_revise, new_hypotheses"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.6)

    async def _score_novelty(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Score the novelty of a hypothesis against existing knowledge."""
        hypothesis_text = payload.get("hypothesis", "")
        existing_hypotheses = payload.get("existing_hypotheses", [])
        existing_claims = payload.get("existing_claims", [])

        prompt = (
            f"Score the novelty of this hypothesis:\n\n"
            f"Hypothesis: {hypothesis_text}\n\n"
            f"Existing hypotheses:\n" +
            "\n".join(f"- {h}" for h in existing_hypotheses[:20]) + "\n\n"
            "Existing claims:\n" +
            "\n".join(f"- {c}" for c in existing_claims[:20]) + "\n\n"
            "Assess novelty along these dimensions:\n"
            "1. Has this exact idea been proposed before?\n"
            "2. Has a similar idea been proposed under different terminology?\n"
            "3. Is this a novel combination of known ideas?\n"
            "4. Does this challenge existing assumptions?\n\n"
            "Return JSON: {\"novelty_score\": 0.0-1.0, \"novelty_type\": "
            "\"completely_new|novel_combination|reframing|incremental_extension\", "
            "\"similar_existing\": [...], \"reasoning\": \"...\"}"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.3)

    async def _rank_hypotheses(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Rank hypotheses by expected research value."""
        hypotheses_data = payload.get("hypotheses", [])

        prompt = (
            "Rank these hypotheses by expected research value.\n\n"
            "Hypotheses:\n"
        )
        for i, h in enumerate(hypotheses_data):
            prompt += f"\n{i+1}. {h}\n"

        prompt += (
            "\nRank by considering:\n"
            "- Novelty and surprise value\n"
            "- Feasibility of testing\n"
            "- Impact if confirmed\n"
            "- Information gain even if refuted\n"
            "- Connection to important open problems\n\n"
            "Return JSON: {\"ranking\": [{\"index\": <int>, \"score\": <float>, "
            "\"rationale\": \"...\"}], \"recommended_portfolio\": \"...\"}"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.4)

    def _parse_hypotheses(self, hypotheses_data: list[dict[str, Any]]) -> list[Hypothesis]:
        """Parse raw hypothesis dicts into Hypothesis model objects."""
        hypotheses: list[Hypothesis] = []
        for raw in hypotheses_data:
            falsification = []
            for fc in raw.get("falsification_criteria", []):
                falsification.append(FalsificationCriteria(
                    description=fc.get("description", ""),
                    test_method=fc.get("test_method", ""),
                    expected_outcome_if_true=fc.get("expected_outcome_if_true", ""),
                    expected_outcome_if_false=fc.get("expected_outcome_if_false", ""),
                ))

            hypotheses.append(Hypothesis(
                id=str(uuid.uuid4()),
                entity_1=raw.get("entity_1", ""),
                relation=raw.get("relation", ""),
                entity_2=raw.get("entity_2", ""),
                conditions=raw.get("conditions", ""),
                confidence=float(raw.get("confidence", 0.5)),
                rationale=raw.get("rationale", ""),
                supporting_evidence=raw.get("supporting_evidence", []),
                counter_evidence=raw.get("counter_evidence", []),
                falsification_criteria=falsification,
                novelty_score=float(raw.get("novelty_score", 0.0)),
                granularity=raw.get("granularity", "medium"),
            ))
        return hypotheses

    def get_generated_hypotheses(self) -> list[Hypothesis]:
        return list(self._generated)
