"""Literature Analyst agent for deep paper reading and claim extraction."""

from __future__ import annotations

import uuid
from typing import Any

from auto_researcher.agents.base import BaseAgent
from auto_researcher.config import ResearchConfig
from auto_researcher.models.agent import AgentMessage, AgentRole, AgentState
from auto_researcher.models.claim import Claim, ClaimRelation, ClaimStatus
from auto_researcher.models.paper import Paper
from auto_researcher.utils.llm import LLMClient

SYSTEM_PROMPT = (
    "You are a Literature Analyst in an autonomous AI research system. "
    "You perform deep, careful reading of scientific papers. You extract "
    "structured claims, identify contradictions, find open questions, and "
    "map key concepts. You read full papers including appendices and "
    "supplementary material. You are precise in your claim extraction and "
    "always note the conditions under which claims hold."
)


class LiteratureAnalyst(BaseAgent):
    """Agent for deep paper reading, claim extraction, and gap identification."""

    role = AgentRole.LITERATURE_ANALYST

    def __init__(self, config: ResearchConfig, llm: LLMClient) -> None:
        super().__init__(config, llm)
        self._controversy_map: dict[str, dict[str, Any]] = {}
        self._claim_cache: dict[str, list[Claim]] = {}

    async def execute(self, task: AgentMessage) -> AgentMessage:
        self.set_state(AgentState.WORKING)
        task_type = task.task_type

        handlers = {
            "analyze_paper": self._analyze_paper,
            "extract_claims": self._extract_claims,
            "identify_gaps": self._identify_gaps,
            "update_controversy_map": self._update_controversy_map,
            "answer_question": self._answer_question,
        }

        handler = handlers.get(task_type)
        if handler is None:
            return self.create_message(
                receiver=task.sender,
                task_type="error",
                payload={"error": f"Unknown task type: {task_type}"},
                in_reply_to=task.message_id,
            )

        result = await handler(task.payload)
        return self.create_message(
            receiver=task.sender,
            task_type=f"{task_type}_result",
            payload=result,
            in_reply_to=task.message_id,
        )

    async def _analyze_paper(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Perform deep analysis of a paper including all sections."""
        paper_text = payload.get("paper_text", "")
        title = payload.get("title", "")
        paper_id = payload.get("paper_id", "")

        prompt = (
            f"Analyze the following research paper in depth.\n\n"
            f"Title: {title}\n\n"
            f"Full Text:\n{paper_text}\n\n"
            "Provide a comprehensive analysis as JSON with these keys:\n"
            "- summary: concise summary of main contributions\n"
            "- key_claims: list of {entity_1, relation, entity_2, conditions, confidence}\n"
            "- methodology: description of methods used\n"
            "- key_results: list of main findings\n"
            "- limitations: list of stated and unstated limitations\n"
            "- open_questions: list of unanswered questions raised\n"
            "- key_concepts: list of important technical concepts\n"
            "- connections: list of connections to other work"
        )
        result = await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.3)

        self.write_episodic(
            f"Analyzed paper: {title}",
            tags=["paper_analysis", paper_id],
            importance=0.8,
        )

        if "key_claims" in result:
            claims = self._parse_claims(result["key_claims"], paper_id)
            self._claim_cache[paper_id] = claims
            result["parsed_claim_count"] = len(claims)

        return result

    async def _extract_claims(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Extract structured claims from paper text."""
        paper_text = payload.get("paper_text", "")
        paper_id = payload.get("paper_id", "")
        focus_areas = payload.get("focus_areas", [])

        focus_str = ""
        if focus_areas:
            focus_str = f"\n\nFocus especially on claims related to: {', '.join(focus_areas)}"

        prompt = (
            f"Extract all scientific claims from this paper text as structured triples.\n\n"
            f"Text:\n{paper_text}{focus_str}\n\n"
            "For each claim provide JSON with:\n"
            "- entity_1: the subject/first entity\n"
            "- relation: one of [outperforms, is_variant_of, refutes, supports, "
            "requires, enables, is_applied_to, improves, extends, contradicts, "
            "is_equivalent_to]\n"
            "- entity_2: the object/second entity\n"
            "- conditions: under what conditions this holds\n"
            "- confidence: your confidence 0.0-1.0 in the claim\n"
            "- evidence_strength: weak/moderate/strong\n\n"
            "Return JSON: {\"claims\": [...]}"
        )
        result = await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.2)

        claims_data = result.get("claims", [])
        claims = self._parse_claims(claims_data, paper_id)
        self._claim_cache[paper_id] = claims

        return {
            "paper_id": paper_id,
            "claim_count": len(claims),
            "claims": [c.model_dump(mode="json") for c in claims],
        }

    async def _identify_gaps(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Identify research gaps from a set of papers."""
        paper_summaries = payload.get("paper_summaries", [])
        domain = payload.get("domain", "AI research")

        prompt = (
            f"Given these paper summaries from {domain}, identify research gaps.\n\n"
            "Paper Summaries:\n" +
            "\n---\n".join(str(s) for s in paper_summaries) + "\n\n"
            "Identify gaps as JSON with:\n"
            "- gaps: list of {description, gap_type, importance, tractability, "
            "novelty, related_concepts, suggested_approaches}\n"
            "- gap_type should be one of: empirical, population, methodological, "
            "contradictory, negative_results, combination, operationalization, "
            "replication, scale\n"
            "- contradictions: list of {claim_a, claim_b, papers, resolution_ideas}\n"
            "- missing_connections: list of {concept_a, concept_b, why_connect}"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.5)

    async def _update_controversy_map(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Update the controversy map with new evidence."""
        topic = payload.get("topic", "")
        new_evidence = payload.get("evidence", {})
        side = payload.get("side", "")

        if topic not in self._controversy_map:
            self._controversy_map[topic] = {
                "description": topic,
                "sides": {},
                "evidence": [],
                "status": "open",
                "last_updated": None,
            }

        controversy = self._controversy_map[topic]
        if side:
            if side not in controversy["sides"]:
                controversy["sides"][side] = []
            controversy["sides"][side].append(new_evidence)
        controversy["evidence"].append(new_evidence)
        controversy["last_updated"] = str(__import__("datetime").datetime.now(__import__("datetime").UTC))

        prompt = (
            f"Analyze the current state of this scientific debate:\n\n"
            f"Topic: {topic}\n"
            f"Sides: {controversy['sides']}\n"
            f"Evidence count: {len(controversy['evidence'])}\n\n"
            "Provide JSON with:\n"
            "- status: open/leaning_towards_side_a/leaning_towards_side_b/resolved\n"
            "- balance_assessment: which side has stronger evidence and why\n"
            "- key_unresolved_points: what experiments would help resolve this\n"
            "- confidence_in_assessment: 0.0-1.0"
        )
        assessment = await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.4)
        controversy["latest_assessment"] = assessment
        return {"topic": topic, "controversy": controversy, "assessment": assessment}

    async def _answer_question(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Answer a question about the paper corpus."""
        question = payload.get("question", "")
        context_papers = payload.get("context_papers", [])

        all_claims = []
        for paper_id, claims in self._claim_cache.items():
            for claim in claims:
                all_claims.append(
                    f"[{paper_id}] {claim.entity_1} {claim.relation.value} {claim.entity_2} "
                    f"(conditions: {claim.conditions}, confidence: {claim.confidence})"
                )

        prompt = (
            f"Based on the following knowledge base, answer this question:\n\n"
            f"Question: {question}\n\n"
            f"Known claims ({len(all_claims)} total):\n" +
            "\n".join(all_claims[:50]) + "\n\n"
            f"Additional context papers: {context_papers}\n\n"
            "Provide JSON with:\n"
            "- answer: your detailed answer\n"
            "- confidence: 0.0-1.0\n"
            "- supporting_claims: list of claim references\n"
            "- caveats: list of important caveats\n"
            "- follow_up_questions: list of related questions worth investigating"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.3)

    def _parse_claims(self, claims_data: list[dict[str, Any]],
                      paper_id: str) -> list[Claim]:
        """Parse raw claim dicts into Claim model objects."""
        claims: list[Claim] = []
        for raw in claims_data:
            relation_str = raw.get("relation", "supports")
            try:
                relation = ClaimRelation(relation_str)
            except ValueError:
                relation = ClaimRelation.SUPPORTS

            claims.append(Claim(
                id=str(uuid.uuid4()),
                entity_1=raw.get("entity_1", ""),
                relation=relation,
                entity_2=raw.get("entity_2", ""),
                conditions=raw.get("conditions", ""),
                confidence=float(raw.get("confidence", 0.5)),
                status=ClaimStatus.EXTRACTED,
                source_paper_ids=[paper_id],
            ))
        return claims

    def get_controversy_map(self) -> dict[str, dict[str, Any]]:
        return dict(self._controversy_map)

    def get_cached_claims(self, paper_id: str | None = None) -> list[Claim]:
        if paper_id:
            return self._claim_cache.get(paper_id, [])
        return [c for claims in self._claim_cache.values() for c in claims]
