"""Critic agent for adversarial evaluation of research outputs."""

from __future__ import annotations

from typing import Any

from auto_researcher.agents.base import BaseAgent
from auto_researcher.config import ResearchConfig
from auto_researcher.models.agent import AgentMessage, AgentRole, AgentState
from auto_researcher.utils.llm import LLMClient

SYSTEM_PROMPT = (
    "You are a Critic agent in an autonomous AI research system. "
    "Your role is deliberately adversarial: you are rewarded for finding "
    "genuine flaws, not for being agreeable. You attack hypotheses, "
    "experimental designs, and draft papers with rigorous scrutiny. "
    "You check for unfalsifiable claims, methodological weaknesses, "
    "statistical errors, missing controls, and prior art under different "
    "terminology. You provide structured critiques with severity ratings "
    "and constructive suggestions for improvement."
)


class CritiqueItem:
    """A single critique point with severity and suggestion."""

    def __init__(self, aspect: str, issue: str, severity: str,
                 suggestion: str, evidence: str = "") -> None:
        self.aspect = aspect
        self.issue = issue
        self.severity = severity  # "minor", "major", "critical"
        self.suggestion = suggestion
        self.evidence = evidence

    def to_dict(self) -> dict[str, str]:
        return {
            "aspect": self.aspect,
            "issue": self.issue,
            "severity": self.severity,
            "suggestion": self.suggestion,
            "evidence": self.evidence,
        }


class Critic(BaseAgent):
    """Adversarial agent for finding flaws in research outputs."""

    role = AgentRole.CRITIC

    def __init__(self, config: ResearchConfig, llm: LLMClient) -> None:
        super().__init__(config, llm)
        self._critique_history: list[dict[str, Any]] = []

    async def execute(self, task: AgentMessage) -> AgentMessage:
        self.set_state(AgentState.WORKING)

        handlers = {
            "critique_hypothesis": self._critique_hypothesis,
            "critique_experiment": self._critique_experiment,
            "critique_paper": self._critique_paper,
            "detect_prior_art": self._detect_prior_art,
            "adversarial_debate": self._adversarial_debate,
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
        self._critique_history.append({
            "task_type": task.task_type,
            "result": result,
        })
        return self.create_message(
            receiver=task.sender,
            task_type=f"{task.task_type}_result",
            payload=result,
            in_reply_to=task.message_id,
        )

    async def _critique_hypothesis(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Rigorously critique a hypothesis."""
        hypothesis = payload.get("hypothesis", "")
        supporting_evidence = payload.get("supporting_evidence", [])
        falsification_criteria = payload.get("falsification_criteria", [])

        prompt = (
            f"Rigorously critique this hypothesis. Be deliberately adversarial.\n\n"
            f"Hypothesis: {hypothesis}\n\n"
            f"Supporting evidence:\n" +
            "\n".join(f"- {e}" for e in supporting_evidence) + "\n\n"
            f"Stated falsification criteria:\n" +
            "\n".join(f"- {f}" for f in falsification_criteria) + "\n\n"
            "Attack along these dimensions:\n"
            "1. Is the hypothesis actually falsifiable? Can you construct a scenario "
            "where it would be considered refuted?\n"
            "2. Is it well-scoped? Or so broad it's trivially true?\n"
            "3. Does the supporting evidence actually support this specific claim?\n"
            "4. Are there confounds that would produce the same prediction?\n"
            "5. Are the falsification criteria sufficient and practical?\n"
            "6. Is this hypothesis novel, or a restatement of known results?\n"
            "7. Are there hidden assumptions?\n"
            "8. What is the strongest argument against this hypothesis?\n\n"
            "Return JSON with:\n"
            "- critiques: list of {aspect, issue, severity, suggestion, evidence}\n"
            "  severity: minor/major/critical\n"
            "- overall_assessment: pass/needs_revision/reject\n"
            "- strongest_objection: the single most damaging critique\n"
            "- hidden_assumptions: list of unstated assumptions\n"
            "- alternative_explanations: list of competing explanations\n"
            "- confidence_in_critique: 0.0-1.0"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.5)

    async def _critique_experiment(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Critique an experimental design for methodological weaknesses."""
        design = payload.get("design", "")
        hypothesis = payload.get("hypothesis", "")

        prompt = (
            f"Critique this experimental design. Find every weakness.\n\n"
            f"Hypothesis being tested: {hypothesis}\n\n"
            f"Experimental Design:\n{design}\n\n"
            "Check for:\n"
            "1. Missing controls or baselines\n"
            "2. Confounds not accounted for\n"
            "3. Statistical issues (multiple comparisons, p-hacking risk, "
            "underpowered tests)\n"
            "4. Dataset selection bias\n"
            "5. Metric gaming potential (optimizing metric without achieving goal)\n"
            "6. Reproducibility concerns\n"
            "7. Information leakage between train/test\n"
            "8. Cherry-picking risk in evaluation\n"
            "9. Scalability of approach\n"
            "10. Does the experiment actually test the hypothesis?\n\n"
            "Return JSON with:\n"
            "- critiques: list of {aspect, issue, severity, suggestion}\n"
            "- missing_controls: list of controls that should be added\n"
            "- statistical_issues: list of statistical methodology problems\n"
            "- overall_assessment: ready/needs_revision/fundamentally_flawed\n"
            "- minimum_changes_required: list of changes needed before running"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.4)

    async def _critique_paper(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Critique a draft paper for research quality issues."""
        paper_text = payload.get("paper_text", "")
        paper_type = payload.get("paper_type", "conference_paper")

        prompt = (
            f"Review this draft {paper_type}. Be a tough but fair reviewer.\n\n"
            f"Paper:\n{paper_text}\n\n"
            "Evaluate along standard review criteria:\n"
            "1. Novelty: Is the contribution genuinely new?\n"
            "2. Significance: Does this matter to the field?\n"
            "3. Soundness: Are the methods and claims valid?\n"
            "4. Clarity: Is the paper well-written and organized?\n"
            "5. Completeness: Are all necessary details present?\n"
            "6. Reproducibility: Could someone replicate this work?\n"
            "7. Related work: Is prior art properly cited and compared?\n"
            "8. Limitations: Are limitations honestly discussed?\n\n"
            "Return JSON with:\n"
            "- summary: one-paragraph summary of the paper\n"
            "- strengths: list of genuine strengths\n"
            "- weaknesses: list of {aspect, issue, severity, suggestion}\n"
            "- questions_for_authors: list of clarifying questions\n"
            "- missing_references: list of papers that should be cited\n"
            "- scores: {novelty, significance, soundness, clarity, overall} (1-10 each)\n"
            "- decision: accept/revise/reject\n"
            "- confidence: 0.0-1.0 in this review"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.4)

    async def _detect_prior_art(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Detect equivalent ideas that may exist under different terminology."""
        idea = payload.get("idea", "")
        domain = payload.get("domain", "machine learning")
        known_related = payload.get("known_related", [])

        prompt = (
            f"Search for prior art for this idea, including under different terminology.\n\n"
            f"Idea: {idea}\n"
            f"Domain: {domain}\n\n"
            f"Already known related work:\n" +
            "\n".join(f"- {r}" for r in known_related) + "\n\n"
            "Think about:\n"
            "1. Has this exact idea been published before?\n"
            "2. Has a very similar idea appeared under different terminology in:\n"
            "   - The same field\n"
            "   - Related fields (statistics, physics, neuroscience, etc.)\n"
            "   - Older literature (pre-deep-learning era)\n"
            "3. Is this a special case of a more general known result?\n"
            "4. Is this essentially equivalent to something known, "
            "just reframed or renamed?\n\n"
            "Return JSON with:\n"
            "- is_novel: true/false\n"
            "- prior_art: list of {description, similarity_score, "
            "original_terminology, field, why_similar}\n"
            "- novelty_assessment: what specifically is new vs known\n"
            "- recommendation: how to position this relative to prior work"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.3)

    async def _adversarial_debate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Conduct multi-round adversarial debate on a claim."""
        claim = payload.get("claim", "")
        initial_defense = payload.get("defense", "")
        max_rounds = int(payload.get("max_rounds", 3))

        rounds: list[dict[str, str]] = []
        current_defense = initial_defense

        for round_num in range(1, max_rounds + 1):
            attack_prompt = (
                f"Round {round_num} of adversarial debate.\n\n"
                f"Claim: {claim}\n\n"
                f"Current defense:\n{current_defense}\n\n"
            )
            if rounds:
                attack_prompt += "Previous rounds:\n"
                for r in rounds:
                    attack_prompt += f"- Attack: {r['attack']}\n  Defense: {r['defense']}\n"

            attack_prompt += (
                "\nProvide your strongest attack on this claim that has not been "
                "addressed by the defense. Return JSON: "
                "{\"attack\": \"...\", \"attack_type\": \"...\", \"severity\": \"...\"}"
            )
            attack_result = await self._ask_llm_structured(
                attack_prompt, system=SYSTEM_PROMPT, temperature=0.5,
            )

            defense_prompt = (
                f"Defend this claim against the following attack.\n\n"
                f"Claim: {claim}\n"
                f"Attack: {attack_result.get('attack', '')}\n\n"
                "Provide the strongest possible defense. "
                "Return JSON: {\"defense\": \"...\", \"concessions\": [\"...\"]}"
            )
            defense_result = await self._ask_llm_structured(
                defense_prompt,
                system="You are defending a scientific claim against adversarial critique.",
                temperature=0.4,
            )

            rounds.append({
                "round": str(round_num),
                "attack": attack_result.get("attack", ""),
                "attack_type": attack_result.get("attack_type", ""),
                "severity": attack_result.get("severity", ""),
                "defense": defense_result.get("defense", ""),
                "concessions": str(defense_result.get("concessions", [])),
            })
            current_defense = defense_result.get("defense", "")

        verdict_prompt = (
            f"After {max_rounds} rounds of adversarial debate on:\n"
            f"Claim: {claim}\n\n"
            "Debate rounds:\n" +
            "\n".join(
                f"Round {r['round']}: Attack ({r['severity']}): {r['attack']} | "
                f"Defense: {r['defense']}"
                for r in rounds
            ) + "\n\n"
            "Provide final verdict as JSON:\n"
            "{\"verdict\": \"claim_stands|claim_weakened|claim_refuted\", "
            "\"surviving_strengths\": [...], \"unresolved_weaknesses\": [...], "
            "\"confidence_after_debate\": 0.0-1.0, \"revised_claim\": \"...\"}"
        )
        verdict = await self._ask_llm_structured(
            verdict_prompt, system=SYSTEM_PROMPT, temperature=0.3,
        )

        return {
            "claim": claim,
            "rounds": rounds,
            "verdict": verdict,
        }

    def get_critique_history(self) -> list[dict[str, Any]]:
        return list(self._critique_history)
