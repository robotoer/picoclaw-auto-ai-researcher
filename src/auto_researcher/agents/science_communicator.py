"""Science Communicator agent for translating findings into clear writing."""

from __future__ import annotations

from typing import Any

from auto_researcher.agents.base import BaseAgent
from auto_researcher.config import ResearchConfig
from auto_researcher.models.agent import AgentMessage, AgentRole, AgentState
from auto_researcher.utils.llm import LLMClient

SYSTEM_PROMPT = (
    "You are a Science Communicator in an autonomous AI research system. "
    "You translate research findings into clear, compelling language for "
    "different audiences and venues. You write conference papers, blog posts, "
    "and grant proposals. You have studied patterns from highly-cited papers "
    "and understand what makes scientific writing effective. You frame claims "
    "accurately without overclaiming or underselling."
)


class ScienceCommunicator(BaseAgent):
    """Agent for translating findings into clear, compelling writing."""

    role = AgentRole.SCIENCE_COMMUNICATOR

    def __init__(self, config: ResearchConfig, llm: LLMClient) -> None:
        super().__init__(config, llm)
        self._drafts: dict[str, dict[str, Any]] = {}

    async def execute(self, task: AgentMessage) -> AgentMessage:
        self.set_state(AgentState.WORKING)

        handlers = {
            "write_paper": self._write_paper,
            "write_blog_post": self._write_blog_post,
            "write_grant_proposal": self._write_grant_proposal,
            "reframe_for_audience": self._reframe_for_audience,
            "improve_writing": self._improve_writing,
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

    async def _write_paper(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Write a conference/journal paper from research results."""
        title = payload.get("title", "")
        hypothesis = payload.get("hypothesis", "")
        methodology = payload.get("methodology", "")
        results = payload.get("results", {})
        related_work = payload.get("related_work", [])
        venue = payload.get("venue", "top ML conference")
        review_feedback = payload.get("review_feedback", [])

        revision_context = ""
        if review_feedback:
            revision_context = (
                "\n\nThis is a revision. Address these reviewer comments:\n" +
                "\n".join(f"- {r}" for r in review_feedback)
            )

        prompt = (
            f"Write a complete research paper for submission to {venue}.\n\n"
            f"Title: {title}\n"
            f"Main hypothesis/contribution: {hypothesis}\n"
            f"Methodology: {methodology}\n"
            f"Results: {results}\n"
            f"Related work to cite: {related_work}\n"
            f"{revision_context}\n\n"
            "Write each section following the patterns of highly-cited papers:\n"
            "1. Abstract: concise, informative, stating contribution clearly\n"
            "2. Introduction: motivate the problem, state contributions\n"
            "3. Related Work: position relative to prior art\n"
            "4. Method: clear, reproducible description\n"
            "5. Experiments: comprehensive evaluation\n"
            "6. Results: honest, well-organized presentation\n"
            "7. Discussion: implications, limitations, future work\n"
            "8. Conclusion: concise summary\n\n"
            "Guidelines:\n"
            "- Frame claims accurately - do not overclaim\n"
            "- Use precise, active language\n"
            "- State limitations honestly\n"
            "- Make the paper self-contained\n\n"
            "Return JSON with:\n"
            "- title: final title\n"
            "- abstract: abstract text\n"
            "- sections: dict of {section_name: content}\n"
            "- key_contributions: list of stated contributions\n"
            "- claim_strength: how strongly claims are framed (conservative/moderate/strong)"
        )
        result = await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.4)

        draft_id = title or str(len(self._drafts))
        self._drafts[draft_id] = result

        self.write_episodic(
            f"Wrote paper draft: {title}",
            tags=["paper_writing", venue],
            importance=0.9,
        )

        return result

    async def _write_blog_post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Write a blog post about research findings."""
        findings = payload.get("findings", "")
        audience = payload.get("audience", "technical ML practitioners")
        tone = payload.get("tone", "informative but accessible")
        key_takeaways = payload.get("key_takeaways", [])

        prompt = (
            f"Write a blog post about these research findings.\n\n"
            f"Findings: {findings}\n\n"
            f"Target audience: {audience}\n"
            f"Tone: {tone}\n"
            f"Key takeaways to convey:\n" +
            "\n".join(f"- {t}" for t in key_takeaways) + "\n\n"
            "Write an engaging, accurate blog post that:\n"
            "1. Opens with a compelling hook\n"
            "2. Explains the problem clearly\n"
            "3. Presents the approach accessibly\n"
            "4. Highlights what's new and why it matters\n"
            "5. Is honest about limitations\n"
            "6. Ends with practical implications or future directions\n\n"
            "Return JSON with:\n"
            "- title: catchy but accurate title\n"
            "- subtitle: one-line summary\n"
            "- content: full blog post text\n"
            "- tldr: 2-3 sentence summary\n"
            "- suggested_figures: list of figures/diagrams to include"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.6)

    async def _write_grant_proposal(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Write a grant proposal section."""
        research_direction = payload.get("research_direction", "")
        preliminary_results = payload.get("preliminary_results", [])
        budget_context = payload.get("budget_context", "")
        funding_agency = payload.get("funding_agency", "")
        duration = payload.get("duration", "3 years")

        prompt = (
            f"Write a grant proposal for this research direction.\n\n"
            f"Research Direction: {research_direction}\n"
            f"Funding Agency: {funding_agency}\n"
            f"Duration: {duration}\n\n"
            f"Preliminary results:\n" +
            "\n".join(f"- {r}" for r in preliminary_results) + "\n\n"
        )
        if budget_context:
            prompt += f"Budget context: {budget_context}\n\n"

        prompt += (
            "Write the key sections:\n"
            "1. Project Summary: concise overview\n"
            "2. Intellectual Merit: why this advances science\n"
            "3. Broader Impacts: societal importance\n"
            "4. Research Plan: detailed technical plan\n"
            "5. Timeline: milestones and deliverables\n\n"
            "Return JSON with:\n"
            "- project_title: compelling title\n"
            "- project_summary: abstract-length summary\n"
            "- sections: dict of {section_name: content}\n"
            "- key_innovations: list of novel aspects\n"
            "- risk_mitigation: how risks are addressed"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.5)

    async def _reframe_for_audience(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Reframe research findings for a different audience."""
        content = payload.get("content", "")
        source_audience = payload.get("source_audience", "researchers")
        target_audience = payload.get("target_audience", "general public")

        prompt = (
            f"Reframe this research content for a different audience.\n\n"
            f"Original content (written for {source_audience}):\n{content}\n\n"
            f"Target audience: {target_audience}\n\n"
            "Adapt the content by:\n"
            "1. Adjusting technical depth appropriately\n"
            "2. Using analogies the target audience would understand\n"
            "3. Emphasizing aspects most relevant to them\n"
            "4. Maintaining accuracy while improving accessibility\n"
            "5. Removing unnecessary jargon\n\n"
            "Return JSON with:\n"
            "- reframed_content: the adapted text\n"
            "- key_adaptations: list of changes made and why\n"
            "- analogies_used: list of analogies introduced\n"
            "- accuracy_notes: any accuracy trade-offs made for clarity"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.5)

    async def _improve_writing(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Improve the quality of existing scientific writing."""
        text = payload.get("text", "")
        focus_areas = payload.get("focus_areas", ["clarity", "conciseness", "precision"])

        prompt = (
            f"Improve this scientific writing.\n\n"
            f"Original text:\n{text}\n\n"
            f"Focus on improving: {', '.join(focus_areas)}\n\n"
            "Apply these principles from highly-cited papers:\n"
            "- Active voice over passive\n"
            "- Precise claims with appropriate hedging\n"
            "- Clear logical flow between sentences\n"
            "- Eliminate redundancy\n"
            "- Strong topic sentences for each paragraph\n"
            "- Concrete examples for abstract claims\n\n"
            "Return JSON with:\n"
            "- improved_text: the improved version\n"
            "- changes: list of {original, revised, reason}\n"
            "- overall_assessment: brief quality assessment"
        )
        return await self._ask_llm_structured(prompt, system=SYSTEM_PROMPT, temperature=0.4)

    def get_drafts(self) -> dict[str, dict[str, Any]]:
        return dict(self._drafts)
