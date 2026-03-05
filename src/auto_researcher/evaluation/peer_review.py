"""Simulated peer review with multi-agent reviewer panel."""

from __future__ import annotations

from typing import Any

from auto_researcher.config import PeerReviewConfig
from auto_researcher.models.research_thread import ResearchThread
from auto_researcher.models.reward import PeerReviewResult, ReviewComment, ReviewDecision
from auto_researcher.utils.llm import LLMClient
from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)

REVIEWER_PROFILES = [
    {
        "id": "reviewer_methodology",
        "focus": "methodology",
        "system": (
            "You are a meticulous methodologist reviewer. Focus on experimental design, "
            "statistical rigor, controls, confounds, and reproducibility. Be critical but fair."
        ),
    },
    {
        "id": "reviewer_novelty",
        "focus": "novelty",
        "system": (
            "You are a reviewer focused on novelty and significance. Assess whether the contribution "
            "is genuinely new, how it advances the field, and its potential impact."
        ),
    },
    {
        "id": "reviewer_clarity",
        "focus": "clarity",
        "system": (
            "You are a reviewer focused on clarity and communication. Assess writing quality, "
            "logical flow, figure quality, and whether claims are well-supported by evidence."
        ),
    },
]

REVIEW_PROMPT = """\
Review the following research paper draft. Provide structured feedback.

Title: {title}
{content}

For each of these aspects, provide a score (0-1) and detailed comments:
1. Methodology
2. Novelty
3. Clarity
4. Significance
5. Reproducibility

Also identify:
- Major issues (must be addressed)
- Minor issues (should be addressed)
- Strengths

Respond with JSON:
{{
    "scores": {{"methodology": <float>, "novelty": <float>, "clarity": <float>, "significance": <float>, "reproducibility": <float>}},
    "comments": [
        {{"aspect": "<string>", "comment": "<string>", "severity": "minor|major|critical", "suggestion": "<string>"}}
    ],
    "overall_score": <float 0-1>,
    "recommendation": "accept|revise|reject",
    "summary": "<string>"
}}
"""

AUTHOR_RESPONSE_PROMPT = """\
You are the author of a research paper. Respond to the following reviewer comments.
For each comment, explain how you will address it or why it does not apply.

Title: {title}
Review comments:
{comments}

Respond with JSON:
{{
    "responses": [
        {{"comment_index": <int>, "response": "<string>", "action": "will_fix|already_addressed|respectfully_disagree"}}
    ],
    "revision_plan": "<string>"
}}
"""

AREA_CHAIR_PROMPT = """\
You are an area chair making a final decision on a paper submission.

Title: {title}
Round: {round_number}

Reviewer scores and summaries:
{reviewer_summaries}

Author responses:
{author_responses}

Make a decision: accept, revise, or reject.

Respond with JSON:
{{
    "decision": "accept|revise|reject",
    "meta_review": "<string>",
    "overall_score": <float 0-1>,
    "key_concerns": ["<string>"]
}}
"""


class SimulatedPeerReview:
    """Runs simulated peer review with a 3-agent reviewer panel."""

    def __init__(self, llm: LLMClient, config: PeerReviewConfig) -> None:
        self._llm = llm
        self._config = config

    async def review(self, thread: ResearchThread) -> PeerReviewResult:
        """Run the full review process, potentially multiple rounds."""
        current_round = 1
        result: PeerReviewResult | None = None

        while current_round <= self._config.max_revision_rounds:
            result = await self._run_review_round(thread, current_round)
            thread.review_history.append({
                "round": str(current_round),
                "decision": result.decision.value,
                "score": str(result.overall_score),
            })

            if result.decision == ReviewDecision.ACCEPT:
                logger.info("peer_review_accepted", thread_id=thread.id, round=current_round)
                break

            if result.decision == ReviewDecision.REJECT:
                logger.info("peer_review_rejected", thread_id=thread.id, round=current_round)
                break

            # Revise: generate author response and loop
            await self._generate_revision(thread, result)
            thread.revision_count += 1
            current_round += 1

        if result is None:
            result = PeerReviewResult(
                thread_id=thread.id,
                decision=ReviewDecision.REJECT,
                overall_score=0.0,
                meta_review="No review completed.",
            )

        return result

    async def quality_gate_check(self, thread: ResearchThread) -> bool:
        """Quick quality gate before full peer review."""
        required_sections = ["abstract", "introduction", "methodology"]
        for section in required_sections:
            if section not in thread.draft_sections:
                logger.info("quality_gate_failed", thread_id=thread.id, missing=section)
                return False
        if not thread.hypothesis_ids:
            logger.info("quality_gate_failed", thread_id=thread.id, reason="no_hypotheses")
            return False
        return True

    async def _run_review_round(
        self, thread: ResearchThread, round_number: int
    ) -> PeerReviewResult:
        """Run a single round of peer review."""
        content = self._format_paper(thread)
        all_comments: list[ReviewComment] = []
        reviewer_summaries: list[str] = []

        for profile in REVIEWER_PROFILES[:self._config.num_reviewers]:
            review_data = await self._get_reviewer_feedback(
                profile, thread.title, content
            )
            for comment_data in review_data.get("comments", []):
                all_comments.append(ReviewComment(
                    reviewer_id=profile["id"],
                    aspect=comment_data.get("aspect", "general"),
                    comment=comment_data.get("comment", ""),
                    severity=comment_data.get("severity", "minor"),
                    suggestion=comment_data.get("suggestion", ""),
                ))
            reviewer_summaries.append(
                f"Reviewer ({profile['focus']}): score={review_data.get('overall_score', 0.5)}, "
                f"rec={review_data.get('recommendation', 'revise')}, "
                f"summary={review_data.get('summary', 'N/A')}"
            )

        # Generate author response
        author_responses = await self._get_author_response(thread.title, all_comments)

        # Area chair decision
        decision_data = await self._area_chair_decision(
            thread.title, round_number, reviewer_summaries, author_responses
        )

        decision_str = decision_data.get("decision", "revise")
        try:
            decision = ReviewDecision(decision_str)
        except ValueError:
            decision = ReviewDecision.REVISE

        return PeerReviewResult(
            thread_id=thread.id,
            decision=decision,
            overall_score=float(decision_data.get("overall_score", 0.5)),
            reviews=all_comments,
            meta_review=decision_data.get("meta_review", ""),
            round_number=round_number,
        )

    async def _get_reviewer_feedback(
        self, profile: dict[str, str], title: str, content: str
    ) -> dict[str, Any]:
        prompt = REVIEW_PROMPT.format(title=title, content=content[:6000])
        try:
            return await self._llm.generate_structured(
                prompt=prompt, system=profile["system"], temperature=0.4
            )
        except Exception:
            logger.exception("reviewer_feedback_failed", reviewer=profile["id"])
            return {"comments": [], "overall_score": 0.5, "recommendation": "revise", "summary": "Review failed."}

    async def _get_author_response(
        self, title: str, comments: list[ReviewComment]
    ) -> str:
        comments_text = "\n".join(
            f"{i+1}. [{c.severity}] ({c.aspect}) {c.comment}"
            for i, c in enumerate(comments)
        )
        prompt = AUTHOR_RESPONSE_PROMPT.format(title=title, comments=comments_text)
        try:
            result = await self._llm.generate_structured(
                prompt=prompt,
                system="You are a research paper author responding to peer review.",
                temperature=0.3,
            )
            return result.get("revision_plan", "Will address all comments.")
        except Exception:
            logger.exception("author_response_failed")
            return "Will address reviewer comments in revision."

    async def _area_chair_decision(
        self,
        title: str,
        round_number: int,
        reviewer_summaries: list[str],
        author_responses: str,
    ) -> dict[str, Any]:
        prompt = AREA_CHAIR_PROMPT.format(
            title=title,
            round_number=round_number,
            reviewer_summaries="\n".join(reviewer_summaries),
            author_responses=author_responses,
        )
        try:
            return await self._llm.generate_structured(
                prompt=prompt,
                system="You are a fair and thorough area chair.",
                temperature=0.3,
            )
        except Exception:
            logger.exception("area_chair_decision_failed")
            return {"decision": "revise", "meta_review": "Decision process failed.", "overall_score": 0.5}

    async def _generate_revision(
        self, thread: ResearchThread, review: PeerReviewResult
    ) -> None:
        """Apply revision suggestions to the thread's draft sections."""
        major_issues = [c for c in review.reviews if c.severity in ("major", "critical")]
        if not major_issues:
            return

        issues_text = "\n".join(
            f"- [{c.severity}] {c.comment} (suggestion: {c.suggestion})"
            for c in major_issues
        )
        for section_name, section_content in thread.draft_sections.items():
            prompt = (
                f"Revise this section based on reviewer feedback.\n\n"
                f"Section: {section_name}\nContent:\n{section_content[:3000]}\n\n"
                f"Issues to address:\n{issues_text}\n\n"
                f"Provide the revised section text."
            )
            try:
                response = await self._llm.generate(
                    prompt=prompt,
                    system="You are a skilled research writer revising a paper based on peer review.",
                    temperature=0.3,
                )
                thread.draft_sections[section_name] = response.content
            except Exception:
                logger.exception("revision_failed", section=section_name)

    @staticmethod
    def _format_paper(thread: ResearchThread) -> str:
        parts = []
        for section in ["abstract", "introduction", "methodology", "results", "discussion", "conclusion"]:
            if section in thread.draft_sections:
                parts.append(f"## {section.title()}\n{thread.draft_sections[section]}")
        return "\n\n".join(parts) if parts else f"Title: {thread.title}"
