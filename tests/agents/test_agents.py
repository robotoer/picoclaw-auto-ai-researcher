"""Tests for all specialized agent classes."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from auto_researcher.agents.critic import Critic, CritiqueItem
from auto_researcher.agents.experiment_designer import ExperimentDesigner
from auto_researcher.agents.hypothesis_generator import HypothesisGenerator
from auto_researcher.agents.literature_analyst import LiteratureAnalyst
from auto_researcher.agents.science_communicator import ScienceCommunicator
from auto_researcher.agents.statistician import Statistician
from auto_researcher.agents.synthesizer import Synthesizer
from auto_researcher.config import ResearchConfig
from auto_researcher.models.agent import AgentMessage, AgentRole, AgentState
from auto_researcher.utils.llm import LLMClient, LLMResponse


@pytest.fixture
def config() -> ResearchConfig:
    return ResearchConfig()


@pytest.fixture
def mock_llm() -> LLMClient:
    llm = MagicMock(spec=LLMClient)
    llm.generate = AsyncMock(return_value=LLMResponse(content="test", model="test"))
    llm.generate_structured = AsyncMock(return_value={})
    return llm


def _make_task(
    task_type: str,
    payload: dict[str, Any] | None = None,
    sender: AgentRole = AgentRole.ORCHESTRATOR,
) -> AgentMessage:
    """Create an AgentMessage. All payload values must conform to the model type."""
    return AgentMessage(
        sender=sender,
        task_type=task_type,
        payload=payload or {},
        message_id="test-msg-1",
    )


# ── LiteratureAnalyst ────────────────────────────────────────────


class TestLiteratureAnalyst:
    @pytest.fixture
    def agent(self, config: ResearchConfig, mock_llm: LLMClient) -> LiteratureAnalyst:
        return LiteratureAnalyst(config, mock_llm)

    def test_role(self, agent: LiteratureAnalyst) -> None:
        assert agent.role == AgentRole.LITERATURE_ANALYST

    @pytest.mark.asyncio
    async def test_analyze_paper(self, agent: LiteratureAnalyst, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "summary": "A paper about X",
            "key_claims": [
                {
                    "entity_1": "method_a",
                    "relation": "outperforms",
                    "entity_2": "method_b",
                    "conditions": "on dataset X",
                    "confidence": 0.9,
                }
            ],
            "methodology": "experimental",
            "key_results": ["result1"],
            "limitations": ["limit1"],
            "open_questions": ["q1"],
            "key_concepts": ["concept1"],
            "connections": [],
        })

        task = _make_task("analyze_paper", {
            "paper_text": "Full paper text here...",
            "title": "Test Paper",
            "paper_id": "paper-1",
        })
        result = await agent.execute(task)
        assert result.task_type == "analyze_paper_result"
        assert len(agent._episodic_memory) == 1

    @pytest.mark.asyncio
    async def test_extract_claims(self, agent: LiteratureAnalyst, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "claims": [
                {
                    "entity_1": "transformer",
                    "relation": "outperforms",
                    "entity_2": "rnn",
                    "conditions": "on NMT",
                    "confidence": 0.85,
                    "evidence_strength": "strong",
                },
                {
                    "entity_1": "attention",
                    "relation": "enables",
                    "entity_2": "parallelism",
                    "conditions": "",
                    "confidence": 0.7,
                    "evidence_strength": "moderate",
                },
            ],
        })

        task = _make_task("extract_claims", {
            "paper_text": "Text...",
            "paper_id": "paper-2",
            "focus_areas": ["transformers"],
        })
        result = await agent.execute(task)
        cached = agent.get_cached_claims("paper-2")
        assert len(cached) == 2

    @pytest.mark.asyncio
    async def test_extract_claims_invalid_relation_defaults(
        self, agent: LiteratureAnalyst, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "claims": [
                {
                    "entity_1": "a",
                    "relation": "invalid_relation",
                    "entity_2": "b",
                    "confidence": 0.5,
                },
            ],
        })

        task = _make_task("extract_claims", {"paper_text": "x", "paper_id": "p3"})
        await agent.execute(task)
        claims = agent.get_cached_claims("p3")
        assert len(claims) == 1
        assert claims[0].relation.value == "supports"  # default fallback

    @pytest.mark.asyncio
    async def test_identify_gaps(self, agent: LiteratureAnalyst, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "gaps": [{"description": "gap1"}],
            "contradictions": [],
            "missing_connections": [],
        })

        task = _make_task("identify_gaps", {
            "paper_summaries": ["summary1", "summary2"],
            "domain": "NLP",
        })
        result = await agent.execute(task)
        assert result.task_type == "identify_gaps_result"

    @pytest.mark.asyncio
    async def test_update_controversy_map(
        self, agent: LiteratureAnalyst, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "status": "open",
            "balance_assessment": "balanced",
            "key_unresolved_points": [],
            "confidence_in_assessment": 0.6,
        })

        task = _make_task("update_controversy_map", {
            "topic": "scaling laws",
            "evidence": {"paper": "paper1", "finding": "finding1"},
            "side": "pro",
        })
        result = await agent.execute(task)
        assert "scaling laws" in agent.get_controversy_map()

    @pytest.mark.asyncio
    async def test_answer_question(self, agent: LiteratureAnalyst, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "answer": "The answer is...",
            "confidence": 0.7,
            "supporting_claims": [],
            "caveats": [],
            "follow_up_questions": [],
        })

        task = _make_task("answer_question", {"question": "What is X?"})
        result = await agent.execute(task)
        assert result.task_type == "answer_question_result"

    @pytest.mark.asyncio
    async def test_unknown_task_type(self, agent: LiteratureAnalyst) -> None:
        task = _make_task("nonexistent_task")
        result = await agent.execute(task)
        assert result.task_type == "error"

    def test_get_cached_claims_all(self, agent: LiteratureAnalyst) -> None:
        assert agent.get_cached_claims() == []


# ── HypothesisGenerator ──────────────────────────────────────────


class TestHypothesisGenerator:
    @pytest.fixture
    def agent(self, config: ResearchConfig, mock_llm: LLMClient) -> HypothesisGenerator:
        return HypothesisGenerator(config, mock_llm)

    def test_role(self, agent: HypothesisGenerator) -> None:
        assert agent.role == AgentRole.HYPOTHESIS_GENERATOR

    @pytest.mark.asyncio
    async def test_generate_hypotheses(
        self, agent: HypothesisGenerator, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "hypotheses": [
                {
                    "entity_1": "method_x",
                    "relation": "outperforms",
                    "entity_2": "method_y",
                    "conditions": "on task Z",
                    "confidence": 0.6,
                    "rationale": "because...",
                    "supporting_evidence": ["ev1"],
                    "counter_evidence": [],
                    "falsification_criteria": [
                        {
                            "description": "test desc",
                            "test_method": "experiment",
                            "expected_outcome_if_true": "higher score",
                            "expected_outcome_if_false": "no difference",
                        }
                    ],
                    "granularity": "medium",
                    "novelty_reasoning": "novel because...",
                },
            ],
        })

        task = _make_task("generate_hypotheses", {
            "gaps": ["gap1"],
            "claims": ["claim1"],
            "num_hypotheses": "1",
        })
        result = await agent.execute(task)
        generated = agent.get_generated_hypotheses()
        assert len(generated) == 1
        assert generated[0].entity_1 == "method_x"
        assert len(generated[0].falsification_criteria) == 1

    @pytest.mark.asyncio
    async def test_score_novelty(
        self, agent: HypothesisGenerator, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "novelty_score": 0.8,
            "novelty_type": "novel_combination",
            "similar_existing": [],
            "reasoning": "unique approach",
        })

        task = _make_task("score_novelty", {
            "hypothesis": "X outperforms Y",
            "existing_hypotheses": [],
            "existing_claims": [],
        })
        result = await agent.execute(task)
        assert result.task_type == "score_novelty_result"

    @pytest.mark.asyncio
    async def test_counterfactual_reasoning(
        self, agent: HypothesisGenerator, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "implications_if_true": ["impl1"],
            "implications_if_false": ["impl2"],
            "surprising_implications": [],
            "beliefs_to_revise": [],
            "new_hypotheses": [],
        })

        task = _make_task("counterfactual_reasoning", {
            "hypothesis": "X is true",
            "known_facts": ["fact1"],
        })
        result = await agent.execute(task)
        assert result.task_type == "counterfactual_reasoning_result"

    @pytest.mark.asyncio
    async def test_rank_hypotheses(
        self, agent: HypothesisGenerator, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "ranking": [{"index": 0, "score": 0.9, "rationale": "best"}],
            "recommended_portfolio": "focused",
        })

        task = _make_task("rank_hypotheses", {"hypotheses": ["hyp1", "hyp2"]})
        result = await agent.execute(task)
        assert result.task_type == "rank_hypotheses_result"

    @pytest.mark.asyncio
    async def test_unknown_task(self, agent: HypothesisGenerator) -> None:
        task = _make_task("bogus")
        result = await agent.execute(task)
        assert result.task_type == "error"


# ── Critic ────────────────────────────────────────────────────────


class TestCritic:
    @pytest.fixture
    def agent(self, config: ResearchConfig, mock_llm: LLMClient) -> Critic:
        return Critic(config, mock_llm)

    def test_role(self, agent: Critic) -> None:
        assert agent.role == AgentRole.CRITIC

    @pytest.mark.asyncio
    async def test_critique_hypothesis(self, agent: Critic, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "critiques": [
                {"aspect": "falsifiability", "issue": "not falsifiable",
                 "severity": "critical", "suggestion": "add criteria", "evidence": ""},
            ],
            "overall_assessment": "needs_revision",
            "strongest_objection": "not falsifiable",
            "hidden_assumptions": ["assumption1"],
            "alternative_explanations": [],
            "confidence_in_critique": 0.9,
        })

        task = _make_task("critique_hypothesis", {
            "hypothesis": "X is better than Y",
            "supporting_evidence": ["ev1"],
            "falsification_criteria": ["fc1"],
        })
        result = await agent.execute(task)
        assert result.task_type == "critique_hypothesis_result"
        assert len(agent.get_critique_history()) == 1

    @pytest.mark.asyncio
    async def test_critique_experiment(self, agent: Critic, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "critiques": [],
            "missing_controls": [],
            "statistical_issues": [],
            "overall_assessment": "ready",
            "minimum_changes_required": [],
        })

        task = _make_task("critique_experiment", {
            "design": "experiment desc",
            "hypothesis": "hypothesis desc",
        })
        result = await agent.execute(task)
        assert result.task_type == "critique_experiment_result"

    @pytest.mark.asyncio
    async def test_critique_paper(self, agent: Critic, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "summary": "A paper",
            "strengths": ["strong method"],
            "weaknesses": [],
            "questions_for_authors": [],
            "missing_references": [],
            "scores": {"overall": "7"},
            "decision": "accept",
            "confidence": 0.8,
        })

        task = _make_task("critique_paper", {"paper_text": "text", "paper_type": "workshop"})
        result = await agent.execute(task)
        assert result.task_type == "critique_paper_result"

    @pytest.mark.asyncio
    async def test_detect_prior_art(self, agent: Critic, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "is_novel": True,
            "prior_art": [],
            "novelty_assessment": "appears new",
            "recommendation": "proceed",
        })

        task = _make_task("detect_prior_art", {"idea": "my idea", "domain": "ML"})
        result = await agent.execute(task)
        assert result.task_type == "detect_prior_art_result"

    @pytest.mark.asyncio
    async def test_adversarial_debate(self, agent: Critic, mock_llm: LLMClient) -> None:
        async def side_effect(*args: Any, **kwargs: Any) -> dict[str, Any]:
            prompt = args[0] if args else kwargs.get("prompt", "")
            if "Attack" in str(prompt) or "attack" in str(prompt):
                return {"attack": "flaw found", "attack_type": "logical", "severity": "major"}
            elif "Defend" in str(prompt) or "defend" in str(prompt):
                return {"defense": "rebuttal", "concessions": []}
            else:
                return {
                    "verdict": "claim_weakened",
                    "surviving_strengths": [],
                    "unresolved_weaknesses": ["flaw"],
                    "confidence_after_debate": 0.4,
                    "revised_claim": "weaker claim",
                }

        mock_llm.generate_structured = AsyncMock(side_effect=side_effect)

        task = _make_task("adversarial_debate", {
            "claim": "X is true",
            "defense": "because reasons",
            "max_rounds": "2",
        })
        result = await agent.execute(task)
        assert result.task_type == "adversarial_debate_result"

    @pytest.mark.asyncio
    async def test_unknown_task(self, agent: Critic) -> None:
        task = _make_task("bogus")
        result = await agent.execute(task)
        assert result.task_type == "error"


class TestCritiqueItem:
    def test_to_dict(self) -> None:
        item = CritiqueItem(
            aspect="methodology",
            issue="missing control",
            severity="major",
            suggestion="add control group",
            evidence="section 4",
        )
        d = item.to_dict()
        assert d["aspect"] == "methodology"
        assert d["severity"] == "major"


# ── ExperimentDesigner ────────────────────────────────────────────


class TestExperimentDesigner:
    @pytest.fixture
    def agent(self, config: ResearchConfig, mock_llm: LLMClient) -> ExperimentDesigner:
        return ExperimentDesigner(config, mock_llm)

    def test_role(self, agent: ExperimentDesigner) -> None:
        assert agent.role == AgentRole.EXPERIMENT_DESIGNER

    @pytest.mark.asyncio
    async def test_design_experiment(
        self, agent: ExperimentDesigner, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "description": "Experiment to test X",
            "methodology": "steps...",
            "datasets": ["dataset1"],
            "models": ["model1"],
            "metrics": ["accuracy"],
            "controls": ["baseline"],
            "confounds": ["confound1"],
            "estimated_compute_hours": 10.0,
            "expected_outcomes": {"if_true": "higher acc"},
            "statistical_power": 0.8,
        })

        task = _make_task("design_experiment", {
            "hypothesis": "X outperforms Y",
            "hypothesis_id": "h1",
        })
        result = await agent.execute(task)
        assert result.task_type == "design_experiment_result"
        designs = agent.get_designs()
        assert len(designs) == 1
        assert designs[0].hypothesis_id == "h1"

    @pytest.mark.asyncio
    async def test_power_analysis(
        self, agent: ExperimentDesigner, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "required_sample_size": 100,
            "total_sample_size": 200,
            "actual_power": 0.82,
        })

        task = _make_task("power_analysis", {"effect_size": "medium"})
        result = await agent.execute(task)
        assert result.task_type == "power_analysis_result"

    @pytest.mark.asyncio
    async def test_simulate_outcomes(
        self, agent: ExperimentDesigner, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "scenarios": [{"name": "confirmed", "probability": 0.6}],
        })

        task = _make_task("simulate_outcomes", {
            "experiment_description": "desc",
            "hypothesis": "hyp",
        })
        result = await agent.execute(task)
        assert result.task_type == "simulate_outcomes_result"

    @pytest.mark.asyncio
    async def test_estimate_compute(
        self, agent: ExperimentDesigner, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "total_gpu_hours": 50,
        })

        task = _make_task("estimate_compute", {"models": ["gpt"], "datasets": ["squad"]})
        result = await agent.execute(task)
        assert result.task_type == "estimate_compute_result"

    @pytest.mark.asyncio
    async def test_generate_code(
        self, agent: ExperimentDesigner, mock_llm: LLMClient
    ) -> None:
        # First create a design so code can be attached
        mock_llm.generate_structured = AsyncMock(return_value={
            "description": "d", "methodology": "m", "datasets": [],
            "models": [], "metrics": [], "controls": [], "confounds": [],
            "estimated_compute_hours": 1.0, "expected_outcomes": {},
        })
        await agent.execute(_make_task("design_experiment", {
            "hypothesis": "h", "hypothesis_id": "h1",
        }))

        mock_llm.generate_structured = AsyncMock(return_value={
            "code": "print('test')",
            "requirements": [],
            "usage_instructions": "",
            "expected_outputs": [],
        })
        task = _make_task("generate_code", {"design": {"desc": "test"}, "framework": "pytorch"})
        result = await agent.execute(task)
        assert result.task_type == "generate_code_result"
        assert agent.get_designs()[-1].code == "print('test')"

    @pytest.mark.asyncio
    async def test_information_gain(
        self, agent: ExperimentDesigner, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "experiments": [],
            "optimal_ordering": [],
            "rationale": "reason",
        })

        task = _make_task("information_gain", {
            "experiments": ["exp1"],
            "hypotheses": ["hyp1"],
        })
        result = await agent.execute(task)
        assert result.task_type == "information_gain_result"

    @pytest.mark.asyncio
    async def test_unknown_task(self, agent: ExperimentDesigner) -> None:
        task = _make_task("bogus")
        result = await agent.execute(task)
        assert result.task_type == "error"


# ── Synthesizer ───────────────────────────────────────────────────


class TestSynthesizer:
    @pytest.fixture
    def agent(self, config: ResearchConfig, mock_llm: LLMClient) -> Synthesizer:
        return Synthesizer(config, mock_llm)

    def test_role(self, agent: Synthesizer) -> None:
        assert agent.role == AgentRole.SYNTHESIZER

    @pytest.mark.asyncio
    async def test_find_connections(self, agent: Synthesizer, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "connections": [{"concept_a": "a", "concept_b": "b", "strength": 0.8}],
            "shared_structures": [],
            "transfer_opportunities": [],
            "novel_combinations": [],
            "confidence": 0.7,
        })

        task = _make_task("find_connections", {
            "domain_a": "NLP",
            "domain_b": "CV",
            "concepts_a": ["attention"],
            "concepts_b": ["convolution"],
        })
        result = await agent.execute(task)
        assert result.task_type == "find_connections_result"

    @pytest.mark.asyncio
    async def test_detect_analogies(self, agent: Synthesizer, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "analogies": [
                {"finding_a_index": 0, "finding_b_index": 1, "shared_structure": "subset selection"}
            ],
            "meta_patterns": [],
            "suggested_abstractions": [],
            "novel_predictions": [],
        })

        task = _make_task("detect_analogies", {"findings": ["f1", "f2"]})
        result = await agent.execute(task)
        assert result.task_type == "detect_analogies_result"
        assert len(agent.get_analogy_database()) == 1

    @pytest.mark.asyncio
    async def test_meta_analysis(self, agent: Synthesizer, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "overall_finding": "positive effect",
            "effect_direction": "positive",
        })

        task = _make_task("meta_analysis", {
            "results": ["r1", "r2"],
            "research_question": "Does X work?",
        })
        result = await agent.execute(task)
        assert result.task_type == "meta_analysis_result"
        assert "Does X work?" in agent.get_synthesis_cache()

    @pytest.mark.asyncio
    async def test_write_survey(self, agent: Synthesizer, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "title": "Survey of X",
            "abstract": "abstract",
            "sections": [],
        })

        task = _make_task("write_survey", {"topic": "transformers", "papers": ["p1"]})
        result = await agent.execute(task)
        assert result.task_type == "write_survey_result"

    @pytest.mark.asyncio
    async def test_identify_patterns(self, agent: Synthesizer, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "patterns": [],
            "convergences": [],
            "divergences": [],
        })

        task = _make_task("identify_patterns", {"threads": ["t1", "t2"]})
        result = await agent.execute(task)
        assert result.task_type == "identify_patterns_result"

    @pytest.mark.asyncio
    async def test_unknown_task(self, agent: Synthesizer) -> None:
        task = _make_task("bogus")
        result = await agent.execute(task)
        assert result.task_type == "error"


# ── ScienceCommunicator ──────────────────────────────────────────


class TestScienceCommunicator:
    @pytest.fixture
    def agent(self, config: ResearchConfig, mock_llm: LLMClient) -> ScienceCommunicator:
        return ScienceCommunicator(config, mock_llm)

    def test_role(self, agent: ScienceCommunicator) -> None:
        assert agent.role == AgentRole.SCIENCE_COMMUNICATOR

    @pytest.mark.asyncio
    async def test_write_paper(self, agent: ScienceCommunicator, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "title": "My Paper",
            "abstract": "We show...",
            "sections": {"intro": "Introduction text"},
            "key_contributions": ["contrib1"],
            "claim_strength": "moderate",
        })

        task = _make_task("write_paper", {
            "title": "My Paper",
            "hypothesis": "X is better",
            "methodology": "experiment",
        })
        result = await agent.execute(task)
        assert result.task_type == "write_paper_result"
        assert "My Paper" in agent.get_drafts()

    @pytest.mark.asyncio
    async def test_write_paper_with_revision(
        self, agent: ScienceCommunicator, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "title": "Revised",
            "abstract": "revised abstract",
            "sections": {},
            "key_contributions": [],
            "claim_strength": "conservative",
        })

        task = _make_task("write_paper", {
            "title": "Revised",
            "hypothesis": "X",
            "methodology": "m",
            "review_feedback": ["fix section 3"],
        })
        result = await agent.execute(task)
        assert result.task_type == "write_paper_result"

    @pytest.mark.asyncio
    async def test_write_blog_post(
        self, agent: ScienceCommunicator, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "title": "Cool Finding",
            "subtitle": "subtitle",
            "content": "blog text",
            "tldr": "summary",
            "suggested_figures": [],
        })

        task = _make_task("write_blog_post", {"findings": "we found X"})
        result = await agent.execute(task)
        assert result.task_type == "write_blog_post_result"

    @pytest.mark.asyncio
    async def test_write_grant_proposal(
        self, agent: ScienceCommunicator, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "project_title": "Grant Title",
            "project_summary": "summary",
            "sections": {},
        })

        task = _make_task("write_grant_proposal", {
            "research_direction": "direction",
            "preliminary_results": ["result1"],
        })
        result = await agent.execute(task)
        assert result.task_type == "write_grant_proposal_result"

    @pytest.mark.asyncio
    async def test_reframe_for_audience(
        self, agent: ScienceCommunicator, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "reframed_content": "simplified text",
            "key_adaptations": [],
            "analogies_used": [],
            "accuracy_notes": [],
        })

        task = _make_task("reframe_for_audience", {
            "content": "technical text",
            "target_audience": "general public",
        })
        result = await agent.execute(task)
        assert result.task_type == "reframe_for_audience_result"

    @pytest.mark.asyncio
    async def test_improve_writing(
        self, agent: ScienceCommunicator, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "improved_text": "better text",
            "changes": [],
            "overall_assessment": "good",
        })

        task = _make_task("improve_writing", {"text": "rough text"})
        result = await agent.execute(task)
        assert result.task_type == "improve_writing_result"

    @pytest.mark.asyncio
    async def test_unknown_task(self, agent: ScienceCommunicator) -> None:
        task = _make_task("bogus")
        result = await agent.execute(task)
        assert result.task_type == "error"


# ── Statistician ──────────────────────────────────────────────────


class TestStatistician:
    @pytest.fixture
    def agent(self, config: ResearchConfig, mock_llm: LLMClient) -> Statistician:
        return Statistician(config, mock_llm)

    def test_role(self, agent: Statistician) -> None:
        assert agent.role == AgentRole.STATISTICIAN

    @pytest.mark.asyncio
    async def test_interpret_results(self, agent: Statistician, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "interpretation": "The results show...",
            "conclusion": "confirmed",
            "confidence_in_conclusion": 0.85,
        })

        task = _make_task("interpret_results", {
            "metrics": {"accuracy": "0.95"},
            "hypothesis": "X > Y",
        })
        result = await agent.execute(task)
        assert result.task_type == "interpret_results_result"
        assert len(agent.get_analyses()) == 1

    @pytest.mark.asyncio
    async def test_power_analysis(self, agent: Statistician, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "required_n_per_group": 50,
            "total_n": 100,
        })

        task = _make_task("power_analysis", {"test_type": "t-test", "effect_size": "0.5"})
        result = await agent.execute(task)
        assert result.task_type == "power_analysis_result"

    @pytest.mark.asyncio
    async def test_effect_size(self, agent: Statistician, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "cohens_d": 0.8,
            "cohens_d_interpretation": "large",
        })

        task = _make_task("effect_size", {
            "metric_name": "accuracy",
            "treatment_values": ["0.9", "0.91", "0.92"],
            "control_values": ["0.8", "0.81", "0.82"],
        })
        result = await agent.execute(task)
        assert result.task_type == "effect_size_result"

    @pytest.mark.asyncio
    async def test_multiple_comparisons(self, agent: Statistician, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "bonferroni": {"adjusted_alpha": 0.0167},
            "recommendation": "use Holm",
        })

        task = _make_task("multiple_comparisons", {
            "comparisons": ["A vs B", "A vs C", "B vs C"],
        })
        result = await agent.execute(task)
        assert result.task_type == "multiple_comparisons_result"

    @pytest.mark.asyncio
    async def test_confidence_intervals(self, agent: Statistician, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "ci_lower": 0.85,
            "ci_upper": 0.95,
        })

        task = _make_task("confidence_intervals", {
            "estimate": "0.90",
            "data_summary": {"n": "100", "std": "0.05"},
        })
        result = await agent.execute(task)
        assert result.task_type == "confidence_intervals_result"

    @pytest.mark.asyncio
    async def test_detect_issues(self, agent: Statistician, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "issues": [{"issue": "p-hacking", "severity": "major"}],
            "overall_statistical_quality": "acceptable",
        })

        task = _make_task("detect_issues", {
            "results": {"p_value": "0.049"},
            "methodology": "desc",
            "claims": ["X works"],
        })
        result = await agent.execute(task)
        assert result.task_type == "detect_issues_result"

    @pytest.mark.asyncio
    async def test_recommend_test(self, agent: Statistician, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "recommended_test": "Mann-Whitney U",
            "alternatives": [],
        })

        task = _make_task("recommend_test", {
            "research_question": "Is A > B?",
            "data_type": "ordinal",
        })
        result = await agent.execute(task)
        assert result.task_type == "recommend_test_result"

    @pytest.mark.asyncio
    async def test_unknown_task(self, agent: Statistician) -> None:
        task = _make_task("bogus")
        result = await agent.execute(task)
        assert result.task_type == "error"
