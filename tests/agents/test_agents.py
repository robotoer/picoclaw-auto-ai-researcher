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
    """Create an AgentMessage with string-only payload values."""
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
        # The execute method creates an AgentMessage with the LLM result as payload.
        # Since the LLM result contains non-string values (floats, nested dicts),
        # we test via direct method call to avoid pydantic payload validation.
        result_payload = await agent._analyze_paper(task.payload)
        assert "summary" in result_payload
        assert result_payload.get("parsed_claim_count") == 1
        assert len(agent._episodic_memory) == 1
        cached = agent.get_cached_claims("paper-1")
        assert len(cached) == 1

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

        result = await agent._extract_claims({
            "paper_text": "Text...",
            "paper_id": "paper-2",
            "focus_areas": ["transformers"],
        })
        assert result["claim_count"] == 2
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

        await agent._extract_claims({"paper_text": "x", "paper_id": "p3"})
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

        result = await agent._identify_gaps({
            "paper_summaries": ["summary1", "summary2"],
            "domain": "NLP",
        })
        assert "gaps" in result

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

        result = await agent._update_controversy_map({
            "topic": "scaling laws",
            "evidence": {"paper": "paper1", "finding": "finding1"},
            "side": "pro",
        })
        assert "scaling laws" in agent.get_controversy_map()
        assert result["topic"] == "scaling laws"

    @pytest.mark.asyncio
    async def test_answer_question(self, agent: LiteratureAnalyst, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "answer": "The answer is...",
            "confidence": 0.7,
            "supporting_claims": [],
            "caveats": [],
            "follow_up_questions": [],
        })

        result = await agent._answer_question({"question": "What is X?"})
        assert result["answer"] == "The answer is..."

    @pytest.mark.asyncio
    async def test_unknown_task_type(self, agent: LiteratureAnalyst) -> None:
        task = _make_task("nonexistent_task")
        result = await agent.execute(task)
        assert result.task_type == "error"

    def test_get_cached_claims_all(self, agent: LiteratureAnalyst) -> None:
        assert agent.get_cached_claims() == []

    def test_parse_claims(self, agent: LiteratureAnalyst) -> None:
        claims_data = [
            {"entity_1": "A", "relation": "outperforms", "entity_2": "B", "confidence": 0.9},
            {"entity_1": "C", "relation": "bad_rel", "entity_2": "D"},
        ]
        claims = agent._parse_claims(claims_data, "paper-x")
        assert len(claims) == 2
        assert claims[0].relation.value == "outperforms"
        assert claims[1].relation.value == "supports"  # fallback
        assert claims[0].source_paper_ids == ["paper-x"]


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

        result = await agent._generate_hypotheses({
            "gaps": ["gap1"],
            "claims": ["claim1"],
            "num_hypotheses": 1,
        })
        assert result["count"] == 1
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

        result = await agent._score_novelty({
            "hypothesis": "X outperforms Y",
            "existing_hypotheses": [],
            "existing_claims": [],
        })
        assert result["novelty_score"] == 0.8

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

        result = await agent._counterfactual_reasoning({
            "hypothesis": "X is true",
            "known_facts": ["fact1"],
        })
        assert "implications_if_true" in result

    @pytest.mark.asyncio
    async def test_rank_hypotheses(
        self, agent: HypothesisGenerator, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "ranking": [{"index": 0, "score": 0.9, "rationale": "best"}],
            "recommended_portfolio": "focused",
        })

        result = await agent._rank_hypotheses({"hypotheses": ["hyp1", "hyp2"]})
        assert "ranking" in result

    @pytest.mark.asyncio
    async def test_unknown_task(self, agent: HypothesisGenerator) -> None:
        task = _make_task("bogus")
        result = await agent.execute(task)
        assert result.task_type == "error"

    def test_parse_hypotheses(self, agent: HypothesisGenerator) -> None:
        data = [
            {
                "entity_1": "A",
                "relation": "improves",
                "entity_2": "B",
                "conditions": "when X",
                "confidence": 0.7,
                "rationale": "reason",
                "falsification_criteria": [
                    {
                        "description": "d",
                        "test_method": "m",
                        "expected_outcome_if_true": "t",
                        "expected_outcome_if_false": "f",
                    }
                ],
                "granularity": "high_risk",
            },
        ]
        hypotheses = agent._parse_hypotheses(data)
        assert len(hypotheses) == 1
        assert hypotheses[0].entity_1 == "A"
        assert hypotheses[0].granularity == "high_risk"
        assert len(hypotheses[0].falsification_criteria) == 1


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

        result = await agent._critique_hypothesis({
            "hypothesis": "X is better than Y",
            "supporting_evidence": ["ev1"],
            "falsification_criteria": ["fc1"],
        })
        assert result["overall_assessment"] == "needs_revision"

    @pytest.mark.asyncio
    async def test_critique_experiment(self, agent: Critic, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "critiques": [],
            "missing_controls": [],
            "statistical_issues": [],
            "overall_assessment": "ready",
            "minimum_changes_required": [],
        })

        result = await agent._critique_experiment({
            "design": "experiment desc",
            "hypothesis": "hypothesis desc",
        })
        assert result["overall_assessment"] == "ready"

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

        result = await agent._critique_paper({"paper_text": "text", "paper_type": "workshop"})
        assert result["decision"] == "accept"

    @pytest.mark.asyncio
    async def test_detect_prior_art(self, agent: Critic, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "is_novel": True,
            "prior_art": [],
            "novelty_assessment": "appears new",
            "recommendation": "proceed",
        })

        result = await agent._detect_prior_art({"idea": "my idea", "domain": "ML"})
        assert result["is_novel"] is True

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

        result = await agent._adversarial_debate({
            "claim": "X is true",
            "defense": "because reasons",
            "max_rounds": 2,
        })
        assert len(result["rounds"]) == 2
        assert "verdict" in result

    @pytest.mark.asyncio
    async def test_execute_stores_critique_history(self, agent: Critic, mock_llm: LLMClient) -> None:
        # Use a task type that returns string-only payload
        mock_llm.generate_structured = AsyncMock(return_value={
            "critiques": [],
            "missing_controls": [],
            "statistical_issues": [],
            "overall_assessment": "ready",
            "minimum_changes_required": [],
        })

        task = _make_task("critique_experiment", {
            "design": "desc",
            "hypothesis": "hyp",
        })
        await agent.execute(task)
        assert len(agent.get_critique_history()) == 1

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

        result = await agent._design_experiment({
            "hypothesis": "X outperforms Y",
            "hypothesis_id": "h1",
        })
        assert "design" in result
        designs = agent.get_designs()
        assert len(designs) == 1
        assert designs[0].hypothesis_id == "h1"
        assert designs[0].estimated_compute_hours == 10.0

    @pytest.mark.asyncio
    async def test_power_analysis(
        self, agent: ExperimentDesigner, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "required_sample_size": 100,
            "total_sample_size": 200,
            "actual_power": 0.82,
        })

        result = await agent._power_analysis({"effect_size": "medium"})
        assert result["required_sample_size"] == 100

    @pytest.mark.asyncio
    async def test_simulate_outcomes(
        self, agent: ExperimentDesigner, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "scenarios": [{"name": "confirmed", "probability": 0.6}],
        })

        result = await agent._simulate_outcomes({
            "experiment_description": "desc",
            "hypothesis": "hyp",
        })
        assert len(result["scenarios"]) == 1

    @pytest.mark.asyncio
    async def test_estimate_compute(
        self, agent: ExperimentDesigner, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "total_gpu_hours": 50,
        })

        result = await agent._estimate_compute({"models": ["gpt"], "datasets": ["squad"]})
        assert result["total_gpu_hours"] == 50

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
        await agent._design_experiment({"hypothesis": "h", "hypothesis_id": "h1"})

        mock_llm.generate_structured = AsyncMock(return_value={
            "code": "print('test')",
            "requirements": [],
            "usage_instructions": "",
            "expected_outputs": [],
        })
        result = await agent._generate_code({"design": {}, "framework": "pytorch"})
        assert result["code"] == "print('test')"
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

        result = await agent._information_gain({
            "experiments": ["exp1"],
            "hypotheses": ["hyp1"],
        })
        assert result["rationale"] == "reason"

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

        result = await agent._find_connections({
            "domain_a": "NLP",
            "domain_b": "CV",
            "concepts_a": ["attention"],
            "concepts_b": ["convolution"],
        })
        assert len(result["connections"]) == 1

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

        result = await agent._detect_analogies({"findings": ["f1", "f2"]})
        assert len(result["analogies"]) == 1
        assert len(agent.get_analogy_database()) == 1

    @pytest.mark.asyncio
    async def test_meta_analysis(self, agent: Synthesizer, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "overall_finding": "positive effect",
            "effect_direction": "positive",
        })

        result = await agent._meta_analysis({
            "results": ["r1", "r2"],
            "research_question": "Does X work?",
        })
        assert result["overall_finding"] == "positive effect"
        assert "Does X work?" in agent.get_synthesis_cache()

    @pytest.mark.asyncio
    async def test_write_survey(self, agent: Synthesizer, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "title": "Survey of X",
            "abstract": "abstract",
            "sections": [],
        })

        result = await agent._write_survey({"topic": "transformers", "papers": ["p1"]})
        assert result["title"] == "Survey of X"

    @pytest.mark.asyncio
    async def test_identify_patterns(self, agent: Synthesizer, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "patterns": [],
            "convergences": [],
            "divergences": [],
        })

        result = await agent._identify_patterns({"threads": ["t1", "t2"]})
        assert "patterns" in result

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

        result = await agent._write_paper({
            "title": "My Paper",
            "hypothesis": "X is better",
            "methodology": "experiment",
        })
        assert result["title"] == "My Paper"
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

        result = await agent._write_paper({
            "title": "Revised",
            "hypothesis": "X",
            "methodology": "m",
            "review_feedback": ["fix section 3"],
        })
        assert result["claim_strength"] == "conservative"

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

        result = await agent._write_blog_post({"findings": "we found X"})
        assert result["title"] == "Cool Finding"

    @pytest.mark.asyncio
    async def test_write_grant_proposal(
        self, agent: ScienceCommunicator, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "project_title": "Grant Title",
            "project_summary": "summary",
            "sections": {},
        })

        result = await agent._write_grant_proposal({
            "research_direction": "direction",
            "preliminary_results": ["result1"],
        })
        assert result["project_title"] == "Grant Title"

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

        result = await agent._reframe_for_audience({
            "content": "technical text",
            "target_audience": "general public",
        })
        assert result["reframed_content"] == "simplified text"

    @pytest.mark.asyncio
    async def test_improve_writing(
        self, agent: ScienceCommunicator, mock_llm: LLMClient
    ) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "improved_text": "better text",
            "changes": [],
            "overall_assessment": "good",
        })

        result = await agent._improve_writing({"text": "rough text"})
        assert result["improved_text"] == "better text"

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

        result = await agent._interpret_results({
            "metrics": {"accuracy": 0.95},
            "hypothesis": "X > Y",
        })
        assert result["conclusion"] == "confirmed"

    @pytest.mark.asyncio
    async def test_power_analysis(self, agent: Statistician, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "required_n_per_group": 50,
            "total_n": 100,
        })

        result = await agent._power_analysis({"test_type": "t-test", "effect_size": "0.5"})
        assert result["required_n_per_group"] == 50

    @pytest.mark.asyncio
    async def test_effect_size(self, agent: Statistician, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "cohens_d": 0.8,
            "cohens_d_interpretation": "large",
        })

        result = await agent._effect_size({
            "metric_name": "accuracy",
            "treatment_values": [0.9, 0.91, 0.92],
            "control_values": [0.8, 0.81, 0.82],
        })
        assert result["cohens_d"] == 0.8

    @pytest.mark.asyncio
    async def test_multiple_comparisons(self, agent: Statistician, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "bonferroni": {"adjusted_alpha": 0.0167},
            "recommendation": "use Holm",
        })

        result = await agent._multiple_comparisons({
            "comparisons": ["A vs B", "A vs C", "B vs C"],
        })
        assert result["recommendation"] == "use Holm"

    @pytest.mark.asyncio
    async def test_confidence_intervals(self, agent: Statistician, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "ci_lower": 0.85,
            "ci_upper": 0.95,
        })

        result = await agent._confidence_intervals({
            "estimate": "0.90",
            "data_summary": {"n": 100, "std": 0.05},
        })
        assert result["ci_lower"] == 0.85

    @pytest.mark.asyncio
    async def test_detect_issues(self, agent: Statistician, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "issues": [{"issue": "p-hacking", "severity": "major"}],
            "overall_statistical_quality": "acceptable",
        })

        result = await agent._detect_issues({
            "results": {"p_value": 0.049},
            "methodology": "desc",
            "claims": ["X works"],
        })
        assert result["overall_statistical_quality"] == "acceptable"

    @pytest.mark.asyncio
    async def test_recommend_test(self, agent: Statistician, mock_llm: LLMClient) -> None:
        mock_llm.generate_structured = AsyncMock(return_value={
            "recommended_test": "Mann-Whitney U",
            "alternatives": [],
        })

        result = await agent._recommend_test({
            "research_question": "Is A > B?",
            "data_type": "ordinal",
        })
        assert result["recommended_test"] == "Mann-Whitney U"

    @pytest.mark.asyncio
    async def test_execute_stores_analysis(self, agent: Statistician, mock_llm: LLMClient) -> None:
        """Test that execute properly stores analysis in history."""
        mock_llm.generate_structured = AsyncMock(return_value={
            "recommended_test": "t-test",
            "alternatives": [],
        })

        task = _make_task("recommend_test", {
            "research_question": "A vs B",
            "data_type": "continuous",
        })
        result = await agent.execute(task)
        assert result.task_type == "recommend_test_result"
        assert len(agent.get_analyses()) == 1

    @pytest.mark.asyncio
    async def test_unknown_task(self, agent: Statistician) -> None:
        task = _make_task("bogus")
        result = await agent.execute(task)
        assert result.task_type == "error"
