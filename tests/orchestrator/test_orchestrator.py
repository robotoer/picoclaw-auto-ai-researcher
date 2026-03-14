"""Tests for ResearchOrchestrator."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from auto_researcher.config import ResearchConfig
from auto_researcher.evaluation.iwpg import IWPGScorer
from auto_researcher.evaluation.peer_review import SimulatedPeerReview
from auto_researcher.evaluation.sunfire import SUNFIREEvaluator
from auto_researcher.models.gap import Gap, GapStatus, GapType
from auto_researcher.models.research_thread import ThreadStatus
from auto_researcher.orchestrator.orchestrator import ResearchOrchestrator
from auto_researcher.orchestrator.resource_manager import ResourceManager
from auto_researcher.orchestrator.task_router import Blackboard, TaskRouter
from auto_researcher.utils.llm import LLMClient


@pytest.fixture
def config() -> ResearchConfig:
    return ResearchConfig()


@pytest.fixture
def mock_llm() -> LLMClient:
    llm = MagicMock(spec=LLMClient)
    llm.generate = AsyncMock()
    llm.generate_structured = AsyncMock(return_value={
        "selected_gap_ids": ["g1"],
        "reasoning": "high priority",
    })
    return llm


@pytest.fixture
def mock_router() -> TaskRouter:
    router = MagicMock(spec=TaskRouter)
    router.route_task = AsyncMock(return_value={"result": "done"})
    router.process_pending = AsyncMock(return_value=0)
    router.blackboard = MagicMock(spec=Blackboard)
    router.blackboard.write = AsyncMock()
    router.blackboard.read = AsyncMock(return_value=None)
    router.get_agent_load = MagicMock(return_value={})
    return router


@pytest.fixture
def mock_resources(config: ResearchConfig) -> ResourceManager:
    rm = MagicMock(spec=ResourceManager)
    rm.can_start_thread = MagicMock(return_value=True)
    rm.allocate_budget = MagicMock(return_value=100.0)
    rm.remaining_budget = MagicMock(return_value=50.0)
    rm.spend = MagicMock(return_value=True)
    rm.release_thread = MagicMock()
    rm.record_insight = MagicMock()
    rm.budget_summary = MagicMock(return_value={"total_spent": 0})
    return rm


@pytest.fixture
def mock_sunfire() -> SUNFIREEvaluator:
    s = MagicMock(spec=SUNFIREEvaluator)
    s.evaluate = AsyncMock(return_value={})
    s.composite_score = MagicMock(return_value=0.75)
    return s


@pytest.fixture
def mock_iwpg() -> IWPGScorer:
    i = MagicMock(spec=IWPGScorer)
    i.compute_reward = AsyncMock(return_value={})
    i.total_reward = MagicMock(return_value=0.8)
    return i


@pytest.fixture
def mock_peer_review() -> SimulatedPeerReview:
    pr = MagicMock(spec=SimulatedPeerReview)
    pr.quality_gate_check = AsyncMock(return_value=True)
    pr.review = AsyncMock(return_value=MagicMock(
        decision=MagicMock(value="accept"),
        overall_score=0.85,
    ))
    return pr


@pytest.fixture
def orchestrator(
    config: ResearchConfig,
    mock_llm: LLMClient,
    mock_router: TaskRouter,
    mock_resources: ResourceManager,
    mock_sunfire: SUNFIREEvaluator,
    mock_iwpg: IWPGScorer,
    mock_peer_review: SimulatedPeerReview,
) -> ResearchOrchestrator:
    return ResearchOrchestrator(
        config=config,
        llm=mock_llm,
        task_router=mock_router,
        resource_manager=mock_resources,
        sunfire=mock_sunfire,
        iwpg=mock_iwpg,
        peer_review=mock_peer_review,
    )


def _make_gap(
    id: str = "g1",
    status: GapStatus = GapStatus.OPEN,
    importance: float = 0.8,
    tractability: float = 0.7,
    novelty: float = 0.6,
) -> Gap:
    return Gap(
        id=id,
        gap_type=GapType.EMPIRICAL,
        description=f"Gap {id}",
        importance=importance,
        tractability=tractability,
        novelty=novelty,
        status=status,
    )


# ── Thread Management ─────────────────────────────────────────────


class TestThreadManagement:
    def test_initialize_thread(self, orchestrator: ResearchOrchestrator) -> None:
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        assert thread.id in orchestrator.threads
        assert thread.gap_id == "g1"
        assert thread.status == ThreadStatus.INITIALIZED
        assert gap.status == GapStatus.IN_PROGRESS

    def test_get_thread(self, orchestrator: ResearchOrchestrator) -> None:
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        assert orchestrator.get_thread(thread.id) is thread
        assert orchestrator.get_thread("nonexistent") is None

    def test_active_threads(self, orchestrator: ResearchOrchestrator) -> None:
        g1 = _make_gap("g1")
        g2 = _make_gap("g2")
        t1 = orchestrator._initialize_thread(g1)
        t2 = orchestrator._initialize_thread(g2)

        assert len(orchestrator.active_threads()) == 2

        t1.status = ThreadStatus.PUBLISHED
        assert len(orchestrator.active_threads()) == 1

        t2.status = ThreadStatus.ABANDONED
        assert len(orchestrator.active_threads()) == 0

    def test_status_summary(
        self, orchestrator: ResearchOrchestrator, mock_resources: ResourceManager
    ) -> None:
        gap = _make_gap("g1")
        orchestrator._initialize_thread(gap)
        summary = orchestrator.status_summary()
        assert summary["total_threads"] == 1
        assert "status_counts" in summary
        assert "budget" in summary


# ── Strategic Planning ────────────────────────────────────────────


class TestStrategicPlanning:
    @pytest.mark.asyncio
    async def test_strategic_cycle_with_open_gaps(
        self, orchestrator: ResearchOrchestrator, mock_llm: LLMClient
    ) -> None:
        gaps = [_make_gap("g1"), _make_gap("g2")]
        orchestrator.load_gaps(gaps)

        await orchestrator._strategic_planning_cycle()

        # Should have initialized at least one thread
        assert len(orchestrator.threads) >= 1

    @pytest.mark.asyncio
    async def test_strategic_cycle_no_gaps(self, orchestrator: ResearchOrchestrator) -> None:
        orchestrator.load_gaps([])
        await orchestrator._strategic_planning_cycle()
        assert len(orchestrator.threads) == 0

    @pytest.mark.asyncio
    async def test_strategic_cycle_max_threads(
        self, orchestrator: ResearchOrchestrator, mock_resources: ResourceManager
    ) -> None:
        gaps = [_make_gap(f"g{i}") for i in range(10)]
        orchestrator.load_gaps(gaps)

        # After first gap, can't start more threads
        call_count = 0
        def can_start_side_effect() -> bool:
            nonlocal call_count
            call_count += 1
            return call_count <= 1

        mock_resources.can_start_thread = MagicMock(side_effect=can_start_side_effect)

        await orchestrator._strategic_planning_cycle()
        assert len(orchestrator.threads) == 1

    @pytest.mark.asyncio
    async def test_select_research_directions_fallback(
        self, orchestrator: ResearchOrchestrator, mock_llm: LLMClient
    ) -> None:
        """When LLM fails, falls back to priority scoring."""
        mock_llm.generate_structured = AsyncMock(side_effect=RuntimeError("LLM error"))
        gaps = [
            _make_gap("g1", importance=0.9, tractability=0.9, novelty=0.9),
            _make_gap("g2", importance=0.1, tractability=0.1, novelty=0.1),
        ]
        selected = await orchestrator._select_research_directions(gaps, n=1)
        assert len(selected) == 1
        assert selected[0].id == "g1"


# ── Thread Review ─────────────────────────────────────────────────


class TestThreadReview:
    @pytest.mark.asyncio
    async def test_abandon_exhausted_thread(
        self, orchestrator: ResearchOrchestrator, mock_resources: ResourceManager
    ) -> None:
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        mock_resources.remaining_budget = MagicMock(return_value=0.0)

        await orchestrator._review_thread_performance()
        assert thread.status == ThreadStatus.ABANDONED

    @pytest.mark.asyncio
    async def test_keep_thread_with_results(
        self, orchestrator: ResearchOrchestrator, mock_resources: ResourceManager
    ) -> None:
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        thread.result_ids = ["r1"]
        mock_resources.remaining_budget = MagicMock(return_value=0.0)

        await orchestrator._review_thread_performance()
        assert thread.status == ThreadStatus.INITIALIZED  # not abandoned

    @pytest.mark.asyncio
    async def test_skip_published_thread(
        self, orchestrator: ResearchOrchestrator, mock_resources: ResourceManager
    ) -> None:
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        thread.status = ThreadStatus.PUBLISHED
        mock_resources.remaining_budget = MagicMock(return_value=0.0)

        await orchestrator._review_thread_performance()
        assert thread.status == ThreadStatus.PUBLISHED


# ── Thread Advancement ────────────────────────────────────────────


class TestThreadAdvancement:
    @pytest.mark.asyncio
    async def test_advance_initialized_to_lit_review(
        self, orchestrator: ResearchOrchestrator, mock_router: TaskRouter
    ) -> None:
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        await orchestrator._advance_thread(thread)
        mock_router.route_task.assert_called_once()
        assert thread.status == ThreadStatus.LITERATURE_REVIEW

    @pytest.mark.asyncio
    async def test_advance_lit_review_to_hypothesis(
        self, orchestrator: ResearchOrchestrator, mock_router: TaskRouter
    ) -> None:
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        thread.status = ThreadStatus.LITERATURE_REVIEW
        await orchestrator._advance_thread(thread)
        assert thread.status == ThreadStatus.HYPOTHESIS_GENERATION

    @pytest.mark.asyncio
    async def test_advance_hypothesis_to_critique(
        self, orchestrator: ResearchOrchestrator, mock_router: TaskRouter
    ) -> None:
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        thread.status = ThreadStatus.HYPOTHESIS_GENERATION
        await orchestrator._advance_thread(thread)
        assert thread.status == ThreadStatus.CRITIQUE

    @pytest.mark.asyncio
    async def test_advance_through_experiment_phases(
        self, orchestrator: ResearchOrchestrator, mock_router: TaskRouter
    ) -> None:
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)

        for expected_status in [
            ThreadStatus.LITERATURE_REVIEW,
            ThreadStatus.HYPOTHESIS_GENERATION,
            ThreadStatus.CRITIQUE,
            ThreadStatus.EXPERIMENT_DESIGN,
            ThreadStatus.EXPERIMENT_EXECUTION,
            ThreadStatus.RESULT_INTERPRETATION,
        ]:
            await orchestrator._advance_thread(thread)
            assert thread.status == expected_status

    @pytest.mark.asyncio
    async def test_advance_synthesis_routes_two_tasks(
        self, orchestrator: ResearchOrchestrator, mock_router: TaskRouter
    ) -> None:
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        thread.status = ThreadStatus.RESULT_INTERPRETATION
        await orchestrator._advance_thread(thread)
        assert thread.status == ThreadStatus.SYNTHESIS
        # Should route both synthesis and writing tasks
        assert mock_router.route_task.call_count == 2

    @pytest.mark.asyncio
    async def test_advance_no_result_stays_same(
        self, orchestrator: ResearchOrchestrator, mock_router: TaskRouter
    ) -> None:
        mock_router.route_task = AsyncMock(return_value=None)
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        await orchestrator._advance_thread(thread)
        assert thread.status == ThreadStatus.INITIALIZED


# ── Peer Review Phase ─────────────────────────────────────────────


class TestPeerReview:
    @pytest.mark.asyncio
    async def test_peer_review_passes_gate(
        self,
        orchestrator: ResearchOrchestrator,
        mock_peer_review: SimulatedPeerReview,
        mock_sunfire: SUNFIREEvaluator,
        mock_iwpg: IWPGScorer,
        mock_router: TaskRouter,
    ) -> None:
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        thread.status = ThreadStatus.SYNTHESIS

        await orchestrator._run_peer_review_phase(thread)

        assert thread.status == ThreadStatus.PEER_REVIEW
        assert thread.sunfire_score == 0.75
        assert thread.iwpg_reward == 0.8
        mock_router.blackboard.write.assert_called()

    @pytest.mark.asyncio
    async def test_peer_review_fails_gate(
        self, orchestrator: ResearchOrchestrator, mock_peer_review: SimulatedPeerReview
    ) -> None:
        mock_peer_review.quality_gate_check = AsyncMock(return_value=False)
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        thread.status = ThreadStatus.SYNTHESIS

        await orchestrator._run_peer_review_phase(thread)
        assert thread.status == ThreadStatus.REVISION


# ── Review Outcome Handling ───────────────────────────────────────


class TestReviewOutcome:
    @pytest.mark.asyncio
    async def test_accept_publishes(
        self, orchestrator: ResearchOrchestrator, mock_router: TaskRouter
    ) -> None:
        mock_router.blackboard.read = AsyncMock(return_value={
            "decision": "accept",
            "score": 0.9,
        })
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        thread.status = ThreadStatus.PEER_REVIEW

        await orchestrator._handle_review_outcome(thread)
        assert thread.status == ThreadStatus.PUBLISHED
        assert thread.published_at is not None

    @pytest.mark.asyncio
    async def test_reject_abandons(
        self, orchestrator: ResearchOrchestrator, mock_router: TaskRouter
    ) -> None:
        mock_router.blackboard.read = AsyncMock(return_value={
            "decision": "reject",
            "score": 0.3,
        })
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        thread.status = ThreadStatus.PEER_REVIEW

        await orchestrator._handle_review_outcome(thread)
        assert thread.status == ThreadStatus.ABANDONED

    @pytest.mark.asyncio
    async def test_revise_goes_to_revision(
        self, orchestrator: ResearchOrchestrator, mock_router: TaskRouter
    ) -> None:
        mock_router.blackboard.read = AsyncMock(return_value={
            "decision": "revise",
            "score": 0.6,
        })
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        thread.status = ThreadStatus.PEER_REVIEW

        await orchestrator._handle_review_outcome(thread)
        assert thread.status == ThreadStatus.REVISION

    @pytest.mark.asyncio
    async def test_revise_max_rounds_abandons(
        self, orchestrator: ResearchOrchestrator, mock_router: TaskRouter, config: ResearchConfig
    ) -> None:
        mock_router.blackboard.read = AsyncMock(return_value={
            "decision": "revise",
            "score": 0.6,
        })
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        thread.status = ThreadStatus.PEER_REVIEW
        thread.revision_count = config.peer_review.max_revision_rounds

        await orchestrator._handle_review_outcome(thread)
        assert thread.status == ThreadStatus.ABANDONED

    @pytest.mark.asyncio
    async def test_no_review_data(
        self, orchestrator: ResearchOrchestrator, mock_router: TaskRouter
    ) -> None:
        mock_router.blackboard.read = AsyncMock(return_value=None)
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        thread.status = ThreadStatus.PEER_REVIEW

        await orchestrator._handle_review_outcome(thread)
        assert thread.status == ThreadStatus.PEER_REVIEW  # unchanged


# ── Execution Cycle ───────────────────────────────────────────────


class TestExecutionCycle:
    @pytest.mark.asyncio
    async def test_execution_cycle_advances_active_threads(
        self, orchestrator: ResearchOrchestrator, mock_router: TaskRouter
    ) -> None:
        g1 = _make_gap("g1")
        g2 = _make_gap("g2")
        t1 = orchestrator._initialize_thread(g1)
        t2 = orchestrator._initialize_thread(g2)

        await orchestrator._execution_cycle()

        # Both threads should have been advanced
        assert t1.status == ThreadStatus.LITERATURE_REVIEW
        assert t2.status == ThreadStatus.LITERATURE_REVIEW
        mock_router.process_pending.assert_called_once()

    @pytest.mark.asyncio
    async def test_execution_cycle_skips_published(
        self, orchestrator: ResearchOrchestrator, mock_router: TaskRouter
    ) -> None:
        gap = _make_gap("g1")
        thread = orchestrator._initialize_thread(gap)
        thread.status = ThreadStatus.PUBLISHED

        await orchestrator._execution_cycle()
        # route_task should not be called for published thread
        mock_router.route_task.assert_not_called()


# ── Ingestion Cycle ───────────────────────────────────────────────


class TestIngestionCycle:
    @pytest.mark.asyncio
    async def test_ingestion_routes_claim_extraction(
        self, orchestrator: ResearchOrchestrator, mock_router: TaskRouter
    ) -> None:
        await orchestrator._ingestion_cycle()
        mock_router.route_task.assert_called_once()
        call_args = mock_router.route_task.call_args
        task = call_args[0][0]
        assert task.task_type == "claim_extraction"


# ── Start/Stop ────────────────────────────────────────────────────


class TestStartStop:
    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self, orchestrator: ResearchOrchestrator) -> None:
        # Start orchestrator in background then immediately stop
        orchestrator._running = True
        task1 = asyncio.create_task(asyncio.sleep(100))
        task2 = asyncio.create_task(asyncio.sleep(100))
        orchestrator._tasks = [task1, task2]

        await orchestrator.stop()
        assert orchestrator._running is False
        assert len(orchestrator._tasks) == 0

    def test_load_gaps(self, orchestrator: ResearchOrchestrator) -> None:
        gaps = [_make_gap("g1"), _make_gap("g2")]
        orchestrator.load_gaps(gaps)
        assert len(orchestrator._gaps) == 2
