"""Main orchestrator with three core loops: outer (strategic), middle (execution), inner (ingestion)."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from auto_researcher.config import ResearchConfig
from auto_researcher.evaluation.iwpg import IWPGScorer
from auto_researcher.evaluation.peer_review import SimulatedPeerReview
from auto_researcher.evaluation.sunfire import SUNFIREEvaluator
from auto_researcher.models.gap import Gap, GapStatus
from auto_researcher.models.research_thread import ResearchThread, ThreadStatus
from auto_researcher.orchestrator.resource_manager import ResourceManager
from auto_researcher.orchestrator.task_router import ResearchTask, TaskRouter
from auto_researcher.utils.llm import LLMClient
from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)

GAP_SELECTION_PROMPT = """\
You are a strategic research planner. Given the following open research gaps, \
select the top {n} most promising opportunities for investigation.

Consider: importance, tractability, novelty, timeliness, and foundational impact.

Gaps:
{gaps_text}

Respond with JSON: {{"selected_gap_ids": ["<id>"], "reasoning": "<string>"}}
"""


class ResearchOrchestrator:
    """Coordinates the three-loop autonomous research process.

    OUTER LOOP (weekly/monthly): Strategic planning - select research directions,
        update gap map, allocate resources.
    MIDDLE LOOP (daily): Research execution - run experiments, generate hypotheses,
        draft outputs, run peer review.
    INNER LOOP (real-time): Knowledge ingestion - process new papers, update
        knowledge graph, detect contradictions.
    """

    def __init__(
        self,
        config: ResearchConfig,
        llm: LLMClient,
        task_router: TaskRouter,
        resource_manager: ResourceManager,
        sunfire: SUNFIREEvaluator,
        iwpg: IWPGScorer,
        peer_review: SimulatedPeerReview,
    ) -> None:
        self._config = config
        self._llm = llm
        self._router = task_router
        self._resources = resource_manager
        self._sunfire = sunfire
        self._iwpg = iwpg
        self._peer_review = peer_review
        self._threads: dict[str, ResearchThread] = {}
        self._gaps: list[Gap] = []
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def threads(self) -> dict[str, ResearchThread]:
        return self._threads

    async def start(self) -> None:
        """Start all three loops concurrently."""
        self._running = True
        logger.info("orchestrator_starting")
        self._tasks = [
            asyncio.create_task(self._outer_loop()),
            asyncio.create_task(self._middle_loop()),
            asyncio.create_task(self._inner_loop()),
        ]
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Gracefully stop all loops."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("orchestrator_stopped")

    def load_gaps(self, gaps: list[Gap]) -> None:
        self._gaps = gaps

    # ------------------------------------------------------------------
    # OUTER LOOP: Strategic research planning
    # ------------------------------------------------------------------

    async def _outer_loop(self) -> None:
        """Weekly/monthly strategic planning loop."""
        interval = self._config.orchestrator.outer_loop_interval_days * 86400
        while self._running:
            try:
                await self._strategic_planning_cycle()
            except Exception:
                logger.exception("outer_loop_error")
            await asyncio.sleep(interval)

    async def _strategic_planning_cycle(self) -> None:
        """One cycle of strategic planning."""
        logger.info("strategic_planning_start")

        # 1. Select top research directions using IWPG scoring
        open_gaps = [g for g in self._gaps if g.status == GapStatus.OPEN]
        if not open_gaps:
            logger.info("no_open_gaps")
            return

        selected = await self._select_research_directions(open_gaps)

        # 2. Initialize research threads for selected gaps
        for gap in selected:
            if not self._resources.can_start_thread():
                logger.info("max_threads_reached")
                break
            thread = self._initialize_thread(gap)
            logger.info("thread_initialized", thread_id=thread.id, gap=gap.description[:80])

        # 3. Review and potentially abandon low-performing threads
        await self._review_thread_performance()

        logger.info(
            "strategic_planning_complete",
            active_threads=len(self._threads),
            open_gaps=len(open_gaps),
        )

    async def _select_research_directions(
        self, gaps: list[Gap], n: int = 3
    ) -> list[Gap]:
        """Select top-n research gaps using LLM + IWPG scoring."""
        # Score gaps by priority
        scored = sorted(gaps, key=lambda g: g.priority_score(), reverse=True)
        top_candidates = scored[:min(10, len(scored))]

        # Use LLM for final selection
        gaps_text = "\n".join(
            f"- ID: {g.id} | Type: {g.gap_type.value} | Priority: {g.priority_score():.3f} | "
            f"Description: {g.description[:200]}"
            for g in top_candidates
        )
        try:
            result = await self._llm.generate_structured(
                prompt=GAP_SELECTION_PROMPT.format(n=n, gaps_text=gaps_text),
                system="You are a strategic research planner.",
                temperature=0.4,
            )
            selected_ids = set(result.get("selected_gap_ids", []))
            selected = [g for g in top_candidates if g.id in selected_ids]
            if selected:
                return selected[:n]
        except Exception:
            logger.exception("gap_selection_llm_failed")

        # Fallback: use priority score
        return top_candidates[:n]

    def _initialize_thread(self, gap: Gap) -> ResearchThread:
        """Create a new research thread from a gap."""
        thread_id = str(uuid.uuid4())
        thread = ResearchThread(
            id=thread_id,
            gap_id=gap.id,
            title=f"Investigation: {gap.description[:100]}",
            compute_budget=self._config.orchestrator.compute_budget_per_thread,
        )
        self._threads[thread_id] = thread
        self._resources.allocate_budget(thread_id, thread.compute_budget)
        gap.status = GapStatus.IN_PROGRESS
        return thread

    async def _review_thread_performance(self) -> None:
        """Review active threads and abandon underperformers."""
        for thread_id, thread in list(self._threads.items()):
            if thread.status in (ThreadStatus.PUBLISHED, ThreadStatus.ABANDONED):
                continue
            # Abandon if budget exhausted with no results
            if (self._resources.remaining_budget(thread_id) <= 0
                    and not thread.result_ids):
                thread.status = ThreadStatus.ABANDONED
                self._resources.release_thread(thread_id)
                logger.info("thread_abandoned_no_budget", thread_id=thread_id)

    # ------------------------------------------------------------------
    # MIDDLE LOOP: Research execution
    # ------------------------------------------------------------------

    async def _middle_loop(self) -> None:
        """Daily research execution loop."""
        interval = self._config.orchestrator.middle_loop_interval_hours * 3600
        while self._running:
            try:
                await self._execution_cycle()
            except Exception:
                logger.exception("middle_loop_error")
            await asyncio.sleep(interval)

    async def _execution_cycle(self) -> None:
        """One cycle of research execution across all active threads."""
        logger.info("execution_cycle_start", active_threads=len(self._threads))

        for thread_id, thread in list(self._threads.items()):
            if thread.status in (ThreadStatus.PUBLISHED, ThreadStatus.ABANDONED):
                continue
            try:
                await self._advance_thread(thread)
            except Exception:
                logger.exception("thread_advance_error", thread_id=thread_id)

        # Process any pending tasks
        await self._router.process_pending()

    async def _advance_thread(self, thread: ResearchThread) -> None:
        """Advance a research thread to its next phase."""
        status = thread.status

        if status == ThreadStatus.INITIALIZED:
            await self._run_literature_review(thread)
        elif status == ThreadStatus.LITERATURE_REVIEW:
            await self._run_hypothesis_generation(thread)
        elif status == ThreadStatus.HYPOTHESIS_GENERATION:
            await self._run_critique(thread)
        elif status == ThreadStatus.CRITIQUE:
            await self._run_experiment_design(thread)
        elif status == ThreadStatus.EXPERIMENT_DESIGN:
            await self._run_experiment_execution(thread)
        elif status == ThreadStatus.EXPERIMENT_EXECUTION:
            await self._run_result_interpretation(thread)
        elif status == ThreadStatus.RESULT_INTERPRETATION:
            await self._run_synthesis(thread)
        elif status == ThreadStatus.SYNTHESIS:
            await self._run_peer_review_phase(thread)
        elif status == ThreadStatus.PEER_REVIEW:
            await self._handle_review_outcome(thread)
        elif status == ThreadStatus.REVISION:
            await self._run_peer_review_phase(thread)

        thread.updated_at = datetime.now(UTC)

    async def _run_literature_review(self, thread: ResearchThread) -> None:
        task = ResearchTask(
            task_type="literature_review",
            payload={"gap_id": thread.gap_id, "title": thread.title},
            priority=3,
            thread_id=thread.id,
        )
        result = await self._router.route_task(task)
        if result is not None:
            thread.status = ThreadStatus.LITERATURE_REVIEW
            self._resources.spend(thread.id, 5.0)

    async def _run_hypothesis_generation(self, thread: ResearchThread) -> None:
        task = ResearchTask(
            task_type="hypothesis_generation",
            payload={"gap_id": thread.gap_id, "literature": thread.literature_context},
            priority=3,
            thread_id=thread.id,
        )
        result = await self._router.route_task(task)
        if result is not None:
            thread.status = ThreadStatus.HYPOTHESIS_GENERATION
            self._resources.spend(thread.id, 5.0)

    async def _run_critique(self, thread: ResearchThread) -> None:
        task = ResearchTask(
            task_type="critique",
            payload={"hypothesis_ids": thread.hypothesis_ids},
            priority=4,
            thread_id=thread.id,
        )
        result = await self._router.route_task(task)
        if result is not None:
            thread.status = ThreadStatus.CRITIQUE
            self._resources.spend(thread.id, 3.0)

    async def _run_experiment_design(self, thread: ResearchThread) -> None:
        task = ResearchTask(
            task_type="experiment_design",
            payload={"hypothesis_ids": thread.hypothesis_ids},
            priority=3,
            thread_id=thread.id,
        )
        result = await self._router.route_task(task)
        if result is not None:
            thread.status = ThreadStatus.EXPERIMENT_DESIGN
            self._resources.spend(thread.id, 5.0)

    async def _run_experiment_execution(self, thread: ResearchThread) -> None:
        task = ResearchTask(
            task_type="experiment_design",
            payload={"experiment_ids": thread.experiment_ids},
            priority=2,
            thread_id=thread.id,
        )
        result = await self._router.route_task(task)
        if result is not None:
            thread.status = ThreadStatus.EXPERIMENT_EXECUTION
            self._resources.spend(thread.id, 20.0)

    async def _run_result_interpretation(self, thread: ResearchThread) -> None:
        task = ResearchTask(
            task_type="statistical_analysis",
            payload={"result_ids": thread.result_ids},
            priority=3,
            thread_id=thread.id,
        )
        result = await self._router.route_task(task)
        if result is not None:
            thread.status = ThreadStatus.RESULT_INTERPRETATION
            self._resources.spend(thread.id, 5.0)
            self._resources.record_insight(thread.id)

    async def _run_synthesis(self, thread: ResearchThread) -> None:
        task = ResearchTask(
            task_type="synthesis",
            payload={
                "hypothesis_ids": thread.hypothesis_ids,
                "result_ids": thread.result_ids,
            },
            priority=3,
            thread_id=thread.id,
        )
        result = await self._router.route_task(task)
        if result is not None:
            thread.status = ThreadStatus.SYNTHESIS
            self._resources.spend(thread.id, 10.0)

        # Also run writing task
        writing_task = ResearchTask(
            task_type="writing",
            payload={"thread_id": thread.id},
            priority=4,
            thread_id=thread.id,
        )
        await self._router.route_task(writing_task)

    async def _run_peer_review_phase(self, thread: ResearchThread) -> None:
        """Run simulated peer review with quality gate."""
        passes_gate = await self._peer_review.quality_gate_check(thread)
        if not passes_gate:
            logger.info("quality_gate_failed", thread_id=thread.id)
            thread.status = ThreadStatus.REVISION
            return

        # SUNFIRE evaluation
        sunfire_score = await self._sunfire.evaluate(thread)
        thread.sunfire_score = self._sunfire.composite_score(sunfire_score)

        # IWPG reward
        iwpg_reward = await self._iwpg.compute_reward(thread)
        thread.iwpg_reward = self._iwpg.total_reward(iwpg_reward)

        # Full peer review
        review_result = await self._peer_review.review(thread)
        thread.status = ThreadStatus.PEER_REVIEW
        self._resources.spend(thread.id, 15.0)

        # Store review outcome for next cycle
        await self._router.blackboard.write(
            f"review:{thread.id}",
            {"decision": review_result.decision.value, "score": review_result.overall_score},
        )

    async def _handle_review_outcome(self, thread: ResearchThread) -> None:
        """Handle the outcome of peer review."""
        review_data = await self._router.blackboard.read(f"review:{thread.id}")
        if review_data is None:
            return

        decision = review_data.get("decision", "revise")
        if decision == "accept":
            thread.status = ThreadStatus.PUBLISHED
            thread.published_at = datetime.now(UTC)
            self._resources.release_thread(thread.id)
            logger.info("thread_published", thread_id=thread.id)
        elif decision == "reject":
            thread.status = ThreadStatus.ABANDONED
            self._resources.release_thread(thread.id)
            logger.info("thread_rejected", thread_id=thread.id)
        else:
            if thread.revision_count < self._config.peer_review.max_revision_rounds:
                thread.status = ThreadStatus.REVISION
            else:
                thread.status = ThreadStatus.ABANDONED
                self._resources.release_thread(thread.id)

    # ------------------------------------------------------------------
    # INNER LOOP: Knowledge ingestion
    # ------------------------------------------------------------------

    async def _inner_loop(self) -> None:
        """Real-time knowledge ingestion loop."""
        interval = self._config.orchestrator.inner_loop_interval_minutes * 60
        while self._running:
            try:
                await self._ingestion_cycle()
            except Exception:
                logger.exception("inner_loop_error")
            await asyncio.sleep(interval)

    async def _ingestion_cycle(self) -> None:
        """One cycle of knowledge ingestion."""
        logger.info("ingestion_cycle_start")

        # Process new papers via task router
        task = ResearchTask(
            task_type="claim_extraction",
            payload={"source": "arxiv_recent"},
            priority=5,
        )
        await self._router.route_task(task)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_thread(self, thread_id: str) -> ResearchThread | None:
        return self._threads.get(thread_id)

    def active_threads(self) -> list[ResearchThread]:
        return [
            t for t in self._threads.values()
            if t.status not in (ThreadStatus.PUBLISHED, ThreadStatus.ABANDONED)
        ]

    def status_summary(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        for thread in self._threads.values():
            status_counts[thread.status.value] = status_counts.get(thread.status.value, 0) + 1
        return {
            "total_threads": len(self._threads),
            "status_counts": status_counts,
            "budget": self._resources.budget_summary(),
            "agent_load": self._router.get_agent_load(),
        }
