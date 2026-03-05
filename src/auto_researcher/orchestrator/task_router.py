"""Task routing and delegation using Contract Net Protocol and Blackboard architecture."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Callable, Awaitable

from auto_researcher.models.agent import AgentMessage, AgentRole, AgentState
from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)


class AgentBid:
    """A bid from an agent to perform a task."""

    def __init__(
        self,
        agent_role: AgentRole,
        task_id: str,
        estimated_cost: float = 0.0,
        estimated_time: float = 0.0,
        confidence: float = 0.5,
    ) -> None:
        self.agent_role = agent_role
        self.task_id = task_id
        self.estimated_cost = estimated_cost
        self.estimated_time = estimated_time
        self.confidence = confidence

    def score(self) -> float:
        """Score a bid: prefer high confidence, low cost, low time."""
        if self.estimated_time == 0:
            time_factor = 1.0
        else:
            time_factor = 1.0 / (1.0 + self.estimated_time)
        if self.estimated_cost == 0:
            cost_factor = 1.0
        else:
            cost_factor = 1.0 / (1.0 + self.estimated_cost)
        return self.confidence * 0.5 + cost_factor * 0.25 + time_factor * 0.25


class ResearchTask:
    """A task to be routed to a specialist agent."""

    def __init__(
        self,
        task_type: str,
        payload: dict[str, Any],
        priority: int = 5,
        thread_id: str = "",
        required_role: AgentRole | None = None,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.task_type = task_type
        self.payload = payload
        self.priority = priority
        self.thread_id = thread_id
        self.required_role = required_role
        self.assigned_to: AgentRole | None = None
        self.status: str = "pending"
        self.result: Any = None
        self.created_at: datetime = datetime.utcnow()

    def to_message(self, sender: AgentRole = AgentRole.ORCHESTRATOR) -> AgentMessage:
        return AgentMessage(
            sender=sender,
            receiver=self.assigned_to,
            task_type=self.task_type,
            priority=self.priority,
            payload=self.payload,
            message_id=self.id,
        )


# Mapping from task type to preferred agent roles
TASK_ROLE_MAP: dict[str, list[AgentRole]] = {
    "literature_review": [AgentRole.LITERATURE_ANALYST],
    "claim_extraction": [AgentRole.LITERATURE_ANALYST],
    "hypothesis_generation": [AgentRole.HYPOTHESIS_GENERATOR],
    "experiment_design": [AgentRole.EXPERIMENT_DESIGNER],
    "critique": [AgentRole.CRITIC],
    "synthesis": [AgentRole.SYNTHESIZER],
    "writing": [AgentRole.SCIENCE_COMMUNICATOR],
    "statistical_analysis": [AgentRole.STATISTICIAN],
}


class Blackboard:
    """Shared state blackboard for inter-agent coordination."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._subscribers: dict[str, list[Callable[[str, Any], Awaitable[None]]]] = {}

    async def write(self, key: str, value: Any) -> None:
        async with self._lock:
            self._state[key] = value
        # Notify subscribers
        for callback in self._subscribers.get(key, []):
            try:
                await callback(key, value)
            except Exception:
                logger.exception("blackboard_subscriber_error", key=key)

    async def read(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            return self._state.get(key, default)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._state.pop(key, None)

    def subscribe(self, key: str, callback: Callable[[str, Any], Awaitable[None]]) -> None:
        self._subscribers.setdefault(key, []).append(callback)

    async def keys(self) -> list[str]:
        async with self._lock:
            return list(self._state.keys())


class TaskRouter:
    """Routes research tasks to specialist agents using Contract Net Protocol."""

    def __init__(self) -> None:
        self._blackboard = Blackboard()
        self._agent_states: dict[AgentRole, AgentState] = {
            role: AgentState.IDLE for role in AgentRole
        }
        self._agent_handlers: dict[AgentRole, Callable[[ResearchTask], Awaitable[Any]]] = {}
        self._pending_tasks: list[ResearchTask] = []
        self._completed_tasks: list[ResearchTask] = []

    @property
    def blackboard(self) -> Blackboard:
        return self._blackboard

    def register_agent(
        self,
        role: AgentRole,
        handler: Callable[[ResearchTask], Awaitable[Any]],
    ) -> None:
        """Register an agent handler for a given role."""
        self._agent_handlers[role] = handler
        self._agent_states[role] = AgentState.IDLE
        logger.info("agent_registered", role=role.value)

    def set_agent_state(self, role: AgentRole, state: AgentState) -> None:
        self._agent_states[role] = state

    async def route_task(self, task: ResearchTask) -> Any:
        """Route a task to the best available agent."""
        # If a specific role is required, use it directly
        if task.required_role is not None:
            return await self._assign_and_execute(task, task.required_role)

        # Contract Net Protocol: find capable agents and select best
        capable_roles = TASK_ROLE_MAP.get(task.task_type, [])
        if not capable_roles:
            logger.warning("no_capable_agents", task_type=task.task_type)
            capable_roles = [AgentRole.ORCHESTRATOR]

        # Collect bids from available agents
        bids = self._collect_bids(task, capable_roles)
        if not bids:
            # All preferred agents busy; queue the task
            self._pending_tasks.append(task)
            logger.info("task_queued", task_id=task.id, task_type=task.task_type)
            return None

        # Award contract to best bidder
        best_bid = max(bids, key=lambda b: b.score())
        return await self._assign_and_execute(task, best_bid.agent_role)

    def _collect_bids(self, task: ResearchTask, roles: list[AgentRole]) -> list[AgentBid]:
        """Collect bids from agents capable of performing the task."""
        bids = []
        for role in roles:
            if self._agent_states.get(role) == AgentState.IDLE:
                if role in self._agent_handlers:
                    bids.append(AgentBid(
                        agent_role=role,
                        task_id=task.id,
                        confidence=0.8 if role in roles[:1] else 0.5,
                    ))
        return bids

    async def _assign_and_execute(self, task: ResearchTask, role: AgentRole) -> Any:
        """Assign task to agent and execute."""
        task.assigned_to = role
        task.status = "in_progress"
        self._agent_states[role] = AgentState.WORKING

        handler = self._agent_handlers.get(role)
        if handler is None:
            logger.error("no_handler", role=role.value)
            task.status = "failed"
            return None

        try:
            # Update blackboard with current task
            await self._blackboard.write(f"task:{task.id}", {
                "type": task.task_type,
                "assigned_to": role.value,
                "thread_id": task.thread_id,
                "status": "in_progress",
            })

            result = await handler(task)
            task.result = result
            task.status = "completed"
            self._completed_tasks.append(task)

            await self._blackboard.write(f"task:{task.id}", {
                "type": task.task_type,
                "assigned_to": role.value,
                "thread_id": task.thread_id,
                "status": "completed",
            })

            logger.info("task_completed", task_id=task.id, role=role.value)
            return result
        except Exception:
            logger.exception("task_failed", task_id=task.id, role=role.value)
            task.status = "failed"
            return None
        finally:
            self._agent_states[role] = AgentState.IDLE

    async def process_pending(self) -> int:
        """Attempt to route any pending tasks. Returns count of tasks processed."""
        processed = 0
        still_pending = []
        for task in self._pending_tasks:
            result = await self.route_task(task)
            if result is not None or task.status == "completed":
                processed += 1
            else:
                still_pending.append(task)
        self._pending_tasks = still_pending
        return processed

    def get_agent_load(self) -> dict[str, str]:
        return {role.value: state.value for role, state in self._agent_states.items()}
