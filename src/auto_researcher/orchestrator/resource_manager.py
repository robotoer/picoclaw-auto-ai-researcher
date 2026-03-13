"""Resource management: compute budgets, rate limits, cost tracking."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any

from auto_researcher.config import OrchestratorConfig
from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)


class ResourceManager:
    """Manages compute budgets, API rate limits, and cost tracking."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self._config = config
        self._thread_budgets: dict[str, float] = {}  # thread_id -> remaining budget
        self._thread_spent: dict[str, float] = {}  # thread_id -> total spent
        self._insights_per_thread: dict[str, int] = defaultdict(int)
        self._rate_limits: dict[str, _RateLimiter] = {}
        self._task_queue: asyncio.PriorityQueue[tuple[int, str, Any]] = asyncio.PriorityQueue()
        self._total_cost: float = 0.0

    def allocate_budget(self, thread_id: str, budget: float | None = None) -> float:
        """Allocate compute budget for a research thread."""
        amount = budget if budget is not None else self._config.compute_budget_per_thread
        self._thread_budgets[thread_id] = amount
        self._thread_spent.setdefault(thread_id, 0.0)
        logger.info("budget_allocated", thread_id=thread_id, budget=amount)
        return amount

    def spend(self, thread_id: str, amount: float) -> bool:
        """Record compute spending for a thread. Returns False if over budget."""
        remaining = self._thread_budgets.get(thread_id, 0.0)
        if amount > remaining:
            logger.warning("budget_exceeded", thread_id=thread_id, requested=amount, remaining=remaining)
            return False
        self._thread_budgets[thread_id] = remaining - amount
        self._thread_spent[thread_id] = self._thread_spent.get(thread_id, 0.0) + amount
        self._total_cost += amount
        return True

    def remaining_budget(self, thread_id: str) -> float:
        return self._thread_budgets.get(thread_id, 0.0)

    def record_insight(self, thread_id: str, count: int = 1) -> None:
        """Record that an insight was generated for cost-per-insight tracking."""
        self._insights_per_thread[thread_id] += count

    def cost_per_insight(self, thread_id: str) -> float:
        """Compute cost per insight for a thread."""
        spent = self._thread_spent.get(thread_id, 0.0)
        insights = self._insights_per_thread.get(thread_id, 0)
        if insights == 0:
            return float("inf")
        return spent / insights

    def register_rate_limit(self, service: str, max_requests: int, window_seconds: float) -> None:
        """Register a rate limiter for an external service."""
        self._rate_limits[service] = _RateLimiter(max_requests, window_seconds)

    async def acquire_rate_limit(self, service: str) -> bool:
        """Acquire a rate limit slot. Blocks if necessary."""
        limiter = self._rate_limits.get(service)
        if limiter is None:
            return True
        return await limiter.acquire()

    async def enqueue_task(self, priority: int, task_id: str, payload: Any) -> None:
        """Add a task to the priority queue. Lower priority number = higher priority."""
        await self._task_queue.put((priority, task_id, payload))

    async def dequeue_task(self) -> tuple[int, str, Any]:
        """Get the highest-priority task from the queue."""
        return await self._task_queue.get()

    @property
    def active_thread_count(self) -> int:
        return len(self._thread_budgets)

    def can_start_thread(self) -> bool:
        return self.active_thread_count < self._config.max_concurrent_threads

    def budget_summary(self) -> dict[str, Any]:
        """Get a summary of budget status across all threads."""
        return {
            "total_spent": self._total_cost,
            "active_threads": self.active_thread_count,
            "threads": {
                tid: {
                    "remaining": self._thread_budgets.get(tid, 0.0),
                    "spent": self._thread_spent.get(tid, 0.0),
                    "insights": self._insights_per_thread.get(tid, 0),
                    "cost_per_insight": self.cost_per_insight(tid),
                }
                for tid in self._thread_budgets
            },
        }

    def release_thread(self, thread_id: str) -> None:
        """Release resources for a completed/abandoned thread."""
        self._thread_budgets.pop(thread_id, None)
        logger.info(
            "thread_released",
            thread_id=thread_id,
            total_spent=self._thread_spent.get(thread_id, 0.0),
            insights=self._insights_per_thread.get(thread_id, 0),
        )


class _RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._lock:
            now = time.monotonic()
            # Remove expired timestamps
            self._timestamps = [t for t in self._timestamps if now - t < self._window]
            if len(self._timestamps) >= self._max:
                # Wait until the oldest request expires
                wait_time = self._window - (now - self._timestamps[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self._timestamps = self._timestamps[1:]
            self._timestamps.append(time.monotonic())
            return True
