"""Registry for tracking ML models and their capabilities."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelCapability(str, Enum):
    TEXT_GENERATION = "text_generation"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    EXTRACTION = "extraction"


class ModelInfo(BaseModel):
    """Metadata about a registered model."""

    name: str
    provider: str
    version: str = "latest"
    capabilities: list[ModelCapability] = Field(default_factory=list)
    context_window: int = 0
    max_output_tokens: int = 0
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModelPerformance(BaseModel):
    """Tracked performance metrics for a model on specific tasks."""

    task: str
    accuracy: float | None = None
    latency_ms: float | None = None
    cost_per_call: float | None = None
    sample_count: int = 0
    last_measured: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModelRegistry:
    """Registry for tracking ML models, their capabilities, and performance."""

    def __init__(self) -> None:
        self._models: dict[str, ModelInfo] = {}
        self._performance: dict[str, list[ModelPerformance]] = {}  # keyed by model name

    async def register(self, model: ModelInfo) -> None:
        key = f"{model.name}:{model.version}"
        self._models[key] = model
        if key not in self._performance:
            self._performance[key] = []
        logger.info("Registered model %s", key)

    async def get(self, name: str, version: str = "latest") -> ModelInfo | None:
        return self._models.get(f"{name}:{version}")

    async def lookup_by_capability(self, capability: ModelCapability) -> list[ModelInfo]:
        """Find all models that have a given capability."""
        return [
            m for m in self._models.values()
            if capability in m.capabilities
        ]

    async def best_for_task(self, capability: ModelCapability, task: str | None = None) -> ModelInfo | None:
        """Find the best model for a capability, optionally considering task performance."""
        candidates = await self.lookup_by_capability(capability)
        if not candidates:
            return None

        if task:
            # Rank by accuracy on the specific task.
            def task_score(model: ModelInfo) -> float:
                key = f"{model.name}:{model.version}"
                perfs = self._performance.get(key, [])
                for p in perfs:
                    if p.task == task and p.accuracy is not None:
                        return p.accuracy
                return 0.0

            candidates.sort(key=task_score, reverse=True)

        return candidates[0]

    async def record_performance(self, name: str, version: str, perf: ModelPerformance) -> None:
        key = f"{name}:{version}"
        if key not in self._performance:
            self._performance[key] = []

        # Update existing entry for same task or append new one.
        existing = [p for p in self._performance[key] if p.task == perf.task]
        if existing:
            entry = existing[0]
            # Running average.
            if perf.accuracy is not None:
                if entry.accuracy is not None:
                    entry.accuracy = (entry.accuracy * entry.sample_count + perf.accuracy) / (entry.sample_count + 1)
                else:
                    entry.accuracy = perf.accuracy
            if perf.latency_ms is not None:
                if entry.latency_ms is not None:
                    entry.latency_ms = (entry.latency_ms * entry.sample_count + perf.latency_ms) / (entry.sample_count + 1)
                else:
                    entry.latency_ms = perf.latency_ms
            entry.sample_count += 1
            entry.last_measured = datetime.now(UTC)
        else:
            perf.sample_count = 1
            self._performance[key].append(perf)

    async def get_performance(self, name: str, version: str = "latest") -> list[ModelPerformance]:
        return self._performance.get(f"{name}:{version}", [])

    async def list_models(self) -> list[ModelInfo]:
        return list(self._models.values())

    async def unregister(self, name: str, version: str = "latest") -> bool:
        key = f"{name}:{version}"
        if key in self._models:
            del self._models[key]
            self._performance.pop(key, None)
            return True
        return False
