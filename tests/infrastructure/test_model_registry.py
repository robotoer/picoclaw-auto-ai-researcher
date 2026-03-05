"""Tests for the ModelRegistry."""

import pytest
import pytest_asyncio

from auto_researcher.infrastructure.model_registry import (
    ModelCapability,
    ModelInfo,
    ModelPerformance,
    ModelRegistry,
)


@pytest_asyncio.fixture
async def registry() -> ModelRegistry:
    return ModelRegistry()


def _model(name: str = "claude", version: str = "latest", **kwargs) -> ModelInfo:
    defaults = dict(
        name=name,
        provider="anthropic",
        version=version,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
        context_window=200000,
        max_output_tokens=4096,
    )
    defaults.update(kwargs)
    return ModelInfo(**defaults)


# ── Registration ─────────────────────────────────────────────────────


class TestRegistration:
    @pytest.mark.asyncio
    async def test_register_and_get(self, registry: ModelRegistry):
        model = _model("claude", "3.5")
        await registry.register(model)
        retrieved = await registry.get("claude", "3.5")
        assert retrieved is not None
        assert retrieved.name == "claude"
        assert retrieved.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_get_default_version(self, registry: ModelRegistry):
        await registry.register(_model("claude"))
        retrieved = await registry.get("claude")
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, registry: ModelRegistry):
        assert await registry.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_register_overwrites(self, registry: ModelRegistry):
        await registry.register(_model("claude", context_window=100000))
        await registry.register(_model("claude", context_window=200000))
        model = await registry.get("claude")
        assert model.context_window == 200000

    @pytest.mark.asyncio
    async def test_list_models(self, registry: ModelRegistry):
        await registry.register(_model("claude"))
        await registry.register(_model("gpt4", provider="openai"))
        models = await registry.list_models()
        assert len(models) == 2


# ── Unregister ───────────────────────────────────────────────────────


class TestUnregister:
    @pytest.mark.asyncio
    async def test_unregister_existing(self, registry: ModelRegistry):
        await registry.register(_model("claude"))
        result = await registry.unregister("claude")
        assert result is True
        assert await registry.get("claude") is None

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self, registry: ModelRegistry):
        result = await registry.unregister("nope")
        assert result is False

    @pytest.mark.asyncio
    async def test_unregister_removes_performance(self, registry: ModelRegistry):
        await registry.register(_model("claude"))
        await registry.record_performance(
            "claude", "latest", ModelPerformance(task="qa", accuracy=0.9)
        )
        await registry.unregister("claude")
        perfs = await registry.get_performance("claude")
        assert perfs == []


# ── Capability lookup ────────────────────────────────────────────────


class TestCapabilityLookup:
    @pytest.mark.asyncio
    async def test_lookup_by_capability(self, registry: ModelRegistry):
        await registry.register(
            _model("claude", capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING])
        )
        await registry.register(
            _model("embed-model", provider="openai", capabilities=[ModelCapability.EMBEDDING])
        )

        text_models = await registry.lookup_by_capability(ModelCapability.TEXT_GENERATION)
        assert len(text_models) == 1
        assert text_models[0].name == "claude"

        embed_models = await registry.lookup_by_capability(ModelCapability.EMBEDDING)
        assert len(embed_models) == 1
        assert embed_models[0].name == "embed-model"

    @pytest.mark.asyncio
    async def test_lookup_no_match(self, registry: ModelRegistry):
        await registry.register(_model("claude"))
        results = await registry.lookup_by_capability(ModelCapability.EMBEDDING)
        assert results == []


# ── Best for task ────────────────────────────────────────────────────


class TestBestForTask:
    @pytest.mark.asyncio
    async def test_best_for_task_no_candidates(self, registry: ModelRegistry):
        result = await registry.best_for_task(ModelCapability.EMBEDDING)
        assert result is None

    @pytest.mark.asyncio
    async def test_best_for_task_without_task_name(self, registry: ModelRegistry):
        await registry.register(_model("claude"))
        result = await registry.best_for_task(ModelCapability.TEXT_GENERATION)
        assert result is not None
        assert result.name == "claude"

    @pytest.mark.asyncio
    async def test_best_for_task_with_performance(self, registry: ModelRegistry):
        await registry.register(_model("claude-a", version="1"))
        await registry.register(_model("claude-b", version="1"))

        await registry.record_performance(
            "claude-a", "1", ModelPerformance(task="qa", accuracy=0.7)
        )
        await registry.record_performance(
            "claude-b", "1", ModelPerformance(task="qa", accuracy=0.95)
        )

        best = await registry.best_for_task(ModelCapability.TEXT_GENERATION, task="qa")
        assert best.name == "claude-b"

    @pytest.mark.asyncio
    async def test_best_for_task_no_perf_data(self, registry: ModelRegistry):
        await registry.register(_model("claude"))
        # No performance recorded, should still return a model
        result = await registry.best_for_task(ModelCapability.TEXT_GENERATION, task="qa")
        assert result is not None


# ── Performance tracking ─────────────────────────────────────────────


class TestPerformanceTracking:
    @pytest.mark.asyncio
    async def test_record_and_get_performance(self, registry: ModelRegistry):
        await registry.register(_model("claude"))
        await registry.record_performance(
            "claude", "latest", ModelPerformance(task="qa", accuracy=0.9, latency_ms=150.0)
        )
        perfs = await registry.get_performance("claude")
        assert len(perfs) == 1
        assert perfs[0].task == "qa"
        assert perfs[0].accuracy == 0.9
        assert perfs[0].sample_count == 1

    @pytest.mark.asyncio
    async def test_running_average_accuracy(self, registry: ModelRegistry):
        await registry.register(_model("claude"))
        await registry.record_performance(
            "claude", "latest", ModelPerformance(task="qa", accuracy=0.8)
        )
        await registry.record_performance(
            "claude", "latest", ModelPerformance(task="qa", accuracy=1.0)
        )
        perfs = await registry.get_performance("claude")
        assert len(perfs) == 1
        # Running average: (0.8 * 1 + 1.0) / 2 = 0.9
        assert perfs[0].accuracy == pytest.approx(0.9)
        assert perfs[0].sample_count == 2

    @pytest.mark.asyncio
    async def test_running_average_latency(self, registry: ModelRegistry):
        await registry.register(_model("claude"))
        await registry.record_performance(
            "claude", "latest", ModelPerformance(task="qa", latency_ms=100.0)
        )
        await registry.record_performance(
            "claude", "latest", ModelPerformance(task="qa", latency_ms=200.0)
        )
        perfs = await registry.get_performance("claude")
        assert perfs[0].latency_ms == pytest.approx(150.0)

    @pytest.mark.asyncio
    async def test_multiple_tasks(self, registry: ModelRegistry):
        await registry.register(_model("claude"))
        await registry.record_performance(
            "claude", "latest", ModelPerformance(task="qa", accuracy=0.9)
        )
        await registry.record_performance(
            "claude", "latest", ModelPerformance(task="summarization", accuracy=0.85)
        )
        perfs = await registry.get_performance("claude")
        assert len(perfs) == 2
        tasks = {p.task for p in perfs}
        assert tasks == {"qa", "summarization"}

    @pytest.mark.asyncio
    async def test_record_for_unregistered_model(self, registry: ModelRegistry):
        # Should not raise; creates performance entry for unknown model
        await registry.record_performance(
            "unknown", "latest", ModelPerformance(task="qa", accuracy=0.5)
        )
        perfs = await registry.get_performance("unknown")
        assert len(perfs) == 1

    @pytest.mark.asyncio
    async def test_get_performance_empty(self, registry: ModelRegistry):
        perfs = await registry.get_performance("nonexistent")
        assert perfs == []

    @pytest.mark.asyncio
    async def test_accuracy_none_then_set(self, registry: ModelRegistry):
        await registry.register(_model("claude"))
        # First record with no accuracy
        await registry.record_performance(
            "claude", "latest", ModelPerformance(task="qa", latency_ms=100.0)
        )
        # Second record sets accuracy
        await registry.record_performance(
            "claude", "latest", ModelPerformance(task="qa", accuracy=0.9)
        )
        perfs = await registry.get_performance("claude")
        assert perfs[0].accuracy == 0.9


# ── ModelCapability enum ─────────────────────────────────────────────


class TestModelCapability:
    def test_values(self):
        assert ModelCapability.TEXT_GENERATION == "text_generation"
        assert ModelCapability.EMBEDDING == "embedding"
        assert ModelCapability.REASONING == "reasoning"
        assert ModelCapability.EXTRACTION == "extraction"


# ── ModelInfo ────────────────────────────────────────────────────────


class TestModelInfo:
    def test_defaults(self):
        m = ModelInfo(name="test", provider="test")
        assert m.version == "latest"
        assert m.capabilities == []
        assert m.context_window == 0
        assert m.cost_per_1k_input == 0.0

    def test_serialization(self):
        m = _model("claude", "3.5")
        data = m.model_dump()
        restored = ModelInfo(**data)
        assert restored.name == "claude"
        assert ModelCapability.TEXT_GENERATION in restored.capabilities
