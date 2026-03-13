"""Tests for the EpisodicMemoryStore."""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from auto_researcher.infrastructure.episodic_memory import EpisodicMemoryStore
from auto_researcher.models.memory import (
    EpisodicEntry,
    MetaMemoryEntry,
    ProceduralEntry,
)


@pytest_asyncio.fixture
async def store() -> EpisodicMemoryStore:
    return EpisodicMemoryStore()


def _episode(content: str = "test", **kwargs) -> EpisodicEntry:
    return EpisodicEntry(content=content, **kwargs)


# ── Episodic store/retrieve ──────────────────────────────────────────


class TestEpisodicStoreRetrieve:
    @pytest.mark.asyncio
    async def test_store_assigns_id(self, store: EpisodicMemoryStore):
        entry = _episode("learned something")
        result = await store.store_episode(entry)
        assert result.id != ""
        assert store.episode_count == 1

    @pytest.mark.asyncio
    async def test_store_preserves_existing_id(self, store: EpisodicMemoryStore):
        entry = _episode("test", id="custom-id")
        # id is truthy so it should be preserved
        # Actually, empty string is falsy but "custom-id" is truthy
        result = await store.store_episode(entry)
        assert result.id == "custom-id"

    @pytest.mark.asyncio
    async def test_get_episode(self, store: EpisodicMemoryStore):
        entry = await store.store_episode(_episode("hello"))
        retrieved = await store.get_episode(entry.id)
        assert retrieved is not None
        assert retrieved.content == "hello"

    @pytest.mark.asyncio
    async def test_get_episode_not_found(self, store: EpisodicMemoryStore):
        assert await store.get_episode("nope") is None

    @pytest.mark.asyncio
    async def test_get_episode_increments_access(self, store: EpisodicMemoryStore):
        entry = await store.store_episode(_episode("test"))
        assert entry.access_count == 0
        await store.get_episode(entry.id)
        assert entry.access_count == 1
        await store.get_episode(entry.id)
        assert entry.access_count == 2

    @pytest.mark.asyncio
    async def test_get_episode_sets_accessed_at(self, store: EpisodicMemoryStore):
        entry = await store.store_episode(_episode("test"))
        assert entry.accessed_at is None
        await store.get_episode(entry.id)
        assert entry.accessed_at is not None


# ── Tag-based retrieval ──────────────────────────────────────────────


class TestTagRetrieval:
    @pytest.mark.asyncio
    async def test_retrieve_by_tags(self, store: EpisodicMemoryStore):
        await store.store_episode(_episode("a", tags=["ml", "nlp"]))
        await store.store_episode(_episode("b", tags=["cv"]))
        await store.store_episode(_episode("c", tags=["nlp", "transformers"]))

        results = await store.retrieve_by_tags(["nlp"])
        assert len(results) == 2
        contents = {r.content for r in results}
        assert contents == {"a", "c"}

    @pytest.mark.asyncio
    async def test_retrieve_by_tags_sorted_by_importance(self, store: EpisodicMemoryStore):
        await store.store_episode(_episode("low", tags=["ml"], importance=0.2))
        await store.store_episode(_episode("high", tags=["ml"], importance=0.9))

        results = await store.retrieve_by_tags(["ml"])
        assert results[0].content == "high"

    @pytest.mark.asyncio
    async def test_retrieve_by_tags_with_limit(self, store: EpisodicMemoryStore):
        for i in range(10):
            await store.store_episode(_episode(f"ep{i}", tags=["shared"]))

        results = await store.retrieve_by_tags(["shared"], limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_retrieve_by_tags_no_match(self, store: EpisodicMemoryStore):
        await store.store_episode(_episode("test", tags=["ml"]))
        results = await store.retrieve_by_tags(["biology"])
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_by_tags_touches_entries(self, store: EpisodicMemoryStore):
        entry = await store.store_episode(_episode("test", tags=["ml"]))
        await store.retrieve_by_tags(["ml"])
        assert entry.access_count == 1


# ── Recency-based retrieval ──────────────────────────────────────────


class TestRecentRetrieval:
    @pytest.mark.asyncio
    async def test_retrieve_recent(self, store: EpisodicMemoryStore):
        old = _episode("old", created_at=datetime(2024, 1, 1))
        new = _episode("new", created_at=datetime(2024, 6, 1))
        await store.store_episode(old)
        await store.store_episode(new)

        results = await store.retrieve_recent(limit=1)
        assert len(results) == 1
        assert results[0].content == "new"

    @pytest.mark.asyncio
    async def test_retrieve_recent_touches_entries(self, store: EpisodicMemoryStore):
        entry = await store.store_episode(_episode("test"))
        await store.retrieve_recent()
        assert entry.access_count == 1


# ── Most accessed retrieval ──────────────────────────────────────────


class TestMostAccessedRetrieval:
    @pytest.mark.asyncio
    async def test_retrieve_most_accessed(self, store: EpisodicMemoryStore):
        e1 = await store.store_episode(_episode("popular"))
        e2 = await store.store_episode(_episode("unpopular"))
        # Manually set access counts
        e1.access_count = 10
        e2.access_count = 1

        results = await store.retrieve_most_accessed(limit=1)
        assert results[0].content == "popular"


# ── Similarity-based retrieval ───────────────────────────────────────


class TestSimilarityRetrieval:
    @pytest.mark.asyncio
    async def test_no_vector_store_returns_empty(self, store: EpisodicMemoryStore):
        results = await store.retrieve_by_similarity([0.1, 0.2])
        assert results == []

    @pytest.mark.asyncio
    async def test_with_vector_store(self):
        mock_vs = AsyncMock()
        mock_vs.search.return_value = [
            {"id": "p1", "payload": {"entry_id": "e1"}, "score": 0.95}
        ]
        store = EpisodicMemoryStore(vector_store=mock_vs)
        entry = _episode("test", id="e1")
        # Store without embedding to skip vector store upsert check
        store._episodes["e1"] = entry

        results = await store.retrieve_by_similarity([0.1, 0.2], limit=5)
        assert len(results) == 1
        assert results[0].content == "test"
        mock_vs.search.assert_awaited_once()


# ── Vector store integration ─────────────────────────────────────────


class TestVectorStoreIntegration:
    @pytest.mark.asyncio
    async def test_store_with_embedding_calls_upsert(self):
        mock_vs = AsyncMock()
        store = EpisodicMemoryStore(vector_store=mock_vs)
        entry = _episode("test", embedding=[0.1, 0.2, 0.3])
        await store.store_episode(entry)
        mock_vs.upsert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_store_without_embedding_skips_upsert(self):
        mock_vs = AsyncMock()
        store = EpisodicMemoryStore(vector_store=mock_vs)
        entry = _episode("test")
        await store.store_episode(entry)
        mock_vs.upsert.assert_not_awaited()


# ── Meta-memory ──────────────────────────────────────────────────────


class TestMetaMemory:
    @pytest.mark.asyncio
    async def test_store_and_get(self, store: EpisodicMemoryStore):
        meta = MetaMemoryEntry(topic="transformers", competence_level=0.8, confidence=0.7)
        await store.store_meta(meta)
        retrieved = await store.get_meta("transformers")
        assert retrieved is not None
        assert retrieved.competence_level == 0.8

    @pytest.mark.asyncio
    async def test_get_missing(self, store: EpisodicMemoryStore):
        assert await store.get_meta("unknown") is None

    @pytest.mark.asyncio
    async def test_list_meta(self, store: EpisodicMemoryStore):
        await store.store_meta(MetaMemoryEntry(topic="a"))
        await store.store_meta(MetaMemoryEntry(topic="b"))
        metas = await store.list_meta()
        assert len(metas) == 2

    @pytest.mark.asyncio
    async def test_assess_competence(self, store: EpisodicMemoryStore):
        await store.store_meta(MetaMemoryEntry(topic="ml", competence_level=0.75))
        assert await store.assess_competence("ml") == 0.75

    @pytest.mark.asyncio
    async def test_assess_competence_unknown(self, store: EpisodicMemoryStore):
        assert await store.assess_competence("unknown") == 0.0

    @pytest.mark.asyncio
    async def test_store_meta_overwrites(self, store: EpisodicMemoryStore):
        await store.store_meta(MetaMemoryEntry(topic="ml", competence_level=0.5))
        await store.store_meta(MetaMemoryEntry(topic="ml", competence_level=0.9))
        meta = await store.get_meta("ml")
        assert meta.competence_level == 0.9


# ── Procedural memory ────────────────────────────────────────────────


class TestProceduralMemory:
    @pytest.mark.asyncio
    async def test_store_and_get(self, store: EpisodicMemoryStore):
        proc = ProceduralEntry(name="search_papers", description="Search arxiv")
        result = await store.store_procedure(proc)
        assert result.id != ""
        assert store.procedure_count == 1

        retrieved = await store.get_procedure(result.id)
        assert retrieved is not None
        assert retrieved.name == "search_papers"

    @pytest.mark.asyncio
    async def test_get_missing_procedure(self, store: EpisodicMemoryStore):
        assert await store.get_procedure("missing") is None

    @pytest.mark.asyncio
    async def test_find_by_name(self, store: EpisodicMemoryStore):
        await store.store_procedure(ProceduralEntry(name="Search Papers", description="d"))
        await store.store_procedure(ProceduralEntry(name="Analyze Claims", description="d"))
        await store.store_procedure(ProceduralEntry(name="Paper Summarizer", description="d"))

        results = await store.find_procedures_by_name("paper")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_record_procedure_use_success(self, store: EpisodicMemoryStore):
        proc = await store.store_procedure(
            ProceduralEntry(name="test", description="d", success_rate=0.5)
        )
        await store.record_procedure_use(proc.id, success=True)
        assert proc.use_count == 1
        assert proc.last_used is not None
        # EMA: 0.5 * 0.9 + 1.0 * 0.1 = 0.55
        assert proc.success_rate == pytest.approx(0.55)

    @pytest.mark.asyncio
    async def test_record_procedure_use_failure(self, store: EpisodicMemoryStore):
        proc = await store.store_procedure(
            ProceduralEntry(name="test", description="d", success_rate=0.5)
        )
        await store.record_procedure_use(proc.id, success=False)
        # EMA: 0.5 * 0.9 + 0.0 * 0.1 = 0.45
        assert proc.success_rate == pytest.approx(0.45)

    @pytest.mark.asyncio
    async def test_record_procedure_use_missing(self, store: EpisodicMemoryStore):
        # Should not raise
        await store.record_procedure_use("missing", success=True)


# ── Consolidation ────────────────────────────────────────────────────


class TestConsolidation:
    @pytest.mark.asyncio
    async def test_consolidate_without_vector_store(self, store: EpisodicMemoryStore):
        removed = await store.consolidate()
        assert removed == 0

    @pytest.mark.asyncio
    async def test_consolidate_deduplicates(self):
        mock_vs = AsyncMock()
        store = EpisodicMemoryStore(vector_store=mock_vs)

        e1 = _episode("high importance", id="e1", importance=0.9, embedding=[0.1])
        e2 = _episode("duplicate", id="e2", importance=0.3, embedding=[0.1], tags=["extra"])
        store._episodes["e1"] = e1
        store._episodes["e2"] = e2

        # When searching for similar to e1, return e2 as near-duplicate
        mock_vs.search.return_value = [
            {"id": "e2", "payload": {"entry_id": "e2"}, "score": 0.95}
        ]

        removed = await store.consolidate(dedup_threshold=0.92)
        assert removed == 1
        assert "e1" in store._episodes
        assert "e2" not in store._episodes
        # Tags should be merged
        assert "extra" in store._episodes["e1"].tags

    @pytest.mark.asyncio
    async def test_consolidate_merges_access_counts(self):
        mock_vs = AsyncMock()
        store = EpisodicMemoryStore(vector_store=mock_vs)

        e1 = _episode("main", id="e1", importance=0.9, embedding=[0.1])
        e1.access_count = 5
        e2 = _episode("dup", id="e2", importance=0.3, embedding=[0.1])
        e2.access_count = 3
        store._episodes["e1"] = e1
        store._episodes["e2"] = e2

        mock_vs.search.return_value = [
            {"id": "e2", "payload": {"entry_id": "e2"}, "score": 0.95}
        ]

        await store.consolidate()
        assert store._episodes["e1"].access_count == 8


# ── Properties ───────────────────────────────────────────────────────


class TestProperties:
    @pytest.mark.asyncio
    async def test_episode_count(self, store: EpisodicMemoryStore):
        assert store.episode_count == 0
        await store.store_episode(_episode("a"))
        assert store.episode_count == 1

    @pytest.mark.asyncio
    async def test_procedure_count(self, store: EpisodicMemoryStore):
        assert store.procedure_count == 0
        await store.store_procedure(ProceduralEntry(name="p", description="d"))
        assert store.procedure_count == 1
