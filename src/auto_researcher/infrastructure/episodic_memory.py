"""Structured episodic memory with semantic retrieval and access tracking."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from auto_researcher.models.memory import EpisodicEntry, MetaMemoryEntry, MemoryType, ProceduralEntry

logger = logging.getLogger(__name__)


class EpisodicMemoryStore:
    """In-memory episodic memory store with semantic retrieval support.

    Uses an optional vector store client for embedding-based retrieval.
    Falls back to tag-based and recency-based retrieval when no vector store is provided.
    """

    def __init__(self, vector_store=None, collection_name: str = "episodic_memory") -> None:
        self._vector_store = vector_store
        self._collection_name = collection_name
        self._episodes: dict[str, EpisodicEntry] = {}
        self._meta_memories: dict[str, MetaMemoryEntry] = {}
        self._procedural: dict[str, ProceduralEntry] = {}

    # ------------------------------------------------------------------
    # Episodic entries
    # ------------------------------------------------------------------

    async def store_episode(self, entry: EpisodicEntry) -> EpisodicEntry:
        if not entry.id:
            entry.id = str(uuid.uuid4())

        self._episodes[entry.id] = entry

        # Store embedding in vector store if available.
        if self._vector_store and entry.embedding:
            await self._vector_store.upsert(
                vector=entry.embedding,
                payload={
                    "entry_id": entry.id,
                    "memory_type": entry.memory_type.value,
                    "content": entry.content,
                    "tags": entry.tags,
                    "source": entry.source,
                    "importance": entry.importance,
                    "created_at": entry.created_at.isoformat(),
                },
                point_id=entry.id,
            )

        logger.debug("Stored episode %s", entry.id)
        return entry

    async def get_episode(self, entry_id: str) -> EpisodicEntry | None:
        entry = self._episodes.get(entry_id)
        if entry:
            self._touch(entry)
        return entry

    async def retrieve_by_tags(self, tags: list[str], limit: int = 20) -> list[EpisodicEntry]:
        """Retrieve episodes that match any of the given tags, sorted by importance."""
        tag_set = set(tags)
        matches = [
            ep for ep in self._episodes.values()
            if tag_set & set(ep.tags)
        ]
        matches.sort(key=lambda e: e.importance, reverse=True)
        for ep in matches[:limit]:
            self._touch(ep)
        return matches[:limit]

    async def retrieve_by_similarity(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[EpisodicEntry]:
        """Retrieve episodes by semantic similarity using vector store."""
        if not self._vector_store:
            logger.warning("No vector store configured for semantic retrieval")
            return []

        results = await self._vector_store.search(
            query_vector=query_embedding,
            limit=limit,
            score_threshold=min_score if min_score > 0 else None,
            filter_conditions={"memory_type": MemoryType.EPISODIC.value},
        )

        episodes: list[EpisodicEntry] = []
        for r in results:
            entry_id = r["payload"].get("entry_id", r["id"])
            ep = self._episodes.get(entry_id)
            if ep:
                self._touch(ep)
                episodes.append(ep)
        return episodes

    async def retrieve_recent(self, limit: int = 20) -> list[EpisodicEntry]:
        """Retrieve the most recently created episodes."""
        sorted_eps = sorted(self._episodes.values(), key=lambda e: e.created_at, reverse=True)
        for ep in sorted_eps[:limit]:
            self._touch(ep)
        return sorted_eps[:limit]

    async def retrieve_most_accessed(self, limit: int = 20) -> list[EpisodicEntry]:
        """Retrieve the most frequently accessed episodes."""
        sorted_eps = sorted(self._episodes.values(), key=lambda e: e.access_count, reverse=True)
        return sorted_eps[:limit]

    # ------------------------------------------------------------------
    # Meta-memory
    # ------------------------------------------------------------------

    async def store_meta(self, entry: MetaMemoryEntry) -> MetaMemoryEntry:
        self._meta_memories[entry.topic] = entry
        return entry

    async def get_meta(self, topic: str) -> MetaMemoryEntry | None:
        return self._meta_memories.get(topic)

    async def list_meta(self) -> list[MetaMemoryEntry]:
        return list(self._meta_memories.values())

    async def assess_competence(self, topic: str) -> float:
        """Return the system's self-assessed competence on a topic, or 0.0 if unknown."""
        meta = self._meta_memories.get(topic)
        return meta.competence_level if meta else 0.0

    # ------------------------------------------------------------------
    # Procedural memory
    # ------------------------------------------------------------------

    async def store_procedure(self, entry: ProceduralEntry) -> ProceduralEntry:
        if not entry.id:
            entry.id = str(uuid.uuid4())
        self._procedural[entry.id] = entry
        return entry

    async def get_procedure(self, procedure_id: str) -> ProceduralEntry | None:
        return self._procedural.get(procedure_id)

    async def find_procedures_by_name(self, name: str) -> list[ProceduralEntry]:
        name_lower = name.lower()
        return [p for p in self._procedural.values() if name_lower in p.name.lower()]

    async def record_procedure_use(self, procedure_id: str, success: bool) -> None:
        """Record that a procedure was used and update its success rate."""
        proc = self._procedural.get(procedure_id)
        if proc is None:
            return
        proc.use_count += 1
        proc.last_used = datetime.utcnow()
        # Exponential moving average for success rate.
        alpha = 0.1
        proc.success_rate = proc.success_rate * (1 - alpha) + (1.0 if success else 0.0) * alpha

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    async def consolidate(self, dedup_threshold: float = 0.92) -> int:
        """Consolidate memory: deduplicate similar episodes, promote important ones.

        Returns the number of episodes removed.
        """
        if not self._vector_store:
            return 0

        removed = 0
        seen_ids: set[str] = set()
        sorted_eps = sorted(self._episodes.values(), key=lambda e: e.importance, reverse=True)

        for ep in sorted_eps:
            if ep.id in seen_ids or ep.embedding is None:
                continue

            # Find near-duplicates.
            similar = await self._vector_store.search(
                query_vector=ep.embedding,
                limit=5,
                score_threshold=dedup_threshold,
            )

            for s in similar:
                dup_id = s["payload"].get("entry_id", s["id"])
                if dup_id != ep.id and dup_id not in seen_ids and dup_id in self._episodes:
                    dup = self._episodes[dup_id]
                    # Merge: keep the higher-importance entry, absorb tags.
                    ep.tags = list(set(ep.tags) | set(dup.tags))
                    ep.access_count += dup.access_count
                    del self._episodes[dup_id]
                    seen_ids.add(dup_id)
                    removed += 1

            seen_ids.add(ep.id)

        if removed:
            logger.info("Consolidated memory: removed %d duplicate episodes", removed)
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _touch(entry: EpisodicEntry) -> None:
        entry.accessed_at = datetime.utcnow()
        entry.access_count += 1

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    @property
    def procedure_count(self) -> int:
        return len(self._procedural)
