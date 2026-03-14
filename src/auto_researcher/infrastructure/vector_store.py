"""Qdrant-based vector store for paper embeddings and semantic search."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from auto_researcher.config import QdrantConfig

logger = logging.getLogger(__name__)


class VectorStoreClient:
    """Async Qdrant client for storing and querying paper embeddings."""

    def __init__(self, config: QdrantConfig) -> None:
        self._config = config
        self._client: AsyncQdrantClient | None = None

    async def connect(self) -> None:
        self._client = AsyncQdrantClient(
            host=self._config.host,
            port=self._config.port,
        )
        await self._ensure_collection()

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("VectorStoreClient is not connected. Call connect() first.")
        return self._client

    async def _ensure_collection(self) -> None:
        collections = await self.client.get_collections()
        existing = {c.name for c in collections.collections}
        if self._config.collection_name not in existing:
            await self.client.create_collection(
                collection_name=self._config.collection_name,
                vectors_config=VectorParams(
                    size=self._config.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created collection %s", self._config.collection_name)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def upsert(
        self,
        vector: list[float],
        payload: dict[str, Any],
        point_id: str | None = None,
    ) -> str:
        """Store a single embedding with metadata payload. Returns the point id."""
        pid = point_id or str(uuid.uuid4())
        point = PointStruct(id=pid, vector=vector, payload=payload)
        await self.client.upsert(
            collection_name=self._config.collection_name,
            points=[point],
        )
        return pid

    async def upsert_batch(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        point_ids: list[str] | None = None,
    ) -> list[str]:
        """Batch upsert embeddings. Returns list of point ids."""
        if point_ids is None:
            point_ids = [str(uuid.uuid4()) for _ in vectors]

        points = [
            PointStruct(id=pid, vector=vec, payload=pl)
            for pid, vec, pl in zip(point_ids, vectors, payloads)
        ]

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self.client.upsert(
                collection_name=self._config.collection_name,
                points=batch,
            )

        logger.info("Upserted %d points to %s", len(points), self._config.collection_name)
        return point_ids

    # ------------------------------------------------------------------
    # Search operations
    # ------------------------------------------------------------------

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float | None = None,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic similarity search. Returns list of {id, score, payload}."""
        qdrant_filter = self._build_filter(filter_conditions) if filter_conditions else None

        results = await self.client.search(
            collection_name=self._config.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )

        return [
            {"id": str(r.id), "score": r.score, "payload": r.payload or {}}
            for r in results
        ]

    async def knn(
        self,
        query_vector: list[float],
        k: int = 10,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """K-nearest neighbor search for novelty computation.

        Returns results sorted by distance (most similar first).
        """
        return await self.search(
            query_vector=query_vector,
            limit=k,
            filter_conditions=filter_conditions,
        )

    async def compute_novelty_score(self, query_vector: list[float], k: int = 10) -> float:
        """Compute a novelty score based on average distance to k nearest neighbors.

        Returns a value in [0, 1] where higher means more novel (more distant from existing).
        Cosine similarity in [0, 1] from Qdrant: novelty = 1 - avg_similarity.
        """
        neighbors = await self.knn(query_vector, k=k)
        if not neighbors:
            return 1.0
        avg_similarity = sum(n["score"] for n in neighbors) / len(neighbors)
        return float(1.0 - avg_similarity)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete(self, point_ids: list[str]) -> None:
        await self.client.delete(
            collection_name=self._config.collection_name,
            points_selector=point_ids,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter(conditions: dict[str, Any]) -> Filter:
        """Build a Qdrant filter from a simple {field: value} dict."""
        must = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in conditions.items()
        ]
        return Filter(must=must)
