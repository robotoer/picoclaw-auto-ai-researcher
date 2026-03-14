"""Infrastructure layer: knowledge graph, vector store, gap map, memory."""

from auto_researcher.infrastructure.episodic_memory import EpisodicMemoryStore
from auto_researcher.infrastructure.gap_map import GapMap
from auto_researcher.infrastructure.model_registry import ModelRegistry

__all__ = [
    "EpisodicMemoryStore",
    "GapMap",
    "KnowledgeGraphClient",
    "ModelRegistry",
    "VectorStoreClient",
]


def __getattr__(name: str) -> type:
    if name == "KnowledgeGraphClient":
        from auto_researcher.infrastructure.knowledge_graph import KnowledgeGraphClient

        return KnowledgeGraphClient
    if name == "VectorStoreClient":
        from auto_researcher.infrastructure.vector_store import VectorStoreClient

        return VectorStoreClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
