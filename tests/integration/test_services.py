"""Integration tests for external service connections.

These tests require running Neo4j and Qdrant instances.
Skip by default; run with: pytest -m integration
"""

from __future__ import annotations

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_neo4j_connection():
    """Test that Neo4j is reachable and accepts queries."""
    from auto_researcher.config import Neo4jConfig
    from auto_researcher.infrastructure.knowledge_graph import KnowledgeGraphClient

    config = Neo4jConfig()
    client = KnowledgeGraphClient(config)
    try:
        await client.connect()
        # Simple connectivity check
        result = await client.run_query("RETURN 1 AS n")
        assert result is not None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_qdrant_connection():
    """Test that Qdrant is reachable."""
    from auto_researcher.config import QdrantConfig
    from auto_researcher.infrastructure.vector_store import VectorStoreClient

    config = QdrantConfig()
    client = VectorStoreClient(config)
    try:
        await client.connect()
        collections = await client.list_collections()
        assert isinstance(collections, list)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_neo4j_claim_crud():
    """Test basic claim CRUD operations in Neo4j."""
    from auto_researcher.config import Neo4jConfig
    from auto_researcher.infrastructure.knowledge_graph import KnowledgeGraphClient
    from auto_researcher.models.claim import Claim, ClaimRelation

    config = Neo4jConfig()
    client = KnowledgeGraphClient(config)
    try:
        await client.connect()
        claim = Claim(
            id="test-integration-1",
            entity_1="TestModel",
            relation=ClaimRelation.OUTPERFORMS,
            entity_2="Baseline",
            confidence=0.9,
            source_paper_ids=["test-paper"],
        )
        await client.add_claim(claim)
        retrieved = await client.get_claims_for_entity("TestModel")
        assert any(c.id == "test-integration-1" for c in retrieved)
    finally:
        # Cleanup
        try:
            await client.run_query("MATCH (n) WHERE n.id = 'test-integration-1' DETACH DELETE n")
        except Exception:
            pass
        await client.close()


@pytest.mark.asyncio
async def test_qdrant_embedding_store():
    """Test storing and retrieving embeddings from Qdrant."""
    from auto_researcher.config import QdrantConfig
    from auto_researcher.infrastructure.vector_store import VectorStoreClient

    config = QdrantConfig()
    client = VectorStoreClient(config)
    try:
        await client.connect()
        test_embedding = [0.1] * config.embedding_dim
        await client.upsert("test-doc-1", test_embedding, {"title": "Test"})
        results = await client.search(test_embedding, limit=1)
        assert len(results) >= 1
    finally:
        try:
            await client.delete("test-doc-1")
        except Exception:
            pass
        await client.close()


@pytest.mark.asyncio
async def test_arxiv_fetch():
    """Test fetching papers from ArXiv API."""
    from auto_researcher.config import ArxivConfig
    from auto_researcher.ingestion.arxiv_monitor import ArxivMonitor

    config = ArxivConfig(max_results_per_query=5)
    monitor = ArxivMonitor(config)
    papers = await monitor.fetch_recent(categories=["cs.AI"], max_results=2)
    assert isinstance(papers, list)
    # ArXiv should return at least 1 paper for cs.AI
    assert len(papers) >= 1
    assert papers[0].metadata.title
