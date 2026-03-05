"""Tests for the GapMap infrastructure component."""

import pytest
import pytest_asyncio

from auto_researcher.config import GapMapConfig
from auto_researcher.infrastructure.gap_map import GapMap, NEGATIVE_EDGE_TYPES
from auto_researcher.models.gap import Gap, GapEdge, GapNode, GapStatus, GapType


@pytest_asyncio.fixture
async def gap_map() -> GapMap:
    config = GapMapConfig(frontier_threshold=0.5)
    return GapMap(config)


def _node(id: str, label: str = "", coverage: float = 0.0, paper_count: int = 0) -> GapNode:
    return GapNode(
        id=id,
        node_type="concept",
        label=label or id,
        coverage_score=coverage,
        paper_count=paper_count,
    )


def _edge(src: str, tgt: str, edge_type: str = "builds_on", **kwargs) -> GapEdge:
    return GapEdge(source_id=src, target_id=tgt, edge_type=edge_type, **kwargs)


def _gap(id: str = "g1", **kwargs) -> Gap:
    defaults = dict(gap_type=GapType.EMPIRICAL, description="test gap")
    defaults.update(kwargs)
    return Gap(id=id, **defaults)


# ── Node operations ──────────────────────────────────────────────────


class TestNodeOperations:
    @pytest.mark.asyncio
    async def test_add_and_get_node(self, gap_map: GapMap):
        node = _node("n1", "Transformers", 0.3)
        await gap_map.add_node(node)
        retrieved = await gap_map.get_node("n1")
        assert retrieved is not None
        assert retrieved.label == "Transformers"
        assert retrieved.coverage_score == 0.3

    @pytest.mark.asyncio
    async def test_get_nonexistent_node(self, gap_map: GapMap):
        assert await gap_map.get_node("nonexistent") is None

    @pytest.mark.asyncio
    async def test_node_count(self, gap_map: GapMap):
        assert gap_map.node_count == 0
        await gap_map.add_node(_node("n1"))
        await gap_map.add_node(_node("n2"))
        assert gap_map.node_count == 2

    @pytest.mark.asyncio
    async def test_update_node_coverage(self, gap_map: GapMap):
        await gap_map.add_node(_node("n1", coverage=0.3))
        await gap_map.update_node_coverage("n1", 0.7, paper_count_delta=2)
        node = await gap_map.get_node("n1")
        assert node.coverage_score == 0.7
        assert node.paper_count == 2

    @pytest.mark.asyncio
    async def test_update_coverage_clamped(self, gap_map: GapMap):
        await gap_map.add_node(_node("n1", coverage=0.5))
        await gap_map.update_node_coverage("n1", 1.5)
        node = await gap_map.get_node("n1")
        assert node.coverage_score == 1.0

        await gap_map.update_node_coverage("n1", -0.5)
        node = await gap_map.get_node("n1")
        assert node.coverage_score == 0.0

    @pytest.mark.asyncio
    async def test_update_nonexistent_node(self, gap_map: GapMap):
        # Should not raise
        await gap_map.update_node_coverage("missing", 0.5)

    @pytest.mark.asyncio
    async def test_all_nodes(self, gap_map: GapMap):
        await gap_map.add_node(_node("n1"))
        await gap_map.add_node(_node("n2"))
        nodes = gap_map.all_nodes()
        assert len(nodes) == 2


# ── Edge operations ──────────────────────────────────────────────────


class TestEdgeOperations:
    @pytest.mark.asyncio
    async def test_add_edge(self, gap_map: GapMap):
        await gap_map.add_node(_node("a"))
        await gap_map.add_node(_node("b"))
        await gap_map.add_edge(_edge("a", "b"))
        assert gap_map.edge_count == 1

    @pytest.mark.asyncio
    async def test_add_edge_updates_adjacency(self, gap_map: GapMap):
        await gap_map.add_node(_node("a"))
        await gap_map.add_node(_node("b"))
        await gap_map.add_edge(_edge("a", "b"))
        a = await gap_map.get_node("a")
        b = await gap_map.get_node("b")
        assert "b" in a.adjacent_node_ids
        assert "a" in b.adjacent_node_ids

    @pytest.mark.asyncio
    async def test_get_edges(self, gap_map: GapMap):
        await gap_map.add_node(_node("a"))
        await gap_map.add_node(_node("b"))
        await gap_map.add_node(_node("c"))
        await gap_map.add_edge(_edge("a", "b"))
        await gap_map.add_edge(_edge("c", "a"))
        edges = await gap_map.get_edges("a")
        # a->b (outgoing) and c->a (incoming)
        assert len(edges) == 2

    @pytest.mark.asyncio
    async def test_negative_edge_explicit(self, gap_map: GapMap):
        await gap_map.add_node(_node("a"))
        await gap_map.add_node(_node("b"))
        await gap_map.add_edge(_edge("a", "b", is_negative=True))
        neg = await gap_map.get_negative_edges()
        assert len(neg) == 1

    @pytest.mark.asyncio
    async def test_negative_edge_by_type(self, gap_map: GapMap):
        # Use different node pairs since DiGraph overwrites edges between same pair.
        types = list(NEGATIVE_EDGE_TYPES)
        for i, edge_type in enumerate(types):
            src, tgt = f"a{i}", f"b{i}"
            await gap_map.add_node(_node(src))
            await gap_map.add_node(_node(tgt))
            await gap_map.add_edge(_edge(src, tgt, edge_type=edge_type))
        neg = await gap_map.get_negative_edges()
        assert len(neg) == len(NEGATIVE_EDGE_TYPES)

    @pytest.mark.asyncio
    async def test_no_duplicate_adjacency(self, gap_map: GapMap):
        await gap_map.add_node(_node("a"))
        await gap_map.add_node(_node("b"))
        await gap_map.add_edge(_edge("a", "b"))
        await gap_map.add_edge(_edge("a", "b", edge_type="improves_upon"))
        a = await gap_map.get_node("a")
        assert a.adjacent_node_ids.count("b") == 1


# ── Gap tracking ─────────────────────────────────────────────────────


class TestGapTracking:
    @pytest.mark.asyncio
    async def test_register_and_get_gap(self, gap_map: GapMap):
        gap = _gap("g1")
        await gap_map.register_gap(gap)
        retrieved = await gap_map.get_gap("g1")
        assert retrieved is not None
        assert retrieved.description == "test gap"

    @pytest.mark.asyncio
    async def test_register_gap_with_nodes(self, gap_map: GapMap):
        await gap_map.add_node(_node("n1"))
        gap = _gap("g1")
        await gap_map.register_gap(gap, node_ids=["n1"])
        node = await gap_map.get_node("n1")
        assert len(node.gaps) == 1

    @pytest.mark.asyncio
    async def test_update_gap_status(self, gap_map: GapMap):
        await gap_map.register_gap(_gap("g1"))
        await gap_map.update_gap_status("g1", GapStatus.FILLED)
        gap = await gap_map.get_gap("g1")
        assert gap.status == GapStatus.FILLED

    @pytest.mark.asyncio
    async def test_get_open_gaps(self, gap_map: GapMap):
        await gap_map.register_gap(_gap("g1"))
        await gap_map.register_gap(_gap("g2"))
        await gap_map.update_gap_status("g1", GapStatus.FILLED)
        open_gaps = await gap_map.get_open_gaps()
        assert len(open_gaps) == 1
        assert open_gaps[0].id == "g2"

    @pytest.mark.asyncio
    async def test_all_gaps(self, gap_map: GapMap):
        await gap_map.register_gap(_gap("g1"))
        await gap_map.register_gap(_gap("g2"))
        assert len(gap_map.all_gaps()) == 2


# ── Coverage density ─────────────────────────────────────────────────


class TestCoverageDensity:
    @pytest.mark.asyncio
    async def test_single_node(self, gap_map: GapMap):
        await gap_map.add_node(_node("n1", coverage=0.8))
        density = await gap_map.coverage_density("n1", radius=0)
        assert density == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_with_neighbors(self, gap_map: GapMap):
        await gap_map.add_node(_node("n1", coverage=0.8))
        await gap_map.add_node(_node("n2", coverage=0.4))
        await gap_map.add_node(_node("n3", coverage=0.6))
        await gap_map.add_edge(_edge("n1", "n2"))
        await gap_map.add_edge(_edge("n1", "n3"))
        density = await gap_map.coverage_density("n1", radius=1)
        assert density == pytest.approx((0.8 + 0.4 + 0.6) / 3)

    @pytest.mark.asyncio
    async def test_nonexistent_node(self, gap_map: GapMap):
        density = await gap_map.coverage_density("missing")
        assert density == 0.0

    @pytest.mark.asyncio
    async def test_region_density(self, gap_map: GapMap):
        await gap_map.add_node(_node("n1", coverage=0.4))
        await gap_map.add_node(_node("n2", coverage=0.6))
        density = await gap_map.region_density(["n1", "n2"])
        assert density == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_region_density_empty(self, gap_map: GapMap):
        assert await gap_map.region_density([]) == 0.0


# ── Frontier calculation ─────────────────────────────────────────────


class TestFrontier:
    @pytest.mark.asyncio
    async def test_frontier_basic(self, gap_map: GapMap):
        # n1 is well-covered, n2 is sparse => n1 is frontier
        await gap_map.add_node(_node("n1", coverage=0.8))
        await gap_map.add_node(_node("n2", coverage=0.2))
        await gap_map.add_edge(_edge("n1", "n2"))

        frontier = await gap_map.compute_frontier()
        assert len(frontier) == 1
        assert frontier[0].id == "n1"

    @pytest.mark.asyncio
    async def test_frontier_no_sparse_neighbors(self, gap_map: GapMap):
        # Both well-covered => no frontier
        await gap_map.add_node(_node("n1", coverage=0.8))
        await gap_map.add_node(_node("n2", coverage=0.9))
        await gap_map.add_edge(_edge("n1", "n2"))

        frontier = await gap_map.compute_frontier()
        assert len(frontier) == 0

    @pytest.mark.asyncio
    async def test_frontier_all_sparse(self, gap_map: GapMap):
        # Both sparse => no frontier (frontier requires well-covered node)
        await gap_map.add_node(_node("n1", coverage=0.1))
        await gap_map.add_node(_node("n2", coverage=0.2))
        await gap_map.add_edge(_edge("n1", "n2"))

        frontier = await gap_map.compute_frontier()
        assert len(frontier) == 0

    @pytest.mark.asyncio
    async def test_frontier_uses_predecessors(self, gap_map: GapMap):
        # n2 is well-covered, n1 is sparse, edge n1->n2
        # n2 should be frontier because predecessor n1 is sparse
        await gap_map.add_node(_node("n1", coverage=0.1))
        await gap_map.add_node(_node("n2", coverage=0.8))
        await gap_map.add_edge(_edge("n1", "n2"))

        frontier = await gap_map.compute_frontier()
        assert len(frontier) == 1
        assert frontier[0].id == "n2"


# ── Gap ranking ──────────────────────────────────────────────────────


class TestGapRanking:
    @pytest.mark.asyncio
    async def test_rank_gaps_by_priority(self, gap_map: GapMap):
        g1 = _gap("g1", importance=0.9, tractability=0.9, novelty=0.9, timeliness=0.1)
        g2 = _gap("g2", importance=0.3, tractability=0.3, novelty=0.3, timeliness=0.9)
        await gap_map.register_gap(g1)
        await gap_map.register_gap(g2)

        ranked = await gap_map.rank_gaps()
        assert ranked[0].id == "g1"
        assert ranked[1].id == "g2"

    @pytest.mark.asyncio
    async def test_rank_gaps_top_k(self, gap_map: GapMap):
        for i in range(5):
            await gap_map.register_gap(_gap(f"g{i}", importance=i * 0.2))
        ranked = await gap_map.rank_gaps(top_k=2)
        assert len(ranked) == 2

    @pytest.mark.asyncio
    async def test_rank_gaps_excludes_closed(self, gap_map: GapMap):
        await gap_map.register_gap(_gap("g1", importance=0.9))
        await gap_map.register_gap(_gap("g2", importance=0.5))
        await gap_map.update_gap_status("g1", GapStatus.FILLED)

        ranked = await gap_map.rank_gaps()
        assert len(ranked) == 1
        assert ranked[0].id == "g2"


# ── Paper ingestion update ───────────────────────────────────────────


class TestIngestPaperUpdate:
    @pytest.mark.asyncio
    async def test_increases_coverage(self, gap_map: GapMap):
        await gap_map.add_node(_node("n1", coverage=0.0))
        await gap_map.ingest_paper_update("paper1", related_node_ids=["n1"])
        node = await gap_map.get_node("n1")
        assert node.coverage_score > 0.0
        assert node.paper_count == 1

    @pytest.mark.asyncio
    async def test_diminishing_returns(self, gap_map: GapMap):
        await gap_map.add_node(_node("n1", coverage=0.0))
        await gap_map.ingest_paper_update("p1", ["n1"])
        c1 = (await gap_map.get_node("n1")).coverage_score
        await gap_map.ingest_paper_update("p2", ["n1"])
        c2 = (await gap_map.get_node("n1")).coverage_score
        increment1 = c1
        increment2 = c2 - c1
        assert increment2 < increment1

    @pytest.mark.asyncio
    async def test_auto_fills_gap(self, gap_map: GapMap):
        await gap_map.add_node(_node("n1", coverage=0.85))
        gap = _gap("g1", adjacent_concepts=["n1"])
        await gap_map.register_gap(gap)
        await gap_map.ingest_paper_update("p1", ["n1"])
        retrieved = await gap_map.get_gap("g1")
        assert retrieved.status == GapStatus.FILLED

    @pytest.mark.asyncio
    async def test_adds_new_edges(self, gap_map: GapMap):
        await gap_map.add_node(_node("a"))
        await gap_map.add_node(_node("b"))
        await gap_map.ingest_paper_update(
            "p1",
            related_node_ids=["a"],
            new_edges=[_edge("a", "b")],
        )
        assert gap_map.edge_count == 1
