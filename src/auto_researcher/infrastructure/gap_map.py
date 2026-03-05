"""Gap Map: the core data structure tracking research coverage, frontiers, and gaps."""

from __future__ import annotations

import logging
from datetime import datetime

import networkx as nx

from auto_researcher.config import GapMapConfig
from auto_researcher.models.gap import Gap, GapEdge, GapNode, GapStatus

logger = logging.getLogger(__name__)

# Edge types classified as negative (missing / failed connections).
NEGATIVE_EDGE_TYPES = {"should_connect_but_doesnt", "tried_and_failed", "theoretically_related"}


class GapMap:
    """Maintains a graph of GapNodes and GapEdges tracking research coverage and gaps."""

    def __init__(self, config: GapMapConfig) -> None:
        self._config = config
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: dict[str, GapNode] = {}
        self._gaps: dict[str, Gap] = {}

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    async def add_node(self, node: GapNode) -> None:
        self._nodes[node.id] = node
        self._graph.add_node(
            node.id,
            node_type=node.node_type,
            label=node.label,
            coverage_score=node.coverage_score,
            paper_count=node.paper_count,
        )

    async def get_node(self, node_id: str) -> GapNode | None:
        return self._nodes.get(node_id)

    async def update_node_coverage(self, node_id: str, coverage_score: float, paper_count_delta: int = 0) -> None:
        node = self._nodes.get(node_id)
        if node is None:
            return
        node.coverage_score = max(0.0, min(1.0, coverage_score))
        node.paper_count += paper_count_delta
        node.last_updated = datetime.utcnow()
        self._graph.nodes[node_id]["coverage_score"] = node.coverage_score
        self._graph.nodes[node_id]["paper_count"] = node.paper_count

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    async def add_edge(self, edge: GapEdge) -> None:
        is_negative = edge.is_negative or edge.edge_type in NEGATIVE_EDGE_TYPES
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type,
            weight=edge.weight,
            is_negative=is_negative,
            conditions=edge.conditions,
            source_paper_ids=edge.source_paper_ids,
        )
        # Update adjacency lists on nodes.
        src = self._nodes.get(edge.source_id)
        tgt = self._nodes.get(edge.target_id)
        if src and edge.target_id not in src.adjacent_node_ids:
            src.adjacent_node_ids.append(edge.target_id)
        if tgt and edge.source_id not in tgt.adjacent_node_ids:
            tgt.adjacent_node_ids.append(edge.source_id)

    async def get_edges(self, node_id: str) -> list[GapEdge]:
        edges: list[GapEdge] = []
        for u, v, data in self._graph.edges(node_id, data=True):
            edges.append(GapEdge(
                source_id=u,
                target_id=v,
                edge_type=data["edge_type"],
                weight=data.get("weight", 1.0),
                is_negative=data.get("is_negative", False),
                conditions=data.get("conditions", ""),
                source_paper_ids=data.get("source_paper_ids", []),
            ))
        # Also include incoming edges.
        for u, v, data in self._graph.in_edges(node_id, data=True):
            edges.append(GapEdge(
                source_id=u,
                target_id=v,
                edge_type=data["edge_type"],
                weight=data.get("weight", 1.0),
                is_negative=data.get("is_negative", False),
                conditions=data.get("conditions", ""),
                source_paper_ids=data.get("source_paper_ids", []),
            ))
        return edges

    async def get_negative_edges(self) -> list[GapEdge]:
        """Return all negative edges (missing / failed connections)."""
        result: list[GapEdge] = []
        for u, v, data in self._graph.edges(data=True):
            if data.get("is_negative", False):
                result.append(GapEdge(
                    source_id=u,
                    target_id=v,
                    edge_type=data["edge_type"],
                    weight=data.get("weight", 1.0),
                    is_negative=True,
                    conditions=data.get("conditions", ""),
                    source_paper_ids=data.get("source_paper_ids", []),
                ))
        return result

    # ------------------------------------------------------------------
    # Gap tracking
    # ------------------------------------------------------------------

    async def register_gap(self, gap: Gap, node_ids: list[str] | None = None) -> None:
        """Register a gap, optionally associating it with specific nodes."""
        self._gaps[gap.id] = gap
        for nid in (node_ids or []):
            node = self._nodes.get(nid)
            if node:
                node.gaps.append(gap)

    async def get_gap(self, gap_id: str) -> Gap | None:
        return self._gaps.get(gap_id)

    async def update_gap_status(self, gap_id: str, status: GapStatus) -> None:
        gap = self._gaps.get(gap_id)
        if gap:
            gap.status = status
            gap.last_updated = datetime.utcnow()

    async def get_open_gaps(self) -> list[Gap]:
        return [g for g in self._gaps.values() if g.status == GapStatus.OPEN]

    # ------------------------------------------------------------------
    # Coverage density
    # ------------------------------------------------------------------

    async def coverage_density(self, node_id: str, radius: int = 1) -> float:
        """Compute average coverage score for a node and its neighbors within radius."""
        if node_id not in self._graph:
            return 0.0

        # Collect nodes within radius via BFS.
        visited: set[str] = set()
        frontier = {node_id}
        for _ in range(radius + 1):
            visited |= frontier
            next_frontier: set[str] = set()
            for nid in frontier:
                next_frontier |= set(self._graph.successors(nid))
                next_frontier |= set(self._graph.predecessors(nid))
            frontier = next_frontier - visited

        if not visited:
            return 0.0

        total = sum(
            self._graph.nodes[nid].get("coverage_score", 0.0)
            for nid in visited
            if nid in self._graph
        )
        return total / len(visited)

    async def region_density(self, node_ids: list[str]) -> float:
        """Average coverage score for a set of nodes."""
        if not node_ids:
            return 0.0
        scores = [
            self._graph.nodes[nid].get("coverage_score", 0.0)
            for nid in node_ids
            if nid in self._graph
        ]
        return sum(scores) / len(scores) if scores else 0.0

    # ------------------------------------------------------------------
    # Frontier calculation
    # ------------------------------------------------------------------

    async def compute_frontier(self) -> list[GapNode]:
        """Find frontier nodes: the boundary between well-covered and sparse regions.

        A frontier node is well-covered itself but has at least one neighbor
        below the frontier threshold.
        """
        threshold = self._config.frontier_threshold
        frontier: list[GapNode] = []

        for nid, node in self._nodes.items():
            if node.coverage_score < threshold:
                continue
            neighbors = list(self._graph.successors(nid)) + list(self._graph.predecessors(nid))
            for neighbor_id in neighbors:
                neighbor = self._nodes.get(neighbor_id)
                if neighbor and neighbor.coverage_score < threshold:
                    frontier.append(node)
                    break

        return frontier

    # ------------------------------------------------------------------
    # Gap ranking
    # ------------------------------------------------------------------

    async def rank_gaps(self, top_k: int | None = None) -> list[Gap]:
        """Rank open gaps by priority_score = (importance * tractability * novelty) / (1 + urgency).

        Returns gaps sorted by priority descending.
        """
        open_gaps = await self.get_open_gaps()
        ranked = sorted(open_gaps, key=lambda g: g.priority_score(), reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    # ------------------------------------------------------------------
    # Update from new papers
    # ------------------------------------------------------------------

    async def ingest_paper_update(
        self,
        paper_id: str,
        related_node_ids: list[str],
        new_edges: list[GapEdge] | None = None,
    ) -> None:
        """Update the gap map when a new paper is ingested.

        Increases coverage for related nodes and adds any new edges.
        """
        for nid in related_node_ids:
            node = self._nodes.get(nid)
            if node is None:
                continue
            # Increment coverage with diminishing returns.
            increment = 0.1 / (1.0 + node.paper_count * 0.1)
            new_coverage = min(1.0, node.coverage_score + increment)
            await self.update_node_coverage(nid, new_coverage, paper_count_delta=1)

        for edge in (new_edges or []):
            await self.add_edge(edge)

        # Check if any gaps are now filled.
        for gap in await self.get_open_gaps():
            related_coverages = [
                self._nodes[nid].coverage_score
                for nid in gap.adjacent_concepts
                if nid in self._nodes
            ]
            if related_coverages and all(c > 0.8 for c in related_coverages):
                await self.update_gap_status(gap.id, GapStatus.FILLED)
                logger.info("Gap %s auto-filled after paper %s", gap.id, paper_id)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def all_nodes(self) -> list[GapNode]:
        return list(self._nodes.values())

    def all_gaps(self) -> list[Gap]:
        return list(self._gaps.values())

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()
