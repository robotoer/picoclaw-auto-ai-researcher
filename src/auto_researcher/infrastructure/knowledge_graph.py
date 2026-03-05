"""Neo4j-based knowledge graph client for storing and querying research claims."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from neo4j import AsyncGraphDatabase, AsyncDriver

from auto_researcher.config import Neo4jConfig
from auto_researcher.models.claim import Claim, ClaimRelation, ClaimStatus

logger = logging.getLogger(__name__)

# Entity types stored as node labels in Neo4j.
ENTITY_TYPES = {"model", "dataset", "metric", "author", "institution", "technique", "concept"}

# Relation types that indicate contradiction.
CONTRADICTING_RELATIONS = {ClaimRelation.REFUTES, ClaimRelation.CONTRADICTS}


class KnowledgeGraphClient:
    """Async Neo4j client for the research knowledge graph."""

    def __init__(self, config: Neo4jConfig) -> None:
        self._config = config
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            self._config.uri,
            auth=(self._config.username, self._config.password),
        )
        await self._ensure_indexes()

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("KnowledgeGraphClient is not connected. Call connect() first.")
        return self._driver

    # ------------------------------------------------------------------
    # Schema / indexes
    # ------------------------------------------------------------------

    async def _ensure_indexes(self) -> None:
        async with self.driver.session(database=self._config.database) as session:
            await session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            await session.run("CREATE INDEX claim_id IF NOT EXISTS FOR ()-[r:CLAIM]-() ON (r.claim_id)")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def store_claim(self, claim: Claim) -> Claim:
        """Store a claim as a relationship between two entity nodes.

        Returns the claim with its id populated.
        """
        if not claim.id:
            claim.id = str(uuid.uuid4())

        contradictions = await self._detect_contradictions(claim)
        if contradictions:
            claim.contradicting_claim_ids.extend([c.id for c in contradictions])
            claim.status = ClaimStatus.CONTRADICTED
            for existing in contradictions:
                await self._mark_contradicted(existing.id, claim.id)

        query = """
        MERGE (e1:Entity {name: $entity_1})
        MERGE (e2:Entity {name: $entity_2})
        CREATE (e1)-[r:CLAIM {
            claim_id: $claim_id,
            relation: $relation,
            conditions: $conditions,
            confidence: $confidence,
            status: $status,
            source_paper_ids: $source_paper_ids,
            extracted_at: $extracted_at,
            half_life_days: $half_life_days,
            contradicting_claim_ids: $contradicting_claim_ids,
            supporting_claim_ids: $supporting_claim_ids
        }]->(e2)
        RETURN r
        """
        params = {
            "entity_1": claim.entity_1,
            "entity_2": claim.entity_2,
            "claim_id": claim.id,
            "relation": claim.relation.value,
            "conditions": claim.conditions,
            "confidence": claim.confidence,
            "status": claim.status.value,
            "source_paper_ids": claim.source_paper_ids,
            "extracted_at": claim.extracted_at.isoformat(),
            "half_life_days": claim.half_life_days,
            "contradicting_claim_ids": claim.contradicting_claim_ids,
            "supporting_claim_ids": claim.supporting_claim_ids,
        }

        async with self.driver.session(database=self._config.database) as session:
            await session.run(query, params)

        logger.info("Stored claim %s: (%s)-[%s]->(%s)", claim.id, claim.entity_1, claim.relation.value, claim.entity_2)
        return claim

    async def store_claims(self, claims: list[Claim]) -> list[Claim]:
        """Batch store multiple claims."""
        results = []
        for claim in claims:
            results.append(await self.store_claim(claim))
        return results

    async def _mark_contradicted(self, existing_claim_id: str, new_claim_id: str) -> None:
        query = """
        MATCH ()-[r:CLAIM {claim_id: $claim_id}]->()
        SET r.status = $status,
            r.contradicting_claim_ids = r.contradicting_claim_ids + $new_id
        """
        async with self.driver.session(database=self._config.database) as session:
            await session.run(query, {
                "claim_id": existing_claim_id,
                "status": ClaimStatus.CONTRADICTED.value,
                "new_id": new_claim_id,
            })

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------

    async def _detect_contradictions(self, claim: Claim) -> list[Claim]:
        """Find existing claims that contradict the new claim.

        A contradiction occurs when:
        - Same entity pair with a contradicting relation (e.g. A outperforms B vs B outperforms A)
        - Same entity pair with explicit refutes/contradicts relation
        """
        contradictions: list[Claim] = []

        # Check for reverse outperforms: if new says A outperforms B, look for B outperforms A
        if claim.relation == ClaimRelation.OUTPERFORMS:
            reverse = await self.get_claims_between(claim.entity_2, claim.entity_1, ClaimRelation.OUTPERFORMS)
            contradictions.extend(reverse)

        # Check for explicit contradictions
        existing_forward = await self.get_claims_between(claim.entity_1, claim.entity_2)
        existing_reverse = await self.get_claims_between(claim.entity_2, claim.entity_1)
        for existing in existing_forward + existing_reverse:
            if existing.relation in CONTRADICTING_RELATIONS:
                contradictions.append(existing)

        return contradictions

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_claim(self, claim_id: str) -> Claim | None:
        query = """
        MATCH (e1)-[r:CLAIM {claim_id: $claim_id}]->(e2)
        RETURN e1.name AS entity_1, e2.name AS entity_2, r
        """
        async with self.driver.session(database=self._config.database) as session:
            result = await session.run(query, {"claim_id": claim_id})
            record = await result.single()
            if record is None:
                return None
            return self._record_to_claim(record)

    async def get_claims_between(
        self,
        entity_1: str,
        entity_2: str,
        relation: ClaimRelation | None = None,
    ) -> list[Claim]:
        if relation:
            query = """
            MATCH (e1:Entity {name: $e1})-[r:CLAIM {relation: $rel}]->(e2:Entity {name: $e2})
            RETURN e1.name AS entity_1, e2.name AS entity_2, r
            """
            params = {"e1": entity_1, "e2": entity_2, "rel": relation.value}
        else:
            query = """
            MATCH (e1:Entity {name: $e1})-[r:CLAIM]->(e2:Entity {name: $e2})
            RETURN e1.name AS entity_1, e2.name AS entity_2, r
            """
            params = {"e1": entity_1, "e2": entity_2}

        async with self.driver.session(database=self._config.database) as session:
            result = await session.run(query, params)
            records = [rec async for rec in result]
            return [self._record_to_claim(r) for r in records]

    async def get_claims_for_entity(self, entity: str) -> list[Claim]:
        query = """
        MATCH (e1)-[r:CLAIM]->(e2)
        WHERE e1.name = $entity OR e2.name = $entity
        RETURN e1.name AS entity_1, e2.name AS entity_2, r
        """
        async with self.driver.session(database=self._config.database) as session:
            result = await session.run(query, {"entity": entity})
            records = [rec async for rec in result]
            return [self._record_to_claim(r) for r in records]

    async def get_entity_neighbors(self, entity: str, max_depth: int = 1) -> list[str]:
        """Get entities connected to the given entity within max_depth hops."""
        query = """
        MATCH (start:Entity {name: $entity})-[:CLAIM*1..$depth]-(neighbor:Entity)
        WHERE neighbor.name <> $entity
        RETURN DISTINCT neighbor.name AS name
        """
        async with self.driver.session(database=self._config.database) as session:
            result = await session.run(query, {"entity": entity, "depth": max_depth})
            records = [rec async for rec in result]
            return [r["name"] for r in records]

    async def get_claims_with_decayed_confidence(
        self,
        min_confidence: float = 0.1,
        now: datetime | None = None,
    ) -> list[Claim]:
        """Get all claims, filtering by decayed confidence."""
        if now is None:
            now = datetime.utcnow()
        query = """
        MATCH (e1)-[r:CLAIM]->(e2)
        RETURN e1.name AS entity_1, e2.name AS entity_2, r
        """
        async with self.driver.session(database=self._config.database) as session:
            result = await session.run(query)
            records = [rec async for rec in result]

        claims = [self._record_to_claim(r) for r in records]
        return [c for c in claims if c.decayed_confidence(now) >= min_confidence]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _record_to_claim(record) -> Claim:
        r = record["r"]
        props = dict(r)
        return Claim(
            id=props["claim_id"],
            entity_1=record["entity_1"],
            entity_2=record["entity_2"],
            relation=ClaimRelation(props["relation"]),
            conditions=props.get("conditions", ""),
            confidence=props["confidence"],
            status=ClaimStatus(props.get("status", "extracted")),
            source_paper_ids=list(props.get("source_paper_ids", [])),
            extracted_at=datetime.fromisoformat(props["extracted_at"]),
            half_life_days=props.get("half_life_days", 365),
            contradicting_claim_ids=list(props.get("contradicting_claim_ids", [])),
            supporting_claim_ids=list(props.get("supporting_claim_ids", [])),
        )
