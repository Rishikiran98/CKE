"""Optional Neo4j graph backend compatible with KnowledgeGraphEngine API."""

from __future__ import annotations

from typing import Any

from cke.models import Statement

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None


class Neo4jBackend:
    def __init__(self, uri: str, user: str, password: str) -> None:
        if GraphDatabase is None:
            raise RuntimeError("neo4j package is not installed")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def add_assertion(
        self,
        subject: str,
        relation: str,
        object_: str,
        context: dict[str, Any] | None = None,
        confidence: float = 1.0,
        source: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        payload = {
            "relation": relation,
            "context": context or {},
            "confidence": confidence,
            "source": source,
            "timestamp": timestamp,
        }
        with self.driver.session() as session:
            session.run(
                """
                MERGE (s:Entity {name: $subject})
                MERGE (o:Entity {name: $object})
                CREATE (s)-[:RELATED {relation: $relation, context: $context,
                    confidence: $confidence, source: $source, timestamp: $timestamp}]->(o)
                """,
                subject=subject,
                object=object_,
                **payload,
            )

    def add_statement(self, *args, **kwargs) -> None:
        self.add_assertion(*args, **kwargs)

    def query_neighbors(self, entity: str) -> list[Statement]:
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (s:Entity {name: $entity})-[r:RELATED]->(o:Entity)
                RETURN s.name AS subject, o.name AS object,
                       r.relation AS relation, r.context AS context,
                       r.confidence AS confidence, r.source AS source,
                       r.timestamp AS timestamp
                """,
                entity=entity,
            )
            return [
                Statement(
                    subject=row["subject"],
                    relation=row.get("relation") or "related_to",
                    object=row["object"],
                    context=dict(row.get("context") or {}),
                    confidence=float(row.get("confidence") or 1.0),
                    source=row.get("source"),
                    timestamp=row.get("timestamp"),
                )
                for row in rows
            ]

    def get_neighbors(self, entity: str) -> list[Statement]:
        return self.query_neighbors(entity)

    def multi_hop_search(
        self, source: str, target: str, max_depth: int = 3
    ) -> list[list[Statement]]:
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH p=(s:Entity {name: $source})-[rels:RELATED*1..$depth]->(t:Entity {name: $target})
                RETURN [n IN nodes(p) | n.name] AS nodes,
                       [r IN relationships(p) | {relation: r.relation, context: r.context,
                        confidence: r.confidence, source: r.source, timestamp: r.timestamp}] AS rels
                """,
                source=source,
                target=target,
                depth=max_depth,
            )

            paths: list[list[Statement]] = []
            for row in rows:
                nodes = row["nodes"]
                rels = row["rels"]
                path: list[Statement] = []
                for idx, rel in enumerate(rels):
                    path.append(
                        Statement(
                            subject=nodes[idx],
                            relation=rel.get("relation") or "related_to",
                            object=nodes[idx + 1],
                            context=dict(rel.get("context") or {}),
                            confidence=float(rel.get("confidence") or 1.0),
                            source=rel.get("source"),
                            timestamp=rel.get("timestamp"),
                        )
                    )
                paths.append(path)
            return paths

    def find_paths(
        self, entity_a: str, entity_b: str, cutoff: int = 3
    ) -> list[list[Statement]]:
        return self.multi_hop_search(entity_a, entity_b, max_depth=cutoff)

    def all_entities(self) -> list[str]:
        with self.driver.session() as session:
            rows = session.run("MATCH (n:Entity) RETURN DISTINCT n.name AS name")
            return [row["name"] for row in rows]
