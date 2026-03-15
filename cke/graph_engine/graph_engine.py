"""Knowledge graph storage.

Uses NetworkX when available and a lightweight internal fallback otherwise.
An optional *storage* backend (e.g. SQLiteStore) can be supplied to persist
entities and statements across process restarts.  When no backend is provided
the engine behaves exactly as before (pure in-memory).
"""

from __future__ import annotations

from collections import defaultdict, deque
from hashlib import sha256
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from cke.models import Statement

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None


class KnowledgeGraphEngine:
    """Manages entity-relation graph operations.

    Parameters
    ----------
    db_path:
        Optional path to a SQLite database file.  When supplied, every
        ``add_statement`` call is also persisted to disk and the in-memory
        graph is pre-loaded from the store on initialisation.  Omit (or
        pass ``None``) to use the original pure in-memory behaviour.
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        backend: Any | None = None,
        shard_count: int = 8,
        shard_strategy: str = "entity_hashing",
    ) -> None:
        self._backend = backend
        self._shard_count = max(1, int(shard_count))
        self._shard_strategy = shard_strategy
        self._shards: dict[int, list[tuple[str, str, dict[str, Any]]]] = defaultdict(
            list
        )
        self._edge_keys: set[tuple[str, str, str, str | None, str | None]] = set()
        if self._backend is not None:
            self.graph = None
            self._storage = None
            self._use_nx = False
            return

        self._storage = self._init_storage(db_path)
        self._entity_display: dict[str, str] = {}

        # --- in-memory graph (always maintained for fast traversal) ---
        self._use_nx = nx is not None
        if self._use_nx:
            self.graph = nx.MultiDiGraph()
            self._adjacency_index: dict[str, list[tuple[str, dict[str, Any]]]] = (
                defaultdict(list)
            )
            self._relation_index: dict[str, list[tuple[str, str, dict[str, Any]]]] = (
                defaultdict(list)
            )
        else:
            self.graph: Dict[str, list[dict[str, Any]]] = defaultdict(list)
            self._nodes: set[str] = set()
            self._relation_index: dict[str, list[tuple[str, dict[str, Any]]]] = (
                defaultdict(list)
            )

        # If a storage backend was provided, warm the in-memory graph.
        if self._storage is not None:
            for st in self._storage.load_all_statements():
                self._add_to_memory(
                    st.subject,
                    st.relation,
                    st.object,
                    context=st.context,
                    confidence=st.confidence,
                    source=st.source,
                    timestamp=st.timestamp,
                    object_display=st.object,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_storage(db_path: Optional[str | Path]):
        """Return a ready storage backend or *None* for in-memory mode."""
        if db_path is None:
            return None
        # Lazy import so the storage module is only required when used.
        from cke.storage.sqlite_store import SQLiteStore  # noqa: PLC0415

        store = SQLiteStore(db_path)
        store.init_schema()
        return store

    def _add_to_memory(
        self,
        subject: str,
        relation: str,
        object_: str,
        context: dict[str, Any] | None = None,
        confidence: float = 1.0,
        source: str | None = None,
        timestamp: str | None = None,
        object_display: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
    ) -> None:
        """Write a statement into the in-memory graph only."""
        self._entity_display.setdefault(subject, subject)
        self._entity_display.setdefault(object_, object_display or object_)

        payload = {
            "relation": relation,
            "context": context or {},
            "confidence": confidence,
            "source": source,
            "timestamp": timestamp,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "object_display": object_display
            or self._entity_display.get(object_, object_),
        }
        dedupe_key = (subject, relation, object_, valid_from, valid_to)
        if dedupe_key in self._edge_keys:
            return
        self._edge_keys.add(dedupe_key)
        if self._use_nx:
            self.graph.add_node(subject)
            self.graph.add_node(object_)
            self.graph.add_edge(subject, object_, **payload)
            self._adjacency_index[subject].append((object_, payload))
            self._relation_index[relation].append((subject, object_, payload))
        else:
            self._nodes.update([subject, object_])
            item = {"object": object_, **payload}
            self.graph[subject].append(item)
            self._relation_index[relation].append((subject, item))

        shard_id = self._compute_shard(subject, payload)
        self._shards[shard_id].append((subject, object_, payload))

    @staticmethod
    def _normalize_entity(entity: str) -> str:
        """Normalize entity surface form to reduce duplicate nodes."""
        cleaned = re.sub(r"[^\w\s]", " ", str(entity).lower().replace("_", " "))
        return " ".join(cleaned.split())

    def _display_entity(self, entity: str) -> str:
        return self._entity_display.get(entity, entity)

    def _normalize_relation(self, relation: str) -> str:
        return "_".join(str(relation).strip().lower().split())

    def _compute_shard(self, entity: str, payload: dict[str, Any] | None = None) -> int:
        if self._shard_strategy == "domain_partitions":
            domain = str((payload or {}).get("context", {}).get("domain", "default"))
            return (
                int(sha256(domain.encode("utf-8")).hexdigest(), 16) % self._shard_count
            )
        if self._shard_strategy == "topic_clusters":
            topic = str((payload or {}).get("context", {}).get("topic", "default"))
            relation = str((payload or {}).get("relation", ""))
            key = f"{topic}:{relation}"
            return int(sha256(key.encode("utf-8")).hexdigest(), 16) % self._shard_count
        return int(sha256(entity.encode("utf-8")).hexdigest(), 16) % self._shard_count

    # ------------------------------------------------------------------
    # Public API (unchanged surface)
    # ------------------------------------------------------------------

    def add_statement(
        self,
        subject: str,
        relation: str,
        object_: str,
        context: dict[str, Any] | None = None,
        confidence: float = 1.0,
        source: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        subject_display = str(subject)
        object_display = str(object_)
        subject = self._normalize_entity(subject_display)
        object_ = self._normalize_entity(object_display)
        self._entity_display.setdefault(subject, subject_display)
        self._entity_display.setdefault(object_, object_display)

        relation = self._normalize_relation(relation)
        valid_from = context.get("valid_from") if context else None
        valid_to = context.get("valid_to") if context else None

        if self._backend is not None:
            self._backend.add_assertion(
                subject,
                relation,
                object_,
                context=context,
                confidence=confidence,
                source=source,
                timestamp=timestamp,
            )
            return

        self._add_to_memory(
            subject,
            relation,
            object_,
            context=context,
            confidence=confidence,
            source=source,
            timestamp=timestamp,
            valid_from=valid_from,
            valid_to=valid_to,
        )
        if self._storage is not None:
            self._storage.upsert_statement(
                subject,
                relation,
                object_,
                context=context,
                confidence=confidence,
                source=source,
                timestamp=timestamp,
            )

    def ingest_delta(self, statements: List[Statement], mode: str = "upsert") -> None:
        """Apply incremental graph updates without requiring full rebuild."""
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"upsert", "append"}:
            raise ValueError("mode must be 'upsert' or 'append'")

        for st in statements:
            if normalized_mode == "append":
                self._edge_keys.discard(
                    (
                        self._normalize_entity(st.subject),
                        self._normalize_relation(st.relation),
                        self._normalize_entity(st.object),
                        (st.context or {}).get("valid_from"),
                        (st.context or {}).get("valid_to"),
                    )
                )
            self.add_statement(
                st.subject,
                st.relation,
                st.object,
                context=st.context,
                confidence=st.confidence,
                source=st.source,
                timestamp=st.timestamp,
            )

    def add_assertion(
        self,
        subject: str,
        relation: str,
        object_: str,
        evidence: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        confidence: float = 1.0,
        source: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        """Add an assertion with optional span-level evidence metadata."""
        merged_context = dict(context or {})
        if evidence is not None:
            merged_context["evidence"] = evidence
        self.add_statement(
            subject,
            relation,
            object_,
            context=merged_context,
            confidence=confidence,
            source=source,
            timestamp=timestamp,
        )

    def add_statements(self, statements: List[Statement]) -> None:
        for st in statements:
            self.add_statement(
                st.subject,
                st.relation,
                st.object,
                context=st.context,
                confidence=st.confidence,
                source=st.source,
                timestamp=st.timestamp,
            )

    def get_neighbors(self, entity: str) -> List[Statement]:
        entity = self._normalize_entity(entity)
        if self._backend is not None:
            return self._backend.query_neighbors(entity)

        if self._use_nx:
            neighbors = self._adjacency_index.get(entity, [])
            if not neighbors:
                return []
            return [
                Statement(
                    subject=self._display_entity(entity),
                    relation=edge_data.get("relation", "related_to"),
                    object=edge_data.get(
                        "object_display", self._display_entity(target)
                    ),
                    confidence=float(edge_data.get("confidence", 1.0)),
                    source=edge_data.get("source"),
                    timestamp=edge_data.get("timestamp"),
                    context={
                        **dict(edge_data.get("context", {})),
                        "valid_from": edge_data.get("valid_from"),
                        "valid_to": edge_data.get("valid_to"),
                    },
                )
                for target, edge_data in neighbors
            ]

        return [
            Statement(
                subject=self._display_entity(entity),
                relation=item.get("relation", "related_to"),
                object=item.get(
                    "object_display", self._display_entity(item.get("object", ""))
                ),
                confidence=float(item.get("confidence", 1.0)),
                source=item.get("source"),
                timestamp=item.get("timestamp"),
                context={
                    **dict(item.get("context", {})),
                    "valid_from": item.get("valid_from"),
                    "valid_to": item.get("valid_to"),
                },
            )
            for item in self.graph.get(entity, [])
        ]

    def find_paths(
        self, entity_a: str, entity_b: str, cutoff: int = 3
    ) -> List[List[Statement]]:
        entity_a = self._normalize_entity(entity_a)
        entity_b = self._normalize_entity(entity_b)
        if self._backend is not None:
            return self._backend.multi_hop_search(entity_a, entity_b, max_depth=cutoff)

        if self._use_nx:
            if entity_a not in self.graph or entity_b not in self.graph:
                return []
            paths: list[list[Statement]] = []
            for node_path in nx.all_simple_paths(
                self.graph, source=entity_a, target=entity_b, cutoff=cutoff
            ):
                statement_path: list[Statement] = []
                for i in range(len(node_path) - 1):
                    src, dst = node_path[i], node_path[i + 1]
                    edge_map = self.graph.get_edge_data(src, dst) or {}
                    edge = (
                        edge_map[min(edge_map.keys())]
                        if edge_map
                        else {"relation": "related_to"}
                    )
                    statement_path.append(
                        Statement(
                            subject=self._display_entity(src),
                            relation=edge.get("relation", "related_to"),
                            object=self._display_entity(dst),
                            context=dict(edge.get("context", {})),
                            confidence=float(edge.get("confidence", 1.0)),
                            source=edge.get("source"),
                            timestamp=edge.get("timestamp"),
                        )
                    )
                paths.append(statement_path)
            return paths

        if entity_a not in self._nodes or entity_b not in self._nodes:
            return []

        results: list[list[Statement]] = []
        queue = deque([(entity_a, [])])
        while queue:
            node, path = queue.popleft()
            if len(path) > cutoff:
                continue
            if node == entity_b and path:
                results.append(path)
                continue
            for item in self.graph.get(node, []):
                nxt = item.get("object", "")
                if any(self._normalize_entity(step.object) == nxt for step in path):
                    continue
                queue.append(
                    (
                        nxt,
                        path
                        + [
                            Statement(
                                subject=self._display_entity(node),
                                relation=item.get("relation", "related_to"),
                                object=self._display_entity(nxt),
                                context=dict(item.get("context", {})),
                                confidence=float(item.get("confidence", 1.0)),
                                source=item.get("source"),
                                timestamp=item.get("timestamp"),
                            )
                        ],
                    )
                )
        return results

    def edges_for_relation(self, relation: str) -> list[Statement]:
        """Return all statements matching a relation label.

        Uses a relation index in memory to avoid graph-wide edge scans.
        """
        relation = str(relation)
        if self._backend is not None:
            return [
                st
                for entity in self._backend.all_entities()
                for st in self._backend.query_neighbors(entity)
                if st.relation == relation
            ]

        if self._use_nx:
            return [
                Statement(
                    subject=self._display_entity(subject),
                    relation=relation,
                    object=edge_data.get(
                        "object_display", self._display_entity(target)
                    ),
                    confidence=float(edge_data.get("confidence", 1.0)),
                    source=edge_data.get("source"),
                    timestamp=edge_data.get("timestamp"),
                )
                for subject, target, edge_data in self._relation_index.get(relation, [])
            ]

        return [
            Statement(
                subject=self._display_entity(subject),
                relation=relation,
                object=item.get(
                    "object_display", self._display_entity(item.get("object", ""))
                ),
                confidence=float(item.get("confidence", 1.0)),
                source=item.get("source"),
                timestamp=item.get("timestamp"),
            )
            for subject, item in self._relation_index.get(relation, [])
        ]

    def all_entities(self) -> list[str]:
        if self._backend is not None:
            return self._backend.all_entities()
        if self._use_nx:
            return [self._display_entity(node) for node in self.graph.nodes]
        return sorted(self._display_entity(node) for node in self._nodes)

    def merge_nodes(self, canonical: str, aliases: list[str]) -> None:
        """Merge aliases into a canonical entity by replaying outgoing edges."""
        canonical_norm = self._normalize_entity(canonical)
        for alias in aliases:
            alias_norm = self._normalize_entity(alias)
            if alias_norm == canonical_norm:
                continue
            for edge in self.get_neighbors(alias_norm):
                self.add_statement(
                    canonical,
                    edge.relation,
                    edge.object,
                    context=edge.context,
                    confidence=edge.confidence,
                    source=edge.source,
                    timestamp=edge.timestamp,
                )

    def shard_stats(self) -> dict[int, int]:
        return {shard: len(edges) for shard, edges in self._shards.items()}

    def get_shard_for_entity(self, entity: str) -> int:
        return self._compute_shard(self._normalize_entity(entity))


def GraphEngine(
    type: str = "memory", **kwargs: Any
) -> KnowledgeGraphEngine:  # noqa: A002
    """Factory for memory or neo4j graph engines."""
    if type == "memory":
        return KnowledgeGraphEngine(db_path=kwargs.get("db_path"))
    if type == "neo4j":
        from cke.graph.neo4j_backend import Neo4jBackend  # noqa: PLC0415

        backend = Neo4jBackend(
            uri=kwargs.get("uri")
            or os.getenv("CKE_NEO4J_URI", "bolt://localhost:7687"),
            user=kwargs.get("user") or os.getenv("CKE_NEO4J_USER", "neo4j"),
            password=kwargs.get("password") or os.getenv("CKE_NEO4J_PASSWORD", "neo4j"),
        )
        return KnowledgeGraphEngine(backend=backend)
    raise ValueError(f"Unknown graph engine type: {type}")
