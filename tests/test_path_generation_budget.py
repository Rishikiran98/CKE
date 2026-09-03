"""Retrieval budgets must actually vary what is retrieved.

The published ablation reported an identical F1 at N=8, N=12 and N=20. That was
not a finding about retrieval budgets; it was two defects in path generation.

  1. Traversal could only follow an edge forward. An edge (A, rel, B) was
     reachable from A and never from B, so BFS dead-ended at any node with no
     outgoing edges and the evidence set stayed far below any budget.
  2. max_results was clamped to 12 in two places, so every budget above 12
     produced the same result whatever the graph contained.
"""

from __future__ import annotations

import pytest

from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.retrieval.graph_retriever import GraphRetriever
from cke.router.query_plan import QueryPlan


def _chain_graph() -> KnowledgeGraphEngine:
    """A connected graph with consistent entity names."""
    engine = KnowledgeGraphEngine()
    for subject, relation, obj in [
        ("Redis", "uses", "RESP"),
        ("RESP", "uses", "TCP"),
        ("TCP", "uses", "IP"),
        ("IP", "uses", "Ethernet"),
        ("Salvatore", "developed", "Redis"),
        ("PubSub", "uses", "RESP"),
        ("Cerf", "developed", "TCP"),
    ]:
        engine.add_statement(subject, relation, obj)
    return engine


def _evidence(engine, *, depth: int, budget: int, seed: str = "Redis") -> list:
    plan = QueryPlan(
        query_text="What does Redis use?",
        seed_entities=[seed],
        intent="factoid",
        max_depth=depth,
        max_results=budget,
    )
    return GraphRetriever(engine).retrieve(plan, mode="bfs").get("evidence", [])


# ---------------------------------------------------------------------------
# The engine must expose an edge from both of its endpoints
# ---------------------------------------------------------------------------


def test_incoming_edges_are_reachable_and_not_reversed():
    """An edge pointing at an entity is reachable from it, direction intact."""
    engine = KnowledgeGraphEngine()
    engine.add_statement("Salvatore", "developed", "Redis")

    assert engine.get_neighbors("Redis") == []

    incoming = engine.get_incoming("Redis")
    assert [(s.subject, s.relation, s.object) for s in incoming] == [
        ("Salvatore", "developed", "Redis")
    ]


def test_incident_returns_both_directions_without_duplicates():
    engine = KnowledgeGraphEngine()
    engine.add_statement("Redis", "uses", "RESP")
    engine.add_statement("Salvatore", "developed", "Redis")

    incident = {(s.subject, s.relation, s.object) for s in engine.get_incident("Redis")}

    assert incident == {
        ("Redis", "uses", "RESP"),
        ("Salvatore", "developed", "Redis"),
    }


def test_a_self_loop_is_reported_once():
    engine = KnowledgeGraphEngine()
    engine.add_statement("Redis", "relates_to", "Redis")

    assert len(engine.get_incident("Redis")) == 1


# ---------------------------------------------------------------------------
# The budget must bind
# ---------------------------------------------------------------------------


def test_the_retrieval_budget_varies_the_evidence():
    """N=8, N=12 and N=20 returned identical evidence in the published run."""
    engine = _chain_graph()

    counts = [len(_evidence(engine, depth=3, budget=n)) for n in (2, 4, 8)]

    assert counts == sorted(counts), "a larger budget must not return less"
    assert len(set(counts)) > 1, "the budget changed nothing"


def test_a_budget_above_twelve_is_not_clamped():
    """max_results was clamped to 12, so 13 and 40 were the same request."""
    engine = KnowledgeGraphEngine()
    for i in range(20):
        engine.add_statement("Hub", "links", f"Node{i}")

    assert len(_evidence(engine, depth=1, budget=12, seed="Hub")) == 12
    assert len(_evidence(engine, depth=1, budget=20, seed="Hub")) == 20


def test_depth_varies_the_evidence():
    """Traversal could not go deep, because it dead-ended on the first hop."""
    engine = _chain_graph()

    shallow = len(_evidence(engine, depth=1, budget=20))
    deep = len(_evidence(engine, depth=3, budget=20))

    assert deep > shallow


def test_the_budget_saturates_at_the_graph_size():
    """A budget larger than the graph returns the graph, not an error."""
    engine = _chain_graph()

    assert len(_evidence(engine, depth=4, budget=100)) == 7


# ---------------------------------------------------------------------------
# The traversal itself
# ---------------------------------------------------------------------------


def test_traversal_reaches_a_neighbour_only_linked_by_an_incoming_edge():
    """The core defect: Cerf is reachable from Redis only against an arrow.

    Redis <- Salvatore is incoming, and Redis -> RESP -> TCP <- Cerf needs two
    forward hops and one backward. Forward-only traversal could never see it.
    """
    engine = _chain_graph()

    evidence = _evidence(engine, depth=3, budget=100)
    subjects = {item["subject"] for item in evidence}

    assert "Salvatore" in subjects, "an incoming edge was never traversed"


def test_a_seed_with_only_incoming_edges_still_retrieves():
    """Such a seed previously returned nothing at all."""
    engine = KnowledgeGraphEngine()
    engine.add_statement("Salvatore", "developed", "Redis")
    engine.add_statement("Antirez", "maintains", "Redis")

    plan = QueryPlan(
        query_text="Who developed Redis?",
        seed_entities=["Redis"],
        intent="factoid",
        max_depth=2,
        max_results=10,
    )
    evidence = GraphRetriever(engine).retrieve(plan, mode="bfs").get("evidence", [])

    assert len(evidence) == 2


@pytest.mark.parametrize("mode", ["bfs", "beam", "astar"])
def test_every_traversal_mode_follows_incoming_edges(mode):
    engine = _chain_graph()
    plan = QueryPlan(
        query_text="Who developed Redis?",
        seed_entities=["Redis"],
        intent="factoid",
        max_depth=2,
        max_results=50,
    )

    evidence = GraphRetriever(engine).retrieve(plan, mode=mode).get("evidence", [])
    subjects = {item["subject"] for item in evidence}

    assert "Salvatore" in subjects, f"{mode} never traversed an incoming edge"
