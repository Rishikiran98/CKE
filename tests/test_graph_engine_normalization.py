"""Tests for entity normalization in KnowledgeGraphEngine."""

from cke.graph_engine.graph_engine import KnowledgeGraphEngine


def test_entity_normalization_merges_surface_forms():
    graph = KnowledgeGraphEngine()
    graph.add_statement("Scott Derrickson", "directed", "Doctor Strange")
    graph.add_statement("Scott_Derrickson", "born_in", "Denver")
    graph.add_statement("scott derrickson", "occupation", "Director")

    entities = set(graph.all_entities())
    assert "Scott Derrickson" in entities
    assert "Scott_Derrickson" not in entities
    assert "scott derrickson" not in entities

    neighbors = graph.get_neighbors("SCOTT_DERRICKSON")
    relations = {edge.relation for edge in neighbors}
    assert {"directed", "born_in", "occupation"}.issubset(relations)
