from cke.graph_engine.graph_engine import KnowledgeGraphEngine


def test_graph_insertion_and_neighbors():
    graph = KnowledgeGraphEngine()
    graph.add_statement("Redis", "supports", "PubSub")
    neighbors = graph.get_neighbors("Redis")
    assert len(neighbors) == 1
    assert neighbors[0].object == "PubSub"


def test_graph_find_paths():
    graph = KnowledgeGraphEngine()
    graph.add_statement("Redis", "supports", "PubSub")
    graph.add_statement("PubSub", "implemented_via", "RESP")
    paths = graph.find_paths("Redis", "RESP")
    assert paths
    assert len(paths[0]) == 2


def test_graph_edges_for_relation_uses_relation_index():
    graph = KnowledgeGraphEngine()
    graph.add_statement("Redis", "supports", "PubSub")
    graph.add_statement("Redis", "supports", "Streams")
    graph.add_statement("Postgres", "supports", "SQL")

    edges = graph.edges_for_relation("supports")

    assert len(edges) == 3
    assert {edge.object for edge in edges} == {"PubSub", "Streams", "SQL"}
