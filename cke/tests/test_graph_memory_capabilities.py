from cke.graph.query_engine import GraphQueryEngine
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.models import Statement


def test_graph_sharding_and_stats() -> None:
    graph = KnowledgeGraphEngine(shard_count=4, shard_strategy="entity_hashing")
    graph.add_statement("Apple", "CEO of", "Tim Cook", context={"domain": "tech"})
    graph.add_statement(
        "Microsoft", "CEO of", "Satya Nadella", context={"domain": "tech"}
    )

    stats = graph.shard_stats()
    assert sum(stats.values()) == 2
    assert graph.get_shard_for_entity("Apple") in {0, 1, 2, 3}


def test_dedup_relation_normalization_and_temporal_context() -> None:
    graph = KnowledgeGraphEngine()
    graph.add_statement(
        "Apple",
        "CEO of",
        "Tim Cook",
        context={"valid_from": "2011", "topic": "leadership"},
    )
    graph.add_statement(
        "Apple",
        "ceo   of",
        "Tim Cook",
        context={"valid_from": "2011", "topic": "leadership"},
    )

    neighbors = graph.get_neighbors("apple")
    assert len(neighbors) == 1
    assert neighbors[0].relation == "ceo_of"
    assert neighbors[0].context["valid_from"] == "2011"


def test_incremental_delta_ingestion_and_node_merge() -> None:
    graph = KnowledgeGraphEngine()
    graph.ingest_delta([Statement("Meta", "CEO of", "Mark Zuckerberg")])
    graph.ingest_delta(
        [Statement("Facebook", "CEO of", "Mark Zuckerberg")], mode="append"
    )
    graph.merge_nodes("Meta", ["Facebook"])

    neighbors = graph.get_neighbors("meta")
    assert neighbors
    assert all(edge.relation == "ceo_of" for edge in neighbors)


def test_graph_query_engine_api() -> None:
    graph = KnowledgeGraphEngine()
    graph.add_statement("Redis", "supports", "PubSub")
    graph.add_statement("PubSub", "implemented via", "RESP")

    query = GraphQueryEngine(graph)
    assert query.neighbors("Redis")
    assert query.paths("Redis", "RESP", cutoff=3)
    assert "supports" in query.relations("Redis")
    subgraph = query.subgraph(["Redis"], depth=2)
    assert subgraph["entities"]
    assert subgraph["edges"]
