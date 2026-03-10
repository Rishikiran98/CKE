from cke.extractor.extractor import RuleBasedExtractor
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.models import Statement
from cke.retrieval.retriever import GraphRetriever


def test_retrieval_bfs_context_depth():
    extractor = RuleBasedExtractor()
    graph = KnowledgeGraphEngine()
    text = "Redis supports PubSub. PubSub implemented_via RESP."
    graph.add_statements(extractor.extract(text))

    retriever = GraphRetriever(graph)
    context = retriever.retrieve("What protocol does Redis pub/sub use?", max_depth=2)
    triples = {(s.subject, s.relation, s.object) for s in context}
    assert ("Redis", "supports", "PubSub") in triples
    assert ("PubSub", "implemented_via", "RESP") in triples


def test_retrieval_prefers_higher_confidence_then_shorter_path():
    graph = KnowledgeGraphEngine()
    graph.add_statements(
        [
            Statement("Redis", "supports", "PubSub", confidence=0.8),
            Statement("Redis", "uses", "RESP", confidence=0.95),
            Statement("PubSub", "implemented_via", "RESP", confidence=0.95),
        ]
    )

    retriever = GraphRetriever(graph)
    context = retriever.retrieve("What does Redis use?", max_depth=2)

    assert context[0].relation == "uses"  # highest confidence, depth 1
    # Same confidence as first, but depth 2 so should come later.
    assert any(s.relation == "implemented_via" for s in context[1:])
