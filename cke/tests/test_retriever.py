from cke.extractor.extractor import RuleBasedExtractor
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
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
