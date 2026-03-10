"""CKE demo script."""

from cke.extractor.extractor import RuleBasedExtractor
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.reasoning.reasoner import TemplateReasoner
from cke.retrieval.retriever import GraphRetriever


def main() -> None:
    corpus = [
        "Redis supports PubSub messaging.",
        "PubSub implemented_via RESP.",
        "Redis uses RESP protocol.",
    ]

    extractor = RuleBasedExtractor()
    graph = KnowledgeGraphEngine()
    for doc in corpus:
        graph.add_statements(extractor.extract(doc))

    query = "What protocol does Redis PubSub use?"
    retriever = GraphRetriever(graph)
    reasoner = TemplateReasoner()

    context = retriever.retrieve(query, max_depth=3)
    answer = reasoner.answer(query, context)

    print(f'Query:\n"{query}"\n')
    print("Graph reasoning:")
    print(reasoner.format_reasoning_path(context))
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
