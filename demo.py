"""CKE demo script."""

from __future__ import annotations

import argparse

from cke.extractor.extractor import BaseExtractor, RuleBasedExtractor
from cke.extractor.llm_extractor import LLMExtractor
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.reasoning.llm_reasoner import LLMReasoner
from cke.reasoning.reasoner import TemplateReasoner
from cke.retrieval.retriever import GraphRetriever


def _build_extractor(name: str) -> BaseExtractor:
    if name == "llm":
        return LLMExtractor()
    return RuleBasedExtractor()


def _build_reasoner(name: str):
    if name == "llm":
        return LLMReasoner()
    return TemplateReasoner()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", choices=["rule", "llm"], default="rule")
    parser.add_argument(
        "--reasoner", choices=["template", "llm"], default="template"
    )
    args = parser.parse_args()

    corpus = [
        "Redis supports PubSub messaging.",
        "PubSub implemented_via RESP.",
        "Redis uses RESP protocol.",
    ]

    extractor = _build_extractor(args.extractor)
    graph = KnowledgeGraphEngine()
    for doc in corpus:
        graph.add_statements(extractor.extract(doc))

    query = "What protocol does Redis PubSub use?"
    retriever = GraphRetriever(graph)
    reasoner = _build_reasoner(args.reasoner)

    context = retriever.retrieve(query, max_depth=3)
    answer = reasoner.answer(query, context)

    print(f"Extractor: {args.extractor}")
    print(f"Reasoner: {args.reasoner}")
    print(f'Query:\n"{query}"\n')
    print("Graph reasoning:")
    print(reasoner.format_reasoning_path(context))
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
