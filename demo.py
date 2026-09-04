"""Run the pipeline end to end on five sentences and one question.

What this shows
---------------
Rule-based extraction from prose, a knowledge graph built from the triples,
graph retrieval around the entity the question names, and path reasoning
across two hops with a transitivity rule. Every component is constructed
with ``strict=True``: if the embedding model cannot be loaded, the demo stops
and says so rather than answering on a hashed stand-in. The environment
report printed first says which optional dependencies are present; the
closing lines say which models actually loaded and whether anything
degraded, because neither is known until the components have been built.

What it does not show
---------------------
A general question answerer. The path reasoner resolves a question's target
relation only from "located in" and "nationality", or from a relation named
as the question's last word, and it prefers the longest chain it can verify,
so "Which city is Hagia Sophia located in?" would be answered "Turkey" too.
The corpus is written in the sentence frames the rule extractor reads, and
subjects carry no leading article because the graph retriever seeds a walk
from the entity name exactly as the question states it. The reasoning trace
printed above the answer is the evidence for it; CI asserts on both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cke.diagnostics import degradation_summary, environment_report
from cke.extractor.rule_extractor import RuleExtractor
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.reasoning.path_reasoner import PathReasoner
from cke.retrieval.retriever import GraphRetriever

CORPUS = [
    "Hagia Sophia is located in Istanbul.",
    "Hagia Sophia was completed in 537.",
    "Blue Mosque is located in Istanbul.",
    "Istanbul is located in Turkey.",
    "Istanbul is a city.",
]

QUESTION = "Which country is Hagia Sophia located in?"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CKE demo: extract five sentences, then answer one question"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="SQLite file to keep the graph in. Omit to keep it in memory.",
    )
    args = parser.parse_args()

    print(environment_report().render(), flush=True)

    db_path = args.db_path
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    graph = KnowledgeGraphEngine(db_path=db_path, strict=True)

    extractor = RuleExtractor()
    print("\nExtracted:")
    for sentence in CORPUS:
        statements = extractor.extract(sentence)
        graph.add_statements(statements)
        for statement in statements:
            print(f"  {statement.subject} -[{statement.relation}]-> {statement.object}")

    retriever = GraphRetriever(graph, strict=True)
    reasoner = PathReasoner(strict=True)
    context = retriever.retrieve(QUESTION, max_depth=3)
    answer = reasoner.answer(QUESTION, context)

    if db_path is not None:
        print(f"\nDB path: {db_path}")
    print(f"\nQuestion: {QUESTION}")
    print("\nReasoning:")
    print(reasoner.format_reasoning_path(context))
    print(f"\nAnswer: {answer}")

    # What ran, as opposed to what was available: the report at the top is
    # taken before any component exists, so it cannot say this.
    print("\nModels loaded:")
    for model in environment_report().loaded_models:
        print(f"  {model.component}: {model.loaded}")
    print(degradation_summary())


if __name__ == "__main__":
    main()
