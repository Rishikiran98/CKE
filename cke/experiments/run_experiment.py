"""Experiment runner for CKE vs baseline RAG."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from cke.extractor.extractor import BaseExtractor, RuleBasedExtractor
from cke.extractor.llm_extractor import LLMExtractor
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.reasoning.llm_reasoner import LLMReasoner
from cke.reasoning.reasoner import TemplateReasoner
from cke.retrieval.rag_baseline import RAGBaseline
from cke.retrieval.retriever import GraphRetriever


@dataclass
class QAItem:
    question: str
    context: str
    answer: str


def load_dataset(path: Path | None = None) -> List[QAItem]:
    if path and path.exists():
        raw = json.loads(path.read_text())
        return [
            QAItem(
                question=item["question"],
                context=item["context"],
                answer=item["answer"],
            )
            for item in raw
        ]

    # Tiny built-in sample for local prototype runs.
    return [
        QAItem(
            question="What protocol does Redis pub/sub use?",
            context=(
                "Redis supports PubSub messaging. "
                "PubSub implemented_via RESP protocol."
            ),
            answer="RESP",
        ),
    ]


def build_extractor(name: str) -> BaseExtractor:
    if name == "llm":
        return LLMExtractor()
    return RuleBasedExtractor()


def build_reasoner(name: str):
    if name == "llm":
        return LLMReasoner()
    return TemplateReasoner()


def evaluate(
    items: Iterable[QAItem],
    extractor_name: str = "rule",
    reasoner_name: str = "template",
) -> dict:
    extractor = build_extractor(extractor_name)
    reasoner = build_reasoner(reasoner_name)
    rag = RAGBaseline()

    items = list(items)
    rag.build_index([item.context for item in items])

    graph_correct = 0
    rag_correct = 0
    graph_tokens = 0
    rag_tokens = 0
    graph_latency = 0.0
    rag_latency = 0.0

    for item in items:
        graph = KnowledgeGraphEngine()
        graph.add_statements(extractor.extract(item.context))
        retriever = GraphRetriever(graph)

        start = time.perf_counter()
        graph_ctx = retriever.retrieve(item.question, max_depth=3)
        graph_answer = reasoner.answer(item.question, graph_ctx)
        graph_latency += (time.perf_counter() - start) * 1000

        rag_ctx, r_latency = rag.retrieve(item.question, top_k=1)
        rag_answer = rag_ctx[0].chunk if rag_ctx else ""
        rag_latency += r_latency

        gold = item.answer.lower()
        graph_correct += int(gold in graph_answer.lower())
        rag_correct += int(gold in rag_answer.lower())
        graph_tokens += sum(
            len(statement.as_text().split()) for statement in graph_ctx
        )
        rag_tokens += sum(len(result.chunk.split()) for result in rag_ctx)

    n = len(items) or 1
    return {
        "graph_accuracy": graph_correct / n,
        "rag_accuracy": rag_correct / n,
        "graph_tokens_retrieved": graph_tokens / n,
        "rag_tokens_retrieved": rag_tokens / n,
        "graph_latency_ms": graph_latency / n,
        "rag_latency_ms": rag_latency / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to HotpotQA-like json sample",
    )
    parser.add_argument("--extractor", choices=["rule", "llm"], default="rule")
    parser.add_argument(
        "--reasoner", choices=["template", "llm"], default="template"
    )
    args = parser.parse_args()

    metrics = evaluate(
        load_dataset(args.dataset),
        extractor_name=args.extractor,
        reasoner_name=args.reasoner,
    )
    print("Experiment results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
