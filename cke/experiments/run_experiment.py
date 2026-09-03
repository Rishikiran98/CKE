"""Experiment runner for CKE vs baseline RAG."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from cke.diagnostics import (
    declare_degradation,
    degradation_summary,
    environment_report,
)
from cke.extractor.extractor import BaseExtractor, RuleBasedExtractor
from cke.extractor.llm_extractor import LLMExtractor
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.reasoning.llm_reasoner import LLMReasoner
from cke.reasoning.path_reasoner import PathReasoner
from cke.reasoning.reasoner import TemplateReasoner
from cke.retrieval.rag_baseline import RAGBaseline
from cke.retrieval.retriever import GraphRetriever


@dataclass
class QAItem:
    question: str
    context: str
    answer: str


#: One hand-written item, for smoke-testing that the pipeline runs at all.
#: Accuracy over it is not a result; ``--dataset`` is required to report one.
_SMOKE_TEST_ITEM = QAItem(
    question="What protocol does Redis pub/sub use?",
    context=("Redis supports PubSub messaging. PubSub implemented_via RESP protocol."),
    answer="RESP",
)


def load_dataset(path: Path | None = None, strict: bool = True) -> List[QAItem]:
    """Load an evaluation dataset, or refuse.

    Previously a missing or nonexistent ``--dataset`` silently substituted a
    single hand-written item and the script went on to print accuracy figures
    over it. That is one authored example scoring itself.
    """
    if path is not None and not path.exists():
        raise FileNotFoundError(
            f"dataset {path} does not exist. This script does not substitute "
            "built-in data for a dataset you asked for."
        )

    if path is not None:
        raw = json.loads(path.read_text())
        return [
            QAItem(
                question=item["question"],
                context=item["context"],
                answer=item["answer"],
            )
            for item in raw
        ]

    declare_degradation(
        "run_experiment.load_dataset",
        "no --dataset was given, so the run uses a single hand-written smoke "
        "test item. Accuracy over one authored example is not a measurement",
        strict=strict,
    )
    return [_SMOKE_TEST_ITEM]


def build_extractor(name: str, strict: bool = False) -> BaseExtractor:
    if name == "llm":
        return LLMExtractor(strict=strict)
    return RuleBasedExtractor()


def build_reasoner(name: str, strict: bool = False):
    if name == "llm":
        return LLMReasoner(strict=strict)
    if name == "template":
        return TemplateReasoner()
    return PathReasoner(strict=strict)


def evaluate(
    items: Iterable[QAItem],
    extractor_name: str = "rule",
    reasoner_name: str = "template",
    strict: bool = True,
) -> dict:
    extractor = build_extractor(extractor_name, strict=strict)
    reasoner = build_reasoner(reasoner_name, strict=strict)
    rag = RAGBaseline(strict=strict)

    items = list(items)
    rag.build_index([item.context for item in items])

    graph_correct = 0
    rag_correct = 0
    graph_tokens = 0
    rag_tokens = 0
    graph_latency = 0.0
    rag_latency = 0.0

    for item in items:
        graph = KnowledgeGraphEngine(strict=strict)
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
        graph_tokens += sum(len(statement.as_text().split()) for statement in graph_ctx)
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
        "--reasoner", choices=["template", "path", "llm"], default="path"
    )
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help=(
            "Permit components to run degraded. Off by default: a run whose "
            "extractor fell back to regexes is not a measurement."
        ),
    )
    args = parser.parse_args()

    print(environment_report().render(), flush=True)
    strict = not args.allow_degraded

    metrics = evaluate(
        load_dataset(args.dataset, strict=strict),
        strict=strict,
        extractor_name=args.extractor,
        reasoner_name=args.reasoner,
    )
    print("Experiment results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(degradation_summary(), flush=True)


if __name__ == "__main__":
    main()
