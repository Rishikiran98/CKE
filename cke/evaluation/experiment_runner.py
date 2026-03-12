"""Baseline evaluation runner for retrieval + generation experiments."""

from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any, Callable

from cke.retrieval.rag_baseline import RAGRetriever
from cke.observability.token_tracker import TokenTracker
from cke.utils.experiment_logger import ExperimentLogger


class ExperimentRunner:
    """Run end-to-end baseline experiments and aggregate metrics."""

    def __init__(
        self,
        retriever: RAGRetriever,
        logger: ExperimentLogger | None = None,
        answer_generator: Callable[[str, list[dict[str, Any]]], str] | None = None,
        token_tracker: TokenTracker | None = None,
    ) -> None:
        self.retriever = retriever
        self.logger = logger or ExperimentLogger()
        self.answer_generator = answer_generator or self._default_answer_generator
        self.token_tracker = token_tracker or TokenTracker()

    def run(self, dataset: list[dict[str, str]], top_k: int = 5) -> dict[str, float]:
        total = max(len(dataset), 1)
        exact_matches = 0
        total_f1 = 0.0
        total_latency = 0.0
        total_tokens = 0

        for item in dataset:
            question = item.get("question", "")
            gold_answer = item.get("answer", "")

            start = time.perf_counter()
            retrieved = self.retriever.retrieve(question, k=top_k)
            answer = self.answer_generator(question, retrieved)
            latency = time.perf_counter() - start

            em = self._exact_match(answer, gold_answer)
            f1 = self._f1_score(answer, gold_answer)
            tokens = sum(len(str(doc.get("text", "")).split()) for doc in retrieved)
            prompt_tokens = len(question.split())
            completion_tokens = len(answer.split())
            self.token_tracker.add_usage(prompt_tokens, completion_tokens)

            exact_matches += int(em)
            total_f1 += f1
            total_latency += latency
            total_tokens += tokens

            retrieved_nodes = self._extract_nodes(retrieved)
            gold_nodes = item.get("gold_nodes", [])
            self.logger.log_query(
                question=question,
                retrieved_items=retrieved,
                answer=answer,
                latency=latency,
                tokens=tokens,
                correct=em,
                retrieved_nodes=retrieved_nodes,
                gold_nodes=gold_nodes,
            )

        return {
            "exact_match": exact_matches / total,
            "f1_score": total_f1 / total,
            "latency": total_latency / total,
            "tokens": total_tokens / total,
            "token_usage": self.token_tracker.to_dict(),
        }


    @staticmethod
    def _extract_nodes(docs: list[dict[str, Any]]) -> list[str]:
        nodes: set[str] = set()
        for item in docs:
            for key in ("subject", "object", "entity", "title"):
                value = item.get(key)
                if value:
                    nodes.add(str(value))
        return sorted(nodes)

    @staticmethod
    def _default_answer_generator(_question: str, docs: list[dict[str, Any]]) -> str:
        return str(docs[0]["text"]) if docs else ""

    @staticmethod
    def _normalize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _exact_match(self, prediction: str, target: str) -> bool:
        return " ".join(self._normalize(prediction)) == " ".join(
            self._normalize(target)
        )

    def _f1_score(self, prediction: str, target: str) -> float:
        pred_tokens = self._normalize(prediction)
        target_tokens = self._normalize(target)
        if not pred_tokens or not target_tokens:
            return 0.0
        common = Counter(pred_tokens) & Counter(target_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(pred_tokens)
        recall = overlap / len(target_tokens)
        return 2 * precision * recall / (precision + recall)
