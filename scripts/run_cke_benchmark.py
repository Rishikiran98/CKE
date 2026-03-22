#!/usr/bin/env python3
"""RAG vs CKE-lite benchmark on HotpotQA and 2WikiMultiHopQA.

Produces:
  results/comparison_table.md   — EM / F1 / median tokens / median latency
  results/ablation.json         — ablation over k and N values
  results/ablation.md           — ablation table (markdown)
  results/token_distribution.png — histogram (or .json fallback)
  results/failure_analysis.json — 10 failure samples
  results/summary.json          — top-level success flags
  results/full_results_*.json   — per-item results per dataset
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cke.datasets.hotpot_loader import HotpotDataset  # noqa: E402
from cke.datasets.wiki2_loader import WikiMultiHopDataset  # noqa: E402
from cke.evaluation.extended_metrics import EvaluationMetrics  # noqa: E402
from cke.extractor.rule_extractor import RuleExtractor  # noqa: E402
from cke.graph_engine.graph_engine import KnowledgeGraphEngine  # noqa: E402
from cke.retrieval.graph_retriever import GraphRetriever  # noqa: E402
from cke.retrieval.hybrid_retrieval import HybridRetrievalMerger  # noqa: E402
from cke.retrieval.rag_baseline import RAGRetriever  # noqa: E402
from cke.retrieval.retrieval_router import RetrievalRouter  # noqa: E402
from cke.retrieval.retriever import GraphRetriever as SimpleGraphRetriever  # noqa: E402
from cke.router.query_plan import QueryPlan  # noqa: E402

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class TokenCounter:
    """Approximate BPE token count via word count × 1.3."""

    @staticmethod
    def count(text: str) -> int:
        words = len(text.split())
        return max(1, int(words * 1.3))


class SeedEntityExtractor:
    """Extract candidate seed entities from a question string."""

    _CAP_PHRASE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
    _QUOTED = re.compile(r'"([^"]+)"')

    def extract(self, question: str) -> list[str]:
        candidates: list[str] = []
        for m in self._QUOTED.finditer(question):
            candidates.append(m.group(1).strip())
        for m in self._CAP_PHRASE.finditer(question):
            phrase = m.group(0).strip()
            if phrase not in candidates:
                candidates.append(phrase)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for c in candidates:
            if c.lower() not in seen and len(c) > 1:
                seen.add(c.lower())
                unique.append(c)
        return unique[:4]


class SimpleExtractiveQA:
    """Extractive answer from retrieved context — same algorithm for both paths."""

    _counter = TokenCounter()

    def answer(self, question: str, context: str) -> str:
        if not context.strip():
            return ""
        q_tokens = set(re.findall(r"\w+", question.lower()))
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", context) if s.strip()]
        if not sentences:
            return ""

        best_sent = ""
        best_score = -1.0
        for sent in sentences:
            s_tokens = set(re.findall(r"\w+", sent.lower()))
            if not s_tokens:
                continue
            overlap = len(q_tokens & s_tokens)
            score = overlap / (len(s_tokens) + 1e-9)
            if score > best_score:
                best_score = score
                best_sent = sent

        # Return first 50 words of best sentence as the predicted answer
        words = best_sent.split()
        return " ".join(words[:50])


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _docs_from_item(item: dict[str, Any]) -> list[dict[str, str]]:
    """Normalise HotpotQA and 2WikiMultiHopQA item formats to a doc list."""
    if "documents" in item:
        return [d for d in item["documents"] if d.get("text")]
    # 2WikiMultiHopQA: contexts is a flat list of strings
    contexts = item.get("contexts", [])
    return [{"doc_id": f"ctx_{i}", "text": str(c)} for i, c in enumerate(contexts) if c]


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    def __init__(self) -> None:
        self._qa = SimpleExtractiveQA()
        self._counter = TokenCounter()

    def run_item(
        self, question: str, docs: list[dict[str, str]], k: int
    ) -> dict[str, Any]:
        t0 = time.perf_counter()

        if not docs:
            return {
                "answer": "",
                "prompt_tokens": self._counter.count(question),
                "latency_ms": 0.0,
                "retrieved_texts": [],
                "k": k,
            }

        retriever = RAGRetriever()
        try:
            retriever.build_index(docs)
            results = retriever.retrieve(question, k=k)
        except Exception:
            results = []

        retrieved_texts = [str(r.get("text", "")) for r in results]
        context = "\n".join(retrieved_texts)
        prompt_tokens = self._counter.count(question) + self._counter.count(context)
        answer = self._qa.answer(question, context)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "answer": answer,
            "prompt_tokens": prompt_tokens,
            "latency_ms": latency_ms,
            "retrieved_texts": retrieved_texts,
            "k": k,
        }


# ---------------------------------------------------------------------------
# CKE-lite pipeline
# ---------------------------------------------------------------------------


class CKELitePipeline:
    def __init__(self) -> None:
        self._extractor = RuleExtractor()
        self._seed_extractor = SeedEntityExtractor()
        self._qa = SimpleExtractiveQA()
        self._counter = TokenCounter()

    @staticmethod
    def _expand_seeds(seeds: list[str], engine: KnowledgeGraphEngine) -> list[str]:
        """Map question entities to actual graph entities via word-level matching.

        Requires all non-stopword seed tokens to appear in the entity, or an exact
        seed-as-prefix match. Falls back to 80% overlap for single-word seeds.
        """
        _STOP = {
            "the",
            "a",
            "an",
            "of",
            "in",
            "is",
            "was",
            "are",
            "be",
            "by",
            "for",
            "with",
        }
        all_entities = engine.all_entities()
        if not all_entities:
            return seeds
        expanded: list[str] = []
        for seed in seeds:
            if len(seed) < 3:
                continue
            seed_toks = set(re.findall(r"\w+", seed.lower())) - _STOP
            if not seed_toks:
                continue
            matched: list[tuple[float, str]] = []
            for e in all_entities:
                if len(e) < 2:
                    continue
                e_lower = e.lower()
                e_toks = set(re.findall(r"\w+", e_lower)) - _STOP
                if not e_toks:
                    continue
                # Require all seed tokens present in entity
                if seed_toks.issubset(e_toks):
                    score = len(seed_toks) / len(e_toks)  # prefer shorter entities
                    matched.append((score, e))
            matched.sort(reverse=True)
            expanded.extend(e for _, e in matched[:3])
        # Deduplicate preserving order
        seen: set[str] = set()
        result: list[str] = []
        for e in expanded:
            if e not in seen:
                seen.add(e)
                result.append(e)
        return result[:6] if result else seeds

    def run_item(
        self, question: str, docs: list[dict[str, str]], n: int
    ) -> dict[str, Any]:
        t0 = time.perf_counter()

        # 1. Extract statements from each document
        engine = KnowledgeGraphEngine()
        total_statements = 0
        for doc in docs:
            text = doc.get("text", "")
            if not text:
                continue
            statements = self._extractor.extract(text)
            for st in statements:
                engine.add_statement(
                    st.subject,
                    st.relation,
                    st.object,
                    confidence=st.confidence,
                )
                total_statements += 1

        # 2. Extract seed entities, then expand to actual graph entities
        seeds = self._seed_extractor.extract(question)
        seeds = self._expand_seeds(seeds, engine)
        plan = QueryPlan(
            query_text=question,
            seed_entities=seeds,
            intent="factoid",
            max_depth=2,
            max_results=n,
        )

        # 3. Graph retrieval
        evidence: list[dict[str, Any]] = []
        if total_statements > 0:
            try:
                retriever = GraphRetriever(engine)
                result = retriever.retrieve(plan, mode="bfs")
                evidence = result.get("evidence", [])[:n]
            except Exception:
                evidence = []

        # 4. Build context from statements
        stmt_texts = [
            f"{e.get('subject', '')} {e.get('relation', '')} {e.get('object', '')}"
            for e in evidence
        ]
        context = "\n".join(stmt_texts)
        prompt_tokens = self._counter.count(question) + self._counter.count(context)
        answer = self._qa.answer(question, context)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "answer": answer,
            "prompt_tokens": prompt_tokens,
            "latency_ms": latency_ms,
            "n_statements": len(evidence),
            "total_extracted": total_statements,
            "retrieved_statements": stmt_texts,
            "n": n,
        }


# ---------------------------------------------------------------------------
# Hybrid pipeline (graph + dense fallback)
# ---------------------------------------------------------------------------


class HybridPipeline:
    """Graph-first retrieval with dense fallback via RetrievalRouter."""

    def __init__(self, evidence_threshold: int = 2, dense_top_k: int = 3) -> None:
        self._extractor = RuleExtractor()
        self._seed_extractor = SeedEntityExtractor()
        self._qa = SimpleExtractiveQA()
        self._counter = TokenCounter()
        self._evidence_threshold = evidence_threshold
        self._dense_top_k = dense_top_k
        self._merger = HybridRetrievalMerger()
        self._total_fallbacks = 0
        self._total_queries = 0

    def run_item(
        self,
        question: str,
        docs: list[dict[str, str]],
        n: int,
        k_fallback: int = 3,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        self._total_queries += 1

        # 1. Build graph and extract statements
        engine = KnowledgeGraphEngine()
        total_statements = 0
        for doc in docs:
            text = doc.get("text", "")
            if not text:
                continue
            statements = self._extractor.extract(text)
            for st in statements:
                engine.add_statement(
                    st.subject, st.relation, st.object, confidence=st.confidence,
                )
                total_statements += 1

        # 2. Build RetrievalRouter with graph + dense retrievers
        graph_retriever = SimpleGraphRetriever(engine)
        dense_retriever = RAGRetriever()
        if docs:
            try:
                dense_retriever.build_index(docs)
            except Exception:
                pass

        router = RetrievalRouter(
            graph_retriever=graph_retriever,
            dense_retriever=dense_retriever,
            evidence_threshold=self._evidence_threshold,
            dense_top_k=k_fallback,
        )

        # 3. Retrieve via router (graph-first, auto dense fallback)
        evidence_pack = router.retrieve(question, max_depth=2)
        graph_texts = [st.as_text() for st in evidence_pack.graph_statements[:n]]
        dense_texts = evidence_pack.fallback_chunks
        n_statements = len(evidence_pack.graph_statements)

        fallback_used = len(dense_texts) > 0
        if fallback_used:
            self._total_fallbacks += 1
        mode = "hybrid" if fallback_used else "graph_only"

        # 4. Merge and answer
        all_texts = graph_texts + dense_texts
        context = "\n".join(all_texts)
        prompt_tokens = self._counter.count(question) + self._counter.count(context)
        answer = self._qa.answer(question, context)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "answer": answer,
            "prompt_tokens": prompt_tokens,
            "latency_ms": latency_ms,
            "n_statements": n_statements,
            "mode": mode,
            "n": n,
            "fallback_rate": (
                self._total_fallbacks / self._total_queries
                if self._total_queries > 0
                else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Dataset benchmark runner
# ---------------------------------------------------------------------------


def run_dataset(
    items: list[dict[str, Any]],
    dataset_name: str,
    limit: int,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Run all pipeline configurations for each item and return per-item results."""

    rag_pipeline = RAGPipeline()
    cke_pipeline = CKELitePipeline()
    hybrid_pipeline = HybridPipeline()
    results: list[dict[str, Any]] = []

    effective = items[:limit]
    total = len(effective)
    print(f"\n[benchmark] {dataset_name}: running {total} items...")

    for idx, item in enumerate(effective):
        question = item.get("question", "")
        gold = item.get("answer", "")
        docs = _docs_from_item(item)

        if verbose or (idx % 50 == 0):
            print(f"  [{idx+1}/{total}] {question[:60]}...")

        row: dict[str, Any] = {
            "dataset": dataset_name,
            "idx": idx,
            "question": question,
            "gold_answer": gold,
            "n_docs": len(docs),
        }

        # --- RAG k=5 ---
        r = rag_pipeline.run_item(question, docs, k=5)
        row["rag_k5"] = {
            "answer": r["answer"],
            "prompt_tokens": r["prompt_tokens"],
            "latency_ms": r["latency_ms"],
            "em": EvaluationMetrics.exact_match(r["answer"], gold),
            "f1": EvaluationMetrics.f1_score(r["answer"], gold),
        }

        # --- RAG k=10 ---
        r = rag_pipeline.run_item(question, docs, k=10)
        row["rag_k10"] = {
            "answer": r["answer"],
            "prompt_tokens": r["prompt_tokens"],
            "latency_ms": r["latency_ms"],
            "em": EvaluationMetrics.exact_match(r["answer"], gold),
            "f1": EvaluationMetrics.f1_score(r["answer"], gold),
        }

        # --- CKE-lite N=8 ---
        r = cke_pipeline.run_item(question, docs, n=8)
        row["cke_n8"] = {
            "answer": r["answer"],
            "prompt_tokens": r["prompt_tokens"],
            "latency_ms": r["latency_ms"],
            "n_statements": r["n_statements"],
            "em": EvaluationMetrics.exact_match(r["answer"], gold),
            "f1": EvaluationMetrics.f1_score(r["answer"], gold),
        }

        # --- CKE-lite N=12 ---
        r = cke_pipeline.run_item(question, docs, n=12)
        row["cke_n12"] = {
            "answer": r["answer"],
            "prompt_tokens": r["prompt_tokens"],
            "latency_ms": r["latency_ms"],
            "n_statements": r["n_statements"],
            "em": EvaluationMetrics.exact_match(r["answer"], gold),
            "f1": EvaluationMetrics.f1_score(r["answer"], gold),
        }

        # --- CKE-lite N=20 ---
        r = cke_pipeline.run_item(question, docs, n=20)
        row["cke_n20"] = {
            "answer": r["answer"],
            "prompt_tokens": r["prompt_tokens"],
            "latency_ms": r["latency_ms"],
            "n_statements": r["n_statements"],
            "em": EvaluationMetrics.exact_match(r["answer"], gold),
            "f1": EvaluationMetrics.f1_score(r["answer"], gold),
        }

        # --- Hybrid N=12, k_fallback=3 ---
        r = hybrid_pipeline.run_item(question, docs, n=12, k_fallback=3)
        row["hybrid_n12"] = {
            "answer": r["answer"],
            "prompt_tokens": r["prompt_tokens"],
            "latency_ms": r["latency_ms"],
            "n_statements": r["n_statements"],
            "mode": r["mode"],
            "em": EvaluationMetrics.exact_match(r["answer"], gold),
            "f1": EvaluationMetrics.f1_score(r["answer"], gold),
        }

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

CONFIGS = ["rag_k5", "rag_k10", "cke_n8", "cke_n12", "cke_n20", "hybrid_n12"]


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute mean EM, mean F1, median tokens, median latency per config."""
    agg: dict[str, dict[str, list[float]]] = {
        c: {"em": [], "f1": [], "tokens": [], "latency_ms": []} for c in CONFIGS
    }

    for row in rows:
        for cfg in CONFIGS:
            if cfg not in row:
                continue
            d = row[cfg]
            agg[cfg]["em"].append(d.get("em", 0.0))
            agg[cfg]["f1"].append(d.get("f1", 0.0))
            agg[cfg]["tokens"].append(d.get("prompt_tokens", 0))
            agg[cfg]["latency_ms"].append(d.get("latency_ms", 0.0))

    result: dict[str, dict[str, float]] = {}
    for cfg, lists in agg.items():
        if not lists["em"]:
            continue
        result[cfg] = {
            "em": round(sum(lists["em"]) / len(lists["em"]), 4),
            "f1": round(sum(lists["f1"]) / len(lists["f1"]), 4),
            "median_tokens": round(statistics.median(lists["tokens"]), 1),
            "median_latency_ms": round(statistics.median(lists["latency_ms"]), 2),
            "n": len(lists["em"]),
        }
    return result


# ---------------------------------------------------------------------------
# Output production
# ---------------------------------------------------------------------------

_CONFIG_LABELS = {
    "rag_k5": "RAG k=5",
    "rag_k10": "RAG k=10",
    "cke_n8": "CKE N=8",
    "cke_n12": "CKE N=12",
    "cke_n20": "CKE N=20",
    "hybrid_n12": "Hybrid N=12",
}


def produce_comparison_table(
    per_dataset: dict[str, dict[str, dict[str, float]]],
    combined: dict[str, dict[str, float]],
) -> str:
    """Generate a markdown comparison table."""
    lines: list[str] = ["# RAG vs CKE-lite Comparison Table", ""]

    for ds_name, metrics in list(per_dataset.items()) + [("combined", combined)]:
        lines.append(f"## {ds_name}")
        lines.append("")

        rag_k10_tokens = metrics.get("rag_k10", {}).get("median_tokens", 1.0)

        header = "| Metric | " + " | ".join(_CONFIG_LABELS[c] for c in CONFIGS) + " |"
        sep = "|--------|" + "|".join("-------" for _ in CONFIGS) + "|"
        lines += [header, sep]

        def row(label: str, key: str, fmt: str = "{:.4f}") -> str:
            vals = [metrics.get(c, {}).get(key, float("nan")) for c in CONFIGS]
            cells = []
            for v in vals:
                try:
                    cells.append(fmt.format(v))
                except (ValueError, TypeError):
                    cells.append("n/a")
            return f"| {label} | " + " | ".join(cells) + " |"

        lines.append(row("Answer EM", "em"))
        lines.append(row("Answer F1", "f1"))
        lines.append(row("Median prompt tokens", "median_tokens", "{:.0f}"))
        lines.append(row("Median latency (ms)", "median_latency_ms", "{:.1f}"))

        # Token reduction vs RAG k=10
        reductions = []
        for c in CONFIGS:
            cke_tokens = metrics.get(c, {}).get("median_tokens", float("nan"))
            try:
                red = rag_k10_tokens / cke_tokens if cke_tokens > 0 else float("nan")
                reductions.append(f"{red:.1f}×")
            except (ZeroDivisionError, TypeError):
                reductions.append("n/a")
        lines.append("| Token reduction vs RAG k=10 | " + " | ".join(reductions) + " |")
        lines.append("")

    return "\n".join(lines)


def produce_ablation_table(
    per_dataset: dict[str, dict[str, dict[str, float]]],
    combined: dict[str, dict[str, float]],
) -> str:
    """Generate a markdown ablation table grouped by RAG vs CKE configurations."""
    lines: list[str] = ["# Ablation: Retrieval Budget", ""]

    for ds_name, metrics in list(per_dataset.items()) + [("combined", combined)]:
        lines.append(f"## {ds_name}")
        lines.append("")

        rag_cfgs = ["rag_k5", "rag_k10"]
        cke_cfgs = ["cke_n8", "cke_n12", "cke_n20"]

        lines.append("### RAG baseline (k ablation)")
        lines.append("")
        lines.append("| Config | EM | F1 | Median tokens | Median latency (ms) |")
        lines.append("|--------|----|----|---------------|---------------------|")
        for c in rag_cfgs:
            m = metrics.get(c, {})
            em = m.get("em", 0)
            f1 = m.get("f1", 0)
            tok = m.get("median_tokens", 0)
            lat = m.get("median_latency_ms", 0)
            lbl = _CONFIG_LABELS[c]
            lines.append(f"| {lbl} | {em:.4f} | {f1:.4f} " f"| {tok:.0f} | {lat:.1f} |")
        lines.append("")

        lines.append("### CKE-lite (N ablation)")
        lines.append("")
        lines.append(
            "| Config | EM | F1 "
            "| Median tokens | Median latency (ms) "
            "| Avg statements |"
        )
        lines.append(
            "|--------|----|----|"
            "---------------|---------------------"
            "|----------------|"
        )
        for c in cke_cfgs:
            m = metrics.get(c, {})
            em = m.get("em", 0)
            f1 = m.get("f1", 0)
            tok = m.get("median_tokens", 0)
            lat = m.get("median_latency_ms", 0)
            lbl = _CONFIG_LABELS[c]
            lines.append(
                f"| {lbl} | {em:.4f} | {f1:.4f} " f"| {tok:.0f} | {lat:.1f} " f"| n/a |"
            )
        lines.append("")

        # Hybrid
        lines.append("### Hybrid (graph + dense fallback)")
        lines.append("")
        lines.append("| Config | EM | F1 | Median tokens | Median latency (ms) |")
        lines.append("|--------|----|----|---------------|---------------------|")
        m = metrics.get("hybrid_n12", {})
        lines.append(
            f"| Hybrid N=12 | {m.get('em', 0):.4f} | {m.get('f1', 0):.4f} "
            f"| {m.get('median_tokens', 0):.0f} | {m.get('median_latency_ms', 0):.1f} |"
        )
        lines.append("")

    return "\n".join(lines)


def produce_token_distribution_plot(
    all_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save a histogram comparing RAG k=10 vs CKE N=12 prompt token distributions."""
    rag_tokens = [r["rag_k10"]["prompt_tokens"] for r in all_rows if "rag_k10" in r]
    cke_tokens = [r["cke_n12"]["prompt_tokens"] for r in all_rows if "cke_n12" in r]

    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(rag_tokens, bins=40, alpha=0.6, color="steelblue", label="RAG k=10")
        ax.hist(
            cke_tokens, bins=40, alpha=0.6, color="darkorange", label="CKE-lite N=12"
        )
        ax.set_xlabel("Prompt tokens (word count × 1.3)")
        ax.set_ylabel("Number of items")
        ax.set_title("Token Distribution: RAG k=10 vs CKE-lite N=12")
        ax.legend()
        ax.axvline(
            statistics.median(rag_tokens) if rag_tokens else 0,
            color="steelblue",
            linestyle="--",
            linewidth=1.5,
            label=(
                f"RAG median={statistics.median(rag_tokens):.0f}" if rag_tokens else ""
            ),
        )
        ax.axvline(
            statistics.median(cke_tokens) if cke_tokens else 0,
            color="darkorange",
            linestyle="--",
            linewidth=1.5,
            label=(
                f"CKE median={statistics.median(cke_tokens):.0f}" if cke_tokens else ""
            ),
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=120)
        plt.close(fig)
        print(f"[output] Token distribution plot → {output_path}")
    except ImportError:
        # Fallback: save data as JSON
        json_path = output_path.with_suffix(".json")
        json_path.write_text(
            json.dumps({"rag_k10": rag_tokens, "cke_n12": cke_tokens}, indent=2),
            encoding="utf-8",
        )
        print(f"[output] matplotlib unavailable — token data → {json_path}")


def produce_failure_analysis(
    all_rows: list[dict[str, Any]],
    n: int = 10,
) -> list[dict[str, Any]]:
    """Select n items where both RAG k=10 and CKE N=12 fail (EM=0)."""
    joint_failures = [
        r
        for r in all_rows
        if r.get("rag_k10", {}).get("em", 1) == 0.0
        and r.get("cke_n12", {}).get("em", 1) == 0.0
    ]

    # Supplement with either-path failures if not enough joint failures
    if len(joint_failures) < n:
        either_failures = [
            r
            for r in all_rows
            if r not in joint_failures
            and (
                r.get("rag_k10", {}).get("em", 1) == 0.0
                or r.get("cke_n12", {}).get("em", 1) == 0.0
            )
        ]
        joint_failures = (joint_failures + either_failures)[:n]

    samples = joint_failures[:n]
    analysis = []
    for r in samples:
        rag = r.get("rag_k10", {})
        cke = r.get("cke_n12", {})

        # Classify failure mode
        rag_tokens = rag.get("prompt_tokens", 0)
        cke_tokens = cke.get("prompt_tokens", 0)
        if cke_tokens == 0 or cke_tokens <= _token_count_static(r.get("question", "")):
            note = "CKE graph empty — no statements extracted"
        elif rag.get("f1", 0) > cke.get("f1", 0) + 0.1:
            note = "RAG outperforms CKE — dense context contained answer"
        elif cke.get("f1", 0) > rag.get("f1", 0) + 0.1:
            note = "CKE outperforms RAG — graph captured relevant relation"
        else:
            note = "Both paths fail — answer not in retrieved context"

        analysis.append(
            {
                "dataset": r.get("dataset", ""),
                "question": r.get("question", ""),
                "gold_answer": r.get("gold_answer", ""),
                "rag_prediction": rag.get("answer", ""),
                "cke_prediction": cke.get("answer", ""),
                "rag_tokens": rag_tokens,
                "cke_tokens": cke_tokens,
                "rag_f1": round(rag.get("f1", 0.0), 4),
                "cke_f1": round(cke.get("f1", 0.0), 4),
                "failure_mode": note,
            }
        )
    return analysis


def _token_count_static(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def produce_summary(
    combined: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Produce top-level success-criterion flags."""
    rag = combined.get("rag_k10", {})
    cke = combined.get("cke_n12", {})

    rag_tokens = rag.get("median_tokens", 1.0)
    cke_tokens = cke.get("median_tokens", 1.0)
    token_reduction = rag_tokens / cke_tokens if cke_tokens > 0 else 0.0

    em_delta = cke.get("em", 0.0) - rag.get("em", 0.0)
    f1_delta = cke.get("f1", 0.0) - rag.get("f1", 0.0)

    return {
        "token_reduction_rag_k10_vs_cke_n12": round(token_reduction, 2),
        "meets_5x_criterion": token_reduction >= 5.0,
        "em_delta_cke_vs_rag": round(em_delta, 4),
        "f1_delta_cke_vs_rag": round(f1_delta, 4),
        "meets_accuracy_criterion": em_delta >= -0.02,
        "rag_k10_median_tokens": rag_tokens,
        "cke_n12_median_tokens": cke_tokens,
        "rag_k10_em": rag.get("em", 0.0),
        "cke_n12_em": cke.get("em", 0.0),
        "rag_k10_f1": rag.get("f1", 0.0),
        "cke_n12_f1": cke.get("f1", 0.0),
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_hotpotqa(path: Path, limit: int) -> list[dict[str, Any]]:
    ds = HotpotDataset()
    ds.load(str(path))
    return ds.items[:limit]


def _load_wiki2(path: Path, limit: int) -> list[dict[str, Any]]:
    ds = WikiMultiHopDataset()
    ds.load(str(path))
    # wiki2 items have 'contexts' (list of str) not 'documents'
    return ds.items[:limit]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_retrieval_mode_ablation(
    items: list[dict[str, Any]],
    dataset_name: str,
    limit: int,
    verbose: bool = False,
) -> dict[str, dict[str, float]]:
    """Ablate across retrieval modes: graph_only, dense_only, hybrid."""
    cke_pipeline = CKELitePipeline()
    rag_pipeline = RAGPipeline()
    hybrid_pipeline = HybridPipeline(evidence_threshold=2, dense_top_k=3)

    effective = items[:limit]
    total = len(effective)
    print(f"\n[ablation] {dataset_name}: retrieval mode ablation on {total} items...")

    mode_results: dict[str, list[dict[str, float]]] = {
        "graph_only": [],
        "dense_only": [],
        "hybrid": [],
    }

    for idx, item in enumerate(effective):
        question = item.get("question", "")
        gold = item.get("answer", "")
        docs = _docs_from_item(item)

        if verbose or (idx % 50 == 0):
            print(f"  [ablation {idx+1}/{total}] {question[:60]}...")

        # graph_only
        r = cke_pipeline.run_item(question, docs, n=12)
        mode_results["graph_only"].append({
            "em": EvaluationMetrics.exact_match(r["answer"], gold),
            "f1": EvaluationMetrics.f1_score(r["answer"], gold),
            "tokens": r["prompt_tokens"],
            "latency_ms": r["latency_ms"],
        })

        # dense_only
        r = rag_pipeline.run_item(question, docs, k=5)
        mode_results["dense_only"].append({
            "em": EvaluationMetrics.exact_match(r["answer"], gold),
            "f1": EvaluationMetrics.f1_score(r["answer"], gold),
            "tokens": r["prompt_tokens"],
            "latency_ms": r["latency_ms"],
        })

        # hybrid
        r = hybrid_pipeline.run_item(question, docs, n=12, k_fallback=3)
        mode_results["hybrid"].append({
            "em": EvaluationMetrics.exact_match(r["answer"], gold),
            "f1": EvaluationMetrics.f1_score(r["answer"], gold),
            "tokens": r["prompt_tokens"],
            "latency_ms": r["latency_ms"],
            "mode": 1.0 if r["mode"] == "hybrid" else 0.0,
        })

    agg: dict[str, dict[str, float]] = {}
    for mode_name, rows in mode_results.items():
        n = max(len(rows), 1)
        agg[mode_name] = {
            "em": round(sum(r["em"] for r in rows) / n, 4),
            "f1": round(sum(r["f1"] for r in rows) / n, 4),
            "median_tokens": round(statistics.median([r["tokens"] for r in rows]), 1),
            "median_latency_ms": round(
                statistics.median([r["latency_ms"] for r in rows]), 2
            ),
            "n": n,
        }
        if mode_name == "hybrid":
            agg[mode_name]["fallback_rate"] = round(
                sum(r.get("mode", 0.0) for r in rows) / n, 4
            )

    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="CKE benchmark: RAG vs CKE-lite")
    parser.add_argument("--limit", type=int, default=500, help="Items per dataset")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--hotpot-path", default=None)
    parser.add_argument("--wiki2-path", default=None)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download (use existing files)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--retrieval-ablation",
        action="store_true",
        help="Run retrieval mode ablation (graph_only vs dense_only vs hybrid)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- Download ---
    if not args.skip_download:
        import importlib.util

        dl_script = ROOT / "scripts" / "download_datasets.py"
        spec = importlib.util.spec_from_file_location("download_datasets", dl_script)
        dl_mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(dl_mod)  # type: ignore[union-attr]
        dl_mod.download_hotpotqa(data_dir / "hotpotqa_dev.json", limit=args.limit)
        dl_mod.download_wiki2(data_dir / "wiki2_dev.json", limit=args.limit)

    hotpot_path = (
        Path(args.hotpot_path) if args.hotpot_path else data_dir / "hotpotqa_dev.json"
    )
    wiki2_path = (
        Path(args.wiki2_path) if args.wiki2_path else data_dir / "wiki2_dev.json"
    )

    # --- Load datasets ---
    datasets: dict[str, list[dict[str, Any]]] = {}
    if hotpot_path.exists():
        try:
            datasets["hotpotqa"] = _load_hotpotqa(hotpot_path, args.limit)
            print(f"[load] HotpotQA: {len(datasets['hotpotqa'])} items")
        except Exception as exc:
            print(f"[load] HotpotQA failed: {exc}")
    else:
        print(f"[load] HotpotQA not found at {hotpot_path}")

    if wiki2_path.exists():
        try:
            datasets["wiki2"] = _load_wiki2(wiki2_path, args.limit)
            print(f"[load] 2WikiMultiHopQA: {len(datasets['wiki2'])} items")
        except Exception as exc:
            print(f"[load] 2WikiMultiHopQA failed: {exc}")
    else:
        print(f"[load] 2WikiMultiHopQA not found at {wiki2_path}")

    if not datasets:
        print("[error] No datasets loaded. Exiting.")
        sys.exit(1)

    # --- Run benchmark ---
    all_rows: list[dict[str, Any]] = []
    per_dataset_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for ds_name, items in datasets.items():
        rows = run_dataset(items, ds_name, limit=args.limit, verbose=args.verbose)
        metrics = aggregate_metrics(rows)
        per_dataset_metrics[ds_name] = metrics
        all_rows.extend(rows)

        (output_dir / f"full_results_{ds_name}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[output] full_results_{ds_name}.json ({len(rows)} items)")

    # --- Combined metrics ---
    combined_metrics = aggregate_metrics(all_rows)

    # --- Comparison table ---
    comparison_md = produce_comparison_table(per_dataset_metrics, combined_metrics)
    (output_dir / "comparison_table.md").write_text(comparison_md, encoding="utf-8")
    print("[output] comparison_table.md")

    # --- Ablation ---
    ablation_json = {
        ds: {cfg: m for cfg, m in metrics.items()}
        for ds, metrics in per_dataset_metrics.items()
    }
    ablation_json["combined"] = combined_metrics
    (output_dir / "ablation.json").write_text(
        json.dumps(ablation_json, indent=2), encoding="utf-8"
    )
    print("[output] ablation.json")

    ablation_md = produce_ablation_table(per_dataset_metrics, combined_metrics)
    (output_dir / "ablation.md").write_text(ablation_md, encoding="utf-8")
    print("[output] ablation.md")

    # --- Token distribution ---
    produce_token_distribution_plot(all_rows, output_dir / "token_distribution.png")

    # --- Failure analysis ---
    failure_samples = produce_failure_analysis(all_rows, n=10)
    (output_dir / "failure_analysis.json").write_text(
        json.dumps(failure_samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[output] failure_analysis.json ({len(failure_samples)} samples)")

    # --- Summary ---
    summary = produce_summary(combined_metrics)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("[output] summary.json")

    # Print key results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    rag = combined_metrics.get("rag_k10", {})
    cke = combined_metrics.get("cke_n12", {})
    r_em = rag.get("em", 0)
    r_f1 = rag.get("f1", 0)
    r_tok = rag.get("median_tokens", 0)
    c_em = cke.get("em", 0)
    c_f1 = cke.get("f1", 0)
    c_tok = cke.get("median_tokens", 0)
    print(
        f"  RAG k=10  — EM: {r_em:.4f}  " f"F1: {r_f1:.4f}  Median tokens: {r_tok:.0f}"
    )
    print(
        f"  CKE N=12  — EM: {c_em:.4f}  " f"F1: {c_f1:.4f}  Median tokens: {c_tok:.0f}"
    )
    tok_red = summary["token_reduction_rag_k10_vs_cke_n12"]
    meets_5x = summary["meets_5x_criterion"]
    em_delta = summary["em_delta_cke_vs_rag"]
    meets_acc = summary["meets_accuracy_criterion"]
    print(f"  Token reduction: {tok_red:.1f}x " f"(>=5x criterion: {meets_5x})")
    print(f"  EM delta (CKE vs RAG): {em_delta:+.4f} " f"(within +/-0.02: {meets_acc})")
    print("=" * 60)

    # --- Retrieval mode ablation ---
    if args.retrieval_ablation:
        retrieval_ablation: dict[str, dict[str, dict[str, float]]] = {}
        for ds_name, items in datasets.items():
            retrieval_ablation[ds_name] = run_retrieval_mode_ablation(
                items, ds_name, limit=args.limit, verbose=args.verbose,
            )
        (output_dir / "retrieval_ablation.json").write_text(
            json.dumps(retrieval_ablation, indent=2), encoding="utf-8",
        )
        print("[output] retrieval_ablation.json")

        print("\n" + "=" * 60)
        print("RETRIEVAL MODE ABLATION")
        print("=" * 60)
        for ds_name, modes in retrieval_ablation.items():
            print(f"\n  {ds_name}:")
            for mode_name, metrics in modes.items():
                fb = f"  fallback_rate={metrics['fallback_rate']:.2%}" if "fallback_rate" in metrics else ""
                print(
                    f"    {mode_name:12s} — EM: {metrics['em']:.4f}  "
                    f"F1: {metrics['f1']:.4f}  "
                    f"Median tokens: {metrics['median_tokens']:.0f}{fb}"
                )
        print("=" * 60)

    print(f"\nAll results written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
