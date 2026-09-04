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
import hashlib
import json
import re
import statistics
import subprocess  # nosec B404 - only `git`, argv list, no shell
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: Read once, so every file this run writes carries the same timestamp.
_STARTED_AT = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cke.datasets.hotpot_loader import HotpotDataset  # noqa: E402
from cke.datasets.musique_loader import MuSiQueDataset  # noqa: E402
from cke.datasets.wiki2_loader import WikiMultiHopDataset  # noqa: E402
from cke.diagnostics import (  # noqa: E402
    degradation_summary,
    environment_report,
)
from cke.evaluation.extended_metrics import EvaluationMetrics  # noqa: E402
from cke.evaluation.llm_qa import LLMAnswerer  # noqa: E402
from cke.evaluation.span_qa import SpanExtractiveQA  # noqa: E402
from cke.evaluation.token_counter import TokenCounter  # noqa: E402
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


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


#: Why a truncation figure can be missing. An answerer behind an endpoint has
#: no tokeniser for the model on the other side, so it sends the context whole
#: and its truncated count stays zero whatever the endpoint did with an
#: over-long prompt. Every output states this rather than printing that zero.
UNMEASURED_TRUNCATION = (
    "the api backend has no tokeniser for the model behind the endpoint, so "
    "the context is sent whole and what was cut, if anything, is not visible "
    "from here"
)


#: What a latency figure here covers. Each arm is timed from an empty index
#: or graph to an answer, so the figure includes building that arm's structure
#: over the item's own documents and the answerer's own time. A deployed
#: system builds its index once and would not pay that per query. Both arms
#: pay their own construction, so the columns are comparable to each other and
#: not to a production query.
LATENCY_INCLUDES = (
    "per item, from an empty index or graph to an answer: building this arm's "
    "structure over the item's documents, retrieval, and the answerer. A "
    "deployed system builds its index once and would not pay that per query"
)


def _unique(values) -> list[str]:
    """The values, in order, without repeats."""
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def gold_documents(item: dict[str, Any], docs: list[dict[str, str]]) -> set[str] | None:
    """The documents a dataset says hold the answer, as doc_ids.

    Every loader emits ``supporting_facts`` and nothing here read them, so
    both arms were scored on their answers alone: an arm could reach the
    right answer from the wrong documents, or the wrong answer while holding
    the right ones, and the tables could not tell those apart.

    A fact is resolved against the item's own documents rather than by
    dataset name. The published shapes:

    ``[title, sentence index]``   HotpotQA, 2WikiMultiHopQA
    ``[title, paragraph index]``  MuSiQue, whose titles repeat within an item
    ``[dia_id, session number]``  LoCoMo, where the first element is the turn

    Returns ``None`` when any fact names nothing in the item. A recall
    measured against a gold set that silently lost entries is a number about
    the resolution failure, not about retrieval.
    """
    facts = item.get("supporting_facts") or []
    if not facts:
        return None

    by_id = {str(doc.get("doc_id")): doc for doc in docs}
    by_title: dict[str, list[str]] = {}
    for doc in docs:
        by_title.setdefault(str(doc.get("title", "")), []).append(
            str(doc.get("doc_id"))
        )

    gold: set[str] = set()
    for fact in facts:
        if not isinstance(fact, (list, tuple)) or not fact:
            return None
        first = str(fact[0])
        if first in by_id:
            gold.add(first)
            continue
        keyed = f"{first}_{fact[1]}" if len(fact) > 1 else first
        if keyed in by_id:
            gold.add(keyed)
            continue
        candidates = by_title.get(first, [])
        if len(candidates) == 1:
            gold.add(candidates[0])
            continue
        # Either the title names no document in this item, or it names
        # several and nothing here says which.
        return None
    return gold or None


def _retrieval_recall(retrieved, gold_docs: set[str] | None) -> float | None:
    """The share of the gold documents this arm's context actually held.

    ``None`` when the arm cannot say which documents it retrieved, or the
    dataset's facts did not resolve. Zero means the arm retrieved none of
    them, which is a measurement; the two must not look alike.
    """
    if gold_docs is None or retrieved is None:
        return None
    return round(len(gold_docs & {str(doc) for doc in retrieved}) / len(gold_docs), 4)


def _truncation_of(answerer) -> tuple[bool | None, int | None]:
    """What the answerer cut on its last call, or ``None`` when it cannot tell.

    ``None`` travels through the per-item rows, the aggregates and the tables
    as "not measured". Recording ``False`` here instead put a substituted zero
    into every file downstream, indistinguishable from a measured zero.
    """
    if not getattr(answerer, "truncation_measured", True):
        return None, None
    return (
        bool(getattr(answerer, "last_truncated", False)),
        int(getattr(answerer, "last_dropped_tokens", 0)),
    )


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
    def __init__(
        self, *, token_counter: TokenCounter, answerer, strict: bool = True
    ) -> None:
        self._strict = strict
        self._qa = answerer
        self._counter = token_counter

    def run_item(
        self, question: str, docs: list[dict[str, str]], k: int
    ) -> dict[str, Any]:
        t0 = time.perf_counter()

        if not docs:
            return {
                "answer": "",
                "prompt_tokens": self._counter.count(question),
                "completion_tokens": 0,
                "latency_ms": 0.0,
                "retrieved_texts": [],
                "k": k,
            }

        retriever = RAGRetriever(strict=self._strict)
        try:
            retriever.build_index(docs)
            results = retriever.retrieve(question, k=k)
        except Exception as exc:  # noqa: BLE001 - index/backends raise varied errors
            # Scoring an empty context as if it were a retrieval result turns a
            # broken arm into a low score rather than an error.
            raise RuntimeError(
                f"dense retrieval failed for question {question!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        retrieved_texts = [str(r.get("text", "")) for r in results]
        retrieved_docs = _unique(str(r["doc_id"]) for r in results if r.get("doc_id"))
        context = "\n".join(retrieved_texts)
        prompt_tokens = self._counter.count(question) + self._counter.count(context)
        answer = self._qa.answer(question, context)
        completion_tokens = self._counter.count(answer)
        truncated, dropped = _truncation_of(self._qa)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "answer": answer,
            "answer_truncated": truncated,
            "answer_dropped_tokens": dropped,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "retrieved_texts": retrieved_texts,
            "retrieved_doc_ids": retrieved_docs,
            "k": k,
        }


# ---------------------------------------------------------------------------
# CKE-lite pipeline
# ---------------------------------------------------------------------------


class CKELitePipeline:
    def __init__(
        self, *, token_counter: TokenCounter, answerer, strict: bool = True
    ) -> None:
        self._strict = strict
        self._extractor = RuleExtractor()
        self._seed_extractor = SeedEntityExtractor()
        self._qa = answerer
        self._counter = token_counter

    def run_item(
        self, question: str, docs: list[dict[str, str]], n: int
    ) -> dict[str, Any]:
        t0 = time.perf_counter()

        # 1. Extract statements from each document
        engine = KnowledgeGraphEngine(strict=self._strict)
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
                    # The document this statement was read out of. The graph
                    # accepted a source all along and this loop dropped it, so
                    # a retrieved statement could not be traced to a document
                    # and this arm's recall against the supporting facts could
                    # not be computed while the dense arm's could.
                    source=doc.get("doc_id"),
                )
                total_statements += 1

        # 2. Extract seed entities. Mapping them onto graph entities is the
        # retriever's EntityResolver's job, which it now does by fan-out. This
        # used to be done here by a second, private implementation matching on
        # token subsets, which shadowed the resolver entirely: every seed
        # reaching the retriever was already an exact graph entity, so no rung
        # of the resolution chain past the first could ever fire.
        seeds = self._seed_extractor.extract(question)
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
                retriever = GraphRetriever(engine, strict=self._strict)
                result = retriever.retrieve(plan, mode="bfs")
                evidence = result.get("evidence", [])[:n]
            except Exception as exc:  # noqa: BLE001 - retriever raises varied errors
                raise RuntimeError(
                    f"graph retrieval failed for question {question!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        # 4. Build context from statements
        stmt_texts = [
            f"{e.get('subject', '')} {e.get('relation', '')} {e.get('object', '')}"
            for e in evidence
        ]
        retrieved_docs = _unique(str(e["source"]) for e in evidence if e.get("source"))
        context = "\n".join(stmt_texts)
        prompt_tokens = self._counter.count(question) + self._counter.count(context)
        answer = self._qa.answer(question, context)
        completion_tokens = self._counter.count(answer)
        truncated, dropped = _truncation_of(self._qa)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "answer": answer,
            "answer_truncated": truncated,
            "answer_dropped_tokens": dropped,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "n_statements": len(evidence),
            "total_extracted": total_statements,
            "retrieved_statements": stmt_texts,
            "retrieved_doc_ids": retrieved_docs,
            "n": n,
        }


# ---------------------------------------------------------------------------
# Hybrid pipeline (graph + dense fallback)
# ---------------------------------------------------------------------------


class HybridPipeline:
    """Graph-first retrieval with dense fallback via RetrievalRouter."""

    def __init__(
        self,
        *,
        token_counter: TokenCounter,
        answerer,
        dense_top_k: int = 3,
        strict: bool = True,
    ) -> None:
        self._strict = strict
        self._extractor = RuleExtractor()
        self._seed_extractor = SeedEntityExtractor()
        self._qa = answerer
        self._counter = token_counter
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
        engine = KnowledgeGraphEngine(strict=self._strict)
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

        # 2. Build RetrievalRouter with graph + dense retrievers
        graph_retriever = SimpleGraphRetriever(engine, strict=self._strict)
        dense_retriever = RAGRetriever(strict=self._strict)
        if docs:
            try:
                dense_retriever.build_index(docs)
            except Exception as exc:  # noqa: BLE001 - index raises varied errors
                # Swallowing this left the hybrid arm running graph-only while
                # still being labelled "hybrid" in every output table.
                raise RuntimeError(
                    "hybrid arm could not build its dense index, so it would "
                    f"have run graph-only while still being reported as "
                    f"hybrid: {type(exc).__name__}: {exc}"
                ) from exc

        router = RetrievalRouter(
            graph_retriever=graph_retriever,
            dense_retriever=dense_retriever,
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
        completion_tokens = self._counter.count(answer)
        truncated, dropped = _truncation_of(self._qa)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "answer": answer,
            "answer_truncated": truncated,
            "answer_dropped_tokens": dropped,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "n_statements": n_statements,
            # This arm cannot say which documents its context came from. Its
            # graph half carries a source per statement, but the dense half
            # arrives as EvidencePack.fallback_chunks, a list[str] with the
            # doc_ids already dropped. A recall computed over the graph half
            # alone would understate an arm that answered from a chunk, so
            # this reports nothing rather than half a measurement.
            "retrieved_doc_ids": None,
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


def _score_row(
    r: dict[str, Any],
    gold: str,
    *extra: str,
    gold_docs: set[str] | None = None,
) -> dict[str, Any]:
    """One per-configuration row, scored.

    Six hand-written dict literals used to do this, each copying a fixed list
    of keys. When the answerer began reporting whether it had truncated the
    context, the pipelines recorded it and every one of the six dropped it on
    the floor, so the per-arm truncation counts read zero against a shared
    total of 596. One function, one list of keys.
    """
    scored = {
        "answer": r["answer"],
        "prompt_tokens": r["prompt_tokens"],
        # What the answer itself cost. The token figures in every table so far
        # counted only what went in, and a context-size comparison that
        # ignores what comes out is half a cost.
        "completion_tokens": r.get("completion_tokens"),
        "latency_ms": r["latency_ms"],
        "retrieval_recall": _retrieval_recall(r.get("retrieved_doc_ids"), gold_docs),
        "answer_truncated": r.get("answer_truncated"),
        "answer_dropped_tokens": r.get("answer_dropped_tokens"),
        "em": EvaluationMetrics.exact_match(r["answer"], gold),
        "f1": EvaluationMetrics.f1_score(r["answer"], gold),
    }
    for key in extra:
        scored[key] = r[key]
    return scored


def run_dataset(
    items: list[dict[str, Any]],
    dataset_name: str,
    limit: int,
    token_counter: TokenCounter,
    answerer,
    verbose: bool = False,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Run all pipeline configurations for each item and return per-item results.

    Every arm counts with the same counter object, so a token figure cannot
    differ between arms because of how it was counted.
    """

    rag_pipeline = RAGPipeline(
        token_counter=token_counter, answerer=answerer, strict=strict
    )
    cke_pipeline = CKELitePipeline(
        token_counter=token_counter, answerer=answerer, strict=strict
    )
    hybrid_pipeline = HybridPipeline(
        token_counter=token_counter, answerer=answerer, strict=strict
    )
    results: list[dict[str, Any]] = []

    effective = items[:limit]
    total = len(effective)
    print(f"\n[benchmark] {dataset_name}: running {total} items...")

    for idx, item in enumerate(effective):
        question = item.get("question", "")
        gold = item.get("answer", "")
        docs = _docs_from_item(item)
        gold_docs = gold_documents(item, docs)

        if verbose or (idx % 50 == 0):
            print(f"  [{idx+1}/{total}] {question[:60]}...")

        row: dict[str, Any] = {
            "dataset": dataset_name,
            "idx": idx,
            "question": question,
            "gold_answer": gold,
            "n_docs": len(docs),
            # Named, not just counted, so a recall figure can be checked
            # against the documents it was computed over.
            "gold_doc_ids": sorted(gold_docs) if gold_docs else None,
        }

        # --- RAG k=5 ---
        r = rag_pipeline.run_item(question, docs, k=5)
        row["rag_k5"] = _score_row(r, gold, gold_docs=gold_docs)

        # --- RAG k=10 ---
        r = rag_pipeline.run_item(question, docs, k=10)
        row["rag_k10"] = _score_row(r, gold, gold_docs=gold_docs)

        # --- CKE-lite N=8 ---
        r = cke_pipeline.run_item(question, docs, n=8)
        row["cke_n8"] = _score_row(r, gold, "n_statements", gold_docs=gold_docs)

        # --- CKE-lite N=12 ---
        r = cke_pipeline.run_item(question, docs, n=12)
        row["cke_n12"] = _score_row(r, gold, "n_statements", gold_docs=gold_docs)

        # --- CKE-lite N=20 ---
        r = cke_pipeline.run_item(question, docs, n=20)
        row["cke_n20"] = _score_row(r, gold, "n_statements", gold_docs=gold_docs)

        # --- Hybrid N=12, k_fallback=3 ---
        r = hybrid_pipeline.run_item(question, docs, n=12, k_fallback=3)
        row["hybrid_n12"] = _score_row(
            r, gold, "n_statements", "mode", gold_docs=gold_docs
        )

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

CONFIGS = ["rag_k5", "rag_k10", "cke_n8", "cke_n12", "cke_n20", "hybrid_n12"]


#: Resamples drawn when estimating an interval. Enough that the interval is
#: stable to the fourth decimal the tables print, few enough to run in a
#: second on a dev-set-sized run.
BOOTSTRAP_REPLICATES = 2000

#: The seed the resampling starts from. Fixed and reported, so two runs of the
#: same command produce the same interval and a reader can reproduce it.
BOOTSTRAP_SEED = 20260904

#: Which headline figures get an interval, and how each is computed over a
#: resample. Every number the tables lead with is here.
_BOOTSTRAP_STATISTICS: dict[str, tuple[str, str]] = {
    "em": ("em", "mean"),
    "f1": ("f1", "mean"),
    "median_tokens": ("prompt_tokens", "median"),
    "median_completion_tokens": ("completion_tokens", "median"),
    "median_latency_ms": ("latency_ms", "median"),
    "retrieval_recall": ("retrieval_recall", "mean"),
}


def bootstrap_intervals(
    rows: list[dict[str, Any]],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, dict[str, float]]]:
    """Percentile intervals for each arm's headline figures.

    A point estimate over fifteen items and one over fifteen thousand print
    identically, and the tables here have shown the first while reading like
    the second. This resamples the items with replacement and reports the
    2.5th and 97.5th percentiles of each figure, so an interval that spans
    half the scale says so.

    One resample of the items is drawn and every arm and figure is computed
    over it, so the arms stay paired: they answered the same items, and
    resampling them independently would break the only thing that makes
    their columns comparable.

    A figure is estimated over the items that reported it — recall over the
    items whose supporting facts resolved — so a resample can miss them all.
    Those replicates are dropped and counted in ``replicates_used``, which is
    below ``replicates`` exactly when the estimate rests on fewer draws than
    it asked for. A figure fewer than two items reported gets no interval: a
    resample of one item is that item.
    """
    if replicates < 1:
        raise ValueError(f"replicates must be at least 1, got {replicates}")

    columns: dict[str, dict[str, np.ndarray]] = {}
    for cfg in CONFIGS:
        for name, (key, _) in _BOOTSTRAP_STATISTICS.items():
            values = [
                (
                    float(row[cfg][key])
                    if cfg in row and row[cfg].get(key) is not None
                    else float("nan")
                )
                for row in rows
            ]
            column = np.asarray(values, dtype=float)
            if np.count_nonzero(np.isfinite(column)) >= 2:
                columns.setdefault(cfg, {})[name] = column
    if not columns:
        return {}

    # Drawn once, over the items, and shared by every arm and figure below.
    draws = np.random.default_rng(seed).integers(
        0, len(rows), size=(replicates, len(rows))
    )

    intervals: dict[str, dict[str, dict[str, float]]] = {}
    for cfg, figures in columns.items():
        cfg_intervals: dict[str, dict[str, float]] = {}
        for name, column in figures.items():
            sample = column[draws]
            _, how = _BOOTSTRAP_STATISTICS[name]
            with warnings.catch_warnings():
                # A resample that drew none of the items reporting this
                # figure yields nan here, and is dropped below rather than
                # counted as a value. numpy warns about that slice; the
                # empty slice is the expected case, not a defect.
                warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
                warnings.filterwarnings(
                    "ignore", "All-NaN slice encountered", RuntimeWarning
                )
                statistic = (
                    np.nanmean(sample, axis=1)
                    if how == "mean"
                    else np.nanmedian(sample, axis=1)
                )
            usable = statistic[np.isfinite(statistic)]
            if usable.size == 0:
                continue
            low, high = np.percentile(usable, [2.5, 97.5])
            cfg_intervals[name] = {
                "low": round(float(low), 4),
                "high": round(float(high), 4),
                "replicates_used": int(usable.size),
            }
        if cfg_intervals:
            intervals[cfg] = cfg_intervals
    return intervals


def with_intervals(
    metrics: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, Any]]:
    """Attach each arm's intervals to the arm's own figures.

    Kept beside the point estimates rather than in a file of their own, so
    every output that already carries a figure carries its interval, and a
    reader cannot pick up one without the other.
    """
    intervals = bootstrap_intervals(rows, replicates=replicates, seed=seed)
    for cfg, figures in intervals.items():
        if cfg in metrics:
            metrics[cfg]["intervals"] = figures
    return metrics


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute mean EM, mean F1, median tokens, median latency per config."""
    agg: dict[str, dict[str, list[Any]]] = {
        c: {
            "em": [],
            "f1": [],
            "tokens": [],
            "completion": [],
            "latency_ms": [],
            "truncated": [],
            "recall": [],
        }
        for c in CONFIGS
    }

    for row in rows:
        for cfg in CONFIGS:
            if cfg not in row:
                continue
            d = row[cfg]
            agg[cfg]["em"].append(d.get("em", 0.0))
            agg[cfg]["truncated"].append(d.get("answer_truncated"))
            agg[cfg]["f1"].append(d.get("f1", 0.0))
            agg[cfg]["tokens"].append(d.get("prompt_tokens", 0))
            agg[cfg]["completion"].append(d.get("completion_tokens"))
            agg[cfg]["latency_ms"].append(d.get("latency_ms", 0.0))
            agg[cfg]["recall"].append(d.get("retrieval_recall"))

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
        completion = [tokens for tokens in lists["completion"] if tokens is not None]
        if len(completion) == len(lists["em"]):
            result[cfg]["median_completion_tokens"] = round(
                statistics.median(completion), 1
            )
            result[cfg]["total_completion_tokens"] = int(sum(completion))
        # Recall over the items whose supporting facts resolved. An arm that
        # cannot name its documents, or a dataset whose facts name nothing in
        # the item, contributes no measurement rather than a zero.
        measured = [value for value in lists["recall"] if value is not None]
        if measured:
            result[cfg]["retrieval_recall"] = round(sum(measured) / len(measured), 4)
            result[cfg]["recall_measured_items"] = len(measured)
        # Which arm the answerer's window cut. One answerer serves every arm,
        # so its shared totals cannot say; this can — but only when the
        # answerer measured. An unmeasured item carries None, and a count
        # taken over those would read as a measured zero in ablation.json.
        cut = lists["truncated"]
        if all(item is not None for item in cut):
            result[cfg]["truncated_items"] = sum(bool(item) for item in cut)
            result[cfg]["truncation_rate"] = round(
                sum(bool(item) for item in cut) / len(cut), 4
            )
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
    token_counter: TokenCounter,
    answerer,
) -> str:
    """Generate a markdown comparison table."""
    lines: list[str] = ["# RAG vs CKE-lite Comparison Table", ""]

    for ds_name, metrics in list(per_dataset.items()) + [("combined", combined)]:
        lines.append(f"## {ds_name}")
        lines.append("")

        header = "| Metric | " + " | ".join(_CONFIG_LABELS[c] for c in CONFIGS) + " |"
        sep = "|--------|" + "|".join("-------" for _ in CONFIGS) + "|"
        lines += [header, sep]

        def row(label: str, key: str, fmt: str = "{:.4f}") -> str:
            cells = []
            for cfg in CONFIGS:
                # Two different absences, which "nan" used to render alike:
                # an arm this run never executed, and an arm that ran without
                # producing this particular figure.
                if cfg not in metrics:
                    cells.append("not run")
                    continue
                if key not in metrics[cfg]:
                    cells.append("not measured")
                    continue
                try:
                    cells.append(fmt.format(metrics[cfg][key]))
                except (ValueError, TypeError):
                    cells.append("n/a")
            return f"| {label} | " + " | ".join(cells) + " |"

        lines.append(row("Answer EM", "em"))
        lines.append(row("Answer F1", "f1"))
        lines.append(row("Median prompt tokens", "median_tokens", "{:.0f}"))
        if any("median_completion_tokens" in metrics.get(c, {}) for c in CONFIGS):
            lines.append(
                row("Median completion tokens", "median_completion_tokens", "{:.0f}")
            )
        if any("retrieval_recall" in metrics.get(c, {}) for c in CONFIGS):
            lines.append(row("Recall of supporting docs", "retrieval_recall"))
            lines.append(
                row("Items recall measured on", "recall_measured_items", "{:.0f}")
            )
        lines.append(row("Median latency (ms)", "median_latency_ms", "{:.1f}"))
        if hasattr(answerer, "truncation"):
            if any("truncated_items" in metrics.get(c, {}) for c in CONFIGS):
                lines.append(
                    row("Items with context truncated", "truncated_items", "{:.0f}")
                )
            else:
                # The figures are missing, so the row says why. Dropping it in
                # silence reads as "nothing to report", and printing the zeros
                # the aggregate no longer holds read as a measured nothing.
                cells = " | ".join("not measured" for _ in CONFIGS)
                lines.append(f"| Items with context truncated | {cells} |")
        lines.append("")
        interval_rows = [
            (label, name)
            for label, name in (
                ("Answer EM", "em"),
                ("Answer F1", "f1"),
                ("Median prompt tokens", "median_tokens"),
                ("Median completion tokens", "median_completion_tokens"),
                ("Recall of supporting docs", "retrieval_recall"),
                ("Median latency (ms)", "median_latency_ms"),
            )
            if any(name in metrics.get(c, {}).get("intervals", {}) for c in CONFIGS)
        ]
        if interval_rows:
            lines.append(
                f"95% bootstrap intervals, {BOOTSTRAP_REPLICATES} resamples of "
                f"the items with replacement, seed {BOOTSTRAP_SEED}. One "
                f"resample is shared by every arm, so the columns stay paired. "
                f"A wide interval is the point: it says how little the figure "
                f"beside it settles."
            )
            lines.append("")
            lines += [header, sep]
            for label, name in interval_rows:
                cells = []
                for cfg in CONFIGS:
                    bounds = metrics.get(cfg, {}).get("intervals", {}).get(name)
                    if bounds is None:
                        cells.append("not measured" if cfg in metrics else "not run")
                        continue
                    cells.append(f"{bounds['low']:.4g}–{bounds['high']:.4g}")
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
            lines.append("")
        # The ratio row that used to sit here was the source of the retracted
        # headline figure. It divided one word-count estimate by another, and
        # that arithmetic objection is now gone: both arms are counted by one
        # tokenizer under one encoding. The row stays out anyway, because the
        # objection that survives is the one that mattered. The arms do not
        # supply the same information, and nothing holds accuracy constant
        # between them, so a context-size ratio is not a result about
        # retrieval. Both columns are printed; a reader who wants the quotient
        # can take it, and own it.
        lines.append(
            f"Prompt tokens counted by {token_counter.description}. A count is "
            f"only comparable to another count made with the same encoding. "
            f"Latency is {LATENCY_INCLUDES}. "
            f"Answers on every arm come from {answerer.description}."
        )
        lines.append("")

    return "\n".join(lines)


def figure_cell(
    metrics_for_config: dict[str, Any], key: str, fmt: str = "{:.4f}"
) -> str:
    """One figure with the interval it carries, or why there is none.

    Used by every markdown table, so a figure cannot appear in one report
    with its interval and in another without it, and a figure an arm never
    produced cannot print as a zero in either.
    """
    if key not in metrics_for_config:
        return "not measured"
    text = fmt.format(metrics_for_config[key])
    bounds = metrics_for_config.get("intervals", {}).get(key)
    if bounds:
        text += f" ({bounds['low']:.4g}–{bounds['high']:.4g})"
    return text


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
            lines.append(
                f"| {_CONFIG_LABELS[c]} "
                f"| {figure_cell(m, 'em')} "
                f"| {figure_cell(m, 'f1')} "
                f"| {figure_cell(m, 'median_tokens', '{:.0f}')} "
                f"| {figure_cell(m, 'median_latency_ms', '{:.1f}')} |"
            )
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
            lines.append(
                f"| {_CONFIG_LABELS[c]} "
                f"| {figure_cell(m, 'em')} "
                f"| {figure_cell(m, 'f1')} "
                f"| {figure_cell(m, 'median_tokens', '{:.0f}')} "
                f"| {figure_cell(m, 'median_latency_ms', '{:.1f}')} "
                f"| n/a |"
            )
        lines.append("")

        # Hybrid
        lines.append("### Hybrid (graph + dense fallback)")
        lines.append("")
        lines.append("| Config | EM | F1 | Median tokens | Median latency (ms) |")
        lines.append("|--------|----|----|---------------|---------------------|")
        m = metrics.get("hybrid_n12", {})
        lines.append(
            f"| Hybrid N=12 "
            f"| {figure_cell(m, 'em')} "
            f"| {figure_cell(m, 'f1')} "
            f"| {figure_cell(m, 'median_tokens', '{:.0f}')} "
            f"| {figure_cell(m, 'median_latency_ms', '{:.1f}')} |"
        )
        lines.append("")

    # This file is read on its own, so it carries the same qualifications the
    # comparison table does. A figure quoted from here without them is quoted
    # without the two things that say what it is worth.
    lines.append(
        f"Parenthesised ranges are 95% percentile bootstrap intervals, "
        f"{BOOTSTRAP_REPLICATES} resamples of the items with replacement, "
        f"seed {BOOTSTRAP_SEED}, one resample shared by every arm."
    )
    lines.append("")
    lines.append(f"Latency is {LATENCY_INCLUDES}.")

    return "\n".join(lines)


def produce_token_distribution_plot(
    all_rows: list[dict[str, Any]],
    output_path: Path,
    token_counter: TokenCounter,
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
        ax.set_xlabel(f"Prompt tokens ({token_counter.description})")
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
    token_counter: TokenCounter,
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
        if cke_tokens == 0 or cke_tokens <= token_counter.count(r.get("question", "")):
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


def produce_summary(
    combined: dict[str, dict[str, float]],
    token_counter: TokenCounter,
    answerer,
) -> dict[str, Any]:
    """Produce the raw per-arm figures, with no verdict attached.

    This used to emit a token-reduction ratio and two pass/fail success flags
    (``meets_5x_criterion``, ``meets_accuracy_criterion``). It was the source
    of the retracted headline claim. Those stay gone.

    One of the two objections to that ratio has been removed: the figures are
    now real token counts from one tokenizer, not a word count multiplied by
    1.3. The other has not. The arms do not supply the same information, and
    nothing here holds accuracy constant between them, so a context-size ratio
    is a statement about how much text each strategy hands over, not about
    what either is worth. And answers on both arms come from
    :class:`SpanExtractiveQA`, a lexical span baseline, not a language model,
    so the accuracy figures are the baseline's own.

    The per-arm numbers stay, each named with the encoding that produced it.
    A verdict belongs with an evaluation harness that can support one.
    """
    rag = combined.get("rag_k10", {})
    cke = combined.get("cke_n12", {})

    return {
        "prompt_token_counter": token_counter.description,
        "prompt_token_figures_are_estimates": token_counter.is_estimate,
        "rag_k10_median_tokens": rag.get("median_tokens", 0.0),
        "cke_n12_median_tokens": cke.get("median_tokens", 0.0),
        # What each arm's answers cost to produce, beside what they cost to
        # ask. Absent rather than zero when an arm did not report it.
        "rag_k10_median_completion_tokens": rag.get("median_completion_tokens"),
        "cke_n12_median_completion_tokens": cke.get("median_completion_tokens"),
        # Reported beside the token counts in the tables, and missing from the
        # file the tables are summarised into.
        "rag_k10_median_latency_ms": rag.get("median_latency_ms"),
        "cke_n12_median_latency_ms": cke.get("median_latency_ms"),
        "latency_includes": LATENCY_INCLUDES,
        # Every figure above, with the range the items support. A point
        # estimate over fifteen items and one over fifteen thousand print
        # identically; these do not.
        "intervals": {
            cfg: combined[cfg]["intervals"]
            for cfg in CONFIGS
            if "intervals" in combined.get(cfg, {})
        },
        "interval_method": (
            f"95% percentile bootstrap, {BOOTSTRAP_REPLICATES} resamples of "
            f"the items with replacement, seed {BOOTSTRAP_SEED}, one resample "
            f"shared by every arm"
        ),
        # Whether the context held the documents the dataset says the answer
        # needs. An answer scored right from the wrong documents and an
        # answer scored wrong while holding the right ones read alike in EM
        # and F1; this is what tells them apart.
        "retrieval_recall": {
            cfg: {
                "recall": combined[cfg]["retrieval_recall"],
                "items": combined[cfg]["recall_measured_items"],
            }
            for cfg in CONFIGS
            if "retrieval_recall" in combined.get(cfg, {})
        },
        "answers_produced_by": answerer.description,
        # What actually ran, captured after the models loaded: their identity
        # at commit level, the versions of the libraries around them, and
        # anything that degraded. The report printed at the start of a run
        # cannot carry this — nothing has been constructed yet — so a results
        # file without it cannot say which weights produced its numbers.
        "environment": environment_report().as_dict(),
        # Whether a model read the context at all. Without it the accuracy
        # columns are a lexical baseline's own figures, and a reader of the
        # results file alone has no way to tell.
        "generator_in_the_loop": bool(getattr(answerer, "uses_language_model", False)),
        "answer_truncation": _truncation_summary(answerer, combined),
        "rag_k10_em": rag.get("em", 0.0),
        "cke_n12_em": cke.get("em", 0.0),
        "rag_k10_f1": rag.get("f1", 0.0),
        "cke_n12_f1": cke.get("f1", 0.0),
    }


def _truncation_summary(answerer, combined: dict[str, dict[str, float]]):
    """What the answerer cut, or that it could not tell.

    The api backend sends a context whole because it has no tokeniser for the
    model behind the endpoint. Its truncated count therefore stays zero
    whatever the endpoint did, and printing that zero as a rate would put a
    substituted value where a measurement belongs.
    """
    if not hasattr(answerer, "truncation"):
        return None

    if not getattr(answerer, "truncation_measured", True):
        return {
            "measured": False,
            "calls": answerer.truncation.calls,
            "reason": UNMEASURED_TRUNCATION,
        }

    return {
        "measured": True,
        "calls": answerer.truncation.calls,
        "truncated": answerer.truncation.truncated,
        "rate": round(answerer.truncation.rate, 4),
        # Per arm, because the shared total cannot say which arm was cut,
        # and that is the whole question about a context window.
        # An arm whose figures were not measured is absent rather than zero,
        # for the same reason the rate above is.
        "by_arm": {
            cfg: {
                "truncated_items": combined[cfg]["truncated_items"],
                "rate": combined[cfg]["truncation_rate"],
            }
            for cfg in CONFIGS
            if "truncated_items" in combined.get(cfg, {})
        },
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


# Each loader takes the run's strictness. Without it a strict run loaded its
# data non-strict: a dataset that dropped malformed entries declared the
# degradation and the run carried on to report metrics computed over fewer
# documents than the items hold, which is what strict exists to refuse.
#: How the evaluated items are chosen from a file. "sample" draws them with a
#: seed; "prefix" takes the first N, which is what this driver always did and
#: what makes a capped run a run on whatever the file happens to list first.
#: MuSiQue's dev split is ordered by hop count, so every capped run of it was
#: a two-hop run reported as a MuSiQue run.
SELECTION_METHODS = ("sample", "prefix")

#: The seed the item sample is drawn from. Fixed and reported, so two runs of
#: one command evaluate the same items.
SAMPLE_SEED = 20260904


def _positive_int(value: str) -> int:
    """argparse type for a count that must be at least one.

    ``--limit 0`` evaluated nothing and reported the empty result as a run.
    """
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def file_digest(path: Path) -> str:
    """The sha256 of a file, read in chunks so a large corpus fits in memory.

    A results file that names its dataset by path says nothing: the path can
    hold different bytes tomorrow. The digest is what makes "the same data"
    checkable rather than asserted.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_indices(
    total: int, limit: int, seed: int = SAMPLE_SEED, method: str = "sample"
) -> list[int]:
    """Which records of a file this run evaluates, in file order.

    Returned sorted rather than in draw order, so the items are evaluated and
    written in the order they appear in the file. The seed decides which
    records are chosen; nothing should depend on the order they were drawn in.
    """
    if method not in SELECTION_METHODS:
        raise ValueError(f"unknown selection method {method!r}")
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    take = min(limit, total)
    if method == "prefix" or take == total:
        return list(range(take))
    drawn = np.random.default_rng(seed).choice(total, size=take, replace=False)
    return sorted(int(index) for index in drawn)


def load_selected(
    dataset, path: Path, limit: int, seed: int, method: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalise only the records this run evaluates, and say which they were.

    The driver used to normalise the whole file and slice the result. Two
    costs: a malformed record this run never evaluates declared a degradation
    and refused a strict run, and the slice made the cap a prefix.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    chosen = select_indices(len(raw), limit, seed, method)
    items = [dataset.normalize_record(index, raw[index]) for index in chosen]
    provenance = {
        "path": str(path),
        "sha256": file_digest(path),
        "bytes": path.stat().st_size,
        "records_in_file": len(raw),
        "items_evaluated": len(items),
        "selection": {"method": method, "seed": seed if method == "sample" else None},
        "item_ids": [str(item.get("id")) for item in items],
    }
    return items, provenance


#: Fields of the results that cannot repeat, and why. A run is reproducible
#: everywhere else; naming the exceptions is what lets "run it twice and diff"
#: be a check rather than a hope.
NON_REPRODUCIBLE_FIELDS = {
    "started_at": "the wall clock reads differently on the second run",
    "median_latency_ms": "a timing, which no seed fixes",
    "rag_k10_median_latency_ms": "a timing, which no seed fixes",
    "cke_n12_median_latency_ms": "a timing, which no seed fixes",
    "latency_ms": "a timing, which no seed fixes",
}


def _git_description() -> dict[str, Any]:
    """The commit this ran from, and whether the tree was clean.

    A commit alone does not determine the code when the tree is dirty, and a
    results file that names a commit while the working tree differed from it
    is claiming a provenance it does not have.
    """

    def _git(*command: str) -> str | None:
        try:
            done = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
                ["git", *command],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        # None, not False: an unknown tree state is not a clean one.
        "clean_tree": (status == "") if status is not None else None,
        "uncommitted_files": (
            len([line for line in status.splitlines() if line.strip()])
            if status
            else 0 if status == "" else None
        ),
    }


def run_provenance(
    args: argparse.Namespace,
    datasets: dict[str, dict[str, Any]],
    answerer,
    token_counter: TokenCounter,
) -> dict[str, Any]:
    """Everything a second run needs to produce these numbers again.

    The question this answers is not "what happened" but "what would I have
    to hold fixed to get this again": the code, the data, the weights, the
    seeds, and the command. Anything that cannot be held fixed is named in
    NON_REPRODUCIBLE_FIELDS rather than left for a reader to discover by
    diffing two runs.
    """
    return {
        "cke": _git_description(),
        "command": list(sys.argv),
        "seeds": {
            "item_sample": args.sample_seed,
            "bootstrap": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "selection_method": args.select,
        "limit": args.limit,
        "strict": not args.allow_degraded,
        "datasets": datasets,
        "answerer": answerer.description,
        "prompt_token_counter": token_counter.description,
        "environment": environment_report().as_dict(),
        "started_at": _STARTED_AT,
        "non_reproducible_fields": NON_REPRODUCIBLE_FIELDS,
    }


def compare_runs(first: Path, second: Path) -> list[str]:
    """Where two runs' results disagree on anything a seed should have fixed.

    The gate for this work is "run it twice and diff". Diffing the files
    themselves always reports a difference, because they carry timings and a
    timestamp, so the useful comparison is of what NON_REPRODUCIBLE_FIELDS
    does not exclude. An empty list is the pass.
    """

    def _walk(left: Any, right: Any, path: str) -> list[str]:
        if isinstance(left, dict) and isinstance(right, dict):
            differences: list[str] = []
            for key in sorted(set(left) | set(right)):
                where = f"{path}.{key}" if path else key
                if key not in left:
                    differences.append(f"{where}: only in the second run")
                elif key not in right:
                    differences.append(f"{where}: only in the first run")
                else:
                    differences += _walk(left[key], right[key], where)
            return differences
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                return [f"{path}: {len(left)} entries then {len(right)}"]
            return [
                difference
                for index, (a, b) in enumerate(zip(left, right))
                for difference in _walk(a, b, f"{path}[{index}]")
            ]
        if left != right:
            return [f"{path}: {left!r} then {right!r}"]
        return []

    with open(first, encoding="utf-8") as handle:
        left = deterministic_view(json.load(handle))
    with open(second, encoding="utf-8") as handle:
        right = deterministic_view(json.load(handle))
    return _walk(left, right, "")


def deterministic_view(payload: Any) -> Any:
    """The part of a results payload that two runs must agree on exactly.

    Drops the fields NON_REPRODUCIBLE_FIELDS names, at any depth. Two runs of
    one command whose deterministic views differ have a defect in this
    harness, not noise.
    """
    if isinstance(payload, dict):
        return {
            key: deterministic_view(value)
            for key, value in payload.items()
            if key not in NON_REPRODUCIBLE_FIELDS
        }
    if isinstance(payload, list):
        return [deterministic_view(value) for value in payload]
    return payload


def _load_hotpotqa(
    path: Path, limit: int, strict: bool, seed: int, method: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return load_selected(HotpotDataset(strict=strict), path, limit, seed, method)


def _load_musique(
    path: Path, limit: int, strict: bool, seed: int, method: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return load_selected(MuSiQueDataset(strict=strict), path, limit, seed, method)


def _load_wiki2(
    path: Path, limit: int, strict: bool, seed: int, method: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return load_selected(WikiMultiHopDataset(strict=strict), path, limit, seed, method)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_retrieval_mode_ablation(
    items: list[dict[str, Any]],
    dataset_name: str,
    limit: int,
    token_counter: TokenCounter,
    answerer,
    verbose: bool = False,
    strict: bool = True,
) -> dict[str, dict[str, float]]:
    """Ablate across retrieval modes: graph_only, dense_only, hybrid."""
    cke_pipeline = CKELitePipeline(
        token_counter=token_counter, answerer=answerer, strict=strict
    )
    rag_pipeline = RAGPipeline(
        token_counter=token_counter, answerer=answerer, strict=strict
    )
    hybrid_pipeline = HybridPipeline(
        token_counter=token_counter, answerer=answerer, dense_top_k=3, strict=strict
    )

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
        mode_results["graph_only"].append(
            {
                "em": EvaluationMetrics.exact_match(r["answer"], gold),
                "f1": EvaluationMetrics.f1_score(r["answer"], gold),
                "tokens": r["prompt_tokens"],
                "latency_ms": r["latency_ms"],
            }
        )

        # dense_only
        r = rag_pipeline.run_item(question, docs, k=5)
        mode_results["dense_only"].append(
            {
                "em": EvaluationMetrics.exact_match(r["answer"], gold),
                "f1": EvaluationMetrics.f1_score(r["answer"], gold),
                "tokens": r["prompt_tokens"],
                "latency_ms": r["latency_ms"],
            }
        )

        # hybrid
        r = hybrid_pipeline.run_item(question, docs, n=12, k_fallback=3)
        mode_results["hybrid"].append(
            {
                "em": EvaluationMetrics.exact_match(r["answer"], gold),
                "f1": EvaluationMetrics.f1_score(r["answer"], gold),
                "tokens": r["prompt_tokens"],
                "latency_ms": r["latency_ms"],
                "mode": 1.0 if r["mode"] == "hybrid" else 0.0,
            }
        )

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
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=500,
        help="Items evaluated per dataset",
    )
    parser.add_argument(
        "--select",
        choices=SELECTION_METHODS,
        default="sample",
        help=(
            "how those items are chosen: 'sample' draws them with --sample-seed, "
            "'prefix' takes the first N. A prefix of a file ordered by anything "
            "-- MuSiQue's dev split is ordered by hop count -- is a run on that "
            "ordering rather than on the dataset."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=SAMPLE_SEED,
        help="seed the item sample is drawn from; recorded in provenance.json",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--hotpot-path", default=None)
    parser.add_argument("--wiki2-path", default=None)
    parser.add_argument("--musique-path", default=None)
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help=(
            "Permit components to run degraded. Off by default: a benchmark "
            "whose embedder fell back to hashing is not a benchmark."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download (use existing files)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--answerer",
        choices=("span", "llm"),
        default="span",
        help=(
            "What reads the retrieved context. 'span' is the lexical baseline; "
            "'llm' puts a language model in the loop, and under strict a run "
            "with no reachable model refuses rather than falling back to span."
        ),
    )
    parser.add_argument("--llm-backend", choices=("local", "api"), default="local")
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model name for the LLM answerer (default: google/flan-t5-base locally)",
    )
    parser.add_argument(
        "--llm-revision",
        default=None,
        help=(
            "The 40-character Hub commit sha to pin a local model to. A branch "
            "or tag is refused: it moves. The default model is pinned already; "
            "any other model must be given one, or a strict run refuses."
        ),
    )
    parser.add_argument(
        "--llm-window",
        type=int,
        default=None,
        help=(
            "Longest prompt, in tokens, handed to a local model. Default is the "
            "window the model reports for itself. Truncation is counted and "
            "reported either way."
        ),
    )
    parser.add_argument(
        "--retrieval-ablation",
        action="store_true",
        help="Run retrieval mode ablation (graph_only vs dense_only vs hybrid)",
    )
    args = parser.parse_args()

    print(environment_report().render(), flush=True)
    strict = not args.allow_degraded
    token_counter = TokenCounter(strict=strict)
    # One answerer for every arm. It is built here, strict, so a run with no
    # model stops before its first item instead of quietly answering with
    # the span baseline while being labelled 'llm'.
    if args.answerer == "llm":
        answerer = LLMAnswerer(
            backend=args.llm_backend,
            model=args.llm_model,
            strict=strict,
            max_input_tokens=args.llm_window,
            model_revision=args.llm_revision,
        )
    else:
        answerer = SpanExtractiveQA()
    print(f"[answerer] {answerer.description}", flush=True)

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
        # No cap: the file is the published split, and the run samples from
        # it with a seed. A capped download made the file a prefix of the
        # split, and no seed downstream can undo an ordering already baked
        # into the data — MuSiQue's dev split is ordered by hop count, so
        # every capped run of it was a two-hop run reported as MuSiQue.
        dl_mod.download_hotpotqa(data_dir / "hotpotqa_dev.json")
        dl_mod.download_wiki2(data_dir / "wiki2_dev.json")
        dl_mod.download_musique(data_dir / "musique_dev.json")

    hotpot_path = (
        Path(args.hotpot_path) if args.hotpot_path else data_dir / "hotpotqa_dev.json"
    )
    wiki2_path = (
        Path(args.wiki2_path) if args.wiki2_path else data_dir / "wiki2_dev.json"
    )
    musique_path = (
        Path(args.musique_path) if args.musique_path else data_dir / "musique_dev.json"
    )

    # --- Load datasets ---
    datasets: dict[str, list[dict[str, Any]]] = {}
    dataset_provenance: dict[str, dict[str, Any]] = {}
    if hotpot_path.exists():
        try:
            datasets["hotpotqa"], dataset_provenance["hotpotqa"] = _load_hotpotqa(
                hotpot_path, args.limit, strict, args.sample_seed, args.select
            )
            print(
                f"[load] HotpotQA: {len(datasets['hotpotqa'])} of "
                f"{dataset_provenance['hotpotqa']['records_in_file']} records, "
                f"{args.select}"
            )
        except Exception as exc:  # noqa: BLE001 - loaders raise varied errors
            # Continuing here computed the report over whichever datasets
            # happened to load, with no note of the one that did not.
            raise RuntimeError(
                f"HotpotQA failed to load from {hotpot_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    else:
        print(f"[load] HotpotQA not found at {hotpot_path}")

    if wiki2_path.exists():
        try:
            datasets["wiki2"], dataset_provenance["wiki2"] = _load_wiki2(
                wiki2_path, args.limit, strict, args.sample_seed, args.select
            )
            print(
                f"[load] 2WikiMultiHopQA: {len(datasets['wiki2'])} of "
                f"{dataset_provenance['wiki2']['records_in_file']} records, "
                f"{args.select}"
            )
        except Exception as exc:  # noqa: BLE001 - loaders raise varied errors
            raise RuntimeError(
                f"2WikiMultiHopQA failed to load from {wiki2_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    else:
        print(f"[load] 2WikiMultiHopQA not found at {wiki2_path}")

    if musique_path.exists():
        try:
            datasets["musique"], dataset_provenance["musique"] = _load_musique(
                musique_path, args.limit, strict, args.sample_seed, args.select
            )
            print(
                f"[load] MuSiQue: {len(datasets['musique'])} of "
                f"{dataset_provenance['musique']['records_in_file']} records, "
                f"{args.select}"
            )
        except Exception as exc:  # noqa: BLE001 - loaders raise varied errors
            raise RuntimeError(
                f"MuSiQue failed to load from {musique_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    else:
        print(f"[load] MuSiQue not found at {musique_path}")

    if not datasets:
        print("[error] No datasets loaded. Exiting.")
        sys.exit(1)

    # --- Run benchmark ---
    all_rows: list[dict[str, Any]] = []
    per_dataset_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for ds_name, items in datasets.items():
        rows = run_dataset(
            items,
            ds_name,
            limit=args.limit,
            token_counter=token_counter,
            answerer=answerer,
            verbose=args.verbose,
            strict=strict,
        )
        metrics = with_intervals(aggregate_metrics(rows), rows)
        per_dataset_metrics[ds_name] = metrics
        all_rows.extend(rows)

        (output_dir / f"full_results_{ds_name}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[output] full_results_{ds_name}.json ({len(rows)} items)")

    # --- Combined metrics ---
    combined_metrics = with_intervals(aggregate_metrics(all_rows), all_rows)

    # --- Comparison table ---
    comparison_md = produce_comparison_table(
        per_dataset_metrics, combined_metrics, token_counter, answerer
    )
    (output_dir / "comparison_table.md").write_text(comparison_md, encoding="utf-8")
    print("[output] comparison_table.md")

    # --- Ablation ---
    ablation_json: dict[str, Any] = {
        ds: {cfg: m for cfg, m in metrics.items()}
        for ds, metrics in per_dataset_metrics.items()
    }
    ablation_json["combined"] = combined_metrics
    # Read on its own, so it says what its latency covers rather than leaving
    # the figure to be taken for a production query time.
    ablation_json["latency_includes"] = LATENCY_INCLUDES
    (output_dir / "ablation.json").write_text(
        json.dumps(ablation_json, indent=2), encoding="utf-8"
    )
    print("[output] ablation.json")

    ablation_md = produce_ablation_table(per_dataset_metrics, combined_metrics)
    (output_dir / "ablation.md").write_text(ablation_md, encoding="utf-8")
    print("[output] ablation.md")

    # --- Token distribution ---
    produce_token_distribution_plot(
        all_rows, output_dir / "token_distribution.png", token_counter
    )

    # --- Failure analysis ---
    failure_samples = produce_failure_analysis(all_rows, token_counter, n=10)
    (output_dir / "failure_analysis.json").write_text(
        json.dumps(failure_samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[output] failure_analysis.json ({len(failure_samples)} samples)")

    # --- Summary ---
    provenance = run_provenance(args, dataset_provenance, answerer, token_counter)
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print("[output] provenance.json")
    if provenance["cke"]["clean_tree"] is False:
        # Not a warning about tidiness: the commit named above does not
        # describe the code that produced these numbers.
        print(
            f"[provenance] the working tree had "
            f"{provenance['cke']['uncommitted_files']} uncommitted file(s), so "
            f"commit {provenance['cke']['commit']} does not fully describe this "
            f"run",
            flush=True,
        )

    summary = produce_summary(combined_metrics, token_counter, answerer)
    summary["provenance"] = provenance
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("[output] summary.json")

    loaded = summary["environment"]["loaded_models"]
    if loaded:
        print("[models] identity recorded in summary.json:")
        for entry in loaded:
            print(f"    {entry['component']}: {entry['loaded']}")
    else:
        print("[models] no model was loaded in this run")

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
    print(f"  Prompt tokens counted by {token_counter.description}.")
    print(f"  Answers on every arm come from {answerer.description}.")
    print("  No success criterion is evaluated: these figures cannot support one.")
    print("=" * 60)

    # --- Retrieval mode ablation ---
    if args.retrieval_ablation:
        retrieval_ablation: dict[str, dict[str, dict[str, float]]] = {}
        for ds_name, items in datasets.items():
            retrieval_ablation[ds_name] = run_retrieval_mode_ablation(
                items,
                ds_name,
                limit=args.limit,
                token_counter=token_counter,
                answerer=answerer,
                verbose=args.verbose,
                strict=strict,
            )
        (output_dir / "retrieval_ablation.json").write_text(
            json.dumps(
                {**retrieval_ablation, "latency_includes": LATENCY_INCLUDES},
                indent=2,
            ),
            encoding="utf-8",
        )
        print("[output] retrieval_ablation.json")

        print("\n" + "=" * 60)
        print("RETRIEVAL MODE ABLATION")
        print("=" * 60)
        for ds_name, modes in retrieval_ablation.items():
            print(f"\n  {ds_name}:")
            for mode_name, metrics in modes.items():
                fb = (
                    f"  fallback_rate={metrics['fallback_rate']:.2%}"
                    if "fallback_rate" in metrics
                    else ""
                )
                print(
                    f"    {mode_name:12s} — EM: {metrics['em']:.4f}  "
                    f"F1: {metrics['f1']:.4f}  "
                    f"Median tokens: {metrics['median_tokens']:.0f}{fb}"
                )
        print("=" * 60)

    print(f"\nAll results written to: {output_dir.resolve()}")
    print(degradation_summary(), flush=True)


if __name__ == "__main__":
    main()
