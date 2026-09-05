"""Dense retrieval evaluation: MS MARCO documents, HotpotQA and LoCoMo queries.

Indexes MS MARCO full documents with a sentence-transformer and a vector
index, retrieves the top-k documents for each HotpotQA and LoCoMo query, and
reports a hit rate against each query's relevance hints.

Contract
--------
This module used to import faiss, pandas and sentence-transformers at the top
and build a ``SentenceTransformer`` directly, so it could not be imported
without all three and had no way to say what it was running on. It now
composes :class:`~cke.retrieval.embedding_model.EmbeddingModel` and
:class:`~cke.retrieval.faiss_index.FaissIndex`, the components every other
evaluation path uses, and inherits their degradation: without
sentence-transformers or faiss it declares what is missing and, under
``strict`` (the command line's default), refuses to run rather than report a
a hit rate measured on a hashed embedder or a numpy scan.

Two readers stay local to this module rather than coming from the dataset
registry, because the registry's are the wrong shape for what the metric
needs. The corpus reader takes the four-column MS MARCO full-document TSV
(id, url, title, body); :class:`~cke.datasets.msmarco_loader.MSMarcoDocumentDataset`
reads two-column rows and would swallow the title the relevance test matches
on. The LoCoMo reader keeps the relevance fields the registry loader drops.
Reconciling them belongs to the evaluation rebuild, not to this change.

What this measures, and what it is called
----------------------------------------
The figure is the fraction of queries with **at least one** relevant document
in the top k. That is a hit rate, and it was called Recall@k. Recall@k is
something else: the share of a query's relevant documents that the top k
holds, which is what ``_retrieval_recall`` in scripts/run_cke_benchmark.py
computes for the benchmark. Two numbers under one name, differing whenever a
query has more than one relevant document — the hit rate reaches 1.0 on the
first of them and recall does not. Neither figure was wrong; reading one as
the other was, and nothing here said which was which.

A retrieved document counts as relevant when its id is among the query's
hints or a hint is a substring of its lower-cased title. Hints come from
HotpotQA's supporting-fact titles and answer, and from whichever relevance
fields a LoCoMo record carries. That is a title-matching proxy, not a judged
relevance set: MS MARCO documents were never labelled against HotpotQA or
LoCoMo queries, and a figure from here says how often a title matched, not
how often the right passage was found.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from cke.datasets.hotpot_loader import HotpotDataset
from cke.diagnostics import (
    DegradationMixin,
    degradation_summary,
    environment_report,
    require_strict_component,
)
from cke.retrieval.embedding_model import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingModel,
)
from cke.retrieval.faiss_index import FaissIndex

__all__ = [
    "CorpusDocument",
    "DenseRetriever",
    "MSMARCOCorpus",
    "QueryExample",
    "evaluate_hit_rate_at_k",
    "load_hotpot_queries",
    "load_locomo_queries",
    "run_pipeline",
]

DEFAULT_MODEL_NAME = DEFAULT_EMBEDDING_MODEL


@dataclass(frozen=True)
class QueryExample:
    """Query and a set of relevant document title/document-id hints."""

    query_id: str
    query_text: str
    relevant_hints: set[str]


@dataclass(frozen=True)
class CorpusDocument:
    """One MS MARCO document: its id, title, and the text that is embedded."""

    doc_id: str
    title: str
    text: str

    @property
    def title_norm(self) -> str:
        return self.title.lower().strip()


class MSMARCOCorpus(DegradationMixin):
    """MS MARCO full documents from the TSV of ``id, url, title, body`` rows.

    A three-column file (``id, title, body``) is accepted too. Rows with fewer
    columns carry no title and no body; they are not padded into empty
    documents, since an empty document that can never be relevant would lower
    every hit rate without anyone seeing why. They are counted and the
    count is declared as a degradation.
    """

    def __init__(
        self, tsv_path: Path, max_docs: int | None = None, strict: bool = False
    ) -> None:
        self._init_degradation(strict)
        self.tsv_path = Path(tsv_path)
        self.max_docs = max_docs
        self.documents = self._load()

    def _load(self) -> list[CorpusDocument]:
        documents: list[CorpusDocument] = []
        malformed = 0
        with self.tsv_path.open("r", encoding="utf-8", newline="") as handle:
            # MS MARCO bodies contain quotation marks that are text, not
            # quoting; a quote-aware reader would join rows across them.
            reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if not row or not any(cell.strip() for cell in row):
                    continue
                if len(row) >= 4:
                    doc_id, _url, title, body = row[:4]
                elif len(row) == 3:
                    doc_id, title, body = row
                else:
                    malformed += 1
                    continue
                text = f"{title.strip()}\n{body.strip()}".strip()
                documents.append(
                    CorpusDocument(doc_id=str(doc_id).strip(), title=title, text=text)
                )
                if self.max_docs is not None and len(documents) >= self.max_docs:
                    break

        if malformed:
            self._degrade(
                f"{malformed} rows of {self.tsv_path} had fewer than three "
                "columns and were dropped, so the corpus is smaller than the "
                "file. Expected id, url, title, body"
            )
        if not documents:
            raise ValueError(
                f"No documents were read from {self.tsv_path}. Expected "
                "tab-separated rows of id, url, title, body"
            )
        return documents

    @property
    def texts(self) -> list[str]:
        return [document.text for document in self.documents]

    @property
    def doc_ids(self) -> list[str]:
        return [document.doc_id for document in self.documents]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, a_min=1e-12, a_max=None)


class DenseRetriever(DegradationMixin):
    """Sentence-transformer embeddings in a vector index, ranked by cosine.

    Vectors are L2-normalised before indexing and before search. On unit
    vectors the index's L2 distance orders documents exactly as cosine
    similarity does, which is the ranking this evaluation has always
    reported; it previously built a raw inner-product index over the same
    normalised vectors.

    Args:
        model_name: sentence-transformers model identifier.
        model_revision: the 40-character Hub commit to load. Defaults to the
            pin for the default model; another model needs its own.
        embedding_model: an embedder to reuse. Under ``strict`` it must
            itself be strict and not already degraded.
        strict: when True, refuse to construct on a hashed embedder or a
            numpy-scan index instead of measuring a hit rate against them.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_revision: str | None = None,
        embedding_model: EmbeddingModel | None = None,
        strict: bool = False,
    ) -> None:
        self._init_degradation(strict)
        require_strict_component(
            type(self).__name__, embedding_model, "embedding model", strict
        )
        self.embedding_model = embedding_model or EmbeddingModel(
            model_name, model_revision, strict=strict
        )
        self.model_name = str(getattr(self.embedding_model, "model_name", model_name))
        #: The identity a reader of the numbers needs: a name says which model
        #: was asked for, the commit says which weights answered.
        self.model_revision = getattr(self.embedding_model, "model_revision", None)
        self.index = FaissIndex(strict=strict)
        self.documents: list[CorpusDocument] = []

        if getattr(self.embedding_model, "degraded", False):
            self._degrade(
                "its embedding model is degraded, so this is not dense "
                "retrieval: "
                f"{getattr(self.embedding_model, 'degraded_reason', 'unknown')}"
            )
        if self.index.degraded:
            self._degrade(f"its vector index is degraded: {self.index.degraded_reason}")

    def build_index(
        self, documents: Sequence[CorpusDocument], batch_size: int = 64
    ) -> None:
        """Embed and index the corpus, keyed by position."""
        self.documents = list(documents)
        vectors = self._embed(
            [document.text for document in self.documents], batch_size
        )
        # The index is keyed by row position, not by the corpus's own ids:
        # a corpus may repeat an id, and a hit has to map back to exactly the
        # row that was embedded.
        self.index.build_index(
            [
                {"doc_id": str(position), "text": document.text, "embedding": vector}
                for position, (document, vector) in enumerate(
                    zip(self.documents, vectors)
                )
            ]
        )

    def search(
        self, queries: Sequence[str], top_k: int = 10, batch_size: int = 64
    ) -> list[list[int]]:
        """Return, per query, the corpus positions of its top-k documents."""
        if top_k < 1:
            # The index clamps k to one, so a caller asking for zero would
            # get one document back and report it under the wrong k.
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        if not self.documents:
            raise RuntimeError("Index has not been built. Call build_index first.")
        vectors = self._embed(list(queries), batch_size)
        return [
            [int(hit["doc_id"]) for hit in self.index.search(vector, top_k)]
            for vector in vectors
        ]

    def _embed(self, texts: list[str], batch_size: int) -> np.ndarray:
        vectors = np.asarray(
            self.embedding_model.embed_texts(texts, batch_size=batch_size),
            dtype=np.float32,
        )
        if vectors.size == 0:
            return vectors
        return _l2_normalize(vectors)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_string_list(value: Any) -> set[str]:
    hints: set[str] = set()
    if value is None:
        return hints
    if isinstance(value, str):
        v = value.strip()
        if v:
            hints.add(v.lower())
        return hints
    if isinstance(value, list):
        for item in value:
            hints.update(_extract_string_list(item))
        return hints
    if isinstance(value, dict):
        for v in value.values():
            hints.update(_extract_string_list(v))
        return hints
    return hints


def _extract_hotpot_relevance(item: dict[str, Any]) -> set[str]:
    # Primary signal: supporting_facts => list[[title, sent_id], ...]
    hints: set[str] = set()
    for sf in item.get("supporting_facts") or []:
        if isinstance(sf, list) and sf:
            title = str(sf[0]).strip().lower()
            if title:
                hints.add(title)

    # Additional fallback signals when available.
    hints.update(_extract_string_list(item.get("answer")))
    return hints


def load_hotpot_queries(
    path: Path, max_queries: int | None = None, strict: bool = False
) -> list[QueryExample]:
    """Read HotpotQA questions through the registry loader, one record at a time.

    The loader carries the degradation contract, so a malformed context is
    declared rather than silently thinned. Records are normalised one by one
    up to the cap: a record after the cap is not evaluated, so a fault in it
    must not refuse the run. Loading the whole file first did exactly that.
    """
    records = _read_json(path)
    if not isinstance(records, list):
        raise ValueError("HotpotQA file should be a JSON list.")
    dataset = HotpotDataset(strict=strict)

    queries: list[QueryExample] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        item = dataset.normalize_record(index, record)
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        queries.append(
            QueryExample(
                query_id=str(item["id"]),
                query_text=question,
                relevant_hints=_extract_hotpot_relevance(item),
            )
        )
        if max_queries is not None and len(queries) >= max_queries:
            break

    return queries


def load_locomo_queries(
    path: Path, max_queries: int | None = None
) -> list[QueryExample]:
    if path.suffix.lower() == ".jsonl":
        data: Iterable[Any] = _read_jsonl(path)
    else:
        payload = _read_json(path)
        data = payload if isinstance(payload, list) else [payload]

    queries: list[QueryExample] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        query_text = ""
        for field in ("query", "question", "prompt"):
            value = item.get(field)
            if isinstance(value, str) and value.strip():
                query_text = value.strip()
                break
        if not query_text:
            continue

        # Flexible relevance extraction due to varying LoCoMo variants.
        hints: set[str] = set()
        for field in (
            "relevant_docs",
            "relevant_doc_ids",
            "evidence",
            "gold_passages",
            "gold_docs",
            "answers",
            "answer",
        ):
            hints.update(_extract_string_list(item.get(field)))

        qid = str(item.get("id", item.get("qid", f"locomo-{i}")))
        queries.append(
            QueryExample(query_id=qid, query_text=query_text, relevant_hints=hints)
        )
        if max_queries is not None and len(queries) >= max_queries:
            break

    return queries


def _is_relevant(document: CorpusDocument, hints: set[str]) -> bool:
    if not hints:
        return False
    if document.doc_id.lower() in hints:
        return True
    title_norm = document.title_norm
    return any(hint and hint in title_norm for hint in hints)


def evaluate_hit_rate_at_k(
    queries: Sequence[QueryExample],
    ranked_positions: Sequence[Sequence[int]],
    documents: Sequence[CorpusDocument],
) -> float:
    """Fraction of queries with at least one relevant document retrieved.

    A hit rate, not Recall@k. Recall@k is the share of a query's relevant
    documents the top k holds; this reaches 1.0 on the first one found. The
    benchmark's ``_retrieval_recall`` computes the other, and for years both
    were printed under the word "Recall".
    """
    if len(queries) == 0:
        return 0.0
    if len(ranked_positions) != len(queries):
        raise ValueError(
            f"{len(ranked_positions)} result lists for {len(queries)} queries"
        )

    hit_count = 0
    for query, positions in zip(queries, ranked_positions):
        if any(
            _is_relevant(documents[position], query.relevant_hints)
            for position in positions
            if 0 <= position < len(documents)
        ):
            hit_count += 1
    return hit_count / len(queries)


def run_pipeline(args: argparse.Namespace, strict: bool = True) -> dict[str, float]:
    """Index the corpus, run both query sets, print and return the hit rate."""
    corpus = MSMARCOCorpus(args.msmarco_path, max_docs=args.max_docs, strict=strict)
    retriever = DenseRetriever(args.model_name, args.model_revision, strict=strict)
    retriever.build_index(corpus.documents, batch_size=args.batch_size)

    query_sets = {
        "HotpotQA": load_hotpot_queries(
            args.hotpot_path, max_queries=args.max_hotpot, strict=strict
        ),
        "LoCoMo": load_locomo_queries(args.locomo_path, max_queries=args.max_locomo),
    }
    if not any(query_sets.values()):
        raise ValueError("No valid queries loaded from HotpotQA or LoCoMo.")

    hit_rate: dict[str, float] = {}
    for name, queries in query_sets.items():
        if not queries:
            continue
        ranked = retriever.search(
            [query.query_text for query in queries],
            top_k=args.top_k,
            batch_size=args.batch_size,
        )
        hit_rate[name] = evaluate_hit_rate_at_k(queries, ranked, corpus.documents)

    print("=== Retrieval Evaluation Summary ===")
    print(f"MS MARCO docs indexed: {len(corpus.documents)}")
    print(f"Embedding model: {retriever.model_name}@{retriever.model_revision}")
    print(f"top_k: {args.top_k}")
    for name, queries in query_sets.items():
        print(f"{name} queries: {len(queries)}")
        if name in hit_rate:
            print(f"{name} hit rate@{args.top_k}: {hit_rate[name]:.4f}")
        else:
            # Not 0.0: a dataset with no queries has no hit rate, and printing
            # a zero for it would read as a measured miss.
            print(f"{name} hit rate@{args.top_k}: not measured, no queries loaded")
    return hit_rate


def _positive_int(value: str) -> int:
    """argparse type for a count that must be at least one.

    ``--top-k 0`` used to be accepted: the index clamps k to one, so one
    document was retrieved and the summary was labelled hit rate@0. A cap of
    zero or less is the same kind of misstatement about what ran.
    """
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MS MARCO + HotpotQA + LoCoMo retrieval evaluator"
    )
    parser.add_argument(
        "--msmarco-path",
        type=Path,
        required=True,
        help="Path to the MS MARCO full-document TSV (id, url, title, body)",
    )
    parser.add_argument(
        "--hotpot-path",
        type=Path,
        required=True,
        help="Path to HotpotQA distractor train JSON",
    )
    parser.add_argument(
        "--locomo-path", type=Path, required=True, help="Path to LoCoMo JSON/JSONL"
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--model-revision",
        type=str,
        default=None,
        help=(
            "the 40-character Hub commit to load. Defaults to the pin for "
            "the default model; any other model needs its own, because a "
            "name alone does not say which weights will arrive."
        ),
    )
    parser.add_argument("--top-k", type=_positive_int, default=10)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument(
        "--max-docs",
        type=_positive_int,
        default=None,
        help="Optional cap for indexed MS MARCO docs",
    )
    parser.add_argument(
        "--max-hotpot",
        type=_positive_int,
        default=None,
        help="Optional cap for HotpotQA queries",
    )
    parser.add_argument(
        "--max-locomo",
        type=_positive_int,
        default=None,
        help="Optional cap for LoCoMo queries",
    )
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help=(
            "Permit components to run degraded. Off by default: a hit rate "
            "from a hashed embedder or a numpy scan is not a measurement of "
            "dense retrieval."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(environment_report().render(), flush=True)
    run_pipeline(args, strict=not args.allow_degraded)
    print(degradation_summary(), flush=True)


if __name__ == "__main__":
    main()
