"""Retrieval and evaluation pipeline for MS MARCO, HotpotQA, and LoCoMo.

This script:
1. Loads MS MARCO fulldocs TSV as the retrieval corpus.
2. Builds sentence-transformer embeddings.
3. Indexes embeddings with FAISS.
4. Loads query sets from HotpotQA and LoCoMo.
5. Retrieves top-k MS MARCO documents for each query.
6. Evaluates retrieval with Recall@k.

The implementation is designed to be robust to slight schema differences
in dataset files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class QueryExample:
    """Query and a set of relevant document title/document-id hints."""

    query_id: str
    query_text: str
    relevant_hints: set[str]


class MSMARCOCorpus:
    """Loads MS MARCO full documents and exposes metadata needed for retrieval."""

    def __init__(self, tsv_path: Path, max_docs: int | None = None) -> None:
        self.tsv_path = tsv_path
        self.max_docs = max_docs
        self.df = self._load()

    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(
            self.tsv_path, sep="\t", header=None, dtype=str, keep_default_na=False
        )

        # Common MS MARCO full-doc layout has 4 columns: doc_id, url, title, body.
        # Fall back gracefully if column count differs.
        if df.shape[1] >= 4:
            df = df.iloc[:, :4].copy()
            df.columns = ["doc_id", "url", "title", "body"]
        elif df.shape[1] == 3:
            df = df.copy()
            df.columns = ["doc_id", "title", "body"]
            df["url"] = ""
        else:
            raise ValueError(
                "Unsupported MS MARCO TSV format. Expected >=3 columns like: "
                "doc_id, url, title, body"
            )

        if self.max_docs is not None:
            df = df.iloc[: self.max_docs].copy()

        df["doc_id"] = df["doc_id"].astype(str)
        df["title"] = df["title"].fillna("")
        df["body"] = df["body"].fillna("")
        df["text"] = (
            df["title"].str.strip() + "\n" + df["body"].str.strip()
        ).str.strip()

        # Normalize title for fuzzy title-based relevance matching.
        df["title_norm"] = df["title"].str.lower().str.strip()
        return df.reset_index(drop=True)

    @property
    def texts(self) -> list[str]:
        return self.df["text"].tolist()

    @property
    def doc_ids(self) -> list[str]:
        return self.df["doc_id"].tolist()


class DenseRetriever:
    """Sentence-transformer embedder + FAISS inner-product index."""

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.doc_embeddings: np.ndarray | None = None

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return x / norms

    def build_index(self, docs: list[str], batch_size: int = 64) -> None:
        embeddings = self.model.encode(
            docs,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype("float32")

        embeddings = self._l2_normalize(embeddings)
        self.doc_embeddings = embeddings

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.index = index

    def search(
        self, queries: list[str], top_k: int = 10, batch_size: int = 64
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index has not been built. Call build_index first.")

        q_emb = self.model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype("float32")
        q_emb = self._l2_normalize(q_emb)
        scores, indices = self.index.search(q_emb, top_k)
        return scores, indices


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
    for sf in item.get("supporting_facts", []):
        if isinstance(sf, list) and sf:
            title = str(sf[0]).strip().lower()
            if title:
                hints.add(title)

    # Additional fallback signals when available.
    hints.update(_extract_string_list(item.get("answer")))
    return hints


def load_hotpot_queries(
    path: Path, max_queries: int | None = None
) -> list[QueryExample]:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError("HotpotQA file should be a JSON list.")

    queries: list[QueryExample] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        hints = _extract_hotpot_relevance(item)
        qid = str(item.get("_id", f"hotpot-{i}"))
        queries.append(
            QueryExample(query_id=qid, query_text=question, relevant_hints=hints)
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


def _is_relevant(doc_id: str, title_norm: str, hints: set[str]) -> bool:
    if not hints:
        return False
    if doc_id.lower() in hints:
        return True
    for hint in hints:
        if hint and hint in title_norm:
            return True
    return False


def evaluate_recall_at_k(
    queries: list[QueryExample],
    topk_indices: np.ndarray,
    corpus_df: pd.DataFrame,
) -> float:
    if len(queries) == 0:
        return 0.0

    hit_count = 0
    for q_idx, query in enumerate(queries):
        found = False
        for doc_row_idx in topk_indices[q_idx]:
            if doc_row_idx < 0:
                continue
            row = corpus_df.iloc[int(doc_row_idx)]
            if _is_relevant(
                str(row["doc_id"]), str(row["title_norm"]), query.relevant_hints
            ):
                found = True
                break
        if found:
            hit_count += 1

    return hit_count / len(queries)


def run_pipeline(args: argparse.Namespace) -> None:
    corpus = MSMARCOCorpus(args.msmarco_path, max_docs=args.max_docs)
    retriever = DenseRetriever(args.model_name)
    retriever.build_index(corpus.texts, batch_size=args.batch_size)

    hotpot_queries = load_hotpot_queries(args.hotpot_path, max_queries=args.max_hotpot)
    locomo_queries = load_locomo_queries(args.locomo_path, max_queries=args.max_locomo)

    if not hotpot_queries and not locomo_queries:
        raise ValueError("No valid queries loaded from HotpotQA or LoCoMo.")

    if hotpot_queries:
        _, hotpot_idx = retriever.search(
            [q.query_text for q in hotpot_queries],
            top_k=args.top_k,
            batch_size=args.batch_size,
        )
        hotpot_recall = evaluate_recall_at_k(hotpot_queries, hotpot_idx, corpus.df)
    else:
        hotpot_recall = 0.0

    if locomo_queries:
        _, locomo_idx = retriever.search(
            [q.query_text for q in locomo_queries],
            top_k=args.top_k,
            batch_size=args.batch_size,
        )
        locomo_recall = evaluate_recall_at_k(locomo_queries, locomo_idx, corpus.df)
    else:
        locomo_recall = 0.0

    print("=== Retrieval Evaluation Summary ===")
    print(f"MS MARCO docs indexed: {len(corpus.df)}")
    print(f"Embedding model: {args.model_name}")
    print(f"top_k: {args.top_k}")
    print(f"HotpotQA queries: {len(hotpot_queries)}")
    print(f"LoCoMo queries: {len(locomo_queries)}")
    print(f"HotpotQA Recall@{args.top_k}: {hotpot_recall:.4f}")
    print(f"LoCoMo Recall@{args.top_k}: {locomo_recall:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MS MARCO + HotpotQA + LoCoMo retrieval evaluator"
    )
    parser.add_argument(
        "--msmarco-path", type=Path, required=True, help="Path to MS MARCO fulldocs.tsv"
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
    parser.add_argument(
        "--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional cap for indexed MS MARCO docs",
    )
    parser.add_argument(
        "--max-hotpot", type=int, default=None, help="Optional cap for HotpotQA queries"
    )
    parser.add_argument(
        "--max-locomo", type=int, default=None, help="Optional cap for LoCoMo queries"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
