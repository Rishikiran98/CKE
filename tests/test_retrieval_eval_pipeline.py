"""The retrieval evaluation pipeline honours the degradation contract.

It used to hard-import faiss, pandas and sentence-transformers and build a
``SentenceTransformer`` directly, so it either ran on real components or did
not import at all, and could never say which. Now it composes the contract
components and inherits their state. Nothing here needs a model: the bare
environment is simulated the way the other contract tests do it, and the
retrieval path runs on a stub embedder that returns fixed vectors.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cke.diagnostics import DegradedComponentError
from cke.experiments import retrieval_eval_pipeline as pipeline
from cke.experiments.retrieval_eval_pipeline import (
    CorpusDocument,
    DenseRetriever,
    MSMARCOCorpus,
    QueryExample,
    evaluate_hit_rate_at_k,
    load_hotpot_queries,
)


@pytest.fixture
def bare(monkeypatch):
    """An environment without sentence-transformers or faiss."""
    from cke.retrieval import embedding_model as em
    from cke.retrieval import faiss_index as fi

    monkeypatch.setattr(em, "SentenceTransformer", None)
    monkeypatch.setattr(em, "_GLOBAL_MODEL_CACHE", {})
    monkeypatch.setattr(em, "_FAILED_MODEL_LOADS", {})
    monkeypatch.setattr(fi, "faiss", None)


class _FixedEmbedder:
    """An embedder whose vectors are read off a table, with the contract flags.

    Vectors are deliberately not unit length: the retriever has to normalise
    them itself, and a test below depends on that.
    """

    strict = True
    degraded = False
    degraded_reason = ""
    model_name = "fixed-table"

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table

    def embed_texts(self, texts, batch_size=32):
        return np.asarray([self.table[text] for text in texts], dtype=np.float32)

    def embed_text(self, text):
        return self.embed_texts([text])[0]


def _corpus(*rows: tuple[str, str, str]) -> list[CorpusDocument]:
    return [
        CorpusDocument(doc_id=doc_id, title=title, text=text)
        for doc_id, title, text in rows
    ]


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_the_module_imports_without_the_optional_dependencies(bare):
    """Importing it used to require faiss, pandas and sentence-transformers."""
    import importlib

    module = importlib.reload(pipeline)
    assert hasattr(module, "DenseRetriever")


def test_strict_retriever_refuses_a_bare_environment(bare):
    with pytest.raises(DegradedComponentError, match="sentence-transformers"):
        DenseRetriever(strict=True)


def test_non_strict_retriever_declares_both_degradations(bare, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        retriever = DenseRetriever(strict=False)

    assert retriever.degraded is True
    assert "sentence-transformers" in retriever.degraded_reason
    assert "faiss" in retriever.degraded_reason
    assert any("DenseRetriever" in record.message for record in caplog.records)


def test_strict_retriever_refuses_a_non_strict_injected_embedder():
    class _NotStrict(_FixedEmbedder):
        strict = False

    with pytest.raises(DegradedComponentError, match="does not declare itself strict"):
        DenseRetriever(embedding_model=_NotStrict({}), strict=True)


def test_strict_retriever_refuses_a_degraded_injected_embedder():
    class _Degraded(_FixedEmbedder):
        degraded = True
        degraded_reason = "hashed vectors"

    with pytest.raises(DegradedComponentError, match="already degraded"):
        DenseRetriever(embedding_model=_Degraded({}), strict=True)


def test_the_command_line_is_strict_by_default(bare, tmp_path):
    """No flag, no model: it must stop, not report a recall figure."""
    paths = _write_inputs(tmp_path)
    with pytest.raises(DegradedComponentError, match="sentence-transformers"):
        pipeline.main(paths)


def test_allow_degraded_runs_and_says_so(bare, tmp_path, capsys):
    paths = _write_inputs(tmp_path)
    pipeline.main(paths + ["--allow-degraded"])

    out = capsys.readouterr().out
    assert "hit rate@10" in out
    assert "DenseRetriever" in out
    assert "sentence-transformers" in out


# ---------------------------------------------------------------------------
# Retrieval and the metric
# ---------------------------------------------------------------------------


def test_search_ranks_by_cosine_not_by_vector_length():
    """The old index was inner product over normalised vectors; keep that.

    Under plain L2 on raw vectors, a long vector pointing the same way as the
    query is far from it, and a short vector pointing elsewhere is near. Only
    after normalisation does direction alone decide, which is the cosine
    ranking this evaluation reported before.
    """
    embedder = _FixedEmbedder(
        {
            "same way, long": [10.0, 0.0],
            "elsewhere, short": [0.5, 0.5],
            "q": [1.0, 0.0],
        }
    )
    retriever = DenseRetriever(embedding_model=embedder, strict=True)
    retriever.build_index(
        _corpus(("a", "Short", "elsewhere, short"), ("b", "Long", "same way, long"))
    )

    assert retriever.search(["q"], top_k=2) == [[1, 0]]


def test_search_returns_corpus_positions_even_when_ids_repeat():
    embedder = _FixedEmbedder({"one": [1.0, 0.0], "two": [0.0, 1.0], "q": [0.0, 1.0]})
    retriever = DenseRetriever(embedding_model=embedder, strict=True)
    retriever.build_index(_corpus(("same", "First", "one"), ("same", "Second", "two")))

    assert retriever.search(["q"], top_k=1) == [[1]]


def test_search_refuses_a_non_positive_top_k():
    """The index clamps k to one; asking for zero must not return a document."""
    embedder = _FixedEmbedder({"one": [1.0, 0.0], "q": [1.0, 0.0]})
    retriever = DenseRetriever(embedding_model=embedder, strict=True)
    retriever.build_index(_corpus(("a", "A", "one")))

    with pytest.raises(ValueError, match="top_k"):
        retriever.search(["q"], top_k=0)


def test_search_before_build_is_an_error():
    retriever = DenseRetriever(embedding_model=_FixedEmbedder({}), strict=True)
    with pytest.raises(RuntimeError, match="build_index"):
        retriever.search(["q"])


@pytest.mark.parametrize(
    "hints, ranked, expected",
    [
        ({"d2"}, [[1, 0]], 1.0),  # by document id
        ({"stanford"}, [[0]], 1.0),  # by title substring
        ({"stanford"}, [[1]], 0.0),  # relevant document not retrieved
        (set(), [[0, 1]], 0.0),  # no hints can never be a hit
    ],
    ids=["doc-id", "title-substring", "missed", "no-hints"],
)
def test_hit_rate_at_k(hints, ranked, expected):
    documents = _corpus(("d1", "Stanford University", "text"), ("d2", "Other", "text"))
    queries = [QueryExample("q1", "who?", hints)]

    assert evaluate_hit_rate_at_k(queries, ranked, documents) == expected


def test_the_hit_rate_is_averaged_over_queries():
    documents = _corpus(("d1", "A", "x"), ("d2", "B", "y"))
    queries = [QueryExample("q1", "?", {"d1"}), QueryExample("q2", "?", {"d2"})]

    assert evaluate_hit_rate_at_k(queries, [[0], [0]], documents) == 0.5


def test_the_hit_rate_is_not_the_benchmark_recall_they_were_both_called():
    """The two figures the word "Recall" used to cover, on one case.

    A query with two relevant documents, one of them retrieved. The hit rate
    is 1.0 — something relevant came back. The benchmark's recall is 0.5 —
    half the relevant documents came back. Both are right about what they
    measure, and reading either as the other is wrong by a factor that grows
    with the number of relevant documents an item has.

    This is here so the names cannot quietly converge again.
    """
    import importlib.util
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "run_cke_benchmark_hitrate", root / "scripts" / "run_cke_benchmark.py"
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    documents = _corpus(("d1", "A", "x"), ("d2", "B", "y"))
    queries = [QueryExample("q1", "?", {"d1", "d2"})]

    hit_rate = evaluate_hit_rate_at_k(queries, [[0]], documents)
    recall = bench._retrieval_recall(["d1"], {"d1", "d2"})

    assert hit_rate == 1.0
    assert recall == 0.5
    assert hit_rate != recall, "the same word covered both of these"


def test_the_hit_rate_refuses_mismatched_result_lists():
    with pytest.raises(ValueError, match="result lists"):
        evaluate_hit_rate_at_k([QueryExample("q", "?", set())], [], [])


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def test_corpus_reads_four_and_three_column_rows(tmp_path):
    path = tmp_path / "docs.tsv"
    path.write_text(
        "D1\thttp://x\tStanford University\tA body.\n"
        "D2\tOther Title\tAnother body.\n",
        encoding="utf-8",
    )

    corpus = MSMARCOCorpus(path, strict=True)

    assert corpus.doc_ids == ["D1", "D2"]
    assert corpus.documents[0].title == "Stanford University"
    assert corpus.documents[1].title == "Other Title"
    assert corpus.texts == [
        "Stanford University\nA body.",
        "Other Title\nAnother body.",
    ]
    assert corpus.degraded is False


def test_corpus_does_not_join_rows_across_a_quotation_mark(tmp_path):
    """MS MARCO bodies contain quotes that are text; pandas treated them as quoting."""
    path = tmp_path / "docs.tsv"
    path.write_text(
        'D1\thttp://x\tOpen "quote\tbody one\n' "D2\thttp://y\tSecond\tbody two\n",
        encoding="utf-8",
    )

    assert MSMARCOCorpus(path, strict=True).doc_ids == ["D1", "D2"]


def test_corpus_declares_short_rows(tmp_path):
    path = tmp_path / "docs.tsv"
    path.write_text("D1\thttp://x\tTitle\tbody\nD2\tonly two\n", encoding="utf-8")

    with pytest.raises(DegradedComponentError, match="1 rows"):
        MSMARCOCorpus(path, strict=True)

    corpus = MSMARCOCorpus(path, strict=False)
    assert corpus.degraded is True
    assert corpus.doc_ids == ["D1"]


def test_corpus_honours_max_docs_without_reading_the_rest(tmp_path):
    path = tmp_path / "docs.tsv"
    path.write_text(
        "D1\tu\tT1\tb\nD2\tu\tT2\tb\nD3\tbroken\n",
        encoding="utf-8",
    )

    corpus = MSMARCOCorpus(path, max_docs=2, strict=True)

    assert corpus.doc_ids == ["D1", "D2"]


def test_an_empty_corpus_is_an_error(tmp_path):
    path = tmp_path / "docs.tsv"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No documents"):
        MSMARCOCorpus(path, strict=True)


def test_hotpot_queries_carry_supporting_titles_and_answer_as_hints(tmp_path):
    path = tmp_path / "hotpot.json"
    path.write_text(
        json.dumps(
            [
                {
                    "_id": "hp1",
                    "question": "Who won?",
                    "answer": "Team A",
                    "supporting_facts": [["Doc One", 0]],
                    "context": [["Doc One", ["Team A won."]]],
                },
                {"_id": "hp2", "question": "   ", "context": []},
            ]
        ),
        encoding="utf-8",
    )

    queries = load_hotpot_queries(path, strict=True)

    assert [q.query_id for q in queries] == ["hp1"]
    assert queries[0].relevant_hints == {"doc one", "team a"}


def test_hotpot_queries_go_through_the_contract_loader(tmp_path):
    """A malformed context entry is declared, and stops a strict run."""
    path = tmp_path / "hotpot.json"
    path.write_text(
        json.dumps([{"_id": "hp1", "question": "Who?", "context": [["only title"]]}]),
        encoding="utf-8",
    )

    with pytest.raises(DegradedComponentError, match="context entries"):
        load_hotpot_queries(path, strict=True)
    assert len(load_hotpot_queries(path, strict=False)) == 1


def test_the_hotpot_cap_stops_before_records_it_does_not_evaluate(tmp_path):
    """A malformed context past the cap must not refuse a strict run.

    Loading the whole file through the registry loader normalised, and so
    declared, every record before the cap was applied.
    """
    path = tmp_path / "hotpot.json"
    path.write_text(
        json.dumps(
            [
                {"_id": "hp1", "question": "Who?", "context": [["Doc", ["text"]]]},
                {"_id": "hp2", "question": "What?", "context": [["only title"]]},
            ]
        ),
        encoding="utf-8",
    )

    queries = load_hotpot_queries(path, max_queries=1, strict=True)
    assert [q.query_id for q in queries] == ["hp1"]

    with pytest.raises(DegradedComponentError, match="context entries"):
        load_hotpot_queries(path, strict=True)


@pytest.mark.parametrize(
    "flag", ["--top-k", "--batch-size", "--max-docs", "--max-hotpot", "--max-locomo"]
)
@pytest.mark.parametrize("value", ["0", "-3"])
def test_the_parser_rejects_counts_below_one(flag, value, tmp_path, capsys):
    """``--top-k 0`` retrieved one document and labelled it Recall@0."""
    paths = _write_inputs(tmp_path)

    with pytest.raises(SystemExit):
        pipeline.parse_args(paths + [flag, value])
    assert "at least 1" in capsys.readouterr().err


def test_a_query_set_with_no_queries_is_not_reported_as_a_zero_hit_rate(
    bare, tmp_path, capsys
):
    paths = _write_inputs(tmp_path, locomo=[])
    pipeline.main(paths + ["--allow-degraded"])

    out = capsys.readouterr().out
    assert "LoCoMo hit rate@10: not measured" in out
    assert "LoCoMo hit rate@10: 0.0000" not in out


def _write_inputs(tmp_path, locomo=None) -> list[str]:
    """Minimal inputs for the plumbing. These test wiring, not retrieval."""
    docs = tmp_path / "docs.tsv"
    docs.write_text("D1\thttp://x\tStanford University\tA body.\n", encoding="utf-8")
    hotpot = tmp_path / "hotpot.json"
    hotpot.write_text(
        json.dumps(
            [
                {
                    "_id": "hp1",
                    "question": "Where is Stanford?",
                    "answer": "California",
                    "supporting_facts": [["Stanford University", 0]],
                    "context": [["Stanford University", ["It is in California."]]],
                }
            ]
        ),
        encoding="utf-8",
    )
    locomo_path = tmp_path / "locomo.json"
    locomo_path.write_text(
        json.dumps(
            [{"id": "l1", "question": "Where?", "evidence": ["D1"]}]
            if locomo is None
            else locomo
        ),
        encoding="utf-8",
    )
    return [
        "--msmarco-path",
        str(docs),
        "--hotpot-path",
        str(hotpot),
        "--locomo-path",
        str(locomo_path),
    ]
