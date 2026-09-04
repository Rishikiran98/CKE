"""Did the context hold the documents the dataset says the answer needs?

Every loader emits supporting facts and the benchmark read none of them, so
both arms were scored on their answers alone. An arm that answered correctly
from the wrong documents and an arm that answered wrongly while holding the
right ones produced the same EM and F1.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from cke.evaluation.token_counter import TokenCounter

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "run_cke_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_cke_benchmark_recall", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_module()


def _docs(*pairs):
    return [{"doc_id": doc_id, "title": title, "text": ""} for doc_id, title in pairs]


# ---------------------------------------------------------------------------
# Resolving what the dataset points at
# ---------------------------------------------------------------------------


def test_a_title_and_sentence_index_resolves_to_the_titled_document():
    """HotpotQA and 2WikiMultiHopQA: [title, sentence index]."""
    docs = _docs(("Ed Wood_4", "Ed Wood"), ("Scott Derrickson_1", "Scott Derrickson"))
    item = {"supporting_facts": [["Ed Wood", 0], ["Scott Derrickson", 3]]}

    assert bench.gold_documents(item, docs) == {"Ed Wood_4", "Scott Derrickson_1"}


def test_a_repeated_title_resolves_by_its_paragraph_index():
    """MuSiQue reuses one title across paragraphs of one item, which is why
    the doc_id carries the index. Matching on the title alone would count a
    different paragraph of the same page as a hit."""
    docs = _docs(("Green_10", "Green"), ("Green_5", "Green"))
    item = {"supporting_facts": [["Green", 5]]}

    assert bench.gold_documents(item, docs) == {"Green_5"}


def test_a_fact_that_names_the_document_itself_resolves():
    """LoCoMo evidence is a dia_id, which is the document's own id."""
    docs = _docs(("D1:3", "session_1"), ("D1:4", "session_1"))
    item = {"supporting_facts": [["D1:3", 1]]}

    assert bench.gold_documents(item, docs) == {"D1:3"}


def test_a_title_that_names_nothing_resolves_to_no_gold_set():
    """A gold set that quietly lost an entry makes recall a number about the
    resolution failure, not about retrieval."""
    docs = _docs(("Ed Wood_4", "Ed Wood"))
    item = {"supporting_facts": [["Ed Wood", 0], ["A page not in this item", 1]]}

    assert bench.gold_documents(item, docs) is None


def test_an_ambiguous_title_resolves_to_no_gold_set():
    docs = _docs(("Green_10", "Green"), ("Green_5", "Green"))
    item = {"supporting_facts": [["Green", 99]]}

    assert bench.gold_documents(item, docs) is None


def test_an_item_with_no_supporting_facts_has_no_gold_set():
    assert bench.gold_documents({}, _docs(("a_1", "a"))) is None


# ---------------------------------------------------------------------------
# Recall, and the difference between zero and unmeasured
# ---------------------------------------------------------------------------


def test_recall_counts_the_gold_documents_the_arm_retrieved():
    assert bench._retrieval_recall(["a", "b"], {"a", "c"}) == 0.5
    assert bench._retrieval_recall(["a", "c"], {"a", "c"}) == 1.0


def test_retrieving_none_of_them_is_a_measured_zero():
    assert bench._retrieval_recall(["x"], {"a"}) == 0.0


def test_an_arm_that_cannot_name_its_documents_reports_nothing():
    """Zero would read as "retrieved none of them" beside arms that did."""
    assert bench._retrieval_recall(None, {"a"}) is None


def test_an_unresolved_gold_set_reports_nothing():
    assert bench._retrieval_recall(["a"], None) is None


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------


def test_the_graph_arm_can_trace_a_statement_to_its_document():
    """The graph accepted a source all along and the driver dropped it, so a
    retrieved statement could not be traced to a document."""
    from cke.graph_engine.graph_engine import KnowledgeGraphEngine
    from cke.retrieval.graph_retriever import GraphRetriever
    from cke.router.query_plan import QueryPlan

    engine = KnowledgeGraphEngine()
    engine.add_statement("Hagia Sophia", "located_in", "Istanbul", source="doc_7")
    retriever = GraphRetriever(engine)
    plan = QueryPlan(
        query_text="Where is Hagia Sophia?",
        seed_entities=["Hagia Sophia"],
        intent="factoid",
        max_depth=2,
        max_results=5,
    )

    evidence = retriever.retrieve(plan, mode="bfs").get("evidence", [])

    assert evidence, "the seed statement must be retrievable"
    assert {e.get("source") for e in evidence} == {"doc_7"}


def test_the_hybrid_arm_reports_no_recall_rather_than_half_of_one():
    """Its dense half arrives as strings with the doc_ids already dropped, so
    a recall over its graph half alone would understate it."""
    row = bench._score_row(
        {
            "answer": "x",
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "latency_ms": 1.0,
            "retrieved_doc_ids": None,
        },
        "x",
        gold_docs={"a"},
    )

    assert row["retrieval_recall"] is None


# ---------------------------------------------------------------------------
# What the aggregates and tables do with it
# ---------------------------------------------------------------------------


def _arm(recall, completion=3):
    return {
        "rag_k10": {
            "em": 1.0,
            "f1": 1.0,
            "prompt_tokens": 100,
            "completion_tokens": completion,
            "latency_ms": 1.0,
            "answer_truncated": False,
            "retrieval_recall": recall,
        }
    }


def test_an_arm_with_no_measured_recall_reports_none():
    metrics = bench.aggregate_metrics([_arm(None), _arm(None)])["rag_k10"]

    assert "retrieval_recall" not in metrics
    assert "recall_measured_items" not in metrics


def test_recall_is_averaged_over_the_items_that_resolved():
    metrics = bench.aggregate_metrics([_arm(1.0), _arm(0.0), _arm(None)])["rag_k10"]

    assert metrics["retrieval_recall"] == 0.5
    assert metrics["recall_measured_items"] == 2, "the unresolved item is not a zero"


def test_completion_tokens_are_aggregated_when_every_item_reported_them():
    metrics = bench.aggregate_metrics([_arm(1.0, 4), _arm(1.0, 2)])["rag_k10"]

    assert metrics["median_completion_tokens"] == 3.0
    assert metrics["total_completion_tokens"] == 6


def test_completion_tokens_are_absent_when_an_arm_did_not_report_them():
    metrics = bench.aggregate_metrics([_arm(1.0, None)])["rag_k10"]

    assert "median_completion_tokens" not in metrics
    assert "total_completion_tokens" not in metrics


def test_the_table_carries_both_new_rows_when_they_were_measured():
    from cke.evaluation.span_qa import SpanExtractiveQA

    metrics = bench.aggregate_metrics([_arm(0.5)])
    table = bench.produce_comparison_table(
        {"hotpotqa": metrics}, metrics, TokenCounter(), SpanExtractiveQA()
    )

    assert "Recall of supporting docs" in table
    assert "Items recall measured on" in table
    assert "Median completion tokens" in table


def test_the_table_omits_the_recall_row_when_nothing_was_measured():
    from cke.evaluation.span_qa import SpanExtractiveQA

    metrics = bench.aggregate_metrics([_arm(None)])
    table = bench.produce_comparison_table(
        {"hotpotqa": metrics}, metrics, TokenCounter(), SpanExtractiveQA()
    )

    assert "Recall of supporting docs" not in table


# ---------------------------------------------------------------------------
# Through the pipelines, not around them
# ---------------------------------------------------------------------------


class _EchoAnswerer:
    """Answers with a fixed string, so the counted completion is known."""

    description = "a stub"
    uses_language_model = False

    def answer(self, question, context):
        return "Istanbul is the answer"


def test_the_graph_arm_names_the_documents_its_statements_came_from():
    """End to end: the pipeline records a source per statement, the graph
    keeps it, and the arm reports the documents behind its context."""
    counter = TokenCounter()
    pipeline = bench.CKELitePipeline(
        token_counter=counter, answerer=_EchoAnswerer(), strict=False
    )
    docs = [
        {
            "doc_id": "doc_a",
            "title": "Hagia Sophia",
            "text": "Hagia Sophia is located in Istanbul.",
        },
        {
            "doc_id": "doc_b",
            "title": "Istanbul",
            "text": "Istanbul is located in Turkey.",
        },
    ]

    result = pipeline.run_item("Where is Hagia Sophia?", docs, n=12)

    assert result["retrieved_doc_ids"], "the arm must name its documents"
    assert set(result["retrieved_doc_ids"]) <= {"doc_a", "doc_b"}
    assert "doc_a" in result["retrieved_doc_ids"]


def test_the_answer_is_counted_by_the_same_counter_as_the_prompt():
    """A completion figure counted another way is not comparable to the
    prompt figure printed beside it."""
    counter = TokenCounter()
    answerer = _EchoAnswerer()
    pipeline = bench.RAGPipeline(token_counter=counter, answerer=answerer, strict=False)
    docs = [{"doc_id": "doc_a", "title": "a", "text": "Istanbul is in Turkey."}]

    result = pipeline.run_item("Where is Istanbul?", docs, k=1)

    expected = counter.count(answerer.answer("", ""))
    assert result["completion_tokens"] == expected
    assert expected != len(
        answerer.answer("", "").split()
    ), "the fixture must distinguish a token count from a word count"


def test_the_scored_row_keeps_the_completion_count():
    row = bench._score_row(
        {
            "answer": "x",
            "prompt_tokens": 10,
            "completion_tokens": 7,
            "latency_ms": 1.0,
            "retrieved_doc_ids": ["a"],
        },
        "x",
        gold_docs={"a"},
    )

    assert row["completion_tokens"] == 7
    assert row["retrieval_recall"] == 1.0


def test_an_arm_that_reported_no_figure_says_so_in_the_table():
    """The hybrid column printed "nan" for recall, which reads as a broken
    number rather than as a figure the arm never produced."""
    from cke.evaluation.span_qa import SpanExtractiveQA

    rows = [
        {
            "rag_k10": {
                "em": 1.0,
                "f1": 1.0,
                "prompt_tokens": 100,
                "completion_tokens": 3,
                "latency_ms": 1.0,
                "answer_truncated": False,
                "retrieval_recall": 0.5,
            },
            "hybrid_n12": {
                "em": 1.0,
                "f1": 1.0,
                "prompt_tokens": 100,
                "completion_tokens": 3,
                "latency_ms": 1.0,
                "answer_truncated": False,
                "retrieval_recall": None,
            },
        }
    ]
    metrics = bench.aggregate_metrics(rows)
    table = bench.produce_comparison_table(
        {"hotpotqa": metrics}, metrics, TokenCounter(), SpanExtractiveQA()
    )

    assert "nan" not in table
    recall_rows = [
        line for line in table.splitlines() if "Recall of supporting docs" in line
    ]
    assert recall_rows
    for line in recall_rows:
        assert "not measured" in line


def test_an_arm_that_never_ran_is_not_confused_with_one_that_measured_nothing():
    from cke.evaluation.span_qa import SpanExtractiveQA

    metrics = bench.aggregate_metrics([_arm(None)])  # only rag_k10 ran
    table = bench.produce_comparison_table(
        {"hotpotqa": metrics}, metrics, TokenCounter(), SpanExtractiveQA()
    )

    em_row = next(line for line in table.splitlines() if line.startswith("| Answer EM"))
    assert "not run" in em_row, "the five arms this run skipped never executed"
    assert "not measured" not in em_row, "EM was measured for the arm that ran"
