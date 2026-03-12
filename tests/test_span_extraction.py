"""Tests for span-level evidence extraction and insertion."""

from cke.extractor.llm_extractor import LLMConfig, LLMExtractor
from cke.graph.assertion import Assertion, Evidence
from cke.graph.deduplicator import AssertionDeduplicator
from cke.graph_engine.graph_engine import KnowledgeGraphEngine


def test_llm_extractor_validates_span_offsets() -> None:
    text = "Redis uses RESP for protocol messaging."
    extractor = LLMExtractor(config=LLMConfig(api_key="dummy"))
    start = text.index("Redis uses RESP")
    end = start + len("Redis uses RESP")

    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        "["
                        '{"subject":"Redis","relation":"uses","object":"RESP",'
                        '"confidence":0.9,'
                        '"evidence":[{"chunk_id":"chunk-1","span_start":'
                        f'{start},"span_end":{end},'
                        '"text":"Redis uses RESP",'
                        '"extractor_confidence":0.93,"source_weight":0.8}]}'
                        "]"
                    )
                }
            }
        ]
    }

    statements = extractor._parse_response(payload, source_text=text)
    assert len(statements) == 1
    evidence = statements[0].context["evidence"]
    assert len(evidence) == 1
    assert evidence[0]["span_start"] == start


def test_llm_extractor_discards_assertion_on_mismatched_span() -> None:
    text = "Redis uses RESP for protocol messaging."
    extractor = LLMExtractor(config=LLMConfig(api_key="dummy"))
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        "["
                        '{"subject":"Redis","relation":"uses","object":"RESP",'
                        '"confidence":0.9,'
                        '"evidence":[{"chunk_id":"chunk-1","span_start":0,'
                        '"span_end":5,"text":"Wrong",'
                        '"extractor_confidence":0.7,"source_weight":0.6}]}'
                        "]"
                    )
                }
            }
        ]
    }

    assert extractor._parse_response(payload, source_text=text) == []


def test_deduplication_uses_span_text_and_graph_stores_evidence() -> None:
    deduplicator = AssertionDeduplicator()
    evidence_a = Evidence(chunk_id="c1", span=(0, 4), text="Redis", source_weight=0.8)
    evidence_b = Evidence(chunk_id="c1", span=(0, 4), text="RESP", source_weight=0.8)

    a1 = Assertion("Redis", "uses", "RESP", evidence=[evidence_a])
    a2 = Assertion("Redis", "uses", "RESP", evidence=[evidence_b])
    merged = deduplicator.deduplicate([a1, a2])
    assert len(merged) == 2

    graph = KnowledgeGraphEngine()
    graph.add_assertion(
        "Redis",
        "uses",
        "RESP",
        evidence=[{"chunk_id": "c1", "span_start": 0, "span_end": 5, "text": "Redis"}],
    )
    neighbors = graph.get_neighbors("Redis")
    assert neighbors[0].context["evidence"][0]["text"] == "Redis"
