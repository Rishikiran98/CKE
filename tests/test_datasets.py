"""Tests for dataset ingestion loaders and normalization schema."""

from __future__ import annotations

import json

from cke.datasets.registry import load_dataset

REQUIRED_KEYS = {
    "id",
    "question",
    "answer",
    "documents",
    "supporting_facts",
    "metadata",
}


def _assert_normalized_item(item: dict) -> None:
    assert REQUIRED_KEYS.issubset(set(item.keys()))
    assert isinstance(item["id"], str)
    assert isinstance(item["documents"], list)
    for doc in item["documents"]:
        assert {"doc_id", "title", "text"}.issubset(set(doc.keys()))
        assert isinstance(doc["doc_id"], str)
        assert isinstance(doc["text"], str)


def test_hotpotqa_loader(tmp_path):
    data = [
        {
            "_id": "hp1",
            "question": "Who won?",
            "answer": "Team A",
            "supporting_facts": [["Doc One", 0]],
            "context": [
                ["Doc One", [" Team A won ", "in 2020. "]],
                ["Doc Two", ["Some", "other info"]],
            ],
            "type": "bridge",
            "level": "easy",
        }
    ]
    path = tmp_path / "hotpot.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    dataset = load_dataset("hotpotqa", str(path))
    assert len(dataset) == 1

    item = dataset.get_item(0)
    _assert_normalized_item(item)
    assert item["question"] == "Who won?"
    assert item["answer"] == "Team A"
    assert item["documents"][0]["title"] == "Doc One"
    assert item["documents"][0]["text"] == "Team A won in 2020."


def test_msmarco_loader(tmp_path):
    path = tmp_path / "msmarco.tsv"
    path.write_text("d1\tA test document.\nd2\tAnother doc.\n", encoding="utf-8")

    dataset = load_dataset("msmarco", str(path))
    assert len(dataset) == 2
    sample = dataset.sample(1)
    assert len(sample) == 1

    item = dataset.get_item(1)
    _assert_normalized_item(item)
    assert item["question"] is None
    assert item["answer"] is None
    assert item["documents"][0]["doc_id"] == "d2"


def test_locomo_loader(tmp_path):
    data = [
        {
            "conversation_id": "c1",
            "turns": [
                {"speaker": "user", "text": "Hello there"},
                {"speaker": "assistant", "text": "Hi!"},
            ],
            "question": "What did the user say?",
            "answer": "Hello there",
        }
    ]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    dataset = load_dataset("locomo", str(path))
    assert len(dataset) == 1

    item = dataset.get_item(0)
    _assert_normalized_item(item)
    assert item["documents"][0]["doc_id"] == "c1_turn_0"
    assert item["documents"][0]["title"] == "conversation"
    assert item["metadata"]["turns"][1]["speaker"] == "assistant"


def test_wiki2_loader(tmp_path):
    data = [
        {
            "_id": "w1",
            "question": "When were both films released?",
            "answer": "2005",
            "context": [
                ["Film A", ["Film A was released ", "in 2005. "]],
                ["Film B", ["Film B came out in 2005."]],
            ],
            "supporting_facts": [["Film A", 1], ["Film B", 0]],
            "type": "comparison",
            "evidences": [["Film A", "Film B"]],
        }
    ]
    path = tmp_path / "wiki2.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    dataset = load_dataset("wiki2", str(path))
    assert len(dataset) == 1

    item = dataset.get_item(0)
    _assert_normalized_item(item)
    assert item["id"] == "w1"
    assert item["question"] == "When were both films released?"
    assert item["answer"] == "2005"
    assert len(item["documents"]) == 2
    assert item["documents"][0]["doc_id"] == "Film A_0"
    assert item["documents"][0]["title"] == "Film A"
    assert "Film A was released" in item["documents"][0]["text"]
    assert item["documents"][1]["doc_id"] == "Film B_1"
    assert item["supporting_facts"] == [["Film A", 1], ["Film B", 0]]
    assert item["metadata"]["type"] == "comparison"


def test_wiki2_loader_via_alias(tmp_path):
    data = [
        {
            "_id": "w2",
            "question": "Q?",
            "answer": "A",
            "context": [["Title", "Plain text body"]],
            "supporting_facts": [],
        }
    ]
    path = tmp_path / "wiki2.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    dataset = load_dataset("2wikimultihopqa", str(path))
    assert len(dataset) == 1

    item = dataset.get_item(0)
    _assert_normalized_item(item)
    assert item["id"] == "w2"
    assert item["documents"][0]["text"] == "Plain text body"
