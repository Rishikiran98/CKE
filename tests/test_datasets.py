"""Tests for dataset ingestion loaders and normalization schema."""

from __future__ import annotations

import json

import pytest

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


def _locomo_record() -> dict:
    """One conversation, shaped as the published file shapes them."""
    return {
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "Ann",
            "speaker_b": "Bo",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Ann", "dia_id": "D1:1", "text": " Hello there "},
                {"speaker": "Bo", "dia_id": "D1:2", "text": "Hi!"},
            ],
            "session_2_date_time": "9:00 am on 9 May, 2023",
            "session_2": [
                {"speaker": "Ann", "dia_id": "D2:1", "text": "I ran a race."}
            ],
            # Out of order in the mapping, and double-digit: the published
            # conversations run to session_19, where sorting the keys as text
            # puts session_10 before session_2.
            "session_10_date_time": "6:00 pm on 20 May, 2023",
            "session_10": [{"speaker": "Bo", "dia_id": "D10:1", "text": "Much later."}],
        },
        "qa": [
            {
                "question": "What did Ann say first?",
                "answer": "Hello there",
                "evidence": ["D1:1"],
                "category": 2,
            },
            {
                "question": "What did Ann realise after the race?",
                "evidence": ["D2:1"],
                "category": 5,
                "adversarial_answer": "self-care is important",
            },
        ],
    }


def test_locomo_loader_reads_the_published_shape(tmp_path):
    """Turns live under session_N in a mapping, not in a flat list."""
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps([_locomo_record()]), encoding="utf-8")

    dataset = load_dataset("locomo", str(path))

    # One item per question, not per conversation.
    assert len(dataset) == 2
    item = dataset.get_item(0)
    _assert_normalized_item(item)
    assert item["id"] == "conv-1::0"
    assert item["answer"] == "Hello there"
    # Every turn of every session, in session then turn order.
    assert [d["doc_id"] for d in item["documents"]] == [
        "D1:1",
        "D1:2",
        "D2:1",
        "D10:1",
    ]
    assert item["documents"][0]["text"] == "Ann: Hello there"
    assert item["documents"][2]["title"] == "session 2, 9:00 am on 9 May, 2023"
    # Evidence names turns by dia_id, and the session number comes with it.
    assert item["supporting_facts"] == [["D1:1", 1]]
    assert item["metadata"]["speakers"] == ["Ann", "Bo"]


def test_locomo_keeps_an_adversarial_answer_out_of_the_answer(tmp_path):
    """Scoring against it would reward inventing what the question catches."""
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps([_locomo_record()]), encoding="utf-8")

    item = load_dataset("locomo", str(path)).get_item(1)

    assert item["answer"] is None
    assert item["metadata"]["answerable"] is False
    assert item["metadata"]["adversarial_answer"] == "self-care is important"


def test_locomo_shares_one_document_list_across_a_conversation(tmp_path):
    """Two hundred questions per conversation; copying the turns each time
    would multiply four hundred documents by two hundred."""
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps([_locomo_record()]), encoding="utf-8")

    dataset = load_dataset("locomo", str(path))

    assert dataset.get_item(0)["documents"] is dataset.get_item(1)["documents"]


def test_locomo_declares_evidence_that_names_no_turn(tmp_path):
    """Nine entries in the published file are malformed ids.

    They are recorded as written rather than guessed at, and the loader says
    that recall against them cannot be measured.
    """
    from cke.diagnostics import DegradedComponentError

    record = _locomo_record()
    record["qa"][0]["evidence"] = ["D1:1", "D8:6; D9:17"]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps([record]), encoding="utf-8")

    from cke.datasets.locomo_loader import LoCoMoDataset

    with pytest.raises(DegradedComponentError, match="match no turn"):
        LoCoMoDataset(strict=True).load(str(path))

    loader = LoCoMoDataset(strict=False).load(str(path))
    assert loader.degraded is True
    # Recorded as written, not split or repaired.
    assert loader.items[0]["supporting_facts"] == [["D1:1", 1], ["D8:6; D9:17", None]]


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


def test_hotpot_normalize_record_matches_load(tmp_path):
    """load() is the per-record method applied in order, nothing more."""
    from cke.datasets.hotpot_loader import HotpotDataset

    records = [
        {
            "_id": "hp1",
            "question": " Who  won? ",
            "answer": "Team A",
            "supporting_facts": [["Doc One", 0]],
            "context": [["Doc One", [" Team A won ", "in 2020. "]]],
            "type": "bridge",
            "level": "easy",
        },
        {"question": "No id, no context"},
    ]
    path = tmp_path / "hotpot.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    loaded = HotpotDataset().load(str(path)).items
    one_by_one = [
        HotpotDataset().normalize_record(index, record)
        for index, record in enumerate(records)
    ]

    assert one_by_one == loaded
    assert one_by_one[1]["id"] == "hotpot_1"


def _musique_records() -> list[dict]:
    return [
        {
            "id": "2hop__460946_294723",
            "question": " Who is the spouse of the Green performer? ",
            "answer": "Miquette Giraudy",
            "answer_aliases": ["Miquette"],
            "answerable": True,
            "paragraphs": [
                {
                    "idx": 0,
                    "title": "Grant's First Stand",
                    "paragraph_text": "An album by Grant Green.",
                    "is_supporting": False,
                },
                {
                    "idx": 5,
                    "title": "Miquette Giraudy",
                    "paragraph_text": " She is a musician. ",
                    "is_supporting": True,
                },
                {
                    "idx": 10,
                    "title": "Green (Steve Hillage album)",
                    "paragraph_text": "Green is an album by Steve Hillage.",
                    "is_supporting": True,
                },
            ],
        },
        {"id": "3hop1__9", "question": "Q?", "answer": "A", "paragraphs": []},
    ]


def test_musique_loader_normalizes_paragraphs_and_supporting_labels(tmp_path):
    from cke.datasets.musique_loader import MuSiQueDataset

    path = tmp_path / "musique.json"
    path.write_text(json.dumps(_musique_records()), encoding="utf-8")

    items = MuSiQueDataset().load(str(path)).items

    first = items[0]
    assert first["id"] == "2hop__460946_294723"
    assert first["question"] == "Who is the spouse of the Green performer?"
    assert first["answer"] == "Miquette Giraudy"
    assert [d["doc_id"] for d in first["documents"]] == [
        "Grant's First Stand_0",
        "Miquette Giraudy_5",
        "Green (Steve Hillage album)_10",
    ]
    assert first["documents"][1]["text"] == "She is a musician."
    # Only the paragraphs the dataset marks, keeping each one's own index.
    assert first["supporting_facts"] == [
        ["Miquette Giraudy", 5],
        ["Green (Steve Hillage album)", 10],
    ]
    assert first["metadata"]["hops"] == "2hop"
    assert first["metadata"]["answer_aliases"] == ["Miquette"]


def test_musique_doc_ids_carry_the_index_because_titles_repeat(tmp_path):
    """Two paragraphs of one article share a title; a title-only id collides."""
    from cke.datasets.musique_loader import MuSiQueDataset

    path = tmp_path / "musique.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "2hop__1_2",
                    "question": "Q?",
                    "answer": "A",
                    "paragraphs": [
                        {"idx": 0, "title": "Same", "paragraph_text": "First."},
                        {"idx": 1, "title": "Same", "paragraph_text": "Second."},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    documents = MuSiQueDataset().load(str(path)).items[0]["documents"]

    assert len({d["doc_id"] for d in documents}) == 2


def test_musique_declares_paragraphs_it_drops(tmp_path):
    from cke.datasets.musique_loader import MuSiQueDataset
    from cke.diagnostics import DegradedComponentError

    path = tmp_path / "musique.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "2hop__1_2",
                    "question": "Q?",
                    "answer": "A",
                    "paragraphs": [
                        {"idx": 0, "title": "Kept", "paragraph_text": "Text."},
                        {"idx": 1, "title": "Empty", "paragraph_text": "   "},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(DegradedComponentError, match="paragraphs"):
        MuSiQueDataset(strict=True).load(str(path))

    loader = MuSiQueDataset(strict=False).load(str(path))
    assert loader.degraded is True
    assert len(loader.items[0]["documents"]) == 1


def test_musique_normalize_record_matches_load(tmp_path):
    """load() is the per-record method applied in order, nothing more."""
    from cke.datasets.musique_loader import MuSiQueDataset

    records = _musique_records()
    path = tmp_path / "musique.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    loaded = MuSiQueDataset().load(str(path)).items
    one_by_one = [
        MuSiQueDataset().normalize_record(index, record)
        for index, record in enumerate(records)
    ]

    assert one_by_one == loaded
    assert one_by_one[1]["documents"] == []


def test_musique_is_in_the_registry(tmp_path):
    path = tmp_path / "musique.json"
    path.write_text(json.dumps(_musique_records()), encoding="utf-8")

    dataset = load_dataset("musique", str(path))

    assert len(dataset.items) == 2


def test_wiki2_declares_context_entries_it_drops(tmp_path):
    """It dropped them in silence while both its siblings declared them."""
    from cke.datasets.wiki2_loader import WikiMultiHopDataset
    from cke.diagnostics import DegradedComponentError

    path = tmp_path / "wiki2.json"
    path.write_text(
        json.dumps(
            [
                {
                    "_id": "w1",
                    "question": "Q?",
                    "answer": "A",
                    "context": [["Kept", ["Text."]], ["malformed"]],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(DegradedComponentError, match="context entries"):
        WikiMultiHopDataset(strict=True).load(str(path))

    loader = WikiMultiHopDataset(strict=False).load(str(path))
    assert loader.degraded is True
    assert len(loader.items[0]["documents"]) == 1
