"""The benchmark reads documents off an item, and refuses one that has none.

`_docs_from_item` carried a fallback for a flat list under "contexts",
commented "2WikiMultiHopQA". wiki2_loader has never produced that shape: it
calls `_context_to_documents` and emits "documents", as every other loader in
the package does. Nothing reached the fallback.

Dead was not the worst of it. An item of an unrecognised shape fell through to
an empty document list, the pipeline retrieved nothing from it, and the run
recorded EM and F1 of zero — a figure indistinguishable from a real retrieval
failure and reported as one. It refuses now.

The premise is checked rather than asserted: every loader in `cke.datasets` is
exercised here and must emit "documents", and the set of loaders is derived
from the package, so adding one without saying how it produces documents fails
a test rather than quietly reinstating the fallback's reason to exist.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "run_cke_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_cke_benchmark_docs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_module()


def test_the_documents_an_item_carries_are_returned():
    item = {"id": "a", "documents": [{"doc_id": "d1", "text": "some text"}]}

    assert bench._docs_from_item(item) == [{"doc_id": "d1", "text": "some text"}]


def test_a_document_with_no_text_is_dropped():
    item = {
        "id": "a",
        "documents": [
            {"doc_id": "d1", "text": "kept"},
            {"doc_id": "d2", "text": ""},
            {"doc_id": "d3"},
        ],
    }

    assert bench._docs_from_item(item) == [{"doc_id": "d1", "text": "kept"}]


def test_an_item_with_no_documents_is_refused_not_scored_as_zero():
    """The failure the deleted fallback used to hide."""
    item = {"id": "wrong-shape", "question": "q", "contexts": ["a", "b"]}

    with pytest.raises(ValueError) as refused:
        bench._docs_from_item(item)

    message = str(refused.value)
    assert "wrong-shape" in message
    assert "contexts" in message, "it must say what the item does carry"


# --- the premise: every loader emits "documents" -----------------------------


def _hotpot_item():
    from cke.datasets.hotpot_loader import HotpotDataset

    return HotpotDataset().normalize_record(
        0, {"_id": "h", "question": "q", "answer": "a", "context": [["T", ["s"]]]}
    )


def _wiki2_item():
    from cke.datasets.wiki2_loader import WikiMultiHopDataset

    return WikiMultiHopDataset().normalize_record(
        0, {"_id": "w", "question": "q", "answer": "a", "context": [["T", ["s"]]]}
    )


def _musique_item():
    from cke.datasets.musique_loader import MuSiQueDataset

    return MuSiQueDataset().normalize_record(
        0,
        {
            "id": "m",
            "question": "q",
            "answer": "a",
            "paragraphs": [{"title": "T", "paragraph_text": "s", "idx": 0}],
        },
    )


def _locomo_item(tmp_path):
    from cke.datasets.locomo_loader import LoCoMoDataset

    path = tmp_path / "locomo.json"
    path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "c1",
                    "conversation": {
                        "session_1_date_time": "1 Jan",
                        "session_1": [
                            {"dia_id": "D1:1", "speaker": "A", "text": "hello"}
                        ],
                    },
                    "qa": [{"question": "q", "answer": "a", "evidence": ["D1:1"]}],
                }
            ]
        ),
        encoding="utf-8",
    )
    return LoCoMoDataset().load(str(path)).items[0]


def _msmarco_item(tmp_path):
    from cke.datasets.msmarco_loader import MSMarcoDocumentDataset

    path = tmp_path / "docs.tsv"
    path.write_text("d1\tsome text\n", encoding="utf-8")
    return MSMarcoDocumentDataset().load(str(path)).items[0]


#: One entry per loader module. The next test holds this to the package.
ONE_ITEM_FROM_EACH_LOADER = {
    "hotpot_loader": lambda tmp_path: _hotpot_item(),
    "locomo_loader": _locomo_item,
    "msmarco_loader": _msmarco_item,
    "musique_loader": lambda tmp_path: _musique_item(),
    "wiki2_loader": lambda tmp_path: _wiki2_item(),
}


def test_every_loader_in_the_package_is_exercised_here():
    """Derived, so a new loader cannot arrive unexamined.

    base_loader is the abstract base and produces no items of its own.
    """
    modules = {
        path.stem
        for path in (ROOT / "cke" / "datasets").glob("*_loader.py")
        if path.stem != "base_loader"
    }

    assert modules == set(ONE_ITEM_FROM_EACH_LOADER), (
        "a loader was added or removed; say how it produces documents, or the "
        "refusal in _docs_from_item rests on a premise nobody checked"
    )


@pytest.mark.parametrize("name", sorted(ONE_ITEM_FROM_EACH_LOADER))
def test_every_loader_emits_documents_the_benchmark_can_read(name, tmp_path):
    item = ONE_ITEM_FROM_EACH_LOADER[name](tmp_path)

    assert "documents" in item, f"{name} emits no 'documents'"
    # The point is not that the key exists but that the benchmark accepts it.
    assert bench._docs_from_item(item), f"{name}'s documents carry no text"
