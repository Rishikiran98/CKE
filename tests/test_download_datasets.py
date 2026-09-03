"""Tests for the benchmark dataset downloader.

These guard one property: the downloader never hands an evaluation a corpus
that this repository generated for itself. It either produces a real dataset
or it raises.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "download_datasets.py"
    spec = importlib.util.spec_from_file_location("download_datasets", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dl = _load_module()


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def _real_row(item_id: str = "5a8b57f25542995d1e6f1371") -> dict:
    return {
        "_id": item_id,
        "question": "Which magazine was started first, Arthur's or First for Women?",
        "answer": "Arthur's Magazine",
        "context": [["Arthur's Magazine", ["Arthur's Magazine was a periodical."]]],
        "supporting_facts": [["Arthur's Magazine", 0]],
        "type": "comparison",
        "level": "medium",
    }


def test_no_synthetic_generator_remains():
    """The generator and its fixtures must stay deleted."""
    source = (ROOT / "scripts" / "download_datasets.py").read_text(encoding="utf-8")
    for name in (
        "_make_synthetic",
        "_SYNTHETIC_TEMPLATES",
        "_ENTITY_POOL",
        "_RELATION_SENTENCES",
    ):
        assert name not in source


def test_missing_dataset_raises_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_try_hf_hotpotqa", lambda *a, **k: False)
    out = tmp_path / "hotpotqa_dev.json"

    with pytest.raises(dl.DatasetUnavailableError) as excinfo:
        dl.download_hotpotqa(out)

    assert not out.exists()
    assert "HotpotQA" in str(excinfo.value)
    assert dl.HOTPOTQA_SOURCE in str(excinfo.value)


@pytest.mark.parametrize(
    "filename, prefix, download",
    [
        ("hotpotqa_dev.json", "synthetic_", "download_hotpotqa"),
        ("wiki2_dev.json", "wiki2_synthetic_", "download_wiki2"),
    ],
)
def test_stale_generated_corpus_is_rejected(tmp_path, filename, prefix, download):
    """A file left by the old generator must not be reused.

    data/ is gitignored, so a checkout that ran the previous downloader still
    holds that corpus on disk after this change.
    """
    out = tmp_path / filename
    _write(out, [_real_row(f"{prefix}{i}") for i in range(3)])

    with pytest.raises(dl.DatasetUnavailableError) as excinfo:
        getattr(dl, download)(out)

    message = str(excinfo.value)
    assert "generated items" in message
    assert f"{prefix}0" in message


def test_unreadable_existing_file_is_rejected(tmp_path):
    out = tmp_path / "hotpotqa_dev.json"
    out.write_text("not json at all", encoding="utf-8")

    with pytest.raises(dl.DatasetUnavailableError):
        dl.download_hotpotqa(out)


def test_empty_existing_file_is_rejected(tmp_path):
    out = tmp_path / "hotpotqa_dev.json"
    _write(out, [])

    with pytest.raises(dl.DatasetUnavailableError):
        dl.download_hotpotqa(out)


def test_real_existing_file_is_reused(tmp_path):
    out = tmp_path / "hotpotqa_dev.json"
    _write(out, [_real_row()])

    dl.download_hotpotqa(out)

    assert json.loads(out.read_text(encoding="utf-8"))[0]["_id"] == _real_row()["_id"]
