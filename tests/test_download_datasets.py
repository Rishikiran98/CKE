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


def _musique_row() -> dict:
    return {
        "id": "2hop__1_2",
        "question": "Who is the spouse of the Green performer?",
        "answer": "Miquette Giraudy",
        "answer_aliases": [],
        "answerable": True,
        "paragraphs": [
            {
                "idx": 0,
                "title": "Green (Steve Hillage album)",
                "paragraph_text": "Green is an album by Steve Hillage.",
                "is_supporting": True,
            }
        ],
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


def test_musique_missing_raises_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_try_hf_musique", lambda *a, **k: False)
    out = tmp_path / "musique_dev.json"

    with pytest.raises(dl.DatasetUnavailableError) as excinfo:
        dl.download_musique(out)

    assert not out.exists()
    assert "MuSiQue" in str(excinfo.value)
    assert dl.MUSIQUE_SOURCE in str(excinfo.value)


def test_musique_reuses_a_real_existing_file(tmp_path):
    out = tmp_path / "musique_dev.json"
    _write(out, [_musique_row()])

    dl.download_musique(out)

    assert json.loads(out.read_text(encoding="utf-8"))[0]["id"] == "2hop__1_2"


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


# ---------------------------------------------------------------------------
# 2WikiMultiHopQA: the authors' archive
# ---------------------------------------------------------------------------


def _wiki2_row(item_id: str = "8813f87c0bdd11eba7f7acde48001122") -> dict:
    return {
        "_id": item_id,
        "type": "compositional",
        "question": "Who is the mother of the director of film Polish-Russian War?",
        "answer": "Małgorzata Braunek",
        "context": [["Polish-Russian War (film)", ["A 2009 Polish film."]]],
        "supporting_facts": [["Polish-Russian War (film)", 1]],
        "evidences": [["Polish-Russian War", "director", "Xawery Żuławski"]],
    }


def _wiki2_archive(rows: list[dict] | None = None) -> bytes:
    """An archive shaped like the published one, __MACOSX entries included."""
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("__MACOSX/data/._dev.json", b"resource fork")
        bundle.writestr("data/test.json", json.dumps([{"_id": "test"}]))
        bundle.writestr(
            "data/dev.json", json.dumps(rows if rows is not None else [_wiki2_row()])
        )
    return buffer.getvalue()


class _FakeResponse:
    """A response that can stop early, the way a dropped connection does."""

    def __init__(self, body: bytes, deliver: int | None = None) -> None:
        self._body = body
        self._delivered = 0
        self._deliver = len(body) if deliver is None else deliver
        self.headers = {"Content-Length": str(len(body))}

    def read(self, size: int = -1) -> bytes:
        remaining = self._deliver - self._delivered
        if remaining <= 0:
            return b""
        take = remaining if size in (-1, None) else min(size, remaining)
        chunk = self._body[self._delivered : self._delivered + take]
        self._delivered += len(chunk)
        return chunk

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _fake_urlopen(bodies: list[tuple[bytes, int | None]]):
    calls = iter(bodies)

    def opener(url, timeout=None):  # noqa: ARG001 - signature matches urlopen
        body, deliver = next(calls)
        return _FakeResponse(body, deliver)

    return opener


def test_wiki2_reads_the_dev_split_out_of_the_official_archive(tmp_path, monkeypatch):
    archive = _wiki2_archive([_wiki2_row(str(i)) for i in range(5)])
    monkeypatch.setattr(dl, "urlopen", _fake_urlopen([(archive, None)]))
    out = tmp_path / "wiki2_dev.json"

    dl.download_wiki2(out, limit=3)

    rows = json.loads(out.read_text(encoding="utf-8"))
    # The dev split, capped, and neither the test split nor the resource forks.
    assert [r["_id"] for r in rows] == ["0", "1", "2"]
    assert rows[0]["evidences"] == _wiki2_row()["evidences"]


def test_wiki2_retries_a_truncated_transfer(tmp_path, monkeypatch):
    """read() returns empty on a dropped connection instead of raising.

    Two of three real transfers of this archive stopped short, so a
    downloader that trusts read() unpacks a corrupt file most of the time.
    """
    archive = _wiki2_archive()
    monkeypatch.setattr(
        dl,
        "urlopen",
        _fake_urlopen([(archive, len(archive) - 50), (archive, None)]),
    )
    out = tmp_path / "wiki2_dev.json"

    dl.download_wiki2(out, limit=1)

    assert json.loads(out.read_text(encoding="utf-8"))[0]["_id"] == _wiki2_row()["_id"]


def test_wiki2_gives_up_when_every_transfer_is_short(tmp_path, monkeypatch):
    archive = _wiki2_archive()
    short = [(archive, len(archive) - 50)] * dl._ARCHIVE_ATTEMPTS
    monkeypatch.setattr(dl, "urlopen", _fake_urlopen(short))
    out = tmp_path / "wiki2_dev.json"

    with pytest.raises(dl.DatasetUnavailableError) as excinfo:
        dl.download_wiki2(out)

    message = str(excinfo.value)
    assert "50 bytes short" in message
    assert dl.WIKI2_SOURCE in message
    assert not out.exists()


def test_wiki2_rejects_an_archive_with_no_dev_split(tmp_path, monkeypatch):
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("data/train.json", json.dumps([{"_id": "train"}]))
    monkeypatch.setattr(dl, "urlopen", _fake_urlopen([(buffer.getvalue(), None)]))
    out = tmp_path / "wiki2_dev.json"

    with pytest.raises(dl.DatasetUnavailableError, match="no dev.json"):
        dl.download_wiki2(out)

    assert not out.exists()


# ---------------------------------------------------------------------------
# LoCoMo
# ---------------------------------------------------------------------------


def _locomo_conversations(n: int = 2) -> list[dict]:
    return [
        {
            "sample_id": f"conv-{i}",
            "conversation": {
                "speaker_a": "Ann",
                "speaker_b": "Bo",
                "session_1": [{"speaker": "Ann", "dia_id": "D1:1", "text": "Hi"}],
            },
            "qa": [{"question": "Q?", "answer": "A", "evidence": ["D1:1"]}],
        }
        for i in range(n)
    ]


def test_locomo_downloads_the_published_conversations(tmp_path, monkeypatch):
    body = json.dumps(_locomo_conversations(3)).encode("utf-8")
    monkeypatch.setattr(dl, "urlopen", _fake_urlopen([(body, None)]))
    out = tmp_path / "locomo.json"

    dl.download_locomo(out)

    rows = json.loads(out.read_text(encoding="utf-8"))
    assert [r["sample_id"] for r in rows] == ["conv-0", "conv-1", "conv-2"]


def test_locomo_limit_caps_conversations_not_questions(tmp_path, monkeypatch):
    body = json.dumps(_locomo_conversations(3)).encode("utf-8")
    monkeypatch.setattr(dl, "urlopen", _fake_urlopen([(body, None)]))
    out = tmp_path / "locomo.json"

    dl.download_locomo(out, limit=2)

    assert len(json.loads(out.read_text(encoding="utf-8"))) == 2


def test_locomo_retries_a_truncated_transfer(tmp_path, monkeypatch):
    body = json.dumps(_locomo_conversations(1)).encode("utf-8")
    monkeypatch.setattr(
        dl, "urlopen", _fake_urlopen([(body, len(body) - 20), (body, None)])
    )
    out = tmp_path / "locomo.json"

    dl.download_locomo(out)

    assert json.loads(out.read_text(encoding="utf-8"))[0]["sample_id"] == "conv-0"


def test_locomo_gives_up_when_every_transfer_is_short(tmp_path, monkeypatch):
    body = json.dumps(_locomo_conversations(1)).encode("utf-8")
    short = [(body, len(body) - 20)] * dl._ARCHIVE_ATTEMPTS
    monkeypatch.setattr(dl, "urlopen", _fake_urlopen(short))
    out = tmp_path / "locomo.json"

    with pytest.raises(dl.DatasetUnavailableError) as excinfo:
        dl.download_locomo(out)

    assert "20 bytes short" in str(excinfo.value)
    assert dl.LOCOMO_SOURCE in str(excinfo.value)
    assert not out.exists()
