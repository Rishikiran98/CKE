#!/usr/bin/env python3
"""Download the multi-hop QA dev splits into the data/ directory.

Datasets are fetched from HuggingFace through the `datasets` library. There is
no fallback. If a dataset cannot be obtained, this script raises
`DatasetUnavailableError` naming the dataset and where to get it, so that no
evaluation can run against substitute data.
"""

from __future__ import annotations

import json
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

HOTPOTQA_SOURCE = "https://huggingface.co/datasets/hotpotqa/hotpot_qa"

#: The Hub commits these splits are pinned to. A dataset name is not a
#: dataset for the same reason a model name is not a model: the Hub can serve
#: different rows under the same name tomorrow, and a number computed over
#: rows that can change underneath it is not reproducible. The download
#: records the revision it fetched in the file's sidecar.
HOTPOTQA_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
MUSIQUE_REVISION = "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"
WIKI2_SOURCE = "https://github.com/Alab-NII/2wikimultihop"

#: The dataset archive the 2WikiMultiHopQA authors publish from that
#: repository. Their own release rather than a mirror: a third-party copy
#: cannot be told from a copy someone has edited, and several on the Hub
#: carry model-generated questions in place of the originals.
_WIKI2_ARCHIVE_URL = "https://www.dropbox.com/s/npidmtadreo6df2/data.zip?dl=1"

#: How many times to fetch an archive before giving up. Transfers of this one
#: are intermittently truncated: of three consecutive attempts here, two
#: stopped 19 MB and 12 MB short and the third completed.
_ARCHIVE_ATTEMPTS = 3
MUSIQUE_SOURCE = "https://huggingface.co/datasets/dgslibisey/MuSiQue"
LOCOMO_SOURCE = "https://github.com/snap-research/locomo"

#: The conversation file published from that repository.
_LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)

# Item ids written by the synthetic generator that this script used to carry.
# data/ is gitignored, so a checkout that ran the old downloader still holds
# that corpus on disk and would otherwise be reused silently.
_LEGACY_SYNTHETIC_ID_PREFIXES = ("synthetic_", "wiki2_synthetic_")


class DatasetUnavailableError(RuntimeError):
    """Raised when a required dataset cannot be downloaded.

    Evaluation data must come from a real, externally maintained source. When
    the download fails the correct outcome is this error, never generated or
    templated stand-in data.
    """

    def __init__(self, dataset: str, source_url: str, reasons: list[str]) -> None:
        detail = "\n".join(f"  - {reason}" for reason in reasons) or "  - no attempts"
        super().__init__(
            f"Could not download {dataset}.\n"
            f"Attempts:\n{detail}\n"
            f"Obtain it from: {source_url}\n"
            f"Install the loader with `pip install datasets`, or download the "
            f"dev split manually and pass its path to the benchmark driver. "
            f"This script does not generate substitute data."
        )
        self.dataset = dataset
        self.source_url = source_url
        self.reasons = reasons


# ---------------------------------------------------------------------------
# HuggingFace helpers
# ---------------------------------------------------------------------------


def _try_hf_hotpotqa(out_path: Path, reasons: list[str] | None = None) -> bool:
    """Download the HotpotQA distractor dev split via HuggingFace datasets."""
    log = reasons if reasons is not None else []
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        log.append(f"`datasets` library not importable: {exc}")
        return False

    ds = None
    # Try multiple dataset identifiers — the API has changed over versions.
    for name, cfg in [
        ("hotpot_qa", "distractor"),
        ("hotpotqa/hotpot_qa", "distractor"),
        ("hotpotqa", None),
    ]:
        try:
            kwargs: dict = {}
            if cfg:
                kwargs["name"] = cfg
            ds = load_dataset(
                name, split="validation", revision=HOTPOTQA_REVISION, **kwargs
            )
            break
        except Exception as exc:  # noqa: BLE001 - the hub raises varied errors
            # Broad by necessity: datasets/huggingface_hub raise many distinct
            # error types. Every one is recorded and surfaced in the final
            # DatasetUnavailableError rather than discarded.
            log.append(f"HuggingFace load of {name!r} failed: {exc}")
            ds = None

    if ds is None:
        return False

    rows = []
    for item in ds:
        context = []
        titles = item.get("context", {}).get("title", [])
        sentences_list = item.get("context", {}).get("sentences", [])
        for title, sents in zip(titles, sentences_list):
            context.append([title, list(sents)])
        rows.append(
            {
                "_id": item.get("id", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "context": context,
                "supporting_facts": list(
                    zip(
                        item.get("supporting_facts", {}).get("title", []),
                        item.get("supporting_facts", {}).get("sent_id", []),
                    )
                ),
                "type": item.get("type", ""),
                "level": item.get("level", ""),
            }
        )
    if not rows:
        log.append("HuggingFace returned an empty validation split")
        return False

    _write_split(
        out_path, rows, "HotpotQA (distractor dev)", HOTPOTQA_SOURCE, HOTPOTQA_REVISION
    )
    print(f"[download] HotpotQA: {len(rows)} items → {out_path}")
    return True


def _stream_archive(url: str, handle, log: list[str]) -> bool:
    """Copy ``url`` into ``handle``, true only if the whole body arrived.

    ``urlopen``'s ``read`` returns empty on a dropped connection rather than
    raising, so a truncated body is indistinguishable from a complete one
    until something downstream chokes on it. Comparing what arrived against
    ``Content-Length`` is what makes the difference visible; without it a
    short read becomes a corrupt file, and a corrupt file that happens to
    parse becomes evaluation data.
    """
    handle.seek(0)
    handle.truncate()
    # The URL is one of this module's own constants, https only.
    with urlopen(url, timeout=600) as response:  # noqa: S310  # nosec B310
        declared = int(response.headers.get("Content-Length") or 0)
        written = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
    handle.flush()

    if declared and written != declared:
        log.append(
            f"transfer of {url} stopped {declared - written} bytes short "
            f"of the {declared} it declared"
        )
        return False
    return True


def _try_official_wiki2(out_path: Path, reasons: list[str] | None = None) -> bool:
    """Download the 2WikiMultiHopQA dev split from the authors' release.

    This replaces a HuggingFace attempt that could never have succeeded: it
    tried one dataset id, ``2wikimultihop``, which does not exist on the Hub,
    while the source URL it printed on failure named ``xanhho/2WikiMultihopQA``
    and never tried it. That copy is a loading script, which ``datasets`` 5
    refuses to execute, so the path was dead at both ends.

    The archive is 246 MB and holds the train, dev and test splits. Only the
    dev split is read, and it is read from inside the zip so the 682 MB train
    file is never written to disk.

    Transfers of it are intermittently truncated, so each attempt is checked
    against the declared length and a short one is retried rather than
    unpacked.
    """
    log = reasons if reasons is not None else []
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip") as archive:
            for attempt in range(1, _ARCHIVE_ATTEMPTS + 1):
                if _stream_archive(_WIKI2_ARCHIVE_URL, archive, log):
                    break
                log.append(f"  (attempt {attempt} of {_ARCHIVE_ATTEMPTS})")
            else:
                return False

            with zipfile.ZipFile(archive.name) as bundle:
                names = [
                    name
                    for name in bundle.namelist()
                    if name.endswith("dev.json") and not name.startswith("__MACOSX")
                ]
                if not names:
                    log.append(
                        f"archive at {_WIKI2_ARCHIVE_URL} holds no dev.json "
                        f"(members: {bundle.namelist()[:6]})"
                    )
                    return False
                with bundle.open(names[0]) as handle:
                    raw = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - network and archive errors vary
        log.append(f"download of {_WIKI2_ARCHIVE_URL} failed: {exc}")
        return False

    if not isinstance(raw, list) or not raw:
        log.append("the archive's dev.json is not a non-empty JSON list")
        return False

    rows = raw
    _write_split(out_path, rows, "2WikiMultiHopQA (dev)", WIKI2_SOURCE)
    print(f"[download] 2WikiMultiHopQA: {len(rows)} items \u2192 {out_path}")
    return True


def _try_hf_musique(out_path: Path, reasons: list[str] | None = None) -> bool:
    """Download the MuSiQue answerable dev split via HuggingFace datasets.

    Paragraphs are written whole, including the ``is_supporting`` flag: it is
    the dataset's own label for which documents an answer needs, and a
    retrieval recall figure has nothing to measure against without it.
    """
    log = reasons if reasons is not None else []
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        log.append(f"`datasets` library not importable: {exc}")
        return False

    ds = None
    for name in ("dgslibisey/MuSiQue", "musique"):
        try:
            ds = load_dataset(name, split="validation", revision=MUSIQUE_REVISION)
            break
        except Exception as exc:  # noqa: BLE001 - the hub raises varied errors
            log.append(f"HuggingFace load of {name!r} failed: {exc}")
            ds = None

    if ds is None:
        return False

    rows = []
    for item in ds:
        rows.append(
            {
                "id": item.get("id", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "answer_aliases": list(item.get("answer_aliases") or []),
                "answerable": item.get("answerable"),
                "paragraphs": [
                    {
                        "idx": p.get("idx"),
                        "title": p.get("title", ""),
                        "paragraph_text": p.get("paragraph_text", ""),
                        "is_supporting": bool(p.get("is_supporting")),
                    }
                    for p in item.get("paragraphs", [])
                ],
            }
        )
    if not rows:
        log.append("HuggingFace returned an empty validation split")
        return False

    _write_split(out_path, rows, "MuSiQue (dev)", MUSIQUE_SOURCE, MUSIQUE_REVISION)
    print(f"[download] MuSiQue: {len(rows)} items \u2192 {out_path}")
    return True


def _try_official_locomo(out_path: Path, reasons: list[str] | None = None) -> bool:
    """Download the LoCoMo conversations from the authors' repository.

    A conversation is the unit
    the file publishes, and its questions cannot be answered without it.
    """
    log = reasons if reasons is not None else []
    try:
        with tempfile.NamedTemporaryFile(suffix=".json") as handle:
            for attempt in range(1, _ARCHIVE_ATTEMPTS + 1):
                if _stream_archive(_LOCOMO_URL, handle, log):
                    break
                log.append(f"  (attempt {attempt} of {_ARCHIVE_ATTEMPTS})")
            else:
                return False
            handle.seek(0)
            raw = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - network and parse errors vary
        log.append(f"download of {_LOCOMO_URL} failed: {exc}")
        return False

    if not isinstance(raw, list) or not raw:
        log.append(f"{_LOCOMO_URL} is not a non-empty JSON list of conversations")
        return False

    rows = raw
    _write_split(out_path, rows, "LoCoMo", LOCOMO_SOURCE)
    questions = sum(len(row.get("qa") or []) for row in rows if isinstance(row, dict))
    print(
        f"[download] LoCoMo: {len(rows)} conversations, "
        f"{questions} questions \u2192 {out_path}"
    )
    return True


def sidecar_path(out_path: Path) -> Path:
    """Where the note about a downloaded file lives."""
    return out_path.with_name(out_path.name + ".source.json")


def _write_split(
    out_path: Path,
    rows: list,
    dataset: str,
    source_url: str,
    revision: str | None = None,
) -> None:
    """Write a split and the note that says it is the whole of one.

    The note is what makes a cached file reusable. Without it a file is just
    some records: this script used to cap downloads, so a file written by an
    older version holds the first N of the split and is indistinguishable, by
    looking at it, from a small dataset.
    """
    out_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sidecar_path(out_path).write_text(
        json.dumps(
            {
                "dataset": dataset,
                "source": source_url,
                "revision": revision,
                "records": len(rows),
                "complete_split": True,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def incomplete_reason(out_path: Path, rows: list) -> str | None:
    """Why the file on disk cannot be shown to hold the whole split.

    ``None`` means it can. Anything else is a reason to fetch it again: a
    truncated split reused as if it were the dataset is the defect this
    guards, and MuSiQue is the sharp case — its dev split is ordered by hop
    count, so a capped file is entirely two-hop and no seed downstream can
    undo that.
    """
    sidecar = sidecar_path(out_path)
    if not sidecar.exists():
        return (
            f"{out_path} has no {sidecar.name} beside it, so it cannot be shown "
            f"to be the whole split. Files written before this script stopped "
            f"capping downloads hold only the first N records"
        )
    try:
        note = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{sidecar} could not be read as JSON: {exc}"
    if not note.get("complete_split"):
        return f"{sidecar} does not record the file as a complete split"
    if note.get("records") != len(rows):
        return (
            f"{sidecar} records {note.get('records')} items and {out_path} holds "
            f"{len(rows)}, so the file has changed since it was fetched"
        )
    return None


def _load_existing(path: Path, dataset: str, source_url: str) -> list:
    """Return the rows of an existing dataset file, or raise.

    An existing file is only reused once it is readable, non-empty, and free of
    the marker ids left by the synthetic generator this script used to contain.
    Anything else raises rather than being silently evaluated against.

    This detects the corpus this repository generated for itself. It is not a
    general provenance check; recording dataset checksums alongside results is
    the durable fix and belongs with the evaluation harness.
    """
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetUnavailableError(
            dataset,
            source_url,
            [f"existing file {path} could not be read as JSON: {exc}"],
        ) from exc

    if not isinstance(rows, list) or not rows:
        raise DatasetUnavailableError(
            dataset,
            source_url,
            [f"existing file {path} is not a non-empty JSON list of items"],
        )

    synthetic = [
        str(row.get("_id", ""))
        for row in rows
        if isinstance(row, dict)
        and str(row.get("_id", "")).startswith(_LEGACY_SYNTHETIC_ID_PREFIXES)
    ]
    if synthetic:
        raise DatasetUnavailableError(
            dataset,
            source_url,
            [
                f"existing file {path} holds {len(synthetic)} generated items "
                f"(for example {synthetic[0]!r}) written by an earlier version "
                f"of this script. Delete the file and download the real dataset; "
                f"it will not be reused"
            ],
        )

    return rows


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def download_hotpotqa(out_path: Path) -> None:
    """Ensure HotpotQA is present at ``out_path`` or raise."""
    if out_path.exists():
        existing = _load_existing(
            out_path, "HotpotQA (distractor dev)", HOTPOTQA_SOURCE
        )
        stale = incomplete_reason(out_path, existing)
        if stale is None:
            n = len(existing)
            print(f"[download] HotpotQA already exists: {n} items at {out_path}")
            return
        print(f"[download] re-fetching HotpotQA: {stale}")

    reasons: list[str] = []
    if _try_hf_hotpotqa(out_path, reasons=reasons):
        return

    raise DatasetUnavailableError("HotpotQA (distractor dev)", HOTPOTQA_SOURCE, reasons)


def download_wiki2(out_path: Path) -> None:
    """Ensure 2WikiMultiHopQA is present at ``out_path`` or raise."""
    if out_path.exists():
        existing = _load_existing(out_path, "2WikiMultiHopQA (dev)", WIKI2_SOURCE)
        stale = incomplete_reason(out_path, existing)
        if stale is None:
            n = len(existing)
            print(f"[download] 2WikiMultiHopQA already exists: {n} items at {out_path}")
            return
        print(f"[download] re-fetching 2WikiMultiHopQA: {stale}")

    reasons: list[str] = []
    if _try_official_wiki2(out_path, reasons=reasons):
        return

    raise DatasetUnavailableError("2WikiMultiHopQA (dev)", WIKI2_SOURCE, reasons)


def download_musique(out_path: Path) -> None:
    """Ensure MuSiQue is present at ``out_path`` or raise."""
    if out_path.exists():
        existing = _load_existing(out_path, "MuSiQue (dev)", MUSIQUE_SOURCE)
        stale = incomplete_reason(out_path, existing)
        if stale is None:
            n = len(existing)
            print(f"[download] MuSiQue already exists: {n} items at {out_path}")
            return
        print(f"[download] re-fetching MuSiQue: {stale}")

    reasons: list[str] = []
    if _try_hf_musique(out_path, reasons=reasons):
        return

    raise DatasetUnavailableError("MuSiQue (dev)", MUSIQUE_SOURCE, reasons)


def download_locomo(out_path: Path) -> None:
    """Ensure LoCoMo is present at ``out_path`` or raise."""
    if out_path.exists():
        existing = _load_existing(out_path, "LoCoMo", LOCOMO_SOURCE)
        stale = incomplete_reason(out_path, existing)
        if stale is None:
            n = len(existing)
            print(f"[download] LoCoMo already exists: {n} conversations at {out_path}")
            return
        print(f"[download] re-fetching LoCoMo: {stale}")

    reasons: list[str] = []
    if _try_official_locomo(out_path, reasons=reasons):
        return

    raise DatasetUnavailableError("LoCoMo", LOCOMO_SOURCE, reasons)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download benchmark datasets")

    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)

    download_hotpotqa(args.data_dir / "hotpotqa_dev.json")
    download_wiki2(args.data_dir / "wiki2_dev.json")
    download_musique(args.data_dir / "musique_dev.json")
    download_locomo(args.data_dir / "locomo.json")

    print("[download] Done.")


if __name__ == "__main__":
    main()
