#!/usr/bin/env python3
"""Download HotpotQA and 2WikiMultiHopQA dev splits into the data/ directory.

Datasets are fetched from HuggingFace through the `datasets` library. There is
no fallback. If a dataset cannot be obtained, this script raises
`DatasetUnavailableError` naming the dataset and where to get it, so that no
evaluation can run against substitute data.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

HOTPOTQA_SOURCE = "https://huggingface.co/datasets/hotpotqa/hotpot_qa"
WIKI2_SOURCE = "https://huggingface.co/datasets/xanhho/2WikiMultihopQA"


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


def _try_hf_hotpotqa(
    out_path: Path, limit: int | None = None, reasons: list[str] | None = None
) -> bool:
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
            ds = load_dataset(name, split="validation", **kwargs)
            break
        except Exception as exc:
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
        if limit and len(rows) >= limit:
            break

    if not rows:
        log.append("HuggingFace returned an empty validation split")
        return False

    out_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[download] HotpotQA: {len(rows)} items → {out_path}")
    return True


def _try_hf_wiki2(
    out_path: Path, limit: int | None = None, reasons: list[str] | None = None
) -> bool:
    """Download the 2WikiMultiHopQA dev split via HuggingFace datasets."""
    log = reasons if reasons is not None else []
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        log.append(f"`datasets` library not importable: {exc}")
        return False

    ds = None
    for name, cfg in [("2wikimultihop", None)]:
        try:
            kwargs: dict = {"trust_remote_code": True}
            if cfg:
                kwargs["name"] = cfg
            ds = load_dataset(name, split="validation", **kwargs)
            break
        except Exception as exc:
            log.append(f"HuggingFace load of {name!r} failed: {exc}")

    if ds is None:
        return False

    rows = []
    for item in ds:
        context = []
        titles = (
            item.get("context", {}).get("title", [])
            if isinstance(item.get("context"), dict)
            else []
        )
        sentences_list = (
            item.get("context", {}).get("sentences", [])
            if isinstance(item.get("context"), dict)
            else []
        )
        if not titles and isinstance(item.get("context"), list):
            context = item["context"]
        else:
            for title, sents in zip(titles, sentences_list):
                context.append([title, list(sents)])
        rows.append(
            {
                "_id": str(item.get("id", item.get("_id", ""))),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "context": context,
                "supporting_facts": item.get("supporting_facts", []),
                "type": item.get("type", ""),
            }
        )
        if limit and len(rows) >= limit:
            break

    if not rows:
        log.append("HuggingFace returned an empty validation split")
        return False

    out_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[download] 2WikiMultiHopQA: {len(rows)} items → {out_path}")
    return True


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def download_hotpotqa(out_path: Path, limit: int = 500) -> None:
    """Ensure HotpotQA is present at ``out_path`` or raise."""
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        print(
            f"[download] HotpotQA already exists: {len(existing)} items at {out_path}"
        )
        return

    reasons: list[str] = []
    if _try_hf_hotpotqa(out_path, limit=limit, reasons=reasons):
        return

    raise DatasetUnavailableError("HotpotQA (distractor dev)", HOTPOTQA_SOURCE, reasons)


def download_wiki2(out_path: Path, limit: int = 500) -> None:
    """Ensure 2WikiMultiHopQA is present at ``out_path`` or raise."""
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        n = len(existing)
        print(f"[download] 2WikiMultiHopQA already exists: {n} items at {out_path}")
        return

    reasons: list[str] = []
    if _try_hf_wiki2(out_path, limit=limit, reasons=reasons):
        return

    raise DatasetUnavailableError("2WikiMultiHopQA (dev)", WIKI2_SOURCE, reasons)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument("--limit", type=int, default=500, help="Max items per dataset")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)

    download_hotpotqa(args.data_dir / "hotpotqa_dev.json", limit=args.limit)
    download_wiki2(args.data_dir / "wiki2_dev.json", limit=args.limit)

    print("[download] Done.")


if __name__ == "__main__":
    main()
