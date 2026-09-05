#!/usr/bin/env python3
"""How much of real prose the rule extractor actually reads.

The benchmark's CKE arm is a graph built by ``RuleExtractor`` from the
documents an item carries. Five sentence frames feed that graph, and nobody has
measured what share of encyclopedic prose they catch. Without that figure the
benchmark's headline comparison cannot be interpreted: if the extractor yields
three triples where the dense arm retrieves ten documents, the token ratio is
fixed by the units rather than measured — which is one of the four defects that
invalidated this project's original result, and it would recur silently.

Three numbers decide it.

**Supporting-sentence yield** is the one that matters. A graph full of triples
drawn from irrelevant sentences cannot answer the question at any retrieval
budget. This counts the sentences the dataset names as supporting facts, and
asks how many of them produced a statement at all.

**Statements per item** is the CKE arm's real retrieval budget. The benchmark
reports a configuration called "CKE N=12". If the median item yields fewer than
twelve statements, that arm was never retrieving twelve of anything and the
token figure beside it is a fact about the extractor.

**Frame breakdown** says which of the five frames carries the work. Two of them
name verbs outright — ``uses`` and ``developed`` — and are left over from a
software demo this repository no longer runs.

The extraction here is the real one. Statements per item come from calling
``RuleExtractor.extract`` on each document exactly as
``scripts/run_cke_benchmark.py`` does, so the count is the count that benchmark
would get. The per-sentence figures call the same method on individual
sentences, which is the only way to attribute a statement to the sentence it
came from; the module docstring of the extractor is the authority on what it
does, not this file.

Run it twice and diff it with ``cke-compare-runs``. A coverage figure that
moves between runs is not a measurement.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Read once, so every figure this run reports carries the same timestamp.
_STARTED_AT = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

from cke.datasets.hotpot_loader import HotpotDataset  # noqa: E402
from cke.datasets.wiki2_loader import WikiMultiHopDataset  # noqa: E402
from cke.extractor.rule_extractor import RuleExtractor  # noqa: E402


def _driver():
    """The benchmark driver, for the helpers this measurement must share.

    ``select_indices`` and ``file_digest`` decide which items a run evaluates
    and how its input is identified. Re-implementing either would let this
    measurement sample a different set from the benchmark it is about to
    inform, which is the whole reason to take them from there.
    """
    path = ROOT / "scripts" / "run_cke_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_cke_benchmark_coverage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Datasets whose published records carry ``context`` as [title, [sentences]]
#: and ``supporting_facts`` as [title, sentence index]. MuSiQue publishes
#: paragraphs instead and is not measured here; saying so beats a figure that
#: silently covers two datasets out of three.
LOADERS = {
    "hotpotqa": HotpotDataset,
    "wiki2": WikiMultiHopDataset,
}


def frame_of(relation: str) -> str:
    """Which of the extractor's frames produced a relation.

    Derived from ``RuleExtractor.PATTERNS`` rather than from a list written
    here, so a frame added or renamed upstream cannot leave this attributing
    statements to a frame that no longer exists. A template's placeholders
    stand for whatever the frame captured, and the first template that fits
    wins — the same order the extractor tries them in, which is what makes
    "directed_by" the passive frame rather than the prepositional one.
    """
    for _, template in RuleExtractor.PATTERNS:
        if "{" not in template:
            if relation == template:
                return template
            continue
        expression = "^" + re.sub(r"\\{\w+\\}", ".+", re.escape(template)) + "$"
        if re.match(expression, relation):
            return template
    return "unattributed"


def _sentences_by_title(record: dict[str, Any]) -> dict[str, list[str]]:
    """The published context, as a sentence list per document title."""
    by_title: dict[str, list[str]] = {}
    for entry in record.get("context") or []:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        title, body = entry
        sentences = body if isinstance(body, list) else [body]
        by_title[str(title)] = [str(sentence) for sentence in sentences]
    return by_title


def _supporting_sentences(record: dict[str, Any]) -> set[tuple[str, int]]:
    """The (title, sentence index) pairs the dataset names as supporting."""
    supporting: set[tuple[str, int]] = set()
    for fact in record.get("supporting_facts") or []:
        if isinstance(fact, (list, tuple)) and len(fact) == 2:
            try:
                supporting.add((str(fact[0]), int(fact[1])))
            except (TypeError, ValueError):
                continue
    return supporting


def measure(
    name: str,
    path: Path,
    limit: int | None,
    seed: int,
    method: str,
    driver: Any,
) -> dict[str, Any]:
    """Every figure this script reports, for one dataset."""
    loader = LOADERS[name](strict=False)
    extractor = RuleExtractor()

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    count = len(raw) if limit is None else limit
    chosen = driver.select_indices(len(raw), count, seed, method)

    per_item_statements: list[int] = []
    per_item_supporting_statements: list[int] = []
    sentences_total = sentences_yielding = 0
    supporting_total = supporting_yielding = 0
    items_with_a_supporting_statement = 0
    frames: dict[str, int] = {}
    relations: dict[str, int] = {}

    for index in chosen:
        record = raw[index]
        item = loader.normalize_record(index, record)

        # The benchmark extracts per document, from the merged text the loader
        # produces. This is that call, so this count is that count.
        item_statements = 0
        for document in item.get("documents") or []:
            for statement in extractor.extract(document.get("text", "")):
                item_statements += 1
                frames[frame_of(statement.relation)] = (
                    frames.get(frame_of(statement.relation), 0) + 1
                )
                relations[statement.relation] = relations.get(statement.relation, 0) + 1
        per_item_statements.append(item_statements)

        # Per sentence, which is the only way to say which sentence a
        # statement came from. Attribution, not a second extraction rule.
        supporting = _supporting_sentences(record)
        supporting_here = 0
        for title, sentences in _sentences_by_title(record).items():
            for position, sentence in enumerate(sentences):
                produced = len(extractor.extract(sentence))
                sentences_total += 1
                sentences_yielding += 1 if produced else 0
                if (title, position) in supporting:
                    supporting_total += 1
                    supporting_yielding += 1 if produced else 0
                    supporting_here += produced
        per_item_supporting_statements.append(supporting_here)
        items_with_a_supporting_statement += 1 if supporting_here else 0

    return {
        "provenance": {
            "path": str(path),
            "sha256": driver.file_digest(path),
            "records_in_file": len(raw),
            "items_evaluated": len(chosen),
            "selection": {
                "method": method,
                "seed": seed if method == "sample" else None,
            },
        },
        "sentences": _yield_block(sentences_yielding, sentences_total),
        "supporting_sentences": _yield_block(supporting_yielding, supporting_total),
        "items_with_a_statement_from_a_supporting_sentence": _yield_block(
            items_with_a_supporting_statement, len(chosen)
        ),
        "statements_per_item": _distribution(per_item_statements),
        "statements_per_item_from_supporting_sentences": _distribution(
            per_item_supporting_statements
        ),
        "frames": dict(sorted(frames.items(), key=lambda pair: -pair[1])),
        "relations_most_common": dict(
            sorted(relations.items(), key=lambda pair: -pair[1])[:20]
        ),
        "distinct_relations": len(relations),
    }


def _yield_block(yielding: int, total: int) -> dict[str, Any]:
    """A count and its share, with the share absent when nothing was counted.

    None rather than 0.0: a share over no observations is not a measured zero,
    and the two must not print alike.
    """
    return {
        "counted": yielding,
        "of": total,
        "share": round(yielding / total, 4) if total else None,
    }


def _distribution(values: list[int]) -> dict[str, Any]:
    """Enough of the shape to see whether "N=12" ever had twelve to retrieve."""
    if not values:
        return {"items": 0}
    ordered = sorted(values)
    return {
        "items": len(ordered),
        "min": ordered[0],
        "p25": ordered[len(ordered) // 4],
        "median": round(statistics.median(ordered), 1),
        "p75": ordered[(3 * len(ordered)) // 4],
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 2),
        "zero": sum(1 for value in ordered if value == 0),
        "at_least_12": sum(1 for value in ordered if value >= 12),
        "share_at_least_12": round(
            sum(1 for value in ordered if value >= 12) / len(ordered), 4
        ),
    }


def render(report: dict[str, Any]) -> str:
    """The figures, and what each one decides."""
    lines = [
        "=" * 72,
        "Rule extractor coverage on published prose",
        "=" * 72,
        "",
    ]
    for name, block in report["datasets"].items():
        provenance = block["provenance"]
        lines += [
            f"{name}: {provenance['items_evaluated']} of "
            f"{provenance['records_in_file']} records, "
            f"{provenance['selection']['method']}",
            f"  sha256 {provenance['sha256'][:16]}...",
            "",
            f"  Sentences producing a statement       " f"{_share(block['sentences'])}",
            f"  SUPPORTING sentences producing one    "
            f"{_share(block['supporting_sentences'])}",
            f"  Items with any supporting statement   "
            f"{_share(block['items_with_a_statement_from_a_supporting_sentence'])}",
            "",
        ]
        per_item = block["statements_per_item"]
        lines += [
            f"  Statements per item: median {per_item['median']}, "
            f"mean {per_item['mean']}, range {per_item['min']}-{per_item['max']}",
            f"    items yielding nothing at all:      {per_item['zero']}",
            f"    items yielding 12 or more:          {per_item['at_least_12']} "
            f"({per_item['share_at_least_12']:.1%})",
            "",
            "  Statements by frame:",
        ]
        for frame, count in block["frames"].items():
            lines.append(f"    {frame:<20} {count}")
        lines.append("")

    lines += [
        "-" * 72,
        "The 'CKE N=12' configuration retrieves twelve statements. The share of",
        "items yielding twelve is how often it had twelve to retrieve; where it",
        "did not, the token figure beside that arm describes the extractor.",
        "",
        "Supporting-sentence yield is the ceiling on what the graph arm can",
        "answer from. Statements drawn from other sentences enlarge the graph",
        "without making the question answerable.",
        "-" * 72,
    ]
    return "\n".join(lines)


def _share(block: dict[str, Any]) -> str:
    if block["share"] is None:
        return "not measured, nothing counted"
    return f"{block['counted']:>7} / {block['of']:<7} {block['share']:.1%}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure what share of published prose RuleExtractor reads, and "
            "how many statements the benchmark's CKE arm actually has to "
            "retrieve from."
        )
    )
    parser.add_argument("--hotpot-path", default=None)
    parser.add_argument("--wiki2-path", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="items per dataset; the whole split by default",
    )
    parser.add_argument("--select", choices=["sample", "prefix"], default="prefix")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        help="write the figures here as JSON, for cke-compare-runs",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    paths = {
        "hotpotqa": Path(args.hotpot_path or data_dir / "hotpotqa_dev.json"),
        "wiki2": Path(args.wiki2_path or data_dir / "wiki2_dev.json"),
    }

    # R1: no substitute data, and no quiet skip either. A coverage figure over
    # whichever dataset happened to be present is not the figure this reports.
    missing = {name: path for name, path in paths.items() if not path.exists()}
    if missing:
        for name, path in missing.items():
            print(
                f"[error] {name} not found at {path}. Fetch it with "
                f"`python scripts/download_datasets.py`; this measurement does "
                f"not substitute generated text for a published split.",
                file=sys.stderr,
            )
        return 2

    driver = _driver()
    report: dict[str, Any] = {
        "started_at": _STARTED_AT,
        "datasets": {
            name: measure(
                name,
                paths[name],
                args.limit,
                args.sample_seed,
                args.select,
                driver,
            )
            for name in LOADERS
        },
    }

    print(render(report))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[output] {out}")

    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
