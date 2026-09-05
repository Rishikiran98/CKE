"""What the package ships that nothing it installs can reach.

Three modules were deleted for being imported by nothing at all: ``cke.sdk``
(a client for an HTTP service nothing runs), ``cke.utils.convergence_stats``
and ``cke.utils.trust_stats``. Together, 157 lines with no importer and no
test, sitting in the coverage denominator.

The deletion is easy to make and easy to undo by accident. What keeps it made
is this census: the reachable set is derived from the import graph, starting
at every entry point the package actually installs, and the modules outside it
are listed below. The list is exact, so a module that stops being reachable
fails this test, and a module that becomes reachable fails it too until
somebody removes the entry and says why in the diff.

The point is that the number is *visible*. Forty-seven modules ship inside the
wheel that no console script, driver or demo can reach; thirty-one of them are
the conversational layer. Retiring that is a decision about what CKE is for,
not a cleanup, so it is recorded here rather than taken quietly.

Reachability here means "imported, transitively, from an entry point". A
module the tests exercise directly is still unreachable by this definition —
that is the distinction being drawn. `cke/conversation/` is well tested and
entirely unreachable, which is exactly the fact worth surfacing.
"""

from __future__ import annotations

import ast
import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Every module inside the package that no entry point can reach. Grouped by
#: subpackage, with counts, because the counts are the finding.
KNOWN_ORPHANS = {
    # api — 2
    "cke.api",
    "cke.api.server",
    # conversation — 31
    "cke.conversation",
    "cke.conversation.answering",
    "cke.conversation.answering.abstention",
    "cke.conversation.answering.confidence",
    "cke.conversation.answering.evidence_selection",
    "cke.conversation.answering.grounded_generation",
    "cke.conversation.config",
    "cke.conversation.consolidation",
    "cke.conversation.extractor",
    "cke.conversation.extractors",
    "cke.conversation.extractors.base",
    "cke.conversation.extractors.heuristic",
    "cke.conversation.extractors.temporal",
    "cke.conversation.ingestion",
    "cke.conversation.memory",
    "cke.conversation.memory_store",
    "cke.conversation.patterns",
    "cke.conversation.reference_resolution",
    "cke.conversation.resolution",
    "cke.conversation.resolution.alias_resolution",
    "cke.conversation.resolution.reference_resolution",
    "cke.conversation.resolution.temporal_resolution",
    "cke.conversation.retrieval",
    "cke.conversation.retrieval.candidate_generation",
    "cke.conversation.retrieval.fact_retrieval",
    "cke.conversation.retrieval.graph_expansion",
    "cke.conversation.retrieval.reranker",
    "cke.conversation.retrieval.summary_retrieval",
    "cke.conversation.retriever",
    "cke.conversation.types",
    "cke.conversation.validation",
    # evaluation — 1
    "cke.evaluation.conversation_cases",
    # experiments — 1
    "cke.experiments",
    # extractor — 1
    "cke.extractor.entity_linker",
    # graph — 7
    "cke.graph.deduplicator",
    "cke.graph.drift_monitor",
    "cke.graph.edge",
    "cke.graph.entity_resolver",
    "cke.graph.snapshot_manager",
    "cke.graph.trust_engine",
    "cke.graph.update_pipeline",
    # pipeline — 1
    "cke.pipeline.conversational_orchestrator",
    # reporting — 2
    "cke.reporting",
    "cke.reporting.report_generator",
    # retrieval — 1
    "cke.retrieval.retrieval_mode",
}


def _import_graph() -> dict[str, set[str]]:
    """Module name to the modules it imports, over the shipped source."""
    graph: dict[str, set[str]] = {}
    paths = (
        list(ROOT.glob("cke/**/*.py"))
        + list(ROOT.glob("scripts/**/*.py"))
        + [ROOT / "demo.py"]
    )
    for path in paths:
        if "__pycache__" in str(path):
            continue
        name = str(path.relative_to(ROOT).with_suffix("")).replace("/", ".")
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        imported: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
        graph.setdefault(name, set()).update(imported)
    return graph


def _entry_points(graph: dict[str, set[str]]) -> set[str]:
    """Everything installation makes runnable, read from pyproject.

    Derived rather than listed: a console script added to [project.scripts]
    joins this set on its own, and whatever it reaches stops being an orphan
    without anybody editing this file.
    """
    with (ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)

    entries = {target.split(":")[0] for target in config["project"]["scripts"].values()}
    entries.add("demo")
    # Read by setuptools for the version, so it ships reachable even though no
    # module imports it.
    entries.add("cke.version")
    entries |= {name for name in graph if name.startswith("scripts.")}
    return entries


def _reachable(graph: dict[str, set[str]]) -> set[str]:
    reached: set[str] = set()
    stack = [entry for entry in _entry_points(graph) if entry in graph]
    while stack:
        module = stack.pop()
        if module in reached:
            continue
        reached.add(module)
        for dependency in graph.get(module, ()):
            if dependency in graph and dependency not in reached:
                stack.append(dependency)
            # Importing a submodule runs its package __init__, which in this
            # package re-exports; without this the census would call those
            # __init__ modules orphans.
            parent = dependency.rsplit(".", 1)[0]
            if parent in graph and parent not in reached:
                stack.append(parent)
    return reached


def census_differences(orphans: set[str], known: set[str]) -> tuple[set[str], set[str]]:
    """What the census must report: modules that appeared, and ones retired.

    Its own function so both directions can be driven with inputs. A mutation
    that stopped the "appeared" half from being computed survived when this
    was inline in the test below — nothing is newly unreachable today, so the
    assertion passed either way and the direction that matters was untested.
    """
    return orphans - known, known - orphans


def test_the_census_reports_a_module_that_became_unreachable():
    appeared, retired = census_differences({"cke.a", "cke.b"}, {"cke.a"})

    assert appeared == {"cke.b"}
    assert retired == set()


def test_the_census_reports_a_module_that_is_no_longer_an_orphan():
    appeared, retired = census_differences({"cke.a"}, {"cke.a", "cke.b"})

    assert appeared == set()
    assert retired == {"cke.b"}


def test_the_deleted_orphans_stay_deleted():
    """Imported by nothing, tested by nothing, and in the coverage denominator."""
    for gone in (
        "cke/sdk",
        "cke/utils/convergence_stats.py",
        "cke/utils/trust_stats.py",
    ):
        assert not (ROOT / gone).exists(), f"{gone} is back; it had no importer"


def test_the_unreachable_modules_are_exactly_the_recorded_ones():
    """Derived from the graph, so the list cannot quietly grow or go stale."""
    graph = _import_graph()
    orphans = {name for name in graph if name.startswith("cke")} - _reachable(graph)

    appeared, retired = census_differences(orphans, KNOWN_ORPHANS)
    assert appeared == set(), (
        f"{sorted(appeared)} became unreachable from every entry point. Either "
        f"wire it up or retire it; shipping it in the wheel where nothing can "
        f"reach it is the third option and the one this census exists to stop."
    )

    assert retired == set(), (
        f"{sorted(retired)} is reachable now, or gone. Remove it from "
        f"KNOWN_ORPHANS in this file so the count keeps meaning something."
    )
