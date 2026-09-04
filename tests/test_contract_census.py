"""Every component that declares the contract must be held to it by name.

README said twenty-six components were under the contract. So did
``tests/test_contract_obligations.py``. Both were consistent with each other
and neither was consistent with ``cke/``, which had thirty-nine classes
declaring it. Prose agreeing with a test is not evidence about code, and
nothing noticed the other thirteen because nothing counted them.

This file counts them. It walks ``cke/`` for classes carrying
``DegradationMixin`` and requires each to appear in exactly one of the three
lists below, keyed by ``module:ClassName`` — qualified because there are two
distinct classes named ``GraphRetriever`` and only one of them had been
brought under the contract, which is precisely the kind of thing an unqualified
name hides.

Each list states what its members owe:

* ``DRIVEN_INTO_DEGRADATION`` — ``test_contract_obligations`` builds it,
  drives it into its own degraded state, and checks all three obligations.
* ``REFUSES_A_DEGRADED_COLLABORATOR`` — its own degradations are not
  reachable from a constructor, but it accepts components that can degrade,
  so a strict run must refuse those. Tested below.
* ``NO_DEGRADATION_PATH`` — it carries the flag so that a strict caller can
  read it, and never substitutes anything. Nothing to drive. A member that
  grows a ``_degrade`` or a ``require_strict_component`` call stops belonging
  here, and this file says so.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from cke.diagnostics import DegradedComponentError

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "cke"

#: Driven into its own degraded state by tests/test_contract_obligations.py.
DRIVEN_INTO_DEGRADATION = {
    "cke/conversation/ingestion.py:ConversationIngestionPipeline",
    "cke/conversation/retrieval/candidate_generation.py:CandidateGenerator",
    "cke/datasets/hotpot_loader.py:HotpotDataset",
    "cke/datasets/locomo_loader.py:LoCoMoDataset",
    "cke/datasets/musique_loader.py:MuSiQueDataset",
    "cke/datasets/wiki2_loader.py:WikiMultiHopDataset",
    "cke/entity_resolution/entity_resolver.py:EntityResolver",
    "cke/evaluation/ablation_runner.py:AblationRunner",
    "cke/evaluation/llm_qa.py:LLMAnswerer",
    "cke/evaluation/token_counter.py:TokenCounter",
    "cke/experiments/retrieval_eval_pipeline.py:MSMARCOCorpus",
    "cke/experiments/retrieval_eval_pipeline.py:DenseRetriever",
    "cke/extractor/coreference_resolver.py:CoreferenceResolver",
    "cke/extractor/llm_extractor.py:LLMExtractor",
    "cke/graph/graph_store.py:GraphStore",
    "cke/graph/trust_engine.py:TrustEngine",
    "cke/graph_engine/graph_engine.py:KnowledgeGraphEngine",
    "cke/observability/system_monitor.py:SystemMonitor",
    "cke/observability/token_tracker.py:TokenTracker",
    "cke/reasoning/llm_reasoner.py:LLMReasoner",
    "cke/reasoning/path_reasoner.py:PathReasoner",
    "cke/reasoning/reasoner_adapter.py:ReasonerAdapter",
    "cke/retrieval/default_evidence_retriever.py:DefaultEvidenceRetriever",
    "cke/retrieval/dense_evidence_retriever.py:DenseEvidenceRetriever",
    "cke/retrieval/embedding_model.py:EmbeddingModel",
    "cke/retrieval/evidence_retriever.py:EvidenceRetriever",
    "cke/retrieval/faiss_index.py:FaissIndex",
    "cke/retrieval/hybrid_evidence_retriever.py:HybridEvidenceRetriever",
    "cke/retrieval/rag_baseline.py:RAGRetriever",
    "cke/schema/relation_mapper.py:RelationMapper",
    "cke/storage/sqlite_store.py:SQLiteStore",
    "cke/trust/calibration.py:TrustCalibrator",
    "cke/trust/confidence_calibrator.py:ConfidenceCalibrator",
    "cke/trust/confidence_model.py:ConfidenceModel",
}

#: Composed of other components. A strict run must refuse a supplied
#: collaborator that has degraded or that cannot say whether it has.
REFUSES_A_DEGRADED_COLLABORATOR = {
    # (component, the slot the collaborator goes in, the label it is refused by)
    "cke/conversation/memory_store.py:ConversationMemoryStore": "graph engine",
    "cke/conversation/retriever.py:ConversationalRetriever": "memory store",
    "cke/evaluation/experiment_runner.py:ExperimentRunner": "retriever",
    "cke/experiments/reasoning_eval_pipeline.py:ReasoningEvalPipeline": "reasoner",
    "cke/extractor/extraction_pipeline.py:ExtractionPipeline": "graph engine",
    "cke/graph/deduplicator.py:AssertionDeduplicator": "trust engine",
    "cke/graph/update_pipeline.py:GraphUpdatePipeline": "entity resolver",
    "cke/pipeline/conversational_orchestrator.py:ConversationalOrchestrator": (
        "memory store"
    ),
    # QueryOrchestrator does declare two substitutions of its own, but both
    # are reached only part-way through answering a query, not from a
    # constructor. tests/test_undeclared_substitutions.py drives those.
    "cke/pipeline/query_orchestrator.py:QueryOrchestrator": "router",
    "cke/retrieval/graph_retriever.py:GraphRetriever": "entity resolver",
    "cke/retrieval/retriever.py:GraphRetriever": "router",
    "cke/router/query_router.py:QueryRouter": "domain classifier",
}

#: Carries the flag, substitutes nothing, composes nothing that can degrade.
NO_DEGRADATION_PATH = {
    "cke/graph/domain_classifier.py:DomainClassifier": (
        "It answers UNCLASSIFIED rather than guessing a domain, which is an "
        "answer and not a substitution, and its only collaborator is an "
        "optional embedding hook with no contract of its own."
    ),
}


def _classes():
    """(qualified name, ClassDef) for every class defined under cke/."""
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef):
                yield f"{relative}:{node.name}", node


def _declaring():
    return {
        key: node
        for key, node in _classes()
        if any(ast.unparse(base) == "DegradationMixin" for base in node.bases)
    }


def _calls(node: ast.ClassDef, name: str) -> int:
    return sum(
        1
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and (
            getattr(child.func, "attr", None) == name
            or getattr(child.func, "id", None) == name
        )
    )


def _obligation_case_names():
    import tempfile

    from tests import test_contract_obligations as obligations

    with tempfile.TemporaryDirectory() as directory:
        return {name for name, _, _ in obligations._cases(pathlib.Path(directory))}


def test_every_declaring_component_is_accounted_for():
    """A new contract component cannot be added without an entry here."""
    listed = (
        DRIVEN_INTO_DEGRADATION
        | set(REFUSES_A_DEGRADED_COLLABORATOR)
        | set(NO_DEGRADATION_PATH)
    )
    declaring = set(_declaring())

    assert declaring - listed == set(), (
        "these classes declare the degradation contract and no test holds "
        "them to it; add each to one of the three lists in this file"
    )
    assert (
        listed - declaring == set()
    ), "these names are listed here but no longer declare the contract"


def test_the_three_lists_do_not_overlap():
    pairs = (
        (DRIVEN_INTO_DEGRADATION, set(REFUSES_A_DEGRADED_COLLABORATOR)),
        (DRIVEN_INTO_DEGRADATION, set(NO_DEGRADATION_PATH)),
        (set(REFUSES_A_DEGRADED_COLLABORATOR), set(NO_DEGRADATION_PATH)),
    )
    for left, right in pairs:
        assert left & right == set(), "a component owes exactly one of the three"


def test_anything_taking_strict_can_say_whether_it_degraded():
    """``require_strict_component``'s degraded check is otherwise vacuous.

    A class that sets ``self.strict`` by hand has no ``degraded`` attribute,
    so an already-degraded instance of it passed the check that exists to
    refuse exactly that, and the component holding it still reported itself
    strict.
    """
    offenders = []
    for key, node in _classes():
        if key.endswith(":DegradationMixin"):
            continue
        declares = any(ast.unparse(b) == "DegradationMixin" for b in node.bases)
        if declares:
            continue
        for child in node.body:
            takes_strict = isinstance(child, ast.FunctionDef) and child.name in {
                "__init__",
                "__post_init__",
            }
            if takes_strict and any(
                arg.arg == "strict" for arg in child.args.args + child.args.kwonlyargs
            ):
                offenders.append(key)
            if (
                isinstance(child, ast.AnnAssign)
                and getattr(child.target, "id", None) == "strict"
            ):
                offenders.append(key)
    assert offenders == [], (
        "these take a strict parameter without inheriting DegradationMixin, "
        "so they cannot report whether they degraded"
    )


def test_the_driven_list_matches_the_obligations_test():
    """The two files must name the same components, or one of them is stale."""
    driven = {key.split(":", 1)[1] for key in DRIVEN_INTO_DEGRADATION}
    assert driven == _obligation_case_names()


def test_a_subclass_that_adds_a_substitution_is_listed_too():
    """Inheriting the contract inherits the entry; adding a fallback does not."""
    declaring = _declaring()
    parents = {key.split(":", 1)[1] for key in declaring}
    for key, node in _classes():
        if key in declaring:
            continue
        if not any(ast.unparse(base) in parents for base in node.bases):
            continue
        assert _calls(node, "_degrade") == 0, (
            f"{key} inherits the contract and declares a substitution of its "
            f"own, so it needs its own entry in this file"
        )


@pytest.mark.parametrize("key", sorted(NO_DEGRADATION_PATH))
def test_nothing_with_a_fallback_is_parked_in_the_exempt_list(key):
    node = _declaring()[key]
    assert _calls(node, "_degrade") == 0, (
        f"{key} declares a substitution; move it to DRIVEN_INTO_DEGRADATION "
        f"and write a case that drives it"
    )
    assert _calls(node, "require_strict_component") == 0, (
        f"{key} guards a collaborator; move it to " f"REFUSES_A_DEGRADED_COLLABORATOR"
    )


class _Degraded:
    """A collaborator that declares itself strict and has already degraded."""

    strict = True
    degraded = True
    degraded_reason = "it ran without the library it measures against"


class _CannotSay:
    """A collaborator with a strict flag and no way to report degradation."""

    strict = True


def _build(key, collaborator, strict=True):
    """Construct *key*'s class with *collaborator* in its guarded slot."""
    from cke.conversation.memory_store import ConversationMemoryStore
    from cke.conversation.retriever import ConversationalRetriever
    from cke.evaluation.experiment_runner import ExperimentRunner
    from cke.experiments.reasoning_eval_pipeline import ReasoningEvalPipeline
    from cke.extractor.extraction_pipeline import ExtractionPipeline
    from cke.graph.deduplicator import AssertionDeduplicator
    from cke.graph.update_pipeline import GraphUpdatePipeline
    from cke.pipeline.conversational_orchestrator import ConversationalOrchestrator
    from cke.pipeline.query_orchestrator import QueryOrchestrator
    from cke.retrieval.graph_retriever import GraphRetriever
    from cke.retrieval.retriever import GraphRetriever as SimpleGraphRetriever
    from cke.router.query_router import QueryRouter

    # Every one of these guards before it builds anything, so None is enough
    # for the arguments the refusal never reaches. If that stops being true
    # the construction raises something other than DegradedComponentError and
    # the test says so rather than passing.
    builders = {
        "cke/conversation/memory_store.py:ConversationMemoryStore": (
            lambda: ConversationMemoryStore(graph_engine=collaborator, strict=strict)
        ),
        "cke/conversation/retriever.py:ConversationalRetriever": (
            lambda: ConversationalRetriever(collaborator, strict=strict)
        ),
        "cke/evaluation/experiment_runner.py:ExperimentRunner": (
            lambda: ExperimentRunner(collaborator, strict=strict)
        ),
        "cke/experiments/reasoning_eval_pipeline.py:ReasoningEvalPipeline": (
            lambda: ReasoningEvalPipeline(reasoner=collaborator, strict=strict)
        ),
        "cke/extractor/extraction_pipeline.py:ExtractionPipeline": (
            lambda: ExtractionPipeline(collaborator, strict=strict)
        ),
        "cke/graph/deduplicator.py:AssertionDeduplicator": (
            lambda: AssertionDeduplicator(trust_engine=collaborator, strict=strict)
        ),
        "cke/graph/update_pipeline.py:GraphUpdatePipeline": (
            lambda: GraphUpdatePipeline(None, resolver=collaborator, strict=strict)
        ),
        "cke/pipeline/conversational_orchestrator.py:ConversationalOrchestrator": (
            lambda: ConversationalOrchestrator(memory_store=collaborator, strict=strict)
        ),
        "cke/pipeline/query_orchestrator.py:QueryOrchestrator": (
            lambda: QueryOrchestrator(None, router=collaborator, strict=strict)
        ),
        "cke/retrieval/graph_retriever.py:GraphRetriever": (
            lambda: GraphRetriever(None, entity_resolver=collaborator, strict=strict)
        ),
        "cke/retrieval/retriever.py:GraphRetriever": (
            lambda: SimpleGraphRetriever(None, router=collaborator, strict=strict)
        ),
        "cke/router/query_router.py:QueryRouter": (
            lambda: QueryRouter(domain_classifier=collaborator, strict=strict)
        ),
    }
    return builders[key]()


@pytest.mark.parametrize("key", sorted(REFUSES_A_DEGRADED_COLLABORATOR))
def test_a_strict_component_refuses_a_degraded_collaborator(key):
    label = REFUSES_A_DEGRADED_COLLABORATOR[key]
    name = key.split(":", 1)[1]

    with pytest.raises(DegradedComponentError) as raised:
        _build(key, _Degraded())

    message = str(raised.value)
    assert message.startswith(name), f"the refusal did not name {name}: {message}"
    assert label in message, f"the refusal did not name the {label}: {message}"
    assert "already degraded" in message


@pytest.mark.parametrize("key", sorted(REFUSES_A_DEGRADED_COLLABORATOR))
def test_a_strict_component_refuses_a_collaborator_that_cannot_report(key):
    """The check above is vacuous against an object with no degraded flag."""
    label = REFUSES_A_DEGRADED_COLLABORATOR[key]
    name = key.split(":", 1)[1]

    with pytest.raises(DegradedComponentError) as raised:
        _build(key, _CannotSay())

    message = str(raised.value)
    assert message.startswith(name), f"the refusal did not name {name}: {message}"
    assert label in message
    assert "cannot report whether it has degraded" in message


@pytest.mark.parametrize("key", sorted(REFUSES_A_DEGRADED_COLLABORATOR))
def test_a_non_strict_component_accepts_what_a_strict_one_refuses(key):
    """A guard that refused everything would pass the two tests above.

    strict=False is a caller saying they accept degraded behaviour. Refusing
    there turns the contract into a ban, and both tests above would still be
    green, because both construct strict.
    """
    try:
        _build(key, _Degraded(), strict=False)
    except DegradedComponentError as error:  # pragma: no cover - the failure
        pytest.fail(f"a non-strict {key} refused a degraded collaborator: {error}")
    except Exception:
        # Building the rest of the component needs arguments this test does
        # not supply. Getting past the guard is the whole assertion.
        pass
