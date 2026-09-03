"""Every contract component must honour all three obligations, at runtime.

The AST sweeps check shape: that a class carries the mixin and takes a strict
parameter. They cannot see a dataclass's generated __init__, which is how
TokenTracker sat in the contract unable to be constructed strict. This drives
each component into its degraded state and checks the behaviour instead.
"""

from __future__ import annotations

import logging
import pathlib
import tempfile

import pytest

from cke.diagnostics import DegradedComponentError, clear_runtime_state
from cke.models import Statement


@pytest.fixture(autouse=True)
def _clean_runtime_state():
    clear_runtime_state()
    yield
    clear_runtime_state()


@pytest.fixture
def bare(monkeypatch):
    """An environment with none of the optional dependencies."""
    from cke.entity_resolution import entity_resolver as er
    from cke.extractor import coreference_resolver as cr
    from cke.extractor import llm_extractor as le
    from cke.graph_engine import graph_engine as ge
    from cke.retrieval import embedding_model as em
    from cke.retrieval import faiss_index as fi
    from cke.schema import relation_mapper as rm

    monkeypatch.setattr(em, "SentenceTransformer", None)
    monkeypatch.setattr(em, "_GLOBAL_MODEL_CACHE", {})
    monkeypatch.setattr(em, "_FAILED_MODEL_LOADS", {})
    monkeypatch.setattr(fi, "faiss", None)
    monkeypatch.setattr(le, "OpenAI", None)
    monkeypatch.setattr(cr, "spacy", None)
    monkeypatch.setattr(rm, "yaml", None)
    monkeypatch.setattr(ge, "nx", None)
    monkeypatch.setattr(er, "fuzz", None)
    monkeypatch.setattr(er, "SentenceTransformer", None)
    monkeypatch.delenv("CKE_LLM_API_KEY", raising=False)


class _NoScoreRetriever:
    def retrieve(self, query, k=5):
        return [{"text": "a passage", "doc_id": "d1"}]


class _EmptyPack:
    graph_statements: list = []
    fallback_chunks = ["a chunk"]


class _FallbackRouter:
    def retrieve(self, *args, **kwargs):
        return _EmptyPack()


def _graph():
    from cke.graph_engine.graph_engine import KnowledgeGraphEngine

    engine = KnowledgeGraphEngine()
    engine.add_statement("Redis", "uses", "RESP")
    return engine


def _cases(tmp: pathlib.Path):
    """(name, build(strict), trigger_or_None) for every contract component."""
    from cke.datasets.hotpot_loader import HotpotDataset
    from cke.datasets.locomo_loader import LoCoMoDataset
    from cke.entity_resolution.entity_resolver import EntityResolver
    from cke.evaluation.ablation_runner import AblationRunner
    from cke.evaluation.token_counter import TokenCounter
    from cke.extractor.coreference_resolver import CoreferenceResolver
    from cke.extractor.llm_extractor import LLMExtractor
    from cke.graph.graph_store import GraphStore
    from cke.graph.trust_engine import TrustEngine
    from cke.graph_engine.graph_engine import KnowledgeGraphEngine
    from cke.observability.token_tracker import TokenTracker
    from cke.reasoning.llm_reasoner import LLMReasoner
    from cke.reasoning.path_reasoner import PathReasoner
    from cke.retrieval.default_evidence_retriever import DefaultEvidenceRetriever
    from cke.retrieval.dense_evidence_retriever import DenseEvidenceRetriever
    from cke.retrieval.embedding_model import EmbeddingModel
    from cke.retrieval.faiss_index import FaissIndex
    from cke.retrieval.hybrid_evidence_retriever import HybridEvidenceRetriever
    from cke.retrieval.rag_baseline import RAGRetriever
    from cke.schema.relation_mapper import RelationMapper
    from cke.storage.sqlite_store import SQLiteStore
    from cke.trust.confidence_model import ConfidenceModel

    assertion = {"subject": "A", "relation": "uses", "object": "B"}
    return [
        ("EmbeddingModel", lambda s: EmbeddingModel(strict=s), None),
        ("FaissIndex", lambda s: FaissIndex(strict=s), None),
        ("RAGRetriever", lambda s: RAGRetriever(strict=s), None),
        ("LLMExtractor", lambda s: LLMExtractor(strict=s), None),
        ("LLMReasoner", lambda s: LLMReasoner(strict=s), None),
        ("EntityResolver", lambda s: EntityResolver(strict=s), None),
        ("CoreferenceResolver", lambda s: CoreferenceResolver(strict=s), None),
        ("RelationMapper", lambda s: RelationMapper(strict=s), None),
        ("KnowledgeGraphEngine", lambda s: KnowledgeGraphEngine(strict=s), None),
        (
            "TrustEngine",
            lambda s: TrustEngine(config_path=tmp / "absent.yaml", strict=s),
            None,
        ),
        ("PathReasoner", lambda s: PathReasoner(strict=s), None),
        (
            "DenseEvidenceRetriever",
            lambda s: DenseEvidenceRetriever(_NoScoreRetriever(), strict=s),
            lambda o: o.retrieve("q"),
        ),
        (
            "HybridEvidenceRetriever",
            lambda s: HybridEvidenceRetriever(_FallbackRouter(), strict=s),
            lambda o: o.retrieve("q"),
        ),
        ("TokenTracker", lambda s: TokenTracker(strict=s), lambda o: o.to_dict()),
        (
            "SQLiteStore",
            lambda s: SQLiteStore(tmp / f"store-{s}.db", strict=s),
            lambda o: o._decode_context("{not json"),
        ),
        (
            "ConfidenceModel",
            lambda s: ConfidenceModel(strict=s),
            lambda o: o.predict(Statement("A", "uses", "B")),
        ),
        (
            "GraphStore",
            lambda s: GraphStore(strict=s),
            lambda o: o.add_assertion(assertion),
        ),
        (
            "HotpotDataset",
            lambda s: HotpotDataset(strict=s),
            lambda o: o._context_to_documents([["T", ["s"]], ["malformed"]]),
        ),
        (
            "LoCoMoDataset",
            lambda s: LoCoMoDataset(strict=s),
            lambda o: o._extract_turns({"unrecognised": []}),
        ),
        (
            "AblationRunner",
            lambda s: AblationRunner(
                evaluator=lambda item, variant: {"answer": "a"}, strict=s
            ),
            lambda o: o.run([{"question": "q"}], output_dir=tmp),
        ),
        (
            "DefaultEvidenceRetriever",
            lambda s: DefaultEvidenceRetriever(_graph(), strict=s),
            lambda o: o.retrieve("nothing at all matches this"),
        ),
        (
            # Driven into degradation by an encoding that cannot resolve,
            # which needs no network and no uninstall.
            "TokenCounter",
            lambda s: TokenCounter(encoding="no_such_encoding_xyz", strict=s),
            None,
        ),
    ]


def _names():
    with tempfile.TemporaryDirectory() as d:
        return [name for name, _, _ in _cases(pathlib.Path(d))]


@pytest.mark.parametrize("name", _names())
def test_component_honours_all_three_obligations(name, bare, caplog, tmp_path):
    """Warn naming the cause, flag the object, and refuse under strict."""
    case = next(c for c in _cases(tmp_path) if c[0] == name)
    _, build, trigger = case

    with caplog.at_level(logging.WARNING):
        component = build(False)
        if trigger:
            trigger(component)

    assert component.degraded is True, f"{name} did not set its degraded flag"
    assert component.degraded_reason, f"{name} degraded with no stated reason"
    assert "degraded" in caplog.text.lower(), f"{name} did not warn"

    clear_runtime_state()
    with pytest.raises(DegradedComponentError):
        strict_component = build(True)
        if trigger:
            trigger(strict_component)
