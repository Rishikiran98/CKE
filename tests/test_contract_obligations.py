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


class _HealthyGraphEngine:
    """Stands in for a graph engine that has not degraded.

    The conversation cases below are about the pipeline and the candidate
    generator. Letting them build a real KnowledgeGraphEngine means networkx
    being absent refuses first, and the case proves nothing about the
    component it names.
    """

    strict = True
    degraded = False
    degraded_reason = ""

    def add_statement(self, *args, **kwargs):
        return None


class _DegradedEmbedder:
    """An embedder that declares itself strict and has already degraded.

    Supplied so that the two retrievers below are shown to refuse. Left to
    build their own, whichever inner component is constructed first refuses,
    and the case says nothing about the retriever it names.
    """

    strict = True
    degraded = True
    degraded_reason = "sentence-transformers was absent when it was built"
    model_name = "stub-embedder"
    model_revision = None

    def embed_text(self, text):
        return [0.0, 0.0]

    def embed_texts(self, texts):
        return [[0.0, 0.0] for _ in texts]


class _RaisingExtractor:
    """An extractor that fails on every turn, losing that turn's candidates."""

    def extract(self, *args, **kwargs):
        raise ValueError("this extractor cannot run")


class _ShortEmbedder:
    """An embedder that returns fewer vectors than it was given texts."""

    strict = True
    degraded = False
    degraded_reason = ""

    def embed_text(self, text):
        return [0.0, 0.0]

    def embed_texts(self, texts):
        return []


class _BareReasoner:
    """A reasoner that answers with a string and no confidence of its own."""

    def answer(self, query, context):
        return "Turkey"


class _OneChunkRetriever:
    def retrieve(self, query, k=5):
        return [{"doc_id": "c1", "text": "Redis uses RESP", "score": 0.9}]


def _conversation_store(strict):
    """A memory store holding one event, so retrieval has something to embed."""
    from cke.conversation.memory_store import ConversationMemoryStore
    from cke.conversation.types import ConversationEvent

    store = ConversationMemoryStore(graph_engine=_HealthyGraphEngine(), strict=strict)
    store.store_event(
        ConversationEvent(
            conversation_id="c1",
            event_id="e1",
            turn_id="t1",
            turn_order=1,
            role="user",
            text="I moved to Berlin",
            timestamp="2026-01-01T00:00:00Z",
        )
    )
    return store


def _chunk_facts():
    """A chunk store whose statement carries no trust score of its own."""
    from cke.retrieval.chunk_fact_store import ChunkFactStore

    store = ChunkFactStore()
    store.add_facts("c1", [Statement("Redis", "uses", "RESP")])
    return store


def _short_corpus(tmp, name):
    """An MS MARCO TSV with one usable row and one that has too few columns."""
    path = tmp / name
    path.write_text("d1\thttp://example/1\tTitle\tBody\nd2\ttwo-columns-only\n")
    return path


def _graph():
    from cke.graph_engine.graph_engine import KnowledgeGraphEngine

    engine = KnowledgeGraphEngine()
    engine.add_statement("Redis", "uses", "RESP")
    return engine


def _cases(tmp: pathlib.Path):
    """(name, build(strict), trigger_or_None) for every contract component."""
    from cke.datasets.hotpot_loader import HotpotDataset
    from cke.datasets.locomo_loader import LoCoMoDataset
    from cke.datasets.musique_loader import MuSiQueDataset
    from cke.datasets.wiki2_loader import WikiMultiHopDataset
    from cke.entity_resolution.entity_resolver import EntityResolver
    from cke.evaluation.ablation_runner import AblationRunner
    from cke.evaluation.token_counter import TokenCounter
    from cke.evaluation.llm_qa import LLMAnswerer
    from cke.experiments.retrieval_eval_pipeline import DenseRetriever
    from cke.extractor.coreference_resolver import CoreferenceResolver
    from cke.extractor.llm_extractor import LLMExtractor
    from cke.graph.graph_store import GraphStore
    from cke.graph.trust_engine import TrustEngine
    from cke.graph_engine.graph_engine import KnowledgeGraphEngine
    from cke.observability.token_tracker import TokenTracker
    from cke.reasoning.llm_reasoner import LLMReasoner
    from cke.reasoning.reasoner import TemplateReasoner
    from cke.reasoning.path_reasoner import PathReasoner
    from cke.retrieval.default_evidence_retriever import DefaultEvidenceRetriever
    from cke.retrieval.dense_evidence_retriever import DenseEvidenceRetriever
    from cke.retrieval.embedding_model import EmbeddingModel
    from cke.retrieval.faiss_index import FaissIndex
    from cke.retrieval.hybrid_evidence_retriever import HybridEvidenceRetriever
    from cke.retrieval.rag_baseline import RAGRetriever
    from cke.schema.relation_mapper import RelationMapper
    from cke.storage.sqlite_store import SQLiteStore
    from cke.conversation.ingestion import ConversationIngestionPipeline
    from cke.conversation.memory_store import ConversationMemoryStore
    from cke.conversation.retrieval.candidate_generation import CandidateGenerator
    from cke.experiments.retrieval_eval_pipeline import MSMARCOCorpus
    from cke.observability.system_monitor import SystemMonitor
    from cke.reasoning.reasoner_adapter import ReasonerAdapter
    from cke.retrieval.evidence_retriever import EvidenceRetriever
    from cke.schema.assertion import Assertion
    from cke.trust.calibration import TrustCalibrator
    from cke.trust.confidence_calibrator import ConfidenceCalibrator
    from cke.trust.confidence_model import ConfidenceModel

    assertion = {"subject": "A", "relation": "uses", "object": "B"}
    return [
        ("EmbeddingModel", lambda s: EmbeddingModel(strict=s), None),
        ("FaissIndex", lambda s: FaissIndex(strict=s), None),
        (
            "RAGRetriever",
            lambda s: RAGRetriever(embedding_model=_DegradedEmbedder(), strict=s),
            None,
        ),
        (
            "DenseRetriever",
            lambda s: DenseRetriever(embedding_model=_DegradedEmbedder(), strict=s),
            None,
        ),
        ("LLMExtractor", lambda s: LLMExtractor(strict=s), None),
        (
            # The fallback is supplied rather than built, so this case is
            # about the missing API key. Left to default, LLMReasoner builds a
            # PathReasoner whose embedding model refuses first, and the case
            # never showed that a strict run refuses to answer without an LLM.
            "LLMReasoner",
            lambda s: LLMReasoner(fallback=TemplateReasoner(), strict=s),
            None,
        ),
        ("EntityResolver", lambda s: EntityResolver(strict=s), None),
        ("CoreferenceResolver", lambda s: CoreferenceResolver(strict=s), None),
        ("RelationMapper", lambda s: RelationMapper(strict=s), None),
        ("KnowledgeGraphEngine", lambda s: KnowledgeGraphEngine(strict=s), None),
        (
            "TrustEngine",
            lambda s: TrustEngine(config_path=tmp / "absent.yaml", strict=s),
            None,
        ),
        (
            "PathReasoner",
            lambda s: PathReasoner(embedding_model=_DegradedEmbedder(), strict=s),
            None,
        ),
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
            lambda o: o._conversation_documents({"speaker_a": "A"}),
        ),
        (
            "WikiMultiHopDataset",
            lambda s: WikiMultiHopDataset(strict=s),
            lambda o: o._context_to_documents([["T", ["s"]], ["malformed"]]),
        ),
        (
            "MuSiQueDataset",
            lambda s: MuSiQueDataset(strict=s),
            lambda o: o._paragraphs_to_documents(
                [{"idx": 0, "title": "T", "paragraph_text": "text"}, "malformed"]
            ),
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
        (
            # The api backend with the key explicitly absent: no network, no
            # download, and the exact state a strict benchmark must refuse in.
            "LLMAnswerer",
            lambda s: LLMAnswerer(backend="api", api_key=None, strict=s),
            None,
        ),
        (
            "MSMARCOCorpus",
            lambda s: MSMARCOCorpus(_short_corpus(tmp, f"corpus-{s}.tsv"), strict=s),
            None,
        ),
        (
            # A metric recorder handed a value it cannot count. The figure it
            # feeds undercounts, and a snapshot could not previously say so.
            "SystemMonitor",
            lambda s: SystemMonitor(strict=s),
            lambda o: o.record_retrieval("not a number"),
        ),
        (
            "ConversationIngestionPipeline",
            lambda s: ConversationIngestionPipeline(
                ConversationMemoryStore(graph_engine=_HealthyGraphEngine(), strict=s),
                extractors=[_RaisingExtractor()],
                strict=s,
            ),
            lambda o: o.ingest_turn("c1", "user", "I moved to Berlin"),
        ),
        (
            "CandidateGenerator",
            lambda s: CandidateGenerator(
                _conversation_store(s), embedding_model=_ShortEmbedder(), strict=s
            ),
            lambda o: o.generate("where did I move", conversation_id="c1"),
        ),
        (
            "ReasonerAdapter",
            lambda s: ReasonerAdapter(_BareReasoner(), strict=s),
            lambda o: o.reason(
                "where is it?", [Statement("Hagia Sophia", "located in", "Istanbul")]
            ),
        ),
        (
            # config_path=None selects the built-in weights deliberately and is
            # not a degradation; the trust substitution below is.
            "EvidenceRetriever",
            lambda s: EvidenceRetriever(
                _OneChunkRetriever(), _chunk_facts(), config_path=None, strict=s
            ),
            lambda o: o.retrieve("what does redis use"),
        ),
        (
            "TrustCalibrator",
            lambda s: TrustCalibrator(strict=s),
            lambda o: o.fit_from_graph(
                [Assertion(subject="A", relation="uses", object="B")]
            ),
        ),
        (
            "ConfidenceCalibrator",
            lambda s: ConfidenceCalibrator(strict=s),
            lambda o: o.calibrate({}),
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
    with pytest.raises(DegradedComponentError) as raised:
        strict_component = build(True)
        if trigger:
            trigger(strict_component)

    # Which component refused matters. Several of these build collaborators
    # that can degrade for reasons of their own, so a bare `raises` passes
    # when the component under test never refused anything and something it
    # constructed did.
    refused = str(raised.value)
    assert refused.startswith(name), (
        f"{name} was expected to refuse under strict, but the refusal came "
        f"from elsewhere: {refused}"
    )
