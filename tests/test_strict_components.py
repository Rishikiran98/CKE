"""Every component with a degradation path must honour strict mode.

These tests exist because a benchmark once ran on a SHA256 hashing embedder
while reporting itself as a dense retrieval baseline. Each case here asserts
that the same substitution now either announces itself or refuses to happen.
"""

from __future__ import annotations

import logging

import pytest

from cke.diagnostics import DegradedComponentError, clear_runtime_state

#: A revision with the shape of a pin, for stubbed models that have no Hub
#: page. Loaders refuse anything that is not a full commit hash.
_A_COMMIT = "0" * 40


@pytest.fixture(autouse=True)
def _clean_runtime_state():
    clear_runtime_state()
    yield
    clear_runtime_state()


# ---------------------------------------------------------------------------
# Embedding model — the fallback that invalidated the prior results
# ---------------------------------------------------------------------------


def test_embedding_model_declares_the_hash_fallback(monkeypatch, caplog):
    from cke.retrieval import embedding_model as module

    monkeypatch.setattr(module, "SentenceTransformer", None)
    monkeypatch.setattr(module, "_GLOBAL_MODEL_CACHE", {})

    with caplog.at_level(logging.WARNING):
        model = module.EmbeddingModel()

    assert model.degraded is True
    assert "sentence-transformers" in model.degraded_reason
    assert "hash" in model.degraded_reason.lower()
    # The reason must say why it matters, not merely that it happened.
    assert "meaningless" in model.degraded_reason
    assert "pip install sentence-transformers" in caplog.text


def test_embedding_model_strict_refuses_to_hash(monkeypatch):
    from cke.retrieval import embedding_model as module

    monkeypatch.setattr(module, "SentenceTransformer", None)
    monkeypatch.setattr(module, "_GLOBAL_MODEL_CACHE", {})

    with pytest.raises(DegradedComponentError):
        module.EmbeddingModel(strict=True)


def test_a_failed_model_load_is_not_cached_as_the_fallback(monkeypatch):
    """A transient load failure must not pin the process to hashing."""
    from cke.retrieval import embedding_model as module

    cache: dict = {}
    monkeypatch.setattr(module, "_GLOBAL_MODEL_CACHE", cache)

    def _fail(name, revision=None):
        raise OSError("network unreachable")

    monkeypatch.setattr(module, "SentenceTransformer", _fail)

    degraded = module.EmbeddingModel()
    assert degraded.degraded is True
    assert cache == {}, "a failed load must not be cached"

    # A later strict construction still sees the real cause and refuses.
    with pytest.raises(DegradedComponentError) as excinfo:
        module.EmbeddingModel(strict=True)
    assert "network unreachable" in str(excinfo.value)


def test_empty_input_uses_the_real_dimension(monkeypatch):
    """The fallback width must not be hardcoded into the healthy path."""
    from cke.retrieval import embedding_model as module

    class FakeModel:
        def get_sentence_embedding_dimension(self):
            return 384

        def encode(self, texts, **kwargs):
            import numpy as np

            return np.zeros((len(texts), 384), dtype="float32")

    monkeypatch.setattr(module, "_GLOBAL_MODEL_CACHE", {("m", _A_COMMIT): FakeModel()})
    model = module.EmbeddingModel(model_name="m", model_revision=_A_COMMIT)

    assert model.embed_texts([]).shape == (0, 384)


# ---------------------------------------------------------------------------
# LLM extractor — the regex substitution
# ---------------------------------------------------------------------------


def test_llm_extractor_declares_the_regex_fallback(monkeypatch, caplog):
    from cke.extractor import llm_extractor as module

    monkeypatch.delenv("CKE_LLM_API_KEY", raising=False)
    monkeypatch.setattr(module, "OpenAI", None)

    with caplog.at_level(logging.WARNING):
        extractor = module.LLMExtractor()

    assert extractor.degraded is True
    assert "no LLM is in the loop" in extractor.degraded_reason


def test_llm_extractor_distinguishes_missing_package_from_missing_key(monkeypatch):
    from cke.extractor import llm_extractor as module

    monkeypatch.delenv("CKE_LLM_API_KEY", raising=False)
    monkeypatch.setattr(module, "OpenAI", None)
    assert "not installed" in module.LLMExtractor().degraded_reason

    monkeypatch.setattr(module, "OpenAI", lambda **kwargs: object())
    assert "no API key" in module.LLMExtractor().degraded_reason


def test_llm_extractor_strict_refuses(monkeypatch):
    from cke.extractor import llm_extractor as module

    monkeypatch.delenv("CKE_LLM_API_KEY", raising=False)
    monkeypatch.setattr(module, "OpenAI", None)

    with pytest.raises(DegradedComponentError):
        module.LLMExtractor(strict=True)


# ---------------------------------------------------------------------------
# Index, baseline, and the rest
# ---------------------------------------------------------------------------


def test_faiss_index_declares_the_numpy_scan(monkeypatch):
    from cke.retrieval import faiss_index as module

    monkeypatch.setattr(module, "faiss", None)
    index = module.FaissIndex()

    assert index.degraded is True
    assert "latency" in index.degraded_reason

    with pytest.raises(DegradedComponentError):
        module.FaissIndex(strict=True)


def test_rag_retriever_inherits_its_components_degradation(monkeypatch):
    """The dense baseline must not report itself healthy on a hashed embedder."""
    from cke.retrieval import embedding_model as embed_module
    from cke.retrieval import faiss_index as faiss_module
    from cke.retrieval import rag_baseline as module

    monkeypatch.setattr(embed_module, "SentenceTransformer", None)
    monkeypatch.setattr(embed_module, "_GLOBAL_MODEL_CACHE", {})
    monkeypatch.setattr(faiss_module, "faiss", None)

    retriever = module.RAGRetriever()

    assert retriever.degraded is True
    assert "not a dense retrieval baseline" in retriever.degraded_reason

    with pytest.raises(DegradedComponentError):
        module.RAGRetriever(strict=True)


def test_rag_retriever_tolerates_a_caller_supplied_embedder():
    """A duck-typed embedder need not implement the contract."""
    from cke.retrieval import rag_baseline as module

    class Stub:
        def embed_texts(self, texts, **kwargs):
            import numpy as np

            return np.zeros((len(list(texts)), 8), dtype="float32")

        def embed_text(self, text):
            import numpy as np

            return np.zeros((8,), dtype="float32")

    retriever = module.RAGRetriever(embedding_model=Stub())
    assert retriever.embedding_model.__class__ is Stub


def test_relation_mapper_declares_the_two_relation_ontology(monkeypatch):
    from cke.schema import relation_mapper as module

    monkeypatch.setattr(module, "yaml", None)
    mapper = module.RelationMapper()

    assert mapper.degraded is True
    assert sorted(mapper.relations) == ["acted_in", "directed"]

    with pytest.raises(DegradedComponentError):
        module.RelationMapper(strict=True)


def test_coreference_resolver_declares_the_regex_fallback(monkeypatch):
    from cke.extractor import coreference_resolver as module

    monkeypatch.setattr(module, "spacy", None)
    resolver = module.CoreferenceResolver()

    assert resolver.degraded is True
    with pytest.raises(DegradedComponentError):
        module.CoreferenceResolver(strict=True)


def test_graph_engine_declares_the_handrolled_graph(monkeypatch):
    from cke.graph_engine import graph_engine as module

    monkeypatch.setattr(module, "nx", None)
    engine = module.KnowledgeGraphEngine()

    assert engine.degraded is True
    with pytest.raises(DegradedComponentError):
        module.KnowledgeGraphEngine(strict=True)


def test_llm_reasoner_declares_the_template_fallback(monkeypatch):
    from cke.reasoning import llm_reasoner as module

    monkeypatch.delenv("CKE_LLM_API_KEY", raising=False)
    reasoner = module.LLMReasoner()

    assert reasoner.degraded is True
    assert "no LLM is in the loop" in reasoner.degraded_reason

    with pytest.raises(DegradedComponentError):
        module.LLMReasoner(strict=True)


def test_trust_engine_declares_a_missing_config(tmp_path):
    from cke.graph.trust_engine import TrustEngine

    engine = TrustEngine(config_path=tmp_path / "absent.yaml")
    assert engine.degraded is True

    with pytest.raises(DegradedComponentError):
        TrustEngine(config_path=tmp_path / "absent.yaml", strict=True)


def test_trust_engine_opting_out_of_config_is_not_a_degradation():
    from cke.graph.trust_engine import TrustEngine

    assert TrustEngine(config_path=None).degraded is False


def test_trust_engine_reads_the_configured_tau(tmp_path):
    """A non-None tau default used to overwrite the configured value."""
    from cke.graph.trust_engine import TrustEngine

    config = tmp_path / "trust.yaml"
    config.write_text("tau: 12345.0\nw_src: 0.5\n", encoding="utf-8")

    engine = TrustEngine(config_path=config)

    assert engine.degraded is False
    assert engine.calibrator.config.tau == 12345.0
    assert engine.calibrator.config.w_src == 0.5


def test_trust_engine_explicit_tau_still_wins(tmp_path):
    from cke.graph.trust_engine import TrustEngine

    config = tmp_path / "trust.yaml"
    config.write_text("tau: 12345.0\n", encoding="utf-8")

    assert TrustEngine(tau=7.0, config_path=config).calibrator.config.tau == 7.0


def test_ranking_config_declares_a_missing_file(tmp_path):
    from cke.retrieval.ranking_config import load_ranking_config

    load_ranking_config(tmp_path / "absent.yaml")

    with pytest.raises(DegradedComponentError):
        load_ranking_config(tmp_path / "absent.yaml", strict=True)


def test_ranking_config_declares_a_typo_in_a_section(tmp_path):
    """A file that parses but names no known section is not a successful load."""
    from cke.retrieval.ranking_config import load_ranking_config

    config = tmp_path / "ranking.yaml"
    config.write_text("chunkk:\n  dense_weight: 0.9\n", encoding="utf-8")

    with pytest.raises(DegradedComponentError):
        load_ranking_config(config, strict=True)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_entity_resolver_fallback_is_deterministic(monkeypatch):
    """The fallback used builtin hash(), which is salted per process."""
    from cke.entity_resolution import entity_resolver as module

    monkeypatch.setattr(module, "SentenceTransformer", None)

    first = module.EntityResolver()._embed("Stanford University research")
    second = module.EntityResolver()._embed("Stanford University research")

    assert first == second
    # Pin the values against the process-salted implementation: a SHA256 digest
    # of a fixed token always lands in the same bucket.
    import hashlib

    expected = int(hashlib.sha256(b"stanford").hexdigest(), 16) % 128
    assert first[expected] > 0


def test_entity_resolver_strict_refuses_hashed_similarity(monkeypatch):
    from cke.entity_resolution import entity_resolver as module

    monkeypatch.setattr(module, "SentenceTransformer", None)

    with pytest.raises(DegradedComponentError):
        module.EntityResolver(strict=True)._embed("anything")


# ---------------------------------------------------------------------------
# Evaluation entry points
# ---------------------------------------------------------------------------


def test_reasoning_eval_pipeline_is_strict_by_default(monkeypatch):
    """It printed the environment report, then ran on a hashed embedder anyway."""
    from cke.experiments import reasoning_eval_pipeline as module
    from cke.retrieval import embedding_model as embed_module

    monkeypatch.setattr(embed_module, "SentenceTransformer", None)
    monkeypatch.setattr(embed_module, "_GLOBAL_MODEL_CACHE", {})

    with pytest.raises(DegradedComponentError):
        module.ReasoningEvalPipeline()

    # The opt-out still works.
    assert module.ReasoningEvalPipeline(strict=False) is not None


def test_run_eval_does_not_retry_a_factory_that_raises_internally():
    """A TypeError from inside a factory must not be read as an unsupported kwarg."""
    from cke.evaluation.run_eval import _accepts_strict

    def takes_strict(strict=False):
        raise TypeError("a real bug inside the factory")

    def takes_nothing():
        return "orchestrator"

    def takes_kwargs(**kwargs):
        return "orchestrator"

    assert _accepts_strict(takes_strict) is True
    assert _accepts_strict(takes_nothing) is False
    assert _accepts_strict(takes_kwargs) is True
    # No inspectable signature: fall back to calling without the keyword.
    assert _accepts_strict(len) is False


def test_entity_resolver_declares_the_fuzzy_fallback_once(monkeypatch):
    """The declaration used to run once per candidate entity."""
    from cke.entity_resolution import entity_resolver as module

    monkeypatch.setattr(module, "fuzz", None)

    resolver = module.EntityResolver()
    before = len(resolver.degraded_reason)
    resolver._best_fuzzy("Entity 7", [f"Entity {i}" for i in range(60)])

    assert len(resolver.degraded_reason) == before
    assert "rapidfuzz" in resolver.degraded_reason


def test_entity_resolver_strict_refuses_without_rapidfuzz(monkeypatch):
    from cke.entity_resolution import entity_resolver as module

    monkeypatch.setattr(module, "fuzz", None)

    with pytest.raises(DegradedComponentError, match="rapidfuzz"):
        module.EntityResolver(strict=True)


def test_llm_reasoner_passes_strict_to_its_fallback(monkeypatch):
    """A strict reasoner built a non-strict PathReasoner, so a strict run
    still reached the hashed embedder."""
    from cke.reasoning import llm_reasoner as module
    from cke.retrieval import embedding_model as embed_module

    monkeypatch.setattr(embed_module, "SentenceTransformer", None)
    monkeypatch.setattr(embed_module, "_GLOBAL_MODEL_CACHE", {})
    monkeypatch.setattr(embed_module, "_FAILED_MODEL_LOADS", {})
    monkeypatch.setenv("CKE_LLM_API_KEY", "a-key-so-the-key-path-is-not-the-cause")

    with pytest.raises(DegradedComponentError, match="sentence-transformers"):
        module.LLMReasoner(strict=True)


def test_llm_extractor_declares_an_empty_llm_response(monkeypatch):
    """The success path substituted regex output when the LLM returned nothing."""
    from cke.extractor import llm_extractor as module

    extractor = module.LLMExtractor(config=module.LLMConfig(api_key="k"))
    extractor.client = object()
    extractor.degraded = False
    extractor.degraded_reason = ""

    monkeypatch.setattr(extractor, "_call_llm", lambda text: {})
    monkeypatch.setattr(extractor, "_parse_response", lambda payload, source_text: [])

    extractor.extract("Redis supports PubSub messaging.")

    assert extractor.degraded is True
    assert "no valid assertions" in extractor.degraded_reason


def test_orchestrator_does_not_swallow_a_strict_refusal():
    """DegradedComponentError subclasses RuntimeError, so a broad except
    caught it and scored a refused run as an abstention."""
    import inspect

    from cke.pipeline import query_orchestrator as module

    source = inspect.getsource(module.QueryOrchestrator._run_reasoner)
    assert "except DegradedComponentError:" in source
    assert source.index("except DegradedComponentError:") < source.index(
        "except Exception:"
    )


def test_failed_model_load_is_attempted_once_per_process(monkeypatch):
    """The benchmark builds an embedder per item; retrying the load each time
    turned one failure into one download attempt per question."""
    from cke.retrieval import embedding_model as module

    attempts = []

    def _fail(name, revision=None):
        attempts.append((name, revision))
        raise OSError("network unreachable")

    monkeypatch.setattr(module, "SentenceTransformer", _fail)
    monkeypatch.setattr(module, "_GLOBAL_MODEL_CACHE", {})
    monkeypatch.setattr(module, "_FAILED_MODEL_LOADS", {})

    for _ in range(20):
        module.EmbeddingModel()

    assert attempts == [
        (module.DEFAULT_EMBEDDING_MODEL, module.DEFAULT_EMBEDDING_REVISION)
    ]

    # The recorded cause is still available to a later strict construction.
    with pytest.raises(DegradedComponentError, match="network unreachable"):
        module.EmbeddingModel(strict=True)


def test_dimension_is_measured_when_a_healthy_model_reports_none(monkeypatch):
    """Reporting the hashed fallback width for a healthy model is misleading."""
    import numpy as np

    from cke.retrieval import embedding_model as module

    class QuietModel:
        def get_sentence_embedding_dimension(self):
            return None

        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 512), dtype="float32")

    monkeypatch.setattr(
        module, "_GLOBAL_MODEL_CACHE", {("quiet", _A_COMMIT): QuietModel()}
    )
    monkeypatch.setattr(module, "_FAILED_MODEL_LOADS", {})

    quiet = module.EmbeddingModel(model_name="quiet", model_revision=_A_COMMIT)
    assert quiet.dimension == 512


def test_a_strict_pipeline_refuses_a_non_strict_injected_reasoner(monkeypatch):
    """A prebuilt reasoner bypassed strict entirely: the pipeline reported
    itself strict while running on a hashed embedder."""
    from cke.experiments import reasoning_eval_pipeline as module
    from cke.reasoning.path_reasoner import PathReasoner
    from cke.retrieval import embedding_model as embed_module

    monkeypatch.setattr(embed_module, "SentenceTransformer", None)
    monkeypatch.setattr(embed_module, "_GLOBAL_MODEL_CACHE", {})
    monkeypatch.setattr(embed_module, "_FAILED_MODEL_LOADS", {})

    weak = PathReasoner(strict=False)
    assert weak.degraded is True

    with pytest.raises(DegradedComponentError, match="already degraded"):
        module.ReasoningEvalPipeline(reasoner=weak, strict=True)

    # Opting out still accepts it.
    assert module.ReasoningEvalPipeline(reasoner=weak, strict=False) is not None


def test_path_reasoner_inherits_its_embedders_degradation(monkeypatch):
    from cke.reasoning.path_reasoner import PathReasoner
    from cke.retrieval import embedding_model as embed_module

    monkeypatch.setattr(embed_module, "SentenceTransformer", None)
    monkeypatch.setattr(embed_module, "_GLOBAL_MODEL_CACHE", {})
    monkeypatch.setattr(embed_module, "_FAILED_MODEL_LOADS", {})

    reasoner = PathReasoner()
    assert reasoner.degraded is True
    assert "embedding model is degraded" in reasoner.degraded_reason

    with pytest.raises(DegradedComponentError):
        PathReasoner(strict=True)


def test_graph_retriever_propagates_strict(monkeypatch):
    """Every graph evaluation path built a non-strict EntityResolver."""
    from cke.entity_resolution import entity_resolver as resolver_module
    from cke.graph_engine.graph_engine import KnowledgeGraphEngine
    from cke.retrieval import graph_retriever as module

    monkeypatch.setattr(resolver_module, "fuzz", None)

    with pytest.raises(DegradedComponentError, match="rapidfuzz"):
        module.GraphRetriever(KnowledgeGraphEngine(), strict=True)


def test_query_orchestrator_propagates_strict(monkeypatch):
    """The orchestrator hardcoded non-strict for its reasoner and resolver."""
    import inspect

    from cke.pipeline import query_orchestrator as module

    signature = inspect.signature(module.QueryOrchestrator.__init__)
    assert "strict" in signature.parameters

    source = inspect.getsource(module.QueryOrchestrator.__init__)
    assert "PathReasoner(strict=strict)" in source
    assert "EntityResolver(strict=strict)" in source


def test_coreference_declares_the_fallback_when_a_model_finds_nothing(monkeypatch):
    """A loaded model that finds no entities silently used the regex."""
    from cke.extractor import coreference_resolver as module

    class ModelWithNoEntities:
        def __call__(self, document):
            class Doc:
                ents: list = []

            return Doc()

    resolver = module.CoreferenceResolver()
    resolver.degraded = False
    resolver.degraded_reason = ""
    resolver._degradation_reasons = []
    resolver._nlp = ModelWithNoEntities()

    resolver.resolve("Ada Lovelace wrote notes. She was a mathematician.")

    assert resolver.degraded is True
    assert "found no entities" in resolver.degraded_reason


def test_relation_mapper_declares_an_ontology_with_no_relations(tmp_path):
    """A file that parses but has no relations key was a silent empty load."""
    from cke.schema.relation_mapper import RelationMapper

    schema = tmp_path / "relations.yaml"
    schema.write_text("something_else: {}\n", encoding="utf-8")

    mapper = RelationMapper(schema_path=str(schema))
    assert mapper.degraded is True
    assert "no 'relations' mapping" in mapper.degraded_reason

    with pytest.raises(DegradedComponentError):
        RelationMapper(schema_path=str(schema), strict=True)


def test_trust_engine_does_not_load_a_config_it_will_discard(tmp_path):
    """A supplied calibrator carries its own config, so a missing file here
    was refusing over values that would never be used."""
    from cke.graph.trust_engine import TrustEngine
    from cke.trust.calibration import TrustCalibrator

    calibrator = TrustCalibrator()
    engine = TrustEngine(
        calibrator=calibrator,
        config_path=tmp_path / "absent.yaml",
        strict=True,
    )

    assert engine.calibrator is calibrator
    assert engine.degraded is False


# ---------------------------------------------------------------------------
# The contract must stay complete
# ---------------------------------------------------------------------------


def test_every_component_in_the_contract_accepts_strict():
    """A class that can degrade but cannot be made strict is a hole.

    Threading a cross-cutting concern is easy to leave half-done, so this
    walks the package rather than trusting a hand-kept list.
    """
    import ast
    import pathlib

    from cke.diagnostics import DegradationMixin

    root = pathlib.Path(DegradationMixin.__module__.split(".")[0])
    if not root.exists():  # pragma: no cover - depends on the working directory
        root = pathlib.Path(__file__).resolve().parents[1] / "cke"

    missing = []
    for path in sorted(root.rglob("*.py")):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if "DegradationMixin" not in [ast.unparse(b) for b in node.bases]:
                continue
            init = next(
                (
                    n
                    for n in node.body
                    if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                ),
                None,
            )
            if init is None:
                # A dataclass has no explicit __init__, so the parameter has
                # to come from an annotated field. TokenTracker sat in the
                # contract unable to be constructed strict because this
                # branch used to skip such classes entirely.
                decorators = [ast.unparse(d) for d in node.decorator_list]
                if any("dataclass" in d for d in decorators):
                    fields = [
                        n.target.id
                        for n in node.body
                        if isinstance(n, ast.AnnAssign)
                        and isinstance(n.target, ast.Name)
                    ]
                    if "strict" not in fields:
                        missing.append(f"{path}:{node.lineno} {node.name} (dataclass)")
                continue
            names = [a.arg for a in init.args.args + init.args.kwonlyargs]
            if "strict" not in names:
                missing.append(f"{path}:{node.lineno} {node.name}")

    assert not missing, "components that can degrade but take no strict: " + ", ".join(
        missing
    )


# ----------------------------------------------------------------------
# Injected components must declare themselves strict
# ----------------------------------------------------------------------


def _injection_cases():
    """Each strict wrapper that accepts a prebuilt component, and one to give it.

    A wrapper constructing its own dependency passes strict down. A wrapper
    handed one cannot: the object was built elsewhere and may already have
    degraded, so a strict run has to refuse it rather than inherit its
    fallbacks while still calling itself strict.
    """
    from cke.entity_resolution.entity_resolver import EntityResolver
    from cke.graph.domain_classifier import DomainClassifier
    from cke.graph_engine.graph_engine import KnowledgeGraphEngine
    from cke.retrieval.graph_retriever import GraphRetriever
    from cke.retrieval.retriever import GraphRetriever as SimpleGraphRetriever
    from cke.router.router import QueryRouter

    graph = KnowledgeGraphEngine()
    return [
        (
            "SimpleGraphRetriever.router",
            lambda strict_dep, strict: SimpleGraphRetriever(
                graph, router=QueryRouter(strict=strict_dep), strict=strict
            ),
        ),
        (
            "GraphRetriever.entity_resolver",
            lambda strict_dep, strict: GraphRetriever(
                graph,
                entity_resolver=EntityResolver(strict=strict_dep),
                strict=strict,
            ),
        ),
        (
            "QueryRouter.domain_classifier",
            lambda strict_dep, strict: QueryRouter(
                domain_classifier=DomainClassifier(strict=strict_dep), strict=strict
            ),
        ),
    ]


@pytest.mark.parametrize("label", [case[0] for case in _injection_cases()])
def test_a_strict_wrapper_refuses_a_non_strict_injection(label):
    build = next(build for name, build in _injection_cases() if name == label)

    with pytest.raises(DegradedComponentError):
        build(False, True)


@pytest.mark.parametrize("label", [case[0] for case in _injection_cases()])
def test_a_strict_wrapper_accepts_a_strict_injection(label):
    build = next(build for name, build in _injection_cases() if name == label)

    assert build(True, True) is not None


@pytest.mark.parametrize("label", [case[0] for case in _injection_cases()])
def test_a_non_strict_wrapper_accepts_anything(label):
    build = next(build for name, build in _injection_cases() if name == label)

    assert build(False, False) is not None
