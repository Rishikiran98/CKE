from cke.entity_resolution.entity_resolver import EntityResolver
from cke.extractor.coreference_resolver import CoreferenceResolver
from cke.extractor.extraction_pipeline import ExtractionPipeline
from cke.extractor.paragraph_extractor import ParagraphExtractor
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.models import Statement
from cke.schema.relation_mapper import RelationMapper
from cke.trust.confidence_model import ConfidenceModel


class StubExtractor:
    def extract(self, text: str) -> list[Statement]:
        if "Christopher Nolan" in text and "Interstellar" in text:
            return [Statement("Christopher Nolan", "director_of", "Interstellar")]
        if "Tony Scott" in text and "Tom Cruise" in text:
            return [Statement("Tony Scott", "directed", "Top Gun")]
        return []


def test_entity_resolver_canonicalises_variants():
    graph = KnowledgeGraphEngine()
    graph.add_statement("Scott Derrickson", "directed", "Doctor Strange")
    resolver = EntityResolver(graph_engine=graph)

    assert resolver.resolve("scott derrickson") == "Scott Derrickson"
    assert resolver.resolve("Scott_Derrickson") == "Scott Derrickson"


def test_relation_mapper_alias_and_fuzzy_mapping():
    mapper = RelationMapper()
    assert mapper.map("director_of") == "directed"
    assert mapper.map("starred in") == "acted_in"


def test_coreference_resolver_rewrites_pronouns():
    resolver = CoreferenceResolver()
    text = "Christopher Nolan directed Inception. He later directed Interstellar."
    resolved = resolver.resolve(text)

    assert "Christopher Nolan later directed Interstellar" in resolved


def test_paragraph_extractor_sliding_windows():
    extractor = ParagraphExtractor(window_size=3)
    windows = extractor.sentence_windows("A. B. C. D.")
    assert windows == ["A. B. C.", "B. C. D."]


def test_confidence_model_outputs_probability():
    model = ConfidenceModel()
    assertion = Statement(
        "A",
        "located_in",
        "B",
        context={
            "span_quality": 0.9,
            "entity_link_confidence": 0.8,
            "llm_logprob": -0.2,
            "source_reliability": 0.95,
        },
    )
    confidence = model.predict(assertion)
    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.5


def test_extraction_pipeline_integrates_components():
    graph = KnowledgeGraphEngine()
    pipeline = ExtractionPipeline(
        graph_engine=graph, extractor=StubExtractor(), window_size=2
    )
    doc = "Christopher Nolan directed Inception. He later directed Interstellar."

    assertions = pipeline.process_document(doc, source="unit-test")

    assert assertions
    assert assertions[0].subject == "Christopher Nolan"
    assert assertions[0].relation == "directed"
    assert assertions[0].object == "Interstellar"
    assert assertions[0].confidence > 0.0

    neighbors = graph.get_neighbors("Christopher Nolan")
    assert any(n.object == "Interstellar" for n in neighbors)
