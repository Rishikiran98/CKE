"""Document-level extraction pipeline orchestrator."""

from __future__ import annotations

from cke.extractor.coreference_resolver import CoreferenceResolver
from cke.extractor.entity_linker import EntityResolver
from cke.extractor.llm_extractor import LLMExtractor
from cke.extractor.paragraph_extractor import ParagraphExtractor
from cke.extractor.rule_extractor import RuleExtractor
from cke.models import Statement
from cke.schema.relation_mapper import RelationMapper
from cke.trust.confidence_model import ConfidenceModel


class ExtractionPipeline:
    """End-to-end extraction from raw documents to graph assertions."""

    def __init__(self, graph_engine, extractor=None, window_size: int = 3) -> None:
        self.graph_engine = graph_engine
        self.coref = CoreferenceResolver()
        self.paragraph_extractor = ParagraphExtractor(window_size=window_size)
        self.extractor = extractor or LLMExtractor(fallback=RuleExtractor())
        self.entity_resolver = EntityResolver(graph_engine)
        self.relation_mapper = RelationMapper()
        self.confidence_model = ConfidenceModel()

    def process_document(
        self, document: str, source: str | None = None
    ) -> list[Statement]:
        resolved_doc = self.coref.resolve(document)
        windows = self.paragraph_extractor.sentence_windows(resolved_doc)
        assertions: list[Statement] = []

        for window in windows:
            for statement in self.extractor.extract(window):
                subject_result = self.entity_resolver.resolve_with_score(
                    statement.subject
                )
                object_result = self.entity_resolver.resolve_with_score(
                    statement.object
                )
                relation = self.relation_mapper.map(statement.relation)

                context = dict(statement.context)
                context.setdefault(
                    "entity_link_confidence",
                    min(subject_result.confidence, object_result.confidence),
                )
                context.setdefault("relation_type", relation)
                confidence_statement = Statement(
                    subject=subject_result.canonical,
                    relation=relation,
                    object=object_result.canonical,
                    context=context,
                    source=source,
                )
                confidence = self.confidence_model.predict(confidence_statement)
                final = Statement(
                    subject=subject_result.canonical,
                    relation=relation,
                    object=object_result.canonical,
                    context=context,
                    confidence=confidence,
                    source=source,
                )
                self.graph_engine.add_statement(
                    final.subject,
                    final.relation,
                    final.object,
                    context=final.context,
                    confidence=final.confidence,
                    source=final.source,
                )
                assertions.append(final)
        return assertions
