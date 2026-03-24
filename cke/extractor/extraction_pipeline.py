"""Document-level extraction pipeline orchestrator."""

from __future__ import annotations

from cke.extractor.coreference_resolver import CoreferenceResolver
from cke.entity_resolution.entity_resolver import EntityResolver
from cke.extractor.llm_extractor import LLMExtractor
from cke.extractor.paragraph_extractor import ParagraphExtractor
from cke.extractor.rule_extractor import RuleExtractor
from cke.graph.conflict_engine import ConflictEngine
from cke.models import Statement
from cke.schema.relation_mapper import RelationMapper
from cke.trust.confidence_model import ConfidenceModel


class ExtractionPipeline:
    """End-to-end extraction from raw documents to graph assertions."""

    def __init__(
        self,
        graph_engine,
        extractor=None,
        window_size: int = 3,
        conflict_engine: ConflictEngine | None = None,
    ) -> None:
        self.graph_engine = graph_engine
        self.coref = CoreferenceResolver()
        self.paragraph_extractor = ParagraphExtractor(window_size=window_size)
        self.extractor = extractor or LLMExtractor(fallback=RuleExtractor())
        self.entity_resolver = EntityResolver(graph_engine=graph_engine)
        self.relation_mapper = RelationMapper()
        self.confidence_model = ConfidenceModel()
        self.conflict_engine = conflict_engine or ConflictEngine()

    def process_document(
        self, document: str, source: str | None = None
    ) -> list[Statement]:
        resolved_doc = self.coref.resolve(document)
        windows = self.paragraph_extractor.sentence_windows(resolved_doc)
        assertions: list[Statement] = []
        source_doc_id = source or "unknown_doc"

        for index, window in enumerate(windows):
            chunk_id = f"{source_doc_id}::chunk-{index}"
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
                qualifiers = dict(statement.qualifiers)
                confidence_statement = Statement(
                    subject=subject_result.canonical,
                    relation=relation,
                    object=object_result.canonical,
                    context=context,
                    qualifiers=qualifiers,
                    source=source,
                )
                confidence = self.confidence_model.predict(confidence_statement)
                context["qualifiers"] = qualifiers
                final = Statement(
                    subject=subject_result.canonical,
                    relation=relation,
                    object=object_result.canonical,
                    context=context,
                    confidence=confidence,
                    qualifiers=qualifiers,
                    source=source,
                    chunk_id=chunk_id,
                    source_doc_id=source_doc_id,
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

        assertions = self._resolve_conflicts(assertions)
        return assertions

    def _resolve_conflicts(self, statements: list[Statement]) -> list[Statement]:
        """Run qualifier-aware conflict resolution on extracted statements."""
        if not statements:
            return statements
        assertions = [s.to_assertion() for s in statements]
        conflicts = self.conflict_engine.detect_conflicts(assertions)
        for left, right in conflicts:
            self.conflict_engine.resolve_conflict(left, right)
        return [Statement.from_assertion(a) for a in assertions]
