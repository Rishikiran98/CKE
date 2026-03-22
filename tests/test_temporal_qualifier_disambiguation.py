"""Tests for temporal qualifier disambiguation across the unified schema."""

from cke.graph.assertion import Assertion
from cke.graph.conflict_engine import ConflictEngine
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.models import Statement
from cke.pipeline.evidence_assembler import EvidenceAssembler, _qualifier_hash
from cke.pipeline.types import EvidenceFact


class TestTemporalConflictDisambiguation:
    """Temporal qualifiers prevent false conflict detection."""

    def test_non_overlapping_temporal_ranges_do_not_conflict(self):
        engine = ConflictEngine()
        german = Assertion(
            subject="Albert Einstein",
            relation="nationality",
            object="German",
            qualifiers={"temporal": {"end": "1940"}},
            trust_score=0.9,
        )
        american = Assertion(
            subject="Albert Einstein",
            relation="nationality",
            object="American",
            qualifiers={"temporal": {"start": "1940"}},
            trust_score=0.9,
        )

        assert not engine.assertions_conflict(german, american)
        conflicts = engine.detect_conflicts([german, american])
        assert len(conflicts) == 0

    def test_overlapping_temporal_ranges_do_conflict(self):
        engine = ConflictEngine()
        a = Assertion(
            subject="Albert Einstein",
            relation="nationality",
            object="German",
            qualifiers={"temporal": {"start": "1879", "end": "1940"}},
            trust_score=0.8,
        )
        b = Assertion(
            subject="Albert Einstein",
            relation="nationality",
            object="Swiss",
            qualifiers={"temporal": {"start": "1901", "end": "1955"}},
            trust_score=0.7,
        )

        assert engine.assertions_conflict(a, b)
        conflicts = engine.detect_conflicts([a, b])
        assert len(conflicts) == 1

    def test_no_qualifiers_still_conflict(self):
        engine = ConflictEngine()
        a = Assertion("Redis", "supports", "pubsub", trust_score=0.9)
        b = Assertion("Redis", "supports", "streams", trust_score=0.4)

        assert engine.assertions_conflict(a, b)

    def test_three_way_temporal_disambiguation(self):
        """Einstein: German (1879-1940), Swiss (1901-1955), American (1940-1955)."""
        engine = ConflictEngine()
        german = Assertion(
            subject="Albert Einstein",
            relation="nationality",
            object="German",
            qualifiers={"temporal": {"start": "1879", "end": "1940"}},
            trust_score=0.9,
        )
        swiss = Assertion(
            subject="Albert Einstein",
            relation="nationality",
            object="Swiss",
            qualifiers={"temporal": {"start": "1901", "end": "1955"}},
            trust_score=0.85,
        )
        american = Assertion(
            subject="Albert Einstein",
            relation="nationality",
            object="American",
            qualifiers={"temporal": {"start": "1940", "end": "1955"}},
            trust_score=0.9,
        )

        conflicts = engine.detect_conflicts([german, swiss, american])
        # German overlaps with Swiss (1901-1940), Swiss overlaps with American (1940-1955)
        # German does NOT overlap with American (German ends 1940, American starts 1940)
        conflict_pairs = {
            (c[0].object, c[1].object) for c in conflicts
        }
        assert ("German", "American") not in conflict_pairs
        assert ("American", "German") not in conflict_pairs


class TestStatementAssertionConversion:
    """Statement <-> Assertion round-trip preserves qualifiers."""

    def test_statement_to_assertion_preserves_qualifiers(self):
        stmt = Statement(
            subject="Einstein",
            relation="nationality",
            object="German",
            qualifiers={"temporal": {"end": "1940"}, "modality": "typical"},
            confidence=0.9,
            source="encyclopedia",
        )
        assertion = stmt.to_assertion()
        assert assertion.qualifiers == {"temporal": {"end": "1940"}, "modality": "typical"}
        assert assertion.subject == "Einstein"
        assert assertion.confidence == 0.9

    def test_assertion_to_statement_preserves_qualifiers(self):
        assertion = Assertion(
            subject="Einstein",
            relation="nationality",
            object="American",
            qualifiers={"temporal": {"start": "1940"}},
            trust_score=0.85,
        )
        stmt = Statement.from_assertion(assertion)
        assert stmt.qualifiers == {"temporal": {"start": "1940"}}
        assert stmt.trust_score == 0.85

    def test_round_trip_preserves_qualifiers(self):
        original = Statement(
            subject="Redis",
            relation="supports",
            object="pubsub",
            qualifiers={"temporal": {"start": "2012"}, "scope": {"version": "2.0"}},
            confidence=0.8,
            source="docs",
        )
        assertion = original.to_assertion()
        restored = Statement.from_assertion(assertion)
        assert restored.qualifiers == original.qualifiers
        assert restored.subject == original.subject
        assert restored.relation == original.relation
        assert restored.object == original.object


class TestQualifiersInGraphRoundTrip:
    """Qualifiers survive storage and retrieval through the graph engine."""

    def test_qualifiers_survive_graph_storage(self):
        engine = KnowledgeGraphEngine()
        engine.add_statement(
            "Albert Einstein",
            "nationality",
            "German",
            context={"qualifiers": {"temporal": {"end": "1940"}}},
            confidence=0.9,
            source="encyclopedia",
        )
        neighbors = engine.get_neighbors("Albert Einstein")
        assert len(neighbors) == 1
        stmt = neighbors[0]
        assert stmt.qualifiers == {"temporal": {"end": "1940"}}

    def test_multiple_qualified_statements_in_graph(self):
        engine = KnowledgeGraphEngine()
        engine.add_statement(
            "Einstein",
            "nationality",
            "German",
            context={"qualifiers": {"temporal": {"end": "1940"}}},
            confidence=0.9,
        )
        engine.add_statement(
            "Einstein",
            "nationality",
            "American",
            context={"qualifiers": {"temporal": {"start": "1940"}}},
            confidence=0.9,
        )
        neighbors = engine.get_neighbors("Einstein")
        assert len(neighbors) == 2
        qualifiers_found = [n.qualifiers for n in neighbors]
        assert {"temporal": {"end": "1940"}} in qualifiers_found
        assert {"temporal": {"start": "1940"}} in qualifiers_found


class TestQualifierAwareDedup:
    """Evidence assembler does not merge facts with different qualifiers."""

    def test_different_temporal_qualifiers_not_deduped(self):
        stmt_german = Statement(
            subject="Einstein",
            relation="nationality",
            object="German",
            qualifiers={"temporal": {"end": "1940"}},
            confidence=0.9,
        )
        stmt_american = Statement(
            subject="Einstein",
            relation="nationality",
            object="American",
            qualifiers={"temporal": {"start": "1940"}},
            confidence=0.9,
        )
        # Same (subject, relation) but different objects and qualifiers
        fact_german = EvidenceFact(
            statement=stmt_german,
            chunk_id="c1",
            source="src",
            trust_score=0.9,
            retrieval_score=0.8,
            entity_alignment_score=1.0,
        )
        fact_american = EvidenceFact(
            statement=stmt_american,
            chunk_id="c2",
            source="src",
            trust_score=0.9,
            retrieval_score=0.8,
            entity_alignment_score=1.0,
        )
        assembler = EvidenceAssembler()
        deduped = assembler._dedupe_facts([fact_german, fact_american])
        assert len(deduped) == 2

    def test_same_qualifiers_are_deduped(self):
        stmt1 = Statement(
            subject="Redis",
            relation="supports",
            object="pubsub",
            qualifiers={"temporal": {"start": "2012"}},
            confidence=0.9,
        )
        stmt2 = Statement(
            subject="Redis",
            relation="supports",
            object="pubsub",
            qualifiers={"temporal": {"start": "2012"}},
            confidence=0.8,
        )
        fact1 = EvidenceFact(
            statement=stmt1,
            chunk_id="c1",
            source="src",
            trust_score=0.9,
            retrieval_score=0.9,
            entity_alignment_score=1.0,
        )
        fact2 = EvidenceFact(
            statement=stmt2,
            chunk_id="c2",
            source="src",
            trust_score=0.8,
            retrieval_score=0.7,
            entity_alignment_score=1.0,
        )
        assembler = EvidenceAssembler()
        deduped = assembler._dedupe_facts([fact1, fact2])
        assert len(deduped) == 1

    def test_qualifier_hash_stability(self):
        q1 = {"temporal": {"start": "1940"}, "modality": "typical"}
        q2 = {"modality": "typical", "temporal": {"start": "1940"}}
        assert _qualifier_hash(q1) == _qualifier_hash(q2)
        assert _qualifier_hash({}) == ""


class TestStatementAsText:
    """as_text() includes qualifier summary."""

    def test_as_text_with_qualifiers(self):
        stmt = Statement(
            subject="Einstein",
            relation="nationality",
            object="German",
            qualifiers={"temporal": {"end": "1940"}, "modality": "typical"},
        )
        text = stmt.as_text()
        assert "Einstein" in text
        assert "nationality" in text
        assert "German" in text
        assert "temporal=" in text
        assert "modality=" in text

    def test_as_text_without_qualifiers(self):
        stmt = Statement(subject="A", relation="B", object="C")
        assert stmt.as_text() == "A B C"
