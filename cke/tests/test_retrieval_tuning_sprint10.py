"""Sprint 10 retrieval tuning coverage."""

from __future__ import annotations

from cke.models import Statement
from cke.pipeline.evidence_assembler import EvidenceAssembler
from cke.pipeline.types import EvidenceFact, ResolvedEntity
from cke.retrieval.chunk_fact_store import ChunkFactStore
from cke.retrieval.evidence_retriever import EvidenceRetriever
from cke.retrieval.path_scorer import CandidatePathScorer
from cke.retrieval.path_types import CandidatePath
from cke.router.query_plan import QueryPlan


class StubDenseRetriever:
    def __init__(self, payload):
        self.payload = payload

    def retrieve(self, query: str, k: int = 5):
        return self.payload[:k]


def _entity(name: str) -> ResolvedEntity:
    return ResolvedEntity(
        surface_form=name,
        canonical_name=name,
        entity_id=name.lower().replace(" ", "_"),
        link_confidence=0.95,
    )


def _fact(
    subject: str,
    relation: str,
    obj: str,
    *,
    retrieval_score: float = 0.4,
    trust_score: float = 0.9,
) -> EvidenceFact:
    return EvidenceFact(
        statement=Statement(
            subject=subject,
            relation=relation,
            object=obj,
            canonical_subject_id=subject.lower().replace(" ", "_"),
            trust_score=trust_score,
        ),
        chunk_id=f"{subject}-{relation}",
        source="test",
        trust_score=trust_score,
        retrieval_score=retrieval_score,
        entity_alignment_score=0.0,
    )


def test_relation_priority_ranking_prefers_target_relation():
    assembler = EvidenceAssembler(max_facts=3)
    facts = [
        _fact("Albert Einstein", "profession", "Physicist", retrieval_score=0.55),
        _fact("Albert Einstein", "born_in", "Ulm", retrieval_score=0.57),
        _fact("Albert Einstein", "nationality", "German", retrieval_score=0.52),
    ]

    context = assembler.assemble(
        query="What is the nationality of Albert Einstein?",
        query_plan=QueryPlan(target_relations=["nationality"]),
        resolved_entities=[_entity("Albert Einstein")],
        chunks=[],
        facts=facts,
    )

    assert context.evidence_facts[0].statement.relation == "nationality"


def test_dual_entity_comparison_preserves_both_sides():
    assembler = EvidenceAssembler(max_facts=2)
    facts = [
        _fact("Scott Derrickson", "nationality", "American", retrieval_score=0.65),
        _fact("Scott Derrickson", "profession", "Director", retrieval_score=0.63),
        _fact("Ed Wood", "nationality", "American", retrieval_score=0.51),
        _fact("Ed Wood", "born_in", "Poughkeepsie", retrieval_score=0.49),
    ]

    context = assembler.assemble(
        query="Were Scott Derrickson and Ed Wood of the same nationality?",
        query_plan=QueryPlan(
            target_relations=["nationality"],
            operator_hint="equality",
        ),
        resolved_entities=[_entity("Scott Derrickson"), _entity("Ed Wood")],
        chunks=[],
        facts=facts,
    )

    subjects = {fact.statement.subject for fact in context.evidence_facts}
    relations = {fact.statement.relation for fact in context.evidence_facts}
    assert "Scott Derrickson" in subjects
    assert "Ed Wood" in subjects
    assert "nationality" in relations


def test_operator_support_ranking_prefers_count_inputs():
    assembler = EvidenceAssembler(max_facts=3)
    facts = [
        _fact("Person X", "employer", "Company Y", retrieval_score=0.61),
        _fact("Person X", "child", "A", retrieval_score=0.45),
        _fact("Person X", "child", "B", retrieval_score=0.44),
    ]

    context = assembler.assemble(
        query="How many children did Person X have?",
        query_plan=QueryPlan(target_relations=["child"], operator_hint="count"),
        resolved_entities=[_entity("Person X")],
        chunks=[],
        facts=facts,
    )

    assert [fact.statement.relation for fact in context.evidence_facts[:2]] == [
        "child",
        "child",
    ]


def test_path_ranking_prefers_useful_multi_hop_chain():
    scorer = CandidatePathScorer()
    direct = CandidatePath(
        statements=[Statement("A", "alias_of", "A1", trust_score=0.95)],
        path_score=0.0,
        entity_overlap_score=0.0,
        relation_match_score=0.0,
        trust_score=0.0,
        summary="A alias_of A1",
    )
    useful = CandidatePath(
        statements=[
            Statement("A", "located_in", "B", trust_score=0.93),
            Statement("B", "located_in", "C", trust_score=0.91),
        ],
        path_score=0.0,
        entity_overlap_score=0.0,
        relation_match_score=0.0,
        trust_score=0.0,
        summary="A located_in B -> B located_in C",
    )

    ranked = scorer.score(
        [direct, useful],
        query_plan=QueryPlan(
            target_relations=["located_in"],
            multi_hop_hint=True,
            bridge_entities_expected=True,
        ),
        resolved_entities=[_entity("A"), _entity("C")],
    )

    assert ranked[0].summary == useful.summary
    assert ranked[0].metadata["query_type"] == "multi_hop"


def test_hybrid_chunk_score_prefers_useful_relation_match():
    fact_store = ChunkFactStore()
    fact_store.add_facts(
        "dense-wrong",
        [Statement("Albert Einstein", "profession", "Physicist", trust_score=0.9)],
    )
    fact_store.add_facts(
        "target-right",
        [Statement("Albert Einstein", "nationality", "German", trust_score=0.9)],
    )
    dense = StubDenseRetriever(
        [
            {
                "doc_id": "dense-wrong",
                "text": "Albert Einstein profession Physicist",
                "score": 0.95,
                "source": "notes",
            },
            {
                "doc_id": "target-right",
                "text": "Albert Einstein nationality German",
                "score": 0.80,
                "source": "wikipedia",
            },
        ]
    )
    retriever = EvidenceRetriever(dense, fact_store)

    chunks, facts = retriever.retrieve(
        "What is the nationality of Albert Einstein?",
        resolved_entities=[_entity("Albert Einstein")],
        target_relations=["nationality"],
        top_k=2,
    )

    assert chunks[0].chunk_id == "target-right"
    assert facts[0].statement.relation == "nationality"
