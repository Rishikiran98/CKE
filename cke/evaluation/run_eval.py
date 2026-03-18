"""CLI entry point for running the Sprint 8 evaluator locally."""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass, field

from cke.entity_resolution.entity_resolver import EntityResolver
from cke.evaluation.default_golden_set import get_default_golden_set
from cke.evaluation.e2e_evaluator import E2EEvaluator
from cke.evaluation.reporting import export_csv, export_json, generate_text_report
from cke.models import Statement
from cke.pipeline.evidence_assembler import EvidenceAssembler
from cke.pipeline.query_orchestrator import QueryOrchestrator
from cke.pipeline.types import ReasonerOutcome
from cke.retrieval.chunk_fact_store import ChunkFactStore
from cke.retrieval.evidence_retriever import EvidenceRetriever


@dataclass(slots=True)
class DemoQueryPlan:
    reasoning_route: str = "advanced_reasoner"
    decomposition: list[dict[str, str | float]] = field(default_factory=list)
    operator_hint: str | None = None
    target_relations: list[str] = field(default_factory=list)
    multi_hop_hint: bool = False
    bridge_entities_expected: bool = False


class DemoRouter:
    def route(self, query: str) -> DemoQueryPlan:
        lowered = query.lower()
        decomposition: list[dict[str, str | float]] = []
        target_relations: list[str] = []
        operator_hint = None
        multi_hop_hint = False

        relation_map = {
            "nationality": ["nationality", "citizenship"],
            "profession": ["profession"],
            "directed": ["direct", "directed", "director"],
            "child": ["children", "child"],
            "release_year": ["released later", "release year", "later"],
            "located_in": ["located in"],
            "portrayed": ["portrayed"],
            "appears_in": ["associated with the character", "appears"],
        }
        for relation, markers in relation_map.items():
            if any(marker in lowered for marker in markers):
                decomposition.append(
                    {"type": "relation", "value": relation, "confidence": 1.0}
                )
                target_relations.append(relation)

        if "same nationality" in lowered:
            operator_hint = "equality"
        elif "how many" in lowered or "count" in lowered:
            operator_hint = "count"
        elif lowered.startswith("did ") or lowered.startswith("does "):
            operator_hint = "existence"
        elif "released later" in lowered or "later" in lowered:
            operator_hint = "temporal_compare"

        if (
            "character portrayed" in lowered
            or "associated with the character" in lowered
        ):
            multi_hop_hint = True
            if "appears_in" not in target_relations:
                target_relations.append("appears_in")
        if "located in" in lowered:
            multi_hop_hint = True

        return DemoQueryPlan(
            decomposition=decomposition,
            operator_hint=operator_hint,
            target_relations=target_relations,
            multi_hop_hint=multi_hop_hint,
            bridge_entities_expected=multi_hop_hint,
        )

    def detect_entities(self, query: str) -> list[str]:
        candidates = [
            "Albert Einstein",
            "Marie Curie",
            "Scott Derrickson",
            "Ed Wood",
            "Christopher Nolan",
            "Chris Nolan",
            "Inception",
            "United States",
            "U.S.",
            "US",
            "USA",
            "Person X",
            "Character Y",
            "Film Z",
            "Film A",
            "Film B",
            "Unknown Film",
            "Unknown Person",
            "A",
            "B",
            "C",
            "Bridge Missing",
        ]
        lowered = query.lower()
        return [candidate for candidate in candidates if candidate.lower() in lowered]


class DemoRAGRetriever:
    def __init__(self, docs: list[dict[str, str | float]]) -> None:
        self.docs = docs

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, str | float]]:
        del query
        return self.docs[:k]


class DemoReasoner:
    def reason(
        self,
        query: str,
        statements: list[Statement],
        candidate_paths=None,
    ) -> ReasonerOutcome:
        lowered = query.lower()
        candidate_paths = list(candidate_paths or [])

        if "what is the nationality" in lowered or "citizenship" in lowered:
            fact = next((s for s in statements if s.relation == "nationality"), None)
            if fact:
                return ReasonerOutcome(
                    answer=fact.object,
                    confidence=0.9,
                    reasoning_path=[fact],
                    required_facts=[(fact.subject, "nationality")],
                    summary="direct_lookup",
                )

        if "who directed inception" in lowered:
            fact = next(
                (
                    s
                    for s in statements
                    if s.relation == "directed" and s.object == "Inception"
                ),
                None,
            )
            if fact:
                return ReasonerOutcome(
                    answer=fact.subject,
                    confidence=0.9,
                    reasoning_path=[fact],
                    required_facts=[(fact.subject, "directed")],
                    summary="direct_lookup",
                )

        if "profession" in lowered:
            fact = next((s for s in statements if s.relation == "profession"), None)
            if fact:
                return ReasonerOutcome(
                    answer=fact.object,
                    confidence=0.9,
                    reasoning_path=[fact],
                    required_facts=[(fact.subject, "profession")],
                    summary="direct_lookup",
                )

        if "where is a located in" in lowered:
            path = next(
                (path for path in candidate_paths if len(path.statements) == 2), None
            )
            if path:
                return ReasonerOutcome(
                    answer=path.statements[-1].object,
                    confidence=0.88,
                    reasoning_path=list(path.statements),
                    summary="path_answer",
                )

        if "associated with the character portrayed by person x" in lowered:
            path = next(
                (
                    path
                    for path in candidate_paths
                    if [statement.relation for statement in path.statements]
                    == ["portrayed", "appears_in"]
                ),
                None,
            )
            if path:
                return ReasonerOutcome(
                    answer=path.statements[-1].object,
                    confidence=0.9,
                    reasoning_path=list(path.statements),
                    summary="path_answer",
                )

        return ReasonerOutcome(
            answer="INSUFFICIENT_EVIDENCE",
            confidence=0.0,
            reasoning_path=[],
            summary="fallback_abstain",
        )


def build_demo_orchestrator() -> QueryOrchestrator:
    docs = [
        {
            "doc_id": "d1::c0",
            "text": "Albert Einstein nationality German",
            "score": 0.95,
            "source": "d1",
        },
        {
            "doc_id": "d1::c1",
            "text": "Albert Einstein nationality Swiss",
            "score": 0.94,
            "source": "d1",
        },
        {
            "doc_id": "d1::c2",
            "text": "Albert Einstein nationality United States",
            "score": 0.93,
            "source": "d1",
        },
        {
            "doc_id": "d1::c3",
            "text": "Albert Einstein profession Physicist",
            "score": 0.92,
            "source": "d1",
        },
        {
            "doc_id": "d2::c0",
            "text": "Marie Curie nationality Polish",
            "score": 0.91,
            "source": "d2",
        },
        {
            "doc_id": "d3::c0",
            "text": "Christopher Nolan directed Inception",
            "score": 0.96,
            "source": "d3",
        },
        {
            "doc_id": "d4::c0",
            "text": "Scott Derrickson nationality American",
            "score": 0.9,
            "source": "d4",
        },
        {
            "doc_id": "d4::c1",
            "text": "Ed Wood nationality American",
            "score": 0.89,
            "source": "d4",
        },
        {"doc_id": "d5::c0", "text": "Person X child A", "score": 0.88, "source": "d5"},
        {"doc_id": "d5::c1", "text": "Person X child B", "score": 0.87, "source": "d5"},
        {"doc_id": "d5::c2", "text": "Person X child C", "score": 0.86, "source": "d5"},
        {
            "doc_id": "d6::c0",
            "text": "Film A release_year 1990",
            "score": 0.85,
            "source": "d6",
        },
        {
            "doc_id": "d6::c1",
            "text": "Film B release_year 2001",
            "score": 0.84,
            "source": "d6",
        },
        {"doc_id": "d7::c0", "text": "A located_in B", "score": 0.83, "source": "d7"},
        {"doc_id": "d7::c1", "text": "B located_in C", "score": 0.82, "source": "d7"},
        {
            "doc_id": "d8::c0",
            "text": "Person X portrayed Character Y",
            "score": 0.81,
            "source": "d8",
        },
        {
            "doc_id": "d8::c1",
            "text": "Character Y appears_in Film Z",
            "score": 0.8,
            "source": "d8",
        },
    ]

    store = ChunkFactStore()
    statements = {
        "d1::c0": [
            Statement("Albert Einstein", "nationality", "German", trust_score=0.95)
        ],
        "d1::c1": [
            Statement("Albert Einstein", "nationality", "Swiss", trust_score=0.94)
        ],
        "d1::c2": [
            Statement(
                "Albert Einstein", "nationality", "United States", trust_score=0.93
            )
        ],
        "d1::c3": [
            Statement("Albert Einstein", "profession", "Physicist", trust_score=0.92)
        ],
        "d2::c0": [Statement("Marie Curie", "nationality", "Polish", trust_score=0.91)],
        "d3::c0": [
            Statement("Christopher Nolan", "directed", "Inception", trust_score=0.96)
        ],
        "d4::c0": [
            Statement("Scott Derrickson", "nationality", "American", trust_score=0.9)
        ],
        "d4::c1": [Statement("Ed Wood", "nationality", "American", trust_score=0.89)],
        "d5::c0": [Statement("Person X", "child", "A", trust_score=0.88)],
        "d5::c1": [Statement("Person X", "child", "B", trust_score=0.87)],
        "d5::c2": [Statement("Person X", "child", "C", trust_score=0.86)],
        "d6::c0": [Statement("Film A", "release_year", "1990", trust_score=0.85)],
        "d6::c1": [Statement("Film B", "release_year", "2001", trust_score=0.84)],
        "d7::c0": [Statement("A", "located_in", "B", trust_score=0.83)],
        "d7::c1": [Statement("B", "located_in", "C", trust_score=0.82)],
        "d8::c0": [Statement("Person X", "portrayed", "Character Y", trust_score=0.81)],
        "d8::c1": [Statement("Character Y", "appears_in", "Film Z", trust_score=0.8)],
    }
    for chunk_id, fact_list in statements.items():
        store.add_facts(chunk_id, fact_list)

    resolver = EntityResolver(
        aliases={
            "Chris Nolan": "Christopher Nolan",
            "US": "United States",
            "U.S.": "United States",
            "USA": "United States",
        }
    )
    return QueryOrchestrator(
        graph_engine=None,
        router=DemoRouter(),
        retriever=EvidenceRetriever(DemoRAGRetriever(docs), store),
        assembler=EvidenceAssembler(max_candidate_paths=10),
        reasoner=DemoReasoner(),
        entity_resolver=resolver,
    )


def _load_factory(path: str):
    module_name, function_name = path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--factory",
        default="cke.evaluation.run_eval:build_demo_orchestrator",
        help="Import path to orchestrator factory in module:function form.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    factory = _load_factory(args.factory)
    orchestrator = factory()
    cases = get_default_golden_set()
    evaluator = E2EEvaluator(orchestrator)
    results, summary = evaluator.evaluate_cases(cases)

    print(generate_text_report(results, summary))
    if args.output_json:
        export_json(results, summary, args.output_json)
    if args.output_csv:
        export_csv(results, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
